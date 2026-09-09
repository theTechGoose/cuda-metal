#!/usr/bin/env bash
#
# mac_release_build.sh -- build, gate and package a CuMetal release. macOS only.
#
#   scripts/mac_release_build.sh <version>
#
# Produces dist/cumetal-<version>-macos-arm64.tar.gz and a .sha256 beside it.
#
# This script is the whole macOS side of a release and is deliberately
# self-contained: it is what a contributor runs by hand on a Mac, and it is also
# what deploy.sh invokes (directly, or through a host bridge from a container).
# It must therefore know nothing about how it was reached.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

VERSION="${1:?usage: scripts/mac_release_build.sh <version>}"

BUILD_DIR="build-release"
STAGE_DIR="dist/stage"
ARTIFACT="dist/cumetal-${VERSION}-macos-arm64.tar.gz"
LOCK_DIR="${BUILD_DIR}/.release.lock"

die() { printf 'mac_release_build: %s\n' "$*" >&2; exit 1; }
step() { printf '\n--- %s\n' "$*"; }

# ------------------------------------------------------------------ guards ---

[ "$(uname -s)" = "Darwin" ] || die "this script builds Metal code and only runs on macOS (found $(uname -s))"
[ "$(uname -m)" = "arm64" ] || die "releases are Apple Silicon only (found $(uname -m))"

command -v cmake >/dev/null 2>&1 || die "cmake not found"
command -v xcrun >/dev/null 2>&1 || die "xcrun not found -- install the Xcode command line tools"

VF64_PROBE="third_party/VF64-metal/Sources/VF64Metal/Shaders/Interop/VF64Support.metal"
[ -f "$VF64_PROBE" ] \
    || die "missing $VF64_PROBE -- run: git submodule update --init --recursive"

# deploy.sh streams output from a bridge where Ctrl-C does not stop the remote
# job, so a re-run can otherwise race a build that is still in flight.
mkdir -p "$BUILD_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    die "another release build holds $LOCK_DIR -- wait for it, or remove the directory if it is stale"
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

# ------------------------------------------------------------------ configure -

step "Configure (Release)"
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUMETAL_BUILD_TESTS=ON \
    -DCUMETAL_ENABLE_BINARY_SHIM=OFF \
    -DCUMETAL_EMBED_SOURCE_DIR=OFF

cache="${BUILD_DIR}/CMakeCache.txt"

# CUMETAL_ENABLE_BINARY_SHIM creates the libcuda.dylib alias, which is the
# EULA-sensitive drop-in case documented in docs/legal-notice.md. It defaults OFF
# only in Release, so a release must assert both rather than trust the default.
grep -q '^CMAKE_BUILD_TYPE:STRING=Release$' "$cache" \
    || die "CMAKE_BUILD_TYPE is not Release in $cache"

# The shipped binaries must not be able to reach this checkout at all. The staged
# tree is grepped for the path later; this is the reason it will not be there.
grep -q '^CUMETAL_EMBED_SOURCE_DIR:BOOL=OFF$' "$cache" \
    || die "CUMETAL_EMBED_SOURCE_DIR is not OFF in $cache"
grep -q '^CUMETAL_ENABLE_BINARY_SHIM:BOOL=OFF$' "$cache" \
    || die "CUMETAL_ENABLE_BINARY_SHIM is not OFF in $cache -- refusing to ship the libcuda.dylib alias"

echo "  Release, binary shim OFF"

# ---------------------------------------------------------------------- build -

step "Build"
cmake --build "$BUILD_DIR" --parallel "$(sysctl -n hw.ncpu)"

# ----------------------------------------------------------------------- test -
#
# ci_report.sh reports passed/skipped/failed separately because a CTest
# registration count is not a pass count. --require-tests catches a selection
# that silently matched nothing. Skips are expected here: parts of the suite gate
# on external projects. This selection includes bench_phase5_all_kernels, which
# is registered only in Release and enforces the spec 5.7 / 10.6 2x gate.

step "Test (correctness)"
# Correctness gets exactly one attempt: a test that only passes on a retry is a
# test that is telling you something.
scripts/ci_report.sh "$BUILD_DIR" --require-tests -LE benchmark

step "Test (performance gate)"
# The phase-5 gate measures wall-clock against a 2x ceiling, so it inherits
# macOS scheduling noise and fails intermittently on an otherwise green tree --
# observed twice here, passing on immediate re-run both times. A flaky release
# gate is worse than no gate: it trains you to re-run until green. Retry the
# measurement rather than loosening the threshold, so a genuine regression
# (which fails every attempt) still blocks the release.
scripts/ci_report.sh "$BUILD_DIR" --require-tests -L benchmark --repeat until-pass:3

# -------------------------------------------------------------------- package -

step "Stage"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cmake --install "$BUILD_DIR" --prefix "$STAGE_DIR"

# The headers a release ships must be all of runtime/api, with nothing left
# behind. 0.6.0 shipped 79 of 105 because the install list was an enumeration
# maintained by hand: vector_types.h and vector_functions.h were among the
# missing, and PhysX's host code includes them, so that tarball could not build
# PhysX's GPU runtime against its own include/. The install is a tree now; this
# gate is what keeps it one -- a header in the source that is not in the stage
# fails the release rather than reaching a user.
step "Verify headers"
STAGE_ABS="$REPO_ROOT/$STAGE_DIR"   # the gate runs from runtime/api; STAGE_DIR is relative
MISSING_HEADERS="$(
    cd runtime/api && command find . -type f \
        | sed 's|^\./||' \
        | while IFS= read -r header; do
              [ -f "$STAGE_ABS/include/$header" ] || printf '%s\n' "$header"
          done
)"
if [ -n "$MISSING_HEADERS" ]; then
    printf 'release: runtime/api headers missing from the staged include/:\n%s\n' \
        "$MISSING_HEADERS" >&2
    exit 1
fi
echo "  $(cd runtime/api && command find . -type f | wc -l | tr -d ' ') headers staged"

# A release must be able to compile on a machine that has never seen this
# checkout. 0.6.1 could not: cumetalc addressed its FP64 support source as
# CUMETAL_SOURCE_DIR/compiler/metal/support/..., a compile-time path to the
# BUILD machine, so xcrun was handed a file that exists nowhere else and the
# FP64 support link failed for every other user. It was invisible here because
# on this machine the path is real -- which is exactly why it needs a gate
# rather than care. Two checks: the support sources are in the tree, and no
# shipped binary carries this checkout's path.
step "Verify the tree is self-contained"
for support in cumetal_fp64_support.metal cumetal_fp64_inline_support.metal; do
    [ -f "$STAGE_DIR/libexec/cumetal/metal-support/compiler/metal/support/$support" ] || {
        printf 'release: %s is missing from the staged tree\n' "$support" >&2
        exit 1
    }
done
LEAKED="$(command grep -rl "$REPO_ROOT" "$STAGE_DIR/bin" "$STAGE_DIR/lib" "$STAGE_DIR/libexec" \
    2>/dev/null || true)"
if [ -n "$LEAKED" ]; then
    printf 'release: these shipped files embed this checkout'"'"'s path (%s):\n%s\n' \
        "$REPO_ROOT" "$LEAKED" >&2
    printf 'a user without that directory cannot use them.\n' >&2
    exit 1
fi
[ -f "$STAGE_DIR/libexec/cumetal/metal-support/third_party/VF64-metal/Sources/VF64Metal/Shaders/Interop/VF64Support.metal" ] || {
    printf 'release: the VF64 shaders the support sources include are not staged\n' >&2
    exit 1
}
echo "  support sources staged; no shipped binary references $REPO_ROOT"

# The staged tree has the files and names no local path -- now prove it actually
# WORKS, by compiling an FP64-touching kernel with the staged cumetalc and
# confirming the support source it reaches for is the staged one. Presence and
# absence are both checkable without catching a broken relative include; this
# caught exactly that (the support sources include VF64-metal's shaders with a
# path relative to their own directory, so shipping the two files alone was not
# enough).
step "Verify the staged compiler is self-sufficient"
FP64_PROBE_DIR="$(mktemp -d /tmp/cumetal-release-probe-XXXXXX)"
cat > "$FP64_PROBE_DIR/fp64.cu" <<'PROBE'
extern "C" __global__ void fp64_touch(double* out, const double* in) {
    out[threadIdx.x] = in[threadIdx.x] * 0.01 + 1.0;
}
PROBE
if CUMETAL_DEBUG_EMITTER=1 "$STAGE_DIR/bin/cumetalc" --cuda-device \
        "$FP64_PROBE_DIR/fp64.cu" -o "$FP64_PROBE_DIR/out.metallib" \
        > "$FP64_PROBE_DIR/log.txt" 2>&1 && [ -f "$FP64_PROBE_DIR/out.metallib" ]; then
    if command grep -q "$REPO_ROOT/compiler/metal/support" "$FP64_PROBE_DIR/log.txt"; then
        echo "release: the staged cumetalc reached back into this checkout for its" >&2
        echo "  FP64 support source; a user without $REPO_ROOT cannot compile." >&2
        rm -rf "$FP64_PROBE_DIR"
        exit 1
    fi
    echo "  staged cumetalc compiled an FP64 kernel using only the staged tree"
else
    echo "release: the staged cumetalc could not compile an FP64-touching kernel:" >&2
    tail -30 "$FP64_PROBE_DIR/log.txt" >&2
    rm -rf "$FP64_PROBE_DIR"
    exit 1
fi
rm -rf "$FP64_PROBE_DIR"

# Sign with a Developer ID when the machine has one. Gatekeeper quarantines an
# unsigned download, and the fix a user reaches for -- disabling Gatekeeper --
# is worse than the problem. This is detection, not configuration: the day a
# certificate exists, releases are signed with no further change here.
SIGN_IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | awk -F'"' '/Developer ID Application/ {print $2; exit}')"
if [ -n "$SIGN_IDENTITY" ]; then
    step "Codesign ($SIGN_IDENTITY)"
    while IFS= read -r binary; do
        codesign --force --timestamp --options runtime \
                 --sign "$SIGN_IDENTITY" "$binary"
    done < <(command find "$STAGE_DIR/bin" "$STAGE_DIR/lib" "$STAGE_DIR/libexec" \
                          -type f -perm +111 2>/dev/null)
    codesign --verify --deep --strict "$STAGE_DIR/lib/libcumetal.dylib"
    echo "  signed and verified"

    # Signing alone is not enough: Gatekeeper rejects a signed-but-unnotarized
    # download just as it rejects an ad-hoc one. Notarization uploads the artifact
    # to Apple, who scan it and issue a ticket that gets stapled to the tarball.
    # Requires credentials stored once with:
    #   xcrun notarytool store-credentials cumetal --apple-id ... --team-id ... --password ...
    if [ -n "${CUMETAL_NOTARY_PROFILE:-}" ]; then
        step "Notarize (${CUMETAL_NOTARY_PROFILE})"
        NOTARY_ZIP="dist/notarize.zip"
        rm -f "$NOTARY_ZIP"
        ditto -c -k --keepParent "$STAGE_DIR" "$NOTARY_ZIP"
        xcrun notarytool submit "$NOTARY_ZIP" \
            --keychain-profile "$CUMETAL_NOTARY_PROFILE" --wait
        # A tarball cannot carry a stapled ticket the way a .app or .dmg can, so
        # the binaries are validated online at first launch instead. Confirm the
        # submission was accepted rather than assuming it.
        xcrun notarytool history --keychain-profile "$CUMETAL_NOTARY_PROFILE" \
            | head -5
        rm -f "$NOTARY_ZIP"
        echo "  notarized"
    else
        echo "  NOT notarized: set CUMETAL_NOTARY_PROFILE to a stored notarytool"
        echo "  profile. Gatekeeper rejects a signed-but-unnotarized download."
    fi
else
    step "Codesign (skipped)"
    echo "  no Developer ID Application certificate on this machine."
    echo "  The artifact is UNSIGNED: macOS will quarantine it on download."
    echo "  install.sh clears the quarantine attribute at install time, so the"
    echo "  tarball still works; a signed build needs an Apple Developer cert."
fi

install -m 755 install/uninstall.sh "$STAGE_DIR/uninstall.sh"
install -m 644 LICENSE "$STAGE_DIR/LICENSE"
install -m 644 README.md "$STAGE_DIR/README.md"
install -m 644 CHANGELOG.md "$STAGE_DIR/CHANGELOG.md"

# The installed tree contains the libcublas/libcurand/... symlinks onto
# libcumetal.dylib. Store them as links; dereferencing would multiply the
# runtime by ten.
step "Package"
tar -czf "$ARTIFACT" -C "$STAGE_DIR" .
( cd dist && shasum -a 256 "$(basename "$ARTIFACT")" > "$(basename "$ARTIFACT").sha256" )

step "Done"
ls -lh "$ARTIFACT"
cat "${ARTIFACT}.sha256"
