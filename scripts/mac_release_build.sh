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
    -DCUMETAL_ENABLE_BINARY_SHIM=OFF

cache="${BUILD_DIR}/CMakeCache.txt"

# CUMETAL_ENABLE_BINARY_SHIM creates the libcuda.dylib alias, which is the
# EULA-sensitive drop-in case documented in docs/legal-notice.md. It defaults OFF
# only in Release, so a release must assert both rather than trust the default.
grep -q '^CMAKE_BUILD_TYPE:STRING=Release$' "$cache" \
    || die "CMAKE_BUILD_TYPE is not Release in $cache"
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

step "Test"
scripts/ci_report.sh "$BUILD_DIR" --require-tests

# -------------------------------------------------------------------- package -

step "Stage"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cmake --install "$BUILD_DIR" --prefix "$STAGE_DIR"

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
