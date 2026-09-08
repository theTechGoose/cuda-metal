#!/usr/bin/env bash
#
# deploy.sh -- cut a CuMetal release from main.
#
#   ./deploy.sh 0.6.0             # build, test, package; stop before anything public
#   ./deploy.sh 0.6.0 --publish   # ...then commit, tag, push, and create the GitHub release
#
# The build itself is macOS-only: the runtime links Metal, MetalPerformanceShaders
# and Accelerate, and compiles Objective-C++. That work lives entirely in
# scripts/mac_release_build.sh, which knows nothing about this script.
#
# This script only decides HOW to reach a machine that can run it. See
# run_macos_build() -- on a Mac it just runs; anywhere else it looks for a
# host bridge and delegates. Nothing here is specific to a container runtime.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO_ROOT"

BUILD_SCRIPT="scripts/mac_release_build.sh"
RELEASE_BRANCH="main"
MAC_RUN_TIMEOUT=7200   # mac-run's documented maximum
NATIVE_HEADER="runtime/api/cumetal_native.h"
VERSIONED_FILES="CMakeLists.txt CHANGELOG.md $NATIVE_HEADER"
BUMPED=0

VERSION=""
PUBLISH=0

die() { printf 'deploy: %s\n' "$*" >&2; exit 1; }
step() { printf '\n==> %s\n' "$*"; }

usage() {
    cat <<'EOF'
usage: ./deploy.sh <version> [--publish]

  <version>   semantic version without the leading v, e.g. 0.6.0
  --publish   actually commit, tag, push and create the GitHub release.
              Without it the release is built and packaged locally and
              nothing leaves this machine.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --publish) PUBLISH=1; shift ;;
        -*) die "unknown option: $1" ;;
        *)
            [ -n "$VERSION" ] && die "unexpected extra argument: $1"
            VERSION="$1"; shift ;;
    esac
done

[ -n "$VERSION" ] || { usage >&2; exit 2; }
case "$VERSION" in
    v*) die "pass the version without the leading v (got '$VERSION')" ;;
esac
printf '%s' "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' \
    || die "version must be MAJOR.MINOR.PATCH (got '$VERSION')"

TAG="v$VERSION"
ARTIFACT="dist/cumetal-${VERSION}-macos-arm64.tar.gz"

# A failed run must not leave a half-applied release in the tree. Preflight has
# already established that these files were clean, so restoring them is safe and
# keeps the next attempt (and any stray `git add -A`) from committing a partial bump.
PUBLISHED=0
on_exit() {
    local code=$?
    if [ "$code" -ne 0 ] && [ "$BUMPED" -eq 1 ] && [ "$PUBLISHED" -eq 0 ]; then
        printf '\ndeploy: run failed -- restoring %s\n' "$VERSIONED_FILES" >&2
        # From HEAD, not from the index. publish() stages these before committing,
        # so a plain `git checkout -- <paths>` restores the staged (already bumped)
        # content over itself and silently changes nothing -- which is worse than
        # not trying, because the message says it worked.
        # shellcheck disable=SC2086
        git checkout HEAD -- $VERSIONED_FILES 2>/dev/null || true
        git reset -q HEAD -- $VERSIONED_FILES 2>/dev/null || true
    fi
    return $code
}
trap on_exit EXIT

# Portable in-place edit. BSD sed (macOS) and GNU sed (the box) disagree about
# -i, so neither is used.
rewrite() {
    local file="$1"; shift
    local tmp
    tmp="$(mktemp)"
    "$@" < "$file" > "$tmp"
    mv "$tmp" "$file"
}

# ---------------------------------------------------------------- preflight --

preflight() {
    step "Preflight"

    [ -f "$BUILD_SCRIPT" ] || die "missing $BUILD_SCRIPT"
    [ -x "$BUILD_SCRIPT" ] || die "$BUILD_SCRIPT is not executable (git update-index --chmod=+x $BUILD_SCRIPT)"

    git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository"

    local branch
    branch="$(git rev-parse --abbrev-ref HEAD)"
    [ "$branch" = "$RELEASE_BRANCH" ] \
        || die "releases are cut from $RELEASE_BRANCH, but HEAD is on '$branch'"

    git diff --quiet && git diff --cached --quiet \
        || die "working tree has uncommitted changes; commit or stash them first"

    if [ -n "$(git ls-files --others --exclude-standard)" ]; then
        die "working tree has untracked files; commit, ignore, or remove them first"
    fi

    git fetch --quiet origin "$RELEASE_BRANCH" \
        || die "could not fetch origin/$RELEASE_BRANCH"
    local local_head remote_head
    local_head="$(git rev-parse HEAD)"
    remote_head="$(git rev-parse "origin/$RELEASE_BRANCH")"
    [ "$local_head" = "$remote_head" ] \
        || die "HEAD ($(git rev-parse --short HEAD)) differs from origin/$RELEASE_BRANCH ($(git rev-parse --short origin/"$RELEASE_BRANCH")); pull or push first"

    git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
        && die "tag $TAG already exists locally"
    git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1 \
        && die "tag $TAG already exists on origin"

    if [ "$PUBLISH" -eq 1 ]; then
        command -v gh >/dev/null 2>&1 || die "gh is required for --publish"
        gh auth status >/dev/null 2>&1 || die "gh is not authenticated (gh auth login)"
        # Checked here rather than discovered at the commit, which happens after a
        # full build and test run: a missing identity should cost a second, not
        # twenty minutes.
        git config user.email >/dev/null 2>&1 \
            || die "git has no author identity; set user.email and user.name"
        git config user.name >/dev/null 2>&1 \
            || die "git has no author identity; set user.email and user.name"
    fi

    grep -q '^## \[Unreleased\]' CHANGELOG.md \
        || die "CHANGELOG.md has no '## [Unreleased]' section to release"

    echo "  branch      $branch @ $(git rev-parse --short HEAD)"
    echo "  version     $VERSION  (tag $TAG)"
    echo "  mode        $([ "$PUBLISH" -eq 1 ] && echo 'publish' || echo 'dry run -- nothing will be pushed')"
}

# The build reads a SHA256 out of third_party/VF64-metal at configure time, so a
# checkout without submodules fails in CMake rather than at compile time.
init_submodules() {
    step "Submodules"
    git submodule update --init --recursive
    local probe="third_party/VF64-metal/Sources/VF64Metal/Shaders/Interop/VF64Support.metal"
    [ -f "$probe" ] || die "submodule checkout incomplete: $probe is missing"
    echo "  ok"
}

# ------------------------------------------------------------------ version --

bump_version() {
    step "Version -> $VERSION"

    local major minor patch rest
    major="${VERSION%%.*}"
    rest="${VERSION#*.}"
    minor="${rest%%.*}"
    patch="${rest#*.}"

    grep -qE '^project\(cumetal VERSION [0-9]+\.[0-9]+\.[0-9]+ ' CMakeLists.txt \
        || die "could not find the project(cumetal VERSION ...) line in CMakeLists.txt"

    rewrite CMakeLists.txt sed -E \
        "s/^project\(cumetal VERSION [0-9]+\.[0-9]+\.[0-9]+ /project(cumetal VERSION ${VERSION} /"

    grep -qE "^project\(cumetal VERSION ${VERSION} " CMakeLists.txt \
        || die "version rewrite did not take effect in CMakeLists.txt"

    # The version is duplicated in the public header. tests/unit/run_version_matches_test.sh
    # is a release gate that fails on any drift between the two (and against
    # cumetalc --version, which CMake bakes in at compile time), so both move together.
    [ -f "$NATIVE_HEADER" ] || die "missing $NATIVE_HEADER"
    rewrite "$NATIVE_HEADER" sed -E \
        -e "s/^(#define CUMETAL_VERSION_MAJOR[[:space:]]+)[0-9]+/\\1${major}/" \
        -e "s/^(#define CUMETAL_VERSION_MINOR[[:space:]]+)[0-9]+/\\1${minor}/" \
        -e "s/^(#define CUMETAL_VERSION_PATCH[[:space:]]+)[0-9]+/\\1${patch}/" \
        -e "s/^(#define CUMETAL_VERSION_STRING[[:space:]]+)\"[^\"]*\"/\\1\"${VERSION}\"/"

    local h_major h_minor h_patch h_string
    h_major="$(sed -n 's/^#define CUMETAL_VERSION_MAJOR[[:space:]][[:space:]]*\([0-9][0-9]*\).*/\1/p' "$NATIVE_HEADER" | head -1)"
    h_minor="$(sed -n 's/^#define CUMETAL_VERSION_MINOR[[:space:]][[:space:]]*\([0-9][0-9]*\).*/\1/p' "$NATIVE_HEADER" | head -1)"
    h_patch="$(sed -n 's/^#define CUMETAL_VERSION_PATCH[[:space:]][[:space:]]*\([0-9][0-9]*\).*/\1/p' "$NATIVE_HEADER" | head -1)"
    h_string="$(sed -n 's/^#define CUMETAL_VERSION_STRING[[:space:]][[:space:]]*"\([^"]*\)".*/\1/p' "$NATIVE_HEADER" | head -1)"

    [ "${h_major}.${h_minor}.${h_patch}" = "$VERSION" ] \
        || die "header macros are ${h_major}.${h_minor}.${h_patch} after rewrite, expected $VERSION"
    [ "$h_string" = "$VERSION" ] \
        || die "CUMETAL_VERSION_STRING is \"$h_string\" after rewrite, expected $VERSION"

    BUMPED=1
    echo "  CMakeLists.txt:    $(grep -E '^project\(cumetal VERSION' CMakeLists.txt)"
    echo "  cumetal_native.h:  ${h_major}.${h_minor}.${h_patch} / \"${h_string}\""
}

roll_changelog() {
    step "CHANGELOG"

    if grep -q "^## \[${VERSION}\]" CHANGELOG.md; then
        die "CHANGELOG.md already has a [${VERSION}] section"
    fi

    # Everything currently under [Unreleased] becomes this release's section.
    local body
    body="$(awk '
        /^## \[Unreleased\]/ { f = 1; next }
        f && /^## \[/        { exit }
        f                    { print }
    ' CHANGELOG.md | sed '/^[[:space:]]*$/d')"
    [ -n "$body" ] || die "the [Unreleased] section is empty -- nothing to release"

    local date
    date="$(date -u +%Y-%m-%d)"
    rewrite CHANGELOG.md awk -v header="## [${VERSION}] - ${date}" '
        { print }
        $0 == "## [Unreleased]" && !done { print ""; print header; done = 1 }
    '

    grep -q "^## \[${VERSION}\] - ${date}\$" CHANGELOG.md \
        || die "changelog rewrite did not take effect"
    echo "  opened ## [${VERSION}] - ${date}"
}

release_notes() {
    awk -v want="## [${VERSION}]" '
        index($0, want) == 1 { f = 1; next }
        f && /^## \[/        { exit }
        f                    { print }
    ' CHANGELOG.md
}

# --------------------------------------------------------------- dispatch ----
#
# The only environment question that matters is "can this machine build Metal
# code, and if not, can it reach one that can". That is a capability question,
# not a container question: probing for Docker vs OrbStack vs Podman would be
# both more fragile and less informative than asking these two directly.

host_shares_this_path() {
    # mac-run execs the script at the SAME absolute path on the Mac, so the repo
    # has to be a share of the host filesystem rather than a copy inside the
    # container image.
    [ -r /proc/mounts ] || return 1
    awk -v root="$REPO_ROOT" '
        $3 == "virtiofs" || $3 == "9p" || $3 == "nfs" || $3 ~ /^fuse/ {
            mp = $2
            if (mp == root) { found = 1; exit }
            if (substr(root, 1, length(mp) + 1) == mp "/") { found = 1; exit }
        }
        END { exit(found ? 0 : 1) }
    ' /proc/mounts
}

run_macos_build() {
    if [ "$(uname -s)" = "Darwin" ]; then
        step "Building here (macOS)"
        "./$BUILD_SCRIPT" "$VERSION"
        return
    fi

    step "Building on the host (this machine cannot run Metal)"

    command -v mac-run >/dev/null 2>&1 || die \
"this machine is $(uname -s), not macOS, and no host bridge (mac-run) is on PATH.

CuMetal links Metal, MetalPerformanceShaders and Accelerate and compiles
Objective-C++, so a release can only be built on Apple Silicon. Run

    ./deploy.sh $VERSION $([ "$PUBLISH" -eq 1 ] && echo '--publish')

on the Mac itself, or from a container that shares this repository at the same
path and provides the mac-run bridge."

    host_shares_this_path || die \
"mac-run is available, but $REPO_ROOT does not look like a share of the host
filesystem. The bridge executes the build script at the same absolute path on
the Mac, so a repository that only exists inside this container cannot be built.
Re-run from a checkout mounted at its real host path."

    mac-run --timeout "$MAC_RUN_TIMEOUT" "$BUILD_SCRIPT" "$VERSION"
}

# ---------------------------------------------------------------- publish ----

publish() {
    step "Publishing $TAG"

    # shellcheck disable=SC2086
    git add $VERSIONED_FILES
    git commit -m "release: $VERSION"
    git tag -a "$TAG" -m "CuMetal $VERSION"

    git push origin "$RELEASE_BRANCH"
    git push origin "$TAG"

    local notes
    notes="$(mktemp)"
    release_notes > "$notes"
    gh release create "$TAG" \
        --title "CuMetal $VERSION" \
        --notes-file "$notes" \
        "$ARTIFACT" "${ARTIFACT}.sha256"
    rm -f "$notes"

    echo "  https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/$TAG"
}

# ------------------------------------------------------------------- main ----

preflight
init_submodules
bump_version
roll_changelog
run_macos_build

[ -f "$ARTIFACT" ] || die "the build reported success but $ARTIFACT is missing"
[ -f "${ARTIFACT}.sha256" ] || die "the build reported success but ${ARTIFACT}.sha256 is missing"

step "Artifact"
ls -lh "$ARTIFACT"
cat "${ARTIFACT}.sha256"

if [ "$PUBLISH" -eq 1 ]; then
    publish
    PUBLISHED=1
    step "Released $TAG"
else
    step "Dry run complete -- nothing was pushed"
    cat <<EOF

  Built and packaged $VERSION. CMakeLists.txt and CHANGELOG.md are modified
  but uncommitted, and no tag exists.

  To publish:   ./deploy.sh $VERSION --publish
  To back out:  git checkout -- $VERSIONED_FILES && rm -rf dist

EOF
fi
