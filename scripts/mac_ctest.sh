#!/usr/bin/env bash
#
# mac_ctest.sh -- build one or more targets and run a ctest selection, on macOS.
#
#   scripts/mac_ctest.sh <ctest-regex> [target ...]
#
# Everything in this repository that touches Metal has to run on the Mac, and a
# box reaches it through mac-run, which execs a file in the repo rather than a
# command string. That is what this script is for: the smallest host-side step
# that builds a target and runs its test, so an investigation does not need a
# full release build to get an answer.
#
# CUMETAL_CTEST_REPEAT=N re-runs the selection N times. Race hunting needs that:
# one clean pass of a concurrency test proves nothing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

[ $# -ge 1 ] || { echo "usage: $0 <ctest-regex> [target ...]" >&2; exit 2; }
SELECTION="$1"; shift

BUILD_DIR="build-release"
cmake -S . -B "$BUILD_DIR" >/dev/null

if [ $# -gt 0 ]; then
    # The build's exit status is load-bearing and used to be discarded. Piping
    # into grep put grep's status at the end of the pipe, and `|| true` swallowed
    # even that -- so a target that failed to compile left the PREVIOUS binary in
    # place and ctest happily ran it. That produced green runs for tests whose
    # source did not compile, three separate times, which is worse than a red run
    # because it is reported as evidence. Capture the status, then filter.
    BUILD_LOG="$(mktemp)"
    if ! cmake --build "$BUILD_DIR" --target "$@" -j8 > "$BUILD_LOG" 2>&1; then
        echo "mac_ctest: BUILD FAILED -- not running tests against stale binaries" >&2
        grep -E 'error:' "$BUILD_LOG" | head -20 >&2
        rm -f "$BUILD_LOG"
        exit 1
    fi
    grep -E 'warning: unused|Linking|Built target' "$BUILD_LOG" || true
    rm -f "$BUILD_LOG"
fi

REPEAT="${CUMETAL_CTEST_REPEAT:-1}"
for attempt in $(seq 1 "$REPEAT"); do
    [ "$REPEAT" -gt 1 ] && printf '\n----- attempt %d/%d -----\n' "$attempt" "$REPEAT"
    # CUMETAL_CTEST_ARGS passes extra ctest flags (label filters, timeouts).
    # shellcheck disable=SC2086
    ctest --test-dir "$BUILD_DIR" -R "$SELECTION" ${CUMETAL_CTEST_ARGS:-} --output-on-failure || exit $?
done
