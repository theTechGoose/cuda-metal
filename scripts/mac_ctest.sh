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
    cmake --build "$BUILD_DIR" --target "$@" -j8 2>&1 | grep -E 'error:|warning: unused|Linking|Built target' || true
fi

REPEAT="${CUMETAL_CTEST_REPEAT:-1}"
for attempt in $(seq 1 "$REPEAT"); do
    [ "$REPEAT" -gt 1 ] && printf '\n----- attempt %d/%d -----\n' "$attempt" "$REPEAT"
    ctest --test-dir "$BUILD_DIR" -R "$SELECTION" --output-on-failure || exit $?
done
