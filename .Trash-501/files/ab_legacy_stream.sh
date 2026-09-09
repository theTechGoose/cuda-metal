#!/usr/bin/env bash
# A/B the multi-worker stress test between the current runtime and 0.6.0's, to
# tell a regression apart from a pre-existing bug. Restores the tree either way.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
RUNS="${RUNS:-8}"
SAVE=$(mktemp)
cp runtime/rt/cuda_runtime.cpp "$SAVE"
restore() { cp "$SAVE" runtime/rt/cuda_runtime.cpp; rm -f "$SAVE"; }
trap restore EXIT

run_n() {
    local label="$1" fails=0
    cmake --build build-release --target cumetal_multi_worker_stream_stress_test -j8 >/dev/null 2>&1 \
        || { echo "$label: BUILD FAILED"; return; }
    for i in $(seq 1 "$RUNS"); do
        ctest --test-dir build-release -R functional_multi_worker_stream_stress >/dev/null 2>&1 || fails=$((fails+1))
    done
    echo "$label: $fails/$RUNS runs failed"
}

echo "=== current (v0.6.1 + fixes) ==="
run_n "current"
echo "=== 0.6.0 runtime (c210af3) ==="
git show c210af3:runtime/rt/cuda_runtime.cpp > runtime/rt/cuda_runtime.cpp
run_n "0.6.0  "
