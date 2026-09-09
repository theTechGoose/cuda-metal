#!/usr/bin/env bash
# Bisect the legacy-stream failure inside 695b33b by neutralising one mechanism
# at a time and re-running the stress test. Restores the tree on exit.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
RUNS="${RUNS:-10}"
SAVE=$(mktemp); cp runtime/rt/cuda_runtime.cpp "$SAVE"
trap 'cp "$SAVE" runtime/rt/cuda_runtime.cpp; rm -f "$SAVE"' EXIT

run_n() {
    local label="$1" fails=0
    cmake --build build-release --target cumetal_multi_worker_stream_stress_test -j8 >/dev/null 2>&1 \
        || { echo "$label: BUILD FAILED"; return; }
    for i in $(seq 1 "$RUNS"); do
        ctest --test-dir build-release -R functional_multi_worker_stream_stress >/dev/null 2>&1 || fails=$((fails+1))
    done
    echo "$label: $fails/$RUNS failed"
}

echo "--- as committed ---"; run_n "baseline        "

echo "--- restore book disabled (note_dtoh_blit is a no-op) ---"
cp "$SAVE" runtime/rt/cuda_runtime.cpp
python3 - <<'PY'
p='runtime/rt/cuda_runtime.cpp'; s=open(p).read()
s=s.replace("void note_dtoh_blit(const void* stream_key, void* host_dst, std::size_t count) {\n    if (host_dst == nullptr || count < sizeof(std::uintptr_t)) return;",
            "void note_dtoh_blit(const void* stream_key, void* host_dst, std::size_t count) {\n    return;  // ISOLATION\n    if (host_dst == nullptr || count < sizeof(std::uintptr_t)) return;",1)
open(p,'w').write(s)
PY
run_n "no restore book "

echo "--- async memsets reverted to the immediate host fill (driver path) ---"
cp "$SAVE" runtime/rt/cuda_runtime.cpp
run_n "baseline again  "
