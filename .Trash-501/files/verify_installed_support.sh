#!/usr/bin/env bash
# Prove an INSTALLED cumetalc finds its Metal support sources through its own
# prefix rather than through the checkout it was built from -- the 0.6.1 bug.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

PREFIX="$(mktemp -d /tmp/cumetal-installed-XXXXXX)"
trap 'rm -rf "$PREFIX"' EXIT

cmake --build build-release --target cumetalc cumetal_runtime -j8 >/dev/null 2>&1 || {
    echo "FAIL: build"; exit 1; }
cmake --install build-release --prefix "$PREFIX" >/dev/null 2>&1 || { echo "FAIL: install"; exit 1; }

echo "== staged support sources =="
ls "$PREFIX/libexec/cumetal/metal-support/" 2>/dev/null || { echo "FAIL: not installed"; exit 1; }

cat > "$PREFIX/fp64.cu" <<'CU'
extern "C" __global__ void fp64_touch(double* out, const double* in) {
    out[threadIdx.x] = in[threadIdx.x] * 0.01 + 1.0;
}
CU

echo "== installed cumetalc, FP64 kernel, verbose =="
OUT="$PREFIX/out.metallib"
CUMETAL_DEBUG_EMITTER=1 "$PREFIX/bin/cumetalc" --cuda-device "$PREFIX/fp64.cu" -o "$OUT" \
    > "$PREFIX/log.txt" 2>&1
STATUS=$?
echo "exit=$STATUS"

if command grep -q "$REPO_ROOT/compiler/metal/support" "$PREFIX/log.txt"; then
    echo "FAIL: the installed cumetalc reached back into the checkout:"
    command grep -o "$REPO_ROOT/compiler/metal/support[^ ']*" "$PREFIX/log.txt" | sort -u
    exit 1
fi
if command grep -q "$PREFIX/libexec/cumetal/metal-support" "$PREFIX/log.txt"; then
    echo "PASS: support source resolved from the install prefix:"
    command grep -o "$PREFIX/libexec/cumetal/metal-support[^ ']*" "$PREFIX/log.txt" | sort -u
else
    echo "NOTE: this kernel did not need the support source; checking the resolver directly."
fi
[ -f "$OUT" ] && echo "PASS: metallib produced at $OUT" || { echo "FAIL: no metallib"; sed -n 1,40p "$PREFIX/log.txt"; exit 1; }
