#!/usr/bin/env bash
# Pin the codegen that depends on the detected AIR/MSL dialect.
#
# The dialect is normally probed from the installed toolchain, so on any one
# machine only one branch of that codegen ever runs -- the other is reasoning,
# not evidence, which is exactly how the AIR 2.8 pinning shipped broken for
# Xcode 15/16 in the first place. CUMETAL_AIR_DIALECT forces each dialect so
# both branches are exercised wherever this suite runs.
set -euo pipefail

PTX2LLVM="${1:?usage: run_dialect_codegen.sh <cumetal-ptx2llvm> <workdir>}"
WORK_DIR="${2:?}"
# Re-runnable: the tool refuses to clobber an existing output, so a stale
# workdir made this pass once and fail every time after.
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

PTX="${WORK_DIR}/probe.ptx"
cat > "${PTX}" <<'PTXEOF'
.version 7.0
.target sm_70
.address_size 64
.visible .entry dialect_probe(.param .u64 o)
{
    .reg .b64 %rd<3>;
    .reg .f32 %f<3>;
    ld.param.u64 %rd1, [o];
    mov.f32 %f1, 1.0;
    st.global.f32 [%rd1], %f1;
    ret;
}
PTXEOF

fail=0
check() {
    local dialect="$1" want_triple="$2" want_air="$3" want_lang="$4"
    local out="${WORK_DIR}/${dialect//\//_}.ll"
    if ! CUMETAL_AIR_DIALECT="${dialect}" "${PTX2LLVM}" "${PTX}" --overwrite -o "${out}" \
            >"${out}.stdout" 2>"${out}.stderr"; then
        echo "FAIL: lowering failed under CUMETAL_AIR_DIALECT=${dialect}"
        sed 's/^/      /' "${out}.stderr" | head -5
        sed 's/^/      /' "${out}.stdout" | head -5
        fail=1
        return
    fi
    if ! grep -q "target triple = \"${want_triple}\"" "${out}"; then
        echo "FAIL: ${dialect}: expected triple ${want_triple}, got: $(grep -m1 'target triple' "${out}")"
        fail=1
    fi
    if ! grep -q "\"air.version\"=\"${want_air}\"" "${out}"; then
        echo "FAIL: ${dialect}: expected air.version ${want_air}"
        fail=1
    fi
    if ! grep -qE "= !\{!\"Metal\", i32 ${want_lang%.*}, i32 ${want_lang#*.}, i32 0\}" "${out}"; then
        echo "FAIL: ${dialect}: expected Metal language version ${want_lang}"
        fail=1
    fi
}

# The Xcode 26 dialect, which no Xcode 15/16 machine would otherwise emit.
check "2.8/4.0" "air64_v28-apple-macosx26.0.0" "2.8" "4.0"
# The Xcode 15/16 dialect.
check "2.7/3.2" "air64_v27-apple-macosx15.0.0" "2.7" "3.2"

# A malformed override must be refused, not obeyed: silently honouring garbage
# would emit a dialect nothing accepts.
if CUMETAL_AIR_DIALECT="nonsense" "${PTX2LLVM}" "${PTX}" --overwrite -o "${WORK_DIR}/bad.ll" 2>"${WORK_DIR}/bad.err" >/dev/null; then
    if ! grep -q "ignoring malformed CUMETAL_AIR_DIALECT" "${WORK_DIR}/bad.err"; then
        echo "FAIL: malformed CUMETAL_AIR_DIALECT was accepted silently"
        fail=1
    fi
else
    echo "FAIL: a malformed dialect override should warn and continue, not abort"
    fail=1
fi

if [[ ${fail} -ne 0 ]]; then
    exit 1
fi
echo "PASS: AIR dialect codegen pinned for both 2.7/3.2 and 2.8/4.0"
