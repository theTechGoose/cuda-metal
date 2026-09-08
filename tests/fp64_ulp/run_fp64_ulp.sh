#!/usr/bin/env bash
# Compile fp64_ulp.cu under one --fp64 policy and assert its worst-case drift
# from hardware binary64 stays inside a budget.
#
# The ieee64 budget is 0 on purpose: that mode claims correctly rounded binary64,
# and a claim of exactness is either true or it is not. The approximate modes get
# generous budgets -- they exist to catch a gross regression (a lost limb, a
# broken normalisation), not to pin their exact error.
set -euo pipefail

CUMETALC="${1:?usage: run_fp64_ulp.sh <cumetalc> <source.cu> <workdir> <mode> <ulp-budget>}"
SOURCE_CU="${2:?}"
WORK_DIR="${3:?}"
MODE="${4:?}"
BUDGET="${5:?}"

if ! command -v xcrun >/dev/null 2>&1; then
    echo "SKIP: xcrun not installed"
    exit 77
fi

mkdir -p "${WORK_DIR}"
OUT_BIN="${WORK_DIR}/fp64_ulp_${MODE}"
BUILD_LOG="${WORK_DIR}/fp64_ulp_${MODE}.build.log"

if ! "${CUMETALC}" "${SOURCE_CU}" "--fp64=${MODE}" -o "${OUT_BIN}" >"${BUILD_LOG}" 2>&1; then
    cat "${BUILD_LOG}"
    echo "FAIL: cumetalc could not build the ULP probe in ${MODE} mode"
    exit 1
fi

"${OUT_BIN}" "${BUDGET}"
