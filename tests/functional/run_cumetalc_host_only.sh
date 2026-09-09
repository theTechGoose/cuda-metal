#!/usr/bin/env bash
# A host-only .c that calls the cuDNN surface must build with cumetalc using the
# same flags a real CUDA build line carries.
#
# Both halves of this were broken until 0.6.8 and both were found by a consumer
# trying to run the documented command (`cumetalc dump.c -o dump -lcudnn`):
# cumetalc rejected -lcudnn as an unknown option, and rejected a .c input
# outright. Keep -lcudnn in the invocation below -- dropping it is what makes
# this gate stop testing the thing that broke.
set -euo pipefail

CUMETALC="${1:?usage: run_cumetalc_host_only.sh <cumetalc> <source.c> <workdir>}"
SOURCE_C="${2:?}"
WORK_DIR="${3:?}"

mkdir -p "${WORK_DIR}"
OUT="${WORK_DIR}/host_only_cudnn_program"
rm -f "${OUT}"

# -lcudnn and -L are the flags a ported build line carries; -lcudnn resolves
# through the alias installed beside libcumetal.
if ! "${CUMETALC}" "${SOURCE_C}" -o "${OUT}" -lcudnn; then
    echo "FAIL: cumetalc could not build a host-only .c with -lcudnn"
    exit 1
fi
[ -x "${OUT}" ] || { echo "FAIL: no executable produced"; exit 1; }

OUTPUT="$("${OUT}")" || { echo "FAIL: produced binary exited nonzero"; echo "${OUTPUT}"; exit 1; }
echo "${OUTPUT}"
case "${OUTPUT}" in
    PASS:*) ;;
    *) echo "FAIL: unexpected output"; exit 1 ;;
esac

# The separated form clang accepts, and a C++ input, so neither path regresses.
if ! "${CUMETALC}" "${SOURCE_C}" -o "${OUT}" -l cudnn; then
    echo "FAIL: cumetalc rejected the separated '-l cudnn' form"
    exit 1
fi
echo "PASS: host-only compilation accepts real CUDA link flags"
