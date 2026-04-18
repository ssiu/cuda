#!/usr/bin/env bash
set -euo pipefail



M="${1:-128}"
N="${2:-128}"
K="${3:-128}"
KERNEL_REGEX='^(Kernel2.*|splitKreduce.*|.*cutlass.*)'

OUT_NAME="sm120_m${M}_n${N}_k${K}"

ncu -f --target-processes all --set full \
    --import-source on \
    --kernel-name ::regex:"${KERNEL_REGEX}" \
    -o "${OUT_NAME}" \
    python3 sm120/profile_compare.py --m "${M}" --n "${N}" --k "${K}" --warmup 0 --iters 0

# ncu -f --target-processes all --set full --import-source on -o sm120_cute_test python sm120/compare.py --mnkl 2048,2048,2048,1 --tile_shape_mnk 128,128,64