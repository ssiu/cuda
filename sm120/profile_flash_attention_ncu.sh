#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-ncu_flash_attention}"
NCU_SET="${NCU_SET:-full}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-.*(cutlass|cudnn).*}"
NCU_PM_SAMPLING="${NCU_PM_SAMPLING:-1}"
NCU_PM_SAMPLING_SECTION="${NCU_PM_SAMPLING_SECTION:-PmSampling}"

BATCH_SIZE="${BATCH_SIZE:-4}"
SEQLEN_Q="${SEQLEN_Q:-8192}"
SEQLEN_K="${SEQLEN_K:-8192}"
NUM_HEAD="${NUM_HEAD:-16}"
HEAD_DIM="${HEAD_DIM:-128}"
DTYPE="${DTYPE:-BFloat16}"
SOFTMAX_SCALE="${SOFTMAX_SCALE:-0.5}"
M_BLOCK_SIZE="${M_BLOCK_SIZE:-128}"
N_BLOCK_SIZE="${N_BLOCK_SIZE:-64}"
NUM_THREADS="${NUM_THREADS:-128}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-3}"
ITERATIONS="${ITERATIONS:-10}"
CAUSAL_FLAG="${CAUSAL_FLAG:-}"

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --dtype "${DTYPE}"
  --batch_size "${BATCH_SIZE}"
  --seqlen_q "${SEQLEN_Q}"
  --seqlen_k "${SEQLEN_K}"
  --num_head "${NUM_HEAD}"
  --head_dim "${HEAD_DIM}"
  --softmax_scale "${SOFTMAX_SCALE}"
  --m_block_size "${M_BLOCK_SIZE}"
  --n_block_size "${N_BLOCK_SIZE}"
  --num_threads "${NUM_THREADS}"
  --warmup_iterations "${WARMUP_ITERATIONS}"
  --iterations "${ITERATIONS}"
)

if [[ -n "${CAUSAL_FLAG}" ]]; then
  COMMON_ARGS+=(--is_causal)
fi

BASE_NAME="b${BATCH_SIZE}_sq${SEQLEN_Q}_sk${SEQLEN_K}_h${NUM_HEAD}_d${HEAD_DIM}_${DTYPE}"

NCU_ARGS=(
  -f
  --target-processes all
  --set "${NCU_SET}"
  --import-source on
  --kernel-name "::regex:${NCU_KERNEL_REGEX}"
)

if [[ "${NCU_PM_SAMPLING}" != "0" ]]; then
  NCU_ARGS+=(--section "${NCU_PM_SAMPLING_SECTION}")
fi

ncu "${NCU_ARGS[@]}" \
  -o "${OUT_DIR}/fa2_vs_cudnn_${BASE_NAME}" \
  "${PYTHON_BIN}" compare_flash_attention.py "${COMMON_ARGS[@]}"
