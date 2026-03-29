#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

DATA_ROOT=${1:-"${REPO_ROOT}/data"}
MODEL_NAME=${2:-"Qwen2.5-1.5B-Instruct"}
PROJ_NOTE=${3:-"trak_norm_seed0_mid0_projdim32768"}
TRAIN_DATA_NAMES=${4:-"gsm8k,math"}
VALID_DATA_NAMES=${5:-"gsm8k,math"}
PROMPT_TYPE=${6:-"qwen25-math-cot"}
TEMPERATURE=${7:-"0.5"}
N_SAMPLES=${8:-"8"}
N_SAMPLES_VAL=${9:-"32"}
SEED=${10:-"0"}
NUM_PARALLEL=${11:-"8"}

cd "${REPO_ROOT}"

echo "[INFO] Computing CROPI influence scores"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] model_name=${MODEL_NAME}"
echo "[INFO] proj_note=${PROJ_NOTE}"
echo "[INFO] train_data_names=${TRAIN_DATA_NAMES}"
echo "[INFO] valid_data_names=${VALID_DATA_NAMES}"

uv run cropi-compute-inf-score \
  --data_root "${DATA_ROOT}" \
  --model_name "${MODEL_NAME}" \
  --proj_note "${PROJ_NOTE}" \
  --train_data_names "${TRAIN_DATA_NAMES}" \
  --valid_data_names "${VALID_DATA_NAMES}" \
  --prompt_type "${PROMPT_TYPE}" \
  --temperature "${TEMPERATURE}" \
  --n_samples "${N_SAMPLES}" \
  --n_samples_val "${N_SAMPLES_VAL}" \
  --seed "${SEED}" \
  --num_parallel "${NUM_PARALLEL}"

echo "[INFO] Score file written under ${DATA_ROOT}/${MODEL_NAME}/"
