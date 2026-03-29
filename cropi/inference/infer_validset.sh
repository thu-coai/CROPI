#!/usr/bin/env bash
set -euo pipefail

PROMPT_TYPE=${1:?prompt type is required}
MODEL_NAME_OR_PATH=${2:?model path is required}
TEMP=${3:?temperature is required}
N_SAMPLE=${4:?n_sample is required}
SEED=${5:-0}
MAX_TOKENS=${6:-2048}

ENTRYPOINT=${MATH_EVAL_ENTRYPOINT:-math_eval_save_logprob.py}
if [[ ! -f "${ENTRYPOINT}" && ! -x "${ENTRYPOINT}" ]]; then
    echo "[ERROR] ${ENTRYPOINT} was not found. Set MATH_EVAL_ENTRYPOINT to your evaluation script."
    exit 1
fi

OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval
SPLIT=${SPLIT:-valid}
NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:--1}

run_eval() {
    local data_name=$1
    TOKENIZERS_PARALLELISM=false \
    python3 -u "${ENTRYPOINT}" \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --data_name "${data_name}" \
        --output_dir "${OUTPUT_DIR}" \
        --split "${SPLIT}" \
        --prompt_type "${PROMPT_TYPE}" \
        --num_test_sample "${NUM_TEST_SAMPLE}" \
        --seed "${SEED}" \
        --temperature "${TEMP}" \
        --n_sampling "${N_SAMPLE}" \
        --max_tokens_per_call "${MAX_TOKENS}" \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --save_log_probs \
        --overwrite
}

run_eval "gsm8k,gaokao2023en,olympiadbench"
run_eval "math"
run_eval "aime24,amc23"
