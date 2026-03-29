#!/usr/bin/env bash
set -euo pipefail

PROMPT_TYPE=${1:?prompt type is required}
MODEL_NAME_OR_PATH=${2:?model path is required}
TEMP=${3:?temperature is required}
N_SAMPLE=${4:?n_sample is required}
DATA_NAMES_TRAIN=${5:?comma-separated dataset names are required}
MAX_TOKENS=${6:-2048}

ENTRYPOINT=${MATH_EVAL_ENTRYPOINT:-math_eval_save_logprob.py}
if [[ ! -f "${ENTRYPOINT}" && ! -x "${ENTRYPOINT}" ]]; then
    echo "[ERROR] ${ENTRYPOINT} was not found. Set MATH_EVAL_ENTRYPOINT to your evaluation script."
    exit 1
fi

OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval
SPLIT=${SPLIT:-sample_200}
N_SPLIT=${N_SPLIT:-10}
NUM_TEST_SAMPLE=${NUM_TEST_SAMPLE:--1}

for DATA_NAME in ${DATA_NAMES_TRAIN//,/ }; do
    for i in $(seq 0 $((N_SPLIT - 1))); do
        SUB_SPLIT="${SPLIT}_${i}"
        echo "[INFO] Processing ${DATA_NAME} ${SUB_SPLIT}"
        TOKENIZERS_PARALLELISM=false \
        python3 -u "${ENTRYPOINT}" \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --data_name "${DATA_NAME}" \
            --output_dir "${OUTPUT_DIR}" \
            --split "${SUB_SPLIT}" \
            --prompt_type "${PROMPT_TYPE}" \
            --num_test_sample "${NUM_TEST_SAMPLE}" \
            --seed 0 \
            --temperature "${TEMP}" \
            --n_sampling "${N_SAMPLE}" \
            --max_tokens_per_call "${MAX_TOKENS}" \
            --top_p 1 \
            --start 0 \
            --end -1 \
            --use_vllm \
            --save_log_probs \
            --save_outputs \
            --overwrite
    done
done
