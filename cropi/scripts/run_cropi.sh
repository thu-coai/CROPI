#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  bash cropi/scripts/run_cropi.sh [mode] [data_root] [initial_model_name]

Modes:
  select-only   Compute CROPI scores and select data once.
  rl-only       Run one RL round from an existing selected parquet.
  full          Run iterative CROPI: select -> RL -> select -> RL ...

If mode is omitted, the script keeps backward-compatible `select-only` behavior.

Core args:
  data_root            Root data directory. Default: <repo>/data
  initial_model_name   Model directory name used by CROPI assets under data/<dataset>/<model_name>.

Important environment variables:
  BASE_MODEL_PATH      HF checkpoint used for the first RL round.
  RL_PYTHON            Python executable with `verl` installed.
  NUM_RL_ROUNDS        Number of RL rounds in `full` mode. Default: 2
  RL_NUM_GPUS          Number of GPUs for RL. Default: 8
  RL_TP_SIZE           vLLM tensor parallel size. Default: 2
  DRY_RUN=1            Print commands instead of executing them.

Example:
  BASE_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct \
  RL_PYTHON=/path/to/verl-env/bin/python \
  bash cropi/scripts/run_cropi.sh full ./data Qwen2.5-1.5B-Instruct_curriculum
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODE=${1:-select-only}
if [[ "${MODE}" != "select-only" && "${MODE}" != "rl-only" && "${MODE}" != "full" ]]; then
  MODE="select-only"
else
  shift
fi

DATA_ROOT=${1:-"${REPO_ROOT}/data"}
INITIAL_MODEL_NAME=${2:-"Qwen2.5-1.5B-Instruct_curriculum"}

TRAIN_DATA_NAMES=${TRAIN_DATA_NAMES:-"gsm_math_dsr_test"}
VALID_DATA_NAMES=${VALID_DATA_NAMES:-"gsm8k,math"}
RL_VAL_DATA_NAMES=${RL_VAL_DATA_NAMES:-"${VALID_DATA_NAMES}"}
SELECT_RATIO=${SELECT_RATIO:-"0.1"}
SCORE_METHOD=${SCORE_METHOD:-"inf_valid_uniform"}
PROMPT_TYPE=${PROMPT_TYPE:-"qwen25-math-cot"}
TEMPERATURE=${TEMPERATURE:-"0.5"}
N_SAMPLES=${N_SAMPLES:-"8"}
N_SAMPLES_VAL=${N_SAMPLES_VAL:-"32"}
SEED=${SEED:-"0"}
NUM_PARALLEL=${NUM_PARALLEL:-"8"}
NUM_RL_ROUNDS=${NUM_RL_ROUNDS:-"2"}

PROJECTION_METHOD=${PROJECTION_METHOD:-"trak_norm"}
PROJ_DIM=${PROJ_DIM:-"32768"}
MODEL_ID=${MODEL_ID:-"0"}
SPARSE_DIM=${SPARSE_DIM:-"15000000"}
PROJ_NOTE=${PROJ_NOTE:-"${PROJECTION_METHOD}_seed${SEED}_mid${MODEL_ID}_projdim${PROJ_DIM}_sparse${SPARSE_DIM}"}
INFER_NOTE=${INFER_NOTE:-"${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES}_s0_e-1"}
VALID_INFER_NOTE=${VALID_INFER_NOTE:-"${PROMPT_TYPE}_-1_seed${SEED}_t${TEMPERATURE}_n${N_SAMPLES_VAL}_s0_e-1"}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-""}
RL_PYTHON=${RL_PYTHON:-python3}
RL_WORKDIR=${RL_WORKDIR:-"${REPO_ROOT}"}
CKPT_ROOT=${CKPT_ROOT:-"${REPO_ROOT}/checkpoints"}
RL_PROJECT_NAME=${RL_PROJECT_NAME:-"cropi_rl"}
RL_NUM_GPUS=${RL_NUM_GPUS:-"8"}
RL_TP_SIZE=${RL_TP_SIZE:-"2"}
RL_GPU_MEMORY_UTILIZATION=${RL_GPU_MEMORY_UTILIZATION:-"0.6"}
RL_N_SAMPLES=${RL_N_SAMPLES:-"${N_SAMPLES}"}
RL_TRAIN_BATCH_SIZE=${RL_TRAIN_BATCH_SIZE:-"128"}
RL_PPO_MINI_BATCH_SIZE=${RL_PPO_MINI_BATCH_SIZE:-"128"}
RL_PPO_MICRO_BATCH_SIZE_PER_GPU=${RL_PPO_MICRO_BATCH_SIZE_PER_GPU:-"16"}
RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-"16"}
RL_MAX_PROMPT_LENGTH=${RL_MAX_PROMPT_LENGTH:-"2048"}
RL_MAX_RESPONSE_LENGTH=${RL_MAX_RESPONSE_LENGTH:-"2048"}
RL_TOTAL_TRAINING_STEPS=${RL_TOTAL_TRAINING_STEPS:-"10"}
RL_SAVE_FREQ=${RL_SAVE_FREQ:-"${RL_TOTAL_TRAINING_STEPS}"}
RL_TEST_FREQ=${RL_TEST_FREQ:-"10"}
RL_USE_WANDB=${RL_USE_WANDB:-"0"}
DRY_RUN=${DRY_RUN:-"0"}

cd "${REPO_ROOT}"

log() {
  echo "[INFO] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] $*"
    return 0
  fi
  eval "$@"
}

split_csv() {
  local csv=$1
  if [[ -z "${csv}" ]]; then
    return 0
  fi
  tr ',' '\n' <<<"${csv}" | sed '/^$/d'
}

join_as_hydra_list() {
  python - "$@" <<'PY'
import sys
items = [x for x in sys.argv[1:] if x]
print("[" + ",".join("'" + item.replace("'", "\\'") + "'" for item in items) + "]")
PY
}

model_size_suffix() {
  local model_name=$1
  local lower=${model_name,,}
  if [[ "${lower}" == *"1.5b"* ]]; then
    echo "1.5b"
  elif [[ "${lower}" == *"7b"* ]]; then
    echo "7b"
  else
    echo "model"
  fi
}

score_note() {
  echo "max_valid_inf_${PROJ_NOTE}_train_${TRAIN_DATA_NAMES}_valid_${VALID_DATA_NAMES}"
}

select_note() {
  local model_name=$1
  echo "$(score_note)_valid_uniform_ratio${SELECT_RATIO}_$(model_size_suffix "${model_name}")"
}

selected_train_path() {
  local train_name=$1
  local model_name=$2
  local iter_idx=$3
  echo "${DATA_ROOT}/${train_name}/${model_name}/train_qwen_selected_$(select_note "${model_name}")_iter${iter_idx}.parquet"
}

score_path_for_model() {
  local model_name=$1
  echo "${DATA_ROOT}/${model_name}/train_valid_score_$(score_note).json"
}

next_model_name() {
  local iter_idx=$1
  local next_idx=$((iter_idx + 1))
  local base_name
  base_name=$(basename "${BASE_MODEL_PATH}")
  echo "${base_name}_curriculum_${SCORE_METHOD}_step${RL_TOTAL_TRAINING_STEPS}_${next_idx}"
}

archive_selection_if_needed() {
  local train_name=$1
  local model_name=$2
  local iter_idx=$3
  local stable_path="${DATA_ROOT}/${train_name}/train_qwen_selected_$(select_note "${model_name}").parquet"
  local stable_stat="${stable_path%.parquet}_stat.json"
  local iter_path
  iter_path=$(selected_train_path "${train_name}" "${model_name}" "${iter_idx}")
  local iter_stat="${iter_path%.parquet}_stat.json"
  if [[ -f "${stable_path}" && ! -f "${iter_path}" ]]; then
    mkdir -p "$(dirname "${iter_path}")"
    run_cmd "cp '${stable_path}' '${iter_path}'"
  fi
  if [[ -f "${stable_stat}" && ! -f "${iter_stat}" ]]; then
    mkdir -p "$(dirname "${iter_stat}")"
    run_cmd "cp '${stable_stat}' '${iter_stat}'"
  fi
}

build_val_files_hydra() {
  python - "${DATA_ROOT}" "${RL_VAL_DATA_NAMES}" <<'PY'
import os
import sys

data_root = sys.argv[1]
names = [x for x in sys.argv[2].split(",") if x]
paths = []
for name in names:
    candidates = [
        os.path.join(data_root, name, "test_qwen_split_valid.parquet"),
        os.path.join(data_root, name, "valid_qwen.parquet"),
    ]
    for path in candidates:
        if os.path.exists(path):
            paths.append(path)
            break
    else:
        raise SystemExit(f"missing validation parquet for {name}: tried {candidates}")
print("[" + ",".join("'" + path.replace("'", "\\'") + "'" for path in paths) + "]")
PY
}

build_train_files_hydra() {
  local model_name=$1
  local iter_idx=$2
  local paths=()
  while IFS= read -r train_name; do
    [[ -n "${train_name}" ]] || continue
    paths+=("$(selected_train_path "${train_name}" "${model_name}" "${iter_idx}")")
  done < <(split_csv "${TRAIN_DATA_NAMES}")
  join_as_hydra_list "${paths[@]}"
}

run_score_and_select() {
  local model_name=$1
  local iter_idx=$2

  log "Scoring model directory ${model_name}"
  run_cmd "'${SCRIPT_DIR}/get_popi_score.sh' '${DATA_ROOT}' '${model_name}' '${PROJ_NOTE}' '${TRAIN_DATA_NAMES}' '${VALID_DATA_NAMES}' '${PROMPT_TYPE}' '${TEMPERATURE}' '${N_SAMPLES}' '${N_SAMPLES_VAL}' '${SEED}' '${NUM_PARALLEL}'"

  local score_path
  score_path=$(score_path_for_model "${model_name}")
  log "Selecting data for ${model_name} with score file ${score_path}"
  run_cmd "uv run cropi-select --data_root '${DATA_ROOT}' --score_method '${SCORE_METHOD}' --score_path '${score_path}' --select_ratio '${SELECT_RATIO}' --train_data_names '${TRAIN_DATA_NAMES}' --valid_data_names '${VALID_DATA_NAMES}' --model_name '${model_name}' --proj_note '${PROJ_NOTE}' --infer_note '${INFER_NOTE}' --num_parallel '${NUM_PARALLEL}' --i_iter '${iter_idx}'"

  while IFS= read -r train_name; do
    [[ -n "${train_name}" ]] || continue
    archive_selection_if_needed "${train_name}" "${model_name}" "${iter_idx}"
  done < <(split_csv "${TRAIN_DATA_NAMES}")
}

prepare_rollout_jsonls() {
  local src_model_name=$1
  local dst_model_name=$2

  log "Copying rollout JSONL files from ${src_model_name} -> ${dst_model_name}"
  while IFS= read -r train_name; do
    [[ -n "${train_name}" ]] || continue
    local src="${DATA_ROOT}/${train_name}/${src_model_name}/train_${INFER_NOTE}.jsonl"
    local dst_dir="${DATA_ROOT}/${train_name}/${dst_model_name}"
    [[ -f "${src}" ]] || die "Missing rollout JSONL: ${src}"
    run_cmd "mkdir -p '${dst_dir}'"
    run_cmd "cp '${src}' '${dst_dir}/'"
  done < <(split_csv "${TRAIN_DATA_NAMES}")

  while IFS= read -r valid_name; do
    [[ -n "${valid_name}" ]] || continue
    local src="${DATA_ROOT}/${valid_name}/${src_model_name}/valid_${VALID_INFER_NOTE}.jsonl"
    local dst_dir="${DATA_ROOT}/${valid_name}/${dst_model_name}"
    [[ -f "${src}" ]] || die "Missing rollout JSONL: ${src}"
    run_cmd "mkdir -p '${dst_dir}'"
    run_cmd "cp '${src}' '${dst_dir}/'"
  done < <(split_csv "${VALID_DATA_NAMES}")
}

compute_grad_for_split() {
  local model_path=$1
  local model_name=$2
  local dataset_name=$3
  local split_name=$4
  local infer_note=$5
  local num_generations=$6

  local src_jsonl="${DATA_ROOT}/${dataset_name}/${model_name}/${split_name}_${infer_note}.jsonl"
  if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -f "${src_jsonl}" ]] || die "Missing rollout file for gradient computation: ${src_jsonl}"
  fi

  run_cmd "python '${REPO_ROOT}/cropi/utils/split_files.py' --input '${src_jsonl}' --split_num '${NUM_PARALLEL}'"

  local rank=0
  while [[ ${rank} -lt ${NUM_PARALLEL} ]]; do
    local shard="${src_jsonl}.${rank}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      [[ -f "${shard}" ]] || die "Missing shard ${shard}"
    fi
    local gpu=$((rank % NUM_PARALLEL))
    run_cmd "CUDA_VISIBLE_DEVICES=${gpu} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run cropi-get-grad --model_name_or_path '${model_path}' --base_model '${BASE_MODEL_PATH}' --rollout_data_path '${shard}' --projection_method '${PROJECTION_METHOD}' --proj_dim '${PROJ_DIM}' --model_id '${MODEL_ID}' --seed '${SEED}' --num_generations '${num_generations}' --max_completion_length '${RL_MAX_RESPONSE_LENGTH}' --beta 0.001 --epsilon 0.2 --cancel_ppo_clip --sparse_dim '${SPARSE_DIM}' > '${src_jsonl}.grad.rank${rank}.log' 2>&1 &"
    rank=$((rank + 1))
  done
  run_cmd "wait"
}

compute_gradients_for_model() {
  local model_path=$1
  local model_name=$2

  log "Recomputing gradients for ${model_name}"
  while IFS= read -r train_name; do
    [[ -n "${train_name}" ]] || continue
    compute_grad_for_split "${model_path}" "${model_name}" "${train_name}" "train" "${INFER_NOTE}" "${N_SAMPLES}"
  done < <(split_csv "${TRAIN_DATA_NAMES}")

  while IFS= read -r valid_name; do
    [[ -n "${valid_name}" ]] || continue
    compute_grad_for_split "${model_path}" "${model_name}" "${valid_name}" "valid" "${VALID_INFER_NOTE}" "${N_SAMPLES_VAL}"
  done < <(split_csv "${VALID_DATA_NAMES}")
}

export_actor_to_hf() {
  local actor_dir=$1
  if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -d "${actor_dir}" ]] || die "Actor directory does not exist: ${actor_dir}"
  fi
  run_cmd "python '${REPO_ROOT}/cropi/utils/model_merger.py' --local_dir '${actor_dir}'"
}

run_rl_round() {
  local iter_idx=$1
  local init_model_path=$2
  local model_name=$3

  [[ -n "${BASE_MODEL_PATH}" ]] || die "BASE_MODEL_PATH is required for RL modes"
  command -v "${RL_PYTHON}" >/dev/null 2>&1 || die "RL_PYTHON not found: ${RL_PYTHON}"

  local train_files_hydra
  train_files_hydra=$(build_train_files_hydra "${model_name}" "${iter_idx}")
  local val_files_hydra
  val_files_hydra=$(build_val_files_hydra)
  local ckpt_dir="${CKPT_ROOT}/${RL_PROJECT_NAME}/iter${iter_idx}"
  local logger_cfg="['console']"
  if [[ "${RL_USE_WANDB}" == "1" ]]; then
    logger_cfg="['console','wandb']"
  fi

  log "Running RL round ${iter_idx} with ${RL_NUM_GPUS} GPUs"
  run_cmd "cd '${RL_WORKDIR}' && '${RL_PYTHON}' -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=\"${train_files_hydra}\" \
    data.val_files=\"${val_files_hydra}\" \
    data.train_batch_size=${RL_TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${RL_MAX_PROMPT_LENGTH} \
    data.max_response_length=${RL_MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path='${init_model_path}' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config._attn_implementation=eager \
    actor_rollout_ref.actor.ppo_mini_batch_size=${RL_PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${RL_PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${RL_TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${RL_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.n=${RL_N_SAMPLES} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${RL_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=${logger_cfg} \
    trainer.project_name='${RL_PROJECT_NAME}' \
    trainer.experiment_name='iter${iter_idx}' \
    trainer.default_local_dir='${ckpt_dir}' \
    trainer.n_gpus_per_node=${RL_NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${RL_SAVE_FREQ} \
    trainer.test_freq=${RL_TEST_FREQ} \
    trainer.total_training_steps=${RL_TOTAL_TRAINING_STEPS}"

  export_actor_to_hf "${ckpt_dir}/global_step_${RL_TOTAL_TRAINING_STEPS}/actor"
}

full_pipeline() {
  [[ -n "${BASE_MODEL_PATH}" ]] || die "BASE_MODEL_PATH is required in full mode"

  local current_model_name="${INITIAL_MODEL_NAME}"
  local current_model_path="${BASE_MODEL_PATH}"
  local round=0

  while [[ ${round} -lt ${NUM_RL_ROUNDS} ]]; do
    run_score_and_select "${current_model_name}" "${round}"
    run_rl_round "${round}" "${current_model_path}" "${current_model_name}"

    if [[ ${round} -lt $((NUM_RL_ROUNDS - 1)) ]]; then
      local next_name
      next_name=$(next_model_name "${round}")
      local next_model_path="${CKPT_ROOT}/${RL_PROJECT_NAME}/iter${round}/global_step_${RL_TOTAL_TRAINING_STEPS}/actor/huggingface"
      prepare_rollout_jsonls "${current_model_name}" "${next_name}"
      compute_gradients_for_model "${next_model_path}" "${next_name}"
      current_model_name="${next_name}"
      current_model_path="${next_model_path}"
    fi

    round=$((round + 1))
  done
}

legacy_select_only() {
  log "Running backward-compatible select-only flow"
  run_score_and_select "${INITIAL_MODEL_NAME}" "0"
}

case "${MODE}" in
  select-only)
    legacy_select_only
    ;;
  rl-only)
    [[ -n "${BASE_MODEL_PATH}" ]] || die "BASE_MODEL_PATH is required in rl-only mode"
    run_rl_round 0 "${BASE_MODEL_PATH}" "${INITIAL_MODEL_NAME}"
    ;;
  full)
    full_pipeline
    ;;
  *)
    die "Unsupported mode: ${MODE}"
    ;;
esac
