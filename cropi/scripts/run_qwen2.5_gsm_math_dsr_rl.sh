#!/bin/bash
select_note=$1
i_iter=$2
iteration_steps=$3
MODEL_PATH=$4
model_shorthand=$5
score_method=$6
CKPT_ROOT=$7

if [[ -z ${model_shorthand} ]]; then
    echo "[ERROR] No model shorthand provided. Please provide the model shorthand as the fifth argument."
    exit 1
fi
if [[ -z ${score_method} ]]; then 
    echo "[ERROR] No score method provided. Please provide the score method "
    exit 1
fi 

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1

PROJECT_NAME=verl_grpo_gsm_math_dsr 

# if selectt_note is none, manually set
if [ -z "$select_note" ]; then
    echo "[ERROR] No select_note provided. Please provide the select_note as the first argument."
    exit 1
fi
echo "select note: $select_note"

if [[ ${model_shorthand,,} == *"r1"* ]]; then
    train_path_extra="_r1"
    echo "Using R1-format data: ${train_path_extra}"
else
    train_path_extra=""
fi

gsm_math_dsr_train_sel_path=./data/gsm_math_dsr_test/${model_shorthand}/train_qwen${train_path_extra}_selected_${select_note}_iter${i_iter}.parquet # Model-based selection

gsm8k_test_path=./data/gsm8k/test_qwen_split_valid.parquet
math_test_path=./data/math/test_qwen_split_valid.parquet
gaokao_test_path=./data/gaokao2023en/test_qwen_split_valid.parquet
olympiad_test_path=./data/olympiadbench/test_qwen_split_valid.parquet
aime24_test_path=./data/aime24/test_qwen_split_valid.parquet
amc23_test_path=./data/amc23/test_qwen_split_valid.parquet

train_files_list=(${gsm_math_dsr_train_sel_path}) # GSM8K-Train + MATH-Train + DeepScaleR
train_files="['$gsm_math_dsr_train_sel_path']"
test_files="['$gsm8k_test_path', '$math_test_path', '$gaokao_test_path', '$olympiad_test_path', '$aime24_test_path', '$amc23_test_path']"

if [[ -z ${MODEL_PATH} ]]; then
    MODEL_PATH=${CKPT_ROOT}/Qwen/Qwen2.5-1.5B-Instruct
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --host="${WANDB_BASE_URL:-https://api.wandb.ai}" || true
else
    echo "[INFO] WANDB_API_KEY is not set. Falling back to console logging only."
fi

if [[ $MODEL_PATH == *"Qwen2.5-3B-Instruct"* ]]; then
    MODEL=qwen2.5_3b
elif [[ $MODEL_PATH == *"Qwen2.5-7B-Instruct"* ]]; then
    MODEL=qwen2.5_7b
elif [[ $MODEL_PATH == *"Qwen2.5-14B-Instruct"* ]]; then
    MODEL=qwen2.5_14b
elif [[ $MODEL_PATH == *"Qwen2.5-32B-Instruct"* ]]; then
    MODEL=qwen2.5_32b
elif [[ $MODEL_PATH == *"Qwen2.5-1.5B-Instruct"* ]]; then
    MODEL=qwen2.5_1.5b
elif [[ $MODEL_PATH == *"Qwen2.5-7B"* ]]; then
    MODEL=qwen2.5_7b_base
elif [[ $MODEL_PATH == *"Qwen2.5-1.5B"* ]]; then
    MODEL=qwen2.5_1.5b_base
elif [[ $MODEL_PATH == *"DeepSeek-R1-Distill-Qwen-1.5B"* ]]; then
    MODEL=qwen2.5_1.5b_r1_distill
else
    echo "${MODEL_PATH} is not correct"
fi

MODEL=${MODEL}_curriculum_${score_method}_step${iteration_steps}

batch_size=128
mini_bsz=128
max_response_len=2048
test_freq=10

if [[ $MODEL_PATH == *"Qwen2.5-7B"* ]]; then
    max_response_len=4096
    extra_args=""
elif [[ $MODEL_PATH == *"R1"* ]]; then
    max_response_len=8192
    log_prob_micro_batch_size_per_gpu=8
    test_freq=10
    extra_args="actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.rollout.max_num_batched_tokens=16384"
fi

n_samples=8
total=0
for f in "${train_files_list[@]}"; do
    num=$(python -c "import pandas as pd; print(len(pd.read_parquet('$f')))")
    echo "[STAT] $f : $num"
    total=$((total + num))
done

echo "[INFO] Total number of training data: [ ${total} ]"
echo "[INFO] Batch size: ${batch_size}"
epoch_steps=$(($total/$batch_size))
echo "[INFO] Epoch steps: ${epoch_steps}"
save_freq=${iteration_steps}
echo "[INFO] Save frequency: ${save_freq}"

if [ $batch_size -eq 1024 ]; then
    select_note_norm="${select_note//,/}"
    EXPERIMENT_NAME=${MODEL}_function_rm_val_qwen_sel_${select_note_norm}
else
    select_note_norm="${select_note//,/}"
    EXPERIMENT_NAME=${MODEL}_function_rm_val_qwen_sel_${select_note_norm}_bsz${batch_size}
fi
CKPT_PATH=${CKPT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}
echo "[INFO] Set checkpoint path: ${CKPT_PATH}. "
mkdir -p ${CKPT_PATH}

TARGET_STEP=$((iteration_steps*(i_iter+1)))
if [[ -d ${CKPT_PATH} ]]; then 
    LAST_STEP=$(cat ${CKPT_PATH}/latest_checkpointed_iteration.txt)
    if [[ ${LAST_STEP} -ge ${TARGET_STEP} ]]; then
        echo "[INFO] Current training step is ahead of target step, no training need. Exiting run_rl_xxx script ..."
        return 0
    fi
fi 

which python3
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=2048 \
    data.max_response_length=${max_response_len} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${n_samples} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${CKPT_PATH} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_training_steps=${TARGET_STEP} ${extra_args}
