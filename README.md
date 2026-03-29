# CROPI: Data-Efficient RLVR via Off-Policy Influence Guidance

<<<<<<< HEAD
=======
**CROPI** is a curriculum reinforcement learning framework for large language models that brings theoretically grounded, rollout-free data selection to RLVR. It leverages influence functions to estimate how individual data points impact the current online policy using only pre-collected trajectories, enabling efficient and scalable training without costly new rollouts.


## News

[251101] We create the github repository. The detailed implementation of **CROPI** will be released as soon as possible. 

## Framework
![framework](figures/CROPI-framework.drawio.png)

1. **Off-Policy Influence Estimation**: quantifies per-datum impact on the online policy from offline trajectories, with theoretical guarantees and no real-time sampling.
2. **Scalable gradient handling**: Sparse Random Projection with a simple pre-projection dropout step to compress high-dimensional gradients, reduce numerical noise, and preserve inner products.
3. **Curriculum RL**: stage-wise training that selects the most influential data at each checkpoint, improving efficiency and performance over full-dataset and heuristic baselines.

## Highlight
On a 1.5B model, it delivers a **$2.66\times$** step-level acceleration while training on only **10%** of the data per stage—demonstrating the practical gains of influence-based data selection for online RLVR.
<!-- ![1.5b_curve](figures/1.5B-main-curve.png)-->
>>>>>>> origin/main
<p align="center">
  <img src="figures/CROPI-logo.png" alt="CROPI logo" width="220">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.26491"><img src="https://img.shields.io/badge/Paper-arXiv-red.svg" alt="Paper"></a>
</p>

**CROPI** is a curriculum reinforcement learning framework for large language models that improves RLVR efficiency through **off-policy influence estimation** over pre-collected trajectories. Instead of relying on repeated online trial-and-error to identify useful training data, CROPI estimates which prompts are most helpful for the current policy, selects a compact high-value subset, and then runs RL on that subset.

Paper: **[Data-Efficient RLVR via Off-Policy Influence Guidance](https://arxiv.org/abs/2510.26491)**

## Why CROPI?

Large-scale RL for reasoning models is expensive because it spends substantial compute on prompts that contribute little to policy improvement. CROPI addresses this by:

- estimating prompt utility with **off-policy influence scores**
- reusing **pre-collected rollout logs** instead of requiring fresh online sampling for every scoring step
- using **sparse random projection** to make gradient-based scoring practical at scale
- building a **curriculum over selected data subsets**, rather than training on the full pool at every stage

The result is a more compute-efficient RL loop that keeps or improves downstream performance while substantially reducing the amount of data trained per stage.

## Main Idea

<p align="center">
  <img src="figures/CROPI-framework.drawio.png" alt="CROPI framework" width="760">
</p>

CROPI follows three core steps:

1. **Off-policy influence estimation**
   Compute projected policy-gradient features from pre-collected rollout trajectories.
2. **Validation-targeted scoring**
   Measure how aligned each train prompt is with the validation objectives via influence scores.
3. **Curriculum RL**
   Select a compact training subset and run RL on that subset, then repeat for the next round.

This makes CROPI particularly attractive when rollout generation is expensive, RL budgets are limited, or you want a more principled curriculum than heuristic filtering.

## Results

### 1.5B Main Training Curve

<p align="center">
  <img src="figures/1.5B-main-curve.png" alt="1.5B main curve" width="720">
</p>

On the 1.5B setting, CROPI achieves **2.66x step-level acceleration** while training on only **10% of the data per stage**. The key takeaway is not just faster training, but **better use of RL compute**: CROPI focuses updates on prompts that matter most for validation-time improvement.

### Overall Comparison

<p align="center">
  <img src="figures/main_result.png" alt="Main result table" width="760">
</p>

The paper shows that CROPI consistently improves the efficiency-quality tradeoff compared with stronger data-hungry baselines. The method is designed to be practical: it works with realistic RL pipelines and avoids introducing another expensive online loop solely for data scoring.

### Sparse Projection Efficiency

<p align="center">
  <img src="figures/1.5B_sparse_projection.png" alt="1.5B sparse projection" width="520">
</p>

<p align="center">
  <img src="figures/3B_sparse_projection.png" alt="3B sparse projection" width="520">
</p>

These results highlight a second advantage of CROPI: the influence-estimation pipeline remains usable even for larger models because it relies on **sparse random projection** rather than storing or comparing raw full-dimensional gradients.

## Repository Overview

This repository contains the open-source implementation of the **CROPI selection and curriculum loop**. The runnable code lives under [`cropi/`](./cropi).

### Repository structure

- `cropi/core`
  Core logic for gradient extraction, influence-score computation, and prompt selection.
- `cropi/utils`
  Utilities for JSONL splitting and RL checkpoint merging/export.
- `cropi/scripts`
  Shell entry points, including a single-script end-to-end controller.
- `cropi/inference`
  Optional wrappers for external rollout-generation scripts.
- `figures`
  Paper figures used in this README.

### What is included

- CROPI scoring and selection pipeline
- multi-round orchestration through one script
- support for iterative `select -> RL -> select -> RL`
- support for exporting RL checkpoints into Hugging Face format for the next CROPI round

### What is not included

- datasets
- model checkpoints
- rollout logs
- experiment logs and tracking artifacts

You are expected to prepare these assets locally.

## How To Run

### 1. Create the environment with `uv`

```bash
cd CROPI

uv venv
source .venv/bin/activate

uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
uv pip install numpy pandas pyarrow tqdm "transformers<5" math-verify tabulate
uv pip install --no-deps traker==0.3.2
uv pip install fast_jl --no-build-isolation
```

If you already have a suitable PyTorch installation on the machine, you can also use:

```bash
uv venv --system-site-packages
source .venv/bin/activate
```

### 2. Sanity-check the installation

```bash
uv run python -m compileall cropi
uv run cropi-select --help
uv run cropi-compute-inf-score --help
uv run cropi-get-grad --help
bash -n cropi/scripts/*.sh cropi/inference/*.sh
```

### 3. Prepare the expected data layout

The public pipeline expects files in the following structure:

```text
data/
  <train_dataset>/
    train_qwen.parquet
    <model_name>/
      train_<infer_note>.jsonl
      train_<infer_note>_grad_<proj_note>.jsonl.<rank>
  <valid_dataset>/
    valid_qwen.parquet
    <model_name>/
      valid_<valid_infer_note>.jsonl
      valid_<valid_infer_note>_grad_<proj_note>.jsonl.<rank>
```

In other words:

- parquet files provide the raw train/validation prompt pool
- rollout JSONL files contain `prompt`, `answer`, `responses`, and `rewards`
- gradient JSONL shards contain projected gradient features for CROPI scoring

### 4. Run one CROPI selection stage

To compute influence scores and select data once:

```bash
cd CROPI

bash cropi/scripts/run_cropi.sh select-only ./data Qwen2.5-1.5B-Instruct_curriculum
```

This runs:

1. `cropi-compute-inf-score`
2. `cropi-select`

and writes the selected parquet under the corresponding dataset directory.

### 5. Run the full CROPI loop from one script

`cropi/scripts/run_cropi.sh` is the top-level entry point for the full pipeline. It supports:

- `select-only`
- `rl-only`
- `full`

The `full` mode executes the iterative pipeline:

```text
select -> RL -> recompute gradients for the new checkpoint -> select -> RL -> ...
```

#### Minimal full-pipeline example

```bash
cd CROPI

BASE_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct \
RL_PYTHON=/path/to/verl-env/bin/python \
TRAIN_DATA_NAMES=gsm_math_dsr_test \
VALID_DATA_NAMES=gsm8k,math,gaokao2023en \
RL_VAL_DATA_NAMES=gsm8k,math,gaokao2023en \
NUM_RL_ROUNDS=2 \
RL_NUM_GPUS=8 \
RL_TP_SIZE=2 \
bash cropi/scripts/run_cropi.sh full ./data Qwen2.5-1.5B-Instruct_curriculum
```

#### Important runtime knobs

- `BASE_MODEL_PATH`
  Initial Hugging Face checkpoint for the first RL round.
- `RL_PYTHON`
  Python executable with `verl` installed.
- `NUM_RL_ROUNDS`
  Number of CROPI+RL stages to run.
- `RL_NUM_GPUS`
  Number of GPUs for RL. Default: `8`.
- `RL_TP_SIZE`
  vLLM tensor-parallel size during rollout. Default: `2`.
- `NUM_PARALLEL`
  Number of gradient shards / `cropi-get-grad` workers.
- `RL_TOTAL_TRAINING_STEPS`
  RL steps per stage.
- `DRY_RUN=1`
  Print the full command chain without executing it.

### 6. Notes on RL support

- `run_cropi.sh` is the **supported public entry point** for the full iterative pipeline.
- The script now supports **8-GPU RL** by default.
- After each RL stage, the script exports the actor checkpoint to `huggingface/` format so the next CROPI stage can reuse it directly.
- If your RL environment is separate from the CROPI environment, point `RL_PYTHON` to that interpreter.

## Acknowledgements

We thank the following projects and teams for making this work possible:

- [VeRL](https://github.com/volcengine/verl) for the RL training framework
- [TRAK / traker](https://github.com/MadryLab/trak) for efficient influence-inspired gradient projection tooling
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) for the base language models
- [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) for math-oriented model and evaluation resources
- [vLLM](https://github.com/vllm-project/vllm) for efficient rollout generation and serving

We also acknowledge the external math-evaluation tooling used by the original project setup, including the Qwen2.5-Math evaluation pipeline.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@misc{zhu2025dataefficientrlvroffpolicyinfluence,
  title={Data-Efficient RLVR via Off-Policy Influence Guidance},
  author={Erle Zhu and Dazhi Jiang and Yuan Wang and Xujun Li and Jiale Cheng and Yuxian Gu and Yilin Niu and Aohan Zeng and Jie Tang and Minlie Huang and Hongning Wang},
  year={2025},
  eprint={2510.26491},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2510.26491}
}
```
