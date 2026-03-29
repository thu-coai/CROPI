import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description="Compute CROPI influence scores from projected gradients.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data files")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct", help="Model name")
    parser.add_argument("--proj_note", type=str, default="trak_norm_seed0_mid0_projdim32768", help="Projection note")
    parser.add_argument("--num_parallel", type=int, default=8, help="Number of gradient shards to read")
    parser.add_argument("--temperature", default=0.5, type=float, help="Sampling temperature")
    parser.add_argument("--n_samples", default=5, type=int, help="Train sampling count")
    parser.add_argument("--n_samples_val", default=5, type=int, help="Validation sampling count")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use_sign", action="store_true", help="Use sign gradients before similarity")
    parser.add_argument("--valid_data_names", type=str, default="gsm8k,math,gaokao2023en,aime24,amc23,olympiadbench")
    parser.add_argument("--valid_dpp", action="store_true", help="Reserved flag kept for compatibility")
    parser.add_argument("--n_valid_dpp_select", type=int, default=8, help="Reserved compatibility flag")
    parser.add_argument("--train_data_names", type=str, default="gsm8k,math", help="Comma-separated train sets")
    parser.add_argument("--ignore_valid_names", type=str, default="", help="Comma-separated valid sets to ignore")
    parser.add_argument("--max_valid_grads", type=int, default=100, help="Max valid gradients to average; <=0 uses all")
    parser.add_argument("--prompt_type", type=str, default="qwen25-math-cot", help="Prompt template name")
    parser.add_argument("--grad_feature_subroot", type=str, default=None, help="Sub-directory holding gradient shards")
    return parser.parse_args()


def load_jsonl_grad_shards(base_path: str, num_parallel: int, device: torch.device) -> tuple[list[str], list[torch.Tensor]]:
    prompts: list[str] = []
    grads: list[torch.Tensor] = []
    for i in range(num_parallel):
        shard_path = f"{base_path}.{i}"
        if not os.path.exists(shard_path):
            continue
        if os.path.getsize(shard_path) == 0:
            continue

        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                grad = item["grad"]
                if isinstance(grad, list):
                    grad = torch.tensor(grad, dtype=torch.float32, device=device).view(-1)
                if grad.abs().sum() < 1e-6:
                    continue
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise ValueError(f"Invalid gradient found for prompt: {item['prompt']}")
                prompts.append(item["prompt"])
                grads.append(grad)
    return prompts, grads


def load_valid_prompt_strings(data_root: str, data_name: str) -> set[str]:
    valid_prompt_data_path = os.path.join(data_root, data_name, "valid_qwen.parquet")
    valid_prompt_data = pd.read_parquet(valid_prompt_data_path).to_dict(orient="records")
    return {row["prompt"][1]["content"] for row in valid_prompt_data}


def build_valid_gradient_bank(args, device: torch.device, valid_data_names: list[str]) -> tuple[dict[str, torch.Tensor], str]:
    valid_name2grad_feature: dict[str, torch.Tensor] = {}
    grad_feature_subroot = args.grad_feature_subroot or args.model_name
    valid_dpp_note = f"_valid_dpp_{args.n_valid_dpp_select}" if args.valid_dpp else ""

    grad_feature_file_name_valid = (
        f"valid_{args.prompt_type}_-1_seed{args.seed}_t{args.temperature}_n{args.n_samples_val}_s0_e-1_grad_{args.proj_note}.jsonl"
    )
    valid_grad_path = os.path.join(
        args.data_root,
        args.model_name,
        f"valid_grad_feature_{args.proj_note}_{args.valid_data_names}{valid_dpp_note}.json",
    )
    os.makedirs(os.path.dirname(valid_grad_path), exist_ok=True)

    if os.path.exists(valid_grad_path):
        with open(valid_grad_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        valid_name2grad_feature = {
            name: torch.tensor(values, dtype=torch.float32, device=device) for name, values in cached.items()
        }
        print(f"[INFO] Loaded valid grad features from {valid_grad_path}.")
        return valid_name2grad_feature, valid_dpp_note

    for data_name in valid_data_names:
        valid_prompts = load_valid_prompt_strings(args.data_root, data_name)
        grad_feature_root = os.path.join(args.data_root, data_name, grad_feature_subroot)
        print(f"[INFO] Loading validation gradients from {grad_feature_root} for `{data_name}`.")

        base_path = os.path.join(grad_feature_root, grad_feature_file_name_valid)
        prompts, grads = load_jsonl_grad_shards(base_path, args.num_parallel, device)
        if not grads:
            fallback_name = (
                f"valid_{args.prompt_type}_-1_seed{args.seed}_t{args.temperature}_n{args.n_samples_val}_log_probs_s0_e-1_grad_{args.proj_note}.jsonl"
            )
            prompts, grads = load_jsonl_grad_shards(os.path.join(grad_feature_root, fallback_name), args.num_parallel, device)

        filtered_grads = [grad for prompt, grad in zip(prompts, grads) if prompt in valid_prompts]
        if not filtered_grads:
            print(f"[WARNING] No usable validation gradients found for `{data_name}`.")
            continue

        grad_tensor = torch.stack(filtered_grads).to(device)
        if args.max_valid_grads > 0:
            torch.manual_seed(0)
            num_sel = min(args.max_valid_grads, len(grad_tensor))
            indices = torch.randperm(len(grad_tensor))[:num_sel]
            grad_tensor = grad_tensor[indices]

        grad_mean = grad_tensor.mean(dim=0)
        if torch.isnan(grad_mean).any() or torch.isinf(grad_mean).any():
            raise ValueError(f"Invalid averaged validation gradient for `{data_name}`")
        valid_name2grad_feature[data_name] = grad_mean

    with open(valid_grad_path, "w", encoding="utf-8") as f:
        json.dump({k: v.cpu().tolist() for k, v in valid_name2grad_feature.items()}, f, indent=2)
    print(f"[INFO] Saved valid grad features to {valid_grad_path}.")
    return valid_name2grad_feature, valid_dpp_note


def check_tensor(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN found in tensor `{name}`")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf found in tensor `{name}`")
    if tensor.ndim == 2 and (tensor.abs().sum(dim=1) == 0).any():
        raise ValueError(f"Zero rows found in tensor `{name}`")


def compute_scores(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_data_names = [name for name in args.valid_data_names.split(",") if name]
    ignore_valid_names = [name for name in args.ignore_valid_names.split(",") if name]
    valid_data_names = [name for name in valid_data_names if name not in ignore_valid_names]
    train_data_names = [name for name in args.train_data_names.split(",") if name]
    grad_feature_subroot = args.grad_feature_subroot or args.model_name
    extra_sign = "_sign" if args.use_sign else ""

    valid_name2grad_feature, valid_dpp_note = build_valid_gradient_bank(args, device, valid_data_names)
    valid_data_names = list(valid_name2grad_feature.keys())
    print(f"[INFO] Valid datasets used for scoring: {valid_data_names}")

    grad_feature_file_name_train = (
        f"train_{args.prompt_type}_-1_seed{args.seed}_t{args.temperature}_n{args.n_samples}_s0_e-1_grad_{args.proj_note}.jsonl"
    )
    data_name2prompt2valid_name2inf_score = defaultdict(lambda: defaultdict(dict))

    for data_name in train_data_names:
        grad_feature_root = os.path.join(args.data_root, data_name, grad_feature_subroot)
        base_path = os.path.join(grad_feature_root, grad_feature_file_name_train)
        prompts, grad_features = load_jsonl_grad_shards(base_path, args.num_parallel, device)
        if not grad_features:
            raise FileNotFoundError(f"No training gradients found under {base_path}.[0-{args.num_parallel - 1}]")

        grad_features_tensor = torch.stack(grad_features).to(device)
        print(f"[INFO] Loaded {len(grad_features)} gradients for training set `{data_name}`.")

        for valid_name in valid_data_names:
            valid_grad = valid_name2grad_feature[valid_name].to(device).view(1, -1)
            grad_feature_ = grad_features_tensor.sign() if args.use_sign else grad_features_tensor
            valid_grad_ = valid_grad.sign() if args.use_sign else valid_grad

            check_tensor(grad_feature_, "grad_feature_")
            check_tensor(valid_grad_, "valid_grad_")

            cos_sims = torch.nn.functional.cosine_similarity(grad_feature_, valid_grad_, dim=1)
            if torch.isnan(cos_sims).any() or torch.isinf(cos_sims).any():
                raise ValueError(f"Invalid cosine similarities for train={data_name}, valid={valid_name}")

            for idx, prompt in enumerate(prompts):
                data_name2prompt2valid_name2inf_score[data_name][prompt][valid_name] = cos_sims[idx].item()

    dense_scores = json.loads(json.dumps(data_name2prompt2valid_name2inf_score))
    save_path = os.path.join(
        args.data_root,
        args.model_name,
        f"train_valid_inf_score_{args.proj_note}_train_{args.train_data_names}_valid_{args.valid_data_names}{extra_sign}{valid_dpp_note}.json",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dense_scores, f)

    max_valid_scores = {}
    for data_name in train_data_names:
        prompt2valid_name2inf_score = dense_scores[data_name]
        prompt2max_inf_score = {}
        for prompt, valid_name2inf_score in tqdm(prompt2valid_name2inf_score.items(), desc=f"Processing {data_name}"):
            max_valid_name = max(valid_name2inf_score, key=valid_name2inf_score.get)
            prompt2max_inf_score[prompt] = {
                "score": valid_name2inf_score[max_valid_name],
                "valid_name": max_valid_name,
                "all_scores": valid_name2inf_score,
            }
        max_valid_scores[data_name] = prompt2max_inf_score

    save_path_max_valid = os.path.join(
        args.data_root,
        args.model_name,
        f"train_valid_score_max_valid_inf_{args.proj_note}_train_{args.train_data_names}_valid_{args.valid_data_names}{extra_sign}{valid_dpp_note}.json",
    )
    with open(save_path_max_valid, "w", encoding="utf-8") as f:
        json.dump(max_valid_scores, f)
    print(f"[INFO] Saved max-valid influence scores to {save_path_max_valid}.")


def main():
    args = parse_args()
    compute_scores(args)


if __name__ == "__main__":
    main()
