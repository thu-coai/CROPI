import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser(description="Select training data with CROPI scores.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data files")
    parser.add_argument(
        "--score_method",
        type=str,
        default="inf",
        choices=["inf", "random", "difficulty", "difficulty_random", "diversity", "inf_valid_uniform", "inf_acg_valid_uniform"],
        help="Method to score prompts",
    )
    parser.add_argument("--score_path", type=str, default=None, help="Path to the influence score file")
    parser.add_argument("--select_ratio", type=float, default=0.22, help="Ratio of prompts to select")
    parser.add_argument("--train_data_names", type=str, default="gsm8k,math", help="Comma-separated train datasets")
    parser.add_argument("--valid_data_names", type=str, default="gsm8k,math,gaokao2023en,olympiadbench", help="Comma-separated valid datasets")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--infer_note", type=str, default="qwen25-math-cot_-1_seed0_t0.5_n8_s0_e-1", help="Inference note")
    parser.add_argument("--proj_note", type=str, default="trak_norm_seed0_mid0_projdim32768", help="Projection note")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct", help="Model name used for inference")
    parser.add_argument("--num_parallel", type=int, default=8, help="Reserved compatibility flag")
    parser.add_argument("--load_from_cache", action="store_true", help="Load cached valid-uniform selections")
    parser.add_argument("--i_iter", default=None, type=int, help="Curriculum iteration index")
    return parser.parse_args()


def load_scores_from_path(score_path: str):
    with open(score_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts_from_raw_path(raw_path: str):
    df = pd.read_parquet(raw_path)
    prompts = df["prompt"].tolist()
    return [item[1]["content"] for item in prompts]


def stat_list(values):
    value2cnt = {}
    for value in values:
        value2cnt[value] = value2cnt.get(value, 0) + 1
    return value2cnt


def build_train_raw_paths(data_root: str, train_data_names: list[str], model_name: str):
    extra_model_note = "_r1" if "r1" in model_name.lower() else ""
    return [f"{data_root}/{data_name}/train_qwen{extra_model_note}.parquet" for data_name in train_data_names]


def load_pass_rate_and_difficulty_maps(args, train_data_names: list[str]):
    pass_rate_maps = {name: {} for name in train_data_names}
    diff_maps = {name: {} for name in train_data_names}

    if "difficulty" not in args.score_method and not args.score_method.startswith("inf"):
        return pass_rate_maps, diff_maps

    for data_name in train_data_names:
        infer_data_path = f"{args.data_root}/{data_name}/{args.model_name}/train_{args.infer_note}.jsonl"
        prompt2pass_rate = {}
        prompt2diff_score = {}
        with open(infer_data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                pass_rate = float(np.mean(item["rewards"]))
                prompt2pass_rate[item["prompt"]] = pass_rate
                prompt2diff_score[item["prompt"]] = pass_rate * (1 - pass_rate)

        pass_rate_maps[data_name] = prompt2pass_rate
        diff_maps[data_name] = prompt2diff_score
        print(f"[INFO] Loaded {len(prompt2pass_rate)} pass-rate entries for `{data_name}`.")

    return pass_rate_maps, diff_maps


def build_prompt_scores(args, train_data_names: list[str], train_raw_paths: list[str], diff_maps):
    score_method = args.score_method
    if score_method == "inf":
        if not args.score_path or not os.path.exists(args.score_path):
            raise FileNotFoundError(f"Method `inf` requires --score_path, got: {args.score_path}")
        score_note = os.path.basename(args.score_path).replace("train_valid_score_", "").replace(".json", "")
        return load_scores_from_path(args.score_path), score_note

    if score_method.startswith("inf_") and score_method.endswith("valid_uniform"):
        if not args.score_path or not os.path.exists(args.score_path):
            raise FileNotFoundError(f"Method `{score_method}` requires --score_path, got: {args.score_path}")
        score_note = os.path.basename(args.score_path).replace("train_valid_score_", "").replace(".json", "")
        return load_scores_from_path(args.score_path), f"{score_note}_valid_uniform"

    if score_method == "random":
        data_name2prompt2score = {}
        for idx, data_name in enumerate(train_data_names):
            train_prompts = load_prompts_from_raw_path(train_raw_paths[idx])
            data_name2prompt2score[data_name] = {
                prompt: {"score": random.uniform(0, 1), "valid_name": "unknown"} for prompt in train_prompts
            }
        return data_name2prompt2score, "random"

    if "difficulty" in score_method:
        if not args.score_path or not os.path.exists(args.score_path):
            raise FileNotFoundError(f"Method `{score_method}` requires --score_path, got: {args.score_path}")
        data_name2prompt2score = load_scores_from_path(args.score_path)
        for data_name in train_data_names:
            prompt2diff_score = diff_maps[data_name]
            for prompt, score_item in data_name2prompt2score[data_name].items():
                prompt2diff_score.setdefault(prompt, 0.0)
                if "random" in score_method:
                    score_item["score"] = random.uniform(0, 1) + 1e-3 if prompt2diff_score[prompt] > 0 else 0.0
                else:
                    score_item["score"] = prompt2diff_score[prompt]
        return data_name2prompt2score, score_method

    raise ValueError(f"Unsupported score method: {score_method}")


def select_prompts_valid_uniform(args, train_name: str, valid_data_names, prompt2score_dict, num_select: int, score_note: str):
    select_prompts_by_valid = {}
    for valid_data_name in valid_data_names:
        cache_path = (
            f"{args.data_root}/{train_name}/{args.model_name}/"
            f"train_{args.infer_note}_valid_uniform_selected_{valid_data_name}_n{num_select}_{score_note}.json"
        )
        if args.load_from_cache and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                select_prompts_by_valid[valid_data_name] = json.load(f)
            continue

        prompt2score_valid = {prompt: prompt2score_dict[prompt]["all_scores"][valid_data_name] for prompt in prompt2score_dict}
        sorted_prompts = sorted(prompt2score_valid.items(), key=lambda x: x[1], reverse=True)
        selected = [prompt for prompt, _ in sorted_prompts[:num_select]]
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)
        select_prompts_by_valid[valid_data_name] = selected

    merged_scores = {}
    for valid_data_name, prompts in select_prompts_by_valid.items():
        for rank, prompt in enumerate(prompts, start=1):
            merged_scores.setdefault(prompt, {})[valid_data_name] = rank

    return {prompt: float(np.sum(1 / np.array(list(valid2rank.values())))) for prompt, valid2rank in merged_scores.items()}


def save_selected_dataset(args, train_name: str, raw_path: str, prompt2score_dict, prompt2score, pass_rate_maps, diff_maps, valid_data_names, score_note):
    select_note = f"{score_note}_ratio{args.select_ratio}"
    if "1.5b" in args.model_name.lower():
        select_note += "_1.5b"
    elif "7b" in args.model_name.lower():
        select_note += "_7b"

    save_path = raw_path.replace(".parquet", f"_selected_{select_note}.parquet")
    if args.i_iter is not None:
        save_dir = os.path.join(os.path.dirname(save_path), args.model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.basename(save_path).replace(".parquet", f"_iter{args.i_iter}.parquet")
        save_path = os.path.join(save_dir, save_name)

    stat_save_path = save_path.replace(".parquet", "_stat.json")
    if os.path.exists(save_path) and os.path.exists(stat_save_path):
        print(f"[INFO] Selection already exists for `{train_name}` at {save_path}.")
        return

    train_raw_data = pd.read_parquet(raw_path)
    num_raw = len(train_raw_data)
    num_select = int(num_raw * args.select_ratio)

    if args.score_method.startswith("inf_valid_uniform"):
        prompt2score = select_prompts_valid_uniform(args, train_name, valid_data_names, prompt2score_dict, num_select, score_note)

    sorted_prompts = sorted(prompt2score.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_prompts) < num_select:
        num_select = len(sorted_prompts)

    selected_prompts = {prompt for prompt, _ in sorted_prompts[:num_select]}
    selected_scores = [score for _, score in sorted_prompts[:num_select]]
    print(f"[INFO] Selected {len(selected_prompts)} prompts from `{train_name}`. Top scores: {selected_scores[:10]}")

    def valid_prompt(prompt):
        return prompt[1]["content"] in selected_prompts

    train_data_selected = train_raw_data[train_raw_data["prompt"].apply(valid_prompt)]
    if len(train_data_selected) == 0:
        raise ValueError(f"No data selected for `{train_name}` with ratio {args.select_ratio}")

    prompt2valid_name = {prompt: item.get("valid_name", "unknown") for prompt, item in prompt2score_dict.items()}
    selected_pass_rates = []
    selected_diff_scores = []
    selected_data_source = []
    selected_valid_names = []
    for row in train_data_selected.itertuples():
        prompt = row.prompt[1]["content"]
        selected_pass_rates.append(pass_rate_maps[train_name].get(prompt, "unknown"))
        selected_diff_scores.append(diff_maps[train_name].get(prompt, "unknown"))
        selected_valid_names.append(prompt2valid_name.get(prompt, "unknown"))
        selected_data_source.append(getattr(row, "data_source", "unknown"))

    selected_stat = {
        "num_selected": len(train_data_selected),
        "pass_rate": stat_list(selected_pass_rates),
        "diff_score": stat_list(selected_diff_scores),
        "data_source": stat_list(selected_data_source),
        "valid_name": stat_list(selected_valid_names),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    train_data_selected.to_parquet(save_path)
    with open(stat_save_path, "w", encoding="utf-8") as f:
        json.dump(selected_stat, f, indent=2)
    print(f"[INFO] Saved selected dataset to {save_path}")
    print(f"[INFO] Saved selection statistics to {stat_save_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_data_names = [name for name in args.train_data_names.split(",") if name]
    valid_data_names = [name for name in args.valid_data_names.split(",") if name]
    train_raw_paths = build_train_raw_paths(args.data_root, train_data_names, args.model_name)

    pass_rate_maps, diff_maps = load_pass_rate_and_difficulty_maps(args, train_data_names)
    data_name2prompt2score, score_note = build_prompt_scores(args, train_data_names, train_raw_paths, diff_maps)

    for raw_path, train_name in zip(train_raw_paths, train_data_names):
        prompt2score_dict = data_name2prompt2score[train_name]
        if not prompt2score_dict:
            raise ValueError(f"No prompt scores found for `{train_name}`")

        first_prompt = next(iter(prompt2score_dict))
        if not isinstance(prompt2score_dict[first_prompt], dict):
            raise TypeError(f"Expected score dict for `{train_name}`, got {type(prompt2score_dict[first_prompt])}")

        local_valid_names = valid_data_names
        if "all_scores" in prompt2score_dict[first_prompt]:
            local_valid_names = list(prompt2score_dict[first_prompt]["all_scores"].keys())

        prompt2score = {prompt: float(item["score"]) for prompt, item in prompt2score_dict.items()}
        save_selected_dataset(
            args,
            train_name,
            raw_path,
            prompt2score_dict,
            prompt2score,
            pass_rate_maps,
            diff_maps,
            local_valid_names,
            score_note,
        )


if __name__ == "__main__":
    main()
