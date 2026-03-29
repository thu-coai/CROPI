import torch
import torch.nn as nn
import numpy as np

def selective_log_softmax(logits, input_ids):
    """
    Computes log probabilities for specific tokens in the vocabulary.

    Args:
        logits (torch.Tensor): The raw logits output from the model.
        input_ids (torch.Tensor): The token IDs for which we want the log probabilities.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens.

    Explanation:
        1. Applies log softmax to convert logits to log probabilities over the vocabulary.
        2. Uses gather to extract only the log probabilities corresponding to the input_ids.
        3. Removes the extra dimension to match the original shape of input_ids.
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # .shape: [batch_size, seq_length]

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """
    Computes the log probabilities for a batch of tokens.

    Args:
        model: The language model.
        input_ids (torch.Tensor): Token IDs for input sequences.
        attention_mask (torch.Tensor): Attention mask for input sequences.
        logits_to_keep (int): Number of tokens to keep from the end of the sequence.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens.

    Explanation:
        1. Gets logits from the model for the input sequence.
        2. Selects logits for all tokens except the last one (as we predict next tokens).
        3. Selects only the last 'logits_to_keep' tokens from both logits and input_ids.
        4. Computes log probabilities for these tokens using selective_log_softmax.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after the EOS token.

    Args:
        completion_ids (torch.Tensor): Token IDs of the generated completions.
        eos_token_id (int): The ID of the end-of-sequence token.

    Returns:
        torch.Tensor: A binary mask with 1s for valid tokens and 0s after the EOS token.

    Explanation:
        1. Identifies positions where EOS tokens occur in each sequence.
        2. Finds the index of the first EOS token in each sequence.
        3. Creates a mask where positions before and including the first EOS are 1, others are 0.
        4. If no EOS token is found in a sequence, all positions are set to 1.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32,
                         load_offline_generation=False,
                         offline_responses_list=None):
    """
    Generates multiple completions for each prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer for encoding and decoding text.
        prompts (list): List of text prompts.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of tokens to generate.
        load_offline_generation (bool): Flag to load offline generated responses.
        offline_responses (list of list of str): Pre-generated responses if load_offline_generation is True. If not provided, the function will generate completions using the model.

    Returns:
        tuple: Containing prompt IDs, prompt mask, completion IDs, and completion mask.

    Explanation:
        1. Encodes the prompts and moves them to the appropriate device.
        2. Repeats each prompt num_generations times to generate multiple completions.
        3. Generates completions using the model with specified parameters.
        4. Extracts the completion IDs (excluding the prompt tokens).
        5. Creates a mask for the completions using create_completion_mask.
    """
    if load_offline_generation:
        assert offline_responses_list is not None, "offline_responses must be provided if load_offline_generation is True"
        assert len(offline_responses_list) == len(prompts), "offline_responses must have the same length as prompts"
        assert len(offline_responses_list[0]) == num_generations, "Each prompt must have the same number of offline responses"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    # print(f"[DEBUG] Input batch size: {prompt_ids.size(0)}, Device before model: {prompt_ids.device}")
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0) # [x,y,z,...] -> [x,x,x,x,y,y,y,y,z,z,z,z,...]
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    if not load_offline_generation:
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False
        )
        # print(f"[DEBUG] Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
        completion_ids = outputs[:, prompt_length:]
        completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    else:
        # Load offline generation, betch encode, right padding
        completion_ids = []
        completion_mask = []
        for offline_responses in offline_responses_list:
            assert len(offline_responses) == num_generations, "Each prompt must have the same number of offline responses"
            assert len(offline_responses) > 0, f"Each prompt must have at least one offline response, but got {len(offline_responses)}"
            _completion_ids = tokenizer(offline_responses, return_tensors="pt", padding=True, padding_side="right")["input_ids"]
            _completion_ids = _completion_ids.to(device)
            _completion_mask = create_completion_mask(_completion_ids, tokenizer.eos_token_id)
            completion_ids.append(_completion_ids)
            completion_mask.append(_completion_mask)
        completion_ids = torch.cat(completion_ids, dim=0)
        completion_mask = torch.cat(completion_mask, dim=0)
        assert completion_ids.size(0) == prompt_ids.size(0), "The number of completions must match the number of prompts"
        assert completion_ids.size(0) == len(prompts) * num_generations, "The number of completions must match the number of prompts times num_generations"
        # print(f"[DEBUG] Completion_ids.shape: {completion_ids.shape}, Device after model: {completion_ids.device}")

    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(rollout_model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length,
                          load_offline_generation=False):
    """
    Generates data for GRPO rollouts including completions and log probabilities.

    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        tokenizer: The tokenizer for encoding and decoding text.
        batch_samples (list): Batch of training samples.
        num_generations (int): Number of completions to generate per sample.
        max_completion_length (int): Maximum completion length.

    Returns:
        dict: Dictionary containing all data needed for GRPO updates.

    Explanation:
        1. Extracts prompts and expected answers from the batch samples.
        2. Generates completions using the current policy model.
        3. Combines prompt and completion tokens.
        4. Computes log probabilities from both the policy model and reference model.
        5. Formats completions for reward calculation.
        6. Repeats prompts and answers to match the number of generated completions.
        7. Returns all data needed for GRPO loss calculation.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    if load_offline_generation:
        assert len(batch_samples[0]) >= 2 or "responses" in batch_samples[0], "If load_offline_generation is True, the batch_samples 'must' contain 'responses' key"
        offline_responses_list = [sample["responses"] if isinstance(sample, dict) else sample[2] for sample in batch_samples]
        assert len(offline_responses_list) == len(prompts), "offline_responses must have the same length as prompts"

    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            rollout_model, tokenizer, prompts, num_generations, max_completion_length,
            load_offline_generation=load_offline_generation,
            offline_responses_list=offline_responses_list if load_offline_generation else None
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        old_log_probs = compute_log_probs(rollout_model, input_ids, attention_mask, logits_to_keep) # rollout model
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    ret = {
        "input_ids": input_ids, # (batch_size * num_generations, seq_length)
        "attention_mask": attention_mask,   # (batch_size * num_generations, seq_length)
        "completion_mask": completion_mask, # (batch_size * num_generations, completion_length)
        "old_log_probs": old_log_probs, # rollout log probs # (batch_size * num_generations, completion_length)
        "ref_log_probs": ref_log_probs, # reference log probs # (batch_size * num_generations, completion_length)
        "formatted_completions": formatted_completions, # list of list of dict (batch_size * num_generations, 1, {"content": str})
        "repeated_prompts": repeated_prompts,   # list of str (batch_size * num_generations, )
        "repeated_answers": repeated_answers,  # list of str (batch_size * num_generations, )
        "logits_to_keep": logits_to_keep,   # int
        "batch_size": len(prompts), # int
        "num_generations": num_generations  # int
    }
    if isinstance(batch_samples[0], dict) and 'rewards' in batch_samples[0]:
        rewards = []
        for sample in batch_samples:
            rewards += sample["rewards"]
        ret["rewards"] = rewards
        assert len(rewards) == len(prompts) * num_generations, f"The number of rewards must match the number of prompts times num_generations, but got {len(rewards)} and {len(prompts)} * {num_generations}"

    return ret

def grpo_loss(model, ref_model, rollout_data, tokenizer, reward_function, 
              beta=0.01, 
              epsilon=0.2,
              baseline=None,
              std=None,
              cancel_ppo_clip=False,
):
    """
    Computes the GRPO loss for updating the policy model...
    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        rollout_data (dict): Data generated by generate_rollout_data.
        tokenizer: The tokenizer for encoding and decoding text.
        reward_function: Function that calculates rewards for completions.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter for PPO.

    Returns:
        torch.Tensor: The GRPO loss to be minimized.

    Explanation:
        1. Computes current token log probabilities using the policy model.
        2. Calculates the probability ratio between current and old policies.
        3. Computes rewards using the provided reward_function.
        4. Calculates advantages by standardizing rewards within each prompt.
        5. Computes the PPO surrogate objective with clipping.
        6. Calculates the KL divergence between reference and policy models.
        7. Combines surrogate loss and KL penalty.
        8. Averages the loss across all tokens and batches.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]

    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    print(f"[DEBUG] Old log probs shape: {old_log_probs.shape}")

    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)  # (batch_size * num_generations, seq_length)
    print(f"[DEBUG] Ratio shape: {ratio.shape}, Old log probs shape: {old_log_probs.shape}, Token log probs shape: {token_log_probs.shape}")  # Debug shapes
    print(f"[DEBUG] Ratio: {ratio}")  # Debug ratio values
    raw_rewards = rollout_data.get("rewards", None)
    if raw_rewards is None:
        raw_rewards = reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"])
    rewards = torch.tensor(
        raw_rewards,
        dtype=torch.float32,
        device=device
    )
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    print(f"[DEBUG] | Rewards: {rewards.shape} | batch_size: {batch_size} | num_generation: {num_generations} | ")  # Debug rewards
    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()

    # print("[DEBUG] Average Reward:", avg_reward)
    if baseline is None:
        baseline = rewards.mean(dim=1)
    if std is None:
        std = rewards.std(dim=1)

    bs = baseline.repeat_interleave(num_generations).to(rewards.device)
    denom = std.repeat_interleave(num_generations).to(rewards.device)
    print(f"[DEBUG] Normalization:\n\t\tbaseline: {bs}\n\t\tdenominator: {denom}") 

    advantages = ((rewards.view(-1) - bs) / (denom + 1e-4)).unsqueeze(1)    # (batch_size * num_generations, 1)
    # print(f"[DEBUG] Advantages shape: {advantages.shape}, Mean rewards shape: {mean_rewards.shape}, Std rewards shape: {std_rewards.shape}")  # Debug shapes
    surr1 = ratio * advantages
    if not cancel_ppo_clip:
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

        # print(f"[DEBUG] Surrogate 1: {surr1}, Surrogate 2: {surr2}")  # Debug surrogate values
        surrogate_loss = torch.min(surr1, surr2)
        # NOTE: stat the clipped tokens
        clipped = torch.where(surr2 < surr1, 1.0, 0.0)  # 当surr2 < surr1时表示被clip
        clipped_tokens = (clipped * completion_mask).sum().item()  # 只统计completion部分的token
    else:
        surrogate_loss = surr1
        clipped_tokens = 0
    total_tokens = completion_mask.sum().item() # 统计所有 token

    # kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    # per_token_loss = surrogate_loss - beta * kl   # NOTE: remove kl loss
    per_token_loss = surrogate_loss
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    return loss, avg_reward, clipped_tokens, total_tokens

def sft_loss(model, rollout_data, tokenizer, reward_function):  # RFT / REINFORCE
    """
    Computes the supervised fine-tuning (SFT) loss for the model.
    Args:
        model: The language model to be fine-tuned.
        rollout_data (dict): Data generated by generate_rollout_data.
        tokenizer: The tokenizer for encoding and decoding text.
    Returns:
        torch.Tensor: The SFT loss to be minimized.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]

    # ratio = torch.exp(token_log_probs - old_log_probs)
    
    raw_rewards = rollout_data.get("rewards", None)
    if raw_rewards is None:
        raw_rewards = reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"])
    rewards = torch.tensor(
        raw_rewards,
        dtype=torch.float32,
        device=device
    )   # (batch_size * num_generations, )
    avg_reward = rewards.mean().item()
    print("Average Reward:", avg_reward)

    reward_positive_indexes = rewards > 0   # (num_samples * num_generations, )
    num_positive_samples = reward_positive_indexes.sum().item()
    print(f"Reward positive indexes: {reward_positive_indexes}, Number of positive samples: {num_positive_samples}")  # Debug indexes and count
    if num_positive_samples == 0:
        print("Warning: No positive rewards found. Returning zero loss.")
        return None, avg_reward

    input_ids_pos = input_ids[reward_positive_indexes]  # (num_positive_samples, seq_length)
    attention_mask_pos = attention_mask[reward_positive_indexes]  # (num_positive_samples, seq_length)
    completion_mask_pos = completion_mask[reward_positive_indexes]  # (num_positive_samples,
    token_log_probs_pos = compute_log_probs(model, input_ids_pos, attention_mask_pos, logits_to_keep)   # (batch_size, seq_length
    print(f"Token log probs positive shape: {token_log_probs_pos.shape}, Completion mask positive shape: {completion_mask_pos.shape}")  # Debug shapes

    loss = -((token_log_probs_pos * completion_mask_pos).sum(dim=1) / completion_mask_pos.sum(dim=1)).mean()

    return loss, avg_reward

try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

def compute_score(model_output: str, ground_truth: str) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception as e:
        pass

    return ret_score

def reward_fn_mathverify(prompts, completions, answer):
    """
    Args:
        prompts (list[str]): List of prompt texts
        completions (list[list[dict]]): List of completion dictionaries
        answer (list[str]): List of expected answers

    Returns:
        list[float]: Combined rewards for each prompt-completion pair
    """
    responses = [completion[0]['content'] for completion in completions]
    
    rewards = []
    for prompt, response, ans in zip(prompts, responses, answer):
        print(f"Prompt: {prompt}, Response: {response}, Answer: {ans}")

        score = compute_score(response, ans)
        print(f"Score: {score}")
        rewards.append(score)
    return rewards
