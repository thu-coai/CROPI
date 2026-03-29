import os 
import torch 
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import copy
import numpy as np
import json
import time
import math
from collections.abc import Sequence

from cropi.utils.rl_utils import grpo_loss, sft_loss,  generate_rollout_data, reward_fn_mathverify
from cropi.utils.rl_utils import generate_completions, compute_log_probs

from tqdm import tqdm

from argparse import ArgumentParser
import pickle

import gc

from trak.projectors import CudaProjector, ProjectionType

def parse_args():
    parser = ArgumentParser(description="Get the RL gradient of the function")
    parser.add_argument("--model_name_or_path", type=str, default=None, required=True, help="Path to the model")
    parser.add_argument("--base_model", type=str, default=None, required=True, help="Path to the base model for KL penalty")
    parser.add_argument("--rollout_data_path", type=str, required=True, help="Path to the data")
    parser.add_argument("--projection_method", type=str, choices=["identity", "trak_norm", "trak_redemacher"], default="trak_redemacher") # TODO: add more methods
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--recompute", action='store_true', help="Recompute the gradient")
    parser.add_argument("--process_batch_size", type=int, default=4, help="Batch size for the process of computing Projections. ")
    parser.add_argument("--micro_batch_size_for_grad_compute", type=int, default=2, help="Micro batch size for the gradient computation (responses per sample)")
    parser.add_argument("--use_valid_responses", action="store_true", help="If true, leave only the valid responses for gradient computation")
    parser.add_argument("--offload_gradient", action="store_true", help="If true, offload the gradient to CPU to save GPU memory")
    parser.add_argument("--offload_ref_model", action="store_true", help="If true, offload the reference model to CPU to save GPU memory")
    parser.add_argument("--max_tokens_per_forward", type=int, default=80000)

    # Arguments for the projector
    parser.add_argument("--model_id", type=int, default=0, help="Model ID for the projector")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for the projector")
    parser.add_argument("--max_batch_size", type=int, default=8, help="Max batch size for the projector")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the projector")
    parser.add_argument("--proj_dim", type=int, default=8192*4, help="Projection dimension for the projector")
    parser.add_argument("--sparse_dim", type=int, default=None, help="Sparse dimension for the gradient projection. If None, use the full gradient.")
    # sparse_dim = None means using the full gradient

    # Arguments for the training LOSS configuration
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "rft"], help="Type of loss to use for training")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations for the training configuration")
    parser.add_argument("--max_completion_length", type=int, default=2048, help="Max completion length for the training configuration")
    parser.add_argument("--beta", type=float, default=0.001, help="KL penalty for the training configuration")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for the training configuration")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clip value for the training configuration")
    parser.add_argument("--mu", type=float, default=1, help="Mu value for the training configuration")
    parser.add_argument("--cancel_ppo_clip", action='store_true', help="Cancel PPO clip in the loss function")
    parser.add_argument("--reweight_group_adv", action='store_true', help="Reweight the group advantage when computing the loss")
    parser.add_argument("--proj_note", type=str, default="", help="Additional note for the projection")

    parser.add_argument("--debug", action='store_true', help="Debug mode, will use a small subset of the data for testing")
    parser.add_argument("--debug_num", type=int, default=50, help="Number of samples to use in debug mode")
    parser.add_argument("--save_dir", type=str, required = False, help="Directory to save the identity projection results")

    args = parser.parse_args()

    assert os.path.exists(args.model_name_or_path), f"Model path {args.model_name_or_path} does not exist"
    assert os.path.exists(args.rollout_data_path), f"Rollout data path {args.rollout_data_path} does not exist"
    if args.output_path is None:

        if args.proj_note:
            proj_note = args.proj_note
        else:
            proj_note = f"{args.projection_method}_seed{args.seed}_mid{args.model_id}_projdim{int(args.proj_dim)}"
            if args.cancel_ppo_clip and args.reweight_group_adv:    # accurate gradient estimation 
                proj_note = f"{args.projection_method}_seed{args.seed}_mid{args.model_id}_projdim{int(args.proj_dim)}_acg"

            if args.sparse_dim is not None:
                proj_note += f"_sparse{args.sparse_dim}"

        args.output_path = args.rollout_data_path.replace(".json", f"_grad_{proj_note}.json")

    return args

def get_random_indices(total_dim, sparse_dim, seed=0):
    """ get random indices for sparse projection. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    begin_time = time.time()

    g = torch.Generator()
    g.manual_seed(seed)
    random_indices = torch.randperm(total_dim, generator=g)[:sparse_dim]
    # random_indices = sorted(random_indices)
    end_time = time.time()
    print(f"[DEBUG] Selected {sparse_dim} random indices from total {total_dim} dimensions in {end_time - begin_time:.2f} seconds. ")
    return random_indices

def get_vectorized_grads(model):
    """ obtain gradients. """
    print(f"[DEBUG] model.dtype: {model.dtype}")
    vectorized_grads = torch.cat(
        [p.grad.view(-1).to(model.device).to(model.dtype) for p in model.parameters() if p.grad is not None])

    return vectorized_grads

def compute_gradient_single_sample(
    args, sample, model, ref_model, tokenizer, training_config, 
    use_offline_responses=False, micro_batch_size=5, loss_type='grpo',
    reweight_group_adv=False, cancel_ppo_clip=False
):
    print(f"\n[DEBUG] Computing Gradient for Prompt: {sample['prompt'][0:50]}...")

    model.train()

    device = model.device
    rollout_model = ref_model if use_offline_responses else model

    # Prepare the data
    rewards_tensor = torch.tensor(sample["rewards"], dtype=torch.float32).to(device)
    num_generations = len(sample['responses'])
    max_completion_length = training_config['max_completion_length']
    
    # Calculate the statistics info
    with torch.no_grad():
        prompts = [sample['prompt']]
        offline_responses_list = [sample["responses"]]
        
        # generate the completions
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            rollout_model, tokenizer, prompts, num_generations, max_completion_length,
            load_offline_generation=True, offline_responses_list=offline_responses_list
        )
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        # compute the number of tokens 
        token_counts = attention_mask.sum(dim=1).cpu().numpy()
        total_token_count = token_counts.sum()
        
        # compute the statistics of rewards
        if reweight_group_adv and use_offline_responses:
            old_log_probs = compute_log_probs(rollout_model, input_ids, attention_mask, logits_to_keep)
            log_probs_current = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

            seq_len = completion_mask.sum(dim=1)
            old_log_prob_seq = (old_log_probs * completion_mask).sum(dim=1) / seq_len
            log_prob_seq = (log_probs_current * completion_mask).sum(dim=1) / seq_len
            prob_ratio = torch.exp(log_prob_seq - old_log_prob_seq)
            
            rewards_mean = (rewards_tensor * prob_ratio).sum() / prob_ratio.sum()
            rewards_std = torch.sqrt(((rewards_tensor - rewards_mean)**2 * prob_ratio).sum() / prob_ratio.sum())
        else:
            rewards_std, rewards_mean = torch.std_mean(rewards_tensor)
    
    # clear the cache of cuda
    torch.cuda.empty_cache()
    
    # create the batch
    def create_batches(token_counts, max_tokens):
        batches = []
        current_batch = []
        current_max_tokens = 0
        
        sorted_indices = np.argsort(-token_counts)
        
        for idx in sorted_indices:
            tokens = token_counts[idx]
            new_max = max(current_max_tokens, tokens)
            
            if new_max * (len(current_batch) + 1) <= max_tokens or not current_batch:
                current_batch.append(int(idx))
                current_max_tokens = new_max
            else:
                batches.append(current_batch)
                current_batch = [int(idx)]
                current_max_tokens = tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    batches = create_batches(token_counts, args.max_tokens_per_forward)
    print(f"[DEBUG] Created {len(batches)} batches from {num_generations} responses")

    model.zero_grad()

    clipped_tokens, total_tokens = 0, 0
    total_loss = 0
    
    for batch_idx, response_indices in enumerate(batches):
        batch_size = len(response_indices)
        print(f"[DEBUG] Processing batch {batch_idx+1}/{len(batches)}: {batch_size} responses")
        
        batch_subsample = {
            "prompt": sample['prompt'],
            "answer": sample['answer'],
            "responses": [sample['responses'][i] for i in response_indices],
            "rewards": [sample['rewards'][i] for i in response_indices]
        }
        
        # Generate rollout data
        batch_rollout_data = generate_rollout_data(
            rollout_model=rollout_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            batch_samples=[batch_subsample],
            num_generations=batch_size,
            max_completion_length=max_completion_length,
            load_offline_generation=use_offline_responses,
        )
        
        # Compute loss
        if loss_type == 'grpo':
            loss, avg_reward, _clipped_tokens, _total_tokens = grpo_loss(
                model=model,
                ref_model=ref_model,
                rollout_data=batch_rollout_data,
                tokenizer=tokenizer,
                reward_function=reward_fn_mathverify,
                beta=training_config['beta'],
                epsilon=training_config['epsilon'],
                baseline=rewards_mean,
                std=rewards_std,
                cancel_ppo_clip=cancel_ppo_clip
            )
            clipped_tokens += _clipped_tokens
            total_tokens += _total_tokens
        elif loss_type == 'rft':
            loss, avg_reward = sft_loss(
                model=model,
                rollout_data=batch_rollout_data,
                tokenizer=tokenizer,
                reward_function=reward_fn_mathverify,
            )
        
        if loss is None:
            print(f"[WARNING] Batch {batch_idx+1} produced None loss")
            continue
        
        print(f"[DEBUG] Batch {batch_idx+1} loss: {loss.item():.6f}")
        total_loss += loss.item()
        
        # Backward propagation - gradients will accumulate automatically
        loss.backward()
        
        # Clear memory (but not gradients!)
        del loss, batch_rollout_data
        torch.cuda.empty_cache()
        
        print(f"[DEBUG] GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    print(f"\n[DEBUG] Total loss across all batches: {total_loss:.6f}")
    # Pre-calculate parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Initialize gradient buffer
    _device = 'cpu' if args.offload_gradient else 'cuda'
    vectorized_grads = torch.zeros(param_count, dtype=torch.float32, device=_device)
    
    # Extract final accumulated gradients
    with torch.no_grad():
        offset = 0
        # grad_norm = 0
        num_params_with_grad = 0
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                numel = p.numel()
                if p.grad is not None:
                    grad_flat = p.grad.view(-1).to(torch.float32)
                    # grad_norm += p.grad.norm().item()
                    num_params_with_grad += 1
                    
                    if num_params_with_grad <= 5:  # Print info for the first few parameters
                        print(f"[DEBUG] {name}: grad_norm={p.grad.norm().item():.6f}, first5={grad_flat[:5].cpu().numpy()}")
                    
                    if args.offload_gradient:
                        vectorized_grads[offset:offset+numel] = grad_flat.cpu()
                    else:
                        vectorized_grads[offset:offset+numel] = grad_flat

                else:
                    print(f"[WARNING] Parameter {name} has no gradient")
                offset += numel

    print(f"\n[DEBUG] Gradient Norm:")
    print(f"[DEBUG] Extracted gradient shape: {vectorized_grads.shape} / First 50 values: {vectorized_grads[:50].cpu().numpy()}")
    print(f"[DEBUG] Gradient Norm: {vectorized_grads.norm().item():.6f} / Number of params with grad: {num_params_with_grad}/{len(list(model.parameters()))}")

    # Cleanup
    model.zero_grad()  # Clear gradients at the end
    torch.cuda.empty_cache()
    gc.collect()
    
    return vectorized_grads, None, clipped_tokens, total_tokens

def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing to reduce memory usage"""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    return model

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 torch_dtype=torch.bfloat16,
                                                 use_cache=False).to(device)
    model = enable_gradient_checkpointing(model)
    # ref_model = copy.deepcopy(model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype=torch.bfloat16,
                                                     use_cache=False).to(device)
    print(f"[INFO] After load model({model.device}), GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3)} GB")

    # ref_model = copy.deepcopy(model)
    print(f"[INFO] Loaded model : {args.model_name_or_path}.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    print(f"[INFO] Loaded tokenizer {args.model_name_or_path}.")
 
    # load rollout data
    data = []
    with open(args.rollout_data_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    print(f"[INFO] Loaded {len(data)} samples from {args.rollout_data_path}.")

    if args.debug:
        # For debugging, use a small subset of the data
        data = data[:args.debug_num]
        print(f"[DEBUG] Debug mode: using {len(data)} samples.")

    training_config = {
        'num_iterations': 1,
        'num_generations': args.num_generations, # number of generations for each sample
        'max_completion_length': args.max_completion_length,  # max completion length
        'beta': args.beta,  # KL penalty
        'learning_rate': args.learning_rate,
        'mu': args.mu,
        'epsilon': args.epsilon  # clip value
    }

    d_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total gradient dimension: {d_grad}")
    if args.sparse_dim is not None:
        assert args.sparse_dim < d_grad, "sparse_dim must be less than the total gradient dimension. "
        print(f"[INFO] Using sparse projection with sparse_dim {args.sparse_dim}. ")
        d_grad = args.sparse_dim

    # d_proj = 8192*4
    time1 = time.time()
    if args.projection_method in {"trak_redemacher", "trak_norm"} and device != "cuda":
        raise RuntimeError("TRAK projection currently requires CUDA. Use a CUDA device or switch to --projection_method identity.")

    if args.projection_method == 'trak_redemacher':
        projector = CudaProjector

        proj = projector(grad_dim=d_grad,
                    proj_dim=args.proj_dim,
                    seed=args.seed,
                    proj_type=ProjectionType.rademacher,
                    device=device,
                    dtype=model.dtype,
                    block_size=args.block_size,
                    max_batch_size=args.max_batch_size)
    elif args.projection_method == 'trak_norm':
        projector = CudaProjector

        proj = projector(grad_dim=d_grad,
                    proj_dim=args.proj_dim,
                    seed=args.seed,
                    proj_type=ProjectionType.normal,
                    device=device,
                    dtype=model.dtype,
                    block_size=args.block_size,
                    max_batch_size=args.max_batch_size)
    time2 = time.time()
    print(f"[DEBUG] After Load projector, GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3)} GB")
    print(f"[DEBUG] Time taken for projector initialization: {time2 - time1:.4f} seconds")

    def check_grad_valid(grad):
        """ Check if the gradient is valid, i.e. not NaN or Inf """
        if isinstance(grad, list):
            grad = torch.tensor(grad, device=model.device)  # Ensure grad is a tensor
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print(f"[ERROR] Gradient contains NaN or Inf.")
            return False
        if torch.all(grad == 0):
            print(f"[ERROR] Gradient is all zeros.")
            return False
        return True

    # NOTE: Filter out the samples that have been processed
    done_prompts = set()
    processed_data = []
    if not args.recompute and os.path.exists(args.output_path):
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                except Exception as e:
                    print(f"[ERR] Error loading item: {item}")
                    continue

                valid_grad = check_grad_valid(item["grad"])

                if valid_grad:
                    processed_data.append(item)
                    done_prompts.add(item["prompt"])

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            json.dump(item, f)
            f.write('\n')

    print(f"[DEBUG] Already processed prompts: {len(done_prompts)}. ")
    data = [sample for sample in data if sample["prompt"] not in done_prompts]
    print(f"[DEBUG] Remaining prompts: {len(data)}. ")

    # NOTE: Process the samples
    num_data = len(data)
    bsz = args.process_batch_size
    num_batch = num_data // bsz + int(num_data % bsz > 0)

    random_indices = None
    if args.sparse_dim is not None:
        random_indices = get_random_indices(d_grad, args.sparse_dim, seed=args.seed)
        print(f"[DEBUG] Using sparse projection with sparse_dim {args.sparse_dim}.")
    
    time_start = time.time()
    for i in tqdm(range(num_batch), unit="batch"):

        # get batch data 
        begin_idx = i * bsz
        end_idx = min(num_data, (i+1)*bsz)
        batch = data[begin_idx:end_idx]
        print(f"[INFO] Processing {i}/{num_batch} batch, batch size: {end_idx - begin_idx}")

        time1 = time.time()

        # NOTE: Got the batch gradients
        grad_batch = []
        batch_clipped_tokens = []
        batch_total_tokens = []

        for idx_sample, sample in enumerate(batch):
            if args.use_valid_responses: # leave the responses that have correct format, i.e. \\box{}
                valid_responses, valid_rewards = [], []
                for response, reward in zip(sample["responses"], sample["rewards"]):
                    if response and '\\boxed' in response:
                        valid_responses.append(response)
                        valid_rewards.append(reward)
                sample["responses"] = valid_responses
                sample["rewards"] = valid_rewards
                print(f"[DEBUG] Valid responses number: {len(sample['responses'])}")
            
            if len(sample["responses"]) == 0:
                print(f"[DEBUG] No valid responses for prompt {sample['prompt']}.")
                continue
            training_config["num_generations"] = num_response =  len(sample["responses"])
            print("[DEBUG] Num generations for this sample: ", num_response)

            # NOTE: Compute the gradient
            # TODO: handle the case the sample is too large: seperate the sample into smaller chunks
            micro_batch_size_for_grad_compute = args.micro_batch_size_for_grad_compute

            if args.projection_method == "identity":
                save_dir = args.save_dir
                os.makedirs(save_dir, exist_ok=True)
                file_name = f"{i}_{idx_sample}_grad.pkl"
                # save prompt & gradient to pkl file
                save_path_for_grad = os.path.join(save_dir, file_name)

            grad, _, clipped_tokens, total_tokens = compute_gradient_single_sample(args, sample, model, ref_model, tokenizer, training_config,
                                                        use_offline_responses=True, 
                                                        micro_batch_size=micro_batch_size_for_grad_compute,
                                                        loss_type=args.loss_type,
                                                        reweight_group_adv=args.reweight_group_adv,
                                                        cancel_ppo_clip=args.cancel_ppo_clip,
                                                        )

            print(f"\n[DEBUG] Gradient values: {grad[0:1000]}.\nClipped tokens: {clipped_tokens}\nTotal tokens: {total_tokens}\n")

            if args.sparse_dim is not None:
                grad = grad[random_indices.to(grad.device)]  # select the sparse indices
                assert grad.shape[0] == args.sparse_dim, f"Gradient shape {grad.shape} does not match sparse_dim {args.sparse_dim}. "
                print(f"[DEBUG] After sparse selection, grad shape: {grad.shape}. ")

            grad = grad.to('cpu')   # move gradient to CPU device 

            # add data to the batch
            grad_batch.append(grad)
            batch_clipped_tokens.append(clipped_tokens)
            batch_total_tokens.append(total_tokens)

            if args.projection_method == "identity":
                with open(save_path_for_grad, 'wb') as f:
                    pickle.dump({
                        "prompt": sample['prompt'],
                        "grad": grad.cpu().to(torch.float16),
                        "clipped_tokens": clipped_tokens,
                        "total_tokens": total_tokens,
                    }, f)
                print(f"[DEBUG] Saved identity gradient to {save_path_for_grad}")

            if args.debug:
                # time.sleep(60)   # debug mode, sleep a while to view the output
                time.sleep(0.5)   # debug mode, sleep a while to view the output
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time2 = time.time()
        print(f"[DEBUG] After calc gradient ({model.device}), GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3)} GB")
        print(f"[DEBUG-TIME] Time taken for gradient computation: {time2 - time1:.4f} seconds")
        
        # NOTE: Calculate the batch projections
        if args.projection_method == "identity":
            # grad_projection = copy.deepcopy(grad)
            # grad_projection = grad_projection.to(torch.float16)
            raise NotImplementedError() # save the identity gradient above
        elif args.projection_method in ["trak_redemacher", "trak_norm"]:    # trak for projection
            print(f"[DEBUG] Projector device: {proj.device}")
            print(f"[DEBUG] grad.device: {grad.device}, dtype: {grad.dtype}, grad.shape: {grad.shape}")
            # time21 = time.time()

            print(f"[DEBUG] grad_batch length: {len(grad_batch)}; grad_batch[0].shape: {grad_batch[0].shape}, dtype: {grad_batch[0].dtype}, device: {grad_batch[0].device}")
            print(f"[DEBUG-CUDA] current cuda memory allocated: {torch.cuda.memory_allocated() / (1024**3)} GB")
            grad_batch = torch.stack(grad_batch)  # stack the gradients
            print(f"[DEBUG-CUDA] current cuda memory allocated: {torch.cuda.memory_allocated() / (1024**3)} GB")
            grad_batch = grad_batch.to(torch.float16)
            grad_batch = grad_batch.to("cuda")  # move to GPU
            print(f"[DEBUG-CUDA] current cuda memory allocated: {torch.cuda.memory_allocated() / (1024**3)} GB")

            grad_projections = proj.project(
                grad_batch, 
                model_id=args.model_id
            )
            print(f"[DEBUG] grad_projection.device: {grad_projections.device}, dtype: {grad_projections.dtype}, grad_projection.shape: {grad_projections.shape}")


        torch.cuda.synchronize()
        time3 = time.time()
        print(f"[DEBUG-CUDA] After computing projection ({model.device}), GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3)} GB")
        print(f"[DEBUG-TIME] Time taken for projection: {time3 - time2:.4f} seconds")

        # NOTE: Save the result of gradient projections
        for i in range(len(batch)):
            sample = batch[i]
            clipped_tokens = batch_clipped_tokens[i]
            total_tokens = batch_total_tokens[i]
            save_item = {
                "prompt": sample['prompt'],
                "grad": grad_projections[i].cpu().numpy().tolist(),
                "clipped_tokens": clipped_tokens,
                "total_tokens": total_tokens
            }

            # if grad is all zero or has nan, continue
            if np.isnan(save_item["grad"]).any() or np.isinf(save_item["grad"]).any():
                print(f"[WARNING] Gradient for prompt {sample['prompt']} is NaN or Inf. Skipping.")
                continue
            if np.all(save_item["grad"] == 0):
                print(f"[WARNING] Gradient for prompt {sample['prompt']} is all zeros. Skipping.")
                continue

            # NOTE: save as json format

            if args.debug:
                # not save the debug data
                continue

            with open(args.output_path, 'a') as f:
                json.dump(save_item, f)
                f.write('\n')

        time4 = time.time()
        print(f"[DEBUG] Time taken for saving the projections: {time4-time3:.4f} seconds. ")

        pass

    time_end = time.time()
    print(f"[DEBUG-TIME] Total time taken: {time_end - time_start:.4f} seconds. ")

if __name__ == '__main__':
    main()
