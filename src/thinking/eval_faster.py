import os
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import thinking.data as data_module
from thinking.grader import extract_solution, normalize_latex_string


def load_all_datasets(mode="eval", turn_off_thinking=False, dataset_names=None, dataset_paths=None):
    """
    Load in datasets specified by user.

    Args:
        mode (str): which split to take from
        turn_off_thinking (bool): whether to append /no_think to end of prompt
        dataset_names (list): list of dataset names to load (e.g., ['calc_mixed', 'gsm8k'])
        dataset_paths (list): list of paths corresponding to each dataset, use "none" for default path

    Returns:
        dict mapping dataset_name -> dataset
    """
    datasets = {}

    if not dataset_names:
        print("No datasets were passed")
        return datasets

    # If dataset_paths not provided, use "none" for all
    if dataset_paths is None:
        dataset_paths = ["none"] * len(dataset_names)

    # Ensure lists are same length
    if len(dataset_names) != len(dataset_paths):
        raise ValueError(f"dataset_names ({len(dataset_names)}) and dataset_paths ({len(dataset_paths)}) must have same length")

    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        try:
            load_fn = getattr(data_module, f"load_{dataset_name}", None)
            if load_fn is None:
                print(f"Warning: No load function 'load_{dataset_name}' found in thinking.data, skipping...")
                continue

            # Build kwargs based on what the function accepts
            kwargs = {"mode": mode}

            # Add data_path if specified
            if dataset_path.lower() != "none":
                kwargs["data_path"] = dataset_path

            # Try with turn_off_thinking first, fall back without if it fails
            try:
                kwargs["turn_off_thinking"] = turn_off_thinking
                datasets[dataset_name] = load_fn(**kwargs)
            except TypeError:
                # Function might not support turn_off_thinking
                del kwargs["turn_off_thinking"]
                try:
                    datasets[dataset_name] = load_fn(**kwargs)
                except TypeError:
                    # Function might not support data_path either
                    datasets[dataset_name] = load_fn(mode=mode)

            print(f"Loaded {dataset_name}: {len(datasets[dataset_name])} examples")

        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {e}")
            continue

    return datasets


def compute_pass_at_k(n, c, k):
    """
    Calculate pass@k metric.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k

    Returns:
        pass@k score
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def calculate_pass_at_k_metrics(results, k_values):
    """
    Calculate pass@k metrics for different k values from evaluation results.

    Args:
        results: dict mapping (dataset_name, problem_id) -> list of correctness bools
        k_values: list of k values to compute pass@k for

    Returns:
        dict of metrics by dataset
    """
    metrics = defaultdict(lambda: defaultdict(list))

    for (dataset_name, problem_id), correctness_list in results.items():
        n = len(correctness_list)
        c = sum(correctness_list)

        for k in k_values:
            if k <= n:
                pass_k = compute_pass_at_k(n, c, k)
                metrics[dataset_name][f"pass@{k}"].append(pass_k)

    # Average across all problems in each dataset
    final_metrics = {}
    for dataset_name, dataset_metrics in metrics.items():
        final_metrics[dataset_name] = {}
        for metric_name, values in dataset_metrics.items():
            final_metrics[dataset_name][metric_name] = np.mean(values) * 100

    return final_metrics


def generate_with_thinking_budget(
    llm,
    tokenizer,
    prompts: List[str],
    thinking_budget: int,
    response_budget: int,
    thinking_end_token: str = "</think>",
    thinking_end_token_id: int = 151668,
    temperature: float = 1.0,
    n_samples: int = 1,
    lora_request=None,
) -> List[List[Dict]]:
    """
    Two-phase generation with thinking and response budgets.

    Phase 1: Generate up to thinking_budget tokens, looking for thinking_end_token
    Phase 2: Truncate at delimiter (or append if missing), then generate response_budget tokens.

    Args:
        llm: vLLM LLM instance
        tokenizer: tokenizer for encoding/decoding
        prompts: list of prompts to generate from
        thinking_budget: max tokens for thinking phase
        response_budget: max tokens for response phase
        thinking_end_token: delimiter string marking end of thinking (default: "</think>")
        thinking_end_token_id: token ID of the thinking delimiter (default: 151668 for Qwen3)
        temperature: sampling temperature
        n_samples: number of samples per prompt
        lora_request: optional LoRA request

    Returns:
        List of lists, where each inner list contains n_samples dicts with:
            - 'full_output': complete generated text
            - 'thinking_output': thinking phase output
            - 'response_output': response phase output
            - 'delimiter_forced': whether delimiter was force-appended
            - 'thinking_tokens': number of tokens in thinking phase
            - 'response_tokens': number of tokens in response phase
    """
    from vllm import SamplingParams

    # Phase 1: Generate thinking tokens
    thinking_params = SamplingParams(
        temperature=temperature,
        max_tokens=thinking_budget,
        skip_special_tokens=False,
        n=n_samples,
    )

    if lora_request:
        thinking_outputs = llm.generate(prompts, thinking_params, lora_request=lora_request)
    else:
        thinking_outputs = llm.generate(prompts, thinking_params)

    # Process each prompt's outputs and prepare for phase 2
    all_results = []
    prompts_for_phase2 = []
    phase2_mapping = []  # Track which (prompt_idx, sample_idx) each phase2 prompt corresponds to

    for prompt_idx, (prompt, thinking_output) in enumerate(zip(prompts, thinking_outputs)):
        prompt_results = []

        for sample_idx, sample_output in enumerate(thinking_output.outputs):
            thinking_text = sample_output.text
            token_ids = list(sample_output.token_ids)

            # Find delimiter by token ID
            delimiter_found = thinking_end_token_id in token_ids
            delimiter_forced = not delimiter_found

            if delimiter_forced:
                # Delimiter not found, force append it
                thinking_text_with_delim = thinking_text + thinking_end_token
                thinking_content = thinking_text
                tokens_after_delim = 0
                thinking_token_count = len(token_ids)
            else:
                # Delimiter found - truncate at delimiter
                delim_idx = token_ids.index(thinking_end_token_id)
                tokens_after_delim = len(token_ids) - delim_idx - 1
                thinking_token_count = delim_idx + 1  # Including delimiter

                # Truncate text at delimiter
                delim_pos = thinking_text.find(thinking_end_token)
                if delim_pos != -1:
                    thinking_text_with_delim = thinking_text[:delim_pos + len(thinking_end_token)]
                    thinking_content = thinking_text[:delim_pos]
                else:
                    # Fallback: decode tokens up to and including delimiter
                    thinking_text_with_delim = tokenizer.decode(token_ids[:delim_idx + 1], skip_special_tokens=False)
                    thinking_content = tokenizer.decode(token_ids[:delim_idx], skip_special_tokens=False)

            # Prepare for phase 2
            phase2_prompt = prompt + thinking_text_with_delim
            prompts_for_phase2.append(phase2_prompt)
            phase2_mapping.append((prompt_idx, sample_idx))

            # Store intermediate result
            prompt_results.append({
                'thinking_output': thinking_content,
                'delimiter_forced': delimiter_forced,
                'thinking_tokens': thinking_token_count,
                'tokens_after_delim': tokens_after_delim,
                'phase2_prompt': phase2_prompt,
            })

        all_results.append(prompt_results)

    # Phase 2: Generate response tokens for all sequences in one batched call
    response_params = SamplingParams(
        temperature=temperature,
        max_tokens=response_budget,
        skip_special_tokens=False,
        n=1,
    )

    if lora_request:
        phase2_raw_outputs = llm.generate(prompts_for_phase2, response_params, lora_request=lora_request)
    else:
        phase2_raw_outputs = llm.generate(prompts_for_phase2, response_params)

    phase2_outputs = [output.outputs[0] for output in phase2_raw_outputs]

    # Combine results
    for phase2_idx in range(len(prompts_for_phase2)):
        prompt_idx, sample_idx = phase2_mapping[phase2_idx]
        result = all_results[prompt_idx][sample_idx]
        thinking_text_with_delim = result['phase2_prompt'][len(prompts[prompt_idx]):]

        output = phase2_outputs[phase2_idx]
        response_text = output.text
        response_tokens = len(output.token_ids)

        result['response_output'] = response_text
        result['response_tokens'] = response_tokens
        result['full_output'] = thinking_text_with_delim + response_text

        # Clean up intermediate data
        del result['phase2_prompt']

    return all_results


def score_vllm_with_budget(
    model_name: str,
    save_name: str,
    save_path: str,
    batch_size: int,
    temperature: float,
    thinking_budget: int,
    response_budget: int,
    thinking_end_token: str = "</think>",
    thinking_end_token_id: int = 151668,
    mode: str = "eval",
    lora_path: Optional[str] = None,
    max_lora_rank: int = 64,
    gpu_memory_utilization: float = 0.9,
    n_samples: int = 1,
    k_values: Optional[List[int]] = None,
    dataset_names: Optional[List[str]] = None,
    dataset_paths: Optional[List[str]] = None,
    turn_off_thinking: bool = False,
):
    """
    Evaluate model with thinking and response budget constraints.

    Args:
        model_name: HuggingFace model name or local path
        save_name: name for output file
        save_path: directory to save results
        batch_size: batch size for generation
        temperature: sampling temperature
        thinking_budget: max tokens for thinking phase
        response_budget: max tokens for response phase
        thinking_end_token_id: token ID of thinking delimiter (default: 151668 for Qwen3)
        thinking_end_token: delimiter marking end of thinking
        mode: data split (train/eval)
        lora_path: optional LoRA adapter path
        max_lora_rank: maximum LoRA rank
        gpu_memory_utilization: GPU memory utilization
        n_samples: samples per prompt for pass@k
        k_values: k values for pass@k metrics
        dataset_names: list of dataset names to evaluate
        dataset_paths: list of paths for each dataset (use "none" for default)
        turn_off_thinking: whether to turn off thinking mode in prompts
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    if k_values is None:
        k_values = [1]

    # Default to calc_mixed if no datasets specified
    if dataset_names is None:
        dataset_names = ["calc_mixed"]

    # Load datasets
    datasets = load_all_datasets(
        mode=mode,
        turn_off_thinking=turn_off_thinking,
        dataset_names=dataset_names,
        dataset_paths=dataset_paths,
    )

    if not datasets:
        raise ValueError("No datasets were loaded. Check dataset_names parameter.")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Initialize vLLM
    print(f"Initializing vLLM with model: {model_name}")
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }

    # Handle lora_path="none" as no LoRA
    if lora_path and lora_path.lower() != "none":
        print(f"Enabling LoRA with adapter: {lora_path}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
    else:
        lora_path = None

    llm = LLM(**llm_kwargs)

    # Setup LoRA request
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("eval_adapter", 1, lora_path)

    # Prepare save path
    os.makedirs(save_path, exist_ok=True)
    save_file = f"{save_name}_{mode}_tb{thinking_budget}_rb{response_budget}.jsonl"
    full_save_path = os.path.join(save_path, save_file)
    print(f"Saving results to: {full_save_path}")

    # Track results
    pass_at_k_results = defaultdict(list)
    dataset_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    # Stats tracking
    total_delimiter_forced = 0
    total_thinking_tokens = 0
    total_response_tokens = 0
    total_samples = 0

    with open(full_save_path, "w") as file:
        for dataset_name, dataset in datasets.items():
            print(f"\n{'='*40}")
            print(f"Evaluating dataset: {dataset_name}")
            print(f"{'='*40}")

            num_batches = (len(dataset) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(dataset))
                print(f"{dataset_name} - batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx})")

                prompts = [dataset[j]["prompt"] for j in range(start_idx, end_idx)]
                ground_truths = [dataset[j]["ground_truth"] for j in range(start_idx, end_idx)]

                # Generate with budget constraints
                batch_results = generate_with_thinking_budget(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    thinking_budget=thinking_budget,
                    response_budget=response_budget,
                    thinking_end_token=thinking_end_token,
                    thinking_end_token_id=thinking_end_token_id,
                    temperature=temperature,
                    n_samples=n_samples,
                    lora_request=lora_request,
                )

                # Process results
                for problem_idx, (prompt, ground_truth, results) in enumerate(
                    zip(prompts, ground_truths, batch_results)
                ):
                    global_problem_id = start_idx + problem_idx
                    problem_correct = []

                    for sample_idx, result in enumerate(results):
                        full_output = result['full_output']
                        model_answer = extract_solution(full_output)

                        # Compare answers
                        gt_normalized = normalize_latex_string(str(ground_truth))
                        if model_answer:
                            answer_normalized = normalize_latex_string(model_answer)
                            is_correct = gt_normalized == answer_normalized
                        else:
                            is_correct = False

                        if sample_idx == 0:
                            if is_correct:
                                dataset_scores[dataset_name]["correct"] += 1

                        problem_correct.append(is_correct)

                        # Update stats
                        total_delimiter_forced += int(result['delimiter_forced'])
                        total_thinking_tokens += result['thinking_tokens']
                        total_response_tokens += result['response_tokens']
                        total_samples += 1

                        # Save result
                        file.write(json.dumps({
                            "dataset": dataset_name,
                            "problem_id": global_problem_id,
                            "sample_id": sample_idx,
                            "prompt": prompt,
                            "ground_truth": str(ground_truth),
                            "thinking_output": result['thinking_output'],
                            "response_output": result['response_output'],
                            "full_output": full_output,
                            "model_answer": model_answer,
                            "correct": is_correct,
                            "delimiter_forced": result['delimiter_forced'],
                            "thinking_tokens": result['thinking_tokens'],
                            "response_tokens": result['response_tokens'],
                            "tokens_after_delim": result['tokens_after_delim'],
                        }))
                        file.write("\n")

                    pass_at_k_results[(dataset_name, global_problem_id)] = problem_correct
                    dataset_scores[dataset_name]["total"] += 1

            # Print per-dataset summary
            ds = dataset_scores[dataset_name]
            acc = 100 * ds["correct"] / ds["total"] if ds["total"] > 0 else 0
            print(f"{dataset_name}: {ds['correct']}/{ds['total']} correct ({acc:.2f}%)")

    # Print overall summary
    total_correct = sum(s["correct"] for s in dataset_scores.values())
    total_problems = sum(s["total"] for s in dataset_scores.values())

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Datasets evaluated: {list(datasets.keys())}")
    print(f"\nPer-dataset results:")
    for ds_name in sorted(dataset_scores.keys()):
        ds = dataset_scores[ds_name]
        acc = 100 * ds["correct"] / ds["total"] if ds["total"] > 0 else 0
        print(f"  {ds_name}: {ds['correct']}/{ds['total']} ({acc:.2f}%)")

    print(f"\nOverall: {total_correct}/{total_problems} ({100*total_correct/total_problems:.2f}%)")
    print(f"\nBudget Stats:")
    print(f"  Thinking budget: {thinking_budget} tokens")
    print(f"  Response budget: {response_budget} tokens")
    print(f"  Delimiter forced: {total_delimiter_forced}/{total_samples} ({100*total_delimiter_forced/total_samples:.2f}%)")
    print(f"  Avg thinking tokens: {total_thinking_tokens/total_samples:.1f}")
    print(f"  Avg response tokens: {total_response_tokens/total_samples:.1f}")

    # Calculate pass@k if multiple samples
    if n_samples > 1:
        pass_k_metrics = calculate_pass_at_k_metrics(pass_at_k_results, k_values)

        print("\n" + "-" * 40)
        print("Pass@k Metrics:")
        for ds_name in sorted(pass_k_metrics.keys()):
            for k in sorted(k_values):
                if f"pass@{k}" in pass_k_metrics[ds_name]:
                    print(f"  {ds_name} pass@{k}: {pass_k_metrics[ds_name][f'pass@{k}']:.2f}%")

        # Save metrics
        metrics_file = full_save_path.replace(".jsonl", "_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump({
                "per_dataset_accuracy": {
                    ds_name: 100 * ds["correct"] / ds["total"] if ds["total"] > 0 else 0
                    for ds_name, ds in dataset_scores.items()
                },
                "overall_accuracy": 100 * total_correct / total_problems,
                "pass_at_k": pass_k_metrics,
                "budget_stats": {
                    "thinking_budget": thinking_budget,
                    "response_budget": response_budget,
                    "delimiter_forced_rate": total_delimiter_forced / total_samples,
                    "avg_thinking_tokens": total_thinking_tokens / total_samples,
                    "avg_response_tokens": total_response_tokens / total_samples,
                }
            }, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")

    print("=" * 60)

    return {
        "per_dataset_accuracy": {
            ds_name: ds["correct"] / ds["total"] if ds["total"] > 0 else 0
            for ds_name, ds in dataset_scores.items()
        },
        "overall_accuracy": total_correct / total_problems,
        "total_correct": total_correct,
        "total_problems": total_problems,
    }


def get_scores_with_budget(file_path: str, k_values: Optional[List[int]] = None):
    """Load and display scores from a budget evaluation results file."""
    scores = defaultdict(int)
    totals = defaultdict(int)
    pass_at_k_results = defaultdict(list)

    budget_stats = {
        'delimiter_forced': 0,
        'thinking_tokens': 0,
        'response_tokens': 0,
        'total_samples': 0,
    }

    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            dataset_name = data["dataset"]

            if "sample_id" not in data or data["sample_id"] == 0:
                scores[dataset_name] += int(data["correct"])
                totals[dataset_name] += 1

            if "sample_id" in data and "problem_id" in data:
                problem_key = (dataset_name, data["problem_id"])
                pass_at_k_results[problem_key].append(data["correct"])

            # Collect budget stats
            if "delimiter_forced" in data:
                budget_stats['delimiter_forced'] += int(data['delimiter_forced'])
                budget_stats['thinking_tokens'] += data.get('thinking_tokens', 0)
                budget_stats['response_tokens'] += data.get('response_tokens', 0)
                budget_stats['total_samples'] += 1

    # Print results
    print("\n" + "=" * 60)
    print("SCORES")
    print("=" * 60)
    for name in sorted(scores.keys()):
        acc = 100 * scores[name] / totals[name] if totals[name] > 0 else 0
        print(f"{name}: {scores[name]}/{totals[name]} ({acc:.2f}%)")

    if budget_stats['total_samples'] > 0:
        print("\nBudget Stats:")
        n = budget_stats['total_samples']
        print(f"  Delimiter forced: {budget_stats['delimiter_forced']}/{n} ({100*budget_stats['delimiter_forced']/n:.2f}%)")
        print(f"  Avg thinking tokens: {budget_stats['thinking_tokens']/n:.1f}")
        print(f"  Avg response tokens: {budget_stats['response_tokens']/n:.1f}")

    if k_values and pass_at_k_results:
        pass_k_metrics = calculate_pass_at_k_metrics(pass_at_k_results, k_values)
        print("\nPass@k Metrics:")
        for dataset_name in sorted(pass_k_metrics.keys()):
            for k in sorted(k_values):
                if f"pass@{k}" in pass_k_metrics[dataset_name]:
                    print(f"  {dataset_name} pass@{k}: {pass_k_metrics[dataset_name][f'pass@{k}']:.2f}%")

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model with thinking/response budget')

    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name or local path')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to LoRA adapter checkpoint')
    parser.add_argument('--max_lora_rank', type=int, default=64,
                        help='Maximum LoRA rank')

    # Budget arguments
    parser.add_argument('--thinking_budget', type=int, required=True,
                        help='Maximum tokens for thinking phase')
    parser.add_argument('--response_budget', type=int, required=True,
                        help='Maximum tokens for response phase')
    parser.add_argument('--thinking_end_token', type=str, default='</think>',
                        help='Token/string marking end of thinking phase')
    parser.add_argument('--thinking_end_token_id', type=int, default=151668,
                        help='Token ID of thinking delimiter (default: 151668 for Qwen3)')

    # Generation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling')

    # Pass@k arguments
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of samples per problem')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1],
                        help='k values for pass@k (e.g., 1 5 10)')

    # Data arguments
    parser.add_argument('--mode', type=str, default='eval',
                        help='Data split (train/eval)')
    parser.add_argument('--dataset_names', type=str, nargs='*', default=None,
                        help='List of dataset names to evaluate (e.g., calc_mixed gsm8k)')
    parser.add_argument('--dataset_paths', type=str, nargs='*', default=None,
                        help='List of paths for each dataset (use "none" for default path)')
    parser.add_argument('--turn_off_thinking', action='store_true',
                        help='Turn off thinking mode by adding /no_think tags to prompts')

    # Save arguments
    parser.add_argument('--save_name', type=str, required=True,
                        help='Name for output file')
    parser.add_argument('--save_path', type=str, default='./results/',
                        help='Directory to save results')

    # vLLM arguments
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization (0.0-1.0)')

    # Other
    parser.add_argument('--force', action='store_true',
                        help='Force re-evaluation even if results exist')
    parser.add_argument('--scores_only', action='store_true',
                        help='Only print scores from existing results file')

    args = parser.parse_args()

    # Determine save file path
    save_file = f"{args.save_name}_{args.mode}_tb{args.thinking_budget}_rb{args.response_budget}.jsonl"
    full_save_path = os.path.join(args.save_path, save_file)

    if args.scores_only:
        if os.path.exists(full_save_path):
            get_scores_with_budget(full_save_path, k_values=args.k_values)
        else:
            print(f"Results file not found: {full_save_path}")
    elif os.path.exists(full_save_path) and not args.force:
        print(f"\nResults file exists: {full_save_path}")
        print("Loading existing scores (use --force to re-evaluate)...\n")
        get_scores_with_budget(full_save_path, k_values=args.k_values)
    else:
        if args.force and os.path.exists(full_save_path):
            print(f"\nForcing re-evaluation (--force flag set)")

        score_vllm_with_budget(
            model_name=args.model_name,
            save_name=args.save_name,
            save_path=args.save_path,
            batch_size=args.batch_size,
            temperature=args.temperature,
            thinking_budget=args.thinking_budget,
            response_budget=args.response_budget,
            thinking_end_token=args.thinking_end_token,
            thinking_end_token_id=args.thinking_end_token_id,
            mode=args.mode,
            lora_path=args.lora_path,
            max_lora_rank=args.max_lora_rank,
            gpu_memory_utilization=args.gpu_memory_utilization,
            n_samples=args.n_samples,
            k_values=args.k_values,
            dataset_names=args.dataset_names,
            dataset_paths=args.dataset_paths,
            turn_off_thinking=args.turn_off_thinking,
        )

        print("\nEvaluation complete! Final scores:")
        get_scores_with_budget(full_save_path, k_values=args.k_values)
