"""
Script to extract and save hidden states from the base model for offline probe training.

This script:
1. Loads the base model (without probe) from Hydra config
2. Runs generation on the specified dataset
3. Saves inputs, outputs, and hidden states to disk

Usage (writes <output_dir>/hidden_states/{train,eval}/):
    python -m miol.extract_hidden_states model=llama3.1_8b data=gsm8k output_dir=outputs/llama3.1_8b-gsm8k
    python -m miol.extract_hidden_states model=qwen3_4b data=gsm8k data.num_train_samples=100

Multi-GPU (each rank loads a full model copy and takes every world_size-th sample):
    python -m torch.distributed.run --nproc_per_node=gpu src/miol/extract_hidden_states.py \
        model=llama3.1_8b data=gsm8k output_dir=outputs/llama3.1_8b-gsm8k
"""

import json
import os
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from miol.data import tokenize_function
from miol.model.generation_utils import (
    build_generation_kwargs,
    extract_completion_with_eos,
)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Seed before any data sampling or generation. With do_sample=true the
    # generated completions depend on the seed; downstream probe-training
    # multi-seed runs share these completions, so this is the *extraction*
    # seed and is independent from the probe-training seed.
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    print_main("=" * 80)
    print_main("EXTRACT HIDDEN STATES")
    print_main(f"SEED: {seed}")
    print_main("=" * 80)
    print_main(OmegaConf.to_yaml(cfg))
    print_main("=" * 80 + "\n")

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    dist_backend: Optional[str] = None
    if local_rank != -1 and not is_distributed():
        dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=dist_backend)
    elif is_distributed():
        backend_obj = dist.get_backend()
        dist_backend = str(backend_obj)

    rank = get_rank()
    world_size = get_world_size()

    # Setup output directory
    output_dir = cfg.get("output_dir", "./outputs")
    hidden_states_dir = Path(output_dir) / "hidden_states"
    hidden_states_dir.mkdir(parents=True, exist_ok=True)

    print_main(f"Output directory: {hidden_states_dir}")

    # Load tokenizer
    model_name = cfg.model.model_name
    print_main(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Left padding for generation
    tokenizer.padding_side = 'left'

    # Load base model (without probe)
    print_main(f"Loading base model: {model_name}")

    if torch.cuda.is_available():
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    print_main(f"Using device: {device}, dtype: {model_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()

    # Load dataset
    dataset_builder = instantiate(cfg.data)
    dataset_type = cfg.data._target_.split('.')[-1]
    print_main(f"Loading data (builder: {dataset_type})...")

    # Extract both train and eval datasets
    train_dataset = dataset_builder.get_dataset(split='train', tokenizer=tokenizer)
    eval_dataset = dataset_builder.get_dataset(split='validation', tokenizer=tokenizer)

    print_main(f"Train dataset size: {len(train_dataset)}")
    print_main(f"Eval dataset size: {len(eval_dataset)}")

    # Get generation config from cfg and override with model-specific config if present
    generation_config = dict(cfg.get("generation", {}))
    if "generation_config" in cfg.model:
        generation_config.update(dict(cfg.model.get("generation_config", {})))
    print_main("\nProcessing generation config...")
    print_main(generation_config)

    # Whether to save the heavy `.safetensors` for samples whose generation did
    # not terminate naturally. The per-sample `.json` metadata is always kept.
    save_truncated = bool(cfg.get("extract", {}).get("save_truncated", False))
    print_main(f"\nsave_truncated = {save_truncated} "
               f"({'all samples saved' if save_truncated else 'truncated samples skipped on disk; JSON kept'})")

    # Process train dataset
    print_main("\nProcessing train dataset...")
    train_saved, train_skipped, train_failed = process_dataset(
        dataset=train_dataset,
        split="train",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        output_dir=hidden_states_dir,
        device=device,
        rank=rank,
        world_size=world_size,
        save_truncated=save_truncated,
    )

    # Process eval dataset
    print_main("\nProcessing eval dataset...")
    eval_saved, eval_skipped, eval_failed = process_dataset(
        dataset=eval_dataset,
        split="eval",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        output_dir=hidden_states_dir,
        device=device,
        rank=rank,
        world_size=world_size,
        save_truncated=save_truncated,
    )

    total_train_saved = aggregate_count(train_saved, dist_backend, device)
    total_train_skipped = aggregate_count(train_skipped, dist_backend, device)
    total_train_failed = aggregate_count(train_failed, dist_backend, device)
    total_eval_saved = aggregate_count(eval_saved, dist_backend, device)
    total_eval_skipped = aggregate_count(eval_skipped, dist_backend, device)
    total_eval_failed = aggregate_count(eval_failed, dist_backend, device)

    train_total = total_train_saved + total_train_skipped + total_train_failed
    eval_total = total_eval_saved + total_eval_skipped + total_eval_failed

    def _pct(n, d):
        return (100.0 * n / d) if d > 0 else 0.0

    print_main(
        f"\nDone! Hidden states saved to: {hidden_states_dir}\n"
        f"  train: saved {total_train_saved} / processed {train_total}"
        f" — {total_train_skipped} truncated"
        f" ({_pct(total_train_skipped, train_total):.1f}%)"
        f", {total_train_failed} failed"
        f" ({_pct(total_train_failed, train_total):.1f}%)\n"
        f"  eval:  saved {total_eval_saved} / processed {eval_total}"
        f" — {total_eval_skipped} truncated"
        f" ({_pct(total_eval_skipped, eval_total):.1f}%)"
        f", {total_eval_failed} failed"
        f" ({_pct(total_eval_failed, eval_total):.1f}%)"
    )

    # Per-split extraction stats — useful when reading the cache months later
    # to remember why some indices are missing.
    if is_main_process():
        for split, saved, skipped, failed_n in (
            ("train", total_train_saved, total_train_skipped, total_train_failed),
            ("eval", total_eval_saved, total_eval_skipped, total_eval_failed),
        ):
            stats = {
                "split": split,
                "saved_with_tensors": int(saved),
                "skipped_truncated": int(skipped),
                "failed": int(failed_n),
                "total_processed": int(saved + skipped + failed_n),
                "save_truncated_flag": save_truncated,
            }
            stats_path = hidden_states_dir / split / "extraction_stats.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def process_dataset(
    dataset,
    split,
    model,
    tokenizer,
    generation_config,
    output_dir,
    device,
    rank: int,
    world_size: int,
    save_truncated: bool = False,
):
    """Process a dataset and save hidden states.

    When `save_truncated=False` (default), samples whose generation did not
    terminate with EOS have their heavy `.safetensors` skipped on disk; only
    the small per-sample `.json` metadata is written so the truncation can
    still be audited after the fact. The trainer's `HiddenStatesDataset`
    discovers samples by globbing `sample_*.safetensors`, so skipped samples
    are automatically excluded from training without any reader-side change.

    The loop is resumable: indices whose `.json` already exists are skipped
    and counted against the existing tally. Per-sample failures (CUDA OOM,
    model errors, etc.) are caught and recorded as an error-marker JSON so
    one bad sample doesn't kill the whole rank, and so a re-run picks up
    where the previous one left off without reprocessing already-handled
    indices. To retry an errored sample, delete its JSON before re-running.

    Returns:
        (saved, skipped, failed): saved is the number of samples written
        with a full safetensors tensor, skipped is the number of has_eos=False
        samples that wrote only the JSON metadata, failed is the number of
        samples that crashed and got an error-marker JSON.
    """
    split_dir = Path(output_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    indices = range(rank, len(dataset), world_size)
    total_local = len(indices)
    iterator = indices
    if is_main_process():
        iterator = tqdm(indices, total=total_local, desc=f"Extracting {split}")

    saved = 0
    skipped = 0
    failed = 0

    for idx in iterator:
        output_base = split_dir / f"sample_{idx:06d}"
        json_path = output_base.with_suffix(".json")

        # Resumability: if this index already has a JSON on disk, count it
        # against the existing tally and skip. To retry an errored sample,
        # delete its JSON before re-running.
        if json_path.exists():
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                if meta.get("error"):
                    failed += 1
                elif meta.get("has_eos", False):
                    saved += 1
                else:
                    skipped += 1
            except Exception:
                # Corrupt JSON — leave it; don't double-process.
                pass
            continue

        try:
            sample = dataset[idx]
            # Tokenize prompt
            tokenized = tokenize_function(
                {"prompt": [sample["prompt"]]},
                tokenizer,
                truncation=False,
            )

            prompt_input_ids = torch.tensor([tokenized["input_ids"][0]]).to(device)
            prompt_attention_mask = torch.tensor([tokenized["attention_mask"][0]]).to(device)

            # Trim padding from prompt
            prompt_len = prompt_attention_mask.sum().item()
            prompt_input_ids_trimmed = prompt_input_ids[:, -prompt_len:] if prompt_len > 0 else prompt_input_ids[:, :0]
            prompt_attention_mask_trimmed = prompt_attention_mask[:, -prompt_len:] if prompt_len > 0 else prompt_attention_mask[:, :0]

            # Generate with hidden states
            with torch.no_grad():
                gen_kwargs = build_generation_kwargs(
                    input_ids=prompt_input_ids_trimmed,
                    attention_mask=prompt_attention_mask_trimmed,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    include_hidden_states=True,
                )

                generation_output = model.generate(**gen_kwargs)
                generated_ids = generation_output.sequences[0]

                # Extract completion
                completion_ids, has_eos, actual_prompt_len = extract_completion_with_eos(
                    generated_ids=generated_ids,
                    prompt_length=prompt_len,
                    eos_token_id=tokenizer.eos_token_id,
                    check_bos=True,
                    bos_token_id=tokenizer.bos_token_id,
                    original_prompt_ids=prompt_input_ids[0, -prompt_len:] if prompt_len > 0 else prompt_input_ids[0, :0],
                )

                # Get actual prompt from generated sequence
                actual_prompt_ids = generated_ids[:actual_prompt_len]
                full_sequence = torch.cat([actual_prompt_ids, completion_ids])

                # Run forward pass to get hidden states for the full sequence
                padded_input_ids = full_sequence.unsqueeze(0)
                attention_mask = torch.ones((1, len(full_sequence)), dtype=torch.long, device=device)

                outputs = model(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Decode completion for metadata (not strictly necessary, but can be useful for debugging and analysis)
                completion = tokenizer.decode(completion_ids.cpu(), skip_special_tokens=False)

                # Extract hidden states (tuple of tensors, one per layer)
                # Each tensor has shape [batch_size, seq_len, hidden_size]
                # Convert to list of tensors and move to CPU
                hidden_states = [h[0].cpu() for h in outputs.hidden_states]  # Remove batch dimension and move to CPU

            # Prepare tensors to save with safetensors
            tensors_to_save = {
                "prompt_ids": actual_prompt_ids.cpu(),
                "completion_ids": completion_ids.cpu(),
                "full_sequence_ids": full_sequence.cpu(),
            }

            # Add hidden states with numeric suffixes
            for i, h in enumerate(hidden_states):
                tensors_to_save[f"hidden_states_{i}"] = h

            # Prepare metadata to save as JSON
            metadata = {
                "prompt": sample["prompt"],
                "completion": completion,
                "has_eos": bool(has_eos),
                "prompt_length": int(actual_prompt_len),
                "completion_length": int(len(completion_ids)),
                "num_hidden_layers": len(hidden_states),
            }

            # Save metadata as JSON for every processed sample (cheap; preserves
            # the audit trail even when we skip the heavy tensors below).
            with open(json_path, "w") as f:
                json.dump(metadata, f)

            # Skip the heavy `.safetensors` for has_eos=False unless explicitly
            # requested. The trainer would loss-mask these positions anyway, so
            # there's no functional change — only disk savings.
            if not has_eos and not save_truncated:
                skipped += 1
                continue

            save_file(tensors_to_save, f"{output_base}.safetensors")
            saved += 1

        except Exception as exc:
            # One bad sample shouldn't kill the rank. Drop an error-marker
            # JSON so the failed slot is auditable and won't be retried by a
            # naive resume (delete the marker JSON to retry). CUDA errors in
            # particular are best-effort: empty_cache may or may not recover.
            failed += 1
            err_meta = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "idx": int(idx),
                "skipped": True,
            }
            try:
                with open(json_path, "w") as f:
                    json.dump(err_meta, f)
            except Exception:
                pass
            print(
                f"[rank {rank}] sample_{idx:06d} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    return saved, skipped, failed


def aggregate_count(local_count: int, backend: Optional[str], device: torch.device) -> int:
    if not is_distributed():
        return local_count

    reduce_device = device if backend == "nccl" else torch.device("cpu")
    tensor = torch.tensor(local_count, dtype=torch.long, device=reduce_device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


if __name__ == "__main__":
    main()
