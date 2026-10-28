"""Sampling-noise floor for prompt-end length prediction (paper, appendix "Sampling-noise floor").

Under stochastic decoding (do_sample, T=0.7, top-p 0.8, top-k 20 — the
extraction settings), the realized completion length T is a random variable
given the prompt, while the prompt-end hidden state is fixed. The Completion
Length Probe can therefore at best predict a central tendency of the length
distribution; its prompt-end MAE is lower-bounded by the intrinsic sampling
spread. This script measures that floor: sample k completions per prompt,
compute each prompt's L1-optimal spread, and compare the probe's MAE to it.

Two floor estimators are reported:
- in-sample: mean_j |T_j - median(T_1..T_k)| per prompt (slightly optimistic:
  the median is fit on the same draws it is scored on);
- leave-one-out (the honest one, reported as the headline): each draw is
  scored against the median of the other k-1, matching the situation of a
  predictor that must guess an independent realization.

Floors use naturally-terminated draws only (has_eos, consistent with the
paper's filter); prompts with fewer than --min-samples surviving draws are
dropped and counted.

Single GPU:
    python -m miol.analysis.sampling_noise_floor \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --num-prompts 1319 --samples-per-prompt 10 \
        --probe-mae 42.29 \
        --output-dir reports/sampling_floor_llama3.1_8b-gsm8k

Multi-GPU (data-parallel prompt sharding, same pattern as steering_experiment.py):
    python -m torch.distributed.run --nproc_per_node=gpu \
        src/miol/analysis/sampling_noise_floor.py \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --num-prompts 1319 --samples-per-prompt 10 \
        --probe-mae 42.29 \
        --output-dir reports/sampling_floor_llama3.1_8b-gsm8k
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from miol.data.tokenization import drop_duplicate_bos

GSM8K_SYSTEM_PROMPT = "You are a helpful math tutor that solves grade school math problems step by step."


def setup_distributed() -> tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def build_prompts(tokenizer, num_prompts: int, seed: int) -> list[str]:
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(num_prompts))
    prompts = []
    for q in ds["question"]:
        messages = [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(drop_duplicate_bos(text, tokenizer))
    return prompts


@torch.no_grad()
def sample_lengths(model, tokenizer, prompt: str, k: int, gen_kwargs: dict, batch_size: int, seed: int):
    """Generate k stochastic completions of one prompt; return (lengths, n_cutoff)."""
    set_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    lengths, n_cutoff = [], 0
    remaining = k
    while remaining > 0:
        b = min(batch_size, remaining)
        input_ids = enc["input_ids"].expand(b, -1)
        attention_mask = enc["attention_mask"].expand(b, -1)
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             **gen_kwargs, pad_token_id=tokenizer.pad_token_id)
        comp = out[:, input_ids.shape[1]:]
        for row in comp:
            toks = row.tolist()
            if tokenizer.eos_token_id in toks:
                lengths.append(toks.index(tokenizer.eos_token_id) + 1)
            else:
                n_cutoff += 1
        remaining -= b
    return lengths, n_cutoff


def per_prompt_stats(lengths: list[int]) -> dict:
    arr = np.asarray(lengths, dtype=np.float64)
    med = float(np.median(arr))
    mad_in = float(np.abs(arr - med).mean())
    loo_errs = [abs(arr[j] - float(np.median(np.delete(arr, j)))) for j in range(len(arr))]
    return {
        "n": int(len(arr)),
        "median_T": med,
        "mean_T": float(arr.mean()),
        "std_T": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "mad_insample": mad_in,
        "mad_loo": float(np.mean(loo_errs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--num-prompts", type=int, default=1319, help="GSM8K test prompts (default: all 1,319)")
    ap.add_argument("--samples-per-prompt", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=5,
                    help="drop prompts with fewer naturally-terminated draws than this")
    ap.add_argument("--probe-mae", type=float, default=None,
                    help="Completion Length Probe prompt-end MAE for this cell (for the ratio)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    all_prompts = build_prompts(tokenizer, args.num_prompts, args.seed)
    shard = list(range(rank, len(all_prompts), world_size))
    if rank == 0:
        print(f"{world_size} rank(s); {len(all_prompts)} prompts x {args.samples_per_prompt} samples")

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

    records = []
    for i in shard:
        # seed per prompt (not per rank) so results are shard-invariant
        lengths, n_cutoff = sample_lengths(
            model, tokenizer, all_prompts[i], args.samples_per_prompt,
            gen_kwargs, args.batch_size, args.seed * 100_000 + i,
        )
        records.append({"prompt_idx": i, "lengths": lengths, "n_cutoff": n_cutoff})

    if world_size > 1:
        bucket = [None] * world_size if rank == 0 else None
        dist.gather_object(records, bucket, dst=0)
        if rank != 0:
            dist.barrier()
            dist.destroy_process_group()
            return
        records = [r for part in bucket for r in part]

    # ---- rank 0: aggregate ----
    records.sort(key=lambda r: r["prompt_idx"])
    kept, dropped = [], 0
    total_cutoff = sum(r["n_cutoff"] for r in records)
    for r in records:
        if len(r["lengths"]) >= args.min_samples:
            kept.append(per_prompt_stats(r["lengths"]))
        else:
            dropped += 1

    mad_loo = np.array([s["mad_loo"] for s in kept])
    mad_in = np.array([s["mad_insample"] for s in kept])
    summary = {
        "model": args.model_name,
        "num_prompts": args.num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "gen_kwargs": gen_kwargs,
        "seed": args.seed,
        "prompts_kept": len(kept),
        "prompts_dropped_min_samples": dropped,
        "frac_draws_cutoff": total_cutoff / (args.num_prompts * args.samples_per_prompt),
        "floor_loo_mean": float(mad_loo.mean()),
        "floor_loo_median": float(np.median(mad_loo)),
        "floor_insample_mean": float(mad_in.mean()),
        "mean_per_prompt_std_T": float(np.mean([s["std_T"] for s in kept])),
        "mean_per_prompt_median_T": float(np.mean([s["median_T"] for s in kept])),
        "per_prompt": kept,
    }
    if args.probe_mae is not None:
        summary["probe_mae"] = args.probe_mae
        summary["probe_over_floor"] = args.probe_mae / summary["floor_loo_mean"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "sampling_floor.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"prompts kept {len(kept)} (dropped {dropped}); draws hitting cutoff: {summary['frac_draws_cutoff']:.1%}")
    print(f"sampling-noise floor (LOO, headline): mean {summary['floor_loo_mean']:.2f}  "
          f"median {summary['floor_loo_median']:.2f}  (in-sample mean {summary['floor_insample_mean']:.2f})")
    print(f"mean per-prompt std of T: {summary['mean_per_prompt_std_T']:.2f}")
    if args.probe_mae is not None:
        print(f"probe prompt-end MAE {args.probe_mae:.2f} = {summary['probe_over_floor']:.2f}x the irreducible floor")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(mad_loo, bins=40, color="steelblue", alpha=0.8)
    ax.axvline(summary["floor_loo_mean"], color="navy", ls="--", lw=1.4,
               label=f"floor (mean LOO MAD) = {summary['floor_loo_mean']:.1f}")
    if args.probe_mae is not None:
        ax.axvline(args.probe_mae, color="firebrick", ls="-", lw=1.6,
                   label=f"probe prompt-end MAE = {args.probe_mae:.1f}")
    ax.set_xlabel("per-prompt sampling spread of $T$ (LOO MAD, tokens)")
    ax.set_ylabel("prompts")
    ax.set_title(f"Sampling-noise floor, {args.model_name}\n"
                 f"{summary['prompts_kept']} prompts x {args.samples_per_prompt} draws", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"sampling_floor.{ext}", dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output_dir}/sampling_floor.json and sampling_floor.pdf/png")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
