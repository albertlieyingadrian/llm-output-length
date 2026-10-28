"""Steering along the Remaining Count Probe direction (paper, Section 4.4).

Tests whether the linearly-decoded length direction is causally meaningful:
add alpha * u to the residual stream during generation, where u is the unit
vector of the count probe's weight slice for one layer, and measure the
realized completion-length distribution as a function of alpha.

Direction extraction. The headline probe is trained on the concatenation of
all layers' hidden states in order [hidden_states_0 (embeddings), ...,
hidden_states_L], so the probe weight w has shape [(L+1) * d]. For steering
index i we take u = normalize(w[i*d : (i+1)*d]) and add alpha * u at the
OUTPUT of decoder layer i-1 (a forward hook), which is exactly the stream the
probe read at concat position i. Positive alpha pushes the stream in the
direction the probe reads as "more tokens remaining"; the causal hypothesis
predicts longer completions for alpha > 0 and shorter for alpha < 0.

Design notes:
- The same prompts and the same torch seed are used for every alpha, so the
  comparison across alphas is paired per prompt; significance is a sign-flip
  test on per-prompt log-length differences vs the alpha=0 arm.
- Quality check: steered completions are re-scored WITHOUT hooks and we report
  mean per-token NLL under the unsteered model, plus the fraction of
  generations that hit max_new_tokens (non-termination).
- The average residual-stream norm at the steering layer is printed first;
  choose |alpha| as a modest fraction of it (typically 2-10%).

Usage (paper settings: the full GSM8K test set, alpha in {-12, ..., 12};
run once per layer, 8, 16 and 24):
    python -m miol.analysis.steering_experiment \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --probe-dir outputs/llama3.1_8b-gsm8k/llama3.1_8b-gsm8k-0/seed_0/trained_probe \
        --steer-index 24 \
        --output-dir reports/steering_llama3.1_8b-gsm8k_layer_24

Multi-GPU (data-parallel prompt sharding, same pattern as extract_hidden_states.py;
each rank loads a full model replica on its own GPU and runs every alpha arm on
its prompt shard; rank 0 gathers and writes outputs):
    python -m torch.distributed.run --nproc_per_node=gpu \
        src/miol/analysis/steering_experiment.py \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --probe-dir outputs/llama3.1_8b-gsm8k/llama3.1_8b-gsm8k-0/seed_0/trained_probe \
        --steer-index 24 --output-dir reports/steering_llama3.1_8b-gsm8k_layer_24
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
    """Init NCCL group when launched under torch.distributed.run; no-op otherwise.

    Returns (rank, world_size). Mirrors extract_hidden_states.py: one process
    per GPU, disjoint data shards, results gathered on rank 0.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def load_direction(probe_dir: Path, steer_index: int, hidden_size: int) -> torch.Tensor:
    state = torch.load(probe_dir / "count_head.pt", map_location="cpu", weights_only=True)
    w = state["head"]["weight"].squeeze(0)  # [(L+1) * d]
    n_layers = w.numel() // hidden_size
    if w.numel() % hidden_size != 0:
        raise ValueError(f"count head dim {w.numel()} is not a multiple of hidden_size {hidden_size}")
    if not (0 <= steer_index < n_layers):
        raise ValueError(f"steer index {steer_index} out of range [0, {n_layers})")
    sl = w[steer_index * hidden_size : (steer_index + 1) * hidden_size]
    print(f"count head covers {n_layers} concat layers; slice {steer_index} norm = {sl.norm():.4f} "
          f"({100 * sl.norm() ** 2 / w.norm() ** 2:.1f}% of total weight mass)")
    return sl / sl.norm()


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
def generate_arm(model, tokenizer, prompts, alpha, direction, steer_module, gen_kwargs, batch_size, seed):
    """Generate all prompts with steering strength alpha; return per-prompt records."""
    direction = direction.to(model.device, dtype=model.dtype)
    handle = None
    if alpha != 0.0:
        def hook(_mod, _inp, out):
            # .to(h.device) is a no-op when devices already match; it guards
            # against sharded device_map runs where the hooked layer lives on
            # a different GPU than the model's first parameter.
            h = out[0] if isinstance(out, tuple) else out
            steered = h + alpha * direction.to(h.device, dtype=h.dtype)
            if isinstance(out, tuple):
                return (steered,) + out[1:]
            return steered
        handle = steer_module.register_forward_hook(hook)

    records = []
    try:
        set_seed(seed)  # same stream for every arm -> paired comparison
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            out = model.generate(**enc, **gen_kwargs, pad_token_id=tokenizer.pad_token_id)
            comp = out[:, enc["input_ids"].shape[1] :]
            for row in comp:
                toks = row.tolist()
                if tokenizer.eos_token_id in toks:
                    length = toks.index(tokenizer.eos_token_id) + 1
                    hit_cutoff = False
                else:
                    length = sum(1 for t in toks if t != tokenizer.pad_token_id)
                    hit_cutoff = True
                records.append({"length": length, "hit_cutoff": hit_cutoff, "completion_ids": toks[:length]})
    finally:
        if handle is not None:
            handle.remove()
    return records


def gather_records(records: list[dict], rank: int, world_size: int) -> list[dict] | None:
    """Gather per-rank records to rank 0, rank-major order (deterministic, so
    the per-prompt pairing across alpha arms is preserved). Returns the merged
    list on rank 0, None elsewhere. completion_ids are dropped before the
    gather to keep the payload small."""
    slim = [{k: r[k] for k in ("length", "hit_cutoff", "nll")} for r in records]
    if world_size == 1:
        return slim
    bucket: list[list[dict] | None] = [None] * world_size if rank == 0 else None
    dist.gather_object(slim, bucket, dst=0)
    if rank != 0:
        return None
    return [r for shard in bucket for r in shard]


@torch.no_grad()
def rescore_nll(model, tokenizer, prompts, records, batch_size) -> None:
    """Mean per-token NLL of each steered completion under the UNSTEERED model."""
    for start in range(0, len(prompts), batch_size):
        for j in range(start, min(start + batch_size, len(prompts))):
            rec = records[j]
            if not rec["completion_ids"]:
                rec["nll"] = float("nan")
                continue
            prompt_ids = tokenizer(prompts[j], return_tensors="pt").input_ids.to(model.device)
            comp_ids = torch.tensor([rec["completion_ids"]], device=model.device)
            full = torch.cat([prompt_ids, comp_ids], dim=1)
            logits = model(full).logits[0, prompt_ids.shape[1] - 1 : -1]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            token_lp = logprobs.gather(1, comp_ids[0].unsqueeze(1)).squeeze(1)
            rec["nll"] = float(-token_lp.mean())


def signflip_p(diffs: np.ndarray, n_perm: int, rng) -> float:
    """Two-sided sign-flip test on paired per-prompt differences."""
    diffs = diffs[~np.isnan(diffs)]
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm = np.abs((signs * diffs).mean(axis=1))
    return float((1 + (perm >= obs).sum()) / (1 + n_perm))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--probe-dir", type=Path, required=True, help="trained_probe/ dir containing count_head.pt")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steer-index", type=int, default=24,
                    help="concat-layer index i (0=embeddings); hook at decoder layer i-1 output (default 24)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0])
    ap.add_argument("--num-prompts", type=int, default=1319, help="GSM8K test prompts (default: all 1,319)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    if 0.0 not in args.alphas:
        args.alphas = [0.0] + args.alphas

    rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    def print_main(*a, **kw):
        if rank == 0:
            print(*a, **kw)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    direction = load_direction(args.probe_dir, args.steer_index, hidden_size)
    if args.steer_index == 0:
        steer_module = model.get_input_embeddings()
    else:
        steer_module = model.model.layers[args.steer_index - 1]

    # Every rank builds the identical full prompt list (deterministic shuffle),
    # then takes a disjoint shard. Each rank runs ALL alpha arms on its shard,
    # so per-prompt pairing across arms is preserved within and across ranks.
    all_prompts = build_prompts(tokenizer, args.num_prompts, args.seed)
    prompts = all_prompts[rank::world_size]
    print_main(f"{world_size} rank(s); ~{len(prompts)} prompts per rank")

    # Calibration aid: typical residual norm at the steering point.
    if rank == 0:
        with torch.no_grad():
            enc = tokenizer(prompts[:4], return_tensors="pt", padding=True, padding_side="left").to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states[args.steer_index]
            print(f"mean residual-stream norm at concat index {args.steer_index}: "
                  f"{hs.norm(dim=-1).float().mean():.1f} (choose |alpha| as a modest fraction of this)")
    if world_size > 1:
        dist.barrier()

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

    results = {}
    for alpha in args.alphas:
        print_main(f"--- alpha = {alpha:+g} ---")
        # seed + rank: identical stream across arms within a rank (paired
        # design), different streams across ranks.
        records = generate_arm(model, tokenizer, prompts, alpha, direction, steer_module,
                               gen_kwargs, args.batch_size, args.seed + rank)
        rescore_nll(model, tokenizer, prompts, records, args.batch_size)
        merged = gather_records(records, rank, world_size)
        if rank != 0:
            continue
        lengths = np.array([r["length"] for r in merged], dtype=float)
        results[alpha] = {
            "lengths": lengths.tolist(),
            "mean_length": float(lengths.mean()),
            "median_length": float(np.median(lengths)),
            "std_length": float(lengths.std(ddof=1)),
            "frac_cutoff": float(np.mean([r["hit_cutoff"] for r in merged])),
            "mean_nll": float(np.nanmean([r["nll"] for r in merged])),
        }
        s = results[alpha]
        print(f"  length mean {s['mean_length']:.1f}  median {s['median_length']:.1f}  "
              f"cutoff {s['frac_cutoff']:.2f}  NLL {s['mean_nll']:.3f}")

    if rank != 0:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    rng = np.random.default_rng(args.seed)
    base = np.array(results[0.0]["lengths"])
    for alpha in args.alphas:
        if alpha == 0.0:
            continue
        arm = np.array(results[alpha]["lengths"])
        diffs = np.log(arm) - np.log(base)  # paired log-length differences
        results[alpha]["mean_log_length_ratio"] = float(np.exp(diffs.mean()))
        results[alpha]["p_vs_alpha0"] = signflip_p(diffs, args.n_perm, rng)
        print(f"alpha {alpha:+g}: length ratio vs alpha=0 = x{results[alpha]['mean_log_length_ratio']:.3f}, "
              f"p = {results[alpha]['p_vs_alpha0']:.4g}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model_name, "probe_dir": str(args.probe_dir), "steer_index": args.steer_index,
        "num_prompts": args.num_prompts, "seed": args.seed, "gen_kwargs": gen_kwargs,
        "arms": {str(a): {k: v for k, v in s.items() if k != "lengths"} for a, s in results.items()},
        "lengths": {str(a): s["lengths"] for a, s in results.items()},
    }
    with open(args.output_dir / "steering_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    alphas = sorted(results)
    means = [results[a]["mean_length"] for a in alphas]
    ses = [results[a]["std_length"] / np.sqrt(args.num_prompts) for a in alphas]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(alphas, means, yerr=ses, marker="o", capsize=3)
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel("mean completion length (tokens)")
    ax.set_title(f"Steering along the count-probe direction\n{args.model_name}, concat index {args.steer_index}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.output_dir / "steering_length_vs_alpha.pdf")
    fig.savefig(args.output_dir / "steering_length_vs_alpha.png", dpi=200)
    print(f"Wrote {args.output_dir}/steering_results.json and steering_length_vs_alpha.pdf/png")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
