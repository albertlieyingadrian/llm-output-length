"""Aggregate retraction analysis with length-matched controls (paper, Section 4.3).

Confirmatory counterpart to token_delta_report.py. The trigger list is fixed
in advance to the single token "Wait" (exact decoded-token match). For every
"Wait" event we compare the one-step probe delta

    delta_r[i] = predictions[i] - predictions[i-1]

against control deltas at non-retraction tokens matched on:
  (a) absolute completion-position bucket (--pos-bucket, default 100 tokens),
  (b) per-completion MAE regime (quartiles of mean per-token count-probe MAE),
  (c) dataset -- satisfied by construction: one eval JSON = one (model, dataset).

Controls are sampled from *other* completions in the same (position bucket,
MAE quartile) stratum, excluding any position within --exclusion-window of a
"Wait" event in its own completion (the spike persists for several tokens, so
neighbours are contaminated, not clean controls).

Reported: mean/median delta at retraction vs control, fraction of upward
shifts, Cohen's d, and two permutation p-values -- a cluster-safe sign-flip
test on per-completion paired differences (primary; deltas within a completion
are not independent) and a stratified label-shuffle on pooled events
(secondary). All stats are reported overall and stratified by MAE quartile.
A histogram of retraction vs matched-control deltas is written alongside.

Usage:
    python -m miol.analysis.retraction_control_analysis \
        --eval-json evals/qwen3_4b-gsm8k.json \
        --output-dir reports/retraction_qwen3_4b-gsm8k
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRIGGER = "Wait"  # fixed in advance; exact decoded-token match


def load_events(eval_json: Path, probe: str, pos_bucket: int, exclusion_window: int):
    """Extract retraction events and the matched-control pool.

    Returns (events, pool, quartile_edges) where
      events: list of dicts {example_id, delta, stratum}
      pool:   stratum -> list of (example_id, delta)
      stratum = (position_bucket, mae_quartile)
    """
    with open(eval_json) as f:
        data = json.load(f)

    examples = [ex for ex in data["examples"] if ex.get("has_eos", True)]

    # (b) per-completion MAE regime: quartiles of mean per-token MAE.
    comp_mae = np.array([float(np.mean(ex["probes"][probe]["mae_per_token"])) for ex in examples])
    edges = np.quantile(comp_mae, [0.25, 0.5, 0.75])
    mae_bin = np.searchsorted(edges, comp_mae)  # 0..3 = Q1..Q4

    events = []
    pool: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)

    for idx, ex in enumerate(examples):
        preds = ex["probes"][probe]["predictions"]
        if len(preds) < 2:
            continue
        prompt_len = ex["prompt_length"]
        tokens = ex["tokens_decoded"]
        q = int(mae_bin[idx])

        wait_positions = [i for i in range(1, len(preds)) if tokens[prompt_len + i] == TRIGGER]
        excluded = set()
        for w in wait_positions:
            excluded.update(range(w - exclusion_window, w + exclusion_window + 1))

        for i in range(1, len(preds)):
            delta = preds[i] - preds[i - 1]
            stratum = (i // pos_bucket, q)
            if i in wait_positions:
                events.append({"example_id": idx, "delta": delta, "stratum": stratum, "position": i})
            elif i not in excluded:
                pool[stratum].append((idx, delta))

    return events, pool, edges, len(examples)


def sample_controls(events, pool, k, rng):
    """K controls per event from the same stratum, other completions.

    Falls back to same position bucket / any MAE bin when a stratum pool
    (excluding the event's own completion) is empty. Returns (controls list of
    (example_id, delta) per event, n_fallbacks).
    """
    by_bucket: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (bucket, _q), items in pool.items():
        by_bucket[bucket].extend(items)

    controls, fallbacks = [], 0
    for ev in events:
        candidates = [c for c in pool.get(ev["stratum"], []) if c[0] != ev["example_id"]]
        if not candidates:
            fallbacks += 1
            candidates = [c for c in by_bucket.get(ev["stratum"][0], []) if c[0] != ev["example_id"]]
        idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=len(candidates) < k)
        controls.append([candidates[j] for j in idx])
    return controls, fallbacks


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def signflip_p(paired_diffs: np.ndarray, n_perm: int, rng) -> float:
    """One-sided cluster-safe test: under H0 the per-completion paired
    differences are symmetric about 0; randomly flip signs."""
    obs = paired_diffs.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(paired_diffs)))
    perm = (signs * paired_diffs).mean(axis=1)
    return float((1 + (perm >= obs).sum()) / (1 + n_perm))


def shuffle_p(ev_deltas, ctl_deltas, ev_strata, ctl_strata, n_perm, rng) -> float:
    """One-sided stratified label shuffle on pooled event/control deltas."""
    obs = ev_deltas.mean() - ctl_deltas.mean()
    labels = np.concatenate([np.ones(len(ev_deltas)), np.zeros(len(ctl_deltas))])
    values = np.concatenate([ev_deltas, ctl_deltas])
    all_strata = np.concatenate([ev_strata, ctl_strata])
    count = 0
    for _ in range(n_perm):
        perm_labels = labels.copy()
        for s in np.unique(all_strata):
            m = all_strata == s
            perm_labels[m] = rng.permutation(perm_labels[m])
        stat = values[perm_labels == 1].mean() - values[perm_labels == 0].mean()
        if stat >= obs:
            count += 1
    return float((1 + count) / (1 + n_perm))


def describe(ev: np.ndarray, ctl: np.ndarray) -> dict:
    return {
        "n_events": int(len(ev)),
        "retraction_mean": float(ev.mean()),
        "retraction_median": float(np.median(ev)),
        "control_mean": float(ctl.mean()),
        "control_median": float(np.median(ctl)),
        "retraction_frac_up": float((ev > 0).mean()),
        "control_frac_up": float((ctl > 0).mean()),
        "cohens_d": cohens_d(ev, ctl),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--probe", default="count")
    ap.add_argument("--pos-bucket", type=int, default=100, help="position bucket width, tokens (control a)")
    ap.add_argument("--exclusion-window", type=int, default=10, help="tokens around a Wait excluded from controls")
    ap.add_argument("--controls-per-event", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    events, pool, mae_edges, n_examples = load_events(
        args.eval_json, args.probe, args.pos_bucket, args.exclusion_window
    )
    if not events:
        raise SystemExit(f'No "{TRIGGER}" events found.')

    controls, fallbacks = sample_controls(events, pool, args.controls_per_event, rng)

    ev_deltas = np.array([e["delta"] for e in events])
    ctl_deltas = np.array([d for ctl in controls for (_i, d) in ctl])
    ev_strata = np.array([hash(e["stratum"]) % (2**31) for e in events])
    # One stratum label per control (an event can have fewer than k controls).
    ctl_strata = np.array([s for s, ctl in zip(ev_strata, controls) for _ in ctl])

    # Cluster-safe primary test: one paired difference per completion.
    per_comp_ev, per_comp_ctl = defaultdict(list), defaultdict(list)
    for e, ctl in zip(events, controls):
        per_comp_ev[e["example_id"]].append(e["delta"])
        per_comp_ctl[e["example_id"]].extend(d for (_i, d) in ctl)
    paired = np.array([np.mean(per_comp_ev[c]) - np.mean(per_comp_ctl[c]) for c in per_comp_ev])

    result = {
        "source": str(args.eval_json),
        "probe": args.probe,
        "trigger": TRIGGER,
        "n_examples_used": n_examples,
        "n_completions_with_trigger": len(per_comp_ev),
        "controls_per_event": args.controls_per_event,
        "pos_bucket": args.pos_bucket,
        "exclusion_window": args.exclusion_window,
        "stratum_fallbacks": fallbacks,
        "mae_quartile_edges": [float(x) for x in mae_edges],
        "overall": describe(ev_deltas, ctl_deltas),
        "p_signflip_per_completion": signflip_p(paired, args.n_perm, rng),
        "p_stratified_shuffle": shuffle_p(ev_deltas, ctl_deltas, ev_strata, ctl_strata, min(args.n_perm, 2000), rng),
        "paired_mean_diff_per_completion": float(paired.mean()),
        "by_mae_quartile": {},
    }

    for q in range(4):
        idx = [j for j, e in enumerate(events) if e["stratum"][1] == q]
        if not idx:
            continue
        ev_q = ev_deltas[idx]
        ctl_q = np.array([d for j in idx for (_i, d) in controls[j]])
        result["by_mae_quartile"][f"Q{q + 1}"] = describe(ev_q, ctl_q)

    # ---- report ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "retraction_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    o = result["overall"]
    print(f'Trigger "{TRIGGER}": {o["n_events"]} events in {result["n_completions_with_trigger"]} completions '
          f"(of {n_examples}); stratum fallbacks: {fallbacks}")
    print(f'  retraction: mean {o["retraction_mean"]:+8.2f}  median {o["retraction_median"]:+8.2f}  '
          f'frac>0 {o["retraction_frac_up"]:.2f}')
    print(f'  control:    mean {o["control_mean"]:+8.2f}  median {o["control_median"]:+8.2f}  '
          f'frac>0 {o["control_frac_up"]:.2f}')
    print(f'  Cohen\'s d = {o["cohens_d"]:.2f}')
    print(f'  p (sign-flip, per-completion, one-sided) = {result["p_signflip_per_completion"]:.4g}')
    print(f'  p (stratified shuffle, pooled, one-sided) = {result["p_stratified_shuffle"]:.4g}')
    print("  By MAE quartile (Q1 = lowest completion MAE):")
    for qname, s in result["by_mae_quartile"].items():
        print(f'    {qname}: n={s["n_events"]:<4d} retraction {s["retraction_mean"]:+8.2f} '
              f'(med {s["retraction_median"]:+7.2f}, frac>0 {s["retraction_frac_up"]:.2f})  '
              f'control {s["control_mean"]:+7.2f}  d={s["cohens_d"]:.2f}')

    # ---- histogram ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    lo = min(np.percentile(ctl_deltas, 0.5), ev_deltas.min()) - 5
    hi = max(np.percentile(ctl_deltas, 99.5), ev_deltas.max()) + 5
    bins = np.linspace(lo, hi, 60)
    ax.hist(ctl_deltas, bins=bins, density=True, alpha=0.55, color="steelblue",
            label=f"matched controls (n={len(ctl_deltas):,})")
    ax.hist(ev_deltas, bins=bins, density=True, alpha=0.65, color="firebrick",
            label=f'"{TRIGGER}" tokens (n={len(ev_deltas)})')
    ax.axvline(float(ctl_deltas.mean()), color="steelblue", ls="--", lw=1.2)
    ax.axvline(float(ev_deltas.mean()), color="firebrick", ls="--", lw=1.2)
    ax.set_xlabel(r"$\Delta\hat{r}_t = \hat{r}_t - \hat{r}_{t-1}$  (tokens)")
    ax.set_ylabel("density")
    ax.set_title(f'Probe-prediction step at "{TRIGGER}" vs length-matched controls\n'
                 f'{Path(str(args.eval_json)).stem}, probe={args.probe}', fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"retraction_histogram.{ext}", dpi=200)
    plt.close(fig)
    print(f"Wrote {args.output_dir}/retraction_analysis.json and retraction_histogram.pdf/png")


if __name__ == "__main__":
    main()
