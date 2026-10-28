"""Per-token average probe-prediction delta report.

Exploratory tool for the aggregate retraction analysis (paper, Section 4.3): for every
completion token in an evaluate_probes.py output JSON, compute the one-step
change in the Remaining Count Probe's prediction

    delta_r[i] = predictions[i] - predictions[i-1]   (completion positions, i >= 1)

attribute it to the token *at* position i (the token whose hidden state produced
predictions[i]), pool the deltas by decoded token string, and report the tokens
with the highest mean delta as a PDF table.

Interpretation notes (also printed on the report's front page):
- The ground-truth r_t decrements by exactly 1 per position, so a well-calibrated
  probe drifts at mean delta ~ -1. Tokens are interesting relative to that
  baseline, not relative to 0.
- This is a DISCOVERY tool. Selecting trigger tokens by max mean delta and then
  reporting that delta is high at those tokens is circular; the confirmatory
  retraction analysis must use a trigger list fixed independently (or a
  split-half design: discover on one half, test on the other).
- Rare tokens dominate any unthresholded ranking by noise; the report keeps
  tokens with at least --min-count occurrences (a separate low-count table is
  included for completeness).

Usage:
    python -m miol.analysis.token_delta_report \
        --eval-json evals/qwen3_4b-gsm8k.json \
        --output reports/token_delta_qwen3_4b-gsm8k.pdf \
        --probe count --top 100 --min-count 20
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def collect_deltas(eval_json: Path, probe: str) -> tuple[dict[str, list[float]], dict]:
    """Pool per-step prediction deltas by decoded token string.

    Returns (token -> list of deltas, global stats dict).
    """
    with open(eval_json) as f:
        data = json.load(f)

    per_token: dict[str, list[float]] = defaultdict(list)
    n_examples = 0
    n_deltas = 0
    total = 0.0
    total_sq = 0.0

    for ex in data["examples"]:
        if not ex.get("has_eos", True):
            continue
        preds = ex["probes"][probe]["predictions"]
        if len(preds) < 2:
            continue
        prompt_len = ex["prompt_length"]
        tokens = ex["tokens_decoded"]
        n_examples += 1
        for i in range(1, len(preds)):
            delta = preds[i] - preds[i - 1]
            tok = tokens[prompt_len + i]
            per_token[tok].append(delta)
            n_deltas += 1
            total += delta
            total_sq += delta * delta

    mean = total / n_deltas if n_deltas else float("nan")
    var = total_sq / n_deltas - mean * mean if n_deltas else float("nan")
    stats = {
        "source": str(eval_json),
        "probe": probe,
        "n_examples": n_examples,
        "n_deltas": n_deltas,
        "global_mean_delta": mean,
        "global_std_delta": math.sqrt(max(var, 0.0)),
        "n_unique_tokens": len(per_token),
    }
    return per_token, stats


def summarize(per_token: dict[str, list[float]]) -> list[dict]:
    rows = []
    for tok, deltas in per_token.items():
        n = len(deltas)
        mean = sum(deltas) / n
        var = sum((d - mean) ** 2 for d in deltas) / n if n > 1 else 0.0
        std = math.sqrt(var)
        rows.append(
            {
                "token": tok,
                "count": n,
                "mean_delta": mean,
                "std_delta": std,
                "se_delta": std / math.sqrt(n) if n > 1 else float("nan"),
                "median_delta": sorted(deltas)[n // 2],
                "frac_positive": sum(1 for d in deltas if d > 0) / n,
            }
        )
    return rows


def render_token(tok: str, max_len: int = 18) -> str:
    """Printable single-line form of a decoded token."""
    out = tok.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    if out.strip() == "" and "\\n" not in out and "\\t" not in out:
        out = f"[{len(tok)} sp]"
    if len(out) > max_len:
        out = out[: max_len - 1] + "…"
    return f'"{out}"'


def make_pdf(rows_main: list[dict], rows_rare: list[dict], stats: dict, output: Path, top: int, min_count: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["token", "n", "mean Δr̂", "median Δr̂", "std", "s.e.", "frac Δ>0"]

    def table_page(pdf: PdfPages, rows: list[dict], title: str, rows_per_page: int = 40) -> None:
        for start in range(0, len(rows), rows_per_page):
            chunk = rows[start : start + rows_per_page]
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            cells = [
                [
                    render_token(r["token"]).replace("$", r"\$"),  # literal $, not mathtext
                    str(r["count"]),
                    f'{r["mean_delta"]:+.2f}',
                    f'{r["median_delta"]:+.2f}',
                    f'{r["std_delta"]:.2f}',
                    f'{r["se_delta"]:.2f}',
                    f'{r["frac_positive"]:.2f}',
                ]
                for r in chunk
            ]
            tbl = ax.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.0, 1.25)
            for j in range(len(header)):
                tbl[0, j].set_text_props(fontweight="bold")
            ax.set_title(f"{title} (rows {start + 1}–{start + len(chunk)})", fontsize=11, pad=20)
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(output) as pdf:
        # Front page: provenance + interpretation notes.
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = [
            "Per-token mean probe-prediction delta — exploratory report",
            "",
            f"Source:            {stats['source']}",
            f"Probe:             {stats['probe']}",
            f"Examples used:     {stats['n_examples']} (has_eos only)",
            f"Deltas pooled:     {stats['n_deltas']:,}",
            f"Unique tokens:     {stats['n_unique_tokens']:,}",
            f"Global mean Δr̂:    {stats['global_mean_delta']:+.3f}   (ground-truth drift is exactly −1/token)",
            f"Global std Δr̂:     {stats['global_std_delta']:.3f}",
            "",
            f"Main table: top {top} tokens by mean Δr̂ with n ≥ {min_count}.",
            "Appendix table: tokens with n < min-count and mean Δr̂ above the",
            "main table's cutoff (noise-prone; shown for completeness only).",
            "",
            "READ ME FIRST",
            "• A calibrated probe drifts at ≈ −1 per token; judge tokens against",
            "  that baseline, not against 0.",
            "• This ranking is for DISCOVERY (does a retraction-like vocabulary",
            "  surface organically?). Do not use the same ranking to both select",
            "  trigger tokens and claim the effect — fix the trigger list",
            "  independently, or split-half: discover here, confirm on held-out.",
            "• Deltas are attributed to the token AT position t (whose hidden",
            "  state produced r̂_t). Prompt positions are excluded.",
        ]
        ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        table_page(pdf, rows_main[:top], f"Top tokens by mean Δr̂ (n ≥ {min_count})")
        if rows_rare:
            table_page(pdf, rows_rare, f"Low-count tokens (n < {min_count}) above the main cutoff — noise-prone")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-json", type=Path, required=True, help="evaluate_probes.py output JSON")
    ap.add_argument("--output", type=Path, required=True, help="PDF report path")
    ap.add_argument("--probe", default="count", help="probe key to read predictions from (default: count)")
    ap.add_argument("--top", type=int, default=100, help="rows in the main table (default: 100)")
    ap.add_argument("--min-count", type=int, default=20, help="min occurrences for the main table (default: 20)")
    args = ap.parse_args()

    per_token, stats = collect_deltas(args.eval_json, args.probe)
    rows = summarize(per_token)

    rows_main = sorted((r for r in rows if r["count"] >= args.min_count), key=lambda r: -r["mean_delta"])
    cutoff = rows_main[min(args.top, len(rows_main)) - 1]["mean_delta"] if rows_main else 0.0
    rows_rare = sorted(
        (r for r in rows if r["count"] < args.min_count and r["mean_delta"] > cutoff),
        key=lambda r: -r["mean_delta"],
    )[:40]

    make_pdf(rows_main, rows_rare, stats, args.output, args.top, args.min_count)

    print(f"Examples: {stats['n_examples']}, deltas: {stats['n_deltas']:,}, unique tokens: {stats['n_unique_tokens']:,}")
    print(f"Global mean Δr̂ = {stats['global_mean_delta']:+.3f} (std {stats['global_std_delta']:.3f})")
    print(f"Top 15 (n ≥ {args.min_count}):")
    for r in rows_main[:15]:
        print(
            f'  {render_token(r["token"]):22s} n={r["count"]:<6d} mean={r["mean_delta"]:+8.2f} '
            f'median={r["median_delta"]:+8.2f} frac>0={r["frac_positive"]:.2f}'
        )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
