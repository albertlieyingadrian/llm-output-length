"""Cross-dataset statistical-baseline evaluation.

The mirror of `cross_dataset_eval.py` for the *non-trained* statistical
baselines (`baseline_constant_mean`, `baseline_constant_median`,
`baseline_mean_plus_position`, `baseline_median_plus_position`). For each
(source_dataset, target_dataset) pair we fit the baseline on the source's
train-split metadata and apply it to the target's eval-split metadata,
producing a length-weighted MAE per cell.

Crucially, this script never touches the saved hidden-state tensors --- it
only reads `sample_*.json` metadata, because the baselines are deterministic
functions of `completion_length` and `has_eos`. The `--metadata-pattern`
flag points at the directory containing those JSONs (typically the same
`hidden_states/` dir that `extract_hidden_states.py` writes; the
`.safetensors` files in there are ignored). End-to-end runtime for a 7×7
dataset matrix is sub-minute on a laptop.

Usage:
    python -m miol.analysis.cross_dataset_baseline_eval \\
        --datasets gsm8k math mmlu_pro openthoughts-1k trivia_qa count countdown \\
        --metadata-pattern outputs/qwen3_4b-{dataset}/hidden_states \\
        --output-dir figures/cross_dataset_baselines/qwen3_4b/

Per-baseline output filenames are auto-generated as
`{baseline}__{metric}.pdf` plus the `_mean.csv` / `_std.csv` pair, so a
single invocation produces one heatmap per baseline.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np

from miol.analysis.cross_dataset_eval import _render_heatmap, _write_heatmap_csvs
from miol.evaluate_probes import build_statistical_baselines, predict_statistical
from miol.statistical_baseline import (
    _iter_sample_json_files,
    _load_metadata,
    _safe_mape,
    _safe_smape,
)


# Per-token error functions for each `--metric` choice. Each returns a
# scalar accumulated over all eval tokens; the cell value is the length-
# weighted mean (sum / total token count), matching the trainer's wandb
# convention `total_err / total_cnt`.
_METRIC_FNS: dict[str, Callable[[float, float, float], float]] = {
    "mae": lambda t, p, _eps: abs(t - p),
    "mse": lambda t, p, _eps: (t - p) ** 2,
    "mape": _safe_mape,
    "smape": _safe_smape,
}


def _resolve_split_dir(metadata_dir: Path, split: str) -> Path:
    """Return `<metadata_dir>/<split>` after light validation."""
    candidate = metadata_dir / split
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Expected `{candidate}` to be a directory of `sample_*.json` "
            f"metadata files. Pass `--metadata-pattern` so it resolves to a "
            f"directory containing `train/` and `eval/` subdirs."
        )
    return candidate


def _eval_completion_lengths(eval_dir: Path) -> list[int]:
    """Walk eval-split JSONs once; return completion lengths for every
    example with `has_eos=True` and `completion_length > 0`."""
    lengths: list[int] = []
    for path in _iter_sample_json_files(eval_dir):
        meta = _load_metadata(path)
        if not bool(meta.get("has_eos", False)):
            continue
        L = int(meta.get("completion_length", 0))
        if L <= 0:
            continue
        lengths.append(L)
    return lengths


def _compute_cell(
    spec: dict,
    eval_completion_lengths: list[int],
    metric_fn: Callable[[float, float, float], float],
    epsilon: float,
) -> float | None:
    """Length-weighted per-token metric of one baseline (fit on source) vs.
    the actual remaining-count targets on the target eval set."""
    total_err = 0.0
    total_cnt = 0
    for L in eval_completion_lengths:
        preds = predict_statistical(spec, L)
        targets = [float(L - k - 1) for k in range(L)]
        for p, t in zip(preds, targets):
            total_err += metric_fn(t, p, epsilon)
            total_cnt += 1
    if total_cnt == 0:
        return None
    return total_err / total_cnt


def _build_matrix(
    datasets: list[str],
    metadata_pattern: str,
    baseline_name: str,
    metric_name: str,
    epsilon: float,
) -> np.ndarray:
    """Length-weighted (source × target) baseline metric matrix.

    Side-cached: reads each source's train metadata once (to fit the
    baseline) and each target's eval metadata once (to get completion
    lengths). The two caches are independent.
    """
    if metric_name not in _METRIC_FNS:
        raise ValueError(
            f"Unknown --metric {metric_name!r}; pick one of {sorted(_METRIC_FNS)}."
        )
    metric_fn = _METRIC_FNS[metric_name]

    # Per-source baseline spec; per-target completion lengths.
    source_specs: dict[str, dict] = {}
    target_lens: dict[str, list[int]] = {}

    for ds in datasets:
        meta_dir = Path(metadata_pattern.format(dataset=ds))
        train_dir = _resolve_split_dir(meta_dir, "train")
        eval_dir = _resolve_split_dir(meta_dir, "eval")
        baselines = build_statistical_baselines(train_dir)
        if baseline_name not in baselines:
            raise KeyError(
                f"Baseline {baseline_name!r} not produced for {ds!r}; "
                f"build_statistical_baselines returned {sorted(baselines)}."
            )
        source_specs[ds] = baselines[baseline_name]
        target_lens[ds] = _eval_completion_lengths(eval_dir)
        if not target_lens[ds]:
            print(
                f"  [warn] {ds}: no has_eos=True eval examples; column will be NaN."
            )

    n = len(datasets)
    Z = np.full((n, n), np.nan)
    for i, src in enumerate(datasets):
        for j, tgt in enumerate(datasets):
            if not target_lens[tgt]:
                continue
            cell = _compute_cell(source_specs[src], target_lens[tgt], metric_fn, epsilon)
            if cell is not None and not (math.isnan(cell) or math.isinf(cell)):
                Z[i, j] = cell
    return Z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        required=True,
        type=str,
        nargs="+",
        help="Dataset slugs. Used as both source rows and target columns.",
    )
    parser.add_argument(
        "--metadata-pattern",
        required=True,
        type=str,
        help="Format string with `{dataset}` placeholder, resolving to a "
        "directory containing `train/` and `eval/` subdirs of "
        "`sample_*.json` metadata files. The conventional path is the "
        "`hidden_states/` directory written by `extract_hidden_states.py` "
        "(e.g. `outputs/qwen3_4b-{dataset}/hidden_states`); any "
        "`.safetensors` tensors in there are ignored — this script reads "
        "only the per-example JSONs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Parent directory for the per-baseline figures + CSVs.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Subset of baseline names to plot; defaults to all four built by "
            "`build_statistical_baselines` (constant_mean / constant_median / "
            "mean_plus_position / median_plus_position). One PDF + CSV pair "
            "is written per baseline."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mae",
        choices=sorted(_METRIC_FNS.keys()),
        help="Per-token metric to aggregate (length-weighted across the eval split).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Blues_r",
        help="matplotlib colormap. `Blues_r` for lower-is-better metrics (default).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Numerical floor used by MAPE / SMAPE; ignored for MAE.",
    )
    args = parser.parse_args()

    # Default: all four baselines. Fit one to discover the available names.
    if args.baselines is None:
        first_meta = Path(args.metadata_pattern.format(dataset=args.datasets[0]))
        first_train = _resolve_split_dir(first_meta, "train")
        baseline_names = list(build_statistical_baselines(first_train).keys())
        # Surface only the four canonical regression baselines by default;
        # downstream readers tend to want the constant + length-minus-position
        # quartet, not the seqlen_* aliases.
        canonical = [
            "baseline_constant_mean",
            "baseline_constant_median",
            "baseline_mean_plus_position",
            "baseline_median_plus_position",
        ]
        baseline_names = [n for n in canonical if n in baseline_names]
    else:
        baseline_names = list(args.baselines)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Datasets   : {args.datasets}")
    print(f"Baselines  : {baseline_names}")
    print(f"Metric     : {args.metric}")
    print(f"Output dir : {output_dir}")

    n = len(args.datasets)
    for name in baseline_names:
        print(f"\n[{name}]")
        Z_mean = _build_matrix(
            datasets=args.datasets,
            metadata_pattern=args.metadata_pattern,
            baseline_name=name,
            metric_name=args.metric,
            epsilon=args.epsilon,
        )
        # Baselines are deterministic — no seeds, no probe weights — so std
        # is zero everywhere (NaN where mean is NaN, to match the heatmap's
        # "no data" rendering).
        Z_std = np.where(np.isnan(Z_mean), np.nan, 0.0)

        for i, src in enumerate(args.datasets):
            cells = " ".join(
                f"{tgt}={Z_mean[i, j]:.3f}" if not np.isnan(Z_mean[i, j]) else f"{tgt}=—"
                for j, tgt in enumerate(args.datasets)
            )
            print(f"  src={src:<22s}  {cells}")

        pdf_path = output_dir / f"{name}__{args.metric}.pdf"
        title = (
            f"Cross-dataset {name} / {args.metric} "
            f"(length-weighted, baseline-only)\n"
            f"rows = source (train), cols = target (eval)"
        )
        _render_heatmap(
            datasets=args.datasets,
            Z_mean=Z_mean,
            Z_std=Z_std,
            output_path=pdf_path,
            title=title,
            cmap=args.cmap,
        )
        print(f"  saved: {pdf_path}")
        _write_heatmap_csvs(args.datasets, Z_mean, Z_std, pdf_path)


if __name__ == "__main__":
    main()
