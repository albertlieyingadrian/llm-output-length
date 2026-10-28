"""Exclusion-rate accounting for naturally-terminated filtering (paper, appendix "Exclusion rates").

Walks the direct subdirectories of --outputs-dir; every one that contains a
`hidden_states/` directory is treated as a (model, dataset) cell. For each
cell and each split (train, eval), reads the `sample_*.json` metadata and
counts examples excluded by the naturally-terminated filter: `has_eos=False`
(hit the max_new_tokens cutoff) or `completion_length <= 0`. Writes one JSON
file mapping cell name -> per-split totals and exclusion rates.

Metadata-only (never touches tensors), CPU, seconds per cell.

Usage:
    python -m miol.analysis.exclusion_rates \
        --outputs-dir outputs \
        --output reports/exclusion_rates.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_counts(split_dir: Path) -> dict:
    total = no_eos = empty = 0
    for path in sorted(split_dir.glob("sample_*.json")):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        total += 1
        if not bool(meta.get("has_eos", False)):
            no_eos += 1
        elif int(meta.get("completion_length", 0)) <= 0:
            empty += 1
    excluded = no_eos + empty
    return {
        "n_total": total,
        "n_excluded": excluded,
        "n_no_eos": no_eos,
        "n_empty_completion": empty,
        "exclusion_rate": (excluded / total) if total else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--output", type=Path, default=Path("reports/exclusion_rates.json"))
    args = ap.parse_args()

    # List subdirectories defensively: on shared clusters even stat() on an
    # entry can be denied (ACLs), which would otherwise raise inside the
    # listing itself — before any per-cell handling could catch it.
    cell_dirs: list[Path] = []
    results: dict[str, dict] = {}
    for p in sorted(args.outputs_dir.iterdir()):
        try:
            if p.is_dir():
                cell_dirs.append(p)
        except PermissionError as e:
            print(f"[warn] skipping {p.name}: {e}")
            results[p.name] = {"error": f"permission denied: {e}"}

    for cell_dir in cell_dirs:
        # Unreadable cells (e.g., directories without read permission) are
        # recorded and skipped rather than aborting the whole sweep.
        try:
            hs_dir = cell_dir / "hidden_states"
            if not hs_dir.is_dir():
                continue
            cell = {}
            for split in ("train", "eval"):
                split_dir = hs_dir / split
                cell[split] = split_counts(split_dir) if split_dir.is_dir() else None
            results[cell_dir.name] = cell
        except PermissionError as e:
            print(f"[warn] skipping {cell_dir.name}: {e}")
            results[cell_dir.name] = {"error": f"permission denied: {e}"}

    if not results:
        raise SystemExit(f"No subdirectories with a hidden_states/ directory found under {args.outputs_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'cell':40s} {'split':6s} {'total':>7s} {'excluded':>9s} {'no_eos':>7s} {'empty':>6s} {'rate':>7s}")
    for name, cell in results.items():
        if "error" in cell:
            print(f"{name:40s} SKIPPED ({cell['error']})")
            continue
        for split in ("train", "eval"):
            s = cell[split]
            if s is None:
                print(f"{name:40s} {split:6s} {'--':>7s}")
            elif s["n_total"] == 0:
                print(f"{name:40s} {split:6s} {0:7d}  (no sample_*.json)")
            else:
                print(f"{name:40s} {split:6s} {s['n_total']:7d} {s['n_excluded']:9d} "
                      f"{s['n_no_eos']:7d} {s['n_empty_completion']:6d} {s['exclusion_rate']:6.2%}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
