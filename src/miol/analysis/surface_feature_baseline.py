"""Surface-feature ridge baseline for prompt-end length prediction (paper, appendix "Control experiments").

Answers: how much of the Completion Length Probe's prompt-end accuracy is
achievable from cheap prompt surface features alone — prompt length and a
TF-IDF bag-of-words — with no access to hidden states?

Reads only the `sample_*.json` metadata written by extract_hidden_states.py
(never the tensors), exactly like cross_dataset_baseline_eval.py: fit on the
train split, score on the eval split, naturally-terminated (has_eos) examples
only. Target is the total completion length T. Ridge regression is solved in
closed form (numpy only, no scikit-learn); the penalty λ is selected on a held-out slice of the train
split by MAE, then the model is refit on the full train split.

Three feature sets are reported to show which surface signal carries what: `length` (prompt token/char/word counts), `tfidf` (bag-of-words
only), `all` (both). Floors: the constant-median and constant-mean predictors
fit on the same train split. Compare the eval MAEs against the Completion
Length Probe's prompt-end MAE for the same (model, dataset) cell (prompt-end
table in the paper's appendix; pass --probe-mae to annotate it into the report).

Usage (one or more cells):
    python -m miol.analysis.surface_feature_baseline \
        --hidden-states-dir outputs/llama3.1_8b-gsm8k/hidden_states \
        --probe-mae 42.29 \
        --output reports/surface_ridge/llama3.1_8b-gsm8k.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")
LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def load_split(split_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(split_dir.glob("sample_*.json")):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        if not bool(meta.get("has_eos", False)):
            continue
        if int(meta.get("completion_length", 0)) <= 0:
            continue
        samples.append(meta)
    return samples


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(train_prompts: list[list[str]], max_features: int, min_df: int = 2) -> dict[str, int]:
    df = Counter()
    for toks in train_prompts:
        df.update(set(toks))
    terms = [t for t, c in df.items() if c >= min_df]
    terms.sort(key=lambda t: (-df[t], t))
    return {t: i for i, t in enumerate(terms[:max_features])}


def tfidf_matrix(prompts_tokens: list[list[str]], vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    X = np.zeros((len(prompts_tokens), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(prompts_tokens):
        for t, c in Counter(toks).items():
            j = vocab.get(t)
            if j is not None:
                X[i, j] = c
    X *= idf[None, :]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def compute_idf(train_prompts: list[list[str]], vocab: dict[str, int]) -> np.ndarray:
    n = len(train_prompts)
    df = np.zeros(len(vocab), dtype=np.float64)
    for toks in train_prompts:
        for t in set(toks):
            j = vocab.get(t)
            if j is not None:
                df[j] += 1
    return (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)


def numeric_features(samples: list[dict]) -> np.ndarray:
    rows = [
        (
            float(s["prompt_length"]),          # tokens
            float(len(s["prompt"])),            # characters
            float(len(tokenize(s["prompt"]))),  # words
        )
        for s in samples
    ]
    return np.asarray(rows, dtype=np.float32)


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with an unpenalized bias column appended last."""
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    d = Xb.shape[1]
    penalty = np.full(d, lam, dtype=np.float64)
    penalty[-1] = 0.0  # never penalize the intercept
    A = Xb.T.astype(np.float64) @ Xb.astype(np.float64) + np.diag(penalty)
    b = Xb.T.astype(np.float64) @ y.astype(np.float64)
    return np.linalg.solve(A, b)


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    return Xb.astype(np.float64) @ beta


def fit_eval_ridge(X_train, y_train, X_eval, y_eval, val_frac: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]

    best_lam, best_val_mae = None, math.inf
    for lam in LAMBDA_GRID:
        beta = ridge_fit(X_train[fit_idx], y_train[fit_idx], lam)
        val_mae = float(np.abs(ridge_predict(X_train[val_idx], beta) - y_train[val_idx]).mean())
        if val_mae < best_val_mae:
            best_lam, best_val_mae = lam, val_mae

    beta = ridge_fit(X_train, y_train, best_lam)  # refit on the full train split
    preds = ridge_predict(X_eval, beta)
    return {
        "lambda": best_lam,
        "val_mae": best_val_mae,
        "eval_mae": float(np.abs(preds - y_eval).mean()),
        "eval_median_ae": float(np.median(np.abs(preds - y_eval))),
    }


def analyze_cell(hs_dir: Path, max_features: int, val_frac: float, seed: int) -> dict:
    train = load_split(hs_dir / "train")
    eval_ = load_split(hs_dir / "eval")
    if not train or not eval_:
        raise SystemExit(f"{hs_dir}: empty train ({len(train)}) or eval ({len(eval_)}) after has_eos filter")

    y_train = np.array([float(s["completion_length"]) for s in train])
    y_eval = np.array([float(s["completion_length"]) for s in eval_])

    # Floors: constant predictors fit on the train split.
    median_floor = float(np.abs(y_eval - np.median(y_train)).mean())
    mean_floor = float(np.abs(y_eval - y_train.mean()).mean())

    # Numeric block, standardized with train statistics.
    num_train, num_eval = numeric_features(train), numeric_features(eval_)
    mu, sd = num_train.mean(axis=0), num_train.std(axis=0)
    sd[sd == 0] = 1.0
    num_train = (num_train - mu) / sd
    num_eval = (num_eval - mu) / sd

    # TF-IDF block from train vocabulary.
    toks_train = [tokenize(s["prompt"]) for s in train]
    toks_eval = [tokenize(s["prompt"]) for s in eval_]
    vocab = build_vocab(toks_train, max_features)
    idf = compute_idf(toks_train, vocab)
    tf_train = tfidf_matrix(toks_train, vocab, idf)
    tf_eval = tfidf_matrix(toks_eval, vocab, idf)

    feature_sets = {
        "length": (num_train, num_eval),
        "tfidf": (tf_train, tf_eval),
        "all": (np.concatenate([num_train, tf_train], axis=1), np.concatenate([num_eval, tf_eval], axis=1)),
    }
    result = {
        "hidden_states_dir": str(hs_dir),
        "n_train": len(train),
        "n_eval": len(eval_),
        "vocab_size": len(vocab),
        "median_floor_eval_mae": median_floor,
        "mean_floor_eval_mae": mean_floor,
        "ridge": {},
    }
    for name, (Xtr, Xev) in feature_sets.items():
        result["ridge"][name] = fit_eval_ridge(Xtr, y_train, Xev, y_eval, val_frac, seed)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden-states-dir", type=Path, nargs="+", required=True,
                    help="one or more hidden_states/ dirs containing train/ and eval/ sample_*.json")
    ap.add_argument("--probe-mae", type=float, nargs="*", default=None,
                    help="optional Completion Length Probe prompt-end MAE per dir (same order), for the report")
    ap.add_argument("--output", type=Path, default=None, help="JSON summary path")
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.probe_mae is not None and len(args.probe_mae) not in (0, len(args.hidden_states_dir)):
        raise SystemExit("--probe-mae must have one value per --hidden-states-dir")

    results = []
    for k, hs_dir in enumerate(args.hidden_states_dir):
        res = analyze_cell(hs_dir, args.max_features, args.val_frac, args.seed)
        if args.probe_mae:
            res["completion_length_probe_mae"] = args.probe_mae[k]
        results.append(res)

        print(f"=== {hs_dir}  (train {res['n_train']}, eval {res['n_eval']}, vocab {res['vocab_size']}) ===")
        print(f"  constant-median floor : eval MAE {res['median_floor_eval_mae']:8.2f}")
        print(f"  constant-mean floor   : eval MAE {res['mean_floor_eval_mae']:8.2f}")
        for name in ("length", "tfidf", "all"):
            r = res["ridge"][name]
            print(f"  ridge[{name:6s}]        : eval MAE {r['eval_mae']:8.2f}  "
                  f"(median AE {r['eval_median_ae']:7.2f}, lambda {r['lambda']:g})")
        if "completion_length_probe_mae" in res:
            print(f"  Completion Length Probe: eval MAE {res['completion_length_probe_mae']:8.2f}  (from Table 1)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
