"""T-005: three candidate signals beyond length (CPU, pre-registered).

No feature selection: StandardScaler -> balanced LogisticRegression,
inner 4-fold CV over C in {0.01, 0.1, 1}; outer StratifiedKFold(5,
shuffle=True, random_state=s) for s=0..4. Same splits for every set.

Pre-registered criterion per candidate: L+X beats L by >= 0.02 mean
balanced accuracy AND wins >= 60% of non-tied folds.

    python scripts/beyond_length.py \
        --dump results/feature-dumps/2026-09-29_v3-features_tf5.13_unbatched-fp16/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz \
        --out results/grok/T-005
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

SEEDS = range(5)
C_GRID = {"logisticregression__C": [0.01, 0.1, 1.0]}
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
N_BOOT = 500
BOOT_SEED = 0
EPS = 1e-6

X1_NAMES = [
    "rename_kl",
    "numerals_kl",
    "typos_kl",
    "distract_kl",
    "rephrase_kl",
    "number_jitter_kl",
    "multidistract_kl_mean",
    "multidistract_kl_max",
]
X2_NAMES = ["base_entropy", "base_top1", "base_margin"]
# X3 built from these base columns via g(c)=log(1-c+1e-6) differences.
X3_BASE = [
    "multidistract_cos_mean_d0.25",
    "rephrase_cos_d0.25",
    "multidistract_poolcos_mean_d0.25",
    "rephrase_poolcos_d0.25",
    "typos_cos_d0.25",
]
X3_DERIVED = [
    "g_md_cos_mean_d0.25_minus_g_rephrase_cos_d0.25",
    "g_md_poolcos_mean_d0.25_minus_g_rephrase_poolcos_d0.25",
    "g_typos_cos_d0.25_minus_g_rephrase_cos_d0.25",
]


def g(c: np.ndarray) -> np.ndarray:
    return np.log(1.0 - c + EPS)


def load_dump(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "features": np.asarray(z["features"], dtype=np.float64),
        "labels": np.asarray(z["labels"], dtype=bool),
        "problems": [str(p) for p in z["problems"]],
        "columns": [str(c) for c in z["columns"]],
        "model_id": str(z["model_id"]),
        "path": str(path),
    }


def length_features(problems: list[str], model_id: str) -> np.ndarray:
    """L = [log n_tokens, log n_words]."""
    n_words = np.array([max(len(p.split()), 1) for p in problems], dtype=np.float64)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    n_tokens = np.array(
        [max(len(tokenizer.encode(p, add_special_tokens=False)), 1) for p in problems],
        dtype=np.float64,
    )
    return np.column_stack([np.log(n_tokens), np.log(n_words)]), n_tokens, n_words


def col_matrix(dump: dict, names: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(dump["columns"])}
    missing = [n for n in names if n not in idx]
    if missing:
        raise KeyError(f"missing columns in dump: {missing}")
    return dump["features"][:, [idx[n] for n in names]]


def build_X3(dump: dict) -> tuple[np.ndarray, list[str]]:
    base = col_matrix(dump, X3_BASE)
    # order: md_cos, rephrase_cos, md_pool, rephrase_pool, typos_cos
    md_cos, reph_cos, md_pool, reph_pool, typos_cos = (base[:, i] for i in range(5))
    X3 = np.column_stack(
        [
            g(md_cos) - g(reph_cos),
            g(md_pool) - g(reph_pool),
            g(typos_cos) - g(reph_cos),
        ]
    )
    return X3, list(X3_DERIVED)


def make_search(seed: int) -> GridSearchCV:
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    return GridSearchCV(
        pipeline,
        C_GRID,
        cv=StratifiedKFold(4, shuffle=True, random_state=seed),
        scoring="balanced_accuracy",
    )


def nested_cv_no_select(
    features: np.ndarray, labels: np.ndarray
) -> tuple[float, float, list[float], list[float]]:
    """Outer 5×5 nested CV; no SelectKBest. Returns mean_bal, mean_auc, bal_folds, auc_folds."""
    balanced, auc = [], []
    for seed in SEEDS:
        outer = StratifiedKFold(5, shuffle=True, random_state=seed)
        for train, test in outer.split(features, labels):
            search = make_search(seed)
            search.fit(features[train], labels[train])
            pred = search.predict(features[test])
            balanced.append(balanced_accuracy_score(labels[test], pred))
            if len(set(labels[test])) > 1:
                auc.append(roc_auc_score(labels[test], search.decision_function(features[test])))
            else:
                auc.append(float("nan"))
    return float(np.mean(balanced)), float(np.nanmean(auc)), list(balanced), list(auc)


def paired_vs_L(x_folds: list[float], l_folds: list[float]) -> dict:
    diffs = np.asarray(x_folds, dtype=np.float64) - np.asarray(l_folds, dtype=np.float64)
    n_wins = int((diffs > 0).sum())
    n_ties = int((diffs == 0).sum())
    n_losses = int((diffs < 0).sum())
    non_tied = n_wins + n_losses
    pct_non_tied_wins = float(n_wins / non_tied * 100.0) if non_tied else float("nan")
    mean_diff = float(diffs.mean())
    # Pre-registered: mean_diff >= 0.02 AND wins >= 60% of non-tied folds.
    criterion = bool(mean_diff >= 0.02 and pct_non_tied_wins >= 60.0)
    return {
        "mean_diff": mean_diff,
        "n_wins": n_wins,
        "n_ties": n_ties,
        "n_losses": n_losses,
        "pct_non_tied_wins": pct_non_tied_wins,
        "criterion_met": criterion,
    }


def bootstrap_coef_over_se(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """Full-fit coef / bootstrap SE for each feature (descriptive)."""
    search = make_search(0)
    search.fit(features, labels)
    # Pipeline: StandardScaler -> LogisticRegression
    coef = search.best_estimator_.named_steps["logisticregression"].coef_.ravel()
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, features.shape[1]), dtype=np.float64)
    n = len(labels)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = labels[idx]
        # Need both classes for LR; redraw if needed.
        tries = 0
        while len(np.unique(yb)) < 2 and tries < 50:
            idx = rng.integers(0, n, size=n)
            yb = labels[idx]
            tries += 1
        if len(np.unique(yb)) < 2:
            boot[b] = np.nan
            continue
        sb = make_search(0)
        sb.fit(features[idx], yb)
        boot[b] = sb.best_estimator_.named_steps["logisticregression"].coef_.ravel()
    se = np.nanstd(boot, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = coef / se
    return pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coef,
            "bootstrap_se": se,
            "coef_over_se": z,
            "best_C": search.best_params_["logisticregression__C"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    dump = load_dump(args.dump)
    assert dump["model_id"] == DEEPSEEK, dump["model_id"]
    labels = dump["labels"]
    n = len(labels)
    print(f"n={n} robust={int(labels.sum())} dump={dump['path']}")

    # Verify names.
    needed = X1_NAMES + X2_NAMES + X3_BASE
    missing = [c for c in needed if c not in dump["columns"]]
    if missing:
        raise SystemExit(f"missing feature names: {missing}")
    print("all requested feature names present exactly")

    L, n_tokens, n_words = length_features(dump["problems"], DEEPSEEK)
    X1 = col_matrix(dump, X1_NAMES)
    X2 = col_matrix(dump, X2_NAMES)
    X3, x3_names = build_X3(dump)

    sets: dict[str, tuple[np.ndarray, list[str]]] = {
        "L": (L, ["log_n_tokens", "log_n_words"]),
        "X1": (X1, list(X1_NAMES)),
        "X2": (X2, list(X2_NAMES)),
        "X3": (X3, x3_names),
        "L+X1": (np.hstack([L, X1]), ["log_n_tokens", "log_n_words"] + list(X1_NAMES)),
        "L+X2": (np.hstack([L, X2]), ["log_n_tokens", "log_n_words"] + list(X2_NAMES)),
        "L+X3": (np.hstack([L, X3]), ["log_n_tokens", "log_n_words"] + x3_names),
    }

    nested = {}
    fold_bal = {}
    fold_auc = {}
    for name, (X, feat_names) in sets.items():
        mean_bal, mean_auc, bal_folds, auc_folds = nested_cv_no_select(X, labels)
        nested[name] = {
            "balanced_mean": mean_bal,
            "auc_mean": mean_auc,
            "n_features": int(X.shape[1]),
            "feature_names": feat_names,
        }
        fold_bal[name] = bal_folds
        fold_auc[name] = auc_folds
        print(f"nested {name:5s}: balanced {mean_bal:.4f} | AUC {mean_auc:.4f} | n_feat={X.shape[1]}")

    l_folds = fold_bal["L"]
    paired = {}
    for cand in ("X1", "X2", "X3"):
        key = f"L+{cand}"
        paired[key] = paired_vs_L(fold_bal[key], l_folds)
        p = paired[key]
        print(
            f"paired {key} vs L: mean_diff={p['mean_diff']:+.4f} "
            f"W/T/L={p['n_wins']}/{p['n_ties']}/{p['n_losses']} "
            f"non_tied_wins%={p['pct_non_tied_wins']:.1f} criterion={p['criterion_met']}"
        )

    # Per-fold audit.
    fold_df = pd.DataFrame({k: fold_bal[k] for k in sets})
    for cand in ("X1", "X2", "X3"):
        fold_df[f"L+{cand}_minus_L"] = (
            np.asarray(fold_bal[f"L+{cand}"]) - np.asarray(fold_bal["L"])
        )
    fold_df.to_csv(out / "nested_cv_fold_balanced.csv", index_label="fold")

    # Bootstrap coef/SE for each L+X full fit (X features only reported, but fit includes L).
    coef_tables = {}
    for cand in ("X1", "X2", "X3"):
        key = f"L+{cand}"
        X, feat_names = sets[key]
        table = bootstrap_coef_over_se(X, labels, feat_names)
        # Keep only X features in the descriptive table (L is control).
        x_only = table[~table["feature"].isin(["log_n_tokens", "log_n_words"])].copy()
        x_only.to_csv(out / f"coef_bootstrap_{key.replace('+', '')}.csv", index=False)
        coef_tables[key] = x_only.to_dict(orient="records")
        print(f"\ncoef/SE {key}:")
        print(x_only.to_string(index=False))

    # Length audit (reproduce L).
    pd.DataFrame(
        {
            "problem_idx": np.arange(n),
            "n_words": n_words,
            "n_tokens": n_tokens,
            "label": labels.astype(int),
        }
    ).to_csv(out / "deepseek_lengths.csv", index=False)

    # Feature matrix for X3 audit.
    pd.DataFrame(X3, columns=x3_names).assign(label=labels.astype(int)).to_csv(
        out / "X3_features.csv", index_label="problem_idx"
    )

    summary = {
        "n_problems": int(n),
        "n_robust": int(labels.sum()),
        "dump": dump["path"],
        "model_id": dump["model_id"],
        "feature_names_verified_exact": True,
        "nested_cv": nested,
        "paired_vs_L": paired,
        "coef_bootstrap": coef_tables,
        "criterion": {
            "mean_balanced_diff_min": 0.02,
            "pct_non_tied_wins_min": 60.0,
            "note": "L+X beats L by >=0.02 mean balanced AND wins >=60% of non-tied folds",
        },
        "pipeline": "StandardScaler -> LogisticRegression(class_weight=balanced); inner CV C in {0.01,0.1,1}; no SelectKBest",
        "outer": "StratifiedKFold(5, shuffle=True, random_state=s) s=0..4",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
