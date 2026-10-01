"""T-004: does drift carry signal beyond problem length? (CPU, tokenizers only)

Pre-registered criterion (collab/tasks/T-004.md): residualized drift adds
signal beyond length if BOTH hold on DeepSeek nested CV:
  (a) L+R beats L in balanced accuracy by >= 0.03 (mean over 25 outer folds)
      and wins in >= 60% of folds;
  (b) R alone has AUC >= 0.55.

    python scripts/length_control.py \
        --dump-dir results/feature-dumps/2026-09-29_v3-features_tf5.13_unbatched-fp16 \
        --csv results/grok/T-003/math-robust-final.csv \
        --out results/grok/T-004
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from transformers import AutoTokenizer

# Ensure sibling evaluate/specificity imports work when run as a script.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluate import GRID, SEEDS, nested_cv  # noqa: E402
from specificity import (  # noqa: E402
    N_SHUFFLES,
    ROW_ALIASES,
    SHUFFLE_SEED,
    auc_predict_failure,
    load_type_failures,
    one_sided_p,
)

V3_SLICE = slice(87, 113)
COS_MEAN = "multidistract_cos_mean_d0.1"
DUMP_NAMES = {
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz",
    "Qwen/Qwen3.5-4B": "Qwen_Qwen3.5-4B.npz",
    "Skywork/Skywork-OR1-Math-7B": "Skywork_Skywork-OR1-Math-7B.npz",
    "allenai/Olmo-3-7B-Think": "allenai_Olmo-3-7B-Think.npz",
}
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


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


def length_features(problems: list[str], model_id: str) -> dict[str, np.ndarray]:
    """n_words, n_chars, n_tokens and L = [log n_tokens, log n_words]."""
    n_words = np.array([max(len(p.split()), 1) for p in problems], dtype=np.float64)
    n_chars = np.array([max(len(p), 1) for p in problems], dtype=np.float64)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    n_tokens = np.array(
        [max(len(tokenizer.encode(p, add_special_tokens=False)), 1) for p in problems],
        dtype=np.float64,
    )
    L = np.column_stack([np.log(n_tokens), np.log(n_words)])
    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "L": L,
    }


def residualize_D(D: np.ndarray, log_n_tokens: np.ndarray) -> np.ndarray:
    """Label-free OLS residualization: each column ~ a + b * log(n_tokens)."""
    x = log_n_tokens.reshape(-1, 1)
    R = np.empty_like(D, dtype=np.float64)
    for j in range(D.shape[1]):
        y = D[:, j]
        if np.nanstd(y) < 1e-12:
            R[:, j] = 0.0
            continue
        reg = LinearRegression()
        reg.fit(x, y)
        R[:, j] = y - reg.predict(x)
    return R


def drop_constant(X: np.ndarray, names: list[str] | None = None):
    varying = X.std(0) > 1e-9
    X2 = X[:, varying]
    if names is None:
        return X2, varying
    return X2, [n for n, keep in zip(names, varying) if keep], varying


def oof_decision(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Average held-out decision_function over seeds 0..4 (higher = more robust)."""
    n = len(labels)
    totals = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for seed in SEEDS:
        outer = StratifiedKFold(5, shuffle=True, random_state=seed)
        for train, test in outer.split(features, labels):
            pipeline = make_pipeline(
                StandardScaler(),
                SelectKBest(f_classif),
                LogisticRegression(max_iter=3000, class_weight="balanced"),
            )
            search = GridSearchCV(
                pipeline,
                GRID,
                cv=StratifiedKFold(4, shuffle=True, random_state=seed),
                scoring="balanced_accuracy",
            )
            search.fit(features[train], labels[train])
            totals[test] += search.decision_function(features[test])
            counts[test] += 1
    if not np.all(counts == len(list(SEEDS))):
        raise RuntimeError(f"OOF coverage unexpected: {np.unique(counts, return_counts=True)}")
    return totals / counts


def train_full_probe(features: np.ndarray, labels: np.ndarray):
    """Fit the evaluate.py pipeline on all rows (for cross-model transfer)."""
    pipeline = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    search = GridSearchCV(
        pipeline,
        GRID,
        cv=StratifiedKFold(4, shuffle=True, random_state=0),
        scoring="balanced_accuracy",
    )
    search.fit(features, labels)
    return search


def specificity_table_for_scores(
    dump: dict,
    csv_path: Path,
    score_sets: dict[str, np.ndarray],
    single_R_cos: np.ndarray,
) -> pd.DataFrame:
    """T-003-style table for multiple OOF scores + residualized cos_mean."""
    problems = dump["problems"]
    model_id = dump["model_id"]
    type_failures = load_type_failures(csv_path, model_id, problems)
    problem_index = {p: i for i, p in enumerate(problems)}
    rng = np.random.default_rng(SHUFFLE_SEED)

    rows = []
    for ptype in sorted(type_failures):
        failed_by_problem = type_failures[ptype]
        idxs = [problem_index[p] for p in failed_by_problem.index]
        failed = failed_by_problem.to_numpy()
        row = {
            "type": ptype,
            "n_problems": int(len(failed)),
            "n_failed": int(failed.sum()),
        }
        for name, score in score_sets.items():
            value = score[idxs]
            auc = auc_predict_failure(failed, value)
            p = one_sided_p(failed, value, auc, rng)
            row[f"AUC {name}"] = auc
            row[f"p {name}"] = p
        # Single residual feature.
        value = single_R_cos[idxs]
        auc = auc_predict_failure(failed, value)
        p = one_sided_p(failed, value, auc, rng)
        row["AUC R[cos_mean_d0.1]"] = auc
        row["p R[cos_mean_d0.1]"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)

    # --- Load DeepSeek ---
    ds_path = arguments.dump_dir / DUMP_NAMES[DEEPSEEK]
    ds = load_dump(ds_path)
    assert ds["model_id"] == DEEPSEEK
    labels = ds["labels"]
    columns = ds["columns"]
    D_raw = ds["features"][:, V3_SLICE]
    d_cols = columns[V3_SLICE]
    assert COS_MEAN in d_cols
    cos_idx = d_cols.index(COS_MEAN)

    lengths = length_features(ds["problems"], DEEPSEEK)
    L = lengths["L"]
    log_tok = np.log(lengths["n_tokens"])
    R_full = residualize_D(D_raw, log_tok)

    D, d_names, d_mask = drop_constant(D_raw, d_cols)
    R, r_names, r_mask = drop_constant(R_full, d_cols)
    L_use, _ = drop_constant(L)

    # Feature sets for nested CV.
    sets = {
        "L": L_use,
        "D": D,
        "R": R,
        "L+D": np.hstack([L_use, D]),
        "L+R": np.hstack([L_use, R]),
    }

    # --- Step 0 ---
    n_words = lengths["n_words"]
    cos_mean = D_raw[:, cos_idx]
    auc_n_words = float(roc_auc_score(labels, n_words))
    spearman_rho = float(spearmanr(n_words, cos_mean).statistic)
    step0 = {
        "AUC_n_words": auc_n_words,
        "Spearman_n_words_cos_mean_d0.1": spearman_rho,
        "n_words_min": float(n_words.min()),
        "n_words_max": float(n_words.max()),
        "n_words_mean_robust": float(n_words[labels].mean()),
        "n_words_mean_nonrobust": float(n_words[~labels].mean()),
    }
    print("step 0:", json.dumps(step0, indent=2))

    # --- Steps 1–2: nested CV ---
    nested = {}
    fold_balanced = {}
    for name, X in sets.items():
        mean_acc, std_acc, mean_bal, mean_auc, bal_folds, auc_folds = nested_cv(
            X, labels, "balanced", return_folds=True
        )
        nested[name] = {
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "balanced_mean": mean_bal,
            "auc_mean": mean_auc,
            "n_features": int(X.shape[1]),
        }
        fold_balanced[name] = bal_folds
        print(
            f"nested {name:4s}: balanced {mean_bal:.4f} | AUC {mean_auc:.4f} "
            f"| n_feat={X.shape[1]}"
        )

    L_folds = np.asarray(fold_balanced["L"], dtype=np.float64)
    LR_folds = np.asarray(fold_balanced["L+R"], dtype=np.float64)
    diffs = LR_folds - L_folds
    mean_diff = float(diffs.mean())
    pct_wins = float((diffs > 0).mean() * 100.0)
    # Ties: count as not wins for the >= 60% criterion.
    criterion_a = bool(mean_diff >= 0.03 and pct_wins >= 60.0)
    criterion_b = bool(nested["R"]["auc_mean"] >= 0.55)
    paired = {
        "mean_balanced_diff_LpR_minus_L": mean_diff,
        "pct_folds_LpR_wins": pct_wins,
        "n_folds": int(len(diffs)),
        "n_wins": int((diffs > 0).sum()),
        "n_ties": int((diffs == 0).sum()),
        "n_losses": int((diffs < 0).sum()),
        "criterion_a_met": criterion_a,
        "criterion_b_met": criterion_b,
        "both_met": bool(criterion_a and criterion_b),
    }
    print("paired L+R vs L:", json.dumps(paired, indent=2))

    # Save per-fold balanced accuracies for audit.
    fold_df = pd.DataFrame({k: fold_balanced[k] for k in sets})
    fold_df["LpR_minus_L"] = diffs
    fold_df.to_csv(out / "nested_cv_fold_balanced.csv", index_label="fold")

    # --- Step 3: T-003 table with R / L+R OOF scores ---
    score_R = oof_decision(R, labels)
    score_LR = oof_decision(sets["L+R"], labels)
    R_cos = R_full[:, cos_idx]
    spec = specificity_table_for_scores(
        ds,
        arguments.csv,
        {"R_score": score_R, "L+R_score": score_LR},
        R_cos,
    )
    spec.to_csv(out / "specificity_R_table.csv", index=False)
    print("\nT-003-style table (R / L+R OOF + R[cos_mean]):")
    print(spec.to_string(index=False))

    # --- Step 4: cross-model proxy ---
    # Train probe on DeepSeek R (and L); apply to each other model's own-fit R/L.
    probe_R = train_full_probe(R, labels)
    probe_L = train_full_probe(L_use, labels)
    deepseek_labels_by_problem = {p: bool(lab) for p, lab in zip(ds["problems"], labels)}

    cross_rows = []
    for model_id, fname in DUMP_NAMES.items():
        other = load_dump(arguments.dump_dir / fname)
        # Align to DeepSeek problem order / shared problems.
        other_index = {p: i for i, p in enumerate(other["problems"])}
        shared = [p for p in ds["problems"] if p in other_index]
        assert len(shared) == len(ds["problems"]), f"problem mismatch for {model_id}"
        idxs = [other_index[p] for p in shared]
        y = np.array([deepseek_labels_by_problem[p] for p in shared], dtype=bool)

        olen = length_features(other["problems"], model_id)
        oL_all = olen["L"]
        oD = other["features"][:, V3_SLICE]
        oR_all = residualize_D(oD, np.log(olen["n_tokens"]))

        # Same column mask as DeepSeek varying features for R/D transfer.
        oR = oR_all[idxs][:, r_mask]
        # L is always [log n_tokens, log n_words]; match DeepSeek L_use columns.
        oL = oL_all[idxs]
        if oL.shape[1] != L_use.shape[1]:
            raise RuntimeError(f"L width mismatch for {model_id}: {oL.shape} vs {L_use.shape}")

        if model_id == DEEPSEEK:
            score_r = probe_R.decision_function(R)
            score_l = probe_L.decision_function(L_use)
        else:
            score_r = probe_R.decision_function(oR)
            score_l = probe_L.decision_function(oL)

        auc_r = float(roc_auc_score(y, score_r))
        auc_l = float(roc_auc_score(y, score_l))
        cross_rows.append(
            {
                "model_id": model_id,
                "n": int(len(y)),
                "AUC_R": auc_r,
                "AUC_L": auc_l,
                "n_tokens_mean": float(olen["n_tokens"][idxs].mean()),
                "n_words_mean": float(olen["n_words"][idxs].mean()),
            }
        )
        print(f"cross-model {model_id}: AUC_R={auc_r:.4f} AUC_L={auc_l:.4f}")

    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(out / "cross_model_auc.csv", index=False)

    # Length table for audit.
    length_audit = pd.DataFrame(
        {
            "problem_idx": np.arange(len(ds["problems"])),
            "n_words": lengths["n_words"],
            "n_chars": lengths["n_chars"],
            "n_tokens": lengths["n_tokens"],
            "label": labels.astype(int),
            "cos_mean_d0.1": cos_mean,
            "R_cos_mean_d0.1": R_cos,
        }
    )
    length_audit.to_csv(out / "deepseek_lengths.csv", index=False)

    summary = {
        "step0": step0,
        "nested_cv": nested,
        "paired_LpR_vs_L": paired,
        "cross_model": cross_rows,
        "dump": ds["path"],
        "csv": str(arguments.csv),
        "n_problems": int(len(labels)),
        "n_robust": int(labels.sum()),
        "v3_n_raw": int(D_raw.shape[1]),
        "v3_n_varying_D": int(D.shape[1]),
        "v3_n_varying_R": int(R.shape[1]),
        "criterion_a_threshold": {"mean_diff": 0.03, "pct_wins": 60.0},
        "criterion_b_threshold_auc_R": 0.55,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
