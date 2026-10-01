"""Failure-type specificity: do distractor features predict distractor failures?

Pre-registered (collab/tasks/T-003.md): among permutation types, AUC is highest
for ``distract``. Computes OOF robustness margins (v3 evaluate.py pipeline on
columns 87-112) and single-feature AUCs against per-type failure labels from
math-robust-final.csv.

    python scripts/specificity.py \
        --dump results/feature-dumps/.../deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz \
        --csv /path/to/math-robust-final.csv \
        --out results/grok/T-003
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Match evaluate.py exactly.
SEEDS = range(5)
GRID = {
    "selectkbest__k": [2, 5, 10, 20, "all"],
    "logisticregression__C": [0.01, 0.1, 1.0],
}
V3_SLICE = slice(87, 113)
N_SHUFFLES = 1000
SHUFFLE_SEED = 0
SINGLE_FEATURES = (
    "multidistract_cos_mean_d0.1",
    "multidistract_cos_min_d0.1",
    "rename_poolcos_d0.25",
    "typos_poolcos_d0.25",
)
ROW_ALIASES = {
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ("qwen3-8b:low",),
}


def oof_robustness_margin(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
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


def auc_predict_failure(failed: np.ndarray, value: np.ndarray) -> float:
    """Higher AUC = value predicts failure (via negation of robustness-like value)."""
    return float(roc_auc_score(failed.astype(bool), -np.asarray(value, dtype=np.float64)))


def one_sided_p(failed: np.ndarray, value: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    """Permutation p: fraction of label-shuffles with AUC >= observed (plus-one)."""
    failed = np.asarray(failed, dtype=bool)
    value = np.asarray(value, dtype=np.float64)
    ge = 0
    for _ in range(N_SHUFFLES):
        shuffled = rng.permutation(failed)
        null_auc = auc_predict_failure(shuffled, value)
        if null_auc >= observed:
            ge += 1
    return (ge + 1) / (N_SHUFFLES + 1)


def load_type_failures(csv_path: Path, model_id: str, problems: list[str]) -> dict[str, pd.Series]:
    frame = pd.read_csv(csv_path)
    wanted = {model_id, *ROW_ALIASES.get(model_id, ())}
    frame = frame[frame["model_id"].isin(wanted)].copy()
    frame["original_problem"] = frame["original_problem"].astype(str)
    frame = frame[~frame["permutation_type"].astype(str).str.upper().eq("OVERALL")]
    problem_set = set(problems)
    frame = frame[frame["original_problem"].isin(problem_set)]
    out: dict[str, pd.Series] = {}
    for ptype, group in frame.groupby(frame["permutation_type"].astype(str), sort=True):
        if group["original_problem"].duplicated().any():
            raise RuntimeError(f"duplicate problems for type {ptype!r}")
        failed = group.set_index("original_problem")["absolute_accuracy_decay"].gt(0)
        out[str(ptype)] = failed
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    dump = np.load(arguments.dump, allow_pickle=True)
    features_all = np.asarray(dump["features"], dtype=np.float64)
    labels = np.asarray(dump["labels"], dtype=bool)
    problems = [str(p) for p in dump["problems"]]
    columns = [str(c) for c in dump["columns"]]
    model_id = str(dump["model_id"])

    v3 = features_all[:, V3_SLICE]
    v3_columns = columns[V3_SLICE]
    varying = v3.std(0) > 1e-9
    v3 = v3[:, varying]
    v3_columns = [c for c, keep in zip(v3_columns, varying) if keep]

    score = oof_robustness_margin(v3, labels)
    feature_values = {
        name: features_all[:, columns.index(name)]
        for name in SINGLE_FEATURES
    }

    type_failures = load_type_failures(arguments.csv, model_id, problems)
    problem_index = {p: i for i, p in enumerate(problems)}

    rng = np.random.default_rng(SHUFFLE_SEED)
    rows = []
    for ptype in sorted(type_failures):
        failed_by_problem = type_failures[ptype]
        idxs = [problem_index[p] for p in failed_by_problem.index]
        failed = failed_by_problem.to_numpy()
        n_problems = int(len(failed))
        n_failed = int(failed.sum())

        values = {
            "score": score[idxs],
            "cos_mean_d0.1": feature_values["multidistract_cos_mean_d0.1"][idxs],
            "cos_min_d0.1": feature_values["multidistract_cos_min_d0.1"][idxs],
            "rename_poolcos": feature_values["rename_poolcos_d0.25"][idxs],
            "typos_poolcos": feature_values["typos_poolcos_d0.25"][idxs],
        }
        cell = {}
        for key, value in values.items():
            auc = auc_predict_failure(failed, value)
            p = one_sided_p(failed, value, auc, rng)
            cell[key] = (auc, p)

        rows.append(
            {
                "type": ptype,
                "n_problems": n_problems,
                "n_failed": n_failed,
                "AUC score": cell["score"][0],
                "p": cell["score"][1],
                "AUC cos_mean_d0.1": cell["cos_mean_d0.1"][0],
                "AUC cos_min_d0.1": cell["cos_min_d0.1"][0],
                "AUC rename_poolcos": cell["rename_poolcos"][0],
                "AUC typos_poolcos": cell["typos_poolcos"][0],
                # Extra p-values for audit (not in report schema table).
                "p_cos_mean_d0.1": cell["cos_mean_d0.1"][1],
                "p_cos_min_d0.1": cell["cos_min_d0.1"][1],
                "p_rename_poolcos": cell["rename_poolcos"][1],
                "p_typos_poolcos": cell["typos_poolcos"][1],
            }
        )

    table = pd.DataFrame(rows)
    arguments.out.mkdir(parents=True, exist_ok=True)
    table_path = arguments.out / "specificity_table.csv"
    table.to_csv(table_path, index=False)

    meta = {
        "dump": str(arguments.dump),
        "csv": str(arguments.csv),
        "model_id": model_id,
        "n_dump_problems": len(problems),
        "v3_n_features": len(v3_columns),
        "n_shuffles": N_SHUFFLES,
        "single_features": list(SINGLE_FEATURES),
    }
    (arguments.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Schema columns for stdout / report.
    schema_cols = [
        "type",
        "n_problems",
        "n_failed",
        "AUC score",
        "p",
        "AUC cos_mean_d0.1",
        "AUC cos_min_d0.1",
        "AUC rename_poolcos",
        "AUC typos_poolcos",
    ]
    print(table[schema_cols].to_string(index=False))
    print(f"\nwrote {table_path}")


if __name__ == "__main__":
    main()
