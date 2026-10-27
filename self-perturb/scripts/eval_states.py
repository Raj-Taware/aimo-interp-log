"""T-019 pre-registered test: does the question-only residual stream add signal beyond length?

Written before the T-019 data exists (collab/tasks/T-019.md). DeepSeek, 137
labelled MATH rows. Outer StratifiedKFold(5, shuffle, seed s) for s=0..4; the
same splits for every set. Inside each training fold: StandardScaler -> PCA(32)
on the states (fitted on the fold only), then balanced LogisticRegression with
inner 4-fold CV over C in {0.01, 0.1, 1}.

L = log(words), log(chars) of the problem; H = PCA of one layer x pooling.
Pass for a (layer, pooling) iff L+H beats L by >= 0.03 mean balanced accuracy
AND wins >= 60% of non-tied folds; the per-candidate p-value (one-sided sign
test over non-tied folds) is Holm-corrected over every candidate tried.

    python scripts/eval_states.py results/grok/T-019/<deepseek math states>.npz --out results/grok/T-019/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

SEEDS = range(5)
PCA_DIMS = 32
MIN_GAIN, MIN_WIN_SHARE = 0.03, 0.60


def length_features(problems) -> np.ndarray:
    return np.log(np.array([[len(p.split()), len(p)] for p in problems], dtype=float))


def model(n_length: int, n_states: int):
    """L columns scaled as-is; state columns scaled then PCA'd inside the fold."""
    parts = []
    if n_length:
        parts.append(("L", StandardScaler(), list(range(n_length))))
    if n_states:
        states = list(range(n_length, n_length + n_states))
        parts.append(("H", make_pipeline(StandardScaler(), PCA(PCA_DIMS, random_state=0)), states))
    pipe = make_pipeline(ColumnTransformer(parts), LogisticRegression(max_iter=5000, class_weight="balanced"))
    return pipe


def fold_scores(X: np.ndarray, y: np.ndarray, n_length: int) -> tuple[list[float], list[float]]:
    bal, auc = [], []
    for seed in SEEDS:
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            search = GridSearchCV(model(n_length, X.shape[1] - n_length), {"logisticregression__C": [0.01, 0.1, 1.0]},
                                  cv=StratifiedKFold(4, shuffle=True, random_state=seed), scoring="balanced_accuracy")
            search.fit(X[train], y[train])
            bal.append(balanced_accuracy_score(y[test], search.predict(X[test])))
            auc.append(roc_auc_score(y[test], search.decision_function(X[test])))
    return bal, auc


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("states", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.states, allow_pickle=True)
    y = data["labels"].astype(bool)
    L = length_features(data["problems"])
    base_bal, base_auc = fold_scores(L, y, 2)
    report = {"n": int(len(y)), "robust": int(y.sum()),
              "L": {"balanced": round(float(np.mean(base_bal)), 4), "auc": round(float(np.mean(base_auc)), 4)}}
    pvalues, rows = {}, {}
    for pooling in ("mean", "last"):
        states = data[pooling].astype(np.float32)
        for index, layer in enumerate(data["layers"]):
            key = f"{pooling}_L{int(layer)}"
            H = states[:, index, :]
            h_bal, h_auc = fold_scores(np.hstack([np.zeros((len(y), 0)), H]), y, 0)
            lh_bal, lh_auc = fold_scores(np.hstack([L, H]), y, 2)
            diff = np.array(lh_bal) - np.array(base_bal)
            wins, losses = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
            p = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0
            pvalues[key] = p
            rows[key] = {"H_balanced": round(float(np.mean(h_bal)), 4), "H_auc": round(float(np.mean(h_auc)), 4),
                         "LH_balanced": round(float(np.mean(lh_bal)), 4), "LH_auc": round(float(np.mean(lh_auc)), 4),
                         "gain": round(float(diff.mean()), 4), "wins": wins, "losses": losses,
                         "win_share": round(wins / max(wins + losses, 1), 3), "p": round(p, 4)}
            print(key, rows[key], flush=True)
    for key, p_holm in holm(pvalues).items():
        row = rows[key]
        row["p_holm"] = round(p_holm, 4)
        row["pass"] = bool(row["gain"] >= MIN_GAIN and row["win_share"] >= MIN_WIN_SHARE and p_holm < 0.05)
    report["candidates"] = rows
    report["any_pass"] = any(r["pass"] for r in rows.values())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_states.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"L": report["L"], "any_pass": report["any_pass"],
                      "best": max(rows.items(), key=lambda kv: kv[1]["gain"])}, indent=2))


if __name__ == "__main__":
    main()
