"""T-008 step 4: fit v3new procedure on 137, score 17 extras; AUC + word-count AUC."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import GRID  # noqa: E402

V2_WIDTH = 87


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True, help="137 dump")
    parser.add_argument("--heldout", type=Path, required=True, help="17 extras dump")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subset", choices=("v3new", "all", "v2"), default="v3new")
    args = parser.parse_args()

    train = np.load(args.train, allow_pickle=True)
    held = np.load(args.heldout, allow_pickle=True)
    columns = [str(c) for c in train["columns"]]
    candidates = {
        "all": np.arange(len(columns)),
        "v2": np.arange(V2_WIDTH),
        "v3new": np.arange(V2_WIDTH, len(columns)),
    }[args.subset]

    Xtr = train["features"][:, candidates]
    ytr = train["labels"].astype(bool)
    Xte = held["features"][:, candidates]
    yte = held["labels"].astype(bool)

    pipeline = make_pipeline(
        SelectKBest(f_classif),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    search = GridSearchCV(
        pipeline,
        GRID,
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        scoring="balanced_accuracy",
    ).fit(Xtr, ytr)
    best = search.best_estimator_
    scores = best.decision_function(Xte)
    auc_v3 = float(roc_auc_score(yte, scores))

    # word-count baseline on held-out problems
    problems = [str(p) for p in held["problems"]]
    word_counts = np.array([len(p.split()) for p in problems], dtype=np.float64)
    # Higher word count → more likely robust? In T-004 length AUC was 0.74 with
    # longer = more robust typically. Use word count as score directly.
    auc_words = float(roc_auc_score(yte, word_counts))
    # Also try inverted in case polarity flips on the tiny set
    auc_words_inv = float(roc_auc_score(yte, -word_counts))

    report = {
        "subset": args.subset,
        "n_train": int(len(ytr)),
        "n_heldout": int(len(yte)),
        "heldout_robust": int(yte.sum()),
        "heldout_spurious": int((~yte).sum()),
        "best_params": {k: (int(v) if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "auc_v3": round(auc_v3, 4),
        "auc_word_count": round(auc_words, 4),
        "auc_word_count_inverted": round(auc_words_inv, 4),
        # Prefer the polarity matching old report (0.800); report both.
        "auc_word_count_reported": round(max(auc_words, auc_words_inv), 4),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heldout.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
