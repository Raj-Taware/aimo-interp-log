"""Honest evaluation of dumped features: nested CV plus a shuffled-label null.

Nested CV chooses the number of features and the regularization strength
inside each training fold, so no number reported here benefits from having
looked at the test folds. Reports accuracy (the competition metric), balanced
accuracy, and AUC, next to the always-majority baseline.

    python scripts/evaluate.py data/features/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = range(5)
GRID = {
    "selectkbest__k": [2, 5, 10, 20, "all"],
    "logisticregression__C": [0.01, 0.1, 1.0],
}


def nested_cv(features, labels, class_weight, return_folds: bool = False):
    """Run nested CV over seeds 0..4 × 5 outer folds.

    When ``return_folds`` is True, also return parallel lists of per-fold
    balanced accuracies and AUCs (length 25) for paired fold comparisons on
    identical splits.
    """
    accuracy, balanced, auc = [], [], []
    for seed in SEEDS:
        outer = StratifiedKFold(5, shuffle=True, random_state=seed)
        for train, test in outer.split(features, labels):
            pipeline = make_pipeline(
                StandardScaler(),
                SelectKBest(f_classif),
                LogisticRegression(max_iter=3000, class_weight=class_weight),
            )
            search = GridSearchCV(
                pipeline,
                GRID,
                cv=StratifiedKFold(4, shuffle=True, random_state=seed),
                scoring="accuracy" if class_weight is None else "balanced_accuracy",
            )
            search.fit(features[train], labels[train])
            prediction = search.predict(features[test])
            accuracy.append((prediction == labels[test]).mean())
            balanced.append(balanced_accuracy_score(labels[test], prediction))
            if len(set(labels[test])) > 1:
                auc.append(roc_auc_score(labels[test], search.decision_function(features[test])))
            else:
                auc.append(float("nan"))
    mean_acc, std_acc = float(np.mean(accuracy)), float(np.std(accuracy))
    mean_bal = float(np.mean(balanced))
    mean_auc = float(np.nanmean(auc))
    if return_folds:
        return mean_acc, std_acc, mean_bal, mean_auc, list(balanced), list(auc)
    return mean_acc, std_acc, mean_bal, mean_auc


def single_feature_scan(features, labels, columns, permutations=200):
    """Rank features by AUC, against the best AUC reachable by chance."""
    observed = sorted(
        ((abs(roc_auc_score(labels, features[:, i]) - 0.5), roc_auc_score(labels, features[:, i]), name)
         for i, name in enumerate(columns)),
        reverse=True,
    )
    rng = np.random.default_rng(0)
    null = [
        max(abs(roc_auc_score(shuffled, features[:, i]) - 0.5) for i in range(features.shape[1]))
        for shuffled in (rng.permutation(labels) for _ in range(permutations))
    ]
    return observed, np.percentile(null, 95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument(
        "--subset",
        choices=("all", "v2", "v3new"),
        default="all",
        help="v2 = the original 87 features (a prefix of every later dump); "
        "v3new = only the features added after them",
    )
    arguments = parser.parse_args()

    dump = np.load(arguments.dump, allow_pickle=True)
    features, labels = dump["features"], dump["labels"]
    columns = [str(name) for name in dump["columns"]]
    v2_width = 87
    if arguments.subset == "v2":
        features, columns = features[:, :v2_width], columns[:v2_width]
    elif arguments.subset == "v3new":
        features, columns = features[:, v2_width:], columns[v2_width:]
    varying = features.std(0) > 1e-9
    features, columns = features[:, varying], [c for c, keep in zip(columns, varying) if keep]

    print(f"{len(labels)} cases, {int(labels.sum())} robust, {features.shape[1]} varying features")
    print(f"majority baseline accuracy: {max(labels.mean(), 1 - labels.mean()):.3f}\n")
    for class_weight in (None, "balanced"):
        accuracy, spread, balanced, auc = nested_cv(features, labels, class_weight)
        print(
            f"class_weight={str(class_weight):9s} accuracy {accuracy:.3f} +/- {spread:.3f}"
            f" | balanced {balanced:.3f} | AUC {auc:.3f}"
        )

    observed, threshold = single_feature_scan(features, labels, columns)
    print(f"\nsingle features beating the shuffled-label 95th pct (|AUC-0.5| > {threshold:.3f}):")
    for distance, auc, name in observed:
        if distance <= threshold:
            break
        print(f"  AUC {auc:.3f}  {name}")


if __name__ == "__main__":
    main()
