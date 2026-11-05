"""Train the robustness classifier on dumped self-perturbation features.

Two evaluations run here, and the second is the one worth writing up:

* **Grouped CV** -- folds never split a problem across train and test. Every
  checkpoint sees the same problems, so a random split lets the classifier
  memorize "problem 7 is hard" from one checkpoint and read the answer off
  another. Grouping by problem removes that leak. Expect a lower, honest number.
* **Leave-one-model-out** -- train on three checkpoints, test on the fourth.
  This answers the question the write-up prize actually rewards: does the method
  find something about reasoning, or just something about these four models?

The exported artifact is plain numpy (scaler statistics + linear weights), not a
pickled sklearn estimator, so it cannot break on a version mismatch inside the
evaluation container.

    python scripts/train_probe.py --features data/features --out artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = (0, 1, 2, 3, 4)
N_SPLITS = 5


def load_features(directory: Path):
    """Stack every per-model dump into one dataset."""
    features, labels, groups, models, columns = [], [], [], [], None
    for path in sorted(directory.glob("*.npz")):
        dump = np.load(path, allow_pickle=True)
        rows = np.asarray(dump["features"], dtype=np.float32)
        features.append(rows)
        labels.append(np.asarray(dump["labels"], dtype=bool))
        # Group key is the problem text: identical problems must not straddle folds.
        groups.extend(str(problem) for problem in dump["problems"])
        models.extend([str(dump["model_id"])] * len(rows))
        columns = [str(name) for name in dump["columns"]]
    if not features:
        raise SystemExit(f"no .npz feature dumps found in {directory}")
    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.asarray(groups),
        np.asarray(models),
        columns,
    )


def build_classifier(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,  # small data, so regularize hard
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def grouped_cv(features, labels, groups) -> dict:
    """Repeated grouped CV. Reports spread, because one split proves nothing."""
    scores = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_index, test_index in splitter.split(features, labels, groups):
            model = build_classifier(seed).fit(features[train_index], labels[train_index])
            prediction = model.predict(features[test_index])
            scores.append(
                {
                    "accuracy": float((prediction == labels[test_index]).mean()),
                    "balanced": float(balanced_accuracy_score(labels[test_index], prediction)),
                }
            )
    accuracy = np.array([score["accuracy"] for score in scores])
    balanced = np.array([score["balanced"] for score in scores])
    return {
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std": float(accuracy.std()),
        "balanced_mean": float(balanced.mean()),
        "balanced_std": float(balanced.std()),
        "n_folds": len(scores),
    }


def leave_one_model_out(features, labels, models) -> dict:
    """Train on every checkpoint but one; test on the one held out."""
    results = {}
    for held_out in sorted(set(models)):
        test_mask = models == held_out
        if test_mask.sum() < 5 or len(set(labels[~test_mask])) < 2:
            continue
        model = build_classifier(0).fit(features[~test_mask], labels[~test_mask])
        prediction = model.predict(features[test_mask])
        results[held_out] = {
            "accuracy": float((prediction == labels[test_mask]).mean()),
            "balanced": float(balanced_accuracy_score(labels[test_mask], prediction)),
            "n_test": int(test_mask.sum()),
        }
    return results


def majority_baseline(labels) -> float:
    """What "always predict the common class" scores. Beat this or go home."""
    return float(max(labels.mean(), 1 - labels.mean()))


def export(features, labels, columns, destination: Path, report: dict) -> None:
    """Refit on everything and save as plain arrays."""
    pipeline = build_classifier(0).fit(features, labels)
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coef=classifier.coef_.astype(np.float32).reshape(-1),
        intercept=np.float32(classifier.intercept_[0]),
        columns=np.asarray(columns, dtype=object),
        report=json.dumps(report),
    )
    print(f"wrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/features"))
    # Inference looks for probe_artifacts/<safe_model_id>.npz, then generic.npz;
    # build_bundle.py ships everything in artifacts/probe_artifacts/.
    parser.add_argument("--out", type=Path, default=Path("artifacts/probe_artifacts/generic.npz"))
    arguments = parser.parse_args()

    features, labels, groups, models, columns = load_features(arguments.features)
    print(f"{len(features)} cases, {len(set(groups))} unique problems, {len(set(models))} models")

    report = {
        "majority_baseline": majority_baseline(labels),
        "grouped_cv": grouped_cv(features, labels, groups),
        "leave_one_model_out": leave_one_model_out(features, labels, models),
    }
    print(json.dumps(report, indent=2))

    cv = report["grouped_cv"]
    if cv["accuracy_mean"] - cv["accuracy_std"] <= report["majority_baseline"]:
        print("\nWARNING: within one std of the majority baseline. No signal yet.")

    export(features, labels, columns, arguments.out, report)


if __name__ == "__main__":
    main()
