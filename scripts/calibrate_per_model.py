"""Train on the labelled model, then calibrate one artifact per checkpoint.

Only one small-track model (DeepSeek) has public labels. Applying its
classifier to the others naively fails in a measured way: features sit on
different scales per checkpoint, and the predicted robust rate swings from
14% (Qwen3.5-4B, Olmo-3) to 60% (Skywork) on the same problems. The hidden
small-track labels are balanced (official all-False = always-True = 0.50), so
those skews cost accuracy directly.

Label-free fix, per checkpoint:

* **standardize** each feature with that model's own mean and spread over a
  shared problem set (no labels needed), so the DeepSeek-trained weights see
  every model on a common scale;
* **threshold** at that model's median margin, so it predicts ~50% robust,
  matching the known class balance.

Writes ``artifacts/probe_artifacts/<safe_model_id>.npz`` per model, which the
inference code loads in preference to ``generic.npz``.

    python scripts/calibrate_per_model.py \
        --labelled data/features/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz \
        --unlabelled data/kaggle_out_v3/features/*.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

OUT = Path("artifacts/probe_artifacts")
C = 0.1


def safe_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def model_scale(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-model standardization statistics; constant columns get unit scale."""
    mean = features.mean(0)
    scale = features.std(0)
    return mean, np.where(scale > 1e-9, scale, 1.0)


def cross_validated(features, labels) -> dict:
    """Out-of-fold AUC and balanced accuracy at the median-margin threshold."""
    margins = np.zeros(len(labels))
    for seed in range(5):
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(features, labels):
            model = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            model.fit(features[train], labels[train])
            margins[test] += model.decision_function(features[test]) / 5
    prediction = margins >= np.median(margins)
    return {
        "oof_auc": round(float(roc_auc_score(labels, margins)), 4),
        "oof_balanced_at_median": round(float(balanced_accuracy_score(labels, prediction)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled", type=Path, required=True)
    parser.add_argument("--unlabelled", type=Path, nargs="+", default=[])
    arguments = parser.parse_args()

    labelled = np.load(arguments.labelled, allow_pickle=True)
    columns = [str(c) for c in labelled["columns"]]
    mean, scale = model_scale(labelled["features"])
    standardized = (labelled["features"] - mean) / scale
    labels = labelled["labels"].astype(bool)

    report = {"trained_on": str(labelled["model_id"]), "cv": cross_validated(standardized, labels)}
    classifier = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(standardized, labels)
    coef = classifier.coef_.reshape(-1)

    OUT.mkdir(parents=True, exist_ok=True)
    report["models"] = {}
    for path in [arguments.labelled, *arguments.unlabelled]:
        dump = np.load(path, allow_pickle=True)
        assert [str(c) for c in dump["columns"]] == columns, f"{path}: feature columns differ"
        model_mean, model_scale_ = model_scale(dump["features"])
        margins = ((dump["features"] - model_mean) / model_scale_) @ coef + classifier.intercept_[0]
        # Shift so this model's median margin sits exactly at the decision boundary.
        intercept = float(classifier.intercept_[0] - np.median(margins))
        model_id = str(dump["model_id"])
        destination = OUT / f"{safe_id(model_id)}.npz"
        np.savez(
            destination,
            mean=model_mean.astype(np.float32),
            scale=model_scale_.astype(np.float32),
            coef=coef.astype(np.float32),
            intercept=np.float32(intercept),
            columns=np.asarray(columns, dtype=object),
            report=json.dumps({"calibrated_on": str(path.name), "trained_on": report["trained_on"]}),
        )
        calibrated = margins - np.median(margins)
        entry = {"artifact": destination.name, "pred_robust_rate": round(float((calibrated >= 0).mean()), 3)}
        # For other models these are DeepSeek's labels: shared-fragility proxy only.
        entry["auc_vs_" + ("own_labels_in_sample" if path == arguments.labelled else "deepseek_labels")] = round(
            float(roc_auc_score(labels, calibrated)), 4
        )
        report["models"][model_id] = entry

    print(json.dumps(report, indent=2))
    (OUT.parent / "calibration_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
