"""Train on the labelled model, then calibrate one artifact per checkpoint.

Only one small-track model (DeepSeek) has public labels. Applying its
classifier to the others naively fails in a measured way: features sit on
different scales per checkpoint, and the predicted robust rate swung from 14%
(Qwen3.5-4B, Olmo-3) to 60% (Skywork) on the same problems. The hidden
small-track labels are balanced (official all-False = always-True = 0.50), so
those skews cost accuracy directly.

Training: the same procedure ``scripts/evaluate.py`` scores with nested CV --
select k features and the regularization strength by inner CV, then fit. Run
once on all labelled cases, so the nested-CV figure is the honest estimate of
this exact artifact.

Label-free calibration, per checkpoint:

* **standardize** each selected feature with that model's own mean and spread
  over a shared problem set, so the DeepSeek-trained weights see every model
  on a common scale;
* **threshold** at that model's median margin, so it predicts ~50% robust,
  matching the known class balance.

Writes ``artifacts/probe_artifacts/<safe_model_id>.npz`` per model; inference
loads it in preference to ``generic.npz``.

    python scripts/calibrate_per_model.py --subset v3new \
        --labelled data/kaggle_out_v5/features/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz \
        --unlabelled data/kaggle_out_v5/features/{Qwen,Skywork,allenai}*.npz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import GRID, nested_cv  # noqa: E402  (same grid the estimate used)

OUT = Path("artifacts/probe_artifacts")
V2_WIDTH = 87


def safe_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def model_scale(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-model standardization statistics; constant columns get unit scale."""
    mean = features.mean(0)
    scale = features.std(0)
    return mean, np.where(scale > 1e-9, scale, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled", type=Path, required=True)
    parser.add_argument("--unlabelled", type=Path, nargs="+", default=[])
    parser.add_argument(
        "--subset",
        choices=("all", "v2", "v3new"),
        default="all",
        help="v2 = the original 87 features; v3new = only those added after them",
    )
    arguments = parser.parse_args()

    labelled = np.load(arguments.labelled, allow_pickle=True)
    columns = [str(c) for c in labelled["columns"]]
    candidates = {
        "all": np.arange(len(columns)),
        "v2": np.arange(V2_WIDTH),
        "v3new": np.arange(V2_WIDTH, len(columns)),
    }[arguments.subset]
    labels = labelled["labels"].astype(bool)
    mean, scale = model_scale(labelled["features"][:, candidates])
    standardized = (labelled["features"][:, candidates] - mean) / scale

    accuracy, spread, balanced, auc = nested_cv(standardized, labels, "balanced")
    pipeline = make_pipeline(
        SelectKBest(f_classif), LogisticRegression(max_iter=3000, class_weight="balanced")
    )
    search = GridSearchCV(
        pipeline,
        GRID,
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        scoring="balanced_accuracy",
    ).fit(standardized, labels)
    best = search.best_estimator_
    keep = best.named_steps["selectkbest"].get_support()
    feature_index = candidates[keep]
    coef = best.named_steps["logisticregression"].coef_.reshape(-1)
    base_intercept = float(best.named_steps["logisticregression"].intercept_[0])

    report = {
        "trained_on": str(labelled["model_id"]),
        "subset": arguments.subset,
        "chosen": {key: str(value) for key, value in search.best_params_.items()},
        "selected_features": [columns[i] for i in feature_index],
        "nested_cv_estimate": {
            "balanced": round(float(balanced), 4),
            "auc": round(float(auc), 4),
            "accuracy": round(float(accuracy), 4),
            "accuracy_std": round(float(spread), 4),
        },
        "models": {},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    for path in [arguments.labelled, *arguments.unlabelled]:
        dump = np.load(path, allow_pickle=True)
        assert [str(c) for c in dump["columns"]] == columns, f"{path}: feature columns differ"
        selected = dump["features"][:, feature_index]
        model_mean, model_scale_ = model_scale(selected)
        margins = ((selected - model_mean) / model_scale_) @ coef + base_intercept
        # Shift so this model's median margin sits exactly at the decision boundary.
        intercept = base_intercept - float(np.median(margins))
        model_id = str(dump["model_id"])
        destination = OUT / f"{safe_id(model_id)}.npz"
        np.savez(
            destination,
            mean=model_mean.astype(np.float32),
            scale=model_scale_.astype(np.float32),
            coef=coef.astype(np.float32),
            intercept=np.float32(intercept),
            feature_index=feature_index.astype(np.int64),
            columns=np.asarray([columns[i] for i in feature_index], dtype=object),
            report=json.dumps({"calibrated_on": path.name, "trained_on": report["trained_on"]}),
        )
        calibrated = margins - np.median(margins)
        key = "auc_in_sample" if path == arguments.labelled else "auc_vs_deepseek_labels"
        report["models"][model_id] = {
            "artifact": destination.name,
            "pred_robust_rate": round(float((calibrated >= 0).mean()), 3),
            # For other models these are DeepSeek's labels: shared-fragility proxy only.
            key: round(float(roc_auc_score(labels, calibrated)), 4),
        }

    print(json.dumps(report, indent=2))
    (OUT.parent / "calibration_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
