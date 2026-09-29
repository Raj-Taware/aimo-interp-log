"""Evaluation-time inference: perturb, encode, score, always return on time.

Three failure modes end a run at zero, and all three are handled here:

* **Exceptions** -- the harness marks the whole model batch invalid, not just
  one case. Every case is wrapped, and a failed case falls back to the prior.
* **Timeout** -- 3600 seconds covers every model and every problem together.
  A deadline is enforced per batch; once it passes, remaining cases take the
  prior instead of a forward pass, so we return a valid list rather than none.
* **Wrong types** -- the harness requires ``type(x) is bool``. numpy bools fail
  the check, so predictions are cast explicitly.

The ``selfperturb`` package is copied next to this file by
``scripts/build_bundle.py``; nothing outside the container runtime is imported.
"""

from __future__ import annotations

import gc
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

from selfperturb.features import case_features, load_model

ARTIFACT_DIR = "probe_artifacts"
# Leave room for model loading and the final scoring pass.
DEFAULT_BUDGET_SECONDS = float(os.environ.get("SELFPERTURB_BUDGET", "2700"))
# Returned when we cannot score a case. Every labelled public dump skews
# non-robust (74% in augmented-sample-math-agg, 68% in val-sample).
FALLBACK = False


def _safe_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def _load_artifact(model_id: str):
    """Per-model artifact first, then a generic one; ``None`` if neither exists."""
    directory = Path(__file__).resolve().parent / ARTIFACT_DIR
    candidates = [directory / f"{_safe_id(model_id)}.npz", directory / "generic.npz"]
    for path in candidates:
        if path.is_file():
            return np.load(path, allow_pickle=True)
    return None


def _score(artifact, features: np.ndarray) -> bool:
    """Standardize, apply the linear probe, threshold at zero."""
    standardized = (features - artifact["mean"]) / np.clip(artifact["scale"], 1e-6, None)
    margin = float(np.dot(standardized, artifact["coef"]) + float(artifact["intercept"]))
    return bool(margin >= 0.0)


def predict_robustness(model_id: str, problem_texts: list[str]) -> list[bool]:
    deadline = time.time() + DEFAULT_BUDGET_SECONDS
    predictions = [FALLBACK] * len(problem_texts)

    artifact = _load_artifact(model_id)
    if artifact is None:
        return predictions

    try:
        tokenizer, model = load_model(model_id)
    except Exception:
        return predictions

    system_prompt = str(artifact["system_prompt"]) if "system_prompt" in artifact else ""
    try:
        for index, problem in enumerate(problem_texts):
            if time.time() >= deadline:
                break  # remaining cases keep the fallback; a late run scores nothing
            try:
                features = case_features(tokenizer, model, problem, system_prompt)
                predictions[index] = _score(artifact, features)
            except Exception:
                predictions[index] = FALLBACK
    finally:
        # The harness calls us once per model in one process. HF models hold
        # reference cycles, so without an explicit collect each checkpoint
        # stays resident and the next one runs out of memory -- measured on
        # Kaggle, where the third and fourth models OOM'd.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [bool(prediction) for prediction in predictions]
