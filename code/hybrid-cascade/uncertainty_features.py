"""Short early-trace uncertainty features (GPU path when models are available).

Design intent (method taste): prefer short generations with hard token caps
<< 4096, motivated by early-trace uncertainty signal. Features are a cheap
first-stage vote in the cascade — not a full reproduction of long-trace
uncertainty-profiling.

CPU / no-HF policy
------------------
This module never downloads models. Without a local cache + usable device it
returns ``None`` so the cascade can abort to the deterministic all-False
baseline without raising (coverage-safe).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Four Small Models Track competition IDs (verbatim HF ids at eval).
SMALL_TRACK_MODEL_IDS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3.5-4B",
        "Skywork/Skywork-OR1-Math-7B",
        "allenai/Olmo-3-7B-Think",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    }
)

# Hard caps for the *short* early-trace path (GPU next step).
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_MAX_PROMPT_TOKENS = 1024


@dataclass(frozen=True)
class UncertaintyArtifactRef:
    """Pointer to a bundled short-trace uncertainty regressor (optional)."""

    path: Path
    model_id: str


def _safe_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def artifact_candidates(model_id: str, root: Path | None = None) -> list[Path]:
    """Most-specific first: per-model joblib, then generic."""
    base = root or Path(__file__).resolve().parent
    artifacts = base / "artifacts"
    safe = _safe_model_id(model_id)
    names = []
    if safe:
        names.append(artifacts / f"uncertainty_{safe}.joblib")
    names.append(artifacts / "uncertainty_short_trace.joblib")
    return names


def find_uncertainty_artifact(model_id: str, root: Path | None = None) -> UncertaintyArtifactRef | None:
    for path in artifact_candidates(model_id, root=root):
        if path.is_file():
            return UncertaintyArtifactRef(path=path, model_id=model_id)
    return None


def gpu_ready() -> bool:
    """True only when CUDA is usable. Never triggers a download."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def short_trace_scores(
    model_id: str,
    problems: Sequence[str],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> list[float] | None:
    """Return soft scores in [0, 1] (higher ≈ more robust) or ``None`` to skip.

    Stub behaviour on CPU / missing artifact / missing HF cache: return ``None``
    immediately. GPU path (next step): load local checkpoint, greedy-generate
    ≤``max_new_tokens``, compute a compact early-trace feature vector, score
    with a bundled regressor keyed per Small-track model.
    """
    del max_new_tokens, max_prompt_tokens  # reserved for GPU implementation
    if not problems:
        return []
    if find_uncertainty_artifact(model_id) is None:
        return None
    if not gpu_ready():
        # Artifact alone is insufficient without a local model forward pass.
        return None
    # GPU next step: implement short greedy gen + 14-ish early-token aggregates
    # + joblib regressor. Intentionally not wired here (no HF download on CPU).
    return None


def scores_to_votes(scores: Sequence[float], threshold: float = 0.5) -> list[bool]:
    """Map soft scores to native bools (True = predicted robust)."""
    return [bool(float(s) >= threshold) for s in scores]


def feature_names() -> tuple[str, ...]:
    """Canonical short-trace feature names (GPU training target)."""
    return (
        "early_mean_logprob",
        "early_mean_entropy",
        "early_mean_top1_prob",
        "early_mean_top2_margin",
        "early_min_logprob",
        "early_frac_selected_is_top1",
        "early_frac_low_conf_tokens",
        "early_std_logprob",
    )
