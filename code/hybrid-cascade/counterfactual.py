"""Cheap K=1–2 behavioural counterfactuals (answer-stability checks).

Transforms are deliberately light (rename / light rephrase / typo-style) so a
few generations fit under the 3600s whole-run budget. Agreement of answers
across the original and K counterfactual prompts is a stability proxy — not
reconstruction of the unpublished private labelling rule H.

CPU / no-HF policy: return ``None`` unless GPU + local checkpoint are ready.
"""

from __future__ import annotations

import re
from typing import Sequence

# Default K kept tiny for cascade budget.
DEFAULT_K = 2


def light_transforms(problem: str, k: int = DEFAULT_K) -> list[str]:
    """Deterministic local transforms; no network paraphrase APIs.

    These are placeholders for GPU-time answer comparison. They must stay
    cheap and offline. Order is stable for reproducibility.
    """
    variants: list[str] = []
    text = problem
    # 1) whitespace / punctuation light touch
    if k >= 1:
        variants.append(re.sub(r"[ \t]+", " ", text).strip())
    # 2) mild rename-style: swap common variable letter if present, else append
    #    a no-op clarifying sentence that should not change the math.
    if k >= 2:
        swapped = re.sub(r"\b([nNmMxX])\b", lambda m: {"n": "m", "N": "M", "m": "n", "M": "N", "x": "y", "X": "Y"}.get(m.group(1), m.group(1)), text, count=3)
        if swapped == text:
            swapped = text + "\n\n(Answer with the same final integer as the original statement.)"
        variants.append(swapped)
    return variants[: max(0, k)]


def gpu_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def counterfactual_stability_votes(
    model_id: str,
    problems: Sequence[str],
    *,
    k: int = DEFAULT_K,
) -> list[bool] | None:
    """Predict robust when original and K transformed answers agree.

    Returns ``None`` on CPU / missing models so the cascade can abort cleanly.
    ``model_id`` is reserved for per-model decoding settings on the GPU path.
    """
    del model_id
    if not problems:
        return []
    if not gpu_ready():
        return None
    # GPU next step: for each problem, greedy-decode original + light_transforms
    # (K=1–2), extract final answers, mark True iff all agree. Cap gen length.
    _ = [light_transforms(p, k=k) for p in problems]
    return None


def agreement_to_bool(answers: Sequence[str | None]) -> bool:
    """True iff every non-None answer equals the first non-None answer."""
    known = [a for a in answers if a is not None]
    if not known:
        return False
    head = known[0]
    return all(a == head for a in known)
