"""Codabench entry point for the budget-aware hybrid cascade (Small track).

Exposes ``are_robust(model_id, reasoning_effort, problems) -> list[bool]``.

On CPU / without local HF weights this solution returns a deterministic
all-False prediction of the correct length (valid coverage baseline). When
GPU + per-model artifacts are present, :mod:`cascade` runs last-token probes
(optional CF agreement veto) → uncertainty stub → all-False, aborting under
the 3600s whole-run budget.
"""

from __future__ import annotations

from cascade import run_cascade


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict robustness for a batch while preserving problem order.

    Returns a list of native Python ``bool`` values, one per input problem, in
    the same order as ``problems``. ``True`` means predicted robust.
    Never raises on missing GPU/artifacts: falls back to all-``False``.
    """
    return run_cascade(model_id, reasoning_effort, problems)
