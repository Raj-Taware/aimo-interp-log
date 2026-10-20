"""Codabench entry point for v5 (behavioural self-test, selfperturb.behave)."""

import traceback

from selfperturb.behave import FALLBACK, log, run_case


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    """One native bool per problem; any failure falls back instead of invalidating the batch."""
    out = []
    for problem in problems:
        try:
            out.append(bool(run_case(model_id, problem)[0]))
        except Exception:
            log(traceback.format_exc())
            out.append(FALLBACK)
    return out
