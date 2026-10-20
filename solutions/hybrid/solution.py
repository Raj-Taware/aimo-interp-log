"""Codabench entry point for v5h: behavioural self-test, v2 probe when it is blind.

v2 (0.71 on the public board, bundle sha256 2b8ebf4c1583, vendored unchanged
as ``sp2``) runs first (~7 s). The behavioural test (selfperturb.behave) then
gets the rest of the per-case budget; when the original never yields an answer
in time it has no information, and the v2 prediction is used instead.
"""

import gc
import time
import traceback

import torch

from selfperturb.behave import CASE_SECONDS, log, run_case
from sp2.infer import predict_robustness as v2_predict


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    out = []
    for problem in problems:
        start = time.time()
        try:
            v2 = bool(v2_predict(model_id, [problem])[0])
        except Exception:
            log(traceback.format_exc())
            v2 = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            robust, info = run_case(model_id, problem, CASE_SECONDS - (time.time() - start))
            final = v2 if "reason" in info else robust
        except Exception:
            log(traceback.format_exc())
            final = v2
        log(f"v2={v2} final={final}")
        out.append(bool(final))
    return out
