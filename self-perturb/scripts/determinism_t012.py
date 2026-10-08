"""T-012 determinism check: first N MATH cases of one model, case_geometry twice in one process.

    python scripts/determinism_t012.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --out results/grok/T-012/determinism

Uses the same rows and order as ``dump_geometry.py --kind math`` (load_rows, then
``random.Random(0).shuffle``). Writes determinism.json (max |diff| over the
feature vectors and over all per-layer profiles, NaN-pattern agreement) and
determinism.npz (run1, run2).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_features import DEFAULT_SOURCES, load_rows  # noqa: E402
from selfperturb.features import load_model  # noqa: E402
from selfperturb.geometry import case_geometry, geometry_names  # noqa: E402


def _maxdiff(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    finite = np.isfinite(a) & np.isfinite(b)
    return (float(np.abs(a[finite] - b[finite]).max()) if finite.any() else 0.0), same_nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    parser.add_argument("--source", action="append")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    rows = [(p, float(y)) for p, y in load_rows(arguments.source or DEFAULT_SOURCES, arguments.model)]
    random.Random(0).shuffle(rows)
    rows = rows[: arguments.n]
    tokenizer, model = load_model(arguments.model, attn_implementation="eager")

    run1, run2, per_case = [], [], []
    for index, (problem, _) in enumerate(rows):
        t0 = time.time()
        v1, p1 = case_geometry(tokenizer, model, problem)
        t1 = time.time()
        v2, p2 = case_geometry(tokenizer, model, problem)
        t2 = time.time()
        vec_diff, vec_nan = _maxdiff(v1, v2)
        prof_diff, prof_nan = 0.0, True
        for family in p1:
            for metric in p1[family]:
                d, s = _maxdiff(p1[family][metric], p2[family][metric])
                prof_diff, prof_nan = max(prof_diff, d), prof_nan and s
        per_case.append({"case": index, "features_max_abs_diff": vec_diff, "features_nan_pattern_equal": vec_nan,
                         "profiles_max_abs_diff": prof_diff, "profiles_nan_pattern_equal": prof_nan,
                         "seconds_run1": round(t1 - t0, 2), "seconds_run2": round(t2 - t1, 2)})
        print(json.dumps(per_case[-1]), flush=True)
        run1.append(v1)
        run2.append(v2)

    arguments.out.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.out / "determinism.npz", run1=np.asarray(run1), run2=np.asarray(run2),
             names=np.asarray(geometry_names(), dtype=object), problems=np.asarray([p for p, _ in rows], dtype=object))
    summary = {
        "model": arguments.model, "cases": len(rows),
        "features_max_abs_diff": max(c["features_max_abs_diff"] for c in per_case),
        "profiles_max_abs_diff": max(c["profiles_max_abs_diff"] for c in per_case),
        "nan_pattern_equal": all(c["features_nan_pattern_equal"] and c["profiles_nan_pattern_equal"] for c in per_case),
        "bitwise_identical": bool(np.array_equal(np.asarray(run1), np.asarray(run2), equal_nan=True)),
        "per_case": per_case,
    }
    (arguments.out / "determinism.json").write_text(json.dumps(summary, indent=2))
    print("DETERMINISM", json.dumps({k: v for k, v in summary.items() if k != "per_case"}), flush=True)


if __name__ == "__main__":
    main()
