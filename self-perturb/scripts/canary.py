"""Canary: recompute features for collab/canary.jsonl and compare to the v3 dump.

Run from the self-perturb/ directory (or anywhere; src/ is put on sys.path):

    python scripts/canary.py \\
        --canary ../collab/canary.jsonl \\
        --reference results/feature-dumps/2026-09-29_v3-features_tf5.13_unbatched-fp16/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz \\
        --out results/grok/T-001

Writes canary_features.npz (run1 + run2 vectors) and metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfperturb.features import case_features, feature_names, load_model  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
V3_SLICE = slice(87, 113)


def load_canary(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize(run1: np.ndarray, run2: np.ndarray, reference: np.ndarray, columns: list[str]) -> dict:
    """Compute the T-001 report numbers. Shapes: (n, 113)."""
    diff_ref = np.abs(run1 - reference)
    max_all = float(diff_ref.max())
    max_v3 = float(diff_ref[:, V3_SLICE].max())

    # Per-feature correlation across the 10 problems (run1 vs reference).
    corrs = []
    for col in range(run1.shape[1]):
        a, b = run1[:, col], reference[:, col]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            # Constant feature (e.g. applied flags that never fire) — treat as 1.0
            # when values match, else 0.0.
            corrs.append(1.0 if np.allclose(a, b) else 0.0)
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    median_corr = float(np.median(corrs))

    # Worst 3 features by max abs diff across problems (then by mean as tiebreak).
    per_feat_max = diff_ref.max(axis=0)
    worst_idx = np.argsort(-per_feat_max)[:3]
    worst = [(columns[i], float(per_feat_max[i])) for i in worst_idx]

    identical = bool(np.array_equal(run1, run2))
    return {
        "max_abs_diff_all": max_all,
        "max_abs_diff_v3": max_v3,
        "median_per_feature_corr": median_corr,
        "worst_3": worst,
        "second_run_identical": identical,
        "per_feature_corr": corrs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    canary = load_canary(arguments.canary)
    ref_data = np.load(arguments.reference, allow_pickle=True)
    ref_features = ref_data["features"]
    columns = list(ref_data["columns"]) if "columns" in ref_data.files else feature_names()
    if list(feature_names()) != columns:
        print("WARNING: live feature_names() differ from reference columns", flush=True)

    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    env = {
        "gpu": gpu_names,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": None,
    }
    try:
        import accelerate

        env["accelerate"] = accelerate.__version__
    except Exception:
        pass
    print(json.dumps(env), flush=True)

    started = time.time()
    tokenizer, model = load_model(arguments.model)
    print(f"model loaded in {time.time() - started:.1f}s", flush=True)

    run1, run2, dump_rows, seconds = [], [], [], []
    for entry in canary:
        problem = entry["problem"]
        dump_row = int(entry["dump_row"])
        t0 = time.time()
        v1 = case_features(tokenizer, model, problem)
        t1 = time.time()
        v2 = case_features(tokenizer, model, problem)
        t2 = time.time()
        run1.append(v1)
        run2.append(v2)
        dump_rows.append(dump_row)
        seconds.append(t1 - t0)
        print(
            f"dump_row={dump_row} dim={v1.shape[0]} "
            f"run1={t1 - t0:.2f}s run2={t2 - t1:.2f}s "
            f"max|r1-r2|={np.abs(v1 - v2).max():.6g}",
            flush=True,
        )

    run1_arr = np.asarray(run1, dtype=np.float32)
    run2_arr = np.asarray(run2, dtype=np.float32)
    dump_rows_arr = np.asarray(dump_rows, dtype=np.int32)
    reference_arr = np.asarray([ref_features[r] for r in dump_rows], dtype=np.float32)

    metrics = summarize(run1_arr, run2_arr, reference_arr, columns)
    metrics["env"] = env
    metrics["seconds_per_problem"] = [round(s, 3) for s in seconds]
    metrics["seconds_per_problem_mean"] = round(float(np.mean(seconds)), 3)
    metrics["dump_rows"] = dump_rows
    metrics["n_problems"] = len(canary)

    out_npz = arguments.out / "canary_features.npz"
    np.savez(
        out_npz,
        run1=run1_arr,
        run2=run2_arr,
        reference=reference_arr,
        dump_rows=dump_rows_arr,
        columns=np.asarray(columns, dtype=object),
        problems=np.asarray([e["problem"] for e in canary], dtype=object),
        model_id=arguments.model,
        seconds=np.asarray(seconds, dtype=np.float32),
    )
    out_json = arguments.out / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in metrics if k != "per_feature_corr"}, indent=2), flush=True)
    print(f"wrote {out_npz} and {out_json}", flush=True)


if __name__ == "__main__":
    main()
