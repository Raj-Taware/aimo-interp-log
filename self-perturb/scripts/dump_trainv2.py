"""T-010: self-perturbation features for aimo-interp/train-main-v2, one model.

``dump_features.py`` reads the older ``model_is_robust``/``original_problem``
schema. train-main-v2 has ``model_id, reasoning_effort, problem, is_robust
(bool or null), max_drop``. This script features every row of one small-track
model (null labels included) on that row's own model, with the repaired
tokenizer from ``selfperturb.features.load_model``.

    python scripts/dump_trainv2.py --model Qwen/Qwen3.5-4B \
        --rows results/grok/T-010/trainv2_small_rows.json --out results/grok/T-010/dumps

``--rows`` is a JSON list of {row_idx, model_id, problem, is_robust, max_drop};
without it the rows are fetched from the HF datasets-server API.
Writes ``<out>/<safe_model_id>.npz`` with features, labels (float: 1/0, NaN
for null), max_drop, problems, columns, row_idx, model_id, seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfperturb.features import case_features, feature_names, load_model  # noqa: E402

ROWS_API = ("https://datasets-server.huggingface.co/rows?dataset=aimo-interp/train-main-v2"
            "&config=default&split=train&offset={offset}&length=100")


def safe_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def fetch_rows() -> list[dict]:
    rows, offset = [], 0
    while True:
        with urllib.request.urlopen(ROWS_API.format(offset=offset)) as response:
            page = json.load(response)
        for item in page["rows"]:
            rows.append({"row_idx": item["row_idx"], **item["row"]})
        offset += len(page["rows"])
        if not page["rows"] or offset >= page["num_rows_total"]:
            return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    rows = json.loads(arguments.rows.read_text(encoding="utf-8")) if arguments.rows else fetch_rows()
    rows = [row for row in rows if row["model_id"] == arguments.model]
    if not rows:
        raise SystemExit(f"no rows for {arguments.model!r}")
    print(f"{len(rows)} rows for {arguments.model}", flush=True)

    started = time.time()
    tokenizer, model = load_model(arguments.model)
    load_seconds = time.time() - started
    print(f"model loaded in {load_seconds:.1f}s; tokenizer {type(tokenizer).__name__}", flush=True)

    features, per_case = [], []
    started = time.time()
    for index, row in enumerate(rows):
        t0 = time.time()
        features.append(case_features(tokenizer, model, row["problem"]))
        per_case.append(time.time() - t0)
        print(f"{index + 1}/{len(rows)}  {per_case[-1]:.2f}s", flush=True)
    feature_seconds = time.time() - started

    matrix = np.asarray(features, dtype=np.float32)
    labels = np.asarray([np.nan if row["is_robust"] is None else float(bool(row["is_robust"]))
                         for row in rows], dtype=np.float32)
    max_drop = np.asarray([np.nan if row.get("max_drop") is None else float(row["max_drop"])
                           for row in rows], dtype=np.float32)
    arguments.out.mkdir(parents=True, exist_ok=True)
    destination = arguments.out / f"{safe_id(arguments.model)}.npz"
    np.savez(
        destination,
        features=matrix,
        labels=labels,
        max_drop=max_drop,
        problems=np.asarray([row["problem"] for row in rows], dtype=object),
        columns=np.asarray(feature_names(), dtype=object),
        row_idx=np.asarray([row["row_idx"] for row in rows], dtype=np.int32),
        model_id=arguments.model,
        tokenizer_class=type(tokenizer).__name__,
        seconds_per_case=np.asarray(per_case, dtype=np.float32),
    )
    summary = {
        "model": arguments.model, "rows": len(rows), "n_features": int(matrix.shape[1]),
        "nan_or_inf": int((~np.isfinite(matrix)).sum()),
        "load_seconds": round(load_seconds, 1), "feature_seconds": round(feature_seconds, 1),
        "tokenizer_class": type(tokenizer).__name__, "path": destination.name,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    (arguments.out / f"{safe_id(arguments.model)}.summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
