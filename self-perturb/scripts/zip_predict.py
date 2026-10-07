"""T-010: run a submission ZIP's ``solution.are_robust`` like the grader does.

The ZIP is unpacked to ``--zip-dir``; this process puts that directory first on
``sys.path`` (and nothing from this repo), imports ``solution``, and calls
``are_robust(model_id, problems)`` ONCE with all of that model's problems in
dataset order -- the grader's batching. Run one process per (ZIP, model) so
two ZIPs' ``selfperturb`` packages never share an import cache.

    python scripts/zip_predict.py --zip-dir /tmp/v3b --model Qwen/Qwen3.5-4B \
        --rows results/grok/T-010/trainv2_small_rows.json --out preds_v3b_qwen.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    rows = [r for r in json.loads(arguments.rows.read_text(encoding="utf-8"))
            if r["model_id"] == arguments.model]
    zip_dir = arguments.zip_dir.resolve()
    sys.path.insert(0, str(zip_dir))
    os.chdir(zip_dir)
    from solution import are_robust  # noqa: E402

    started = time.time()
    error = None
    try:
        predictions = are_robust(arguments.model, [r["problem"] for r in rows])
    except Exception as exc:  # grader would mark the batch invalid
        predictions, error = None, repr(exc)
    seconds = time.time() - started
    valid = (error is None and isinstance(predictions, list) and len(predictions) == len(rows)
             and all(type(p) is bool for p in predictions))
    record = {
        "zip_dir": str(zip_dir), "model": arguments.model, "seconds": round(seconds, 1),
        "valid": valid, "error": error,
        "row_idx": [r["row_idx"] for r in rows],
        "predictions": predictions if valid else None,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=2))
    print("PRED " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
