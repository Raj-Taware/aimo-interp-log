"""T-011: compare v4 ZIP GPU predictions to Claude's expected per-row predictions.

    python scripts/compare_t011.py --preds-dir results/grok/T-011/preds \
        --expected results/expected/v4_trainv2_predictions.json --out results/grok/T-011/compare.json

Expects ``<preds-dir>/v4__<safe_model>.json`` from ``zip_predict.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    expected = json.loads(arguments.expected.read_text())
    table, mismatches = [], []
    for safe, exp in expected.items():
        rec = json.loads((arguments.preds_dir / f"v4__{safe}.json").read_text())
        actual = dict(zip(rec["row_idx"], rec["predictions"] or [None] * len(rec["row_idx"])))
        same_order = rec["row_idx"] == exp["row_idx"]
        match = 0
        for idx, e in zip(exp["row_idx"], exp["predictions"]):
            a = actual.get(idx)
            if a == e:
                match += 1
            else:
                mismatches.append({"model": safe, "row_idx": idx, "expected": e, "actual": a,
                                   "expected_margin": exp.get("margins", [None] * 99)[exp["row_idx"].index(idx)]})
        table.append({"model": safe, "rows": len(exp["row_idx"]), "matching": match,
                      "robust_predicted": sum(bool(p) for p in actual.values()),
                      "seconds": rec["seconds"], "valid": rec["valid"], "error": rec["error"],
                      "same_row_order": same_order})
    out = {"table": table, "mismatches": mismatches}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
