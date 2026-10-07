"""T-010: score submission predictions on the 34 labelled train-main-v2 rows.

    python scripts/leaderboard_trainv2.py --rows results/grok/T-010/trainv2_small_rows.json \
        --preds-dir results/grok/T-010/preds --out results/grok/T-010/leaderboard

Expects ``<preds-dir>/<candidate>__<safe_model>.json`` from ``zip_predict.py``.
Adds an always-True baseline. Writes leaderboard.json and leaderboard.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MODELS = {"Qwen": "Qwen/Qwen3.5-4B", "Sky": "Skywork/Skywork-OR1-Math-7B",
          "Olmo": "allenai/Olmo-3-7B-Think", "DS": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}


def safe_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def score(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (label, prediction) over labelled rows."""
    n = len(pairs)
    correct = sum(l == p for l, p in pairs)
    pos = [p for l, p in pairs if l]
    neg = [p for l, p in pairs if not l]
    recalls = []
    if pos:
        recalls.append(sum(pos) / len(pos))
    if neg:
        recalls.append(sum(not p for p in neg) / len(neg))
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None,
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
            "robust_predicted": sum(p for _, p in pairs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--preds-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["v2", "v3b"])
    arguments = parser.parse_args()

    rows = json.loads(arguments.rows.read_text(encoding="utf-8"))
    by_idx = {r["row_idx"]: r for r in rows}
    predictions: dict[str, dict[int, bool | None]] = {"always-True": {r["row_idx"]: True for r in rows}}
    validity: dict[str, dict[str, bool]] = {"always-True": {k: True for k in MODELS}}
    for candidate in arguments.candidates:
        predictions[candidate], validity[candidate] = {}, {}
        for short, model in MODELS.items():
            record = json.loads((arguments.preds_dir / f"{candidate}__{safe_id(model)}.json").read_text())
            validity[candidate][short] = record["valid"]
            for idx, pred in zip(record["row_idx"], record["predictions"] or [None] * len(record["row_idx"])):
                predictions[candidate][idx] = pred

    table = []
    for candidate, preds in predictions.items():
        # An invalid batch scores as wrong on every labelled row (grader: invalid).
        def pairs(model=None):
            return [(bool(r["is_robust"]), bool(preds[r["row_idx"]]) if preds[r["row_idx"]] is not None
                     else (not bool(r["is_robust"])))
                    for r in rows if r["is_robust"] is not None and (model is None or r["model_id"] == model)]
        entry = {"candidate": candidate, **score(pairs()), "valid_batches": validity[candidate],
                 "per_model": {short: score(pairs(model)) for short, model in MODELS.items()},
                 "robust_predicted_all37": sum(bool(p) for p in preds.values())}
        table.append(entry)

    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "leaderboard.json").write_text(json.dumps(table, indent=2))
    with open(arguments.out / "leaderboard.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "correct", "n", "accuracy", "balanced_accuracy", "robust_predicted",
                         *[f"acc_{s}" for s in MODELS], *[f"n_{s}" for s in MODELS]])
        for e in table:
            writer.writerow([e["candidate"], e["correct"], e["n"], round(e["accuracy"], 4),
                             round(e["balanced_accuracy"], 4), e["robust_predicted"],
                             *[round(e["per_model"][s]["accuracy"], 4) for s in MODELS],
                             *[e["per_model"][s]["n"] for s in MODELS]])
    for e in table:
        per = "/".join(f"{e['per_model'][s]['correct']}/{e['per_model'][s]['n']}" for s in MODELS)
        print(f"{e['candidate']}: {e['correct']}/{e['n']} acc={e['accuracy']:.3f} "
              f"bal={e['balanced_accuracy']:.3f} robust_pred={e['robust_predicted']} per-model {per}")


if __name__ == "__main__":
    main()
