"""T-015 gate: our labels on the train-main-v2 small-track problems vs the official ones.

    python scripts/label_gen/gate_check.py --trainv2 <train-main-v2.parquet> \
        --labels <run_dir>/gate/labels.jsonl --out <run_dir>/gate/gate_report.json

Pass iff agreement >= 0.70 and Cohen's kappa >= 0.30 on pairs labelled by both
(official is_robust not null, ours not null). Reported for all such pairs and
for complete=True only. max_drop Pearson r over pairs where both have max_drop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl

WRITER = "openai/gpt-oss-120b"


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return None if not sx or not sy else round(sxy / (sx * sy), 4)


def compare(official: dict, ours: dict, complete_only: bool) -> dict:
    both, drops, uncovered = [], [], []
    for key, off in official.items():
        mine = ours.get(key)
        if mine is None or (complete_only and not mine["complete"]):
            if off["is_robust"] is not None:
                uncovered.append(key)
            continue
        if off["max_drop"] is not None and mine["max_drop"] is not None:
            drops.append((off["max_drop"], mine["max_drop"]))
        if off["is_robust"] is not None and mine["is_robust"] is not None:
            both.append((off["is_robust"], mine["is_robust"]))
    n = len(both)
    agree = round(sum(a == b for a, b in both) / n, 4) if n else None
    k = kappa(both)
    return {
        "n_pairs_both_labelled": n, "agreement": agree, "kappa": k,
        "confusion_official_x_ours": {f"{a}->{b}": sum(p == (a, b) for p in both)
                                      for a in (True, False) for b in (True, False)},
        "max_drop_r": pearson([a for a, _ in drops], [b for _, b in drops]), "n_max_drop": len(drops),
        "official_labelled_but_not_by_us": len(uncovered),
        "passes": bool(n and agree >= 0.70 and k is not None and k >= 0.30),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainv2", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.trainv2)
    df = df[df.model_id != WRITER]
    official = {(r.model_id, r.problem): {"is_robust": None if pd.isna(r.is_robust) else bool(r.is_robust),
                                          "max_drop": None if pd.isna(r.max_drop) else float(r.max_drop)}
                for r in df.itertuples()}
    ours = {(r["model_id"], r["problem"]): r for r in read_jsonl(args.labels)}
    report = {"all": compare(official, ours, False), "complete_only": compare(official, ours, True),
              "n_official_rows": len(official), "n_official_labelled": sum(v["is_robust"] is not None
                                                                           for v in official.values())}
    report["gate_passes"] = report["all"]["passes"]
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
