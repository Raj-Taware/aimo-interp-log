"""T-023 pre-registered gates (written before the Kaggle data exists).

Score = prose_enrichment from dump_saliency.py. Sign fixed now: higher = leans on wording =
predicted non-robust. Nothing is fitted, so there is no train/test leak and no threshold.

Gate 0 (measurement): mean zero-gradient share < 0.5 on every model (fp16 underflow check),
  >= 90% of rows scored.
Gate 1 (train-main-v2): within-model rank, mean over models, problem non-robust if any model
  is non-robust. Pass iff AUC >= 0.70, AUC > length AUC, label-permutation p < 0.05.
Gate 2 (transfer, a domain never used here): DeepSeek self-domain set (137 problems from
  T-021). Pass iff AUC >= 0.60, AUC > length AUC, label-permutation p < 0.05.
Positive signal = all three pass. Otherwise the method is dropped.

    python scripts/eval_saliency.py results/grok/T-023
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from difficulty_probe import STATES, key

ROOT = Path(__file__).resolve().parent.parent
SELFDOMAIN = ROOT / "results/grok/T-021/dump/selfdomain__deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz"
DEEPSEEK = "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"


def gate(y, score, length, floor, rng, draws=2000):
    auc, length_auc = roc_auc_score(y, score), roc_auc_score(y, length)
    null = np.array([roc_auc_score(rng.permutation(y), score) for _ in range(draws)])
    p = float((null >= auc).mean())
    return {"n": int(len(y)), "positives": int(y.sum()), "auc": round(float(auc), 3), "length_auc": round(float(length_auc), 3),
            "p": round(p, 4), "pass": bool(auc >= floor and auc > length_auc and p < 0.05)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    report, rows, dumps = {"gate0": {}}, [], {}
    for f in sorted(args.run.glob("saliency__*.npz")):
        r = np.load(f, allow_pickle=True)
        safe = f.stem.split("__", 1)[1]
        dumps[safe] = {key(p): s for p, s in zip(r["problems"], r["score"].astype(float))}
        scored = float(np.isfinite(r["score"].astype(float)).mean())
        zero = float(np.nanmean(r["zero_grad"].astype(float)))
        report["gate0"][safe] = {"scored": round(scored, 3), "zero_grad": round(zero, 4), "pass": scored >= 0.9 and zero < 0.5}
        tr = np.load(STATES / f"trainv2__{safe}__states.npz", allow_pickle=True)
        ok = ~np.isnan(tr["labels"])
        frame = pd.DataFrame({"problem": [key(p) for p in tr["problems"][ok]], "robust": tr["labels"][ok].astype(bool)})
        frame["score"] = frame["problem"].map(dumps[safe]).rank(pct=True)
        frame["length"] = frame["problem"].str.split().str.len().rank(pct=True)
        rows.append(frame)
    df = pd.concat(rows).dropna(subset=["score"])
    prob = df.groupby("problem").agg(nonrobust=("robust", lambda v: not v.all()), score=("score", "mean"), length=("length", "mean"))
    report["gate1"] = gate(prob["nonrobust"].values, prob["score"].values, prob["length"].values, 0.70, rng)
    sd = np.load(SELFDOMAIN, allow_pickle=True)
    probs = [key(p) for p in sd["problems"]]
    score = np.array([dumps[DEEPSEEK].get(p, np.nan) for p in probs])
    ok = np.isfinite(score)
    report["gate2"] = gate(~sd["labels"][ok].astype(bool), score[ok], np.array([len(p.split()) for p in probs], float)[ok], 0.60, rng)
    report["pass"] = all(g["pass"] for g in report["gate0"].values()) and report["gate1"]["pass"] and report["gate2"]["pass"]
    (args.run / "eval_saliency.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
