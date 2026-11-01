"""T-025 pre-registered gates (written before the Kaggle data exists).

Score = mean drift over typos and distract from dump_trace.py. Sign fixed now: higher = the
reasoning state depends more on surface text = predicted non-robust. Nothing is fitted.

Gate 0 (measurement): on every model median identity_drift < 1e-3, median drift above it,
  median trace >= 32 tokens, >= 90% of problems scored.
Gate 1 (train-main-v2): within-model rank, mean over models, problem non-robust if any model is
  non-robust. Pass iff AUC >= 0.70, AUC > best length AUC, label-permutation p < 0.05.
Gate 2 (transfer): DeepSeek self-domain set from T-021. Pass iff AUC >= 0.60, AUC > best length
  AUC (max(auc, 1 - auc)), label-permutation p < 0.05.
Positive signal = all three pass. Otherwise the method is dropped.

    python scripts/eval_trace.py results/grok/T-025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from difficulty_probe import STATES, key
from eval_patch import DEEPSEEK, SELFDOMAIN, gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    report, rows, dumps = {"gate0": {}}, [], {}
    for f in sorted(args.run.glob("trace__*.npz")):
        r = np.load(f, allow_pickle=True)
        safe = f.stem.split("__", 1)[1]
        drift = r["drift"].astype(float)
        score = np.nanmean(drift, 1)
        dumps[safe] = {key(p): s for p, s in zip(r["problems"], score)}
        identity, median, trace = float(np.nanmedian(r["identity_drift"])), float(np.nanmedian(drift)), float(np.median(r["trace_tokens"]))
        scored = float(np.isfinite(score).mean())
        report["gate0"][safe] = {"identity": identity, "drift": median, "trace": trace, "scored": round(scored, 3),
                                 "pass": identity < 1e-3 and median > identity and trace >= 32 and scored >= 0.9}
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
    (args.run / "eval_trace.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
