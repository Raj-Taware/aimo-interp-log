"""T-024 pre-registered gates (written before the Kaggle data exists).

recovery[problem, perturbation] = 1 - kl_patch / kl_pert at depth 0.5. Score = -median recovery
over perturbations. Sign fixed now: a low recovery means the output change arrives through
context tokens rather than the answer-position state, read as the wording reaching the answer
directly, so higher score = predicted non-robust. Nothing is fitted.

Gate 0 (measurement): on every model median identity_kl < 1e-4 (the hook is a no-op when
  patching a prompt with itself), median kl_patch at depth 0.75 <= median kl_pert (patching
  removes change), >= 90% of problems scored.
Gate 1 (train-main-v2): within-model rank, mean over models, problem non-robust if any model is
  non-robust. Pass iff AUC >= 0.70, AUC > best length AUC, label-permutation p < 0.05.
Gate 2 (transfer): DeepSeek self-domain set from T-021. Pass iff AUC >= 0.60, AUC > best length
  AUC, label-permutation p < 0.05.
Best length AUC = max(auc, 1 - auc): T-023 showed a one-way comparison hides that short
problems are the non-robust ones on the self-domain set.
Positive signal = all three pass. Otherwise the method is dropped.

    python scripts/eval_patch.py results/grok/T-024
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
MID = 1  # index of depth 0.5


def gate(y, score, length, floor, rng, draws=2000):
    auc, length_auc = roc_auc_score(y, score), roc_auc_score(y, length)
    length_auc = max(length_auc, 1 - length_auc)
    null = np.array([roc_auc_score(rng.permutation(y), score) for _ in range(draws)])
    p = float((null >= auc).mean())
    return {"n": int(len(y)), "positives": int(y.sum()), "auc": round(float(auc), 3), "best_length_auc": round(float(length_auc), 3),
            "p": round(p, 4), "pass": bool(auc >= floor and auc > length_auc and p < 0.05)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    report, rows, dumps = {"gate0": {}}, [], {}
    for f in sorted(args.run.glob("patch__*.npz")):
        r = np.load(f, allow_pickle=True)
        safe = f.stem.split("__", 1)[1]
        pert, patch = r["kl_pert"].astype(float), r["kl_patch"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = -np.nanmedian(np.where(pert > 0, 1 - patch[:, :, MID] / pert, np.nan), 1)
        dumps[safe] = {key(p): s for p, s in zip(r["problems"], score)}
        identity, scored = float(np.nanmedian(r["identity_kl"])), float(np.isfinite(score).mean())
        report["gate0"][safe] = {"identity_kl": identity, "kl_pert": float(np.nanmedian(pert)), "kl_patch_075": float(np.nanmedian(patch[:, :, 2])),
                                 "scored": round(scored, 3)}
        report["gate0"][safe]["pass"] = identity < 1e-4 and np.nanmedian(patch[:, :, 2]) <= np.nanmedian(pert) and scored >= 0.9
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
    (args.run / "eval_patch.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
