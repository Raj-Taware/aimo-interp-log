"""T-022 pre-registered gates (written before the Kaggle data exists).

Gate 1 (per model, on dump_recall.py output): enough contrast to build a direction.
  knows = immediate answer to the original is the gold answer (no reasoning, so mostly recall).
  Pass iff >= 12 problems with knows & pm (still gives the old answer after a number changes)
  AND >= 12 with knows & ~pm. Fallback label if a model fails: pull = lp_alt - lp_orig
  (log-prob of the old answer, altered minus original), continuous, all altered problems.
Gate 2 (offline, models that pass Gate 1; trainv2 problems excluded from the fit):
  direction = mean(T-019 state | pm) - mean(state | ~pm) among known problems, one
  (pooling, layer) picked by leave-one-out AUC inside the pool only. Score = projection.
  Higher = more template-like -> predicted non-robust (sign fixed now).
  Pass iff problem-level AUC on train-main-v2 >= 0.70, above length AUC, and
  shuffled-pm null (200 draws) p < 0.05.

    python scripts/eval_recall.py results/grok/T-022
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from difficulty_probe import STATES, TRAINV2, key

MIN_EACH = 12


def direction(X, y):
    return X[y].mean(0) - X[~y].mean(0)


def loo_auc(X, y):
    s = np.empty(len(y))
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        s[i] = X[i] @ direction(X[m], y[m])
    return roc_auc_score(y, s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--null-draws", type=int, default=200)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    tv = pd.read_parquet(TRAINV2)
    exclude = {key(p) for p in tv["problem"]}
    report, rows = {"gate1": {}, "gate2": {}}, []
    for f in sorted(args.run.glob("recall__*.npz")):
        r = np.load(f, allow_pickle=True)
        safe = f.stem.split("__", 1)[1]
        knows, pm = r["knows"].astype(bool), r["pm"].astype(bool)
        g1 = {"knows": int(knows.sum()), "knows_pm": int((knows & pm).sum()), "knows_not_pm": int((knows & ~pm).sum())}
        g1["pass"] = g1["knows_pm"] >= MIN_EACH and g1["knows_not_pm"] >= MIN_EACH
        report["gate1"][safe] = g1
        if not g1["pass"]:
            continue
        pool = np.load(STATES / f"pool__{safe}__states.npz", allow_pickle=True)
        idx = {key(p): i for i, p in enumerate(pool["problems"])}
        sel = [idx[key(p)] for p, k in zip(r["problems"], knows) if k and key(p) in idx and key(p) not in exclude]
        y = np.array([pm[i] for i, (p, k) in enumerate(zip(r["problems"], knows)) if k and key(p) in idx and key(p) not in exclude])
        best = max(((loo_auc(pool[pl][sel][:, li].astype(np.float64), y), pl, li)
                    for pl in ("mean", "last") for li in range(1, len(pool["layers"]))), key=lambda t: t[0])
        auc, pl, li = best
        X = pool[pl][sel][:, li].astype(np.float64)
        w = direction(X, y)
        tr = np.load(STATES / f"trainv2__{safe}__states.npz", allow_pickle=True)
        ok = ~np.isnan(tr["labels"])
        H = tr[pl][ok][:, li].astype(np.float64)
        s = H @ w
        null = [H @ direction(X, rng.permutation(y)) for _ in range(args.null_draws)]
        report["gate2"][safe] = {"pooling": pl, "layer": int(pool["layers"][li]), "pool_loo_auc": round(float(auc), 3), "n_fit": len(y)}
        for j, p in enumerate(tr["problems"][ok]):
            rows.append({"model": safe, "problem": key(p), "robust": bool(tr["labels"][ok][j]),
                         "score": float(pd.Series(s).rank(pct=True)[j]),
                         "length": float(pd.Series([len(q.split()) for q in tr["problems"][ok]]).rank(pct=True)[j]),
                         "null": [float(pd.Series(n).rank(pct=True)[j]) for n in null]})
    if rows:
        df = pd.DataFrame(rows)
        prob = df.groupby("problem").agg(nonrobust=("robust", lambda v: not v.all()), score=("score", "mean"),
                                         length=("length", "mean"))
        nulls = df.groupby("problem")["null"].apply(lambda c: np.mean(np.stack(c.values), 0))
        auc = roc_auc_score(prob["nonrobust"], prob["score"])
        null_auc = np.array([roc_auc_score(prob["nonrobust"], nulls.loc[prob.index].map(lambda v: v[k])) for k in range(args.null_draws)])
        length_auc = roc_auc_score(prob["nonrobust"], prob["length"])
        report["gate2"]["problem_auc"] = round(float(auc), 3)
        report["gate2"]["length_auc"] = round(float(length_auc), 3)
        report["gate2"]["null_p"] = round(float((null_auc >= auc).mean()), 4)
        report["gate2"]["n_problems"] = int(len(prob))
        report["gate2"]["pass"] = bool(auc >= 0.70 and auc > length_auc and report["gate2"]["null_p"] < 0.05)
    (args.run / "eval_recall.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
