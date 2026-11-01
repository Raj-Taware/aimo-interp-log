"""T-026 pre-registered gate (written before this comparison is computed): does a signal add
anything beyond length? Offline on saved scores, DeepSeek self-domain set (137 problems).

The T-024/T-025 gates compared a signal with length and failed even v2's own drift, so they
could not detect the signal behind v2b. Here length is held fixed instead of competed against:

  Stratified AUC = pairs of (non-robust, robust) problems inside the same length quartile only.
  Null = labels permuted inside each quartile, 5000 draws.

Control: v2 hidden block, out-of-fold logistic score (20 seeds, 5 folds). The gate is usable only
if the control reaches stratified AUC >= 0.60 with p < 0.05. If it does not, this gate is no
better than the old one and no verdict is drawn for any signal.
Signals, sign fixed by each task's own registration: T-023 enrichment, T-024 -recovery, T-025
drift. Each passes iff stratified AUC >= 0.60 and p < 0.0167 (Bonferroni over three).

    python scripts/eval_length_matched.py results/grok/positive-control/<v2 dump>.npz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from difficulty_probe import key
from eval_patch import DEEPSEEK, MID, SELFDOMAIN

ROOT = Path(__file__).resolve().parent.parent
GROK = ROOT / "results/grok"


def stratified_auc(y, score, bins):
    pairs, wins = 0.0, 0.0
    for b in np.unique(bins):
        m = bins == b
        pos, neg = int(y[m].sum()), int((~y[m]).sum())
        if pos and neg:
            pairs += pos * neg
            wins += pos * neg * roc_auc_score(y[m], score[m])
    return wins / pairs


def test(y, score, bins, rng, draws=5000):
    ok = np.isfinite(score)
    y, score, bins = y[ok], score[ok], bins[ok]
    auc = stratified_auc(y, score, bins)
    null = np.empty(draws)
    for i in range(draws):
        shuffled = y.copy()
        for b in np.unique(bins):
            m = np.flatnonzero(bins == b)
            shuffled[m] = rng.permutation(y[m])
        null[i] = stratified_auc(shuffled, score, bins)
    return {"n": int(len(y)), "stratified_auc": round(float(auc), 3), "p": round(float((null >= auc).mean()), 4)}


def scores_from(path, field, problems):
    r = np.load(path, allow_pickle=True)
    if field == "enrichment":
        s = r["score"].astype(float)
    elif field == "recovery":
        pert, patch = r["kl_pert"].astype(float), r["kl_patch"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -np.nanmedian(np.where(pert > 0, 1 - patch[:, :, MID] / pert, np.nan), 1)
    else:
        s = np.nanmean(r["drift"].astype(float), 1)
    table = {key(p): v for p, v in zip(r["problems"], s)}
    return np.array([table.get(p, np.nan) for p in problems])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("v2_dump", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    sd = np.load(SELFDOMAIN, allow_pickle=True)
    problems = [key(p) for p in sd["problems"]]
    y = ~sd["labels"].astype(bool)
    length = np.array([len(p.split()) for p in problems], float)
    bins = np.searchsorted(np.quantile(length, [0.25, 0.5, 0.75]), length, side="left")
    v2 = np.load(args.v2_dump, allow_pickle=True)
    assert [key(p) for p in v2["problems"]] == problems
    assert (~v2["labels"].astype(bool) == y).all()
    hidden = [i for i, c in enumerate(v2["columns"]) if re.search(r"_(cos|norm)_d[0-9.]+$", c)]
    X = v2["features"].astype(float)[:, hidden]
    oof = np.zeros(len(y))
    for seed in range(20):
        fold = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof += cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000)),
                                 X, y, cv=fold, method="predict_proba")[:, 1]
    report = {"bins": np.bincount(bins).tolist(), "nonrobust_by_bin": [int(y[bins == b].sum()) for b in range(4)],
              "control_v2_hidden": test(y, oof, bins, rng)}
    c = report["control_v2_hidden"]
    report["gate_usable"] = bool(c["stratified_auc"] >= 0.60 and c["p"] < 0.05)
    for name, file, field in (("T-023", f"T-023/saliency__{DEEPSEEK}.npz", "enrichment"),
                              ("T-024", f"T-024/patch__{DEEPSEEK}.npz", "recovery"),
                              ("T-025", f"T-025/trace__{DEEPSEEK}.npz", "drift")):
        report[name] = test(y, scores_from(GROK / file, field, problems), bins, rng)
        report[name]["pass"] = report["gate_usable"] and report[name]["stratified_auc"] >= 0.60 and report[name]["p"] < 0.0167
    (GROK / "positive-control/eval_length_matched.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
