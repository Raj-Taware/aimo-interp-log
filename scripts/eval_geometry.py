"""Pre-registered evaluation of phase 1 geometry (collab/tasks/T-012.md).

Written before any T-012 data existed. Inputs: the MATH DeepSeek dump and the
train-main-v2 (TV2) dumps from ``dump_geometry.py``.

    python scripts/eval_geometry.py --math <math__deepseek.npz> --tv2 <trainv2__*.npz ...>

Fixed choices (stated before data):
* TV2 rows are pooled across models after ranking each feature within its
  model (label-free percentile ranks), since raw scales differ by model.
* TV2 significance = problem-level label permutation: labels are shuffled
  across the 20 problems and every row of a problem takes its problem's label
  (labels agree across models in TV2).
* MATH "beyond length" = the T-004 rule: L + X beats L by >= 0.03 mean
  balanced accuracy and wins >= 60% of the 25 paired outer folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solutions" / "self-perturb"))
from evaluate import GRID, SEEDS  # noqa: E402

H2_COLUMNS = ("distract_cka_d0.4", "distract_cka_d0.55", "distract_cka_d0.7")
PERMUTATIONS = 5000


def load(path):
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["features"].astype(float),
        "names": [str(n) for n in d["names"]],
        "y": d["labels"].astype(float),
        "problems": [str(p) for p in d["problems"]],
        "model": str(d["model_id"]),
    }


def column(dump, name):
    return dump["X"][:, dump["names"].index(name)]


def h2_score(dump):
    return np.nanmean(np.stack([column(dump, c) for c in H2_COLUMNS], axis=1), axis=1)


# ---------------------------------------------------------------- TV2 pooling


def pooled(tv2, score_fn):
    """Within-model percentile ranks of a score, pooled over labelled rows."""
    scores, labels, problems = [], [], []
    for dump in tv2:
        s = score_fn(dump)
        ranks = np.full(len(s), np.nan)
        ok = np.isfinite(s)
        if ok.sum():
            ranks[ok] = rankdata(s[ok]) / ok.sum()
        keep = ~np.isnan(dump["y"])
        scores += list(ranks[keep]); labels += list(dump["y"][keep].astype(bool)); problems += list(np.array(dump["problems"])[keep])
    return np.array(scores), np.array(labels), np.array(problems)


def problem_permutation_p(scores, labels, problems, auc_fn):
    ok = np.isfinite(scores)
    scores, labels, problems = scores[ok], labels[ok], problems[ok]
    observed = auc_fn(labels, scores)
    unique = sorted(set(problems))
    problem_label = {p: labels[problems == p].mean() >= 0.5 for p in unique}
    rng = np.random.default_rng(0)
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = dict(zip(unique, rng.permutation([problem_label[p] for p in unique])))
        perm = np.array([shuffled[p] for p in problems])
        if perm.all() or not perm.any():
            continue
        null.append(auc_fn(perm, scores))
    return float(observed), float(np.mean(np.array(null) >= observed)), int(ok.sum())


def auc_spurious(labels_robust, score):
    """AUC for 'higher score = spurious'."""
    return roc_auc_score(~labels_robust, score)


# ------------------------------------------------------------- MATH vs length


def nested_folds(X, y):
    out = []
    for seed in SEEDS:
        for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced"),
            )
            search = GridSearchCV(model, {"logisticregression__C": [0.01, 0.1, 1.0]},
                                  cv=StratifiedKFold(4, shuffle=True, random_state=seed),
                                  scoring="balanced_accuracy").fit(X[train], y[train])
            out.append(balanced_accuracy_score(y[test], search.predict(X[test])))
    return np.array(out)


def beyond_length(math, x):
    y = math["y"].astype(bool)
    length = np.log([len(p.split()) for p in math["problems"]])[:, None]
    base = nested_folds(length, y)
    plus = nested_folds(np.c_[length, x], y)
    diff = plus - base
    wins, ties = int((diff > 1e-12).sum()), int((np.abs(diff) <= 1e-12).sum())
    return {
        "L": round(float(base.mean()), 4), "L+X": round(float(plus.mean()), 4),
        "mean_diff": round(float(diff.mean()), 4), "wins": wins, "ties": ties, "losses": len(diff) - wins - ties,
        "pass": bool(diff.mean() >= 0.03 and wins >= 0.6 * len(diff)),
    }


# ------------------------------------------------------------------ H3 model


def h3(math, tv2):
    names = math["names"]
    y = math["y"].astype(bool)
    imputer = SimpleImputer(strategy="median").fit(math["X"])
    Xm = imputer.transform(math["X"])
    mu, sd = Xm.mean(0), np.where(Xm.std(0) > 1e-9, Xm.std(0), 1.0)
    pipe = make_pipeline(SelectKBest(f_classif), LogisticRegression(max_iter=3000, class_weight="balanced"))
    search = GridSearchCV(pipe, GRID, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          scoring="balanced_accuracy").fit((Xm - mu) / sd, y)
    margins, labels, problems, preds = [], [], [], []
    for dump in tv2:
        assert dump["names"] == names
        Xt = imputer.transform(dump["X"])
        m_mu, m_sd = Xt.mean(0), np.where(Xt.std(0) > 1e-9, Xt.std(0), 1.0)   # this model's batch
        m = search.decision_function((Xt - m_mu) / m_sd)
        decided = m > np.median(m)
        keep = ~np.isnan(dump["y"])
        margins += list(m[keep]); preds += list(decided[keep])
        labels += list(dump["y"][keep].astype(bool)); problems += list(np.array(dump["problems"])[keep])
    margins, labels, problems, preds = map(np.array, (margins, labels, problems, preds))
    auc, p, n = problem_permutation_p(margins, labels, problems, roc_auc_score)
    return {
        "chosen": {k: str(v) for k, v in search.best_params_.items()},
        "tv2_auc": round(auc, 4), "tv2_p": round(p, 4), "n": n,
        "tv2_balanced_at_batch_median": round(float(balanced_accuracy_score(labels, preds)), 4),
        "pass": bool(auc > 0.675 and p < 0.05),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--tv2", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()
    math = load(arguments.math)
    tv2 = [load(p) for p in arguments.tv2]
    report = {}
    for key, fn in [("H1_attnx", lambda d: column(d, "distract_attnx_mean")),
                    ("H2_mid_cka", lambda d: -h2_score(d))]:        # lower CKA = spurious
        s, y, pr = pooled(tv2, fn)
        auc, p, n = problem_permutation_p(s, y, pr, auc_spurious)
        x = fn(math)[:, None]
        report[key] = {"tv2_auc_spurious": round(auc, 4), "tv2_p": round(p, 4), "n": n,
                       "math_beyond_length": beyond_length(math, x)}
        report[key]["pass"] = bool(auc > 0.60 and p < 0.05 and report[key]["math_beyond_length"]["pass"])
        report[key]["tv2_only"] = bool(auc > 0.60 and p < 0.05 and not report[key]["pass"])
    report["H3_classifier"] = h3(math, tv2)
    text = json.dumps(report, indent=2)
    print(text)
    if arguments.out:
        arguments.out.write_text(text)


if __name__ == "__main__":
    main()
