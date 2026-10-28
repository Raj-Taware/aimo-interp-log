"""Competence-margin probe: a linear "difficulty" direction in the question-only residual
stream, trained without robustness labels, predicts non-robustness on train-main-v2.

Mechanism (see collab/reports/T-021.md follow-up): on competition problems robustness
labels agree across models (35/35 pairs) and non-robust problems are the hard ones; a
perturbation removes >= 0.25 accuracy only where the model is near its competence limit.

Probe: per model, ridge from T-019 question states (one layer x pooling, chosen by 5-fold
CV Spearman on AIME only) to the AIME problem number (1-15, public AoPS URL; 90 problems of
AI-MO/aimo-validation-aime, train-main-v2 problems excluded). Applied to train-main-v2;
score = predicted difficulty, higher -> non-robust (sign fixed in advance).

Controls: length; a text-only ridge (TF-IDF word 1-2 grams + char 3-5 grams) on the same 90
AIME problems; the same probe pipeline with shuffled AIME numbers (null). Problem-level AUC
averages the within-model rank scores of a problem; its label is non-robust if any model's is.

    python scripts/difficulty_probe.py --out results/difficulty_probe.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
STATES = ROOT / "results/grok/T-019/states"
POOL = ROOT / "results/grok/T-014/problems.jsonl"
TRAINV2 = "https://huggingface.co/api/datasets/aimo-interp/train-main-v2/parquet/default/train/0.parquet"
MODELS = ["Qwen_Qwen3.5-4B", "Skywork_Skywork-OR1-Math-7B", "allenai_Olmo-3-7B-Think",
          "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"]
ALPHAS = np.logspace(1, 5, 9)


def key(text) -> str:
    return " ".join(str(text).split())


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def aime_rows(pool_problems, meta, exclude):
    idx = [i for i, p in enumerate(pool_problems)
           if p in meta and meta[p]["source"].endswith("aime") and p not in exclude]
    numbers = np.array([int(re.search(r"Problem_(\d+)", meta[pool_problems[i]]["url"]).group(1)) for i in idx])
    return idx, numbers


def pick_layer(pool, idx, d):
    """(cv rho, pooling, layer index) with the best 5-fold CV Spearman on AIME."""
    best = None
    for pooling in ("mean", "last"):
        for li in range(1, len(pool["layers"])):
            X = pool[pooling][idx][:, li].astype(np.float32)
            pred = cross_val_predict(ridge(), X, d, cv=KFold(5, shuffle=True, random_state=0))
            rho = spearmanr(pred, d).correlation
            if best is None or rho > best[0]:
                best = (rho, pooling, li)
    return best


def auc_with_perm(y, s, rng, n=5000):
    auc = roc_auc_score(y, s)
    null = np.array([roc_auc_score(rng.permutation(y), s) for _ in range(n)])
    return round(float(auc), 3), round(float((null >= auc).mean()), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--null-draws", type=int, default=50)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    meta = {key(r["problem"]): r for r in map(json.loads, POOL.open(encoding="utf-8"))}
    tv = pd.read_parquet(TRAINV2)
    exclude = {key(p) for p in tv["problem"]}

    report, rows = {"models": {}}, []
    text_fit = None
    for m in MODELS:
        pool = np.load(STATES / f"pool__{m}__states.npz", allow_pickle=True)
        pp = [key(p) for p in pool["problems"]]
        idx, d = aime_rows(pp, meta, exclude)
        rho, pooling, li = pick_layer(pool, idx, d)
        X = pool[pooling][idx][:, li].astype(np.float32)
        probe = ridge().fit(X, d)
        tr = np.load(STATES / f"trainv2__{m}__states.npz", allow_pickle=True)
        ok = ~np.isnan(tr["labels"])
        H = tr[pooling][ok][:, li].astype(np.float32)
        probs = [key(p) for p in tr["problems"][ok]]
        score = probe.predict(H)
        # null: same pipeline, shuffled AIME numbers
        null = [ridge().fit(X, rng.permutation(d)).predict(H) for _ in range(args.null_draws)]
        if text_fit is None:  # text control: same 90 AIME problems, same target
            texts = [pp[i] for i in idx]
            vw, vc = TfidfVectorizer(ngram_range=(1, 2), min_df=2), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
            T = hstack([vw.fit_transform(texts), vc.fit_transform(texts)]).tocsr()
            alpha = min(ALPHAS, key=lambda a: -spearmanr(cross_val_predict(Ridge(alpha=a), T, d, cv=KFold(5, shuffle=True, random_state=0)), d).correlation)
            text_cv = spearmanr(cross_val_predict(Ridge(alpha=alpha), T, d, cv=KFold(5, shuffle=True, random_state=0)), d).correlation
            text_fit = (vw, vc, Ridge(alpha=alpha).fit(T, d))
            report["text_control"] = {"alpha": float(alpha), "aime_cv_rho": round(float(text_cv), 3)}
        vw, vc, tm = text_fit
        tscore = tm.predict(hstack([vw.transform(probs), vc.transform(probs)]).tocsr())
        report["models"][m] = {"layer": int(pool["layers"][li]), "pooling": pooling, "aime_cv_rho": round(float(rho), 3)}
        for j, p in enumerate(probs):
            rows.append({"model": m, "problem": p, "robust": bool(tr["labels"][ok][j]),
                         "probe": rankdata(score)[j] / len(score), "text": rankdata(tscore)[j] / len(tscore),
                         "length": rankdata([len(q.split()) for q in probs])[j] / len(probs),
                         "null": [rankdata(n)[j] / len(n) for n in null]})
    df = pd.DataFrame(rows)
    nr = ~df["robust"].values
    for col in ("probe", "text", "length"):
        report[f"row_{col}"] = dict(zip(("auc", "perm_p"), auc_with_perm(nr, df[col].values, rng)))
    null_auc = [roc_auc_score(nr, [r[k] for r in df["null"]]) for k in range(args.null_draws)]
    report["row_probe_shuffled_null"] = {"mean": round(float(np.mean(null_auc)), 3), "max": round(float(np.max(null_auc)), 3)}
    report["per_model_probe_auc"] = {m: round(float(roc_auc_score(~g["robust"], g["probe"])), 3) if g["robust"].nunique() > 1 else None
                                     for m, g in df.groupby("model")}
    prob = df.groupby("problem").agg(nonrobust=("robust", lambda s: not s.all()), probe=("probe", "mean"),
                                     text=("text", "mean"), length=("length", "mean"))
    for col in ("probe", "text", "length"):
        report[f"problem_{col}"] = dict(zip(("auc", "perm_p"), auc_with_perm(prob["nonrobust"].values, prob[col].values, rng)))
    report["n_rows"], report["n_problems"] = int(len(df)), int(len(prob))
    report["nonrobust_rows"], report["nonrobust_problems"] = int(nr.sum()), int(prob["nonrobust"].sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
