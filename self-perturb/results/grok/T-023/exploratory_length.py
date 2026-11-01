"""Exploratory, after the T-023 verdict: is the self-domain pass only a length effect?"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from difficulty_probe import key

sd = np.load("results/grok/T-021/dump/selfdomain__deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz", allow_pickle=True)
r = np.load("results/grok/T-023/saliency__deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz", allow_pickle=True)
m = {key(p): s for p, s in zip(r["problems"], r["score"].astype(float))}
probs = [key(p) for p in sd["problems"]]
s = np.array([m.get(p, np.nan) for p in probs])
ok = np.isfinite(s)
y = ~sd["labels"][ok].astype(bool)
s = s[ok]
ln = np.log(np.array([len(p.split()) for p in probs], float)[ok])
print("n", len(y), "dropped", int((~ok).sum()), "nonrobust", int(y.sum()))
print("auc enrichment %.3f | auc short=nonrobust %.3f | spearman(enrichment, log length) %.3f" % (roc_auc_score(y, s), roc_auc_score(y, -ln), spearmanr(s, ln)[0]))


def cv(X):
    aucs = []
    for seed in range(20):
        p = cross_val_predict(LogisticRegression(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs)


zs = (s - s.mean()) / s.std()
zl = (ln - ln.mean()) / ln.std()
print("cv auc: length %.3f | enrichment %.3f | both %.3f" % (cv(zl[:, None]), cv(zs[:, None]), cv(np.c_[zl, zs])))
for lo, hi in zip(np.quantile(ln, [0, 1 / 3, 2 / 3]), np.quantile(ln, [1 / 3, 2 / 3, 1])):
    b = (ln >= lo) & (ln <= hi)
    print("length tercile n=%d nonrobust=%d auc=%.3f" % (b.sum(), y[b].sum(), roc_auc_score(y[b], s[b])))
