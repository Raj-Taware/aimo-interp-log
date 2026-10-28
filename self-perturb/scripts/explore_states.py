"""Exploratory (post pre-registration) sweep on T-019 DeepSeek states. Every candidate counts in Holm."""
import json, sys, numpy as np
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

R = "C:/fun/aimo-interp-log/self-perturb/results/grok/T-019/states/"
d = np.load(R + "math__deepseek-ai_DeepSeek-R1-0528-Qwen3-8B__states.npz", allow_pickle=True)
p = np.load(R + "pool__deepseek-ai_DeepSeek-R1-0528-Qwen3-8B__states.npz", allow_pickle=True)
y = d["labels"].astype(bool)
L = np.log(np.array([[len(t.split()), len(t)] for t in d["problems"]], float))
layers = list(d["layers"])


def H(pool, idx, src=d):
    return np.hstack([src[pool][:, layers.index(i)].astype(np.float32) for i in idx])


def scores(X_fn, ncomp, trans):
    """X_fn() -> (math states, pool states). trans: PCA fit on math+pool without labels."""
    S, P = X_fn()
    out = []
    for seed in range(5):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(L, y):
            if S is None:
                Z = L
            else:
                sc = StandardScaler().fit(np.vstack([S, P]) if trans else S[tr])
                pca = PCA(ncomp, random_state=0).fit(sc.transform(np.vstack([S, P]) if trans else S[tr]))
                Z = np.hstack([L, pca.transform(sc.transform(S))])
            Zs = StandardScaler().fit(Z[tr])
            m = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0], cv=StratifiedKFold(4, shuffle=True, random_state=seed),
                                     class_weight="balanced", scoring="balanced_accuracy", max_iter=5000)
            m.fit(Zs.transform(Z[tr]), y[tr])
            out.append(balanced_accuracy_score(y[te], m.predict(Zs.transform(Z[te]))))
    return np.array(out)


base = scores(lambda: (None, None), 0, False)
cands = {}
for pool in ("mean", "last"):
    for lay in (4, 8, 12, 20):
        for k in (4, 8, 16):
            for trans in (False, True):
                cands[f"{pool}_L{lay}_k{k}_{'trans' if trans else 'fold'}"] = (pool, [lay], k, trans)
    cands[f"{pool}_L4-8-12_k8_trans"] = (pool, [4, 8, 12], 8, True)
cands["mean8+last4_k8_trans"] = None
rows, pv = {}, {}
for name, spec in cands.items():
    if spec is None:
        fn = lambda: (np.hstack([H("mean", [8]), H("last", [4])]), np.hstack([H("mean", [8], p), H("last", [4], p)]))
        k, trans = 8, True
    else:
        pool, idx, k, trans = spec
        fn = lambda pool=pool, idx=idx: (H(pool, idx), H(pool, idx, p))
    s = scores(fn, k, trans)
    diff = s - base
    w, l = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
    pv[name] = binomtest(w, w + l, 0.5, alternative="greater").pvalue if w + l else 1.0
    rows[name] = dict(gain=round(float(diff.mean()), 4), wins=w, losses=l, p=round(pv[name], 4))
    print(name, rows[name], flush=True)
m = len(pv) + 20  # + the 20 pre-registered candidates
order = sorted(pv, key=pv.get)
run = 0
for r, kname in enumerate(order):
    run = max(run, min(1, (m - r) * pv[kname]))
    rows[kname]["p_holm"] = round(run, 4)
print("L base", round(base.mean(), 4))
print(json.dumps(sorted(rows.items(), key=lambda kv: -kv[1]["gain"])[:5], indent=1))
