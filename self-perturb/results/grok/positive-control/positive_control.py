"""Positive control: score v2's own unfitted drift with the T-024/T-025 gate 2 recipe."""
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

d = np.load(sys.argv[1], allow_pickle=True)
cols, X = list(d["columns"]), d["features"].astype(float)
y = ~d["labels"].astype(bool)
ln = np.array([len(p.split()) for p in d["problems"]], float)
print("n", len(y), "non-robust", int(y.sum()), "length best", round(max(roc_auc_score(y, ln), roc_auc_score(y, -ln)), 3))
cos = [i for i, c in enumerate(cols) if re.search(r"_cos_d[0-9.]+$", c)]
print("cos columns", len(cos), [cols[i] for i in cos[:4]])
rng = np.random.default_rng(0)
s = (1 - X[:, cos]).mean(1)
auc = roc_auc_score(y, s)
null = np.array([roc_auc_score(rng.permutation(y), s) for _ in range(2000)])
print("mean drift auc %.3f p %.4f" % (auc, (null >= auc).mean()))
pert = sorted({cols[i].split("_cos_d")[0] for i in cos})
for name in pert:
    idx = [i for i in cos if cols[i].startswith(name + "_cos_d")]
    print(" %-16s auc %.3f" % (name, roc_auc_score(y, (1 - X[:, idx]).mean(1))))
