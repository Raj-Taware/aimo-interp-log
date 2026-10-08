import json, sys, re, numpy as np
from pathlib import Path
root = Path(sys.argv[1]); klog = Path(sys.argv[2])
rows = json.loads(klog.read_text())
# seconds per case from kernel log timestamps
step = None; t_load = {}; t_wrote = {}
for r in rows:
    for line in r["data"].splitlines():
        if line.startswith("STEP "): step = line[5:].strip()
        if "model loaded" in line and step: t_load[step] = r["time"]
        if line.startswith("wrote") and step: t_wrote[step] = r["time"]
for f in sorted((root / "dumps").glob("*.npz")):
    d = np.load(f, allow_pickle=True)
    X = d["features"]; names = d["names"]
    key = f.stem.replace("__", "::", 1).replace("_", "/", 1) if False else None
    kind, mid = f.stem.split("__", 1)
    stepkey = [k for k in t_load if k.startswith(kind + "::") and k.split("::")[1].replace("/", "_") == mid]
    spc = (t_wrote[stepkey[0]] - t_load[stepkey[0]]) / len(X) if stepkey and stepkey[0] in t_wrote else float("nan")
    nan_only = [str(n) for j, n in enumerate(names) if not np.isfinite(X[:, j]).any()]
    lab = d["labels"]
    print(f"{f.stem}: cases={len(X)} finite={np.isfinite(X).mean():.3f} spc={spc:.2f} inf={int(np.isinf(X).sum())} unlabelled={int(np.isnan(lab).sum())}")
    print("   NaN-only:", nan_only)
    a = d["profile__distract__attnx"]
    print("   attnx finite idx (any case):", np.flatnonzero(np.isfinite(a).any(0)).tolist(), " all cases:", np.flatnonzero(np.isfinite(a).all(0)).tolist(), "n_states", a.shape[1])
    last = a.shape[1] - 1
    for m in ["cka", "wcos", "ans", "lens_kl"]:
        p = d[f"profile__distract__{m}"]
        print(f"   distract {m} last-idx finite share {np.isfinite(p[:, last]).mean():.2f}", end=";")
    print()
    d1 = [j for j, n in enumerate(names) if str(n).endswith("_d1.0")]
    print("   d1.0 cols:", len(d1), "finite share", float(np.isfinite(X[:, d1]).mean()), "all-NaN d1.0:", [str(names[j]) for j in d1 if not np.isfinite(X[:, j]).any()])
