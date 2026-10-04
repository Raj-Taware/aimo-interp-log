"""T-008 step 2: max |diff| and median per-feature corr between old and new 137 dumps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    old = np.load(args.old, allow_pickle=True)
    new = np.load(args.new, allow_pickle=True)
    old_probs = [str(p) for p in old["problems"]]
    new_probs = [str(p) for p in new["problems"]]
    # Align by problem text
    new_index = {p: i for i, p in enumerate(new_probs)}
    pairs = [(i, new_index[p]) for i, p in enumerate(old_probs) if p in new_index]
    if len(pairs) != len(old_probs) or len(pairs) != len(new_probs):
        print(f"WARNING: aligned {len(pairs)} / old={len(old_probs)} new={len(new_probs)}")
    oi = np.array([a for a, _ in pairs])
    ni = np.array([b for _, b in pairs])
    A = np.asarray(old["features"][oi], dtype=np.float64)
    B = np.asarray(new["features"][ni], dtype=np.float64)
    diff = np.abs(A - B)
    corrs = []
    for col in range(A.shape[1]):
        a, b = A[:, col], B[:, col]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            corrs.append(1.0 if np.allclose(a, b) else 0.0)
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    report = {
        "n_aligned": int(len(pairs)),
        "max_abs_diff": float(diff.max()),
        "median_per_feature_corr": float(np.median(corrs)),
        "mean_abs_diff": float(diff.mean()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "old_vs_new.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
