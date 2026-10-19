"""Exploratory H1-head on train-main-v2 (T-017). NOT the pre-registered test:
that one needed T-015 labels, which will not exist (labelling stopped
2026-10-19). Same head rule as the pre-registration, so the number is
comparable: per model, the top 5% of attention heads by mean excess attention
over all of that model's rows (labels unused), score = mean over those heads.

    python scripts/eval_heads.py --tv2 <trainv2__*.npz ...> [--frac 0.05]

Prints AUC(spurious) and problem-level permutation p for H1 (all heads, the
existing distract_attnx_mean) and H1-head, on the same rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_geometry import auc_spurious, column, load, pooled, problem_permutation_p  # noqa: E402


def head_score(path: Path, frac: float):
    d = np.load(path, allow_pickle=True)
    n = len(d["features"])
    heads = d["profile__distract__attnx_heads"].astype(float)          # (cases, attn layers, heads)
    flat = heads.reshape(len(heads), -1)
    k = max(1, round(frac * flat.shape[1]))
    top = np.argsort(np.nanmean(flat, axis=0))[::-1][:k]                # label-free selection
    score = np.full(n, np.nan)
    score[d["profile__distract__attnx_heads_case"].astype(int)] = np.nanmean(flat[:, top], axis=1)
    return score, [divmod(int(t), heads.shape[2]) for t in top]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tv2", type=Path, nargs="+", required=True)
    parser.add_argument("--frac", type=float, default=0.05)
    args = parser.parse_args()

    dumps, selected = [], {}
    for path in args.tv2:
        dump = load(path)
        dump["head_score"], selected[dump["model"]] = head_score(path, args.frac)
        dumps.append(dump)
    report = {"selected_heads_(attn_layer_idx,head)": selected}
    for name, fn in [("H1_attnx_all_heads", lambda d: column(d, "distract_attnx_mean")),
                     ("H1_head_top", lambda d: d["head_score"])]:
        auc, p, n = problem_permutation_p(*pooled(dumps, fn), auc_spurious)
        report[name] = {"auc_spurious": round(auc, 4), "perm_p": round(p, 4), "n": n}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
