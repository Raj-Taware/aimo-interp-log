"""T-021 pre-registered test (written before the data exists).

One white-box candidate, no Holm: W = every non-text column of dump_selfdomain.py.
Same recipe as eval_states.py (5 seeds x 5 outer folds, balanced LR, inner CV over C;
W scaled then PCA to min(32, width) inside the fold). Pass iff L+W beats L by >= 0.03
mean balanced accuracy AND wins >= 60% of non-tied folds AND one-sided sign p < 0.05.
Reported, not part of the pass: L+text (retain, rew_words) and L+text+W, to show
whether any gain is the rewrite's number retention (black-box) rather than the internals.

    python scripts/eval_selfdomain.py results/grok/T-021/selfdomain__<model>.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import eval_states as es
from dump_selfdomain import TEXT_COLUMNS


def compare(base, other) -> dict:
    diff = np.array(other) - np.array(base)
    wins, losses = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
    p = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0
    return {"balanced": round(float(np.mean(other)), 4), "gain": round(float(diff.mean()), 4), "wins": wins,
            "losses": losses, "win_share": round(wins / max(wins + losses, 1), 3), "p": round(float(p), 4)}


def run(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    y, columns = data["labels"].astype(bool), list(data["columns"])
    X = data["features"].astype(float)
    text = [columns.index(c) for c in TEXT_COLUMNS]
    white = [i for i in range(len(columns)) if i not in text]
    L = es.length_features(data["problems"])
    es.PCA_DIMS = min(32, len(white))
    base, _ = es.fold_scores(L, y, 2)
    report = {"n": int(len(y)), "robust": int(y.sum()), "L": round(float(np.mean(base)), 4)}
    report["L+W"] = compare(base, es.fold_scores(np.hstack([L, X[:, white]]), y, 2)[0])
    es.PCA_DIMS = len(text)
    report["L+text"] = compare(base, es.fold_scores(np.hstack([L, X[:, text]]), y, 2)[0])
    es.PCA_DIMS = min(32, len(white) + len(text))
    report["L+text+W"] = compare(base, es.fold_scores(np.hstack([L, X[:, text + white]]), y, 2)[0])
    w = report["L+W"]
    report["pass"] = bool(w["gain"] >= es.MIN_GAIN and w["win_share"] >= es.MIN_WIN_SHARE and w["p"] < 0.05)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path)
    args = parser.parse_args()
    report = run(args.npz)
    (args.npz.parent / "eval_selfdomain.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
