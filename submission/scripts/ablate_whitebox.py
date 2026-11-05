"""White-box ablation on the labelled DeepSeek problems (White-Box Award evidence).

Same nested-CV recipe as ``evaluate.py`` (the one v2 was selected with), run on
feature blocks of one dump:

* ``full``      all v2 columns (87)
* ``hidden``    hidden-state drift only: ``*_cos_d*`` / ``*_norm_d*`` (72)
* ``no_hidden`` everything else: logit KL/entropy/margin + perturbation-applied flags (15)
* ``length``    problem length in words and characters (text-only control)
* ``shuffled``  hidden block with rows permuted within the label (null; breaks problem-feature link)

Balanced accuracy and AUC are means over 5 seeds x 5 outer folds.

    python scripts/ablate_whitebox.py results/feature-dumps/<dump>/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import nested_cv  # noqa: E402

HIDDEN = re.compile(r"_(cos|norm)_d[0-9.]+$")


def blocks(columns: list[str], features: np.ndarray, problems: np.ndarray, rng) -> dict[str, np.ndarray]:
    hidden = np.array([bool(HIDDEN.search(c)) for c in columns])
    length = np.array([[len(p.split()), len(p)] for p in problems], dtype=float)
    shuffled = features[:, hidden][rng.permutation(len(features))]
    return {"full": features, "hidden": features[:, hidden], "no_hidden": features[:, ~hidden],
            "length": length, "shuffled": shuffled}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--columns-from", type=Path, help="restrict to this artifact's columns (e.g. the v2 87)")
    args = parser.parse_args()
    dump = np.load(args.dump, allow_pickle=True)
    columns, features = list(dump["columns"]), dump["features"].astype(float)
    if args.columns_from:
        keep = list(np.load(args.columns_from, allow_pickle=True)["columns"])
        features, columns = features[:, [columns.index(c) for c in keep]], keep
    labels = dump["labels"].astype(bool)
    rng = np.random.default_rng(0)
    report = {"dump": str(args.dump), "n": int(len(labels)), "robust": int(labels.sum())}
    for name, block in blocks(columns, features, dump["problems"], rng).items():
        acc, acc_sd, bal, auc = nested_cv(block, labels, "balanced")
        report[name] = {"width": int(block.shape[1]), "balanced": round(float(bal), 3), "auc": round(float(auc), 3),
                        "accuracy": round(float(acc), 3), "accuracy_sd": round(float(acc_sd), 3)}
        print(name, report[name], flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
