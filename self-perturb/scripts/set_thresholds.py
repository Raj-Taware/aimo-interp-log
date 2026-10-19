"""v4b: per-model decision threshold from unlabelled competition-like problems.

Since Oct 2026 the grader calls ``are_robust`` once per problem in a fresh
process, so v3b/v4's batch-median split never fires and the artifact's zero
threshold (training median, MATH-length problems) makes v4 near all-True on
competition-length text. This sets ``threshold`` = median margin over the 173
AIME/AMC pool problems (no labels used) and writes new artifacts.

    # 1. pool as a dump_features source (labels are dummies, never used)
    python scripts/set_thresholds.py --make-source data/pool_unlabelled.parquet \
        --problems <aimo-interp-log>/self-perturb/results/grok/T-014/problems.jsonl
    # 2. (GPU) dump_features.py --model <m> --source data/pool_unlabelled.parquet --out <feat_dir>
    # 3. thresholds
    python scripts/set_thresholds.py --features <feat_dir> --artifacts results/artifacts-v4 --out results/artifacts-v4b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "solutions" / "self-perturb"))
from selfperturb_infer import _margin  # noqa: E402

MODELS = ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")


def safe_id(model_id: str) -> str:
    return model_id.replace("/", "_")


def make_source(problems: Path, out: Path) -> None:
    import pandas as pd

    texts = [json.loads(line)["problem"] for line in problems.open(encoding="utf-8") if line.strip()]
    frame = pd.DataFrame([{"model_id": m, "original_problem": t, "model_is_robust": False}
                          for m in MODELS for t in texts])
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    print(f"wrote {out}: {len(texts)} problems x {len(MODELS)} models")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-source", type=Path)
    parser.add_argument("--problems", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.make_source:
        return make_source(args.problems, args.make_source)

    args.out.mkdir(parents=True, exist_ok=True)
    report, every = {}, []
    for model in MODELS:
        art_path = args.artifacts / f"{safe_id(model)}.npz"
        artifact = dict(np.load(art_path, allow_pickle=True))
        feats = np.load(args.features / f"{safe_id(model)}.npz", allow_pickle=True)["features"]
        margins = np.array([_margin(artifact, f) for f in feats])
        margins = margins[np.isfinite(margins)]
        threshold = float(np.median(margins))
        every += list(feats)
        np.savez(args.out / art_path.name, **artifact, threshold=np.float32(threshold))
        report[model] = {"n": int(len(margins)), "threshold": round(threshold, 4),
                         "share_above_zero": round(float((margins >= 0).mean()), 3)}
    generic = dict(np.load(args.artifacts / "generic.npz", allow_pickle=True))
    gm = np.array([_margin(generic, f) for f in every])
    np.savez(args.out / "generic.npz", **generic, threshold=np.float32(np.median(gm[np.isfinite(gm)])))
    for extra in args.artifacts.glob("*.json"):
        shutil.copy(extra, args.out / extra.name)
    (args.out / "thresholds.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
