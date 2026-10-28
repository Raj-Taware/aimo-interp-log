"""T-020 problems: hendrycks MATH train, the organisers' labelled source (dataset_id
hendrycks-math-train-reference-500). Gold = last \\boxed{} of the reference solution.

* gate: problems of the organisers' labelled DeepSeek set (augmented-sample-math-agg)
  found verbatim in MATH train, ``--gate`` of them stratified by label; carry
  ``official_label`` so math_gate.py can measure agreement.
* new: MATH train problems in no organiser table, sampled to the labelled set's
  level mix (``--new`` rows).

    python scripts/label_gen/fetch_math.py --out results/grok/T-020/problems.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import extract_last_boxed, write_jsonl

MATH = "https://huggingface.co/api/datasets/EleutherAI/hendrycks_math/parquet/{}/train/0.parquet"
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")
ORG = "https://huggingface.co/api/datasets/aimo-interp/{}/parquet/default/validation/0.parquet"
LABELLED, FULL = ORG.format("augmented-sample-math-agg"), ORG.format("augmented-sample-math-full-filtered")
DEEPSEEK_ALIAS = "qwen3-8b:low"


def key(text: str) -> str:
    return " ".join(str(text).split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=int, default=40)
    parser.add_argument("--new", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    math = pd.concat([pd.read_parquet(MATH.format(s)).assign(subject=s) for s in SUBJECTS], ignore_index=True)
    math["key"] = math["problem"].map(key)
    math["answer"] = math["solution"].map(extract_last_boxed)
    math = math.dropna(subset=["answer"]).drop_duplicates("key")

    lab = pd.read_parquet(LABELLED)
    lab = lab[lab["model_id"] == DEEPSEEK_ALIAS].drop_duplicates("original_problem")
    official = {key(p): bool(y) for p, y in zip(lab["original_problem"], lab["model_is_robust"])}
    seen = set(official) | {key(p) for p in pd.read_parquet(FULL)["original_problem"]}

    matched = math[math["key"].isin(official)].copy()
    matched["official_label"] = matched["key"].map(official)
    gate = pd.concat([g.sample(min(len(g), round(args.gate * len(g) / len(matched))), random_state=args.seed)
                      for _, g in matched.groupby("official_label")])
    pool = math[~math["key"].isin(seen)]
    mix = matched["level"].value_counts(normalize=True)
    new = pd.concat([pool[pool["level"] == lv].sample(round(args.new * share), random_state=args.seed)
                     for lv, share in mix.items()])

    rows = []
    for split, frame in (("gate", gate), ("new", new)):
        for r in frame.itertuples():
            rows.append({"problem_id": f"math-{hashlib.sha1(r.key.encode()).hexdigest()[:10]}", "problem": r.problem,
                         "answer": r.answer, "source": f"hendrycks_math/train/{r.subject}", "level": r.level,
                         "split": split, "official_label": getattr(r, "official_label", None)})
    assert len({r["problem_id"] for r in rows}) == len(rows)
    assert not ({key(r["problem"]) for r in rows if r["split"] == "new"} & seen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    print(json.dumps({"labelled": len(official), "matched_in_math_train": len(matched),
                      "gate": len(gate), "gate_robust": int(gate["official_label"].sum()), "new": len(new),
                      "level_mix": {k: round(v, 3) for k, v in mix.items()}}, indent=1))


if __name__ == "__main__":
    main()
