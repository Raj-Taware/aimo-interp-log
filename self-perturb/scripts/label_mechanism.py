"""Why problem length predicts the organisers' robustness label (public training tables only).

1. Which perturbation type caused each non-robust label (agg ``permutations_causing_decay``).
2. Mean decay per type (full-filtered table).
3. How often a harmful variant lacks a number present in the original problem, and the
   model's accuracy on those (caveat: the stored variant text may be truncated).
4. Words per original problem by label.

    python scripts/label_mechanism.py
"""

from __future__ import annotations

import json
import re

import pandas as pd

URL = "https://huggingface.co/api/datasets/aimo-interp/{}/parquet/default/validation/0.parquet"
ALIAS = "qwen3-8b:low"


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def main() -> None:
    agg = pd.read_parquet(URL.format("augmented-sample-math-agg"))
    agg = agg[agg["model_id"] == ALIAS].drop_duplicates("original_problem")
    full = pd.read_parquet(URL.format("augmented-sample-math-full-filtered"))
    y = agg["model_is_robust"].astype(bool)
    harmful = []
    for r in agg.itertuples():
        for v in json.loads(r.permutations_causing_decay or "[]"):
            missing = bool(numbers(r.original_problem) - numbers(v["question"]))
            harmful.append({"type": v["permutation_type"], "acc": v["variant_accuracy"], "missing_number": missing})
    harmful = pd.DataFrame(harmful)
    causes = agg[~y]["permutations_causing_decay"].map(
        lambda s: ",".join(sorted({v["permutation_type"] for v in json.loads(s or "[]")})))
    report = {
        "n": int(len(agg)), "robust": int(y.sum()),
        "non_robust_cause": causes.value_counts().to_dict(),
        "mean_decay_by_type": full[full["model_id"] == ALIAS].groupby("permutation_type")
        ["absolute_accuracy_decay"].mean().round(3).to_dict(),
        "harmful_by_type": harmful.groupby("type").agg(n=("acc", "size"), acc=("acc", "mean"),
                                                       missing_number=("missing_number", "mean")).round(3).to_dict("index"),
        "domain_acc_by_missing_number": harmful[harmful["type"] == "domain"].groupby("missing_number")["acc"]
        .agg(["mean", "size"]).round(3).to_dict("index"),
        "words_by_label": agg.assign(w=agg["original_problem"].str.split().str.len()).groupby(y)["w"].mean().round(1).to_dict(),
    }
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
