"""Recover labelled DeepSeek problems from public per-type tables.

Rule (Claude-verified on overlaps): a problem is robust iff
max(absolute_accuracy_decay) over that problem's rows <= 0.

    python scripts/build_extra_labels.py \
        --out results/grok/T-002/extra_labels.parquet

Drops problems already in the labelled agg set. If per-type A and B disagree
on a problem's label, drops it and records it in conflicts.json beside the
parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HF_PARQUET = (
    "https://huggingface.co/api/datasets/aimo-interp/{name}/parquet/default/validation/0.parquet"
)
LABELLED = HF_PARQUET.format(name="augmented-sample-math-agg")
PER_TYPE_A = HF_PARQUET.format(name="augmented-sample-math-full-filtered")
DEFAULT_B = (
    "https://raw.githubusercontent.com/aimo-interp/baselines/main/data/math-robust-final.csv"
)
MODEL_ALIAS = "qwen3-8b:low"


def load_table(path_or_url: str) -> pd.DataFrame:
    if path_or_url.endswith(".csv") or path_or_url.endswith(".csv?raw=1"):
        return pd.read_csv(path_or_url)
    return pd.read_parquet(path_or_url)


def label_by_problem(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """One row per original_problem with the max-decay robust rule."""
    frame = frame[frame["model_id"] == MODEL_ALIAS].copy()
    frame["original_problem"] = frame["original_problem"].astype(str)
    grouped = (
        frame.groupby("original_problem", sort=False)
        .agg(
            max_decay=("absolute_accuracy_decay", "max"),
            types_present=("permutation_type", lambda s: sorted({str(x) for x in s.dropna()})),
        )
        .reset_index()
    )
    grouped["model_is_robust"] = grouped["max_decay"] <= 0
    grouped["model_id"] = MODEL_ALIAS
    grouped["source"] = source
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled", default=LABELLED)
    parser.add_argument("--per-type-a", default=PER_TYPE_A)
    parser.add_argument("--per-type-b", default=DEFAULT_B)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    labelled = load_table(arguments.labelled)
    labelled = labelled[labelled["model_id"] == MODEL_ALIAS]
    already = set(labelled["original_problem"].astype(str))

    a = label_by_problem(load_table(arguments.per_type_a), "A")
    b = label_by_problem(load_table(arguments.per_type_b), "B")

    a = a[~a["original_problem"].isin(already)]
    b = b[~b["original_problem"].isin(already)]

    a_map = a.set_index("original_problem")
    b_map = b.set_index("original_problem")
    both = sorted(set(a_map.index) & set(b_map.index))

    conflicts = []
    keep_both = []
    for problem in both:
        la = bool(a_map.loc[problem, "model_is_robust"])
        lb = bool(b_map.loc[problem, "model_is_robust"])
        if la != lb:
            conflicts.append(
                {
                    "original_problem": problem,
                    "label_A": la,
                    "label_B": lb,
                    "max_decay_A": float(a_map.loc[problem, "max_decay"]),
                    "max_decay_B": float(b_map.loc[problem, "max_decay"]),
                }
            )
        else:
            keep_both.append(problem)

    conflict_problems = {c["original_problem"] for c in conflicts}
    a_only = a[~a["original_problem"].isin(set(both))]
    b_only = b[~b["original_problem"].isin(set(both))]

    both_rows = []
    for problem in keep_both:
        row_a = a_map.loc[problem]
        row_b = b_map.loc[problem]
        types = sorted(set(row_a["types_present"]) | set(row_b["types_present"]))
        # Prefer the more conservative (larger) max_decay when both agree on label.
        max_decay = max(float(row_a["max_decay"]), float(row_b["max_decay"]))
        both_rows.append(
            {
                "original_problem": problem,
                "model_id": MODEL_ALIAS,
                "model_is_robust": bool(row_a["model_is_robust"]),
                "source": "both",
                "max_decay": max_decay,
                "types_present": types,
            }
        )

    out = pd.concat(
        [
            a_only.assign(source="A"),
            b_only.assign(source="B"),
            pd.DataFrame(both_rows),
        ],
        ignore_index=True,
    )
    # Drop any residual conflict rows if they slipped through.
    out = out[~out["original_problem"].isin(conflict_problems)].reset_index(drop=True)
    out = out[
        [
            "original_problem",
            "model_id",
            "model_is_robust",
            "source",
            "max_decay",
            "types_present",
        ]
    ]

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(arguments.out, index=False)

    notes = {
        "n_labelled_already": len(already),
        "n_new": len(out),
        "from_A_only": int((out["source"] == "A").sum()),
        "from_B_only": int((out["source"] == "B").sum()),
        "in_both": int((out["source"] == "both").sum()),
        "robust": int(out["model_is_robust"].sum()),
        "spurious": int((~out["model_is_robust"]).sum()),
        "conflicts_dropped": len(conflicts),
        "conflicts": conflicts,
    }
    notes_path = arguments.out.with_name("extra_labels_notes.json")
    notes_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in notes.items() if k != "conflicts"}, indent=2))
    print(f"wrote {arguments.out} ({len(out)} rows) and {notes_path}")


if __name__ == "__main__":
    main()
