"""Match labelled problems to Hendrycks MATH and extract ground-truth answers.

    python scripts/fetch_answers.py \
        --extra results/grok/T-002/extra_labels.parquet \
        --out results/grok/T-006/answers.parquet

CPU only. Sources:
  - 137 unique problems from HF aimo-interp/augmented-sample-math-agg (validation)
  - 17 problems from T-002 extra_labels.parquet
  - Answers from EleutherAI/hendrycks_math (all subject configs, train+test)

Matching is by problem text (exact, then whitespace-normalized). problem_id is
a cross-check only. Answer = argument of the LAST \\boxed{...} or \\fbox{...}
with balanced-brace parsing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

AGG_DATASET = "aimo-interp/augmented-sample-math-agg"
MATH_DATASET = "EleutherAI/hendrycks_math"
BOX_OPENERS = (r"\boxed{", r"\fbox{")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_last_boxed(solution: str) -> tuple[str | None, int]:
    """Return (answer, n_boxed_commands) from solution.

    Finds every \\boxed{...} / \\fbox{...} with balanced braces and returns the
    argument of the last one. Nested braces inside the argument are allowed.
    """
    n = 0
    last: str | None = None
    i = 0
    while i < len(solution):
        matched_opener = None
        for opener in BOX_OPENERS:
            if solution.startswith(opener, i):
                matched_opener = opener
                break
        if matched_opener is None:
            i += 1
            continue
        n += 1
        start = i + len(matched_opener)
        depth = 1
        j = start
        while j < len(solution) and depth > 0:
            ch = solution[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = solution[start : j - 1]
            i = j
        else:
            # Unbalanced — skip past opener and keep searching.
            i = start
    return last, n


def load_math() -> pd.DataFrame:
    configs = get_dataset_config_names(MATH_DATASET)
    rows: list[dict] = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(MATH_DATASET, cfg, split=split)
            for row in ds:
                answer, n_boxed = extract_last_boxed(row["solution"])
                rows.append(
                    {
                        "problem": row["problem"],
                        "solution": row["solution"],
                        "subject": row.get("type") or cfg,
                        "level": row.get("level"),
                        "answer": answer,
                        "n_boxed": n_boxed,
                        "math_split": split,
                        "math_config": cfg,
                    }
                )
    return pd.DataFrame(rows)


def load_labelled(extra_path: Path) -> pd.DataFrame:
    agg = load_dataset(AGG_DATASET, split="validation").to_pandas()
    agg = agg.drop_duplicates(subset=["original_problem"], keep="first").copy()
    agg["source_set"] = "agg"
    agg["problem_id"] = agg["problem_id"].astype(str)
    agg = agg[["original_problem", "source_set", "problem_id"]]

    extra = pd.read_parquet(extra_path)
    extra = extra.drop_duplicates(subset=["original_problem"], keep="first").copy()
    extra["source_set"] = "extra"
    # T-002 parquet has no problem_id; leave blank.
    if "problem_id" not in extra.columns:
        extra["problem_id"] = ""
    else:
        extra["problem_id"] = extra["problem_id"].astype(str)
    extra = extra[["original_problem", "source_set", "problem_id"]]

    overlap = set(agg["original_problem"]) & set(extra["original_problem"])
    if overlap:
        raise SystemExit(f"agg/extra overlap unexpected: {len(overlap)} problems")

    out = pd.concat([agg, extra], ignore_index=True)
    if len(out) != 154:
        raise SystemExit(f"expected 154 labelled problems, got {len(out)}")
    return out


def match_problems(labelled: pd.DataFrame, math_df: pd.DataFrame) -> pd.DataFrame:
    exact_map: dict[str, list[int]] = {}
    norm_map: dict[str, list[int]] = {}
    for idx, problem in enumerate(math_df["problem"].tolist()):
        exact_map.setdefault(problem, []).append(idx)
        norm_map.setdefault(normalize_ws(problem), []).append(idx)

    records = []
    for row in labelled.itertuples(index=False):
        text = str(row.original_problem)
        match = "none"
        hit_idx = None

        exact_hits = exact_map.get(text, [])
        if exact_hits:
            match = "exact"
            hit_idx = exact_hits[0]
        else:
            norm_hits = norm_map.get(normalize_ws(text), [])
            if norm_hits:
                match = "normalized"
                hit_idx = norm_hits[0]

        if hit_idx is None:
            records.append(
                {
                    "original_problem": text,
                    "source_set": row.source_set,
                    "problem_id": row.problem_id,
                    "answer": None,
                    "subject": None,
                    "level": None,
                    "match": match,
                    "n_boxed": 0,
                }
            )
            continue

        mrow = math_df.iloc[hit_idx]
        records.append(
            {
                "original_problem": text,
                "source_set": row.source_set,
                "problem_id": row.problem_id,
                "answer": mrow["answer"],
                "subject": mrow["subject"],
                "level": mrow["level"],
                "match": match,
                "n_boxed": int(mrow["n_boxed"]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extra",
        type=Path,
        default=Path("results/grok/T-002/extra_labels.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/grok/T-006/answers.parquet"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path (defaults to answers.summary.json beside out).",
    )
    args = parser.parse_args()

    labelled = load_labelled(args.extra)
    math_df = load_math()
    answers = match_problems(labelled, math_df)

    # Drop helper column from parquet schema (task lists specific cols).
    parquet_cols = [
        "original_problem",
        "source_set",
        "problem_id",
        "answer",
        "subject",
        "level",
        "match",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    answers[parquet_cols].to_parquet(args.out, index=False)

    summary_path = args.summary or args.out.with_name("answers.summary.json")
    counts = answers["match"].value_counts().to_dict()
    multi_boxed = int((answers["n_boxed"] > 1).sum())
    unmatched = answers.loc[answers["match"] == "none", "original_problem"].tolist()
    samples = []
    for _, r in answers.head(5).iterrows():
        samples.append(
            {
                "problem_60": str(r["original_problem"])[:60],
                "answer": r["answer"],
                "match": r["match"],
            }
        )
    summary = {
        "n_problems": int(len(answers)),
        "match_counts": {
            "exact": int(counts.get("exact", 0)),
            "normalized": int(counts.get("normalized", 0)),
            "none": int(counts.get("none", 0)),
        },
        "multi_boxed_count": multi_boxed,
        "unmatched_count": len(unmatched),
        "unmatched_first80": [u[:80] for u in unmatched],
        "samples": samples,
        "math_rows_indexed": int(len(math_df)),
        "answers_missing_boxed": int(
            ((answers["match"] != "none") & answers["answer"].isna()).sum()
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
