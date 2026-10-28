"""T-020 variants without an LLM writer: the rule-based answer-preserving
perturbations in ``selfperturb.perturbations`` for the organisers' four MATH types
(distract, rename, rephrase, typos). One variant per (problem, type); a type whose
rule leaves the text unchanged is skipped (stage 3 then labels from the others).

    python scripts/label_gen/rule_variants.py --problems <problems.jsonl> --out <dir>
writes <dir>/stage2_variants.jsonl in the stage-2 schema (writer = "rule").
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from selfperturb.perturbations import PERTURBATIONS  # noqa: E402

from common import load_problems, stable_seed, write_jsonl  # noqa: E402

TYPES = ("distract", "rename", "rephrase", "typos")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for p in load_problems(args.problems):
        for t in TYPES:
            variant = PERTURBATIONS[t](p["problem"], stable_seed(f"{p['problem_id']}|{t}") % 2**31)
            if variant and variant.strip() != p["problem"].strip():
                rows.append({"problem_id": p["problem_id"], "type": t, "attempt": 0, "variant": variant,
                             "answer": p["answer"], "writer": "rule", "missing_numbers": [], "writer_tokens": 0})
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "stage2_variants.jsonl", rows)
    print(dict(Counter(r["type"] for r in rows)), "variants for", len(load_problems(args.problems)), "problems")


if __name__ == "__main__":
    main()
