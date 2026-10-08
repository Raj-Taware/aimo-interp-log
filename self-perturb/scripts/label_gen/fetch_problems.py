"""Fetch the olympiad problem pool: AI-MO/aimo-validation-aime (90) + -amc (83).

    python scripts/label_gen/fetch_problems.py --out results/grok/T-014/problems.jsonl

One row per problem: problem_id (aime-<row> / amc-<row>, the row index in the
dataset's parquet -- the AMC ``id`` column is NOT unique: 49 ids over 83 rows),
dataset_id, source, problem, answer
(normalized integer string), url. Reads the datasets' parquet exports over HTTP
(no ``datasets`` dependency).
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path

from common import normalize_answer

SOURCES = {
    "aime": "https://huggingface.co/api/datasets/AI-MO/aimo-validation-aime/parquet/default/train/0.parquet",
    "amc": "https://huggingface.co/api/datasets/AI-MO/aimo-validation-amc/parquet/default/train/0.parquet",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd

    rows = []
    for name, url in SOURCES.items():
        with urllib.request.urlopen(url) as response:
            frame = pd.read_parquet(io.BytesIO(response.read()))
        for row_index, rec in enumerate(frame.to_dict("records")):
            answer = normalize_answer(rec["answer"])
            rows.append({"problem_id": f"{name}-{row_index}", "dataset_id": int(rec["id"]), "source": f"AI-MO/aimo-validation-{name}",
                         "problem": rec["problem"], "answer": answer, "answer_raw": str(rec["answer"]),
                         "url": rec.get("url")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert len({r["problem_id"] for r in rows}) == len(rows)
    non_int = [r["problem_id"] for r in rows if not r["answer"].lstrip("-").isdigit()]
    print(f"wrote {len(rows)} problems to {args.out}; non-integer answers: {non_int}")


if __name__ == "__main__":
    main()
