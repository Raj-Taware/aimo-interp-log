"""Extract self-perturbation features for every case of one model. Resumable.

Run once per checkpoint. Qwen3.5-4B fits a single T4 in fp16; the 7-8B
checkpoints want 2xT4 (``device_map="auto"``) or the H100 slot.

    python scripts/dump_features.py \
        --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out data/features

Writes ``<out>/<safe_model_id>.npz`` holding the feature matrix, labels, and
problem texts. Progress is flushed every ``--flush-every`` cases, so a killed
Kaggle or Lightning session resumes instead of restarting -- which matters when
the H100 time arrives in fixed five-hour slots.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfperturb.features import case_features, feature_names, load_model  # noqa: E402

HF_PARQUET = "https://huggingface.co/api/datasets/aimo-interp/{name}/parquet/default/validation/0.parquet"
DEFAULT_SOURCES = (HF_PARQUET.format(name="augmented-sample-math-agg"),)

# The public dumps label rows with evaluation-time aliases, not checkpoint ids.
# The getting-started README confirms qwen3-8b:low resolves to this checkpoint.
ROW_ALIASES = {
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ("qwen3-8b:low",),
}


def safe_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def load_rows(sources, model_id: str) -> list[tuple[str, bool]]:
    """(problem_text, label) pairs for one checkpoint, one row per problem."""
    frame = pd.concat([pd.read_parquet(source) for source in sources], ignore_index=True)
    wanted = {model_id, *ROW_ALIASES.get(model_id, ())}
    frame = frame[frame["model_id"].isin(wanted)].dropna(subset=["model_is_robust"])
    # One label per (model, problem); the public dumps repeat a few problems,
    # and a duplicate would leak across CV folds.
    frame = frame.drop_duplicates(subset=["original_problem"])
    return list(zip(frame["original_problem"].astype(str), frame["model_is_robust"].astype(bool)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", action="append", help="parquet path or URL; repeatable")
    parser.add_argument("--out", type=Path, default=Path("data/features"))
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--problems-of",
        help="take the problem set (and labels) of another model. The labels then "
        "describe THAT model, not --model: use only for unlabelled analysis such "
        "as per-model threshold calibration, never for training.",
    )
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    destination = arguments.out / f"{safe_id(arguments.model)}.npz"

    labels_from = arguments.problems_of or arguments.model
    rows = load_rows(arguments.source or DEFAULT_SOURCES, labels_from)
    # The public dumps arrive sorted by label, so a --limit prefix would be one
    # class only. A fixed seed keeps the order identical across resumed runs.
    random.Random(0).shuffle(rows)
    if arguments.limit:
        rows = rows[: arguments.limit]
    if not rows:
        raise SystemExit(f"no labelled rows for {arguments.model!r}")
    print(f"{len(rows)} labelled cases for {arguments.model}", flush=True)

    # Resume: keep whatever a previous slot already computed.
    features: list[np.ndarray] = []
    labels: list[bool] = []
    problems: list[str] = []
    if destination.exists():
        cached = np.load(destination, allow_pickle=True)
        features = list(cached["features"])
        labels = list(cached["labels"])
        problems = list(cached["problems"])
        print(f"resuming from {len(features)}/{len(rows)} cases", flush=True)

    def flush() -> None:
        np.savez(
            destination,
            features=np.asarray(features, dtype=np.float32),
            labels=np.asarray(labels, dtype=bool),
            problems=np.asarray(problems, dtype=object),
            columns=np.asarray(feature_names(), dtype=object),
            model_id=arguments.model,
            labels_from=labels_from,
        )

    started = time.time()
    tokenizer, model = load_model(arguments.model)
    print(f"model loaded in {time.time() - started:.1f}s", flush=True)

    started, done_before = time.time(), len(features)
    for index in range(len(features), len(rows)):
        problem, label = rows[index]
        features.append(case_features(tokenizer, model, problem, arguments.system_prompt))
        labels.append(label)
        problems.append(problem)
        if (index + 1) % arguments.flush_every == 0:
            flush()
            rate = (time.time() - started) / (index + 1 - done_before)
            print(f"{index + 1}/{len(rows)} cases  {rate:.2f}s/case", flush=True)

    flush()
    print(f"wrote {destination} with {len(features)} cases", flush=True)


if __name__ == "__main__":
    main()
