"""Extract self-perturbation features for every case of one model. Resumable.

Run once per checkpoint. Qwen3.5-4B fits a single T4 in fp16; the 7-8B
checkpoints want 2xT4 (``device_map="auto"``) or the H100 slot.

    python scripts/dump_features.py \
        --model Qwen/Qwen3.5-4B \
        --dataset aimo-interp/aimo-interp-challenge-sample-full \
        --out data/features

Writes ``<out>/<safe_model_id>.npz`` holding the feature matrix, labels, and
problem texts. Progress is flushed every ``--flush-every`` cases, so a killed
Kaggle or Lightning session resumes instead of restarting -- which matters when
the H100 time arrives in fixed five-hour slots.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selfperturb.features import case_features, feature_names, load_model  # noqa: E402

# Column-name candidates, since the public dumps are not consistent with each other.
PROBLEM_COLUMNS = ("original_problem", "problem", "problem_text")
LABEL_COLUMNS = ("model_is_robust", "is_robust")
MODEL_COLUMNS = ("model_id", "model")


def _pick(row, candidates, default=None):
    for name in candidates:
        if name in row and row[name] is not None:
            return row[name]
    return default


def safe_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def load_rows(dataset: str, model_id: str, split: str):
    """Rows for one checkpoint: (problem_text, label) pairs."""
    from datasets import load_dataset  # dev-time only; absent from the eval image

    data = load_dataset(dataset, split=split)
    rows = []
    for row in data:
        problem = _pick(row, PROBLEM_COLUMNS)
        label = _pick(row, LABEL_COLUMNS)
        row_model = _pick(row, MODEL_COLUMNS, "")
        if problem is None or label is None:
            continue
        # Match loosely: dumps mix aliases ("qwen3-8b:low") with full checkpoint ids.
        if model_id and row_model and model_id.split("/")[-1].lower() not in str(row_model).lower():
            continue
        rows.append((str(problem), bool(label)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="aimo-interp/aimo-interp-challenge-sample-full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("data/features"))
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    destination = arguments.out / f"{safe_id(arguments.model)}.npz"

    rows = load_rows(arguments.dataset, arguments.model, arguments.split)
    if arguments.limit:
        rows = rows[: arguments.limit]
    if not rows:
        raise SystemExit(f"no rows matched model {arguments.model!r} in {arguments.dataset}")

    # Resume: keep whatever a previous slot already computed.
    features: list[np.ndarray] = []
    labels: list[bool] = []
    problems: list[str] = []
    if destination.exists():
        cached = np.load(destination, allow_pickle=True)
        features = list(cached["features"])
        labels = list(cached["labels"])
        problems = list(cached["problems"])
        print(f"resuming from {len(features)}/{len(rows)} cases")

    def flush() -> None:
        np.savez(
            destination,
            features=np.asarray(features, dtype=np.float32),
            labels=np.asarray(labels, dtype=bool),
            problems=np.asarray(problems, dtype=object),
            columns=np.asarray(feature_names(), dtype=object),
            model_id=arguments.model,
        )

    tokenizer, model = load_model(arguments.model)
    started = time.time()
    for index in range(len(features), len(rows)):
        problem, label = rows[index]
        features.append(case_features(tokenizer, model, problem, arguments.system_prompt))
        labels.append(label)
        problems.append(problem)
        if (index + 1) % arguments.flush_every == 0:
            flush()
            rate = (time.time() - started) / (index + 1 - 0)
            print(f"{index + 1}/{len(rows)} cases  {rate:.2f}s/case", flush=True)

    flush()
    print(f"wrote {destination} with {len(features)} cases")


if __name__ == "__main__":
    main()
