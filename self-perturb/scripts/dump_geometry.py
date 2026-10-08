"""Dump token-aligned geometry features (and per-layer profiles) for one model.

    # MATH problems with DeepSeek labels (labelled for DeepSeek, borrowed otherwise)
    python scripts/dump_geometry.py --kind math --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    python scripts/dump_geometry.py --kind math --model Qwen/Qwen3.5-4B \
        --problems-of deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

    # competition-like rows of aimo-interp/train-main-v2 for that model
    python scripts/dump_geometry.py --kind trainv2 --model Qwen/Qwen3.5-4B

Writes ``<out>/<kind>__<safe_model_id>.npz`` with ``features`` (n, F),
``names``, ``labels`` (float; NaN = unlabelled), ``problems``, and
``profile__<family>__<metric>`` arrays (n, n_states) for figures. Flushes every
``--flush-every`` cases and resumes from an existing file.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_features import DEFAULT_SOURCES, load_rows, safe_id  # noqa: E402
from selfperturb.features import load_model  # noqa: E402
from selfperturb.geometry import FAMILIES, METRICS, case_geometry, geometry_names  # noqa: E402

TRAINV2 = "https://huggingface.co/api/datasets/aimo-interp/train-main-v2/parquet/default/train/0.parquet"


def trainv2_rows(source: str, model_id: str) -> list[tuple[str, float]]:
    frame = pd.read_parquet(source)
    frame = frame[frame["model_id"] == model_id]
    return [
        (str(p), float("nan") if pd.isna(y) else float(bool(y)))
        for p, y in zip(frame["problem"], frame["is_robust"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("math", "trainv2"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", help="parquet path or URL (default depends on --kind)")
    parser.add_argument("--problems-of", help="math only: use another model's labelled problems")
    parser.add_argument("--out", type=Path, default=Path("data/geometry"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=10)
    arguments = parser.parse_args()

    if arguments.kind == "math":
        labels_from = arguments.problems_of or arguments.model
        rows = [(p, float(y)) for p, y in load_rows([arguments.source] if arguments.source else DEFAULT_SOURCES, labels_from)]
        random.Random(0).shuffle(rows)
    else:
        labels_from = arguments.model
        rows = trainv2_rows(arguments.source or TRAINV2, arguments.model)
    if arguments.limit:
        rows = rows[: arguments.limit]
    if not rows:
        raise SystemExit(f"no rows for {arguments.model} ({arguments.kind})")

    arguments.out.mkdir(parents=True, exist_ok=True)
    destination = arguments.out / f"{arguments.kind}__{safe_id(arguments.model)}.npz"
    features, labels, problems, profiles = [], [], [], {}
    if destination.exists():
        cached = np.load(destination, allow_pickle=True)
        features, labels, problems = list(cached["features"]), list(cached["labels"]), list(cached["problems"])
        profiles = {k: list(cached[k]) for k in cached.files if k.startswith("profile__")}
        print(f"resuming from {len(features)}/{len(rows)}", flush=True)

    def flush() -> None:
        np.savez(
            destination,
            features=np.asarray(features, dtype=np.float32),
            names=np.asarray(geometry_names(), dtype=object),
            labels=np.asarray(labels, dtype=np.float32),
            problems=np.asarray(problems, dtype=object),
            model_id=arguments.model,
            labels_from=labels_from,
            kind=arguments.kind,
            **{k: np.asarray(v, dtype=np.float32) for k, v in profiles.items()},
        )

    started = time.time()
    tokenizer, model = load_model(arguments.model, attn_implementation="eager")
    print(f"{len(rows)} cases | model loaded in {time.time() - started:.1f}s", flush=True)
    started, done_before = time.time(), len(features)
    for index in range(len(features), len(rows)):
        problem, label = rows[index]
        vector, profile = case_geometry(tokenizer, model, problem)
        features.append(vector)
        labels.append(label)
        problems.append(problem)
        for family in FAMILIES:
            for metric in list(METRICS) + (["attnx"] if family == "distract" else []):
                profiles.setdefault(f"profile__{family}__{metric}", []).append(profile[family][metric])
        if (index + 1) % arguments.flush_every == 0:
            flush()
            rate = (time.time() - started) / (index + 1 - done_before)
            print(f"{index + 1}/{len(rows)}  {rate:.2f}s/case", flush=True)
    flush()
    print(f"wrote {destination} ({len(features)} cases, {len(geometry_names())} features)", flush=True)


if __name__ == "__main__":
    main()
