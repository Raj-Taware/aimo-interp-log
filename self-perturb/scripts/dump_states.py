"""T-019: dump the question-only residual stream (no generation) for one model x one source.

    python scripts/dump_states.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --source data/math.parquet --out results/grok/T-019/states_math_deepseek

One forward pass per problem (``output_hidden_states=True``), prompt built exactly
as ``dump_features.py`` does (``selfperturb.features.build_prompt``, same chat
template and system prompt, no generation). For hidden-state indices
``range(0, n_layers + 1, 4)`` plus the last one (index 0 = embeddings), store

* ``mean``: mean over the prompt's tokens (single prompt, so no padding), float16
* ``last``: the final prompt token's state, float16

``<out>/states.npz`` keys: ``mean [n, L, d]``, ``last [n, L, d]``, ``layers``,
``n_tokens``, ``problems``, ``labels``, ``model_id``. Sources: the MATH validation
parquet (rows and order as ``dump_features.py``), ``trainv2`` (``problem`` /
``is_robust``), or the unlabelled pool (``original_problem``; labels are dummy).
Flushes every ``--flush-every`` rows and resumes. A file over ``--max-mb`` is split
by layer into ``states.part0.npz`` ... (each part holds every key, a layer subset).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_features import load_rows  # noqa: E402
from selfperturb.features import MAX_PROMPT_TOKENS, build_prompt, load_model  # noqa: E402

LAYER_STEP = 4
KEYS = ("mean", "last", "layers", "n_tokens", "problems", "labels", "model_id")


def state_layers(n_layers: int) -> list[int]:
    """Hidden-state indices ``range(0, n_layers + 1, 4)`` plus the last (``n_layers``)."""
    layers = list(range(0, n_layers + 1, LAYER_STEP))
    if layers[-1] != n_layers:
        layers.append(n_layers)
    return layers


@torch.no_grad()
def extract_states(tokenizer, model, problem: str, system_prompt: str = ""):
    """(mean [L, d] f16, last [L, d] f16, n_tokens, layers) for one problem, one forward pass."""
    prompt = build_prompt(tokenizer, problem, system_prompt)
    batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
    batch = {name: value.to(model.device) for name, value in batch.items()}
    output = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = output.hidden_states  # n_layers + 1 tensors of (1, T, d); may sit on different GPUs
    layers = state_layers(len(hidden) - 1)
    # Mean over real tokens in fp32 (batch size is 1: nothing is padded), stored as fp16.
    mean = torch.stack([hidden[i][0].float().cpu().mean(0) for i in layers]).half().numpy()
    last = torch.stack([hidden[i][0, -1, :].float().cpu() for i in layers]).half().numpy()
    return mean, last, int(batch["input_ids"].shape[1]), layers


def load_source_rows(source: str, model_id: str) -> list[tuple[str, float]]:
    """(problem, label) rows for one model from any of the three parquet layouts."""
    frame = pd.read_parquet(source)
    if "is_robust" in frame.columns:  # train-main-v2
        frame = frame[frame["model_id"] == model_id]
        return [(str(p), float("nan") if pd.isna(y) else float(bool(y)))
                for p, y in zip(frame["problem"], frame["is_robust"])]
    # MATH validation parquet / unlabelled pool: same selection and order as dump_features.py
    rows = [(p, float(y)) for p, y in load_rows([source], model_id)]
    random.Random(0).shuffle(rows)
    return rows


def _save(destination: Path, mean, last, layers, n_tokens, problems, labels, model_id) -> None:
    tmp = destination.with_suffix(".tmp.npz")
    np.savez(tmp, mean=np.asarray(mean, dtype=np.float16), last=np.asarray(last, dtype=np.float16),
             layers=np.asarray(layers, dtype=np.int32), n_tokens=np.asarray(n_tokens, dtype=np.int32),
             problems=np.asarray(problems, dtype=object), labels=np.asarray(labels, dtype=np.float32),
             model_id=model_id)
    tmp.replace(destination)


def run(tokenizer, model, rows, out_dir: Path, model_id: str, flush_every: int = 10,
        system_prompt: str = "") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "states.npz"
    mean, last, n_tokens, problems, labels, layers = [], [], [], [], [], None
    if destination.exists():
        cached = np.load(destination, allow_pickle=True)
        mean, last = list(cached["mean"]), list(cached["last"])
        n_tokens, problems = list(cached["n_tokens"]), list(cached["problems"])
        labels, layers = list(cached["labels"]), cached["layers"].tolist()
        print(f"resuming from {len(problems)}/{len(rows)}", flush=True)

    def flush() -> None:
        _save(destination, mean, last, layers, n_tokens, problems, labels, model_id)

    started, done_before = time.time(), len(problems)
    for index in range(len(problems), len(rows)):
        problem, label = rows[index]
        m, l, t, these = extract_states(tokenizer, model, problem, system_prompt)
        if layers is None:
            layers = these
        assert these == layers, "layer list changed between rows"
        if not (np.isfinite(m).all() and np.isfinite(l).all()):
            print(f"WARNING non-finite state at row {index}", flush=True)
        mean.append(m), last.append(l), n_tokens.append(t), problems.append(problem), labels.append(label)
        if (index + 1) % flush_every == 0:
            flush()
            print(f"{index + 1}/{len(rows)}  {(time.time() - started) / (index + 1 - done_before):.2f}s/row", flush=True)
    if layers is None:
        raise SystemExit("no rows")
    flush()
    return destination


def split_by_layer(path: Path, max_bytes: int) -> list[Path]:
    """Split ``path`` into ``<stem>.part<k>.npz`` files (all keys, layer subset) of at most ~max_bytes."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    layers = data["layers"]
    per_layer = (data["mean"][:, :1].nbytes + data["last"][:, :1].nbytes)
    per_part = max(1, int(max_bytes // per_layer))
    parts = []
    for k, start in enumerate(range(0, len(layers), per_part)):
        sel = slice(start, start + per_part)
        part = path.with_name(f"{path.stem}.part{k}.npz")
        _save(part, data["mean"][:, sel], data["last"][:, sel], layers[sel], data["n_tokens"], data["problems"],
              data["labels"], str(data["model_id"]))
        parts.append(part)
    path.unlink()
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="parquet path or URL")
    parser.add_argument("--out", type=Path, required=True, help="directory; writes states.npz")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--max-mb", type=float, default=90, help="split by layer above this size")
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()

    rows = load_source_rows(arguments.source, arguments.model)
    if arguments.limit:
        rows = rows[: arguments.limit]
    if not rows:
        raise SystemExit(f"no rows for {arguments.model} in {arguments.source}")
    started = time.time()
    tokenizer, model = load_model(arguments.model)  # default attention, fp16, same as dump_features.py
    print(f"{len(rows)} rows | model loaded in {time.time() - started:.1f}s", flush=True)
    destination = run(tokenizer, model, rows, arguments.out, arguments.model, arguments.flush_every,
                      arguments.system_prompt)
    size = destination.stat().st_size
    if size > arguments.max_mb * 1e6:
        parts = split_by_layer(destination, int(arguments.max_mb * 1e6))
        print(f"split {size / 1e6:.1f} MB into {[p.name for p in parts]}", flush=True)
    else:
        print(f"wrote {destination} ({size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
