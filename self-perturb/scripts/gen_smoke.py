"""Generation smoke test: trace length + accuracy for four small models.

Selects 5 robust + 5 spurious DeepSeek-labelled problems (from
``aimo-interp/augmented-sample-math-agg`` ``model_is_robust``, alias
``qwen3-8b:low``) with ``random.Random(0)``, then runs plain transformers
``generate`` on each model.

    python scripts/gen_smoke.py \
        --answers results/grok/T-006/answers.parquet \
        --out results/grok/T-007 \
        --model Qwen/Qwen3.5-4B

Kaggle T4x2, env per PROTOCOL (transformers==5.13.0, accelerate==1.13.0, fp16).
No vLLM. Stop a model early if wall time would exceed --max-minutes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

import pandas as pd

AGG_DATASET = "aimo-interp/augmented-sample-math-agg"
DEEPSEEK_ALIAS = "qwen3-8b:low"
DEFAULT_MODELS = (
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
)
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 0


def extract_last_boxed(text: str) -> str | None:
    """Argument of the LAST \\boxed{...} with balanced braces."""
    opener = r"\boxed{"
    last: str | None = None
    i = 0
    while i < len(text):
        if not text.startswith(opener, i):
            i += 1
            continue
        start = i + len(opener)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[start : j - 1]
            i = j
        else:
            i = start
    return last


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    return text.replace(" ", "").replace("$", "")


def select_problems(answers_path: Path, n_robust: int = 5, n_spurious: int = 5, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    answers = pd.read_parquet(answers_path)
    agg = load_dataset(AGG_DATASET, split="validation").to_pandas()
    agg = agg[agg["model_id"] == DEEPSEEK_ALIAS].drop_duplicates("original_problem")
    merged = agg[["original_problem", "model_is_robust"]].merge(
        answers[["original_problem", "answer", "problem_id"]],
        on="original_problem",
        how="inner",
    )
    robust = merged[merged["model_is_robust"] == True].sort_values("problem_id")  # noqa: E712
    spurious = merged[merged["model_is_robust"] == False].sort_values("problem_id")  # noqa: E712
    rng = random.Random(seed)
    sel_r = rng.sample(robust["original_problem"].tolist(), n_robust)
    sel_s = rng.sample(spurious["original_problem"].tolist(), n_spurious)
    rows: list[dict] = []
    for problem in sel_r + sel_s:
        row = merged[merged["original_problem"] == problem].iloc[0]
        rows.append(
            {
                "original_problem": problem,
                "answer": str(row["answer"]),
                "problem_id": str(row["problem_id"]),
                "model_is_robust": bool(row["model_is_robust"]),
                "label": "robust" if bool(row["model_is_robust"]) else "spurious",
            }
        )
    return rows


def load_problems(path: Path | None, answers_path: Path | None) -> list[dict]:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if answers_path is None:
        raise SystemExit("need --problems or --answers")
    return select_problems(answers_path)


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def summarize(token_counts: list[int], n_cap: int, n_boxed: int, n_correct: int, wall: float, n: int) -> dict:
    ordered = sorted(token_counts)
    total_tok = sum(token_counts)
    return {
        "n": n,
        "median_tokens": percentile(ordered, 50) if ordered else float("nan"),
        "mean_tokens": (total_tok / len(token_counts)) if token_counts else float("nan"),
        "p90_tokens": percentile(ordered, 90) if ordered else float("nan"),
        "max_tokens": max(token_counts) if token_counts else 0,
        "pct_hit_8192_cap": 100.0 * n_cap / n if n else float("nan"),
        "pct_with_boxed": 100.0 * n_boxed / n if n else float("nan"),
        "accuracy_of_10": n_correct,  # absolute count of 10; also fine as /10 later
        "n_correct": n_correct,
        "wall_seconds": round(wall, 1),
        "generated_tok_per_s": round(total_tok / wall, 2) if wall > 0 else float("nan"),
        "total_generated_tokens": total_tok,
    }


def run_model(
    model_id: str,
    problems: list[dict],
    max_minutes: float,
    dtype_name: str = "float16",
) -> tuple[list[dict], dict, str | None]:
    """Generate for one model. Returns (rows, summary, early_stop_reason)."""
    import gc

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    print(f"\n=== {model_id} ===", flush=True)
    print(
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"gpus {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}",
        flush=True,
    )

    dtype = torch.float16 if dtype_name == "float16" else getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        **{dtype_key: dtype},
    )
    model.eval()

    prompts: list[str] = []
    for item in problems:
        messages = [{"role": "user", "content": item["original_problem"]}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            print(f"chat template failed ({exc}); falling back to raw problem", flush=True)
            prompt = item["original_problem"] + "\n"
        prompts.append(prompt)

    deadline = time.time() + max_minutes * 60.0
    first_device = next(model.parameters()).device
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    early: str | None = None
    decoded: list[tuple[int, str]] = []  # (n_tok, text)

    def _strip_decode(token_ids) -> tuple[int, str]:
        tok_list = list(token_ids)
        if tokenizer.pad_token_id is not None:
            while tok_list and tok_list[-1] == tokenizer.pad_token_id:
                tok_list.pop()
        # Also drop eos for cleaner text (optional); keep for count
        n_tok = len(tok_list)
        text = tokenizer.decode(tok_list, skip_special_tokens=True)
        return n_tok, text

    set_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    batch = {k: v.to(first_device) for k, v in batch.items()}
    prompt_len = batch["input_ids"].shape[1]
    print(f"batched {len(prompts)} prompts; padded length={prompt_len}", flush=True)

    if time.time() > deadline:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return [], {"model": model_id}, "exceeded time budget before generate() (load too slow)"

    t0 = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(**batch, **gen_kwargs)
        wall = time.time() - t0
        print(f"generate wall {wall:.1f}s (batched)", flush=True)
        for i in range(len(problems)):
            new_tokens = outputs[i, prompt_len:].tolist()
            decoded.append(_strip_decode(new_tokens))
        del outputs
    except torch.cuda.OutOfMemoryError:
        print("batched generate OOM; falling back to sequential left-padded singles", flush=True)
        torch.cuda.empty_cache()
        t0 = time.time()
        for i, prompt in enumerate(prompts):
            if time.time() > deadline:
                early = f"stopped sequential at {i}/{len(prompts)} — {max_minutes} min budget"
                print(early, flush=True)
                break
            set_seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            one = tokenizer(prompt, return_tensors="pt")
            one = {k: v.to(first_device) for k, v in one.items()}
            with torch.no_grad():
                out_i = model.generate(**one, **gen_kwargs)
            new_tokens = out_i[0, one["input_ids"].shape[1] :].tolist()
            decoded.append(_strip_decode(new_tokens))
            print(f"  seq {i + 1}/{len(prompts)} toks={decoded[-1][0]}", flush=True)
            del out_i, one
            torch.cuda.empty_cache()
        wall = time.time() - t0
        print(f"generate wall {wall:.1f}s (sequential, {len(decoded)}/{len(problems)})", flush=True)

    if wall > max_minutes * 60.0 and early is None:
        early = f"generate exceeded {max_minutes} min budget ({wall / 60:.1f} min)"

    rows: list[dict] = []
    token_counts: list[int] = []
    n_cap = n_boxed = n_correct = 0
    for i, item in enumerate(problems):
        if i >= len(decoded):
            break
        n_tok, text = decoded[i]
        extracted = extract_last_boxed(text)
        gold = normalize_answer(item["answer"])
        pred = normalize_answer(extracted)
        correct = bool(extracted is not None and pred == gold and gold != "")
        hit_cap = n_tok >= MAX_NEW_TOKENS
        finish_reason = "length" if hit_cap else "stop"
        if hit_cap:
            n_cap += 1
        if extracted is not None:
            n_boxed += 1
        if correct:
            n_correct += 1
        token_counts.append(n_tok)
        rows.append(
            {
                "model": model_id,
                "problem": item["original_problem"],
                "problem_id": item["problem_id"],
                "label": item["label"],
                "gold": item["answer"],
                "tokens": n_tok,
                "finish_reason": finish_reason,
                "extracted": extracted,
                "correct": correct,
                "text": text,
            }
        )
        print(
            f"  [{i}] id={item['problem_id']} toks={n_tok} boxed={extracted!r} "
            f"gold={item['answer']!r} correct={correct} finish={finish_reason}",
            flush=True,
        )

    summary = summarize(token_counts, n_cap, n_boxed, n_correct, wall, len(rows) or len(problems))
    # If we stopped early, n in summarize should reflect completed count
    summary["n"] = len(rows)
    summary["model"] = model_id
    summary["early_stop"] = early
    summary["accuracy_of_10"] = n_correct  # of completed; report notes if <10

    del model, tokenizer, batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary, early



def clear_hf_cache() -> None:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    shutil.rmtree(hub, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=None)
    parser.add_argument("--problems", type=Path, default=None, help="precomputed problems.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--max-minutes", type=float, default=40.0)
    parser.add_argument("--select-only", action="store_true", help="write problems.json and exit")
    parser.add_argument("--clear-cache-between", action="store_true", default=True)
    parser.add_argument("--no-clear-cache-between", action="store_false", dest="clear_cache_between")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    problems = load_problems(args.problems, args.answers)
    (args.out / "problems.json").write_text(json.dumps(problems, indent=2, ensure_ascii=False))
    print(f"selected {len(problems)} problems", flush=True)
    if args.select_only:
        return

    models = args.model or list(DEFAULT_MODELS)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    jsonl_path = args.out / "generations.jsonl"
    # Resume: keep prior rows for other models
    done_models: set[str] = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            all_rows.append(row)
            done_models.add(row["model"])
        print(f"resuming; already have rows for {sorted(done_models)}", flush=True)

    for model_id in models:
        if model_id in done_models:
            print(f"skip {model_id} (already in generations.jsonl)", flush=True)
            continue
        # Rough pre-check: if previous models used most of a shared wall we still
        # enforce per-model --max-minutes inside run_model.
        try:
            rows, summary, early = run_model(model_id, problems, args.max_minutes)
        except Exception as exc:
            summary = {
                "model": model_id,
                "error": f"{type(exc).__name__}: {exc}",
                "early_stop": f"exception: {exc}",
                "n": 0,
                "median_tokens": float("nan"),
                "mean_tokens": float("nan"),
                "p90_tokens": float("nan"),
                "max_tokens": 0,
                "pct_hit_8192_cap": float("nan"),
                "pct_with_boxed": float("nan"),
                "accuracy_of_10": 0,
                "n_correct": 0,
                "wall_seconds": 0,
                "generated_tok_per_s": float("nan"),
            }
            rows = []
            print(f"FAILED {model_id}: {exc}", flush=True)
        summaries.append(summary)
        # Append to jsonl (omit full text? keep it for debugging; report schema wants extracted)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                # Store full text; parent can trim if needed. Schema fields present.
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_rows.extend(rows)
        (args.out / "summaries.json").write_text(json.dumps(summaries, indent=2))
        # Free GPU / disk before next checkpoint
        del rows
        if args.clear_cache_between:
            clear_hf_cache()
        # Try to free CUDA
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    (args.out / "summaries.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
