"""T-009 generation smoke v2: fixed protocol after T-007 review.

Changes vs T-007:
  - tokenizer via selfperturb.features.load_tokenizer (repairs DeepSeek)
  - user turn = problem + boxed instruction
  - sampling from model.generation_config when set; do NOT pass eos_token_id
  - extraction: last \\boxed{}; else last **...**; else last number after "answer is"
  - batch size 2 (Olmo: 1); 8192 max new tokens; 30-min cap per model
  - problems: first 6 of T-007 problems.json

    python scripts/gen_smoke_v2.py \\
        --problems results/grok/T-007/problems.json \\
        --out results/grok/T-009
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

DEFAULT_MODELS = (
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
)
MAX_NEW_TOKENS = 8192
SEED = 0
INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
OLMO_ID = "allenai/Olmo-3-7B-Think"


def extract_last_boxed(text: str) -> str | None:
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


def extract_last_bold(text: str) -> str | None:
    matches = list(re.finditer(r"\*\*([^*]+)\*\*", text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_answer_is_number(text: str) -> str | None:
    matches = list(re.finditer(r"answer is[^\d\-]*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE))
    if not matches:
        return None
    return matches[-1].group(1)


def extract_answer(text: str) -> tuple[str | None, str]:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed, "boxed"
    bold = extract_last_bold(text)
    if bold is not None:
        return bold, "bold"
    num = extract_answer_is_number(text)
    if num is not None:
        return num, "answer_is"
    return None, "none"


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    return text.replace(" ", "").replace("$", "")


def is_looping(text: str) -> bool:
    """Last 4000 chars repeat a 40-char chunk at least 4 times."""
    tail = text[-4000:] if len(text) > 4000 else text
    if len(tail) < 160:
        return False
    # sliding 40-char chunks; count max frequency
    from collections import Counter

    chunks = [tail[i : i + 40] for i in range(0, max(len(tail) - 39, 1))]
    if not chunks:
        return False
    counts = Counter(chunks)
    return counts.most_common(1)[0][1] >= 4


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


def sampling_from_config(model) -> dict:
    """temperature / top_p / top_k from generation_config when set, else defaults."""
    cfg = model.generation_config
    def _get(name, default):
        val = getattr(cfg, name, None)
        if val is None:
            return default
        # HF sometimes uses unset sentinels
        if isinstance(val, float) and val <= 0 and name in ("temperature", "top_p"):
            return default
        if isinstance(val, int) and val <= 0 and name == "top_k":
            return default
        return val

    temperature = _get("temperature", 0.6)
    top_p = _get("top_p", 0.95)
    top_k = getattr(cfg, "top_k", None)
    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        top_k = None
    used = {"temperature": float(temperature), "top_p": float(top_p), "top_k": top_k}
    return used


def run_model(
    model_id: str,
    problems: list[dict],
    max_minutes: float,
    batch_size: int,
) -> tuple[list[dict], dict]:
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "src"))
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, set_seed
    from selfperturb.features import load_tokenizer

    print(f"\n=== {model_id} ===", flush=True)
    print(
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"gpus {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}",
        flush=True,
    )

    tokenizer = load_tokenizer(model_id, False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        **{dtype_key: torch.float16},
    )
    model.eval()

    used = sampling_from_config(model)
    print(f"sampling used: {used}", flush=True)
    print(f"generation_config.eos_token_id={getattr(model.generation_config, 'eos_token_id', None)}", flush=True)
    print(f"tokenizer.eos_token_id={tokenizer.eos_token_id}", flush=True)

    prompts: list[str] = []
    for item in problems:
        user = item["original_problem"] + INSTRUCTION
        messages = [{"role": "user", "content": user}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            print(f"chat template failed ({exc}); raw fallback", flush=True)
            prompt = user + "\n"
        prompts.append(prompt)

    deadline = time.time() + max_minutes * 60.0
    first_device = next(model.parameters()).device

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=used["temperature"],
        top_p=used["top_p"],
        pad_token_id=tokenizer.pad_token_id,
        # deliberately omit eos_token_id — let generation_config supply stop tokens
    )
    if used["top_k"] is not None:
        gen_kwargs["top_k"] = used["top_k"]

    set_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    rows: list[dict] = []
    token_counts: list[int] = []
    n_cap = n_stop = n_correct = n_loop = 0
    rule_counts = {"boxed": 0, "bold": 0, "answer_is": 0, "none": 0}
    early: str | None = None
    t_wall0 = time.time()
    total_new_tokens = 0

    def _decode_new(token_ids) -> tuple[int, str, str]:
        tok_list = list(token_ids)
        # finish reason heuristic later via length
        if tokenizer.pad_token_id is not None:
            while tok_list and tok_list[-1] == tokenizer.pad_token_id:
                tok_list.pop()
        n_tok = len(tok_list)
        text = tokenizer.decode(tok_list, skip_special_tokens=True)
        return n_tok, text

    i = 0
    while i < len(problems):
        if time.time() > deadline:
            early = f"stopped at {i}/{len(problems)} — {max_minutes} min budget"
            print(early, flush=True)
            break
        bs = min(batch_size, len(problems) - i)
        batch_prompts = prompts[i : i + bs]
        batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)
        batch = {k: v.to(first_device) for k, v in batch.items()}
        prompt_len = batch["input_ids"].shape[1]
        print(f"batch {i}:{i+bs} padded_len={prompt_len}", flush=True)
        try:
            with torch.no_grad():
                outputs = model.generate(**batch, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            print("OOM; retrying with batch_size=1 for this chunk", flush=True)
            torch.cuda.empty_cache()
            if bs == 1:
                early = f"OOM at problem {i}"
                break
            # shrink and retry without advancing
            batch_size = 1
            continue

        for j in range(bs):
            new_ids = outputs[j, prompt_len:].tolist()
            n_tok, text = _decode_new(new_ids)
            finish = "length" if n_tok >= MAX_NEW_TOKENS else "stop"
            extracted, rule = extract_answer(text)
            correct = normalize_answer(extracted) == normalize_answer(problems[i + j]["answer"])
            looping = is_looping(text)
            token_counts.append(n_tok)
            total_new_tokens += n_tok
            if finish == "length":
                n_cap += 1
            else:
                n_stop += 1
            if correct:
                n_correct += 1
            if looping:
                n_loop += 1
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
            rows.append(
                {
                    "model": model_id,
                    "problem_id": problems[i + j]["problem_id"],
                    "original_problem": problems[i + j]["original_problem"],
                    "answer": problems[i + j]["answer"],
                    "label": problems[i + j].get("label"),
                    "tokens": n_tok,
                    "finish_reason": finish,
                    "extracted": extracted,
                    "extraction_rule": rule,
                    "correct": bool(correct),
                    "looping": bool(looping),
                    "text": text,
                    "sampling": used,
                }
            )
            print(
                f"  [{i+j}] tokens={n_tok} finish={finish} rule={rule} "
                f"correct={correct} looping={looping} extracted={extracted!r}",
                flush=True,
            )
        del outputs, batch
        torch.cuda.empty_cache()
        i += bs

    wall = time.time() - t_wall0
    n = len(rows)
    ordered = sorted(token_counts)
    summary = {
        "model": model_id,
        "sampling_used": used,
        "n": n,
        "median_tokens": percentile(ordered, 50) if ordered else float("nan"),
        "pct_hit_cap": round(100.0 * n_cap / n, 1) if n else float("nan"),
        "pct_stopped": round(100.0 * n_stop / n, 1) if n else float("nan"),
        "correct_of_6": n_correct,
        "extraction_rule_counts": rule_counts,
        "looping_count": n_loop,
        "tok_per_s": round(total_new_tokens / wall, 2) if wall > 0 else float("nan"),
        "wall_seconds": round(wall, 1),
        "early_stop": early,
        "batch_size": batch_size,
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # clear HF cache between models (~50GB)
    shutil.rmtree(Path.home() / ".cache" / "huggingface" / "hub", ignore_errors=True)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--max-minutes", type=float, default=30.0)
    parser.add_argument("--limit-problems", type=int, default=6)
    arguments = parser.parse_args()

    problems = json.loads(arguments.problems.read_text(encoding="utf-8"))[: arguments.limit_problems]
    models = arguments.model or list(DEFAULT_MODELS)
    arguments.out.mkdir(parents=True, exist_ok=True)
    out_jsonl = arguments.out / "generations.jsonl"
    # resume: skip models already fully present
    done_models = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_models.add(json.loads(line)["model"])
        # only skip if we have limit_problems rows
        from collections import Counter
        counts = Counter()
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[json.loads(line)["model"]] += 1
        done_models = {m for m, c in counts.items() if c >= arguments.limit_problems}

    summaries = []
    sum_path = arguments.out / "summaries.json"
    if sum_path.exists():
        summaries = json.loads(sum_path.read_text(encoding="utf-8"))

    with out_jsonl.open("a", encoding="utf-8") as fh:
        for model_id in models:
            if model_id in done_models:
                print(f"skip {model_id} (already have {arguments.limit_problems} rows)", flush=True)
                continue
            bs = 1 if model_id == OLMO_ID else 2
            rows, summary = run_model(model_id, problems, arguments.max_minutes, bs)
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            summaries.append(summary)
            sum_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2), flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
