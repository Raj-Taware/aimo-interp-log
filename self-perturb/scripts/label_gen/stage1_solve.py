"""Stage 1: each small model solves each original problem ``--samples`` times.

    # H100 (vLLM), one model per process:
    python scripts/label_gen/stage1_solve.py --backend vllm --model Qwen/Qwen3.5-4B \
        --problems results/grok/T-014/problems.jsonl --out <run_dir>
    # Kaggle T4 dry run (transformers generate):
    python scripts/label_gen/stage1_solve.py --backend hf --model Skywork/Skywork-OR1-Math-7B \
        --problems ... --out <run_dir> --samples 2 --max-new-tokens 4096 --batch-size 6

Raw samples  -> <out>/stage1_generations.jsonl  (append-only, key model|problem_id|sample)
Per pair     -> <out>/stage1_summary.jsonl      (derived: base_acc, tokens, kept)
A pair is kept for stages 2-3 iff base_acc >= --min-base-acc (0.5).
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from common import (JsonlStore, Request, add_backend_args, answers_match, chat_ids, extract_answer,
                    free_backend, is_looping, load_problems, load_tokenizer, log, make_backend,
                    resolve_sampling, runtime_env, solve_prompt, stable_seed, write_jsonl)


def summarize(store: JsonlStore, problems: list[dict], min_base_acc: float, samples: int) -> list[dict]:
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for rec in store.values():
        by_pair.setdefault((rec["model"], rec["problem_id"]), []).append(rec)
    order = {p["problem_id"]: i for i, p in enumerate(problems)}
    rows = []
    for (model, pid), recs in sorted(by_pair.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 1e9))):
        if pid not in order:
            continue
        recs = sorted(recs, key=lambda r: r["sample"])
        n = len(recs)
        base_acc = sum(r["correct"] for r in recs) / n
        rows.append({
            "model": model, "problem_id": pid, "n_samples": n, "complete": n >= samples,
            "base_acc": base_acc, "kept": bool(n >= samples and base_acc >= min_base_acc),
            "tokens": [r["tokens"] for r in recs],
            "median_tokens": statistics.median(r["tokens"] for r in recs),
            "n_hit_cap": sum(r["finish_reason"] == "length" for r in recs),
            "extracted": [r["extracted"] for r in recs],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12288)
    parser.add_argument("--min-base-acc", type=float, default=0.5)
    parser.add_argument("--problem-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    add_backend_args(parser)
    args = parser.parse_args()
    if args.backend == "vllm" and len(args.model) > 1:
        parser.error("vllm: run one --model per process (engine memory is not reliably freed)")

    problems = load_problems(args.problems, args.problem_id, args.limit)
    store = JsonlStore(args.out / "stage1_generations.jsonl")
    log(f"stage1: {len(problems)} problems, {len(store.records)} samples on disk "
        f"(bad lines {store.bad_lines}, duplicates {store.duplicate_keys})")

    for model_id in args.model:
        todo = [(p, s) for p in problems for s in range(args.samples)
                if f"{model_id}|{p['problem_id']}|{s}" not in store]
        log(f"stage1 {model_id}: {args.samples * len(problems) - len(todo)} done, {len(todo)} pending")
        if not todo:
            continue
        tokenizer = load_tokenizer(model_id)
        sampling = resolve_sampling(model_id)
        log(f"sampling {sampling}")
        prompt_cache = {p["problem_id"]: chat_ids(tokenizer, solve_prompt(p["problem"])) for p, _ in todo}
        requests = [Request(f"{model_id}|{p['problem_id']}|{s}", prompt_cache[p["problem_id"]],
                            stable_seed(f"stage1|{model_id}|{p['problem_id']}|{s}"), (p, s)) for p, s in todo]
        backend = make_backend(args, model_id, tokenizer, sampling, args.max_new_tokens)
        env = runtime_env()
        for batch, results in backend.generate(requests):
            recs = []
            for req, res in zip(batch, results):
                problem, sample = req.meta
                extracted, rule = extract_answer(res.text)
                recs.append({
                    "key": req.key, "stage": 1, "model": model_id, "problem_id": problem["problem_id"],
                    "sample": sample, "seed": req.seed, "answer": problem["answer"],
                    "extracted": extracted, "extraction_rule": rule,
                    "correct": answers_match(extracted, problem["answer"]),
                    "tokens": res.n_tokens, "prompt_tokens": len(req.prompt_ids),
                    "finish_reason": res.finish_reason, "looping": is_looping(res.text),
                    "batch_seconds": res.seconds, "backend": backend.name, "sampling": sampling,
                    "max_new_tokens": args.max_new_tokens, "env": env,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "text": res.text,
                })
            store.append(recs)
            log(f"stage1 {model_id}: +{len(recs)} (tokens {[r['tokens'] for r in recs]}, "
                f"correct {[int(r['correct']) for r in recs]}) -> {len(store.records)} on disk")
        free_backend(backend)

    rows = summarize(store, problems, args.min_base_acc, args.samples)
    write_jsonl(args.out / "stage1_summary.jsonl", rows)
    kept = sum(r["kept"] for r in rows)
    log(f"stage1 done: {len(rows)} pairs, kept {kept} (base_acc >= {args.min_base_acc})")


if __name__ == "__main__":
    main()
