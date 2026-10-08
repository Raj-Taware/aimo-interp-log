"""Stage 3: each kept (model, problem) solves every variant ``--samples`` times -> labels.

    python scripts/label_gen/stage3_solve_variants.py --backend vllm --model Qwen/Qwen3.5-4B \
        --problems ... --stage1 <run_dir>/stage1_summary.jsonl \
        --variants <run_dir>/stage2_variants.jsonl --out <run_dir>

Same solve protocol as stage 1 (T-009). Per pair: perm_acc per type,
max_drop = base_acc - min(perm_acc); robust iff max_drop <= 0.10, spurious iff
>= 0.25, else null (organizers' rule, aimo-interp/train-main-v2 README).

Raw samples -> <out>/stage3_generations.jsonl (append-only, key model|problem_id|type|sample)
Labels      -> <out>/labels.jsonl (+ labels.parquet when pandas is present), train-main-v2
               schema (model_id, problem, is_robust, max_drop) + base_acc, perm_acc_<type>, ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import (PERTURBATION_TYPES, ROBUST_MAX_DROP, SPURIOUS_MAX_DROP, JsonlStore, Request,
                    add_backend_args, answers_match, chat_ids, extract_answer, free_backend, is_looping,
                    load_problems, load_tokenizer, log, make_backend, read_jsonl, resolve_sampling,
                    runtime_env, solve_prompt, stable_seed, write_jsonl)


def label(max_drop: float) -> bool | None:
    if max_drop <= ROBUST_MAX_DROP + 1e-9:
        return True
    if max_drop >= SPURIOUS_MAX_DROP - 1e-9:
        return False
    return None


def build_labels(store, stage1_rows, variants, problems, types, samples, source) -> list[dict]:
    have = {(v["problem_id"], v["type"]) for v in variants}
    rows = []
    for s1 in stage1_rows:
        if not s1["kept"]:
            continue
        model, pid = s1["model"], s1["problem_id"]
        perm, n_used = {}, {}
        for t in types:
            if (pid, t) not in have:
                continue
            recs = [store.records[k] for k in (f"{model}|{pid}|{t}|{s}" for s in range(samples)) if k in store]
            if len(recs) < samples:
                continue
            perm[t] = sum(r["correct"] for r in recs) / len(recs)
            n_used[t] = len(recs)
        row = {"model_id": model, "problem": problems[pid]["problem"], "problem_id": pid,
               "source": source, "base_acc": s1["base_acc"], "n_base": s1["n_samples"],
               "n_types": len(perm), "complete": len(perm) == len(types)}
        for t in types:
            row[f"perm_acc_{t}"] = perm.get(t)
        if perm:
            row["max_drop"] = round(s1["base_acc"] - min(perm.values()), 4)
            row["worst_type"] = min(perm, key=perm.get)
            row["is_robust"] = label(row["max_drop"])
        else:
            row["max_drop"], row["worst_type"], row["is_robust"] = None, None, None
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=12288)
    parser.add_argument("--types", nargs="+", default=list(PERTURBATION_TYPES))
    parser.add_argument("--source", default="self-labelled (T-014 pipeline)")
    add_backend_args(parser)
    args = parser.parse_args()
    if args.backend == "vllm" and len(args.model) > 1:
        parser.error("vllm: run one --model per process (engine memory is not reliably freed)")

    problems = {p["problem_id"]: p for p in load_problems(args.problems)}
    stage1_rows = read_jsonl(args.stage1)
    variants = [v for v in read_jsonl(args.variants) if v["type"] in args.types]
    by_problem: dict[str, list[dict]] = {}
    for v in variants:
        by_problem.setdefault(v["problem_id"], []).append(v)
    store = JsonlStore(args.out / "stage3_generations.jsonl")
    log(f"stage3: {len(variants)} variants; {len(store.records)} samples on disk "
        f"(bad lines {store.bad_lines}, duplicates {store.duplicate_keys})")

    for model_id in args.model:
        kept = [r["problem_id"] for r in stage1_rows if r["model"] == model_id and r["kept"]]
        todo = [(v, s) for pid in kept for v in by_problem.get(pid, []) for s in range(args.samples)
                if f"{model_id}|{pid}|{v['type']}|{s}" not in store]
        total = sum(len(by_problem.get(pid, [])) for pid in kept) * args.samples
        log(f"stage3 {model_id}: {len(kept)} kept problems, {total - len(todo)} done, {len(todo)} pending")
        if not todo:
            continue
        tokenizer = load_tokenizer(model_id)
        sampling = resolve_sampling(model_id)
        log(f"sampling {sampling}")
        cache = {}
        requests = []
        for v, s in todo:
            ck = (v["problem_id"], v["type"])
            if ck not in cache:
                cache[ck] = chat_ids(tokenizer, solve_prompt(v["variant"]))
            key = f"{model_id}|{v['problem_id']}|{v['type']}|{s}"
            requests.append(Request(key, cache[ck], stable_seed(f"stage3|{key}"), (v, s)))
        backend = make_backend(args, model_id, tokenizer, sampling, args.max_new_tokens)
        env = runtime_env()
        for batch, results in backend.generate(requests):
            recs = []
            for req, res in zip(batch, results):
                v, sample = req.meta
                extracted, rule = extract_answer(res.text)
                recs.append({
                    "key": req.key, "stage": 3, "model": model_id, "problem_id": v["problem_id"],
                    "type": v["type"], "variant_attempt": v["attempt"], "sample": sample, "seed": req.seed,
                    "answer": v["answer"], "extracted": extracted, "extraction_rule": rule,
                    "correct": answers_match(extracted, v["answer"]), "tokens": res.n_tokens,
                    "prompt_tokens": len(req.prompt_ids), "finish_reason": res.finish_reason,
                    "looping": is_looping(res.text), "batch_seconds": res.seconds, "backend": backend.name,
                    "sampling": sampling, "max_new_tokens": args.max_new_tokens, "env": env,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "text": res.text,
                })
            store.append(recs)
            log(f"stage3 {model_id}: +{len(recs)} (tokens {[r['tokens'] for r in recs]}, "
                f"correct {[int(r['correct']) for r in recs]}) -> {len(store.records)} on disk")
        free_backend(backend)

    rows = build_labels(store, stage1_rows, variants, problems, args.types, args.samples, args.source)
    write_jsonl(args.out / "labels.jsonl", rows)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(args.out / "labels.parquet", index=False)
    except Exception as exc:  # pandas / pyarrow absent in a lean vLLM env
        log(f"labels.parquet skipped ({exc})")
    counts = {"robust": sum(r["is_robust"] is True for r in rows),
              "spurious": sum(r["is_robust"] is False for r in rows),
              "null": sum(r["is_robust"] is None for r in rows)}
    log(f"stage3 done: {len(rows)} labelled pairs {counts}")


if __name__ == "__main__":
    main()
