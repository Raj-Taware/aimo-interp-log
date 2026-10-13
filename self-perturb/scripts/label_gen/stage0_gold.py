"""Stage 0 (T-015 gate): gold answers for the train-main-v2 small-track problems.

train-main-v2 has no answer column, so the gate cannot grade the small models
without one. ``--model`` (default gpt-oss-120b; DeepSeek-R1-0528-Qwen3-8B on Kaggle)
solves each problem ``--samples`` times; gold is the
majority normalized answer if at least ``--min-agree`` samples give it, else
the problem is dropped from the gate (reported, never silently).

    python scripts/label_gen/stage0_gold.py --backend vllm --dtype auto \
        --trainv2 <train-main-v2.parquet> --out <run_dir>/gate

Raw samples -> <out>/stage0_generations.jsonl (append-only, key problem_id|sample)
Gate pool   -> <out>/problems.jsonl (problem_id tv2-<i>, problem, answer, n_agree)
Report      -> <out>/stage0_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from common import (JsonlStore, Request, add_backend_args, chat_ids, extract_answer, free_backend,
                    load_tokenizer, log, make_backend, normalize_answer, resolve_sampling, runtime_env,
                    solve_prompt, stable_seed, write_jsonl)

WRITER = "openai/gpt-oss-120b"


def gate_problems(trainv2: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(trainv2)
    small = df[df.model_id != WRITER]
    texts = sorted(set(small.problem))  # stable ids across runs
    return [{"problem_id": f"tv2-{i}", "problem": t, "source": "aimo-interp/train-main-v2"}
            for i, t in enumerate(texts)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainv2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--min-agree", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort", default="high", help="gpt-oss only")
    parser.add_argument("--model", default=WRITER)
    add_backend_args(parser)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    problems = gate_problems(args.trainv2)
    store = JsonlStore(args.out / "stage0_generations.jsonl")
    tokenizer = load_tokenizer(args.model)
    kwargs = {"reasoning_effort": args.reasoning_effort} if "gpt-oss" in args.model else {}
    requests = [Request(f"{p['problem_id']}|{s}", chat_ids(tokenizer, solve_prompt(p["problem"]), **kwargs),
                        stable_seed(f"stage0|{p['problem_id']}|{s}"), (p, s))
                for p in problems for s in range(args.samples)]
    pending = [r for r in requests if r.key not in store]
    log(f"stage0: {len(requests) - len(pending)} done, {len(pending)} pending")
    if pending:
        sampling = resolve_sampling(args.model)
        backend = make_backend(args, args.model, tokenizer, sampling, args.max_new_tokens)
        env = runtime_env()
        try:
            for batch, results in backend.generate(pending):
                recs = []
                for req, res in zip(batch, results):
                    p, s = req.meta
                    extracted, rule = extract_answer(res.text)
                    recs.append({
                        "key": req.key, "stage": 0, "model": args.model, "problem_id": p["problem_id"],
                        "sample": s, "seed": req.seed, "extracted": extracted, "extraction_rule": rule,
                        "normalized": normalize_answer(extracted), "tokens": res.n_tokens,
                        "finish_reason": res.finish_reason, "backend": backend.name, "sampling": sampling,
                        "reasoning_effort": kwargs.get("reasoning_effort"), "env": env,
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "text": res.text,
                    })
                store.append(recs)
        finally:
            free_backend(backend)

    by_pid: dict[str, list[str]] = {}
    for rec in store.values():
        by_pid.setdefault(rec["problem_id"], []).append(rec["normalized"])
    kept, dropped = [], []
    for p in problems:
        votes = Counter(a for a in by_pid.get(p["problem_id"], []) if a)
        top, n = votes.most_common(1)[0] if votes else ("", 0)
        row = {**p, "answer": top, "n_agree": n, "n_samples": len(by_pid.get(p["problem_id"], []))}
        (kept if n >= args.min_agree else dropped).append(row)
    write_jsonl(args.out / "problems.jsonl", kept)
    summary = {"n_problems": len(problems), "n_kept": len(kept), "min_agree": args.min_agree,
               "dropped": [{"problem_id": r["problem_id"], "votes": r["n_agree"], "top": r["answer"]}
                           for r in dropped]}
    (args.out / "stage0_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"stage0: gold for {len(kept)}/{len(problems)} problems; dropped {[r['problem_id'] for r in dropped]}")


if __name__ == "__main__":
    main()
