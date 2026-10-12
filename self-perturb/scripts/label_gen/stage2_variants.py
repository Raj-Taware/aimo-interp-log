"""Stage 2: a writer model produces one answer-preserving variant per type.

    # H100 (vLLM, the organizers' permutation source):
    python scripts/label_gen/stage2_variants.py --backend vllm --writer openai/gpt-oss-120b \
        --reasoning-effort medium --problems ... --stage1 <run_dir>/stage1_summary.jsonl --out <run_dir>
    # Kaggle dry run: --backend hf --writer Skywork/Skywork-OR1-Math-7B (stand-in; gpt-oss-120b does not fit a T4)

Problems: every problem kept (base_acc >= 0.5) by at least one model in stage 1.
Types: domain, distract, rephrase, rename, typos. Each (problem, type) gets up to
``--attempts`` generations; the first one that passes validation is the variant.

Raw attempts -> <out>/stage2_raw.jsonl       (append-only, key problem_id|type|attempt)
Variants     -> <out>/stage2_variants.jsonl  (derived: first valid attempt per problem x type)
Spot check   -> <out>/stage2_spotcheck.jsonl + .md (20 random variants, seed 0, for Claude)
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

from common import (PERTURBATION_TYPES, JsonlStore, Request, add_backend_args, chat_ids, free_backend,
                    load_problems, load_tokenizer, log, make_backend, normalize_answer, read_jsonl,
                    resolve_sampling, runtime_env, stable_seed, write_jsonl)

TYPE_DEFINITIONS = {
    "domain": ("Re-set the problem in a different domain or real-world context (for example, turn an "
               "abstract number, algebra or geometry question into a story about objects, people, money, "
               "games or physics, or turn a story into abstract mathematics) while keeping the underlying "
               "mathematical structure and every quantity identical."),
    "distract": ("Add one or two sentences of irrelevant but plausible-sounding information (an unrelated "
                 "fact, or an extra number that plays no role in the solution). Keep all of the original "
                 "information unchanged."),
    "rephrase": ("Reword the problem with different sentence structure and vocabulary while keeping its "
                 "exact mathematical meaning. Do not add or remove any information."),
    "rename": ("Consistently rename variables, functions, points, sets, people and objects (for example "
               "x -> u, triangle ABC -> triangle PQR, Alice -> Priya). Change nothing else."),
    "typos": ("Introduce 3 to 6 realistic typing mistakes (swapped, dropped or doubled letters) in ordinary "
              "English words only. Do not touch any number, variable or LaTeX math."),
}

WRITER_PROMPT = """You are helping build a robustness benchmark for competition math problems. Rewrite the problem below according to the perturbation type "{ptype}".

Perturbation type "{ptype}": {definition}

Hard requirements:
- The rewritten problem must have exactly the same final answer as the original, which is {answer}. Do not change any quantity, condition or relationship that the answer depends on, and keep the question asking for the same quantity.
- Do not solve the problem, do not add hints, and never state the answer.
- Keep any LaTeX valid, and keep every math delimiter ($ signs) exactly where the original has them.
- Output only the rewritten problem, enclosed in <variant> and </variant> tags, with nothing after the closing tag.

Original problem:
<original>
{problem}
</original>"""

_VARIANT = re.compile(r"<variant>(.*?)</variant>", re.DOTALL)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_variant(text: str, original: str, answer: str, finish: str) -> tuple[str | None, str]:
    matches = _VARIANT.findall(text)
    if not matches:
        return None, "no_tags" if finish != "length" else "hit_cap"
    variant = matches[-1].strip()
    if len(variant) < 20:
        return None, "too_short"
    if variant == original.strip():
        return None, "unchanged"
    if "\\boxed" in variant:
        return None, "contains_boxed"
    ans = normalize_answer(answer)
    if ans and ans not in set(_NUMBER.findall(original)) and ans in set(_NUMBER.findall(variant)):
        return None, "leaks_answer"
    return variant, "ok"


def missing_numbers(original: str, variant: str) -> list[str]:
    have = set(_NUMBER.findall(variant))
    return sorted({n for n in _NUMBER.findall(original) if n not in have})


def derive(store: JsonlStore, problems: dict, types, attempts: int):
    """First valid attempt per (problem, type); plus status of every pair."""
    variants, status = [], {}
    for pid in problems:
        for ptype in types:
            recs = sorted((store.records[k] for k in (f"{pid}|{ptype}|{a}" for a in range(attempts))
                           if k in store), key=lambda r: r["attempt"])
            valid = [r for r in recs if r["valid"]]
            if valid:
                r = valid[0]
                variants.append({k: r[k] for k in ("problem_id", "type", "attempt", "variant", "answer",
                                                   "writer", "missing_numbers", "writer_tokens")}
                                | {"original": problems[pid]["problem"]})
                status[(pid, ptype)] = "ok"
            else:
                status[(pid, ptype)] = "failed" if len(recs) >= attempts else "pending"
    return variants, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True, help="stage1_summary.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--writer", default="openai/gpt-oss-120b")
    parser.add_argument("--types", nargs="+", default=list(PERTURBATION_TYPES))
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", default=None, help="gpt-oss chat-template kwarg (low/medium/high)")
    parser.add_argument("--spotcheck-n", type=int, default=20)
    add_backend_args(parser)
    args = parser.parse_args()

    kept_ids = {r["problem_id"] for r in read_jsonl(args.stage1) if r["kept"]}
    problems = {p["problem_id"]: p for p in load_problems(args.problems) if p["problem_id"] in kept_ids}
    store = JsonlStore(args.out / "stage2_raw.jsonl")
    log(f"stage2: {len(problems)} kept problems x {len(args.types)} types; {len(store.records)} attempts on disk "
        f"(bad lines {store.bad_lines}, duplicates {store.duplicate_keys})")

    backend = tokenizer = None
    template_kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    sampling = None
    for attempt in range(args.attempts):
        _, status = derive(store, problems, args.types, args.attempts)
        todo = [(pid, t) for (pid, t), s in status.items()
                if s == "pending" and f"{pid}|{t}|{attempt}" not in store
                and all(f"{pid}|{t}|{a}" in store for a in range(attempt))]
        log(f"stage2 attempt {attempt}: {len(todo)} pending")
        if not todo:
            continue
        if backend is None:
            tokenizer = load_tokenizer(args.writer)
            sampling = resolve_sampling(args.writer, {"temperature": args.temperature, "top_p": args.top_p})
            log(f"writer sampling {sampling} template {template_kwargs}")
            backend = make_backend(args, args.writer, tokenizer, sampling, args.max_new_tokens)
        env = runtime_env()
        requests = []
        for pid, ptype in todo:
            p = problems[pid]
            prompt = WRITER_PROMPT.format(ptype=ptype, definition=TYPE_DEFINITIONS[ptype],
                                          answer=normalize_answer(p["answer"]), problem=p["problem"])
            requests.append(Request(f"{pid}|{ptype}|{attempt}", chat_ids(tokenizer, prompt, **template_kwargs),
                                    stable_seed(f"stage2|{args.writer}|{pid}|{ptype}|{attempt}"), (p, ptype)))
        for batch, results in backend.generate(requests):
            recs = []
            for req, res in zip(batch, results):
                p, ptype = req.meta
                variant, reason = parse_variant(res.text, p["problem"], p["answer"], res.finish_reason)
                recs.append({
                    "key": req.key, "stage": 2, "problem_id": p["problem_id"], "type": ptype,
                    "attempt": attempt, "seed": req.seed, "writer": args.writer, "answer": p["answer"],
                    "valid": variant is not None, "reason": reason, "variant": variant,
                    "missing_numbers": missing_numbers(p["problem"], variant) if variant else None,
                    "writer_tokens": res.n_tokens, "prompt_tokens": len(req.prompt_ids),
                    "finish_reason": res.finish_reason, "batch_seconds": res.seconds,
                    "backend": backend.name, "sampling": sampling, "template_kwargs": template_kwargs,
                    "max_new_tokens": args.max_new_tokens, "env": env,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "text": res.text,
                })
            store.append(recs)
            log(f"stage2: +{len(recs)} (valid {[int(r['valid']) for r in recs]}, "
                f"reasons {[r['reason'] for r in recs]}) -> {len(store.records)} on disk")
    if backend is not None:
        free_backend(backend)

    variants, status = derive(store, problems, args.types, args.attempts)
    write_jsonl(args.out / "stage2_variants.jsonl", variants)
    rng = random.Random(0)
    spot = rng.sample(variants, min(args.spotcheck_n, len(variants)))
    write_jsonl(args.out / "stage2_spotcheck.jsonl", spot)
    lines = ["# Stage 2 spot check (random sample, seed 0)", "",
             "For each: does the variant keep the answer? (yes / no / unsure)", ""]
    for i, v in enumerate(spot, 1):
        lines += [f"## {i}. {v['problem_id']} / {v['type']} (answer {v['answer']}, writer {v['writer']})", "",
                  "**Original**", "", v["original"], "", "**Variant**", "", v["variant"], "",
                  f"missing numbers: {v['missing_numbers']}", "", "verdict: ", ""]
    (args.out / "stage2_spotcheck.md").write_text("\n".join(lines), encoding="utf-8")
    counts = {s: sum(1 for x in status.values() if x == s) for s in ("ok", "failed", "pending")}
    log(f"stage2 done: {counts}")


if __name__ == "__main__":
    main()
