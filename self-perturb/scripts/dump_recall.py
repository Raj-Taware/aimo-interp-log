"""T-022 Gate 1: does the model answer from memory when a number in the problem changes?

For each pool problem (173 AIME/AMC with gold answers) build an altered copy with one
number changed (first integer >= 2 inside $...$, else in prose: n -> n + 1). Same prompt as
the T-019 states (build_prompt, which ends in "The final answer is $\\boxed{"), so the model
must answer immediately, no reasoning. Per problem and version: greedy decode up to "}"
(<= 16 tokens) and the teacher-forced log-prob of the ORIGINAL gold answer + "}".

Pattern matching = the model gives the original answer to the altered problem.

    python scripts/dump_recall.py --model <hf id> --pool data/pool.jsonl --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from selfperturb.features import MAX_PROMPT_TOKENS, build_prompt, load_model

MATH_INT = re.compile(r"(?<![\d.])(\d{1,4})(?![\d.])")


def alter(problem: str) -> str | None:
    """Change the first integer >= 2 (math spans first, then prose) to n + 1; None if none."""
    for span in re.finditer(r"\$[^$]+\$", problem):
        for m in MATH_INT.finditer(span.group(0)):
            if int(m.group(1)) >= 2:
                start = span.start() + m.start(1)
                return problem[:start] + str(int(m.group(1)) + 1) + problem[span.start() + m.end(1):]
    for m in MATH_INT.finditer(problem):
        if int(m.group(1)) >= 2:
            return problem[:m.start(1)] + str(int(m.group(1)) + 1) + problem[m.end(1):]
    return None


def norm(ans: str) -> str:
    return ans.strip().strip("$").replace(" ", "").lstrip("0") or "0"


@torch.no_grad()
def answer_and_logp(tokenizer, model, problem: str, gold: str) -> tuple[str, float]:
    prompt = build_prompt(tokenizer, problem)
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)["input_ids"].to(model.device)
    out = model.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("}")[0]
    target = tokenizer(gold + "}", add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    logits = model(torch.cat([ids, target], 1), use_cache=False).logits[0, ids.shape[1] - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, target[0][:, None]).sum().item()
    return text, lp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.pool.open(encoding="utf-8")][: args.limit]
    tokenizer, model = load_model(args.model)
    rec = {k: [] for k in ("problems", "altered", "gold", "dec_orig", "dec_alt", "lp_orig", "lp_alt")}
    t0 = time.time()
    for i, r in enumerate(rows):
        alt = alter(r["problem"])
        gold = str(r["answer"]).strip()
        d0, l0 = answer_and_logp(tokenizer, model, r["problem"], gold)
        d1, l1 = answer_and_logp(tokenizer, model, alt, gold) if alt else ("", float("nan"))
        for k, v in zip(rec, (r["problem"], alt or "", gold, d0, d1, l0, l1)):
            rec[k].append(v)
        print(f"[{i}] gold={gold} orig={d0!r} alt={d1!r} lp={l0:.2f}/{l1:.2f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    knows = np.array([norm(a) == norm(g) for a, g in zip(rec["dec_orig"], rec["gold"])])
    pm = np.array([bool(alt) and norm(a) == norm(g) for a, g, alt in zip(rec["dec_alt"], rec["gold"], rec["altered"])])
    np.savez(args.out / f"recall__{args.model.replace('/', '_')}.npz", model_id=args.model, knows=knows, pm=pm,
             **{k: np.array(v, dtype=object if k in ("problems", "altered", "gold", "dec_orig", "dec_alt") else float)
                for k, v in rec.items()})
    print("SUMMARY", json.dumps({"model": args.model, "n": len(rows), "altered": int(sum(map(bool, rec["altered"]))),
                                 "knows": int(knows.sum()), "knows_and_pm": int((knows & pm).sum()),
                                 "knows_not_pm": int((knows & ~pm).sum()), "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
