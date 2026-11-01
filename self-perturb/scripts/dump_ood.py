"""T-027 pilot: an out-of-distribution set whose robustness labels are cheap to make.

Every gate so far failed its positive control, partly because labels are scarce: the official
set is DeepSeek only and self-labelling long MATH problems cost 4 h per 40 (T-020). Multi-step
arithmetic word problems are far from the training domain, reason in hundreds of tokens, and
have a known integer answer, so each model can be labelled directly.

Per problem and model: greedy reasoned answer to the original and to each changed
answer-preserving perturbation. robust = original correct and every changed variant correct.
Also saves the v2 white-box feature row, so drift can be scored on this set if the pilot passes.

Pilot gate (fixed before the run), per model: >= 20% robust, >= 20% non-robust, truncated
answers < 20%. A model that fails gives no contrast; if none pass, the OOD set is dropped.

    python scripts/dump_ood.py --model <hf id> --rows ignored.jsonl --out <dir> [--limit 3]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from selfperturb.features import case_features, load_model
from selfperturb.perturbations import ANSWER_PRESERVING, perturb_all

N_PROBLEMS = 30
MAX_NEW_TOKENS = 1024
INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
NAMES = ("Maya", "Omar", "Priya", "Lucas", "Hana", "Diego")


def problem_set(n: int = N_PROBLEMS, seed: int = 0) -> list[dict]:
    """Integer-answer word problems, three size tiers so some sit near each model's limit."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tier = i % 3
        hi = (30, 300, 3000)[tier]
        a, b = rng.randint(12, hi), rng.randint(12, 99)
        c, d, r = rng.randint(2, a * b // 3), rng.randint(3, 17), rng.randint(1, 2)
        name, kind = rng.choice(NAMES), i % 4
        if kind == 0:
            total = a * b - c
            q = (total - total % d) // d
            text = (f"{name} packs {a} boxes with {b} pens each and sells {c} pens. The rest are shared equally "
                    f"among {d} classrooms, as many as possible. How many pens does each classroom get?")
        elif kind == 1:
            q = a * b - c + d * r
            text = (f"A depot starts with {a} crates of {b} cans. {name} removes {c} cans, then adds {d} trays "
                    f"of {r} cans each. How many cans are in the depot now?")
        elif kind == 2:
            q = (a * b + c) * d
            text = (f"Each of {a} shelves holds {b} books, and {name} adds {c} more books. Every book has {d} "
                    f"stickers. How many stickers are there in total?")
        else:
            q = a * b - c * d
            text = (f"{name} earns {b} coins a day for {a} days, then buys {d} tickets costing {c} coins each. "
                    f"How many coins are left? The result may be negative.")
        rows.append({"problem": text, "answer": str(q), "tier": tier})
    return rows


def boxed(text: str) -> str | None:
    found = re.findall(r"\\boxed\{([^{}]*)\}", text)  # shortcut: flat boxes only, answers here are integers
    return found[-1].replace(",", "").replace("$", "").strip() if found else None


@torch.no_grad()
def solve(tokenizer, model, texts: list[str]) -> list[tuple[str | None, bool]]:
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": t + INSTRUCTION}], tokenize=False,
                                             add_generation_prompt=True) for t in texts]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
    out = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new = out[:, batch["input_ids"].shape[1]:]
    done = (new == tokenizer.eos_token_id).any(1).tolist()
    return list(zip(map(boxed, tokenizer.batch_decode(new, skip_special_tokens=True)), done))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, help="unused, the set is generated")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = problem_set()[: args.limit]
    tokenizer, model = load_model(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    robust, base_ok, truncated, features = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        variants = perturb_all(r["problem"])
        texts = [r["problem"]] + [variants[n] for n in ANSWER_PRESERVING if variants[n] != r["problem"]]
        answers = solve(tokenizer, model, texts)
        correct = [a == r["answer"] for a, _ in answers]
        robust.append(all(correct)); base_ok.append(correct[0])
        truncated.append(sum(not d for _, d in answers) / len(answers))
        features.append(case_features(tokenizer, model, r["problem"]))
        print(f"[{i}] tier={r['tier']} gold={r['answer']} got={[a for a, _ in answers]} robust={robust[-1]} trunc={truncated[-1]:.2f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"ood__{args.model.replace('/', '_')}.npz", model_id=args.model,
             problems=np.array([r["problem"] for r in rows], dtype=object), answers=np.array([r["answer"] for r in rows], dtype=object),
             tier=np.array([r["tier"] for r in rows]), robust=np.array(robust), base_ok=np.array(base_ok),
             truncated=np.array(truncated), features=np.array(features))
    share = float(np.mean(robust))
    print("SUMMARY", json.dumps({"model": args.model, "n": len(rows), "robust_share": share, "base_acc": float(np.mean(base_ok)),
                                 "truncated": float(np.mean(truncated)),
                                 "pilot_pass": 0.2 <= share <= 0.8 and float(np.mean(truncated)) < 0.2,
                                 "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
