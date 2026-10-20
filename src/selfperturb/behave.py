"""v5: run the label recipe at test time (behavioural self-test).

The organizers label a (model, problem) pair robust when the model's accuracy
survives answer-preserving perturbations (max_drop <= 0.10; spurious >= 0.25;
pairs the model does not solve consistently are excluded). We have the model
and ~300 s per problem on a 96 GB card, so we measure that directly:

1. sample the original ``N_ORIG`` times and each variant family ``N_VAR``
   times, all in one batched ``generate`` capped by wall clock (``max_time``);
2. the reference answer is the majority \\boxed{} answer on the original
   (no gold answer is given; excluded low-accuracy pairs make the majority
   a good proxy);
3. drop = agreement(original) - min over families agreement(family);
   robust iff drop < DROP_CUT.

Every step logs to stderr so Codabench logs show throughput and decisions.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

import torch

from .features import load_model
from .perturbations import distract, numerals, rename, typos

INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
FAMILIES = {"rename": rename, "typos": typos, "distract": distract, "numerals": numerals}
N_ORIG = int(os.environ.get("BEHAVE_N_ORIG", "4"))
N_VAR = int(os.environ.get("BEHAVE_N_VAR", "2"))
MAX_NEW = int(os.environ.get("BEHAVE_MAX_NEW", "8192"))
CASE_SECONDS = float(os.environ.get("BEHAVE_SECONDS", "250"))  # job budget 4200 s / 14 cases
DROP_CUT = float(os.environ.get("BEHAVE_DROP_CUT", "0.25"))
FALLBACK = False  # original never finished within budget: hard problem


def log(*parts) -> None:
    print("[behave]", *parts, file=sys.stderr, flush=True)


def last_boxed(text: str) -> str | None:
    i, last = 0, None
    while (i := text.find("\\boxed{", i)) >= 0:
        j, depth = i + 7, 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth:
            break
        last, i = text[i + 7 : j - 1], j
    return None if last is None else last.replace(" ", "").replace("$", "").strip().rstrip(".")


def prompts_for(problem: str) -> list[tuple[str, str]]:
    groups = [("original", problem)] * N_ORIG
    for name, fn in FAMILIES.items():
        try:
            text = fn(problem)
        except Exception:
            continue
        if text and text != problem:
            groups += [(name, text)] * N_VAR
    return groups


def decide(groups: list[str], answers: list[str | None]) -> tuple[bool, dict]:
    orig = [a for g, a in zip(groups, answers) if g == "original"]
    votes = Counter(a for a in orig if a)
    if not votes:
        return FALLBACK, {"reason": "no original answer"}
    ref = votes.most_common(1)[0][0]
    acc = {}
    for g, a in zip(groups, answers):
        acc.setdefault(g, []).append(a == ref)
    acc = {g: sum(v) / len(v) for g, v in acc.items()}
    fams = {g: v for g, v in acc.items() if g != "original"}
    drop = acc["original"] - min(fams.values()) if fams else 0.0
    return bool(drop < DROP_CUT), {"ref": ref, "acc": acc, "drop": round(drop, 3)}


@torch.no_grad()
def run_case(model_id: str, problem: str, seconds: float = CASE_SECONDS) -> tuple[bool, dict]:
    start = time.time()
    tokenizer, model = load_model(model_id, dtype=torch.bfloat16)
    groups_texts = prompts_for(problem)
    groups = [g for g, _ in groups_texts]
    chats = [tokenizer.apply_chat_template([{"role": "user", "content": t + INSTRUCTION}],
                                           tokenize=False, add_generation_prompt=True)
             for _, t in groups_texts]
    batch = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    budget = seconds - (time.time() - start) - 10
    log(f"{model_id} load {time.time() - start:.1f}s, {len(chats)} seqs, gen budget {budget:.0f}s")
    t0 = time.time()
    out = model.generate(**batch, do_sample=True, max_new_tokens=MAX_NEW, max_time=max(budget, 5.0),
                         pad_token_id=tokenizer.pad_token_id)
    new = out[:, batch["input_ids"].shape[1]:]
    texts = tokenizer.batch_decode(new, skip_special_tokens=True)
    answers = [last_boxed(t) for t in texts]
    ntok = int((new != tokenizer.pad_token_id).sum())
    secs = time.time() - t0
    robust, info = decide(groups, answers)
    log(f"gen {secs:.0f}s {ntok} tok ({ntok / max(secs, 1e-6):.0f} tok/s), steps {new.shape[1]}, "
        f"answered {sum(a is not None for a in answers)}/{len(answers)} -> robust={robust} {info}")
    del model
    torch.cuda.empty_cache()
    return robust, info
