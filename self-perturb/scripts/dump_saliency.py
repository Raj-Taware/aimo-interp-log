"""T-023 sample: where does the model's immediate answer look? (gradient x input saliency)

Failure this targets: every signal read from the question-only state fails to transfer, and
T-022 showed the models do not recall answers, yet problem rewrites still change their
behaviour. A non-robust answer should lean on the wording of the problem rather than on its
mathematics. That is measurable without any perturbation: differentiate the log-prob of the
model's top next token after "The final answer is $\\boxed{" w.r.t. the input embeddings.

Per problem token: saliency = ||grad * embedding||. Tokens split into math (inside $...$ or
containing a digit) and prose. Score = prose_enrichment = log(share of saliency on prose /
share of tokens that are prose), so a long-winded problem does not score higher by being long.

    python scripts/dump_saliency.py --model <hf id> --rows rows.jsonl --out <dir>
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


def math_chars(problem: str) -> np.ndarray:
    """True for characters inside $...$ or that are digits."""
    mask = np.array([c.isdigit() for c in problem], dtype=bool)
    for span in re.finditer(r"\$[^$]+\$", problem):
        mask[span.start():span.end()] = True
    return mask


def enrichment(saliency: np.ndarray, prose: np.ndarray) -> float:
    if prose.all() or not prose.any() or saliency.sum() == 0:
        return float("nan")
    return float(np.log((saliency[prose].sum() / saliency.sum()) / prose.mean()))


def saliency_of(tokenizer, model, problem: str) -> tuple[float, float, int]:
    prompt = build_prompt(tokenizer, problem)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].numpy()
    ids = enc["input_ids"].to(model.device)
    start = prompt.find(problem)
    if start < 0:
        return float("nan"), float("nan"), 0
    chars = math_chars(problem)
    inside = (offsets[:, 0] >= start) & (offsets[:, 1] <= start + len(problem)) & (offsets[:, 1] > offsets[:, 0])
    is_math = np.array([chars[a - start:b - start].any() if keep else False
                        for (a, b), keep in zip(offsets, inside)])
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model(inputs_embeds=embeds, use_cache=False).logits[0, -1].float()
        torch.log_softmax(logits, -1).max().backward()
    sal = (embeds.grad[0].float() * embeds[0].detach().float()).norm(dim=-1).cpu().numpy()
    return enrichment(sal[inside], ~is_math[inside]), float((sal[inside] == 0).mean()), int(inside.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.open(encoding="utf-8")][: args.limit]
    tokenizer, model = load_model(args.model)
    model.requires_grad_(False)
    scores, zeros, tokens = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            s, z, n = saliency_of(tokenizer, model, r["problem"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            s, z, n = float("nan"), float("nan"), -1
        scores.append(s); zeros.append(z); tokens.append(n)
        print(f"[{i}] tokens={n} enrichment={s:.3f} zero_grad={z:.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"saliency__{args.model.replace('/', '_')}.npz", model_id=args.model,
             problems=np.array([r["problem"] for r in rows], dtype=object), score=np.array(scores),
             zero_grad=np.array(zeros), tokens=np.array(tokens))
    print("SUMMARY", json.dumps({"model": args.model, "n": len(rows), "scored": int(np.isfinite(scores).sum()),
                                 "oom": tokens.count(-1), "mean_zero_grad": float(np.nanmean(zeros)),
                                 "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
