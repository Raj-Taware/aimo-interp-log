"""T-025 sample: hidden-state drift inside the reasoning, aligned by teacher forcing.

Every probe after v2b forces an immediate answer, yet the labels come from reasoned answers.
Two free generations cannot be compared token by token, so: generate a greedy reasoning trace
on the original problem, then replay that exact trace after the perturbed problem. Positions
now match one to one, and any state difference is caused by the problem text alone.

    drift = mean over trace tokens of 1 - cos(state_original, state_perturbed) at depth 0.5
    nll_gap = mean NLL of the trace on the perturbed problem - on the original (secondary)

Only typos and distract: the trace stays a valid continuation under both, while rename,
numerals and rephrase would make it mention text the perturbed problem no longer contains.
identity_drift replays the original against itself and must be ~0.

    python scripts/dump_trace.py --model <hf id> --rows rows.jsonl --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from selfperturb.features import ANSWER_PREFIX, MAX_PROMPT_TOKENS, _layer_indices, build_prompt, load_model
from selfperturb.perturbations import perturb_all

PERTURBATIONS = ("typos", "distract")
TRACE_TOKENS = 64
DEPTH = 0.5


def prefix_ids(tokenizer, model, problem: str) -> torch.Tensor:
    prompt = build_prompt(tokenizer, problem).removesuffix(ANSWER_PREFIX)
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)["input_ids"].to(model.device)


@torch.no_grad()
def replay(model, prefix: torch.Tensor, trace: torch.Tensor) -> tuple[torch.Tensor, float]:
    out = model(input_ids=torch.cat([prefix, trace], 1), output_hidden_states=True, use_cache=False)
    n = trace.shape[1]
    layer = _layer_indices(len(out.hidden_states), (DEPTH,))[0]
    logp = torch.log_softmax(out.logits[0, -n - 1:-1].float(), -1).cpu()
    nll = -logp.gather(1, trace[0, :, None].cpu()).mean().item()
    return out.hidden_states[layer][0, -n:].float().cpu(), nll


def drift(a: torch.Tensor, b: torch.Tensor) -> float:
    return (1 - torch.cosine_similarity(a, b, dim=-1)).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.open(encoding="utf-8")][: args.limit]
    tokenizer, model = load_model(args.model)
    n, m = len(rows), len(PERTURBATIONS)
    drifts, gaps = np.full((n, m), np.nan), np.full((n, m), np.nan)
    identity, lengths = np.full(n, np.nan), np.zeros(n, dtype=int)
    t0 = time.time()
    for i, r in enumerate(rows):
        prefix = prefix_ids(tokenizer, model, r["problem"])
        with torch.no_grad():
            trace = model.generate(prefix, max_new_tokens=TRACE_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)[:, prefix.shape[1]:]
        lengths[i] = trace.shape[1]
        base, base_nll = replay(model, prefix, trace)
        identity[i] = drift(base, replay(model, prefix, trace)[0])
        variants = perturb_all(r["problem"])
        for j, name in enumerate(PERTURBATIONS):
            if variants[name] != r["problem"]:
                states, nll = replay(model, prefix_ids(tokenizer, model, variants[name]), trace)
                drifts[i, j], gaps[i, j] = drift(base, states), nll - base_nll
        print(f"[{i}] trace={lengths[i]} identity={identity[i]:.2e} drift={drifts[i].round(5)} nll_gap={gaps[i].round(4)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"trace__{args.model.replace('/', '_')}.npz", model_id=args.model,
             problems=np.array([r["problem"] for r in rows], dtype=object), drift=drifts, nll_gap=gaps,
             identity_drift=identity, trace_tokens=lengths, perturbations=np.array(PERTURBATIONS))
    print("SUMMARY", json.dumps({"model": args.model, "n": n, "median_identity": float(np.nanmedian(identity)),
                                 "median_drift": float(np.nanmedian(drifts)), "median_trace": float(np.median(lengths)),
                                 "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
