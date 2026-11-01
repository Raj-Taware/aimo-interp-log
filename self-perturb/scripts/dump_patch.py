"""T-024 sample: activation patching of the answer position on perturbed problems.

v2b's drift says hidden states moved; it cannot say the move matters. Here: run a perturbed
prompt, overwrite the state of the last token (the "\\boxed{" position, the same in both
prompts) at one layer with the state from the ORIGINAL prompt, and let later layers run.

    kl_pert[i, j]     = KL(p_orig || p_perturbed)
    kl_patch[i, j, d] = KL(p_orig || p_perturbed with last-token state patched at depth d)

recovery = 1 - kl_patch / kl_pert: the share of the output change that travels through the
answer-position state at that depth. The rest arrives from context tokens (attention in later
layers). identity_kl patches a prompt with its own state, so it must be ~0 if the hook is right.

    python scripts/dump_patch.py --model <hf id> --rows rows.jsonl --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from selfperturb.features import MAX_PROMPT_TOKENS, _kl, _layer_indices, build_prompt, load_model
from selfperturb.perturbations import ANSWER_PRESERVING, perturb_all

DEPTHS = (0.25, 0.5, 0.75)


def patch_hook(state: torch.Tensor):
    def hook(_module, _inputs, output):
        hidden = (output[0] if isinstance(output, tuple) else output).clone()
        hidden[0, -1] = state.to(hidden.device, hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    return hook


@torch.no_grad()
def forward(tokenizer, model, problem: str, layer=None, state=None):
    batch = tokenizer(build_prompt(tokenizer, problem), return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
    batch = {name: value.to(model.device) for name, value in batch.items()}
    handle = layer.register_forward_hook(patch_hook(state)) if layer is not None else None
    try:
        out = model(**batch, output_hidden_states=layer is None, return_dict=True, use_cache=False)
    finally:
        if handle:
            handle.remove()
    probs = torch.softmax(out.logits[0, -1].float(), -1).cpu().numpy()
    return probs, ([s[0, -1].clone() for s in out.hidden_states] if layer is None else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.open(encoding="utf-8")][: args.limit]
    tokenizer, model = load_model(args.model)
    layers = model.get_decoder().layers
    n, m = len(rows), len(ANSWER_PRESERVING)
    kl_pert, kl_patch = np.full((n, m), np.nan), np.full((n, m, len(DEPTHS)), np.nan)
    identity = np.full(n, np.nan)
    t0 = time.time()
    for i, r in enumerate(rows):
        p0, states = forward(tokenizer, model, r["problem"])
        idx = _layer_indices(len(states), DEPTHS)
        identity[i] = _kl(p0, forward(tokenizer, model, r["problem"], layers[idx[1] - 1], states[idx[1]])[0])
        variants = perturb_all(r["problem"])
        for j, name in enumerate(ANSWER_PRESERVING):
            text = variants[name]
            if text == r["problem"]:
                continue
            kl_pert[i, j] = _kl(p0, forward(tokenizer, model, text)[0])
            for d, k in enumerate(idx):
                # hidden_states[k] is the output of layers[k - 1]
                kl_patch[i, j, d] = _kl(p0, forward(tokenizer, model, text, layers[k - 1], states[k])[0])
        print(f"[{i}] identity={identity[i]:.2e} kl_pert={np.nanmean(kl_pert[i]):.4f} kl_patch={np.nanmean(kl_patch[i], 0).round(4)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"patch__{args.model.replace('/', '_')}.npz", model_id=args.model,
             problems=np.array([r["problem"] for r in rows], dtype=object), kl_pert=kl_pert, kl_patch=kl_patch,
             identity_kl=identity, depths=np.array(DEPTHS), perturbations=np.array(ANSWER_PRESERVING))
    print("SUMMARY", json.dumps({"model": args.model, "n": n, "median_identity_kl": float(np.nanmedian(identity)),
                                 "median_kl_pert": float(np.nanmedian(kl_pert)), "median_kl_patch": np.nanmedian(kl_patch, (0, 1)).tolist(),
                                 "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
