"""T-021: self-domain-rewrite white-box features (targets the label mechanism, see label_mechanism.py).

The organisers' non-robust labels are mostly "domain" rewrites (problem recast as a
real-world story) that the model then fails. Here the model under test writes its own
domain rewrite (greedy, thinking skipped), and we read how far its internals move
between the original and the rewrite, at the answer-commit position (build_prompt ends
with ANSWER_PREFIX):

* ``cos_mean_L*`` / ``cos_last_L*``: cosine of question-mean / last-token states
* ``norm_L*``: log norm ratio of mean states (rewrite / original)
* ``kl``, ``top1_same``, ``p_keep``, ``ent_orig``, ``ent_rew``: next-token answer distribution
* text diagnostics (black-box, kept apart for the control): ``retain`` = share of the
  original's numbers present in the rewrite, ``rew_words``

    python scripts/dump_selfdomain.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --source data/math.parquet --out results/grok/T-021 [--limit 5]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_features import safe_id  # noqa: E402
from dump_states import load_source_rows, state_layers  # noqa: E402
from selfperturb.features import MAX_PROMPT_TOKENS, build_prompt, load_model  # noqa: E402

REWRITE = ("Rewrite the following math problem as a short real-world word problem (a story about "
           "people, objects or money). Keep every number and condition so that the answer is "
           "unchanged. Output only the rewritten problem.\n\nProblem: ")
REWRITE_TOKENS = 256
TEXT_COLUMNS = ("retain", "rew_words")


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


@torch.no_grad()
def rewrite(tokenizer, model, problem: str) -> str:
    chat = tokenizer.apply_chat_template([{"role": "user", "content": REWRITE + problem}],
                                         tokenize=False, add_generation_prompt=True)
    # Skip the reasoning trace: an empty think block, then the answer.
    chat += "</think>\n\n" if chat.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"
    batch = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**batch, do_sample=False, max_new_tokens=REWRITE_TOKENS,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    text = tokenizer.decode(out[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("</think>")[-1].strip() or problem


@torch.no_grad()
def read(tokenizer, model, problem: str):
    batch = tokenizer(build_prompt(tokenizer, problem), return_tensors="pt", truncation=True,
                      max_length=MAX_PROMPT_TOKENS).to(model.device)
    out = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
    layers = state_layers(len(out.hidden_states) - 1)
    mean = torch.stack([out.hidden_states[i][0].float().cpu().mean(0) for i in layers])
    last = torch.stack([out.hidden_states[i][0, -1].float().cpu() for i in layers])
    logp = torch.log_softmax(out.logits[0, -1].float().cpu(), -1)
    return mean, last, logp, layers


def features(orig, rew) -> dict[str, float]:
    (m0, l0, p0, layers), (m1, l1, p1, _) = orig, rew
    row = {}
    for k, layer in enumerate(layers):
        row[f"cos_mean_L{layer}"] = float(torch.cosine_similarity(m0[k], m1[k], 0))
        row[f"cos_last_L{layer}"] = float(torch.cosine_similarity(l0[k], l1[k], 0))
        row[f"norm_L{layer}"] = float(torch.log(m1[k].norm() / m0[k].norm()))
    top = int(p0.argmax())
    row["kl"] = float((p0.exp() * (p0 - p1)).sum())
    row["top1_same"] = float(int(p1.argmax()) == top)
    row["p_keep"] = float(p1[top].exp())
    row["ent_orig"] = float(-(p0.exp() * p0).sum())
    row["ent_rew"] = float(-(p1.exp() * p1).sum())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tokenizer, model = load_model(args.model, dtype=torch.float16)
    rows = load_source_rows(args.source, args.model)[: args.limit]
    table, rewrites, started = [], [], time.time()
    for index, (problem, label) in enumerate(rows):
        text = rewrite(tokenizer, model, problem)
        row = features(read(tokenizer, model, problem), read(tokenizer, model, text))
        have = numbers(problem)
        row["retain"] = len(have & numbers(text)) / len(have) if have else 1.0
        row["rew_words"] = float(len(text.split()))
        table.append(row), rewrites.append(text)
        if index < 5 or (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(rows)} {(time.time() - started) / (index + 1):.1f}s/row "
                  f"kl={row['kl']:.3f} retain={row['retain']:.2f} | {text[:160]!r}", flush=True)
    columns = list(table[0])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"selfdomain__{safe_id(args.model)}.npz",
             features=np.array([[r[c] for c in columns] for r in table], dtype=np.float32),
             columns=np.array(columns, dtype=object), labels=np.array([y for _, y in rows], dtype=np.float32),
             problems=np.array([p for p, _ in rows], dtype=object), rewrites=np.array(rewrites, dtype=object),
             model_id=args.model)


if __name__ == "__main__":
    main()
