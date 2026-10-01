"""Self-perturbation drift features from a single batch of forward passes.

The idea in one line: perturb the problem ourselves, push original and
perturbed statements through the model, and measure how far its internal state
moves. A model that has genuinely parsed the problem should represent a renamed
or typo'd version almost identically, and should react when a quantity changes.
A model leaning on surface patterns behaves the other way around.

Cost: one forward pass per variant, prompt only, no generation. With six
perturbations that is seven short forward passes per case -- roughly two orders
of magnitude cheaper than the generation-based uncertainty baseline, which is
what makes it viable inside the 3600-second budget.

Everything here uses only torch + transformers + numpy, all present in the
evaluation container.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .perturbations import PERTURBATIONS, distract_variants, perturb_all

# Fractions of network depth to read. Absolute layer indices differ across the
# four checkpoints (36 to 48 blocks), so depth fractions keep features
# comparable when we train one model on several checkpoints.
DEPTH_FRACTIONS = (0.25, 0.5, 0.65, 0.8, 0.9, 1.0)
# v3: the distractor signal lives at shallow depth, so sample it more densely.
MULTI_DEPTHS = (0.1, 0.25, 0.4)
N_DISTRACTORS = 5
MAX_PROMPT_TOKENS = 2048
# Appended after the chat template's assistant turn, so the last position is
# the model committing to the first token of its answer. Reasoning models
# otherwise always emit the same opening token (``<think>``) here, which made
# the next-token distribution constant -- entropy 0, top-1 1.0 -- on every case.
ANSWER_PREFIX = "The final answer is $\\boxed{"


def load_model(model_id: str, dtype: torch.dtype | None = None):
    """Load tokenizer + model in eval mode.

    Offline-only when ``HF_HUB_OFFLINE=1`` (set inside the evaluation container);
    elsewhere -- Kaggle, Lightning -- the checkpoint is downloaded on first use.
    """
    local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    if torch.cuda.is_available():
        # Always fp16 on GPU. Training features come from T4s, which have no
        # bf16; picking bf16 on the eval card (which has it) would score test
        # features at lower precision than the classifier was trained on --
        # and these features live at rounding scale. All four small models ran
        # fp16 with zero NaN/inf across 137 problems.
        resolved = dtype or torch.float16
        device_map = "auto"
    else:
        resolved, device_map = torch.float32, None

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # keeps the last token at index -1
    # transformers 5 renamed ``torch_dtype`` to ``dtype``; 4.x silently ignores
    # the new name and loads fp32, which would not fit an 8B model on a T4 pair.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_only,
        device_map=device_map,
        **{dtype_key: resolved},
    )
    if device_map is None:
        model.to("cpu")
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, problem: str, system_prompt: str = "") -> str:
    """Format a problem the way the model expects to receive it."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": problem})
        try:
            templated = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return templated + ANSWER_PREFIX
        except Exception:
            pass
    body = f"{system_prompt}\n\n{problem}" if system_prompt else problem
    return f"{body}\n\n{ANSWER_PREFIX}"


@torch.no_grad()
def encode(tokenizer, model, prompts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return last-token and mean-pooled hidden states per layer, plus
    next-token distributions.

    Shapes: ``last`` and ``pooled`` ``(n_prompts, n_layers + 1, hidden_dim)`` in
    float32, and ``probs`` ``(n_prompts, vocab)`` -- the model's belief about
    what comes next after reading the problem, our generation-free "doubt"
    signal. ``pooled`` averages over real tokens only (the attention mask
    excludes left padding), so it reflects the whole statement rather than the
    answer-prefix position alone.
    """
    # One prompt per forward pass, deliberately. The drift features are
    # differences in the third decimal of a cosine, at fp16 rounding scale, and
    # batching made each prompt's numerics depend on its batch-mates' padding:
    # the same feature on the same 137 problems correlated only r=0.62 (median)
    # between two runs whose batches differed. Unbatched, a prompt's features
    # depend on that prompt alone.
    lasts, pools, dists = [], [], []
    for prompt in prompts:
        batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
        batch = {name: value.to(model.device) for name, value in batch.items()}
        output = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
        # With device_map="auto" over two GPUs, layers live on different
        # devices, so each slice moves to CPU before stacking.
        lasts.append(torch.stack([state[0, -1, :].float().cpu() for state in output.hidden_states]))
        pools.append(torch.stack([state[0].float().cpu().mean(0) for state in output.hidden_states]))
        dists.append(torch.softmax(output.logits[0, -1, :].float(), dim=-1).cpu())
    return torch.stack(lasts).numpy(), torch.stack(pools).numpy(), torch.stack(dists).numpy()


def _layer_indices(n_states: int, fractions=DEPTH_FRACTIONS) -> list[int]:
    """Map depth fractions onto this model's actual hidden-state indices."""
    return [max(1, min(n_states - 1, round(f * (n_states - 1)))) for f in fractions]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _norm_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-9))


def _entropy(probs: np.ndarray) -> float:
    return float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.clip(p, 1e-12, None), np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())


def case_features(tokenizer, model, problem: str, system_prompt: str = "") -> np.ndarray:
    """Feature vector for one (model, problem) case, from one batched pass.

    v2 block (87 features, unchanged order): per perturbation and depth, last-
    token cosine drift and norm ratio; per perturbation, next-token KL and an
    applied flag; three absolute doubt features of the original prompt.

    v3 block: the distractor drift was the only single feature beating a
    shuffled-label null, and one inserted sentence is a noisy probe. So
    ``N_DISTRACTORS`` distracted copies are aggregated (mean and worst case) at
    denser shallow depths, from both last-token and mean-pooled states; plus
    mean-pooled drift for every original perturbation at depth 0.25.
    """
    variants = perturb_all(problem)
    names = list(PERTURBATIONS)
    extra = distract_variants(problem, N_DISTRACTORS)
    texts = [problem] + [variants[name] for name in names] + extra
    prompts = [build_prompt(tokenizer, text, system_prompt) for text in texts]

    last, pooled, probs = encode(tokenizer, model, prompts)
    n_states = last.shape[1]
    layers = _layer_indices(n_states)

    features: list[float] = []
    for index in range(1, len(names) + 1):
        for layer in layers:
            features.append(_cosine(last[0][layer], last[index][layer]))
            features.append(_norm_ratio(last[0][layer], last[index][layer]))
        features.append(_kl(probs[0], probs[index]))
        # Some perturbations cannot apply (no symbols to rename, no numbers to
        # respell). Unchanged text gives cosine exactly 1.0, which would read as
        # "perfectly stable"; this flag lets the classifier tell the two apart.
        features.append(float(variants[names[index - 1]] != problem))

    # Absolute doubt on the untouched problem: how sure is it before it starts?
    top = np.sort(probs[0])[::-1]
    features.extend([_entropy(probs[0]), float(top[0]), float(top[0] - top[1])])

    # v3: multi-distractor aggregates.
    extra_rows = range(len(names) + 1, len(prompts))
    for layer in _layer_indices(n_states, MULTI_DEPTHS):
        cos_last = [_cosine(last[0][layer], last[row][layer]) for row in extra_rows]
        norm_last = [_norm_ratio(last[0][layer], last[row][layer]) for row in extra_rows]
        cos_pool = [_cosine(pooled[0][layer], pooled[row][layer]) for row in extra_rows]
        features.extend([np.mean(cos_last), np.min(cos_last),
                         np.mean(norm_last), np.max(np.abs(np.log(norm_last))),
                         np.mean(cos_pool), np.min(cos_pool)])
    kls = [_kl(probs[0], probs[row]) for row in extra_rows]
    features.extend([np.mean(kls), np.max(kls)])

    # v3: mean-pooled drift for each original perturbation.
    pool_layer = _layer_indices(n_states, (0.25,))[0]
    for index in range(1, len(names) + 1):
        features.append(_cosine(pooled[0][pool_layer], pooled[index][pool_layer]))
    return np.asarray(features, dtype=np.float32)


def feature_names() -> list[str]:
    """Column names matching :func:`case_features`, for inspection and write-up."""
    names: list[str] = []
    for perturbation in PERTURBATIONS:
        for fraction in DEPTH_FRACTIONS:
            names.append(f"{perturbation}_cos_d{fraction}")
            names.append(f"{perturbation}_norm_d{fraction}")
        names.append(f"{perturbation}_kl")
        names.append(f"{perturbation}_applied")
    names.extend(["base_entropy", "base_top1", "base_margin"])
    for fraction in MULTI_DEPTHS:
        names.extend(f"multidistract_{stat}_d{fraction}" for stat in (
            "cos_mean", "cos_min", "norm_mean", "lognorm_absmax", "poolcos_mean", "poolcos_min"))
    names.extend(["multidistract_kl_mean", "multidistract_kl_max"])
    names.extend(f"{perturbation}_poolcos_d0.25" for perturbation in PERTURBATIONS)
    return names
