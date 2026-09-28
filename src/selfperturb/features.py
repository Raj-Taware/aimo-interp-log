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

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .perturbations import PERTURBATIONS, perturb_all

# Fractions of network depth to read. Absolute layer indices differ across the
# four checkpoints (36 to 48 blocks), so depth fractions keep features
# comparable when we train one model on several checkpoints.
DEPTH_FRACTIONS = (0.25, 0.5, 0.65, 0.8, 0.9, 1.0)
MAX_PROMPT_TOKENS = 2048


def load_model(model_id: str, dtype: torch.dtype | None = None):
    """Load tokenizer + model from the offline cache, in eval mode."""
    if torch.cuda.is_available():
        # T4 (compute 7.5) has no bf16; the eval worker's Blackwell card does.
        resolved = dtype or (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        device_map = "auto"
    else:
        resolved, device_map = torch.float32, None

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # keeps the last token at index -1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=resolved,
        local_files_only=True,
        device_map=device_map,
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
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{problem}" if system_prompt else problem


@torch.no_grad()
def encode(tokenizer, model, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return last-token hidden states per layer, and next-token distributions.

    Shapes: hidden ``(n_prompts, n_layers + 1, hidden_dim)`` in float32, and
    ``probs`` ``(n_prompts, vocab)`` -- the model's belief about what comes next
    after reading the problem, which is our generation-free "doubt" signal.
    """
    batch = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    )
    batch = {name: value.to(model.device) for name, value in batch.items()}
    output = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)

    # Left padding means the real final token is always at position -1.
    hidden = torch.stack([state[:, -1, :] for state in output.hidden_states], dim=1)
    probs = torch.softmax(output.logits[:, -1, :].float(), dim=-1)
    return hidden.float().cpu().numpy(), probs.cpu().numpy()


def _layer_indices(n_states: int) -> list[int]:
    """Map depth fractions onto this model's actual hidden-state indices."""
    return [max(1, min(n_states - 1, round(f * (n_states - 1)))) for f in DEPTH_FRACTIONS]


def _entropy(probs: np.ndarray) -> float:
    return float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.clip(p, 1e-12, None), np.clip(q, 1e-12, None)
    return float((p * np.log(p / q)).sum())


def case_features(tokenizer, model, problem: str, system_prompt: str = "") -> np.ndarray:
    """Feature vector for one (model, problem) case.

    Per perturbation and per sampled depth: cosine drift and relative norm
    change of the hidden state. Plus, per perturbation, the KL divergence
    between original and perturbed next-token distributions. Plus three
    absolute doubt features of the original prompt itself.
    """
    variants = perturb_all(problem)
    names = list(PERTURBATIONS)
    prompts = [build_prompt(tokenizer, problem, system_prompt)]
    prompts += [build_prompt(tokenizer, variants[name], system_prompt) for name in names]

    hidden, probs = encode(tokenizer, model, prompts)
    layers = _layer_indices(hidden.shape[1])

    base_hidden, base_probs = hidden[0], probs[0]
    features: list[float] = []
    for index in range(1, len(prompts)):
        for layer in layers:
            a, b = base_hidden[layer], hidden[index][layer]
            denominator = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
            features.append(float(np.dot(a, b) / denominator))  # cosine drift
            features.append(
                float(np.linalg.norm(b) / (np.linalg.norm(a) + 1e-9))  # norm ratio
            )
        features.append(_kl(base_probs, probs[index]))

    # Absolute doubt on the untouched problem: how sure is it before it starts?
    top = np.sort(base_probs)[::-1]
    features.extend([_entropy(base_probs), float(top[0]), float(top[0] - top[1])])
    return np.asarray(features, dtype=np.float32)


def feature_names() -> list[str]:
    """Column names matching :func:`case_features`, for inspection and write-up."""
    names: list[str] = []
    for perturbation in PERTURBATIONS:
        for fraction in DEPTH_FRACTIONS:
            names.append(f"{perturbation}_cos_d{fraction}")
            names.append(f"{perturbation}_norm_d{fraction}")
        names.append(f"{perturbation}_kl")
    names.extend(["base_entropy", "base_top1", "base_margin"])
    return names
