"""Token-aligned representational geometry between a problem and its edits.

Phase 1 of the interpretability rework (see the adversarial critique in the
project log). The earlier features compared *unaligned summaries* — the last
token, or a mean over all tokens — with raw cosine. Three flaws followed:

* inserted tokens entered the comparison, so drift measured "how much text
  changed", i.e. problem length;
* in a causal model every token *before* an edit has an identical state, and
  averaging those in diluted the effect by an amount that depends on where
  the edit sits;
* raw cosine is dominated by a few huge, content-blind residual directions,
  so every comparison read ~0.99 and the signal sat at fp16 rounding scale.

Here each edited prompt is aligned to the original token by token
(``difflib`` on token ids). Only matched tokens *after the first edit* are
compared — the same piece of the problem, read in a changed context — and
inserted tokens are tracked separately. States are centred and scaled per
layer with the original prompt's own token statistics (label-free, sink token
excluded). Per layer and per edit we record:

* ``cka``     linear CKA between the aligned token clouds (relational geometry,
              invariant to rotation and isotropic scale);
* ``wcos``    mean whitened cosine of aligned token pairs;
* ``ans``     whitened cosine at the answer position (last token);
* ``lens_kl`` KL between logit-lens next-token distributions at the answer
              position — where in depth the edit changes the answer;
* ``attnx``   (distractors only) attention mass that later tokens and the
              answer position place on the inserted sentence, divided by the
              sentence's share of the visible context: >1 means the model
              actively attends to irrelevant text. Length-controlled by
              construction.

``case_geometry`` returns a compact feature vector (metrics sampled at
``LAYER_FRACTIONS``) and the full per-layer profiles for figures.
"""

from __future__ import annotations

import difflib

import numpy as np
import torch

from .features import MAX_PROMPT_TOKENS, build_prompt
from .perturbations import PERTURBATIONS, distract_variants

LAYER_FRACTIONS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
FAMILIES = ("rename", "numerals", "typos", "distract", "rephrase", "number_jitter")
METRICS = ("cka", "wcos", "ans", "lens_kl")
N_DISTRACT = 5


# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------


def _final_norm_and_head(model):
    """Final normalization and unembedding, for the logit lens. None if unknown."""
    head = model.get_output_embeddings()
    for path in ("model.norm", "model.language_model.norm", "transformer.ln_f", "model.final_layernorm"):
        module = model
        try:
            for part in path.split("."):
                module = getattr(module, part)
        except AttributeError:
            continue
        return module, head
    return None, head


@torch.no_grad()
def _forward(tokenizer, model, prompt: str, attentions: bool):
    batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
    ids = batch["input_ids"][0].tolist()
    batch = {name: value.to(model.device) for name, value in batch.items()}
    output = model(
        **batch,
        output_hidden_states=True,
        output_attentions=attentions,
        return_dict=True,
        use_cache=False,
    )
    device = model.device
    # (n_states, T, d) on one device; layers may sit on different GPUs.
    states = torch.stack([h[0].to(device, torch.float32) for h in output.hidden_states])
    attn = list(output.attentions) if attentions and output.attentions is not None else None
    return ids, states, attn


# --------------------------------------------------------------------------
# alignment and metrics
# --------------------------------------------------------------------------


def align(original_ids: list[int], variant_ids: list[int]):
    """Match tokens; return post-edit aligned pairs and inserted variant positions.

    Pairs before the first divergence are dropped: in a causal model those
    states are identical in both prompts and carry no information.
    """
    matcher = difflib.SequenceMatcher(None, original_ids, variant_ids, autojunk=False)
    pairs = [
        (block.a + k, block.b + k)
        for block in matcher.get_matching_blocks()
        for k in range(block.size)
    ]
    prefix = 0
    limit = min(len(original_ids), len(variant_ids))
    while prefix < limit and original_ids[prefix] == variant_ids[prefix]:
        prefix += 1
    post = [(i, j) for i, j in pairs if i >= prefix]
    matched = {j for _, j in pairs}
    inserted = [j for j in range(len(variant_ids)) if j not in matched]
    return post, inserted


def _cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA via n x n Gram matrices (n tokens << d dims)."""
    if x.shape[0] < 3:
        return float("nan")
    x = x - x.mean(0)
    y = y - y.mean(0)
    kx, ky = x @ x.T, y @ y.T
    denom = torch.linalg.norm(kx) * torch.linalg.norm(ky)
    return float((kx * ky).sum() / denom) if denom > 0 else float("nan")


@torch.no_grad()
def _layer_profiles(orig_states, var_states, pairs, norm, head, lens_layers):
    """Per-layer metrics for one (original, variant) pair."""
    n_states = orig_states.shape[0]
    device = orig_states.device
    out = {metric: np.full(n_states, np.nan) for metric in METRICS}
    for layer in range(n_states):
        # fp16 can overflow at a few positions in a model's last block (seen in
        # Skywork's final layer); drop non-finite token states rather than let
        # one of them turn the whole layer's statistics into NaN.
        ok_o = torch.isfinite(orig_states[layer]).all(-1)
        ok_v = torch.isfinite(var_states[layer]).all(-1)
        reference = orig_states[layer, 1:][ok_o[1:]]       # drop the sink token
        if reference.shape[0] < 2:
            continue
        mu = reference.mean(0)
        sd = reference.std(0).clamp_min(1e-4)
        o = (orig_states[layer] - mu) / sd
        v = (var_states[layer] - mu) / sd
        kept = [(i, j) for i, j in pairs if ok_o[i] and ok_v[j]]
        if kept:
            i_idx = torch.tensor([i for i, _ in kept], device=device, dtype=torch.long)
            j_idx = torch.tensor([j for _, j in kept], device=device, dtype=torch.long)
            x, y = o[i_idx], v[j_idx]
            out["wcos"][layer] = float(torch.nn.functional.cosine_similarity(x, y, dim=-1).mean())
            out["cka"][layer] = _cka(x, y)
        if ok_o[-1] and ok_v[-1]:
            out["ans"][layer] = float(torch.nn.functional.cosine_similarity(o[-1], v[-1], dim=0))
        if layer in lens_layers and head is not None:
            pair = torch.stack([orig_states[layer, -1], var_states[layer, -1]])
            # The last hidden state already has the final norm applied.
            if norm is not None and layer < n_states - 1:
                pair = norm(pair.to(next(norm.parameters()).dtype)).float()
            logp = torch.log_softmax(head(pair.to(head.weight.dtype)).float(), dim=-1)
            out["lens_kl"][layer] = float((logp[0].exp() * (logp[0] - logp[1])).sum())
    return out


@torch.no_grad()
def _attention_layer_ids(model, n_attentions: int, n_states: int):
    """Hidden-state index (1..n_layers) of each returned attention tensor.

    Hybrid models (Qwen3.5) return attention only for their full-attention
    layers; ``config.layer_types`` says which those are. Returns None when the
    mapping cannot be established.
    """
    n_layers = n_states - 1
    if n_attentions == n_layers:
        return list(range(1, n_layers + 1))
    config = getattr(model.config, "text_config", None) or model.config
    types = getattr(config, "layer_types", None)
    if types and len(types) == n_layers:
        full = [i + 1 for i, kind in enumerate(types) if "full" in str(kind)]
        if len(full) == n_attentions:
            return full
    return None


@torch.no_grad()
def _attention_excess(attentions, inserted, pairs, n_states, layer_ids=None):
    """Excess attention on the inserted span, per layer (NaN where unavailable).

    ``layer_ids`` maps each attention tensor to its hidden-state index; without
    it the profile stays NaN but the mean over layers is still returned.
    """
    out = np.full(n_states, np.nan)                         # index = hidden-state layer
    values: list[float] = []
    if not attentions or not inserted:
        return out, values
    span_end = max(inserted)
    queries = sorted({j for _, j in pairs if j > span_end})
    if not queries:
        return out, values
    span = torch.tensor(inserted, dtype=torch.long)
    q = torch.tensor(queries, dtype=torch.long)
    expected = torch.tensor([len(inserted) / (t + 1) for t in queries], dtype=torch.float32)
    present = [a for a in attentions if a is not None]      # linear-attention layers give None
    if layer_ids is None and len(present) == len(attentions) == n_states - 1:
        layer_ids = list(range(1, n_states))
    for k, a in enumerate(present):
        a = a[0].float()                                    # (heads, T, T)
        mass = a[:, q.to(a.device)][:, :, span.to(a.device)].sum(-1)   # (heads, |q|)
        value = float((mass.cpu() / expected).mean())
        if layer_ids is not None:
            out[layer_ids[k]] = value
        # fp16 attention can overflow in a model's last block (Skywork); keep
        # it out of the mean, which then matches the earlier nanmean definition.
        if np.isfinite(value):
            values.append(value)
    return out, values


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def _sample(profile: np.ndarray) -> list[float]:
    last = len(profile) - 1
    return [float(profile[max(1, min(last, round(f * last)))]) for f in LAYER_FRACTIONS]


def _sample_nearest(profile: np.ndarray) -> list[float]:
    """Like ``_sample`` but takes the nearest layer that has a value.

    Hybrid models have attention only at some depths (every 4th layer in
    Qwen3.5), so a fixed fraction can land on a layer without attention.
    """
    available = np.flatnonzero(np.isfinite(profile))
    if available.size == 0:
        return [float("nan")] * len(LAYER_FRACTIONS)
    last = len(profile) - 1
    return [float(profile[available[np.argmin(np.abs(available - f * last))]]) for f in LAYER_FRACTIONS]


def case_geometry(tokenizer, model, problem: str, system_prompt: str = ""):
    """Geometry features and per-layer profiles for one (model, problem) case."""
    norm, head = _final_norm_and_head(model)
    orig_ids, orig_states, _ = _forward(tokenizer, model, build_prompt(tokenizer, problem, system_prompt), False)
    n_states = orig_states.shape[0]
    lens_layers = {max(1, min(n_states - 1, round(f * (n_states - 1)))) for f in LAYER_FRACTIONS}

    texts = {name: [PERTURBATIONS[name](problem)] for name in FAMILIES}
    texts["distract"] = texts["distract"] + distract_variants(problem, N_DISTRACT)

    profiles: dict[str, dict[str, np.ndarray]] = {}
    applied: dict[str, float] = {}
    attn_runs: list[np.ndarray] = []
    attn_means: list[float] = []
    for family in FAMILIES:
        runs = []
        for text in texts[family]:
            if text == problem:
                continue
            want_attn = family == "distract"
            var_ids, var_states, attentions = _forward(
                tokenizer, model, build_prompt(tokenizer, text, system_prompt), want_attn
            )
            pairs, inserted = align(orig_ids, var_ids)
            runs.append(_layer_profiles(orig_states, var_states, pairs, norm, head, lens_layers))
            if want_attn:
                present = [a for a in (attentions or []) if a is not None]
                layer_ids = _attention_layer_ids(model, len(present), n_states) if present else None
                profile, values = _attention_excess(attentions, inserted, pairs, n_states, layer_ids)
                attn_runs.append(profile)
                if values:
                    attn_means.append(float(np.mean(values)))
            del var_states, attentions
        applied[family] = float(bool(runs))
        profiles[family] = {
            metric: (np.nanmean([r[metric] for r in runs], axis=0) if runs else np.full(n_states, np.nan))
            for metric in METRICS
        }
    attn_profile = np.nanmean(attn_runs, axis=0) if attn_runs else np.full(n_states, np.nan)
    profiles["distract"]["attnx"] = attn_profile

    values: list[float] = []
    for family in FAMILIES:
        for metric in METRICS:
            values += _sample(profiles[family][metric])
    values += _sample_nearest(attn_profile) + [float(np.mean(attn_means)) if attn_means else float("nan")]
    values += [applied[family] for family in FAMILIES]
    return np.asarray(values, dtype=np.float32), profiles


def geometry_names() -> list[str]:
    names = [f"{family}_{metric}_d{f}" for family in FAMILIES for metric in METRICS for f in LAYER_FRACTIONS]
    names += [f"distract_attnx_d{f}" for f in LAYER_FRACTIONS] + ["distract_attnx_mean"]
    names += [f"{family}_applied" for family in FAMILIES]
    return names
