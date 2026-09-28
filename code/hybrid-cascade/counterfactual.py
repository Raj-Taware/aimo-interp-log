"""Cheap K=1–2 behavioural counterfactuals (answer-stability checks).

Transforms are deliberately light (whitespace / rename / distractor) so a few
generations fit under the 3600s whole-run budget. Agreement of answers across
the original and K counterfactual prompts is a stability proxy — not
reconstruction of the unpublished private labelling rule H.

On Codabench GPU the cascade uses these votes as an **agreement veto** on top
of last-token probes (True only if probe says True AND answers agree). Pure
CPU / missing local weights ⇒ ``None`` so the cascade can skip cleanly.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# Default K kept tiny for cascade budget.
DEFAULT_K = 2
# Thinking models need headroom; still << uncertainty baseline's 4096.
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_PROMPT_TOKENS = 1024

# Shared system prompt (aligned with probe training / trained-probe baseline).
SYSTEM_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)

_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*$"
)
_BOXED_RE = re.compile(
    r"\\boxed\s*\{([^{}]+)\}"
)
_INT_RE = re.compile(
    r"(?<![A-Za-z0-9_/])(-?\d+)(?![A-Za-z0-9_/])"
)


def light_transforms(problem: str, k: int = DEFAULT_K) -> list[str]:
    """Deterministic local transforms; no network paraphrase APIs.

    Order is stable for reproducibility. Prefer meaning-preserving wording
    changes that still stress brittle pattern-matchers.
    """
    variants: list[str] = []
    text = problem
    # 1) whitespace / punctuation light touch
    if k >= 1:
        variants.append(re.sub(r"[ \t]+", " ", text).strip())
    # 2) mild rename-style: swap common variable letter if present, else append
    #    a no-op clarifying sentence that should not change the math.
    if k >= 2:
        swapped = re.sub(
            r"\b([nNmMxX])\b",
            lambda m: {
                "n": "m",
                "N": "M",
                "m": "n",
                "M": "N",
                "x": "y",
                "X": "Y",
            }.get(m.group(1), m.group(1)),
            text,
            count=3,
        )
        if swapped == text:
            swapped = (
                text
                + "\n\n(Answer with the same final integer as the original statement.)"
            )
        variants.append(swapped)
    return variants[: max(0, k)]


def extract_final_answer(text: str | None) -> str | None:
    """Pull a normalized final answer from model output, or ``None`` if absent.

    Preference order: ``ANSWER:`` line → ``\\boxed{...}`` → last standalone integer.
    Normalization: strip, collapse spaces, lower-case ``nan``.
    """
    if not text:
        return None
    raw = str(text)

    matches = list(_ANSWER_LINE_RE.finditer(raw))
    if matches:
        return _normalize_answer(matches[-1].group(1))

    boxed = list(_BOXED_RE.finditer(raw))
    if boxed:
        return _normalize_answer(boxed[-1].group(1))

    # Prefer the last line that looks like a lone integer / NaN.
    for line in reversed(raw.strip().splitlines()):
        s = line.strip().rstrip(".")
        if not s:
            continue
        if s.lower() in {"nan", "none", "n/a"}:
            return "nan"
        if re.fullmatch(r"-?\d+", s):
            return s

    ints = list(_INT_RE.finditer(raw))
    if ints:
        return ints[-1].group(1)
    return None


def _normalize_answer(value: str) -> str:
    s = value.strip().strip(".")
    s = re.sub(r"\s+", " ", s)
    if s.lower() in {"nan", "none", "n/a"}:
        return "nan"
    # Drop surrounding $ / \text{} when trivial.
    s = s.strip("$")
    m = re.fullmatch(r"\\text\{([^{}]+)\}", s)
    if m:
        s = m.group(1).strip()
    if re.fullmatch(r"-?\d+", s):
        return s
    return s


def agreement_to_bool(answers: Sequence[str | None]) -> bool:
    """True iff every answer is non-None and equals the first."""
    if not answers:
        return False
    if any(a is None for a in answers):
        return False
    head = answers[0]
    return all(a == head for a in answers)


def gpu_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _build_prompt(tokenizer: Any, problem_text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{SYSTEM_PROMPT}\n\n{problem_text}"


def _load_hf_model(model_id: str) -> tuple[Any, Any]:
    """Load tokenizer + causal LM from the offline HF cache only (no download)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def _generate_one(
    tokenizer: Any,
    model: Any,
    problem_text: str,
    *,
    max_new_tokens: int,
    max_prompt_tokens: int,
) -> str:
    import torch

    prompt = _build_prompt(tokenizer, problem_text)
    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    )
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    prompt_len = int(model_inputs["input_ids"].shape[-1])

    with torch.no_grad():
        out = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _predict_with_gpu(
    model_id: str,
    problems: Sequence[str],
    *,
    k: int,
    max_new_tokens: int,
    max_prompt_tokens: int,
) -> list[bool]:
    tokenizer, model = _load_hf_model(model_id)
    votes: list[bool] = []
    for problem in problems:
        prompts = [problem, *light_transforms(problem, k=k)]
        answers: list[str | None] = []
        for p in prompts:
            try:
                text = _generate_one(
                    tokenizer,
                    model,
                    p,
                    max_new_tokens=max_new_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                )
                answers.append(extract_final_answer(text))
            except Exception:
                answers.append(None)
        votes.append(agreement_to_bool(answers))
    return votes


def counterfactual_stability_votes(
    model_id: str,
    problems: Sequence[str],
    *,
    k: int = DEFAULT_K,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> list[bool] | None:
    """Predict robust when original and K transformed answers agree.

    Returns ``None`` on CPU / missing models so the cascade can abort cleanly.
    On CUDA + local cache: greedy-decode original + ``light_transforms``, extract
    final answers, mark True iff all parse and agree.
    """
    if not problems:
        return []
    if not gpu_ready():
        return None
    try:
        return _predict_with_gpu(
            model_id,
            problems,
            k=k,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
        )
    except Exception:
        # Missing local weights, OOM, tokenizer issues, etc.
        return None


def fuse_probe_with_cf_veto(
    probe_votes: Sequence[bool],
    cf_votes: Sequence[bool],
) -> list[bool]:
    """Agreement veto: True only if probe says True AND CF answers agree.

    Keeps native bools; length must match. CF False (disagreement / unparseable)
    forces False even when the probe is bullish — the failure mode of 949901.
    """
    if len(probe_votes) != len(cf_votes):
        raise ValueError(
            f"length mismatch: probe={len(probe_votes)} cf={len(cf_votes)}"
        )
    return [bool(p and c) for p, c in zip(probe_votes, cf_votes)]
