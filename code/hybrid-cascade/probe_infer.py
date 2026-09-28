"""Per-model last-token linear probes for the four Small Models Track IDs.

Mirrors the trained-probe baseline contract (final prompt-token @ selected
layer → policy gate ⇒ robust) but is cascade-aware: missing artifact or no
GPU/local model ⇒ ``None`` (skip), never a raised exception that would
invalidate the batch.

Decision policies (see ``DECISION_POLICY``):
- ``legacy`` — mean margin >= 0 (Codabench 949901 → 0.4286; over-predicted True)
- ``conservative`` — high True bar only (949937 → 0.5714; best so far)
- ``symmetric_confident`` — ±margin bars + stronger seed agreement; cascade
  allows True-rate up to 0.50 on balanced hidden set (candidate to beat 0.571)

Public val-sample labels are noisy proxies of hidden H — not a reconstruction
target. Artifacts should be trained offline per Competition model id.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SMALL_TRACK_MODEL_IDS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3.5-4B",
        "Skywork/Skywork-OR1-Math-7B",
        "allenai/Olmo-3-7B-Think",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    }
)

RECOMMENDED_STRATEGY = "best_layer_mean_margin"

# ---------------------------------------------------------------------------
# Decision policy (single switch; cascade reads the same name)
# ---------------------------------------------------------------------------
# Active candidate for Codabench A/B vs 949937 (0.5714). Flip to "conservative"
# to restore the prior winning config without code surgery.
DECISION_POLICY = "symmetric_confident"

POLICY_THRESHOLDS: dict[str, dict[str, float | None]] = {
    # Codabench 949901 — mean margin >= 0
    "legacy": {
        "true_margin_threshold": 0.0,
        "false_margin_threshold": None,
        "min_agree_frac": 0.0,
    },
    # Codabench 949937 — asymmetric True gate (best official so far)
    "conservative": {
        "true_margin_threshold": 0.5,
        "false_margin_threshold": None,
        "min_agree_frac": 0.75,
    },
    # New: soften True bar, harden seed agreement, gate False symmetrically.
    # Hypothesis vs 0.571: catch more genuine Trues (τ 0.5→0.30) while requiring
    # ~90% seed agreement for precision; ±τ keeps the rule balanced-set-native.
    "symmetric_confident": {
        "true_margin_threshold": 0.30,
        "false_margin_threshold": -0.30,
        "min_agree_frac": 0.90,
    },
}

if DECISION_POLICY not in POLICY_THRESHOLDS:
    raise RuntimeError(f"unknown DECISION_POLICY {DECISION_POLICY!r}")

_active = POLICY_THRESHOLDS[DECISION_POLICY]

# Back-compat aliases used by tests / README / cascade.
CONSERVATIVE = DECISION_POLICY != "legacy"
TRUE_MARGIN_THRESHOLD = float(_active["true_margin_threshold"] or 0.0)
FALSE_MARGIN_THRESHOLD = (
    None
    if _active["false_margin_threshold"] is None
    else float(_active["false_margin_threshold"])
)
MIN_POSITIVE_MARGIN_FRAC = float(_active["min_agree_frac"] or 0.0)
MIN_AGREE_FRAC = MIN_POSITIVE_MARGIN_FRAC  # alias (True and False sides)


@dataclass(frozen=True)
class ProbeArtifactRef:
    path: Path
    model_id: str


@dataclass(frozen=True)
class ProbeArtifact:
    """Loaded schema-v2 ensemble pickle (same contract as trained-probe)."""

    model_id: str
    system_prompt: str
    data: dict[str, Any]
    path: Path


def _safe_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def artifact_candidates(model_id: str, root: Path | None = None) -> list[Path]:
    base = root or Path(__file__).resolve().parent
    artifacts = base / "artifacts"
    safe = _safe_model_id(model_id)
    names: list[Path] = []
    if safe:
        names.extend(
            [
                artifacts / f"probe_{safe}.pkl",
                artifacts / f"{safe}.pkl",
            ]
        )
    names.append(artifacts / "probe_artifact.pkl")
    return names


def find_probe_artifact(model_id: str, root: Path | None = None) -> ProbeArtifactRef | None:
    for path in artifact_candidates(model_id, root=root):
        if path.is_file():
            return ProbeArtifactRef(path=path, model_id=model_id)
    return None


def load_pickle_artifact(path: Path) -> ProbeArtifact:
    """Load and validate a schema-v2 probe-ensemble pickle.

    Raises ``RuntimeError`` on schema mismatch so callers can skip the stage.
    """
    with path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise RuntimeError(f"{path.name} must contain a dictionary artifact")
    if data.get("schema_version") != 2:
        raise RuntimeError(
            f"{path.name} has unsupported schema_version {data.get('schema_version')!r}"
        )
    if data.get("artifact_type") != "all_folds_layers_seed_probe_ensemble":
        raise RuntimeError(
            f"{path.name} has unsupported artifact_type {data.get('artifact_type')!r}"
        )
    if not data.get("groups"):
        raise RuntimeError(f"{path.name} contains no probe groups")
    if "best_layer_index" not in data:
        raise RuntimeError(f"{path.name} is missing best_layer_index")

    strategy = data.get("recommended_strategy") or {}
    if strategy and strategy.get("name") != RECOMMENDED_STRATEGY:
        raise RuntimeError(
            f"{path.name} recommends unsupported strategy {strategy.get('name')!r}"
        )

    return ProbeArtifact(
        model_id=str(data.get("model_id") or ""),
        system_prompt=str(data.get("system_prompt") or ""),
        data=data,
        path=path,
    )


def load_probe_for_model(model_id: str, root: Path | None = None) -> ProbeArtifact | None:
    """Find and load the matching per-model artifact, or ``None`` if absent/bad."""
    ref = find_probe_artifact(model_id, root=root)
    if ref is None:
        return None
    try:
        return load_pickle_artifact(ref.path)
    except Exception:
        return None


def gpu_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _layer_index(artifact: ProbeArtifact) -> int:
    strategy = artifact.data.get("recommended_strategy") or {}
    return int(strategy.get("layer_index", artifact.data["best_layer_index"]))


def ensemble_margins(vector: Any, artifact: ProbeArtifact) -> list[float]:
    """Signed margins for every seed/fold probe at the selected layer."""
    import numpy as np

    layer_index = _layer_index(artifact)
    margins: list[float] = []
    for group in artifact.data["groups"]:
        probes = group.get("probes", {})
        probe = probes.get(layer_index, probes.get(str(layer_index)))
        if probe is None:
            continue

        weights = np.asarray(probe["weights"], dtype=np.float32)
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        if weights.ndim != 2 or weights.shape[1] != vector.shape[0]:
            raise RuntimeError(
                f"probe dimension mismatch: got {vector.shape[0]}, expected {weights.shape[-1]}"
            )

        bias = np.asarray(probe["bias"], dtype=np.float32).reshape(-1)
        threshold = np.asarray(probe["threshold"], dtype=np.float32).reshape(-1)
        if bias.size != weights.shape[0] or threshold.size != weights.shape[0]:
            raise RuntimeError("probe ensemble arrays have inconsistent seed dimensions")

        margins.extend((weights @ vector + bias - threshold).astype(float).tolist())

    if not margins:
        raise RuntimeError(f"artifact contains no probes for layer {layer_index}")
    return margins


def mean_ensemble_margin(vector: Any, artifact: ProbeArtifact) -> float:
    """Average signed margin across every probe in the ensemble (trained-probe)."""
    import numpy as np

    margins = ensemble_margins(vector, artifact)
    return float(np.mean(np.asarray(margins, dtype=np.float64)))


def decide_robust_from_margins(
    margins: Sequence[float],
    *,
    policy: str | None = None,
    conservative: bool | None = None,
    margin_threshold: float | None = None,
    min_positive_frac: float | None = None,
) -> bool:
    """Map seed margins → native bool under the selected decision policy.

    Policies
    --------
    ``legacy``
        True iff mean(margin) >= 0.
    ``conservative``
        True only if mean >= ``true_margin_threshold`` AND >= ``min_agree_frac``
        of seed margins are >= 0. (Asymmetric; prefers False.)
    ``symmetric_confident``
        True if mean >= +τ AND pos_frac >= φ;
        False if mean <= −τ AND neg_frac >= φ;
        dead zone (−τ, +τ) → False (abstain-as-False on balanced set).

    Back-compat: ``conservative=False`` forces legacy; ``conservative=True``
    forces the conservative policy (unless ``policy`` is set). Optional
    ``margin_threshold`` / ``min_positive_frac`` override the True-side bars.
    Empty margins → False (never invent True).
    """
    if not margins:
        return False
    import numpy as np

    if policy is not None:
        pol = policy
    elif conservative is False:
        pol = "legacy"
    elif conservative is True:
        pol = "conservative"
    else:
        pol = DECISION_POLICY

    if pol not in POLICY_THRESHOLDS:
        raise ValueError(f"unknown policy {pol!r}")

    cfg = POLICY_THRESHOLDS[pol]
    true_thr = (
        float(margin_threshold)
        if margin_threshold is not None
        else float(cfg["true_margin_threshold"] or 0.0)
    )
    false_thr = cfg["false_margin_threshold"]
    min_agree = (
        float(min_positive_frac)
        if min_positive_frac is not None
        else float(cfg["min_agree_frac"] or 0.0)
    )

    arr = np.asarray(list(margins), dtype=np.float64)
    mean_m = float(np.mean(arr))
    pos_frac = float(np.mean(arr >= 0.0))
    neg_frac = float(np.mean(arr <= 0.0))

    if pol == "legacy":
        return bool(mean_m >= 0.0)

    if pol == "conservative":
        return bool(mean_m >= true_thr and pos_frac >= min_agree)

    # symmetric_confident (and any future ±τ policy using false_margin_threshold)
    if mean_m >= true_thr and pos_frac >= min_agree:
        return True
    if false_thr is not None and mean_m <= float(false_thr) and neg_frac >= min_agree:
        return False
    # Dead zone / weak agreement → False (prefer not inventing True on balanced H)
    return False


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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def _build_prompt(tokenizer: Any, problem_text: str, system_prompt: str) -> str:
    if system_prompt and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{problem_text}" if system_prompt else problem_text


def _encode_problem(problem_text: str, artifact: ProbeArtifact, tokenizer: Any, model: Any) -> Any:
    """Last prompt-token hidden state at the probe layer (float32 numpy)."""
    import torch

    prompt = _build_prompt(tokenizer, problem_text, artifact.system_prompt)
    model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    model_inputs = {name: value.to(model.device) for name, value in model_inputs.items()}

    with torch.no_grad():
        output = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("model forward pass returned no hidden states")

    layer_index = _layer_index(artifact)
    if not -len(hidden_states) <= layer_index < len(hidden_states):
        raise RuntimeError(
            f"probe layer {layer_index} is outside model hidden-state range "
            f"0..{len(hidden_states) - 1}"
        )
    return hidden_states[layer_index][0, -1, :].detach().float().cpu().numpy()


def _predict_with_gpu(artifact: ProbeArtifact, problems: Sequence[str]) -> list[bool]:
    """Run HF encode + mean-ensemble margin on CUDA. Raises on any failure."""
    import numpy as np

    # Prefer the artifact's recorded HF id (training id) over the caller's alias.
    hf_id = str(artifact.data.get("hf_model_id") or artifact.model_id or "")
    if not hf_id:
        raise RuntimeError("artifact missing model_id / hf_model_id")

    tokenizer, model = _load_hf_model(hf_id)
    out: list[bool] = []
    for problem_text in problems:
        vector = np.asarray(
            _encode_problem(problem_text, artifact, tokenizer, model),
            dtype=np.float32,
        )
        margins = ensemble_margins(vector, artifact)
        out.append(decide_robust_from_margins(margins))
    return out


def last_token_probe_votes(
    model_id: str,
    problems: Sequence[str],
) -> list[bool] | None:
    """Score last-token probes or return ``None`` to skip this cascade stage.

    On CPU / without a matching per-model artifact / without a local HF cache,
    returns ``None`` so the cascade falls through without invalidating coverage.
    Never invents GPU predictions on a CPU box.
    """
    if not problems:
        return []

    artifact = load_probe_for_model(model_id)
    if artifact is None:
        return None
    if not gpu_ready():
        return None

    try:
        return _predict_with_gpu(artifact, problems)
    except Exception:
        # Missing local weights, OOM, schema mismatch at score time, etc.
        return None


def describe_expected_artifact() -> dict[str, Any]:
    """Document expected on-disk schema for GPU training/export."""
    return {
        "schema_version": 2,
        "artifact_type": "all_folds_layers_seed_probe_ensemble",
        "recommended_strategy": "best_layer_mean_margin",
        "decision_policy": DECISION_POLICY,
        "keys": [
            "model_id",
            "system_prompt",
            "best_layer_index",
            "groups[*].probes[layer].{weights,bias,threshold}",
        ],
        "small_track_model_ids": sorted(SMALL_TRACK_MODEL_IDS),
    }
