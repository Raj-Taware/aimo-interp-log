# %% Cell 0: title / overview
"""
Kaggle GPU notebook-style script: last-token linear probes for AIMO Small Models Track.

Target artifact (hybrid-cascade probe_infer / trained-probe schema-v2):
  - schema_version = 2
  - artifact_type  = "all_folds_layers_seed_probe_ensemble"
  - recommended_strategy.name = "best_layer_mean_margin"
  - Inference rule: mean(weights @ h + bias - threshold) >= 0  ⇒  robust
    (LogisticRegression decision score uses threshold=0.0)

Outputs (under /kaggle/working/probe_artifacts/):
  probe_<safe_model_id>.pkl   e.g. probe_Qwen_Qwen3.5-4B.pkl

Data expected (any one path works):
  Columns needed:
    - problem text: problem_full | problem | original_problem
    - label:        is_robust | model_is_robust  (bool / 0-1)
    - optional:     id, model_id, problem_id (grouping uses problem text)
  Preferred CSV: val_sample_enriched.csv
    Upload as a Kaggle Dataset (e.g. "aimo-val-sample-enriched") and attach it,
    OR place at /kaggle/working/val_sample_enriched.csv.
  Also accepts val-sample jsonl pair:
    .../input/cases.jsonl  +  .../reference/labels.jsonl
    (id, model_id, problem) + (id, is_robust)

Public val-sample labels are noisy proxies of hidden H — not reconstruction targets.
Prints ONLY measured train/CV numbers from this run; never invents metrics.
"""

# %% Cell 1: verify CUDA
from __future__ import annotations

import os
import sys

print("=== Cell 1: CUDA check ===")
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "torch is required on the Kaggle GPU image but was not importable: "
        f"{exc}"
    ) from exc

print(f"torch.__version__ = {torch.__version__}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is required for this probe-training script. "
        "Switch the Kaggle notebook Accelerator to GPU (T4/P100/etc.) and re-run."
    )
print(f"cuda device count = {torch.cuda.device_count()}")
print(f"cuda device 0     = {torch.cuda.get_device_name(0)}")
print(f"cuda capability   = {torch.cuda.get_device_capability(0)}")

# %% Cell 2: install / upgrade deps (torch already present)
print("\n=== Cell 2: deps ===")
import subprocess


def _pip_install(*pkgs: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


# Do NOT reinstall torch; only packages commonly missing / stale on Kaggle.
_pip_install("transformers", "accelerate", "scikit-learn", "joblib", "pandas")

import json
import pickle
import re
import gc
from pathlib import Path
from typing import Any

import joblib  # noqa: F401  — ensure import works for downstream hybrid-cascade
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"transformers OK; sklearn OK; pandas={pd.__version__}; numpy={np.__version__}")

# %% Cell 3: config (Small Models Track IDs + OOM-safe knobs)
print("\n=== Cell 3: config ===")

SMALL_MODELS: list[str] = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]

# Train the first Small model first; set to SMALL_MODELS to run all four.
TRAIN_MODELS: list[str] = SMALL_MODELS[:1]

SYSTEM_PROMPT = (
    "You are a precise math problem solver. Solve the given problem step by step, "
    "then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)

# OOM-safe defaults for Kaggle T4/P100 (~15–16 GB).
BATCH_SIZE = 1
MAX_LENGTH = 512  # truncate long prompts; clear hard cap
SEEDS = [42, 43, 44]
N_CV_SPLITS = 4
LR_C = 1.0
LR_MAX_ITER = 2000
# Candidate layer fractions of (n_hidden_states - 1); resolved after first forward.
LAYER_FRACS = [0.25, 0.50, 0.75, 0.90, 1.00]

OUT_DIR = Path("/kaggle/working/probe_artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = Path("/kaggle/working/probe_train_metrics.json")

print(f"SMALL_MODELS = {SMALL_MODELS}")
print(f"TRAIN_MODELS = {TRAIN_MODELS}")
print(f"BATCH_SIZE={BATCH_SIZE} MAX_LENGTH={MAX_LENGTH} SEEDS={SEEDS}")
print(f"OUT_DIR = {OUT_DIR}")

# %% Cell 4: HF auth note (gated models)
print("\n=== Cell 4: HF auth ===")


def configure_hf_token() -> None:
    """Use Kaggle Secret named HF_TOKEN / HUGGINGFACE_TOKEN if present."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("huggingface_token")
    )
    if not token:
        # Kaggle Secrets often injected via userdata
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            secrets = UserSecretsClient()
            for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "huggingface_token"):
                try:
                    token = secrets.get_secret(key)
                    if token:
                        break
                except Exception:
                    continue
        except Exception:
            token = None

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
            print("HF token configured from Kaggle Secret / env.")
        except Exception as exc:
            print(f"HF token present but login helper failed ({exc}); transformers may still pick it up via env.")
    else:
        print(
            "No HF token found. If a model is gated (401/403), add a Kaggle Secret "
            "named HF_TOKEN with a Hugging Face read token that has accepted the "
            "model license, then re-run. Settings → Secrets → Add → HF_TOKEN."
        )


configure_hf_token()

# %% Cell 5: load labels / problems (CSV or val-sample jsonl)
print("\n=== Cell 5: load data ===")

PROBLEM_COLS = ("problem_full", "problem", "original_problem")
LABEL_COLS = ("is_robust", "model_is_robust")


def _find_csv_candidates() -> list[Path]:
    candidates: list[Path] = [
        Path("/kaggle/working/val_sample_enriched.csv"),
        Path("/workspace/aimo-research-loop/kaggle_upload/val_sample_enriched.csv"),
        Path("/workspace/aimo-research-loop/artifacts/val_sample_enriched.csv"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for p in sorted(input_root.rglob("val_sample_enriched.csv")):
            candidates.append(p)
        for p in sorted(input_root.rglob("*.csv")):
            if p not in candidates:
                candidates.append(p)
    return candidates


def _find_jsonl_pair() -> tuple[Path, Path] | None:
    roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("/workspace/aimo-interp-getting-started/data/val-sample"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for cases in root.rglob("cases.jsonl"):
            # Prefer .../input/cases.jsonl with sibling ../reference/labels.jsonl
            ref = cases.parent.parent / "reference" / "labels.jsonl"
            if ref.is_file():
                return cases, ref
            alt = cases.parent / "labels.jsonl"
            if alt.is_file():
                return cases, alt
    return None


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    problem_col = next((c for c in PROBLEM_COLS if c in df.columns), None)
    label_col = next((c for c in LABEL_COLS if c in df.columns), None)
    if problem_col is None or label_col is None:
        raise ValueError(
            f"CSV must include a problem column in {PROBLEM_COLS} and a label "
            f"column in {LABEL_COLS}. Got columns={list(df.columns)}"
        )
    out = pd.DataFrame(
        {
            "id": df["id"].astype(str) if "id" in df.columns else [f"row-{i}" for i in range(len(df))],
            "model_id": df["model_id"].astype(str) if "model_id" in df.columns else "",
            "problem": df[problem_col].astype(str),
            "is_robust": df[label_col].map(lambda v: bool(v) if not isinstance(v, str) else v.strip().lower() in {"1", "true", "yes"}),
        }
    )
    out = out[out["problem"].str.len() > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError("No non-empty problem rows after normalization.")
    return out


def _load_from_jsonl(cases_path: Path, labels_path: Path) -> pd.DataFrame:
    cases = [json.loads(l) for l in cases_path.read_text().splitlines() if l.strip()]
    labels = {
        j["id"]: bool(j["is_robust"])
        for j in (json.loads(l) for l in labels_path.read_text().splitlines() if l.strip())
    }
    rows = []
    for c in cases:
        if c["id"] not in labels:
            continue
        rows.append(
            {
                "id": c["id"],
                "model_id": c.get("model_id", ""),
                "problem": c["problem"],
                "is_robust": labels[c["id"]],
            }
        )
    return pd.DataFrame(rows)


def load_training_frame() -> tuple[pd.DataFrame, str]:
    for path in _find_csv_candidates():
        if not path.is_file():
            continue
        try:
            raw = pd.read_csv(path)
            frame = _normalize_frame(raw)
            print(f"Loaded CSV: {path}  shape={frame.shape}")
            return frame, f"csv:{path}"
        except Exception as exc:
            print(f"Skip CSV {path}: {exc}")
            continue

    pair = _find_jsonl_pair()
    if pair is not None:
        cases_path, labels_path = pair
        frame = _load_from_jsonl(cases_path, labels_path)
        print(f"Loaded jsonl: {cases_path} + {labels_path}  shape={frame.shape}")
        return frame, f"jsonl:{cases_path}"

    raise FileNotFoundError(
        "Could not find training labels. Expected one of:\n"
        "  - Kaggle Dataset / upload containing val_sample_enriched.csv "
        "(columns: problem_full, is_robust; optional id, model_id)\n"
        "  - /kaggle/working/val_sample_enriched.csv\n"
        "  - val-sample cases.jsonl + labels.jsonl under /kaggle/input\n"
        "Attach the dataset in the notebook sidebar, or upload the CSV to /kaggle/working/."
    )


df = load_training_frame()[0]
n = len(df)
n_rob = int(df["is_robust"].sum())
n_groups = int(df["problem"].nunique())
print(f"MEASURED n_rows={n} n_robust={n_rob} n_not_robust={n - n_rob} base_rate_robust={n_rob / n:.6f}")
print(f"MEASURED n_unique_problems={n_groups}")

# %% Cell 6: helpers — prompt, safe id, hidden-state extraction
print("\n=== Cell 6: helpers ===")


def safe_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def build_prompt(tokenizer: Any, problem_text: str, system_prompt: str) -> str:
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


def load_model_tokenizer(model_id: str):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {model_id} dtype={dtype} ...", flush=True)
    try:
        print("tokenizer.from_pretrained ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("model.from_pretrained (device_map=auto, low_cpu_mem_usage) ...", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            print("retry from_pretrained with torch_dtype= ...", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        print("from_pretrained returned; calling eval() ...", flush=True)
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("401", "403", "gated", "authorized", "authentication", "token")):
            print(
                "HF AUTH / GATED MODEL ERROR.\n"
                "Add a Kaggle Secret named HF_TOKEN (HF read token with model access),\n"
                "accept the model license on huggingface.co, then Restart & Run All.",
                flush=True,
            )
        raise
    model.eval()
    try:
        dev = next(model.parameters()).device
        print(
            f"model ready on {dev}; cuda mem_alloc={torch.cuda.memory_allocated()/1e9:.2f}GB",
            flush=True,
        )
    except Exception:
        print("model.eval() done", flush=True)
    return tokenizer, model


def resolve_candidate_layers(n_hidden_states: int) -> list[int]:
    """Map fractions onto hidden_states indices (0 = embeddings, last = final layer)."""
    last = n_hidden_states - 1
    layers: list[int] = []
    for frac in LAYER_FRACS:
        idx = int(round(frac * last))
        idx = max(0, min(last, idx))
        layers.append(idx)
    # unique, sorted
    return sorted(set(layers))


@torch.inference_mode()
def extract_last_token_layers(
    model: Any,
    tokenizer: Any,
    problems: list[str],
    system_prompt: str,
    layer_indices: list[int] | None = None,
    max_length: int = MAX_LENGTH,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """
    Returns:
      features[layer_index] -> (n_problems, hidden_dim) float32
      resolved layer_indices
    """
    device = next(model.parameters()).device
    cache: dict[str, dict[int, np.ndarray]] = {}
    resolved_layers: list[int] | None = None
    hidden_dim: int | None = None

    for i, problem in enumerate(problems):
        if problem in cache:
            continue
        prompt = build_prompt(tokenizer, problem, system_prompt)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        try:
            out = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Retry with an even shorter cap.
            shorter = max(128, max_length // 2)
            print(f"OOM on encode; retrying with max_length={shorter}")
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=shorter,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Model returned no hidden_states")
        if resolved_layers is None:
            n_hs = len(hs)
            hidden_dim = int(hs[0].shape[-1])
            resolved_layers = (
                resolve_candidate_layers(n_hs)
                if layer_indices is None
                else [int(x) for x in layer_indices]
            )
            for li in resolved_layers:
                if not (0 <= li < n_hs):
                    raise RuntimeError(f"layer {li} out of range 0..{n_hs - 1}")
            print(
                f"MEASURED n_hidden_states={n_hs} hidden_dim={hidden_dim} "
                f"candidate_layers={resolved_layers}"
            )

        cache[problem] = {
            li: hs[li][0, -1, :].detach().float().cpu().numpy()
            for li in resolved_layers
        }
        if (i + 1) % 5 == 0 or (i + 1) == len(problems):
            print(f"  encoded unique-progress ~{len(cache)} (row {i + 1}/{len(problems)})")

    assert resolved_layers is not None
    features: dict[int, np.ndarray] = {}
    for li in resolved_layers:
        features[li] = np.stack([cache[p][li] for p in problems], axis=0).astype(np.float32)
    return features, resolved_layers


def _to_python_lists(weights: np.ndarray, bias: np.ndarray, threshold: np.ndarray) -> dict[str, Any]:
    """Schema stores plain Python lists (trained-probe README contract)."""
    return {
        "weights": np.asarray(weights, dtype=np.float64).tolist(),
        "bias": np.asarray(bias, dtype=np.float64).reshape(-1).tolist(),
        "threshold": np.asarray(threshold, dtype=np.float64).reshape(-1).tolist(),
    }


# %% Cell 7: train LogisticRegression probes with problem-grouped CV
print("\n=== Cell 7: train probes ===")


def make_splitter(y: np.ndarray, groups: np.ndarray, n_splits: int):
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups >= 3:
        splits = min(n_splits, n_unique_groups)
        print(f"Using GroupKFold n_splits={splits} (n_unique_groups={n_unique_groups})")
        return GroupKFold(n_splits=splits), True
    print(
        f"Too few problem groups ({n_unique_groups}) for GroupKFold; "
        f"falling back to StratifiedKFold"
    )
    # Stratified needs at least 2 per class ideally
    counts = np.bincount(y.astype(int))
    max_splits = int(counts.min()) if len(counts) > 1 else 2
    splits = max(2, min(n_splits, max_splits, len(y)))
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=42), False


def train_layer_probes(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seeds: list[int],
) -> dict[str, Any]:
    """Fit CV folds × seeds of LogisticRegression; return group probe packs + metrics."""
    splitter, grouped = make_splitter(y, groups, N_CV_SPLITS)
    split_iter = (
        splitter.split(X, y, groups)
        if grouped
        else splitter.split(X, y)
    )

    fold_groups: list[dict[str, Any]] = []
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    fold_val_accs: list[float] = []

    for fold_i, (train_idx, val_idx) in enumerate(split_iter):
        seed_weights: list[np.ndarray] = []
        seed_bias: list[float] = []
        seed_thr: list[float] = []
        fold_val_scores: list[np.ndarray] = []

        for seed in seeds:
            clf = LogisticRegression(
                C=LR_C,
                max_iter=LR_MAX_ITER,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            # Degenerate fold guard (single-class train)
            if len(np.unique(y[train_idx])) < 2:
                print(f"  fold={fold_i} seed={seed}: single-class train — skipping seed")
                continue
            clf.fit(X[train_idx], y[train_idx])
            w = np.asarray(clf.coef_, dtype=np.float64).reshape(-1)
            b = float(np.asarray(clf.intercept_).reshape(-1)[0])
            # threshold=0 ⇒ margin = score - 0; mean margin >= 0 matches predict class 1
            seed_weights.append(w)
            seed_bias.append(b)
            seed_thr.append(0.0)
            # decision_function on val for ensemble later
            fold_val_scores.append(clf.decision_function(X[val_idx]))

        if not seed_weights:
            continue

        W = np.stack(seed_weights, axis=0)  # (n_seeds, dim)
        B = np.asarray(seed_bias, dtype=np.float64)
        T = np.asarray(seed_thr, dtype=np.float64)

        # OOF via mean seed margin on val fold
        mean_scores = np.mean(np.stack(fold_val_scores, axis=0), axis=0)
        pred = (mean_scores >= 0.0).astype(np.int64)
        oof_pred[val_idx] = pred
        acc = float(accuracy_score(y[val_idx], pred))
        fold_val_accs.append(acc)
        print(f"  MEASURED fold={fold_i} val_acc={acc:.6f} n_train={len(train_idx)} n_val={len(val_idx)}")

        fold_groups.append(
            {
                "fold_index": fold_i,
                "seeds": list(seeds[: len(seed_weights)]),
                "aggregation": "mean_margin",
                "selection_metric": "val accuracy",
                "probes": {},  # filled by caller per layer
                "_arrays": _to_python_lists(W, B, T),
            }
        )

    covered = oof_pred >= 0
    metrics = {
        "n_folds_used": len(fold_groups),
        "fold_val_accs": fold_val_accs,
        "mean_fold_val_acc": float(np.mean(fold_val_accs)) if fold_val_accs else float("nan"),
    }
    if covered.any():
        metrics["oof_accuracy"] = float(accuracy_score(y[covered], oof_pred[covered]))
        metrics["oof_balanced_accuracy"] = float(
            balanced_accuracy_score(y[covered], oof_pred[covered])
        )
        metrics["oof_n"] = int(covered.sum())
        print(
            f"  MEASURED oof_accuracy={metrics['oof_accuracy']:.6f} "
            f"oof_balanced_accuracy={metrics['oof_balanced_accuracy']:.6f} "
            f"oof_n={metrics['oof_n']}"
        )
    print(f"  MEASURED mean_fold_val_acc={metrics['mean_fold_val_acc']:.6f}")

    # Also fit full-data probes (all seeds) for a reported train accuracy.
    full_W, full_B, full_T = [], [], []
    train_pred_votes = []
    if len(np.unique(y)) >= 2:
        for seed in seeds:
            clf = LogisticRegression(
                C=LR_C,
                max_iter=LR_MAX_ITER,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            clf.fit(X, y)
            full_W.append(np.asarray(clf.coef_, dtype=np.float64).reshape(-1))
            full_B.append(float(np.asarray(clf.intercept_).reshape(-1)[0]))
            full_T.append(0.0)
            train_pred_votes.append((clf.decision_function(X) >= 0.0).astype(np.int64))
        mean_train_pred = (
            np.mean(np.stack(train_pred_votes, axis=0), axis=0) >= 0.5
        ).astype(np.int64)
        metrics["train_accuracy_full_fit"] = float(accuracy_score(y, mean_train_pred))
        print(f"  MEASURED train_accuracy_full_fit={metrics['train_accuracy_full_fit']:.6f}")
    metrics["full_fit_arrays"] = (
        _to_python_lists(np.stack(full_W, axis=0), np.asarray(full_B), np.asarray(full_T))
        if full_W
        else None
    )
    return {"fold_groups": fold_groups, "metrics": metrics}


def build_artifact(
    model_id: str,
    layer_results: dict[int, dict[str, Any]],
    best_layer: int,
    feature_dim: int,
) -> dict[str, Any]:
    """Assemble schema-v2 ensemble pickle matching probe_infer / trained-probe."""
    # Align groups by fold index across layers.
    n_folds = max(len(layer_results[li]["fold_groups"]) for li in layer_results)
    groups: list[dict[str, Any]] = []
    for fold_i in range(n_folds):
        probes: dict[int, Any] = {}
        seeds_ref: list[int] = SEEDS
        for li, result in layer_results.items():
            folds = result["fold_groups"]
            # find matching fold_index
            match = next((g for g in folds if g["fold_index"] == fold_i), None)
            if match is None:
                continue
            packed = dict(match["_arrays"])
            packed["seeds"] = match["seeds"]
            probes[int(li)] = packed
            seeds_ref = match["seeds"]
        if not probes:
            continue
        # Prefer storing CV fold probes; also ensure best_layer present.
        groups.append(
            {
                "fold_index": fold_i,
                "seeds": seeds_ref,
                "aggregation": "mean_margin",
                "selection_metric": "val accuracy",
                "best_layer_index": int(best_layer),
                "layer_indices": sorted(probes.keys()),
                "probes": probes,
            }
        )

    # If CV produced no groups (degenerate), fall back to full-fit single group.
    if not groups:
        probes = {}
        for li, result in layer_results.items():
            arr = result["metrics"].get("full_fit_arrays")
            if arr is None:
                continue
            packed = dict(arr)
            packed["seeds"] = list(SEEDS)
            probes[int(li)] = packed
        groups.append(
            {
                "fold_index": 0,
                "seeds": list(SEEDS),
                "aggregation": "mean_margin",
                "selection_metric": "train accuracy",
                "best_layer_index": int(best_layer),
                "layer_indices": sorted(probes.keys()),
                "probes": probes,
            }
        )

    layer_scores = {
        int(li): float(layer_results[li]["metrics"]["mean_fold_val_acc"])
        for li in layer_results
    }

    return {
        "schema_version": 2,
        "artifact_type": "all_folds_layers_seed_probe_ensemble",
        "model_id": model_id,
        "hf_model_id": model_id,
        "system_prompt": SYSTEM_PROMPT,
        "target_col": "is_robust",
        "task_type": "classification",
        "aggregation": "mean_margin",
        "selection_metric": "val accuracy",
        "seeds": list(SEEDS),
        "best_layer_index": int(best_layer),
        "layer_indices": sorted(layer_results.keys()),
        "layer_scores": layer_scores,
        "feature_dim": int(feature_dim),
        "recommended_strategy": {
            "name": "best_layer_mean_margin",
            "layer_index": int(best_layer),
            "description": (
                "At inference, encode best_layer_index (last prompt token) and "
                "average score-threshold margins across every stored seed/fold "
                "probe for that layer. Robust iff mean(weights @ h + bias - threshold) >= 0."
            ),
        },
        "groups": groups,
        "threshold_rule": {
            "margin": "weights @ h + bias - threshold",
            "decision": "mean_margin >= 0 ⇒ True (robust)",
            "threshold_value_for_logreg": 0.0,
        },
    }


all_run_metrics: dict[str, Any] = {"models": {}}

for model_id in TRAIN_MODELS:
    print(f"\n######## MODEL {model_id} ########")
    safe = safe_model_id(model_id)
    tokenizer, model = load_model_tokenizer(model_id)

    problems = df["problem"].tolist()
    y = df["is_robust"].astype(int).to_numpy()
    groups_arr = pd.factorize(df["problem"])[0]

    # Encode once; collect candidate layers.
    features_by_layer, candidate_layers = extract_last_token_layers(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        system_prompt=SYSTEM_PROMPT,
        layer_indices=None,
        max_length=MAX_LENGTH,
    )
    feature_dim = int(next(iter(features_by_layer.values())).shape[1])
    print(f"MEASURED feature_dim={feature_dim}")

    # Free GPU before sklearn (CPU) fitting.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    layer_results: dict[int, dict[str, Any]] = {}
    for li in candidate_layers:
        print(f"\n-- layer {li} --")
        X = features_by_layer[li]
        layer_results[li] = train_layer_probes(X, y, groups_arr, SEEDS)

    # Pick best layer by measured mean CV fold accuracy (nan-safe).
    scored = []
    for li, result in layer_results.items():
        m = result["metrics"]["mean_fold_val_acc"]
        # Prefer OOF accuracy when available
        oof = result["metrics"].get("oof_accuracy", m)
        scored.append((li, float(oof), float(m)))
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    best_layer = scored[0][0]
    print(
        f"MEASURED best_layer_index={best_layer} "
        f"best_oof_or_mean={scored[0][1]:.6f} "
        f"layer_table={[(a, round(b, 6), round(c, 6)) for a, b, c in scored]}"
    )

    artifact = build_artifact(model_id, layer_results, best_layer, feature_dim)
    out_path = OUT_DIR / f"probe_{safe}.pkl"
    with out_path.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {out_path} size_bytes={out_path.stat().st_size}")

    # Validate load contract locally (schema checks mirroring trained-probe).
    with out_path.open("rb") as handle:
        reloaded = pickle.load(handle)
    assert reloaded["schema_version"] == 2
    assert reloaded["artifact_type"] == "all_folds_layers_seed_probe_ensemble"
    assert reloaded["recommended_strategy"]["name"] == "best_layer_mean_margin"
    assert "groups" in reloaded and reloaded["groups"]
    assert best_layer in reloaded["groups"][0]["probes"] or str(best_layer) in reloaded["groups"][0]["probes"]
    g0_probe = reloaded["groups"][0]["probes"][best_layer]
    W = np.asarray(g0_probe["weights"])
    print(
        f"MEASURED artifact_ok weights_shape={W.shape} "
        f"bias_len={len(g0_probe['bias'])} threshold={g0_probe['threshold']}"
    )

    all_run_metrics["models"][model_id] = {
        "safe_model_id": safe,
        "artifact_path": str(out_path),
        "best_layer_index": best_layer,
        "feature_dim": feature_dim,
        "candidate_layers": candidate_layers,
        "per_layer": {
            str(li): {
                k: v
                for k, v in result["metrics"].items()
                if k != "full_fit_arrays"
            }
            for li, result in layer_results.items()
        },
    }

    # Drop large feature arrays
    del features_by_layer, layer_results, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# %% Cell 8: write metrics JSON (measured only)
print("\n=== Cell 8: metrics summary ===")
all_run_metrics["n_rows"] = int(n)
all_run_metrics["n_robust"] = int(n_rob)
all_run_metrics["n_unique_problems"] = int(n_groups)
all_run_metrics["train_models"] = TRAIN_MODELS
METRICS_PATH.write_text(json.dumps(all_run_metrics, indent=2, sort_keys=True))
print(f"Wrote {METRICS_PATH}")
print(json.dumps(all_run_metrics, indent=2, sort_keys=True))

print(
    "\nDone. Copy probe_*.pkl into "
    "solutions/hybrid-cascade/artifacts/ for cascade packaging.\n"
    "Inference: last prompt token @ best_layer_index → "
    "mean(weights @ h + bias - threshold) >= 0 ⇒ robust."
)
