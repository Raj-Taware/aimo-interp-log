"""v6 bundle: competence-margin (difficulty) probe, see difficulty_probe.py.

Per model: ridge on one question-only state (layer x pooling picked by AIME CV) -> AIME
problem number, standardized by that model's own (label-free) score mean / sd over the
173-problem pool. Decision: robust iff standardized difficulty < CUT, one CUT shared by all models, chosen for accuracy on train-main-v2 (43 labelled rows; its
leave-one-problem-out accuracy is reported). Writes <out>/ (solution.py,
difficulty_infer.py, selfperturb/, probe_artifacts/<model>.npz, small.txt) and <out>.zip.

    python scripts/build_v6.py --out ../../AIMO/results/submissions/solution-v6-difficulty-small

White-Box ablations (same cut procedure): ``--ablate shuffled`` fits the probe at the same
layer on permuted AIME numbers (seed 0), removing the difficulty direction; ``--ablate length``
replaces the probe by log word count (no model forward pass).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from difficulty_probe import MODELS, POOL, STATES, TRAINV2, aime_rows, key, pick_layer, ridge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HF_IDS = {"Qwen_Qwen3.5-4B": "Qwen/Qwen3.5-4B", "Skywork_Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
          "allenai_Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think",
          "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}

SOLUTION = '''"""Codabench entry point for v6 (competence-margin difficulty probe)."""

from difficulty_infer import predict_robustness


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    """Predict one native bool per problem, in the order received."""
    return predict_robustness(model_id, problems)
'''

INFER = '''"""v6 inference: one forward pass per problem, read one residual state, project on the
model's difficulty direction (ridge to AIME problem number), standardize by the model's
pool score mean / sd, robust iff below the cut.

Same prompt and state as the training dumps (scripts/dump_states.py): build_prompt with the
answer prefix, mean over prompt tokens or the last token, at one hidden-state index.
Failures fall back to FALLBACK (robust: 72% of train-main-v2 small-model rows are robust).
"""

from __future__ import annotations

import gc
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from selfperturb.features import MAX_PROMPT_TOKENS, build_prompt, load_model

ARTIFACT_DIR = "probe_artifacts"
BUDGET_SECONDS = float(os.environ.get("V6_BUDGET", "240"))  # per call; job budget 4200 s / 14 cases
FALLBACK = True


def _safe_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


@torch.no_grad()
def difficulty(tokenizer, model, artifact, problem: str) -> float:
    batch = tokenizer(build_prompt(tokenizer, problem), return_tensors="pt", truncation=True,
                      max_length=MAX_PROMPT_TOKENS)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
    h = out.hidden_states[int(artifact["layer"])][0].float().cpu()
    x = (h.mean(0) if str(artifact["pooling"]) == "mean" else h[-1]).half().float().numpy()
    z = (x - artifact["mean"]) / artifact["scale"]
    raw = float(z @ artifact["coef"] + float(artifact["intercept"]))
    return (raw - float(artifact["pool_mu"])) / float(artifact["pool_sd"])


def predict_robustness(model_id: str, problems: list[str]) -> list[bool]:
    deadline = time.time() + BUDGET_SECONDS
    preds = [FALLBACK] * len(problems)
    path = Path(__file__).resolve().parent / ARTIFACT_DIR / f"{_safe_id(model_id)}.npz"
    if not path.is_file():
        return preds
    artifact = np.load(path, allow_pickle=True)
    if str(artifact["kind"]) == "length":  # ablation control: text length only
        return [bool((np.log(len(p.split()) + 1) - float(artifact["pool_mu"])) / float(artifact["pool_sd"])
                     < float(artifact["cut"])) for p in problems]
    try:
        tokenizer, model = load_model(model_id)
    except Exception:
        return preds
    try:
        for i, problem in enumerate(problems):
            if time.time() >= deadline:
                print(f"[v6] {model_id} deadline at case {i}", file=sys.stderr, flush=True)
                break
            try:
                z = difficulty(tokenizer, model, artifact, problem)
                preds[i] = bool(z < float(artifact["cut"]))
                print(f"[v6] {model_id} case {i} z={z:.3f} robust={preds[i]}", file=sys.stderr, flush=True)
            except Exception as error:  # keep the case, log why it fell back
                preds[i] = FALLBACK
                print(f"[v6] {model_id} case {i} FALLBACK {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [bool(p) for p in preds]
'''


def best_cut(scores: np.ndarray, robust: np.ndarray) -> float:
    """Midpoint cut maximizing accuracy of 'robust iff score < cut' (ties: the middle one)."""
    s = np.sort(np.unique(scores))
    cands = np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]])
    acc = np.array([((scores < c) == robust).mean() for c in cands])
    best = np.flatnonzero(acc == acc.max())
    return float(cands[best[len(best) // 2]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ablate", choices=("shuffled", "length"), default=None)
    parser.add_argument("--cut", type=float, default=None,
                        help="fixed cut instead of the train-main-v2 fit (0 = pool mean, label-free)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    meta = {key(r["problem"]): r for r in map(json.loads, POOL.open(encoding="utf-8"))}
    exclude = {key(p) for p in pd.read_parquet(TRAINV2)["problem"]}
    arts, rows = {}, []
    for m in MODELS:
        pool = np.load(STATES / f"pool__{m}__states.npz", allow_pickle=True)
        idx, d = aime_rows([key(p) for p in pool["problems"]], meta, exclude)
        rho, pooling, li = pick_layer(pool, idx, d)
        tr = np.load(STATES / f"trainv2__{m}__states.npz", allow_pickle=True)
        ok = ~np.isnan(tr["labels"])
        if args.ablate == "length":
            ps = np.log([len(p.split()) + 1 for p in pool["problems"]])
            arts[m] = {"kind": "length", "pool_mu": float(ps.mean()), "pool_sd": float(ps.std())}
            s = (np.log([len(p.split()) + 1 for p in tr["problems"][ok]]) - ps.mean()) / ps.std()
        else:
            target = rng.permutation(d) if args.ablate == "shuffled" else d
            probe = ridge().fit(pool[pooling][idx][:, li].astype(np.float32), target)
            sc, rg = probe.named_steps["standardscaler"], probe.named_steps["ridgecv"]
            ps = probe.predict(pool[pooling][:, li].astype(np.float32))  # all 173 pool problems, no labels
            arts[m] = {"kind": args.ablate or "probe", "layer": int(pool["layers"][li]), "pooling": pooling,
                       "mean": sc.mean_.astype(np.float32), "scale": sc.scale_.astype(np.float32),
                       "coef": rg.coef_.astype(np.float32), "intercept": float(rg.intercept_),
                       "alpha": float(rg.alpha_), "aime_cv_rho": float(rho),
                       "pool_mu": float(ps.mean()), "pool_sd": float(ps.std())}
            s = (probe.predict(tr[pooling][ok][:, li].astype(np.float32)) - ps.mean()) / ps.std()
        rows += [{"model": m, "problem": key(p), "robust": bool(y), "score": float(v)}
                 for p, y, v in zip(tr["problems"][ok], tr["labels"][ok], s)]
    df = pd.DataFrame(rows)
    cut = best_cut(df["score"].values, df["robust"].values) if args.cut is None else args.cut
    loo = []  # leave-one-problem-out: cut chosen without the held-out problem
    for p in df["problem"].unique():
        held = df["problem"] == p
        c = best_cut(df.loc[~held, "score"].values, df.loc[~held, "robust"].values) if args.cut is None else args.cut
        loo += list((df.loc[held, "score"] < c) == df.loc[held, "robust"])
    summary = {"ablate": args.ablate, "fixed_cut": args.cut is not None, "cut": round(cut, 3), "train_acc": round(float(((df["score"] < cut) == df["robust"]).mean()), 3),
               "loo_problem_acc": round(float(np.mean(loo)), 3), "always_robust_acc": round(float(df["robust"].mean()), 3),
               "models": {m: {k: a[k] for k in ("kind", "layer", "pooling", "alpha", "aime_cv_rho", "pool_mu", "pool_sd")
                              if k in a} for m, a in arts.items()}}

    out = args.out
    shutil.rmtree(out, ignore_errors=True)
    (out / "probe_artifacts").mkdir(parents=True)
    shutil.copytree(ROOT / "src/selfperturb", out / "selfperturb", ignore=shutil.ignore_patterns("__pycache__"))
    (out / "solution.py").write_text(SOLUTION)
    (out / "difficulty_infer.py").write_text(INFER)
    (out / "small.txt").write_text("")
    for m, a in arts.items():
        np.savez(out / "probe_artifacts" / f"{m}.npz", cut=cut, model_id=HF_IDS[m], **a)
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2))
    with zipfile.ZipFile(out.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.name != "build_summary.json":
                z.write(f, f.relative_to(out).as_posix())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
