# AIMO Interpretability Challenge — small-models track

Entry targeting the **Small Models Track** and the **write-up prize**.
Deadline: submissions **Nov 1 2026**, technical report **Nov 15 2026**.

## Thesis

Everyone gets the same handicap: the organizers' perturbations are hidden, so
the field will treat the original problem as all there is and hunt for a signal
inside it. But the perturbation *families* are published — rephrase, rename,
domain, distract, typos, expert edit, no-solution — and five of them are plain
string manipulation.

So: **recreate the perturbations ourselves and measure how far the model's
internal state moves.** Not whether the answer stays right, which would need
generation we cannot afford, but whether the representation stays put.

A model that has parsed the problem should encode a renamed version almost
identically, and should react when a quantity changes. One leaning on surface
patterns does the opposite. That asymmetry is the feature.

Cost is seven prompt-only forward passes per case, no generation — roughly two
orders of magnitude under the generation-based uncertainty baseline, which also
wins the runtime tiebreak.

## Official Codabench scores (Small Model Track)

Hidden labels are balanced: all-False and always-True both score 0.50 (other
agent's controls 949938 / 949942). The board is ~14 cases, so one case is
~7 points.

| Submission | Method | Accuracy | Coverage | Invalid |
| --- | --- | ---: | ---: | ---: |
| 949937 | other agent's conservative hybrid probes | 0.5714 (8/14) | 1.0 | 0 |
| **951038** | **self-perturb v2, per-model calibrated** (tag `v2-calibrated-submission`) | **0.7143 (10/14)** | 1.0 | 0 |
| (2026-10-02) | self-perturb v3, multi-distractor, env-matched (tag `v3-multidistract-submission`) | 0.5000 (7/14) — **all 14 predicted robust** | 1.0 | 0 |
| 955589 | v3b: v3 + per-batch median threshold (tag `v3b-batch-median`) | 0.4286 (6/14) | 1.0 | 0 |
| 968513 | v4: v3b recipe, repaired DeepSeek tokenizer (tag `v4-repaired-tokenizer`) | 0.4286 (6/14) | 1.0 | 0 |

Two small sets disagree. On 34 labelled train-main-v2 rows (balanced accuracy):
v2 0.475, v3b 0.692, v4 0.692 — the reverse of the leaderboard. Labels look
problem-level, so 14 cases are only a handful of distinct problems and 34
rows only 20; neither set can rank these versions.

P(>=10/14 by chance) = 1471/16384 ~= 0.09. Offline estimate beforehand:
DeepSeek OOF balanced accuracy 0.648 (~9/14). Consistent, not conclusive.

**v3 failure:** competition problems are ~3x longer than the MATH training
problems (median 54 vs 16 words); v3's margin tracks length, so every case
cleared the training-median threshold. v3b splits each model's batch at its
own median instead (label-free; hidden labels are balanced). Replay on 17
unseen problems with all margins shifted positive: old threshold 17/17
robust, v3b 8/17 with balanced accuracy 0.80.

## Layout

```text
src/selfperturb/perturbations.py  deterministic stdlib perturbations, LaTeX-safe
src/selfperturb/features.py       drift, pooled drift, multi-distractor, KL features
scripts/dump_features.py          per-checkpoint feature dump, resumable
scripts/evaluate.py               nested CV + shuffled-label null; --subset v2/v3new
scripts/calibrate_per_model.py    train once, calibrate one artifact per checkpoint
scripts/train_probe.py            earlier grouped-CV trainer (superseded)
scripts/build_bundle.py           package the Codabench ZIP
scripts/make_kernel.py            Kaggle kernel for feature dumps
scripts/make_gpu_check.py         Kaggle kernel running the exact ZIP on all models
scripts/codabench_submit.py       user-run API submission (hidden password prompt)
solutions/self-perturb/           the Codabench bundle
results/                          frozen dumps, artifacts, submitted ZIPs
docs/                             review of the other agent's log; noise study
getting-started/                  organizers' repo (harness, baselines)
baselines/                        organizers' probe/kernel baselines
```

## Evaluation environment, in short

One RTX PRO 6000 (96 GB), 3600 s for *all* models and problems together,
offline, no `pip install`. Available: torch 2.12.1, transformers 5.13.0,
sklearn 1.8.0, numpy, pandas, joblib. Nothing else.

Small-track checkpoints: `Qwen/Qwen3.5-4B`, `Skywork/Skywork-OR1-Math-7B`,
`allenai/Olmo-3-7B-Think`, `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.

## Compute plan

| Hardware | Budget | Use |
| --- | --- | --- |
| Kaggle T4 (16 GB) | 30 h/week | Qwen3.5-4B in fp16; all probe training |
| Kaggle 2×T4 (32 GB) | same pool | 7–8B checkpoints if the H100 is unavailable |
| Lightning H100 | 4 slots × 5 h | bulk dumps for the three 7–8B checkpoints, SAE work |

Every script checkpoints its progress, because the H100 arrives in fixed slots.

## Open questions for the organizers

1. Where is the full training set (~150 problems × models)? The public dumps
   are small and mostly cover models that are *not* in the competition phase.
2. Is there a submission ZIP size limit? This decides whether SAE-derived
   artifacts can ship at all.
3. Does "do not submit model weights" forbid bundling third-party SAE weights
   trained on a competition checkpoint's base model?
4. Is a self-generated perturbation of the input problem within the rules?
   (Reading the rules, yes — it uses only the given problem and the given
   model — but worth confirming before it becomes the whole method.)
