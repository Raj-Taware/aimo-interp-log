# Results archive

Frozen copies of everything needed to reproduce the numbers in this branch's
docs without a GPU. Working copies live in git-ignored `data/`, `artifacts/`
and `dist/`.

## Feature dumps

Each `.npz` holds `features` (cases x features), `labels`, `problems`,
`columns`, `model_id`, and (newer dumps) `labels_from`. All runs: Kaggle T4x2,
fp16, the 137 DeepSeek-labelled problems of `aimo-interp/augmented-sample-math-agg`.

| Folder | Commit | Features | transformers | Encoding | Use |
| --- | --- | ---: | --- | --- | --- |
| `2026-09-28_v2-features_tf-unpinned_batched` | `23de22e` | 87 | Kaggle default | batched | v2 artifact (submission 951038) |
| `2026-09-29_v2-features_tf5.13_batched_borrowed-problems` | `3181d0c` | 87 | 5.13.0 | batched | v2 per-model calibration |
| `2026-09-29_v3-features_tf5.13_batched` | `5df87b1` | 113 | 5.13.0 | batched | v3 evaluation; noise study |
| `2026-09-29_v3-features_tf5.13_unbatched-fp16` | `2058aba` | 113 | 5.13.0 | one prompt per pass | **v3 artifact**; noise study |

**Labels are DeepSeek's in every dump.** For Qwen3.5-4B, Skywork-OR1-Math-7B
and Olmo-3-7B-Think, `labels_from` says so: those rows are borrowed labels
for unlabelled analysis (scale calibration, shared-fragility AUC), never
training targets for those models.

The first two folders differ from the last two in library version; the last
two differ only in batching. That pair of contrasts is the evidence in
[`docs/measurement-noise.md`](../docs/measurement-noise.md): run-to-run r = 0.62
across library versions, 0.997+ across batching.

## Artifacts (v3)

`artifacts-v3/` — one calibrated probe per checkpoint, written by
`scripts/calibrate_per_model.py --subset v3new` from the unbatched-fp16 dumps.
`calibration_report.json` records the chosen hyperparameters (k=20, C=0.01),
the selected features, and the nested-CV estimate (balanced 0.643, AUC 0.682).

## Submissions

| File | Tag | sha256 (prefix) | Official |
| --- | --- | --- | --- |
| `solution-self-perturb-small-v2.zip` | `v2-calibrated-submission` | `2b8ebf4c15830380` | 951038: 0.7143 (10/14) |
| `solution-self-perturb-small-v3.zip` | `v3-multidistract-submission` | `cde7b78257fd5e86` | 0.50 (all predicted robust) |
| `solution-self-perturb-small-v3b.zip` | `v3b-batch-median` | see `git log` | 955589: 0.4286 |
| `solution-self-perturb-small-v4.zip` | `v4-repaired-tokenizer` | `8ccca29cfe53dca1` | 968513: 0.4286 |

Reproduce the v3 evaluation from this folder:

```bash
python scripts/evaluate.py results/feature-dumps/2026-09-29_v3-features_tf5.13_unbatched-fp16/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz --subset v3new
```
