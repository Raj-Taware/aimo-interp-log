# Submitted probe and its ablations

This folder holds the code, features and Codabench bundles for the probe in Sections 2 and 3 of the report.

## Method

The organisers publish the perturbation families but not the perturbed problems. Five of the families are plain string edits, so the probe makes its own perturbations of each problem: rename, numerals, typos, distract, rephrase and number jitter. Each prompt ends in `The final answer is $\boxed{`, and one fp16 forward pass per prompt gives the last-token hidden state at six depths. The features are the cosine and norm ratio of each perturbed state against the original (72 features), plus 15 features from the next-token distribution and the perturbation flags. No answer is generated.

A logistic classifier is trained on the 137 labelled DeepSeek-R1-0528-Qwen3-8B problems. Each of the four models gets its own artifact, which standardises the features with that model's statistics. Its threshold is the median score over the 173 unlabelled AIME/AMC problems in `data/pool_unlabelled.parquet`.

## Bundles

All bundles are in `results/submissions/`.

| Bundle | Codabench ID | Public accuracy |
|---|---|---:|
| `solution-v2b-poolthr-small.zip` (full probe) | 1014195 | 0.71 |
| `solution-v2b-ablate-wbonly-small.zip` (hidden-state block only) | 1026158 | 0.71 |
| `solution-v2b-ablate-nowb-small.zip` (no hidden-state block) | 1019496 | 0.43 |
| `solution-v2-diag-small.zip` (v2b with per-case margin logging) | 1014184 | 0.71 |
| `solution-self-perturb-small-v2.zip` | 951038 | 0.71 |
| `solution-self-perturb-small-v3b.zip` | 955589 | 0.43 |
| `solution-self-perturb-small-v4.zip` | 968513 | 0.43 |

The two ablation bundles are the v2b bundle with the dropped block's coefficients set to zero and each threshold set again at the pool median. The other ZIPs in that folder are earlier versions.

## Reproducing

1. `scripts/dump_features.py` extracts the features for one model. `scripts/make_kernel.py` wraps it in a Kaggle kernel.
2. `scripts/calibrate_per_model.py` trains on the labelled model and writes one artifact per checkpoint.
3. `scripts/set_thresholds.py` sets each model's threshold at the median margin over the unlabelled pool. Its docstring lists the commands.
4. `scripts/build_bundle.py` packages `solutions/self-perturb/` and the artifacts into a Codabench ZIP.
5. `scripts/make_gpu_check.py` makes a Kaggle kernel that runs the exact ZIP on all four models.
6. `scripts/ablate_whitebox.py` runs the offline white-box ablation with nested CV and length and shuffled-label controls.

`scripts/evaluate.py` gives nested-CV estimates with a shuffled-label null for any feature dump. Prompts must not be batched, because batching changes the hidden-state features.

## Layout

| Path | Contents |
|---|---|
| `src/selfperturb/` | Perturbations and feature extraction |
| `solutions/self-perturb/` | The Codabench solution code |
| `solutions/behave/`, `solutions/hybrid/` | Later behavioural and hybrid variants (v5) |
| `results/` | Feature dumps, artifacts and bundles |
| `data/` | The unlabelled AIME/AMC pool |
| `docs/` | Notes on a DeepSeek tokenizer bug, measurement noise, and a review of the earlier attempt |

`scripts/codabench_submit.py` asks for a password when it runs. No credentials are stored in the repository.
