# Self-perturbation probes for the AIMO Interpretability Challenge

This repository holds the code, data and results behind the report in `report/`. The work targets the Small Models track of the [AIMO Interpretability Challenge](https://aimo-interp.github.io/).

## Method

The organisers publish the perturbation families but not the perturbed problems. Five of the families are plain string edits, so the probe makes its own perturbations of each problem: rename, numerals, typos, distract, rephrase and number jitter. Each prompt ends in `The final answer is $\boxed{`, and one fp16 forward pass per prompt gives the last-token hidden state at six depths. The features are the cosine and norm ratio of each perturbed state against the original (72 features), plus 15 features from the next-token distribution and the perturbation flags. No answer is generated.

A logistic classifier is trained on the 137 labelled DeepSeek-R1-0528-Qwen3-8B problems. Each of the four models gets its own artifact, which standardises the features with that model's statistics. Its threshold is the median score over the 173 unlabelled AIME/AMC problems in `submission/data/pool_unlabelled.parquet`.

## Results on the public Small Models test

All bundles are in `submission/results/submissions/`.

| Bundle | Codabench ID | Accuracy |
|---|---|---:|
| `solution-v2b-poolthr-small.zip` (full probe) | 1014195 | 0.71 |
| `solution-v2b-ablate-wbonly-small.zip` (hidden-state block only) | 1026158 | 0.71 |
| `solution-v2b-ablate-nowb-small.zip` (no hidden-state block) | 1019496 | 0.43 |
| `solution-v2-diag-small.zip` (v2b with per-case margin logging) | 1014184 | 0.71 |
| `solution-self-perturb-small-v2.zip` | 951038 | 0.71 |
| `solution-self-perturb-small-v3b.zip` | 955589 | 0.43 |
| `solution-self-perturb-small-v4.zip` | 968513 | 0.43 |

The two ablation bundles are the v2b bundle with the dropped block's coefficients set to zero and each threshold set again at the pool median. The public test has 14 cases, so the ablation difference is not statistically significant (exact McNemar, two-sided p = 0.29). The other ZIPs in that folder are earlier versions.

## Reproducing the submission

Run these scripts from `submission/`:

1. `scripts/dump_features.py` extracts the features for one model. `scripts/make_kernel.py` wraps it in a Kaggle kernel.
2. `scripts/calibrate_per_model.py` trains on the labelled model and writes one artifact per checkpoint.
3. `scripts/set_thresholds.py` sets each model's threshold at the median margin over the unlabelled pool. Its docstring lists the commands.
4. `scripts/build_bundle.py` packages `solutions/self-perturb/` and the artifacts into a Codabench ZIP.
5. `scripts/make_gpu_check.py` makes a Kaggle kernel that runs the exact ZIP on all four models.
6. `scripts/ablate_whitebox.py` runs the offline white-box ablation with nested CV and length and shuffled-label controls.

`scripts/evaluate.py` gives nested-CV estimates with a shuffled-label null for any feature dump. Prompts must not be batched, because batching changes the hidden-state features. A full 14-case run over the four models took about 83 s on the challenge's evaluation machine. Offline features were computed on Kaggle with 2×T4 GPUs, torch 2.11.0 and transformers 5.13.0, in fp16. Appendix A of the report lists the same details.

## Follow-up studies

The studies in Sections 5 and 6 of the report are in `self-perturb/`. Each study follows the same steps. We named what was failing, chose a method to address it, picked a small sample, and wrote the pass criteria before the run. Task specs for T-001 to T-021 are in `collab/tasks/`, with reports in `collab/reports/`. For later studies the pass criteria are in the docstring of the `eval_*` script. All outputs are in `self-perturb/results/grok/T-0xx/`.

| Study | Scripts in `self-perturb/scripts/` | Result |
|---|---|---|
| T-022 answer-recall contrast | `dump_recall.py`, `eval_recall.py` | Failed: models recall 3 to 12 of 173 pool problems |
| T-023 gradient×input on prose vs. mathematics | `dump_saliency.py`, `eval_saliency.py` | Failed on train-main-v2 (AUC 0.36) |
| T-024 activation patching at the answer position | `dump_patch.py`, `eval_patch.py` | Did not predict labels. Shows where the perturbation's effect reaches the answer position |
| T-025 drift along a replayed reasoning trace | `dump_trace.py`, `eval_trace.py` | Failed (AUC 0.30 and 0.59) |
| T-026 length-matched gate | `eval_length_matched.py` | No verdict: the positive control did not pass |
| T-027 out-of-distribution arithmetic set | `dump_ood.py` | Failed its pilot: answers hit the 1024-token cap |

The positive control and the white-box ablation are in `self-perturb/results/grok/positive-control/` and `self-perturb/results/grok/ablation-v2b/`. `make_kernel_t022.py` builds the Kaggle kernel for T-022. `make_kernel_t023.py` builds the later ones, with `--task` and `--dump` choosing the study.

## Layout

| Path | Contents |
|---|---|
| `report/` | Report source (`report.tex`), PDF, and the script that draws Figure 2 |
| `submission/src/selfperturb/` | Perturbations and feature extraction |
| `submission/solutions/self-perturb/` | The Codabench solution code |
| `submission/solutions/behave/`, `submission/solutions/hybrid/` | Later behavioural and hybrid variants (v5) |
| `submission/results/` | Feature dumps, artifacts and bundles |
| `submission/docs/` | Notes on a DeepSeek tokenizer bug, measurement noise, and a review of the earlier attempt |
| `self-perturb/` | Follow-up studies, with shared code in `src/` and outputs in `results/grok/` |
| `collab/` | Task board, task specs and reports for the studies |
| `docs/`, `LOG.md` | Lab notebook and chronological log |
| `code/`, `artifacts/`, `submissions/`, `training/`, `notes/`, `tests/` | An earlier attempt, a hybrid cascade of last-token probes. See `code/hybrid-cascade/README.md` |

## Earlier attempt

Before the self-perturbation probe, we tried a budget-aware cascade of per-model last-token probes. Its official scores were:

| Codabench ID | Method | Accuracy |
|---|---|---:|
| 949901 | Hybrid cascade, default policy | 0.4286 |
| 949937 | Conservative hybrid | 0.5714 |
| 949966 | Symmetric confident policy | 0.5000 |
| 949938 | All-False control | 0.5000 |
| 949942 | Always-True control | 0.5000 |

The two controls both scored 0.50, so the hidden labels are balanced.

## Not included

The repository has no passwords, API tokens or model weights, and no competition test data. `submission/scripts/codabench_submit.py` asks for a password when it runs.

## Sources

- Challenge: https://aimo-interp.github.io/
- Starter kit: https://github.com/aimo-interp/getting-started
- Proposal paper: arXiv 2607.13899
