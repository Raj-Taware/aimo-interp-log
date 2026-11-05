# Follow-up studies

This folder holds the studies in Sections 5 and 6 of the report and the earlier work that led to the submitted probe. The submitted code itself is in `../submission/`.

Each study follows the same steps. We named what was failing, chose a method to address it, picked a small sample, and wrote the pass criteria before the run. Task specs for T-001 to T-021 are in `../collab/tasks/`, with reports in `../collab/reports/`. For later studies the pass criteria are in the docstring of the `eval_*` script. All outputs are in `results/grok/T-0xx/`.

## Studies in the report

| Study | Scripts | Result |
|---|---|---|
| T-022 answer-recall contrast | `dump_recall.py`, `eval_recall.py` | Failed: models recall 3 to 12 of 173 pool problems |
| T-023 gradient×input on prose vs. mathematics | `dump_saliency.py`, `eval_saliency.py` | Failed on train-main-v2 (AUC 0.36) |
| T-024 activation patching at the answer position | `dump_patch.py`, `eval_patch.py` | Did not predict labels. Shows where the perturbation's effect reaches the answer position |
| T-025 drift along a replayed reasoning trace | `dump_trace.py`, `eval_trace.py` | Failed (AUC 0.30 and 0.59) |
| T-026 length-matched gate | `eval_length_matched.py` | No verdict: the positive control did not pass |
| T-027 out-of-distribution arithmetic set | `dump_ood.py` | Failed its pilot: answers hit the 1024-token cap |

The positive control and the white-box ablation are in `results/grok/positive-control/` and `results/grok/ablation-v2b/`. `make_kernel_t022.py` builds the Kaggle kernel for T-022. `make_kernel_t023.py` builds the later ones, with `--task` and `--dump` choosing the study.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | `dump_*` scripts compute features on GPU; `eval_*` scripts score them |
| `results/grok/` | Outputs, one folder per study |
| `src/selfperturb/` | Shared perturbation and feature code |
| `solutions/` | Solution variants built during the studies |
| `docs/`, `tests/` | Notes and tests |
