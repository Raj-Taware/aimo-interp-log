# Self-perturbation probes for the AIMO Interpretability Challenge

This repository holds the code, data and results behind the report in `report/`. The work targets the Small Models track of the [AIMO Interpretability Challenge](https://aimo-interp.github.io/).

The submitted probe predicts whether a model's answer is robust to rewording without generating an answer. It perturbs each problem six ways, runs one forward pass per prompt, and measures how far the last-token hidden state moves at six depths. A logistic classifier trained on the public DeepSeek labels turns these features into a score, and each model's threshold is the median score over 173 unlabelled AIME/AMC problems.

## Results on the public Small Models test

| Submission | Codabench ID | Accuracy |
|---|---|---:|
| Full probe (v2b) | 1014195 | 0.71 |
| Hidden-state block only | 1026158 | 0.71 |
| Without the hidden-state block | 1019496 | 0.43 |

The public test has 14 cases, so these differences are not statistically significant (exact McNemar, two-sided p = 0.29). The report gives the details and the follow-up studies.

## Layout

| Path | Contents |
|---|---|
| `report/` | Report source (`report.tex`), PDF, and the script that draws Figure 2 |
| `submission/` | Code, features and Codabench bundles for the submitted probe and its two ablations |
| `self-perturb/` | Follow-up studies: `scripts/dump_*` and `scripts/eval_*`, with outputs in `results/grok/T-0xx/` |
| `collab/` | Task board, task specs and reports for the studies |
| `docs/`, `LOG.md` | Lab notebook and chronological log |
| `code/`, `artifacts/`, `submissions/`, `training/`, `notes/`, `tests/` | An earlier attempt, a hybrid cascade of last-token probes. See `code/hybrid-cascade/README.md` |

## Reproducing

Appendix A of the report lists the scripts, bundles, training data and software versions. Paths in that appendix for the submission are relative to `submission/`. The pass criteria for each follow-up study are in the docstring of its `eval_*` script, committed before the run.

A full 14-case run over the four models took about 83 s on the challenge's evaluation machine. Offline features were computed on Kaggle with 2×T4 GPUs, torch 2.11.0 and transformers 5.13.0, in fp16 and without batching. Batching changes the features.

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

The repository has no passwords, API tokens or model weights, and no competition test data.

## Sources

- Challenge: https://aimo-interp.github.io/
- Starter kit: https://github.com/aimo-interp/getting-started
- Proposal paper: arXiv 2607.13899
