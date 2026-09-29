# Review: `Raj-Taware/aimo-interp-log` (other LLM's work, reviewed 2026-09-29)

## Keep — facts that change our decisions

- **Hidden small-track labels are balanced.** Official all-False and always-True
  submissions both scored exactly 0.50. Our 0.737 majority bar comes from the
  DeepSeek public dump (26% robust) and does not apply to the hidden set;
  **balanced accuracy is the number that predicts the leaderboard**.
- **Leaderboard denominator is ~14 cases now** (0.4286 = 6/14, 0.5714 = 8/14).
  Standard error ~13 points; one case is ~7 points.
- **Label construction:** on the public agg table, `model_is_robust` is exactly
  `absolute_accuracy_decay == 0`.
- **Damage by perturbation family** (DeepSeek, `math-robust-final.csv`):
  domain 0.692 mean decay, distract 0.147, rephrase 0.044, rename 0.028,
  typos 0.005.
- **Text-only control loses to majority** (OOF acc 0.617 vs 0.745, AUC 0.606):
  problem text alone does not explain the labels.
- **Working Codabench API submission path** (Small task pk `36241`, phase
  `30038`). Standing rule from Raj: **ask before any submission**.

## Discard — why its probes cannot work

- All four probes were trained on the 24-row val-sample, whose labels belong
  to *other* models (GPT-5.2, Gemini, GLM, gpt-oss). None of the four small
  checkpoints' own behaviour was ever supervised. Its OOF balanced accuracy is
  0.50-0.63 on n=24: noise, as its own diagnosis concludes.
- Its best official score, 0.5714, is 8/14 -- one case above chance, within
  noise. The threshold A/Bs (conservative vs symmetric) tune to leaderboard
  noise; they are not evidence of a better method.
- The counterfactual and uncertainty stages never ran on a GPU.

## How the two efforts combine

- Its ranked experiment #1 ("model-matched probes on
  `augmented-sample-math-agg`, grouped CV") is what `scripts/evaluate.py`
  already measures here: nested-CV AUC 0.66, balanced accuracy ~0.60.
- Our features are **architecture-agnostic** (cosines, norms, KL, flags at
  depth *fractions*), so a classifier trained on DeepSeek labels can be applied
  to all four checkpoints. Raw hidden-state probes cannot transfer (2560-dim vs
  4096-dim).

## Dead ends checked

- `augmented-sample-math` (Llama-3.1-8B-Instruct, 156 rows): **all labels
  False** -- single class, useless for training or a cross-model test.
- Full perturbed texts are not published. `permutations_causing_decay` lists
  only the variants that *broke* DeepSeek, so they cannot produce unbiased
  labels for other models.
