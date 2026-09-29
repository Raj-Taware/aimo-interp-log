# Measurement noise in self-perturbation features (2026-09-29)

Drift features are differences in the third decimal of a cosine (states under
answer-preserving edits sit at cosine ~0.99), so they are sensitive to anything
that changes low-order numerics.

## What was measured

Same 137 DeepSeek problems, same feature code, Kaggle T4x2, fp16:

| Run pair | What differed | Median run-to-run r (same feature) |
| --- | --- | ---: |
| v2 dump (2026-09-28) vs v3 dump | Kaggle's default transformers vs pinned **5.13.0** (and batching) | **0.62** |
| v3 dump vs v3.1 dump | batched vs one prompt per pass, both on 5.13.0 | **0.997** (v2 block) / **0.999** (v3 block) |

## Conclusion — and a correction

Commit `2058aba` attributed the r=0.62 instability to batch padding. The second
comparison rules that out: batching barely matters. The instability tracks the
**library environment** (different attention / RoPE implementations across
transformers versions). Unbatched encoding and fp16 pinning were kept anyway:
unbatched makes a prompt's features independent of its batch-mates by
construction, and fp16 pinning fixes a real precision mismatch (training on
T4s, which lack bf16; the eval card would otherwise choose bf16).

## Consequences

- Train and evaluate features in the **grader's exact library versions**
  (transformers 5.13.0, accelerate 1.13.0); every dump since v3 does.
- The first scored submission (951038, 0.7143) used an artifact trained on
  features from the unpinned environment. The v2 feature set measured in the
  pinned environment scores lower (nested-CV balanced 0.575 vs 0.602).
- A single feature can clear a shuffled-label permutation null and still be
  noise: `rephrase_norm_d0.25` reached AUC 0.709 in one run with a run-to-run
  r of 0.17. Permutation tests guard against label luck, not measurement luck;
  replicate across runs before trusting a feature.
- The multi-distractor aggregates replicated: `multidistract_cos_mean_d0.1`
  scored AUC 0.714 in both v3 and v3.1.
