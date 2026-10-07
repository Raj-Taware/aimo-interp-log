# Plan: label all four small models ourselves (H100, 1-2 slots)

Status: **ON HOLD (2026-10-07).** `aimo-interp/train-main-v2` shows the four
small models never disagree on the 9 problems labelled for >= 2 of them, so
the within-problem design below may have almost no discordant pairs to test.
Revisit after T-010.

## Why

Five pre-registered tests (T-003..T-005, re-run in T-008 with the repaired
tokenizer) agree: on public DeepSeek labels, prompt-only drift carries the
same information as problem length and nothing more. Public data cannot ask
the question that matters, because only one model is labelled — "model" and
"problem" are confounded.

**Within-problem design.** Label all four models on the same problems. For a
fixed problem, length is identical across models, so if a model's drift
(relative to the other models on that problem) predicts *its* label, length
cannot be the explanation. This is the test the method needs, and the
write-up's strongest possible result either way.

## Pipeline (mirrors the organizers', whose `permutation_source` is gpt-oss-120b)

1. **Variants.** `openai/gpt-oss-120b` (~62 GB MXFP4, fits one 80 GB H100)
   writes one answer-preserving variant per type — domain, distract,
   rephrase, rename, typos — for each of the 154 problems (770 variants,
   ~0.3M tokens, minutes). Prompt it with the type definition and the
   original answer; ask it to keep the answer unchanged.
2. **Sampling.** vLLM, each small model: 4 samples on the original, 2 per
   variant (14 generations per problem). T-009 protocol: boxed instruction,
   sampling from `generation_config`, no eos override, 8192 max tokens,
   extraction cascade boxed → bold → "answer is", repaired tokenizer.
3. **Labels.** `base_acc`, per-type `perm_acc`; spurious iff any type's
   `base_acc - perm_acc > 0` (the organizers' rule) — plus a noise-tolerant
   variant (> 0.25) reported alongside.

## Gate (before spending slot 2)

Re-derive **DeepSeek's** labels with this pipeline and compare with the 137
official ones. Proceed only if agreement >= 0.75 or Cohen's kappa >= 0.4.
If it fails, our labels measure something else and the plan stops there —
itself a reportable finding about label reproducibility.

## Budget (from T-009 measurements)

- ~4.5k tokens per generation (medians 2.2k-5.9k, 17-33% hit 8192).
- Per model: 154 x 14 = 2,156 generations ~= 9.7M tokens.
- vLLM on one H100, 4-8B models, high concurrency: ~3-6k tok/s aggregate
  -> ~0.5-1 h per model.
- **Slot 1**: gpt-oss variants + DeepSeek labels + gate. **Slot 2**: Qwen3.5-4B,
  Skywork, Olmo. Leaves two slots spare.

## Pre-registered analysis (fixed now)

Within-problem test on problems where the four models' labels disagree:
conditional logistic regression of label on each model's v3 drift score
standardized within problem (problem fixed effect, so length drops out).
Success: drift coefficient > 0 with p < 0.05 (one-sided), and AUC > 0.6 on
discordant pairs. Report either way.

## Risks

- vLLM support for Qwen3.5-4B (hybrid attention) — fall back to HF generate
  for that model only.
- gpt-oss variants may change the answer: spot-check 20 by hand-solving or by
  majority-of-models agreement on the variant.
- Two samples per variant is noisy; the gate measures how noisy.

## Prep that needs no H100 (Grok, Kaggle)

T-010: write the pipeline (`self-perturb/scripts/label_gen/`), dry-run it at
toy scale (3 problems, Skywork, HF generate) so the H100 slot is pure
execution.
