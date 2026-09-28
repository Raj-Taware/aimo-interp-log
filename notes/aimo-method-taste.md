# AIMO Interpretability Challenge — Method Taste Notes (Small Models Track)

**Audience:** Raj Taware (independent; Small Models Track primary)  
**Written:** 2026-09-21 IST  
**Scope:** Research findings + recommended attack plan. **No Fields compute form.**  
**Prior brainstorm (not ground truth):** `/workspace/boundarybet_aimo_playbook.md`

### Source tags
| Tag | Source |
|---|---|
| **[SITE]** | https://aimo-interp.github.io/ (fetched 2026-09-21) |
| **[PAPER]** | arXiv:2607.13899 (ar5iv HTML; §§1.3–1.5, 2.1; App. B) |
| **[GS]** | github.com/aimo-interp/getting-started README + local `/workspace/aimo-interp-getting-started` |
| **[BASE]** | github.com/aimo-interp/baselines README + local `/workspace/aimo-interp-baselines` |
| **[U-ABS]** | arXiv:2605.07776 Atom abstract only |
| **[LOCAL]** | val-sample import, probe/uncertainty artifacts, `notes/aimo-scoreboard.md` |
| **[PLAY]** | boundarybet playbook — brainstorm only |

Where **[SITE]/[GS]** disagree with **[PAPER]**, prefer website/repo (**[SITE]** FAQ §1).

---

## A. Task facts that change method design

### Objective & ranking
- Classify each **problem–model** pair as **robust** vs **spurious**. Submission must expose `are_robust(model_id, reasoning_effort, problems) -> list[bool]` (`True` = robust). Legacy 2-arg form still accepted. **[GS]**
- Ranked by **accuracy** on private Codabench test; Main and Small ranked separately. Ties broken by **faster mean runtime over three repeated trials**. Also report `coverage`, `invalid_predictions`. **[SITE]**, **[GS]**

### Hidden robustness criterion (design-critical)
- Public gloss: robust iff answers hold under perturbations that leave the deductive chain unchanged; spurious iff counterfactuals reveal ≥1 spurious pattern. **[SITE]**
- **Exact labelling rule is unpublished** and **differs from the proposal-paper outline**, deliberately discouraging label reconstruction vs label-agnostic methods. Validation cases have a **substantially large boundary** between positives and negatives. **[SITE]** FAQ §7
- Paper formalization (label 0 if a *majority* of ΔX flips correct→incorrect; else 1) is **not** the operative private rule. **[PAPER]** §1.4 vs **[SITE]** FAQ §7

### Signals & data rules
- Allowed: weights, activations, token probabilities, other internals; white-box encouraged; **black-box behavioural analyses within the eval budget** (e.g. reasoning chains) permitted. **[SITE]** Rules §2, **[GS]**
- **Clarification 19 Sep:** ban on extra labeled data was **retired**. Large-scale test-time perturbation generation is **disincentivised by the time limit**, not explicitly banned. Evaluation-data leakage still disqualifies. **[SITE]** Rules §1  
  (Paper’s older “no extra labels for same perturbation types” is superseded. **[PAPER]** §2.1)

### Time / hardware (paper overridden)
- **3600 seconds for the entire prediction run** (all models × all problems); scoring run limited separately. **[SITE]** FAQ §8, **[GS]**
- Ignore paper’s “1 hour per problem / multi-GPU node” for planning. **[PAPER]** vs **[GS]**
- Worker: **1× NVIDIA RTX PRO 6000 (Blackwell, 96 GB VRAM)**, 192 CPU cores; network off; `HF_HUB_OFFLINE=1`; fixed image only (PyTorch 2.12.1+CUDA 13.2, Transformers 5.13.0, scikit-learn 1.8.0, joblib 1.5.3, …). No runtime `pip install`. **[GS]**
- `openai/gpt-oss-120b` cached MXFP4 ~62 GiB; Small track excludes it. **[GS]**

### Competition models
| `model_id` | Main | Small |
|---|:---:|:---:|
| `Qwen/Qwen3.5-4B` | yes | yes |
| `Skywork/Skywork-OR1-Math-7B` | yes | yes |
| `allenai/Olmo-3-7B-Think` | yes | yes |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | yes | yes |
| `openai/gpt-oss-120b` | yes | **no** |

`are_robust` receives these IDs **verbatim** (no `:low` aliases). Warm-up models (`allenai/Olmo-3-7B-Instruct`, `google/gemma-3-27b-it`) are unused in Competition phase. **[SITE]** FAQ §6, **[GS]**

### Public vs private
- At eval: `model_id`, `reasoning_effort`, problem strings (grouped by model×effort). No case IDs, labels, network, or other HF models. **[GS]**
- Public: `aimo-interp/val-sample` (and broader public validation on site/HF). Dev labels only; ranking uses private problems/labels. Validation covers all **model types** in test, not the same problems. **[SITE]**, **[GS]**
- Paper design intent: ~150 train / ~30 test problems × ≥8 models; symbolic chains enable rephrase/rename/domain/distract/typos/expert/expert-no-solution with large accuracy drops. **[PAPER]** §1.3–1.4, App. B
- Small track exists so compute-heavy methods (SAEs, Transcoders) remain feasible on ≤10B models. **[SITE]**
- Awards: technical report + open source by **2026-11-15**; reproducibility from report alone; peer-review ≤3 reports. Writeup pool $5,000 (ranking-independent). Final submit **2026-11-01**. **[SITE]**

### Packaging
- Small-track ZIP must include root `small.txt` (contents ignored). **[GS]**
- Predictions: native Python `bool`, correct length/order; else `invalid_predictions`. **[GS]**
- Uncertainty starter maps alias `qwen3-8b:low` → `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`; **Competition will not pass aliases** — key artifacts on exact HF IDs. **[GS]**

---

## B. Official baselines: mechanics & limits

### B1. Trained representation probe
**Train ([BASE]):** one forward pass; **final prompt-token hidden state at every layer**; linear `probe` and optional RBF `kernel` (+ optional PCA) predicting `absolute_accuracy_decay` / robustness labels; CV over layers/seeds/controls (`NONE`, `RANDOMIZATION`). Unique prompts encoded once. **[BASE]**

**Submit ([GS] `solutions/trained-probe`):** load schema-v2 ensemble pickle → same system prompt → last prompt-token @ `best_layer_index` → mean seed/fold margin → `mean ≥ 0` ⇒ robust. Missing artifact ⇒ all `False`. **[GS]**

**Local artifact ([LOCAL]):** ~53 MB pickle; `model_id=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`; `target_col=model_is_robust`; classification; `best_layer_index=31`; strategy `best_layer_mean_margin`; seeds 42–46; weights `(5, 4096)`; thresholds 0; **layer-31 selection score ≈ 0.5059** (≈chance on that metric).

**Paper CV ([PAPER] §1.5):** probing classifier up to **58.37% (±7.5)** on DeepSeek sample val, stratified 10-fold.

**Limits:** (1) last-prompt-token ignores CoT; (2) DeepSeek-only artifact won’t score the other three Small models; (3) local selection ~0.50 ⇒ weak standalone signal; (4) kernel/PCA still last-token-bound; no Codabench numbers in [BASE] README; (5) on this CPU box artifact path **raises** (no HF cache/GPU) → all invalid. **[LOCAL]**

### B2. Uncertainty profiling
**What ([GS] `solutions/uncertainty-profiling`):** greedy generate (prompt≤2048, gen≤4096, batch 8, static KV) → **14 aggregate** token-distribution features → sklearn regressor predicts `absolute_accuracy_decay` → score **&lt; threshold** ⇒ robust. Explicitly simpler than **[U-ABS]** temporal profiles (slope/linearity). **[GS]**

**14 features ([GS] config):** `generation_mean_logprob`, `generation_mean_prob`, `generation_mean_nll`, `generation_ppl`, `generation_min_logprob`, `generation_max_logprob`, `generation_std_logprob`, `generation_min_k_logprob`, `generation_mean_entropy`, `generation_mean_top1_prob`, `generation_mean_top2_margin`, `generation_frac_selected_is_top1`, `generation_frac_high_conf_tokens`, `generation_frac_low_conf_tokens`.

**Checked-in artifact ([GS] README):** RF 500 trees; threshold **0.5760**; OOF balanced accuracy **0.6262**, plain accuracy **0.6879**, MAE **0.3429**; trained on `aimo-interp/augmented-sample-math-agg` pinned revision; DeepSeek / `qwen3-8b:low` only.

**Paper CV ([PAPER] §1.5):** uncertainty classifier up to **69.23% (±10.13)** — stronger than probe in that experiment.

**[U-ABS]:** uncertainty *trace profiles* predict answer **correctness** with AUROC up to 0.807; early tokens alone ≈0.801 — motivates short-trace features; target ≠ robustness.

**Limits:** (1) 4k-token gens × many problems × 4 models can blow **3600s**; (2) aggregates drop temporal shape; (3) DeepSeek-only; (4) decay→threshold may misalign with private H; (5) not runnable here without checkpoint. **[GS]**, **[LOCAL]**

### B3. Data attribution (promised, not shipped)
- Listed in **[PAPER]** §1.5; **not** present as a Codabench solution in [BASE] or [GS] `solutions/` as of this research.

### B4. Constants
- `solutions/always-true` / all-`False` via missing-artifact probe (and `solutions/all-false` if present) for contract tests. **[GS]**, **[LOCAL]**

---

## C. Measured local numbers (no Codabench LB invented)

Source: **[LOCAL]** `aimo-scoreboard.md` + re-inspected artifacts/metadata 2026-09-21.

**val-sample:** `aimo-interp/val-sample` revision `1ae454ec1fad9727084eda8f9f3c9ae2239b21de`; **24** cases after dropping 4 duplicate rows (28→24); labels **9 robust / 15 not** (`is_robust`). **[LOCAL]** `metadata.json`

**Local `run_local` metrics (CPU, no GPU, no HF cache) — [LOCAL] scoreboard:**

| Method | accuracy | coverage | invalid | Notes |
|---|---:|---:|---:|---|
| `solutions/always-true` | 0.375 | 1.0 | 0 | = base rate of True |
| `solutions/trained-probe` **with** artifact | 0.0 | 0.0 | 24 | All invalid — model load fails |
| `solutions/trained-probe` **without** artifact (all-False) | 0.625 | 1.0 | 0 | = base rate of False |
| `solutions/uncertainty-profiling` | *not run* | — | — | Same GPU/cache blocker |

**val-sample model mix ≠ Competition models:** `Qwen/Qwen3.5-397B-A17B-FP8:low` (5), `gemini-3.1-pro-preview:low` (3), `gpt-5.2-2025-12-11:low` (3), `gpt-oss-120b:low` (5), `huikang-gpt-oss-120b-aimo3:low` (3), `lukealonso/GLM-5.1-NVFP4:low` (4), `qwen3-8b:low` (1). Only `qwen3-8b:low` maps (via starter alias) toward a Competition Small checkpoint. **[LOCAL]**  
⇒ Local accuracy is a **packaging / base-rate check**, not a Small-track private preview. **[GS]**

**No Codabench public/private leaderboard scores are cited** (none retrieved; none invented).

---

## D. Recommended attack plan — Small Models Track

1. **Ship a valid Codabench Small ZIP first** (`small.txt`, coverage=1, invalid=0, Docker-parity). Use all-False or majority prior until real models run on GPU. Local all-False=0.625 on this 9/15 sample is **not** transferable; validity is. **[LOCAL]**, **[GS]**

2. **Treat public robustness labels as noisy proxies of hidden H.** Prefer features that predict **stability / accuracy decay under self-generated or organiser-style perturbations**, then map to bool with **problem-grouped** CV — do **not** reverse-engineer secret thresholds. Extra labeled data is now allowed (**[SITE]** Clarification 19 Sep); build **own** robustness proxies offline, don’t chase val-sample closed-model IDs.

3. **Try first (priority):**  
   (a) **short-trace uncertainty** with **hard token caps** ≪4096 (early-trace motivation from **[U-ABS]**; paper already shows uncertainty ≫ last-token probe: ~69% vs ~58%);  
   (b) **cheap behavioural counterfactuals** (rename / light rephrase / typo; K=1–2) measuring answer agreement — legal within time;  
   (c) **per-model last-token / selected-layer linear probes** as a cheap third vote.  
   Fuse with nested grouped-by-problem CV. **[PAPER]**, **[U-ABS]**, **[GS]**

4. **Per-model artifacts for all four Small checkpoints:** `Qwen/Qwen3.5-4B`, `Skywork/Skywork-OR1-Math-7B`, `allenai/Olmo-3-7B-Think`, `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`. Do not ship DeepSeek-only pickles. **[GS]**, **[LOCAL]**

5. **Budget cascade is mandatory:** cost ≈ Σ_models (load + N_problems × (encode_ms + K×gen_ms)). Target finish ≤~80% of 3600s before freeze. Abort expensive gens; never risk timeout/invalid. Prefer one forward or one short gen reused for multiple features. **[GS]**

6. **Discard / deprioritize:** reconstructing private H; SAE/Transcoder megaprojects unless pure-Torch + offline-trainable with clear lift (eval image has no SAE stack); Main/`openai/gpt-oss-120b` until Small is timed with headroom; copying starter uncertainty config (4096 tokens) unchanged; optimizing for val-sample’s closed-model mix.

7. **Compute sense (independent):** request Fields/H200 hours for **offline training** (feature caches × 4 models, grouped CV, optional self-label perturbation gens). Eval is organiser GPU — design for **1×96GB, 1 hour wall**. Serialize with runtime sklearn/joblib versions. **[SITE]**, **[GS]**, **[PAPER]** §3.2

8. **Writeup as parallel prize:** document generalization tests, negatives (probe≈chance on selection metric), efficiency cascade, and why label-agnostic stability signals beat H-reconstruction — aligns with $5k interest areas. **[SITE]**

---

## E. Risks / unknowns

| Risk / unknown | Why it matters | Source |
|---|---|---|
| Exact private H / thresholds | Overfitting public labels may not transfer | **[SITE]** FAQ §7 |
| Private base rate & N_problems | Majority baseline & whether gens fit in 3600s unknown | **[UNKNOWN]**; measure via timing |
| Public sample models ≠ Competition models | Local scores nearly useless for ranking prediction | **[LOCAL]**, **[GS]** |
| `reasoning_effort` in Competition | Field exists; starters ignore it; may matter for Think/OR1 | **[GS]** |
| Whether 3-trial timing runs always or only on ties | Still design for deterministic fast path | **[SITE]** |
| Self-generated labels ≠ organiser H | Extra labels allowed but wrong proxy still fails privately | **[SITE]** |
| OOM / shared-worker VRAM contention | GS notes possible clash; resubmit | **[GS]** |
| Data-attribution baseline if later released | Could move community meta | **[PAPER]** vs missing in **[BASE]** |
| Independent GPU lag | Blocks real probe/uncertainty measurement until GPU | **[LOCAL]** blockers |

---

## F. Taste (plain technical prose; no product name)

What should win on the Small Models Track is not a prettier linear probe on the last prompt token, and not a maximal SAE demo that cannot finish in an hour. The private label is a **stability** judgment under hidden but structured perturbations; the organisers’ own published numbers already show that **token-level uncertainty over the model’s generated reasoning** carries more signal than a linear read of the prefill state (~69% vs ~58% on their DeepSeek sample CV), while still leaving a large accuracy gap. The competitive solution is therefore a **budget-aware hybrid**: short, early-trace uncertainty features plus a few **cheap answer-stability checks** under light local transforms, fused and calibrated with **problem-grouped** validation, with one artifact pack per ≤8B Competition model and a hard cascade that always returns valid bools under 3600s. Methods that memorize public robustness labels or assume the paper’s majority-flip formula will look strong on the wrong sample and fail privately; methods that measure **whether the model’s belief state behaves like robust deduction**—quickly, per model, without reconstructing H—are what the current rules, baselines, and time budget jointly point toward.

---

## Source checklist
- [SITE] https://aimo-interp.github.io/ (2026-09-21)
- [PAPER] arXiv 2607.13899 via ar5iv HTML + Atom abstract
- [GS] github.com/aimo-interp/getting-started (+ `/workspace/aimo-interp-getting-started`)
- [BASE] github.com/aimo-interp/baselines (+ `/workspace/aimo-interp-baselines`)
- [U-ABS] arXiv 2605.07776 Atom abstract only
- [LOCAL] val-sample metadata/cases/labels; probe+uncertainty artifacts; aimo-scoreboard.md
- [PLAY] `/workspace/boundarybet_aimo_playbook.md` — brainstorm only
