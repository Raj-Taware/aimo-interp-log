# Reading the Room Without the GPU: CPU-Only Findings for the AIMO Interpretability Challenge

**Author:** Raj Taware  
**Date:** 2026-09-21 (IST)  
**Track focus:** Small Models  
**Status:** Draft mini-paper from a ~2h CPU research loop — measured numbers only; no Codabench submission in this pass.

---

## Motivation

The AIMO Interpretability Challenge asks a deceptively simple question: for a given math problem and a given language model, is the model’s answer **robust**, or is it hanging on a **spurious** pattern that a small rewrite would break?

Think of it like checking whether a student really understands a proof, or just memorised the layout of the textbook page. A robust solver still gets the right answer when you rename variables or change the story wrapping; a spurious solver collapses.

Two practical constraints shaped this note. First, the private label rule is **hidden** (website FAQ), so chasing public `model_is_robust` as if it were the secret exam key is a trap. Second, this research pass ran on a **CPU-only** shared box with no NVIDIA GPU and no local Hugging Face weights for the contest 4–8B models. That ruled out live probe or uncertainty inference. So the goal became: squeeze honest signal out of public tables, starter artifacts, and control experiments — then write a clear GPU todo list.

Organiser-reported sample CV from the proposal paper (cite as paper, not local run): uncertainty-style classifier up to about **69%**, representation probe about **58%**. Our local packaging baselines on `val-sample` remain the trivial rates: always-`True` **0.375**, always-`False` **0.625**.

---

## Methods tried

All work lives under `/workspace/aimo-research-loop/`. No Forms, no Codabench upload.

1. **Val-sample autopsy.** Joined the 24 imported cases with labels; measured base rates, `model_id` mix, problem-length stats. (`permutation_type` is not in the imported JSONL.)

2. **Probe artifact inspection without model load.** `pickle.load` on the 53 MB starter ensemble; recorded layer, weight dims, thresholds, seeds, and the artifact’s own layer-selection scores.

3. **Uncertainty artifact inspection without generation.** `joblib.load` on the DeepSeek regressor; read threshold, 14 feature names, baked-in OOF metrics, and random-forest importances.

4. **Text-only control classifier.** Sklearn logistic regression on shallow problem-text features (length, digit density, latex-ish markers, a few topic flags). This is deliberately a **negative control**: a real submission must call `are_robust(model_id, problems)` using **model** signals. Text-only only answers, “how much of the public label leaks from the problem string alone?”

5. **Baselines CSV analysis.** `math-robust-agg.csv` (141 rows) and `math-robust-final.csv` (675 rows) from the baselines repo — both `qwen3-8b:low` — for decay stats and permutation-type breakdowns.

6. **Hybrid plan diagram.** A budget-cascade sketch: short uncertainty + cheap counterfactuals + per-model probes → calibrated fusion → always-valid fallback.

---

## Results

### 1. Val-sample is a packaging mirror, not a Small-track preview

We measure **24** cases: **9** robust, **15** not → base rates **0.375 / 0.625**. There are only **8** unique problem texts reused across models. Mean problem length is **262.5** characters (**47.4** words); robust cases are a bit shorter on average (**227** vs **284** chars), but with n=24 that is a hint, not a method.

The `model_id` mix is dominated by closed or non-Competition aliases (Qwen3.5-397B, Gemini, GPT-5.2, gpt-oss-120b, GLM-5.1, …) with a single `qwen3-8b:low` row. So the familiar local scores — always-true **0.375**, all-False **0.625**, probe-with-artifact **24/24 invalid** for lack of weights — check the **contract**, not private ranking.

Figures: `figures/01_val_sample_label_base_rate.png`, `02_val_sample_model_mix.png`, `03_val_sample_problem_length.png`.

### 2. The shipped probe is DeepSeek-only and nearly chance at layer selection

The pickle is schema v2, `model_id=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, target `model_is_robust`, strategy `best_layer_mean_margin`, **best_layer_index=31**, seeds **42–46**, **8** fold groups, per-seed weight shape **(4096,)**, thresholds **0**. Decision rule: mean margin ≥ 0.

The uncomfortable number: the artifact’s own **layer selection score at layer 31 is 0.5059** (other layers sit at 0.5). One individual fold/seed entry hits **0.7375**, but the selection summary that the starter actually trusts is essentially a coin flip. That aligns with treating last-prompt-token linear probes as a **weak third vote**, not the whole product.

Figure: `figures/04_probe_layer_scores.png`.

### 3. The shipped uncertainty regressor is stronger on paper — still one model

Joblib artifact: **14** generation-confidence features, RF with **500** trees, **decision_threshold ≈ 0.576**. Baked-in OOF on its training provenance (`augmented-sample-math-agg`, 141 rows / 137 groups): accuracy **0.6879**, balanced accuracy **0.6262**, MAE **0.3429**. Top importances: `generation_min_logprob` (**0.219**), `generation_frac_selected_is_top1` (**0.210**), `generation_std_logprob` (**0.118**).

So the starter’s own bookkeeping already ranks uncertainty above the probe — same direction as the organiser paper (~69% vs ~58%). We could not re-run it live without a GPU and the DeepSeek checkpoint.

Figure: `figures/05_uncertainty_feature_importance.png`.

### 4. Text-only control: public labels are not “just hard problems”

On `math-robust-agg` (n=**141**, robust rate **0.255**), majority accuracy is **0.745**. A text-only logistic model gets OOF accuracy **0.617**, balanced accuracy **0.579**, AUC **0.606**. It **loses to majority** on accuracy and sits near chance when class balance is respected.

On val-sample with GroupKFold by problem text: OOF accuracy **0.583** vs majority **0.625**; balanced accuracy **0.511**.

Analogy: if robustness were mostly “this problem looks scary,” a bag of length and latex features would sail past the majority baseline. It does not. The label behaves like a property of **how the model reacts**, not of the worksheet alone — which is exactly what the competition interface forces you to measure.

Figures: `figures/06_text_only_control_metrics.png`, `07_text_only_coefficients.png`, `08_text_only_val_sample.png`.

### 5. Public decay tables: domain hurts; typos barely tickle

On `math-robust-agg`, mean `absolute_accuracy_decay` is **0.622** while mean base accuracy is **0.994**. Robust rows have mean decay **0.0**; not-robust rows **0.835**. In this particular AGG table, `model_is_robust` coincides with zero overall decay (oracle threshold accuracy **1.0**). That is **how the public column is constructed**, not a feature you will be handed at eval time.

By permutation type on `math-robust-final` (excluding OVERALL):

| Type | Mean abs decay |
|---|---:|
| domain | **0.692** |
| distract | 0.147 |
| rephrase | 0.044 |
| rename | 0.028 |
| typos | **0.005** |

If we later generate our own cheap counterfactuals under the time budget, **domain / distract** look like higher-yield stress tests than typos.

Figures: `figures/09_agg_decay_by_label.png` … `12_length_vs_decay.png`.

### 6. Hybrid sketch

Short early-trace uncertainty (inspired by stronger uncertainty numbers in paper and starter OOF) + a few cheap answer-agreement checks + per-model last-token probes, fused under a **budget cascade** that always returns valid bools before the 3600s wall. Figure: `figures/13_hybrid_pipeline_diagram.png`.

---

## Discussion

Three takeaways survived contact with measurement.

**Validity before cleverness.** On this box, the only runnable accurate scores are base-rate constants. Shipping a Small-track ZIP with `small.txt`, coverage 1, and zero invalid predictions remains the first real milestone; probe/uncertainty paths currently raise without weights.

**Model signals, not text hacks.** The text-only control bounds label leakage from the problem string: useful as a sanity check, useless as a submission. Combined with the hidden private criterion, the winning direction is label-agnostic **stability** measurement (uncertainty traces, light counterfactuals), not reconstructing H from public bits.

**One DeepSeek pickle is not a Small-track entry.** Competition Small models are four checkpoints. Artifacts must be keyed on exact HF IDs. Layer-selection ≈0.51 also argues for not betting the farm on a single last-token probe.

Limitations of this pass are blunt: no live generation, no activation probes, val-sample model mix mismatched to eval, and baselines CSVs cover only `qwen3-8b:low`. Every accuracy number above that is either a base rate, an artifact’s own OOF, organiser paper CV, or a control — all labelled as such.

---

## Next on GPU (Kaggle / Fields offline train / Codabench image)

1. **Warm the four Small models** into the offline cache: `Qwen/Qwen3.5-4B`, `Skywork/Skywork-OR1-Math-7B`, `allenai/Olmo-3-7B-Think`, `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`.

2. **Re-measure starters** on a Competition-like validation slice with GPU: trained-probe and uncertainty-profiling accuracy/coverage — replace today’s invalid/not-run rows with real numbers.

3. **Short-trace uncertainty:** keep the 14-feature family but cut max new tokens far below 4096; prefer early-trace aggregates; re-tune threshold with **problem-grouped** CV.

4. **Cheap CF loop (K=1–2):** rename / light distract / domain-flavoured rewrites; score answer agreement; lean on Trial 5’s finding that domain ≫ typos.

5. **Per-model probes** at selected layers (not DeepSeek-only); treat as a fast third vote given weak layer-selection scores.

6. **Budget cascade:** load once per model; abort gens when remaining time is tight; fall back to calibrated prior / all-False so `invalid_predictions` stays 0.

7. **Ship** a Small ZIP (`small.txt`) timed to finish with headroom under 3600s; write up negatives (text-only fail, probe≈chance selection) for the writeup pool.

---

## Artifact index

| Path | Role |
|---|---|
| `/workspace/aimo-research-loop/lab-notebook.md` | Chronological trials |
| `/workspace/aimo-research-loop/figures/*.png` | 13 charts |
| `/workspace/aimo-research-loop/artifacts/*.json` | Raw measured dumps |
| `/workspace/aimo-research-loop/mini-paper-draft.md` | This draft |
| `/workspace/aimo-research-loop/aimo-eli5-today.html` | ELI5 page for today’s findings |
| `/home/box/agent-data/agents/9cb4b6eb-bbed-4e9d-8fe4-97d4de22c4f4/notes/aimo-scoreboard.md` | Updated local scoreboard |

**Citations (external):** organiser proposal paper arXiv:2607.13899 (uncertainty ~69%, probe ~58% sample CV); getting-started / baselines repos; site FAQ on hidden label criterion.
