# AIMO Research Loop — Lab Notebook

**Operator:** Raj Taware (via box agent)  
**Machine:** shared CPU box (no GPU / no nvidia)  
**Workspace:** `/workspace/aimo-research-loop/`  
**Started:** 2026-09-21 ~16:53 IST  
**Repo:** `/workspace/aimo-interp-getting-started`  
**Baselines data:** `/workspace/aimo-interp-baselines/data/`

Rule for this notebook: only measured numbers. No invented Codabench scores.

---

## Trial 0 — Setup (16:53 IST)

**Why:** Create a clean research-loop directory and confirm CPU-only constraints.  
**How:** `mkdir` under `/workspace/aimo-research-loop/{figures,code,artifacts}`; reuse getting-started `.venv`; `uv pip install matplotlib` (pandas already present).  
**Result:** Workspace ready. Confirmed prior scoreboard: always-true **0.375**, all-False **0.625**, trained-probe with artifact **invalid 24/24** (no HF cache / no GPU).

---

## Trial 1 — Val-sample deep dive

**Why:** Understand base rates, model mix, and whether problem text length correlates with labels before trusting any local accuracy.  
**How:** Join `data/val-sample/input/cases.jsonl` with `reference/labels.jsonl`; compute counts, length stats, unique problems; matplotlib charts.  
**Result (measured):**

| Quantity | Value |
|---|---|
| Cases | 24 (metadata: 28 source rows − 4 duplicates) |
| Robust / not | **9 / 15** |
| Base rate True / False | **0.375 / 0.625** |
| Unique problem texts | **8** |
| Chars mean / median / min–max | **262.5 / 282 / 142–405** |
| Words mean / median / min–max | **47.4 / 54 / 22–70** |
| Mean chars (robust vs not) | **227.2 vs 283.7** |
| Mean words (robust vs not) | **41.7 vs 50.9** |

**model_id mix:**

| model_id | n | n_robust | robust_rate |
|---|---:|---:|---:|
| Qwen/Qwen3.5-397B-A17B-FP8:low | 5 | 1 | 0.20 |
| gpt-oss-120b:low | 5 | 0 | 0.00 |
| lukealonso/GLM-5.1-NVFP4:low | 4 | 4 | 1.00 |
| gpt-5.2-2025-12-11:low | 3 | 3 | 1.00 |
| gemini-3.1-pro-preview:low | 3 | 0 | 0.00 |
| huikang-gpt-oss-120b-aimo3:low | 3 | 1 | 0.33 |
| qwen3-8b:low | 1 | 0 | 0.00 |

**permutation_type:** **absent** from imported `cases.jsonl` (fields are only `id`, `model_id`, `problem`). Breakdown deferred to baselines CSVs (Trial 5).  
**Figures:** `figures/01_val_sample_label_base_rate.png`, `02_val_sample_model_mix.png`, `03_val_sample_problem_length.png`  
**Artifact:** `artifacts/exp1_val_sample.json`, `artifacts/val_sample_enriched.csv`

**Interpretation:** Local always-true/all-False numbers are exactly these base rates. Val-sample models are mostly closed / non-Competition IDs — packaging check only, not a Small-track preview.

---

## Trial 2 — Probe pickle metadata (no model load)

**Why:** Learn what the shipped 53 MB artifact actually contains without loading DeepSeek.  
**How:** `pickle.load` on `solutions/trained-probe/probe_artifacts/probe_artifact.pkl`; walk schema; plot `layer_scores`.  
**Result (measured):**

| Field | Value |
|---|---|
| Size | 54,729,307 bytes (~53 MB) |
| schema_version | 2 |
| artifact_type | `all_folds_layers_seed_probe_ensemble` |
| model_id / hf_model_id | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` |
| target_col | `model_is_robust` (classification) |
| best_layer_index | **31** |
| recommended_strategy | `best_layer_mean_margin` |
| seeds | 42–46 (5 seeds) |
| layers scored | 37 (indices 0–36) |
| groups / folds | 8 |
| weight shape (per seed @ layer) | **(4096,)** float64 |
| thresholds (layer 31, group 0) | **[0,0,0,0,0]** |
| bias example (seed 0, L31) | −0.00584 |
| aggregation | `mean_margin` (≥0 ⇒ robust) |
| selection_metric | `val balanced_acc` |
| **layer selection best score** | **0.5059375** (min across layers also 0.5) |
| best single probe_scores entry @ L31 NONE | 0.7375 (fold 1, seed 42) |
| mean probe_scores @ L31 NONE (n=20) | 0.511875 |

**Figures:** `figures/04_probe_layer_scores.png`  
**Artifacts:** `artifacts/exp2_probe_metadata.json`, `exp2_probe_metadata_rich.json`, `exp2_weight_dims.json`

**Interpretation:** Layer-selection metric is ~chance (0.506). A single lucky fold hit 0.74, but the artifact’s own layer pick is barely above 0.5. DeepSeek-only; cannot score other Small models.

---

## Trial 3 — Uncertainty joblib metadata (no generation)

**Why:** Confirm feature schema, threshold, and CV numbers baked into the starter without running generation.  
**How:** `joblib.load` on `uncertainty_artifacts/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib`; read `cv_results`, RF importances via pipeline `regressor`.  
**Result (measured):**

| Field | Value |
|---|---|
| Size | 1,007,536 bytes |
| artifact_type | `uncertainty_regressor` |
| model_id | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` |
| n_features | **14** (names match starter config) |
| decision_threshold | **0.5759658815999995** |
| estimator | Pipeline(`imputer` → `RandomForestRegressor` n_estimators=**500**) |
| Train source | `aimo-interp/augmented-sample-math-agg` rev `f972ced0…` |
| Train rows / groups | **141 / 137** (source_model_ids: `qwen3-8b:low`) |
| OOF accuracy (in artifact) | **0.6879** |
| OOF balanced accuracy | **0.6262** |
| OOF MAE | **0.3429** |
| OOF R² | **−0.0626** |

**Top RF importances:**

| Feature | Importance |
|---|---:|
| generation_min_logprob | 0.219 |
| generation_frac_selected_is_top1 | 0.210 |
| generation_std_logprob | 0.118 |
| generation_min_k_logprob | 0.071 |
| generation_frac_high_conf_tokens | 0.065 |

**Figures:** `figures/05_uncertainty_feature_importance.png`  
**Artifacts:** `artifacts/exp3_uncertainty_metadata.json`, `exp3_feature_importance.json`

**Interpretation:** Starter uncertainty beat the probe on its own OOF (≈0.69 vs probe layer-select ≈0.51), matching organiser paper ordering (~69% vs ~58%). Still DeepSeek / qwen3-8b:low only; needs GPU to run on eval.

---

## Trial 4 — Text-only control classifier (CPU)

**Why:** Bound how much `model_is_robust` is predictable from problem text alone. This is a **negative / control** experiment — submissions must use model signals (`are_robust(model_id, problems)`), not problem-only heuristics.  
**How:** Hand-crafted features (length, digit fraction, latex-ish markers, topic flags) → `StandardScaler` + balanced `LogisticRegression`.  
- On `math-robust-agg` (n=141): Stratified 5-fold OOF.  
- On val-sample (n=24): GroupKFold by problem text (8 unique problems).  

**Result (measured) — math-robust-agg:**

| Metric | Value |
|---|---:|
| n / n_robust | 141 / 36 |
| base_rate_robust | 0.255 |
| majority accuracy | **0.745** |
| OOF accuracy | **0.617** |
| OOF balanced accuracy | **0.579** |
| OOF F1 | **0.400** |
| OOF ROC AUC | **0.606** |

**Result (measured) — val-sample grouped:**

| Metric | Value |
|---|---:|
| OOF accuracy | **0.583** |
| OOF balanced accuracy | **0.511** |
| OOF F1 | **0.286** |
| OOF ROC AUC | **0.667** |
| majority accuracy | **0.625** |

Largest |coefs| on agg: `words` (+0.82), `digit_frac` (−0.75), `n_numbers` (+0.67), `chars` (−0.42).

**Figures:** `06_text_only_control_metrics.png`, `07_text_only_coefficients.png`, `08_text_only_val_sample.png`  
**Artifact:** `artifacts/exp4_text_only.json`

**Interpretation:** Text-only **loses to majority** on accuracy (0.617 < 0.745) and is near chance on balanced accuracy. On val-sample, grouped OOF accuracy (0.583) also sits under majority (0.625). Conclusion: public robustness is **not** well explained by shallow problem text; any competitive method needs model internals / behaviour.

---

## Trial 5 — Baselines public tables (`math-robust-*.csv`)

**Why:** permutation_type and decay stats exist here even though val-sample import dropped them.  
**How:** Analyze `math-robust-agg.csv` (141 rows) and `math-robust-final.csv` (675 rows), both `qwen3-8b:low` only.

**Result (measured):**

| Quantity | Value |
|---|---|
| agg robust rate | **0.255** (36/141) |
| mean absolute_accuracy_decay | **0.622** (median 0.75) |
| mean base_accuracy | **0.994** |
| mean permuted_accuracy | **0.372** |
| robust rows: mean abs decay | **0.000** |
| not-robust rows: mean abs decay | **0.835** |

**Oracle note:** On this AGG table, `model_is_robust` is exactly equivalent to `absolute_accuracy_decay == 0` (threshold sweep best thr=0.01 → acc=1.0, bal=1.0, AUC of −decay = 1.0). That is **label construction**, not a deployable method — eval does not provide decay.

**Mean abs decay by permutation_type (final, excl. OVERALL):**

| permutation_type | n | mean_abs_decay | robust_rate |
|---|---:|---:|---:|
| domain | 98 | **0.692** | 0.122 |
| distract | 127 | 0.147 | 0.260 |
| rephrase | 105 | 0.044 | 0.229 |
| rename | 105 | 0.028 | 0.276 |
| typos | 99 | **0.005** | 0.232 |

**Figures:** `09_agg_decay_by_label.png`, `10_final_decay_by_perm_type.png`, `11_decay_threshold_sweep.png`, `12_length_vs_decay.png`  
**Artifact:** `artifacts/exp5_baselines_tables.json`

**Interpretation:** **Domain** perturbations hurt most; **typos** barely move accuracy. Cheap CF budget should prefer domain/distract-style transforms over typos if generating own stability labels offline.

---

## Trial 6 — Hybrid pipeline diagram

**Why:** Capture the planned Small-track system for the mini-paper / writeup.  
**How:** matplotlib flowchart + mermaid snippet.  
**Result:** `figures/13_hybrid_pipeline_diagram.png`, `artifacts/hybrid_pipeline.mmd.md`

---

## Trial 7 — Packaging / docs (no Codabench submit)

**Why:** Deliverables for parent agent / Raj.  
**How:** Write `mini-paper-draft.md`, extend ELI5 HTML, append scoreboard.  
**Result:** See paths in mini-paper and final report. Forms/Codabench **not** submitted (per constraints).

---

## Blockers still open

1. No GPU / no local HF weights → cannot measure live probe or uncertainty accuracy on val-sample.  
2. Val-sample model mix ≠ Competition Small IDs.  
3. Probe + uncertainty artifacts are DeepSeek-only.

## Next on GPU (Kaggle / Fields / Codabench image)

1. Cache all 4 Small models; train **per-model** probes + short-trace uncertainty.  
2. Time-box generation (≪4096 tokens); budget cascade with valid fallback.  
3. Cheap CF (K=1–2) emphasizing domain/distract over typos.  
4. Grouped-by-problem calibration; ship `small.txt` ZIP.
