# Hybrid Cascade (budget-aware Small Models Track)

Method taste: **per-model last-token probes** (primary) with an optional
**K=1–2 counterfactual agreement veto**, plus a reserved short early-trace
uncertainty stage, fused under a **cascade abort &lt; 3600s**.
Public `model_is_robust` / val-sample labels are **proxies**, not a
reconstruction of the unpublished private labelling rule H.

## Layout

```text
hybrid-cascade/
├── solution.py              # are_robust(model_id, reasoning_effort, problems)
├── cascade.py               # probe-first + CF veto, wall budget, all-False
├── uncertainty_features.py  # short early-trace stage (GPU stub without artifact)
├── counterfactual.py        # K=1–2 light transforms + GPU generate (veto)
├── probe_infer.py           # per-model last-token probes (GPU)
├── artifacts/               # per-model probe_*.pkl (4 Small IDs)
└── README.md
```

Small-track packaging: build with `uv run scripts/build.py solutions --small`
so the ZIP root contains blank `small.txt`
→ `dist/solution-hybrid-cascade-small.zip`.

## Cascade order (Codabench GPU)

| Step | Module | What it does | Needs GPU? | Needs local HF weights? | Needs bundled artifact? |
|---:|---|---|---|---|---|
| 1 | `probe_infer` | Last prompt-token @ selected layer; conservative policy gate | **Yes** | **Yes** | **Yes** — one probe pack per Small ID |
| 1b | `counterfactual` veto | Original + K=1–2 transforms; fuse via probe∧agree | **Yes** | **Yes** | No (behavioural) |
| 2 | standalone CF | Only if probes unavailable | **Yes** | **Yes** | No |
| 3 | `uncertainty_features` | Short gen early-trace aggregates → soft score | **Yes** | **Yes** | Optional short-trace `.joblib` |
| 4 | fallback | Deterministic `False` × N | No | No | No |

- If a stage returns `None` (no CUDA, no artifact, no cache) or raises, the
  cascade **skips** to the next stage.
- CF **agreement veto**: when both probe and CF votes are available, a case is
  True only if the probe says True **and** answers agree across transforms.
  Disagreement / unparseable answers force False (precision bias; 949901 failure mode).
- If wall time exceeds ~80% of **3600s**, remaining expensive stages are
  aborted and the fallback is used so coverage stays 1.0 / invalid=0.
- Predictions are always **native Python `bool`**, same length/order as
  `problems`.

## Small Models Track IDs (verbatim)

Artifacts and checkpoints must key on these exact HF ids (Competition does
**not** pass `:low` aliases):

- `Qwen/Qwen3.5-4B`
- `Skywork/Skywork-OR1-Math-7B`
- `allenai/Olmo-3-7B-Think`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

Suggested on-disk names under `artifacts/`:

- `probe_<safe-model-id>.pkl` (schema-v2 ensemble, `best_layer_mean_margin`)
- `uncertainty_<safe-model-id>.joblib` (short-trace regressor + threshold)

`<safe-model-id>` replaces `/`, `:`, and whitespace with `_`.

## CPU box behaviour (this workspace)

No GPU (`torch.cuda.is_available() == False`) and no large HF downloads by
policy ⇒ every GPU stage returns `None` ⇒ **all-False** baseline. That is a
**valid** Codabench contract smoke (coverage=1, invalid=0), not a claim about
private-test accuracy.

## Decision policies (`DECISION_POLICY` in `probe_infer.py`)

Official Small (rmtaware) anchors (measured):

| Sub | method | accuracy |
|---|---|---:|
| 949901 | hybrid default (`legacy`) | 0.4286 |
| 949938 | all-False | 0.5000 |
| 949942 | always-True | 0.5000 |
| 949966 | `symmetric_confident` | 0.5000 |
| **949937** | **conservative hybrid** | **0.5714** ← best / **active default** |

Hidden labels are **balanced** (chance = 0.50). Local val-sample False-rate
0.625 is a misleading prior.

| Policy | Probe True gate | Cascade `MAX_STAGE_TRUE_RATE` | Status |
|---|---|---:|---|
| `legacy` | mean margin ≥ 0 | 1.01 (off) | 949901 |
| **`conservative`** (active) | mean ≥ 0.5 **and** ≥75% seeds ≥ 0 | 0.35 | **949937 (best)** |
| `symmetric_confident` | mean ≥ +0.30 **and** ≥90% seeds ≥ 0; ±τ False gate; dead zone → False | 0.50 | 949966 → 0.50; do not ship |

`are_robust(model_id, reasoning_effort, problems) -> list[bool]` unchanged.
No CUDA / missing artifact / missing local weights still → all-False.

## Local validate & package

```bash
uv run python -m unittest tests.test_hybrid_cascade_conservative -v
uv run scripts/build.py solutions --small
uv run scripts/run_local.py dist/solution-hybrid-cascade-small.zip --small \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

Expect on CPU: coverage `1.0`, invalid_predictions `0`, accuracy equal to the
all-False base rate on the local label mix (report the measured JSON only).

**Do not submit to Codabench without Raj approval** (10/day, 300 total).
