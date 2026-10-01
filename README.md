# AIMO Interpretability Challenge — Research Log (Raj Taware)

> **Active work (2026-10-01 on):** `self-perturb/` holds the current method
> (official best 0.7143, submission 951038). Claude and Grok coordinate through
> `collab/` — start at `collab/BOARD.md`, rules in `collab/PROTOCOL.md`.
> Everything below this note documents the earlier hybrid-cascade attempt.

**Competition:** [AIMO Interpretability Challenge](https://aimo-interp.github.io/) · Codabench [#16180](https://www.codabench.org/competitions/16180/)  
**Track:** Small Models  
**Codabench user:** `rmtaware`  
**Team:** solo (Raj Taware) · contact `rajmtaware@gmail.com`  
**Repo purpose:** durable archive of code, probe artifacts, submission ZIPs, measured results, and method notes from the AIMO Interp Runner agent (NeurIPS 2026 / deadline 1 Nov 2026).

> **Private archive.** Do not commit Codabench passwords, API tokens, or Kaggle credentials. Official scores below are copied from the Codabench API only — never invented.

## Official Codabench Small scores (measured)

| Sub ID | Method | Accuracy | Coverage | Invalid |
|--------|--------|---------:|---------:|--------:|
| 949901 | hybrid-cascade (default 4 probes) | 0.4286 | 1.0 | 0 |
| 949938 | all-False control | 0.5000 | 1.0 | 0 |
| 949942 | always-True control | 0.5000 | 1.0 | 0 |
| **949937** | **conservative hybrid** | **0.5714** | **1.0** | **0** |
| 949966 | symmetric_confident | 0.5000 | 1.0 | 0 |

**Best official so far: 0.5714** (submission 949937).  
Hidden Small labels are **balanced** (all-False = always-True = 0.50). Local val-sample False-rate 0.625 is **not** the official ceiling.

`symmetric_confident` (949966) **regressed to 0.50** — keep conservative as default.

## Local val-sample (`run_local --small`, n=24)

On CPU without CUDA, probe stages no-op → deterministic all-False:

| Method | Accuracy | Coverage | Invalid |
|--------|---------:|---------:|--------:|
| all-False / hybrid CPU fallback / conservative / symmetric_confident | 0.625 | 1.0 | 0 |

## Layout

```
README.md                 ← this file
LOG.md                    ← chronological work log
SHA256SUMS.txt            ← checksums for probes + submissions
MANIFEST.txt              ← byte sizes
code/hybrid-cascade/      ← solution.py + cascade / probe_infer / helpers
artifacts/probes/         ← 4 last-token probe pickles + train metrics
artifacts/research-loop/  ← CPU analysis JSON/CSV from early research
submissions/              ← Codabench-ready ZIPs (Small track)
training/                 ← Kaggle probe training script
notes/                    ← living scoreboard + method taste + Fields draft
tests/                    ← conservative / policy smoke tests
docs/                     ← lab notebook + mini-paper draft (if present)
```

## Small Models Track IDs

- `Qwen/Qwen3.5-4B`
- `Skywork/Skywork-OR1-Math-7B`
- `allenai/Olmo-3-7B-Think`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

## Probe artifacts (Kaggle T4×2, notebook `rajtaware10/aimo-interp-small-probes`)

| File | Model | Best layer (recorded) | Notes |
|------|-------|----------------------:|-------|
| `probe_Qwen_Qwen3.5-4B.pkl` | Qwen3.5-4B | 16 | OOF≈0.625 on n=24 proxy |
| `probe_Skywork_Skywork-OR1-Math-7B.pkl` | Skywork-OR1-Math-7B | 7 | OOF≈0.625 |
| `probe_allenai_Olmo-3-7B-Think.pkl` | Olmo-3-7B-Think | 8 | layer-select≈0.583 |
| `probe_deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.pkl` | DeepSeek-R1-0528-Qwen3-8B | 9 | OOF≈0.625 |

See `artifacts/probes/probe_train_metrics.json` for machine-readable metrics. Val n=24 is smoke-only.

## Method (current best official)

**Budget-aware hybrid cascade** with last-token representation probes:

1. Optional early uncertainty / CF stages (may no-op without CUDA / budget).
2. Per-`model_id` probe inference from `artifacts/`.
3. **Conservative policy** (949937): True only if mean margin ≥ 0.5 and ≥75% seed margins ≥ 0; cascade aborts to all-False if batch True-rate > 0.35.
4. Coverage-safe fallback: all-False.

Later variant `symmetric_confident` (in code + ZIP): softer True τ (0.30), harder seed agree (0.90), True-rate abort 0.50, symmetric ±τ False gate. See `code/hybrid-cascade/README.md`.

## Codabench upload path (API)

Browser upload from the agent box fails (gibberish HTML). Working API flow for account `rmtaware`:

1. `POST /api/api-token-auth/`
2. `POST /api/datasets/` (`type=submission`, `file_size`, `request_sassy_file_name`)
3. `PUT` bytes to `sassy_url`
4. `PUT /api/datasets/completed/<key>/`
5. `POST /api/submissions/` with `phase=30038`, `tasks=[36241]` (Small Model Track), `data=<dataset UUID>`

Phase **30038** = Competition phase. Small task pk **36241** / UUID `5b30ceda-7a5f-442c-a58d-9fe770e2f69c`.

## Reproduce locally (getting-started)

Upstream kit: https://github.com/aimo-interp/getting-started

```bash
# from a checkout of getting-started, copy code/hybrid-cascade → solutions/hybrid-cascade
uv run scripts/run_local.py --small --solution-dir solutions/hybrid-cascade
```

Expect coverage=1.0, invalid_predictions=0. Accuracy on CPU without models ≈ 0.625 (all-False on val-sample).

## What is intentionally omitted

- Codabench / Kaggle passwords and tokens
- Full HF model weights (probes only)
- `solution-trained-probe-small.zip` (~27 MB starting-kit DeepSeek probe) — not our primary path
- Fields Model compute form (draft parked pending Raj review)

## Sources

- https://aimo-interp.github.io/
- https://github.com/aimo-interp/getting-started
- https://www.codabench.org/competitions/16180/
- Proposal paper: arXiv 2607.13899

## License / competition rules

Respect Codabench offline Docker rules (no internet at eval). This archive is for Raj’s research continuity; redistributing competition data or solutions outside the team may violate competition terms — keep **private** until you decide otherwise.
