# Chronological work log — AIMO Interp (Raj Taware / rmtaware)

Times in Asia/Calcutta (IST) unless noted. Scores are **measured** only.

## 2026-09-21 — Bootstrap

- Cloned `aimo-interp/getting-started`; reproduced baselines on val-sample (n=24).
- Local always-true **0.375**; trained-probe with artifact invalid; trivial all-False **0.625** cov=1 inv=0.
- CPU research loop under `/workspace/aimo-research-loop` (text-only leakage bounds, probe metadata, uncertainty FI). Box: **no GPU**.
- Fields Model compute-form draft parked (ask ~200 A100-hours); Raj prefers draft review before submit.
- Standing 2h progress heartbeat routine created on AIMO Interp Runner agent.

## 2026-09-28 — Resume + Kaggle GPU probes

- Raj signed into Kaggle on box desktop (`rajtaware10`); notebook  
  https://www.kaggle.com/code/rajtaware10/aimo-interp-small-probes/edit (T4×2).
- Trained last-token probes for all 4 Small IDs; pickles landed in hybrid-cascade `artifacts/`.
- Built `solution-hybrid-cascade-small.zip` (4 pkls). Local `--small`: acc 0.625 / cov 1 / inv 0.
- Codabench username confirmed: **rmtaware**.
- Box Chrome cannot render Codabench (literary gibberish); curl/API sees real site.
- Raj’s browser upload toast “status 0” (client/network); smoke ZIP same failure → not ZIP contents.

## 2026-09-28 evening — API submits + A/B

| Time (approx IST) | Event |
|-------------------|--------|
| ~19:15 | API submit **949901** hybrid default → Finished **acc 0.4286** |
| ~19:19 | Conservative gates landed in code; ZIP rebuilt |
| ~19:19 | API submit **949937** conservative → Finished **acc 0.5714** (best) |
| ~19:20 | API submit **949938** all-False → **0.50** |
| ~19:23 | API submit **949942** always-True → **0.50** (balanced H confirmed) |
| ~19:25 | Variant `symmetric_confident` packaged for next A/B |

### Decision policy evolution

1. **legacy / default:** mean margin ≥ 0 → True (over-predicted True on H → 0.43).
2. **conservative (best official):** mean margin ≥ 0.5 + ≥75% seeds ≥ 0; True-rate abort 0.35.
3. **symmetric_confident (candidate):** τ=0.30, seed agree 0.90, True-rate abort 0.50, ± False gate.

## Open / next

- Score `symmetric_confident` on Codabench vs 0.5714.
- Improve probe signal (retrain, layers, features) beyond threshold tweaks.
- Fields compute form still parked pending Raj review.
- Deadline: **1 Nov 2026**.

## Artifact provenance

- Probe training: Kaggle notebook + `training/kaggle_aimo_small_probes.py`.
- Packaging: root `solution.py` + helpers + `artifacts/*.pkl` for offline Docker.
- Checksums: `SHA256SUMS.txt`.

| ~19:30 | API submit **949966** symmetric_confident → **0.50** (regression; drop) |

## 2026-09-28 ~19:35 IST — Diagnosis + CF veto hardening (no submit)

- Audited cascade: only probes fire on Codabench; uncertainty + CF were no-ops.
- Root cause of 0.5714 ceiling: probes trained on n=24 **cross-model** val-sample proxies (bal_acc≈chance); threshold A/Bs exhausted.
- Offline plan ranked: (1) model-matched denser probe retrain on Kaggle →0.70 gate, (2) short-trace uncertainty artifacts, (3) CF veto smoke, (4) multi-layer probes, (5) calibrated fusion →0.80+.
- Implemented: restore `DECISION_POLICY=conservative`; wire CF GPU generate+extract+agree; probe-first cascade with `fuse_probe_with_cf_veto`; 16 unit tests OK.
- **No Codabench submit.** Kaggle retrain needs Raj GPU/login.
