# AIMO Interpretability Challenge — Compute Request Proposal

**Applicant:** Raj Taware (independent researcher, solo)
**Email:** rajmtaware@gmail.com
**Competition:** AIMO Interpretability Challenge (NeurIPS 2026), Codabench #16180
**Primary track:** Small Models Track
**Requested:** 1× NVIDIA A100 (40GB), 200 GPU-hours, stop-and-restart through 1 Nov 2026

## PASTE INTO FORM (research proposal)

I am an independent researcher competing solo in the AIMO Interpretability Challenge (NeurIPS 2026 / Codabench). I request a modest training-time allocation of one A100 (40GB) for 200 GPU-hours, used with stop-and-restart across the competition window through 1 November 2026.

Goal. Build BoundaryBet, a hybrid classifier that, for each problem–model pair, predicts whether a frontier math model’s answer is robust or spurious under the challenge protocol. Official submissions run offline in the Codabench Docker interface (are_robust → list[bool], 3600s, no network) with Small Models Track checkpoints already cached on evaluation workers. The allocation is therefore for developing artifacts offline—not for interactive cluster login during scoring.

Method. BoundaryBet fuses (1) a white-box representation probe bank (layer-wise linear margins on final prompt-token hidden states, following the official trained-probe starter) with (2) a behavioural/uncertainty path (token-level confidence statistics over greedy traces, following the uncertainty-profiling baseline). A calibrated fusion head and a Budget Cascade abort expensive probes early so Small Models Track inference stays well under the hour limit. Public model_is_robust labels are treated as development proxies only; the focus is mechanisms that generalize rather than reconstructing the hidden robustness criterion.

Compute plan (200 A100-hours on 1 GPU). Rough split: ~70h forward passes and feature dumps across the four Small Models Track checkpoints (Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think, DeepSeek-R1-0528-Qwen3-8B) on public development data; ~40h probe and uncertainty regressor training with grouped cross-validation; ~50h ablation and fusion calibration; ~40h contingency, failed runs, and Docker-parity packaging checks. Restartable use lets experiments pause between Codabench submission cycles.

Openness and constraints. Training code, probe artifacts, and the technical writeup will be released under an open licence consistent with Fields Model and competition award rules (reproduce-from-report). No evaluation-set leakage and no external APIs at evaluation time. Local CPU validation already confirms packaging (coverage=1.0, invalid_predictions=0); this grant unblocks loading 4–8B checkpoints for real probe training. I can use limited Kaggle GPU hours as a thin interim for the smallest experiments until the allocation starts.

Time constraints. Prefer flexible scheduling anytime through the competition deadline (1 November 2026); no hard blackout windows.
