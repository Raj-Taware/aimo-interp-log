# Board

Protocol: `collab/PROTOCOL.md`. Status: open â†’ doing â†’ review â†’ done | redo | blocked.

| Task | Status | Owner | Goal | Needs GPU |
| --- | --- | --- | --- | --- |
| T-001 | done | grok | Canary: does Grok's GPU reproduce the reference features? | yes, ~10 min |
| T-002 | done | grok | Recover ~26 more labelled DeepSeek problems; dump their features | yes, ~30 min |
| T-003 | done | grok | Failure-type specificity: do distractor features predict distractor failures? | no |
| T-004 | done | grok | Does drift carry signal beyond problem length? (pre-registered) | no |
| T-005 | done | grok | Three candidate signals beyond length (pre-registered) | no |
| T-006 | done | grok | Ground-truth answers for the 154 labelled problems | no |
| T-007 | done | grok | Generation smoke test: trace length + accuracy per model (sizes H100 plan) | yes, ~40 min |
| T-008 | done | grok | Re-measure DeepSeek with repaired tokenizer; re-run T-004/T-005 (do first) | yes, ~15 min |
| T-009 | done | grok | Generation smoke v2 with fixed protocol (after T-008) | yes, ~2 h |
| T-010 | done | grok | Local leaderboard (v2, v3b, always-True) + feature dump on train-main-v2 | yes, ~30 min |
| T-011 | done | grok | GPU check of v4 ZIP on train-main-v2 vs expected predictions | yes, ~10 min |
| T-012 | done | grok | Phase 1 geometry dumps (MATH DeepSeek + train-main-v2 x 4 models) | yes, ~45 min |
| T-013 | done | grok | Re-run T-012 dumps with fixed geometry.py (hybrid attention, fp16 overflow) | yes, ~20 min |
| T-014 | done | grok | Build olympiad self-labelling pipeline (vLLM, 3 stages) + Kaggle dry run | yes, ~1 h |
| T-015 | dropped | claude | Lightning H100 run: gate vs official labels, then label 173 olympiad problems x 4 models (after T-014; ask Raj for access) | yes, <=2 H100 slots |
| T-016 | review | grok | T-015 on Kaggle T4x2: vLLM-on-T4 check, gate gold (DeepSeek majority), gate stages with Claude-written variants, gate check; stop for review | yes, <=15 Kaggle GPU h |
| T-017 | blocked | grok | Exploratory per-head distractor attention on train-main-v2 (dumps with per-head matrix, canary, eval_heads.py) | yes, ~1-2 Kaggle h |
| T-018 | done | grok | v4 features on 173 unlabelled AIME/AMC problems -> v4b per-model thresholds (grader now calls once per problem) | yes, ~1-2 Kaggle h |
| T-019 | done | grok | Question-only residual-stream dump (137 MATH DeepSeek, train-main-v2, 173 pool; 4 models) for the pre-registered beyond-length test | yes, ~40 Kaggle min; due Thu 30 Oct 18:00 IST |
| T-020 | doing | grok | Self-made DeepSeek labels on MATH train (40 gate + 161 new, rule variants, vLLM) + v2/state dumps; gate pre-registered | yes, ~10-14 Kaggle GPU h, Raj confirms budget before Phase 2; gate due Wed 29 Oct 12:00 IST |
