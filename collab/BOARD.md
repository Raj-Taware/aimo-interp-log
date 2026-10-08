# Board

Protocol: `collab/PROTOCOL.md`. Status: open → doing → review → done | redo | blocked.

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
| T-012 | review | grok | Phase 1 geometry dumps (MATH DeepSeek + train-main-v2 x 4 models) | yes, ~45 min |
