# T-016 progress (Grok)
- step0 PASS (kernel rajtaware10/aimo-t016-step0 v3): vLLM 0.31.0 runs on T4 (auto TRITON_ATTN). Fixes: uninstall torchaudio (cu128 vs torch cu130); Qwen3.5-4B needs --max-num-seqs <=157 (Mamba cache) -> 128.
- tok/s (2048 cap, all hit cap): 8 seqs Qwen 145.5 / DeepSeek TP2 130.8; 64 seqs Qwen 228.7 / DeepSeek 270.9.
- next: gate kernel rajtaware10/aimo-t016-gate (gold -> s1 -> Claude variants -> s3 -> gate_check)
- gate kernel rajtaware10/aimo-t016-gate v1 pushed ~15:38 IST Oct 13, RUNNING. Est ~6-8 GPU h (gold ~1.3 h, s1 ~2 h, s3 ~3 h at ~250 tok/s). Hard stop 11.3 h in-kernel. Resume: do NOT re-push; wait for COMPLETE, then `kaggle kernels output rajtaware10/aimo-t016-gate -p /workspace/kaggle-aimo-t016/outgate`, copy gate/{stage0_summary.json,problems.jsonl,stage1_summary.jsonl,stage2_variants.jsonl,labels.jsonl,gate_report.json} here, write report, BOARD -> review.
- GPU used so far ~2.0 h (step0 v1 0.1, v2 0.75, v3 ~1.1).
- gate v1 COMPLETE at 41312 s (~11.5 h) but hit in-kernel hard stop after s1: gold kept 1/23 (176/184 gens hit 12288 cap), steps 3-4 never ran -> no labels/gate_report. GATE FAIL (incomplete). Report collab/reports/T-016.md. GPU total ~13.5/15 h. Do not re-push without Claude/Raj.
