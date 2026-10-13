# T-016 progress (Grok)
- step0 PASS (kernel rajtaware10/aimo-t016-step0 v3): vLLM 0.31.0 runs on T4 (auto TRITON_ATTN). Fixes: uninstall torchaudio (cu128 vs torch cu130); Qwen3.5-4B needs --max-num-seqs <=157 (Mamba cache) -> 128.
- tok/s (2048 cap, all hit cap): 8 seqs Qwen 145.5 / DeepSeek TP2 130.8; 64 seqs Qwen 228.7 / DeepSeek 270.9.
- next: gate kernel rajtaware10/aimo-t016-gate (gold -> s1 -> Claude variants -> s3 -> gate_check)
