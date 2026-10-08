# label_gen — olympiad self-labelling pipeline (T-014)

Three resumable stages; see the docstring of each script and `collab/tasks/T-014.md`.

| stage | script | unit of work / resume key | raw output (append-only) | derived output |
| --- | --- | --- | --- | --- |
| 1 solve originals | `stage1_solve.py` | `model\|problem_id\|sample` | `stage1_generations.jsonl` | `stage1_summary.jsonl` (base_acc, kept = base_acc >= 0.5) |
| 2 write variants | `stage2_variants.py` | `problem_id\|type\|attempt` | `stage2_raw.jsonl` | `stage2_variants.jsonl`, `stage2_spotcheck.{jsonl,md}` (20 random) |
| 3 solve variants | `stage3_solve_variants.py` | `model\|problem_id\|type\|sample` | `stage3_generations.jsonl` | `labels.jsonl` / `labels.parquet` (train-main-v2 schema + base_acc, perm_acc_<type>, complete) |

Problems: `fetch_problems.py` -> `self-perturb/results/grok/T-014/problems.jsonl`
(173 = 90 AIME + 83 AMC; `problem_id` = `aime-<row>` / `amc-<row>` because the
AMC `id` column repeats).

Resume: re-run the identical command. Records are written (flush + fsync) after
every batch (hf) / chunk (vllm), so a kill loses at most the batch in flight;
a torn last line is skipped. Re-running a finished stage exits before loading a model.

## H100 commands (vLLM 0.31.0 + transformers 5.13.0, one model per process)

```bash
cd self-perturb
P=results/grok/T-014/problems.jsonl; R=<run_dir>   # raw traces: storage Raj names, not git
for M in Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B; do
  X=""; [ "$M" = Qwen/Qwen3.5-4B ] && X="--text-only"
  python scripts/label_gen/stage1_solve.py --backend vllm --model $M --problems $P --out $R $X   # 4 samples, 12288 tok
done
python scripts/label_gen/stage2_variants.py --backend vllm --writer openai/gpt-oss-120b --reasoning-effort medium \
    --dtype auto --problems $P --stage1 $R/stage1_summary.jsonl --out $R
for M in ...same four...; do
  python scripts/label_gen/stage3_solve_variants.py --backend vllm --model $M --problems $P \
      --stage1 $R/stage1_summary.jsonl --variants $R/stage2_variants.jsonl --out $R $X   # 2 samples, 12288 tok
done
```

`--dtype` defaults to float16 (protocol rule 2); gpt-oss-120b (MXFP4) needs `--dtype auto`.
Fallback for any model vLLM cannot run: `--backend hf` (plain transformers generate,
same prompts, sampling, tokenizer and output schema).

Dry run: `dryrun_t014.py` (kill / resume / no-op per stage), Kaggle kernel from `make_kernel_t014.py`.
