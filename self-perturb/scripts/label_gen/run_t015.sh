#!/usr/bin/env bash
# T-015 driver (Lightning H100). Resumable: every step writes $R/.done/<step>
# and every stage script skips finished keys, so re-running the same phase
# after a kill or a Studio restart continues where it stopped.
#
#   bash scripts/label_gen/run_t015.sh setup|gate|pool1|pool3
#     setup : venv (vLLM 0.31.0 + transformers 5.13.0), GPU/disk checks, background model downloads
#     gate  : stage 0 gold -> stage 1/2/3 on train-main-v2 small-track problems -> gate_check (exit 3 on fail)
#     pool1 : stage 1 on the 173 AIME/AMC problems (4 models) -> stage 2 variants (gpt-oss-120b)
#     pool3 : stage 3 on kept pool pairs (S3_SAMPLES, default 2)
# Run from self-perturb/. Logs: $R/logs/<phase>.log
set -euo pipefail

PHASE=${1:?phase}
HOME_DIR=${HOME_DIR:-/teamspace/studios/this_studio}
R=${R:-$HOME_DIR/t015}
VENV=${VENV:-$HOME_DIR/venv-vllm}
TV2=${TV2:-$HOME_DIR/aimo/data/train-main-v2.parquet}
POOL=results/grok/T-014/problems.jsonl
S3_SAMPLES=${S3_SAMPLES:-2}
MODELS=(Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
WRITER=openai/gpt-oss-120b
export HF_HOME=${HF_HOME:-$HOME_DIR/hf}
export PYTHONUNBUFFERED=1

mkdir -p "$R/.done" "$R/logs"
exec > >(tee -a "$R/logs/$PHASE.log") 2>&1
echo "=== $PHASE start $(date -Is)"

done_() { touch "$R/.done/$1"; echo "--- done $1 $(date -Is)"; }
step() { [ -e "$R/.done/$1" ] && { echo "--- skip $1 (done)"; return 1; }; echo "--- run $1 $(date -Is)"; }

py() { "$VENV/bin/python" "$@"; }

# Qwen3.5-4B: --text-only; fp16 first (rule 2), bfloat16 only if fp16 fails (generation only).
model_flags() { [ "$1" = Qwen/Qwen3.5-4B ] && echo "--text-only" || true; }
run_model() {  # run_model <stage-script> <model> <args...>
  local script=$1 model=$2; shift 2
  local extra; extra=$(model_flags "$model")
  if ! py "scripts/label_gen/$script" --backend vllm --model "$model" $extra "$@"; then
    [ "$model" = Qwen/Qwen3.5-4B ] || return 1
    echo "!!! $model failed in fp16 -> retry bfloat16"
    py "scripts/label_gen/$script" --backend vllm --model "$model" $extra --dtype bfloat16 "$@"
  fi
}

case "$PHASE" in
setup)
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  df -h "$HOME_DIR" | tail -1
  if step venv; then
    python3 -m venv "$VENV"
    py -m pip install -q --upgrade pip
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if [ "$DRV" -ge 580 ]; then py -m pip install -q "vllm==0.31.0"
    else echo "driver $DRV < 580 -> cu129 wheel"; py -m pip install -q "vllm==0.31.0" \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      -f https://github.com/vllm-project/vllm/releases/expanded_assets/v0.31.0; fi
    py -m pip install -q "transformers==5.13.0" pandas pyarrow
    py -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"
    done_ venv
  fi
  if step downloads; then
    for M in "$WRITER" "${MODELS[@]}"; do
      X=(); [ "$M" = "$WRITER" ] && X=(--exclude "original/*" --exclude "metal/*")
      nohup "$VENV/bin/hf" download "$M" "${X[@]}" > "$R/logs/dl_${M//\//_}.log" 2>&1 &
    done
    echo "downloads started in background"; done_ downloads
  fi
  ;;
gate)
  G=$R/gate
  step gold && { py scripts/label_gen/stage0_gold.py --backend vllm --dtype auto --trainv2 "$TV2" --out "$G"; done_ gold; }
  for M in "${MODELS[@]}"; do
    k=gate_s1_${M//\//_}
    step "$k" && { run_model stage1_solve.py "$M" --problems "$G/problems.jsonl" --out "$G"; done_ "$k"; }
  done
  step gate_s2 && { py scripts/label_gen/stage2_variants.py --backend vllm --dtype auto --writer "$WRITER" \
      --reasoning-effort medium --problems "$G/problems.jsonl" --stage1 "$G/stage1_summary.jsonl" --out "$G"; done_ gate_s2; }
  for M in "${MODELS[@]}"; do
    k=gate_s3_${M//\//_}
    step "$k" && { run_model stage3_solve_variants.py "$M" --problems "$G/problems.jsonl" \
      --stage1 "$G/stage1_summary.jsonl" --variants "$G/stage2_variants.jsonl" --out "$G" \
      --source "self-labelled gate (T-015)"; done_ "$k"; }
  done
  py scripts/label_gen/gate_check.py --trainv2 "$TV2" --labels "$G/labels.jsonl" --out "$G/gate_report.json"
  py -c "import json,sys; sys.exit(0 if json.load(open('$G/gate_report.json'))['gate_passes'] else 3)" \
    || { echo "!!! GATE FAILED - stop, Claude decides"; exit 3; }
  done_ gate
  ;;
pool1)
  [ -e "$R/.done/gate" ] || { echo "gate not passed"; exit 3; }
  P=$R/pool
  for M in "${MODELS[@]}"; do
    k=pool_s1_${M//\//_}
    step "$k" && { run_model stage1_solve.py "$M" --problems "$POOL" --out "$P"; done_ "$k"; }
  done
  step pool_s2 && { py scripts/label_gen/stage2_variants.py --backend vllm --dtype auto --writer "$WRITER" \
      --reasoning-effort medium --problems "$POOL" --stage1 "$P/stage1_summary.jsonl" --out "$P"; done_ pool_s2; }
  ;;
pool3)
  P=$R/pool
  [ -e "$R/.done/pool_s2" ] || { echo "pool stage 2 not done"; exit 3; }
  for M in "${MODELS[@]}"; do
    k=pool_s3_${M//\//_}
    step "$k" && { run_model stage3_solve_variants.py "$M" --problems "$POOL" --stage1 "$P/stage1_summary.jsonl" \
      --variants "$P/stage2_variants.jsonl" --out "$P" --samples "$S3_SAMPLES"; done_ "$k"; }
  done
  ;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
echo "=== $PHASE end $(date -Is)"
