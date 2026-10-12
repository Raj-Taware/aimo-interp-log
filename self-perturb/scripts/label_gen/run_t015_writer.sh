# T-015 hybrid: the only big-GPU work. gpt-oss-120b -> gate gold answers + variants for gate + 173 pool problems.
set -euo pipefail
export HF_HOME=~/hf PYTHONUNBUFFERED=1
PY=~/venv-vllm/bin/python
cd ~/aimo/self-perturb
R=~/t015; G=$R/gate; V=$R/variants; mkdir -p $G $V
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
date -Is; echo "--- stage0 gold"
[ -e $G/problems.jsonl ] || $PY scripts/label_gen/stage0_gold.py --backend vllm --dtype auto \
    --trainv2 ~/aimo/data/train-main-v2.parquet --out $G --max-model-len 20480
cat $G/stage0_summary.json
date -Is; echo "--- stage2 variants (gate + pool)"
cat $G/problems.jsonl results/grok/T-014/problems.jsonl > $V/problems_all.jsonl
$PY -c "
import json
rows=[json.loads(l) for l in open('$V/problems_all.jsonl')]
open('$V/all_kept.jsonl','w').write(''.join(json.dumps({'model':'any','problem_id':r['problem_id'],'kept':True})+'\n' for r in rows))
print(len(rows),'problems')"
$PY scripts/label_gen/stage2_variants.py --backend vllm --dtype auto --writer openai/gpt-oss-120b \
    --reasoning-effort medium --problems $V/problems_all.jsonl --stage1 $V/all_kept.jsonl --out $V
date -Is; echo H200_DONE
