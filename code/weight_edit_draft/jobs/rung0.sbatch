#!/bin/bash
#SBATCH --job-name=wed-rung0
#SBATCH --account=PCS0296
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Rung 0: MIT-edited Qwen3.5-0.8B vs base 0.8B (and 2B) as drafts for Qwen3.5-9B, greedy, 256 MATH-500 problems.
set -euo pipefail
source /fs/scratch/PCS0296/specsched/code/weight_edit_draft/jobs/env.sh
RUN=${RUN:-r0}
TARGET=${TARGET:-Qwen/Qwen3.5-9B}
MAXNEW=${MAXNEW:-2048}
THINK=${THINK:-1}
OUT=$PROJ/runs/$RUN
mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "RUN=$RUN TARGET=$TARGET MAXNEW=$MAXNEW THINK=$THINK"

echo "== 0. verify the MIT edit diff (CPU, once)"
python verify_mit_edit.py --manifest data/mit_p50c_manifest.json | tee $OUT/verify_mit_edit.txt

echo "== 1. target greedy trajectories"
if [ ! -s $OUT/target.jsonl ]; then
  python gen_target.py --model $TARGET --problems data/math500_rung0_256.jsonl --out $OUT/target.jsonl \
      --engine vllm --thinking $THINK --max-new $MAXNEW --batch 256 --gpu-util 0.85 || echo "vllm generation failed"
fi
if [ ! -s $OUT/target.jsonl ]; then
  echo "== 1b. fallback: HF batched greedy generation"
  python gen_target.py --model $TARGET --problems data/math500_rung0_256.jsonl --out $OUT/target.jsonl \
      --engine hf --thinking $THINK --max-new $MAXNEW --batch 32
fi

echo "== 2. drafts teacher-forced on the frozen trajectories"
python draft_agree.py --model Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 \
    --target $OUT/target.jsonl --out $OUT/draft_base.npz
python draft_agree.py --model RikiyaT/Qwen3.5-0.8B-weight-space-programming \
    --target $OUT/target.jsonl --out $OUT/draft_edited.npz
python draft_agree.py --model Qwen/Qwen3.5-2B --target $OUT/target.jsonl --out $OUT/draft_2b.npz

echo "== 3. exact accounting"
python analyze.py --target $OUT/target.jsonl --drafts base=$OUT/draft_base.npz edited=$OUT/draft_edited.npz \
    2b=$OUT/draft_2b.npz --ref base --gamma 5 --tokenizer Qwen/Qwen3.5-0.8B --out $OUT/summary.md
echo "== done"
