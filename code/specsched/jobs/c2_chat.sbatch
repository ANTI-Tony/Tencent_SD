#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-c2-chat
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Adapt the head to chat (domain A) and measure every head's acceptance on all domains: gives the
# "adapted-to-A then shifted-to-B" acceptance drops that the drift experiments need.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
B=$WS/runs/c2/m1/chat
if [ ! -f $B/N3200/model.safetensors ]; then
  python train_head.py --mode stream --model m1 --domain chat --budgets 100,400,3200 || echo FAIL_stream
fi
DEADLINE=$(( $(date +%s) + 50*60 ))
for H in none N0 N3200 N400 N100; do
  [ -f $B/$H/eval.json ] && continue
  [ $(date +%s) -gt $DEADLINE ] && { echo "deadline: stop before $H"; break; }
  python c2_eval.py --model m1 --domain chat --head $H --also sql,code,zh,fc --n-lat 6 || echo FAIL_eval_$H
done
echo "=== STEP done"
