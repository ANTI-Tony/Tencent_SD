#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-c2mix
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Mixed-traffic domains (make_mix_domains.py): one C2 slice for domain DOM: one-pass online updates (if not done) then evaluate HEADS in order,
# skipping heads whose eval.json already exists (so slices can be re-submitted until complete).
source /fs/scratch/PCS0296/specsched/code/jobs/env_ss312.sh
set -x
DOM=${DOM:?}
MODEL=${MODEL:-m1}
B=$WS/runs/c2/$MODEL/$DOM
if [ ! -f $B/N3200/model.safetensors ]; then
  python train_head.py --mode stream --model $MODEL --domain $DOM --budgets 100,200,400,800,1600,3200 || echo FAIL_stream
fi
DEADLINE=$(( $(date +%s) + 50*60 ))
for H in ${HEADS:-none N0 N100 N200 N400 N800 N1600 N3200}; do
  [ -f $B/$H/eval.json ] && continue
  [ $(date +%s) -gt $DEADLINE ] && { echo "deadline: stop before $H"; break; }
  python c2_eval.py --model $MODEL --domain $DOM --head $H || echo FAIL_eval_$H
done
echo "=== STEP done"
