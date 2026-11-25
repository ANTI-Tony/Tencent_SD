#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-gen
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Full traffic for C2 (M1, five short domains, greedy, <=512 tokens) and the cache for C1's trainer.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
python gen_traffic.py --model m1 --domains ${DOMS:-chat,sql,code,zh,fc} || echo FAIL_gen
python train_head.py --mode cache --model m1 --domain chat --n 48 --cache $WS/runs/cache/m1_chat.pt || echo FAIL_cache
echo "=== STEP done"
