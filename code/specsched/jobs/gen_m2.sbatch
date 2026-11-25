#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-gen-m2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# MoE (Qwen3-30B-A3B) traffic for C2 + the code cache for the hot-swap loop
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
python gen_traffic.py --model m2 --domains code,sql,zh,chat || echo FAIL_gen
python train_head.py --mode cache --model m2 --domain code --n 800 --cache $WS/runs/cache/m2_code800.pt || echo FAIL_cache
echo "=== STEP done"
