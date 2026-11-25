#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-m2prep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# M2 (Qwen3-30B-A3B MoE): a little chat traffic, aux-layer calibration of the stock head, trainer cache for C1.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
python gen_traffic.py --model m2 --domains chat --limit 64 || echo FAIL_gen
python train_head.py --mode calib --model m2 --domain chat --n 32 --variants "2,24,45;3,25,46" || echo FAIL_calib
python train_head.py --mode cache --model m2 --domain chat --n 48 --cache $WS/runs/cache/m2_chat.pt || echo FAIL_cache
python train_head.py --mode load --model m2 --cache $WS/runs/cache/m2_chat.pt --duration 40 \
  --log $WS/runs/cache/load_alone_m2_${SLURM_JOB_ID}.jsonl --log-every 5 || echo FAIL_load
cat $WS/runs/cache/load_alone_m2_${SLURM_JOB_ID}.jsonl
echo "=== STEP done"
