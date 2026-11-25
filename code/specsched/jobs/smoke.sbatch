#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# End-to-end smoke test of every Rung-0 component at toy scale.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
step() { echo "=== STEP $1 $(date +%T)"; }

step gen;    python gen_traffic.py --model m1 --domains chat,sql --limit 64 || echo FAIL_gen
step calib;  python train_head.py --mode calib --model m1 --domain chat --n 32 --variants "2,18,33;3,19,34" || echo FAIL_calib
step cache;  mkdir -p $WS/runs/cache && python train_head.py --mode cache --model m1 --domain chat --n 48 --cache $WS/runs/cache/m1_chat.pt || echo FAIL_cache
step load;   python train_head.py --mode load --model m1 --cache $WS/runs/cache/m1_chat.pt --duration 45 --log $WS/runs/cache/load_alone_${SLURM_JOB_ID}.jsonl --log-every 5 || echo FAIL_load
cat $WS/runs/cache/load_alone_${SLURM_JOB_ID}.jsonl
step c1;     python c1_run.py --model m1 --regime short --conds S1,S3 --levels 1,16 --tag smoke --deadline $(( $(date +%s) + 1500 )) || echo FAIL_c1
step stream; python train_head.py --mode stream --model m1 --domain sql --budgets 32,64 || echo FAIL_stream
step eval0;  python c2_eval.py --model m1 --domain sql --head N0 --n-acc 32 --n-lat 4 || echo FAIL_eval0
step eval64; python c2_eval.py --model m1 --domain sql --head N64 --n-acc 32 --n-lat 4 || echo FAIL_eval64
step done
