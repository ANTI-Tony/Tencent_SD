#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-tp-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# TP exactness probe at TP=2 (run-to-run vs no-op reload vs zeroed reload) + serve hot-swap at TP=4
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
export VLLM_LOGGING_LEVEL=WARNING
python tp_exactness_probe.py --tp 2 2>&1 | grep -v -i "warning\|INFO " | tail -5 || echo FAIL_probe
: # TP=4 serve already done (runs/tp/m1_tp4_code)
echo "=== STEP done"
