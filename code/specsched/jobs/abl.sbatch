#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-abl
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Ablation-matrix worker: runs whatever is still open in MANIFEST until BUDGET_MIN minutes have passed.
# Submit with --partition=gpu --time=HH:MM:00 and BUDGET_MIN accordingly for a long worker.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-55}*60 ))
python abl_runner.py --manifest ${MANIFEST:-$WS/runs/abl/manifest.jsonl} --deadline $DEADLINE ${ONLY:+--only $ONLY} ${MAXRUNS:+--max-runs $MAXRUNS}
echo "=== STEP done"
