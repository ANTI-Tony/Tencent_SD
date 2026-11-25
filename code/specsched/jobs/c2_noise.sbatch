#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-c2-noise
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Cross-instance noise of the C2 saturation measurement: the same head evaluated in 3 separate engine
# instances (code: none / N400; sql: none / N3200). Writes eval_rep{1,2,3}.json next to eval.json.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
DEADLINE=$(( $(date +%s) + 52*60 ))
for spec in "code none" "code N400" "sql none" "sql N3200"; do
  set -- $spec
  for R in 1 2 3; do
    [ $(date +%s) -gt $DEADLINE ] && { echo "deadline: stop"; break 2; }
    [ -f $WS/runs/c2/m1/$1/$2/eval_rep$R.json ] && continue
    python c2_eval.py --model m1 --domain $1 --head $2 --rep $R || echo FAIL_noise_$1_$2_$R
  done
done
echo "=== STEP done"
