#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-c2-fixed
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Fixed-length (ignore_eos) saturation measurement: identical token counts and drain shape for every head,
# to separate head quality from output-set effects in the saturation comparison. Writes eval_fixed.json.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
DEADLINE=$(( $(date +%s) + 52*60 ))
for spec in "code none" "code N0" "code N400" "code N800" "code N3200" "sql none" "sql N0" "sql N3200" "fc none" "fc N0" "fc N3200"; do
  set -- $spec
  [ $(date +%s) -gt $DEADLINE ] && { echo "deadline: stop"; break; }
  [ -f $WS/runs/c2/m1/$1/$2/eval_fixed.json ] && continue
  python c2_eval.py --model m1 --domain $1 --head $2 --fixed-len 256 --n-acc 32 --n-lat 2 || echo FAIL_fixed_$1_$2
done
echo "=== STEP done"
