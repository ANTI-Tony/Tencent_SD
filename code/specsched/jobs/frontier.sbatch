#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-frontier
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Load x checkpoint speculation frontier for one or more domains. DOMS and HEADS use '+' as separator
# (sbatch --export splits on commas). Re-submittable: completed sweeps are skipped.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
DEADLINE=$(( $(date +%s) + 52*60 ))
HEADS=${HEADS:-none+N0+N100+N400+N3200}
for D in ${DOMS//+/ }; do
  python frontier_run.py --model ${MODEL:-m1} --domain $D --heads ${HEADS//+/,} --deadline $DEADLINE || echo FAIL_frontier_$D
done
echo "=== STEP done"
