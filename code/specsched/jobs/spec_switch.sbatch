#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-switch
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Runtime speculation switch: on -> off -> on at saturation on one server (DOMS '+'-separated)
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
for D in ${DOMS//+/ }; do
  python spec_switch_check.py --model ${MODEL:-m1} --dom $D || echo FAIL_switch_$D
done
echo "=== STEP done"
