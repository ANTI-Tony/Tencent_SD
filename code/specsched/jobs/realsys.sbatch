#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-realsys
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Real-system duty control under a load profile, one run per controller policy.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
DEADLINE=$(( $(date +%s) + 52*60 ))
python realsys_run.py --model m1 --domain chat --policies ${POLICIES:-fixed0,costmodel,osd_idle,fixed1} \
  --profile ${PROFILE:-4,64,256,4} --phase-s ${PHASE_S:-60} --deadline $DEADLINE || echo FAIL_realsys
echo "=== STEP done"
