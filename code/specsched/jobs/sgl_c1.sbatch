#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-sglc1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# SGLang C1 slice (sgl_calib.py c1 = c1_run.py on SGLang, output runs/sgl/c1): MODEL, CONDS, DUTIES, LEVELS,
# TRAINER_GB, TAG from the environment; '+' separates list items (sbatch --export splits on commas).
source /fs/scratch/PCS0296/specsched/code/jobs/env_ss312.sh
set -x
DEADLINE=$(( $(date +%s) + 52*60 ))
CONDS=${CONDS//+/,}
LEVELS=${LEVELS//+/,}
DUTIES=${DUTIES:-1.0}
for D in ${DUTIES//+/ }; do
  python sgl_calib.py c1 --model ${MODEL:-m1} --regime short --conds ${CONDS:-S0} --duty $D \
    ${LEVELS:+--levels $LEVELS} --trainer-gb ${TRAINER_GB:-12} ${TAG:+--tag $TAG} --deadline $DEADLINE || echo FAIL_c1
done
echo "=== STEP done"
