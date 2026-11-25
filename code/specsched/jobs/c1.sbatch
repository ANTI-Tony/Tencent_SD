#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-c1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# One C1 slice: MODEL, REGIME, CONDS, DUTY, LEVELS, TRAINER_GB, TAG come from the environment (sbatch --export).
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
DEADLINE=$(( $(date +%s) + 52*60 ))
CONDS=${CONDS//+/,}   # sbatch --export splits on commas, so conditions are passed as S0+S1+...
LEVELS=${LEVELS//+/,}
DUTIES=${DUTIES:-${DUTY:-1.0}}
if [ "${MPS:-0}" = "1" ]; then   # spatial sharing: both processes become clients of one MPS server
  export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps_log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && echo "MPS daemon started" || echo FAIL_mps
fi
for D in ${DUTIES//+/ }; do
  python c1_run.py --model ${MODEL:-m1} --regime ${REGIME:-short} --conds ${CONDS:-S0} --duty $D \
    ${LEVELS:+--levels $LEVELS} --trainer-gb ${TRAINER_GB:-12} ${TAG:+--tag $TAG} \
    ${MPS_PCT:+--trainer-mps-pct $MPS_PCT} --deadline $DEADLINE || echo FAIL_c1
done
[ "${MPS:-0}" = "1" ] && echo quit | nvidia-cuda-mps-control
echo "=== STEP done"
