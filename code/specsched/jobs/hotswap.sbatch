#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-hotswap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Hot-swap closed loop; MODES='+'-separated (static / train_noswap / train_swap); LIVE=1 streams training data
# from the server through the live tap instead of the precomputed cache.
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
DEADLINE=$(( $(date +%s) + 52*60 ))
for M in ${MODES//+/ }; do
  python hotswap_run.py --model ${MODEL:-m1} --mode $M ${CACHE:+--cache $CACHE} --trainer-gb ${TRAINER_GB:-18} --live ${LIVE:-0} --dom-b ${DOMB:-code} --policy ${POLICY:-costmodel} --spec-switchable ${SPECSW:-0} --spec-switch ${SPECSWITCH:-0} ${SPECSCHED:+--spec-schedule ${SPECSCHED//+/,}} --deadline $DEADLINE || echo FAIL_hotswap_$M
done
echo "=== STEP done"
