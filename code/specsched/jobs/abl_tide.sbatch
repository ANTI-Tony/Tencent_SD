#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-ablt
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Worker for baselines added after the freeze (TIDE-style): like abl.sbatch, but runs abl_runner_any.py, which takes
# the harness script from the manifest entry. Same frozen code; environment = env_ss312.sh (the rebuilt frozen
# environment) unless ENVSH says otherwise.
source /fs/scratch/PCS0296/specsched/code/jobs/${ENVSH:-env_ss312.sh}
set -x
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-55}*60 ))
python abl_runner_any.py --manifest ${MANIFEST:-$WS/runs/abl/manifest_tide.jsonl} --deadline $DEADLINE ${ONLY:+--only $ONLY} ${MAXRUNS:+--max-runs $MAXRUNS}
echo "=== STEP done"
