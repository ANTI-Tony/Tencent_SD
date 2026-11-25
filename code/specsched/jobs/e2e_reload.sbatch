#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-e2e-reload
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
export VLLM_LOGGING_LEVEL=WARNING
python -m pytest -q -x -s test_reload_draft_e2e_standalone.py -p no:cacheprovider --basetemp=$WS/cache/pytest 2>&1 | grep -v -i "warning" | tail -30
echo "=== STEP done"
