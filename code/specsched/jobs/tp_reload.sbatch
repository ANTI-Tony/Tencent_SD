#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-tp-reload
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Multi-GPU: reload_draft_weights under tensor parallelism (e2e at TP=2 and TP=4) + serve -tp 2 reload before/after bench
source /fs/scratch/PCS0296/specsched/code/jobs/env.sh
set -x
export VLLM_LOGGING_LEVEL=WARNING
for TP in 2 4; do
  TP=$TP python -m pytest -q -x -s test_reload_draft_e2e_standalone.py -p no:cacheprovider --basetemp=$WS/cache/pytest_tp$TP -W ignore 2>&1 | grep -E "passed|failed|error|Error" | tail -3
done
for TP in 2 4; do
  python tp_serve_reload.py --tp $TP || echo FAIL_tp_serve_$TP
done
echo "=== STEP done"
