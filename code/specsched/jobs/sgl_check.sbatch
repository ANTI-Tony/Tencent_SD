#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-sglchk
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Functional (WHAT=func) or speed (WHAT=perf) checks of the SGLang port, sgl_port_check.py.
source /fs/scratch/PCS0296/specsched/code/jobs/env_ss312.sh
set -x
python sgl_port_check.py ${WHAT:-func} --model ${MODEL:-m1} --mem ${MEM:-0.67} --levels ${LEVELS:-1,16,64,256}
echo "=== STEP done"
