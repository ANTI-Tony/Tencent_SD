#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-sgl
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Stock SGLang smoke test (server in env/sgl0510; benchmark client and caches from env_ss312.sh).
source /fs/scratch/PCS0296/specsched/code/jobs/env_ss312.sh
set -x
python sgl_smoke.py --model ${MODEL:-m1} --mem ${MEM:-0.67} --levels ${LEVELS:-1,16,256}
echo "=== STEP done"
