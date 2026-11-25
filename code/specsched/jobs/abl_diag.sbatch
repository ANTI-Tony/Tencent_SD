#!/bin/bash
#SBATCH --account=PCS0296
#SBATCH --job-name=ss-diag
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:58:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Memory diagnosis (2026-11-24): abl_tide.sbatch plus per-process GPU memory samplers next to the manifest.
source /fs/scratch/PCS0296/specsched/code/jobs/${ENVSH:-env_ss312.sh}
set -x
D=$(dirname $MANIFEST)
J=$SLURM_JOB_ID
echo "$(date +%s.%N) $(date '+%Y/%m/%d %H:%M:%S.%N') $(hostname)" > $D/gpumem_clock_$J.txt
nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 500 > $D/gpumem_apps_$J.csv 2>/dev/null &
S1=$!
nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -lms 500 > $D/gpumem_gpu_$J.csv 2>/dev/null &
S2=$!
( set +x; while true; do echo "T $(date +%s.%N)"; ps -u $(id -un) -o pid=,ppid=,args= | grep -E 'vllm|train_head|controller' | grep -v grep | cut -c1-220; sleep 2; done ) > $D/gpumem_ps_$J.txt 2>/dev/null &
S3=$!
python alloc_preflight.py > $D/alloc_preflight_$J.txt 2>&1 || true
DEADLINE=$(( $(date +%s) + ${BUDGET_MIN:-55}*60 ))
python abl_runner_any.py --manifest $MANIFEST --deadline $DEADLINE ${ONLY:+--only $ONLY} ${MAXRUNS:+--max-runs $MAXRUNS}
kill $S1 $S2 $S3
echo "=== STEP done"
