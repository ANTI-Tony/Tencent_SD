#!/bin/bash
#SBATCH --job-name=wed-vocab
#SBATCH --account=PCS0296
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Vocabulary surgery on the AngelSlim Qwen3-8B EAGLE-3 head: activation-space bridge + offline screen, then real
# vLLM evaluation (specsched c2_eval) of the exported heads on zh (+chat).
set -uo pipefail
source /fs/scratch/PCS0296/specsched/code/jobs/env_ss312.sh
cd $WS/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
STAGE=${STAGE:-both}
if [ "$STAGE" != "eval" ]; then
  python weight_edit_draft/bridge_fit.py --n-fit ${NFIT:-2000} --out-report weight_edit_draft/runs/vocab/bridge.md || { echo "bridge_fit FAILED"; exit 1; }
fi
if [ "$STAGE" != "fit" ]; then
  DEADLINE=$(( $(date +%s) + 52*60 ))
  for H in ${HEADS:-k_union b_union b_full}; do
    [ $(date +%s) -gt $DEADLINE ] && { echo "deadline: stop before $H"; break; }
    echo "== c2_eval $H"
    if [ "$H" = "N0" ]; then
      python c2_eval.py --model m1 --domain zh --head N0 --rep 9 || echo "FAIL_eval_$H"
    else
      python c2_eval.py --model m1 --domain zh --head $H || echo "FAIL_eval_$H"
    fi
  done
fi
echo "== done"
