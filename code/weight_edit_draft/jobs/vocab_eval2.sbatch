#!/bin/bash
#SBATCH --job-name=wed-vocab2
#SBATCH --account=PCS0296
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/fs/scratch/PCS0296/specsched/logs/%x-%j.out
# Round 2: grid-calibrated k_* heads, mixed-domain bridge, and "surgery + online training" (specsched stream trainer
# run in private workspaces whose models/ symlink points at a surgically built head).
set -uo pipefail
source /fs/scratch/PCS0296/specsched/code/jobs/env_ss312.sh
cd $WS/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
DEADLINE=$(( $(date +%s) + 52*60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
echo "== fit (zh bridge, grid calibration, zero-init)"
python weight_edit_draft/bridge_fit.py --n-fit 2000 --out-report weight_edit_draft/runs/vocab/bridge2.md || { echo "bridge_fit FAILED"; exit 1; }
echo "== fit (zh+chat bridge) $(left) min left"
python weight_edit_draft/bridge_fit.py --n-fit 2000 --fit-domains zh,chat --tag _mix --out-report weight_edit_draft/runs/vocab/bridge2_mix.md || echo "mix fit FAILED"
# private workspaces for "surgery + training": models/qwen3-8b-eagle3 -> the surgical head
train_from() {  # $1 = head folder name under runs/c2/m1/zh, $2 = workspace name
  local W=$WS/ws_$2
  mkdir -p $W/models $W/runs
  ln -sfn $WS/data $W/data
  ln -sfn $WS/runs/c2/m1/zh/$1 $W/models/qwen3-8b-eagle3
  echo "== train from $1 in $W ($(left) min left)"
  WS=$W python train_head.py --mode stream --model m1 --domain zh --budgets 3200 || echo "FAIL_train_$1"
}
train_from g_union gu
train_from z_union zu
train_from b_union bu
eval_head() {  # $1 = head name, $2 = workspace (default main)
  [ $(date +%s) -gt $DEADLINE ] && { echo "deadline: skip $1"; return; }
  echo "== c2_eval $1 in ${2:-main} ($(left) min left)"
  if [ -z "${2:-}" ]; then python c2_eval.py --model m1 --domain zh --head $1 || echo "FAIL_eval_$1"
  else WS=$WS/ws_$2 python c2_eval.py --model m1 --domain zh --head $1 || echo "FAIL_eval_$1"; fi
}
eval_head g_union
eval_head N3200 gu
eval_head N3200 zu
eval_head N3200 bu
eval_head g_union_mix
eval_head b_union_mix
echo "== done"
