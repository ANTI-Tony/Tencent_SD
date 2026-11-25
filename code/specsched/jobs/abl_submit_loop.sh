#!/bin/bash
# Keeps up to MAXJOBS ss-abl workers queued/running on the debug partition until every run in the manifest
# has done.json. Runs on the login node (sleep + squeue only). Usage: nohup bash abl_submit_loop.sh &
WS=/fs/scratch/PCS0296/specsched
MANIFEST=${MANIFEST:-$WS/runs/abl/manifest.jsonl}
MAXJOBS=${MAXJOBS:-4}
LOG=$WS/logs/abl_submit_loop.log
PY=/fs/scratch/PCS0296/bursar/env/bsr312/bin/python
open_runs() {
  $PY - "$MANIFEST" <<'EOF'
import json, os, sys
runs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
open_ = [r for r in runs if not os.path.exists(r["outdir"] + "/done.json")
         and not (os.path.exists(r["outdir"] + "/fail.txt") and int(open(r["outdir"] + "/fail.txt").read().strip() or 0) >= 3)]
print(len(open_))
EOF
}
while true; do
  n_open=$(open_runs)
  n_jobs=$(squeue -u $USER -h -n ss-abl | wc -l)
  echo "$(date +%F_%T) open_runs=$n_open ss-abl_jobs=$n_jobs" >> $LOG
  if [ "$n_open" -eq 0 ]; then echo "$(date +%F_%T) all runs done" >> $LOG; break; fi
  # each worker does ~4 runs: never queue more workers than the open runs justify
  want=$(( (n_open + 3) / 4 )); [ $want -gt $MAXJOBS ] && want=$MAXJOBS
  while [ "$n_jobs" -lt "$want" ]; do
    sbatch $WS/code/jobs/abl.sbatch >> $LOG 2>&1
    n_jobs=$((n_jobs + 1))
    sleep 2
  done
  sleep 120
done
