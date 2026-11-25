"""Matrix worker like abl_runner.py (imported unchanged for claiming and stale-claim takeover), but the harness
script comes from the manifest entry ("script", default hotswap_run.py). Used for baselines added after the freeze,
e.g. the TIDE-style policy (hotswap_run_tide.py), so that the frozen abl_runner.py stays untouched."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from abl_runner import claim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--deadline", type=float, required=True)
    ap.add_argument("--margin-min", type=float, default=3.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-runs", type=int, default=0)
    a = ap.parse_args()
    me = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
    runs = [json.loads(l) for l in open(a.manifest) if l.strip()]
    if a.only:
        pre = a.only.split(",")
        runs = [r for r in runs if any(r["run_id"].startswith(p) for p in pre)]
    n_done = 0
    for r in runs:
        out = r["outdir"]
        if os.path.exists(f"{out}/done.json"):
            continue
        fails = int(open(f"{out}/fail.txt").read().strip() or 0) if os.path.exists(f"{out}/fail.txt") else 0
        if fails >= 3:
            continue
        left_min = (a.deadline - time.time()) / 60
        if left_min < r["est_min"] + a.margin_min:
            print(f"[runner] {r['run_id']} needs {r['est_min']} min, {left_min:.0f} left: stop", flush=True)
            break
        if not claim(out, me):
            continue
        print(f"[runner] {time.strftime('%H:%M:%S')} start {r['run_id']} ({left_min:.0f} min left)", flush=True)
        t0 = time.time()
        for fn in os.listdir(out):
            if fn not in (".claim", "fail.txt"):
                p = f"{out}/{fn}"
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        script = r.get("script", "hotswap_run.py")
        p = subprocess.run([sys.executable, os.path.join(HERE, script), *r["args"], "--deadline", str(a.deadline)],
                           cwd=HERE, stdout=open(f"{out}/harness.out", "w"), stderr=subprocess.STDOUT)
        ok = p.returncode == 0 and os.path.exists(f"{out}/done.json")
        print(f"[runner] {time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} {r['run_id']} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if not ok:
            open(f"{out}/fail.txt", "w").write(f"{fails + 1}\n")
            shutil.rmtree(f"{out}/.claim", ignore_errors=True)
        n_done += 1
        if a.max_runs and n_done >= a.max_runs:
            break
    remaining = sum(1 for r in runs if not os.path.exists(f"{r['outdir']}/done.json"))
    print(f"[runner] exit: {remaining} runs still open", flush=True)


if __name__ == "__main__":
    main()
