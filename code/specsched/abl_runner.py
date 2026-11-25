"""Worker for the ablation matrix: pulls unfinished runs from the manifest, one at a time, until the job's
deadline. Several jobs (debug and gpu partitions) share one manifest; a run is claimed by creating
<outdir>/.claim (atomic mkdir on the shared file system) and is finished when the harness wrote done.json.
A claim left by a job that is no longer running (timeout, node failure) is stale and gets taken over.
Failures are retried twice; the third failure parks the run (fail.txt) for a human.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))


def job_alive(jobid):
    if not jobid:
        return False
    r = subprocess.run(["squeue", "-h", "-j", str(jobid), "-o", "%T"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() not in ("", "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED")


def claim(outdir, me):
    os.makedirs(outdir, exist_ok=True)
    lock = f"{outdir}/.claim"
    try:
        os.mkdir(lock)
    except FileExistsError:
        try:
            owner = open(f"{lock}/job").read().strip()
        except Exception:
            owner = ""
        if owner == me or job_alive(owner):
            return False
        print(f"[runner] stale claim by job {owner!r} on {outdir}: taking over", flush=True)
        shutil.rmtree(lock, ignore_errors=True)
        try:
            os.mkdir(lock)
        except FileExistsError:
            return False
    open(f"{lock}/job", "w").write(me + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--deadline", type=float, required=True, help="unix time the job must be done by")
    ap.add_argument("--margin-min", type=float, default=3.0, help="start a run only if est_min + margin fits")
    ap.add_argument("--only", default="", help="comma-separated run_id prefixes to restrict to")
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
        # a clean directory: a previous failed attempt must not leave heads / logs behind
        for fn in os.listdir(out):
            if fn not in (".claim", "fail.txt"):
                p = f"{out}/{fn}"
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        p = subprocess.run([sys.executable, os.path.join(CODE, "hotswap_run.py"), *r["args"], "--deadline", str(a.deadline)],
                           cwd=CODE, stdout=open(f"{out}/harness.out", "w"), stderr=subprocess.STDOUT)
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
