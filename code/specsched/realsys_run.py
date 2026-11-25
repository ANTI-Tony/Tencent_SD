"""Real-system duty control: one vLLM server (spec on, stock head) + co-located trainer whose duty is set every
second by controller.py, driven through a load profile. One run per controller policy; identical profiles.

Output: runs/realsys/<policy>/{phase_<i>_C<c>.json, controller.jsonl, trainer.jsonl, server.log, summary.json}
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

from c1_run import PY, VLLM, bench, default_port, kill, scrape, start_server
from cfg import DATA, RUNS, model

CODE = os.path.dirname(os.path.abspath(__file__))


def start_trainer_ctl(model_name, cache, duty_file, tlog, outpath):
    cmd = [PY, os.path.join(CODE, "train_head.py"), "--mode", "load", "--model", model_name, "--cache", cache,
           "--duty", "0", "--duty-file", duty_file, "--log", tlog, "--log-every", "1"]
    f = open(outpath, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid, cwd=CODE)
    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            raise RuntimeError(f"trainer died, see {outpath}")
        if "LOAD_READY" in open(outpath).read():
            return p
        time.sleep(2)
    raise RuntimeError("trainer start timeout")


def start_controller(port, duty_file, log, policy, domain, outpath, fixed_duty=0.0):
    cmd = [PY, os.path.join(CODE, "controller.py"), "--port", str(port), "--duty-file", duty_file, "--log", log,
           "--policy", policy, "--domain", domain, "--fixed-duty", str(fixed_duty)]
    f = open(outpath, "w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid, cwd=CODE)


def window(path, t0, t1, key_steps="steps", key_tok="tokens"):
    steps = toks = 0
    busy = win = 0.0
    duties = []
    if not os.path.exists(path):
        return {}
    for line in open(path):
        r = json.loads(line)
        if "steps" in r and t0 <= r["t"] <= t1:
            steps += r["steps"]; toks += r["tokens"]; busy += r["busy_s"]; win += r["win_s"]
            duties.append(r.get("duty", 0.0))
    return dict(trainer_steps=steps, trainer_tok_s=toks / win if win else 0.0, trainer_busy_frac=busy / win if win else 0.0,
                trainer_duty_mean=sum(duties) / len(duties) if duties else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--domain", default="chat")
    ap.add_argument("--policies", default="fixed0,costmodel,osd_idle,fixed1")
    ap.add_argument("--profile", default="4,64,256,4", help="concurrency phases")
    ap.add_argument("--phase-s", type=float, default=60)
    ap.add_argument("--trainer-gb", type=float, default=12)
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args()
    mc = model(a.model)
    port = default_port()
    rg = dict(dom=a.domain, thinking=False, outlen=512, maxlen=4096, nprompt=lambda c: max(16, 2 * c))
    cache = f"{RUNS}/cache/{a.model}_chat.pt"
    import subprocess as sp
    total = float(sp.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()[0]) / 1024
    util = 0.90 - a.trainer_gb / total
    phases = [int(x) for x in a.profile.split(",")]
    for pol in a.policies.split(","):
        if a.deadline and time.time() > a.deadline - 420:
            print(f"[{pol}] not enough time, skipping", flush=True)
            continue
        outdir = f"{RUNS}/realsys/{a.model}_{a.domain}/{pol}"
        os.makedirs(outdir, exist_ok=True)
        duty_file = f"{outdir}/duty.txt"
        open(duty_file, "w").write("0\n")
        srv = trn = ctl = None
        rows = []
        try:
            srv, ts = start_server(mc, port, util, rg["maxlen"], True, 3, f"{outdir}/server.log")
            print(f"[{pol}] server up in {ts:.0f}s on port {port}", flush=True)
            trn = start_trainer_ctl(a.model, cache, duty_file, f"{outdir}/trainer.jsonl", f"{outdir}/trainer.out")
            policy, fixed = ("fixed", 0.0) if pol == "fixed0" else ("fixed", 1.0) if pol == "fixed1" else (pol, 0.0)
            ctl = start_controller(port, duty_file, f"{outdir}/controller.jsonl", policy, a.domain, f"{outdir}/controller.out", fixed)
            bench(mc, port, dict(rg, nprompt=lambda c: 8), 4, outdir, "warmup.json")
            for i, c in enumerate(phases):
                # size the phase to roughly phase_s seconds: ~out tokens per request / tokens per second per request
                m0, t0 = scrape(port), time.time()
                n_req = max(16, int(c * a.phase_s / 6.0))  # ~6 s per request at 512 tokens, ~80 tok/s per stream
                res = bench(mc, port, dict(rg, nprompt=lambda _c: n_req), c, outdir, f"phase{i}_C{c}.json")
                t1, m1 = time.time(), scrape(port)
                d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
                drafts = sum(v for k, v in d.items() if "num_drafts" in k)
                acc = sum(v for k, v in d.items() if "num_accepted_tokens" in k and "per_pos" not in k)
                row = dict(policy=pol, phase=i, C=c, t0=t0, t1=t1, dur=t1 - t0, acc_len=(1 + acc / drafts) if drafts else None,
                           **{k: res.get(k) for k in res if not isinstance(res.get(k), list)},
                           **window(f"{outdir}/trainer.jsonl", t0, t1))
                rows.append(row)
                print(f"[{pol}] phase {i} C={c} {row['dur']:.0f}s out_tok/s={res.get('output_throughput',0):.0f} "
                      f"tpot_p50={res.get('median_tpot_ms',0):.1f} p99={res.get('p99_tpot_ms',0):.1f} "
                      f"duty={row.get('trainer_duty_mean',0):.2f} trn_tok/s={row.get('trainer_tok_s',0):.0f}", flush=True)
                json.dump(rows, open(f"{outdir}/summary.json", "w"), indent=1)
        finally:
            kill(ctl); kill(trn); kill(srv)
            time.sleep(5)


if __name__ == "__main__":
    main()
