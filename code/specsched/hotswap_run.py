"""End-to-end hot-swap loop on one GPU: a vLLM server (speculation on, stock head) serves a traffic profile
A(chat, light) -> B(code, saturated) -> A(chat, light); during B a co-located trainer learns the new domain
from the same requests the server serves (hidden states precomputed into a cache) and exports a head every
N requests; the swap loop reloads each exported head into the running server through POST /collective_rpc
("reload_draft_weights", backported into the installed vLLM). A /metrics sampler records throughput and
acceptance every few seconds.

Modes: static (no trainer), train_noswap (trainer runs, heads never loaded), train_swap (full loop),
controller (duty / exports / swaps / speculation switch all decided by controller.py).
Output: runs/hotswap/<mode>/{timeline.jsonl, events.jsonl, phase*.json, server.log, trainer.*}

Ablation matrix (10-23): every strategy of a scenario shares one engine configuration (--reserve-always 1 keeps
the trainer's memory reserve even when nothing trains), one prompt order per repeat (--file-a1/--file-a2/--file-b
+ --seed) and the frozen code; --spec 0 starts the server without a drafter (plain decoding), --train-from start
runs a fixed-duty trainer through all phases, --outdir / --strategy-name put the run where the matrix expects it.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from c1_run import PY, VLLM, bench, default_port, kill, scrape, start_server
from cfg import DATA, RUNS, model

CODE = os.path.dirname(os.path.abspath(__file__))


def post_rpc(port, method, kwargs):
    body = json.dumps({"method": method, "kwargs": kwargs}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/collective_rpc", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        txt = r.read().decode()
    return time.time() - t0, txt


def gen_tokens(d):
    """Exact generated-token counter delta (the histogram series with the same stem would inflate it)."""
    for k in ("vllm:generation_tokens_total", "vllm:generation_tokens"):
        if k in d:
            return d[k]
    return sum(v for k, v in d.items() if k.endswith("generation_tokens_total") or k.endswith("generation_tokens"))


def sampler(port, path, stop, every=5.0):
    prev, t_prev = scrape(port), time.time()
    with open(path, "w") as f:
        while not stop.is_set():
            time.sleep(every)
            try:
                cur = scrape(port)
            except Exception:
                continue
            now = time.time()
            d = {k: cur.get(k, 0) - prev.get(k, 0) for k in cur}
            drafts = sum(v for k, v in d.items() if "num_drafts" in k)
            acc = sum(v for k, v in d.items() if "num_accepted_tokens" in k and "per_pos" not in k)
            gen = gen_tokens(d)
            f.write(json.dumps(dict(t=now, gen_tok_s=gen / (now - t_prev), acc_len=(1 + acc / drafts) if drafts else None,
                                    running=cur.get("vllm:num_requests_running", 0),
                                    preempt=sum(v for k, v in d.items() if "preemption" in k))) + "\n")
            f.flush()
            prev, t_prev = cur, now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--mode", default="train_swap", choices=["static", "train_noswap", "train_swap", "controller"],
                    help="controller = duty, exports and swaps all decided by controller.py (needs --live 1)")
    ap.add_argument("--policy", default="costmodel", choices=["costmodel", "osd_idle", "accdrop", "fixed", "switch_only", "simple"])
    ap.add_argument("--fixed-duty", type=float, default=1.0, help="controller policies fixed / simple")
    ap.add_argument("--max-duty", type=float, default=1.0)
    ap.add_argument("--spec-switchable", type=int, default=0, help="controller assumption about the engine (see controller.py)")
    ap.add_argument("--spec-switch", type=int, default=0,
                    help="1 = the server has the set_speculation backport and the controller enforces the frontier "
                         "switch through it (mode directory gets a _sw suffix)")
    ap.add_argument("--spec-schedule", default="",
                    help="static per-batch-size speculative-token table, e.g. '1,64,3;65,256,0' (vLLM 0.27 "
                         "num_speculative_tokens_per_batch_size): the frontier switch as it exists today")
    ap.add_argument("--spec", type=int, default=1, help="0 = start the server without a drafter (plain decoding)")
    ap.add_argument("--reserve-always", type=int, default=0,
                    help="1 = subtract the trainer's memory reserve from the server even in static mode (equal KV budget)")
    ap.add_argument("--dom-a", default="chat")
    ap.add_argument("--dom-b", default="code")
    ap.add_argument("--cache", default=f"{RUNS}/cache/m1_code800.pt")
    ap.add_argument("--export-at", default="100,200,400,800",
                    help="trainer export schedule (consumed requests); 'doubling:100' = 100,200,400,... without end")
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--train-from", default="B", choices=["B", "start"],
                    help="fixed-duty modes: start the trainer at phase B (regime known) or before A1 and never stop")
    ap.add_argument("--CA", type=int, default=16)
    ap.add_argument("--CB", type=int, default=256)
    ap.add_argument("--nA", type=int, default=120, help="requests in each A phase")
    ap.add_argument("--nB", type=int, default=9000, help="requests in the B phase (~5 min at saturation)")
    ap.add_argument("--file-a1", default="", help="prompt file for phase A1 (default data/bench/<dom-a>_eval.jsonl)")
    ap.add_argument("--file-a2", default="", help="prompt file for phase A2 (default: same as A1)")
    ap.add_argument("--file-b", default="", help="prompt file for phase B (default data/bench/<dom-b>_pool.jsonl)")
    ap.add_argument("--seed", type=int, default=0, help="bench sampling seed and trainer seed")
    ap.add_argument("--trainer-gb", type=float, default=18)
    ap.add_argument("--deadline", type=float, default=0)
    ap.add_argument("--live", type=int, default=0,
                    help="1 = the trainer learns from the server's live tap (hidden states exported by the engine) "
                         "instead of the precomputed cache; in static mode the tap runs without a consumer (overhead)")
    ap.add_argument("--tap-rate", type=float, default=1.0, help="fraction of requests the engine exports")
    ap.add_argument("--tap-gb", type=float, default=4.0, help="ring-buffer size in /dev/shm")
    ap.add_argument("--outdir", default="", help="explicit output directory (default runs/hotswap/<model>_<dom-b>/<mode...>)")
    ap.add_argument("--strategy-name", default="", help="label written into done.json")
    a = ap.parse_args()
    mc = model(a.model)
    port = default_port()
    schedule = [tuple(int(v) for v in r.split(",")) for r in a.spec_schedule.split(";") if r.strip()]
    outdir = a.outdir or (f"{RUNS}/hotswap/{a.model}_{a.dom_b}/{a.mode}" + ("_sw" if a.spec_switch else "")
                          + ("_live" if a.live else "") + ("_sched" if schedule else ""))
    os.makedirs(outdir, exist_ok=True)
    heads = f"{outdir}/heads"
    total = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()[0]) / 1024
    util = 0.90 - (a.trainer_gb / total if (a.mode != "static" or a.reserve_always) else 0.0)
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    tap = f"/dev/shm/specsched_tap_{os.environ.get('SLURM_JOB_ID', 'x')}_{port}"
    if a.live:
        os.environ["VLLM_SPEC_LIVE_TAP"] = tap
        os.environ["VLLM_SPEC_LIVE_TAP_GB"] = str(a.tap_gb)
        os.environ["VLLM_SPEC_LIVE_TAP_RATE"] = str(a.tap_rate)
    else:
        os.environ.pop("VLLM_SPEC_LIVE_TAP", None)
    events = open(f"{outdir}/events.jsonl", "w")
    t_run0 = time.time()

    def ev(**kw):
        kw["t"] = time.time()
        events.write(json.dumps(kw) + "\n")
        events.flush()
        print(json.dumps(kw), flush=True)

    srv = trn = ctl = None
    stop = threading.Event()
    duty_file, cmd_file, dom_file = f"{outdir}/duty.txt", f"{outdir}/cmd.txt", f"{outdir}/domain.txt"
    ctl_mode = a.mode == "controller"
    assert not ctl_mode or a.live, "controller mode needs the live tap (--live 1)"
    assert a.spec or a.mode == "static", "without a drafter nothing can be trained or swapped"
    export_args = (["--export-doubling", a.export_at.split(":")[1]] if a.export_at.startswith("doubling:")
                   else ["--export-at", a.export_at])

    def set_domain(dom):
        open(dom_file, "w").write(dom + "\n")

    def start_trainer(duty):
        src = ["--mode", "live", "--tap", tap] if a.live else ["--mode", "load", "--cache", a.cache, "--order", "seq"]
        cmd = [PY, os.path.join(CODE, "train_head.py"), *src, "--model", a.model,
               *(["--export-at", ""] if ctl_mode else export_args), "--export-dir", heads, "--duty", str(duty),
               "--duty-file", duty_file, "--cmd-file", cmd_file, "--log", f"{outdir}/trainer.jsonl", "--log-every", "1",
               "--seed", str(a.seed)]
        p = subprocess.Popen(cmd, stdout=open(f"{outdir}/trainer.out", "w"), stderr=subprocess.STDOUT,
                             preexec_fn=os.setsid, cwd=CODE)
        ev(event="trainer_started", duty=duty)
        return p

    def wait_trainer(p):
        t0 = time.time()
        while "LOAD_READY" not in open(f"{outdir}/trainer.out").read():
            if p.poll() is not None or time.time() - t0 > 600:
                raise RuntimeError(f"trainer did not start, see {outdir}/trainer.out")
            time.sleep(2)

    done = False
    try:
        srv, ts = start_server(mc, port, util, 4096, bool(a.spec), 3, f"{outdir}/server.log", spec_schedule=schedule)
        ev(event="server_up", seconds=ts, port=port, util=util, spec=a.spec, spec_schedule=schedule)
        # phase B serves a large pool of unique prompts (data/bench/<dom>_pool.jsonl): the trainer learns from the
        # served requests, so a small cycling pool would let it memorise the very prompts it is later measured on
        pool = a.file_b or f"{DATA}/bench/{a.dom_b}_pool.jsonl"
        if not os.path.exists(pool):
            raise RuntimeError(f"missing unique prompt pool {pool} (make_bench_pool.py)")
        file_a1 = a.file_a1 or None
        file_a2 = a.file_a2 or file_a1
        rg = lambda dom, n, file=None: dict(dom=dom, thinking=False, outlen=512, maxlen=4096, nprompt=lambda c: n, file=file)
        ev(event="pool", file=pool, rows=sum(1 for _ in open(pool)), file_a1=file_a1, file_a2=file_a2, seed=a.seed)
        bench(mc, port, rg(a.dom_a, 8), 4, outdir, "warmup.json", seed=a.seed)
        th = threading.Thread(target=sampler, args=(port, f"{outdir}/timeline.jsonl", stop), daemon=True)
        th.start()
        if ctl_mode:
            # the trainer idles at duty 0 from the start; the controller owns duty / exports / swaps from here on
            set_domain(a.dom_a)
            open(duty_file, "w").write("0\n")
            trn = start_trainer(0.0)
            wait_trainer(trn)
            ctl = subprocess.Popen([PY, os.path.join(CODE, "controller.py"), "--port", str(port), "--duty-file", duty_file,
                                    "--cmd-file", cmd_file, "--domain-file", dom_file, "--heads", heads,
                                    "--trainer-log", f"{outdir}/trainer.jsonl", "--log", f"{outdir}/controller.jsonl",
                                    "--policy", a.policy, "--fixed-duty", str(a.fixed_duty), "--hot-swap", "1",
                                    "--max-duty", str(a.max_duty),
                                    "--max-seqs", "256", "--spec-switchable", str(a.spec_switchable),
                                    "--spec-switch", str(a.spec_switch),
                                    *(["--model-json", mj] if (mj := os.path.join(CODE, "sched", f"model_{a.model}.json")) and os.path.exists(mj) else [])],
                                   stdout=open(f"{outdir}/controller.out", "w"), stderr=subprocess.STDOUT,
                                   preexec_fn=os.setsid, cwd=CODE)
            ev(event="controller_started", policy=a.policy)
        swapped = set()

        def swap_watcher():
            while not stop.is_set():
                time.sleep(2)
                if a.mode != "train_swap" or not os.path.isdir(heads):
                    continue
                for d in sorted(os.listdir(heads), key=lambda x: int(x[1:])):
                    p = f"{heads}/{d}"
                    if d in swapped or not os.path.exists(f"{p}/model.safetensors"):
                        continue
                    try:
                        dt, txt = post_rpc(port, "reload_draft_weights", {"weights_path": p})
                        ev(event="swap", head=d, seconds=dt, resp=txt[:120])
                    except Exception as e:
                        ev(event="swap_error", head=d, error=str(e)[:200])
                    swapped.add(d)
        sw = threading.Thread(target=swap_watcher, daemon=True)
        if not ctl_mode and a.mode != "static" and a.train_from == "start":
            # fixed-duty trainer without regime knowledge: runs (and swaps) through A1, B and A2
            open(duty_file, "w").write(f"{a.duty}\n")
            trn = start_trainer(a.duty)
            wait_trainer(trn)
            sw.start()
        # phase A1
        ev(event="phase", name="A1", dom=a.dom_a, C=a.CA)
        bench(mc, port, rg(a.dom_a, a.nA, file_a1), a.CA, outdir, "phaseA1.json", seed=a.seed)
        # phase B: start the trainer (duty-controlled), the swap watcher, and the saturating bench
        ev(event="phase", name="B", dom=a.dom_b, C=a.CB)
        if ctl_mode:
            set_domain(a.dom_b)
        elif a.train_from == "B":
            open(duty_file, "w").write(f"{a.duty if a.mode != 'static' else 0}\n")
            if a.mode != "static":
                trn = start_trainer(a.duty)
        if not sw.is_alive():
            sw.start()
        bench(mc, port, rg(a.dom_b, a.nB, pool), a.CB, outdir, "phaseB.json", seed=a.seed)
        if not ctl_mode and a.train_from == "B":
            open(duty_file, "w").write("0\n")
        ev(event="phase_end", name="B")
        # phase A2
        ev(event="phase", name="A2", dom=a.dom_a, C=a.CA)
        if ctl_mode:
            set_domain(a.dom_a)
        bench(mc, port, rg(a.dom_a, a.nA, file_a2), a.CA, outdir, "phaseA2.json", seed=a.seed)
        ev(event="done")
        done = True
    finally:
        stop.set()
        kill(ctl)
        kill(trn)
        kill(srv)
        time.sleep(5)
        if os.path.exists(tap):
            os.remove(tap)
        if done:
            json.dump(dict(vars(a), outdir=outdir, util=util, port=port, server_seconds=ts, wall_seconds=time.time() - t_run0,
                           job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename, finished_at=time.time()),
                      open(f"{outdir}/done.json", "w"), indent=1)


if __name__ == "__main__":
    main()
