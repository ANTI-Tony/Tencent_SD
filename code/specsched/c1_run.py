"""C1: one (model, regime) x a list of conditions on one GPU.

S0 no spec | S1 spec | S2 no spec + co-located trainer | S3 spec + co-located trainer.
For each condition: start `vllm serve`, optionally start `train_head.py --mode load` on the same GPU,
sweep closed-loop concurrency levels with `vllm bench serve`, record bench JSON + /metrics deltas +
trainer throughput inside each level's window.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

from cfg import DATA, RUNS, model

CODE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
VLLM = os.path.join(os.path.dirname(PY), "vllm")

REGIMES = {
    # domain, thinking, max output tokens, max_model_len, concurrency levels, prompts per level
    "short": dict(dom="chat", thinking=False, outlen=512, maxlen=4096,
                  levels=[1, 2, 4, 8, 16, 32, 64, 128, 256], nprompt=lambda c: min(max(32, 4 * c), 1024)),
    "long": dict(dom="math", thinking=True, outlen=4096, maxlen=6144,
                 levels=[1, 4, 16, 64, 96], nprompt=lambda c: {1: 6, 4: 12}.get(c, 3 * c)),
}


def get(url, timeout=5):
    return urllib.request.urlopen(url, timeout=timeout).read().decode()


def scrape(port):
    vals = {}
    for line in get(f"http://127.0.0.1:{port}/metrics", 10).splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if any(s in name for s in ("spec_decode", "preemption", "generation_tokens", "prompt_tokens",
                                   "kv_cache_usage", "num_requests")):
            try:
                vals[name] = vals.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return vals


def gpu_total_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()[0]
    return float(out) / 1024


def kill(proc):
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def port_is_free(port):
    try:
        get(f"http://127.0.0.1:{port}/health", 3)
        return False      # something answers already (another job on the same node!)
    except Exception as e:
        return "refused" in str(e).lower() or "connection" in str(e).lower()


def default_port():
    """Jobs sharing a node share the network namespace: derive the port from the Slurm job id."""
    base = 8000 + int(os.environ.get("SLURM_JOB_ID", "123")) % 1000
    for p in range(base, base + 50):
        if port_is_free(p):
            return p
    raise RuntimeError("no free port")


def start_server(mc, port, util, maxlen, spec, nspec, logpath, tp=1, spec_schedule=None):
    """spec_schedule: vLLM's static per-batch-size table [(range_start, range_end, k), ...] (k=0 = no drafts)."""
    assert port_is_free(port), f"port {port} already answers /health: another server is alive on this node"
    cmd = [VLLM, "serve", mc["target"], "--port", str(port), "--gpu-memory-utilization", f"{util:.3f}",
           "--max-model-len", str(maxlen), "--max-num-seqs", "256", "--no-enable-prefix-caching",
           "--seed", "0", "--served-model-name", "target"]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    if spec:
        sc = {"method": "eagle3", "model": mc["head"], "num_speculative_tokens": nspec}
        if spec_schedule:
            sc["num_speculative_tokens_per_batch_size"] = [list(x) for x in spec_schedule]
        cmd += ["--speculative-config", json.dumps(sc)]
    logf = open(logpath, "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            raise RuntimeError(f"server died, see {logpath}")
        try:
            if get(f"http://127.0.0.1:{port}/health") is not None:
                return proc, time.time() - t0
        except Exception:
            time.sleep(3)
    raise RuntimeError("server start timeout")


def start_trainer(model_name, cache, duty, tlog, outpath, mps_pct=0):
    cmd = [PY, os.path.join(CODE, "train_head.py"), "--mode", "load", "--model", model_name, "--cache", cache,
           "--duty", str(duty), "--log", tlog, "--log-every", "2"]
    env = dict(os.environ)
    if mps_pct:  # only meaningful when an MPS daemon is running for this GPU
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_pct)
    f = open(outpath, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid, cwd=CODE, env=env)
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"trainer died, see {outpath}")
        if "LOAD_READY" in open(outpath).read():
            return proc
        time.sleep(2)
    raise RuntimeError("trainer start timeout")


def trainer_window(tlog, t0, t1):
    steps = toks = 0
    busy = win = 0.0
    peak = 0.0
    if not os.path.exists(tlog):
        return {}
    for line in open(tlog):
        r = json.loads(line)
        if "steps" in r and t0 <= r["t"] <= t1:
            steps += r["steps"]
            toks += r["tokens"]
            busy += r["busy_s"]
            win += r["win_s"]
            peak = max(peak, r["peak_mem_gb"])
    return dict(trainer_steps=steps, trainer_tok_s=toks / win if win else 0.0,
                trainer_busy_frac=busy / win if win else 0.0, trainer_peak_mem_gb=peak)


def bench(mc, port, rg, c, outdir, fname=None, seed=0):
    fname = fname or f"C{c}.json"
    cmd = [VLLM, "bench", "serve", "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
           "--base-url", f"http://127.0.0.1:{port}", "--model", mc["target"], "--served-model-name", "target",
           "--dataset-name", "custom", "--dataset-path", rg.get("file") or f"{DATA}/bench/{rg['dom']}_eval.jsonl",
           "--skip-chat-template", "--custom-output-len", str(rg["outlen"]), "--num-prompts", str(rg["nprompt"](c)),
           "--max-concurrency", str(c), "--request-rate", "inf", "--temperature", "0",
           "--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": rg["thinking"]}}),
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--save-result", "--save-detailed", "--result-dir", outdir, "--result-filename", fname,
           "--seed", str(seed), "--disable-tqdm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise RuntimeError(f"bench failed at C={c}")
    return json.load(open(os.path.join(outdir, fname)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--regime", default="short", choices=list(REGIMES))
    ap.add_argument("--conds", default="S0,S1,S2,S3")
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--levels", default="")
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--trainer-gb", type=float, default=14.0)
    ap.add_argument("--nspec", type=int, default=3)
    ap.add_argument("--cache", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=0, help="unix time; skip remaining levels after it")
    ap.add_argument("--trainer-mps-pct", type=int, default=0)
    a = ap.parse_args()
    a.port = a.port or default_port()
    print(f"using port {a.port}", flush=True)
    mc = model(a.model)
    rg = dict(REGIMES[a.regime])
    if a.levels:
        rg["levels"] = [int(x) for x in a.levels.split(",")]
    cache = a.cache or f"{RUNS}/cache/{a.model}_chat.pt"
    total = gpu_total_gb()
    for cond in a.conds.split(","):
        if a.deadline and time.time() > a.deadline - 360:
            print(f"[{cond}] not enough time left before deadline, skipping", flush=True)
            continue
        spec = cond in ("S1", "S3")
        coloc = cond in ("S2", "S3")
        util = a.util - (a.trainer_gb / total if coloc else 0.0)
        name = f"{cond}" + (f"_d{a.duty:g}" if coloc else "") + (f"_{a.tag}" if a.tag else "")
        outdir = f"{RUNS}/c1/{a.model}_{a.regime}/{name}"
        os.makedirs(outdir, exist_ok=True)
        rows = []
        srv = trn = None
        try:
            srv, t_start = start_server(mc, a.port, util, rg["maxlen"], spec, a.nspec, f"{outdir}/server.log")
            print(f"[{name}] server up in {t_start:.0f}s util={util:.3f}", flush=True)
            if coloc:
                trn = start_trainer(a.model, cache, a.duty, f"{outdir}/trainer.jsonl", f"{outdir}/trainer.out",
                                    a.trainer_mps_pct)
            bench(mc, a.port, dict(rg, nprompt=lambda c: 8), 4, outdir, "warmup.json")
            for c in rg["levels"]:
                if a.deadline and time.time() > a.deadline:
                    print(f"[{name}] deadline reached, skipping C>={c}", flush=True)
                    break
                m0, t0 = scrape(a.port), time.time()
                res = bench(mc, a.port, rg, c, outdir)
                t1, m1 = time.time(), scrape(a.port)
                d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
                drafts = sum(v for k, v in d.items() if "num_drafts" in k)
                acc = sum(v for k, v in d.items() if "num_accepted_tokens" in k and "per_pos" not in k)
                dtok = sum(v for k, v in d.items() if "num_draft_tokens" in k)
                row = dict(cond=cond, duty=a.duty if coloc else 0, util=util, C=c, t0=t0, t1=t1,
                           acc_len=(1 + acc / drafts) if drafts else None,
                           draft_acc_rate=(acc / dtok) if dtok else None,
                           preemptions=sum(v for k, v in d.items() if "preemption" in k),
                           **{k: res.get(k) for k in res if not isinstance(res.get(k), list)},
                           **trainer_window(f"{outdir}/trainer.jsonl", t0, t1))
                rows.append(row)
                print(f"[{name}] C={c} out_tok/s={res.get('output_throughput', 0):.0f} "
                      f"tpot_p50={res.get('median_tpot_ms', 0):.1f} p99={res.get('p99_tpot_ms', 0):.1f} "
                      f"acc_len={row['acc_len']} trn_tok/s={row.get('trainer_tok_s', 0):.0f}", flush=True)
                json.dump(rows, open(f"{outdir}/summary.json", "w"), indent=1)
        finally:
            kill(trn)
            kill(srv)
            time.sleep(5)


if __name__ == "__main__":
    main()
