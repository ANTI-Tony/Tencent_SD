"""Functional and speed checks of the SGLang port (sgl/specsched_sgl.py) before any calibration run.

func: one patched speculative server with the live tap on.
  1. /metrics carries every counter the controller and the sampler read.
  2. Greedy outputs with speculation on and off (set_speculation through /collective_rpc), batch size 1.
  3. Switch under load: 64 requests at concurrency 16 while speculation is toggled every 0.5 s; every request must
     finish, drafts must stop while it is off.
  4. Draft reload: acceptance with the factory head, the factory head reloaded, a head trained on code in the vLLM
     study, a head with its fc layer zeroed, the factory head again; the target's greedy output must not change.
  5. Tap (server stopped, ring kept): for the batch-1 requests every position from 0 to the last input appears once,
     tokens equal prompt + output[:-1], a FINISH record follows; the aux hidden states match Hugging Face's
     (inputs of layers 2, L/2, L-3) row by row.
perf: output throughput at C = 1, 16, 64, 256 of the patched speculative server with speculation on and off, and of
  the patched server without a drafter (the stock plain server is in runs/sgl_smoke).
Output: runs/sgl_check/<model>/{func,perf}/report.json
Usage: python sgl_port_check.py func|perf --model m1 [--mem 0.67]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c1_run import bench, default_port, get, kill  # noqa: E402
from cfg import DATA, RUNS, model  # noqa: E402
from sgl_common import start_sgl_server  # noqa: E402

REQUIRED = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:generation_tokens_total",
            "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"]
TRAINED = {"m1": f"{RUNS}/abl/m1_code/fixed800/r1/heads/N800", "m2": f"{RUNS}/abl/m2_code/fixed800/r1/heads/N800"}


def gpu_mem_mib():
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(r.stdout.split()[0])


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def metrics(port):
    vals = {}
    for line in get(f"http://127.0.0.1:{port}/metrics", 10).splitlines():
        if line.startswith("#") or not line.strip():
            continue
        vals[line.split("{")[0].split(" ")[0]] = float(line.rsplit(" ", 1)[1])
    return vals


def rpc(port, method, **kw):
    t0 = time.time()
    r = dict(post(port, "/collective_rpc", dict(method=method, kwargs=kw))["results"][0])
    r["http_seconds"] = round(time.time() - t0, 3)
    return r


def gen(port, ids, rid, n):
    r = post(port, "/generate", dict(input_ids=ids, rid=rid, sampling_params=dict(temperature=0, max_new_tokens=n)))
    return r["output_ids"]


def prompts(tok, k, file):
    out = []
    for line in open(file):
        p = json.loads(line)["prompt"]
        out.append(list(tok.apply_chat_template([dict(role="user", content=p)], add_generation_prompt=True,
                                                enable_thinking=False, tokenize=True, return_dict=False)))
        if len(out) == k:
            return out
    return out


def acceptance(port, ids, tag, conc=4, n=128):
    m0 = metrics(port)
    with cf.ThreadPoolExecutor(conc) as ex:
        outs = list(ex.map(lambda x: gen(port, x[1], f"{tag}-{x[0]}", n), enumerate(ids)))
    m1 = metrics(port)
    d = {k: m1[k] - m0.get(k, 0) for k in m1}
    drafts, acc = d["vllm:spec_decode_num_drafts_total"], d["vllm:spec_decode_num_accepted_tokens_total"]
    return dict(acc_len=round(1 + acc / max(drafts, 1), 4), drafts=drafts, accepted=acc,
                tokens=sum(len(o) for o in outs))


def prepare_heads(mc, out):
    """factory head re-exported in the trainer's format (model.safetensors), and a copy with fc zeroed"""
    import torch
    from safetensors.torch import load_file, save_file

    from head import EagleHead, load_cfg, load_sd
    fac, zero = f"{out}/heads/factory", f"{out}/heads/fc_zero"
    EagleHead(load_sd(mc["head"]), load_cfg(mc["head"])).export(fac, mc["head"])
    sd = load_file(f"{fac}/model.safetensors")
    sd["fc.weight"] = torch.zeros_like(sd["fc.weight"])
    os.makedirs(zero, exist_ok=True)
    save_file(sd, f"{zero}/model.safetensors", metadata={"format": "pt"})
    return fac, zero


def read_tap(path):
    import numpy as np

    from livetap import K_CHUNK, K_FINISH, K_GAP, Ring, decode_chunk, decode_id
    ring = Ring(path)
    ring.rewind_to_oldest()
    rows, fin = {}, {}
    while True:
        got = 0
        for kind, payload in ring.read(max_records=100000):
            got += 1
            if kind == K_CHUNK:
                rid, n, start, plen, hid, tok, aux = decode_chunk(payload)
                rows.setdefault(rid, []).append((start, plen, tok.copy(), aux.copy()))
            elif kind in (K_FINISH, K_GAP):
                fin[decode_id(payload)] = "finish" if kind == K_FINISH else "gap"
        if not got:
            break
    return rows, fin, ring.lapped, np


def check_tap(tap, reqs, mc, n_hf=4):
    rows, fin, lapped, np = read_tap(tap)
    res = dict(lapped=lapped, requests=len(rows), finish=sum(v == "finish" for v in fin.values()),
               gap=sum(v == "gap" for v in fin.values()), per_request={})
    seqs = {}
    for rid, (ids, out) in reqs.items():
        ch = sorted(rows.get(rid, []), key=lambda r: r[0])
        pos, ok = 0, bool(ch)
        for start, plen, tok, aux in ch:
            ok &= start == pos and plen == len(ids)
            pos += len(tok)
        want = ids + out[:-1]
        toks = np.concatenate([c[2] for c in ch]).tolist() if ch else []
        res["per_request"][rid] = dict(contiguous=ok, rows=pos, expected_rows=len(want), tokens_equal=toks == want,
                                       end=fin.get(rid), chunks=len(ch))
        if ch:
            seqs[rid] = (want, np.concatenate([c[3] for c in ch]))
    # aux hidden states against Hugging Face
    import torch
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(mc["target"], dtype=torch.bfloat16).to("cuda").eval()
    L = hf.config.num_hidden_layers
    layers = [2, L // 2, L - 3]
    cmp = {}
    for rid in list(seqs)[:n_hf]:
        want, aux = seqs[rid]
        with torch.no_grad():
            hs = hf(torch.tensor([want], device="cuda"), output_hidden_states=True).hidden_states
        ref = torch.cat([hs[i][0] for i in layers], dim=-1).float()
        got = torch.from_numpy(aux.astype(np.int16)).view(torch.bfloat16).float().cuda()
        n = min(len(ref), len(got))
        cos = torch.nn.functional.cosine_similarity(ref[:n], got[:n], dim=-1)
        rel = (ref[:n] - got[:n]).norm(dim=-1) / ref[:n].norm(dim=-1)
        cmp[rid] = dict(rows=n, cos_min=round(cos.min().item(), 5), cos_mean=round(cos.mean().item(), 5),
                        rel_err_mean=round(rel.mean().item(), 4), rel_err_max=round(rel.max().item(), 4))
    res["hf_aux"] = dict(layers=layers, per_request=cmp)
    return res


def func(a):
    from transformers import AutoTokenizer
    mc = model(a.model)
    out = f"{RUNS}/sgl_check/{a.model}/func"
    os.makedirs(out, exist_ok=True)
    port = default_port()
    tap = f"/dev/shm/specsched_tap_{os.environ.get('SLURM_JOB_ID', 'x')}_{port}"
    env = dict(os.environ, VLLM_SPEC_LIVE_TAP=tap, VLLM_SPEC_LIVE_TAP_GB="2", VLLM_SPEC_LIVE_TAP_RATE="1")
    fac, zero = prepare_heads(mc, out)
    tok = AutoTokenizer.from_pretrained(mc["target"])
    ids = prompts(tok, 16, f"{DATA}/bench/code_eval.jsonl")
    rep = dict(model=a.model, mem=a.mem, tap=tap)
    srv, rep["server_seconds"] = start_sgl_server(mc, port, a.mem, 1, f"{out}/server.log", env=env)
    reqs = {}
    try:
        _func_body(a, port, ids, reqs, rep, fac, zero)
    except Exception:
        import traceback
        rep["error"] = traceback.format_exc()
        print(rep["error"], flush=True)
    finally:
        json.dump(rep, open(f"{out}/report.json", "w"), indent=1, default=str)
        kill(srv)
        time.sleep(5)
    try:
        rep["tap"] = check_tap(tap, reqs, mc)
    except Exception:
        import traceback
        rep["tap_error"] = traceback.format_exc()
    if os.path.exists(tap):
        os.remove(tap)
    json.dump(rep, open(f"{out}/report.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str)[:8000], flush=True)


def _func_body(a, port, ids, reqs, rep, fac, zero):
    m = metrics(port)
    rep["metrics_missing"] = [k for k in REQUIRED if k not in m]
    print("metrics", sorted(m), flush=True)
    on = [gen(port, x, f"on-{i}", 128) for i, x in enumerate(ids[:8])]
    rep["rpc_off"] = rpc(port, "set_speculation", enabled="0")
    m0 = metrics(port)
    off = [gen(port, x, f"off-{i}", 128) for i, x in enumerate(ids[:8])]
    m1 = metrics(port)
    rep["rpc_on"] = rpc(port, "set_speculation", enabled="1")
    rep["off_period_drafts"] = m1["vllm:spec_decode_num_drafts_total"] - m0["vllm:spec_decode_num_drafts_total"]
    rep["off_period_plain_steps"] = m1["specsched:plain_steps_total"] - m0["specsched:plain_steps_total"]
    div = []
    for o, f in zip(on, off):
        k = next((j for j, (x, y) in enumerate(zip(o, f)) if x != y), None)
        div.append(None if k is None and len(o) == len(f) else (k if k is not None else min(len(o), len(f))))
    rep["greedy_on_vs_off"] = dict(identical=sum(d is None for d in div), of=len(div), first_divergence=div)
    for i in range(8):
        reqs[f"on-{i}"], reqs[f"off-{i}"] = (ids[i], on[i]), (ids[i], off[i])
    print("greedy", rep["greedy_on_vs_off"], flush=True)
    # speculation toggled every 0.5 s under load
    stop, flips = threading.Event(), [0]

    def toggler():
        on_ = True
        while not stop.is_set():
            time.sleep(0.5)
            on_ = not on_
            rpc(port, "set_speculation", enabled="1" if on_ else "0")
            flips[0] += 1
        rpc(port, "set_speculation", enabled="1")
    m0 = metrics(port)
    th = threading.Thread(target=toggler)
    th.start()
    t0 = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda i: gen(port, ids[i % 16], f"load-{i}", 256), range(64)))
    stop.set()
    th.join()
    m1 = metrics(port)
    d = {k: m1[k] - m0.get(k, 0) for k in m1}
    rep["switch_under_load"] = dict(completed=len(res), tokens=sum(len(r) for r in res), flips=flips[0],
                                    seconds=round(time.time() - t0, 1), drafts=d["vllm:spec_decode_num_drafts_total"],
                                    plain_steps=d["specsched:plain_steps_total"],
                                    decode_steps=d["specsched:decode_steps_total"],
                                    gen_tokens=d["vllm:generation_tokens_total"])
    print("switch", rep["switch_under_load"], flush=True)
    # draft reload
    acc = {"factory_start": acceptance(port, ids, "acc0")}
    for name, path in [("factory_reloaded", fac), ("trained_code_N800", TRAINED[a.model]), ("fc_zero", zero),
                       ("factory_again", fac)]:
        acc[name] = dict(rpc=rpc(port, "reload_draft_weights", weights_path=path),
                         **acceptance(port, ids, f"acc-{name}"))
        print("reload", name, acc[name], flush=True)
    rep["reload"] = acc
    rep["target_unchanged_after_reloads"] = gen(port, ids[0], "after-0", 128) == on[0]
    rep["metrics_end"] = metrics(port)
    rep["gpu_mem_mib_end"] = gpu_mem_mib()


def perf(a):
    mc = model(a.model)
    out = f"{RUNS}/sgl_check/{a.model}/perf"
    os.makedirs(out, exist_ok=True)
    port = default_port()
    rg = lambda dom, n, file=None: dict(dom=dom, thinking=False, outlen=512, maxlen=4096, nprompt=lambda c: n, file=file)
    levels = [int(x) for x in a.levels.split(",")]
    rep = dict(model=a.model, mem=a.mem, runs={})
    # the plain server once as deployed (SGLang's overlap scheduler on) and once without the overlap scheduler,
    # which a speculative server cannot use (spec v1): the switch's plain steps can at best match the latter
    for spec, extra, modes in [(1, (), ["on", "off"]), (0, (), ["plain"]), (0, ("--disable-overlap-schedule",), ["plain_nooverlap"])]:
        srv, ts = start_sgl_server(mc, port, a.mem, spec, f"{out}/server_{modes[0]}.log", extra=extra)
        try:
            bench(mc, port, rg("chat", 8), 4, out, "warmup.json", seed=0)
            for mode in modes:
                if spec:
                    rpc(port, "set_speculation", enabled="1" if mode == "on" else "0")
                for c in levels:
                    n = {1: 16, 16: 64}.get(c, 3 * c)
                    f = f"{DATA}/bench/code_eval.jsonl" if c <= 16 else f"{DATA}/bench/code_pool.jsonl"
                    m0 = metrics(port)
                    r = bench(mc, port, rg("code", n, f), c, out, f"{mode}_C{c}.json", seed=0)
                    m1 = metrics(port)
                    d = {k: m1[k] - m0.get(k, 0) for k in m1}
                    dr = d.get("vllm:spec_decode_num_drafts_total", 0)
                    rep["runs"][f"{mode}_C{c}"] = dict(
                        out_tok_s=r["output_throughput"], median_tpot_ms=r["median_tpot_ms"], completed=r["completed"],
                        gpu_mem_mib=gpu_mem_mib(), preemptions=d.get("vllm:num_preemptions_total"),
                        acc_len=(1 + d["vllm:spec_decode_num_accepted_tokens_total"] / dr) if dr else None,
                        plain_steps=d.get("specsched:plain_steps_total"), decode_steps=d.get("specsched:decode_steps_total"))
                    print(f"[sgl-perf] {mode} C={c}: {r['output_throughput']:.0f} tok/s, TPOT p50 "
                          f"{r['median_tpot_ms']:.2f} ms, acc {rep['runs'][f'{mode}_C{c}']['acc_len']}", flush=True)
                    json.dump(rep, open(f"{out}/report.json", "w"), indent=1)
        finally:
            kill(srv)
            time.sleep(10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["func", "perf"])
    ap.add_argument("--model", default="m1")
    ap.add_argument("--mem", type=float, default=0.67)
    ap.add_argument("--levels", default="1,16,64,256")
    a = ap.parse_args()
    (func if a.what == "func" else perf)(a)
