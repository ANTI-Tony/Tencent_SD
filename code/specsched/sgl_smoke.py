"""Smoke test of stock SGLang (no port patches yet): EAGLE-3 vs plain serving of the code domain on one H100.

For each of {speculation on, off}: start the server, warm up, run the closed-loop benchmark at C = 1, 16, 256 and save
the raw /metrics text after each level (to learn SGLang's metric names and its acceptance counters) and the GPU
memory in use. Output: runs/sgl_smoke/<model>/{spec,plain}/...
Usage: python sgl_smoke.py --model m1 [--mem 0.67]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c1_run import bench, default_port, get, kill  # noqa: E402
from cfg import DATA, RUNS, model  # noqa: E402
from sgl_common import start_sgl_server  # noqa: E402


def gpu_mem():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--mem", type=float, default=0.67)
    ap.add_argument("--levels", default="1,16,256")
    a = ap.parse_args()
    mc = model(a.model)
    port = default_port()
    rg = lambda dom, n, file=None: dict(dom=dom, thinking=False, outlen=512, maxlen=4096, nprompt=lambda c: n, file=file)
    for spec in (1, 0):
        out = f"{RUNS}/sgl_smoke/{a.model}/{'spec' if spec else 'plain'}"
        os.makedirs(out, exist_ok=True)
        srv, ts = start_sgl_server(mc, port, a.mem, spec, f"{out}/server.log", patched=False)
        res = dict(server_seconds=ts, port=port, mem=a.mem, spec=spec, levels={})
        try:
            bench(mc, port, rg("chat", 8), 4, out, "warmup.json", seed=0)
            res["gpu_mem_after_warmup"] = gpu_mem()
            for c in [int(x) for x in a.levels.split(",")]:
                n = {1: 16, 16: 64}.get(c, 3 * c)
                f = f"{DATA}/bench/code_eval.jsonl" if c <= 16 else f"{DATA}/bench/code_pool.jsonl"
                t0 = time.time()
                r = bench(mc, port, rg("code", n, f), c, out, f"C{c}.json", seed=0)
                open(f"{out}/metrics_C{c}.txt", "w").write(get(f"http://127.0.0.1:{port}/metrics", 10))
                res["levels"][c] = dict(out_tok_s=r["output_throughput"], median_tpot_ms=r["median_tpot_ms"],
                                        completed=r["completed"], wall=time.time() - t0)
                print(f"[sgl-smoke] spec={spec} C={c}: {r['output_throughput']:.0f} tok/s, "
                      f"TPOT p50 {r['median_tpot_ms']:.2f} ms, completed {r['completed']}", flush=True)
            res["gpu_mem_end"] = gpu_mem()
        finally:
            kill(srv)
            json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
            time.sleep(10)


if __name__ == "__main__":
    main()
