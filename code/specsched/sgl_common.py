"""SGLang server launch for the SGLang port of the study.

Only the server process runs in the SGLang environment (env/sgl0510); the harness, benchmark client, controller and
trainer keep running in env/ss312, exactly as in the vLLM study. The server configuration mirrors c1_run.start_server:
256 running requests, context 4,096, no prefix cache, seed 0, served model name "target", EAGLE-3 chain drafting
with k draft tokens (topk 1, k + 1 verified tokens), CUDA graphs up to batch 256. SGLang's default EAGLE-3 path is
used (spec v1, overlap scheduler off).
"""
import os
import subprocess
import time

from c1_run import get, port_is_free

WS = os.environ.get("WS", "/fs/scratch/PCS0296/specsched")
SGLPY = os.environ.get("SGLPY", f"{WS}/env/sgl0510/bin/python")
# JIT caches whose defaults live under the home directory. The home quota is full, and the tvm-ffi JIT build then
# retries its lock file forever (seen on 2026-10-28: the server hung in CUDA graph capture), so all go to scratch.
SGL_CACHE_ENV = {"TVM_FFI_CACHE_DIR": f"{WS}/cache/tvm-ffi", "SGLANG_CACHE_DIR": f"{WS}/cache/sglang",
                 "SGLANG_DG_CACHE_DIR": f"{WS}/cache/deep_gemm"}
HERE = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = os.path.join(HERE, "sgl")      # specsched_sgl.py: tap, draft reload, speculation switch, /metrics, RPC
# vLLM's --gpu-memory-utilization u caps the server's whole footprint at u x total; SGLang's --mem-fraction-static
# covers weights + KV cache only, and CUDA graphs, activations and the CUDA context come on top. A harness `util`
# therefore maps to mem_fraction_static = util - NONSTATIC_GB / total, so that the server's peak footprint, and the
# memory left to the trainer, are those of the vLLM runs. NONSTATIC_GB: peak nvidia-smi use minus the static pool,
# measured as peak nvidia-smi use minus the static pool after C = 256 (runs/sgl_check/<model>/perf, 2026-10-28:
# 8B 7.22 / 1.14 GiB, MoE 5.65 / 0.80 GiB with / without a drafter), plus 0.45-0.8 GiB margin. A speculative server
# needs far more than a plain one (verify, draft, draft-extend and plain-decode CUDA graphs, 4 tokens per request).
NONSTATIC_GB = {("m1", True): 8.0, ("m1", False): 1.75, ("m2", True): 6.5, ("m2", False): 1.25}


def mem_fraction(model_name, util, total_gb, spec):
    return util - NONSTATIC_GB[(model_name, bool(spec))] / total_gb


def sgl_cmd(mc, port, mem, spec, nspec=3, maxlen=4096, extra=(), patched=True):
    """patched: start through specsched_sgl_launch (the port's mechanisms and its /metrics); otherwise stock SGLang
    with its own Prometheus metrics."""
    cmd = [SGLPY, "-m", "specsched_sgl_launch" if patched else "sglang.launch_server", "--model-path", mc["target"],
           "--host", "127.0.0.1", "--port", str(port), "--mem-fraction-static", f"{mem:.3f}",
           "--context-length", str(maxlen), "--max-running-requests", "256", "--cuda-graph-max-bs", "256",
           "--disable-radix-cache", "--random-seed", "0", "--served-model-name", "target",
           # SGLang turns on its experimental piecewise CUDA graphs for prefill only without a drafter, and they
           # crash at start-up here (illegal memory access, 2026-10-28); off for all servers, so that plain and
           # speculative servers prefill on the same path
           "--disable-piecewise-cuda-graph"]
    if not patched:
        cmd += ["--enable-metrics"]
    if spec:
        cmd += ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", mc["head"],
                "--speculative-num-steps", str(nspec), "--speculative-eagle-topk", "1",
                "--speculative-num-draft-tokens", str(nspec + 1)]
    return cmd + list(extra)


def start_sgl_server(mc, port, mem, spec, logpath, nspec=3, maxlen=4096, extra=(), env=None, timeout=1200,
                     patched=True):
    assert port_is_free(port), f"port {port} already answers /health: another server is alive on this node"
    env = dict(os.environ if env is None else env, **SGL_CACHE_ENV)
    if patched:
        env["PYTHONPATH"] = os.pathsep.join([PATCH_DIR] + [p for p in [env.get("PYTHONPATH")] if p])
    for d in SGL_CACHE_ENV.values():
        os.makedirs(d, exist_ok=True)
    logf = open(logpath, "w")
    proc = subprocess.Popen(sgl_cmd(mc, port, mem, spec, nspec, maxlen, extra, patched), stdout=logf,
                            stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server died, see {logpath}")
        try:
            if get(f"http://127.0.0.1:{port}/health") is not None:
                return proc, time.time() - t0
        except Exception:
            time.sleep(3)
    raise RuntimeError("SGLang server start timeout")
