"""Runs the frozen harness hotswap_run.py (or, with --tide, hotswap_run_tide.py) unchanged on an SGLang server.

The harness is imported, not copied: phases, prompt files, benchmark client, trainer, tap reader, sampler, controller
code and the done.json record are those of the frozen matrix. Two things change:
  - start_server starts SGLang through the patched launcher (sgl_common.start_sgl_server: tap, draft reload,
    speculation switch, /metrics with vLLM's names, /collective_rpc) instead of `vllm serve`; vLLM's memory
    utilisation u becomes SGLang's static fraction u - NONSTATIC_GB/total (sgl_common.mem_fraction);
  - the controller reads SGLang's calibration, sched/model_sgl_<model>.json, through --model-json.
Usage: python hotswap_run_sgl.py [--tide] <hotswap_run.py arguments>
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TIDE = "--tide" in sys.argv
if TIDE:
    sys.argv.remove("--tide")
    import hotswap_run_tide  # noqa: E402,F401  (patches subprocess.Popen: controller.py -> controller_tide.py)
import hotswap_run  # noqa: E402  (frozen harness)
from sgl_common import mem_fraction, start_sgl_server  # noqa: E402

MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "m1"
MODEL_JSON = os.path.join(HERE, "sched", f"model_sgl_{MODEL}.json")


def start_server(mc, port, util, maxlen, spec, nspec, logpath, tp=1, spec_schedule=None):
    assert tp == 1 and not spec_schedule, "the SGLang port covers one GPU and no static per-batch-size table"
    total = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()[0]) / 1024
    mem = mem_fraction(MODEL, util, total, spec)
    print(f"[sgl] util {util:.3f} -> mem_fraction_static {mem:.3f}", flush=True)
    return start_sgl_server(mc, port, mem, spec, logpath, nspec=nspec, maxlen=maxlen)


hotswap_run.start_server = start_server
_Popen = subprocess.Popen


def _popen(cmd, *args, **kw):
    if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and os.path.basename(str(cmd[1])) in ("controller.py",
                                                                                             "controller_tide.py"):
        cmd = list(cmd)
        if "--model-json" in cmd:
            cmd[cmd.index("--model-json") + 1] = MODEL_JSON
        else:
            cmd += ["--model-json", MODEL_JSON]
    return _Popen(cmd, *args, **kw)


subprocess.Popen = _popen

if __name__ == "__main__":
    assert os.path.exists(MODEL_JSON), f"missing SGLang calibration {MODEL_JSON} (sgl_calib.py fit)"
    hotswap_run.main()
