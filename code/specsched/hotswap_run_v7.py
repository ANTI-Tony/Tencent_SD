"""Runs the frozen harness hotswap_run.py unchanged, but starts controller_v7.py (the self-calibrating controller)
instead of controller.py. Optional --model-json P replaces the calibration file the harness passes (the mixed-traffic
scenarios); v7 reads only its per-target fields. Everything else (phases, prompt files, engine configuration,
trainer, sampler, done.json) is the frozen matrix's. Usage: the arguments of hotswap_run.py (+ --model-json P)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _take(flag, has_value):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    val = sys.argv[i + 1] if has_value else True
    del sys.argv[i:i + 2 if has_value else i + 1]
    return val


MODEL_JSON = _take("--model-json", True)
if MODEL_JSON:
    MODEL_JSON = os.path.abspath(MODEL_JSON)
    assert os.path.exists(MODEL_JSON), MODEL_JSON
import hotswap_run  # noqa: E402  (frozen harness)

_Popen = subprocess.Popen
FROZEN = os.path.join(HERE, "controller.py")


def _popen(cmd, *args, **kw):
    if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and str(cmd[1]) == FROZEN:
        cmd = list(cmd)
        cmd[1] = os.path.join(HERE, "controller_v7.py")
        if MODEL_JSON:
            if "--model-json" in cmd:
                cmd[cmd.index("--model-json") + 1] = MODEL_JSON
            else:
                cmd += ["--model-json", MODEL_JSON]
    return _Popen(cmd, *args, **kw)


subprocess.Popen = _popen

if __name__ == "__main__":
    hotswap_run.main()
