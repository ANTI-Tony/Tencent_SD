"""Runs the frozen harness with controller_v7.py in its v7.0 configuration (proportional frontier with c = 1, no
off probe, one-step look-ahead): --pair-frontier 0 --off-probe 0 --lookahead 1. Otherwise identical to
hotswap_run_v7.py (optional --model-json P for the mixed-traffic scenarios)."""
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
        cmd += ["--pair-frontier", "0", "--off-probe", "0", "--lookahead", "1"]
        if MODEL_JSON:
            if "--model-json" in cmd:
                cmd[cmd.index("--model-json") + 1] = MODEL_JSON
            else:
                cmd += ["--model-json", MODEL_JSON]
    return _Popen(cmd, *args, **kw)


subprocess.Popen = _popen

if __name__ == "__main__":
    hotswap_run.main()
