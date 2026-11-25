"""Runs the frozen harness hotswap_run.py unchanged, but starts controller_budget.py with --policy switch_budget (the
calibrated Switch + Budget baseline) instead of controller.py. The harness is given --policy switch_only (one of its
accepted choices); the controller command is rewritten. Optional --model-json P replaces the calibration file (the
mixed-traffic scenarios); optional --ctl-args "..." appends arguments to the controller command (the payback-wait
experiment). Everything else is the frozen matrix's. Usage: the arguments of hotswap_run.py (+ --model-json P --ctl-args "...")."""
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
CTL_ARGS = _take("--ctl-args", True)          # extra controller arguments, e.g. "--budget-n 3200 --wait-mode payback"
if MODEL_JSON:
    MODEL_JSON = os.path.abspath(MODEL_JSON)
    assert os.path.exists(MODEL_JSON), MODEL_JSON
import hotswap_run  # noqa: E402  (frozen harness)

_Popen = subprocess.Popen
FROZEN = os.path.join(HERE, "controller.py")


def _popen(cmd, *args, **kw):
    if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and str(cmd[1]) == FROZEN:
        cmd = list(cmd)
        cmd[1] = os.path.join(HERE, "controller_budget.py")
        assert "--policy" in cmd
        cmd[cmd.index("--policy") + 1] = "switch_budget"
        if "--fixed-duty" in cmd:
            cmd[cmd.index("--fixed-duty") + 1] = "0.5"
        else:
            cmd += ["--fixed-duty", "0.5"]
        if CTL_ARGS:
            cmd += CTL_ARGS.split()
        if MODEL_JSON:
            if "--model-json" in cmd:
                cmd[cmd.index("--model-json") + 1] = MODEL_JSON
            else:
                cmd += ["--model-json", MODEL_JSON]
    return _Popen(cmd, *args, **kw)


subprocess.Popen = _popen

if __name__ == "__main__":
    hotswap_run.main()
