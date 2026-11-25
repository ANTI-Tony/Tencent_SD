"""Runs the frozen harness hotswap_run.py unchanged, with three optional interventions on the controller process it
starts (and nothing else: phases, prompt files, engine configuration, trainer, sampler and done.json are the frozen
matrix's):
  --ablate X        start controller_ablate.py --ablate X instead of controller.py (X = novalue | nostop | nocurve)
  --tide            start controller_tide.py (as hotswap_run_tide.py does)
  --model-json P    give the controller (any of the three) this calibration file instead of sched/model_<model>.json;
                    used by the mixed-traffic runs, whose domains are not in the frozen file
These options are stripped before hotswap_run parses its own arguments. Usage: the arguments of hotswap_run.py plus
the options above; the noswitch cell of the ablation needs no option (--spec-switch 0 --spec-switchable 0).
"""
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


ABLATE = _take("--ablate", True)
TIDE = _take("--tide", False)
MODEL_JSON = _take("--model-json", True)
assert not (ABLATE and TIDE), "--ablate and --tide exclude each other"
if MODEL_JSON:
    MODEL_JSON = os.path.abspath(MODEL_JSON)
    assert os.path.exists(MODEL_JSON), MODEL_JSON
if TIDE:
    import hotswap_run_tide  # noqa: E402,F401  (patches subprocess.Popen: controller.py -> controller_tide.py)
import hotswap_run  # noqa: E402  (frozen harness)

_Popen = subprocess.Popen
CONTROLLERS = {os.path.join(HERE, n) for n in ("controller.py", "controller_tide.py", "controller_ablate.py")}


def _popen(cmd, *args, **kw):
    if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and str(cmd[1]) in CONTROLLERS:
        cmd = list(cmd)
        if ABLATE and str(cmd[1]) == os.path.join(HERE, "controller.py"):
            cmd[1] = os.path.join(HERE, "controller_ablate.py")
            cmd += ["--ablate", ABLATE]
        if MODEL_JSON:
            if "--model-json" in cmd:
                cmd[cmd.index("--model-json") + 1] = MODEL_JSON
            else:
                cmd += ["--model-json", MODEL_JSON]
    return _Popen(cmd, *args, **kw)


subprocess.Popen = _popen

if __name__ == "__main__":
    hotswap_run.main()
