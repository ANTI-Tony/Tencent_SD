"""Runs the frozen harness hotswap_run.py unchanged, except that the controller process it starts is
controller_tide.py (TIDE-style training triggers) instead of controller.py.

The harness is imported, not copied, so phases, prompt files, engine configuration, trainer, sampler and the
done.json record are exactly those of the frozen matrix. The only intervention is in subprocess.Popen: a command
that would start controller.py starts controller_tide.py with the same arguments, plus --factory-head (the server's
initial drafter, restored if TIDE's first round does not improve acceptance) and --policy tide.
Usage: the arguments of hotswap_run.py with --mode controller --policy switch_only --spec-switch 1 --live 1.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hotswap_run  # noqa: E402  (frozen harness)
from cfg import model  # noqa: E402

_Popen = subprocess.Popen


def _model_name(argv):
    return argv[argv.index("--model") + 1] if "--model" in argv else "m1"


FACTORY = model(_model_name(sys.argv))["head"]


def _popen(cmd, *args, **kw):
    if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and str(cmd[1]) == os.path.join(HERE, "controller.py"):
        cmd = list(cmd)
        if "--policy" in cmd:
            cmd[cmd.index("--policy") + 1] = "tide"
        cmd = [cmd[0], os.path.join(HERE, "controller_tide.py"), *cmd[2:], "--factory-head", FACTORY]
    return _Popen(cmd, *args, **kw)


subprocess.Popen = _popen

if __name__ == "__main__":
    hotswap_run.main()
