"""TIDE-style baseline with other trigger parameters (sensitivity check of the TIDE-style policy).

Runs hotswap_run_tide.py unchanged (which itself runs the frozen harness hotswap_run.py and swaps controller.py
for controller_tide.py) and appends extra controller_tide.py options, e.g. "--round 800 --epsilon 0.3", to the
controller command. The options are also written to <outdir>/tide_variant.json so that every run records them.
Usage: the arguments of hotswap_run_tide.py plus --tide-extra "<controller_tide.py options>".
"""
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# take our own option out of argv before the frozen harness parses it
EXTRA = []
if "--tide-extra" in sys.argv:
    _i = sys.argv.index("--tide-extra")
    EXTRA = shlex.split(sys.argv[_i + 1])
    del sys.argv[_i:_i + 2]

import hotswap_run_tide  # noqa: E402  (patches subprocess.Popen: controller.py -> controller_tide.py)

_tide_popen = subprocess.Popen
CONTROLLER = os.path.join(HERE, "controller.py")


def _popen(cmd, *args, **kw):
    if EXTRA and isinstance(cmd, (list, tuple)) and len(cmd) > 1 and str(cmd[1]) == CONTROLLER:
        cmd = [*cmd, *EXTRA]
    return _tide_popen(cmd, *args, **kw)


subprocess.Popen = _popen


def _record():
    if "--outdir" in sys.argv:
        out = sys.argv[sys.argv.index("--outdir") + 1]
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "tide_variant.json"), "w") as f:
            json.dump({"controller_tide_extra_args": EXTRA}, f)


if __name__ == "__main__":
    _record()
    hotswap_run_tide.hotswap_run.main()
