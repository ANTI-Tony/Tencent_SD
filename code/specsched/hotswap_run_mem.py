"""GPU-memory arrangement wrapper (written 2026-11-24 after the E28 open-loop smokes). hotswap_run_arm.py and the
frozen harness are imported and run unchanged; this wrapper only adds
  --trainer-reserve-gb G   starts the trainer through train_head_reserve.py: the trainer process takes G GiB of GPU
                           memory into its own allocator cache before it loads anything
  --server-alloc-conf S    starts the serving process (and only it) with PYTORCH_CUDA_ALLOC_CONF=S
  --memprobe 1             diagnosis only: the serving process and the trainer log their own allocator counters to
                           <outdir>/memprobe_<role>_<pid>.jsonl (memprobe/sitecustomize.py)
Every other argument goes on to hotswap_run_arm.py (--arm, --arm-n, --model-json, --open-rate) and the frozen harness.
Without these flags the run is identical to one started through hotswap_run_arm.py."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _take(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    val = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return val


RESERVE = _take("--trainer-reserve-gb")
ALLOC = _take("--server-alloc-conf")
PROBE = _take("--memprobe") == "1"
if RESERVE:
    assert float(RESERVE) > 0
OUTDIR = sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else None
assert OUTDIR or not PROBE, "--memprobe needs --outdir"

import hotswap_run_arm  # noqa: E402,F401  (takes its own flags, patches the controller start and the phase-B client)
import hotswap_run  # noqa: E402  (frozen harness)
from c1_run import VLLM  # noqa: E402

_prev = subprocess.Popen                      # hotswap_run_arm's patched Popen
TRAINER = os.path.join(HERE, "train_head.py")
PROBE_DIR = os.path.join(HERE, "memprobe")


def _probe(env, role):
    if PROBE:
        env["PYTHONPATH"] = PROBE_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["MEMPROBE_DIR"], env["MEMPROBE_ROLE"] = OUTDIR, role
    return env


def _popen(cmd, *args, **kw):
    if isinstance(cmd, (list, tuple)) and len(cmd) > 1:
        if str(cmd[1]) == TRAINER and (RESERVE or PROBE):
            env = _probe(dict(kw.get("env") or os.environ), "trainer")
            if RESERVE:
                cmd = list(cmd)
                cmd[1] = os.path.join(HERE, "train_head_reserve.py")
                env["TRAINER_RESERVE_GB"] = str(RESERVE)
                print('{"event": "trainer_reserve", "gb": %s}' % RESERVE, flush=True)
            kw["env"] = env
        elif str(cmd[0]) == VLLM and str(cmd[1]) == "serve" and (ALLOC or PROBE):
            env = _probe(dict(kw.get("env") or os.environ), "server")
            if ALLOC:
                env["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC
                print('{"event": "server_alloc_conf", "conf": "%s"}' % ALLOC, flush=True)
            kw["env"] = env
    return _prev(cmd, *args, **kw)


subprocess.Popen = _popen

if __name__ == "__main__":
    hotswap_run.main()
