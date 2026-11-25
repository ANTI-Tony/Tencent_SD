"""In-process GPU allocator probe for the memory diagnosis (2026-11-24; diagnosis only, never used in E28a).

Active only when MEMPROBE_DIR is set and this directory is on PYTHONPATH; hotswap_run_mem.py --memprobe 1 does both
for the serving process and for the trainer, and for nothing else. Once a process has initialised CUDA, a daemon
thread appends one JSON line every MEMPROBE_EVERY seconds (default 0.5) to
$MEMPROBE_DIR/memprobe_<role>_<pid>.jsonl: bytes allocated and reserved by PyTorch's caching allocator, their peaks
since the process last reset them, and the allocator's retry and out-of-memory counters. It only reads counters: it
resets no statistic and frees nothing. The serving engine forks its GPU process, so the thread is restarted in forked
children."""
import json
import os
import sys
import threading
import time

KEYS = dict(alloc="allocated_bytes.all.current", resv="reserved_bytes.all.current",
            alloc_peak="allocated_bytes.all.peak", resv_peak="reserved_bytes.all.peak",
            inactive="inactive_split_bytes.all.current", retries="num_alloc_retries", ooms="num_ooms",
            dev_alloc="num_device_alloc", dev_free="num_device_free")


def _loop():
    out = os.environ["MEMPROBE_DIR"]
    every = float(os.environ.get("MEMPROBE_EVERY", "0.5"))
    role = os.environ.get("MEMPROBE_ROLE", "proc")
    pid, f = os.getpid(), None
    while True:
        time.sleep(every)
        try:
            torch = sys.modules.get("torch")
            if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_initialized():
                continue
            s = torch.cuda.memory_stats(0)
            if f is None:
                f = open(os.path.join(out, f"memprobe_{role}_{pid}.jsonl"), "a", buffering=1)
                f.write(json.dumps(dict(t=time.time(), event="start", pid=pid, ppid=os.getppid(), role=role,
                                        argv=[str(x)[-80:] for x in sys.argv[:4]], torch=torch.__version__)) + "\n")
            f.write(json.dumps(dict(t=round(time.time(), 3), **{k: s.get(v, 0) for k, v in KEYS.items()})) + "\n")
        except Exception:
            pass


def _start():
    threading.Thread(target=_loop, name="memprobe", daemon=True).start()


if os.environ.get("MEMPROBE_DIR"):
    _start()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_start)
