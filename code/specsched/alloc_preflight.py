"""Allocator-option preflight for the memory diagnosis (2026-11-24). Runs for a few seconds on the job's GPU before
any server starts. It answers three questions about this torch build: is per_process_memory_fraction accepted as a
PYTORCH_CUDA_ALLOC_CONF key, does the cap then reject an allocation above it, and does the cap survive the scoped
max_split_size_mb change that the serving engine applies while it loads weights (gpu_worker.py, load_model)."""
import json
import os
import subprocess
import sys

CHILD = r'''
import json, torch
out = {"torch": torch.__version__}
def step(name, fn):
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = "ERR " + type(e).__name__ + ": " + str(e)[:220]
def grab(gib):
    x = torch.empty(int(gib * 2**30), dtype=torch.uint8, device="cuda"); del x
    torch.cuda.empty_cache()
    return "ok"
def scoped():                                   # what the engine does around weight loading, with no max_split in the env
    torch._C._accelerator_setAllocatorSettings("max_split_size_mb:20")
    torch._C._accelerator_setAllocatorSettings("max_split_size_mb:%d" % ((2**64 - 1) // (1024 * 1024)))
    return "ok"
step("total_gib", lambda: round(torch.cuda.mem_get_info()[1] / 2**30, 2))
step("fraction_at_start", torch.cuda.get_per_process_memory_fraction)
step("alloc_1gib", lambda: grab(1))
step("alloc_6gib", lambda: grab(6))
step("scoped_setting", scoped)
step("fraction_after_scoped_setting", torch.cuda.get_per_process_memory_fraction)
step("alloc_6gib_after_scoped_setting", lambda: grab(6))
step("ooms", lambda: torch.cuda.memory_stats(0).get("num_ooms"))
print("PREFLIGHT " + json.dumps(out))
'''

for conf in ["", "per_process_memory_fraction:0.04"]:          # 0.04 of a 93 GiB card = 3.7 GiB: 1 GiB fits, 6 GiB does not
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = conf
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300)
    line = [x for x in r.stdout.splitlines() if x.startswith("PREFLIGHT ")]
    print(json.dumps(dict(conf=conf, returncode=r.returncode, result=json.loads(line[0][10:]) if line else None,
                          stderr_tail=r.stderr[-600:])), flush=True)
