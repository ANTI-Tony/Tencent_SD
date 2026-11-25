"""Freeze record of the SGLang checkpoint: runs/sgl/MANIFEST.json, written once before the first matrix run.

Hashes (first 16 hex digits of SHA-256, as in runs/abl/MANIFEST.json) of the port's files, of the SGLang calibration,
of every file the SGLang runs reuse from the vLLM checkpoint (each must equal its hash in runs/abl/MANIFEST.json,
checked here), of the drafters, prompt pools and target weights, and the SGLang environment's versions and freeze.
Usage: python sgl_freeze.py [--check]   (--check compares the current files with an existing MANIFEST.json)
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cfg import RUNS  # noqa: E402

WS = os.environ.get("WS", "/fs/scratch/PCS0296/specsched")
PORT = ["sgl/specsched_sgl.py", "sgl/specsched_sgl_launch.py", "sgl_common.py", "hotswap_run_sgl.py", "sgl_calib.py",
        "abl_matrix_sgl.py", "sgl_freeze.py", "sched/model_sgl_m1.json", "sched/model_sgl_m2.json"]
REUSED = ["hotswap_run.py", "hotswap_run_tide.py", "controller.py", "controller_tide.py", "train_head.py", "head.py",
          "livetap.py", "c1_run.py", "frontier_run.py", "cfg.py", "abl_matrix.py", "abl_matrix_tide.py", "abl_runner.py",
          "abl_runner_any.py", "sched/sim.py", "sched/fit_model.py"]


def h(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            d.update(b)
    return d.hexdigest()[:16]


def build():
    vman = json.load(open(f"{RUNS}/abl/MANIFEST.json"))
    m = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), port={}, reused={}, reused_vllm_manifest={})
    for f in PORT:
        m["port"][f"code/{f}"] = h(os.path.join(HERE, f))
    for f in REUSED:
        m["reused"][f"code/{f}"] = h(os.path.join(HERE, f))
        m["reused_vllm_manifest"][f"code/{f}"] = vman["code"].get(f"code/{f}")
    m["heads"], m["pools"] = vman["heads"], vman["pools"]
    m["target_m1"], m["target_m2"] = vman["target_m1"], vman["target_m2"]
    py = f"{WS}/env/sgl0510/bin/python"
    m["sglang_env"] = subprocess.run(
        [py, "-c", "import importlib.metadata as md; print(';'.join(f'{p} {md.version(p)}' for p in "
                   "['sglang', 'sglang-kernel', 'torch', 'flashinfer-python', 'transformers', 'triton']))"],
        capture_output=True, text=True).stdout.strip()
    m["sglang_env_freeze"] = h(f"{WS}/env/sgl0510_freeze.txt")
    return m


def main():
    out = f"{RUNS}/sgl/MANIFEST.json"
    m = build()
    changed = {k: (m["reused"][k], v) for k, v in m["reused_vllm_manifest"].items() if v is not None and v != m["reused"][k]}
    if "--check" in sys.argv:
        old = json.load(open(out))
        diff = {k: (old[g].get(k), m[g].get(k)) for g in ("port", "reused") for k in m[g] if old[g].get(k) != m[g].get(k)}
        print("unchanged since the freeze" if not diff else f"CHANGED: {diff}")
        sys.exit(1 if diff else 0)
    assert not changed, f"reused files differ from the vLLM checkpoint: {changed}"
    assert not os.path.exists(out), f"{out} exists: the SGLang checkpoint is already frozen"
    json.dump(m, open(out, "w"), indent=1)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
