"""Manifest of the GPU-memory diagnosis (2026-11-24). Not part of E28: it produces no comparison data.

runs/abl_arm_diag/<scenario>/<name>/r1, seed 1, arm duty05 (the trainer works through all of B, so it is exposed for
the whole phase). Every run goes through hotswap_run_mem.py with the in-process allocator probes on.
Usage: python abl_matrix_mem.py manifest [out.jsonl]"""
import json
import os
import sys

import abl_matrix_arm as M
from cfg import RUNS

ROOT = f"{RUNS}/abl_arm_diag"
DIAG = [  # scenario, name, B prompts, open-loop requests/s (None = closed loop), trainer reservation GiB, est. minutes
    ("m1_code", "ref_closed", 3000, None, None, 10),          # the frozen arrangement, closed loop: reference trace
    ("m1_code", "open69", 4000, 69.4, None, 10),              # the cancelled E28b condition, unchanged
    ("m1_code", "open200_reserve", 4000, 200.0, 12, 10),      # stress (backlog of a full-length run) with the reservation
    ("m1_code", "open69_reserve", 4000, 69.4, 12, 10),        # the E28b condition with the reservation
    ("m2_code", "open63", 4000, 63.0, None, 14),              # MoE at its E28b rate, unchanged
    ("m2_code", "open200_reserve", 4000, 200.0, 7, 14),       # MoE stress with the reservation
]


def manifest(path=None):
    rows = []
    for scen, name, nB, rate, reserve, est in DIAG:
        r = M.row(ROOT, scen, "duty05", 1, open_rate=rate, nB=nB)
        outdir = f"{ROOT}/{scen}/{name}/r1"
        args = list(r["args"])
        args[args.index("--outdir") + 1] = outdir
        extra = ["--memprobe", "1"] + (["--trainer-reserve-gb", str(reserve)] if reserve else [])
        rows.append(dict(r, run_id=f"{scen}.{name}.r1", strategy=name, est_min=est, outdir=outdir,
                         script="hotswap_run_mem.py", args=extra + args))
    path = path or f"{ROOT}/manifest.jsonl"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        manifest(sys.argv[2] if len(sys.argv) > 2 else None)
