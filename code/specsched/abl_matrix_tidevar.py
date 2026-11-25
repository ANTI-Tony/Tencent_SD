"""Manifest for the TIDE-style sensitivity setting on the frozen checkpoint: rounds of 800 requests (4x) and a drift
margin of 0.3 (3x), every other parameter as in the TIDE-style baseline. 3 scenarios x 3 repeats with the prompt
files, seeds, engine configuration and harness of the 72-run matrix (abl_matrix.py, imported unchanged). Runs land in
runs/abl_tidevar/<scenario>/<VAR>/r<seed>/ (kept out of the main matrix) and are executed by abl_runner_any.py with
hotswap_run_tidevar.py.
Usage: python abl_matrix_tidevar.py manifest [path]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402  (frozen matrix definition)
import abl_matrix_tide  # noqa: E402,F401  (registers the "tide" strategy)

VAR = "tide_r800e03"
EXTRA = "--round 800 --epsilon 0.3"
abl_matrix.STRATEGIES[VAR] = dict(abl_matrix.STRATEGIES["tide"])


def manifest(path):
    rows = []
    root = f"{abl_matrix.RUNS}/abl_tidevar"
    for seed in abl_matrix.SEEDS:                       # repeat 1 of every scenario first, then 2, then 3
        for scen in abl_matrix.SCENARIOS:
            outdir = f"{root}/{scen}/{VAR}/r{seed}"
            est = abl_matrix.SCENARIOS[scen]["est"]["spec_off"] + 2
            rows.append(dict(run_id=f"{scen}.{VAR}.r{seed}", scenario=scen, strategy=VAR, seed=seed, est_min=est,
                             outdir=outdir, script="hotswap_run_tidevar.py",
                             args=abl_matrix.run_args(scen, VAR, seed, outdir) + ["--tide-extra", EXTRA]))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        manifest(sys.argv[2] if len(sys.argv) > 2 else f"{abl_matrix.RUNS}/abl_tidevar/manifest.jsonl")
