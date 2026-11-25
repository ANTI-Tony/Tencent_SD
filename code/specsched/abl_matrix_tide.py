"""Manifest for the TIDE-style baseline on the frozen checkpoint: 3 scenarios x 3 repeats, same prompt files,
seeds, engine configuration and harness as the 72-run matrix (abl_matrix.py, imported unchanged); runs land next to
the matrix in runs/abl/<scenario>/tide/r<seed>/ and are executed by abl_runner_any.py with hotswap_run_tide.py.
Usage: python abl_matrix_tide.py manifest [path] [--smoke]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402  (frozen matrix definition)

# the harness launches controller.py in controller mode; hotswap_run_tide.py replaces it by controller_tide.py
abl_matrix.STRATEGIES["tide"] = dict(mode="controller", live=1, policy="switch_only", spec_switch=1)


def manifest(path, smoke=False):
    rows = []
    root = f"{abl_matrix.RUNS}/abl_smoke" if smoke else f"{abl_matrix.RUNS}/abl"
    for seed in ([1] if smoke else abl_matrix.SEEDS):
        for scen in (["m1_zh"] if smoke else abl_matrix.SCENARIOS):
            outdir = f"{root}/{scen}/tide/r{seed}"
            est = 7 if smoke else abl_matrix.SCENARIOS[scen]["est"]["spec_off"] + 2
            rows.append(dict(run_id=f"{scen}.tide.r{seed}", scenario=scen, strategy="tide", seed=seed, est_min=est,
                             outdir=outdir, script="hotswap_run_tide.py",
                             args=abl_matrix.run_args(scen, "tide", seed, outdir, smoke)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


def envcheck(path):
    """Reference runs of frozen strategies in the rebuilt environment, compared with the frozen matrix: the two
    noise-free static policies on Qwen3-8B code and static speculation on the MoE target, repeat 1."""
    rows = []
    for scen, strat in (("m1_code", "static"), ("m1_code", "nospec"), ("m2_code", "static")):
        outdir = f"{abl_matrix.RUNS}/abl_envcheck/{scen}/{strat}/r1"
        rows.append(dict(run_id=f"{scen}.{strat}.r1", scenario=scen, strategy=strat, seed=1, est_min=13, outdir=outdir,
                         script="hotswap_run.py", args=abl_matrix.run_args(scen, strat, 1, outdir)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} env-check runs -> {path}")


def fullrerun(path):
    """FULL (v6) on Qwen3-8B code, 3 repeats, in the rebuilt environment: a same-environment partner for the
    TIDE-style runs where the two are close (the frozen-matrix FULL runs used the original environment)."""
    rows = []
    for seed in abl_matrix.SEEDS:
        outdir = f"{abl_matrix.RUNS}/abl_envcheck/m1_code/v6/r{seed}"
        rows.append(dict(run_id=f"m1_code.v6.r{seed}", scenario="m1_code", strategy="v6", seed=seed, est_min=13, outdir=outdir,
                         script="hotswap_run.py", args=abl_matrix.run_args("m1_code", "v6", seed, outdir)))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} FULL re-runs -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "fullrerun":
        fullrerun(sys.argv[2] if len(sys.argv) > 2 else f"{abl_matrix.RUNS}/abl_envcheck/manifest_full.jsonl")
        sys.exit(0)
    if sys.argv[1] == "envcheck":
        envcheck(sys.argv[2] if len(sys.argv) > 2 else f"{abl_matrix.RUNS}/abl_envcheck/manifest.jsonl")
        sys.exit(0)
    if sys.argv[1] == "manifest":
        default = f"{abl_matrix.RUNS}/abl_smoke/manifest_tide.jsonl" if "--smoke" in sys.argv else f"{abl_matrix.RUNS}/abl/manifest_tide.jsonl"
        args = [x for x in sys.argv[2:] if not x.startswith("--")]
        manifest(args[0] if args else default, smoke="--smoke" in sys.argv)
