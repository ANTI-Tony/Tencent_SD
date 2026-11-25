"""SGLang matrix: 3 scenarios x 5 strategies x 3 repeats on SGLang (hotswap_run_sgl.py), with the scenarios, prompt
files, seeds and harness arguments of the frozen vLLM matrix (abl_matrix.run_args, imported unchanged).

Strategies: static, nospec (Plain), switch, tide (TIDE-style triggers, hotswap_run_tide.py's controller swap), v6
(Full). Output: runs/sgl/abl/<scenario>/<strategy>/r<seed>/, executed by abl_runner_any.py.
Usage: python abl_matrix_sgl.py manifest [path] [--smoke]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402  (frozen)
import abl_matrix_tide  # noqa: E402,F401  (registers abl_matrix.STRATEGIES["tide"])
from cfg import RUNS  # noqa: E402

STRATS = ["static", "nospec", "switch", "tide", "v6"]
EST = {"m1_code": 17, "m1_zh": 22, "m2_code": 22}        # minutes per run on SGLang (server start ~5 min)


def manifest(path, smoke=False):
    rows = []
    root = f"{RUNS}/sgl/abl_smoke" if smoke else f"{RUNS}/sgl/abl"
    for seed in ([1] if smoke else abl_matrix.SEEDS):
        for scen in (["m1_code"] if smoke else abl_matrix.SCENARIOS):
            for strat in STRATS:
                outdir = f"{root}/{scen}/{strat}/r{seed}"
                args = abl_matrix.run_args(scen, strat, seed, outdir, smoke)
                rows.append(dict(run_id=f"sgl.{scen}.{strat}.r{seed}", scenario=scen, strategy=strat, seed=seed,
                                 est_min=8 if smoke else EST[scen], outdir=outdir, script="hotswap_run_sgl.py",
                                 args=(["--tide"] if strat == "tide" else []) + args))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        smoke = "--smoke" in sys.argv
        rest = [x for x in sys.argv[2:] if x != "--smoke"]
        manifest(rest[0] if rest else f"{RUNS}/sgl/{'abl_smoke' if smoke else 'abl'}/manifest.jsonl", smoke)
