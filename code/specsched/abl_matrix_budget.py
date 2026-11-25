"""Calibrated Switch + Budget baseline (controller_budget.py, policy switch_budget) on the frozen checkpoint: the three
single-domain scenarios and the two mixtures, 3 repeats each (15 runs), same prompt files, seeds, engine configuration
and harness as the matrix. Runs land in runs/abl_budget/<scenario>/switch_budget/r<seed>/ and are executed by
abl_runner_any.py with hotswap_run_budget.py; paired by seed with the matrix (analyze_cells.py).
Usage: python abl_matrix_budget.py manifest [path] [--smoke]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402
import abl_matrix_mix  # noqa: E402  (mixture scenarios + model_mix.json)
from cfg import RUNS  # noqa: E402

STRAT = dict(mode="controller", live=1, policy="switch_only", spec_switch=1)   # the wrapper rewrites the policy
abl_matrix.STRATEGIES["switch_budget"] = STRAT
SCEN = ["m1_code", "m1_zh", "m2_code", "m1_mixcc", "m1_mixzc"]


def manifest(path, smoke=False):
    rows = []
    root = f"{RUNS}/abl_budget_smoke" if smoke else f"{RUNS}/abl_budget"
    for seed in ([1] if smoke else abl_matrix.SEEDS):
        for scen in (["m1_code"] if smoke else SCEN):
            outdir = f"{root}/{scen}/switch_budget/r{seed}"
            args = abl_matrix.run_args(scen, "switch_budget", seed, outdir, smoke)
            extra = ["--model-json", abl_matrix_mix.MODEL_JSON] if scen.startswith("m1_mix") else []
            est = 6 if smoke else abl_matrix.SCENARIOS[scen]["est"]["spec_off"]
            rows.append(dict(run_id=f"{scen}.switch_budget.r{seed}", scenario=scen, strategy="switch_budget", seed=seed,
                             est_min=est, outdir=outdir, script="hotswap_run_budget.py", args=extra + args))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        smoke = "--smoke" in sys.argv
        rest = [x for x in sys.argv[2:] if x != "--smoke"]
        manifest(rest[0] if rest else f"{RUNS}/{'abl_budget_smoke' if smoke else 'abl_budget'}/manifest.jsonl", smoke)
