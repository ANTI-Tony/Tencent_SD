"""Self-calibrating controller (v7) on the frozen checkpoint: the three single-domain scenarios of the 72-run matrix
plus the two mixed-traffic scenarios, 3 repeats each (15 runs), same prompt files, seeds, engine configuration and
harness as the matrix (abl_matrix.py and abl_matrix_mix.py, imported unchanged). Runs land in runs/abl_v7/<scenario>/
v7/r<seed>/ and are executed by abl_runner_any.py with hotswap_run_v7.py; they are paired by seed with the matrix's
runs (analyze_cells.py). Usage: python abl_matrix_v7.py manifest [path] [--smoke] [--root abl_v71]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402  (frozen matrix definition)
import abl_matrix_mix  # noqa: E402  (adds the mixture scenarios to abl_matrix.SCENARIOS)
from cfg import RUNS  # noqa: E402

STRAT = dict(mode="controller", live=1, policy="costmodel", spec_switch=1)
abl_matrix.STRATEGIES["v7"] = STRAT
SCEN = ["m1_code", "m1_zh", "m2_code", "m1_mixcc", "m1_mixzc"]


def manifest(path, smoke=False, name="abl_v7"):
    rows = []
    root = f"{RUNS}/{name}_smoke" if smoke else f"{RUNS}/{name}"
    for seed in ([1] if smoke else abl_matrix.SEEDS):
        for scen in (["m1_code", "m1_zh"] if smoke else SCEN):
            outdir = f"{root}/{scen}/v7/r{seed}"
            args = abl_matrix.run_args(scen, "v7", seed, outdir, smoke)
            extra = ["--model-json", abl_matrix_mix.MODEL_JSON] if scen.startswith("m1_mix") else []
            est = 6 if smoke else abl_matrix.SCENARIOS[scen]["est"]["spec_off"]
            rows.append(dict(run_id=f"{scen}.v7.r{seed}", scenario=scen, strategy="v7", seed=seed, est_min=est,
                             outdir=outdir, script="hotswap_run_v7.py", args=extra + args))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        smoke = "--smoke" in sys.argv
        name = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "abl_v7"   # e.g. abl_v71 for v7.1
        rest = [x for i, x in enumerate(sys.argv[2:]) if x not in ("--smoke", "--root", name)]
        manifest(rest[0] if rest else f"{RUNS}/{name + '_smoke' if smoke else name}/manifest.jsonl", smoke, name)
