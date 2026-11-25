"""Ablation of the controller's tests on the frozen checkpoint: 4 cells x 3 scenarios x 3 repeats, same prompt files,
seeds, engine configuration and harness as the 72-run matrix (abl_matrix.py, imported unchanged).

  v6_novalue   Full without the payback rule (fixed duty 0.5 under saturation; stop tests kept)
  v6_nostop    Full without the three explicit stop tests (structural ceiling, at ceiling, converged)
  v6_noswitch  Full on an engine without a speculation switch (controller.py with --spec-switch 0 --spec-switchable 0:
               speculation stays on, and training is valued with speculation stuck on)
  v6_nocurve   Full without per-domain knowledge (no label, no learning curve, proportional frontier; see
               controller_ablate.py)
Runs land in runs/abl_ablate/<scenario>/<strategy>/r<seed>/ and are executed by abl_runner_any.py with
hotswap_run_ablate.py. Compared by seed with the matrix's Full / Static runs (same request sequences).
Usage: python abl_matrix_ablate.py manifest [path] [--smoke]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402  (frozen matrix definition)
from cfg import RUNS  # noqa: E402

CELLS = {
    "v6_novalue":  (dict(mode="controller", live=1, policy="costmodel", spec_switch=1), ["--ablate", "novalue"]),
    "v6_nostop":   (dict(mode="controller", live=1, policy="costmodel", spec_switch=1), ["--ablate", "nostop"]),
    "v6_noswitch": (dict(mode="controller", live=1, policy="costmodel", spec_switch=0, spec_switchable=0), []),
    "v6_nocurve":  (dict(mode="controller", live=1, policy="costmodel", spec_switch=1), ["--ablate", "nocurve"]),
}
for name, (st, _extra) in CELLS.items():
    abl_matrix.STRATEGIES[name] = st
SPEC_ON_B = {"v6_noswitch"}       # speculation never off: the long zh estimate


def manifest(path, smoke=False):
    rows = []
    root = f"{RUNS}/abl_ablate_smoke" if smoke else f"{RUNS}/abl_ablate"
    for seed in ([1] if smoke else abl_matrix.SEEDS):
        for scen in (["m1_code", "m1_zh"] if smoke else abl_matrix.SCENARIOS):
            for strat, (_st, extra) in CELLS.items():
                outdir = f"{root}/{scen}/{strat}/r{seed}"
                est = 6 if smoke else abl_matrix.SCENARIOS[scen]["est"]["spec_on" if strat in SPEC_ON_B else "spec_off"]
                rows.append(dict(run_id=f"{scen}.{strat}.r{seed}", scenario=scen, strategy=strat, seed=seed, est_min=est,
                                 outdir=outdir, script="hotswap_run_ablate.py",
                                 args=extra + abl_matrix.run_args(scen, strat, seed, outdir, smoke)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        smoke = "--smoke" in sys.argv
        rest = [x for x in sys.argv[2:] if x != "--smoke"]
        manifest(rest[0] if rest else f"{RUNS}/{'abl_ablate_smoke' if smoke else 'abl_ablate'}/manifest.jsonl", smoke)
