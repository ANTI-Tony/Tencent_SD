"""Experiment E (prereg 2026-11-09): the payback-wait rule of the rent-or-buy model versus training at once, no
training, and the frozen controller, at segment lengths placed around the waiting threshold s = b / Delta and the
training time tau. All policies that train use the same fixed checkpoint n* (one head, exported at n*) and the same
calibration; the chat phases are not trained by the budget policies (budget domains = the B domain only).

Scenario base -> (checkpoint n*, segment lengths in requests: 0.5 s, s + tau/2, s + tau + 0.5 s, s + tau + 2 s):
  m2_code   n* 3200, s 38 s, tau ~71 s  -> 1000, 3850, 6700, 9650
  m1_mixcc  n* 1600, s 71 s, tau ~55 s  -> 1250, 3500, 5750, 9550   (the 16,000-prompt pool caps n* 3200's longest length)
  m1_code   n* 3200, s 50 s, tau ~72 s  -> 1300, 4400, 7550, 11350
Strategies: switch (no training), train_now (wait 0), wait (payback s), v6 (frozen controller); variants of wait with
--delta-scale 0.5 / 2 (biased benefit estimate, same segment lengths) for the second round.
Runs land in runs/abl_wait/<base>_w<k>/<strategy>/r<seed>/. Usage: python abl_matrix_wait.py manifest [path] [--only s1,s2] [--bases b1,b2]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402
import abl_matrix_mix  # noqa: E402
from cfg import RUNS  # noqa: E402

CHECKPOINT = {"m2_code": 3200, "m1_mixcc": 1600, "m1_code": 3200}
LENGTHS = {"m2_code": [1000, 3850, 6700, 9650], "m1_mixcc": [1250, 3500, 5750, 9550], "m1_code": [1300, 4400, 7550, 11350]}
EST = {1: 5, 2: 7, 3: 9, 4: 12}
BUDGET_ST = dict(mode="controller", live=1, policy="switch_only", spec_switch=1)      # the wrapper rewrites the policy
STRATS = {
    "switch":    (abl_matrix.STRATEGIES["switch"], "hotswap_run_ablate.py", None),
    "train_now": (BUDGET_ST, "hotswap_run_budget.py", "--budget-n {n} --budget-domains {dom} --wait-mode none --single-export 1"),
    "wait":      (BUDGET_ST, "hotswap_run_budget.py", "--budget-n {n} --budget-domains {dom} --wait-mode payback --single-export 1"),
    "wait_d05":  (BUDGET_ST, "hotswap_run_budget.py", "--budget-n {n} --budget-domains {dom} --wait-mode payback --single-export 1 --delta-scale 0.5"),
    "wait_d2":   (BUDGET_ST, "hotswap_run_budget.py", "--budget-n {n} --budget-domains {dom} --wait-mode payback --single-export 1 --delta-scale 2"),
    "v6":        (abl_matrix.STRATEGIES["v6"], "hotswap_run_ablate.py", None),
}
for name, (st, _s, _c) in STRATS.items():
    abl_matrix.STRATEGIES[name] = st


def manifest(path, only=None, bases=None):
    rows = []
    for seed in abl_matrix.SEEDS:
        for base in (bases or ["m2_code", "m1_mixcc", "m1_code"]):
            dom = abl_matrix.SCENARIOS[base]["dom_b"]
            for k, nB in enumerate(LENGTHS[base], start=1):
                scen = f"{base}_w{k}"
                abl_matrix.SCENARIOS[scen] = dict(abl_matrix.SCENARIOS[base], nB=nB)
                for strat, (_st, script, ctl) in STRATS.items():
                    if only and strat not in only:
                        continue
                    outdir = f"{RUNS}/abl_wait/{scen}/{strat}/r{seed}"
                    args = abl_matrix.run_args(scen, strat, seed, outdir)
                    extra = ["--model-json", abl_matrix_mix.MODEL_JSON] if base.startswith("m1_mix") else []
                    if ctl:
                        extra += ["--ctl-args", ctl.format(n=CHECKPOINT[base], dom=dom)]
                    rows.append(dict(run_id=f"{scen}.{strat}.r{seed}", scenario=scen, strategy=strat, seed=seed, est_min=EST[k],
                                     outdir=outdir, script=script, args=extra + args))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        argv = sys.argv[2:]
        only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
        bases = argv[argv.index("--bases") + 1].split(",") if "--bases" in argv else None
        skip = set()
        for flag in ("--only", "--bases"):
            if flag in argv:
                skip.update({flag, argv[argv.index(flag) + 1]})
        rest = [x for x in argv if x not in skip]
        tag = ("_" + "_".join(only) if only else "") + ("_" + "_".join(bases) if bases else "")
        manifest(rest[0] if rest else f"{RUNS}/abl_wait/manifest{tag}.jsonl", only, bases)
