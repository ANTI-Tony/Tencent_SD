"""Manifests of the same-start comparison (E28, preregistered 2026-11-24).

E28a (closed loop, frozen protocol), runs/abl_arm/<scenario>/<arm>/r<seed>:
  m1_code, m2_code, m1_mixcc : switch sw round800 budget duty05 novalue full   (63 runs)
  m1_zh, m1_mixzc            : switch sw round800 duty05 full                  (30 runs; budget = sw and novalue = full there)
E28b (open loop, identical arrival timestamps), runs/abl_arm_open/<scenario>/<arm>/r<seed>:
  m1_code, m2_code, m1_mixcc : sw round800 duty05 full at 1.2 x the frozen Switch request rate   (36 runs)
`switch` and `full` of E28a are the frozen strategies on their original scripts; every other row goes through
hotswap_run_arm.py. Usage: python abl_matrix_arm.py manifest {a|b|smoke} [out.jsonl]"""
import json
import os
import sys

import abl_matrix
import abl_matrix_mix  # registers the mixture scenarios; MODEL_JSON = sched/model_mix.json
from cfg import RUNS

V6 = dict(abl_matrix.STRATEGIES["v6"])
ARMS = {                       # arm -> (frozen strategy it is built on, wrapper arguments or None = frozen script)
    "switch":   ("switch", None),
    "sw":       ("v6", ["--arm", "sw"]),
    "round800": ("v6", ["--arm", "round", "--arm-n", "800"]),
    "budget":   ("v6", ["--arm", "budget"]),
    "duty05":   ("v6", ["--arm", "duty"]),
    "novalue":  ("v6", ["--arm", "novalue"]),
    "full":     ("v6", None),
}
for _arm, (_base, _x) in ARMS.items():
    abl_matrix.STRATEGIES.setdefault(_arm, dict(abl_matrix.STRATEGIES[_base]))
abl_matrix.STRATEGIES["full"] = V6

TRAIN_SCEN = ["m1_code", "m2_code", "m1_mixcc"]
VETO_SCEN = ["m1_zh", "m1_mixzc"]
ARMS_TRAIN = ["switch", "sw", "round800", "budget", "duty05", "novalue", "full"]
ARMS_VETO = ["switch", "sw", "round800", "duty05", "full"]
OPEN_RATE = {"m1_code": 69.4, "m2_code": 63.0, "m1_mixcc": 51.0}     # requests/s = 1.2 x frozen Switch (57.86, 52.46, 42.51)
ARMS_OPEN = ["sw", "round800", "duty05", "full"]


def row(root, scen, arm, seed, smoke=False, open_rate=None, nB=None):
    outdir = f"{root}/{scen}/{arm}/r{seed}"
    args = abl_matrix.run_args(scen, arm, seed, outdir, smoke)
    if nB:
        args[args.index("--nB") + 1] = str(nB)
    mix = scen.startswith("m1_mix")
    extra = ["--model-json", abl_matrix_mix.MODEL_JSON] if mix else []
    wrap = ARMS[arm][1]
    if wrap is None and open_rate is None:
        script = "hotswap_run_ablate.py" if mix else "hotswap_run.py"       # the frozen arms, exactly as in abl / abl_mix
    else:
        script = "hotswap_run_arm.py"
        extra = (wrap or []) + extra + (["--open-rate", str(open_rate)] if open_rate else [])
    # frozen estimate + 1 min: an arm that trains through all of B runs about a minute longer than Switch
    est = 6 if smoke else abl_matrix.SCENARIOS[scen]["est"]["spec_off"] + 1
    return dict(run_id=f"{scen}.{arm}.r{seed}", scenario=scen, strategy=arm, seed=seed, est_min=est, outdir=outdir,
                script=script, args=extra + args)


def manifest(kind, path=None):
    rows = []
    if kind == "a":
        root = f"{RUNS}/abl_arm"
        for scens, arms in ((TRAIN_SCEN, ARMS_TRAIN), (VETO_SCEN, ARMS_VETO)):
            for seed in abl_matrix.SEEDS:
                for scen in scens:
                    rows += [row(root, scen, arm, seed) for arm in arms]
    elif kind == "b":
        root = f"{RUNS}/abl_arm_open"
        for seed in abl_matrix.SEEDS:
            for scen in TRAIN_SCEN:
                rows += [row(root, scen, arm, seed, open_rate=OPEN_RATE[scen]) for arm in ARMS_OPEN]
    elif kind == "smoke":
        root = f"{RUNS}/abl_arm_smoke"

        def named(r, name, est, swap=None):                      # rename a row and overwrite single argument values
            r = dict(r, run_id=f"{r['scenario']}.{name}.r1", strategy=name, outdir=f"{root}/{r['scenario']}/{name}/r1",
                     est_min=est)
            for flag, val in (swap or {}).items():
                r["args"][r["args"].index(flag) + 1] = val
            r["args"][r["args"].index("--outdir") + 1] = r["outdir"]
            return r
        # closed loop, arm controller on 8B: full-length A1 (one chat head before B), stop after 300, final export
        rows.append(named(row(root, "m1_code", "round800", 1, nB=3000), "round300", 10, {"--arm-n": "300"}))
        # the preregistered open-loop smoke: 4,000 prompts at the E28b rate, frozen controller through the wrapper
        rows.append(named(row(root, "m1_code", "full", 1, open_rate=OPEN_RATE["m1_code"], nB=4000), "full_open", 9))
        # open-loop stress: 4,000 prompts at 200 requests/s reach the backlog depth of a full run within 20 s
        rows.append(named(row(root, "m1_code", "duty05", 1, open_rate=200.0, nB=4000), "duty05_open200", 9))
        # MoE, calibrated budget (659): no head before B, stop and final export
        rows.append(named(row(root, "m2_code", "budget", 1, nB=3000), "budget3k", 13))
    else:
        raise SystemExit(f"unknown manifest kind {kind}")
    path = path or f"{root}/manifest.jsonl"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        manifest(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
