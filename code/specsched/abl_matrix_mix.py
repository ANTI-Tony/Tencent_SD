"""Mixed-traffic matrix on the frozen checkpoint: phase B serves a 50/50 mixture of two domains (make_mix_domains.py)
instead of one domain, 7 strategies x 2 mixtures x 3 repeats, A phases, seeds, engine configuration and harness as in
the 72-run matrix (abl_matrix.py, imported unchanged).

Scenarios: m1_mixcc (code + chat), m1_mixzc (Chinese + chat).
Strategies (all through hotswap_run_ablate.py, which hands the controller sched/model_mix.json):
  static, nospec, switch, tide     as in the matrix (the switch and TIDE read the mixture's frontier)
  v6          Full with the mixture's own label and calibration (learning curve + frontier of the mixture)
  v6_stale    Full told that phase B is still chat: the operator did not notice the change (--dom-b chat with the
              mixture's prompt file), so it uses chat's curve and frontier
  v6_nocurve  Full without any per-domain knowledge (controller_ablate.py --ablate nocurve)
Runs land in runs/abl_mix/<scenario>/<strategy>/r<seed>/. Usage: python abl_matrix_mix.py manifest [path] [--smoke]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402  (frozen matrix definition)
from cfg import RUNS  # noqa: E402

MODEL_JSON = os.path.join(HERE, "sched", "model_mix.json")
SCENARIOS = {
    "m1_mixcc": dict(model="m1", dom_b="mixcc", trainer_gb=18, nB=16000, est=dict(spec_on=14, spec_off=13)),
    "m1_mixzc": dict(model="m1", dom_b="mixzc", trainer_gb=18, nB=16000, est=dict(spec_on=16, spec_off=13)),
}
STRATS = {
    "static":     (abl_matrix.STRATEGIES["static"], []),
    "nospec":     (abl_matrix.STRATEGIES["nospec"], []),
    "switch":     (abl_matrix.STRATEGIES["switch"], []),
    "tide":       (dict(mode="controller", live=1, policy="switch_only", spec_switch=1), ["--tide"]),
    "v6":         (abl_matrix.STRATEGIES["v6"], []),
    "v6_stale":   (abl_matrix.STRATEGIES["v6"], []),
    "v6_nocurve": (abl_matrix.STRATEGIES["v6"], ["--ablate", "nocurve"]),
}
SPEC_ON_B = {"static"}
abl_matrix.SCENARIOS.update(SCENARIOS)
for name, (st, _e) in STRATS.items():
    abl_matrix.STRATEGIES[name] = st


def manifest(path, smoke=False):
    rows = []
    root = f"{RUNS}/abl_mix_smoke" if smoke else f"{RUNS}/abl_mix"
    for seed in ([1] if smoke else abl_matrix.SEEDS):
        for scen in (["m1_mixzc"] if smoke else SCENARIOS):
            for strat, (_st, extra) in STRATS.items():
                outdir = f"{root}/{scen}/{strat}/r{seed}"
                args = abl_matrix.run_args(scen, strat, seed, outdir, smoke)
                if strat == "v6_stale":                       # the label stays chat; the prompts are the mixture's
                    args[args.index("--dom-b") + 1] = "chat"
                assert args[args.index("--file-b") + 1].endswith(f"{SCENARIOS[scen]['dom_b']}_r{seed}.jsonl")
                est = 6 if smoke else SCENARIOS[scen]["est"]["spec_on" if strat in SPEC_ON_B else "spec_off"]
                rows.append(dict(run_id=f"{scen}.{strat}.r{seed}", scenario=scen, strategy=strat, seed=seed, est_min=est,
                                 outdir=outdir, script="hotswap_run_ablate.py",
                                 args=extra + ["--model-json", MODEL_JSON] + args))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        smoke = "--smoke" in sys.argv
        rest = [x for x in sys.argv[2:] if x != "--smoke"]
        manifest(rest[0] if rest else f"{RUNS}/{'abl_mix_smoke' if smoke else 'abl_mix'}/manifest.jsonl", smoke)
