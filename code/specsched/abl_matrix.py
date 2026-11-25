"""Ablation / repeat matrix (10-23): 3 scenarios x 8 strategies x 3 repeats, one frozen configuration.

Scenario = model + new domain B (A is always chat at C=16; B saturates at C=256).
Strategy (all share the engine configuration except `vanilla`, the production reference):
  vanilla   static speculation as deployed today: stock head, no tap, full KV memory
  nospec    plain decoding: server started without a drafter (same KV budget as the system)
  static    static speculation + the system's infrastructure (tap on, trainer reserve), nothing trains
  switch    runtime frontier switch only (controller rule 1), nothing trains
  simple    threshold rules: speculation off iff saturated, train at duty 0.5 iff within SLO (OSD's idle rule)
  fixed     fixed duty 0.5 from the first request to the last, heads swapped at 100, 200, 400 ... consumed
  fixed800  fixed duty 0.5 during B only (regime change known), stop after the 800-request head (10-08 baseline)
  v6        the full controller: frontier switch + free duty within SLO + value rule at saturation + stop rules
Repeat r = prompt order seed r (bench + pools) and trainer seed r; strategies are paired by repeat.

`python abl_matrix.py manifest <out.tsv>` writes one line per run: run_id, scenario, strategy, seed, est_min, args.
"""
import json
import os
import sys

from cfg import DATA, RUNS

SCENARIOS = {
    "m1_code": dict(model="m1", dom_b="code", trainer_gb=18, nB=16000, est=dict(spec_on=12, spec_off=11)),
    "m1_zh":   dict(model="m1", dom_b="zh",   trainer_gb=18, nB=16000, est=dict(spec_on=17, spec_off=12)),
    "m2_code": dict(model="m2", dom_b="code", trainer_gb=5,  nB=16000, est=dict(spec_on=13, spec_off=13)),
}
NA = 200
CA, CB = 16, 256

STRATEGIES = {
    "vanilla":  dict(mode="static", live=0, spec=1, reserve_always=0),
    "nospec":   dict(mode="static", live=0, spec=0, reserve_always=1),
    "static":   dict(mode="static", live=1, spec=1, reserve_always=1),
    "switch":   dict(mode="controller", live=1, policy="switch_only", spec_switch=1),
    "simple":   dict(mode="controller", live=1, policy="simple", spec_switch=1, fixed_duty=0.5),
    "fixed":    dict(mode="train_swap", live=1, duty=0.5, train_from="start", export_at="doubling:100"),
    "fixed800": dict(mode="train_swap", live=1, duty=0.5, train_from="B", export_at="100,200,400,800"),
    "v6":       dict(mode="controller", live=1, policy="costmodel", spec_switch=1),
}
# strategies whose B phase runs with speculation on throughout (zh: long B)
SPEC_ON_B = {"vanilla", "static", "fixed", "fixed800"}
SEEDS = [1, 2, 3]


def run_args(scen, strat, seed, outdir, smoke=False):
    sc, st = SCENARIOS[scen], STRATEGIES[strat]
    nA, nB = (40, 600) if smoke else (NA, sc["nB"])
    args = ["--model", sc["model"], "--dom-a", "chat", "--dom-b", sc["dom_b"], "--CA", str(CA), "--CB", str(CB),
            "--nA", str(nA), "--nB", str(nB), "--trainer-gb", str(sc["trainer_gb"]), "--seed", str(seed),
            "--file-a1", f"{DATA}/bench/abl/chat_r{seed}_A1.jsonl", "--file-a2", f"{DATA}/bench/abl/chat_r{seed}_A2.jsonl",
            "--file-b", f"{DATA}/bench/abl/{sc['dom_b']}_r{seed}.jsonl",
            "--outdir", outdir, "--strategy-name", strat]
    for k, v in st.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return args


def manifest(path, smoke=False):
    rows = []
    root = f"{RUNS}/abl_smoke" if smoke else f"{RUNS}/abl"
    for seed in ([1] if smoke else SEEDS):           # repeat 1 of everything first, then 2, then 3
        for scen in (["m1_code"] if smoke else SCENARIOS):
            for strat in STRATEGIES:
                outdir = f"{root}/{scen}/{strat}/r{seed}"
                est = 6 if smoke else SCENARIOS[scen]["est"]["spec_on" if strat in SPEC_ON_B else "spec_off"]
                rows.append(dict(run_id=f"{scen}.{strat}.r{seed}", scenario=scen, strategy=strat, seed=seed,
                                 est_min=est, outdir=outdir, args=run_args(scen, strat, seed, outdir, smoke)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        manifest(sys.argv[2] if len(sys.argv) > 2 else f"{RUNS}/abl/manifest.jsonl", smoke="--smoke" in sys.argv)
