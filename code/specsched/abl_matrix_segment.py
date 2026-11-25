"""Kill test B (prereg 2026-11-07): does the best training budget depend on the length of the traffic segment?
Phase B is shortened to nB = 2000 or 6000 requests (16000 is the frozen matrix); strategies per segment length:
  switch     frontier switch only, no training
  fixed200   fixed duty 0.5 from the start of B until the 200-request head (exports at 100, 200), then nothing
  fixed800   = the matrix's Informed schedule (exports at 100, 200, 400, 800)
  v6         the frozen controller (Full)
  v7         the self-calibrating controller (hotswap_run_v7.py)
Scenarios m1_code, m2_code, m1_mixcc; 3 repeats; same prompt files, seeds, engine configuration and harness as the
matrix. Runs land in runs/abl_seg/<scenario>_n<nB>/<strategy>/r<seed>/; for nB = 16000 only fixed200 is new (switch,
fixed800 and v6 are in runs/abl and runs/abl_mix, v7 in runs/abl_v7).
Usage: python abl_matrix_segment.py manifest [path]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abl_matrix  # noqa: E402
import abl_matrix_mix  # noqa: E402  (mixture scenarios + model_mix.json)
from cfg import RUNS  # noqa: E402

STRATS = {
    "switch":   (abl_matrix.STRATEGIES["switch"], "hotswap_run_ablate.py"),
    "fixed200": (dict(mode="train_swap", live=1, duty=0.5, train_from="B", export_at="100,200"), "hotswap_run_ablate.py"),
    "fixed800": (abl_matrix.STRATEGIES["fixed800"], "hotswap_run_ablate.py"),
    "v6":       (abl_matrix.STRATEGIES["v6"], "hotswap_run_ablate.py"),
    "v7":       (dict(mode="controller", live=1, policy="costmodel", spec_switch=1), "hotswap_run_v7.py"),   # = v7.1 when run
    "v70":      (dict(mode="controller", live=1, policy="costmodel", spec_switch=1), "hotswap_run_v70.py"),  # v7.0 configuration
}
for name, (st, _s) in STRATS.items():
    abl_matrix.STRATEGIES[name] = st
BASE = ["m1_code", "m2_code", "m1_mixcc"]
LENGTHS = [2000, 6000, 16000]
EST = {2000: 5, 6000: 8, 16000: 13}


def manifest(path, only=None):
    rows = []
    for seed in abl_matrix.SEEDS:
        for base in BASE:
            for nB in LENGTHS:
                scen = f"{base}_n{nB}"
                abl_matrix.SCENARIOS[scen] = dict(abl_matrix.SCENARIOS[base], nB=nB)
                for strat, (_st, script) in STRATS.items():
                    if only and strat not in only:
                        continue
                    if nB == 16000 and strat != "fixed200":
                        continue                                  # already in the frozen / mixture / v7 matrices
                    outdir = f"{RUNS}/abl_seg/{scen}/{strat}/r{seed}"
                    args = abl_matrix.run_args(scen, strat, seed, outdir)
                    extra = ["--model-json", abl_matrix_mix.MODEL_JSON] if base.startswith("m1_mix") and strat != "v7" else []
                    if base.startswith("m1_mix") and strat == "v7":
                        extra = ["--model-json", abl_matrix_mix.MODEL_JSON]
                    rows.append(dict(run_id=f"{scen}.{strat}.r{seed}", scenario=scen, strategy=strat, seed=seed, est_min=EST[nB],
                                     outdir=outdir, script=script, args=extra + args))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} runs, {sum(r['est_min'] for r in rows)} est. minutes -> {path}")


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
        rest = [x for x in sys.argv[2:] if x != "--only" and (only is None or x != ",".join(only))]
        manifest(rest[0] if rest else f"{RUNS}/abl_seg/manifest{'_' + '_'.join(only) if only else ''}.jsonl", only)
