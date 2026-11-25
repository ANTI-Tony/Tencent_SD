"""Kill test B: training budget versus segment length. Builds the run table of runs/abl_seg (pseudo-scenarios
<base>_n<nB>) plus the long-segment references (switch / fixed800 / v6 from runs/abl and runs/abl_mix, v7 from
runs/abl_v7, relabelled <base>_n16000), then per (base, nB) the paired % difference of phase-B throughput of every
strategy against switch and against the best fixed budget of that seed.
Usage: python analyze_segment.py --out results/abl_seg"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_abl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = ["m1_code", "m2_code", "m1_mixcc"]


def rows_from(root, scen_map=None):
    out = []
    for d in sorted(glob.glob(f"{root}/*/*/r*")):
        if not os.path.exists(f"{d}/done.json"):
            continue
        r = analyze_abl.run_row(d)
        if scen_map is not None:
            if r["scenario"] not in scen_map:
                continue
            r["scenario"] = scen_map[r["scenario"]]
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/results/abl_seg")
    a = ap.parse_args()
    rows = rows_from(f"{ROOT}/data/runs/abl_seg")
    long = {b: f"{b}_n16000" for b in BASE}
    for root in ("abl", "abl_mix", "abl_v7"):
        rows += [r for r in rows_from(f"{ROOT}/data/runs/{root}", long) if r["strategy"] in ("switch", "fixed800", "v6", "v7")]
    df = pd.DataFrame(rows).drop_duplicates(subset=["scenario", "strategy", "seed"], keep="first")
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/seg_runs.csv", index=False)
    res = []
    for sc in sorted(set(df.scenario)):
        x = df[df.scenario == sc].pivot_table(index="seed", columns="strategy", values="B_thr")
        if "switch" not in x:
            continue
        fixed = [c for c in ("fixed200", "fixed800") if c in x]
        for strat in x.columns:
            d_sw = (x[strat] - x["switch"]) / x["switch"] * 100
            best_fixed = x[fixed].max(axis=1) if fixed else None
            d_bf = ((x[strat] - best_fixed) / best_fixed * 100) if fixed else None
            res.append(dict(scenario=sc, strategy=strat, n=int(d_sw.notna().sum()), thr=float(x[strat].mean()),
                            vs_switch=float(d_sw.mean()), vs_switch_sd=float(d_sw.std(ddof=1)) if d_sw.notna().sum() > 1 else np.nan,
                            vs_best_fixed=float(d_bf.mean()) if d_bf is not None else np.nan,
                            per_seed_vs_switch=json.dumps([None if np.isnan(v) else round(v, 2) for v in d_sw.tolist()])))
    r = pd.DataFrame(res)
    r.to_csv(f"{a.out}/seg_pairs.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
