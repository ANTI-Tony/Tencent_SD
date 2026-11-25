"""Paired comparison of added matrix cells (ablation, mixed traffic) with reference policies, by repeat.

Builds the run table of every given run root with analyze_abl.run_row (same metrics as the matrix), concatenates
them, and for every (scenario, strategy) in the cells computes the per-seed % difference of phase-B throughput
against each reference strategy of the same scenario and seed (same request sequence), with mean, std and sign
agreement. References may live in another root (the frozen matrix's runs) or in the same one.
Usage: python analyze_cells.py --roots data/runs/abl_ablate data/runs/abl_all --cells v6_novalue,v6_nostop,...
                               --refs static,v6,switch,fixed800,nospec,tide --out results/abl_ablate
"""
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


def table(root):
    dirs = sorted(d for d in glob.glob(f"{root}/*/*/r*") if os.path.exists(f"{d}/done.json"))
    rows = [analyze_abl.run_row(d) for d in dirs]
    for r, d in zip(rows, dirs):
        r["root"] = root
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="run roots; the first holds the cells")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--refs", default="static,v6,switch,fixed800,nospec,tide")
    ap.add_argument("--metric", default="B_thr")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.concat([table(r) for r in a.roots], ignore_index=True)
    df = df.drop_duplicates(subset=["scenario", "strategy", "seed"], keep="first")   # first root wins
    os.makedirs(a.out, exist_ok=True)
    cells = a.cells.split(",")
    df[df.strategy.isin(cells)].to_csv(f"{a.out}/cell_runs.csv", index=False)
    out = []
    for sc in sorted(set(df[df.strategy.isin(cells)].scenario)):
        for cell in cells:
            x = df[(df.scenario == sc) & (df.strategy == cell)].set_index("seed")[a.metric]
            if x.empty:
                continue
            for ref in a.refs.split(","):
                y = df[(df.scenario == sc) & (df.strategy == ref)].set_index("seed")[a.metric]
                common = sorted(set(x.index) & set(y.index))
                if not common:
                    continue
                d = np.array([(x[s] - y[s]) / y[s] * 100 for s in common])
                out.append(dict(scenario=sc, cell=cell, ref=ref, n=len(d), mean_pct=d.mean(),
                                std_pct=d.std(ddof=1) if len(d) > 1 else float("nan"), min_pct=d.min(), max_pct=d.max(),
                                same_sign=bool((d > 0).all() or (d < 0).all()), per_seed=json.dumps([round(v, 2) for v in d]),
                                cell_mean=float(x.loc[common].mean()), ref_mean=float(y.loc[common].mean())))
    res = pd.DataFrame(out)
    res.to_csv(f"{a.out}/cell_pairs.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["scenario", "strategy", "seed", "host", "B_thr", "B_p99", "A1_p50", "A2_p50", "acc_last", "duty_B", "spec_on_B", "swaps_B", "train_tok", "reasons"]
    print(df[df.strategy.isin(cells)].sort_values(["scenario", "strategy", "seed"])[cols].round(2).to_string(index=False))
    print()
    print(res[["scenario", "cell", "ref", "n", "mean_pct", "std_pct", "same_sign", "per_seed"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
