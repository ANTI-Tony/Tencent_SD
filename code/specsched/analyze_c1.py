"""C1 analysis: speculation gain vs co-located update cost across load, per (model, regime).

Input : <runs>/c1/<model>_<regime>/<cond>/{summary.json, C*.json}
Output: <out>/c1_<model>_<regime>.png and rows for the Excel workbook (returned by analyze()).
Goodput SLO (pre-registered rule): TPOT <= 2x and TTFT <= 5x of S0's median at concurrency 1.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COND_LABEL = {"S0": "no spec", "S1": "spec", "S2": "no spec + update", "S3": "spec + update"}
COLORS = {"S0": "#7f7f7f", "S1": "#1f77b4", "S2": "#d62728", "S3": "#9467bd"}


def per_request_goodput(detail, slo_tpot, slo_ttft):
    """Requests/s meeting both SLOs, from a `vllm bench serve --save-detailed` result."""
    ttft = np.array(detail.get("ttfts", []), dtype=float) * 1e3
    lens = np.array(detail.get("output_lens", []), dtype=float)
    e2el = None
    if "itls" in detail:
        itl_sum = np.array([sum(x) for x in detail["itls"]], dtype=float) * 1e3
        tpot = np.where(lens > 1, itl_sum / np.maximum(lens - 1, 1), 0.0)
    else:
        return None
    ok = (tpot <= slo_tpot) & (ttft <= slo_ttft)
    return float(ok.sum()) / float(detail["duration"])


def load_group(group_dir):
    rows = []
    for cond_dir in sorted(glob.glob(f"{group_dir}/S*")):
        sp = f"{cond_dir}/summary.json"
        if not os.path.exists(sp):
            continue
        for r in json.load(open(sp)):
            r["cond_dir"] = os.path.basename(cond_dir)
            det = f"{cond_dir}/C{r['C']}.json"
            r["_detail"] = det if os.path.exists(det) else None
            rows.append(r)
    return pd.DataFrame(rows)


def analyze(group_dir, out_dir):
    df = load_group(group_dir)
    if df.empty:
        return None
    name = os.path.basename(group_dir)
    base = df[(df.cond == "S0") & (df.C == df.C.min())]
    slo_tpot = 2.0 * float(base["median_tpot_ms"].iloc[0]) if len(base) else None
    slo_ttft = max(5.0 * float(base["median_ttft_ms"].iloc[0]), 1000.0) if len(base) else None
    gp = []
    for _, r in df.iterrows():
        g = None
        if slo_tpot and r["_detail"]:
            g = per_request_goodput(json.load(open(r["_detail"])), slo_tpot, slo_ttft)
        gp.append(g)
    df["goodput_rps"] = gp
    main = df[df.cond_dir.isin(["S0", "S1", "S2_d1", "S3_d1"])].copy()
    piv = main.pivot_table(index="C", columns="cond", values=["output_throughput", "p99_tpot_ms",
                                                               "median_tpot_ms", "goodput_rps", "acc_len"])
    der = pd.DataFrame(index=piv.index)
    thr = piv["output_throughput"]
    if {"S0", "S1"} <= set(thr.columns):
        der["spec_gain"] = thr["S1"] / thr["S0"] - 1
    if {"S0", "S2"} <= set(thr.columns):
        der["update_cost"] = 1 - thr["S2"] / thr["S0"]
        der["update_p99_tpot_infl"] = piv["p99_tpot_ms"]["S2"] / piv["p99_tpot_ms"]["S0"] - 1
    if {"S0", "S1", "S2", "S3"} <= set(thr.columns):
        # pre-registered additive form (scale-confounded: a proportional loss on a faster S1 looks "larger")
        der["interaction"] = ((thr["S1"] - thr["S3"]) - (thr["S0"] - thr["S2"])) / thr["S0"]
        der["interaction_rel"] = der["interaction"] / der["update_cost"].where(der["update_cost"].abs() > 1e-3)
        # multiplicative form: does co-located updating erode the *relative* speculation gain?
        der["spec_gain_coloc"] = thr["S3"] / thr["S2"] - 1
        der["interaction_mult"] = (thr["S3"] / thr["S2"]) / (thr["S1"] / thr["S0"]) - 1
    gp = piv["goodput_rps"] if "goodput_rps" in piv else None
    if gp is not None:
        for cnd in ["S0", "S1", "S2", "S3"]:
            if cnd in gp.columns:
                der[f"goodput_{cnd}"] = gp[cnd]
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for cond in ["S0", "S1", "S2", "S3"]:
        d = main[main.cond == cond].sort_values("C")
        if d.empty:
            continue
        ax[0].plot(d.C, d.output_throughput, "o-", color=COLORS[cond], label=COND_LABEL[cond])
        ax[1].plot(d.C, d.p99_tpot_ms, "o-", color=COLORS[cond], label=COND_LABEL[cond])
    for col, lab, c in [("spec_gain", "spec gain (S1/S0-1)", "#1f77b4"),
                        ("update_cost", "update cost (1-S2/S0)", "#d62728"),
                        ("interaction", "interaction I(L)/thr(S0)", "#9467bd")]:
        if col in der:
            ax[2].plot(der.index, der[col], "o-", color=c, label=lab)
    ax[2].axhline(0, color="k", lw=0.6)
    for a_, t in zip(ax, ["output tokens/s", "P99 TPOT (ms)", "relative to S0 throughput"]):
        a_.set_xscale("log", base=2)
        a_.set_xlabel("concurrency")
        a_.set_title(t)
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/c1_{name}.png", dpi=150)
    plt.close(fig)
    df = df.drop(columns=["_detail"])
    return dict(name=name, raw=df, derived=der.reset_index(), slo=dict(tpot_ms=slo_tpot, ttft_ms=slo_ttft))


if __name__ == "__main__":
    runs, out = sys.argv[1], sys.argv[2]
    for g in sorted(glob.glob(f"{runs}/c1/*")):
        r = analyze(g, out)
        if r:
            print(r["name"], r["slo"])
            print(r["derived"].round(3).to_string(index=False))
