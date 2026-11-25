"""Drift experiments in the simulator: same acceptance drop, different future speculation opportunity.

Type I  persistence : A -> B burst at saturation, duration swept (B = sql, adaptable)
Type II ceiling     : A -> B persistent at saturation, B = sql (ceiling above the frontier) vs zh (ceiling below it)
Type III future load: A -> B persistent at moderate load, with vs without a later surge above base capacity

Writes results/summaries/drift_sim.csv and prints the tables.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import POLICIES, Model, drift_series, run  # noqa: E402

POLS = ["static_off", "spec_only", "osd_idle", "accdrop", "costmodel_oracle", "costmodel_persist", "costmodel_lindy"]


def lam_for(m, dom, C):
    return m.thr_base(C) / m.out_per_req(dom)


def table(rows):
    df = pd.DataFrame(rows)
    return df


def main(A="code", NA=3200, out="results/summaries/drift_sim.csv"):
    m = Model()
    rows = []
    # Type I: persistence at saturation (demand = no-spec capacity; the stock/transferred head is net negative;
    # an adapted head flips it). B = code (strong frontier; stock head, no prior adaptation so the drop is real)
    # and B = sql (flat frontier: marginal even at the ceiling).
    for B, N0 in [("code", {}), ("sql", {A: NA})]:
        for dur in [3, 10, 30, 120, 600, 1800]:
            s = drift_series(lam_for(m, A, 16), lam_for(m, B, 256), A, B, 300, dur, 600)
            for p in POLS:
                sm, _ = run(m, s, POLICIES[p], N0=dict(N0))
                rows.append(dict(exp="I_persistence", B=B, dur_s=dur, C_B=256, policy=p, **{k: sm[k] for k in
                            ["goodput", "slo_viol_s", "train_tok", "duty_mean", "spec_frac"]}, acc_B=sm["final_acc"][B]))
    # Type II: ceiling. same saturation, persistent 20 min: code (adaptable, strong), sql (adaptable, marginal),
    # zh (structural ceiling below the frontier)
    for B, N0 in [("code", {}), ("sql", {A: NA}), ("zh", {A: NA})]:
        s = drift_series(lam_for(m, A, 16), lam_for(m, B, 256), A, B, 300, 1200, 300)
        for p in POLS:
            sm, _ = run(m, s, POLICIES[p], N0=dict(N0))
            rows.append(dict(exp="II_ceiling", B=B, dur_s=1200, C_B=256, policy=p, **{k: sm[k] for k in
                        ["goodput", "slo_viol_s", "train_tok", "duty_mean", "spec_frac"]}, acc_B=sm["final_acc"][B]))
    # Type III: moderate load (C~128, within SLO) for 10 min, then a 10-min surge whose magnitude is swept
    # (fraction of the no-speculation capacity); "none" keeps the moderate load
    for surge in ["none", 0.95, 1.0, 1.05, 1.10]:
        lamB = lam_for(m, "sql", 128)
        s = drift_series(lam_for(m, A, 16), lamB, A, "sql", 300, 600, 0)
        s += [((lam_for(m, "sql", 256) * surge) if surge != "none" else lamB, "sql")] * 600
        s += [(lam_for(m, A, 16), A)] * 300
        for p in POLS:
            sm, _ = run(m, s, POLICIES[p], N0={A: NA})
            rows.append(dict(exp="III_future_load", B="sql", dur_s=1200, C_B=f"128->surge{surge}", policy=p,
                             **{k: sm[k] for k in ["goodput", "slo_viol_s", "train_tok", "duty_mean", "spec_frac"]},
                             acc_B=sm["final_acc"]["sql"]))
    df = table(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    for exp, g in df.groupby("exp", sort=False):
        print(f"\n=== {exp}")
        piv = g.pivot_table(index=["B", "C_B", "dur_s"], columns="policy", values="goodput", sort=False) / 1e6
        print("goodput (M tokens)\n", piv[[p for p in POLS if p in piv.columns]].round(2).to_string())
        piv = g.pivot_table(index=["B", "C_B", "dur_s"], columns="policy", values="slo_viol_s", sort=False)
        print("SLO-violation seconds\n", piv[[p for p in POLS if p in piv.columns]].round(0).to_string())
        piv = g.pivot_table(index=["B", "C_B", "dur_s"], columns="policy", values="train_tok", sort=False) / 1e6
        print("training tokens (M)\n", piv[[p for p in POLS if p in piv.columns]].round(2).to_string())
        piv = g.pivot_table(index=["B", "C_B", "dur_s"], columns="policy", values="acc_B", sort=False)
        print("final acceptance on B\n", piv[[p for p in POLS if p in piv.columns]].round(2).to_string())
    return df


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else "results/summaries/drift_sim.csv")
