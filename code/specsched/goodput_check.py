"""Post hoc (not preregistered; E28's primary metric is B throughput): the reviewer's suggested metric, B-phase
goodput under the controller's own TPOT SLO = output tokens of requests whose TPOT <= SLO, per second of phase B.
Same seeds paired as in the preregistered contrasts. Closed loop, so there is no queueing term."""
import os
import numpy as np, pandas as pd
ROOT = "/Applications/proj_hl/brainstorm/spec_decoding"
RED = f"{ROOT}/results/_reduced/abl_arm"
SLO = {"m1": 16.69212223804753, "m2": 10.612297103406213}
PAIRS = [("full", "sw"), ("full", "novalue"), ("full", "round800"), ("full", "budget"), ("full", "duty05"), ("sw", "switch"), ("full", "switch")]


def main():
    r = pd.read_csv(f"{ROOT}/results/abl_arm/runs_full.csv")
    g = []
    for x in r.itertuples():
        z = np.load(f"{RED}/{x.scenario}__{x.strategy}__r{int(x.seed)}.npz")
        ok = z["ok"] & (z["out"] >= 2)
        tpot = (z["e2e"] - z["ttft"]) / np.maximum(z["out"] - 1, 1) * 1e3
        good = ok & (tpot <= SLO["m2" if x.scenario.startswith("m2") else "m1"])
        g.append(dict(scenario=x.scenario, strategy=x.strategy, seed=int(x.seed), B_dur_s=x.B_dur_s,
                      goodput_tok_s=float(z["out"][good].sum()) / x.B_dur_s, slo_share=float(good.mean()), B_thr=x.B_thr))
    G = pd.DataFrame(g)
    rows = []
    for sc, s in G.groupby("scenario"):
        for a, b in PAIRS:
            xa, xb = s[s.strategy == a].set_index("seed"), s[s.strategy == b].set_index("seed")
            k = sorted(set(xa.index) & set(xb.index))
            if len(k) < 3:
                continue
            if (xb.goodput_tok_s[k] <= 0).any():
                rows.append(dict(scenario=sc, a=a, b=b, mean_pct=np.nan, per_seed="no request meets the SLO", verdict="n/a"))
                continue
            d = np.array([100 * (xa.goodput_tok_s[i] / xb.goodput_tok_s[i] - 1) for i in k])
            same = (d > 0).all() or (d < 0).all()
            v = "better" if d.mean() >= 2 and same else ("worse" if d.mean() <= -2 and same else "not separable")
            rows.append(dict(scenario=sc, a=a, b=b, mean_pct=d.mean(), per_seed="; ".join(f"{u:+.2f}" for u in d), verdict=v,
                             a_share=xa.slo_share[k].mean(), b_share=xb.slo_share[k].mean()))
    P = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/results/abl_arm/plan_rows", exist_ok=True)
    G.to_csv(f"{ROOT}/results/abl_arm/plan_rows/goodput_runs.csv", index=False)
    P.to_csv(f"{ROOT}/results/abl_arm/plan_rows/goodput_pairs.csv", index=False)
    pd.set_option("display.width", 200)
    print(P.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
