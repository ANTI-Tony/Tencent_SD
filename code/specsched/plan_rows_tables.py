"""Plan rows 2, 3 and 19 from existing per-run tables and reduced request arrays (analysis only, no new runs).
Row 2: lifecycle throughput (A1, B, A2, whole run), completion time, training volume.
Row 3: B-phase request latency (TPOT, TTFT, end-to-end) and the post hoc share of requests with TPOT within the
controller's SLO (no hard-SLO claim). Row 19: paired per-seed differences with 95% t-intervals (n = 3) and A/A floors."""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "abl_arm", "plan_rows")
RED = os.path.join(ROOT, "results", "_reduced")
T975 = 4.302652729911275          # t quantile, 2 degrees of freedom
SLO_MS = {"m1": 16.69212223804753, "m2": 10.612297103406213}
NAME = {"v6": "Full", "nospec": "Plain", "static": "Static", "switch": "Switch", "fixed800": "Informed", "fixed": "Always",
        "tide": "TIDE-style", "simple": "Simple", "vanilla": "Vanilla", "v6_nocurve": "No curve", "v6_stale": "Stale label",
        "full": "Full", "sw": "sw", "round800": "round800", "budget": "budget", "duty05": "duty05", "novalue": "novalue"}
METRICS2 = ["A1_thr", "B_thr", "A2_thr", "all_thr", "all_dur_s", "train_tok"]
LAT = ["tpot_mean", "tpot_p50", "tpot_p95", "tpot_p99", "ttft_p50", "ttft_p99", "e2e_p50", "e2e_p99"]


def load():
    a = pd.read_csv(os.path.join(ROOT, "results", "abl_arm", "runs_full.csv")).assign(batch="E28a", set="abl_arm")
    f = pd.read_csv(os.path.join(ROOT, "results", "abl_arm", "frozen_abl", "runs_full.csv")).assign(batch="frozen", set="abl")
    x = pd.read_csv(os.path.join(ROOT, "results", "abl_arm", "frozen_abl_mix", "runs_full.csv")).assign(batch="frozen", set="abl_mix")
    return pd.concat([a, f, x], ignore_index=True)


def slo_share(row):
    p = os.path.join(RED, row["set"], f"{row['scenario']}__{row['strategy']}__r{int(row['seed'])}.npz")
    if not os.path.exists(p):
        return np.nan, np.nan
    z = np.load(p)
    ok = z["ok"] & (z["out"] >= 2)
    tpot = (z["e2e"][ok] - z["ttft"][ok]) / (z["out"][ok] - 1) * 1e3
    slo = SLO_MS["m2" if row["scenario"].startswith("m2") else "m1"]
    return float((tpot <= slo).mean()), float(tpot.mean())


def paired(df, metric, a, b, lower_better=False):
    x = df[df.strategy == a].set_index("seed")[metric]
    y = df[df.strategy == b].set_index("seed")[metric]
    s = sorted(set(x.index) & set(y.index))
    if len(s) < 2:
        return None
    d = np.array([100 * (x[k] / y[k] - 1) for k in s])
    m, sd = d.mean(), d.std(ddof=1)
    hw = T975 * sd / np.sqrt(len(d)) if len(d) == 3 else np.nan
    same = (d > 0).all() or (d < 0).all()
    good = -m if lower_better else m
    verdict = "better" if good >= 2 and same else ("worse" if good <= -2 and same else "not separable")
    return dict(metric=metric, a=a, b=b, n=len(d), mean_pct=m, sd_pct=sd, ci_lo=m - hw, ci_hi=m + hw,
                per_seed="; ".join(f"{v:+.2f}" for v in d), verdict=verdict)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    sh = df.apply(slo_share, axis=1, result_type="expand")
    df["slo_share"], df["tpot_mean_req"] = sh[0], sh[1]
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 400)
    # ---- row 2
    means = df.groupby(["batch", "scenario", "strategy"])[METRICS2 + ["slo_share"] + LAT].mean().reset_index()
    means.insert(3, "name", means.strategy.map(NAME))
    means.to_csv(os.path.join(OUT, "means_by_strategy.csv"), index=False)
    pairs = []
    main_frozen = [("v6", b) for b in ["nospec", "static", "switch", "fixed800", "fixed", "tide", "simple", "vanilla", "v6_nocurve", "v6_stale"]]
    arm_pairs = [("full", "switch"), ("sw", "switch"), ("full", "sw"), ("full", "round800"), ("full", "budget"), ("full", "duty05"),
                 ("full", "novalue"), ("round800", "sw"), ("budget", "sw"), ("duty05", "sw"), ("novalue", "sw")]
    for (batch, sc), g in df.groupby(["batch", "scenario"]):
        for a, b in (main_frozen if batch == "frozen" else arm_pairs):
            for mt in METRICS2[:5] + ["tpot_p99", "ttft_p99", "e2e_p99", "slo_share"]:
                lb = mt in ("all_dur_s", "tpot_p99", "ttft_p99", "e2e_p99")
                r = paired(g, mt, a, b, lower_better=lb)
                if r:
                    pairs.append(dict(batch=batch, scenario=sc, **r))
    P = pd.DataFrame(pairs)
    P.to_csv(os.path.join(OUT, "pairs_all_metrics.csv"), index=False)
    print("ROW 2  frozen main matrix: Full (v6) against each baseline, per-seed paired % (whole run and phases)")
    v = P[(P.batch == "frozen") & P.metric.isin(["A1_thr", "B_thr", "A2_thr", "all_thr", "all_dur_s"])]
    print(v.pivot_table(index=["scenario", "b"], columns="metric", values="mean_pct").round(2).to_string())
    print("\nROW 2  E28a: A1 pre-adaptation (sw - switch) and Full, per-seed paired %")
    v = P[(P.batch == "E28a") & P.metric.isin(["A1_thr", "B_thr", "A2_thr", "all_thr", "all_dur_s"]) &
          P.apply(lambda r: (r.a, r.b) in [("sw", "switch"), ("full", "sw"), ("full", "switch")], axis=1)]
    print(v.pivot_table(index=["scenario", "a", "b"], columns="metric", values="mean_pct").round(2).to_string())
    print("\nROW 2  means (tok/s; s; tokens)")
    print(means[["batch", "scenario", "name"] + METRICS2].round(1).to_string(index=False))
    # ---- row 3
    print("\nROW 3  B-phase latency means over seeds (ms) and share of B requests with TPOT within the controller SLO "
          "(m1 %.2f ms, m2 %.2f ms)" % (SLO_MS["m1"], SLO_MS["m2"]))
    print(means[["batch", "scenario", "name", "slo_share"] + LAT].round(3).to_string(index=False))
    chk = df[["tpot_mean", "tpot_mean_req"]].dropna()
    print("  check: request TPOT from reduced arrays vs table tpot_mean: max abs diff %.4f ms over %d runs" % (
        (chk.tpot_mean - chk.tpot_mean_req).abs().max(), len(chk)))
    # ---- row 19
    print("\nROW 19  E28a preregistered contrasts on B throughput with 95% t-intervals (n = 3)")
    pre = [("sw", "switch"), ("full", "sw"), ("full", "novalue"), ("full", "round800"), ("full", "duty05"), ("full", "budget")]
    v = P[(P.batch == "E28a") & (P.metric == "B_thr") & P.apply(lambda r: (r.a, r.b) in pre, axis=1)]
    print(v[["scenario", "a", "b", "mean_pct", "sd_pct", "ci_lo", "ci_hi", "per_seed", "verdict"]].round(2).to_string(index=False))
    v.to_csv(os.path.join(OUT, "row19_e28a_ci.csv"), index=False)
    print("\nROW 19  frozen main matrix: Full (v6) vs baselines on B throughput with 95% t-intervals")
    v = P[(P.batch == "frozen") & (P.metric == "B_thr")]
    print(v[["scenario", "a", "b", "mean_pct", "sd_pct", "ci_lo", "ci_hi", "per_seed", "verdict"]].round(2).to_string(index=False))
    v.to_csv(os.path.join(OUT, "row19_frozen_ci.csv"), index=False)
    # A/A floors
    aa = []
    e = df[df.batch == "E28a"]
    for sc, a, b, kind in [("m2_code", "sw", "switch", "exact A/A"), ("m1_zh", "sw", "full", "near A/A"), ("m1_mixzc", "sw", "full", "near A/A")]:
        r = paired(e[e.scenario == sc], "B_thr", a, b)
        aa.append(dict(kind=kind, scenario=sc, a=a, b=b, **{k: r[k] for k in ("mean_pct", "sd_pct", "ci_lo", "ci_hi", "per_seed")}))
    fz = df[df.batch == "frozen"]
    for sc in ["m1_code", "m2_code", "m1_mixcc", "m1_zh", "m1_mixzc"]:
        for arm, frozen_name in [("switch", "switch"), ("full", "v6")]:
            x = e[(e.scenario == sc) & (e.strategy == arm)].set_index("seed").B_thr
            y = fz[(fz.scenario == sc) & (fz.strategy == frozen_name)].set_index("seed").B_thr
            s = sorted(set(x.index) & set(y.index))
            if not s:
                continue
            d = np.array([100 * (x[k] / y[k] - 1) for k in s])
            hw = T975 * d.std(ddof=1) / np.sqrt(3) if len(d) == 3 else np.nan
            aa.append(dict(kind="cross-batch A/A", scenario=sc, a=f"E28a {arm}", b=f"frozen {frozen_name}", mean_pct=d.mean(),
                           sd_pct=d.std(ddof=1), ci_lo=d.mean() - hw, ci_hi=d.mean() + hw, per_seed="; ".join(f"{v:+.2f}" for v in d)))
    A = pd.DataFrame(aa)
    A.to_csv(os.path.join(OUT, "row19_aa.csv"), index=False)
    print("\nROW 19  A/A noise floors on B throughput (%)")
    print(A.round(2).to_string(index=False))
    df.to_csv(os.path.join(OUT, "runs_with_slo_share.csv"), index=False)


if __name__ == "__main__":
    main()
