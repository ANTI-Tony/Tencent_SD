"""Plan row 13, descriptive (not preregistered): configured duty vs actual trainer activity, and the serving
throughput lost while the trainer is busy, measured at matched acceptance length in the E28a saturated B phases.
Logs only. Windows: harness timeline (5 s), >= 250 running, speculation on, no swap / toggle in the 13 s before the
window end, first 10 s of B skipped. Trainer busy seconds are the exact overlap of trainer windows with each window."""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "sched"))
from sim import Model
from frontier_check import jl, SKIP0, RUN_MIN, SETTLE

RUNS = os.path.join(ROOT, "data", "runs", "abl_arm")
OUT = os.path.join(ROOT, "results", "abl_arm", "row13")
MODEL = {"m1_code": "model.json", "m2_code": "model_m2.json", "m1_mixcc": "model_mix.json"}


def windows(d):
    ev = jl(os.path.join(d, "events.jsonl"))
    tB = [e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B"][0]
    tE = [e["t"] for e in ev if e.get("event") == "phase_end" and e.get("name") == "B"][0]
    tr = [w for w in jl(os.path.join(d, "trainer.jsonl")) if "busy_s" in w]
    ctl = jl(os.path.join(d, "controller.jsonl"))
    ev_t = np.array([r.get("at", r["t"]) for r in ctl if r.get("event") in ("swap", "spec")])
    out = []
    for s in jl(os.path.join(d, "timeline.jsonl")):
        t, a = s["t"], s.get("acc_len")
        if not (tB + SKIP0 + 5.0 <= t <= tE) or s.get("running", 0.0) < RUN_MIN or a is None or a <= 1.0:
            continue
        if len(ev_t) and ((ev_t > t - SETTLE) & (ev_t <= t)).any():
            continue
        busy, duty = 0.0, 0.0
        for w in tr:
            lo, hi = w["t"] - w["win_s"], w["t"]
            ov = max(0.0, min(hi, t) - max(lo, t - 5.0))
            if ov > 0 and w["win_s"] > 0:
                busy += w["busy_s"] * ov / w["win_s"]
                duty = max(duty, w["duty"])
        out.append(dict(t_B=t - tB, tok_s=s["gen_tok_s"], acc=a, running=s["running"], busy_share=busy / 5.0, duty=duty))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sc in MODEL:
        for d in sorted(glob.glob(os.path.join(RUNS, sc, "*", "r[0-9]"))):
            for w in windows(d):
                rows.append(dict(scenario=sc, arm=d.split(os.sep)[-2], seed=int(d[-1]), **w))
    df = pd.DataFrame(rows)
    df["bin"] = (np.floor(df.acc / 0.05) * 0.05).round(2)
    idle = df[df.busy_share <= 0.0].groupby(["scenario", "bin"]).tok_s.agg(["count", "mean"]).rename(columns={"count": "n_idle", "mean": "idle_tok_s"})
    b = df[df.busy_share >= 0.1].join(idle, on=["scenario", "bin"])
    b = b[b.n_idle >= 5].copy()
    b["drop"] = 1 - b.tok_s / b.idle_tok_s
    res = []
    for sc, g in b.groupby("scenario"):
        kap = Model(os.path.join(HERE, "sched", MODEL[sc])).kappa(256)
        k_fit = float((g["drop"] * g.busy_share).sum() / (g.busy_share ** 2).sum())
        res.append(dict(scenario=sc, n_busy_windows=len(g), duty_mean=g.duty.mean(), busy_share_mean=g.busy_share.mean(),
                        busy_share_p10=g.busy_share.quantile(0.1), busy_share_p90=g.busy_share.quantile(0.9),
                        drop_mean_pct=100 * g["drop"].mean(), pred_drop_mean_pct=100 * kap * g.busy_share.mean(),
                        kappa_fit=k_fit, kappa_model=kap, arms=",".join(sorted(set(g.arm)))))
    r = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, "windows.csv"), index=False)
    b.to_csv(os.path.join(OUT, "busy_windows_matched.csv"), index=False)
    r.to_csv(os.path.join(OUT, "kappa_check.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("windows: %d (spec on, saturated, settled); trainer idle: %d; busy share >= 0.1: %d; matched to an idle bin: %d" % (
        len(df), int((df.busy_share <= 0).sum()), int((df.busy_share >= 0.1).sum()), len(b)))
    print(r.round(4).to_string(index=False))
    per_arm = b.groupby(["scenario", "arm"]).agg(n=("drop", "size"), duty=("duty", "mean"), busy=("busy_share", "mean"),
                                                drop_pct=("drop", lambda x: 100 * x.mean())).round(3)
    print(per_arm.to_string())


if __name__ == "__main__":
    main()
