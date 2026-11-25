"""Descriptive check for plan rows 11 and 14 (not preregistered): does the calibrated frontier (model*.json) predict
the speculative speedup measured in the real saturated B phases? Logs only; frozen files are read, never changed.

Samples: harness timeline (5-s windows) inside phase B, skipping the first 10 s, with >= 250 running requests, no
trainer activity in the window, and no swap / speculation toggle in the 13 s before the sample's end.
Speedup = generated tok/s with speculation on / plain tok/s of the same scenario (spec-off samples)."""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "sched"))
from sim import Model

RUNS = os.path.join(ROOT, "data", "runs")
OUT = os.path.join(ROOT, "results", "abl_arm", "frontier_check")
SETS = {"abl_arm": ["m1_code", "m2_code", "m1_mixcc"], "abl": ["m1_code", "m2_code"], "abl_mix": ["m1_mixcc"]}
MODEL = {"m1_code": ("model.json", "code"), "m2_code": ("model_m2.json", "code"), "m1_mixcc": ("model_mix.json", "mixcc")}
SKIP0, RUN_MIN, SETTLE = 10.0, 250.0, 13.0


def jl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def samples(d):
    ev = jl(os.path.join(d, "events.jsonl"))
    tb = [e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B"]
    te = [e["t"] for e in ev if e.get("event") == "phase_end" and e.get("name") == "B"]
    if not tb or not te:
        return []
    tB, tE = tb[0], te[0]
    tr = jl(os.path.join(d, "trainer.jsonl"))
    tw = np.array([[r["t"], r.get("busy_s", 0.0)] for r in tr if "busy_s" in r]) if tr else np.zeros((0, 2))
    ctl = jl(os.path.join(d, "controller.jsonl"))
    ev_t = np.array([r["at"] if "at" in r else r["t"] for r in ctl if r.get("event") in ("swap", "spec")]) if ctl else np.zeros(0)
    rows = []
    for s in jl(os.path.join(d, "timeline.jsonl")):
        t = s["t"]
        if not (tB + SKIP0 + 5.0 <= t <= tE) or s.get("running", 0.0) < RUN_MIN:
            continue
        busy = float(tw[(tw[:, 0] > t - 5.0) & (tw[:, 0] <= t + 1.0), 1].sum()) if len(tw) else 0.0
        near = bool(len(ev_t) and ((ev_t > t - SETTLE) & (ev_t <= t)).any())
        a = s.get("acc_len")
        rows.append(dict(t_B=t - tB, tok_s=s["gen_tok_s"], acc=a if a is not None else np.nan, running=s["running"],
                         spec=a is not None and a > 1.0, busy=busy, near_event=near))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    allr = []
    for st, scens in SETS.items():
        for sc in scens:
            for d in sorted(glob.glob(os.path.join(RUNS, st, sc, "*", "r[0-9]"))):
                arm, seed = d.split(os.sep)[-2], int(d[-1])
                for r in samples(d):
                    allr.append(dict(set=st, scenario=sc, arm=arm, seed=seed, **r))
    df = pd.DataFrame(allr)
    df["clean"] = (df.busy <= 0.0) & ~df.near_event
    df.to_csv(os.path.join(OUT, "samples.csv"), index=False)
    c = df[df.clean]
    print("samples in B with >= %d running: %d; clean: %d" % (RUN_MIN, len(df), len(c)))
    # plain reference: spec-off clean samples, pooled per batch and scenario (MoE has none in abl_arm)
    plain = c[~c.spec].groupby(["set", "scenario"]).tok_s.agg(["count", "mean", "std", "median"])
    print("\nPLAIN tok/s (spec off, clean, saturated)")
    print(plain.round(1).to_string())
    pa = c[~c.spec].groupby(["set", "scenario", "arm"]).tok_s.agg(["count", "mean"]).round(1)
    print(pa.to_string())
    ref = {}
    for (st, sc), r in plain.iterrows():
        ref[(st, sc)] = r["mean"]
    for sc in ["m2_code"]:
        if ("abl_arm", sc) not in ref and ("abl", sc) in ref:
            ref[("abl_arm", sc)] = ref[("abl", sc)]    # cross-batch: frozen nospec runs, same memory share
    rows, fits = [], []
    for (st, sc), g in c[c.spec].groupby(["set", "scenario"]):
        if (st, sc) not in ref:
            continue
        mf, dom = MODEL[sc]
        m = Model(os.path.join(HERE, "sched", mf))
        m.cur_dom = dom
        g = g.copy()
        g["speedup"] = g.tok_s / ref[(st, sc)]
        g["model"] = [m.gain(a, C) for a, C in zip(g.acc, g.running)]
        g["bin"] = (np.floor(g.acc / 0.05) * 0.05).round(2)
        for b, h in g.groupby("bin"):
            rows.append(dict(set=st, scenario=sc, acc_bin=b, n=len(h), acc_mean=h.acc.mean(), tok_s=h.tok_s.mean(),
                             speedup=h.speedup.mean(), model=h.model.mean(), diff_pts=100 * (h.model.mean() - h.speedup.mean()),
                             arms=",".join(sorted(set(h.arm)))))
        A = np.vstack([g.acc, np.ones(len(g))]).T
        (al, be), *_ = np.linalg.lstsq(A, g.speedup.to_numpy(), rcond=None)
        f = m.fr[dom]
        lc = np.log2(256.0)
        ma, mb = float(np.interp(lc, f[0], f[1])), float(np.interp(lc, f[0], f[2]))
        fits.append(dict(set=st, scenario=sc, n=len(g), acc_min=g.acc.min(), acc_max=g.acc.max(), plain_ref=ref[(st, sc)],
                         plain_ref_source=("abl" if (st == "abl_arm" and sc == "m2_code") else st),
                         fit_alpha=al, fit_beta=be, fit_breakeven=(1 - be) / al if al > 0 else np.nan,
                         model_alpha=ma, model_beta=mb, model_breakeven=(1 - mb) / ma,
                         mean_diff_pts=100 * (g.model - g.speedup).mean(),
                         model_base256=m.thr_base(256)))
    bins = pd.DataFrame(rows)
    fit = pd.DataFrame(fits)
    bins.to_csv(os.path.join(OUT, "speedup_bins.csv"), index=False)
    fit.to_csv(os.path.join(OUT, "speedup_fit.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 300)
    print("\nSPEEDUP by acceptance bin (measured vs calibrated frontier; diff = model - measured, points)")
    print(bins[bins.n >= 5].round(3).to_string(index=False))
    print("\nLINEAR FIT of measured speedup on acceptance vs the calibrated frontier at C=256")
    print(fit.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
