"""Experiment E diagnostics beyond the paired totals of analyze_wait.py:
  (1) trainer scan per run: host, consumption rate (requests/s over the training ticks), dropped tap tokens, crash;
  (2) time profile: throughput relative to the switch run of the same seed in windows of time since the B traffic
      started (longest segment w4 by default), one row per (base, strategy, window);
  (3) realized gain of the trained head: throughput in the last 40 s of the w4 segment (after the head went live)
      relative to switch in the same window, next to the calibrated gain (g1 vs the off/base state).
Usage: python analyze_wait_profile.py [--out results/abl_wait] [--seg w4]"""
import argparse
import glob
import json
import os
import statistics as st
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = f"{ROOT}/data/runs/abl_wait"
WINS = [(0, 10), (10, 40), (40, 80), (80, 120), (120, 160), (160, 240)]
TRAIN_REASONS = ("budget_train", "saturated_value", "free_within_slo")


def jl(p):
    out = []
    if os.path.exists(p):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def bounds(d):
    ev, tl = jl(f"{d}/events.jsonl"), jl(f"{d}/timeline.jsonl")
    tB = next(e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B")
    tBe = next(e["t"] for e in ev if e.get("event") == "phase_end")
    t0 = next((r["t"] for r in tl if r["t"] >= tB and r["gen_tok_s"] > 2000), tB)
    return tB, t0, tBe, tl


def win(tl, a, b, min_ticks=5):
    v = [r["gen_tok_s"] for r in tl if a <= r["t"] < b and r["gen_tok_s"] > 2000]
    return st.mean(v) if len(v) >= min_ticks else None


def calib(d):
    """(g0, g1) of the B domain from the controller start event of a budget-policy run; None for other policies."""
    start = next((r for r in jl(f"{d}/controller.jsonl") if r.get("event") == "start"), {})
    c = start.get("calib") or {}
    if not c:
        return None
    v = next(iter(c.values()))
    return v.get("g0"), v.get("g1")


def trainer_scan(dirs):
    rows = []
    for d in dirs:
        sc, strat, seed = d.split("/")[-3:]
        done = json.load(open(f"{d}/done.json"))
        ticks = [(r["t"], r.get("consumed", 0), r.get("at")) for r in jl(f"{d}/controller.jsonl")
                 if r.get("reason") in TRAIN_REASONS and r.get("duty", 0) > 0]
        stop = next((r for r in jl(f"{d}/trainer.jsonl") if r.get("event") == "stop"), {})
        out = open(f"{d}/trainer.out", errors="replace").read() if os.path.exists(f"{d}/trainer.out") else ""
        rate = (ticks[-1][1] - ticks[0][1]) / max(ticks[-1][0] - ticks[0][0], 1.0) if len(ticks) > 1 else None
        rows.append(dict(scenario=sc, strategy=strat, seed=int(seed[1:]), host=done.get("host", "?").split(".")[0],
                         started=datetime.fromtimestamp(ticks[0][2]).strftime("%m-%d %H:%M") if ticks and ticks[0][2] else "",
                         train_ticks=len(ticks), consumed=ticks[-1][1] if ticks else 0, rate_req_s=round(rate, 1) if rate is not None else None,
                         total_steps=stop.get("total_steps"), dropped_gap=stop.get("dropped_gap"), dropped_overflow=stop.get("dropped_overflow"),
                         crash=("Traceback" in out or "RuntimeError" in out), oom_warn=("OOM on device" in out)))
    df = pd.DataFrame(rows)
    # slow trainer: < 0.6 x the median rate of the base, outside w1 (where the idle ramp dominates), >= 20 ticks
    df["base"] = df.scenario.str.rsplit("_", n=1).str[0]
    med = df[(df.train_ticks >= 20) & (df.strategy != "v6") & (~df.scenario.str.endswith("_w1"))].groupby("base").rate_req_s.median()
    df["slow_trainer"] = [(r.rate_req_s is not None and r.train_ticks >= 20 and r.strategy != "v6" and not r.scenario.endswith("_w1")
                           and r.rate_req_s < 0.6 * med.get(r.base, float("inf"))) for r in df.itertuples()]
    return df, med


def profile(seg, wins):
    rows = []
    for base in ["m2_code", "m1_code", "m1_mixcc"]:
        for strat in ["v6", "train_now", "wait", "wait_d05", "wait_d2"]:
            for a, b in wins:
                vals = []
                for seed in (1, 2, 3):
                    d, ds = f"{RUNS}/{base}_{seg}/{strat}/r{seed}", f"{RUNS}/{base}_{seg}/switch/r{seed}"
                    if not (os.path.exists(f"{d}/done.json") and os.path.exists(f"{ds}/done.json")):
                        continue
                    _, t0, tBe, tl = bounds(d)
                    _, t0s, tBes, tls = bounds(ds)
                    x, y = win(tl, t0 + a, min(t0 + b, tBe)), win(tls, t0s + a, min(t0s + b, tBes))
                    if x and y:
                        vals.append((x / y - 1) * 100)
                if vals:
                    rows.append(dict(base=base, strategy=strat, window=f"{a}-{b}s", vs_switch_pct=round(st.mean(vals), 2), n=len(vals),
                                     per_seed=json.dumps([round(v, 1) for v in vals])))
    return pd.DataFrame(rows)


def realized(seg, tail_s=40):
    rows = []
    for base in ["m2_code", "m1_code", "m1_mixcc"]:
        g = None
        for strat in ["train_now", "wait", "v6"]:
            vals, lives = [], []
            for seed in (1, 2, 3):
                d, ds = f"{RUNS}/{base}_{seg}/{strat}/r{seed}", f"{RUNS}/{base}_{seg}/switch/r{seed}"
                if not (os.path.exists(f"{d}/done.json") and os.path.exists(f"{ds}/done.json")):
                    continue
                g = g or calib(d)
                tB, t0, tBe, tl = bounds(d)
                _, _, tBes, tls = bounds(ds)
                sw = [r["at"] for r in jl(f"{d}/controller.jsonl") if r.get("event") == "swap" and r.get("at", 0) >= tB]
                a = tBe - tail_s
                if sw and sw[0] + 5 > a:
                    a = sw[0] + 5
                x, y = win(tl, a, tBe), win(tls, tBes - (tBe - a), tBes)
                if x and y:
                    vals.append((x / y - 1) * 100)
                    lives.append(bool(sw))
            if vals:
                g0, g1 = g if g else (None, None)
                off = (0.97 if g0 is not None and g0 < 1 else g0) if g0 is not None else None
                rows.append(dict(base=base, strategy=strat, realized_gain_pct=round(st.mean(vals), 2), per_seed=json.dumps([round(v, 1) for v in vals]),
                                 heads_live=sum(lives), calibrated_gain_pct=round((g1 - off) / off * 100, 2) if g1 and off else None, g0=g0, g1=g1))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/results/abl_wait")
    ap.add_argument("--seg", default="w4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(d for d in glob.glob(f"{RUNS}/*/*/r*") if os.path.exists(f"{d}/done.json"))
    pd.set_option("display.width", 250)
    tr, med = trainer_scan(dirs)
    tr.to_csv(f"{a.out}/wait_trainer.csv", index=False)
    print(f"trainer scan: {len(tr)} runs; median consumption rate (req/s, w2-w4, budget policies): {med.round(1).to_dict()}")
    flagged = tr[tr.slow_trainer | tr.crash]
    print("flagged (slow trainer or crash):")
    print(flagged[["scenario", "strategy", "seed", "host", "started", "rate_req_s", "consumed", "dropped_gap", "crash"]].to_string(index=False) if len(flagged) else "  none")
    print(f"OOM-retry warnings in trainer.out by base: {tr.groupby('base').oom_warn.sum().to_dict()}")
    pr = profile(a.seg, WINS)
    pr.to_csv(f"{a.out}/wait_profile_{a.seg}.csv", index=False)
    print(f"\ntime profile ({a.seg}): throughput vs switch (%), windows of seconds since B traffic start")
    print(pr.pivot_table(index=["base", "strategy"], columns="window", values="vs_switch_pct", sort=False).reindex(columns=[f"{x}-{y}s" for x, y in WINS]).to_string())
    re = realized(a.seg)
    re.to_csv(f"{a.out}/wait_realized_{a.seg}.csv", index=False)
    print(f"\nrealized gain in the last 40 s of {a.seg} vs calibrated gain:")
    print(re.to_string(index=False))


if __name__ == "__main__":
    main()
