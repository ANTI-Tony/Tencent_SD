"""Experiment E (rent-or-buy): per (base, segment length) the phase-B throughput of switch / train_now / wait /
wait_d05 / wait_d2 / v6, paired by seed; differences against switch and against train_now; the opportunity loss
against the hindsight best of {switch, train_now} of the same seed (absolute tokens and % of the segment); and per
run the timing facts from the controller log: when training started (s after B traffic), when the head went live,
whether it went live before the segment ended, p99 TPOT.
Usage: python analyze_wait.py [--out results/abl_wait]"""
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


def jl(p):
    out = []
    if os.path.exists(p):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def timing(d):
    ev, tl, ctl = jl(f"{d}/events.jsonl"), jl(f"{d}/timeline.jsonl"), jl(f"{d}/controller.jsonl")
    tB = next(e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B")
    tBe = next(e["t"] for e in ev if e.get("event") == "phase_end")
    t0 = next((r["t"] for r in tl if r["t"] >= tB and r["gen_tok_s"] > 2000), tB)
    train = [r["at"] for r in ctl if r.get("reason") in ("budget_train", "saturated_value", "free_within_slo") and t0 <= r.get("at", 0) <= tBe and r.get("duty", 0) > 0]
    swaps = [r["at"] for r in ctl if r.get("event") == "swap" and tB <= r.get("at", 0) <= tBe]
    start = next((r for r in ctl if r.get("event") == "start"), {})
    return dict(seg_s=round(tBe - t0, 1), train_start_s=round(train[0] - t0, 1) if train else None,
                train_s=len(train), live_s=round(swaps[0] - t0, 1) if swaps else None, live_in_segment=bool(swaps),
                swaps=len(swaps), wait_s=(start.get("wait_s") or {}).get(start.get("dom", ""), None) if start else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/results/abl_wait")
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/data/runs/abl_wait/*/*/r*")):
        if not os.path.exists(f"{d}/done.json"):
            continue
        r = analyze_abl.run_row(d)
        r.update(timing(d))
        r["B_tokens"] = r["B_thr"] * r["B_dur_s"]
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/wait_runs.csv", index=False)
    res = []
    for sc in sorted(set(df.scenario)):
        x = df[df.scenario == sc]
        thr = x.pivot_table(index="seed", columns="strategy", values="B_thr")
        tok = x.pivot_table(index="seed", columns="strategy", values="B_tokens")
        if "switch" not in thr or "train_now" not in thr:
            continue
        best = tok[["switch", "train_now"]].max(axis=1)
        for strat in thr.columns:
            d_sw = (thr[strat] - thr["switch"]) / thr["switch"] * 100
            d_tn = (thr[strat] - thr["train_now"]) / thr["train_now"] * 100
            loss = best - tok[strat]
            seg = x[x.strategy == strat].seg_s.mean()
            res.append(dict(scenario=sc, strategy=strat, n=int(d_sw.notna().sum()), seg_s=round(seg, 1), thr=float(thr[strat].mean()),
                            vs_switch=float(d_sw.mean()), vs_switch_sd=float(d_sw.std(ddof=1)) if d_sw.notna().sum() > 1 else np.nan,
                            vs_train_now=float(d_tn.mean()),
                            loss_tok=float(loss.mean()), loss_pct=float((loss / best * 100).mean()),
                            train_start_s=x[x.strategy == strat].train_start_s.mean(), live_s=x[x.strategy == strat].live_s.mean(),
                            live_frac=float(x[x.strategy == strat].live_in_segment.mean()), p99=float(x[x.strategy == strat].B_p99.mean()),
                            per_seed_vs_switch=json.dumps([None if np.isnan(v) else round(v, 2) for v in d_sw.tolist()])))
    r = pd.DataFrame(res)
    r.to_csv(f"{a.out}/wait_pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
