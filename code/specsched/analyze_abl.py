"""Ablation / repeat matrix analysis: one row per run, mean +- std per (scenario, strategy), paired differences
by repeat against `static` (same infrastructure, nothing trains) and `vanilla` (production reference).

Per run (from <outdir>/{done.json, phase*.json, timeline.jsonl, events.jsonl, trainer.jsonl, controller.jsonl}):
  B_thr        output tokens/s of the saturated phase B (bench)      B_dur_s     phase B wall time
  B_p50/p99    TPOT percentiles in B (ms)                            A1/A2_p50   TPOT p50 in the light phases
  all_thr      output tokens of A1+B+A2 / their wall time (the makespan objective: every strategy serves the
               same requests)                                        all_dur_s
  acc_first/last  acceptance length in the first 3 / last 6 sampler windows of B (None when speculation is off)
  duty_B       trainer duty mean over B      train_tok   tokens the trainer consumed (whole run)
  spec_on_B    fraction of controller ticks in B with speculation on   swaps   number of head swaps (whole run)
Usage: python analyze_abl.py <runs/abl dir> <out dir>
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ORDER = ["vanilla", "nospec", "static", "switch", "simple", "fixed", "fixed800", "v6"]


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def run_row(d):
    done = json.load(open(f"{d}/done.json"))
    ev, tl, tr, ctl = jl(f"{d}/events.jsonl"), jl(f"{d}/timeline.jsonl"), jl(f"{d}/trainer.jsonl"), jl(f"{d}/controller.jsonl")
    ph = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{d}/phase*.json")}
    tB = next(e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B")
    tBe = next(e["t"] for e in ev if e.get("event") == "phase_end")
    tA1 = next(e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "A1")
    tEnd = next(e["t"] for e in ev if e.get("event") == "done")
    # the bench tokenises its 16000-prompt pool for ~50 s before the first request: B's traffic window starts at
    # the first sampler window with real generation, and the behaviour metrics are taken over that window
    t0 = next((r["t"] for r in tl if r["t"] >= tB and r["gen_tok_s"] > 2000), tB)
    acc = [r["acc_len"] for r in tl if t0 <= r["t"] <= tBe and r["acc_len"]]
    duty = [r["duty"] for r in tr if "duty" in r and t0 <= r["t"] <= tBe]
    tok = sum(r.get("tokens", 0) for r in tr if "tokens" in r)
    ticks = [r for r in ctl if "spec" in r and "at" in r and t0 <= r["at"] <= tBe]
    duty_ticks = [r for r in ctl if "duty" in r and "at" in r and t0 <= r["at"] <= tBe]
    # harness swap events carry an absolute "t"; controller swap events carry an absolute "at" (their "t" is relative)
    swaps = [e["t"] for e in ev if e.get("event") == "swap"] + [r["at"] for r in ctl if r.get("event") == "swap" and "at" in r]
    switches = [r for r in ctl if r.get("event") == "spec"]
    out_tok = sum(ph[p]["total_output_tokens"] for p in ("phaseA1", "phaseB", "phaseA2"))
    dur = sum(ph[p]["duration"] for p in ("phaseA1", "phaseB", "phaseA2"))
    reasons = {}
    for r in ctl:
        if "reason" in r and t0 <= r.get("at", 0) <= tBe:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    gen_B = [r["gen_tok_s"] for r in tl if t0 <= r["t"] <= tBe]
    return dict(scenario=d.split("/")[-3], strategy=d.split("/")[-2], seed=done["seed"], job=done.get("job"), host=done.get("host"),
                util=round(done["util"], 3), server_s=round(done["server_seconds"]), wall_min=round(done["wall_seconds"] / 60, 1),
                B_thr=ph["phaseB"]["output_throughput"], B_dur_s=ph["phaseB"]["duration"],
                B_p50=ph["phaseB"]["median_tpot_ms"], B_p99=ph["phaseB"]["p99_tpot_ms"], B_req=ph["phaseB"]["completed"],
                A1_p50=ph["phaseA1"]["median_tpot_ms"], A2_p50=ph["phaseA2"]["median_tpot_ms"],
                A1_thr=ph["phaseA1"]["output_throughput"], A2_thr=ph["phaseA2"]["output_throughput"],
                all_thr=out_tok / dur, all_dur_s=dur, out_tok=out_tok,
                acc_first=float(np.mean(acc[:3])) if acc else None, acc_last=float(np.mean(acc[-6:])) if acc else None,
                duty_B=float(np.mean(duty)) if duty else 0.0, train_tok=tok,
                spec_on_B=(sum(1 for r in ticks if r["spec"]) / len(ticks)) if ticks else (0.0 if done["spec"] == 0 else 1.0),
                swaps=len(swaps), swaps_B=sum(1 for t in swaps if t0 <= t <= tBe),
                switches=len(switches), reasons=json.dumps(reasons, ensure_ascii=False),
                tok_gap_s=round(t0 - tB, 1), ctl_duty_frac_B=(sum(1 for r in duty_ticks if r["duty"] > 0) / len(duty_ticks)) if duty_ticks else 0.0,
                gen_cv_B=float(np.std(gen_B[1:-1]) / np.mean(gen_B[1:-1])) if len(gen_B) > 4 else None,
                preempt=sum(r.get("preempt", 0) for r in tl))


def summarize(df, metrics):
    rows = []
    for (sc, st), g in df.groupby(["scenario", "strategy"], sort=False):
        r = dict(scenario=sc, strategy=st, n=len(g))
        for m in metrics:
            v = g[m].astype(float)
            r[f"{m}_mean"] = v.mean()
            r[f"{m}_std"] = v.std(ddof=1) if len(v) > 1 else float("nan")
            r[f"{m}_min"], r[f"{m}_max"] = v.min(), v.max()
        rows.append(r)
    return pd.DataFrame(rows)


def paired(df, metric, ref):
    """paired (same seed) differences strategy - ref in % of ref, mean/std/all-same-sign/t-test p."""
    rows = []
    for sc, g in df.groupby("scenario", sort=False):
        base = g[g.strategy == ref].set_index("seed")[metric]
        for st, h in g.groupby("strategy", sort=False):
            if st == ref:
                continue
            h = h.set_index("seed")
            common = sorted(set(h.index) & set(base.index))
            if not common:
                continue
            d = np.array([(h.loc[s, metric] - base.loc[s]) / base.loc[s] * 100 for s in common])
            n = len(d)
            sd = d.std(ddof=1) if n > 1 else float("nan")
            t = d.mean() / (sd / math.sqrt(n)) if n > 1 and sd > 0 else float("nan")
            p = None
            if n > 1 and sd > 0:
                try:
                    from scipy import stats
                    p = float(2 * stats.t.sf(abs(t), n - 1))
                except Exception:
                    p = None
            rows.append(dict(scenario=sc, strategy=st, ref=ref, metric=metric, n=n, mean_pct=d.mean(), std_pct=sd,
                             min_pct=d.min(), max_pct=d.max(), same_sign=bool((d > 0).all() or (d < 0).all()),
                             t=t, p=p, per_seed=json.dumps([round(x, 2) for x in d])))
    return pd.DataFrame(rows)


def figure(df, sm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scens = list(dict.fromkeys(df.scenario))
    fig, axes = plt.subplots(1, len(scens), figsize=(5.2 * len(scens), 4.2), squeeze=False)
    for ax, sc in zip(axes[0], scens):
        g = sm[sm.scenario == sc]
        base = float(g[g.strategy == "static"].B_thr_mean.iloc[0]) if (g.strategy == "static").any() else None
        xs = list(g.strategy)
        ys = [float(v) for v in g.B_thr_mean]
        es = [0.0 if not np.isfinite(v) else float(v) for v in g.B_thr_std]
        cols = ["#9467bd" if x == "v6" else ("#7f7f7f" if x in ("vanilla", "static", "nospec") else "#1f77b4") for x in xs]
        ax.bar(range(len(xs)), ys, yerr=es, color=cols, capsize=3)
        for i, (x, y) in enumerate(zip(xs, ys)):
            raw = df[(df.scenario == sc) & (df.strategy == x)].B_thr
            ax.plot([i] * len(raw), raw, "k.", ms=4)
            if base:
                ax.text(i, y + (max(ys) * 0.01), f"{y / base - 1:+.0%}", ha="center", fontsize=7)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{sc}: phase B output tok/s (mean ± sd, dots = runs; % vs static)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/abl_B_thr.png", dpi=150)
    plt.close(fig)


def main(runs, out):
    dirs = sorted(d for d in glob.glob(f"{runs}/*/*/r*") if os.path.exists(f"{d}/done.json"))
    df = pd.DataFrame([run_row(d) for d in dirs])
    df["strategy"] = pd.Categorical(df["strategy"], [s for s in ORDER if s in set(df.strategy)] + sorted(set(df.strategy) - set(ORDER)))
    df = df.sort_values(["scenario", "strategy", "seed"]).reset_index(drop=True)
    os.makedirs(out, exist_ok=True)
    df.to_csv(f"{out}/abl_runs.csv", index=False)
    metrics = ["B_thr", "B_p50", "B_p99", "B_dur_s", "A1_p50", "A2_p50", "all_thr", "all_dur_s", "acc_last", "duty_B", "train_tok", "spec_on_B", "swaps",
               "swaps_B", "ctl_duty_frac_B", "gen_cv_B", "tok_gap_s"]
    sm = summarize(df, metrics)
    sm.to_csv(f"{out}/abl_summary.csv", index=False)
    pairs = pd.concat([paired(df, m, ref) for ref in ("static", "vanilla") for m in ("B_thr", "all_thr", "B_p99", "A2_p50")
                       if ref in set(df.strategy)], ignore_index=True)
    pairs.to_csv(f"{out}/abl_pairs.csv", index=False)
    try:
        figure(df, sm, out)
    except Exception as e:  # the figure is a convenience; never block the CSVs
        print("figure skipped:", e)
    pd.set_option("display.width", 250)
    for sc, g in sm.groupby("scenario", sort=False):
        print(f"\n=== {sc}")
        cols = ["strategy", "n", "B_thr_mean", "B_thr_std", "B_p99_mean", "all_thr_mean", "all_thr_std", "A2_p50_mean", "acc_last_mean", "duty_B_mean", "spec_on_B_mean", "swaps_mean"]
        print(g[cols].round(2).to_string(index=False))
    print("\n=== paired vs static (B_thr, % per seed)")
    print(pairs[(pairs.ref == "static") & (pairs.metric == "B_thr")][["scenario", "strategy", "n", "mean_pct", "std_pct", "same_sign", "p", "per_seed"]].round(2).to_string(index=False))
    print("\n=== paired vs vanilla (all_thr, % per seed)")
    print(pairs[(pairs.ref == "vanilla") & (pairs.metric == "all_thr")][["scenario", "strategy", "n", "mean_pct", "std_pct", "same_sign", "p", "per_seed"]].round(2).to_string(index=False))
    return df, sm, pairs


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
