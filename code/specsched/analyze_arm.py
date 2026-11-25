"""Same-start comparison (E28) and per-run reductions shared with the earlier run sets (plan rows 2, 3, 6, 13, 19).

Analysis only. The criteria are the ones written in ideas/2026-11-24_同起点对照与累计回本_预注册.md before any run:
  primary   B-phase output throughput, % difference paired by seed; mean >= +2.0 % with three same-sign seeds = better,
            mean <= -2.0 % with three same-sign seeds = worse, anything else = not separable
  secondary cumulative completed tokens from the client's per-request arrays (origin = first B request, a request's
            tokens spread evenly over its stream chunks), fixed window W, difference to `sw`, deepest loss, break-even

  python analyze_arm.py reduce <runs root> [...]           per-request arrays -> results/_reduced/<set>/<scen>__<arm>__r<k>.npz
  python analyze_arm.py runs   <runs root> <out dir>       per-run table: throughput, lifecycle, latency, B-start state, training
  python analyze_arm.py e28a   <runs/abl_arm> <out dir>    verdicts, A/A, anomaly rules (b)-(e), cumulative curves
  python analyze_arm.py e28b   <runs/abl_arm_open> <out dir>
"""
import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_abl  # noqa: E402  (per-run row of the frozen analysis: B_thr, A1_thr, all_thr, ...)

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(ROOT, "results", "_reduced")
GRID = 1.0                                   # seconds; cumulative curves and in-flight counts
ARM_ORDER = ["switch", "sw", "round800", "budget", "duty05", "novalue", "full"]
SCEN_ORDER = ["m1_code", "m2_code", "m1_mixcc", "m1_zh", "m1_mixzc"]
SCEN_NAME = {"m1_code": "8B code", "m2_code": "MoE code", "m1_mixcc": "8B code+chat", "m1_zh": "8B zh",
             "m1_mixzc": "8B zh+chat"}
TRAINABLE = ["m1_code", "m2_code", "m1_mixcc"]
CONTRASTS = [("sw", "switch"), ("full", "sw"), ("full", "novalue"), ("full", "round800"), ("full", "duty05"),
             ("full", "budget"),                                   # the six preregistered contrasts
             ("round800", "sw"), ("budget", "sw"), ("duty05", "sw"), ("novalue", "sw"), ("full", "switch")]  # descriptive
OPEN_W = {"m1_code": 230.0, "m2_code": 250.0, "m1_mixcc": 310.0}   # preregistered windows of E28b
MARKS = [30, 60, 120, 180, 240]


def jl(path):
    return analyze_abl.jl(path)


def run_dirs(root):
    return sorted(d for d in glob.glob(f"{root}/*/*/r*") if os.path.exists(f"{d}/done.json") and "_flagged" not in d)


def key_of(d):
    p = os.path.abspath(d).rstrip("/").split("/")
    return p[-4], p[-3], p[-2], p[-1]          # set, scenario, arm, r<seed>


# ----------------------------------------------------------------------------- per-request arrays

def reduce_run(d, force=False):
    """Phase-B per-request arrays reduced to what the analysis needs; cached as npz."""
    st, scen, arm, rep = key_of(d)
    path = os.path.join(CACHE, st, f"{scen}__{arm}__{rep}.npz")
    src = f"{d}/phaseB.json"
    if os.path.exists(path) and not force and os.path.getmtime(path) >= os.path.getmtime(src):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    b = json.load(open(src))
    start = np.asarray(b["start_times"], dtype=float)
    origin = start.min()
    start = start - origin
    ttft = np.asarray(b["ttfts"], dtype=float)
    out = np.asarray(b["output_lens"], dtype=float)
    nch = np.array([len(x) + 1 for x in b["itls"]], dtype=float)          # first chunk at ttft, then one per itl
    e2e = ttft + np.array([float(np.sum(x)) for x in b["itls"]])
    ok = np.array([not e for e in b["errors"]]) & (out > 0)
    # chunk times of every successful request; each chunk carries out_len / n_chunks tokens
    times, weights = [], []
    for i in np.nonzero(ok)[0]:
        it = b["itls"][i]
        t = start[i] + ttft[i] + np.concatenate([[0.0], np.cumsum(it)]) if len(it) else np.array([start[i] + ttft[i]])
        times.append(t)
        weights.append(np.full(len(t), out[i] / len(t)))
    times = np.concatenate(times) if times else np.zeros(0)
    weights = np.concatenate(weights) if weights else np.zeros(0)
    end = start + e2e
    n = int(np.ceil((end[ok].max() if ok.any() else 0.0) / GRID)) + 2
    edges = np.arange(0, n + 1) * GRID
    hist, _ = np.histogram(times, bins=edges, weights=weights)
    cum = np.concatenate([[0.0], np.cumsum(hist)])                         # cum[k] = tokens completed by k*GRID
    grid = edges
    started = np.searchsorted(np.sort(start), grid, side="right")
    ended = np.searchsorted(np.sort(end[ok]), grid, side="right")
    red = dict(start=start, ttft=ttft, e2e=e2e, out=out, nch=nch, ok=ok, cum=cum, inflight=(started - ended).astype(float),
               scalars=np.array([b["duration"], b["output_throughput"], b["completed"], b["failed"],
                                 b["total_output_tokens"], b.get("max_concurrent_requests") or 0,
                                 b["mean_tpot_ms"], b["mean_ttft_ms"], b["mean_e2el_ms"]], dtype=float))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **red)
    return red


def latency(red):
    """Request-level latency statistics from the arrays (ms)."""
    ok = red["ok"].astype(bool)
    out, ttft, e2e = red["out"][ok], red["ttft"][ok], red["e2e"][ok]
    m = out > 1
    tpot = (e2e[m] - ttft[m]) / (out[m] - 1)
    r = {}
    for name, x in (("tpot", tpot), ("ttft", ttft), ("e2e", e2e)):
        x = x * 1e3
        r[f"{name}_mean"] = float(x.mean())
        for q in (50, 95, 99):
            r[f"{name}_p{q}"] = float(np.percentile(x, q))
    return r


def cum_at(red, t):
    """Tokens completed by time t (seconds after the first B request), linear between grid points."""
    cum = red["cum"]
    x = np.clip(np.asarray(t, dtype=float) / GRID, 0, len(cum) - 1)
    return np.interp(x, np.arange(len(cum)), cum)


def last_busy(red, level=230):
    """Last time the client had at least `level` requests in flight."""
    idx = np.nonzero(red["inflight"] >= level)[0]
    return float(idx.max() * GRID) if len(idx) else 0.0


# ----------------------------------------------------------------------------- controller / trainer logs

def run_extra(d):
    """B-start state, training activity and switch state of one run, from the controller and trainer logs."""
    ev = jl(f"{d}/events.jsonl")
    done = json.load(open(f"{d}/done.json"))
    dom_b = done.get("dom_b")
    tB = next((e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B"), None)
    tBe = next((e["t"] for e in ev if e.get("event") == "phase_end"), None)
    tEnd = next((e["t"] for e in ev if e.get("event") == "done"), None)
    ctl = jl(f"{d}/controller.jsonl")
    tr = jl(f"{d}/trainer.jsonl")
    r = dict(dom_b=dom_b, tB=tB, tBe=tBe)
    for name in ("trainer", "controller"):       # a process that died leaves its Python traceback in <name>.out
        try:
            txt = open(f"{d}/{name}.out", errors="replace").read()
            r[f"{name}_traceback"] = "Traceback" in txt
            if name == "trainer":
                r["trainer_oom"] = "OutOfMemoryError" in txt
        except FileNotFoundError:
            r[f"{name}_traceback"] = None
    if tB is None or tBe is None:
        return r
    ticks = [c for c in ctl if "event" not in c and "at" in c]
    evs = [c for c in ctl if "event" in c and "at" in c]
    # the label period as the controller saw it
    tlab = next((c["at"] for c in evs if c["event"] == "domain" and c.get("dom") == dom_b), tB)
    tlab_end = next((c["at"] for c in evs if c["event"] == "domain" and c["at"] > tlab), tBe)
    r["consumed_at_label"] = next((c.get("consumed") for c in evs if c["event"] == "domain" and c.get("dom") == dom_b), None)
    # first served B traffic (server side): first tick after the label change that generated tokens under load
    first = next((c for c in ticks if c["at"] > tlab + 2 and (c.get("gen_tok_s") or 0) > 0 and (c.get("C") or 0) > 1), None)
    t0 = (first["at"] - 1.0) if first else tB
    r["t0_srv"] = t0
    r["gap_label_to_traffic_s"] = t0 - tlab
    swaps = [c for c in evs if c["event"] == "swap"]
    evals = {c["head"]: c.get("realized") for c in evs if c["event"] == "swap_eval"}
    r["heads_before_B"] = sum(1 for c in swaps if c["at"] < tlab)
    r["head_at_B_start"] = next((c["head"] for c in reversed(swaps) if c["at"] < tlab), "stock")
    inB = [c for c in swaps if tlab <= c["at"] < tlab_end]
    r["swaps_in_B"] = len(inB)
    r["swap_times_B"] = ";".join(f"{c['head']}@{c['at'] - t0:.0f}s" for c in inB)
    r["swap_realized_B"] = ";".join(f"{c['head']}:{evals.get(c['head'])}" for c in inB)
    r["swaps_after_B"] = sum(1 for c in swaps if c["at"] >= tlab_end)
    r["exports_in_B"] = sum(1 for c in evs if c["event"] == "export_request" and tlab <= c["at"] < tlab_end)
    r["swap_errors"] = sum(1 for c in evs if c["event"] in ("swap_error", "spec_error"))
    sat = [c for c in ticks if tlab <= c["at"] < tlab_end and c.get("saturated")]
    r["sat_ticks_B"] = len(sat)
    acc = [c["acc_meas"] for c in sat if c.get("acc_meas") is not None]
    r["acc_B_first5"] = float(np.mean(acc[:5])) if acc else None
    late = [c["acc_meas"] for c in sat if c.get("acc_meas") is not None and c["at"] >= sat[-1]["at"] - 30] if sat else []
    r["acc_B_last30s"] = float(np.mean(late)) if late else None
    r["spec_on_frac_B"] = float(np.mean([bool(c.get("spec")) for c in sat])) if sat else None
    r["spec_toggles_B"] = sum(1 for c in evs if c["event"] == "spec" and tlab <= c["at"] < tlab_end)
    r["duty_ticks_B"] = json.dumps({str(k): v for k, v in sorted(collections.Counter(c["duty"] for c in sat).items())})
    r["reasons_B"] = json.dumps(dict(collections.Counter(c["reason"] for c in sat).most_common()))
    r["ctl_ticks_after_B"] = sum(1 for c in ticks if c["at"] >= tBe)
    r["ctl_alive"] = bool(ticks) and (tEnd is None or ticks[-1]["at"] >= tEnd - 10)
    # server-side token integral over the label period (cross-check of the client's total)
    r["srv_tok_B"] = float(sum((c.get("gen_tok_s") or 0) * dt for c, dt in
                               zip(ticks[1:], np.diff([c["at"] for c in ticks])) if tlab <= c["at"] < tlab_end + 2))
    # trainer windows inside the label period
    win = [w for w in tr if "busy_s" in w and tlab <= w["t"] < tlab_end]
    r["trainer_stop_event"] = any(w.get("event") == "stop" for w in tr)
    r["train_tok_B"] = float(sum(w["tokens"] for w in win))
    r["train_busy_s_B"] = float(sum(w["busy_s"] for w in win))
    r["train_req_B"] = float(win[-1]["consumed"] - win[0]["consumed"]) if win else 0.0
    allw = [w for w in tr if "busy_s" in w]
    inc = {id(w): w["consumed"] - p["consumed"] for p, w in zip(allw, allw[1:])}     # requests learned in each window
    on = [w for w in win if w["duty"] > 0 and w["t"] >= t0]
    r["train_windows_on"] = len(on)
    # anomaly rule (b): consumption rate over the windows at duty >= 0.5 that lie inside saturated B
    sat_t = np.array([c["at"] for c in sat])
    hi = [w for w in win if w["duty"] >= 0.5 and id(w) in inc and len(sat_t) and np.min(np.abs(sat_t - w["t"])) <= 1.5]
    r["train_sat_windows"] = len(hi)
    r["train_req_s_sat"] = float(sum(inc[id(w)] for w in hi) / max(sum(w["win_s"] for w in hi), 1e-9)) if hi else None
    if on:
        secs = sum(w["win_s"] for w in on)
        r["duty_cfg_on"] = float(np.mean([w["duty"] for w in on]))
        r["busy_frac_on"] = float(sum(w["busy_s"] for w in on) / secs)
        r["train_tok_s_on"] = float(sum(w["tokens"] for w in on) / secs)
        r["train_tok_per_busy_s"] = float(sum(w["tokens"] for w in on) / max(sum(w["busy_s"] for w in on), 1e-9))
        r["train_req_s_on"] = float(sum(inc.get(id(w), 0) for w in on) / secs)      # requests learned per second of training window
        r["train_first_s"] = float(on[0]["t"] - t0)
        r["train_last_s"] = float(on[-1]["t"] - t0)
        # serving throughput while the trainer is on / off (saturated ticks only; server counter)
        on_t = np.array([w["t"] for w in on])
        srv_on, srv_off = [], []
        for c in sat:
            if c["at"] < t0 + 3 or (c.get("gen_tok_s") or 0) <= 0:
                continue
            (srv_on if np.any(np.abs(on_t - c["at"]) <= 1.0) else srv_off).append(c["gen_tok_s"])
        r["srv_tok_s_train_on"] = float(np.mean(srv_on)) if srv_on else None
        r["srv_tok_s_train_off"] = float(np.mean(srv_off)) if srv_off else None
    return r


def run_table(root, with_arrays=True, verbose=True):
    rows = []
    for d in run_dirs(root):
        r = analyze_abl.run_row(d)
        r.update(run_extra(d))
        if with_arrays and os.path.exists(f"{d}/phaseB.json"):
            red = reduce_run(d)
            r.update(latency(red))
            sc = red["scalars"]
            r.update(B_failed=int(sc[3]), B_max_inflight=int(sc[5]), B_last_busy_s=last_busy(red),
                     B_tok_client=float(red["cum"][-1]), B_tok_bench=float(sc[4]),
                     tpot_mean_bench=float(sc[6]), ttft_mean_bench=float(sc[7]), e2e_mean_bench=float(sc[8]))
        r["dir"] = d
        rows.append(r)
        if verbose:
            print(f"  {r['scenario']}/{r['strategy']}/r{r['seed']}  B {r['B_thr']:.0f} tok/s", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- paired verdicts

def verdict(diffs, thr=2.0):
    diffs = [float(x) for x in diffs]
    m = float(np.mean(diffs))
    same = all(x > 0 for x in diffs) or all(x < 0 for x in diffs)
    if len(diffs) >= 3 and same and m >= thr:
        return "better"
    if len(diffs) >= 3 and same and m <= -thr:
        return "worse"
    return "not separable"


def paired(df, metric, a, b, scen):
    x = df[(df.scenario == scen) & (df.strategy == a)].set_index("seed")[metric]
    y = df[(df.scenario == scen) & (df.strategy == b)].set_index("seed")[metric]
    seeds = sorted(set(x.index) & set(y.index))
    if not seeds:
        return None
    d = [(x[s] / y[s] - 1) * 100 for s in seeds]
    return dict(scenario=scen, a=a, b=b, metric=metric, n=len(seeds), mean_pct=float(np.mean(d)),
                sd_pct=float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"), min_pct=float(min(d)), max_pct=float(max(d)),
                per_seed=";".join(f"{v:+.2f}" for v in d), verdict=verdict(d),
                a_mean=float(np.mean([x[s] for s in seeds])), b_mean=float(np.mean([y[s] for s in seeds])))


def contrasts(df, metric="B_thr", pairs=CONTRASTS):
    rows = []
    for scen in [s for s in SCEN_ORDER if s in set(df.scenario)]:
        have = set(df[df.scenario == scen].strategy)
        for a, b in pairs:
            if a in have and b in have:
                rows.append(paired(df, metric, a, b, scen))
    return pd.DataFrame([r for r in rows if r])


def conclusions(pairs):
    """The four pre-written conclusions, evaluated mechanically."""
    v = {(r.scenario, r.a, r.b): r.verdict for r in pairs.itertuples()}
    out = []
    simple = ["sw", "round800", "duty05"]
    worse = [(s, b) for b in simple for s in SCEN_ORDER if v.get((s, "full", b)) == "worse"]
    better = {b: [s for s in SCEN_ORDER if v.get((s, "full", b)) == "better"] for b in simple}
    A = not worse and all(len(better[b]) >= 2 for b in simple)
    out.append(dict(item="A", holds=A, detail=f"worse: {worse or 'none'}; better per baseline: {better}"))
    bud = {s: v.get((s, "full", "budget")) for s in TRAINABLE}
    B = A and sum(x == "better" for x in bud.values()) >= 2 and "worse" not in bud.values()
    out.append(dict(item="B", holds=B, detail=f"full vs budget: {bud}"))
    C = all(x in ("not separable", "worse") for x in bud.values()) and None not in bud.values()
    out.append(dict(item="C", holds=C, detail="budget not separable from, or better than, full on all three trainable scenarios"
                    if C else f"full vs budget: {bud}"))
    nov = {s: v.get((s, "full", "novalue")) for s in TRAINABLE}
    D = any(x == "better" for x in nov.values()) and "worse" not in nov.values()
    out.append(dict(item="D", holds=D, detail=f"full vs novalue: {nov}"))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- cumulative curves

def curves(df, window, ref="sw"):
    """Per scenario: cumulative tokens of every arm on a 1-s grid, difference to `ref`, deepest loss, break-even.
    window: dict scenario -> W (seconds)."""
    long, summ = [], []
    for scen in [s for s in SCEN_ORDER if s in set(df.scenario)]:
        W = window[scen]
        t = np.arange(0, int(W) + 1, 1.0)
        sub = df[df.scenario == scen]
        cum = {(r.strategy, r.seed): cum_at(reduce_run(r.dir), t) for r in sub.itertuples()}
        arms = [a for a in ARM_ORDER if a in set(sub.strategy)]
        seeds = sorted(set(sub.seed))
        for a in arms:
            ss = [s for s in seeds if (a, s) in cum and (ref, s) in cum]
            if not ss:
                continue
            delta = np.array([cum[(a, s)] - cum[(ref, s)] for s in ss])           # seeds x time
            base = np.array([cum[(ref, s)] for s in ss])
            mean_d = delta.mean(axis=0)
            for k, tt in enumerate(t):
                long.append(dict(scenario=scen, arm=a, t=tt, cum_mean=float(np.mean([cum[(a, s)][k] for s in ss])),
                                 delta_mean=float(mean_d[k]), delta_min=float(delta[:, k].min()), delta_max=float(delta[:, k].max())))

            def breakeven(dl):
                okk = dl >= 0
                tail = np.flip(np.cumprod(np.flip(okk)))                         # 1 where >= 0 from here to W
                idx = [i for i in range(len(t)) if tail[i] and t[i] >= 10]
                return float(t[idx[0]]) if idx else float("inf")
            row = dict(scenario=scen, arm=a, ref=ref, W=W, n=len(ss),
                       cum_W=float(np.mean([cum[(a, s)][-1] for s in ss])), ref_cum_W=float(base[:, -1].mean()),
                       delta_W=float(mean_d[-1]), delta_W_pct=float(np.mean(delta[:, -1] / base[:, -1]) * 100),
                       delta_W_per_seed_pct=";".join(f"{x:+.2f}" for x in delta[:, -1] / base[:, -1] * 100),
                       verdict_W=verdict(list(delta[:, -1] / base[:, -1] * 100)) if a != ref else "",
                       deepest=float(mean_d.min()), deepest_t=float(t[int(mean_d.argmin())]),
                       deepest_s_of_ref=float(mean_d.min() / (base[:, -1].mean() / W)),
                       breakeven_s=breakeven(mean_d), breakeven_per_seed=";".join(
                           ("%.0f" % breakeven(x)) if np.isfinite(breakeven(x)) else ">W" for x in delta))
            for mk in MARKS:
                if mk <= W:
                    row[f"delta_{mk}s"] = float(mean_d[int(mk)])
            summ.append(row)
    return pd.DataFrame(long), pd.DataFrame(summ)


def closed_window(df):
    """W_a of the preregistration: per scenario, min over arms and seeds of the last time >= 230 requests were in
    flight, floored to 10 s."""
    return {scen: float(np.floor(g.B_last_busy_s.min() / 10.0) * 10.0) for scen, g in df.groupby("scenario")}


# ----------------------------------------------------------------------------- anomaly rules (b)-(e)

def anomaly_rules(df):
    """Rules (b)-(e) of the preregistration, evaluated once. Rule (a) is scan_anomaly.py."""
    flags = []
    for scen, g in df.groupby("scenario"):
        tr = g[(g.train_sat_windows >= 20) & g.train_req_s_sat.notna()]
        med = tr.train_req_s_sat.median() if len(tr) else None
        for r in g.itertuples():
            why = []
            if med and r.train_sat_windows >= 20 and pd.notna(r.train_req_s_sat) and r.train_req_s_sat < 0.6 * med:
                why.append(f"(b) trainer {r.train_req_s_sat:.1f} req/s < 0.6 x median {med:.1f}")
            # (c) as recorded in the change log: no trainer stop event, a traceback in trainer.out or controller.out,
            # controller ticks ending before the harness's done, or failed requests in B (swap errors are reported, not a trigger)
            dead = (not r.ctl_alive) or (not r.trainer_stop_event) or bool(r.trainer_traceback) or bool(r.controller_traceback)
            if dead or r.B_failed > 0:
                why.append(f"(c) ctl_alive={r.ctl_alive} trainer_stop={r.trainer_stop_event} trainer_traceback={r.trainer_traceback} "
                           f"controller_traceback={r.controller_traceback} failed={r.B_failed}")
            if r.strategy != "switch":
                want = 0 if scen.startswith("m2") else 1
                if r.heads_before_B != want:
                    why.append(f"(e) heads before B = {r.heads_before_B}, expected {want}")
            if why:
                flags.append(dict(run=f"{scen}.{r.strategy}.r{r.seed}", host=r.host, why=" | ".join(why)))
    # (d) exact A/A: MoE switch vs sw
    g = df[df.scenario == "m2_code"]
    for s in sorted(set(g.seed)):
        a, b = g[(g.strategy == "switch") & (g.seed == s)], g[(g.strategy == "sw") & (g.seed == s)]
        if len(a) and len(b):
            dd = (b.B_thr.iloc[0] / a.B_thr.iloc[0] - 1) * 100
            if abs(dd) > 3.0:
                for arm, x in (("switch", a), ("sw", b)):
                    flags.append(dict(run=f"m2_code.{arm}.r{s}", host=x.host.iloc[0], why=f"(d) exact A/A pair differs {dd:+.2f}%"))
    return pd.DataFrame(flags, columns=["run", "host", "why"])


# ----------------------------------------------------------------------------- entry points

def tab(df, cols, fmt=None):
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400):
        print(df[cols].to_string(index=False, float_format=fmt or (lambda x: f"{x:.2f}")))


def main_runs(root, out):
    os.makedirs(out, exist_ok=True)
    df = run_table(root)
    df.to_csv(f"{out}/runs_full.csv", index=False)
    print(f"{len(df)} runs -> {out}/runs_full.csv")
    return df


def main_e28(root, out, open_loop=False):
    os.makedirs(out, exist_ok=True)
    df = run_table(root, verbose=False)
    df.to_csv(f"{out}/runs_full.csv", index=False)
    print(f"{len(df)} runs")
    if open_loop:
        window = {s: OPEN_W[s] for s in set(df.scenario)}
    else:
        window = closed_window(df)
        pr = contrasts(df)
        pr.to_csv(f"{out}/pairs_B_thr.csv", index=False)
        print("\n== primary metric: B-phase output throughput, % paired by seed")
        tab(pr, ["scenario", "a", "b", "n", "a_mean", "b_mean", "mean_pct", "sd_pct", "min_pct", "max_pct", "per_seed", "verdict"])
        cc = conclusions(pr)
        cc.to_csv(f"{out}/conclusions.csv", index=False)
        print("\n== pre-written conclusions")
        tab(cc, ["item", "holds", "detail"])
        for m in ("all_thr", "A1_thr", "A2_thr"):
            contrasts(df, m).to_csv(f"{out}/pairs_{m}.csv", index=False)
    print("\n== fixed windows (s):", window)
    long, summ = curves(df, window)
    long.to_csv(f"{out}/cum_curves.csv", index=False)
    summ.to_csv(f"{out}/cum_summary.csv", index=False)
    print("\n== cumulative tokens against sw inside the fixed window")
    tab(summ, [c for c in ["scenario", "arm", "W", "n", "cum_W", "delta_W", "delta_W_pct", "delta_W_per_seed_pct", "verdict_W",
                           "deepest", "deepest_t", "deepest_s_of_ref", "breakeven_s", "breakeven_per_seed"] if c in summ])
    fl = anomaly_rules(df)
    fl.to_csv(f"{out}/anomaly_rules.csv", index=False)
    print(f"\n== anomaly rules (b)-(e): {len(fl)} flagged")
    if len(fl):
        tab(fl, ["run", "host", "why"])
    return df


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "reduce":
        for root in sys.argv[2:]:
            for d in run_dirs(root):
                if os.path.exists(f"{d}/phaseB.json"):
                    reduce_run(d)
                    print("reduced", d, flush=True)
    elif cmd == "runs":
        main_runs(sys.argv[2], sys.argv[3])
    elif cmd == "e28a":
        main_e28(sys.argv[2], sys.argv[3])
    elif cmd == "e28b":
        main_e28(sys.argv[2], sys.argv[3], open_loop=True)
    else:
        raise SystemExit(__doc__)
