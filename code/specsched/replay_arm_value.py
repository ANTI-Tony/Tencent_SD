"""Plan rows 14 and 15: replay of the payback rule on logged saturated decisions. Analysis only, no new runs.

Row 14  predicted n, tau(d), V(d) recomputed from the tick log of every Full run of E28a, next to what happened
        (requests learned, swap times, acceptance, cumulative tokens against `sw` of the same seed).
Row 15  the same logged decisions with the horizon floor at 30, 60, 180 s instead of the deployed 90 s:
        H = clip(regime age, floor, 600). Counts train / no-train flips and where in the regime they fall.
Definitions: ideas/2026-11-24_同起点对照与累计回本_预注册.md (rows 14 and 15, written before this computation).

The rule is the one of the frozen controller.py (rule 3); `invert` and the value expression are copied from
replay_value_baseline.py, which cannot be imported (it runs its replay at import time). n and tau are not logged,
so they are recomputed from the logged tick fields; acc_est is logged with 3 decimals, which is why a replayed value
can differ from the logged one by a few hundred tokens.

  python replay_arm_value.py            -> results/abl_arm/value_replay/
"""
import collections
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sched"))
from sim import Model  # noqa: E402
import analyze_arm as aa  # noqa: E402

ROOT = aa.ROOT
OUT = os.path.join(ROOT, "results", "abl_arm", "value_replay")
MODELS = {k: Model(os.path.join(HERE, "sched", f)) for k, f in
          (("m1", "model.json"), ("m2", "model_m2.json"), ("mix", "model_mix.json"))}
K, FRAC, DUT = 10.0, 0.8, [0.1, 0.25, 0.5]          # controller defaults: --K, --target-frac, duties <= sat_max_duty
H_MIN, H_MAX = 90.0, 600.0                           # --horizon-min, --horizon-max
FLOORS = [30.0, 60.0, 90.0, 180.0]
RULE3 = ("saturated_value", "saturated_no_value")
BUCKETS = [(0, 30), (30, 60), (60, 120), (120, 1e9)]  # seconds since the first served B traffic


def mkey(scen):
    return "mix" if "mix" in scen else ("m2" if scen.startswith("m2") else "m1")


def curve(m, dom):
    return sorted((float(k), v) for k, v in m.dom[dom]["acc_curve"].items())


def invert(m, dom, acc):
    pts = curve(m, dom)
    if acc <= pts[0][1]:
        return 0.0
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 <= acc <= a1 and a1 > a0:
            return n0 + (n1 - n0) * (acc - a0) / (a1 - a0)
    return pts[-1][0]


def acc_lin(m, dom, n):
    """The controller's own curve reading (linear between calibration points), the inverse of `invert`."""
    pts = curve(m, dom)
    return float(np.interp(n, [p[0] for p in pts], [p[1] for p in pts]))


def plan(m, r, H):
    """Rule 3 of the frozen controller for one logged tick with horizon H. Returns every intermediate quantity."""
    dom, C, a = r["dom"], r["C"], r["acc_est"]
    m.cur_dom = dom if dom in m.fr else None
    ceiling = max(m.dom[dom]["acc_curve"].values())
    a_t = min(ceiling, a + FRAC * (ceiling - a))
    tpr, opr = m.tok_per_req(dom), m.out_per_req(dom)
    n_now = max(invert(m, dom, a), 0.0)
    n_tgt = invert(m, dom, a_t)
    need = max(0.0, n_tgt - n_now) * tpr
    on, off = m.thr_at(C, True, a, 0.0), m.thr_at(C, False, a, 0.0)
    now = on if r["spec"] else off
    tgt = max(m.thr_at(C, True, a_t, 0.0), m.thr_at(C, False, a_t, 0.0))
    feed = r["buffered_tokens"] / K + r["gen_tok_s"] * tpr / max(opr, 1.0)
    kap = m.kappa(C)
    p = dict(a=a, ceiling=ceiling, a_target=a_t, n_now=n_now, n_target=n_tgt, need_req=need / tpr, need_tok=need,
             feed_tok_s=feed, thr_now=now, thr_target=tgt, gain_pct=(tgt / now - 1) * 100 if now > 0 else float("nan"),
             kappa=kap, H=H)
    best_v, duty = 0.0, 0.0
    for dd in DUT:
        speed = min(m.rate * dd, feed)
        if speed <= 0 or need <= 0:
            continue
        T = need / speed
        busy = min(dd, speed / m.rate)
        loss = now * kap * busy * T
        v = (tgt - now) * max(0.0, H - T) - loss
        p[f"tau_{dd}"] = T
        p[f"busy_{dd}"] = busy
        p[f"loss_{dd}"] = loss
        p[f"V_{dd}"] = v
        p[f"tstar_{dd}"] = T + loss / (tgt - now) if tgt > now else float("inf")   # V > 0  <=>  H > tstar
        if v > best_v:
            best_v, duty = v, dd
    p["duty"], p["V"] = duty, best_v
    ts = [p[f"tstar_{dd}"] for dd in DUT if f"tstar_{dd}" in p]
    p["tstar_min"] = min(ts) if ts else float("inf")
    return p


def load_ticks(d):
    """Controller ticks in file order with the regime clock rebuilt (controller.py: the clock restarts at a label
    change and starts running at the first tick that sees traffic)."""
    t_dom, out = None, []
    for r in aa.jl(f"{d}/controller.jsonl"):
        if r.get("event") == "domain":
            t_dom = None
            continue
        if "event" in r or "reason" not in r:
            continue
        traffic = (r.get("gen_tok_s") or 0) > 0 or (r.get("C") or 0) > 1 or (r.get("waiting") or 0) > 0
        if t_dom is None and traffic:
            t_dom = r["t"]
        out.append(dict(r, age=(r["t"] - t_dom) if t_dom is not None else 0.0))
    return out


def replay_run(d, scen):
    m = MODELS[mkey(scen)]
    rows = []
    for r in load_ticks(d):
        if r["reason"] not in RULE3:
            continue
        p = plan(m, r, r["H"])
        row = dict(t=r["t"], at=r["at"], age=r["age"], dom=r["dom"], C=r["C"], spec=bool(r["spec"]), acc_est=r["acc_est"],
                   acc_meas=r.get("acc_meas"), gen_tok_s=r["gen_tok_s"], consumed=r["consumed"],
                   buffered_tokens=r["buffered_tokens"], H_log=r["H"], duty_log=r["duty"], value_log=r["value"],
                   duty_replay=p["duty"], value_replay=p["V"])
        row.update({k: v for k, v in p.items() if k not in ("duty", "V", "H", "a")})
        for f in FLOORS:
            Hf = min(H_MAX, max(f, r["age"]))
            pf = plan(m, r, Hf)
            row[f"H_f{f:.0f}"], row[f"duty_f{f:.0f}"], row[f"V_f{f:.0f}"] = Hf, pf["duty"], pf["V"]
        rows.append(row)
    return rows


def episodes(ticks_all, rule3):
    """Maximal runs of consecutive controller ticks that train under rule 3 (reason saturated_value)."""
    by_t = {r["t"]: r for r in rule3}
    eps, cur = [], []
    for r in ticks_all:
        if r["reason"] == "saturated_value" and r["duty"] > 0:
            cur.append(r)
        elif cur:
            eps.append(cur)
            cur = []
    if cur:
        eps.append(cur)
    return [[by_t[r["t"]] for r in e] for e in eps]


def consumed_after(ticks_all, at, wait=5.0, stop_at=None):
    """Requests consumed once the trainer has drained after a stop: the tick `wait` s later (or the next episode)."""
    lim = at + wait if stop_at is None else min(at + wait, stop_at)
    later = [r for r in ticks_all if r["at"] >= lim]
    return later[0]["consumed"] if later else ticks_all[-1]["consumed"]


def run_row14(d, scen, seed, rule3, sw_dir, W):
    """One Full run: the plan at the first decision and at the end of the first training episode, next to the actuals."""
    m = MODELS[mkey(scen)]
    ex = aa.run_extra(d)
    t0, dom = ex["t0_srv"], ex["dom_b"]
    ticks_all = load_ticks(d)
    out = dict(scenario=scen, seed=seed, dom=dom, W=W, rule3_ticks=len(rule3),
               train_ticks=sum(1 for r in rule3 if r["duty_log"] > 0), reasons_B=ex.get("reasons_B"),
               a_first5=ex.get("acc_B_first5"), a_last30=ex.get("acc_B_last30s"), learned_B=ex.get("train_req_B"),
               swaps_B=ex.get("swaps_in_B"), swap_times_B=ex.get("swap_times_B"), head_at_B=ex.get("head_at_B_start"),
               consumed_at_label=ex.get("consumed_at_label"))
    ceiling = max(m.dom[dom]["acc_curve"].values())
    out["ceiling"] = ceiling
    # the head that served last in B: how many B requests it had learned, and what the curve says for it
    heads = [int(s.split("@")[0][1:]) for s in (ex.get("swap_times_B") or "").split(";") if s]
    if heads and ex.get("consumed_at_label") is not None and ex.get("acc_B_first5") is not None:
        in_head = heads[-1] - ex["consumed_at_label"]
        n0 = invert(m, dom, ex["acc_B_first5"])
        out.update(req_in_last_head=in_head, req_unserved=ex["train_req_B"] - in_head, n_at_B_start=n0,
                   a_curve_last_head=acc_lin(m, dom, n0 + in_head))
    if not rule3:
        return out, []
    full, sw = aa.reduce_run(d), aa.reduce_run(sw_dir)

    def delta(t):
        t = min(float(t), W)
        return float(aa.cum_at(full, t) - aa.cum_at(sw, t))
    eps = episodes([r for r in ticks_all if r["at"] >= t0 - 2], rule3)
    ep_rows = []
    for i, e in enumerate(eps):
        nxt = eps[i + 1][0]["at"] if i + 1 < len(eps) else None
        first, last = e[0], e[-1]
        c0 = first["consumed"]
        c1 = consumed_after(ticks_all, last["at"], stop_at=nxt)
        s = first["at"] - t0
        ep_rows.append(dict(scenario=scen, seed=seed, episode=i + 1, start_s=s, end_s=last["at"] - t0, ticks=len(e),
                            duty=collections.Counter(r["duty_log"] for r in e).most_common(1)[0][0],
                            H_start=first["H_log"], a_est_start=first["acc_est"], a_target_start=first["a_target"],
                            n_pred_start=first["need_req"], tau_pred_start=first.get(f"tau_{first['duty_log']}"),
                            V_pred_start=first["value_replay"], tstar_start=first.get(f"tstar_{first['duty_log']}"),
                            n_pred_end=last["need_req"], V_pred_end=last["value_replay"],
                            learned=c1 - c0, duration_s=last["at"] - first["at"] + 1.0,
                            delta_over_H=delta(s + first["H_log"]) - delta(max(s, 0.0))))
    f = rule3[0]                                              # first saturated decision of the regime
    s = f["at"] - t0
    out.update(first_s=s, first_H=f["H_log"], first_a_est=f["acc_est"], first_a_meas=f["acc_meas"],
               first_a_target=f["a_target"], first_n=f["need_req"], first_feed=f["feed_tok_s"],
               first_gain_pct=f["gain_pct"], first_duty=f["duty_log"], first_V=f["value_replay"],
               first_tstar=f["tstar_min"], first_actual_delta_H=delta(s + f["H_log"]) - delta(max(s, 0.0)))
    for dd in DUT:
        out[f"first_tau_{dd}"], out[f"first_V_{dd}"] = f.get(f"tau_{dd}"), f.get(f"V_{dd}")
        out[f"first_loss_{dd}"] = f.get(f"loss_{dd}")
    if eps:
        e = eps[0]
        last = e[-1]
        s2 = last["at"] - t0
        out.update(ep1_start_s=e[0]["at"] - t0, ep1_dur_s=last["at"] - e[0]["at"] + 1.0, ep1_ticks=len(e),
                   ep1_learned=ep_rows[0]["learned"], last_s=s2, last_H=last["H_log"], last_a_est=last["acc_est"],
                   last_a_meas=last["acc_meas"], last_a_target=last["a_target"], last_n=last["need_req"],
                   last_gain_pct=last["gain_pct"], last_duty=last["duty_log"], last_V=last["value_replay"],
                   last_tau=last.get(f"tau_{last['duty_log']}"), last_loss=last.get(f"loss_{last['duty_log']}"),
                   last_tstar=last.get(f"tstar_{last['duty_log']}"),
                   last_actual_delta_H=delta(s2 + last["H_log"]) - delta(s2), n_episodes=len(eps),
                   learned_in_episodes=sum(r["learned"] for r in ep_rows))
    for t in (30, 60, 90, 120, 180):
        out[f"delta_{t}s"] = delta(t)
    out["delta_W"] = delta(W)
    out["sw_cum_W"] = float(aa.cum_at(sw, W))
    return out, ep_rows


def flips(rows, t0):
    """Row 15: decisions under each floor against the deployed floor (90 s), by time since the first B traffic."""
    out = collections.Counter()
    for r in rows:
        s = r["at"] - t0
        b = next((f"{lo}-{hi if hi < 1e8 else 'end'}s" for lo, hi in BUCKETS if lo <= s < hi), "before")
        base = r["duty_f90"]
        out["ticks"] += 1
        out[f"ticks@{b}"] += 1
        out["train_f90"] += int(base > 0)
        for f in FLOORS:
            if f == 90.0:
                continue
            k = f"f{f:.0f}"
            d = r[f"duty_{k}"]
            out[f"train_{k}"] += int(d > 0)
            if (d > 0) != (base > 0):
                kind = "to_train" if d > 0 else "to_notrain"
                out[f"{kind}_{k}"] += 1
                out[f"{kind}_{k}@{b}"] += 1
            elif d != base:
                out[f"duty_change_{k}"] += 1
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    sets = [("abl_arm", "full"), ("abl", "v6"), ("abl_mix", "v6")]
    win = pd.read_csv(os.path.join(ROOT, "results", "abl_arm", "cum_summary.csv"))
    W = {s: float(g["W"].iloc[0]) for s, g in win.groupby("scenario")}
    actual = {(r.scenario, r.arm): r for r in win.itertuples()}
    tick_rows, run_rows, ep_rows, h_rows = [], [], [], []
    check = collections.Counter()
    vdiff = []
    for st, arm in sets:
        for d in sorted(glob.glob(os.path.join(ROOT, "data", "runs", st, "*", arm, "r*"))):
            if not os.path.exists(f"{d}/done.json") or "_flagged" in d:
                continue
            scen, seed = d.split("/")[-3], int(d.split("/")[-1][1:])
            rows = replay_run(d, scen)
            ex = aa.run_extra(d)
            t0 = ex["t0_srv"]
            for r in rows:
                check["ticks"] += 1
                check["duty_reproduced"] += int(abs(r["duty_replay"] - r["duty_log"]) < 1e-9)
                check["train_decision_reproduced"] += int((r["duty_replay"] > 0) == (r["duty_log"] > 0))
                check["H_checked"] += 1
                check["H_reproduced"] += int(abs(r["H_f90"] - r["H_log"]) <= 0.15)
                vdiff.append((st, scen, seed, r["t"], r["value_log"], r["value_replay"], r["duty_log"], r["duty_replay"],
                              r["H_log"], r["H_f90"]))
                tick_rows.append(dict(set=st, scenario=scen, seed=seed, rel_s=r["at"] - t0, **r))
            c = flips(rows, t0)
            h_rows.append(dict(set=st, scenario=scen, seed=seed, **c))
            if st == "abl_arm":
                sw_dir = os.path.join(ROOT, "data", "runs", st, scen, "sw", f"r{seed}")
                rr, er = run_row14(d, scen, seed, rows, sw_dir, W[scen])
                a = actual.get((scen, "full"))
                if a is not None:
                    be = str(a.breakeven_per_seed).split(";")
                    ds = str(a.delta_W_per_seed_pct).split(";")
                    rr["breakeven_actual"] = be[seed - 1].strip() if len(be) >= seed else None
                    rr["delta_W_pct_actual"] = ds[seed - 1].strip() if len(ds) >= seed else None
                run_rows.append(rr)
                ep_rows += er
            print(f"  {st}/{scen}/{arm}/r{seed}: {len(rows)} rule-3 ticks", flush=True)
    ticks = pd.DataFrame(tick_rows)
    ticks.to_csv(f"{OUT}/ticks.csv", index=False)
    runs = pd.DataFrame(run_rows)
    runs.to_csv(f"{OUT}/row14_runs.csv", index=False)
    eps = pd.DataFrame(ep_rows)
    eps.to_csv(f"{OUT}/row14_episodes.csv", index=False)
    h = pd.DataFrame(h_rows).fillna(0)
    h.to_csv(f"{OUT}/row15_runs.csv", index=False)
    # ---- replay check
    v = pd.DataFrame(vdiff, columns=["set", "scenario", "seed", "t", "value_log", "value_replay", "duty_log", "duty_replay",
                                     "H_log", "H_replay"])
    v["abs_diff"] = (v.value_replay.round() - v.value_log).abs()
    bad = v[(v.duty_log - v.duty_replay).abs() > 1e-9]
    bad.to_csv(f"{OUT}/replay_mismatches.csv", index=False)
    chk = dict(check, value_abs_diff_median=float(v.abs_diff.median()), value_abs_diff_p95=float(v.abs_diff.quantile(0.95)),
               value_abs_diff_max=float(v.abs_diff.max()), H_abs_diff_max=float((v.H_log - v.H_replay).abs().max()),
               duty_mismatches=len(bad),
               train_mismatches=int(((v.duty_log > 0) != (v.duty_replay > 0)).sum()))
    json.dump(chk, open(f"{OUT}/replay_check.json", "w"), indent=1)
    print("REPLAY CHECK", json.dumps(chk))
    if len(bad):
        print(bad.to_string(index=False))
    # ---- row 14 summary per scenario
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 400)
    if len(runs):
        cols = ["scenario", "seed", "rule3_ticks", "train_ticks", "first_s", "first_H", "first_a_est", "first_a_meas",
                "first_a_target", "first_n", "first_tau_0.5", "first_V", "first_tstar", "first_gain_pct", "first_actual_delta_H",
                "ep1_dur_s", "ep1_learned", "last_a_est", "last_n", "last_tau", "last_V", "last_tstar", "last_gain_pct",
                "last_actual_delta_H", "n_episodes", "learned_in_episodes", "learned_B", "req_in_last_head", "req_unserved",
                "a_first5", "a_last30", "a_curve_last_head", "ceiling", "breakeven_actual", "delta_W_pct_actual"]
        print("ROW 14 per run")
        print(runs[[c for c in cols if c in runs.columns]].round(3).to_string(index=False))
        num = [c for c in runs.columns if c not in ("scenario", "seed", "dom", "reasons_B", "swap_times_B", "head_at_B",
                                                    "breakeven_actual", "delta_W_pct_actual")]
        summ = runs.groupby("scenario", sort=False)[num].mean(numeric_only=True).reset_index()
        summ.to_csv(f"{OUT}/row14_scenarios.csv", index=False)
        print("ROW 14 episodes")
        print(eps.round(2).to_string(index=False))
    # ---- row 15 summary
    keys = [c for c in h.columns if c not in ("set", "scenario", "seed")]
    hs = h.groupby(["set", "scenario"], sort=False)[keys].sum().reset_index()
    hs.to_csv(f"{OUT}/row15_scenarios.csv", index=False)
    main_cols = ["set", "scenario", "ticks", "train_f30", "train_f60", "train_f90", "train_f180", "to_notrain_f30",
                 "to_train_f30", "to_notrain_f60", "to_train_f60", "to_notrain_f180", "to_train_f180"]
    print("ROW 15 per scenario (sum over 3 seeds)")
    print(hs[[c for c in main_cols if c in hs.columns]].to_string(index=False))
    where = [c for c in hs.columns if "@" in c]
    print(hs[["set", "scenario"] + sorted(where)].to_string(index=False))
    print("ROW 15 per run")
    print(h[[c for c in main_cols if c in h.columns] + ["seed"]].to_string(index=False))
    # distribution of the model's break-even time against the horizon, per scenario (what a floor can change)
    if len(ticks):
        q = ticks.replace([np.inf, -np.inf], np.nan).groupby(["set", "scenario"], sort=False).agg(
            ticks=("t", "size"), tstar_p10=("tstar_min", lambda x: x.quantile(0.1)), tstar_p50=("tstar_min", "median"),
            tstar_p90=("tstar_min", lambda x: x.quantile(0.9)), tstar_inf=("tstar_min", lambda x: int(x.isna().sum())),
            age_min=("age", "min"), age_p50=("age", "median"), gain_p50=("gain_pct", "median")).reset_index()
        q.to_csv(f"{OUT}/row15_tstar.csv", index=False)
        print("ROW 15 break-even time of the plan (s) against the regime age")
        print(q.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
