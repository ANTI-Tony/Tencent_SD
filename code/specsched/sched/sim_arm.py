"""Row 10: the frozen simulator with three more baselines (switch only, fixed-round and calibrated-budget training).

Nothing frozen is edited: sim.py, model.json and validate_heldout.py are imported as they are, and the held-out
scenario set is reused unchanged (35 trace windows, drift I 60, drift II 40, drift III 20 x 7 surge levels).
Preregistered in ideas/2026-11-24_同起点对照与累计回本_预注册.md ("第 10 项的口径"), before this file was run.

New policies. The speculation switch is the frozen cost-model policy's own expression in every one of them:
  switch_only      switch, never trains
  round800_safe    trains until the current domain has learned 800 requests, then stops
  budget_safe      same, with the calibrated budget of the domain (the frozen budget rule: 0.8 of the curve's
                   headroom, 0 when even the ceiling cannot beat the frontier at saturation)
  duty: inside the SLO and not saturated -> the largest duty that stays unsaturated with TPOT <= 0.9 x SLO (the
        cost-model policy's rule 2, may be 0); otherwise 0.5
  count: requests the domain has learned in total (the simulator never forgets, so this favours the simple policies)
Sensitivity variants, appendix only, not part of the reading:
  *_naive   0.5 whenever below the target, whatever the SLO
  *_regime  the count restarts at every domain change (the way the hardware arms count); duty as *_safe

Gate: the 8 frozen policies are rerun first, exactly as validate_heldout.py runs them, and every numeric column of
the four frozen result tables has to be reproduced (relative 1e-9) before any number of a new policy is read.

Output (results/sim_arm/): sim_arm_{trace,driftI,driftII,driftIII}.csv, sim_arm_gate.json, sim_arm_reading.csv,
sim_arm_items.csv, sim_arm_policy_totals.csv, sim_arm_verdict.json
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import DUTIES, POLICIES, Model, drift_series, operating_point, run, trace_series  # noqa: E402
from validate_heldout import (KEYS, POLS, SIMPLE, TRACES, lam_for, sha, specs_drift_I, specs_drift_II,  # noqa: E402
                              specs_drift_III, specs_trace, verdicts)

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "sim_arm")
FROZEN = os.path.join(ROOT, "results", "summaries")
MAIN = ["switch_only", "round800_safe", "budget_safe"]
SENS = ["round800_naive", "budget_naive", "round800_regime", "budget_regime"]
NEW = MAIN + SENS
ALL = POLS + NEW
CM = "costmodel_lindy"
BUDGET_EXPECTED = {"chat": 153, "code": 428, "fc": 90, "sql": 653, "zh": 0}
GROUPS = {"trace": ["trace", "start_hour", "scale"], "driftI": ["id"], "driftII": ["id"], "driftIII": ["id", "surge"]}


def budgets(m, frac=0.8, max_seqs=256):
    """The frozen budget rule (controller_budget.py, the budget-frac branch), read off model.json."""
    out = {}
    for dname, dv in m.dom.items():
        cv = sorted((float(k), v) for k, v in dv["acc_curve"].items())
        a_first, a_max = cv[0][1], max(v for _, v in cv)
        m.cur_dom = dname if dname in m.fr else None
        Csat = float(max_seqs)
        if m.gain(a_max, Csat) <= 1.0:
            out[dname] = 0                                                     # structural ceiling
        else:
            tgt = a_first + frac * (a_max - a_first)
            n = next((n0 + (n1 - n0) * (tgt - v0) / (v1 - v0) for (n0, v0), (n1, v1) in zip(cv, cv[1:])
                      if v0 <= tgt <= v1 and v1 > v0), cv[-1][0])
            out[dname] = int(round(n))
    m.cur_dom = None
    return out


BUDGET = budgets(Model())
assert BUDGET == BUDGET_EXPECTED, BUDGET


def load_state(ctx):
    """Speculation switch and load state, copied from the head of sim.pol_costmodel."""
    m, st, dom, a, dt = ctx["model"], ctx["state"], ctx["dom"], ctx["a"], ctx["dt"]
    slo_tpot = ctx["slo_tpot"]
    demand_now = ctx["lam"] * m.out_per_req(dom)
    demand_q = demand_now + st.backlog / dt
    C0, _ = operating_point(m, demand_q, False, a, 0.0)
    Cs, _ = operating_point(m, demand_q, True, a, 0.0)
    spec_now = m.gain(a, Cs) > 1.0 and (m.tpot(Cs, True, a, 0.0) <= slo_tpot or m.tpot(C0, False, a, 0.0) > slo_tpot)
    C = Cs if spec_now else C0
    saturated_deficit = C >= 255.9 and m.thr_at(C, spec_now, a, 0.0) < demand_q * 0.999
    ok_now = (m.tpot(C, spec_now, a, 0.0) <= slo_tpot and ctx.get("wait", 0.0) <= ctx.get("slo_wait", 1.0)
              and not saturated_deficit)
    return spec_now, ok_now, demand_q


def make_policy(name):
    """A fresh policy per run: the *_regime variants keep the count at the start of the regime."""
    if name == "switch_only":
        return lambda ctx: (load_state(ctx)[0], 0.0)
    kind, mode = name.split("_")
    base = {"n": 0.0}

    def pol(ctx):
        m, st, dom, a = ctx["model"], ctx["state"], ctx["dom"], ctx["a"]
        spec_now, ok_now, demand_q = load_state(ctx)
        if mode == "regime":
            if st.regime_t == 0.0:
                base["n"] = st.N[dom]
            learned = st.N[dom] - base["n"]
        else:
            learned = st.N[dom]
        target = 800.0 if kind == "round800" else float(BUDGET[dom])
        if target <= 0 or learned >= target:
            return spec_now, 0.0
        if mode == "naive" or not ok_now:
            return spec_now, 0.5
        d_free = 0.0
        for d in DUTIES[1:]:
            Cd, satd = operating_point(m, demand_q, spec_now, a, d)
            if not satd and m.tpot(Cd, spec_now, a, d) <= 0.9 * ctx["slo_tpot"]:
                d_free = d
        return spec_now, d_free
    return pol


def pol_of(p):
    return POLICIES[p] if p in POLICIES else make_policy(p)


def job_trace(spec):
    m = Model()
    s = trace_series(os.path.join(TRACES, spec["file"]), spec["kind"], spec["dom"], m, spec["hours"], spec["scale"],
                     dt=spec["dt"], start_hour=spec["start_hour"])
    rows = []
    for p in ALL:
        sm, df = run(m, s, pol_of(p), dt=spec["dt"], log=True)
        rows.append(dict(spec, policy=p, **{k: sm[k] for k in KEYS}, acc=sm["final_acc"][spec["dom"]],
                         N_dom=sm["N"][spec["dom"]], train_s=float((df.duty > 0).sum() * spec["dt"])))
    return rows


def job_drift(spec):
    m = Model()
    lamA = lam_for(m, spec["A"], spec["C_A"])
    lamB = spec["lamB_mult"] * lam_for(m, spec["B"], 256) if spec.get("lamB_mult") else lam_for(m, spec["B"], spec["C_B"])
    s = drift_series(lamA, lamB, spec["A"], spec["B"], spec["pre"], spec["dur"], 0)
    if spec.get("surge") is not None:
        s += [(lam_for(m, spec["B"], 256) * spec["surge"], spec["B"])] * int(spec["surge_dur"])
    s += [(lamA, spec["A"])] * int(spec["post"])
    N0 = {spec["A"]: spec["NA"]} if spec["NA"] else {}
    i0 = int(spec["pre"])
    i1 = i0 + int(spec["dur"])
    i2 = i1 + (int(spec["surge_dur"]) if spec.get("surge") is not None else 0)
    rows = []
    for p in ALL:
        sm, df = run(m, s, pol_of(p), N0=dict(N0), log=True)
        r = dict(spec, policy=p, **{k: sm[k] for k in KEYS}, acc_B=sm["final_acc"][spec["B"]])
        if spec.get("surge") is not None:
            r["surge_viol_s"] = float((~df.ok.iloc[i1:i2]).sum())
        r.update(g_pre=float(df.goodput.iloc[:i0].sum()), g_B=float(df.goodput.iloc[i0:i1].sum()),
                 g_surge=float(df.goodput.iloc[i1:i2].sum()), g_post=float(df.goodput.iloc[i2:].sum()),
                 train_pre=float(df.train_tok.iloc[:i0].sum()), train_B=float(df.train_tok.iloc[i0:i2].sum()),
                 train_post=float(df.train_tok.iloc[i2:].sum()), viol_B=float((~df.ok.iloc[i0:i2]).sum()),
                 spec_frac_B=float(df.spec.iloc[i0:i2].mean()), N_A=sm["N"][spec["A"]], N_B=sm["N"][spec["B"]])
        rows.append(r)
    return rows


def selfcheck_switch():
    """The copied switch expression must agree with the one inside pol_costmodel at every step."""
    n = bad = 0

    def wrap(ctx):
        nonlocal n, bad
        mine = load_state(ctx)[0]
        spec, duty = POLICIES[CM](ctx)
        n += 1
        bad += int(bool(mine) != bool(spec))
        return spec, duty
    m = Model()
    for spec in specs_drift_I()[:6] + specs_drift_II()[:6]:
        lamA = lam_for(m, spec["A"], spec["C_A"])
        lamB = spec["lamB_mult"] * lam_for(m, spec["B"], 256) if spec.get("lamB_mult") else lam_for(m, spec["B"], spec["C_B"])
        s = drift_series(lamA, lamB, spec["A"], spec["B"], spec["pre"], spec["dur"], 0) + [(lamA, spec["A"])] * int(spec["post"])
        N0 = {spec["A"]: spec["NA"]} if spec["NA"] else {}
        a, _ = run(m, s, wrap, N0=dict(N0))
        b, _ = run(m, s, POLICIES[CM], N0=dict(N0))
        assert a["goodput"] == b["goodput"]
    return dict(steps=n, disagreements=bad)


# ----------------------------------------------------------------------------- gate

def gate_table(new, name):
    keys = GROUPS[name]
    old = pd.read_csv(os.path.join(FROZEN, f"sim_heldout_{name}.csv"))
    cur = new[new.policy.isin(POLS)]
    mg = old.merge(cur, on=keys + ["policy"], how="outer", suffixes=("_old", "_new"), indicator=True)
    res = dict(table=name, rows_frozen=int(len(old)), rows_rerun=int(len(cur)), rows_matched=int((mg._merge == "both").sum()))
    worst, bad = 0.0, []
    for c in [c for c in old.columns if c not in keys + ["policy"]]:
        a, b = mg[c + "_old"], mg[c + "_new"]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):   # text columns (pandas 3: str dtype)
            same = (a.astype(str).to_numpy(dtype=object) == b.astype(str).to_numpy(dtype=object))
        else:
            a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            same = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
            worst = max(worst, float(np.nanmax(np.where(np.isfinite(rel), rel, 0.0))))
        if not same.all():
            bad.append(dict(column=c, n_diff=int((~same).sum())))
    res.update(columns_checked=int(len(old.columns) - len(keys) - 1), max_rel_err=worst, mismatches=bad,
               passed=bool(res["rows_matched"] == len(old) == len(cur) and not bad))
    return res


def close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return bool(np.isclose(float(a), float(b), rtol=tol, atol=1e-12, equal_nan=True))


# ----------------------------------------------------------------------------- reading

def outcome(cm, other):
    """win / tie / loss of the cost model against one policy, the frozen +-0.1% band."""
    if cm == 0 and other == 0:
        return "tie", 1.0
    if other == 0:
        return "win", float("inf")
    r = cm / other
    return ("win" if r > 1.001 else "loss" if r < 0.999 else "tie"), r


def label(win, tie, loss):
    if win >= 0.5 and loss <= 0.1:
        return "代价模型更好"
    if tie >= 0.8:
        return "分不开"
    if loss >= 0.5 and win <= 0.1:
        return "代价模型落后"
    return "混合"


def reading(frames):
    rows, items = [], []
    for name, df in frames.items():
        keys = GROUPS[name]
        views = {name: df.pivot_table(index=keys, columns="policy", values="goodput")}
        if name == "driftIII":       # descriptive: the 7 surge levels of a base scenario summed
            views["driftIII_per_base"] = df.groupby(["id", "policy"]).goodput.sum().unstack()
        for vname, g in views.items():
            comps = {p: g[p] for p in NEW}
            comps["best_of_main3"] = g[MAIN].max(axis=1)
            for comp, col in comps.items():
                res = [outcome(c, o) for c, o in zip(g[CM].to_numpy(), col.to_numpy())]
                lab = np.array([x[0] for x in res])
                rat = np.array([x[1] for x in res])
                fin = rat[np.isfinite(rat)]
                w, t, l = [float((lab == k).mean()) for k in ("win", "tie", "loss")]
                rows.append(dict(group=vname, comparator=comp, role="main" if comp in MAIN + ["best_of_main3"] else "sensitivity",
                                 preregistered=bool(not vname.endswith("per_base") and comp in MAIN + ["best_of_main3"]),
                                 n=int(len(lab)), win=int((lab == "win").sum()), tie=int((lab == "tie").sum()),
                                 loss=int((lab == "loss").sum()), win_frac=w, tie_frac=t, loss_frac=l, verdict=label(w, t, l),
                                 ratio_min=float(fin.min()) if len(fin) else None, ratio_median=float(np.median(rat)),
                                 ratio_max=float(rat.max()), sum_ratio=float(g[CM].sum() / col.sum()) if col.sum() > 0 else None))
                for idx, c, o, (lb, r) in zip(g.index, g[CM].to_numpy(), col.to_numpy(), res):
                    items.append(dict(group=vname, comparator=comp, scenario=str(idx), cm_goodput=float(c),
                                      other_goodput=float(o), ratio=r, outcome=lb))
    return pd.DataFrame(rows), pd.DataFrame(items)


def verdicts_ext(tr, d1, d2, d3):
    """V1'..V4': the frozen checks with the three main policies added to the comparison sets; thresholds unchanged."""
    v = {}
    idx = ["trace", "start_hour", "scale"]
    g = tr.pivot_table(index=idx, columns="policy", values="goodput")
    t = tr.pivot_table(index=idx, columns="policy", values="train_tok")
    simple2 = SIMPLE + MAIN
    ratio = g[CM] / g[simple2].max(axis=1)
    train_ratio = t[CM] / t["osd_idle"].replace(0, np.nan)
    rank = g[simple2 + [CM]].rank(axis=1, ascending=False)[CM]
    best_name = g[simple2].idxmax(axis=1)
    v["V1p"] = dict(n=int(len(ratio)), min_ratio=float(ratio.min()), median_ratio=float(ratio.median()),
                    n_pass_ratio=int((ratio >= 0.97).sum()), n_best=int((rank <= 1.0).sum()),
                    n_pass_train=int(((train_ratio <= 0.10) | train_ratio.isna()).sum()),
                    passed=bool((ratio >= 0.97).all() and ((train_ratio <= 0.10) | train_ratio.isna()).all()),
                    worst=str(ratio.idxmin()), best_simple_counts={k: int(n) for k, n in best_name.value_counts().items()},
                    train_M={p: [float(t[p].min() / 1e6), float(t[p].median() / 1e6), float(t[p].max() / 1e6)] for p in [CM] + MAIN[1:]})
    g1 = d1.pivot_table(index="id", columns="policy", values="goodput")
    meta = d1.drop_duplicates("id").set_index("id")
    long = meta.index[meta.dur >= 10]
    den = g1[["spec_only"] + MAIN].max(axis=1)
    r1 = g1[CM] / den
    wins = r1.loc[long] >= 0.999
    by_dur = {}
    for dur, h in meta.groupby("dur"):
        rr = r1.loc[h.index]
        by_dur[int(dur)] = dict(n=int(len(rr)), strict_win=float((rr > 1.001).mean()), tie=float(((rr <= 1.001) & (rr >= 0.999)).mean()),
                                loss=float((rr < 0.999).mean()), median=float(rr.median()), min=float(rr.min()), max=float(rr.max()))
    v["V2p"] = dict(n_long=int(len(long)), n_win=int(wins.sum()), frac=float(wins.mean()), n_all=int(len(g1)),
                    n_win_all=int((r1 >= 0.999).sum()), passed=bool(wins.mean() >= 0.8),
                    losses=[int(i) for i in long[~wins.values]], max_loss_pct=float((1 - r1.min()) * 100), by_dur=by_dur,
                    best_counts={k: int(n) for k, n in g1[["spec_only"] + MAIN].idxmax(axis=1).value_counts().items()})
    g2 = d2.pivot_table(index="id", columns="policy", values="goodput")
    m2 = d2.drop_duplicates("id").set_index("id")
    sat = m2.index[(m2.C_B >= 256) | m2.lamB_mult.notna()]
    r_spec = g2[CM] / g2[["spec_only", "switch_only"]].max(axis=1)
    r_react = g2[CM] / g2[["osd_idle", "accdrop", "round800_safe", "budget_safe"]].max(axis=1)
    v["V3p"] = dict(n=int(len(g2)), n_sat=int(len(sat)), min_ratio_vs_spec_sat=float(r_spec.loc[sat].min()),
                    median_ratio_vs_spec_sat=float(r_spec.loc[sat].median()), max_ratio_vs_spec_sat=float(r_spec.loc[sat].max()),
                    min_ratio_vs_spec_all=float(r_spec.min()), min_ratio_vs_reactive=float(r_react.min()),
                    passed=bool((r_spec.loc[sat] >= 1.5).all() and (r_react >= 0.999).all()),
                    n_sat_ge_1p5=int((r_spec.loc[sat] >= 1.5).sum()), n_cm_ge_spec=int((r_spec >= 0.999).sum()),
                    n_cm_ge_reactive=int((r_react >= 0.999).sum()), n_cm_gt_reactive=int((r_react > 1.001).sum()),
                    reactive_losses=[int(i) for i in r_react.index[(r_react < 0.999).values]])
    ids = sorted(d3.id.unique())

    def max_surge(tol):
        ok = d3[d3.surge_viol_s <= tol].groupby(["id", "policy"]).surge.max().unstack()
        return ok.reindex(index=ids, columns=ALL).fillna(0.0)
    ok = max_surge(0)
    cm = ok[CM]
    others = ["osd_idle", "accdrop"] + MAIN
    ge = pd.concat([cm >= ok[p] - 1e-9 for p in others], axis=1).all(axis=1)
    ok10 = max_surge(10)
    ge10 = pd.concat([ok10[CM] >= ok10[p] - 1e-9 for p in others], axis=1).all(axis=1)
    v["V4p"] = dict(n=int(len(ok)), n_ge=int(ge.sum()), frac=float(ge.mean()), passed=bool(ge.mean() >= 0.8),
                    max_surge_mean={p: float(ok[p].mean()) for p in ALL}, losses=[int(i) for i in ok.index[~ge.values]],
                    vs={p: dict(better=int((cm > ok[p] + 1e-9).sum()), equal=int((abs(cm - ok[p]) <= 1e-9).sum()),
                                worse=int((cm < ok[p] - 1e-9).sum())) for p in others},
                    tol10s=dict(n_ge=int(ge10.sum()), max_surge_mean={p: float(ok10[p].mean()) for p in ALL}))
    return v


def totals(frames):
    rows = []
    for name, df in frames.items():
        for p in ALL:
            h = df[df.policy == p]
            ref = df[df.policy == CM]
            rows.append(dict(group=name, policy=p, n=int(len(h)), goodput_sum=float(h.goodput.sum()),
                             goodput_vs_cm=float(h.goodput.sum() / ref.goodput.sum()), served_sum=float(h.served.sum()),
                             slo_viol_s_mean=float(h.slo_viol_s.mean()), train_tok_mean=float(h.train_tok.mean()),
                             duty_mean=float(h.duty_mean.mean()), spec_frac_mean=float(h.spec_frac.mean())))
    return pd.DataFrame(rows)


def main(workers=8):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    frozen = {f: sha(os.path.join(HERE, f)) for f in ["sim.py", "model.json", "validate_heldout.py"]}
    print("frozen", frozen, flush=True)
    print("budgets", BUDGET, flush=True)
    sc = selfcheck_switch()
    print("switch self-check", sc, flush=True)
    assert sc["disagreements"] == 0, sc
    frames = {}
    with Pool(workers) as pool:
        for name, job, specs in (("trace", job_trace, specs_trace()), ("driftI", job_drift, specs_drift_I()),
                                 ("driftII", job_drift, specs_drift_II()), ("driftIII", job_drift, specs_drift_III())):
            frames[name] = pd.DataFrame([r for rows in pool.map(job, specs) for r in rows])
            frames[name].to_csv(os.path.join(OUT, f"sim_arm_{name}.csv"), index=False)
            print(f"{name} done {len(frames[name])} rows, {time.time() - t0:.0f} s", flush=True)
    finish(frames, frozen, sc)


def finish(frames, frozen, sc):
    gates = [gate_table(frames[k], k) for k in GROUPS]
    v_old = json.load(open(os.path.join(FROZEN, "sim_heldout_verdict.json")))
    v_new = json.loads(json.dumps(verdicts(*[frames[k][frames[k].policy.isin(POLS)] for k in GROUPS]), default=str))
    v_same = {k: close(v_old[k], v_new[k]) for k in ("V1", "V2", "V3", "V4")}
    gate = dict(tables=gates, frozen_verdict_reproduced=v_same, switch_selfcheck=sc, frozen=frozen,
                frozen_at_validation=v_old.get("frozen"),
                passed=bool(all(g["passed"] for g in gates) and all(v_same.values())))
    json.dump(gate, open(os.path.join(OUT, "sim_arm_gate.json"), "w"), indent=1, default=str)
    print("GATE", json.dumps(gate, indent=1, default=str), flush=True)
    if not gate["passed"]:
        print("gate failed: the new policies are not read", flush=True)
        sys.exit(2)
    rd, items = reading(frames)
    rd.to_csv(os.path.join(OUT, "sim_arm_reading.csv"), index=False)
    items.to_csv(os.path.join(OUT, "sim_arm_items.csv"), index=False)
    tot = totals(frames)
    tot.to_csv(os.path.join(OUT, "sim_arm_policy_totals.csv"), index=False)
    ext = verdicts_ext(*[frames[k] for k in GROUPS])
    out = dict(frozen=frozen, budgets=BUDGET, frozen_verdict={k: v_old[k]["passed"] for k in ("V1", "V2", "V3", "V4")},
               extended=ext, reading=rd[rd.preregistered].to_dict("records"))
    json.dump(out, open(os.path.join(OUT, "sim_arm_verdict.json"), "w"), indent=1, default=str, ensure_ascii=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    print("\nREADING (preregistered rows)")
    print(rd[rd.preregistered].drop(columns=["role", "preregistered"]).to_string(index=False))
    print("\nREADING (sensitivity and per-base, descriptive)")
    print(rd[~rd.preregistered].drop(columns=["role", "preregistered"]).to_string(index=False))
    print("\nTOTALS")
    print(tot.to_string(index=False))
    print("\nEXTENDED", json.dumps(ext, indent=1, default=str, ensure_ascii=False))


if __name__ == "__main__":
    if "--finish-only" in sys.argv:      # recompute the gate and the reading from the saved tables
        fr = {k: pd.read_csv(os.path.join(OUT, f"sim_arm_{k}.csv")) for k in GROUPS}
        finish(fr, {f: sha(os.path.join(HERE, f)) for f in ["sim.py", "model.json", "validate_heldout.py"]}, selfcheck_switch())
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8)
