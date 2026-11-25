"""Row 10 follow-up (analysis only): item lists for the mixed groups and step-level checks of why the
simulator's cost model loses to the fixed-budget policies. The frozen simulator is imported unchanged; the
rule-3 branch trace runs on a copy of pol_costmodel's source with logging lines inserted at run time."""
import inspect, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim
from sim import Model, run, drift_series, operating_point
import validate_heldout as vh
import sim_arm as sa

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "sim_arm")
PREREG = ["switch_only", "round800_safe", "budget_safe", "best_of_main3"]


def series_of(m, spec):
    lamA = vh.lam_for(m, spec["A"], spec["C_A"])
    lamB = spec["lamB_mult"] * vh.lam_for(m, spec["B"], 256) if spec.get("lamB_mult") else vh.lam_for(m, spec["B"], spec["C_B"])
    return drift_series(lamA, lamB, spec["A"], spec["B"], spec["pre"], spec["dur"], 0) + [(lamA, spec["A"])] * int(spec["post"]), lamA, lamB


def sim_log(spec, pol):
    s, _, _ = series_of(Model(), spec)
    N0 = {spec["A"]: spec["NA"]} if spec["NA"] else {}
    sm, df = run(Model(), s, pol, N0=dict(N0), log=True)
    return sm, df


def mixed_items():
    rd = pd.read_csv(os.path.join(OUT, "sim_arm_reading.csv"))
    it = pd.read_csv(os.path.join(OUT, "sim_arm_items.csv"))
    it["scenario"] = it["scenario"].astype(str)
    rows = []
    for _, g in rd[(rd.verdict == "混合") & rd.group.isin(["trace", "driftI", "driftII", "driftIII"]) & rd.comparator.isin(PREREG)].iterrows():
        sub = it[(it.group == g.group) & (it.comparator == g.comparator)].sort_values("ratio")
        for oc in ("loss", "win"):
            ids = sub[sub.outcome == oc]
            if oc == "win":
                ids = ids.iloc[::-1]
            rows.append(dict(group=g.group, comparator=g.comparator, outcome=oc, n=len(ids),
                             items="; ".join(f"{r.scenario} ({r.ratio:.3f})" for r in ids.itertuples())))
        rows.append(dict(group=g.group, comparator=g.comparator, outcome="tie", n=int((sub.outcome == "tie").sum()),
                         items="; ".join(sub[sub.outcome == "tie"].scenario)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "sim_arm_mixed_items.csv"), index=False)
    print("MIXED GROUPS, ITEM BY ITEM (sorted by ratio; losses worst first, wins best first)")
    for r in t.itertuples():
        print(f"  {r.group:8s} {r.comparator:14s} {r.outcome:4s} {r.n:3d}: {r.items}")


def trace_rule3(spec, lo=-2, hi=16):
    """Which branch of the frozen rule chain returned, tick by tick (copy of the source with logging)."""
    log = []
    src = inspect.getsource(sim.pol_costmodel).split("\n")
    new = []
    for k, ln in enumerate(src):
        ind = ln[: len(ln) - len(ln.lstrip())]
        if ln.strip().startswith("return spec_now"):
            new.append(f"{ind}LOG.append(dict(t=ctx['t'], branch={k}, a=a, ok_now=ok_now))")
        if ln.strip().startswith("if value - loss > best_v:"):
            new.append(f"{ind}LOG.append(dict(t=ctx['t'], d=d, a=a, dN=dN, da=da, w=fut[0][2], value=value, loss=loss))")
        new.append(ln)
    ns = dict(sim.__dict__, LOG=log)
    exec("\n".join(new), ns)
    sm_ref, _ = sim_log(spec, sim.POLICIES["costmodel_lindy"])
    sm, df = sim_log(spec, lambda ctx: ns["pol_costmodel"](ctx, forecast="lindy"))
    assert sm["goodput"] == sm_ref["goodput"], "instrumented copy diverged from the frozen policy"
    names = {k: src[k - 1].strip() for k in range(len(src)) if src[k].strip().startswith("return spec_now")}
    lg = pd.DataFrame(log)
    t0 = int(spec["pre"])
    return lg[(lg.t >= t0 + lo) & (lg.t < t0 + hi)], names, df


def check_id25():
    spec = vh.specs_drift_I()[25]
    m = Model(); m.cur_dom = spec["B"]
    _, _, lamB = series_of(m, spec)
    dem = lamB * m.out_per_req(spec["B"])
    lg, names, df = trace_rule3(spec)
    print("\nDRIFT I id 25 (fc -> code, lamB x1.05, 60 s, NA 3200): frozen rule chain, tick by tick")
    print("  branch lines:", names)
    print(lg.round(4).to_string())
    print("  steady state at the burst demand %.1f tok/s (slo %.2f ms):" % (dem, 2 * m.T_base(1)))
    for a in (2.1573, 2.3693, 2.4178, 2.6386):
        Cs, sat = operating_point(m, dem, True, a, 0.0)
        print("    a %.4f  capacity at C=256 %+.3f%% of demand  steady C %.1f (sat %s) tpot %.2f  tpot at 256 %.2f" % (
            a, 100 * (m.thr_at(256, True, a, 0.0) / dem - 1), Cs, sat, m.tpot(Cs, True, a, 0.0), m.tpot(256, True, a, 0.0)))
    i0, i1 = int(spec["pre"]), int(spec["pre"]) + int(spec["dur"])
    for p in ("costmodel_lindy", "budget_safe", "round800_safe"):
        _, d = sim_log(spec, sa.pol_of(p))
        B = d.iloc[i0:i1]
        okB = B.ok.to_numpy()
        print("    %-16s burst: train s %2d  a_end %.4f  tpot_end %.2f  wait first/max/end %.3f/%.3f/%.3f  ok s %2d (first %s)  goodput %.0f" % (
            p, int((B.duty > 0).sum()), B.a.iloc[-1], B.tpot.iloc[-1], B.wait.iloc[0], B.wait.max(), B.wait.iloc[-1],
            okB.sum(), (int(okB.argmax()) if okB.any() else "-"), B.goodput.sum()))


def check_backlog():
    """Drift I losses against a fixed-budget policy where nobody earns goodput in the burst."""
    d1 = pd.read_csv(os.path.join(OUT, "sim_arm_driftI.csv"))
    it = pd.read_csv(os.path.join(OUT, "sim_arm_items.csv"))
    it = it[(it.group == "driftI") & it.comparator.isin(["round800_safe", "budget_safe"]) & (it.outcome == "loss")]
    ids = sorted(set(int(x) for x in it.scenario))
    gB = d1[d1.policy.isin(["costmodel_lindy", "round800_safe", "budget_safe"])].pivot(index="id", columns="policy", values="g_B")
    ids0 = [i for i in ids if (gB.loc[i] == 0).all()]
    specs = vh.specs_drift_I()
    rows = []
    for i in ids0:
        sp = specs[i]
        i0, i1 = int(sp["pre"]), int(sp["pre"]) + int(sp["dur"])
        r = dict(id=i, A=sp["A"], B=sp["B"], lamB_mult=sp["lamB_mult"], dur=sp["dur"], NA=sp["NA"])
        for p, tag in (("costmodel_lindy", "cm"), ("budget_safe", "bud"), ("round800_safe", "r800")):
            _, d = sim_log(sp, sa.pol_of(p))
            B, post = d.iloc[i0:i1], d.iloc[i1:]
            okp = post.ok.to_numpy()
            r.update({f"{tag}_trainB_tok": float(B.train_tok.sum()), f"{tag}_specB": float(B.spec.mean()),
                      f"{tag}_servedB": float(B.served_tok.sum()), f"{tag}_wait_end": float(B.wait.iloc[-1]),
                      f"{tag}_post_first_ok": int(okp.argmax()) if okp.any() else -1, f"{tag}_post_ok_s": int(okp.sum()),
                      f"{tag}_a_post": float(post.a.iloc[0]), f"{tag}_g_post": float(post.goodput.sum())})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "sim_arm_mech_backlog.csv"), index=False)
    print("\nDRIFT I losses with zero burst goodput for all three policies: %d of %d losses" % (len(ids0), len(ids)))
    cols = ["id", "A", "B", "lamB_mult", "dur", "NA", "cm_trainB_tok", "bud_trainB_tok", "r800_trainB_tok", "cm_specB", "bud_specB",
            "cm_servedB", "bud_servedB", "r800_servedB", "cm_wait_end", "bud_wait_end", "r800_wait_end",
            "cm_post_first_ok", "bud_post_first_ok", "r800_post_first_ok", "cm_g_post", "bud_g_post", "r800_g_post"]
    print(t[cols].round(3).to_string(index=False))
    smaller = ((t.bud_wait_end < t.cm_wait_end) | (t.r800_wait_end < t.cm_wait_end))
    print("  fixed policy ends the burst with a shorter queue than the cost model in %d of %d; cost model trains 0 B tokens in %d of %d" % (
        int(smaller.sum()), len(t), int((t.cm_trainB_tok < 1).sum()), len(t)))
    print(t[["id", "dur", "cm_wait_end", "bud_wait_end", "r800_wait_end", "cm_a_post", "bud_a_post", "r800_a_post"]].round(4).to_string(index=False))
    longer = ~smaller
    hi_a = (t.bud_a_post > t.cm_a_post + 1e-9) & (t.r800_a_post > t.cm_a_post + 1e-9)
    print("  queue not shorter for either fixed policy: %d (ids %s); of these, both fixed policies enter the post phase with higher acceptance: %d" % (
        int(longer.sum()), list(t.id[longer]), int((longer & hi_a).sum())))
    print("  shorter queue for a fixed policy: %d (ids %s); of these, both fixed policies enter the post phase with higher acceptance: %d" % (
        int(smaller.sum()), list(t.id[smaller]), int((smaller & hi_a).sum())))
    # the run serves the same total in every policy, so post goodput can rise either because the burst ends with a
    # shorter queue (more compliant seconds afterwards) or because tokens not served during the non-compliant burst
    # are served after it, inside compliant windows
    for tag in ("bud", "r800"):
        t[f"{tag}_servedB_pct"] = 100 * (t[f"{tag}_servedB"] / t.cm_servedB - 1)
        t[f"{tag}_ok_gain_s"] = t[f"{tag}_post_ok_s"] - t.cm_post_ok_s
        t[f"{tag}_deferred_tok"] = t.cm_servedB - t[f"{tag}_servedB"]
        t[f"{tag}_gpost_gain"] = t[f"{tag}_g_post"] - t.cm_g_post
    print(t[["id", "dur", "bud_servedB_pct", "r800_servedB_pct", "bud_ok_gain_s", "r800_ok_gain_s",
             "bud_deferred_tok", "bud_gpost_gain", "r800_deferred_tok", "r800_gpost_gain"]].round(2).to_string(index=False))
    t.to_csv(os.path.join(OUT, "sim_arm_mech_backlog.csv"), index=False)


def check_transfer():
    """Drift II losses against round800_safe: acceptance of domain A after a zh burst."""
    m = Model()
    print("\nTRANSFER coefficients in the frozen model:", {d: round(m.transfer(d), 3) for d in m.dom})
    it = pd.read_csv(os.path.join(OUT, "sim_arm_items.csv"))
    it = it[(it.group == "driftII") & (it.comparator == "round800_safe") & (it.outcome == "loss")]
    specs = vh.specs_drift_II()
    rows = []
    for i in sorted(int(x) for x in it.scenario):
        sp = specs[i]
        i0, i1 = int(sp["pre"]), int(sp["pre"]) + int(sp["dur"])
        r = dict(id=i, A=sp["A"], C_A=sp["C_A"], C_B=sp["C_B"], lamB_mult=sp["lamB_mult"], dur=sp["dur"], NA=sp["NA"])
        for p, tag in (("costmodel_lindy", "cm"), ("round800_safe", "r800")):
            sm, d = sim_log(sp, sa.pol_of(p))
            B, post = d.iloc[i0:i1], d.iloc[i1:]
            r.update({f"{tag}_N_zh": sm["N"]["zh"], f"{tag}_gB": float(B.goodput.sum()), f"{tag}_wait_end": float(B.wait.iloc[-1]),
                      f"{tag}_a_post": float(post.a.iloc[0]), f"{tag}_g_post": float(post.goodput.sum())})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "sim_arm_mech_transfer.csv"), index=False)
    print("DRIFT II losses against round800_safe (%d):" % len(t))
    print(t.round(3).to_string(index=False))
    print("  round800_safe enters the post phase with higher A-domain acceptance in %d of %d; with a longer queue in %d of %d" % (
        int((t.r800_a_post > t.cm_a_post).sum()), len(t), int((t.r800_wait_end > t.cm_wait_end + 1e-9).sum()), len(t)))


if __name__ == "__main__":
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 400)
    mixed_items()
    check_id25()
    check_backlog()
    check_transfer()
