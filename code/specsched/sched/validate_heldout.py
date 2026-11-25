"""Held-out validation of the simulator conclusions (E08 trace replay, E09 drift classes) with the FROZEN simulator.

Frozen: sim.py (policies, parameters: DUTIES, K=10 s, buffer 4M tokens, lindy floor 30 s, oracle bins 60 s,
SLO = 2 x T_base(1) / 1 s queue) and model.json (measured frontier, kappa, learning curves). Their hashes are
written into the verdict file; nothing in here is tuned on the outcome.

The trajectories are new: trace windows never replayed before (BurstGPT days 4/6/8/12/16/20 for the
conversation log, days 15/21/22 for the API log, both Azure 2023 hour-long traces) at three demand scales,
and drift scenarios drawn at random (seeded) over domains, loads, durations and prior adaptation.

Pre-registered checks (written before running):
  V1 trace   : on every held-out window costmodel_lindy goodput >= 0.97 x the best simple policy
               (static_off / spec_only / osd_idle / fixed_interval / accdrop) and trains <= 10% of osd_idle's tokens.
  V2 drift I : saturated bursts of an adaptable domain: for bursts >= 10 s costmodel_lindy >= spec_only goodput
               in >= 80% of scenarios; the duration at which training first pays is <= 10 s in the median scenario.
  V3 drift II: B = zh (ceiling below the frontier) at saturation: costmodel_lindy goodput >= 1.5 x spec_only and
               >= every reactive trainer (osd_idle, accdrop) in every scenario.
  V4 drift III: the largest surge served without SLO violation by costmodel_lindy >= that of osd_idle and of
               accdrop in >= 80% of scenarios.
Output: results/summaries/sim_heldout_{trace,driftI,driftII,driftIII}.csv + sim_heldout_verdict.json
"""
import hashlib
import json
import os
import random
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import POLICIES, Model, drift_series, run, trace_series  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "summaries")
TRACES = os.path.join(ROOT, "data", "traces")
POLS = ["static_off", "spec_only", "osd_idle", "fixed_interval", "accdrop", "costmodel_oracle", "costmodel_persist", "costmodel_lindy"]
SIMPLE = ["static_off", "spec_only", "osd_idle", "fixed_interval", "accdrop"]
KEYS = ["goodput", "served", "slo_viol_s", "train_tok", "duty_mean", "spec_frac"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def lam_for(m, dom, C):
    return m.thr_base(C) / m.out_per_req(dom)


def job_trace(spec):
    m = Model()
    s = trace_series(os.path.join(TRACES, spec["file"]), spec["kind"], spec["dom"], m, spec["hours"], spec["scale"],
                     dt=spec["dt"], start_hour=spec["start_hour"])
    rows = []
    for p in POLS:
        sm, _ = run(m, s, POLICIES[p], dt=spec["dt"])
        rows.append(dict(spec, policy=p, **{k: sm[k] for k in KEYS}, acc=sm["final_acc"][spec["dom"]]))
    return rows


def job_drift(spec):
    """spec: pre/post at (A, C_A), B at (B, C_B or demand multiple) for dur; optional surge segment."""
    m = Model()
    lamA = lam_for(m, spec["A"], spec["C_A"])
    lamB = spec["lamB_mult"] * lam_for(m, spec["B"], 256) if spec.get("lamB_mult") else lam_for(m, spec["B"], spec["C_B"])
    s = drift_series(lamA, lamB, spec["A"], spec["B"], spec["pre"], spec["dur"], 0)
    if spec.get("surge") is not None:
        s += [(lam_for(m, spec["B"], 256) * spec["surge"], spec["B"])] * int(spec["surge_dur"])
    s += [(lamA, spec["A"])] * int(spec["post"])
    N0 = {spec["A"]: spec["NA"]} if spec["NA"] else {}
    rows = []
    for p in POLS:
        sm, df = run(m, s, POLICIES[p], N0=dict(N0), log=spec.get("surge") is not None)
        r = dict(spec, policy=p, **{k: sm[k] for k in KEYS}, acc_B=sm["final_acc"][spec["B"]])
        if df is not None:
            i0 = int(spec["pre"] + spec["dur"])
            i1 = i0 + int(spec["surge_dur"])
            r["surge_viol_s"] = float((~df.ok.iloc[i0:i1]).sum())
        rows.append(r)
    return rows


def specs_trace():
    out = []
    for sh in [72, 120, 168, 264, 360, 456]:                       # conversation log: days 4, 6, 8, 12, 16, 20
        for sc in [0.6, 0.8, 1.0]:
            out.append(dict(trace="burstgpt_conv", file="BurstGPT_without_fails_1.csv", kind="conv", dom="chat",
                            hours=24, dt=10.0, start_hour=sh, scale=sc))
    for sh in [336, 480, 504]:                                     # API log: days 15, 21, 22 (busy, unused so far)
        for sc in [0.6, 0.8, 1.0]:
            out.append(dict(trace="burstgpt_api", file="BurstGPT_without_fails_1.csv", kind="api", dom="code",
                            hours=24, dt=10.0, start_hour=sh, scale=sc))
    for name, dom in [("AzureLLMInferenceTrace_conv.csv", "chat"), ("AzureLLMInferenceTrace_code.csv", "code")]:
        for sc in [0.6, 0.8, 1.0, 1.2]:
            out.append(dict(trace=name.split("_")[1][:-4] + "_azure", file=name, kind="azure", dom=dom,
                            hours=0.95, dt=1.0, start_hour=0.0, scale=sc))
    return out


def specs_drift_I(n=60, seed=1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        A = rng.choice(["chat", "code", "fc"])
        B = rng.choice([b for b in ["code", "sql", "fc"] if b != A])
        out.append(dict(exp="I", id=i, A=A, B=B, C_A=rng.choice([4, 8, 16, 32, 64]), C_B=256,
                        lamB_mult=rng.choice([1.0, 1.0, 1.05, 1.2]), dur=rng.choice([5, 10, 20, 60, 180, 600]),
                        NA=rng.choice([0, 3200]), pre=300, post=600))
    return out


def specs_drift_II(n=40, seed=2):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        A = rng.choice(["chat", "code"])
        out.append(dict(exp="II", id=i, A=A, B="zh", C_A=rng.choice([4, 16, 64]), C_B=rng.choice([128, 192, 256]),
                        lamB_mult=rng.choice([None, None, 1.1]), dur=rng.choice([600, 1200, 1800]),
                        NA=rng.choice([0, 3200]), pre=300, post=300))
    return out


def specs_drift_III(n=20, seed=3):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        A = rng.choice(["chat", "code"])
        B = rng.choice([b for b in ["sql", "code", "fc"] if b != A])
        base = dict(exp="III", id=i, A=A, B=B, C_A=rng.choice([8, 16, 32]), C_B=rng.choice([96, 128, 160]),
                    dur=rng.choice([300, 600, 900]), NA=rng.choice([0, 3200]), pre=rng.choice([120, 300, 900]),
                    post=300, surge_dur=600)
        for surge in [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
            out.append(dict(base, surge=surge))
    return out


def verdicts(tr, d1, d2, d3):
    """Pre-registered checks (passed flags) plus the descriptive statistics the write-up needs."""
    v = {}
    # V1
    g = tr.pivot_table(index=["trace", "start_hour", "scale"], columns="policy", values="goodput")
    t = tr.pivot_table(index=["trace", "start_hour", "scale"], columns="policy", values="train_tok")
    best_simple = g[SIMPLE].max(axis=1)
    ratio = g["costmodel_lindy"] / best_simple
    train_ratio = t["costmodel_lindy"] / t["osd_idle"].replace(0, np.nan)
    rank = g[SIMPLE + ["costmodel_lindy"]].rank(axis=1, ascending=False)["costmodel_lindy"]
    exc = train_ratio[train_ratio > 0.10]
    v["V1"] = dict(n=int(len(ratio)), min_ratio=float(ratio.min()), median_ratio=float(ratio.median()),
                   n_pass_ratio=int((ratio >= 0.97).sum()), n_best=int((rank <= 1.0).sum()),
                   max_train_ratio=float(train_ratio.max()), n_pass_train=int(((train_ratio <= 0.10) | train_ratio.isna()).sum()),
                   passed=bool((ratio >= 0.97).all() and ((train_ratio <= 0.10) | train_ratio.isna()).all()),
                   worst=ratio.idxmin(), cm_train_M=[float(t["costmodel_lindy"].min() / 1e6), float(t["costmodel_lindy"].max() / 1e6)],
                   osd_train_M=[float(t["osd_idle"].min() / 1e6), float(t["osd_idle"].max() / 1e6)],
                   train_exceptions=[dict(window=str(i), cm_M=round(float(t.loc[i, "costmodel_lindy"]) / 1e6, 2),
                                          osd_M=round(float(t.loc[i, "osd_idle"]) / 1e6, 3)) for i in exc.index],
                   gain_vs_spec_only=[float((g["costmodel_lindy"] / g["spec_only"]).min()), float((g["costmodel_lindy"] / g["spec_only"]).max())],
                   gain_vs_static_off=[float((g["costmodel_lindy"] / g["static_off"]).min()), float((g["costmodel_lindy"] / g["static_off"]).max())])
    # V2
    g1 = d1.pivot_table(index="id", columns="policy", values="goodput")
    meta = d1.drop_duplicates("id").set_index("id")
    long = meta.index[meta.dur >= 10]
    r1 = g1["costmodel_lindy"] / g1["spec_only"]
    wins = (r1.loc[long] >= 0.999)
    by_dur = {}
    for dur, h in meta.groupby("dur"):
        rr = r1.loc[h.index]
        adapt = rr.loc[h.index[h.B.isin(["code", "sql"])]]
        by_dur[int(dur)] = dict(n=int(len(rr)), strict_win=float((rr > 1.001).mean()), tie=float(((rr <= 1.001) & (rr >= 0.999)).mean()),
                                loss=float((rr < 0.999).mean()), median=float(rr.median()), max=float(rr.max()),
                                strict_win_adaptable=float((adapt > 1.001).mean()) if len(adapt) else None)
    v["V2"] = dict(n_long=int(len(long)), n_win=int(wins.sum()), frac=float(wins.mean()),
                   n_all=int(len(g1)), n_win_all=int((r1 >= 0.999).sum()), passed=bool(wins.mean() >= 0.8),
                   losses=[int(i) for i in long[~wins.values]], max_loss_pct=float((1 - r1.min()) * 100), by_dur=by_dur)
    # V3
    g2 = d2.pivot_table(index="id", columns="policy", values="goodput")
    m2 = d2.drop_duplicates("id").set_index("id")
    sat = m2.index[(m2.C_B >= 256) | m2.lamB_mult.notna()]
    feas = m2.index[(m2.C_B >= 256) & m2.lamB_mult.isna()]
    over = m2.index[m2.lamB_mult.notna()]
    r_spec = g2["costmodel_lindy"] / g2["spec_only"]
    r_react = g2["costmodel_lindy"] / g2[["osd_idle", "accdrop"]].max(axis=1)
    v["V3"] = dict(n=int(len(g2)), n_sat=int(len(sat)), min_ratio_vs_spec_sat=float(r_spec.loc[sat].min()),
                   min_ratio_vs_spec_all=float(r_spec.min()), min_ratio_vs_reactive=float(r_react.min()),
                   passed=bool((r_spec.loc[sat] >= 1.5).all() and (r_react >= 0.999).all()),
                   worst_sat=int(r_spec.loc[sat].idxmin()), worst_react=int(r_react.idxmin()),
                   n_cm_ge_spec=int((r_spec >= 0.999).sum()), n_cm_gt_reactive=int((r_react > 1.001).sum()),
                   feasible_sat=dict(n=int(len(feas)), min=float(r_spec.loc[feas].min()), median=float(r_spec.loc[feas].median()), max=float(r_spec.loc[feas].max())),
                   overload=dict(n=int(len(over)), min=float(r_spec.loc[over].min()), median=float(r_spec.loc[over].median()), max=float(r_spec.loc[over].max())),
                   light=dict(n=int(len(g2) - len(sat)), min=float(r_spec.drop(sat).min()), median=float(r_spec.drop(sat).median())))
    # V4
    ids = sorted(d3.id.unique())

    def max_surge(tol):
        ok = d3[d3.surge_viol_s <= tol].groupby(["id", "policy"]).surge.max().unstack()
        return ok.reindex(index=ids, columns=POLS).fillna(0.0)
    ok = max_surge(0)
    cm = ok["costmodel_lindy"]
    ge = (cm >= ok["osd_idle"] - 1e-9) & (cm >= ok["accdrop"] - 1e-9)
    ok10 = max_surge(10)
    ge10 = (ok10["costmodel_lindy"] >= ok10["osd_idle"] - 1e-9) & (ok10["costmodel_lindy"] >= ok10["accdrop"] - 1e-9)
    v["V4"] = dict(n=int(len(ok)), n_ge=int(ge.sum()), frac=float(ge.mean()), passed=bool(ge.mean() >= 0.8),
                   max_surge_mean={p: float(ok[p].mean()) for p in POLS}, losses=[int(i) for i in ok.index[~ge.values]],
                   vs_accdrop=dict(better=int((cm > ok["accdrop"] + 1e-9).sum()), equal=int((abs(cm - ok["accdrop"]) <= 1e-9).sum()),
                                   worse=int((cm < ok["accdrop"] - 1e-9).sum())),
                   vs_osd=dict(better=int((cm > ok["osd_idle"] + 1e-9).sum()), equal=int((abs(cm - ok["osd_idle"]) <= 1e-9).sum()),
                               worse=int((cm < ok["osd_idle"] - 1e-9).sum())),
                   tol10s=dict(n_ge=int(ge10.sum()), max_surge_mean={p: float(ok10[p].mean()) for p in POLS}))
    return v


def main(workers=8, verdict_only=False):
    os.makedirs(OUT, exist_ok=True)
    frozen = {f: sha(os.path.join(HERE, f)) for f in ["sim.py", "model.json"]}
    print("frozen", frozen, flush=True)
    if verdict_only:
        tr, d1, d2, d3 = (pd.read_csv(os.path.join(OUT, f"sim_heldout_{k}.csv")) for k in ("trace", "driftI", "driftII", "driftIII"))
        v = verdicts(tr, d1, d2, d3)
        v["frozen"] = frozen
        json.dump(v, open(os.path.join(OUT, "sim_heldout_verdict.json"), "w"), indent=1, default=str)
        print(json.dumps(v, indent=1, default=str))
        return
    with Pool(workers) as pool:
        tr = pd.DataFrame([r for rows in pool.map(job_trace, specs_trace()) for r in rows])
        tr.to_csv(os.path.join(OUT, "sim_heldout_trace.csv"), index=False)
        print("trace done", len(tr), flush=True)
        d1 = pd.DataFrame([r for rows in pool.map(job_drift, specs_drift_I()) for r in rows])
        d1.to_csv(os.path.join(OUT, "sim_heldout_driftI.csv"), index=False)
        print("drift I done", len(d1), flush=True)
        d2 = pd.DataFrame([r for rows in pool.map(job_drift, specs_drift_II()) for r in rows])
        d2.to_csv(os.path.join(OUT, "sim_heldout_driftII.csv"), index=False)
        print("drift II done", len(d2), flush=True)
        d3 = pd.DataFrame([r for rows in pool.map(job_drift, specs_drift_III()) for r in rows])
        d3.to_csv(os.path.join(OUT, "sim_heldout_driftIII.csv"), index=False)
        print("drift III done", len(d3), flush=True)
    v = verdicts(tr, d1, d2, d3)
    v["frozen"] = frozen
    json.dump(v, open(os.path.join(OUT, "sim_heldout_verdict.json"), "w"), indent=1, default=str)
    print(json.dumps(v, indent=1, default=str))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8, verdict_only="--verdict-only" in sys.argv)
