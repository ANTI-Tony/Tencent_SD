"""Collect C1/C2 results into results/Rung0_数据总结.xlsx (one sheet per experiment, with a source column).

Usage: python build_excel.py <local runs mirror> <results dir>
"""
import os
import sys

import pandas as pd

import analyze_c1
import analyze_c2


def main(runs, res):
    os.makedirs(f"{res}/figures", exist_ok=True)
    sheets = {}
    for g in sorted(os.listdir(f"{runs}/c1")) if os.path.isdir(f"{runs}/c1") else []:
        r = analyze_c1.analyze(f"{runs}/c1/{g}", f"{res}/figures")
        if r is None:
            continue
        raw = r["raw"].copy()
        raw.insert(0, "来源文件", f"runs/c1/{g}/<cond>/summary.json")
        der = r["derived"].copy()
        der.insert(0, "来源文件", f"runs/c1/{g} (derived)")
        der["SLO_tpot_ms"], der["SLO_ttft_ms"] = r["slo"]["tpot_ms"], r["slo"]["ttft_ms"]
        sheets[f"C1_{g}_raw"[:31]] = raw
        sheets[f"C1_{g}_derived"[:31]] = der
    for mname, sheet in [("m1", "C2_breakeven"), ("m2", "C2_breakeven_m2")]:
        if not os.path.isdir(f"{runs}/c2/{mname}"):
            continue
        c2 = analyze_c2.analyze(runs, f"{res}/figures", model=mname)
        if c2 is not None:
            c2.insert(0, "来源文件", f"runs/c2/{mname}/<domain>/N*/{{eval,train_stats}}.json")
            sheets[sheet] = c2
    # Rung 1: frontier sweeps, per-C linear fits, drift simulations, real-system duty control
    import glob
    import json
    rows = []
    for g in sorted(glob.glob(f"{runs}/frontier/m*_*")):
        mname, dom = os.path.basename(g).split("_", 1)
        s0 = {int(r["C"]): r["output_throughput"] for r in json.load(open(f"{g}/S0/summary.json"))} if os.path.exists(f"{g}/S0/summary.json") else {}
        for cd in sorted(glob.glob(f"{g}/S*")):
            for r in (json.load(open(f"{cd}/summary.json")) if os.path.exists(f"{cd}/summary.json") else []):
                rows.append(dict(来源文件=f"runs/frontier/{mname}_{dom}/{os.path.basename(cd)}/summary.json", model=mname, domain=dom,
                                 head=os.path.basename(cd), C=r["C"], acc_len=r.get("acc_len"), output_throughput=r["output_throughput"],
                                 speedup_vs_S0=(r["output_throughput"] / s0[int(r["C"])]) if int(r["C"]) in s0 else None,
                                 median_tpot_ms=r.get("median_tpot_ms"), p99_tpot_ms=r.get("p99_tpot_ms")))
    if rows:
        sheets["R1_frontier_raw"] = pd.DataFrame(rows)
    mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sched", "model.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        fr = [dict(来源文件="sched/model.json (fit_model.py)", domain=d, C=int(c), alpha=f.get("alpha"), beta=f.get("beta"),
                   n_heads=f.get("n"), max_abs_err=f.get("max_abs_err"),
                   a_star=((1 - f["beta"]) / f["alpha"]) if f.get("alpha") else None)
              for d, v in (m.get("frontier") or {}).items() for c, f in v["fit"].items()]
        if fr:
            sheets["R1_frontier_fit"] = pd.DataFrame(fr)
    for name in ["drift_sim", "drift_sim_chatA"]:
        p = f"{res}/summaries/{name}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            d.insert(0, "来源文件", f"results/summaries/{name}.csv (sched/drift_exp.py)")
            sheets[f"R1_{name}"[:31]] = d
    rs = []
    for pol_dir in sorted(glob.glob(f"{runs}/realsys/m1_chat/*")):
        p = f"{pol_dir}/summary.json"
        if os.path.exists(p):
            for r in json.load(open(p)):
                rs.append(dict(来源文件=f"runs/realsys/m1_chat/{os.path.basename(pol_dir)}/summary.json",
                               policy=os.path.basename(pol_dir), **{k: r[k] for k in ["phase", "C", "dur", "acc_len", "output_throughput",
                                                                                      "median_tpot_ms", "p99_tpot_ms"] if k in r},
                               trainer_duty_mean=r.get("trainer_duty_mean"), trainer_tok_s=r.get("trainer_tok_s")))
    if rs:
        sheets["R1_realsys"] = pd.DataFrame(rs)
    hs = []
    for mdir in sorted(glob.glob(f"{runs}/hotswap/*/*")):
        if not os.path.exists(f"{mdir}/events.jsonl"):
            continue
        ev = [json.loads(l) for l in open(f"{mdir}/events.jsonl")]
        if os.path.exists(f"{mdir}/controller.jsonl"):   # controller-driven swaps are logged by the controller
            ev += [dict(event="swap", head=c["head"], t=c["at"], seconds=c.get("seconds", 0))
                   for c in (json.loads(l) for l in open(f"{mdir}/controller.jsonl") if l.strip())
                   if c.get("event") == "swap" and "at" in c]
        tB = next((e["t"] for e in ev if e.get("event") == "phase" and e.get("name") == "B"), None)
        for ph in sorted(glob.glob(f"{mdir}/phase*.json")):
            r = json.load(open(ph))
            hs.append(dict(来源文件=f"runs/hotswap/{os.path.basename(os.path.dirname(mdir))}/{os.path.basename(mdir)}/{os.path.basename(ph)}",
                           mode=os.path.basename(mdir), phase=os.path.basename(ph)[:-5], duration_s=r.get("duration"),
                           output_tok_s=r.get("output_throughput"), median_tpot_ms=r.get("median_tpot_ms"),
                           p99_tpot_ms=r.get("p99_tpot_ms"),
                           swaps=";".join(f"{e['head']}@{e['t'] - tB:.1f}s/{e.get('seconds', 0):.2f}s" for e in ev
                                          if e.get("event") == "swap" and tB) or ""))
    if hs:
        sheets["R1_hotswap"] = pd.DataFrame(hs)
    # Rung 2: hot-swap under tensor parallelism
    tp = []
    for sj in sorted(glob.glob(f"{runs}/tp/*/summary.json")):
        for r in json.load(open(sj)):
            tp.append(dict(来源文件=f"runs/tp/{os.path.basename(os.path.dirname(sj))}/summary.json", **r))
    if tp:
        sheets["R2_tp_hotswap"] = pd.DataFrame(tp)
    # Rung 2: runtime speculation switch on one server (on -> off -> on)
    sw = []
    for sj in sorted(glob.glob(f"{runs}/switch/*/summary.json")):
        for r in json.load(open(sj)):
            sw.append(dict(来源文件=f"runs/switch/{os.path.basename(os.path.dirname(sj))}/summary.json",
                           group=os.path.basename(os.path.dirname(sj)), **r))
    if sw:
        sheets["R2_spec_switch"] = pd.DataFrame(sw)
    path = f"{res}/Rung0_数据总结.xlsx"
    with pd.ExcelWriter(path) as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
    print("wrote", path, list(sheets))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
