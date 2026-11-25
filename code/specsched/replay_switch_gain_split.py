"""Does the frozen value rule credit switch-only gains to training? (1) On every saturated tick of the frozen v6
runs where the server sat in the worse speculation state: how often it trained, the switch-only share of the
computed gain dTheta = Theta*(a_target) - Theta_current(a), whether the training-only gain (baseline = better state,
the paper's Eq. 6) stays positive, seconds since the label change and since the last swap. (2) Split Full's
phase-B gain over Static into the switch part (Switch vs Static) and the training part (Full - Switch), in points
of Static's mean throughput. Analysis only; the frozen controller is not touched."""
import glob, json, sys, collections, statistics as st
import pandas as pd
P = "/Applications/proj_hl/brainstorm/spec_decoding"
sys.path.insert(0, f"{P}/code/specsched/sched")
from sim import Model
M = {k: Model(f"{P}/code/specsched/sched/{f}") for k, f in (("m1", "model.json"), ("m2", "model_m2.json"), ("mix", "model_mix.json"))}
c, where, shares, ages, after_swap = collections.Counter(), collections.Counter(), [], [], 0
for root in ["abl", "abl_mix", "abl_seg", "abl_wait"]:
    for d in glob.glob(f"{P}/data/runs/{root}/*/v6/r*/controller.jsonl"):
        sc = d.split("/")[-4]; m = M["mix" if "mix" in sc else ("m2" if sc.startswith("m2") else "m1")]
        t_dom, t_swap = None, -1e9
        for line in open(d):
            r = json.loads(line)
            if r.get("event") == "domain": t_dom = r["t"]
            if r.get("event") == "swap": t_swap = r["at"]
            if r.get("reason") not in ("saturated_value", "saturated_no_value"): continue
            c["saturated_ticks"] += 1
            dom, C, a = r["dom"], r["C"], r["acc_est"]; m.cur_dom = dom if dom in m.fr else None
            on, off = m.thr_at(C, True, a, 0.0), m.thr_at(C, False, a, 0.0); cur = on if r["spec"] else off
            if cur >= max(on, off) - 1e-9: continue
            ceil = max(m.dom[dom]["acc_curve"].values()); at = min(ceil, a + 0.8 * (ceil - a))
            tgt = max(m.thr_at(C, True, at, 0.0), m.thr_at(C, False, at, 0.0))
            c["worse_state_ticks"] += 1; c["of_which_trained"] += int(r["duty"] > 0); c["training_only_gain_positive"] += int(tgt - max(on, off) > 0)
            where[f"{root}:{sc.split('_w')[0].split('_n')[0]}:{'spec_on' if r['spec'] else 'spec_off'}"] += 1
            if tgt - cur > 0: shares.append((max(on, off) - cur) / (tgt - cur))
            ages.append(r["t"] - t_dom); after_swap += int(r["at"] - t_swap < 12)
print("(1)", dict(c)); print("   by run set:", dict(where))
print(f"   switch-only share of the computed gain: median {st.median(shares):.0%}, max {max(shares):.0%}; seconds since label change {min(ages):.0f}-{max(ages):.0f} (median {st.median(ages):.0f}); within 12 s after a swap: {after_swap}")
print("(2) Full vs Static = switch part + training part (points of Static's mean phase-B throughput)")
for f, scs in ((f"{P}/results/abl_all/abl_runs.csv", ["m1_code", "m1_zh", "m2_code"]), (f"{P}/results/abl_mix/cell_runs.csv", ["m1_mixcc", "m1_mixzc"])):
    df = pd.read_csv(f)
    for sc in scs:
        x = df[df.scenario == sc]; S, W, F = (x[x.strategy == s].B_thr.mean() for s in ("static", "switch", "v6"))
        print(f"   {sc:9s} {100*(F-S)/S:+6.1f} = {100*(W-S)/S:+6.1f} + {100*(F-W)/S:+5.1f}")
