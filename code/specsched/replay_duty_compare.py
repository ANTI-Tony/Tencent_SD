"""Duty-level comparison of the frozen value rule (controller.py 302-333) under two baselines on every saturated tick
of the frozen v6 runs: (A) as implemented, Theta_0 in the server's current speculation state; (B) the paper's Eq. 6,
Theta_0 in the better state (pi_mode). Reports, per tick, the logged duty, the replayed duty under A and under B,
V(0.5) under both, and the data-limit facts; writes worse_state_ticks.csv (the ticks where A and B differ in input)
and mismatches.csv (every tick where any of logged / A / B disagree). Analysis only."""
import csv, glob, json, os, sys, collections
P = "/Applications/proj_hl/brainstorm/spec_decoding"
sys.path.insert(0, f"{P}/code/specsched/sched")
from sim import Model
M = {k: Model(f"{P}/code/specsched/sched/{f}") for k, f in (("m1", "model.json"), ("m2", "model_m2.json"), ("mix", "model_mix.json"))}
K, FRAC, DUT = 10.0, 0.8, [0.1, 0.25, 0.5]
OUT = f"{P}/results/value_rule_replay"

def invert(m, dom, acc):
    pts = sorted((float(k), v) for k, v in m.dom[dom]["acc_curve"].items())
    if acc <= pts[0][1]: return 0.0
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 <= acc <= a1 and a1 > a0: return n0 + (n1 - n0) * (acc - a0) / (a1 - a0)
    return pts[-1][0]

def rule(m, r, base):
    dom, C, a, H = r["dom"], r["C"], r["acc_est"], r["H"]
    m.cur_dom = dom if dom in m.fr else None
    ceil = max(m.dom[dom]["acc_curve"].values()); at = min(ceil, a + FRAC * (ceil - a))
    tpr, opr = m.tok_per_req(dom), m.out_per_req(dom)
    need = max(0.0, invert(m, dom, at) - max(invert(m, dom, a), 0.0)) * tpr
    on, off = m.thr_at(C, True, a, 0.0), m.thr_at(C, False, a, 0.0)
    now = (on if r["spec"] else off) if base == "A" else max(on, off)
    tgt = max(m.thr_at(C, True, at, 0.0), m.thr_at(C, False, at, 0.0))
    feed = r["buffered_tokens"] / K + r["gen_tok_s"] * tpr / max(opr, 1.0)
    best_v, duty, vals = 0.0, 0.0, {}
    for dd in DUT:
        speed = min(m.rate * dd, feed)
        if speed <= 0 or need <= 0: continue
        T = need / speed; busy = min(dd, speed / m.rate)
        v = (tgt - now) * max(0.0, H - T) - now * m.kappa(C) * busy * T; vals[dd] = v
        if v > best_v: best_v, duty = v, dd
    return duty, vals.get(0.5), tgt - now, feed, m.rate, on > off

worse, mism, cnt = [], [], collections.Counter()
for root in ["abl", "abl_mix", "abl_seg", "abl_wait"]:
    for d in sorted(glob.glob(f"{P}/data/runs/{root}/*/v6/r*/controller.jsonl")):
        sc, seed = d.split("/")[-4], d.split("/")[-2]; m = M["mix" if "mix" in sc else ("m2" if sc.startswith("m2") else "m1")]
        for line in open(d):
            r = json.loads(line)
            if r.get("reason") not in ("saturated_value", "saturated_no_value"): continue
            dA, vA, gA, feed, rate, on_better = rule(m, r, "A")
            dB, vB, gB, _, _, _ = rule(m, r, "B")
            is_worse = bool(r["spec"]) != on_better
            row = dict(run=f"{root}/{sc}/{seed}", t=r["t"], spec=int(r["spec"]), C=r["C"], acc=r["acc_est"], H=r["H"],
                       feed_tok_s=round(feed), r_tok_s=round(rate), duty_logged=r["duty"], duty_A=dA, duty_B=dB,
                       V05_A=round(vA) if vA is not None else "", V05_B=round(vB) if vB is not None else "",
                       dTheta_A=round(gA, 1), dTheta_B=round(gB, 1), worse_state=int(is_worse))
            cnt["ticks"] += 1; cnt["A==logged"] += int(dA == r["duty"]); cnt["B==logged"] += int(dB == r["duty"]); cnt["A==B"] += int(dA == dB)
            if is_worse:
                worse.append(row); cnt["worse"] += 1
                cnt["worse A==logged"] += int(dA == r["duty"]); cnt["worse B==logged"] += int(dB == r["duty"]); cnt["worse A==B"] += int(dA == dB)
            if not (dA == dB == r["duty"]): mism.append(row)
for name, rows in (("worse_state_ticks.csv", worse), ("mismatches.csv", mism)):
    with open(f"{OUT}/{name}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(dict(cnt))
print("worse-state ticks by (logged, A, B) duty:", dict(collections.Counter((x["duty_logged"], x["duty_A"], x["duty_B"]) for x in worse)))
print(f"\nall ticks where logged / A / B are not all equal ({len(mism)}):")
for x in mism:
    print(f"  {x['run']:34s} t={x['t']:7.1f} spec={x['spec']} C={x['C']:5.0f} acc={x['acc']:.3f} H={x['H']} feed={x['feed_tok_s']} logged={x['duty_logged']} A={x['duty_A']} B={x['duty_B']} V05_A={x['V05_A']} V05_B={x['V05_B']} worse={x['worse_state']}")
