"""Re-evaluate the frozen controller's saturated value rule (controller.py lines 302-333) offline on every logged
saturated tick of the v6 runs, once with the implemented baseline (current speculation state) and once with the
better of speculation on/off as the baseline; report reproduction, how often the current state was the worse mode,
decision flips, and whether every duty below 0.5 was data-limited."""
import glob, json, os, sys, collections
P = "/Applications/proj_hl/brainstorm/spec_decoding"
sys.path.insert(0, f"{P}/code/specsched/sched")
from sim import Model
MODELS = {k: Model(f"{P}/code/specsched/sched/{f}") for k, f in (("m1", "model.json"), ("m2", "model_m2.json"), ("mix", "model_mix.json"))}
K, FRAC, DUT = 10.0, 0.8, [0.1, 0.25, 0.5]

def jl(p):
    out = []
    for line in open(p):
        try: out.append(json.loads(line))
        except ValueError: pass
    return out

def invert(m, dom, acc):
    pts = sorted((float(k), v) for k, v in m.dom[dom]["acc_curve"].items())
    if acc <= pts[0][1]: return 0.0
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 <= acc <= a1 and a1 > a0: return n0 + (n1 - n0) * (acc - a0) / (a1 - a0)
    return pts[-1][0]

def decide(m, r, base_mode):
    dom, C, a, H = r["dom"], r["C"], r["acc_est"], r["H"]
    m.cur_dom = dom if dom in m.fr else None
    ceiling = max(m.dom[dom]["acc_curve"].values()); a_t = min(ceiling, a + FRAC * (ceiling - a))
    tpr, opr = m.tok_per_req(dom), m.out_per_req(dom)
    need = max(0.0, invert(m, dom, a_t) - max(invert(m, dom, a), 0.0)) * tpr
    on, off = m.thr_at(C, True, a, 0.0), m.thr_at(C, False, a, 0.0)
    now = (on if r["spec"] else off) if base_mode == "current" else max(on, off)
    tgt = max(m.thr_at(C, True, a_t, 0.0), m.thr_at(C, False, a_t, 0.0))
    feed = r["buffered_tokens"] / K + r["gen_tok_s"] * tpr / max(opr, 1.0)
    best_v, duty = 0.0, 0.0
    for dd in DUT:
        speed = min(m.rate * dd, feed)
        if speed <= 0 or need <= 0: continue
        T = need / speed; busy = min(dd, speed / m.rate)
        v = (tgt - now) * max(0.0, H - T) - now * m.kappa(C) * busy * T
        if v > best_v: best_v, duty = v, dd
    return duty, best_v, feed, (on > off)

rows = collections.defaultdict(collections.Counter)
roots = ["abl", "abl_mix", "abl_seg", "abl_wait"]
for root in roots:
    for d in sorted(glob.glob(f"{P}/data/runs/{root}/*/v6/r*")):
        if not os.path.exists(f"{d}/done.json"): continue
        sc = d.split("/")[-3]; key = "mix" if "mix" in sc else ("m2" if sc.startswith("m2") else "m1")
        m = MODELS[key]
        grp = f"{root}:{sc.split('_w')[0].split('_n')[0]}"
        for r in jl(f"{d}/controller.jsonl"):
            if r.get("reason") not in ("saturated_value", "saturated_no_value"): continue
            c = rows[grp]; c["ticks"] += 1
            dc, vc, feed, on_better = decide(m, r, "current")
            db, vb, _, _ = decide(m, r, "best")
            c["reproduced"] += int(abs(dc - r["duty"]) < 1e-9)
            worse = bool(r["spec"]) != on_better
            c["current_is_worse_mode"] += int(worse)
            c["train_logged"] += int(r["duty"] > 0)
            c["flip_train_decision"] += int((dc > 0) != (db > 0))
            c["flip_to_no_train"] += int(dc > 0 and db == 0)
            c["flip_to_train"] += int(dc == 0 and db > 0)
            if r["duty"] > 0 and r["duty"] < 0.5:
                c["low_duty"] += 1; c["low_duty_data_limited"] += int(m.rate * r["duty"] >= feed - 1e-6)
for g, c in rows.items():
    print(f"{g:22s} " + "  ".join(f"{k}={v}" for k, v in c.items()))
tot = collections.Counter()
for c in rows.values(): tot.update(c)
print("TOTAL", dict(tot))
