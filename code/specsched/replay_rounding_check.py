"""For every saturated tick where the replayed duty differs from the logged one, test whether the logged inputs'
rounding explains it: the controller logs acc_est to 3 decimals, H to 0.1 s and gen_tok_s to an integer, so the
true inputs lie within +-0.0005, +-0.05 s and +-0.5 tok/s. Re-evaluate the rule on a grid over that box (implemented
baseline) and report whether the logged duty is reachable, plus the size of V(0.5) against its two terms."""
import csv, itertools, json, sys
P = "/Applications/proj_hl/brainstorm/spec_decoding"
sys.path.insert(0, f"{P}/code/specsched"); sys.path.insert(0, f"{P}/code/specsched/sched")
from sim import Model
M = {k: Model(f"{P}/code/specsched/sched/{f}") for k, f in (("m1", "model.json"), ("m2", "model_m2.json"), ("mix", "model_mix.json"))}
def invert(m, dom, acc):
    pts = sorted((float(k), v) for k, v in m.dom[dom]["acc_curve"].items())
    if acc <= pts[0][1]: return 0.0
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 <= acc <= a1 and a1 > a0: return n0 + (n1 - n0) * (acc - a0) / (a1 - a0)
    return pts[-1][0]
def rule(m, dom, C, a, H, buf, gen, spec):
    m.cur_dom = dom if dom in m.fr else None
    ceil = max(m.dom[dom]["acc_curve"].values()); at = min(ceil, a + 0.8 * (ceil - a))
    tpr, opr = m.tok_per_req(dom), m.out_per_req(dom)
    need = max(0.0, invert(m, dom, at) - max(invert(m, dom, a), 0.0)) * tpr
    now = m.thr_at(C, spec, a, 0.0); tgt = max(m.thr_at(C, True, at, 0.0), m.thr_at(C, False, at, 0.0))
    feed = buf / 10.0 + gen * tpr / max(opr, 1.0); best, duty, terms = 0.0, 0.0, None
    for dd in (0.1, 0.25, 0.5):
        speed = min(m.rate * dd, feed)
        if speed <= 0 or need <= 0: continue
        T = need / speed; busy = min(dd, speed / m.rate)
        g, l = (tgt - now) * max(0.0, H - T), now * m.kappa(C) * busy * T
        if dd == 0.5: terms = (g, l)
        if g - l > best: best, duty = g - l, dd
    return duty, terms
rows = list(csv.DictReader(open(f"{P}/results/value_rule_replay/mismatches.csv")))
for x in rows:
    root_sc_seed = x["run"]; t = float(x["t"])
    d = f"{P}/data/runs/{root_sc_seed.split('/')[0]}/{root_sc_seed.split('/')[1]}/v6/{root_sc_seed.split('/')[2]}/controller.jsonl"
    r = next(json.loads(l) for l in open(d) if f'"t": {x["t"]}' in l or (json.loads(l).get("t") == t and "reason" in json.loads(l)))
    sc = root_sc_seed.split("/")[1]; m = M["mix" if "mix" in sc else ("m2" if sc.startswith("m2") else "m1")]
    reach = set(); terms0 = rule(m, r["dom"], r["C"], r["acc_est"], r["H"], r["buffered_tokens"], r["gen_tok_s"], r["spec"])[1]
    for da, dh, dg in itertools.product([-5e-4, -2.5e-4, 0, 2.5e-4, 5e-4], [-0.05, 0, 0.05], [-0.5, 0, 0.5]):
        reach.add(rule(m, r["dom"], r["C"], r["acc_est"] + da, r["H"] + dh, r["buffered_tokens"], r["gen_tok_s"] + dg, r["spec"])[0])
    g, l = terms0
    print(f"{root_sc_seed:30s} t={t:7.1f} logged={r['duty']} reachable={sorted(reach)} -> {'explained by rounding' if r['duty'] in reach else 'NOT explained'};"
          f" V(0.5)={g-l:+.0f} tokens vs gain {g:.0f} / loss {l:.0f} ({abs(g-l)/max(g,l):.2%})")
