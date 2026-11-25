"""Fit the scheduler's system model from measured runs -> sched/model.json.

Pieces (all from data/runs, nothing hand-copied):
  base[C]      : S0 output tokens/s and TPOT (ms) per concurrency (chat eval prompts, M1 short regime)
  spec_step[C] : per-sequence step time with speculation, T_spec(C) = C * acc_meas(C) / thr_S1(C)  [ms]
                 -> gain(a, C) = a * T_base(C) / T_spec(C)   (linear in acceptance length a)
  frontier     : if runs/frontier/<m>_<dom>/S1_<head> sweeps exist, per-C linear fit gain = alpha*a + beta
                 for validation of the linear model
  kappa[C]     : serving-throughput loss per unit duty at concurrency C (from the duty sweep, spec on)
  trainer_rate : trainer tokens/s per unit duty (median over sweep)
  domains[d]   : learning curve a(N) (own domain), transfer-to-chat curve, tokens per request,
                 output tokens per request, draft-vocab coverage
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_summary(p):
    return json.load(open(p)) if os.path.exists(p) else []


def by_C(rows, key):
    return {int(r["C"]): r[key] for r in rows if r.get(key) is not None}


def fit(runs, model="m1"):
    m = {"model": model}
    s0 = load_summary(f"{runs}/c1/{model}_short/S0/summary.json")
    s1 = load_summary(f"{runs}/c1/{model}_short/S1/summary.json")
    thr0, tpot0 = by_C(s0, "output_throughput"), by_C(s0, "median_tpot_ms")
    p99_0 = by_C(s0, "p99_tpot_ms")
    thr1, acc1 = by_C(s1, "output_throughput"), by_C(s1, "acc_len")
    Cs = sorted(set(thr0) & set(thr1))
    m["C"] = Cs
    m["base"] = {str(c): dict(thr=thr0[c], tpot=tpot0[c], p99=p99_0[c]) for c in Cs}
    # per-sequence base step time: C tokens per step / thr  -> ms
    m["spec_step"] = {str(c): dict(T_base=1e3 * c / thr0[c], T_spec=1e3 * c * acc1[c] / thr1[c], acc_meas=acc1[c])
                      for c in Cs}
    # duty sweep (spec on): kept(d, C) = thr_S3(d)/thr_S1 ; kappa(C) = (1-kept)/d, averaged over duties
    kap = {}
    for d, tag in [(0.1, "S3_d0.1_sweep"), (0.25, "S3_d0.25_sweep"), (0.5, "S3_d0.5_sweep"), (1.0, "S3_d1")]:
        rows = load_summary(f"{runs}/c1/{model}_short/{tag}/summary.json")
        for r in rows:
            c = int(r["C"])
            if c in thr1:
                kap.setdefault(c, []).append((d, 1 - r["output_throughput"] / thr1[c], r.get("trainer_tok_s", 0) / d))
    m["kappa"] = {str(c): float(np.median([l / d for d, l, _ in v])) for c, v in sorted(kap.items())}
    m["kept_table"] = {str(c): [(d, 1 - l) for d, l, _ in sorted(v)] for c, v in sorted(kap.items())}
    m["trainer_rate_per_duty"] = float(np.median([t for v in kap.values() for _, _, t in v]))
    m["base_chat_acc"] = acc1[min(Cs)]   # stock head on the chat regime
    # domains
    m["domains"] = {}
    for ddir in sorted(glob.glob(f"{runs}/c2/{model}/*")):
        dom = os.path.basename(ddir)
        curve, chat_curve = {}, {}
        for ev in glob.glob(f"{ddir}/N*/eval.json"):
            e = json.load(open(ev))
            n = int(os.path.basename(os.path.dirname(ev))[1:])
            curve[n] = e["sets"][dom]["acc_len"]
            if "chat" in e["sets"]:
                chat_curve[n] = e["sets"]["chat"]["acc_len"]
        if not curve:
            continue
        tl = f"{ddir}/train_log.jsonl"
        tok_per_req = None
        if os.path.exists(tl):
            last = json.loads(open(tl).readlines()[-1])
            # packed ~4096 tokens per step (prompt+output, capped at max_len)
            tok_per_req = 4096.0 * last["step"] / max(last["requests"], 1)
        m["domains"][dom] = dict(acc_curve=dict(sorted(curve.items())), chat_curve=dict(sorted(chat_curve.items())),
                                 tokens_per_request=tok_per_req)
    # traffic stats: output tokens per request per domain (for converting req/s to tokens/s)
    return m


def add_traffic_stats(m, data):
    for dom in list(m["domains"]):
        p = f"{data}/traffic/{m['model']}/{dom}.jsonl"
        if os.path.exists(p):
            n_out, n_in = [], []
            for line in open(p):
                r = json.loads(line)
                n_out.append(r["n_out"])
                n_in.append(len(r["prompt_token_ids"]))
            m["domains"][dom]["out_tokens_per_request"] = float(np.mean(n_out))
            m["domains"][dom]["prompt_tokens_per_request"] = float(np.mean(n_in))
    return m


def add_frontier(m, runs):
    """Per-C linear fit of measured gain vs acceptance across checkpoint heads, when frontier sweeps exist."""
    fr = {}
    for gdir in sorted(glob.glob(f"{runs}/frontier/{m['model']}_*")):
        dom = os.path.basename(gdir).split("_", 1)[1]
        s0 = by_C(load_summary(f"{gdir}/S0/summary.json"), "output_throughput")
        pts = {}
        for hd in glob.glob(f"{gdir}/S1_*"):
            for r in load_summary(f"{hd}/summary.json"):
                c = int(r["C"])
                if c in s0 and r.get("acc_len"):
                    pts.setdefault(c, []).append((r["acc_len"], r["output_throughput"] / s0[c], os.path.basename(hd)))
        fit_c = {}
        for c, v in sorted(pts.items()):
            a = np.array([x[0] for x in v])
            g = np.array([x[1] for x in v])
            if len(v) >= 2 and np.ptp(a) > 0.05:
                alpha, beta = np.polyfit(a, g, 1)
                pred = alpha * a + beta
                fit_c[str(c)] = dict(alpha=float(alpha), beta=float(beta), n=len(v),
                                     max_abs_err=float(np.abs(pred - g).max()), points=[(float(x), float(y), h) for x, y, h in v])
            else:
                fit_c[str(c)] = dict(points=[(float(x), float(y), h) for x, y, h in v])
        if fit_c:
            fr[dom] = dict(thr_S0=s0, fit=fit_c)
    if fr:
        m["frontier"] = fr
    return m


if __name__ == "__main__":
    runs, data = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "m1"
    m = add_frontier(add_traffic_stats(fit(runs, model), data), runs)
    out = os.path.join(HERE, "model.json" if model == "m1" else f"model_{model}.json")
    json.dump(m, open(out, "w"), indent=1)
    print("wrote", out)
    print("C:", m["C"])
    print("kappa:", {k: round(v, 3) for k, v in m["kappa"].items()}, "trainer tok/s per duty:", round(m["trainer_rate_per_duty"]))
    for c in m["C"]:
        s = m["spec_step"][str(c)]
        print(f"C={c:>3} T_base={s['T_base']:.2f}ms T_spec={s['T_spec']:.2f}ms acc_meas={s['acc_meas']:.2f} "
              f"-> gain(a)= a*{s['T_base']/s['T_spec']:.3f}; break-even a*={s['T_spec']/s['T_base']:.2f}")
    for d, v in m["domains"].items():
        print(d, "acc_curve", {k: round(x, 2) for k, x in v["acc_curve"].items()}, "tok/req", round(v.get("tokens_per_request") or 0),
              "out/req", round(v.get("out_tokens_per_request") or 0))
    if "frontier" in m:
        for d, v in m["frontier"].items():
            print("frontier", d, {c: (round(f.get("alpha", 0), 3), round(f.get("beta", 0), 3), f.get("max_abs_err")) for c, f in v["fit"].items()})
