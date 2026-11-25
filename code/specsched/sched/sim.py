"""Minimal cost-model scheduler for co-located drafter self-evolution, evaluated by trace replay.

System model (all numbers from sched/model.json, fitted from measured runs by fit_model.py):
  - fluid serving model on one GPU: arrival lam (req/s) of domain `dom` -> operating concurrency C solving
    thr(C) = demand, where thr(C) = thr_base(C) * [gain(a,C) if spec] * (1 - kappa(C)*duty)
    gain(a, C) = a * T_base(C) / T_spec(C)   (linear in acceptance length a; validated by the frontier sweeps)
  - saturation: demand beyond thr(256) accumulates a request backlog (queueing delay -> SLO violation)
  - latency: TPOT(C) = T_base(C) or T_spec(C)/a, inflated by 1/(1 - kappa*duty)
  - data: every served request contributes its tokens to an on-policy buffer (FIFO, tagged by domain)
  - training at duty d consumes min(buffer, rate*d*dt) tokens; the head's acceptance on domain X follows the
    measured own-domain curve a_X(N_X); training on Y also moves a_X by a measured transfer fraction
  - objective: goodput = served output tokens in windows where TPOT <= SLO_tpot and queueing <= SLO_wait

Decisions every dt seconds: spec on/off, duty in DUTIES.
Policies: static_off, spec_only, osd_idle, fixed_interval, accdrop, costmodel(forecast=oracle|persist|lindy).
"""
import argparse
import json
import math
import os
from collections import deque

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DUTIES = [0.0, 0.1, 0.25, 0.5, 1.0]


class Model:
    def __init__(self, path=os.path.join(HERE, "model.json")):
        m = json.load(open(path))
        self.C = np.array(m["C"], dtype=float)
        self.lC = np.log2(self.C)
        self.thr = np.array([m["base"][str(int(c))]["thr"] for c in self.C])
        self.tb = np.array([m["spec_step"][str(int(c))]["T_base"] for c in self.C])
        self.ts = np.array([m["spec_step"][str(int(c))]["T_spec"] for c in self.C])
        self.kap = np.array([m["kappa"].get(str(int(c)), 0.5) for c in self.C])
        self.rate = m["trainer_rate_per_duty"]
        self.dom = m["domains"]
        self.base_chat_acc = m.get("base_chat_acc", 1.90)  # stock head on chat (C1 S1 at C=1)
        # per-domain measured frontier gain = alpha(C)*a + beta(C) (from checkpoint sweeps), else proportional
        self.fr = {}
        for d, v in (m.get("frontier") or {}).items():
            cs = sorted(int(c) for c, f in v["fit"].items() if "alpha" in f)
            if len(cs) >= 2:
                self.fr[d] = (np.log2(np.array(cs, dtype=float)),
                              np.array([v["fit"][str(c)]["alpha"] for c in cs]),
                              np.array([v["fit"][str(c)]["beta"] for c in cs]))
        self.cur_dom = None   # set by the runner; selects the frontier

    def _i(self, arr, C):
        return float(np.interp(np.log2(max(C, 1.0)), self.lC, arr))

    def thr_base(self, C): return self._i(self.thr, C)
    def T_base(self, C): return self._i(self.tb, C)
    def T_spec(self, C): return self._i(self.ts, C)
    def kappa(self, C): return self._i(self.kap, C)
    def gain(self, a, C):
        f = self.fr.get(self.cur_dom)
        if f is None:
            return a * self.T_base(C) / self.T_spec(C)
        lc = np.log2(max(C, 1.0))
        return float(np.interp(lc, f[0], f[1])) * a + float(np.interp(lc, f[0], f[2]))
    def tpot(self, C, spec, a, duty):
        # with speculation, per-token time = base step time / gain (consistent with the per-domain frontier)
        t = self.T_base(C) / max(self.gain(a, C), 1e-6) if spec else self.T_base(C)
        return t / max(1e-6, 1 - self.kappa(C) * duty)

    def thr_at(self, C, spec, a, duty):
        g = self.gain(a, C) if spec else 1.0
        return self.thr_base(C) * g * (1 - self.kappa(C) * duty)

    def acc(self, dom, N):
        cv = self.dom[dom]["acc_curve"]
        xs = np.array([math.log1p(float(k)) for k in cv]); ys = np.array(list(cv.values()))
        return float(np.interp(math.log1p(max(N, 0.0)), xs, ys))

    def transfer(self, dom):
        """fraction of own-domain acceptance gain that transfers to another domain (measured to chat)."""
        d = self.dom[dom]
        own = max(d["acc_curve"].values()) - list(d["acc_curve"].values())[0]
        ch = d.get("chat_curve") or {}
        if ch and own > 1e-6:
            return float(np.clip((max(ch.values()) - list(ch.values())[0]) / own, 0.0, 1.0))
        return 0.3

    def tok_per_req(self, dom): return self.dom[dom].get("tokens_per_request") or 300.0
    def out_per_req(self, dom): return self.dom[dom].get("out_tokens_per_request") or 300.0


class State:
    def __init__(self, model, N0=None):
        self.N = {d: 0.0 for d in model.dom}
        if N0:
            self.N.update(N0)
        self.buffer = deque()   # (dom, tokens); consumed newest-first (on-policy = most recent traffic)
        self.buf_tokens = 0.0
        self.backlog = 0.0      # queued output tokens (domain-agnostic once queued)
        self.regime_t = 0.0     # seconds since the last domain change (for lindy forecast)

    def acceptance(self, model, dom):
        """own-domain curve plus transfer from training on other domains."""
        a = model.acc(dom, self.N[dom])
        for d, n in self.N.items():
            if d != dom and n > 0:
                a += model.transfer(d) * (model.acc(d, n) - model.acc(d, 0)) * 0.5
        return a


def operating_point(model, demand_tok_s, spec, a, duty):
    """concurrency at which supply meets demand (fluid), capped at 256 (saturated)."""
    lo, hi = 1.0, 256.0
    if model.thr_at(hi, spec, a, duty) <= demand_tok_s:
        return hi, True
    if model.thr_at(lo, spec, a, duty) >= demand_tok_s:
        return max(1.0, demand_tok_s / max(model.thr_at(lo, spec, a, duty), 1e-6) * lo), False
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        if model.thr_at(mid, spec, a, duty) < demand_tok_s:
            lo = mid
        else:
            hi = mid
    return hi, False


def step(model, st, lam, dom, spec, duty, dt, slo_tpot, slo_wait):
    model.cur_dom = dom
    a = st.acceptance(model, dom)
    out_tok = model.out_per_req(dom)
    demand_tok = lam * out_tok + st.backlog / dt          # tokens/s to serve, incl. the queue
    C, sat = operating_point(model, demand_tok, spec, a, duty)
    supply = model.thr_at(C, spec, a, duty)
    served_tok = min(demand_tok, supply) * dt
    st.backlog = max(0.0, st.backlog + lam * out_tok * dt - served_tok)
    wait = st.backlog / max(supply, 1e-6)                  # time to drain the queue at the current service rate
    tp = model.tpot(C, spec, a, duty)
    ok = (tp <= slo_tpot) and (wait <= slo_wait)
    served_req = served_tok / out_tok
    # on-policy data
    if served_req > 0:
        st.buffer.append((dom, served_req * model.tok_per_req(dom)))
        st.buf_tokens += served_req * model.tok_per_req(dom)
        while st.buf_tokens > 4e6:
            d0, t0 = st.buffer.popleft(); st.buf_tokens -= t0
    # training: newest data first
    train_tok = min(st.buf_tokens, model.rate * duty * dt) if duty > 0 else 0.0
    rem = train_tok
    while rem > 0 and st.buffer:
        d0, t0 = st.buffer[-1]
        take = min(t0, rem)
        st.N[d0] += take / model.tok_per_req(d0)
        rem -= take; st.buf_tokens -= take
        if take >= t0:
            st.buffer.pop()
        else:
            st.buffer[-1] = (d0, t0 - take)
    # GPU time: serving busy time at the operating point (slowed by contention) + trainer time (full-rate equivalent),
    # capped at wall time. Under-load this is the quantity a replica-count / energy cost sees.
    gpu_s = min(dt, served_tok / max(model.thr_at(C, spec, a, duty), 1e-6) + train_tok / max(model.rate, 1e-6))
    return dict(C=C, sat=sat, a=a, tpot=tp, wait=wait, ok=ok, served_tok=served_tok, goodput=served_tok if ok else 0.0,
                train_tok=train_tok, gpu_s=gpu_s, spec=spec, duty=duty, dom=dom, lam=lam)


# ----------------------------------------------------------------------------- policies

def pol_static_off(ctx): return False, 0.0
def pol_spec_only(ctx): return True, 0.0


def pol_osd_idle(ctx, c_idle=16):
    return True, (1.0 if ctx["C"] < c_idle else 0.0)


def pol_fixed_interval(ctx, period=300, burst=30):
    return True, (1.0 if (ctx["t"] % period) < burst else 0.0)


def pol_accdrop(ctx, frac=0.95):
    m, st, dom = ctx["model"], ctx["state"], ctx["dom"]
    ceiling = max(m.dom[dom]["acc_curve"].values())
    return True, (1.0 if ctx["a"] < frac * ceiling else 0.0)


def pol_costmodel(ctx, horizon=1800, forecast="oracle"):
    """Three rules.
    1. speculation on iff it pays at the current load: gain(a, C) > 1 and TPOT within SLO (the frontier switch).
    2. training is free while the window stays within SLO (under-load: the only cost is latency): use the
       largest duty that keeps TPOT within SLO, but only if the head can still improve on this domain
       and the improvement can matter at loads expected in the horizon (structural-ceiling check).
    3. when the window is already outside the SLO (saturation), training costs served tokens now; train at the
       duty maximizing forecast value - immediate loss.
    """
    m, st, dom, a, dt = ctx["model"], ctx["state"], ctx["dom"], ctx["a"], ctx["dt"]
    slo_tpot = ctx["slo_tpot"]
    demand_now = ctx["lam"] * m.out_per_req(dom)
    demand_q = demand_now + st.backlog / dt            # what must be served this window incl. the queue
    # operating point implied by the current arrivals (the previous window's C is stale right after a switch)
    C0, _ = operating_point(m, demand_q, False, a, 0.0)
    Cs, _ = operating_point(m, demand_q, True, a, 0.0)
    # speculation on iff it raises throughput, unless it would push an otherwise-compliant window out of the SLO
    spec_now = m.gain(a, Cs) > 1.0 and (m.tpot(Cs, True, a, 0.0) <= slo_tpot or m.tpot(C0, False, a, 0.0) > slo_tpot)
    C = Cs if spec_now else C0
    saturated_deficit = C >= 255.9 and m.thr_at(C, spec_now, a, 0.0) < demand_q * 0.999
    ok_now = (m.tpot(C, spec_now, a, 0.0) <= slo_tpot and ctx.get("wait", 0.0) <= ctx.get("slo_wait", 1.0)
              and not saturated_deficit)
    fut = ctx["forecast"](ctx["t"], horizon, forecast, st)
    N_now = st.N[dom]
    ceiling = max(m.dom[dom]["acc_curve"].values())
    headroom = ceiling - m.acc(dom, N_now)
    if headroom <= 0.005 or not fut:
        return spec_now, 0.0
    # can the ceiling ever beat the frontier at the loads in the forecast? (zh-style structural limit)
    a_ceiling = st.acceptance(m, dom) + headroom
    useful = False
    for (lam_f, dom_f, w) in fut:
        if dom_f != dom:
            continue
        Cf, _ = operating_point(m, lam_f * m.out_per_req(dom_f), True, a_ceiling, 0.0)
        if m.gain(a_ceiling, Cf) > 1.0:
            useful = True
            break
    if not useful:
        return spec_now, 0.0
    if ok_now:
        # rule 2: largest duty whose own operating point still meets demand within the SLO (training is free
        # in goodput terms; the only cost is latency)
        d_free = 0.0
        for d in DUTIES[1:]:
            Cd, satd = operating_point(m, demand_q, spec_now, a, d)
            if not satd and m.tpot(Cd, spec_now, a, d) <= 0.9 * slo_tpot:   # 10% safety margin
                d_free = d
        return spec_now, d_free
    # rule 3: already outside SLO -> value-based. If the head is already good enough (demand met within the SLO
    # at the implied operating point), the violation is only the queue draining: let it drain.
    def goodput_rate(demand, a_, dom_):
        """tokens/s that count as goodput at steady state for acceptance a_ (spec per the frontier rule)."""
        m.cur_dom = dom_
        Cb, satb = operating_point(m, demand, False, a_, 0.0)
        Cs_, sats = operating_point(m, demand, True, a_, 0.0)
        use_spec = m.gain(a_, Cs_) > 1.0 and (m.tpot(Cs_, True, a_, 0.0) <= slo_tpot or m.tpot(Cb, False, a_, 0.0) > slo_tpot)
        Cx, satx, sp = (Cs_, sats, True) if use_spec else (Cb, satb, False)
        ok = (not satx or m.thr_at(Cx, sp, a_, 0.0) >= demand * 0.999) and m.tpot(Cx, sp, a_, 0.0) <= slo_tpot
        return (min(demand, m.thr_at(Cx, sp, a_, 0.0)) if ok else 0.0)
    if goodput_rate(demand_now, a, dom) > 0:
        m.cur_dom = dom
        return spec_now, 0.0
    # value of training at duty d for the next K seconds (data-limited), amortized against K windows of loss
    K = 10.0
    best_d, best_v = 0.0, 0.0
    for d in DUTIES[1:]:
        train_tok = min(st.buf_tokens + demand_now / m.out_per_req(dom) * m.tok_per_req(dom) * K, m.rate * d * K)
        if train_tok <= 0:
            continue
        dN = train_tok / m.tok_per_req(dom)
        da = m.acc(dom, N_now + dN) - m.acc(dom, N_now)
        if da <= 0:
            continue
        value = 0.0
        for (lam_f, dom_f, w) in fut:
            demand = lam_f * m.out_per_req(dom_f)
            a_f = st.acceptance(m, dom_f)
            da_f = da if dom_f == dom else da * m.transfer(dom) * 0.5
            value += (goodput_rate(demand, a_f + da_f, dom_f) - goodput_rate(demand, a_f, dom_f)) * w
        m.cur_dom = dom
        loss = min(m.thr_at(C, spec_now, a, 0.0) * m.kappa(C), demand_now * m.kappa(C)) * d * K
        if value - loss > best_v:
            best_v, best_d = value - loss, d
    return spec_now, best_d


POLICIES = {
    "static_off": pol_static_off,
    "spec_only": pol_spec_only,
    "osd_idle": pol_osd_idle,
    "fixed_interval": pol_fixed_interval,
    "accdrop": pol_accdrop,
    "costmodel_oracle": lambda ctx: pol_costmodel(ctx, forecast="oracle"),
    "costmodel_persist": lambda ctx: pol_costmodel(ctx, forecast="persist"),
    "costmodel_lindy": lambda ctx: pol_costmodel(ctx, forecast="lindy"),
}


# ----------------------------------------------------------------------------- scenarios

def make_forecaster(series, dt):
    """series: list of (lam, dom) per dt. Returns forecast(t, horizon, mode, state) -> [(lam, dom, seconds)]."""
    n = len(series)

    def fc(t, horizon, mode, st):
        i = int(t // dt)
        if mode == "oracle":
            j = min(n, i + int(horizon // dt))
            # split the horizon at domain changes, then bin each run into <=60 s segments (mean rate)
            k = max(1, int(60 // dt))
            out, s = [], i
            while s < j:
                e = s
                while e < j and series[e][1] == series[s][1] and e - s < k:
                    e += 1
                seg = series[s:e]
                out.append((float(np.mean([x[0] for x in seg])), series[s][1], len(seg) * dt))
                s = e
            return out
        lam, dom = series[min(i, n - 1)]
        if mode == "persist":
            return [(lam, dom, float(horizon))]
        if mode == "lindy":  # expected remaining time of the current regime ~ its elapsed time
            return [(lam, dom, float(min(horizon, max(st.regime_t, 30.0))))]
        raise ValueError(mode)
    return fc


def run(model, series, policy, dt=1.0, slo_tpot=None, slo_wait=1.0, N0=None, log=False):
    st = State(model, N0)
    slo_tpot = slo_tpot or 2.0 * model.T_base(1)
    fc = make_forecaster(series, dt)
    rows, last_dom = [], None
    C, a = 1.0, None
    for i, (lam, dom) in enumerate(series):
        t = i * dt
        st.regime_t = 0.0 if dom != last_dom else st.regime_t + dt
        last_dom = dom
        model.cur_dom = dom
        a = st.acceptance(model, dom)
        ctx = dict(model=model, state=st, t=t, dt=dt, lam=lam, dom=dom, C=C, a=a, slo_tpot=slo_tpot, slo_wait=slo_wait,
                   wait=rows[-1]["wait"] if rows else 0.0, forecast=fc)
        spec, duty = policy(ctx)
        r = step(model, st, lam, dom, spec, duty, dt, slo_tpot, slo_wait)
        C = r["C"]
        r["t"] = t
        rows.append(r)
    df = pd.DataFrame(rows)
    summary = dict(goodput=df.goodput.sum(), served=df.served_tok.sum(), slo_viol_s=float((~df.ok).sum() * dt),
                   gpu_s=df.gpu_s.sum(), gpu_s_per_ktok=1e3 * df.gpu_s.sum() / max(df.served_tok.sum(), 1.0),
                   train_tok=df.train_tok.sum(), duty_mean=df.duty.mean(), spec_frac=df.spec.mean(),
                   final_acc={d: round(st.acceptance(model, d), 3) for d in model.dom},
                   N={d: round(v) for d, v in st.N.items()})
    return summary, (df if log else None)


def drift_series(lam_A, lam_B, dom_A, dom_B, t_pre, dur_B, t_post, dt=1.0):
    s = [(lam_A, dom_A)] * int(t_pre / dt) + [(lam_B, dom_B)] * int(dur_B / dt) + [(lam_A, dom_A)] * int(t_post / dt)
    return s


def trace_series(path, kind, dom, model, hours, scale_p95=0.8, dt=1.0, start_hour=None):
    """Per-second arrival rates from a trace, scaled so that the p95 per-minute demand = scale_p95 * capacity."""
    if "BurstGPT" in path:
        b = pd.read_csv(path)
        b = b[b["Log Type"].str.contains("API" if kind == "api" else "Conversation")]
        ts = b["Timestamp"].values.astype(float)
        if start_hour is None:   # conversation log: steady diurnal from day 1; API log: bursty, use day 9-10
            start_hour = 24.0 if kind != "api" else 9 * 24.0
    else:  # Azure 2023 traces: TIMESTAMP,ContextTokens,GeneratedTokens
        b = pd.read_csv(path)
        ts = pd.to_datetime(b.iloc[:, 0]).values.astype("datetime64[ns]").astype(np.int64) / 1e9
        start_hour = start_hour or 0.0
    ts = ts - ts.min()
    t0 = start_hour * 3600
    ts = ts[(ts >= t0) & (ts < t0 + hours * 3600)] - t0
    counts = np.bincount((ts // dt).astype(int), minlength=int(hours * 3600 / dt)) / dt
    # smooth to 60 s to get the rate signal, then scale
    k = int(60 / dt)
    sm = np.convolve(counts, np.ones(k) / k, mode="same")
    cap = model.thr_base(256) / model.out_per_req(dom)
    p95 = np.quantile(sm, 0.95) if sm.max() > 0 else 1.0
    lam = sm * (scale_p95 * cap / p95)
    return [(float(x), dom) for x in lam]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("drift")
    d.add_argument("--A", default="chat"); d.add_argument("--B", default="sql")
    d.add_argument("--lamA", type=float, default=None); d.add_argument("--lamB", type=float, default=None)
    d.add_argument("--CA", type=float, default=16); d.add_argument("--CB", type=float, default=16)
    d.add_argument("--durations", default="10,30,60,300,1200,3600")
    d.add_argument("--pre", type=float, default=300); d.add_argument("--post", type=float, default=1800)
    d.add_argument("--NA", type=float, default=3200, help="head already adapted to A with this many requests")
    d.add_argument("--policies", default="spec_only,osd_idle,accdrop,costmodel_oracle,costmodel_persist,costmodel_lindy")
    t = sub.add_parser("trace")
    t.add_argument("--trace", default="burstgpt_conv"); t.add_argument("--hours", type=float, default=24)
    t.add_argument("--dom", default="chat"); t.add_argument("--scale", type=float, default=0.8)
    t.add_argument("--dt", type=float, default=10.0)
    t.add_argument("--policies", default="static_off,spec_only,osd_idle,fixed_interval,accdrop,costmodel_oracle,costmodel_persist")
    a = ap.parse_args()
    m = Model()
    if a.cmd == "drift":
        # arrival rates that put the base system at concurrency CA / CB
        lamA = a.lamA or m.thr_base(a.CA) / m.out_per_req(a.A)
        lamB = a.lamB or m.thr_base(a.CB) / m.out_per_req(a.B)
        print(f"lamA={lamA:.2f} req/s ({a.A}, C~{a.CA})  lamB={lamB:.2f} req/s ({a.B}, C~{a.CB})")
        N0 = {a.A: a.NA}
        for dur in [float(x) for x in a.durations.split(",")]:
            s = drift_series(lamA, lamB, a.A, a.B, a.pre, dur, a.post)
            print(f"\n== B={a.B} for {dur:.0f}s at C~{a.CB}")
            for p in a.policies.split(","):
                sm, _ = run(m, s, POLICIES[p], N0=dict(N0))
                print(f"  {p:>18}: goodput {sm['goodput']/1e6:7.2f}M  gpu_s/ktok {sm['gpu_s_per_ktok']:6.3f}  "
                      f"slo_viol {sm['slo_viol_s']:5.0f}s  train {sm['train_tok']/1e6:5.2f}M  duty {sm['duty_mean']:.3f}  "
                      f"spec {sm['spec_frac']:.2f}  acc_B {sm['final_acc'][a.B]:.2f}")
    else:
        traces = {"burstgpt_conv": ("BurstGPT_without_fails_1.csv", "conv"), "burstgpt_api": ("BurstGPT_without_fails_1.csv", "api"),
                  "azure_conv": ("AzureLLMInferenceTrace_conv.csv", "conv"), "azure_code": ("AzureLLMInferenceTrace_code.csv", "code")}
        path, kind = traces[a.trace]
        path = os.path.join(HERE, "..", "..", "..", "data", "traces", path)
        s = trace_series(path, kind, a.dom, m, a.hours, a.scale, dt=a.dt)
        lam = np.array([x[0] for x in s])
        print(f"trace {a.trace}: {len(s)} steps of {a.dt:g}s, lam mean {lam.mean():.2f} p50 {np.median(lam):.2f} "
              f"p95 {np.quantile(lam,.95):.2f} max {lam.max():.2f} req/s")
        for p in a.policies.split(","):
            sm, _ = run(m, s, POLICIES[p], dt=a.dt)
            print(f"  {p:>18}: goodput {sm['goodput']/1e6:8.2f}M  served {sm['served']/1e6:8.2f}M  gpu_s/ktok {sm['gpu_s_per_ktok']:6.3f}  "
                  f"slo_viol {sm['slo_viol_s']:6.0f}s  train {sm['train_tok']/1e6:6.2f}M  duty {sm['duty_mean']:.3f}  "
                  f"spec {sm['spec_frac']:.2f}  acc {sm['final_acc'][a.dom]:.2f}")


if __name__ == "__main__":
    main()
