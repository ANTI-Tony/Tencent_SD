"""Self-calibrating controller (v7): the frozen controller's loop without traffic labels and without per-domain
calibration. It reads only the per-target calibration of sched/model.json (step times T_base / T_spec, plain
throughput, interference kappa, trainer rate); the domain fields (learning curves, frontier fits, token counts)
are never used. The frozen controller.py is not modified; this is a new file.

Online estimates
  frontier       G(a, C) = c * a * T_base(C) / T_spec(C). The slope comes from the per-target step times; the level c
                 comes from a paired measurement inside the regime: the served token rate with speculation on (at
                 acceptance a_on) over the rate with speculation off, both at saturation and on clean ticks. When
                 speculation is on, no off-state rate was measured in the regime and the model puts the gain within
                 --probe-band of 1, the controller switches speculation off for 8 s to measure it (an off probe, the
                 mirror image of the frozen controller's probe after a swap). Without a pair, c = 1 (v7.1, 11-08:
                 v7.0 referenced the measured rate to the calibration's plain throughput, which is traffic-dependent)
  acceptance     EMA of the measured acceptance (weight 0.3), probed for 8 s after every swap (frozen rule)
  learning curve a(N) = a0 + D (1 - exp(-N / N0)) over the requests N learned in the regime; grid posterior over
                 (D, N0) with weak log-normal priors (--prior-d, --prior-n0), updated by the realized gain of every
                 swapped head (likelihood N(., --obs-sd^2))
  tokens/request EMA of (prompt + generated tokens) / finished requests from the metrics (fallback --tpr, --opr)
Decisions
  switch   rule 1 of the frozen controller with the online frontier, same hysteresis and post-swap probe
  train    under saturation, one-step look-ahead: train the next round iff the expected throughput gain of the next
           exported head, over the forecast horizon minus the training time, exceeds the throughput lost to
           interference (busy x T = n / r, independent of the duty); the duty is the data-limited maximum
           (<= --sat-max-duty). Light load: the frozen SLO-slack rule.
  ceiling  if speculation does not pay now and P[a0 + D > a*(C)] < --p-ceiling, nothing is trained (rule 1 switches
           speculation off instead)
  regime   label-free: an acceptance drift (|short EMA - long EMA| > --drift, not within 15 s after a swap) or a
           saturation flip that persists for 3 ticks starts a new regime: regime clock, posterior, export schedule
           and c are reset and the trainer's buffer is flushed
One JSON line per tick to --log (+ swap / export / regime events), as the frozen controller.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sched"))
from sim import DUTIES, Model  # noqa: E402


def scrape(port):
    txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    vals = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            vals[name] = vals.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except ValueError:
            pass
    return vals


def post_rpc(port, method, kwargs):
    body = json.dumps({"method": method, "kwargs": kwargs}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/collective_rpc", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        txt = r.read().decode()
    return time.time() - t0, txt


def gen_tokens(d):
    for k in ("vllm:generation_tokens_total", "vllm:generation_tokens"):
        if k in d:
            return d[k]
    return sum(v for k, v in d.items() if k.endswith("generation_tokens_total") or k.endswith("generation_tokens"))


def counter(d, *names):
    for k in names:
        if k in d:
            return d[k]
    return None


class TrainerLog:
    """Incremental reader of train_head.py's JSONL log (live mode): latest consumed / buffered counts."""

    def __init__(self, path):
        self.path, self.pos = path, 0
        self.state = dict(consumed=0, buffered=0, buffered_tokens=0, finished=0, exports=[])

    def poll(self):
        if not self.path or not os.path.exists(self.path):
            return self.state
        with open(self.path) as f:
            f.seek(self.pos)
            for line in f:
                if not line.endswith("\n"):
                    break
                self.pos += len(line)
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("event") == "export":
                    self.state["exports"].append(r)
                for k in ("consumed", "buffered", "buffered_tokens", "finished"):
                    if k in r:
                        self.state[k] = r[k]
        return self.state


class CurvePosterior:
    """Grid posterior over the learning curve a(N) = a0 + D (1 - exp(-N / N0)) of the current regime."""

    def __init__(self, prior_d, sd_d, prior_n0, sd_n0, obs_sd, n=48):
        self.D = np.exp(np.linspace(math.log(0.03), math.log(3.0), n))
        self.N0 = np.exp(np.linspace(math.log(20.0), math.log(8000.0), n))
        ld = (np.log(self.D) - math.log(prior_d)) / sd_d
        ln = (np.log(self.N0) - math.log(prior_n0)) / sd_n0
        self.logprior = -0.5 * (ld[:, None] ** 2 + ln[None, :] ** 2)
        self.obs_sd = obs_sd
        self.reset()

    def reset(self):
        self.logw = self.logprior.copy()
        self.rounds = []

    def _w(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def pred_gain(self, n1, n2):
        """Expected gain of the round from n1 to n2 requests under every grid point (matrix over (D, N0))."""
        e1 = np.exp(-n1 / self.N0)[None, :]
        e2 = np.exp(-n2 / self.N0)[None, :]
        return self.D[:, None] * (e1 - e2)

    def update(self, n1, n2, realized):
        self.rounds.append((n1, n2, realized))
        self.logw += -0.5 * ((realized - self.pred_gain(n1, n2)) / self.obs_sd) ** 2

    def expected_gain(self, n1, n2):
        return float((self._w() * self.pred_gain(n1, n2)).sum())

    def p_ceiling_above(self, needed):
        """P[D > needed]: the regime's ceiling a0 + D can reach the break-even acceptance."""
        w = self._w()
        return float(w[self.D > needed, :].sum())

    def summary(self):
        w = self._w()
        return dict(D=float((w * self.D[:, None]).sum()), N0=float((w * self.N0[None, :]).sum()), rounds=len(self.rounds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--duty-file", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--tick", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--domain", default="")                      # accepted and ignored (harness compatibility)
    ap.add_argument("--domain-file", default="")                 # accepted and ignored: v7 reads no label
    ap.add_argument("--policy", default="costmodel")             # accepted for compatibility; only costmodel exists
    ap.add_argument("--fixed-duty", type=float, default=1.0)
    ap.add_argument("--max-duty", type=float, default=1.0)
    ap.add_argument("--spec-on", type=int, default=1)
    ap.add_argument("--slo-mult", type=float, default=2.0)
    ap.add_argument("--hot-swap", type=int, default=0)
    ap.add_argument("--heads", default="")
    ap.add_argument("--trainer-log", default="")
    ap.add_argument("--cmd-file", default="")
    ap.add_argument("--sat-currency", default="throughput", choices=["throughput"])
    ap.add_argument("--swap-min", type=int, default=100)
    ap.add_argument("--horizon-min", type=float, default=90.0)
    ap.add_argument("--horizon-max", type=float, default=600.0)
    ap.add_argument("--K", type=float, default=10.0)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--sat-frac", type=float, default=0.95)
    ap.add_argument("--export-interval", type=float, default=5.0)
    ap.add_argument("--model-json", default="")
    ap.add_argument("--target-frac", type=float, default=0.8)    # accepted and ignored
    ap.add_argument("--sat-max-duty", type=float, default=0.5)
    ap.add_argument("--min-swap-gain", type=float, default=0.02)  # accepted and ignored: no explicit stop test
    ap.add_argument("--spec-switch", type=int, default=0)
    ap.add_argument("--spec-switchable", type=int, default=0)
    # v7
    ap.add_argument("--prior-d", type=float, default=0.45, help="prior median of the regime's total headroom D")
    ap.add_argument("--prior-d-sd", type=float, default=0.8, help="log-sd of the D prior")
    ap.add_argument("--prior-n0", type=float, default=300.0, help="prior median of the curve's scale N0 (requests)")
    ap.add_argument("--prior-n0-sd", type=float, default=0.9)
    ap.add_argument("--obs-sd", type=float, default=0.04, help="sd of a measured swap gain")
    ap.add_argument("--p-ceiling", type=float, default=0.25, help="train only if P[ceiling > break-even] >= this")
    ap.add_argument("--drift", type=float, default=0.25, help="acceptance drift that starts a new regime")
    ap.add_argument("--probe-band", type=float, default=0.12, help="off probe when |model gain - 1| is below this")
    ap.add_argument("--pair-age", type=float, default=600.0, help="an off-state rate older than this (s) is remeasured")
    ap.add_argument("--lookahead", type=int, default=4, help="rounds the training decision looks ahead (best of 1..K)")
    ap.add_argument("--pair-frontier", type=int, default=1, help="0 = never use the paired on/off rates (c = 1, v7.0)")
    ap.add_argument("--off-probe", type=int, default=1, help="0 = never probe the off state (v7.0)")
    ap.add_argument("--rate-ema", type=float, default=0.2, help="EMA weight of the measured token rates")
    ap.add_argument("--tpr", type=float, default=400.0, help="fallback tokens per request (prompt + output)")
    ap.add_argument("--opr", type=float, default=300.0, help="fallback output tokens per request")
    a = ap.parse_args()
    m = Model(a.model_json) if a.model_json else Model()
    slo_tpot = a.slo_mult * m.T_base(1)
    if a.spec_switch:
        a.spec_switchable = 1
    spec = bool(a.spec_on)
    t_switch, want_count, probe_until = -1e9, 0, 0.0
    post = CurvePosterior(a.prior_d, a.prior_d_sd, a.prior_n0, a.prior_n0_sd, a.obs_sd)
    c = 1.0                                   # frontier level from the paired on/off measurement (1 = proportional model)
    tpr, opr = a.tpr, a.opr                   # tokens per request (prompt + output), output tokens per request
    thr_on = thr_off = None                   # clean served token rates at saturation, speculation on / off (this regime)
    a_on, t_thr_off, t_off_probe = None, -1e9, -1e9

    def gain7(acc, C):
        return c * acc * m.T_base(C) / m.T_spec(C)

    def thr7(C, on, acc):
        base = thr_off if (thr_off is not None and C >= 0.9 * a.max_seqs) else m.thr_base(C)
        return base * (gain7(acc, C) if on else 1.0)

    def tpot7(C, on, acc, d):
        t = m.T_base(C) / max(gain7(acc, C), 1e-6) if on else m.T_base(C)
        return t / max(1e-6, 1 - m.kappa(C) * d)

    def a_star(C):
        return m.T_spec(C) / (c * m.T_base(C))

    def set_spec(on, why):
        nonlocal spec, t_switch, want_count
        try:
            dt, _ = post_rpc(a.port, "set_speculation", {"enabled": "1" if on else "0"})
        except Exception as e:
            emit(t=time.time() - t0, event="spec_error", error=str(e)[:200])
            return
        spec, t_switch, want_count = bool(on), time.time(), 0
        emit(t=time.time() - t0, event="spec", on=spec, why=why, seconds=round(dt, 3))

    tl = TrainerLog(a.trainer_log)
    log = open(a.log, "a")

    def emit(**kw):
        kw.setdefault("at", time.time())
        log.write(json.dumps(kw) + "\n")
        log.flush()

    prev, t0 = scrape(a.port), time.time()
    t_last = t0
    regime, t_dom = 0, None
    acc_est = acc_long = None
    a0 = None                                 # acceptance at the start of the regime (after transfer)
    duty = 0.0
    N_reg0 = last_export = 0
    next_export = a.swap_min
    pending, t_pending, t_export = False, 0.0, -1e9
    swapped, head_n = set(), {}
    swap_eval, t_last_swap = None, -1e9
    n_prev_head = 0                           # requests learned (in the regime) by the head currently served
    sat_state, sat_count = False, 0

    def new_regime(why, consumed):
        nonlocal regime, t_dom, acc_long, a0, N_reg0, last_export, next_export, pending, swap_eval, c, n_prev_head
        nonlocal thr_on, thr_off, a_on, t_thr_off, t_off_probe
        regime += 1
        t_dom, acc_long, a0 = None, None, None
        post.reset()
        c, thr_on, thr_off, a_on, t_thr_off, t_off_probe = 1.0, None, None, None, -1e9, -1e9
        N_reg0 = last_export = consumed
        next_export = consumed + a.swap_min
        n_prev_head = 0
        pending, swap_eval = False, None
        if a.cmd_file:                        # buffered sequences belong to the previous regime
            open(a.cmd_file, "w").write("flush\n")
        emit(t=time.time() - t0, event="regime", regime=regime, why=why, consumed=consumed)

    open(a.duty_file, "w").write("0\n")
    emit(t=0.0, event="start", slo_tpot_ms=slo_tpot, policy="v7", prior=dict(D=a.prior_d, N0=a.prior_n0))
    while a.duration <= 0 or time.time() - t0 < a.duration:
        time.sleep(a.tick)
        try:
            cur = scrape(a.port)
        except Exception:
            continue
        d = {k: cur.get(k, 0) - prev.get(k, 0) for k in cur}
        prev = cur
        now = time.time()
        win, t_last = now - t_last, now
        C = max(1.0, cur.get("vllm:num_requests_running", 1.0))
        drafts = sum(v for k, v in d.items() if "spec_decode_num_drafts" in k)
        acc = sum(v for k, v in d.items() if "spec_decode_num_accepted_tokens" in k and "per_pos" not in k)
        a_meas = (1 + acc / drafts) if drafts > 20 else None
        gen = gen_tokens(d) / max(win, 1e-6)
        ts = tl.poll()
        waiting = cur.get("vllm:num_requests_waiting", 0.0)
        traffic = gen > 0 or C > 1 or waiting > 0
        # tokens per request from the metrics, when the engine exposes the counters
        fin = counter(d, "vllm:request_success_total", "vllm:request_success")
        ptk = counter(d, "vllm:prompt_tokens_total", "vllm:prompt_tokens")
        if fin is not None and fin >= 5 and ptk is not None:
            gtk = gen_tokens(d)
            tpr = 0.9 * tpr + 0.1 * (ptk + gtk) / fin
            opr = 0.9 * opr + 0.1 * gtk / fin
        # acceptance estimates: short EMA (decisions) and long EMA (drift detection)
        if a_meas is not None:
            acc_est = a_meas if acc_est is None else 0.7 * acc_est + 0.3 * a_meas
            if acc_long is None:
                acc_long = acc_est
            else:
                acc_long = 0.98 * acc_long + 0.02 * a_meas
        a_used = acc_est if acc_est is not None else m.base_chat_acc
        if t_dom is None and traffic:
            t_dom = now
        if a0 is None and acc_est is not None:
            a0 = acc_est
        # regime detection without labels: saturation flip (5 consistent ticks) or acceptance drift
        saturated = waiting > 0 or C >= a.sat_frac * a.max_seqs
        sat_count = sat_count + 1 if saturated != sat_state else 0
        if sat_count >= 3 and traffic:
            sat_state, sat_count = saturated, 0
            new_regime("saturation_%s" % ("on" if saturated else "off"), ts["consumed"])
        elif (acc_long is not None and acc_est is not None and abs(acc_est - acc_long) > a.drift
              and now - t_last_swap > 15 and now >= probe_until):
            new_regime("acceptance_drift", ts["consumed"])
            acc_long, a0 = acc_est, acc_est
        # swap every new exported head
        if a.hot_swap and a.heads and os.path.isdir(a.heads):
            for hd in sorted(os.listdir(a.heads), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
                p = f"{a.heads}/{hd}"
                if hd in swapped or not os.path.exists(f"{p}/model.safetensors"):
                    continue
                swapped.add(hd)
                pending = False
                try:
                    dt, txt = post_rpc(a.port, "reload_draft_weights", {"weights_path": p})
                    n_head = int(hd[1:]) if hd[1:].isdigit() else ts["consumed"]
                    head_n[hd] = n_head
                    emit(t=time.time() - t0, event="swap", head=hd, seconds=dt, acc_before=a_used, n_head=n_head, regime=regime)
                    swap_eval = dict(a=a_used, t=time.time(), head=hd, n1=n_prev_head, n2=max(n_head - N_reg0, 0), regime=regime)
                    t_last_swap = time.time()
                    acc_est, acc_long = None, None          # re-measure the new head; drift detection restarts
                    thr_on, a_on = None, None               # and its token rate
                    if a.spec_switch and not spec:
                        set_spec(True, "probe_after_swap")
                        probe_until = time.time() + 8.0
                except Exception as e:
                    emit(t=time.time() - t0, event="swap_error", head=hd, error=str(e)[:200])
        # realized gain of the last swap -> posterior update (same regime only)
        if swap_eval is not None and acc_est is not None and now - swap_eval["t"] > 8:
            realized = acc_est - swap_eval["a"]
            if swap_eval["regime"] == regime and swap_eval["n2"] > swap_eval["n1"]:
                post.update(swap_eval["n1"], swap_eval["n2"], max(realized, -0.5))
                n_prev_head = swap_eval["n2"]
            emit(t=now - t0, event="swap_eval", head=swap_eval["head"], realized=round(realized, 3),
                 n1=swap_eval["n1"], n2=swap_eval["n2"], post=post.summary())
            swap_eval = None
        # paired on/off token rates at saturation (clean ticks: no swap in 8 s, 2 s after a switch, duty divided out)
        saturated_now = waiting > 0 or C >= a.sat_frac * a.max_seqs
        if traffic and gen > 0 and saturated_now and now - t_last_swap > 8 and now - t_switch > 2:
            rate = gen / max(1e-6, 1 - m.kappa(C) * duty)
            if spec and a_meas is not None:
                thr_on = rate if thr_on is None else (1 - a.rate_ema) * thr_on + a.rate_ema * rate
                a_on = a_used
            elif not spec:
                thr_off = rate if thr_off is None else (1 - a.rate_ema) * thr_off + a.rate_ema * rate
                t_thr_off = now
        if a.pair_frontier and thr_on is not None and thr_off is not None and a_on is not None and now - t_thr_off <= a.pair_age:
            c = min(1.5, max(0.5, (thr_on / thr_off) / max(a_on * m.T_base(C) / m.T_spec(C), 1e-6)))
        else:
            c = 1.0
        # off probe: speculation is on, the model puts it near break-even, and the off-state rate is unknown or stale
        if (a.off_probe and a.spec_switch and spec and traffic and saturated_now and now >= probe_until and now - t_last_swap > 10
                and now - t_switch > 20 and now - t_off_probe > a.pair_age and t_dom is not None and now - t_dom > 10
                and (thr_off is None or now - t_thr_off > a.pair_age) and abs(gain7(a_used, C) - 1.0) < a.probe_band):
            set_spec(False, "probe_off")
            probe_until, t_off_probe = time.time() + 8.0, time.time()
        # rules
        tpot_now = tpot7(C, spec, a_used, duty)
        ok_now = tpot_now <= slo_tpot and not saturated
        kap = m.kappa(C)
        reason, H, best_v, loss, e_gain, p_ceil = "", None, 0.0, 0.0, 0.0, None
        new_duty = 0.0
        N_now = max(ts["consumed"] - N_reg0, 0)
        N_next = max(next_export - N_reg0, N_now + a.swap_min)   # logged; the decision looks up to --lookahead rounds ahead
        if not traffic and ts["buffered_tokens"] <= 0:
            reason = "no_traffic"
        elif ok_now:
            tpot_free = tpot_now * (1 - kap * duty)
            for dd in DUTIES[1:]:
                if dd <= a.max_duty and tpot_free / max(1e-6, 1 - kap * dd) <= 0.9 * slo_tpot:
                    new_duty = dd
            reason = "free_within_slo" if new_duty > 0 else "no_slo_room"
        elif not a.hot_swap:
            reason = "saturated_no_hotswap"
        else:
            pays_now = gain7(a_used, C) > 1.0
            needed = a_star(C) - (a0 if a0 is not None else a_used)
            p_ceil = post.p_ceiling_above(needed) if needed > 0 else 1.0
            if a.spec_switchable and not pays_now and p_ceil < a.p_ceiling:
                reason = "ceiling_unlikely"              # rule 1 switches speculation off instead
            else:
                H = min(a.horizon_max, max(a.horizon_min, now - (t_dom if t_dom is not None else now)))
                thr_now = thr7(C, spec, a_used)
                arrival_tok = gen * tpr / max(opr, 1.0)
                feed = ts["buffered_tokens"] / a.K + arrival_tok
                d_max = min(a.max_duty, a.sat_max_duty)
                speed = min(m.rate * d_max, feed)
                # best of training 1..K more rounds (export gaps keep doubling): a single round may not cross
                # break-even where two would, so the decision is not myopic; stopping is still when no horizon pays
                gap = max(next_export - last_export, a.swap_min)
                n_abs, best_net = next_export, -1e18
                for k in range(1, a.lookahead + 1):
                    N_k = max(n_abs - N_reg0, N_now + a.swap_min)
                    g_k = post.expected_gain(N_now, N_k)
                    a_k = a_used + g_k
                    thr_k = max(thr7(C, True, a_k), thr7(C, False, a_k)) if a.spec_switch else thr7(C, spec, a_k)
                    tokens_k = max(n_abs - ts["consumed"], a.swap_min) * tpr
                    if speed > 0:
                        T = tokens_k / speed
                        busy = min(d_max, speed / m.rate)
                        v_k = (thr_k - thr_now) * max(0.0, H - T)
                        l_k = thr_now * kap * busy * T
                        if v_k - l_k > best_net:
                            best_net, best_v, loss, e_gain = v_k - l_k, v_k, l_k, g_k
                    n_abs += gap * 2 ** k
                if best_net > 0:
                    new_duty = d_max
                reason = "saturated_value" if new_duty > 0 else "saturated_no_value"
        duty = new_duty
        open(a.duty_file, "w").write(f"{duty}\n")
        # export schedule: doubling gaps from the start of the regime; rounds are sequential: no new export while
        # the last swapped head is still being measured (its realized gain is one posterior observation)
        if (a.cmd_file and a.heads and ts["consumed"] >= next_export and not pending and swap_eval is None
                and now - t_last_swap >= 8 and now - t_export >= a.export_interval and not os.path.exists(a.cmd_file)):
            open(a.cmd_file, "w").write("export\n")
            pending, t_pending, t_export = True, now, now
            emit(t=now - t0, event="export_request", consumed=ts["consumed"], next_at=next_export)
            gap = max(a.swap_min, 2 * (ts["consumed"] - last_export))
            next_export = ts["consumed"] + gap
            last_export = ts["consumed"]
        if pending and now - t_pending > 120:
            pending = False
        spec_should = gain7(a_used, C) > 1.0 and (tpot7(C, True, a_used, 0.0) <= slo_tpot or tpot7(C, False, a_used, 0.0) > slo_tpot)
        if a.spec_switch and traffic and now >= probe_until:
            want_count = want_count + 1 if spec_should != spec else 0
            if want_count >= 3 and now - t_switch >= 10.0:
                set_spec(spec_should, f"frontier gain={gain7(a_used, C):.2f} c={c:.2f} C={C:.0f} a={a_used:.2f}")
        emit(t=round(now - t0, 2), C=C, regime=regime, acc_est=round(a_used, 3), acc_meas=None if a_meas is None else round(a_meas, 3),
             acc_long=None if acc_long is None else round(acc_long, 3), a0=None if a0 is None else round(a0, 3),
             c=round(c, 3), thr_on=None if thr_on is None else round(thr_on), thr_off=None if thr_off is None else round(thr_off),
             tpot_ms=round(tpot_now, 2), gen_tok_s=round(gen), ok_now=ok_now, duty=duty, reason=reason,
             H=None if H is None else round(H, 1), value=round(best_v), loss=round(loss), e_gain=round(e_gain, 3),
             p_ceil=None if p_ceil is None else round(p_ceil, 3), N_now=N_now, N_next=N_next, tpr=round(tpr), consumed=ts["consumed"],
             buffered=ts["buffered"], buffered_tokens=ts["buffered_tokens"], spec_should=spec_should, swapped=len(swapped),
             waiting=waiting, saturated=saturated, spec=spec)
    open(a.duty_file, "w").write("0\n")
    emit(t=time.time() - t0, event="stop")


if __name__ == "__main__":
    main()
