"""Ablation variant of the frozen controller.py (copy of its code plus one switch, --ablate):
  none     identical to controller.py
  novalue  Full without the payback rule: under saturation the trainer runs at the latency-guard duty (0.5) whenever
           the stop tests allow it, instead of the value V(d) of the paper's Eq. (3)
  nostop   Full without the three explicit stop tests (structural ceiling, at ceiling, converged); the unknown-domain
           guard, the SLO-slack rule and the value rule stay
  nocurve  Full without any per-domain knowledge: the traffic label is ignored, the frontier is the per-target
           proportional model (gain = a * T_base / T_spec, as controller.py uses for a domain without a frontier),
           the next round is assumed to raise acceptance by --prior-gain and that prior is replaced by the measured
           gain of every swap; a round that does not pay (gain below --min-swap-gain) stops training for --retry-s
           seconds, after which one more round is tried with the prior
The frozen controller.py is not modified; this file carries the same code with the edits marked ABLATE.

Original docstring of controller.py:
Real-system controller: polls vLLM /metrics and the trainer's log, applies the cost-model rules, and owns every
decision of the self-adapting drafter: the trainer's duty cycle (duty file), when to export a head (command
file) and when to hot-swap it into the running server (POST /collective_rpc reload_draft_weights).

Observations per tick: running requests C, acceptance length (spec counters), measured TPOT (histogram),
served output tok/s, the traffic domain label (--domain-file, written by the operator/harness), the trainer's
state (consumed requests, buffered tokens). Model: sched/model.json (frontier, kappa, trainer rate, curves).

Rules (sched/sim.py:pol_costmodel ported to measured quantities):
  0. stop: no headroom on this domain (acc within 2% of the learning-curve ceiling) or structural ceiling
     (even the ceiling cannot beat the frontier at this load, e.g. zh at saturation) -> duty 0.
  1. frontier switch: speculation should be on iff gain(a, C) > 1 within SLO (advisory: vLLM cannot toggle
     speculation at runtime) -> logged as spec_should.
  2. within SLO: training is free in goodput terms -> largest duty keeping measured TPOT within 0.9 x SLO.
  3. outside SLO (saturation): training costs served tokens now -> duty maximizing
       (throughput gain of the improved head) x (forecast horizon - training window) - (tokens lost training)
     horizon = lindy estimate (the regime lasts as long as it has lasted, clipped to [--horizon-min, --horizon-max]).
     --sat-currency goodput reproduces the simulator's stricter objective (value only if the SLO can be restored).
Swap policy: request an export when the requests consumed since the last export reach a doubling schedule
(--swap-min, 2x, 4x ...) counted from the start of the current domain; swap every exported head immediately.
One JSON line per tick to --log (+ swap/export/domain events).
"""
import argparse
import json
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
    """Exact generated-token counter delta (the histogram series with the same stem would inflate it)."""
    for k in ("vllm:generation_tokens_total", "vllm:generation_tokens"):
        if k in d:
            return d[k]
    return sum(v for k, v in d.items() if k.endswith("generation_tokens_total") or k.endswith("generation_tokens"))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--duty-file", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--tick", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--domain", default="chat", help="traffic domain when no --domain-file is given")
    ap.add_argument("--domain-file", default="", help="re-read every tick: the operator's label of the current traffic")
    ap.add_argument("--policy", default="costmodel", choices=["costmodel", "osd_idle", "accdrop", "fixed", "switch_only", "simple"],
                    help="switch_only = rule 1 only, never trains; simple = threshold rules without the cost model: "
                         "speculation off iff saturated, train at --fixed-duty iff within SLO and not saturated")
    ap.add_argument("--fixed-duty", type=float, default=1.0)
    ap.add_argument("--max-duty", type=float, default=1.0)
    ap.add_argument("--spec-on", type=int, default=1, help="whether the server runs with speculation")
    ap.add_argument("--slo-mult", type=float, default=2.0)
    ap.add_argument("--hot-swap", type=int, default=0,
                    help="1 if the engine takes updated heads live (rule 3 is pointless otherwise)")
    ap.add_argument("--heads", default="", help="directory the trainer exports heads into; new heads are swapped in")
    ap.add_argument("--trainer-log", default="")
    ap.add_argument("--cmd-file", default="", help="write 'export' here to request a head export from the trainer")
    ap.add_argument("--sat-currency", default="throughput", choices=["throughput", "goodput"])
    ap.add_argument("--swap-min", type=int, default=100, help="requests consumed before the first export of a domain")
    ap.add_argument("--horizon-min", type=float, default=90.0,
                    help="prior on how long a new traffic regime lasts (s); with the measured numbers a full "
                         "adaptation breaks even after ~45 s of serving, so 90 s is the scale that makes it worth it")
    ap.add_argument("--horizon-max", type=float, default=600.0)
    ap.add_argument("--K", type=float, default=10.0, help="training window (s) evaluated by rule 3")
    ap.add_argument("--max-seqs", type=int, default=256, help="server max_num_seqs: saturation = running >= sat-frac x this")
    ap.add_argument("--sat-frac", type=float, default=0.95)
    ap.add_argument("--export-interval", type=float, default=5.0, help="min seconds between export requests")
    ap.add_argument("--model-json", default="", help="calibrated system model (default sched/model.json)")
    ap.add_argument("--target-frac", type=float, default=0.8, help="rule 3 aims at this fraction of the remaining headroom")
    ap.add_argument("--sat-max-duty", type=float, default=0.5,
                    help="latency guard while saturated: the trainer never gets more than this share (TPOT x 1/(1-kappa d))")
    ap.add_argument("--min-swap-gain", type=float, default=0.02,
                    help="a swapped head that raises measured acceptance by less than this ends training on the domain")
    ap.add_argument("--spec-switch", type=int, default=0,
                    help="1 = the engine exposes set_speculation (runtime switch): rule 1 is then enforced through "
                         "POST /collective_rpc, with a 3-tick / 10 s hysteresis; implies --spec-switchable 1")
    ap.add_argument("--spec-switchable", type=int, default=0,
                    help="1 if the engine can switch speculation off at runtime. Then a domain whose ceiling cannot "
                         "beat the no-speculation frontier is not worth training (rule 1 switches speculation off "
                         "instead); with speculation stuck on (vLLM today), training still raises throughput")
    ap.add_argument("--ablate", default="none", choices=["none", "novalue", "nostop", "nocurve"])   # ABLATE
    ap.add_argument("--prior-gain", type=float, default=0.3, help="nocurve: assumed acceptance gain of the next round")
    ap.add_argument("--retry-s", type=float, default=120.0, help="nocurve: seconds before a converged domain is retried")
    a = ap.parse_args()
    if a.ablate == "nocurve":
        a.domain_file = ""                 # ABLATE: label-free; the label is never read
    m = Model(a.model_json) if a.model_json else Model()
    tpr_default = float(np.mean([v.get("tokens_per_request") or 300.0 for v in m.dom.values()]))     # ABLATE nocurve
    opr_default = float(np.mean([v.get("out_tokens_per_request") or 300.0 for v in m.dom.values()]))
    delta, converged_until = a.prior_gain, 0.0
    slo_tpot = a.slo_mult * m.T_base(1)
    if a.spec_switch:
        a.spec_switchable = 1
    spec = bool(a.spec_on)                 # current engine state (changes only through set_speculation)
    t_switch, want_count, probe_until = -1e9, 0, 0.0

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

    def read_domain():
        if a.domain_file and os.path.exists(a.domain_file):
            try:
                d = open(a.domain_file).read().strip()
                if d:
                    return d        # may be a domain the model has no curve for: then nothing is trained
            except Exception:
                pass
        return a.domain

    def invert_curve(dom, acc):
        """Effective position on the domain's learning curve for a measured acceptance length."""
        cv = m.dom[dom]["acc_curve"]
        pts = sorted((float(k), v) for k, v in cv.items())
        if acc <= pts[0][1]:
            return 0.0
        for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
            if a0 <= acc <= a1 and a1 > a0:
                return n0 + (n1 - n0) * (acc - a0) / (a1 - a0)
        return pts[-1][0]

    prev, t0 = scrape(a.port), time.time()
    t_last = t0
    dom, t_dom = read_domain(), None      # the regime clock starts when traffic of the domain is observed
    acc_est = None
    duty = 0.0
    consumed_at_dom = last_export = 0
    next_export = a.swap_min
    pending, t_pending, t_export = False, 0.0, -1e9
    swapped = set()
    swap_eval, converged = None, False
    open(a.duty_file, "w").write("0\n")
    emit(t=0.0, event="start", dom=dom, slo_tpot_ms=slo_tpot, policy=a.policy, sat_currency=a.sat_currency)
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
        cnt = d.get("vllm:time_per_output_token_seconds_count", 0.0)
        tpot_meas = (d.get("vllm:time_per_output_token_seconds_sum", 0.0) / cnt * 1000) if cnt > 20 else None
        gen = gen_tokens(d) / max(win, 1e-6)
        ts = tl.poll()
        # domain label (operator) -> regime start, export schedule and acceptance estimate restart
        dnow = read_domain()
        waiting = cur.get("vllm:num_requests_waiting", 0.0)
        traffic = gen > 0 or C > 1 or waiting > 0
        if dnow != dom:
            dom, t_dom, acc_est = dnow, None, None
            swap_eval, converged = None, False
            consumed_at_dom = last_export = ts["consumed"]
            next_export = ts["consumed"] + a.swap_min
            pending = False
            if a.cmd_file:  # buffered sequences belong to the previous domain: drop them
                open(a.cmd_file, "w").write("flush\n")
            emit(t=now - t0, event="domain", dom=dom, consumed=ts["consumed"])
        if t_dom is None and traffic:
            t_dom = now
        if a_meas is not None:
            acc_est = a_meas if acc_est is None else 0.7 * acc_est + 0.3 * a_meas
        a_used = acc_est if acc_est is not None else m.base_chat_acc
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
                    emit(t=time.time() - t0, event="swap", head=hd, seconds=dt, dom=dom, acc_before=a_used)
                    swap_eval = dict(a=a_used, t=time.time(), head=hd)
                    acc_est = None          # re-measure the new head
                    if a.spec_switch and not spec:
                        set_spec(True, "probe_after_swap")     # measure the new head's acceptance
                        probe_until = time.time() + 8.0
                except Exception as e:
                    emit(t=time.time() - t0, event="swap_error", head=hd, error=str(e)[:200])
        # did the last swap pay? (measured on the same domain, after the estimate has settled)
        if swap_eval is not None and acc_est is not None and now - swap_eval["t"] > 8:
            realized = acc_est - swap_eval["a"]
            emit(t=now - t0, event="swap_eval", head=swap_eval["head"], realized=round(realized, 3), dom=dom)
            if realized < a.min_swap_gain:
                converged = True
                converged_until = now + a.retry_s           # ABLATE nocurve: retried after --retry-s
            if a.ablate == "nocurve":
                delta = max(realized, 0.0)                  # ABLATE nocurve: the measured gain replaces the prior
            swap_eval = None
        if a.ablate == "nocurve" and converged and now >= converged_until:    # ABLATE nocurve: one more round
            converged, delta = False, a.prior_gain
        # rules
        m.cur_dom = dom if dom in m.fr else None
        known = dom in m.dom
        ceiling = max(m.dom[dom]["acc_curve"].values()) if known else a_used
        headroom = ceiling - a_used
        useful = known and m.gain(ceiling, C) > 1.0
        if a.ablate == "nocurve":                           # ABLATE nocurve: no label, no curve, no domain frontier
            m.cur_dom, known = None, True
            ceiling, headroom = a_used + delta, delta
            useful = m.gain(ceiling, C) > 1.0
        tpot_now = tpot_meas if tpot_meas is not None else m.tpot(C, spec, a_used, duty)
        saturated = waiting > 0 or C >= a.sat_frac * a.max_seqs
        ok_now = tpot_now <= slo_tpot and not saturated
        kap = m.kappa(C)
        reason, H, best_v = "", None, 0.0
        if a.policy == "fixed":
            duty, reason = a.fixed_duty, "fixed"
        elif a.policy == "switch_only":
            duty, reason = 0.0, "switch_only"
        elif a.policy == "simple":
            # OSD's rule: train only while the service is not on a spike (TPOT within SLO, no queue)
            duty, reason = (a.fixed_duty if (tpot_now <= slo_tpot and not saturated) else 0.0), "simple_idle_rule"
        elif a.policy == "osd_idle":
            duty, reason = (a.max_duty if C < 16 else 0.0), "idle_rule"
        elif a.policy == "accdrop":
            duty, reason = (a.max_duty if a_used < 0.95 * ceiling else 0.0), "accdrop_rule"
        else:
            new_duty = 0.0
            if not traffic and ts["buffered_tokens"] <= 0:
                reason = "no_traffic"
            elif not known:
                reason = "unknown_domain"       # no learning curve for this traffic: cannot value training
            elif a.ablate != "nostop" and headroom <= 0.02:            # ABLATE nostop skips the three tests
                reason = "at_ceiling"
            elif a.ablate != "nostop" and converged:
                reason = "converged"           # the last swapped head did not move acceptance: stop paying
            elif a.ablate != "nostop" and not useful and a.spec_switchable:
                reason = "structural_ceiling"      # rule 1 will switch speculation off here instead
            elif ok_now:
                # rule 2: measured TPOT already carries the current duty's load; divide it out, then pick the
                # largest duty that stays within the SLO with a 10% margin
                tpot_free = tpot_now * (1 - kap * duty)
                for dd in DUTIES[1:]:
                    if dd <= a.max_duty and tpot_free / max(1e-6, 1 - kap * dd) <= 0.9 * slo_tpot:
                        new_duty = dd
                reason = "free_within_slo" if new_duty > 0 else "no_slo_room"
            elif not a.hot_swap:
                reason = "saturated_no_hotswap"
            elif a.ablate == "novalue":                     # ABLATE novalue: fixed duty instead of V(d)
                new_duty, reason = min(a.max_duty, a.sat_max_duty), "saturated_fixed"
            else:
                # rule 3 (time-to-target): train no faster than the data arrives, and only if reaching the
                # target acceptance within the forecast horizon pays for the tokens lost while training
                H = min(a.horizon_max, max(a.horizon_min, now - (t_dom if t_dom is not None else now)))
                K = a.K
                # where the head stands on this domain's curve, read off the measured acceptance (what was
                # consumed so far is a poor proxy: live data and transfer move the head faster than offline)
                if a.ablate == "nocurve":                   # ABLATE nocurve: one round at the prior gain
                    arrival_tok = gen * tpr_default / max(opr_default, 1.0)
                    thr_now = m.thr_at(C, spec, a_used, 0.0)
                    a_target = a_used + delta
                    tokens_needed = max(next_export - ts["consumed"], a.swap_min) * tpr_default
                else:
                    N_now = max(invert_curve(dom, a_used), 0.0)
                    arrival_tok = gen * m.tok_per_req(dom) / max(m.out_per_req(dom), 1.0)
                    thr_now = m.thr_at(C, spec, a_used, 0.0)
                    a_target = min(ceiling, a_used + a.target_frac * headroom)
                    tokens_needed = max(0.0, invert_curve(dom, a_target) - N_now) * m.tok_per_req(dom)
                thr_target = m.thr_at(C, spec, a_target, 0.0)
                if a.spec_switch:   # after adapting, the controller picks the better mode
                    thr_target = max(m.thr_at(C, True, a_target, 0.0), m.thr_at(C, False, a_target, 0.0))
                feed = ts["buffered_tokens"] / K + arrival_tok      # tokens/s the trainer can actually get
                for dd in DUTIES[1:]:
                    if dd > min(a.max_duty, a.sat_max_duty):
                        continue
                    speed = min(m.rate * dd, feed)
                    if speed <= 0 or tokens_needed <= 0:
                        continue
                    T = tokens_needed / speed
                    busy = min(dd, speed / m.rate)                  # the trainer idles when starved of data
                    if a.sat_currency == "goodput":
                        value = (thr_target if m.tpot(C, spec, a_target, 0.0) <= slo_tpot else 0.0) * max(0.0, H - T)
                    else:
                        value = (thr_target - thr_now) * max(0.0, H - T)
                    loss = thr_now * kap * busy * T
                    if value - loss > best_v:
                        best_v, new_duty = value - loss, dd
                reason = "saturated_value" if new_duty > 0 else "saturated_no_value"
            duty = new_duty
        open(a.duty_file, "w").write(f"{duty}\n")
        # export schedule: doubling gaps from the start of the domain; the trainer exports, the loop above swaps
        if (a.cmd_file and a.heads and a.policy != "switch_only" and ts["consumed"] >= next_export and not pending
                and now - t_export >= a.export_interval and not os.path.exists(a.cmd_file)):
            open(a.cmd_file, "w").write("export\n")
            pending, t_pending, t_export = True, now, now
            emit(t=now - t0, event="export_request", consumed=ts["consumed"], next_at=next_export)
            gap = max(a.swap_min, 2 * (ts["consumed"] - last_export))   # 100, 200, 400 ... requests between heads
            next_export = ts["consumed"] + gap
            last_export = ts["consumed"]
        if pending and now - t_pending > 120:
            pending = False
        if a.policy == "simple":
            spec_should = not saturated      # the load-threshold rule (vLLM's static per-batch-size table, live)
        else:
            spec_should = m.gain(a_used, C) > 1.0 and (m.tpot(C, True, a_used, 0.0) <= slo_tpot
                                                       or m.tpot(C, False, a_used, 0.0) > slo_tpot)
        # rule 1 enforced: switch after 3 consistent ticks, at most once per 10 s, never during a probe
        if a.spec_switch and traffic and now >= probe_until:
            want_count = want_count + 1 if spec_should != spec else 0
            if want_count >= 3 and now - t_switch >= 10.0:
                set_spec(spec_should, (f"threshold saturated={saturated} C={C:.0f}" if a.policy == "simple" else
                                       f"frontier gain={m.gain(a_used, C):.2f} C={C:.0f} a={a_used:.2f}"))
        emit(t=round(now - t0, 2), C=C, dom=dom, acc_est=round(a_used, 3), acc_meas=None if a_meas is None else round(a_meas, 3),
             tpot_ms=round(tpot_now, 2), gen_tok_s=round(gen), ok_now=ok_now, useful=useful, headroom=round(headroom, 3),
             duty=duty, reason=reason, H=None if H is None else round(H, 1), value=round(best_v), consumed=ts["consumed"],
             buffered=ts["buffered"], buffered_tokens=ts["buffered_tokens"], spec_should=spec_should, swapped=len(swapped),
             waiting=waiting, saturated=saturated, spec=spec, ablate=a.ablate, delta=round(delta, 3))
    open(a.duty_file, "w").write("0\n")
    emit(t=time.time() - t0, event="stop")


if __name__ == "__main__":
    main()
