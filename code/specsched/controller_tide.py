"""TIDE-style controller: the baseline for 'acceptance-triggered training' in the co-located setting.

It differs from the frozen controller.py only in WHEN and HOW MUCH it trains. Everything else is the same code:
  - measurements (/metrics deltas every second, acceptance moving average, TPOT) are taken exactly as in controller.py;
  - speculation follows the same frontier rule (rule 1 with the SLO clause, 3-tick / 10 s hysteresis, 8 s probe after
    every swap), selected by the same domain label, so SWITCH, FULL and this policy speculate identically;
  - every exported head is hot-swapped immediately, as in controller.py.

Training follows Algorithm 1 of TIDE (Park et al., SC'26, arXiv:2602.05145) instead of the cost-model rules:
  - collection is enabled when serving starts ("so that the draft model can adapt to the initial workload");
  - two exponential moving averages of the measured acceptance length, short and long, are initialised from the
    first --n-init measured seconds; whenever short < long - epsilon (a distribution shift), collection is enabled;
  - while collection is enabled the co-located trainer runs at --tide-duty (TIDE trains on separate GPUs; here it has
    to share the serving GPU, at the same duty as the other fixed-duty baselines);
  - a round ends when the trainer has learned --round requests since the last export: the head is exported and
    swapped in; once its acceptance has been measured (>= 8 s after the swap), it is kept if its acceptance is higher
    than that of the previous head just before the swap, and collection continues; otherwise the previous head is
    restored and collection is disabled until the next shift.
TIDE does not specify epsilon, the decay rates, N_init or N_threshold; the defaults below are ours. TIDE evaluates a
candidate offline on held-out samples of the round before deploying it; the frozen trainer has no evaluation path, so
here the candidate is evaluated live, on the current traffic, right after the swap. Live evaluation is only valid if
old and new head are measured on the same traffic: if the traffic label changed within --eval-wait seconds before the
swap or during the evaluation, the comparison is declared inconclusive and the new head is kept (TIDE would have
trained it on the mixed round and deployed it). Apart from this, the training logic never reads the label (TIDE has
none): no buffer flush and no reset on a label change.

Accepts every argument the harness (hotswap_run.py) passes to controller.py, so the harness can launch it unchanged.
One JSON line per tick to --log, with the fields of controller.py plus ema_short / ema_long / collecting.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sched"))
sys.path.insert(0, HERE)
from controller import TrainerLog, gen_tokens, post_rpc, scrape  # noqa: E402  (the frozen helpers, imported unchanged)
from sim import Model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    # ---- the arguments hotswap_run.py passes (same meaning as in controller.py)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--duty-file", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--tick", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--domain", default="chat")
    ap.add_argument("--domain-file", default="")
    ap.add_argument("--policy", default="tide", help="ignored (the harness passes its own policy name)")
    ap.add_argument("--fixed-duty", type=float, default=1.0, help="ignored")
    ap.add_argument("--max-duty", type=float, default=1.0, help="ignored")
    ap.add_argument("--spec-on", type=int, default=1)
    ap.add_argument("--slo-mult", type=float, default=2.0)
    ap.add_argument("--hot-swap", type=int, default=1)
    ap.add_argument("--heads", default="")
    ap.add_argument("--trainer-log", default="")
    ap.add_argument("--cmd-file", default="")
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--sat-frac", type=float, default=0.95)
    ap.add_argument("--export-interval", type=float, default=5.0)
    ap.add_argument("--model-json", default="")
    ap.add_argument("--spec-switch", type=int, default=0)
    ap.add_argument("--spec-switchable", type=int, default=0, help="ignored")
    # ---- TIDE-style training trigger
    ap.add_argument("--factory-head", default="", help="path of the server's initial drafter (restored if round 1 fails)")
    ap.add_argument("--tide-duty", type=float, default=0.5, help="trainer duty while collection is enabled")
    ap.add_argument("--round", type=int, default=200, help="requests learned per round (TIDE's N_threshold)")
    ap.add_argument("--ema-short", type=float, default=0.3, help="weight of the newest second in the short average")
    ap.add_argument("--ema-long", type=float, default=0.02, help="weight of the newest second in the long average")
    ap.add_argument("--epsilon", type=float, default=0.1, help="drift margin, acceptance-length units")
    ap.add_argument("--n-init", type=int, default=10, help="measured seconds that initialise both averages")
    ap.add_argument("--eval-wait", type=float, default=8.0, help="seconds after a swap before the head is judged")
    a = ap.parse_args()
    m = Model(a.model_json) if a.model_json else Model()
    slo_tpot = a.slo_mult * m.T_base(1)
    spec = bool(a.spec_on)
    t_switch, want_count, probe_until = -1e9, 0, 0.0
    log = open(a.log, "a")

    def emit(**kw):
        kw.setdefault("at", time.time())
        log.write(json.dumps(kw) + "\n")
        log.flush()

    def set_spec(on, why):
        nonlocal spec, t_switch, want_count
        try:
            dt, _ = post_rpc(a.port, "set_speculation", {"enabled": "1" if on else "0"})
        except Exception as e:
            emit(t=time.time() - t0, event="spec_error", error=str(e)[:200])
            return
        spec, t_switch, want_count = bool(on), time.time(), 0
        emit(t=time.time() - t0, event="spec", on=spec, why=why, seconds=round(dt, 3))

    def read_domain():
        if a.domain_file and os.path.exists(a.domain_file):
            try:
                d = open(a.domain_file).read().strip()
                if d:
                    return d
            except Exception:
                pass
        return a.domain

    tl = TrainerLog(a.trainer_log)
    prev, t0 = scrape(a.port), time.time()
    t_last = t0
    dom = read_domain()
    acc_est = None                  # the speculation rule's acceptance estimate (as in controller.py)
    ema_s = ema_l = None            # TIDE's drift detector (never reset by the label)
    init_vals = []
    collecting = True               # TIDE: collection enabled at start
    duty = a.tide_duty
    last_export = 0
    pending, t_pending, t_export = False, 0.0, -1e9
    swapped = set()
    current_head, previous_head = a.factory_head, a.factory_head
    evaluating = None               # dict(a_before, t, head, prev) while a swapped head is being judged
    n_round = 0
    t_domchange = -1e9
    open(a.duty_file, "w").write(f"{duty}\n")
    emit(t=0.0, event="start", dom=dom, slo_tpot_ms=slo_tpot, policy="tide", tide_duty=a.tide_duty, round=a.round,
         ema_short=a.ema_short, ema_long=a.ema_long, epsilon=a.epsilon, n_init=a.n_init)
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
        waiting = cur.get("vllm:num_requests_waiting", 0.0)
        traffic = gen > 0 or C > 1 or waiting > 0
        # the label only selects the frontier of the speculation rule (as for SWITCH / FULL)
        dnow = read_domain()
        if dnow != dom:
            dom, acc_est, t_domchange = dnow, None, now
            emit(t=now - t0, event="domain", dom=dom, consumed=ts["consumed"])
        if a_meas is not None:
            acc_est = a_meas if acc_est is None else 0.7 * acc_est + 0.3 * a_meas
            # TIDE drift detector
            if ema_s is None:
                init_vals.append(a_meas)
                if len(init_vals) >= a.n_init:
                    ema_s = ema_l = sum(init_vals) / len(init_vals)
            else:
                ema_s = (1 - a.ema_short) * ema_s + a.ema_short * a_meas
                ema_l = (1 - a.ema_long) * ema_l + a.ema_long * a_meas
                if ema_s < ema_l - a.epsilon and not collecting:
                    collecting = True
                    emit(t=now - t0, event="drift", ema_short=round(ema_s, 3), ema_long=round(ema_l, 3))
        a_used = acc_est if acc_est is not None else m.base_chat_acc
        # swap every new exported head (as controller.py), then judge it once its acceptance is measured
        if a.heads and os.path.isdir(a.heads):
            for hd in sorted(os.listdir(a.heads), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
                p = f"{a.heads}/{hd}"
                if hd in swapped or not os.path.exists(f"{p}/model.safetensors"):
                    continue
                swapped.add(hd)
                pending = False
                try:
                    dt, _ = post_rpc(a.port, "reload_draft_weights", {"weights_path": p})
                    emit(t=time.time() - t0, event="swap", head=hd, seconds=dt, dom=dom, acc_before=a_used)
                    evaluating = dict(a_before=a_used, t=time.time(), head=hd, path=p, prev=current_head)
                    previous_head, current_head = current_head, p
                    acc_est = None
                    if a.spec_switch and not spec:
                        set_spec(True, "probe_after_swap")
                        probe_until = time.time() + a.eval_wait
                except Exception as e:
                    emit(t=time.time() - t0, event="swap_error", head=hd, error=str(e)[:200])
        if evaluating is not None and acc_est is not None and now - evaluating["t"] > a.eval_wait:
            gain = acc_est - evaluating["a_before"]
            n_round += 1
            if t_domchange >= evaluating["t"] - a.eval_wait:
                emit(t=now - t0, event="tide_eval", head=evaluating["head"], gain=round(gain, 3), keep=True, round=n_round,
                     inconclusive=True)
            elif gain > 0:
                emit(t=now - t0, event="tide_eval", head=evaluating["head"], gain=round(gain, 3), keep=True, round=n_round)
            else:
                emit(t=now - t0, event="tide_eval", head=evaluating["head"], gain=round(gain, 3), keep=False, round=n_round)
                if evaluating["prev"]:
                    try:
                        dt, _ = post_rpc(a.port, "reload_draft_weights", {"weights_path": evaluating["prev"]})
                        emit(t=time.time() - t0, event="swap", head="restore:" + os.path.basename(evaluating["prev"]),
                             seconds=dt, dom=dom, acc_before=acc_est)
                        current_head = evaluating["prev"]
                        acc_est = None
                    except Exception as e:
                        emit(t=time.time() - t0, event="swap_error", head="restore", error=str(e)[:200])
                collecting = False
            evaluating = None
        # duty: fixed while collecting, otherwise 0 (TIDE has no latency or value test)
        duty = a.tide_duty if collecting else 0.0
        open(a.duty_file, "w").write(f"{duty}\n")
        # a round: export after --round requests learned since the last export, one head under evaluation at a time
        if (collecting and a.cmd_file and a.heads and evaluating is None and not pending
                and ts["consumed"] - last_export >= a.round and now - t_export >= a.export_interval
                and not os.path.exists(a.cmd_file)):
            open(a.cmd_file, "w").write("export\n")
            pending, t_pending, t_export = True, now, now
            last_export = ts["consumed"]
            emit(t=now - t0, event="export_request", consumed=ts["consumed"])
        if pending and now - t_pending > 120:
            pending = False
        # speculation: rule 1 of controller.py, unchanged
        m.cur_dom = dom if dom in m.fr else None
        saturated = waiting > 0 or C >= a.sat_frac * a.max_seqs
        tpot_now = tpot_meas if tpot_meas is not None else m.tpot(C, spec, a_used, duty)
        spec_should = m.gain(a_used, C) > 1.0 and (m.tpot(C, True, a_used, 0.0) <= slo_tpot
                                                   or m.tpot(C, False, a_used, 0.0) > slo_tpot)
        if a.spec_switch and traffic and now >= probe_until:
            want_count = want_count + 1 if spec_should != spec else 0
            if want_count >= 3 and now - t_switch >= 10.0:
                set_spec(spec_should, f"frontier gain={m.gain(a_used, C):.2f} C={C:.0f} a={a_used:.2f}")
        reason = "tide_collect" if collecting else "tide_off"
        emit(t=round(now - t0, 2), C=C, dom=dom, acc_est=round(a_used, 3), acc_meas=None if a_meas is None else round(a_meas, 3),
             tpot_ms=round(tpot_now, 2), gen_tok_s=round(gen), duty=duty, reason=reason, collecting=collecting,
             ema_short=None if ema_s is None else round(ema_s, 3), ema_long=None if ema_l is None else round(ema_l, 3),
             consumed=ts["consumed"], buffered=ts["buffered"], buffered_tokens=ts["buffered_tokens"],
             spec_should=spec_should, swapped=len(swapped), waiting=waiting, saturated=saturated, spec=spec)
    open(a.duty_file, "w").write("0\n")
    emit(t=time.time() - t0, event="stop")


if __name__ == "__main__":
    main()
