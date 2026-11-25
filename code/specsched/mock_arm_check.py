"""Offline check of controller_arm.py (E28) against a deterministic mock world. No GPU, no server, no real time.

The controller modules are loaded unchanged; only their `time` and `urllib` names are replaced, so every decision
runs on a virtual clock against an in-process world: a serving model built from the calibration (sched/sim.Model),
a trainer that obeys the duty file and the export / flush commands, and a harness that moves the traffic label by
served-request counts (A1 200 requests at C=16 -> gap -> B 16,000 requests at C=256 -> drain -> A2 200 requests).

Checks
  1. `controller_arm.py --arm none` reproduces frozen controller.py record by record (ticks and events).
  2. every arm is identical to Full up to the label change into B.
  3. inside the B label period each arm follows its preregistered rule (duty, stop, exports, final export).
Usage: python mock_arm_check.py [--verbose]
"""
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sched"))
from sim import Model  # noqa: E402

VERBOSE = "--verbose" in sys.argv


class Done(BaseException):
    pass


class Resp:
    def __init__(self, b):
        self.b = b

    def read(self):
        return self.b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class World:
    def __init__(self, root, model_json, dom_a, dom_b, open_loop, cap_per_duty, nA=200, nB=16000, flat=False, fail=()):
        self.root = root
        os.makedirs(f"{root}/heads", exist_ok=True)
        self.m = Model(model_json) if model_json else Model()
        self.t0 = self.t = 1.7e9
        self.dom_a, self.dom_b, self.open_loop, self.cap = dom_a, dom_b, open_loop, cap_per_duty
        self.nA, self.nB = nA, nB
        self.flat, self.fail = flat, set(fail)   # flat: new heads do not move acceptance; fail: ticks whose scrape raises
        self.spec, self.head, self.N, self.heads = True, {}, {}, {}
        self.buffered = self.consumed = self.finished = 0.0
        self.c = dict(gen=0.0, drafts=0.0, acc=0.0, cnt=0.0, sum=0.0)
        self.phase, self.t_phase, self.f_phase = "pre", self.t, 0.0
        self.running = self.waiting = 0.0
        self.k = 0
        self.label = None
        self.tl = open(f"{root}/trainer.jsonl", "a")
        self.set_label(dom_a)
        self.rpcs = []
        self.label_times = {}

    def set_label(self, dom):
        self.label = dom
        open(f"{self.root}/domain.txt", "w").write(dom + "\n")

    def acc_now(self, dom):
        m = self.m
        if dom not in m.dom:
            return m.base_chat_acc
        if self.flat:
            return m.acc(dom, 0.0) + 0.01 * math.sin(self.k * 0.7)
        a = m.acc(dom, self.head.get(dom, 0.0))
        for d, n in self.head.items():
            if d != dom and n > 0 and d in m.dom:
                a += m.transfer(d) * (m.acc(d, n) - m.acc(d, 0)) * 0.5
        return a + 0.01 * math.sin(self.k * 0.7)

    def harness(self):
        """the frozen harness: phases move by served requests, the label is written at the phase change."""
        age, done = self.t - self.t_phase, self.finished - self.f_phase
        nxt = None
        if self.phase == "pre" and age >= 5:
            nxt = "A1"
        elif self.phase == "A1" and done >= self.nA:
            nxt = "gapB"
            self.set_label(self.dom_b)
            self.label_times["B"] = self.t
        elif self.phase == "gapB" and age >= 30:
            nxt = "B"
        elif self.phase == "B" and done >= self.nB - 256:
            nxt = "drainB"
        elif self.phase == "drainB" and age >= 7:
            nxt = "gapA2"
            self.set_label(self.dom_a)
            self.label_times["A2"] = self.t
        elif self.phase == "gapA2" and age >= 3:
            nxt = "A2"
        elif self.phase == "A2" and done >= self.nA:
            nxt = "end"
        elif self.phase == "end" and age >= 5:
            raise Done()
        if nxt:
            self.phase, self.t_phase = nxt, self.t
            if nxt in ("A1", "B", "A2"):
                self.f_phase = self.finished
        self.running = {"A1": 16.0, "A2": 16.0, "B": 256.0, "drainB": 120.0}.get(self.phase, 0.0)
        self.waiting = 600.0 if (self.open_loop and self.phase == "B") else 0.0

    def advance(self, dt):
        self.t += dt
        self.k += 1
        self.harness()
        cmdf = f"{self.root}/cmd.txt"
        if os.path.exists(cmdf):                       # the trainer handles commands at the top of its loop
            c = open(cmdf).read().strip()
            os.remove(cmdf)
            if c == "export":
                name = f"N{int(self.consumed)}"
                p = f"{self.root}/heads/{name}"
                if not os.path.exists(f"{p}/model.safetensors"):
                    os.makedirs(p, exist_ok=True)
                    open(f"{p}/model.safetensors", "w").write("x")
                    self.heads[name] = dict(self.N)
                    self.tl.write(json.dumps(dict(event="export", N=int(self.consumed), consumed=int(self.consumed))) + "\n")
            elif c == "flush":
                self.buffered = 0.0
                self.tl.write(json.dumps(dict(event="flush", consumed=int(self.consumed))) + "\n")
        duty = float(open(f"{self.root}/duty.txt").read().strip() or 0)
        cap = self.cap * duty * dt
        take = min(self.buffered, cap)
        busy = duty * (take / cap) if cap > 0 else 0.0
        self.buffered -= take
        self.consumed += take
        self.N[self.label] = self.N.get(self.label, 0.0) + take
        if self.running > 0:
            m, dom = self.m, self.label
            m.cur_dom = dom if dom in m.fr else None
            a = self.acc_now(dom)
            thr = m.thr_at(self.running, self.spec, a, busy) * (1 + 0.01 * math.sin(self.k * 1.3))
            gen = thr * dt
            opr = m.out_per_req(dom) if dom in m.dom else 438.0
            fin = gen / opr
            self.c["gen"] += gen
            self.c["cnt"] += fin
            self.c["sum"] += fin * m.tpot(self.running, self.spec, a, busy) / 1000.0
            if self.spec:
                self.c["drafts"] += gen / a
                self.c["acc"] += gen * (a - 1) / a
            self.finished += fin
            self.buffered = min(self.buffered + fin, 4000.0)
        tpr = self.m.tok_per_req(self.label) if self.label in self.m.dom else 510.0
        self.tl.write(json.dumps(dict(t=round(self.t - self.t0, 2), duty=duty, consumed=int(self.consumed),
                                      buffered=int(self.buffered), buffered_tokens=int(self.buffered * tpr),
                                      finished=int(self.finished))) + "\n")
        self.tl.flush()

    # --- what the controller sees ---
    def metrics(self):
        if self.k in self.fail:
            raise OSError("mock scrape failure")
        c = self.c
        return (f'vllm:num_requests_running{{model_name="target"}} {self.running}\n'
                f'vllm:num_requests_waiting{{model_name="target"}} {self.waiting}\n'
                f'vllm:generation_tokens_total{{model_name="target"}} {c["gen"]}\n'
                f'vllm:spec_decode_num_drafts_total{{model_name="target"}} {c["drafts"]}\n'
                f'vllm:spec_decode_num_accepted_tokens_total{{model_name="target"}} {c["acc"]}\n'
                f'vllm:time_per_output_token_seconds_count{{model_name="target"}} {c["cnt"]}\n'
                f'vllm:time_per_output_token_seconds_sum{{model_name="target"}} {c["sum"]}\n').encode()

    def rpc(self, body):
        r = json.loads(body)
        kw = {k: (os.path.basename(v) if k == "weights_path" else v) for k, v in r["kwargs"].items()}   # tmp dirs differ per run
        self.rpcs.append((round(self.t - self.t0, 2), r["method"], kw))
        if r["method"] == "set_speculation":
            self.spec = r["kwargs"]["enabled"] == "1"
            self.t += 0.05
        elif r["method"] == "reload_draft_weights":
            self.head = dict(self.heads[os.path.basename(r["kwargs"]["weights_path"])])
            self.t += 0.2
        else:
            raise ValueError(r["method"])
        return b"{}"


class FakeTime:
    def __init__(self, w):
        self.w = w

    def time(self):
        return self.w.t

    def sleep(self, dt):
        self.w.advance(dt)


class FakeRequest:
    def __init__(self, url, data=None, headers=None):
        self.url, self.data = url, data


class FakeUrllibRequest:
    Request = FakeRequest

    def __init__(self, w):
        self.w = w

    def urlopen(self, x, timeout=None):
        if isinstance(x, str):
            assert x.endswith("/metrics"), x
            return Resp(self.w.metrics())
        assert x.url.endswith("/collective_rpc"), x.url
        return Resp(self.w.rpc(x.data))


class FakeUrllib:
    def __init__(self, w):
        self.request = FakeUrllibRequest(w)


SCEN = OrderedDict(
    m1_code=dict(mj="sched/model.json", dom_b="code", cap=70.0),
    m1_zh=dict(mj="sched/model.json", dom_b="zh", cap=70.0),
    m2_code=dict(mj="sched/model_m2.json", dom_b="code", cap=96.0),
    m1_mixcc=dict(mj="sched/model_mix.json", dom_b="mixcc", cap=70.0),
    m1_mixzc=dict(mj="sched/model_mix.json", dom_b="mixzc", cap=70.0),
)
_n = [0]


def run(ctl, scen, extra, open_loop=False, policy="costmodel", fixed_duty="1.0", **world_kw):
    """one mock run; returns (records of controller.jsonl, world)."""
    sc = SCEN[scen]
    root = tempfile.mkdtemp(prefix="mockarm_")
    w = World(root, os.path.join(HERE, sc["mj"]), "chat", sc["dom_b"], open_loop, sc["cap"], **world_kw)
    open(f"{root}/duty.txt", "w").write("0\n")
    _n[0] += 1
    spec = importlib.util.spec_from_file_location(f"ctl_mock_{_n[0]}", os.path.join(HERE, ctl))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.time, mod.urllib = FakeTime(w), FakeUrllib(w)
    argv = ["x", "--port", "1", "--duty-file", f"{root}/duty.txt", "--cmd-file", f"{root}/cmd.txt",
            "--domain-file", f"{root}/domain.txt", "--heads", f"{root}/heads", "--trainer-log", f"{root}/trainer.jsonl",
            "--log", f"{root}/controller.jsonl", "--policy", policy, "--fixed-duty", fixed_duty, "--hot-swap", "1",
            "--max-duty", "1.0", "--max-seqs", "256", "--spec-switchable", "0", "--spec-switch", "1",
            "--model-json", os.path.join(HERE, sc["mj"])] + extra          # the harness' command line, verbatim
    old = sys.argv
    sys.argv = argv
    try:
        mod.main()
    except Done:
        pass
    finally:
        sys.argv = old
    recs = [json.loads(x) for x in open(f"{root}/controller.jsonl")]
    w.tl.close()
    shutil.rmtree(root)
    return recs, w


def same(r1, r2, keys=None):
    ks = keys or [k for k in r1 if k in r2]
    return all(r1.get(k) == r2.get(k) for k in ks)


def summary(recs, w):
    out = []
    st = recs[0]
    out.append("   start: " + json.dumps({k: st.get(k) for k in ("policy", "arm", "arm_domain", "arm_n", "arm_bn") if k in st}))
    groups = OrderedDict()
    for r in recs[1:]:
        if "event" in r:
            if r["event"] in ("stop",):
                continue
            key = None
            out_ev = {k: v for k, v in r.items() if k not in ("at",)}
            out_ev["t"] = round(out_ev["t"], 1)
            groups[("EV", len(groups))] = out_ev
            continue
        key = (r["dom"], r["reason"], r["duty"], r["spec"], r.get("in_arm"))
        last = next(reversed(groups)) if groups else None
        if last is not None and last[0] == "TK" and last[1] == key:
            g = groups[last]
            g["n"] += 1
            g["t1"], g["c1"] = r["t"], r["consumed"]
        else:
            groups[("TK", key, len(groups))] = dict(n=1, t0=r["t"], t1=r["t"], c0=r["consumed"], c1=r["consumed"])
    for k, g in groups.items():
        if k[0] == "EV":
            out.append(f"   {g}")
        else:
            dom, reason, duty, spec, in_arm = k[1]
            out.append(f"   {g['n']:4d} ticks  t {g['t0']:.0f}-{g['t1']:.0f}  consumed {g['c0']}-{g['c1']}  dom={dom} "
                       f"reason={reason} duty={duty} spec={'on' if spec else 'off'}" + (" [arm]" if in_arm else ""))
    return "\n".join(out)


def b_window(recs, dom_b):
    """index range of the records between the label change into B and the label change out of it."""
    i0 = next(i for i, r in enumerate(recs) if r.get("event") == "domain" and r["dom"] == dom_b)
    i1 = next((i for i, r in enumerate(recs) if i > i0 and r.get("event") == "domain"), len(recs))
    return i0, i1


def main():
    fails = []

    def check(cond, msg):
        print(("   ok   " if cond else "   FAIL ") + msg)
        if not cond:
            fails.append(msg)

    full = {}
    print("== 1. controller_arm.py --arm none vs frozen controller.py (record by record)")
    for scen in SCEN:
        for ol in (False, True):
            fr, w1 = run("controller.py", scen, [], ol)
            ar, w2 = run("controller_arm.py", scen, [], ol)
            n = min(len(fr), len(ar))
            bad = [i for i in range(n) if not same(fr[i], ar[i])]
            ticks = sum(1 for r in fr if "event" not in r)
            reasons = sorted({r["reason"] for r in fr if "event" not in r})
            check(len(fr) == len(ar) and not bad and w1.rpcs == w2.rpcs,
                  f"{scen:9s} {'open  ' if ol else 'closed'} {len(fr)} records, {ticks} ticks, {len(w1.rpcs)} RPCs identical; "
                  f"paths: {','.join(reasons)}")
            if bad:
                print("      first difference:", fr[bad[0]], ar[bad[0]])
            elif len(fr) != len(ar) or w1.rpcs != w2.rpcs:
                print(f"      lengths {len(fr)} vs {len(ar)}; RPCs {len(w1.rpcs)} vs {len(w2.rpcs)}")
                print("      first RPC difference:", next(((x, y) for x, y in zip(w1.rpcs, w2.rpcs) if x != y), None))
            full[(scen, ol)] = fr
            if VERBOSE and not ol:
                print(summary(fr, w1))
    print("== 1a. same check on streams where new heads do not help (stop test `converged`) and where scrapes fail")
    for scen in ("m1_code", "m2_code", "m1_mixcc"):
        for kw in (dict(flat=True), dict(fail=(40, 41, 150, 151, 152, 300)), dict(flat=True, fail=(60, 200))):
            fr, w1 = run("controller.py", scen, [], False, **kw)
            ar, w2 = run("controller_arm.py", scen, [], False, **kw)
            bad = [i for i in range(min(len(fr), len(ar))) if not same(fr[i], ar[i])]
            reasons = sorted({r["reason"] for r in fr if "event" not in r})
            check(len(fr) == len(ar) and not bad and w1.rpcs == w2.rpcs,
                  f"{scen:9s} {kw} {len(fr)} records identical; paths: {','.join(reasons)}")
            if bad:
                print("      first difference:", fr[bad[0]], ar[bad[0]])
    print("== 1b. frozen switch_only vs arm sw on MoE (no chat curve: nothing trains in A1)")
    fr, w1 = run("controller.py", "m2_code", [], False, policy="switch_only")
    ar, w2 = run("controller_arm.py", "m2_code", ["--arm", "sw", "--arm-domain", "code"], False, fixed_duty="0.5")
    keys = ["t", "C", "dom", "acc_est", "duty", "spec", "spec_should", "consumed", "swapped", "gen_tok_s"]
    tf, ta = [r for r in fr if "event" not in r], [r for r in ar if "event" not in r]
    check(len(tf) == len(ta) and all(same(x, y, keys) for x, y in zip(tf, ta)) and w1.rpcs == w2.rpcs,
          f"m2_code switch vs sw: {len(tf)} ticks identical in duty, speculation state, acceptance and throughput")

    arms = OrderedDict(sw=["--arm", "sw"], round800=["--arm", "round", "--arm-n", "800"], budget=["--arm", "budget"],
                       duty05=["--arm", "duty"], novalue=["--arm", "novalue"])
    expect_bn = dict(m1_code=428, m2_code=659, m1_mixcc=194, m1_zh=0, m1_mixzc=0)
    print("== 2/3. arms")
    for scen in SCEN:
        dom_b = SCEN[scen]["dom_b"]
        for ol in (False, True):
            ref = full[(scen, ol)]
            r0, _ = b_window(ref, dom_b)
            for arm, extra in arms.items():
                recs, w = run("controller_arm.py", scen, extra + ["--arm-domain", dom_b], ol, fixed_duty="0.5")
                tag = f"{scen:9s} {'open  ' if ol else 'closed'} {arm:9s}"
                i0, i1 = b_window(recs, dom_b)
                pre_ok = i0 == r0 and all(same(ref[i], recs[i]) for i in range(1, i0 + 1))
                inb = [r for r in recs[i0:i1] if "event" not in r]
                evb = [r for r in recs[i0:i1] if "event" in r]
                exports = [r for r in evb if r["event"] == "export_request"]
                c_dom = recs[i0]["consumed"]
                ok, note = True, ""
                if not all(r.get("in_arm") for r in inb) or any(r.get("in_arm") for r in recs[:i0] + recs[i1:] if "event" not in r):
                    ok, note = False, "in_arm flag does not match the label period"
                if arm == "sw":
                    ok = ok and all(r["duty"] == 0.0 and r["reason"] == "arm_sw" for r in inb) and not exports
                    note = f"{len(inb)} ticks duty 0, {len(exports)} exports"
                elif arm == "duty05":
                    ok = ok and all(r["duty"] == 0.5 and r["reason"] == "arm_duty" for r in inb)
                    note = f"{len(inb)} ticks duty 0.5, {len(exports)} exports, consumed {c_dom}-{inb[-1]['consumed']}"
                elif arm in ("round800", "budget"):
                    bn = 800 if arm == "round800" else expect_bn[scen]
                    ok = ok and recs[0]["arm_bn"] == bn
                    for r in inb:
                        used = r["consumed"] - c_dom
                        want = ("arm_train", 0.5) if (bn > 0 and used < bn) else (("arm_done" if bn > 0 else "arm_zero"), 0.0)
                        if (r["reason"], r["duty"]) != want:
                            ok, note = False, f"tick t={r['t']} used={used}: {r['reason']} {r['duty']} != {want}"
                            break
                    stop_i = next((i for i, r in enumerate(recs[i0:i1]) if r.get("reason") == "arm_done"), None)
                    after = [r for r in recs[i0:i1][stop_i:] if r.get("event") == "export_request"] if stop_i is not None else []
                    if stop_i is not None:
                        before = [r for r in recs[i0:i1][:stop_i] if r.get("event") == "export_request"]
                        last_exp = before[-1]["consumed"] if before else c_dom
                        c_stop = recs[i0:i1][stop_i]["consumed"]
                        need = c_stop - last_exp >= 100
                        ok = ok and len(after) == (1 if need else 0)
                        note = note or (f"bn={bn}: trains to +{c_stop - c_dom}, {len(before)} heads before the stop, "
                                        f"final export {'yes' if after else 'no'} ({c_stop - last_exp} learned since the last head)")
                    else:
                        note = note or f"bn={bn}: no training tick, {len(exports)} exports"
                        ok = ok and (bn == 0) and not exports
                elif arm == "novalue":
                    sat = [r for r in inb if r["saturated"]]
                    ok = ok and not any(r["reason"] in ("saturated_value", "saturated_no_value") for r in inb)
                    nfix = sum(1 for r in sat if r["reason"] == "saturated_fixed" and r["duty"] == 0.5)
                    note = f"{nfix}/{len(sat)} saturated ticks at fixed 0.5; other reasons: " + ",".join(
                        sorted({r["reason"] for r in inb if r["reason"] != "saturated_fixed"}))
                check(pre_ok and ok, f"{tag} same as Full before B: {pre_ok}; {note}")
                if VERBOSE and not ol:
                    print(summary(recs, w))
    print("\nRESULT:", "all checks passed" if not fails else f"{len(fails)} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
