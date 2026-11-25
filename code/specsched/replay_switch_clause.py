"""Does the SLO clause of the switch rule (controller.py 349-350) ever override 'gain > 1' in the frozen model?"""
import glob, json, os, sys, collections
P = "/Applications/proj_hl/brainstorm/spec_decoding"
sys.path.insert(0, f"{P}/code/specsched/sched")
from sim import Model
M = {k: Model(f"{P}/code/specsched/sched/{f}") for k, f in (("m1", "model.json"), ("m2", "model_m2.json"), ("mix", "model_mix.json"))}
c = collections.Counter()
for root in ["abl", "abl_mix", "abl_seg", "abl_wait"]:
    for d in glob.glob(f"{P}/data/runs/{root}/*/v6/r*/controller.jsonl"):
        sc = d.split("/")[-4]; m = M["mix" if "mix" in sc else ("m2" if sc.startswith("m2") else "m1")]
        slo = None
        for line in open(d):
            r = json.loads(line)
            if r.get("event") == "start": slo = r["slo_tpot_ms"]; continue
            if "spec_should" not in r or "acc_est" not in r: continue
            m.cur_dom = r["dom"] if r["dom"] in m.fr else None
            C, a = r["C"], r["acc_est"]; g = m.gain(a, C)
            c["ticks"] += 1
            c["gain>1"] += int(g > 1)
            c["gain>1 but spec_should=False"] += int(g > 1 + 1e-3 and not r["spec_should"])
            c["clause false (Tspec>SLO and Tplain<=SLO)"] += int(m.tpot(C, True, a, 0) > slo and m.tpot(C, False, a, 0) <= slo)
            c["clause false while gain>1"] += int(g > 1 and m.tpot(C, True, a, 0) > slo and m.tpot(C, False, a, 0) <= slo)
print(dict(c))
