"""Calibration file for the mixed-traffic runs: sched/model_mix.json = the frozen sched/model.json plus the learning
curves, traffic statistics and frontier of the mixture domains (make_mix_domains.py), fitted with sched/fit_model.py's
own functions from runs/c2/m1/<mix> and runs/frontier/m1_<mix>. Every field of the frozen file is copied unchanged
(asserted), so the controller sees the frozen step times, interference and the frozen domains, plus the mixtures.
Usage: python fit_mix_model.py [mixcc,mixzc]
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sched"))
import fit_model  # noqa: E402
from cfg import DATA, RUNS  # noqa: E402

FROZEN = os.path.join(HERE, "sched", "model.json")
FROZEN_HASH = "18c3723adbe9b858"


def main(names):
    raw = open(FROZEN, "rb").read()
    assert hashlib.sha256(raw).hexdigest()[:16] == FROZEN_HASH, "sched/model.json is not the frozen file"
    base = json.loads(raw)
    fresh = fit_model.add_frontier(fit_model.add_traffic_stats(fit_model.fit(RUNS, "m1"), DATA), RUNS)
    out = json.loads(raw)
    for mix in names:
        assert mix in fresh["domains"], f"no learning curve for {mix} (runs/c2/m1/{mix})"
        assert mix in fresh.get("frontier", {}), f"no frontier for {mix} (runs/frontier/m1_{mix})"
        out["domains"][mix] = fresh["domains"][mix]
        out["frontier"][mix] = fresh["frontier"][mix]
    # nothing of the frozen file changed
    for k in base:
        if k not in ("domains", "frontier"):
            assert out[k] == base[k], k
    for d in base["domains"]:
        assert out["domains"][d] == base["domains"][d], d
    for d in base["frontier"]:
        assert out["frontier"][d] == base["frontier"][d], d
    out["mixtures"] = {mix: dict(source_file="sched/model.json", source_hash=FROZEN_HASH) for mix in names}
    path = os.path.join(HERE, "sched", "model_mix.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
    for mix in names:
        d, fr = out["domains"][mix], out["frontier"][mix]
        print(mix, "acc_curve", {k: round(v, 3) for k, v in d["acc_curve"].items()}, "ceiling", round(max(d["acc_curve"].values()), 3),
              "| chat_curve", {k: round(v, 3) for k, v in d.get("chat_curve", {}).items()},
              "| tok/req", round(d.get("tokens_per_request") or 0), "out/req", round(d.get("out_tokens_per_request") or 0))
        print("   frontier fit", {c: (round(f.get("alpha", 0), 3), round(f.get("beta", 0), 3), round(f.get("max_abs_err", 0), 3))
                                  for c, f in fr["fit"].items()})


if __name__ == "__main__":
    main((sys.argv[1] if len(sys.argv) > 1 else "mixcc,mixzc").split(","))
