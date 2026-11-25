"""SGLang calibration with the vLLM study's own scripts, imported unchanged:
  c1        c1_run.py: load sweeps without (S0) and with (S1) speculation, and with a co-located trainer at duty d
            (S3_d<d>): step times T_base(C), T_spec(C) and the interference coefficient kappa(C)
  frontier  frontier_run.py: load x checkpoint sweeps on a domain's prompts: throughput gain against acceptance
  fit       sched/fit_model.py's functions -> sched/model_sgl_<model>.json (the vLLM files are not touched)
The server is SGLang through the patched launcher (sgl_common.start_sgl_server; vLLM's memory utilisation u maps to
SGLang's static fraction, sgl_common.mem_fraction) and the output root is runs/sgl/. Plain decoding (S0, frontier
"none") is measured by default (--s0 off) as the controller gets it at run time: the speculative server with
speculation switched off, which in SGLang is slower than a server without a drafter (no overlap scheduler with a
drafter loaded); --s0 server measures the latter instead (give it a --tag). The heads of the learning
curves (runs/c2) and the trainer's hidden-state caches (runs/cache) do not depend on the engine: runs/sgl/c2 and
runs/sgl/cache link to them, so the learning curves in the SGLang model are the vLLM study's.
Usage: python sgl_calib.py c1 <c1_run.py args> | frontier <frontier_run.py args> | fit --model m1
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cfg import DATA, RUNS  # noqa: E402

SGL_RUNS = f"{RUNS}/sgl"
ENGINE = "sglang 0.5.10.post1 (EAGLE-3 spec v1, overlap scheduler off), sgl/specsched_sgl.py"


def links():
    os.makedirs(SGL_RUNS, exist_ok=True)
    for d in ("c2", "cache"):
        if not os.path.lexists(f"{SGL_RUNS}/{d}"):
            os.symlink(f"{RUNS}/{d}", f"{SGL_RUNS}/{d}")


def sgl_start_server(model_name, s0):
    from c1_run import get
    from controller import post_rpc
    from sgl_common import mem_fraction, start_sgl_server

    def start_server(mc, port, util, maxlen, spec, nspec, logpath, tp=1, spec_schedule=None):
        assert tp == 1 and not spec_schedule
        total = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout.split()[0]) / 1024
        off = not spec and s0 == "off"
        mem = mem_fraction(model_name, util, total, spec or off)
        print(f"[sgl] util {util:.3f} -> mem_fraction_static {mem:.3f}" + (" (speculation switched off)" if off else ""),
              flush=True)
        proc, ts = start_sgl_server(mc, port, mem, spec or off, logpath, nspec=nspec, maxlen=maxlen)
        if off:
            post_rpc(port, "set_speculation", {"enabled": "0"})
            on = [float(x.rsplit(" ", 1)[1]) for x in get(f"http://127.0.0.1:{port}/metrics", 10).splitlines()
                  if x.startswith("specsched:spec_enabled")]
            assert on == [0.0], f"speculation still on: {on}"
        return proc, ts
    return start_server


def main():
    what = sys.argv.pop(1)
    model_name = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "m1"
    s0 = "off"
    if "--s0" in sys.argv:
        i = sys.argv.index("--s0")
        s0 = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    assert s0 in ("off", "server")
    if what in ("c1", "frontier"):
        links()
        mod = __import__("c1_run" if what == "c1" else "frontier_run")
        mod.RUNS = SGL_RUNS
        mod.start_server = sgl_start_server(model_name, s0)
        mod.main()
    elif what == "fit":
        sys.path.insert(0, os.path.join(HERE, "sched"))
        import fit_model
        m = fit_model.add_frontier(fit_model.add_traffic_stats(fit_model.fit(SGL_RUNS, model_name), DATA), SGL_RUNS)
        m["engine"] = ENGINE
        out = os.path.join(HERE, "sched", f"model_sgl_{model_name}.json")
        json.dump(m, open(out, "w"), indent=1)
        print("wrote", out)
        for c in m["C"]:
            s = m["spec_step"][str(c)]
            print(f"C={c:>3} T_base={s['T_base']:.2f}ms T_spec={s['T_spec']:.2f}ms acc_meas={s['acc_meas']:.2f} "
                  f"kappa={m['kappa'].get(str(c))} break-even a*={s['T_spec'] / s['T_base']:.2f}")
        for d, v in m.get("frontier", {}).items():
            print("frontier", d, {c: (round(f.get("alpha", 0), 3), round(f.get("beta", 0), 3), f.get("max_abs_err"))
                                  for c, f in v["fit"].items()})
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
