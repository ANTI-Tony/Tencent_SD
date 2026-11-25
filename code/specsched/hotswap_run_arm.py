"""Same-start comparison harness (E28, preregistered 2026-11-24). The frozen harness hotswap_run.py is imported and
run unchanged; this wrapper only
  --arm X [--arm-n N]   starts controller_arm.py instead of controller.py, with `--arm X --arm-domain <dom_b>`
                        (policy stays costmodel, --fixed-duty 0.5): v6 rules outside the B label period, the arm rule
                        inside it. Without --arm the frozen controller.py runs.
  --model-json FILE     passes a calibration file to the controller (mixtures: sched/model_mix.json)
  --open-rate L         serves phase B open loop: Poisson arrivals at L requests/s with the run seed and no
                        concurrency cap, so every arm of a seed sees the same arrival timestamps. Warmup, A1 and A2
                        keep the frozen closed-loop client.
All other arguments go to hotswap_run.py as they are."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _take(flag, has_value=True):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    val = sys.argv[i + 1] if has_value else True
    del sys.argv[i:i + 2 if has_value else i + 1]
    return val


ARM = _take("--arm")
ARM_N = _take("--arm-n")
MODEL_JSON = _take("--model-json")
OPEN_RATE = _take("--open-rate")
assert ARM in (None, "sw", "round", "budget", "duty", "novalue"), ARM
assert "--dom-b" in sys.argv, "the manifest always names the B domain"
DOM_B = sys.argv[sys.argv.index("--dom-b") + 1]
if MODEL_JSON:
    MODEL_JSON = os.path.abspath(MODEL_JSON)
    assert os.path.exists(MODEL_JSON), MODEL_JSON
if OPEN_RATE:
    assert float(OPEN_RATE) > 0

import hotswap_run  # noqa: E402  (frozen harness)
from c1_run import VLLM  # noqa: E402
from cfg import DATA  # noqa: E402

_Popen = subprocess.Popen
FROZEN = os.path.join(HERE, "controller.py")


def _popen(cmd, *args, **kw):
    if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and str(cmd[1]) == FROZEN:
        cmd = list(cmd)
        if ARM:
            cmd[1] = os.path.join(HERE, "controller_arm.py")
            assert cmd[cmd.index("--policy") + 1] == "costmodel", "arms run on the v6 strategy"
            cmd[cmd.index("--fixed-duty") + 1] = "0.5"
            cmd += ["--arm", ARM, "--arm-domain", DOM_B] + (["--arm-n", ARM_N] if ARM_N else [])
        if MODEL_JSON:
            if "--model-json" in cmd:
                cmd[cmd.index("--model-json") + 1] = MODEL_JSON
            else:
                cmd += ["--model-json", MODEL_JSON]
    return _Popen(cmd, *args, **kw)


subprocess.Popen = _popen

_bench = hotswap_run.bench


def _bench_open(mc, port, rg, c, outdir, fname=None, seed=0):
    """c1_run.bench with the concurrency cap removed and a finite request rate; phase B only."""
    if fname != "phaseB.json":
        return _bench(mc, port, rg, c, outdir, fname, seed=seed)
    cmd = [VLLM, "bench", "serve", "--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
           "--base-url", f"http://127.0.0.1:{port}", "--model", mc["target"], "--served-model-name", "target",
           "--dataset-name", "custom", "--dataset-path", rg.get("file") or f"{DATA}/bench/{rg['dom']}_eval.jsonl",
           "--skip-chat-template", "--custom-output-len", str(rg["outlen"]), "--num-prompts", str(rg["nprompt"](c)),
           "--request-rate", str(OPEN_RATE), "--burstiness", "1.0", "--temperature", "0",
           "--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": rg["thinking"]}}),
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--save-result", "--save-detailed", "--result-dir", outdir, "--result-filename", fname,
           "--seed", str(seed), "--disable-tqdm"]
    print(json.dumps(dict(event="open_loop", rate=float(OPEN_RATE), n=rg["nprompt"](c), seed=seed)), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise RuntimeError(f"open-loop bench failed at rate={OPEN_RATE}")
    return json.load(open(os.path.join(outdir, fname)))


if OPEN_RATE:
    hotswap_run.bench = _bench_open

if __name__ == "__main__":
    hotswap_run.main()
