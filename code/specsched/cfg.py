"""Shared paths and model registry for the Rung-0 characterization (OSC Cardinal)."""
import json
import os

WS = os.environ.get("WS", "/fs/scratch/PCS0296/specsched")
DATA = f"{WS}/data"
RUNS = f"{WS}/runs"

MODELS = {
    # vLLM 0.27 registers aux id L as the residual stream after layer L-1, i.e. HF hidden_states[L].
    "m1": dict(
        target="/fs/scratch/PCS0296/bursar/models/hub/models--Qwen--Qwen3-8B/snapshots/"
               "b968826d9c46dd6066d109eabc6255188de91218",
        head=f"{WS}/models/qwen3-8b-eagle3",
        aux=(2, 18, 33),
    ),
    "m2": dict(
        target=f"{WS}/models/Qwen3-30B-A3B",
        head=f"{WS}/models/qwen3-a3b-eagle3",
        aux=(2, 24, 45),
    ),
}

DOMAINS = ["chat", "sql", "code", "zh", "fc", "math"]
B_DOMAINS = ["sql", "code", "zh", "fc"]


def model(name):
    m = dict(MODELS[name])
    aux = os.environ.get("AUXL")
    if aux:
        m["aux"] = tuple(int(x) for x in aux.split(","))
    return m


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
