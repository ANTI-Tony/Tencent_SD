"""Is greedy output byte-exact across draft reloads under tensor parallelism? Separate the reload from TP
numerics: generate with the stock head, reload the *same* head (no-op), generate again; then reload a copy
whose lm_head is zeroed, generate again. Reports, per step, how many prompts are identical and the first
divergence position. Output: runs/tp/probe_tp<N>.json
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from vllm import LLM, SamplingParams

from cfg import DATA, RUNS, model, read_jsonl


def zeroed_copy(src, dst):
    """Copy of the draft checkpoint with its lm_head zeroed, written as model.safetensors (the AngelSlim
    heads ship pytorch_model.bin; vLLM loads either, so the .bin is dropped from the copy)."""
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    tensors = {}
    for f in sorted(os.listdir(dst)):
        fp = os.path.join(dst, f)
        if f.endswith(".safetensors"):
            tensors.update(load_file(fp))
            os.remove(fp)
        elif f.endswith(".bin"):
            tensors.update(torch.load(fp, map_location="cpu", weights_only=True))
            os.remove(fp)
    heads = [k for k in tensors if k.endswith("lm_head.weight")]
    assert heads, f"no lm_head in {src}: {list(tensors)[:5]}"
    for k in heads:
        tensors[k] = torch.zeros_like(tensors[k])
    save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    return dst


def compare(a, b):
    same = sum(1 for x, y in zip(a, b) if x == y)
    first = [next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None) for x, y in zip(a, b)]
    return dict(identical=same, total=len(a), first_divergence=[f for f in first if f is not None][:16])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    mc = model(a.model)
    prompts = [r["prompt"] for r in read_jsonl(f"{DATA}/bench/code_eval.jsonl")[: a.n]]
    llm = LLM(model=mc["target"], tensor_parallel_size=a.tp, max_model_len=2048, gpu_memory_utilization=0.8, seed=0,
              enable_prefix_caching=False,
              speculative_config={"method": "eagle3", "model": mc["head"], "num_speculative_tokens": 3})
    sp = SamplingParams(temperature=0, max_tokens=128)
    chat = [[{"role": "user", "content": p}] for p in prompts]

    def gen():
        return [list(o.outputs[0].token_ids) for o in llm.chat(chat, sp, chat_template_kwargs={"enable_thinking": False})]

    os.makedirs(f"{RUNS}/tp", exist_ok=True)
    out = {"tp": a.tp, "n": a.n}

    def record(key, res):
        out[key] = res
        json.dump(out, open(f"{RUNS}/tp/probe_tp{a.tp}.json", "w"), indent=1)
        print(key, json.dumps(res), flush=True)

    base = gen()
    record("rerun_same_head", compare(base, gen()))                  # pure run-to-run nondeterminism
    llm.collective_rpc("reload_draft_weights", kwargs={"weights_path": mc["head"]})
    record("after_noop_reload", compare(base, gen()))                # reload of the identical checkpoint
    zdir = zeroed_copy(mc["head"], f"{RUNS}/tp/zeroed_head_tp{a.tp}")
    llm.collective_rpc("reload_draft_weights", kwargs={"weights_path": zdir})
    record("after_zeroed_reload", compare(base, gen()))              # a head that proposes garbage
    llm.collective_rpc("reload_draft_weights", kwargs={"weights_path": mc["head"]})
    record("after_restore", compare(base, gen()))


if __name__ == "__main__":
    main()
