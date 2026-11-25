"""C2: evaluate one draft head (or no speculation) for a domain with vLLM offline.

Writes runs/c2/<model>/<domain>/<head>/eval.json with, per eval set (domain, chat):
  acceptance length, draft acceptance rate, lossless check vs reference traffic,
  saturation throughput (eval+reserve submitted at once), and batch-1 per-token latency.
One head per process: vLLM's metrics registry and GPU memory are not reused across engines.
"""
import argparse
import json
import os
import time

from cfg import DATA, RUNS, model, read_jsonl


def counters(llm):
    out = {}
    for m in llm.get_metrics():
        if "spec_decode" in m.name and hasattr(m, "value"):
            out[m.name] = out.get(m.name, 0) + m.value
    return out


def spec_stats(c0, c1):
    d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
    drafts = sum(v for k, v in d.items() if "num_drafts" in k)
    acc = sum(v for k, v in d.items() if "num_accepted_tokens" in k and "per_pos" not in k)
    dtok = sum(v for k, v in d.items() if "num_draft_tokens" in k)
    return dict(acc_len=(1 + acc / drafts) if drafts else None, draft_acc_rate=(acc / dtok) if dtok else None,
                drafts=drafts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--head", required=True, help="head folder name under runs/c2/<m>/<dom>/, or 'none'")
    ap.add_argument("--nspec", type=int, default=3)
    ap.add_argument("--also", default="chat")
    ap.add_argument("--n-acc", type=int, default=300)
    ap.add_argument("--n-lat", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rep", type=int, default=0, help=">0: write eval_rep<N>.json (cross-instance noise runs)")
    ap.add_argument("--fixed-len", type=int, default=0,
                    help=">0: also measure saturation with ignore_eos at this output length (identical token "
                         "counts and drain shape for every head); written as eval_fixed.json")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    mc = model(a.model)
    base = f"{RUNS}/c2/{a.model}/{a.domain}"
    outdir = f"{base}/{a.head}" if a.head != "none" else f"{base}/none"
    os.makedirs(outdir, exist_ok=True)
    spec = None if a.head == "none" else {"method": "eagle3", "model": f"{base}/{a.head}",
                                          "num_speculative_tokens": a.nspec}
    t_load = time.time()
    llm = LLM(model=mc["target"], speculative_config=spec, gpu_memory_utilization=0.90, max_model_len=4096,
              enable_prefix_caching=False, disable_log_stats=False, seed=0)
    t_load = time.time() - t_load
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    also = [d for d in a.also.split(",") if d]
    traffic = {d: read_jsonl(f"{DATA}/traffic/{a.model}/{d}.jsonl") for d in {a.domain, *also}}
    res = dict(model=a.model, domain=a.domain, head=a.head, nspec=a.nspec, load_s=t_load, sets={})
    warm = [r for r in traffic[a.domain] if r["split"] == "reserve"][:32]
    llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in warm], sp, use_tqdm=False)
    for name in dict.fromkeys([a.domain, *also]):
        ev = [r for r in traffic[name] if r["split"] == "eval"][:a.n_acc]
        c0, t0 = counters(llm), time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in ev], sp, use_tqdm=False)
        dt, c1 = time.time() - t0, counters(llm)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        same = sum(list(o.outputs[0].token_ids) == r["output_token_ids"] for o, r in zip(outs, ev))
        res["sets"][name] = dict(n=len(ev), wall_s=dt, out_tokens=ntok, tok_s=ntok / dt,
                                 identical_frac=same / len(ev), **spec_stats(c0, c1))
        print(name, json.dumps(res["sets"][name]), flush=True)
    # saturation throughput on the domain (eval + reserve, all submitted at once)
    sat = [r for r in traffic[a.domain] if r["split"] in ("eval", "reserve")]
    walls, ntok = [], 0
    for _ in range(3):  # single runs are noisy for short-output domains; report the median of three
        t0 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in sat], sp, use_tqdm=False)
        walls.append(time.time() - t0)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    dt = sorted(walls)[1]
    res["saturation"] = dict(n=len(sat), wall_s=dt, walls=walls, out_tokens=ntok, tok_s=ntok / dt,
                             gpu_s_per_req=dt / len(sat))
    # batch-1 latency (one request at a time)
    lat = []
    for r in [r for r in traffic[a.domain] if r["split"] == "eval"][:a.n_lat]:
        t0 = time.time()
        o = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"])], sp, use_tqdm=False)
        n = len(o[0].outputs[0].token_ids)
        lat.append(((time.time() - t0) / max(n, 1)) * 1e3)
    res["batch1_ms_per_token"] = sum(lat) / len(lat)
    if a.fixed_len:
        spf = SamplingParams(temperature=0.0, max_tokens=a.fixed_len, ignore_eos=True)
        prompts = [TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in sat]
        llm.generate(prompts[:64], spf, use_tqdm=False)
        walls = []
        for _ in range(3):
            c0, t0 = counters(llm), time.time()
            outs = llm.generate(prompts, spf, use_tqdm=False)
            walls.append(time.time() - t0)
            c1 = counters(llm)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        dt = sorted(walls)[1]
        res["saturation_fixed"] = dict(n=len(sat), out_len=a.fixed_len, wall_s=dt, walls=walls, out_tokens=ntok,
                                       tok_s=ntok / dt, gpu_s_per_req=dt / len(sat), **spec_stats(c0, c1))
        print("FIXED", json.dumps(res["saturation_fixed"]), flush=True)
    fname = f"eval_rep{a.rep}.json" if a.rep else ("eval_fixed.json" if a.fixed_len else "eval.json")
    json.dump(res, open(f"{outdir}/{fname}", "w"), indent=1)
    print("EVAL_DONE", json.dumps({k: v for k, v in res.items() if k != "sets"}), flush=True)


if __name__ == "__main__":
    main()
