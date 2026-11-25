#!/usr/bin/env python
"""Rung 0, step 1: the target model's greedy trajectories on the Rung 0 problems.

One JSON line per problem: uid, prompt_ids, gen_ids (the stop token included when the model stopped), finish
("stop" | "length"), n_gen. Every draft is later teacher-forced on exactly these ids, so this file is the frozen
reference of the run. Greedy + fixed target means the accepted tokens of speculative decoding ARE this trajectory.

  python gen_target.py --model Qwen/Qwen3.5-9B --problems data/math500_rung0_256.jsonl --out runs/r0/target.jsonl \
      --engine vllm --thinking 1 --max-new 2048
"""
import argparse
import json
import os
import time

from common import chat_prompt_ids, load_tokenizer, log, read_jsonl, stop_token_ids, write_jsonl


def run_vllm(args, pids, stops):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    max_prompt = max(len(p) for p in pids)
    llm = LLM(model=args.model, revision=args.revision, dtype="bfloat16", seed=0,
              max_model_len=max_prompt + args.max_new + 16, gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=False, max_num_seqs=args.batch)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new, seed=0, stop_token_ids=stops,
                        skip_special_tokens=False)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp)
    gens = []
    for o in outs:
        g = list(o.outputs[0].token_ids)
        fin = "length" if o.outputs[0].finish_reason == "length" else "stop"
        gens.append((g, fin))
    return gens, time.time() - t0


def run_hf(args, pids, stops, pad_id):
    import torch
    from common import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, args.revision, device)
    order = sorted(range(len(pids)), key=lambda i: -len(pids[i]))
    gens = [None] * len(pids)
    t0 = time.time()
    for s in range(0, len(order), args.batch):
        idx = order[s:s + args.batch]
        L = max(len(pids[i]) for i in idx)
        inp = torch.full((len(idx), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            p = pids[i]
            inp[r, L - len(p):] = torch.tensor(p)
            att[r, L - len(p):] = 1
        with torch.no_grad():
            out = model.generate(input_ids=inp.to(device), attention_mask=att.to(device), do_sample=False,
                                 num_beams=1, max_new_tokens=args.max_new, eos_token_id=stops, pad_token_id=pad_id)
        for r, i in enumerate(idx):
            g = out[r, L:].tolist()
            cut = len(g)
            for k, t in enumerate(g):
                if t in stops:
                    cut = k + 1
                    break
            g = g[:cut]
            gens[i] = (g, "stop" if (g and g[-1] in stops) else "length")
        log(f"  hf batch {s // args.batch}: {len(idx)} seqs, {time.time() - t0:.0f}s")
    return gens, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256, help="vllm: max_num_seqs; hf: generate batch size")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N problems")
    args = ap.parse_args()

    probs = read_jsonl(args.problems)
    if args.limit:
        probs = probs[:args.limit]
    tok = load_tokenizer(args.model, args.revision)
    stops = stop_token_ids(tok, args.model, args.revision)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else stops[0]
    pids, texts = [], []
    for p in probs:
        ids, text = chat_prompt_ids(tok, p["problem"], args.thinking)
        pids.append(ids)
        texts.append(text)
    log(f"{len(probs)} problems; prompt tokens min/mean/max = {min(map(len, pids))}/"
        f"{sum(map(len, pids)) / len(pids):.0f}/{max(map(len, pids))}; stop ids {stops}; thinking={args.thinking}")
    log("prompt example:\n" + texts[0])

    if args.engine == "vllm":
        gens, dt = run_vllm(args, pids, stops)
    else:
        gens, dt = run_hf(args, pids, stops, pad_id)

    rows, appended = [], 0
    for p, ids, (g, fin) in zip(probs, pids, gens):
        if fin == "stop" and (not g or g[-1] not in stops):
            g = g + [stops[0]]          # the engine stripped the stop token: put the chat EOS back
            appended += 1
        rows.append(dict(uid=p["unique_id"], prompt_ids=ids, gen_ids=g, finish=fin, n_gen=len(g)))
    write_jsonl(args.out, rows)
    n_len = sum(r["finish"] == "length" for r in rows)
    meta = dict(model=args.model, revision=args.revision, engine=args.engine, thinking=args.thinking,
                max_new=args.max_new, n=len(rows), n_length=n_len, eos_appended=appended,
                mean_gen=sum(r["n_gen"] for r in rows) / len(rows), gen_seconds=dt, stop_ids=stops,
                prompt_example=texts[0])
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    log(f"done: {len(rows)} trajectories, {n_len} hit the cap, mean {meta['mean_gen']:.0f} tokens, "
        f"{dt:.0f}s, eos appended {appended}")


if __name__ == "__main__":
    main()
