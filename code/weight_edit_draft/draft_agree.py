#!/usr/bin/env python
"""Rung 0, step 2: teacher-force one draft checkpoint on the frozen target trajectories.

For every generated position t of every problem: the draft's argmax token, its probability, and the probability the
draft gives the target's token. Under greedy decoding the accepted tokens of speculative decoding are exactly the
target trajectory, so these argmaxes determine the accepted lengths exactly (see analyze.py).

  python draft_agree.py --model Qwen/Qwen3.5-0.8B --target runs/r0/target.jsonl --out runs/r0/draft_base.npz
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import load_model, log, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1024, help="positions per logits chunk")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(args.target)
    if args.limit:
        rows = rows[:args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, args.revision, device)
    argmax, p_top, p_tgt, offsets, uids, n_gen = [], [], [], [0], [], []
    t0 = time.time()
    for k, r in enumerate(rows):
        ids = list(r["prompt_ids"]) + list(r["gen_ids"])
        P, G = len(r["prompt_ids"]), len(r["gen_ids"])
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            try:
                out = model(input_ids=x, logits_to_keep=G + 1, use_cache=False)
                logits = out.logits[0]
                if logits.shape[0] == len(ids):          # the kwarg was accepted but ignored
                    logits = logits[P - 1:len(ids) - 1]
                else:
                    logits = logits[:-1]                  # kept positions P-1..T-1; drop the one after the end
            except TypeError:
                out = model(input_ids=x, use_cache=False)
                logits = out.logits[0][P - 1:len(ids) - 1]
            assert logits.shape[0] == G, (logits.shape, G)
            tgt = torch.tensor(r["gen_ids"], device=device)
            am, pt, pg = [], [], []
            for s in range(0, G, args.chunk):
                lp = torch.log_softmax(logits[s:s + args.chunk].float(), dim=-1)
                am.append(lp.argmax(-1))
                pt.append(lp.max(-1).values.exp())
                pg.append(lp.gather(1, tgt[s:s + args.chunk, None])[:, 0].exp())
            argmax.append(torch.cat(am).cpu().numpy().astype(np.int32))
            p_top.append(torch.cat(pt).cpu().numpy().astype(np.float16))
            p_tgt.append(torch.cat(pg).cpu().numpy().astype(np.float16))
        offsets.append(offsets[-1] + G)
        uids.append(r["uid"])
        n_gen.append(G)
        if k % 32 == 0:
            agree = float((argmax[-1] == np.asarray(r["gen_ids"], dtype=np.int32)).mean())
            log(f"  {k}/{len(rows)} {time.time() - t0:.0f}s  last seq agree={agree:.3f} (G={G})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, argmax=np.concatenate(argmax), p_top=np.concatenate(p_top),
                        p_tgt=np.concatenate(p_tgt), offsets=np.asarray(offsets, dtype=np.int64),
                        uids=np.asarray(uids), n_gen=np.asarray(n_gen, dtype=np.int64),
                        meta=json.dumps(dict(model=args.model, revision=args.revision, target=args.target,
                                             seconds=time.time() - t0)))
    tot = offsets[-1]
    a = np.concatenate(argmax) == np.concatenate([np.asarray(r["gen_ids"], dtype=np.int32) for r in rows])
    log(f"done: {len(rows)} seqs, {tot} positions, pooled agreement {a.mean():.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
