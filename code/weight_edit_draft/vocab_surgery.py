#!/usr/bin/env python
"""Vocabulary surgery for the AngelSlim Qwen3-8B EAGLE-3 head: lift the 32k draft-vocabulary ceiling without training.

1. Fit a ridge-regression map A from the target's lm_head rows to the head's trained lm_head rows on the 32k tokens
   both have (held-out rows measure how faithful the map is).
2. Choose new draft vocabularies from the zh TRAIN traffic's output-token frequencies (eval traffic never used).
3. Transplant rows for newly added tokens as W_target[new] @ A + intercept; keep the trained rows for tokens the head
   already had.
4. Export vLLM-loadable head folders under runs/c2/m1/zh/<variant>/ and write a report.

Run from the specsched code dir on OSC (CPU is enough):
  python ../weight_edit_draft/vocab_surgery.py --out-report ../weight_edit_draft/runs/vocab/surgery.md
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/fs/scratch/PCS0296/specsched/code")
from cfg import DATA, RUNS, model, read_jsonl  # noqa: E402
from head import load_cfg, load_sd  # noqa: E402

VOCAB = 151936


def target_lm_head(target_dir):
    idx = os.path.join(target_dir, "model.safetensors.index.json")
    files = sorted(glob.glob(os.path.join(target_dir, "*.safetensors")))
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        name = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
        files = [os.path.join(target_dir, wm[name])]
    else:
        name = None
    for fp in files:
        with safe_open(fp, "pt") as f:
            keys = list(f.keys())
            for n in ([name] if name else ["lm_head.weight", "model.embed_tokens.weight"]):
                if n in keys:
                    return f.get_tensor(n).float(), n
    raise RuntimeError("no lm_head in target")


def fit_map(X, Y, lam, seed=0):
    """ridge Y ~ (X - xm) A + ym; returns A, xm, ym and held-out metrics."""
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    va, tr = perm[: n // 10], perm[n // 10:]
    xm, ym = X[tr].mean(0), Y[tr].mean(0)
    Xc, Yc = (X[tr] - xm).double(), (Y[tr] - ym).double()
    G = Xc.T @ Xc
    A = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype) * G.diagonal().mean(), Xc.T @ Yc).float()
    pred = (X[va] - xm) @ A + ym
    cos = F.cosine_similarity(pred, Y[va], dim=1)
    rel = (pred - Y[va]).norm(dim=1) / Y[va].norm(dim=1)
    cos_mean_row = F.cosine_similarity(ym.expand_as(Y[va]), Y[va], dim=1)
    cos_identity = F.cosine_similarity(X[va], Y[va], dim=1)
    return A, xm, ym, dict(lam=lam, holdout_cos=float(cos.mean()), holdout_cos_p10=float(cos.quantile(0.1)),
                           holdout_rel_err=float(rel.mean()), baseline_cos_mean_row=float(cos_mean_row.mean()),
                           baseline_cos_identity=float(cos_identity.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--domain", default="zh")
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--lams", default="0.001,0.01,0.1,1")
    ap.add_argument("--union-cov", type=float, default=0.995)
    args = ap.parse_args()
    t0 = time.time()
    mc = model(args.model)
    sd, cfg = load_sd(mc["head"]), load_cfg(mc["head"])
    W_d = sd["lm_head.weight"].float()
    K0 = W_d.shape[0]
    S_old = (torch.arange(K0) + sd["d2t"]).long()          # target ids of the head's vocabulary
    assert S_old.unique().numel() == K0
    W_t, tname = target_lm_head(mc["target"])
    assert W_t.shape[0] == VOCAB, W_t.shape
    print(f"head vocab {K0}, target {tname} {tuple(W_t.shape)}  ({time.time() - t0:.0f}s)", flush=True)

    # 1. fit the row map on the shared tokens
    X, Y = W_t[S_old], W_d
    fits = []
    for lam in [float(z) for z in args.lams.split(",")]:
        A, xm, ym, m = fit_map(X, Y, lam)
        fits.append((m["holdout_cos"], lam, A, xm, ym, m))
        print("fit", json.dumps(m), flush=True)
    fits.sort(key=lambda z: -z[0])
    _, lam, A, xm, ym, best = fits[0]
    # refit on all shared rows with the chosen lambda
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = (X - xm).double(), (Y - ym).double()
    G = Xc.T @ Xc
    A = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype) * G.diagonal().mean(), Xc.T @ Yc).float()
    proj = lambda ids: (W_t[ids] - xm) @ A + ym  # noqa: E731

    # 2. frequencies from TRAIN traffic only
    traffic = read_jsonl(f"{DATA}/traffic/{args.model}/{args.domain}.jsonl")
    chat = read_jsonl(f"{DATA}/traffic/{args.model}/chat.jsonl")
    cnt = Counter()
    for r in traffic:
        if r["split"] == "train":
            cnt.update(r["output_token_ids"])
    ranked = [t for t, _ in cnt.most_common()]
    in_old = torch.zeros(VOCAB, dtype=torch.bool); in_old[S_old] = True

    def coverage(ids_mask, rows):
        tot = hit = 0
        for r in rows:
            o = r["output_token_ids"]
            tot += len(o); hit += int(ids_mask[torch.tensor(o, dtype=torch.long)].sum())
        return hit / max(tot, 1)
    zh_eval = [r for r in traffic if r["split"] == "eval"]
    zh_train = [r for r in traffic if r["split"] == "train"]
    chat_eval = [r for r in chat if r["split"] == "eval"]

    variants = {}
    # zh-ranked 32k: top tokens by zh-train frequency, padded with the old vocabulary's tokens
    top = ranked[:K0]
    if len(top) < K0:
        top += [int(t) for t in S_old.tolist() if t not in set(top)][: K0 - len(top)]
    variants["vz32k"] = sorted(set(int(t) for t in top))
    # union: old vocabulary + zh tokens until zh-train coverage >= union-cov
    tot = sum(cnt.values()); acc = sum(c for t, c in cnt.items() if in_old[t])
    add = []
    for t in ranked:
        if acc / tot >= args.union_cov:
            break
        if not in_old[t]:
            add.append(t); acc += cnt[t]
    variants["vunion"] = sorted(set(S_old.tolist()) | set(add))
    variants["vfull"] = list(range(VOCAB))
    variants["vproj"] = sorted(S_old.tolist())            # diagnostic: same vocabulary, every row projected

    report = dict(head_vocab=K0, fit=best, lam=lam, zh_train_tokens=int(tot), zh_train_distinct=len(cnt), variants={})
    base_cov = dict(zh_eval=coverage(in_old, zh_eval), zh_train=coverage(in_old, zh_train), chat_eval=coverage(in_old, chat_eval))
    report["factory_coverage"] = base_cov
    print("factory coverage", json.dumps(base_cov), flush=True)
    other = {k: v for k, v in sd.items() if k not in ("lm_head.weight", "d2t", "t2d")}
    for name, ids in variants.items():
        ids_t = torch.tensor(ids, dtype=torch.long)
        mask = torch.zeros(VOCAB, dtype=torch.bool); mask[ids_t] = True
        if name == "vproj":
            W_new = proj(ids_t)
        else:
            W_new = proj(ids_t)
            # keep the trained rows for tokens the head already had
            old_pos = {int(t): i for i, t in enumerate(S_old.tolist())}
            keep = [(i, old_pos[t]) for i, t in enumerate(ids) if t in old_pos]
            ki = torch.tensor([i for i, _ in keep]); kj = torch.tensor([j for _, j in keep])
            W_new[ki] = W_d[kj]
        out = f"{RUNS}/c2/{args.model}/{args.domain}/{name}"
        os.makedirs(out, exist_ok=True)
        tensors = dict(other)
        tensors["lm_head.weight"] = W_new.to(torch.bfloat16).contiguous()
        tensors["d2t"] = (ids_t - torch.arange(len(ids))).contiguous()
        tensors["t2d"] = mask.contiguous()
        save_file(tensors, f"{out}/model.safetensors", metadata={"format": "pt"})
        c = dict(cfg); c["draft_vocab_size"] = len(ids)
        json.dump(c, open(f"{out}/config.json", "w"), indent=2)
        cov = dict(zh_eval=coverage(mask, zh_eval), zh_train=coverage(mask, zh_train), chat_eval=coverage(mask, chat_eval))
        n_kept = int(in_old[ids_t].sum())
        report["variants"][name] = dict(size=len(ids), kept_trained_rows=n_kept, transplanted_rows=len(ids) - n_kept, coverage=cov, dir=out)
        print(name, json.dumps(report["variants"][name]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_report)), exist_ok=True)
    json.dump(report, open(args.out_report.replace(".md", ".json"), "w"), indent=1)
    L = ["# 词表手术报告（AngelSlim Qwen3-8B EAGLE-3 头）\n",
         f"头词表 {K0}；行映射 ridge λ={lam}：留出 10% 共享 token 行的余弦 {best['holdout_cos']:.3f}（10% 分位 {best['holdout_cos_p10']:.3f}），"
         f"相对误差 {best['holdout_rel_err']:.3f}；对照：只用均值行 {best['baseline_cos_mean_row']:.3f}，直接用 target 行 {best['baseline_cos_identity']:.3f}。\n",
         f"zh 训练流量输出 token 共 {tot}，不同 token {len(cnt)}。出厂头覆盖率：zh 评测 {base_cov['zh_eval']:.3f}，zh 训练 {base_cov['zh_train']:.3f}，chat 评测 {base_cov['chat_eval']:.3f}。\n",
         "| 变体 | 词表大小 | 保留训练行 | 移植行 | zh 评测覆盖 | chat 评测覆盖 |", "|---|---|---|---|---|---|"]
    for name, v in report["variants"].items():
        L.append(f"| {name} | {v['size']} | {v['kept_trained_rows']} | {v['transplanted_rows']} | {v['coverage']['zh_eval']:.3f} | {v['coverage']['chat_eval']:.3f} |")
    open(args.out_report, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
