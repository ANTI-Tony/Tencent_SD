#!/usr/bin/env python
"""Rung 0, step 3: exact greedy speculative-decoding accounting from the frozen trajectories.

Greedy + fixed target: the accepted tokens are the target's own greedy trajectory, so with a_t = [draft argmax at
position t == target token t] each round accepts the run of 1s from its start, capped at gamma, and then the target
supplies one token (correction or bonus). Everything below is computed from a_t; no speculative engine is involved.

  python analyze.py --target runs/r0/target.jsonl --drafts base=runs/r0/draft_base.npz edited=runs/r0/draft_edited.npz \
      2b=runs/r0/draft_2b.npz --ref base --gamma 5 --tokenizer Qwen/Qwen3.5-0.8B --out runs/r0/summary.md
"""
import argparse
import json
import os
import string

import numpy as np

from common import read_jsonl


def simulate(a, gamma):
    """accepted draft tokens per round, and the first-rejection position (1..gamma) or 0 for a full round."""
    T = len(a)
    pos = 0
    acc, first = [], []
    while pos < T:
        g = min(gamma, T - pos)
        k = 0
        while k < g and a[pos + k]:
            k += 1
        acc.append(k)
        first.append(0 if k == g else k + 1)
        pos += k + 1
    return np.asarray(acc), np.asarray(first)


def load_draft(path):
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    return dict(argmax=z["argmax"], p_top=z["p_top"].astype(np.float32), p_tgt=z["p_tgt"].astype(np.float32),
                offsets=off, uids=list(z["uids"]), meta=json.loads(str(z["meta"])))


def copy_flags(prompt_ids, gen_ids, proposals, n=4):
    """For each generated position: is (last n-1 context tokens + proposal) an n-gram already seen in the context?
    Returned for the draft's proposal and for the target's own token."""
    x = list(prompt_ids) + list(gen_ids)
    P = len(prompt_ids)
    seen = set()
    for i in range(n - 1, P):
        seen.add(tuple(x[i - n + 1:i + 1]))
    cp, ct = np.zeros(len(gen_ids), bool), np.zeros(len(gen_ids), bool)
    for t in range(len(gen_ids)):
        i = P + t
        if i >= n - 1:
            ctx = tuple(x[i - n + 1:i])
            cp[t] = (ctx + (int(proposals[t]),)) in seen
            ct[t] = (ctx + (x[i],)) in seen
            seen.add(ctx + (x[i],))
    return cp, ct


def token_category(tok, tid, special):
    if tid in special:
        return "eos/think"
    s = tok.decode([tid])
    st = s.strip()
    if st == "":
        return "空白/换行"
    if all(c in string.digits + ".,-" for c in st):
        return "数字"
    if "\\" in st:
        return "LaTeX"
    if all((c in string.punctuation) or (c in "{}$^_") for c in st):
        return "标点/符号"
    if len(st) <= 2 and not st.isalpha():
        return "标点/符号"
    return "词"


def bootstrap_diff(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = y - x
    m = len(d)
    idx = rng.integers(0, m, size=(n, m))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafts", nargs="+", required=True, help="name=path.npz ...")
    ap.add_argument("--ref", default="base")
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--ngram", type=int, default=4)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.target)
    uids = [r["uid"] for r in rows]
    drafts = {}
    for spec in args.drafts:
        name, path = spec.split("=", 1)
        d = load_draft(path)
        assert d["uids"] == uids[:len(d["uids"])], f"{name}: problem order differs from the target file"
        drafts[name] = d
    names = list(drafts)
    assert args.ref in drafts
    n = min(len(d["uids"]) for d in drafts.values())
    rows = rows[:n]

    tok = special = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        special = set(int(i) for i in tok.all_special_ids)
        for t in ("</think>", "<think>"):
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                special.add(i)

    per = {nm: [] for nm in names}        # per-problem records
    agree = {}                            # name -> list of a_t arrays
    props = {}
    for nm, d in drafts.items():
        agree[nm], props[nm] = [], []
        for k, r in enumerate(rows):
            s, e = d["offsets"][k], d["offsets"][k + 1]
            am = d["argmax"][s:e]
            g = np.asarray(r["gen_ids"], dtype=np.int32)
            assert len(am) == len(g)
            a = (am == g)
            agree[nm].append(a)
            props[nm].append(am)
            acc, first = simulate(a, args.gamma)
            cp, ct = copy_flags(r["prompt_ids"], r["gen_ids"], am, args.ngram)
            dis = ~a
            # positions visited as first rejections in the simulation
            vis = np.zeros(len(g), bool)
            pos = 0
            for kk, f in zip(acc, first):
                if f:
                    vis[pos + kk] = True
                pos += kk + 1
            per[nm].append(dict(
                uid=r["uid"], n_gen=len(g), finish=r["finish"], agree=float(a.mean()),
                rounds=len(acc), acc_mean=float(acc.mean()), full=float((first == 0).mean()),
                first_hist=np.bincount(first, minlength=args.gamma + 1)[:args.gamma + 1].tolist(),
                n_dis=int(dis.sum()), dis_copy=int((dis & cp).sum()), dis_loop=int((dis & cp & ~ct).sum()),
                n_first=int(vis.sum()), first_copy=int((vis & cp).sum()), first_loop=int((vis & cp & ~ct).sum()),
                tgt_copy_all=int(ct.sum()), p_tgt_mean=float(d["p_tgt"][s:e].mean()),
            ))

    L = []
    L.append(f"# Rung 0 结果：greedy 投机解码精确核算（γ={args.gamma}，{n} 题）\n")
    L.append(f"target 文件：`{args.target}`；每题 target 轨迹平均 {np.mean([r['n_gen'] for r in rows]):.0f} token，"
             f"{sum(r['finish'] == 'length' for r in rows)} 题撞上限。\n")
    L.append("## 表 1：各 draft 的接受统计（按题平均 ± 题间标准差）\n")
    L.append("| draft | 逐 token 一致率 | 每轮接受草稿 token 数 | 每轮产出 token（含 target 的 1 个） | 整轮全接受占比 | "
             + " | ".join(f"首拒在第 {i} 个" for i in range(1, args.gamma + 1)) + " | 首拒中复制型占比 | 首拒中循环型占比 | 所有分歧中复制型占比 |")
    L.append("|" + "---|" * (8 + args.gamma))
    stats = {}
    for nm in names:
        p = per[nm]
        acc = np.array([q["acc_mean"] for q in p])
        agr = np.array([q["agree"] for q in p])
        full = np.array([q["full"] for q in p])
        hist = np.sum([q["first_hist"] for q in p], axis=0)
        rounds = hist.sum()
        nf, fc, fl = sum(q["n_first"] for q in p), sum(q["first_copy"] for q in p), sum(q["first_loop"] for q in p)
        nd, dc = sum(q["n_dis"] for q in p), sum(q["dis_copy"] for q in p)
        stats[nm] = dict(acc_mean=float(acc.mean()), acc_sd=float(acc.std(ddof=1)), agree=float(agr.mean()),
                         full=float(full.mean()), first_hist=hist.tolist(), rounds=int(rounds),
                         first_copy_share=fc / max(nf, 1), first_loop_share=fl / max(nf, 1),
                         dis_copy_share=dc / max(nd, 1), n_first=nf, n_dis=nd)
        L.append(f"| {nm} | {agr.mean():.4f} | {acc.mean():.3f} ± {acc.std(ddof=1):.3f} | {acc.mean() + 1:.3f} | "
                 f"{full.mean():.3f} | " + " | ".join(f"{hist[i] / rounds:.3f}" for i in range(1, args.gamma + 1))
                 + f" | {fc / max(nf, 1):.3f} | {fl / max(nf, 1):.3f} | {dc / max(nd, 1):.3f} |")
    L.append("\n复制型 = draft 提议的 token 使（前 %d 个上下文 token + 提议）构成的 %d-gram 已在上文出现过；"
             "循环型 = 复制型且 target 自己的 token 不是复制型（draft 想重复、target 不想）。\n" % (args.ngram - 1, args.ngram))

    L.append(f"## 表 2：相对 `{args.ref}` 的配对差（按题配对 bootstrap 10000 次，95% 区间）\n")
    L.append("| draft | Δ每轮接受草稿 token | 相对变化 | 95% 区间 | 修好的位置数（ref 错→它对） | 其中复制型 | 其中循环型 | 弄坏的位置数（ref 对→它错） | 其中复制型（ref 的提议） |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    ref_acc = np.array([q["acc_mean"] for q in per[args.ref]])
    comp = {}
    for nm in names:
        if nm == args.ref:
            continue
        acc = np.array([q["acc_mean"] for q in per[nm]])
        d, lo, hi = bootstrap_diff(ref_acc, acc)
        fixed = broken = fixed_cp = fixed_loop = broken_cp = 0
        for k, r in enumerate(rows):
            a0, a1 = agree[args.ref][k], agree[nm][k]
            cp1, ct = copy_flags(r["prompt_ids"], r["gen_ids"], props[nm][k], args.ngram)
            cp0, _ = copy_flags(r["prompt_ids"], r["gen_ids"], props[args.ref][k], args.ngram)
            f = (~a0) & a1
            b = a0 & (~a1)
            fixed += int(f.sum()); broken += int(b.sum())
            fixed_cp += int((f & cp0).sum()); fixed_loop += int((f & cp0 & ~ct).sum())
            broken_cp += int((b & cp1).sum())
        comp[nm] = dict(delta=d, rel=d / ref_acc.mean(), lo=lo, hi=hi, fixed=fixed, broken=broken,
                        fixed_copy=fixed_cp, fixed_loop=fixed_loop, broken_copy=broken_cp)
        L.append(f"| {nm} | {d:+.3f} | {100 * d / ref_acc.mean():+.1f}% | [{lo:+.3f}, {hi:+.3f}] | {fixed} | "
                 f"{fixed_cp} ({fixed_cp / max(fixed, 1):.2f}) | {fixed_loop} ({fixed_loop / max(fixed, 1):.2f}) | {broken} | "
                 f"{broken_cp} ({broken_cp / max(broken, 1):.2f}) |")
    L.append("\n“修好的位置”里复制型占比衡量 MIT 去循环编辑在 target 前缀上到底改掉了哪类分歧。\n")

    if tok is not None:
        L.append(f"## 表 3：`{args.ref}` 首拒位置上 target token 的类别（与全部 token 的类别分布对比）\n")
        cat_first, cat_all = {}, {}
        for k, r in enumerate(rows):
            a = agree[args.ref][k]
            acc, first = simulate(a, args.gamma)
            vis = np.zeros(len(a), bool)
            pos = 0
            for kk, f in zip(acc, first):
                if f:
                    vis[pos + kk] = True
                pos += kk + 1
            for t, tid in enumerate(r["gen_ids"]):
                c = token_category(tok, int(tid), special)
                cat_all[c] = cat_all.get(c, 0) + 1
                if vis[t]:
                    cat_first[c] = cat_first.get(c, 0) + 1
        tf, ta = sum(cat_first.values()), sum(cat_all.values())
        L.append("| 类别 | 首拒位置占比 | 全部 token 占比 | 富集倍数 |")
        L.append("|---|---|---|---|")
        for c in sorted(cat_all, key=lambda c: -cat_first.get(c, 0)):
            pf, pa = cat_first.get(c, 0) / max(tf, 1), cat_all[c] / ta
            L.append(f"| {c} | {pf:.3f} | {pa:.3f} | {pf / pa if pa else 0:.2f} |")
        L.append("")

    L.append("## 表 4：按输出是否撞上限分组（每轮接受草稿 token 数）\n")
    L.append("| draft | 自然停止的题 | 撞上限的题 |")
    L.append("|---|---|---|")
    for nm in names:
        p = per[nm]
        s = [q["acc_mean"] for q in p if q["finish"] == "stop"]
        l = [q["acc_mean"] for q in p if q["finish"] == "length"]
        L.append(f"| {nm} | {np.mean(s) if s else float('nan'):.3f} (n={len(s)}) | {np.mean(l) if l else float('nan'):.3f} (n={len(l)}) |")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    base = os.path.splitext(args.out)[0]
    with open(base + ".json", "w") as f:
        json.dump(dict(gamma=args.gamma, n=n, stats=stats, comparisons=comp,
                       drafts={nm: d["meta"] for nm, d in drafts.items()}), f, indent=1, ensure_ascii=False)
    import csv
    with open(base + "_per_problem.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "n_gen", "finish"] + [f"{nm}_acc" for nm in names] + [f"{nm}_agree" for nm in names])
        for k, r in enumerate(rows):
            w.writerow([r["uid"], r["n_gen"], r["finish"]] + [f"{per[nm][k]['acc_mean']:.4f}" for nm in names]
                       + [f"{per[nm][k]['agree']:.4f}" for nm in names])
    print("\n".join(L))


if __name__ == "__main__":
    main()
