#!/usr/bin/env python
"""Optimistic upper bounds for "fix one class of disagreements": on the frozen Rung 0 data, set a_t := 1 for every
disagreement in a class (nothing else changes), re-run the exact gamma=5 accounting, and report the gain.
Classes are ranked on the diagnostic half (first 128 problems) and reported on the held-out half (last 128).
Prefix-identifiable classes use only what an editor can see before the target answers (the draft's proposal and
confidence, the context); the others need the target token and are references only.

  python bound_analysis.py --run runs/r0 --tokenizer Qwen/Qwen3.5-0.8B --out runs/r0/bounds.md
"""
import argparse
import json
import string

import numpy as np

from analyze import copy_flags, load_draft, simulate
from common import read_jsonl

CATS = ["词", "标点/符号", "数字", "空白/换行", "LaTeX", "eos/think"]


def categorize(tok, special):
    cache = {}

    def f(tid):
        tid = int(tid)
        if tid in cache:
            return cache[tid]
        if tid in special:
            c = "eos/think"
        else:
            s = tok.decode([tid])
            st = s.strip()
            if st == "":
                c = "空白/换行"
            elif all(ch in string.digits + ".,-" for ch in st):
                c = "数字"
            elif "\\" in st:
                c = "LaTeX"
            elif all((ch in string.punctuation) or (ch in "{}$^_") for ch in st) or (len(st) <= 2 and not st.isalpha()):
                c = "标点/符号"
            else:
                c = "词"
        cache[tid] = c
        return c
    return f


def acc_mean(a, gamma):
    acc, _ = simulate(a, gamma)
    return float(acc.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = read_jsonl(f"{args.run}/target.jsonl")
    base = load_draft(f"{args.run}/draft_base.npz")
    big = load_draft(f"{args.run}/draft_2b.npz")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    special = set(int(i) for i in tok.all_special_ids)
    think_end = tok.convert_tokens_to_ids("</think>")
    for t in ("</think>", "<think>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            special.add(i)
    cat = categorize(tok, special)
    rng = np.random.default_rng(args.seed)

    # per-problem feature tables
    probs = []
    for k, r in enumerate(rows):
        s, e = base["offsets"][k], base["offsets"][k + 1]
        g = np.asarray(r["gen_ids"], dtype=np.int32)
        am = base["argmax"][s:e]
        a = am == g
        a2 = big["argmax"][big["offsets"][k]:big["offsets"][k + 1]] == g
        p_top, p_tgt = base["p_top"][s:e], base["p_tgt"][s:e]
        cp, ct = copy_flags(r["prompt_ids"], r["gen_ids"], am, 4)
        tcat = np.array([cat(t) for t in g])
        pcat = np.array([cat(t) for t in am])
        prev = np.array(["起始"] + [cat(t) for t in g[:-1]])
        after_think = np.zeros(len(g), bool)
        if isinstance(think_end, int) and think_end >= 0:
            w = np.where(g == think_end)[0]
            if len(w):
                after_think[w[0] + 1:] = True
        # positions actually visited as first rejections
        acc, first = simulate(a, args.gamma)
        vis = np.zeros(len(g), bool)
        pos = 0
        for kk, f in zip(acc, first):
            if f:
                vis[pos + kk] = True
            pos += kk + 1
        probs.append(dict(a=a, a2=a2, p_top=p_top, p_tgt=p_tgt, cp=cp, ct=ct, tcat=tcat, pcat=pcat, prev=prev,
                          after_think=after_think, vis=vis, base_acc=float(acc.mean())))

    classes = []  # (group, name, fn(prob) -> bool mask over positions)
    classes.append(("天花板", "全部分歧都修对", lambda p: np.ones(len(p["a"]), bool)))
    classes.append(("前缀可识别", "提议是复制型（4-gram 已出现）", lambda p: p["cp"]))
    for c in CATS:
        classes.append(("前缀可识别", f"draft 提议的 token 是「{c}」", lambda p, c=c: p["pcat"] == c))
    for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
        classes.append(("前缀可识别", f"draft 置信 p_top ∈ [{lo},{hi if hi <= 1 else 1}]",
                        lambda p, lo=lo, hi=hi: (p["p_top"] >= lo) & (p["p_top"] < hi)))
    for c in CATS[:5]:
        classes.append(("前缀可识别", f"上一个 token 是「{c}」", lambda p, c=c: p["prev"] == c))
    classes.append(("前缀可识别", "在 </think> 之后（答案段）", lambda p: p["after_think"]))
    classes.append(("前缀可识别", "p_top<0.5 且提议是「词」", lambda p: (p["p_top"] < .5) & (p["pcat"] == "词")))
    classes.append(("需知道 target", "循环型（draft 想重复、target 不）", lambda p: p["cp"] & ~p["ct"]))
    for c in CATS:
        classes.append(("需知道 target", f"target token 是「{c}」", lambda p, c=c: p["tcat"] == c))
    classes.append(("需知道 target", "近失：p_tgt ≥ 0.5·p_top", lambda p: p["p_tgt"] >= .5 * p["p_top"]))
    classes.append(("需知道 target", "近失：p_tgt ≥ 0.2·p_top", lambda p: p["p_tgt"] >= .2 * p["p_top"]))
    classes.append(("需知道 target", "换 2B 能修对的位置", lambda p: p["a2"]))

    diag, held = probs[:128], probs[128:]

    def evaluate(cls_fn, subset, frac=1.0):
        gains, cover, n_cls, n_first = [], 0, 0, 0
        for p in subset:
            dis = ~p["a"]
            m = cls_fn(p) & dis
            if frac < 1.0:
                m = m & (rng.random(len(m)) < frac)
            a_new = p["a"] | m
            gains.append(acc_mean(a_new, args.gamma) - p["base_acc"])
            n_cls += int(m.sum()); cover += int(dis.sum()); n_first += int((m & p["vis"]).sum())
        return float(np.mean(gains)), n_cls, cover, n_first

    base_h = float(np.mean([p["base_acc"] for p in held]))
    base_d = float(np.mean([p["base_acc"] for p in diag]))
    big_h = float(np.mean([acc_mean(p["a2"], args.gamma) for p in held]))
    first_h = sum(int(p["vis"].sum()) for p in held)
    res = []
    for grp, name, fn in classes:
        gd, _, _, _ = evaluate(fn, diag)
        gh, n_cls, cover, n_first = evaluate(fn, held)
        gh50, _, _, _ = evaluate(fn, held, 0.5)
        res.append(dict(group=grp, name=name, diag_rel=gd / base_d, held_abs=gh, held_rel=gh / base_h,
                        held50_rel=gh50 / base_h, share_dis=n_cls / cover, share_first=n_first / first_h, n=n_cls))
    res.sort(key=lambda x: (x["group"] != "天花板", x["group"] != "前缀可识别", -x["diag_rel"]))

    L = [f"# 修复一类分歧的乐观上限（γ={args.gamma}；诊断集 128 题排序，留出集 128 题报告）\n",
         f"留出集 base 每轮接受 {base_h:.3f}；换 2B 实测 {big_h:.3f}（{100 * (big_h / base_h - 1):+.1f}%）。"
         f"上限假设该类里所有分歧全部修对、其他位置完全不受损；“修对一半”是随机修对该类 50% 的位置。\n",
         "| 组 | 类 | 占留出集分歧的比例 | 占留出集首拒的比例 | 上限（留出，相对） | 上限（留出，绝对） | 修对一半（相对） | 上限（诊断，相对） |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['group']} | {r['name']} | {r['share_dis']:.3f} | {r['share_first']:.3f} | {100 * r['held_rel']:+.1f}% | "
                 f"{r['held_abs']:+.3f} | {100 * r['held50_rel']:+.1f}% | {100 * r['diag_rel']:+.1f}% |")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(dict(base_held=base_h, big_held=big_h, classes=res), f, indent=1, ensure_ascii=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
