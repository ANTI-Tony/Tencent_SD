#!/usr/bin/env python
"""CPU screen for targeted token replacement rules (the editor's would-be task), on the frozen Rung 0 data.

Step 1  On the diagnostic half, at positions where 0.8B and 2B propose different tokens (a -> b), count what 9B
        actually chose: b (fixed), a (broken: 0.8B was right), other (still wrong). Rank directed pairs by net.
Step 2  For each top pair, drop the 2B: look at ALL diagnostic positions where 0.8B proposes a and search simple
        prefix conditions (previous token, previous two tokens, 0.8B confidence, combinations) that separate
        "9B says b" from "9B says a". Control: always replace a -> b.
Step 3  Freeze the rules, replay them on the held-out half: force the proposal to b wherever the rule fires, redo the
        exact gamma=5 accounting, report fixed / broken / accepted-length change. Also the reverse fold.

  python pair_screen.py --run runs/r0 --tokenizer-json <tokenizer.json> --out runs/r0/pair_screen.md
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from analyze import load_draft, simulate
from common import read_jsonl


def load(run):
    rows = read_jsonl(f"{run}/target.jsonl")
    b08, b2 = load_draft(f"{run}/draft_base.npz"), load_draft(f"{run}/draft_2b.npz")
    probs = []
    for k, r in enumerate(rows):
        s, e = b08["offsets"][k], b08["offsets"][k + 1]
        x = np.asarray(r["gen_ids"], dtype=np.int64)
        ctx = np.asarray(list(r["prompt_ids"]) + list(r["gen_ids"]), dtype=np.int64)
        P = len(r["prompt_ids"])
        idx = np.arange(len(x)) + P
        probs.append(dict(x=x, a08=b08["argmax"][s:e].astype(np.int64), p08=b08["p_top"][s:e].astype(np.float32),
                          a2b=b2["argmax"][b2["offsets"][k]:b2["offsets"][k + 1]].astype(np.int64),
                          prev1=ctx[idx - 1], prev2=ctx[idx - 2], k=k))
    return probs


def acc_of(a_bool):
    acc, _ = simulate(a_bool, 5)
    return float(acc.mean())


def step1(probs, min_probs=3):
    pairs = defaultdict(lambda: dict(fixed=0, broken=0, other=0, probs=set()))
    for p in probs:
        m = p["a08"] != p["a2b"]
        for t in np.where(m)[0]:
            a, b, xt = int(p["a08"][t]), int(p["a2b"][t]), int(p["x"][t])
            d = pairs[(a, b)]
            if xt == b:
                d["fixed"] += 1; d["probs"].add(p["k"])
            elif xt == a:
                d["broken"] += 1
            else:
                d["other"] += 1
    out = []
    for (a, b), d in pairs.items():
        out.append(dict(a=a, b=b, fixed=d["fixed"], broken=d["broken"], other=d["other"], net=d["fixed"] - d["broken"],
                        n_probs=len(d["probs"])))
    out.sort(key=lambda d: -d["net"])
    return [d for d in out if d["n_probs"] >= min_probs], len(pairs)


def conditions_for_pair(probs, a, b, min_support=8, min_probs=3):
    """All positions where 0.8B proposes a. Returns candidate rules with diagnostic fixed/broken counts."""
    recs = []  # (prob k, prev1, prev2, p08, a2b, x)
    for p in probs:
        for t in np.where(p["a08"] == a)[0]:
            recs.append((p["k"], int(p["prev1"][t]), int(p["prev2"][t]), float(p["p08"][t]), int(p["a2b"][t]), int(p["x"][t])))
    if not recs:
        return []
    K, P1, P2, C, A2, X = (np.asarray(z) for z in zip(*recs))
    isb, isa = X == b, X == a

    def score(mask, name, kind):
        n = int(mask.sum())
        if n < min_support:
            return None
        f, br = int((mask & isb).sum()), int((mask & isa).sum())
        np_ = len(set(K[mask & isb].tolist()))
        if np_ < min_probs:
            return None
        return dict(name=name, kind=kind, support=n, fixed=f, broken=br, net=f - br, n_probs=np_)

    cands = []
    # union of previous-token contexts whose diagnostic net is positive (support >= 5): a stronger, still simple rule
    good = []
    for c1 in np.unique(P1):
        m1 = P1 == c1
        if m1.sum() >= 5 and int((m1 & isb).sum()) - int((m1 & isa).sum()) > 0:
            good.append(int(c1))
    if good:
        r = score(np.isin(P1, good), "前一个 token∈{" + ",".join(map(str, good)) + "}", "prev1set")
        if r: cands.append(r)
    r = score(np.ones(len(X), bool), "总是替换", "control")
    if r: cands.append(r)
    r = score(A2 == b, "2B 也选 b（不可部署，参考）", "reference")
    if r: cands.append(r)
    for tau in (0.3, 0.5, 0.7, 0.9):
        r = score(C < tau, f"置信 p_top<{tau}", "conf")
        if r: cands.append(r)
    for c1 in np.unique(P1):
        m1 = P1 == c1
        if m1.sum() < min_support:
            continue
        r = score(m1, f"前一个 token={c1}", "prev1")
        if r: cands.append(r)
        for tau in (0.5, 0.7):
            r = score(m1 & (C < tau), f"前一个 token={c1} 且 p_top<{tau}", "prev1+conf")
            if r: cands.append(r)
        for c2 in np.unique(P2[m1]):
            m2 = m1 & (P2 == c2)
            if m2.sum() < min_support:
                continue
            r = score(m2, f"前两个 token=({c2},{c1})", "prev2")
            if r: cands.append(r)
    return cands


def apply_rule(p, a, b, kind, name):
    """mask of positions in problem p where the rule fires."""
    m = p["a08"] == a
    if kind == "control":
        return m
    if kind == "reference":
        return m & (p["a2b"] == b)
    if kind == "conf":
        tau = float(name.split("<")[1])
        return m & (p["p08"] < tau)
    if kind == "prev1":
        c1 = int(name.split("=")[1])
        return m & (p["prev1"] == c1)
    if kind == "prev1set":
        good = [int(z) for z in name.split("{")[1].rstrip("}").split(",")]
        return m & np.isin(p["prev1"], good)
    if kind == "prev1+conf":
        c1 = int(name.split("=")[1].split(" ")[0]); tau = float(name.split("<")[1])
        return m & (p["prev1"] == c1) & (p["p08"] < tau)
    if kind == "prev2":
        c2, c1 = (int(z) for z in name.split("(")[1].rstrip(")").split(","))
        return m & (p["prev1"] == c1) & (p["prev2"] == c2)
    raise ValueError(kind)


def replay(probs, rules):
    """rules: list of (a, b, kind, name). Returns per-problem accepted lengths (base, edited) and fixed/broken."""
    base, edited, fixed, broken = [], [], 0, 0
    for p in probs:
        prop = p["a08"].copy()
        fire = np.zeros(len(prop), bool)
        for a, b, kind, name in rules:
            m = apply_rule(p, a, b, kind, name) & ~fire
            prop[m] = b
            fire |= m
        a0, a1 = p["a08"] == p["x"], prop == p["x"]
        fixed += int((~a0 & a1).sum()); broken += int((a0 & ~a1).sum())
        base.append(acc_of(a0)); edited.append(acc_of(a1))
    base, edited = np.asarray(base), np.asarray(edited)
    return base.mean(), edited.mean(), fixed, broken


def run_fold(probs, diag_idx, held_idx, dec, top_pairs=40, L=None):
    diag = [probs[i] for i in diag_idx]; held = [probs[i] for i in held_idx]
    pairs, n_pairs = step1(diag)
    L.append(f"诊断集上 0.8B 与 2B 首选不同的有向 token 对共 {n_pairs} 个，其中修好的位置分布在 ≥3 题上的 {len(pairs)} 个。"
             f"前 20 个按净收益（修好 − 弄坏）排序：\n")
    L.append("| a → b | 修好（9B 选 b） | 弄坏（9B 选 a，0.8B 原本对） | 仍错（9B 选其他） | 净 | 涉及题数 |")
    L.append("|---|---|---|---|---|---|")
    for d in pairs[:20]:
        L.append(f"| {dec(d['a'])} → {dec(d['b'])} | {d['fixed']} | {d['broken']} | {d['other']} | {d['net']:+d} | {d['n_probs']} |")
    tot_fixed = sum(d["fixed"] for d in pairs); tot_net = sum(max(d["net"], 0) for d in pairs)
    L.append(f"\n所有 ≥3 题的对：修好合计 {tot_fixed}，正净值合计 {tot_net}。\n")

    # Step 2: rule search per top pair (by net), evaluated on held-out
    L.append(f"## 第 2–3 步：拿掉 2B，只看前缀能不能定位（前 {top_pairs} 对；规则在诊断集选，留出集回放）\n")
    L.append("| a → b | 触发条件 | 诊断：支持数 / 修好 / 弄坏 / 净 | 留出：修好 / 弄坏 | 留出每轮接受变化 | 对照“总是替换”留出：修好 / 弄坏 / 接受变化 | 参考“2B 也选 b”留出接受变化 |")
    L.append("|---|---|---|---|---|---|---|")
    chosen, summary = [], []
    base_held = np.mean([acc_of(p["a08"] == p["x"]) for p in held])
    for d in pairs[:top_pairs]:
        a, b = d["a"], d["b"]
        cands = conditions_for_pair(diag, a, b)
        if not cands:
            continue
        ctrl = next((c for c in cands if c["kind"] == "control"), None)
        ref = next((c for c in cands if c["kind"] == "reference"), None)
        deploy = [c for c in cands if c["kind"] not in ("control", "reference")]
        best = max(deploy, key=lambda c: (c["net"], c["fixed"])) if deploy else None
        if best is None or best["net"] <= 0:
            continue
        _, e, f_h, b_h = replay(held, [(a, b, best["kind"], best["name"])])
        rel = (e - base_held) / base_held
        if ctrl:
            _, ec, fc, bc = replay(held, [(a, b, "control", "总是替换")]); relc = (ec - base_held) / base_held
        else:
            fc = bc = 0; relc = 0.0
        if ref:
            _, er, _, _ = replay(held, [(a, b, "reference", ref["name"])]); relr = (er - base_held) / base_held
        else:
            relr = 0.0
        name_h = best["name"]
        for tok_id in [z for z in name_h.replace("(", " ").replace(")", " ").replace(",", " ").replace("=", " ").split() if z.isdigit()]:
            name_h = name_h.replace(tok_id, f"{tok_id}‹{dec(int(tok_id))}›", 1)
        if len(name_h) > 160: name_h = name_h[:160] + "…"
        L.append(f"| {dec(a)} → {dec(b)} | {name_h} | {best['support']} / {best['fixed']} / {best['broken']} / {best['net']:+d} | "
                 f"{f_h} / {b_h} | {100 * rel:+.2f}% | {fc} / {bc} / {100 * relc:+.2f}% | {100 * relr:+.2f}% |")
        chosen.append((a, b, best["kind"], best["name"]))
        summary.append(dict(a=dec(a), b=dec(b), rule=best["name"], diag_net=best["net"], held_fixed=f_h, held_broken=b_h,
                            held_rel=rel, control_rel=relc, ref_rel=relr))
    # ceiling of pair-based replacement: perfect trigger for every positive-net pair chosen on the diagnostic set
    pos_pairs = {(d["a"], d["b"]) for d in pairs if d["net"] > 0}
    base_l, ceil_l, n_fix = [], [], 0
    for p in held:
        a0 = p["a08"] == p["x"]
        m = np.zeros(len(a0), bool)
        for t in np.where(~a0)[0]:
            if (int(p["a08"][t]), int(p["x"][t])) in pos_pairs:
                m[t] = True
        n_fix += int(m.sum()); base_l.append(acc_of(a0)); ceil_l.append(acc_of(a0 | m))
    L.append(f"\n**对级替换的天花板**：诊断集净值为正的 {len(pos_pairs)} 个对，在留出集上假设触发条件完美（恰好在 9B 选 b 的位置改、别处不动）："
             f"修好 {n_fix} 个位置，每轮接受 {np.mean(base_l):.3f} → {np.mean(ceil_l):.3f}（{100 * (np.mean(ceil_l) / np.mean(base_l) - 1):+.2f}%）。\n")
    # combined
    if chosen:
        _, e_all, f_all, b_all = replay(held, chosen)
        pos = [c for c, s in zip(chosen, summary) if s["held_fixed"] > s["held_broken"]]
        _, e_pos, f_pos, b_pos = replay(held, pos) if pos else (0, base_held, 0, 0)
        ctrl_all = [(a, b, "control", "总是替换") for a, b, _, _ in chosen]
        _, e_ctrl, f_ctrl, b_ctrl = replay(held, ctrl_all)
        ref_all = [(a, b, "reference", "2B") for a, b, _, _ in chosen]
        _, e_ref, _, _ = replay(held, ref_all)
        L.append(f"\n**合并回放（留出集 base 每轮接受 {base_held:.3f}）**\n")
        L.append("| 方案 | 规则数 | 留出修好 | 留出弄坏 | 留出每轮接受 | 相对变化 |")
        L.append("|---|---|---|---|---|---|")
        L.append(f"| 诊断集选出的全部规则 | {len(chosen)} | {f_all} | {b_all} | {e_all:.3f} | {100 * (e_all / base_held - 1):+.2f}% |")
        L.append(f"| 其中留出集上净值仍为正的规则（事后挑选，偏乐观） | {len(pos)} | {f_pos} | {b_pos} | {e_pos:.3f} | {100 * (e_pos / base_held - 1):+.2f}% |")
        L.append(f"| 对照：同样的对，总是替换 | {len(ctrl_all)} | {f_ctrl} | {b_ctrl} | {e_ctrl:.3f} | {100 * (e_ctrl / base_held - 1):+.2f}% |")
        L.append(f"| 参考：同样的对，只在 2B 也选 b 时替换（不可部署） | {len(ref_all)} | — | — | {e_ref:.3f} | {100 * (e_ref / base_held - 1):+.2f}% |")
        n_pos_held = sum(1 for s in summary if s["held_fixed"] > s["held_broken"])
        L.append(f"\n诊断集净值为正的 {len(summary)} 条规则里，留出集上修好 > 弄坏的有 {n_pos_held} 条。\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-pairs", type=int, default=40)
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer_json)
    cache = {}

    def dec(i):
        if i not in cache:
            s = tok.decode([int(i)], skip_special_tokens=False)
            cache[i] = "`" + repr(s)[1:-1].replace("|", "\\|") + "`"
        return cache[i]

    probs = load(args.run)
    n = len(probs)
    L = [f"# 定向替换规则筛查（0.8B 提议 a → 改成 b；2B 提候选、9B 事后判分；{n} 题，γ=5 精确回放）\n"]
    L.append("## 折 1：诊断 = 前 128 题，留出 = 后 128 题\n")
    L.append("### 第 1 步：有向 token 对的净收益（诊断集，仅 0.8B≠2B 的位置）\n")
    s1 = run_fold(probs, list(range(n // 2)), list(range(n // 2, n)), dec, args.top_pairs, L)
    L.append("\n## 折 2（反向）：诊断 = 后 128 题，留出 = 前 128 题\n")
    L.append("### 第 1 步\n")
    s2 = run_fold(probs, list(range(n // 2, n)), list(range(n // 2)), dec, args.top_pairs, L)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(dict(fold1=s1, fold2=s2), f, indent=1, ensure_ascii=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
