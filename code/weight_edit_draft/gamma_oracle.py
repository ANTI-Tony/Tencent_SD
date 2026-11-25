#!/usr/bin/env python
"""Dynamic draft length: oracle upper bound vs best fixed gamma, replayed exactly on the frozen greedy trajectories.

Cost model per round: gamma * c_d + c_v  (draft forward per token c_d, one target verification c_v; both in units of
c_v, r = c_d / c_v). Target-only decoding costs 1 per token. With greedy and a known agreement sequence a_t the oracle
never drafts past a disagreement and never splits a run of agreements: round length = min(run, gamma_max).

  python gamma_oracle.py --run runs/r0 --out runs/r0/gamma_oracle.md
"""
import argparse
import json

import numpy as np

from analyze import load_draft
from common import read_jsonl


def rounds_fixed(a, g):
    T, pos, acc = len(a), 0, []
    while pos < T:
        gg = min(g, T - pos)
        k = 0
        while k < gg and a[pos + k]:
            k += 1
        acc.append((gg, k))            # (drafted, accepted)
        pos += k + 1
    return acc


def rounds_oracle(a, gmax):
    T, pos, acc = len(a), 0, []
    while pos < T:
        k = 0
        while pos + k < T and a[pos + k] and k < gmax:
            k += 1
        acc.append((k, k))             # draft exactly the run (0 if the next token is a disagreement)
        pos += k + 1
    return acc


def cost(rounds, r):
    return sum(d * r + 1.0 for d, _ in rounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = read_jsonl(f"{args.run}/target.jsonl")
    drafts = {"0.8B": load_draft(f"{args.run}/draft_base.npz"), "2B": load_draft(f"{args.run}/draft_2b.npz")}
    A = {}
    for nm, d in drafts.items():
        A[nm] = [d["argmax"][d["offsets"][k]:d["offsets"][k + 1]] == np.asarray(r["gen_ids"], dtype=np.int32)
                 for k, r in enumerate(rows)]
    T_total = sum(len(r["gen_ids"]) for r in rows)
    gammas = list(range(1, 9))
    ratios = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    L = ["# 动态草稿长度的 oracle 上限（greedy 轨迹精确回放，256 题）\n",
         "成本单位 = 一次 target 验证；r = 每个草稿 token 的起草成本 / 一次验证成本。target 单独解码 = 每 token 1。"
         "固定 γ 在 1–8 里按同一 r 选最优（相当于验证集调好的固定长度）；oracle 提前知道逐位匹配，每轮恰好起草整段匹配（上限 γmax）。"
         "加速比 = target 单独解码成本 / 该方案成本。\n"]
    res = {}
    for nm in drafts:
        fixed = {g: sum(cost(rounds_fixed(a, g), 1.0) for a in A[nm]) for g in gammas}      # placeholder, recomputed per r
        L.append(f"## draft {nm}\n")
        L.append("| r | 最优固定 γ | 固定 γ 加速比 | oracle γmax=8 加速比 | oracle γmax=16 加速比 | oracle(8) 比最优固定省时 | oracle(16) 比最优固定省时 | 固定 γ=5 加速比 |")
        L.append("|---|---|---|---|---|---|---|---|")
        res[nm] = []
        pre_f = {g: [rounds_fixed(a, g) for a in A[nm]] for g in gammas}
        pre_o = {gm: [rounds_oracle(a, gm) for a in A[nm]] for gm in (8, 16)}
        for r in ratios:
            cf = {g: sum(cost(rr, r) for rr in pre_f[g]) for g in gammas}
            gbest = min(cf, key=cf.get)
            co = {gm: sum(cost(rr, r) for rr in pre_o[gm]) for gm in (8, 16)}
            row = dict(r=r, gbest=gbest, sp_fixed=T_total / cf[gbest], sp_o8=T_total / co[8], sp_o16=T_total / co[16],
                       save8=1 - co[8] / cf[gbest], save16=1 - co[16] / cf[gbest], sp_g5=T_total / cf[5])
            res[nm].append(row)
            L.append(f"| {r} | {gbest} | {row['sp_fixed']:.2f}× | {row['sp_o8']:.2f}× | {row['sp_o16']:.2f}× | "
                     f"{100 * row['save8']:.1f}% | {100 * row['save16']:.1f}% | {row['sp_g5']:.2f}× |")
        L.append("")
    # run-length distribution (what the oracle exploits)
    runs = []
    for a in A["0.8B"]:
        k = 0
        for x in a:
            if x:
                k += 1
            else:
                runs.append(k); k = 0
        runs.append(k)
    runs = np.asarray(runs)
    L.append("## 0.8B 一致游程分布（从每个 target 补的 token 之后数起）\n")
    L.append("| 统计 | 值 |\n|---|---|")
    L.append(f"| 游程数 | {len(runs)} |\n| 均值 | {runs.mean():.2f} |\n| 中位数 | {np.median(runs):.0f} |\n| ≥5 的比例 | {(runs >= 5).mean():.3f} |\n| ≥8 的比例 | {(runs >= 8).mean():.3f} |\n| ≥16 的比例 | {(runs >= 16).mean():.3f} |\n| 0（下一个就拒）的比例 | {(runs == 0).mean():.3f} |")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    with open(args.out.replace(".md", ".json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
