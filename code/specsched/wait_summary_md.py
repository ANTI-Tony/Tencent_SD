"""Print the experiment-E summary table (markdown) from results/abl_wait/wait_pairs.csv: phase-B throughput relative to
switch per (base, segment length, strategy) with the training start / head-live times, plus the E1-E4 verdict lines
exactly as the preregistration defines them. Usage: python wait_summary_md.py"""
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE3 = [("m2_code", "MoE"), ("m1_code", "8B code"), ("m1_mixcc", "code+chat")]
LENS = ["w1", "w2", "w3", "w4"]
STR = ["train_now", "wait", "v6", "wait_d05", "wait_d2"]


def main():
    wp = pd.read_csv(f"{ROOT}/results/abl_wait/wait_pairs.csv")

    def row(sc, st):
        x = wp[(wp.scenario == sc) & (wp.strategy == st)]
        return x.iloc[0] if len(x) else None

    def f(v, d=0):
        return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"

    def cell(sc, st):
        x = row(sc, st)
        if x is None:
            return ""
        s = f"{x.vs_switch:+.1f}"
        if st == "v6" or st.startswith("wait") or st == "train_now":
            if math.isnan(x.train_start_s):
                s += "（未训）"
            else:
                s += f" ({f(x.train_start_s)}/{f(x.live_s)})"
        return s
    print("| 场景 | 段长 | train_now | wait | v6 | wait_d05 | wait_d2 |")
    print("|---|---|---|---|---|---|---|")
    for b, nm in BASE3:
        for l in LENS:
            sc = f"{b}_{l}"
            sw = row(sc, "switch")
            cells = [cell(sc, st) for st in STR]
            best = max((st for st in STR if row(sc, st) is not None), key=lambda st: row(sc, st).vs_switch)
            cells = [f"**{c}**" if st == best and row(sc, st).vs_switch > 0 else c for st, c in zip(STR, cells)]
            print(f"| {nm} | {sw.seg_s:.0f} s | " + " | ".join(cells) + " |")
    print()
    e1 = {b: (row(f"{b}_w1", "wait").vs_switch, row(f"{b}_w1", "train_now").vs_switch) for b, _ in BASE3}
    e2 = {b: row(f"{b}_w4", "wait").vs_train_now for b, _ in BASE3}
    print("E1 (0.5s: |wait−switch| ≤ 1.5 且 train_now ≤ −3): " + "; ".join(f"{nm} wait {e1[b][0]:+.1f} train_now {e1[b][1]:+.1f} -> {'✓' if abs(e1[b][0]) <= 1.5 and e1[b][1] <= -3 else '✗'}" for b, nm in BASE3))
    print("E2 (s+τ+2s: |wait−train_now| ≤ 1): " + "; ".join(f"{nm} {e2[b]:+.1f} -> {'✓' if abs(e2[b]) <= 1 else '✗'}" for b, nm in BASE3))
    for b, nm in BASE3:
        wins = [l for l in LENS if max(STR, key=lambda st: row(f'{b}_{l}', st).vs_switch if row(f'{b}_{l}', st) is not None else -1e9) == "v6" and row(f"{b}_{l}", "v6").vs_switch > 0]
        print(f"v6 best (and > switch) in {nm}: {wins}")
    for st in ("wait_d05", "wait_d2"):
        print(st, "train_start_s by length (MoE):", [f(row(f"m2_code_{l}", st).train_start_s) for l in LENS], "| wait:", [f(row(f"m2_code_{l}", "wait").train_start_s) for l in LENS])


if __name__ == "__main__":
    main()
