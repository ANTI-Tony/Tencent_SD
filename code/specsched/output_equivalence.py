"""Reviewer priority 3 (analysis only): are greedy B-phase outputs the same across strategies for the same prompts?
Every run of a scenario and seed serves the same 16,000 prompts in the same order (same pool file, same bench seed).
Reference = the frozen PLAIN run (nospec, no drafter) of that scenario and seed. Plain-vs-plain controls: frozen
SWITCH on 8B code (speculation off for almost all of B) and the E28a switch arm (other batch)."""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = "/Applications/proj_hl/brainstorm/spec_decoding"
OUT = f"{ROOT}/results/output_equivalence"
SETS = [("abl", ["m1_code", "m1_zh", "m2_code"]), ("abl_mix", ["m1_mixcc", "m1_mixzc"]), ("abl_arm", ["m1_code", "m1_zh", "m2_code", "m1_mixcc", "m1_mixzc"])]


def load(d):
    j = json.load(open(f"{d}/phaseB.json"))
    return j["generated_texts"], np.array(j["output_lens"]), np.array(j["input_lens"]), j["total_output_tokens"]


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    refs = {}
    for st, scens in SETS:
        for sc in scens:
            for seed in (1, 2, 3):
                ref_dir = f"{ROOT}/data/runs/{'abl_mix' if 'mix' in sc else 'abl'}/{sc}/nospec/r{seed}"
                if (sc, seed) not in refs:
                    refs[(sc, seed)] = load(ref_dir)
                rt, rl, ri, rtot = refs[(sc, seed)]
                for d in sorted(glob.glob(f"{ROOT}/data/runs/{st}/{sc}/*/r{seed}")):
                    arm = d.split("/")[-2]
                    if st != "abl_arm" and arm == "nospec":
                        continue
                    t, l, i, tot = load(d)
                    n = min(len(t), len(rt))
                    same_in = float((i[:n] == ri[:n]).mean())
                    eq = np.array([t[k] == rt[k] for k in range(n)])
                    leq = l[:n] == rl[:n]
                    diff = np.where(~eq)[0]
                    pref = [common_prefix(t[k], rt[k]) / max(1, min(len(t[k]), len(rt[k]))) for k in diff[:4000]]
                    rows.append(dict(set=st, scenario=sc, seed=seed, arm=arm, n=n, same_input=same_in, exact=float(eq.mean()),
                                     same_len=float(leq.mean()), out_tok_ratio=tot / rtot,
                                     mismatch_prefix_frac_median=float(np.median(pref)) if pref else np.nan))
                    print(st, sc, seed, arm, "same_input %.4f exact %.4f same_len %.4f tok_ratio %.5f" % (
                        same_in, eq.mean(), leq.mean(), tot / rtot), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/pairs_vs_plain.csv", index=False)
    g = df.groupby(["set", "scenario", "arm"]).agg(exact=("exact", "mean"), exact_min=("exact", "min"), same_len=("same_len", "mean"),
                                                   tok_ratio=("out_tok_ratio", "mean"), tok_ratio_min=("out_tok_ratio", "min"),
                                                   tok_ratio_max=("out_tok_ratio", "max"), same_input=("same_input", "min"),
                                                   prefix_med=("mismatch_prefix_frac_median", "median")).reset_index()
    g.to_csv(f"{OUT}/summary_vs_plain.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(g.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
