"""C2 analysis: does a one-pass online update pay for itself, and how fast, per domain?

Input : <runs>/c2/m1/<dom>/{train_log.jsonl, N*/train_stats.json, N*/eval.json, none/eval.json}
Output: <out>/c2_breakeven.png, <out>/c2_acc.png and a tidy table (returned by analyze()).

Currencies
  C_train(N) : head-only GPU seconds spent updating on the first N requests (one pass, arrival order)
  s(N)       : GPU seconds per request saved at saturation, gpu_s_per_req(N0) - gpu_s_per_req(N)
  T_be(N)    : C_train(N) / s(N) future requests at saturation; minutes = T_be * gpu_s_per_req(N) / 60
  acc-model  : same, but s(N) = gpu_s_per_req(N0) * (1 - acc_len(N0)/acc_len(N)): with the step time and
               the output fixed, GPU time per request scales with 1/acceptance length. Far less noisy than
               differencing two separate throughput runs, and checked against s_measured.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cross_instance_noise(ddir):
    """CV of saturation gpu_s_per_req and of acc_len across separate engine instances (eval_rep*.json)."""
    out = {}
    for hd in sorted(glob.glob(f"{ddir}/*/")):
        reps = [json.load(open(p)) for p in sorted(glob.glob(f"{hd}/eval_rep*.json"))]
        if os.path.exists(f"{hd}/eval.json"):
            reps.append(json.load(open(f"{hd}/eval.json")))
        if len(reps) < 2:
            continue
        g = np.array([r["saturation"]["gpu_s_per_req"] for r in reps])
        dom = reps[0]["domain"]
        acc = np.array([r["sets"][dom]["acc_len"] or np.nan for r in reps], dtype=float)
        out[os.path.basename(hd.rstrip("/"))] = dict(n=len(reps), sat_mean=g.mean(), sat_cv=g.std(ddof=1) / g.mean(),
                                                     sat_min=g.min(), sat_max=g.max(),
                                                     acc_cv=float(np.nanstd(acc, ddof=1) / np.nanmean(acc)) if np.isfinite(acc).sum() > 1 else None)
    return out


def load_domain(ddir):
    rows = []
    none = json.load(open(f"{ddir}/none/eval.json")) if os.path.exists(f"{ddir}/none/eval.json") else None
    for ev in sorted(glob.glob(f"{ddir}/N*/eval.json"), key=lambda p: int(p.split("/N")[-1].split("/")[0])):
        hd = os.path.dirname(ev)
        n = int(os.path.basename(hd)[1:])
        e = json.load(open(ev))
        ts = json.load(open(f"{hd}/train_stats.json")) if os.path.exists(f"{hd}/train_stats.json") else {}
        dom = e["domain"]
        fx = json.load(open(f"{hd}/eval_fixed.json"))["saturation_fixed"] if os.path.exists(f"{hd}/eval_fixed.json") else None
        fx_none = (json.load(open(f"{ddir}/none/eval_fixed.json"))["saturation_fixed"]
                   if os.path.exists(f"{ddir}/none/eval_fixed.json") else None)
        rows.append(dict(domain=dom, N=n, C_train_gpu_s=ts.get("head_gpu_s", 0.0), steps=ts.get("step", 0),
                         fixed_gpu_s_per_req=fx["gpu_s_per_req"] if fx else None,
                         fixed_speedup_vs_none=(fx_none["gpu_s_per_req"] / fx["gpu_s_per_req"]) if fx and fx_none else None,
                         acc_len=e["sets"][dom]["acc_len"], acc_len_chat=e["sets"].get("chat", {}).get("acc_len"),
                         identical=e["sets"][dom]["identical_frac"],
                         sat_gpu_s_per_req=e["saturation"]["gpu_s_per_req"], sat_tok_s=e["saturation"]["tok_s"],
                         b1_ms_tok=e["batch1_ms_per_token"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, none
    base = df[df.N == 0].iloc[0]
    df["s_measured"] = base.sat_gpu_s_per_req - df.sat_gpu_s_per_req
    df["s_accmodel"] = base.sat_gpu_s_per_req * (1.0 - base.acc_len / df.acc_len)
    if none is not None:
        g_none = none["saturation"]["gpu_s_per_req"]
        df["spec_speedup_sat"] = g_none / df.sat_gpu_s_per_req
        # value under a load-aware on/off policy: an operator who would rather switch speculation off
        # only gains from an update if the updated head beats the better of (no spec, old head)
        best_before = min(g_none, base.sat_gpu_s_per_req)
        df["s_policy"] = best_before - np.minimum(g_none, df.sat_gpu_s_per_req)
    for col in ["s_measured", "s_accmodel", "s_policy"]:
        if col in df:
            s = df[col].where(df[col] > 0)
            df[f"Tbe_req_{col[2:]}"] = df.C_train_gpu_s / s
            df[f"Tbe_min_{col[2:]}"] = df[f"Tbe_req_{col[2:]}"] * df.sat_gpu_s_per_req / 60
    return df, none


def analyze(runs, out_dir, model="m1"):
    frames, nones = [], {}
    for ddir in sorted(glob.glob(f"{runs}/c2/{model}/*")):
        df, none = load_domain(ddir)
        if not df.empty:
            frames.append(df)
            if none is not None:
                nones[df.domain.iloc[0]] = none
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    for dom, d in df.groupby("domain"):
        d = d.sort_values("N")
        x = d.N.clip(lower=50)
        ax[0].plot(x, d.acc_len, "o-", label=dom)
        ax[1].plot(x, d.acc_len_chat, "o--", label=f"{dom}-trained, chat eval")
        if "spec_speedup_sat" in d:
            ax[2].plot(x, d.spec_speedup_sat, "o-", label=dom)
        ln, = ax[3].plot(x, d.b1_ms_tok, "o-", label=dom)
        if dom in nones:
            ax[3].axhline(nones[dom]["batch1_ms_per_token"], color=ln.get_color(), ls=":", lw=0.8)
        dd = d[d.N > 0]
        ax[4].plot(dd.N, dd.Tbe_min_accmodel, "o-", label=dom)
        if "Tbe_min_policy" in dd:
            ax[5].plot(dd.N, dd.Tbe_min_policy.fillna(1e4), "o-", label=dom)
    ax[0].set_title("acceptance length, own domain")
    ax[1].set_title("acceptance length on chat (transfer / forgetting)")
    ax[2].set_title("saturation speedup vs no-spec (1 = tie)")
    ax[2].axhline(1.0, color="k", lw=0.6)
    ax[3].set_title("batch-1 ms/token (dotted = no-spec)")
    ax[4].set_title("break-even, spec always on (min at saturation)")
    ax[4].set_yscale("log")
    ax[5].set_title("break-even, load-aware spec on/off (1e4 = never)")
    ax[5].set_yscale("log")
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("requests consumed N (N=0 drawn at 50)")
        a_.grid(alpha=0.3)
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/c2_{model}.png", dpi=150)
    plt.close(fig)
    return df


if __name__ == "__main__":
    d = analyze(sys.argv[1], sys.argv[2])
    if d is not None:
        print(d.round(4).to_string(index=False))
    for ddir in sorted(glob.glob(f"{sys.argv[1]}/c2/m1/*")):
        nz = cross_instance_noise(ddir)
        if nz:
            print("cross-instance noise", os.path.basename(ddir), json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in nz.items()}))
