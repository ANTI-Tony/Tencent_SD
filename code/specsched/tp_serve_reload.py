"""Hot-swap under tensor parallelism: `vllm serve -tp N` with EAGLE-3, bench one domain with the stock head,
reload the domain-adapted head (runs/c2/<model>/<dom>/N3200) through POST /collective_rpc, bench again, then
reload the stock head and bench a third time. Records throughput / TPOT / acceptance per phase and the swap
seconds. Output: runs/tp/<model>_tp<N>_<dom>/{summary.json, phase*.json, server.log}
"""
import argparse
import json
import os
import time

from c1_run import bench, default_port, kill, scrape, start_server
from cfg import RUNS, model
from hotswap_run import post_rpc


def acc_len(d):
    drafts = sum(v for k, v in d.items() if "num_drafts" in k)
    acc = sum(v for k, v in d.items() if "num_accepted_tokens" in k and "per_pos" not in k)
    return (1 + acc / drafts) if drafts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dom", default="code")
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--adapted", default="", help="head dir; default runs/c2/<model>/<dom>/N3200")
    ap.add_argument("--util", type=float, default=0.85)
    a = ap.parse_args()
    mc = model(a.model)
    adapted = a.adapted or f"{RUNS}/c2/{a.model}/{a.dom}/N3200"
    heads = [("stock", mc["head"]), ("adapted", adapted), ("stock_again", mc["head"])]
    port = default_port()
    outdir = f"{RUNS}/tp/{a.model}_tp{a.tp}_{a.dom}"
    os.makedirs(outdir, exist_ok=True)
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    rg = dict(dom=a.dom, thinking=False, outlen=512, maxlen=4096, nprompt=lambda c: a.n)
    rows, srv = [], None
    try:
        srv, ts = start_server(mc, port, a.util, 4096, True, 3, f"{outdir}/server.log", tp=a.tp)
        print(f"server up in {ts:.0f}s (tp={a.tp})", flush=True)
        bench(mc, port, dict(rg, nprompt=lambda c: 16), 8, outdir, "warmup.json")
        for i, (name, path) in enumerate(heads):
            swap_s = None
            if i > 0:
                swap_s, txt = post_rpc(port, "reload_draft_weights", {"weights_path": path})
                print(f"swap -> {name} in {swap_s:.2f}s resp={txt[:100]}", flush=True)
            m0 = scrape(port)
            r = bench(mc, port, rg, a.C, outdir, f"phase{i}_{name}.json")
            d = {k: v - m0.get(k, 0) for k, v in scrape(port).items()}
            rows.append(dict(phase=i, head=name, head_path=path, tp=a.tp, C=a.C, swap_s=swap_s,
                             output_throughput=r["output_throughput"], median_tpot_ms=r.get("median_tpot_ms"),
                             p99_tpot_ms=r.get("p99_tpot_ms"), acc_len=acc_len(d)))
            print(json.dumps(rows[-1]), flush=True)
            json.dump(rows, open(f"{outdir}/summary.json", "w"), indent=1)
    finally:
        kill(srv)
        time.sleep(5)


if __name__ == "__main__":
    main()
