"""Runtime speculation switch check on one server: bench a domain at saturation with speculation on, switch it
off through POST /collective_rpc set_speculation, bench again, switch back on, bench a third time.
Output: runs/switch/<model>_<dom>/{summary.json, phase*.json, server.log}
"""
import argparse
import json
import os
import time

from c1_run import bench, default_port, kill, scrape, start_server
from cfg import DATA, RUNS, model
from hotswap_run import post_rpc


def acc_len(d):
    drafts = sum(v for k, v in d.items() if "num_drafts" in k)
    acc = sum(v for k, v in d.items() if "num_accepted_tokens" in k and "per_pos" not in k)
    return (1 + acc / drafts) if drafts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--dom", default="code")
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    mc = model(a.model)
    port = default_port()
    outdir = f"{RUNS}/switch/{a.model}_{a.dom}"
    os.makedirs(outdir, exist_ok=True)
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    pool = f"{DATA}/bench/{a.dom}_pool.jsonl"
    rg = dict(dom=a.dom, thinking=False, outlen=512, maxlen=4096, nprompt=lambda c: a.n, file=pool)
    rows, srv = [], None
    try:
        srv, ts = start_server(mc, port, 0.90, 4096, True, 3, f"{outdir}/server.log")
        print(f"server up in {ts:.0f}s", flush=True)
        bench(mc, port, dict(rg, nprompt=lambda c: 16, file=None), 8, outdir, "warmup.json")
        for i, on in enumerate([True, False, True]):
            dt = None
            if i > 0:
                dt, txt = post_rpc(port, "set_speculation", {"enabled": "1" if on else "0"})
                print(f"set_speculation {on} in {dt:.3f}s resp={txt[:80]}", flush=True)
            m0 = scrape(port)
            r = bench(mc, port, rg, a.C, outdir, f"phase{i}_{'on' if on else 'off'}.json")
            d = {k: v - m0.get(k, 0) for k, v in scrape(port).items()}
            rows.append(dict(phase=i, spec=on, C=a.C, switch_s=dt, output_throughput=r["output_throughput"],
                             median_tpot_ms=r.get("median_tpot_ms"), p99_tpot_ms=r.get("p99_tpot_ms"), acc_len=acc_len(d),
                             drafts=sum(v for k, v in d.items() if "num_drafts" in k)))
            print(json.dumps(rows[-1]), flush=True)
            json.dump(rows, open(f"{outdir}/summary.json", "w"), indent=1)
    finally:
        kill(srv)
        time.sleep(5)


if __name__ == "__main__":
    main()
