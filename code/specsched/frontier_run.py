"""Load x checkpoint speculation frontier: for one domain, sweep concurrency with no speculation (S0) and with
each checkpoint head (S1/<head>), on the domain's own eval prompts (natural stop, <=512 tokens).

Output: runs/frontier/<model>_<domain>/{S0,S1_<head>}/{summary.json, C*.json, server.log}
Reuses c1_run's server / bench / metrics helpers.
"""
import argparse
import json
import os
import time

from c1_run import bench, default_port, kill, scrape, start_server
from cfg import RUNS, model

LEVELS = [1, 4, 16, 64, 128, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--heads", default="none,N0,N100,N400,N3200", help="'none' = S0; others = runs/c2 head dirs")
    ap.add_argument("--levels", default=",".join(map(str, LEVELS)))
    ap.add_argument("--nspec", type=int, default=3)
    ap.add_argument("--outlen", type=int, default=512)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=0)
    a = ap.parse_args()
    a.port = a.port or default_port()
    print(f"using port {a.port}", flush=True)
    mc = model(a.model)
    levels = [int(x) for x in a.levels.split(",")]
    rg = dict(dom=a.domain, thinking=False, outlen=a.outlen, maxlen=4096,
              nprompt=lambda c: min(max(32, 4 * c), 1024))
    for head in a.heads.split(","):
        if a.deadline and time.time() > a.deadline - 300:
            print(f"[{head}] not enough time left, skipping", flush=True)
            continue
        name = "S0" if head == "none" else f"S1_{head}"
        outdir = f"{RUNS}/frontier/{a.model}_{a.domain}/{name}"
        if os.path.exists(f"{outdir}/summary.json") and len(json.load(open(f"{outdir}/summary.json"))) >= len(levels):
            print(f"[{name}] already complete, skipping", flush=True)
            continue
        os.makedirs(outdir, exist_ok=True)
        m = dict(mc)
        if head != "none":
            m["head"] = f"{RUNS}/c2/{a.model}/{a.domain}/{head}"
        rows, srv = [], None
        try:
            srv, t_start = start_server(m, a.port, 0.90, rg["maxlen"], head != "none", a.nspec, f"{outdir}/server.log")
            print(f"[{name}] server up in {t_start:.0f}s", flush=True)
            bench(m, a.port, dict(rg, nprompt=lambda c: 8), 4, outdir, "warmup.json")
            for c in levels:
                if a.deadline and time.time() > a.deadline:
                    print(f"[{name}] deadline reached, skipping C>={c}", flush=True)
                    break
                m0, t0 = scrape(a.port), time.time()
                res = bench(m, a.port, rg, c, outdir)
                t1, m1 = time.time(), scrape(a.port)
                d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
                drafts = sum(v for k, v in d.items() if "num_drafts" in k)
                acc = sum(v for k, v in d.items() if "num_accepted_tokens" in k and "per_pos" not in k)
                row = dict(cond=name, head=head, domain=a.domain, C=c, t0=t0, t1=t1,
                           acc_len=(1 + acc / drafts) if drafts else None,
                           **{k: res.get(k) for k in res if not isinstance(res.get(k), list)})
                rows.append(row)
                print(f"[{name}] C={c} out_tok/s={res.get('output_throughput', 0):.0f} "
                      f"tpot_p50={res.get('median_tpot_ms', 0):.1f} p99={res.get('p99_tpot_ms', 0):.1f} "
                      f"acc_len={row['acc_len']}", flush=True)
                json.dump(rows, open(f"{outdir}/summary.json", "w"), indent=1)
        finally:
            kill(srv)
            time.sleep(5)


if __name__ == "__main__":
    main()
