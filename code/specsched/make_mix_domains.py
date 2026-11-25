"""Mixed-traffic domains for the mixed-traffic experiment: every file a domain has, built by interleaving two existing
domains' files row by row (so the mixture is uniform at every scale), nothing generated anew:
  data/domains/<mix>.jsonl         rows {id, prompt, split, src}   (train 1600+1600, eval 150+150, reserve 150+150)
  data/traffic/m1/<mix>.jsonl      the same rows' served traffic (token ids, outputs) copied from the components
  data/bench/<mix>_eval.jsonl      150 + 150 of the components' eval prompts (frontier sweeps)
  data/bench/<mix>_pool.jsonl      8000 + 8000 of the components' unique pools, minus the chat prompts that the
                                   matrix's A1/A2 phases use (data/bench/abl/chat_r*_A{1,2}.jsonl)
  data/bench/abl/<mix>_r<seed>.jsonl   the pool reshuffled per repeat (as make_abl_pools.py)
Usage: python make_mix_domains.py [mixcc,mixzc]
"""
import glob
import random
import sys

from cfg import DATA, read_jsonl, write_jsonl

MIXES = {"mixcc": ("code", "chat"), "mixzc": ("zh", "chat")}
N_POOL = 8000           # per component: 16,000 unique B requests in total, as in the matrix
SEEDS = [1, 2, 3]


def interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out += [x, y]
    return out


def main(names):
    a_used = {r["prompt"] for p in glob.glob(f"{DATA}/bench/abl/chat_r*_A[12].jsonl") for r in read_jsonl(p)}
    for mix in names:
        d1, d2 = MIXES[mix]
        # domain rows and traffic rows, per split, in the components' order
        dom = {d: read_jsonl(f"{DATA}/domains/{d}.jsonl") for d in (d1, d2)}
        trf = {d: {r["id"]: r for r in read_jsonl(f"{DATA}/traffic/m1/{d}.jsonl")} for d in (d1, d2)}
        rows, traffic = [], []
        for split, n in (("train", 1600), ("eval", 150), ("reserve", 150)):
            parts = [[r for r in dom[d] if r["split"] == split][:n] for d in (d1, d2)]
            assert all(len(p) == n for p in parts), (mix, split, [len(p) for p in parts])
            for r in interleave(*parts):
                i = len(rows)
                rows.append(dict(id=f"{mix}-{i}", prompt=r["prompt"], split=split, src=r["id"]))
                t = dict(trf[r["id"].split("-")[0]][r["id"]])
                t["id"], t["src"] = f"{mix}-{i}", r["id"]
                traffic.append(t)
        write_jsonl(f"{DATA}/domains/{mix}.jsonl", rows)
        write_jsonl(f"{DATA}/traffic/m1/{mix}.jsonl", traffic)
        ev = [read_jsonl(f"{DATA}/bench/{d}_eval.jsonl")[:150] for d in (d1, d2)]
        write_jsonl(f"{DATA}/bench/{mix}_eval.jsonl", interleave(*ev))
        pools = []
        for d in (d1, d2):
            p = [r for r in read_jsonl(f"{DATA}/bench/{d}_pool.jsonl") if r["prompt"] not in a_used]
            assert len(p) >= N_POOL, (d, len(p))
            pools.append(p[:N_POOL])
        pool = interleave(*pools)
        write_jsonl(f"{DATA}/bench/{mix}_pool.jsonl", pool)
        for s in SEEDS:
            rng = random.Random(1000 + s)
            rs = list(pool)
            rng.shuffle(rs)
            write_jsonl(f"{DATA}/bench/abl/{mix}_r{s}.jsonl", rs)
        print(f"{mix} = {d1}+{d2}: domains {len(rows)}, traffic {len(traffic)}, eval {sum(len(e) for e in ev)}, "
              f"pool {len(pool)}, abl pools x{len(SEEDS)}", flush=True)


if __name__ == "__main__":
    main((sys.argv[1] if len(sys.argv) > 1 else "mixcc,mixzc").split(","))
