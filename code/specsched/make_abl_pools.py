"""Per-repeat prompt orders for the ablation matrix: data/bench/abl/<dom>_r<seed>.jsonl (the unique pool of
make_bench_pool.py reshuffled with the repeat's seed) and, for the light A phases, two disjoint chat slices
chat_r<seed>_A1.jsonl / chat_r<seed>_A2.jsonl so that a head trained on A1's prompts is never measured on them.
"""
import random
import sys

from cfg import DATA, read_jsonl, write_jsonl


def main(seeds, doms, n_a=400):
    for s in seeds:
        rng = random.Random(1000 + s)
        for dom in doms:
            rows = read_jsonl(f"{DATA}/bench/{dom}_pool.jsonl")
            rng.shuffle(rows)
            write_jsonl(f"{DATA}/bench/abl/{dom}_r{s}.jsonl", rows)
            print(f"ABL {dom} r{s}: {len(rows)} prompts", flush=True)
        chat = read_jsonl(f"{DATA}/bench/chat_pool.jsonl")
        rng.shuffle(chat)
        write_jsonl(f"{DATA}/bench/abl/chat_r{s}_A1.jsonl", chat[:n_a])
        write_jsonl(f"{DATA}/bench/abl/chat_r{s}_A2.jsonl", chat[n_a:2 * n_a])
        print(f"ABL chat r{s}: A1/A2 {n_a}+{n_a} of {len(chat)}", flush=True)


if __name__ == "__main__":
    main([int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3").split(",")],
         (sys.argv[2] if len(sys.argv) > 2 else "code,zh").split(","))
