"""Large pools of unique prompts for the closed-loop harness: data/bench/<dom>_pool.jsonl.

The online trainer learns from the requests the server serves, so the served prompts must not repeat (the
300-prompt *_eval.jsonl files cycle 30x in a 9000-request phase and let the head memorise them). Pools are
built from the raw sources with the same prompt formatting as prep_domains.py, minus every prompt already in
data/domains/<dom>.jsonl (train/eval/reserve), shuffled with a fixed seed.
"""
import random
import sys

import prep_domains as pdm
from cfg import DATA, read_jsonl, write_jsonl


def main(doms):
    rng = random.Random(1)
    for name in doms:
        used = {r["prompt"] for r in read_jsonl(f"{DATA}/domains/{name}.jsonl")}
        prompts = [p for p in dict.fromkeys(getattr(pdm, name)()) if p not in used]
        rng.shuffle(prompts)
        write_jsonl(f"{DATA}/bench/{name}_pool.jsonl", [{"prompt": p} for p in prompts])
        print(f"POOL {name}: {len(prompts)} unique prompts (excluded {len(used)} traffic prompts)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else ["code", "zh", "sql", "fc", "chat"])
