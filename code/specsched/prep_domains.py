"""Build per-domain prompt files: data/domains/<dom>.jsonl with {id, prompt, split}.

Every prompt is a single user message (system text / schema folded in), so the same file
serves vLLM offline generation, `vllm bench serve --dataset-name custom`, and HF caching.
Splits: train (observed traffic, in order), eval (held-out), reserve.
"""
import json
import random

import pandas as pd

from cfg import DATA, write_jsonl

RAW = f"{DATA}/raw"
N_TRAIN, N_EVAL, N_RES = 3200, 300, 300
rng = random.Random(0)


def chat():
    df = pd.read_parquet(f"{RAW}/chat/data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet")
    return [p for p in df["prompt"].tolist() if 20 <= len(p) <= 2000]


def sql():
    rows = json.load(open(f"{RAW}/sql/sql_create_context_v4.json"))
    return [f"Database schema:\n{r['context']}\n\nWrite a SQL query that answers the question below. "
            f"Return only the SQL.\nQuestion: {r['question']}" for r in rows]


def code():
    rows = json.load(open(f"{RAW}/code/code_alpaca_20k.json"))
    out = []
    for r in rows:
        p = r["instruction"] + (f"\n\nInput:\n{r['input']}" if r.get("input") else "")
        out.append(p)
    return out


def zh():
    rows = json.load(open(f"{RAW}/zh/alpaca_gpt4_data_zh.json"))
    out = []
    for r in rows:
        p = r["instruction"] + (f"\n\n{r['input']}" if r.get("input") else "")
        if 4 <= len(p) <= 1500:
            out.append(p)
    return out


def fc():
    rows = json.load(open(f"{RAW}/fc/glaive-function-calling-v2.json"))
    out = []
    for r in rows:
        sys_txt, chat_txt = r.get("system", ""), r.get("chat", "")
        if "access to the following functions" not in sys_txt or '"parameters"' not in sys_txt \
                or "USER:" not in chat_txt:
            continue
        user = chat_txt.split("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip()
        funcs = sys_txt.split("-", 1)[1].strip() if "-" in sys_txt else sys_txt
        if not user or len(funcs) > 3000:
            continue
        out.append("You can call the following functions. If a call is needed, reply with a JSON object "
                   "{\"name\": ..., \"arguments\": {...}} and nothing else.\n"
                   f"Functions:\n{funcs}\n\nUser request: {user}")
    return out


def math():
    rows = [json.loads(l) for l in open(f"{RAW}/math500/test.jsonl")]
    return [r["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
            for r in rows]


def main():
    for name, fn in [("chat", chat), ("sql", sql), ("code", code), ("zh", zh), ("fc", fc), ("math", math)]:
        prompts = list(dict.fromkeys(fn()))  # dedupe, keep order
        rng.shuffle(prompts)
        if name == "math":
            split = ["eval"] * len(prompts)
        else:
            need = N_TRAIN + N_EVAL + N_RES
            assert len(prompts) >= need, (name, len(prompts))
            prompts = prompts[:need]
            split = ["train"] * N_TRAIN + ["eval"] * N_EVAL + ["reserve"] * N_RES
        rows = [{"id": f"{name}-{i}", "prompt": p, "split": s} for i, (p, s) in enumerate(zip(prompts, split))]
        write_jsonl(f"{DATA}/domains/{name}.jsonl", rows)
        # vLLM bench custom dataset (eval split only); output_tokens set by the caller
        write_jsonl(f"{DATA}/bench/{name}_eval.jsonl",
                    [{"prompt": r["prompt"]} for r in rows if r["split"] == "eval"])
        print(name, len(rows), "mean chars", sum(len(r["prompt"]) for r in rows) // len(rows))


if __name__ == "__main__":
    main()
