"""Generate the 'served traffic' of each domain with the target model (greedy), saving token ids.

Output: data/traffic/<model>/<domain>.jsonl rows {id, split, prompt_token_ids, output_token_ids, n_out, finish}
"""
import argparse
import os

from cfg import DATA, model, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--domains", default="chat,sql,code,zh,fc")
    ap.add_argument("--splits", default="train,eval,reserve")
    ap.add_argument("--limit", type=int, default=0, help="max rows per split (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    mc = model(a.model)
    tok = AutoTokenizer.from_pretrained(mc["target"])
    llm = LLM(model=mc["target"], gpu_memory_utilization=0.90, max_model_len=a.max_model_len,
              enable_prefix_caching=False, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    splits = a.splits.split(",")
    for dom in a.domains.split(","):
        rows = [r for r in read_jsonl(f"{DATA}/domains/{dom}.jsonl") if r["split"] in splits]
        if a.limit:
            per = {}
            rows = [r for r in rows if per.setdefault(r["split"], []).append(1) or len(per[r["split"]]) <= a.limit]
        prompts = []
        for r in rows:
            ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=True,
                                          add_generation_prompt=True, enable_thinking=bool(a.thinking))
            if hasattr(ids, "keys"):  # transformers 5 returns BatchEncoding
                ids = ids["input_ids"]
            ids = [int(x) for x in ids]
            prompts.append(ids[: a.max_model_len - a.max_tokens])
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
        res = []
        for r, p, o in zip(rows, prompts, outs):
            c = o.outputs[0]
            res.append(dict(id=r["id"], split=r["split"], prompt_token_ids=p,
                            output_token_ids=list(c.token_ids), n_out=len(c.token_ids),
                            finish=c.finish_reason))
        out = f"{DATA}/traffic/{a.model}/{dom}.jsonl"
        write_jsonl(out, res)
        n_out = [x["n_out"] for x in res]
        print(f"TRAFFIC {dom}: {len(res)} rows, mean out {sum(n_out)/len(n_out):.0f}, "
              f"truncated {sum(x['finish']=='length' for x in res)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
