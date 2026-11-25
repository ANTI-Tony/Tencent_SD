"""Shared helpers for the Rung 0 scripts (weight-edited draft vs. base draft, greedy speculative decoding)."""
import json
import os
import time

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."   # MIT exp8 user turn


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_tokenizer(path, revision=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, revision=revision)


def chat_prompt_ids(tok, problem, thinking):
    """Token ids of the chat-templated user turn (problem + MIT suffix), generation prompt added.
    enable_thinking is passed explicitly: the 0.8B/2B templates default to thinking OFF, the 4B/9B ones to ON."""
    msgs = [{"role": "user", "content": problem + SUFFIX}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return list(ids), text


def stop_token_ids(tok, model_path, revision=None):
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", -1):
            ids.add(int(i))
    try:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained(model_path, revision=revision)
        e = gc.eos_token_id
        if isinstance(e, int):
            ids.add(e)
        elif e:
            ids.update(int(x) for x in e)
    except Exception:
        pass
    return sorted(ids)


def load_model(path, revision=None, device="cuda"):
    """bf16 causal LM. Qwen3.5 checkpoints are multimodal releases (Qwen3_5ForConditionalGeneration); text-only
    forward with input_ids works through the image-text-to-text class when the causal-LM mapping is missing."""
    import torch
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    def _load(cls):
        try:
            return cls.from_pretrained(path, revision=revision, dtype=dtype)
        except TypeError:
            return cls.from_pretrained(path, revision=revision, torch_dtype=dtype)
    try:
        m = _load(AutoModelForCausalLM)
    except Exception as e:  # noqa: BLE001
        log(f"AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:160]}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        m = _load(AutoModelForImageTextToText)
    return m.to(device).eval()
