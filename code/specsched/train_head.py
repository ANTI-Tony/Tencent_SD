"""EAGLE-3 head: aux-layer calibration, cache building, one-pass online updates, co-located load.

Modes
  calib  : teacher-forced depth-1 top-1 match of the stock head for several aux-layer variants
  cache  : target forward on the first N train requests of a domain -> packed tensors for `load`
  stream : one-pass online distillation over a domain's train traffic, in arrival order;
           exports vLLM draft folders after N requests in --budgets; logs head-only GPU time
  load   : co-located trainer for C1 -- trains on the cache forever at a duty cycle, logs tok/s
  live   : co-located trainer fed by the serving engine's live tap (livetap.py): assembles finished
           requests from the shared-memory ring, trains newest-first at a duty cycle, exports heads
"""
import argparse
import json
import math
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from cfg import DATA, RUNS, model, read_jsonl
from head import EagleHead, load_cfg, load_sd

DEV = "cuda:0"
VOCAB = 151936


def traffic_path(m, dom):
    return f"{DATA}/traffic/{m}/{dom}.jsonl"


def load_target(path):
    from transformers import AutoModelForCausalLM
    t = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV)  # no accelerate in env
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def load_embed(path):
    """Only the input embedding matrix (for `load`, which must not hold a second target copy)."""
    from safetensors import safe_open
    idx = os.path.join(path, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    for fn in files:
        with safe_open(os.path.join(path, fn), "pt", device="cpu") as f:
            if "model.embed_tokens.weight" in f.keys():
                return f.get_tensor("model.embed_tokens.weight").to(DEV, torch.bfloat16)
    raise KeyError("embed_tokens not found")


@torch.no_grad()
def target_aux(target, seqs, aux_ids, max_tok=8192):
    """Right-padded batched forward; returns per-sequence aux [L, 3H] (bf16, on GPU)."""
    out, batch = [], []

    def flush():
        L = max(len(s) for s in batch)
        ids = torch.zeros(len(batch), L, dtype=torch.long, device=DEV)
        att = torch.zeros(len(batch), L, dtype=torch.long, device=DEV)
        for i, s in enumerate(batch):
            ids[i, :len(s)] = torch.tensor(s, device=DEV)
            att[i, :len(s)] = 1
        hs = target(input_ids=ids, attention_mask=att, output_hidden_states=True).hidden_states
        a = torch.cat([hs[i] for i in aux_ids], -1)
        for i, s in enumerate(batch):
            out.append(a[i, :len(s)].clone())
        del hs, a
        batch.clear()

    for s in seqs:
        if batch and (len(batch) + 1) * max(max(len(x) for x in batch), len(s)) > max_tok:
            flush()
        batch.append(s)
    if batch:
        flush()
    return out


def pack(items, embed_w, t2d_idx):
    """items: (ids [L] long gpu, aux [L,3H], out_start). Returns packed head inputs + draft labels/mask."""
    E, A, P, S, Y, M = [], [], [], [], [], []
    for sid, (ids, aux, out_start) in enumerate(items):
        L = ids.shape[0]
        if L < 3:
            continue
        toks, a, lab = ids[1:-1], aux[:-2], ids[2:]
        n = L - 2
        E.append(F.embedding(toks, embed_w))
        A.append(a)
        P.append(torch.arange(n, device=DEV))
        S.append(torch.full((n,), sid, device=DEV))
        d = t2d_idx[lab]
        Y.append(d.clamp(min=0))
        # supervise only positions whose label is a generated (output) token present in the draft vocab
        M.append((torch.arange(n, device=DEV) + 2 >= out_start) & (d >= 0))
    return (torch.cat(E), torch.cat(A).to(torch.bfloat16), torch.cat(P), torch.cat(S),
            torch.cat(Y), torch.cat(M))


def head_loss(head, batch):
    emb, aux, pos, seg, y, m = batch
    with torch.autocast("cuda", torch.bfloat16):
        logits = head(emb, aux, pos, seg)
    lg = logits[m].float()
    loss = F.cross_entropy(lg, y[m])
    acc = (lg.argmax(-1) == y[m]).float().mean()
    return loss, acc, int(m.sum())


def make_opt(head, lr):
    return torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))


def opt_step(head, opt, loss, step, lr, warmup=10):
    loss.backward()
    for g in opt.param_groups:
        g["lr"] = lr * min(1.0, (step + 1) / warmup)
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)


def items_from_rows(target, rows, aux_ids, max_len):
    seqs = [(r["prompt_token_ids"] + r["output_token_ids"])[:max_len] for r in rows]
    auxs = target_aux(target, seqs, aux_ids)
    return [(torch.tensor(s, device=DEV), a, len(r["prompt_token_ids"])) for s, a, r in zip(seqs, auxs, rows)]


# ----------------------------------------------------------------------------- modes

def mode_calib(a, mc):
    target = load_target(mc["target"])
    head = EagleHead(load_sd(mc["head"]), load_cfg(mc["head"])).to(DEV).eval()
    t2d_idx = head.target_to_draft_index(VOCAB)
    rows = [r for r in read_jsonl(traffic_path(a.model, a.domain)) if r["split"] == "eval"][:a.n]
    res = {}
    for var in a.variants.split(";"):
        aux_ids = tuple(int(x) for x in var.split(","))
        with torch.no_grad():
            items = items_from_rows(target, rows, aux_ids, a.max_len)
            hit = tot = 0
            for it in items:
                _, acc, n = head_loss(head, pack([it], target.model.embed_tokens.weight, t2d_idx))
                hit += float(acc) * n
                tot += n
        res[var] = hit / max(tot, 1)
        print(f"CALIB aux={var} depth1_top1_match={res[var]:.4f} over {tot} tokens", flush=True)
    os.makedirs(f"{RUNS}/calib", exist_ok=True)
    json.dump(res, open(f"{RUNS}/calib/{a.model}_{a.domain}.json", "w"), indent=1)


def mode_cache(a, mc):
    target = load_target(mc["target"])
    rows = [r for r in read_jsonl(traffic_path(a.model, a.domain)) if r["split"] == "train"][:a.n]
    items = items_from_rows(target, rows, mc["aux"], a.max_len)
    torch.save([(ids.cpu(), aux.cpu(), s) for ids, aux, s in items], a.cache)
    print(f"CACHE wrote {len(items)} seqs, {sum(i[0].numel() for i in items)} tokens -> {a.cache}", flush=True)


def mode_stream(a, mc):
    target = load_target(mc["target"])
    head = EagleHead(load_sd(mc["head"]), load_cfg(mc["head"])).to(DEV).train()
    t2d_idx = head.target_to_draft_index(VOCAB)
    embed_w = target.model.embed_tokens.weight
    opt = make_opt(head, a.lr)
    rows = [r for r in read_jsonl(traffic_path(a.model, a.domain)) if r["split"] == "train"]
    budgets = sorted(int(x) for x in a.budgets.split(","))
    outdir = f"{RUNS}/c2/{a.model}/{a.domain}"
    os.makedirs(outdir, exist_ok=True)
    head.export(f"{outdir}/N0", mc["head"])
    log = open(f"{outdir}/train_log.jsonl", "w")
    used = step = 0
    head_ms = aux_s = 0.0
    tok_sup = 0
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    while used < budgets[-1]:
        chunk, ntok = [], 0
        while used < len(rows) and used < budgets[-1] and ntok < a.tok_per_step:
            r = rows[used]
            chunk.append(r)
            ntok += min(len(r["prompt_token_ids"]) + len(r["output_token_ids"]), a.max_len)
            used += 1
            if used in budgets:
                break
        t0 = time.time()
        items = items_from_rows(target, chunk, mc["aux"], a.max_len)
        torch.cuda.synchronize()
        aux_s += time.time() - t0
        ev0.record()
        batch = pack(items, embed_w, t2d_idx)
        loss, acc, n = head_loss(head, batch)
        opt_step(head, opt, loss, step, a.lr)
        ev1.record()
        torch.cuda.synchronize()
        head_ms += ev0.elapsed_time(ev1)
        step += 1
        tok_sup += n
        rec = dict(step=step, requests=used, loss=float(loss), acc=float(acc), sup_tokens=n,
                   head_gpu_s=head_ms / 1e3, aux_s=aux_s)
        log.write(json.dumps(rec) + "\n")
        if step % 10 == 0:
            print(json.dumps(rec), flush=True)
        if used in budgets:
            head.export(f"{outdir}/N{used}", mc["head"])
            json.dump(dict(rec, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9),
                      open(f"{outdir}/N{used}/train_stats.json", "w"), indent=1)
            print(f"EXPORT N{used} step={step} head_gpu_s={head_ms/1e3:.2f}", flush=True)
        if used >= len(rows):
            break
    log.close()


def mode_load(a, mc):
    """Stand-in for an online trainer sharing the GPU with the server: head-only compute and memory."""
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    embed_w = load_embed(mc["target"])
    head = EagleHead(load_sd(mc["head"]), load_cfg(mc["head"])).to(DEV).train()
    t2d_idx = head.target_to_draft_index(VOCAB)
    opt = make_opt(head, a.lr)
    cache = [(i.to(DEV), x.to(DEV), s) for i, x, s in torch.load(a.cache, weights_only=False)]
    rng = random.Random(0)
    log = open(a.log, "a")
    # sequential one-pass mode with checkpoint exports (hot-swap harness): consume the cache in arrival
    # order, export a vLLM-loadable head when the consumed-request count crosses each threshold
    export_at = sorted(int(x) for x in a.export_at.split(",")) if a.export_at else []
    one_pass = bool(export_at)   # with exports configured, the cache is the served traffic: train on it once
    consumed, next_item = 0, 0
    mc_head = model(a.model)["head"]
    step, t_last, tok_win, steps_win = 0, time.time(), 0, 0
    t_start, busy_win = time.time(), 0.0
    print("LOAD_READY", flush=True)
    log.write(json.dumps(dict(t=time.time(), event="ready")) + "\n")
    log.flush()
    def current_duty():
        if not a.duty_file:
            return a.duty
        try:
            return max(0.0, min(1.0, float(open(a.duty_file).read().strip() or 0)))
        except Exception:
            return a.duty

    while not stop["flag"] and (a.duration <= 0 or time.time() - t_start < a.duration):
        duty = current_duty()
        if duty <= 0.0:   # paused by the controller
            time.sleep(0.2)
            now = time.time()
            if now - t_last >= a.log_every:
                log.write(json.dumps(dict(t=now, steps=steps_win, tokens=tok_win, busy_s=busy_win, win_s=now - t_last,
                                          duty=0.0, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)) + "\n")
                log.flush()
                t_last, tok_win, steps_win, busy_win = now, 0, 0, 0.0
            continue
        if a.order == "seq":
            if next_item >= len(cache):
                if one_pass:  # one pass done: nothing left to learn from, release the GPU
                    log.write(json.dumps(dict(t=time.time(), event="exhausted", consumed=consumed)) + "\n")
                    log.flush()
                    print("EXHAUSTED", consumed, flush=True)
                    break
                next_item = 0
            batch_items, ntok = [], 0
            while next_item < len(cache) and ntok < a.tok_per_step:
                it = cache[next_item]
                batch_items.append(it)
                ntok += it[0].numel()
                next_item += 1
        else:
            rng.shuffle(cache)
            batch_items, ntok = [], 0
            for it in cache:
                if ntok >= a.tok_per_step:
                    break
                batch_items.append(it)
                ntok += it[0].numel()
        t0 = time.time()
        loss, _, n = head_loss(head, pack(batch_items, embed_w, t2d_idx))
        opt_step(head, opt, loss, step, a.lr)
        torch.cuda.synchronize()
        dt = time.time() - t0
        step += 1
        steps_win += 1
        tok_win += ntok
        busy_win += dt
        if a.order == "seq":
            consumed += len(batch_items)
            while export_at and consumed >= export_at[0]:
                n = export_at.pop(0)
                out = os.path.join(a.export_dir, f"N{n}")
                head.export(out, mc_head)
                log.write(json.dumps(dict(t=time.time(), event="export", N=n, consumed=consumed, step=step, path=out)) + "\n")
                log.flush()
                print(f"EXPORT N{n} consumed={consumed} step={step}", flush=True)
        if duty < 1.0:
            time.sleep(dt * (1.0 - duty) / duty)
        now = time.time()
        if now - t_last >= a.log_every:
            log.write(json.dumps(dict(t=now, steps=steps_win, tokens=tok_win, busy_s=busy_win,
                                      win_s=now - t_last, duty=duty, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                                      loss=float(loss))) + "\n")
            log.flush()
            t_last, tok_win, steps_win, busy_win = now, 0, 0, 0.0
    log.write(json.dumps(dict(t=time.time(), event="stop", total_steps=step)) + "\n")
    log.close()


def mode_live(a, mc):
    """Trainer fed by the live tap: no precomputed hidden states, the server's own EAGLE-3 aux states and
    committed tokens arrive through the ring buffer as requests complete."""
    from collections import deque

    from livetap import HDR, K_CHUNK, K_FINISH, K_GAP, Ring, decode_chunk, decode_id
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    embed_w = load_embed(mc["target"])
    head = EagleHead(load_sd(mc["head"]), load_cfg(mc["head"])).to(DEV).train()
    t2d_idx = head.target_to_draft_index(VOCAB)
    opt = make_opt(head, a.lr)
    export_at = sorted(int(x) for x in a.export_at.split(",")) if a.export_at else []
    one_pass = bool(export_at)   # fixed schedule: learn until the last export, then release the GPU (as `load`)
    next_doubling = a.export_doubling   # open-ended schedule 100, 200, 400 ...: never releases the GPU
    mc_head = model(a.model)["head"]
    log = open(a.log, "a")
    print("LOAD_READY", flush=True)
    log.write(json.dumps(dict(t=time.time(), event="ready")) + "\n")
    log.flush()
    while not stop["flag"] and (not os.path.exists(a.tap) or os.path.getsize(a.tap) < HDR + 16):
        time.sleep(0.5)
    if stop["flag"]:
        return
    ring = Ring(a.tap)
    if a.tap_replay:
        ring.rewind_to_oldest()
    log.write(json.dumps(dict(t=time.time(), event="tap_open", path=a.tap, write_seq=ring.write_seq)) + "\n")
    partial = {}        # req_id -> [token chunks, aux chunks, next expected position, prompt_len, broken]
    ready = deque()     # finished sequences (tokens int32[L], aux int16[L,hidden], prompt_len), newest right
    st = dict(finished=0, dropped_gap=0, dropped_short=0, dropped_overflow=0, buffered_tokens=0, chunks=0)

    def drain():
        for kind, payload in ring.read(max_records=4000):
            if kind == K_CHUNK:
                rid, n, start, plen, _hid, tok, aux = decode_chunk(payload)
                st["chunks"] += 1
                s = partial.get(rid)
                if s is None:
                    s = partial[rid] = [[], [], 0, plen, start != 0]
                if s[4]:
                    continue
                if start < s[2]:
                    # the engine recomputed a prefix (preemption + recompute): keep the committed prefix up to
                    # `start` and continue from there; the recomputed tokens are the same, the aux states fresher
                    st["recomputed"] = st.get("recomputed", 0) + 1
                    t_all, a_all = np.concatenate(s[0])[:start], np.concatenate(s[1])[:start]
                    s[0], s[1], s[2] = [t_all], [a_all], start
                elif start > s[2]:
                    s[4] = True          # a step of this request was never exported: unusable
                    continue
                s[0].append(tok)
                s[1].append(aux)
                s[2] += n
            elif kind in (K_FINISH, K_GAP):
                rid = decode_id(payload)
                s = partial.pop(rid, None)
                if s is None:
                    continue
                if s[4] or kind == K_GAP:
                    st["dropped_gap"] += 1
                    continue
                if s[2] < 3:
                    st["dropped_short"] += 1
                    continue
                tok = np.concatenate(s[0])[:a.max_len]
                aux = np.concatenate(s[1])[:a.max_len]
                ready.append((tok, aux, s[3], time.time()))
                st["finished"] += 1
                st["buffered_tokens"] += len(tok)
                while st["buffered_tokens"] > a.buffer_tokens and ready:
                    old = ready.popleft()
                    st["buffered_tokens"] -= len(old[0])
                    st["dropped_overflow"] += 1

    def current_duty():
        if not a.duty_file:
            return a.duty
        try:
            return max(0.0, min(1.0, float(open(a.duty_file).read().strip() or 0)))
        except Exception:
            return a.duty

    def flush_log(now, duty, loss):
        nonlocal t_last, tok_win, steps_win, busy_win
        log.write(json.dumps(dict(t=now, steps=steps_win, tokens=tok_win, busy_s=busy_win, win_s=now - t_last,
                                  duty=duty, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, loss=loss,
                                  consumed=consumed, buffered=len(ready), lapped=ring.lapped, **st)) + "\n")
        log.flush()
        t_last, tok_win, steps_win, busy_win = now, 0, 0, 0.0

    consumed = step = 0
    t_start = t_last = time.time()
    tok_win, steps_win, busy_win, loss_v = 0, 0, 0.0, None

    def export_now(by):
        out = os.path.join(a.export_dir, f"N{consumed}")
        if not os.path.exists(f"{out}/model.safetensors"):
            head.export(out, mc_head)
            log.write(json.dumps(dict(t=time.time(), event="export", N=consumed, consumed=consumed, step=step,
                                      path=out, by=by)) + "\n")
            log.flush()
            print(f"EXPORT N{consumed} ({by}) step={step}", flush=True)

    while not stop["flag"] and (a.duration <= 0 or time.time() - t_start < a.duration):
        drain()
        if a.cmd_file and os.path.exists(a.cmd_file):      # controller commands: export / flush
            try:
                cmd = open(a.cmd_file).read().strip()
                os.remove(a.cmd_file)
            except Exception:
                cmd = ""
            if cmd.startswith("export"):
                export_now("controller")
            elif cmd.startswith("flush"):             # traffic changed domain: buffered sequences are stale
                st["flushed"] = st.get("flushed", 0) + len(ready) + len(partial)
                ready.clear()
                partial.clear()
                st["buffered_tokens"] = 0
                log.write(json.dumps(dict(t=time.time(), event="flush", consumed=consumed)) + "\n")
                log.flush()
        now = time.time()
        while ready and a.max_age > 0 and now - ready[0][3] > a.max_age:   # oldest on the left
            old = ready.popleft()
            st["buffered_tokens"] -= len(old[0])
            st["dropped_stale"] = st.get("dropped_stale", 0) + 1
        duty = current_duty()
        if duty <= 0.0 or not ready:
            time.sleep(0.05)
            now = time.time()
            if now - t_last >= a.log_every:
                flush_log(now, duty, loss_v)
            continue
        batch_items, ntok = [], 0
        while ready and ntok < a.tok_per_step:       # newest first: the buffer is LIFO
            tok, aux, plen, _t_fin = ready.pop()
            st["buffered_tokens"] -= len(tok)
            batch_items.append((torch.from_numpy(tok).to(DEV, dtype=torch.long),
                                torch.from_numpy(aux).view(torch.bfloat16).to(DEV), plen))
            ntok += len(tok)
        t0 = time.time()
        loss, _, n = head_loss(head, pack(batch_items, embed_w, t2d_idx))
        opt_step(head, opt, loss, step, a.lr)
        torch.cuda.synchronize()
        dt = time.time() - t0
        loss_v = float(loss)
        step += 1
        steps_win += 1
        tok_win += ntok
        busy_win += dt
        consumed += len(batch_items)
        while export_at and consumed >= export_at[0]:
            n = export_at.pop(0)
            out = os.path.join(a.export_dir, f"N{n}")
            head.export(out, mc_head)
            log.write(json.dumps(dict(t=time.time(), event="export", N=n, consumed=consumed, step=step, path=out)) + "\n")
            log.flush()
            print(f"EXPORT N{n} consumed={consumed} step={step}", flush=True)
        if next_doubling and consumed >= next_doubling:
            export_now("doubling")
            while next_doubling <= consumed:
                next_doubling *= 2
        if one_pass and not export_at:
            log.write(json.dumps(dict(t=time.time(), event="exhausted", consumed=consumed)) + "\n")
            print("EXHAUSTED", consumed, flush=True)
            break
        if duty < 1.0:
            time.sleep(dt * (1.0 - duty) / duty)
        now = time.time()
        if now - t_last >= a.log_every:
            flush_log(now, duty, loss_v)
    log.write(json.dumps(dict(t=time.time(), event="stop", total_steps=step, consumed=consumed, **st)) + "\n")
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["calib", "cache", "stream", "load", "live"])
    ap.add_argument("--tap", default="", help="live mode: ring-buffer file written by the serving engine")
    ap.add_argument("--tap-replay", type=int, default=0, help="live mode: 1 = also consume what is already in the ring")
    ap.add_argument("--buffer-tokens", type=int, default=600_000, help="live mode: max tokens held in the CPU buffer")
    ap.add_argument("--cmd-file", default="", help="live mode: a file containing 'export' / 'flush' (consumed and removed)")
    ap.add_argument("--max-age", type=float, default=60.0, help="live mode: drop buffered sequences older than this (s)")
    ap.add_argument("--model", default="m1")
    ap.add_argument("--domain", default="chat")
    ap.add_argument("--variants", default="2,18,33;3,19,34")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cache", default="")
    ap.add_argument("--budgets", default="100,200,400,800,1600,3200")
    ap.add_argument("--tok-per-step", type=int, default=4096)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--duty-file", default="", help="if set, re-read the duty (a float, 0 = pause) from this file every step")
    ap.add_argument("--order", default="shuffle", choices=["shuffle", "seq"])
    ap.add_argument("--export-at", default="", help="load mode, seq order: export heads at these consumed-request counts")
    ap.add_argument("--export-dir", default="")
    ap.add_argument("--export-doubling", type=int, default=0,
                    help="live mode: export at this many consumed requests, then at 2x, 4x ... without stopping")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--log", default="")
    ap.add_argument("--log-every", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    mc = model(a.model)
    {"calib": mode_calib, "cache": mode_cache, "stream": mode_stream, "load": mode_load, "live": mode_live}[a.mode](a, mc)


if __name__ == "__main__":
    main()
