#!/usr/bin/env python
"""Activation-space vocabulary surgery for the AngelSlim Qwen3-8B EAGLE-3 head (GPU, no gradient descent).

Fit a linear bridge B from the head's final hidden state h_d to the target's final hidden state h_t on zh TRAIN traffic
(least squares, closed form), so that for any token v the row  r_v = B^T w_v^target  gives  r_v . h_d ~ w_v . h_t.
That lets the head score tokens outside its 32k draft vocabulary without training. Variants are exported as vLLM
head folders under runs/c2/m1/zh/<name>/ and screened offline by first-draft-token agreement on zh/chat EVAL traffic.

  python bridge_fit.py --n-fit 2000 --out-report ../weight_edit_draft/runs/vocab/bridge.md
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, "/fs/scratch/PCS0296/specsched/code")
from cfg import DATA, RUNS, model, read_jsonl  # noqa: E402
from head import EagleHead, load_cfg, load_sd  # noqa: E402
from train_head import load_target  # noqa: E402

DEV = "cuda:0"
VOCAB = 151936


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@torch.no_grad()
def head_hidden(h, emb, aux, pos, seg):
    """EagleHead.forward without the lm_head: the normalized hidden state the lm_head reads."""
    T = emb.shape[0]
    e = h.rms(emb, h.in_ln)
    g = aux @ h.fc.T.to(aux.dtype)
    residual = g.float()
    x = torch.cat([e, h.rms(g, h.hid_ln)], -1)
    q = h.rope((x @ h.q.T.to(x.dtype)).view(T, h.n_q, h.hd), pos)
    k = h.rope((x @ h.k.T.to(x.dtype)).view(T, h.n_kv, h.hd), pos)
    v = (x @ h.v.T.to(x.dtype)).view(T, h.n_kv, h.hd)
    rep = h.n_q // h.n_kv
    idx = torch.arange(T, device=emb.device)
    mask = (seg[:, None] == seg[None, :]) & (idx[:, None] >= idx[None, :])
    a = F.scaled_dot_product_attention(
        q.permute(1, 0, 2)[None], k.repeat_interleave(rep, 1).permute(1, 0, 2)[None],
        v.repeat_interleave(rep, 1).permute(1, 0, 2)[None], attn_mask=mask[None, None])
    a = a[0].permute(1, 0, 2).reshape(T, -1) @ h.o.T.to(x.dtype)
    residual = residual + a.float()
    hh = h.rms(residual, h.post_ln)
    hh = (F.silu(hh @ h.gate.T.to(hh.dtype)) * (hh @ h.up.T.to(hh.dtype))) @ h.down.T.to(hh.dtype)
    residual = residual + hh.float()
    return h.rms(residual, h.norm).float()


@torch.no_grad()
def features(target, head, rows, aux_ids, max_len, group=8, final_check=[False]):
    """Yields per group: h_d [T,H] fp32, h_t [T,H] fp32 (target final hidden at the position that predicts the same
    token), label [T] (target ids), is_out [T] (label is an output token). Packing follows train_head.pack."""
    embed_w = target.model.embed_tokens.weight
    seqs = [((r["prompt_token_ids"] + r["output_token_ids"])[:max_len], len(r["prompt_token_ids"])) for r in rows]
    for s in range(0, len(seqs), group):
        batch = seqs[s:s + group]
        L = max(len(x) for x, _ in batch)
        ids = torch.zeros(len(batch), L, dtype=torch.long, device=DEV)
        att = torch.zeros(len(batch), L, dtype=torch.long, device=DEV)
        for i, (x, _) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x, device=DEV)
            att[i, :len(x)] = 1
        out = target(input_ids=ids, attention_mask=att, output_hidden_states=True)
        hs = out.hidden_states
        if not final_check[0]:
            lg = target.lm_head(hs[-1][0, :4]).float()
            err = (lg - out.logits[0, :4].float()).abs().max().item()
            log(f"final hidden check: lm_head(hidden_states[-1]) vs logits max abs diff {err:.4f}")
            final_check[0] = True
        E, A, P, S, Y, O, HT = [], [], [], [], [], [], []
        for i, (x, out_start) in enumerate(batch):
            n = len(x) - 2
            if n < 1:
                continue
            xt = ids[i, :len(x)]
            E.append(F.embedding(xt[1:-1], embed_w))
            A.append(torch.cat([hs[j][i, :len(x)] for j in aux_ids], -1)[:-2])
            P.append(torch.arange(n, device=DEV))
            S.append(torch.full((n,), i, device=DEV))
            Y.append(xt[2:])
            O.append(torch.arange(n, device=DEV) + 2 >= out_start)
            HT.append(hs[-1][i, 1:len(x) - 1])
        with torch.autocast("cuda", torch.bfloat16):
            hd = head_hidden(head, torch.cat(E), torch.cat(A).to(torch.bfloat16), torch.cat(P), torch.cat(S))
        yield hd, torch.cat(HT).float(), torch.cat(Y), torch.cat(O)
        del out, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m1")
    ap.add_argument("--domain", default="zh")
    ap.add_argument("--n-fit", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--fit-domains", default="zh", help="comma list; the bridge is fitted on these domains' TRAIN traffic")
    ap.add_argument("--n-fit-other", type=int, default=1000, help="train rows taken from each extra fit domain")
    ap.add_argument("--tag", default="", help="suffix for exported variant names")
    ap.add_argument("--out-report", required=True)
    a = ap.parse_args()
    t0 = time.time()
    mc = model(a.model)
    target = load_target(mc["target"])
    head = EagleHead(load_sd(mc["head"]), load_cfg(mc["head"])).to(DEV).eval()
    sd, cfg = load_sd(mc["head"]), load_cfg(mc["head"])
    W_d = head.lm_head.detach().float()                       # [32000, H] trained rows
    K0 = W_d.shape[0]
    S_old = (torch.arange(K0) + sd["d2t"]).long().to(DEV)
    W_t = target.lm_head.weight.detach().float()              # [V, H]
    H = W_t.shape[1]
    in_old = torch.zeros(VOCAB, dtype=torch.bool, device=DEV); in_old[S_old] = True
    old_pos = torch.full((VOCAB,), -1, dtype=torch.long, device=DEV); old_pos[S_old] = torch.arange(K0, device=DEV)
    log(f"loaded target + head ({time.time() - t0:.0f}s); head vocab {K0}")

    traffic = read_jsonl(f"{DATA}/traffic/{a.model}/{a.domain}.jsonl")
    chat = read_jsonl(f"{DATA}/traffic/{a.model}/chat.jsonl")
    fit_rows = [r for r in traffic if r["split"] == "train"][:a.n_fit]
    n_hold = max(len(fit_rows) // 10, 50)
    hold_rows, fit_rows = fit_rows[-n_hold:], fit_rows[:-n_hold]
    extra = [d for d in a.fit_domains.split(",") if d and d != a.domain]
    for d in extra:
        rows_d = [r for r in read_jsonl(f"{DATA}/traffic/{a.model}/{d}.jsonl") if r["split"] == "train"][:a.n_fit_other]
        nh = max(len(rows_d) // 10, 20)
        hold_rows += rows_d[-nh:]; fit_rows += rows_d[:-nh]
    log(f"fit domains {[a.domain] + extra}: {len(fit_rows)} fit rows, {len(hold_rows)} holdout rows")
    zh_eval = [r for r in traffic if r["split"] == "eval"][:a.n_eval]
    chat_eval = [r for r in chat if r["split"] == "eval"][:a.n_eval]

    # ---- 1. accumulate normal equations on fit rows (output positions only) + logit-scale pairs on old tokens
    G = torch.zeros(H, H, dtype=torch.float64, device=DEV)
    M = torch.zeros(H, H, dtype=torch.float64, device=DEV)
    n_pos = 0
    hsum = torch.zeros(H, dtype=torch.float64, device=DEV)
    # logit-scale calibration between the head's trained rows and the target over the OLD vocabulary, per position:
    # ld_v - mean_v(ld) ~ s * (lt_v - mean_v(lt)),  mean_v(ld) ~ s * mean_v(lt) + b
    c_xy = c_xx = c_yy = 0.0; m_ld = m_lt = 0.0; n_cal = 0
    gen = torch.Generator(device=DEV).manual_seed(0)
    W_t_old = W_t[S_old]
    for hd, ht, y, o in features(target, head, fit_rows, mc["aux"], a.max_len):
        hd, ht, y = hd[o], ht[o], y[o]
        G += hd.double().T @ hd.double()
        M += hd.double().T @ ht.double()
        hsum += hd.double().sum(0)
        n_pos += hd.shape[0]
        k = min(32, hd.shape[0])
        sel = torch.randperm(hd.shape[0], generator=gen, device=DEV)[:k]
        ld = hd[sel] @ W_d.T; lt = ht[sel] @ W_t_old.T                     # [k, 32000]
        ldc = ld - ld.mean(1, keepdim=True); ltc = lt - lt.mean(1, keepdim=True)
        c_xy += float((ldc * ltc).sum()); c_xx += float((ltc * ltc).sum()); c_yy += float((ldc * ldc).sum())
        m_ld += float(ld.mean(1).sum()); m_lt += float(lt.mean(1).sum()); n_cal += k
    s_cal = c_xy / c_xx
    b_cal = m_ld / n_cal - s_cal * (m_lt / n_cal)
    corr_cal = c_xy / (c_xx * c_yy) ** 0.5
    log(f"normal equations over {n_pos} output positions; per-position logit scale head ~ s*target + b over the old "
        f"vocab: s={s_cal:.3f} b={b_cal:.3f} (within-position corr {corr_cal:.3f}, {n_cal} positions) ({time.time() - t0:.0f}s)")

    # ---- 2. holdout: choose lambda by full-vocab bridged agreement on held-out fit rows
    hold = [(hd[o], ht[o], y[o]) for hd, ht, y, o in features(target, head, hold_rows, mc["aux"], a.max_len)]
    u = torch.linalg.solve(G + 1e-6 * G.diagonal().mean() * torch.eye(H, dtype=G.dtype, device=DEV), hsum).float()
    uh = torch.cat([hd @ u for hd, _, _ in hold])
    log(f"constant direction u: u.h_d on holdout mean {uh.mean():.3f} sd {uh.std():.3f} (want 1, 0)")

    def agreement(rows_feats, W_rows, ids):
        """first-draft-token agreement: argmax over the variant's vocabulary (rows W_rows for target ids `ids`)."""
        hit = tot = 0
        for hd, _, y in rows_feats:
            for s in range(0, hd.shape[0], 2048):
                lg = hd[s:s + 2048] @ W_rows.T
                hit += int((ids[lg.argmax(-1)] == y[s:s + 2048]).sum()); tot += lg.shape[0]
        return hit / max(tot, 1)
    all_ids = torch.arange(VOCAB, device=DEV)
    best = None
    for lam in [float(z) for z in a.lams.split(",")]:
        B = torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(H, dtype=G.dtype, device=DEV), M)  # B^T: hd -> ht
        W_full = (W_t @ B.float().T)                 # h_t ~ B^T h_d  ->  row_v = B w_v  ->  rows = W_t @ B^T
        ag = agreement(hold, W_full, all_ids)
        rel = None
        log(f"lambda {lam}: holdout full-vocab bridged agreement {ag:.4f}")
        if best is None or ag > best[0]:
            best = (ag, lam, B.float())
    ag_hold, lam, B = best
    factory_hold = agreement(hold, W_d, S_old)
    log(f"chosen lambda {lam} (holdout bridged {ag_hold:.4f} vs factory {factory_hold:.4f})")

    # ---- 3. variants
    cnt = Counter()
    for r in fit_rows + hold_rows:
        cnt.update(r["output_token_ids"])
    ranked = [t for t, _ in cnt.most_common()]
    top = ranked[:K0]
    if len(top) < K0:
        seen = set(top); top += [t for t in S_old.tolist() if t not in seen][:K0 - len(top)]
    tot = sum(cnt.values()); acc = sum(c for t, c in cnt.items() if in_old[t]); add = []
    for t in ranked:
        if acc / tot >= 0.995:
            break
        if not in_old[t]:
            add.append(t); acc += cnt[t]
    vocabs = {"z32k": sorted(set(top)), "union": sorted(set(S_old.tolist()) | set(add)), "full": list(range(VOCAB)),
              "old": sorted(S_old.tolist())}
    W_bridge_all = W_t @ B.T                                  # [V,H] bridged rows for every token

    def rows_for(ids_t, keep_trained):
        W = W_bridge_all[ids_t].clone()
        if keep_trained:
            W = W * s_cal + b_cal * u[None, :]               # bridged rows live on the target's logit scale: map to the head's
            m = in_old[ids_t]
            W[m] = W_d[old_pos[ids_t[m]]]
        return W
    @torch.no_grad()
    def grid_calib(ids_t):
        """choose (s, b) for  new rows = s * bridged + b * u  by holdout first-step agreement (old rows kept)."""
        m_old = in_old[ids_t]
        ids_old_v, ids_new_v = ids_t[m_old], ids_t[~m_old]
        W_old_v, W_new_v = W_d[old_pos[ids_old_v]], W_bridge_all[ids_new_v]
        stats = []
        for hd, _, y in hold:
            for s0 in range(0, hd.shape[0], 1024):
                h = hd[s0:s0 + 1024]
                lo = h @ W_old_v.T; ln = h @ W_new_v.T
                mo, io = lo.max(-1); mn, inn = ln.max(-1)
                stats.append(torch.stack([mo, ids_old_v[io].float(), mn, ids_new_v[inn].float(), h @ u, y[s0:s0 + 1024].float()], 1))
        st = torch.cat(stats)
        mo, io, mn, inn, uh, y = st.T
        best = None
        for s_ in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0):
            for b_ in (-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6):
                new_wins = s_ * mn + b_ * uh > mo
                ag = float(torch.where(new_wins, inn == y, io == y).float().mean())
                if best is None or ag > best[0]:
                    best = (ag, s_, b_, float(new_wins.float().mean()))
        return best

    def rows_grid(ids_t, s_, b_):
        W = W_bridge_all[ids_t] * s_ + b_ * u[None, :]
        m = in_old[ids_t]
        W[m] = W_d[old_pos[ids_t[m]]]
        return W

    variants, grid = {}, {}
    for vn, ids in vocabs.items():
        ids_t = torch.tensor(ids, dtype=torch.long, device=DEV)
        variants[f"b_{vn}"] = (ids_t, rows_for(ids_t, False))
        if vn != "old":
            variants[f"k_{vn}"] = (ids_t, rows_for(ids_t, True))
            ag, s_, b_, share = grid_calib(ids_t)
            grid[vn] = dict(holdout_agreement=ag, s=s_, b=b_, new_row_win_share=share)
            log(f"grid calibration {vn}: s={s_} b={b_} holdout agreement {ag:.4f} (new rows win at {share:.3f} of positions)")
            variants[f"g_{vn}"] = (ids_t, rows_grid(ids_t, s_, b_))
            if vn == "union":
                Wz = torch.zeros(ids_t.numel(), H, device=DEV); mz = in_old[ids_t]; Wz[mz] = W_d[old_pos[ids_t[mz]]]
                variants["z_union"] = (ids_t, Wz)

    # ---- 4. offline screening on zh / chat eval traffic
    ev = {"zh": [(hd[o], ht[o], y[o]) for hd, ht, y, o in features(target, head, zh_eval, mc["aux"], a.max_len)],
          "chat": [(hd[o], ht[o], y[o]) for hd, ht, y, o in features(target, head, chat_eval, mc["aux"], a.max_len)]}
    res = {"factory": {d: agreement(ev[d], W_d, S_old) for d in ev}}
    for vn, (ids_t, W) in variants.items():
        res[vn] = {d: agreement(ev[d], W, ids_t) for d in ev}
        res[vn]["size"] = int(ids_t.numel())
        log(f"{vn}: " + json.dumps(res[vn]))
    cov = {}
    for vn, ids in vocabs.items():
        m = torch.zeros(VOCAB, dtype=torch.bool, device=DEV); m[torch.tensor(ids, device=DEV)] = True
        cov[vn] = {d: float(torch.cat([m[y] for _, _, y in ev[d]]).float().mean()) for d in ev}

    # ---- 5. export head folders
    other = {k: v for k, v in sd.items() if k not in ("lm_head.weight", "d2t", "t2d")}
    for vn, (ids_t, W) in variants.items():
        out = f"{RUNS}/c2/{a.model}/{a.domain}/{vn}{a.tag}"
        os.makedirs(out, exist_ok=True)
        ids_c = ids_t.cpu()
        tensors = dict(other)
        tensors["lm_head.weight"] = W.to(torch.bfloat16).cpu().contiguous()
        tensors["d2t"] = (ids_c - torch.arange(ids_c.numel())).contiguous()
        t2d = torch.zeros(VOCAB, dtype=torch.bool); t2d[ids_c] = True
        tensors["t2d"] = t2d
        save_file(tensors, f"{out}/model.safetensors", metadata={"format": "pt"})
        c = dict(cfg); c["draft_vocab_size"] = int(ids_c.numel())
        json.dump(c, open(f"{out}/config.json", "w"), indent=2)
    report = dict(n_fit_rows=len(fit_rows), n_hold_rows=len(hold_rows), n_fit_positions=n_pos, lam=lam,
                  holdout_bridged_full=ag_hold, holdout_factory=factory_hold, scale=s_cal, bias=b_cal, corr_cal=corr_cal,
                  u_mean=float(uh.mean()), u_sd=float(uh.std()),
                  coverage=cov, agreement=res, grid=grid, fit_domains=[a.domain] + extra, tag=a.tag, seconds=time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_report)), exist_ok=True)
    json.dump(report, open(a.out_report.replace(".md", ".json"), "w"), indent=1)
    L = ["# 激活空间桥接：词表手术离线筛查（第一步草稿 token 一致率，评测流量输出位置）\n",
         f"拟合 {len(fit_rows)} 条 zh 训练序列、{n_pos} 个输出位置；λ={lam}；留出序列上全词表桥接一致率 {ag_hold:.4f}，出厂头 {factory_hold:.4f}。"
         f"头 logit ≈ {s_cal:.3f}×target logit + {b_cal:.2f}（位置内相关 {corr_cal:.3f}；常数方向 u·h 均值 {uh.mean():.2f}±{uh.std():.2f}）。\n",
         "| 变体 | 词表 | zh 覆盖 | zh 第一步一致率 | chat 覆盖 | chat 第一步一致率 |", "|---|---|---|---|---|---|",
         f"| factory（出厂头） | {K0} | {cov['old']['zh']:.3f} | {res['factory']['zh']:.4f} | {cov['old']['chat']:.3f} | {res['factory']['chat']:.4f} |"]
    for vn in variants:
        base = vn.split("_", 1)[1]
        L.append(f"| {vn} | {res[vn]['size']} | {cov[base]['zh']:.3f} | {res[vn]['zh']:.4f} | {cov[base]['chat']:.3f} | {res[vn]['chat']:.4f} |")
    L.append("\nb_* = 该词表所有行都走桥接（target logit 尺度）；k_* = 原 32k 的训练行保留、新增行走桥接并按位置内回归的 s,b 校准；"
             "g_* = 同上但 s,b 在留出序列上按第一步一致率网格搜索；z_union = 新增行全零（训练消融的起点）；b_old = 原词表全部换成桥接行。")
    for vn, gv in grid.items():
        L.append(f"- 网格校准 {vn}: s={gv['s']} b={gv['b']}，留出一致率 {gv['holdout_agreement']:.4f}，新行胜出位置占比 {gv['new_row_win_share']:.3f}")
    open(a.out_report, "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    log(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
