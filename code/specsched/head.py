"""Trainable replica of vLLM's Eagle3LlamaForCausalLM (AngelSlim key layout), packed-sequence forward.

Input convention (EAGLE "s1" shift): at packed position j of a sequence, the head sees the
target embedding of token x_{j+1} and the target aux hidden state h_j, and predicts x_{j+2}.
"""
import json
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

NAME_MAP = {
    "fc": "fc.weight",
    "in_ln": "midlayer.input_layernorm.weight",
    "hid_ln": "midlayer.hidden_norm.weight",
    "post_ln": "midlayer.post_attention_layernorm.weight",
    "q": "midlayer.self_attn.q_proj.weight",
    "k": "midlayer.self_attn.k_proj.weight",
    "v": "midlayer.self_attn.v_proj.weight",
    "o": "midlayer.self_attn.o_proj.weight",
    "gate": "midlayer.mlp.gate_proj.weight",
    "up": "midlayer.mlp.up_proj.weight",
    "down": "midlayer.mlp.down_proj.weight",
    "norm": "norm.weight",
    "lm_head": "lm_head.weight",
}


def load_sd(head_dir):
    for n in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(head_dir, n)
        if os.path.exists(p):
            sd = load_file(p) if n.endswith(".safetensors") else torch.load(p, map_location="cpu", weights_only=True)
            return {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    raise FileNotFoundError(f"no weights in {head_dir}")


def load_cfg(head_dir):
    return json.load(open(os.path.join(head_dir, "config.json")))


class EagleHead(nn.Module):
    def __init__(self, sd, cfg):
        super().__init__()
        for attr, key in NAME_MAP.items():
            setattr(self, attr, nn.Parameter(sd[key].float().clone()))
        self.register_buffer("d2t", sd["d2t"].clone())
        self.register_buffer("t2d", sd["t2d"].clone() if "t2d" in sd else torch.zeros(0))
        self.extra = {k: v for k, v in sd.items() if k not in NAME_MAP.values() and k not in ("d2t", "t2d")}
        self.n_q = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or cfg["hidden_size"] // self.n_q
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        theta = float(cfg.get("rope_theta", 1e6))
        self.register_buffer("inv", 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd)))

    def rms(self, x, w):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * w).to(d)

    def rope(self, x, pos):
        freqs = torch.outer(pos.float(), self.inv)
        cos, sin = freqs.cos()[:, None, :], freqs.sin()[:, None, :]
        h = self.hd // 2
        x1, x2 = x.float()[..., :h], x.float()[..., h:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)

    def forward(self, emb, aux, pos, seg):
        """emb [T,H], aux [T,3*Ht], pos [T] (position inside its sequence), seg [T] sequence id."""
        T = emb.shape[0]
        e = self.rms(emb, self.in_ln)
        g = aux @ self.fc.T.to(aux.dtype)
        residual = g.float()
        x = torch.cat([e, self.rms(g, self.hid_ln)], -1)
        q = self.rope((x @ self.q.T.to(x.dtype)).view(T, self.n_q, self.hd), pos)
        k = self.rope((x @ self.k.T.to(x.dtype)).view(T, self.n_kv, self.hd), pos)
        v = (x @ self.v.T.to(x.dtype)).view(T, self.n_kv, self.hd)
        rep = self.n_q // self.n_kv
        idx = torch.arange(T, device=emb.device)
        mask = (seg[:, None] == seg[None, :]) & (idx[:, None] >= idx[None, :])
        a = F.scaled_dot_product_attention(
            q.permute(1, 0, 2)[None], k.repeat_interleave(rep, 1).permute(1, 0, 2)[None],
            v.repeat_interleave(rep, 1).permute(1, 0, 2)[None], attn_mask=mask[None, None])
        a = a[0].permute(1, 0, 2).reshape(T, -1) @ self.o.T.to(x.dtype)
        residual = residual + a.float()
        h = self.rms(residual, self.post_ln)
        h = (F.silu(h @ self.gate.T.to(h.dtype)) * (h @ self.up.T.to(h.dtype))) @ self.down.T.to(h.dtype)
        residual = residual + h.float()
        return self.rms(residual, self.norm) @ self.lm_head.T.to(residual.dtype)

    def draft_to_target(self, d):
        return d + self.d2t[d]

    def target_to_draft_index(self, vocab_size):
        t2d = torch.full((vocab_size,), -1, dtype=torch.long, device=self.d2t.device)
        base = torch.arange(self.d2t.numel(), device=self.d2t.device)
        t2d[base + self.d2t] = base
        return t2d

    @torch.no_grad()
    def export(self, out_dir, src_dir):
        """Write a vLLM-loadable draft folder (original key names, bf16) next to the source config."""
        os.makedirs(out_dir, exist_ok=True)
        sd = {key: getattr(self, attr).detach().to(torch.bfloat16).cpu().contiguous()
              for attr, key in NAME_MAP.items()}
        sd["d2t"] = self.d2t.cpu().contiguous()
        if self.t2d.numel():
            sd["t2d"] = self.t2d.cpu().contiguous()
        for k, v in self.extra.items():
            sd[k] = v.cpu().contiguous()
        tmp = os.path.join(out_dir, "model.safetensors.tmp")
        save_file(sd, tmp, metadata={"format": "pt"})
        os.replace(tmp, os.path.join(out_dir, "model.safetensors"))
        shutil.copy(os.path.join(src_dir, "config.json"), os.path.join(out_dir, "config.json"))
