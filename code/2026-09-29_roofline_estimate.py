# Llama-3.1-8B on one H100-SXM (bf16). Crude roofline, ignores kernel inefficiency.
P=8.0e9; W=2*P                       # weight bytes
kv_tok=2*32*8*128*2                  # K+V bytes per token per seq (32L, 8 kv heads, hd128, bf16) = 128KB
att_flop_tok=4*32*128*32             # per query token per context token: QK+AV over 32 heads, 32 layers
BW=3.35e12; PEAK=989e12; HBM=80e9
def step(B, L, q=1):
    """one decode/verify step: B seqs, context L, q query tokens per seq"""
    flops = 2*P*B*q + att_flop_tok*L*B*q
    byts  = W + kv_tok*L*B
    return max(flops/PEAK, byts/BW), flops/byts
print("ridge FLOP/byte =", round(PEAK/BW))
print("\n(1) marginal cost of verifying k=4 extra tokens (step time ratio q=5 vs q=1)")
for L in (1024, 8192, 32768):
    row=[]
    for B in (1,8,32,64,128,256):
        if W+kv_tok*L*B > 0.9*HBM: row.append(f"B{B}:OOM"); continue
        t1,_=step(B,L,1); t5,_=step(B,L,5)
        row.append(f"B{B}:{t5/t1:.2f}x")
    print(f"  L={L:>6}: "+"  ".join(row))
print("\n(2) throughput cost of giving 10GB HBM to drafter training (KV pool shrinks)")
for L in (1024, 8192, 32768):
    pool=0.9*HBM-W
    Bmax=int(pool//(kv_tok*L)); Btr=int((pool-10e9)//(kv_tok*L))
    thr=lambda B: B/step(B,L,1)[0]
    print(f"  L={L:>6}: Bmax {Bmax:>4} -> {Btr:>4} ({(Btr/Bmax-1)*100:+.0f}% batch)  throughput {(thr(Btr)/thr(Bmax)-1)*100:+.1f}%")
# EAGLE-3 head training state for 8B target (approx): 1 decoder layer + fc(3d->d) + draft lm_head(32k)
d=4096; ff=14336
layer = 4*d*d*0.625 + 3*d*ff          # GQA attn approx + SwiGLU MLP
head_params = layer + 3*d*d + 32000*d
state = head_params*(2+2+4+4+4)       # bf16 w + bf16 grad + fp32 master + adam m,v
print(f"\n(3) EAGLE-3 head ~{head_params/1e6:.0f}M params -> weights+grads+Adam ~{state/1e9:.1f} GB (+ activations)")
print("    hidden-state log per token (3 layers, bf16):", 3*d*2/1024, "KB  (Aurora reports ~22KB/token after logit top-k filtering)")
