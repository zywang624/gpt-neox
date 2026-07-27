"""DAR-abs (MLA-style decoupled RoPE) absorption: reparameterization + equivalence gates.

The DAR-abs global (distant-token) path, in training/reference form, computes for
each distant key j and query t (per head, NoPE content + shared decoupled RoPE):

    h^D_j     = GELU(W_down x_j + b_down)          # bottleneck latent, dim d_down
    lt_hid_j  = W_up h^D_j + b_up                  # dim = hidden
    k_C_j     = W_K lt_hid_j + b_K                 # NoPE content key (per head)
    lt_V_j    = W_V lt_hid_j + b_V                 # distant value (per head)
    score_tj  = (q_C_t . k_C_j + q_R_t . k_R_j) / sqrt(d_head)
    ctx_t    += Σ_j p_tj lt_V_j                    # over distant j (window j use full V)

Absorbed inference caches ONLY h^D_j (dim d_down) for distant tokens instead of the
full k_C_j + lt_V_j (2 * n_heads * d_head), and folds W_up into W_K / W_V:

    W_K' = W_K W_up ,  c_K = W_K b_up + b_K        # score side
    W_V' = W_V W_up ,  c_V = W_V b_up + b_V        # value side
    q'_t = q_C_t W_K'                              # absorb W_K' into the query (-> dim d_down)
    score_content_tj = q'_t . h^D_j + q_C_t . c_K  # == q_C_t . k_C_j   (exact reparam)
    ctx_distant_t    = W_V' (Σ_j p_tj h^D_j) + c_V (Σ_j p_tj)

c_K is a per-(t,head) shift added to DISTANT scores only; with a window it does NOT
cancel (window scores don't carry it). Under Uniform (no window) every score carries
it -> constant per-row shift -> cancels under softmax (harmless no-op).

This module is a standalone equivalence harness (no deepspeed/mpu). It reuses the real
RotaryEmbedding + apply_rotary from megatron.model.positional_embeddings so RoPE is
identical to the model, and loads real checkpoint weights.

Gates:
  core    -- k_C / lt_V reconstruction from cached h^D matches the standard forward.
  attn    -- full attention-level output (window/distant fusion + RoPE) matches std vs absorbed.
  logits  -- full-model next-token logits match std vs absorbed across all layers.

Usage:  python tools/dar_abs_absorb.py {core,attn,logits,all} --ckpt <global_step_dir> [--dtype fp32|fp16] [--device cuda]
"""
import argparse
import glob
import math
import os
import sys

import torch

# real RoPE (pure-torch, no mpu dependency)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from megatron.model.positional_embeddings import (  # noqa: E402
    RotaryEmbedding,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb_torch_k,
)

# ---- 70M DAR-abs geometry (fixed by the checkpoint) -------------------------
HIDDEN = 512
N_HEADS = 8
D_HEAD = 64
D_DOWN = 128
D_R = 32
ROTARY_NDIMS = 16          # int(d_head * rotary_pct) = int(64 * 0.25)
LN_EPS = 1.0e-5
NORM_FACTOR = math.sqrt(D_HEAD)   # 8.0 ; apply_query_key_layer_scaling off. Cancels in std-vs-absorbed.


def _layer_ids(ckpt):
    ids = []
    for f in glob.glob(os.path.join(ckpt, "layer_*-model_00-model_states.pt")):
        sd = torch.load(f, map_location="cpu", weights_only=False)
        if "attention.query_key_value.weight" in sd:
            ids.append(int(os.path.basename(f).split("_")[1].split("-")[0]))
    return sorted(ids)


def load_ckpt(ckpt, dtype, device):
    """Return (embed, [layer_dicts], final_norm, unembed) as tensors in `dtype`/`device`."""
    def L(i):
        return torch.load(
            os.path.join(ckpt, f"layer_{i:02d}-model_00-model_states.pt"),
            map_location="cpu", weights_only=False,
        )

    def t(x):
        return x.to(dtype=dtype, device=device)

    embed = t(L(0)["word_embeddings.weight"])                    # [vocab, hidden]
    # final norm + unembed live in the two highest layer indices
    all_ids = sorted(
        int(os.path.basename(f).split("_")[1].split("-")[0])
        for f in glob.glob(os.path.join(ckpt, "layer_*-model_00-model_states.pt"))
    )
    fn = L(all_ids[-2])
    final_norm = (t(fn["norm.weight"]), t(fn["norm.bias"]))
    unembed = t(L(all_ids[-1])["final_linear.weight"])           # [vocab, hidden]

    layers = []
    for i in _layer_ids(ckpt):
        sd = L(i)
        Wqkv = t(sd["attention.query_key_value.weight"]).view(N_HEADS, 3, D_HEAD, HIDDEN)
        bqkv = t(sd["attention.query_key_value.bias"]).view(N_HEADS, 3, D_HEAD)
        layers.append(dict(
            ln_w=t(sd["input_layernorm.weight"]), ln_b=t(sd["input_layernorm.bias"]),
            pln_w=t(sd["post_attention_layernorm.weight"]), pln_b=t(sd["post_attention_layernorm.bias"]),
            Wdown=t(sd["attention.down_up_proj.0.weight"]), bdown=t(sd["attention.down_up_proj.0.bias"]),
            Wup=t(sd["attention.down_up_proj.2.weight"]), bup=t(sd["attention.down_up_proj.2.bias"]),
            W_Q=Wqkv[:, 0], b_Q=bqkv[:, 0], W_K=Wqkv[:, 1], b_K=bqkv[:, 1], W_V=Wqkv[:, 2], b_V=bqkv[:, 2],
            Wqr=t(sd["attention.lskv_W_qr.weight"]), Wkr=t(sd["attention.lskv_W_kr.weight"]),
            dense_w=t(sd["attention.dense.weight"]), dense_b=t(sd["attention.dense.bias"]),
            h4h_w=t(sd["mlp.dense_h_to_4h.weight"]), h4h_b=t(sd["mlp.dense_h_to_4h.bias"]),
            f4h_w=t(sd["mlp.dense_4h_to_h.weight"]), f4h_b=t(sd["mlp.dense_4h_to_h.bias"]),
        ))
    return embed, layers, final_norm, unembed


def absorb(lay):
    """Fold W_up into W_K / W_V. Returns (W_K'[nh,dh,d_down], c_K[nh,dh], W_V', c_V)."""
    Wup, bup = lay["Wup"], lay["bup"]                     # Wup: [hidden, d_down]
    WKp = torch.einsum("ndh,hc->ndc", lay["W_K"], Wup)
    cK = torch.einsum("ndh,h->nd", lay["W_K"], bup) + lay["b_K"]
    WVp = torch.einsum("ndh,hc->ndc", lay["W_V"], Wup)
    cV = torch.einsum("ndh,h->nd", lay["W_V"], bup) + lay["b_V"]
    return WKp, cK, WVp, cV


def layernorm(x, w, b):
    xf = x.float()
    m = xf.mean(-1, keepdim=True)
    v = xf.var(-1, keepdim=True, unbiased=False)
    return ((xf - m) / torch.sqrt(v + LN_EPS)).to(x.dtype) * w + b


# ---- shared RoPE pieces (identical between reference and absorbed) -----------
def _rope_modules(device, dtype):
    win = RotaryEmbedding(ROTARY_NDIMS, precision=dtype).to(device)
    dec = RotaryEmbedding(D_R, precision=dtype).to(device)
    return win, dec


def _window_rope(q, k, win_rot, S):
    """q,k: [S, nh, d_head] -> partial RoPE on first ROTARY_NDIMS dims. Returns rotated q,k."""
    qb = q.unsqueeze(1)  # [S,1,nh,dh]
    kb = k.unsqueeze(1)
    q_rot, q_pass = qb[..., :ROTARY_NDIMS], qb[..., ROTARY_NDIMS:]
    k_rot, k_pass = kb[..., :ROTARY_NDIMS], kb[..., ROTARY_NDIMS:]
    cos, sin = win_rot(kb, seq_len=S)
    q_rot, k_rot = apply_rotary_pos_emb_torch(q_rot, k_rot, cos, sin)
    q = torch.cat([q_rot, q_pass], -1).squeeze(1)
    k = torch.cat([k_rot, k_pass], -1).squeeze(1)
    return q, k


def _decoupled_rope(x, lay, dec_rot, S):
    """Returns q_R [S,nh,d_r], k_R [S,d_r] after full RoPE (via dedicated instance)."""
    q_R = (x @ lay["Wqr"].t()).view(S, 1, N_HEADS, D_R)   # [S,1,nh,dr]
    k_R = (x @ lay["Wkr"].t()).view(S, 1, 1, D_R)         # [S,1,1,dr]
    cos, sin = dec_rot(k_R, seq_len=S)
    q_R = apply_rotary_pos_emb_torch_k(q_R, q_R, cos, sin)
    k_R = apply_rotary_pos_emb_torch_k(k_R, k_R, cos, sin)
    return q_R.squeeze(1), k_R.view(S, D_R)


def attention(x, lay, mode, window, win_rot, dec_rot, absorbed=None):
    """Faithful DAR-abs attention. x:[S,hidden]. mode in {reference, absorbed}.
    Returns dense output [S,hidden]."""
    S = x.shape[0]
    dev, dt = x.device, x.dtype

    # window q/k/v (full path)
    q = torch.einsum("sh,ndh->snd", x, lay["W_Q"]) + lay["b_Q"]   # [S,nh,dh]
    k = torch.einsum("sh,ndh->snd", x, lay["W_K"]) + lay["b_K"]
    v = torch.einsum("sh,ndh->snd", x, lay["W_V"]) + lay["b_V"]

    # bottleneck latent + reconstructed distant content key / value
    hD = torch.nn.functional.gelu(x @ lay["Wdown"].t() + lay["bdown"])   # [S,d_down]
    lt_hidden = hD @ lay["Wup"].t() + lay["bup"]                          # [S,hidden]
    kC = torch.einsum("sh,ndh->snd", lt_hidden, lay["W_K"]) + lay["b_K"]  # [S,nh,dh]
    ltV = torch.einsum("sh,ndh->snd", lt_hidden, lay["W_V"]) + lay["b_V"]

    qC = q.clone()                             # NoPE content query (pre-window-RoPE)
    q, k = _window_rope(q, k, win_rot, S)      # window path gets partial RoPE
    q_R, k_R = _decoupled_rope(x, lay, dec_rot, S)

    # ---- masks ----
    idx = torch.arange(S, device=dev)
    causal = idx[None, :] <= idx[:, None]                              # [S,S]
    if window > 0:
        win = causal & (idx[None, :] >= (idx[:, None] - window + 1))
    else:
        win = torch.zeros_like(causal)
    dist = causal & (~win)

    inv = 1.0 / NORM_FACTOR
    win_scores = torch.einsum("qnd,knd->nqk", q, k) * inv             # [nh,S,S]
    score_R = torch.einsum("qnd,kd->nqk", q_R, k_R) * inv

    if mode == "reference":
        dist_content = torch.einsum("qnd,knd->nqk", qC, kC) * inv
    else:
        WKp, cK, WVp, cV = absorbed
        qprime = torch.einsum("qnd,ndc->qnc", qC, WKp)               # [S,nh,d_down]
        dist_content = torch.einsum("qnc,kc->nqk", qprime, hD) * inv
        cK_term = torch.einsum("qnd,nd->qn", qC, cK) * inv           # [S,nh]
        dist_content = dist_content + cK_term.transpose(0, 1)[:, :, None]
    dist_scores = dist_content + score_R

    scores = torch.where(win[None], win_scores, dist_scores)
    scores = scores.masked_fill(~causal[None], float("-inf"))
    p = torch.softmax(scores.float(), dim=-1).to(dt)                 # [nh,S,S]

    p_win = p * win[None].to(dt)
    p_dist = p * dist[None].to(dt)
    ctx_win = torch.einsum("nqk,knd->qnd", p_win, v)                 # [S,nh,dh]
    if mode == "reference":
        ctx = ctx_win + torch.einsum("nqk,knd->qnd", p_dist, ltV)
    else:
        WKp, cK, WVp, cV = absorbed
        sum_phD = torch.einsum("nqk,kc->qnc", p_dist, hD)           # [S,nh,d_down]
        pmass = p_dist.sum(-1).transpose(0, 1)                       # [S,nh]
        ctx_dist = torch.einsum("qnc,ndc->qnd", sum_phD, WVp) + torch.einsum("qn,nd->qnd", pmass, cV)
        ctx = ctx_win + ctx_dist

    ctx = ctx.reshape(S, HIDDEN)
    return ctx @ lay["dense_w"].t() + lay["dense_b"]


# ==================== batched variants (correctness stress-test) ============
# [S, B, ...] layout (seq-first, matching megatron/NeoX's native convention --
# the single-sequence functions above already use this with an implicit B=1
# via unsqueeze(1)/squeeze(1)). Generalizing B lets us test the EXACT same
# RoPE/einsum machinery at real batch sizes and diff against a per-item loop
# of the already-validated single-sequence `attention()`/`full_forward()`.
def _window_rope_batch(q, k, win_rot, S):
    """q,k: [S,B,nh,d_head] -> partial RoPE on first ROTARY_NDIMS dims (batched)."""
    q_rot, q_pass = q[..., :ROTARY_NDIMS], q[..., ROTARY_NDIMS:]
    k_rot, k_pass = k[..., :ROTARY_NDIMS], k[..., ROTARY_NDIMS:]
    cos, sin = win_rot(k_rot, seq_len=S)          # [S,1,1,ndims] (batch-independent, position-only)
    q_rot, k_rot = apply_rotary_pos_emb_torch(q_rot, k_rot, cos, sin)  # q.shape[0]==S==cos.shape[0]: correct slice
    return torch.cat([q_rot, q_pass], -1), torch.cat([k_rot, k_pass], -1)


def _decoupled_rope_batch(x, lay, dec_rot, S, B):
    """x:[S,B,hidden] -> q_R:[S,B,nh,d_r], k_R:[S,B,d_r]."""
    q_R = (x @ lay["Wqr"].t()).view(S, B, N_HEADS, D_R)
    k_R = (x @ lay["Wkr"].t()).view(S, B, 1, D_R)
    cos, sin = dec_rot(k_R, seq_len=S)             # [S,1,1,d_r]
    q_R = apply_rotary_pos_emb_torch_k(q_R, q_R, cos, sin)
    k_R = apply_rotary_pos_emb_torch_k(k_R, k_R, cos, sin)
    return q_R, k_R.view(S, B, D_R)


def attention_batch(x, lay, mode, window, win_rot, dec_rot, absorbed=None):
    """Batched twin of `attention()`. x:[S,B,hidden]. Returns [S,B,hidden].
    Every einsum keeps 'b' (batch) as a PRESERVED (non-contracted) index in both
    operands and the output -- i.e. per-batch-item dot products only, no summation
    or mixing across the batch dimension. This is exactly the property the
    equivalence tests below verify empirically (don't just trust the comment)."""
    S, B = x.shape[0], x.shape[1]
    dt = x.dtype

    q = torch.einsum("sbh,ndh->sbnd", x, lay["W_Q"]) + lay["b_Q"]     # [S,B,nh,dh]
    k = torch.einsum("sbh,ndh->sbnd", x, lay["W_K"]) + lay["b_K"]
    v = torch.einsum("sbh,ndh->sbnd", x, lay["W_V"]) + lay["b_V"]

    hD = torch.nn.functional.gelu(x @ lay["Wdown"].t() + lay["bdown"])   # [S,B,d_down]
    lt_hidden = hD @ lay["Wup"].t() + lay["bup"]                          # [S,B,hidden]
    kC = torch.einsum("sbh,ndh->sbnd", lt_hidden, lay["W_K"]) + lay["b_K"]
    ltV = torch.einsum("sbh,ndh->sbnd", lt_hidden, lay["W_V"]) + lay["b_V"]

    qC = q.clone()
    q, k = _window_rope_batch(q, k, win_rot, S)
    q_R, k_R = _decoupled_rope_batch(x, lay, dec_rot, S, B)

    idx = torch.arange(S, device=x.device)
    causal = idx[None, :] <= idx[:, None]
    if window > 0:
        win = causal & (idx[None, :] >= (idx[:, None] - window + 1))
    else:
        win = torch.zeros_like(causal)
    dist = causal & (~win)

    inv = 1.0 / NORM_FACTOR
    win_scores = torch.einsum("qbnd,kbnd->bnqk", q, k) * inv           # [B,nh,S,S]
    score_R = torch.einsum("qbnd,kbd->bnqk", q_R, k_R) * inv

    if mode == "reference":
        dist_content = torch.einsum("qbnd,kbnd->bnqk", qC, kC) * inv
    else:
        WKp, cK, WVp, cV = absorbed
        qprime = torch.einsum("qbnd,ndc->qbnc", qC, WKp)               # [S,B,nh,d_down]
        dist_content = torch.einsum("qbnc,kbc->bnqk", qprime, hD) * inv
        cK_term = torch.einsum("qbnd,nd->qbn", qC, cK) * inv           # [S,B,nh]
        dist_content = dist_content + cK_term.permute(1, 2, 0)[:, :, :, None]
    dist_scores = dist_content + score_R

    scores = torch.where(win[None], win_scores, dist_scores)
    scores = scores.masked_fill(~causal[None], float("-inf"))
    p = torch.softmax(scores.float(), dim=-1).to(dt)                   # [B,nh,S,S]

    p_win = p * win[None].to(dt)
    p_dist = p * dist[None].to(dt)
    ctx_win = torch.einsum("bnqk,kbnd->qbnd", p_win, v)                # [S,B,nh,dh]
    if mode == "reference":
        ctx = ctx_win + torch.einsum("bnqk,kbnd->qbnd", p_dist, ltV)
    else:
        WKp, cK, WVp, cV = absorbed
        sum_phD = torch.einsum("bnqk,kbc->qbnc", p_dist, hD)           # [S,B,nh,d_down]
        pmass = p_dist.sum(-1).permute(2, 0, 1)                         # [S,B,nh]
        ctx_dist = (torch.einsum("qbnc,ndc->qbnd", sum_phD, WVp)
                    + torch.einsum("qbn,nd->qbnd", pmass, cV))
        ctx = ctx_win + ctx_dist

    ctx = ctx.reshape(S, B, HIDDEN)
    return ctx @ lay["dense_w"].t() + lay["dense_b"]


def mlp(x, lay):
    h = torch.nn.functional.gelu(x @ lay["h4h_w"].t() + lay["h4h_b"])
    return h @ lay["f4h_w"].t() + lay["f4h_b"]


def full_forward(tokens, model, mode, window, win_rot, dec_rot, absorbed_all):
    embed, layers, (fnw, fnb), unembed = model
    x = embed[tokens]                                                # [S,hidden]
    for li, lay in enumerate(layers):
        x1 = layernorm(x, lay["ln_w"], lay["ln_b"])
        x2 = layernorm(x, lay["pln_w"], lay["pln_b"])
        a = attention(x1, lay, mode, window, win_rot, dec_rot,
                      absorbed=absorbed_all[li] if mode == "absorbed" else None)
        m = mlp(x2, lay)
        x = x + a + m                                                # untied gpt_j_residual
    x = layernorm(x, fnw, fnb)
    return x @ unembed.t()                                           # [S,vocab]


def full_forward_batch(tokens, model, mode, window, win_rot, dec_rot, absorbed_all):
    """Batched twin of `full_forward`. tokens:[S,B] (DIFFERENT sequence per batch
    item). layernorm/mlp are already shape-agnostic (elementwise / matmul-broadcast
    over leading dims); only attention needed a batched rewrite."""
    embed, layers, (fnw, fnb), unembed = model
    x = embed[tokens]                                                # [S,B,hidden]
    for li, lay in enumerate(layers):
        x1 = layernorm(x, lay["ln_w"], lay["ln_b"])
        x2 = layernorm(x, lay["pln_w"], lay["pln_b"])
        a = attention_batch(x1, lay, mode, window, win_rot, dec_rot,
                            absorbed=absorbed_all[li] if mode == "absorbed" else None)
        m = mlp(x2, lay)
        x = x + a + m
    x = layernorm(x, fnw, fnb)
    return x @ unembed.t()                                           # [S,B,vocab]


# ============================ gates =========================================
def check_core(ckpt, dtype, device):
    print("== core: k_C / lt_V reconstruction from cached h^D ==")
    _, layers, _, _ = load_ckpt(ckpt, dtype, device)
    worst_k = worst_v = 0.0
    for li, lay in enumerate(layers):
        WKp, cK, WVp, cV = absorb(lay)
        x = torch.randn(7, HIDDEN, dtype=dtype, device=device)
        hD = torch.nn.functional.gelu(x @ lay["Wdown"].t() + lay["bdown"])
        lt_hidden = hD @ lay["Wup"].t() + lay["bup"]
        kC_std = torch.einsum("sh,ndh->snd", lt_hidden, lay["W_K"]) + lay["b_K"]
        ltV_std = torch.einsum("sh,ndh->snd", lt_hidden, lay["W_V"]) + lay["b_V"]
        kC_abs = torch.einsum("sc,ndc->snd", hD, WKp) + cK
        ltV_abs = torch.einsum("sc,ndc->snd", hD, WVp) + cV
        dk = (kC_std - kC_abs).abs().max().item()
        dv = (ltV_std - ltV_abs).abs().max().item()
        worst_k, worst_v = max(worst_k, dk), max(worst_v, dv)
    print(f"   worst |Δk_C|={worst_k:.2e}  |Δlt_V|={worst_v:.2e}")
    return worst_k, worst_v


def check_attention(ckpt, dtype, device, window, S=192):
    print(f"== attn: full attention output std-vs-absorbed (window={window}, S={S}) ==")
    _, layers, _, _ = load_ckpt(ckpt, dtype, device)
    win_rot, dec_rot = _rope_modules(device, dtype)
    worst = 0.0
    for li, lay in enumerate(layers):
        x = torch.randn(S, HIDDEN, dtype=dtype, device=device)
        absorbed = absorb(lay)
        ref = attention(x, lay, "reference", window, win_rot, dec_rot)
        abs_ = attention(x, lay, "absorbed", window, win_rot, dec_rot, absorbed=absorbed)
        d = (ref - abs_).abs().max().item()
        scale = ref.abs().max().item()
        worst = max(worst, d)
        if li == 0:
            print(f"   layer0: max|Δout|={d:.2e}  (out scale ~{scale:.1f}, rel {d/max(scale,1e-9):.1e})")
    print(f"   worst over {len(layers)} layers: max|Δout|={worst:.2e}")
    return worst


def check_logits(ckpt, dtype, device, window, S=256):
    print(f"== logits: full-model next-token logits std-vs-absorbed (window={window}, S={S}) ==")
    model = load_ckpt(ckpt, dtype, device)
    win_rot, dec_rot = _rope_modules(device, dtype)
    layers = model[1]
    absorbed_all = [absorb(l) for l in layers]
    vocab = model[0].shape[0]
    torch.manual_seed(0)
    tokens = torch.randint(0, vocab, (S,), device=device)
    ref = full_forward(tokens, model, "reference", window, win_rot, dec_rot, absorbed_all)
    abs_ = full_forward(tokens, model, "absorbed", window, win_rot, dec_rot, absorbed_all)
    d = (ref - abs_).abs().max().item()
    scale = ref.abs().max().item()
    # argmax agreement (what actually matters for generation)
    agree = (ref.argmax(-1) == abs_.argmax(-1)).float().mean().item()
    print(f"   max|Δlogit|={d:.2e}  (logit scale ~{scale:.1f}, rel {d/max(scale,1e-9):.1e})")
    print(f"   argmax agreement over {S} positions: {agree*100:.2f}%")
    return d, agree


def check_batched(ckpt, dtype, device, window, batch_sizes, S=256, layer_idxs=(0, -1)):
    """Attention-level batching stress test. For each batch size B, four
    independent probes (all must agree to fp32/fp16 tolerance):
      1. loop-vs-batch (reference mode): does batching corrupt the reference path?
      2. loop-vs-batch (absorbed mode):  does batching corrupt the absorbed path?
         (the PRIMARY concern -- a bug here could look like a "speedup".)
      3. reference-vs-absorbed AT this batch size directly (not inferred from B=1).
      4. shuffle-invariance: permute the INPUT batch order, check outputs permute
         identically -- catches leakage that (1)-(3) could miss by coincidence.
    'loop' = per-item calls to the ALREADY gate-validated single-sequence
    `attention()` (proven vs the real megatron model earlier). 'batch' =
    `attention_batch()`, the new code under test.
    """
    print(f"== batched: attention_batch vs per-item loop + shuffle-invariance "
          f"(window={window}, S={S}, batches={list(batch_sizes)}) ==")
    _, layers, _, _ = load_ckpt(ckpt, dtype, device)
    win_rot, dec_rot = _rope_modules(device, dtype)
    nl = len(layers)
    idxs = sorted(set((i if i >= 0 else nl + i) for i in layer_idxs))
    all_ok = True
    for li in idxs:
        lay = layers[li]
        absorbed = absorb(lay)
        for B in batch_sizes:
            torch.manual_seed(10_000 + li * 1000 + B)
            x_sb = torch.randn(S, B, HIDDEN, dtype=dtype, device=device)  # distinct content per batch item

            loop_ref = torch.stack(
                [attention(x_sb[:, b, :], lay, "reference", window, win_rot, dec_rot) for b in range(B)], dim=1)
            loop_abs = torch.stack(
                [attention(x_sb[:, b, :], lay, "absorbed", window, win_rot, dec_rot, absorbed=absorbed) for b in range(B)], dim=1)

            batch_ref = attention_batch(x_sb, lay, "reference", window, win_rot, dec_rot)
            batch_abs = attention_batch(x_sb, lay, "absorbed", window, win_rot, dec_rot, absorbed=absorbed)

            d_ref = (loop_ref - batch_ref).abs().max().item()
            d_abs = (loop_abs - batch_abs).abs().max().item()
            d_refabs = (batch_ref - batch_abs).abs().max().item()

            perm = torch.randperm(B, device=device)
            batch_abs_shuf = attention_batch(x_sb[:, perm, :], lay, "absorbed", window, win_rot, dec_rot, absorbed=absorbed)
            d_shuf = (batch_abs_shuf - batch_abs[:, perm, :]).abs().max().item()

            tol = 1e-3 if dtype == torch.float32 else 5e-1
            ok = max(d_ref, d_abs, d_refabs, d_shuf) < tol
            all_ok &= ok
            print(f"   layer{li:2d} B={B:4d}: loop-vs-batch(ref)={d_ref:.2e}  loop-vs-batch(abs)={d_abs:.2e}  "
                  f"ref-vs-abs@batch={d_refabs:.2e}  shuffle-invariance={d_shuf:.2e}  {'PASS' if ok else 'FAIL'}")
    print(f"   {'ALL PASS' if all_ok else '*** SOME FAILED -- DO NOT TRUST THE BATCH SWEEP ***'}")
    return all_ok


def check_logits_batch(ckpt, dtype, device, window, batch_sizes, S=256):
    """Full-model (all layers) version of check_batched, on next-token logits.
    Directly answers: 're-run the equivalence gate at the larger batch sizes'."""
    print(f"== logits@batch: full-model std-vs-absorbed + loop-vs-batch + shuffle "
          f"(window={window}, S={S}, batches={list(batch_sizes)}) ==")
    model = load_ckpt(ckpt, dtype, device)
    win_rot, dec_rot = _rope_modules(device, dtype)
    layers = model[1]
    absorbed_all = [absorb(l) for l in layers]
    vocab = model[0].shape[0]
    all_ok = True
    for B in batch_sizes:
        torch.manual_seed(20_000 + B)
        tokens = torch.randint(0, vocab, (S, B), device=device)   # different random sequence per batch item

        # free each tensor as soon as its diff is taken -- with B=256, S=256,
        # vocab=50304, fp32 logits are ~13GB EACH; keeping 5 live at once OOMs
        # (a test-harness memory issue, not a property of the code under test).
        batch_ref = full_forward_batch(tokens, model, "reference", window, win_rot, dec_rot, absorbed_all)
        loop_ref = torch.stack(
            [full_forward(tokens[:, b], model, "reference", window, win_rot, dec_rot, absorbed_all) for b in range(B)], dim=1)
        d_ref = (loop_ref - batch_ref).abs().max().item()
        del loop_ref

        batch_abs = full_forward_batch(tokens, model, "absorbed", window, win_rot, dec_rot, absorbed_all)
        loop_abs = torch.stack(
            [full_forward(tokens[:, b], model, "absorbed", window, win_rot, dec_rot, absorbed_all) for b in range(B)], dim=1)
        d_abs = (loop_abs - batch_abs).abs().max().item()
        del loop_abs

        d_refabs = (batch_ref - batch_abs).abs().max().item()
        agree = (batch_ref.argmax(-1) == batch_abs.argmax(-1)).float().mean().item()
        del batch_ref

        perm = torch.randperm(B, device=device)
        batch_abs_perm_ref = batch_abs[:, perm, :].clone()
        batch_abs_shuf = full_forward_batch(tokens[:, perm], model, "absorbed", window, win_rot, dec_rot, absorbed_all)
        d_shuf = (batch_abs_shuf - batch_abs_perm_ref).abs().max().item()
        del batch_abs, batch_abs_shuf, batch_abs_perm_ref
        torch.cuda.empty_cache()

        tol = 1e-3 if dtype == torch.float32 else 5e-1
        ok = max(d_ref, d_abs, d_refabs, d_shuf) < tol and agree > 0.999
        all_ok &= ok
        print(f"   B={B:4d}: loop-vs-batch(ref)={d_ref:.2e}  loop-vs-batch(abs)={d_abs:.2e}  "
              f"ref-vs-abs@batch={d_refabs:.2e}  argmax={agree*100:.2f}%  shuffle={d_shuf:.2e}  {'PASS' if ok else 'FAIL'}")
    print(f"   {'ALL PASS' if all_ok else '*** SOME FAILED -- DO NOT TRUST THE BATCH SWEEP ***'}")
    return all_ok


# ==================== cache-cost model + decode microbench ==================
def cache_bytes_per_token(method, window, T, dtype_bytes, n_layers):
    """Per-layer KV-cache footprint (bytes) for a length-T context.

    DAR-abs design (matches the real cache: LSKVCache appends to BOTH the
    regular K/V cache AND long_term_components_cache on every token's first
    forward pass -- h^D/k_R are computed from x_j and cached immediately,
    since x_j is not retained afterward and cannot be recovered from K_h/V_h
    (different, non-invertible projections). So EVERY token (window or not)
    carries d_down+d_r, and the w most-recent (window) tokens ADDITIONALLY
    carry the full 2d K/V -- i.e. w*2d + T*(d_down+d_r), NOT w*2d + (T-w)*
    (d_down+d_r) (which would implicitly assume h^D/k_R appear for free once
    a token exits the window, with no mechanism to actually compute them)."""
    full = 2 * N_HEADS * D_HEAD              # K + V, full heads
    if method == "vanilla":
        per_layer = T * full
    elif method == "dar_abs":                # window (2d + d_down+d_r) + distant (d_down+d_r)
        w = min(window, T)
        per_layer = w * full + T * (D_DOWN + D_R)
    else:
        raise ValueError(method)
    return per_layer * n_layers * dtype_bytes


def _decode_bench(model, T, n_steps, window, mode, device, B=1, vanilla_impl="naive",
                  warmup=5, seed=0):
    """Prefill a length-T cache for B sequences, time n_steps of single-token decode.
    Each step advances B sequences by one token -> B tokens/step. B=1 reproduces the
    original batch-1 benchmark exactly (same math; only a leading batch dim is threaded
    through).

    vanilla_impl (mode=="full" only):
      "naive" -- the original hand-written einsum attention (what produced the first
                 batch-sweep numbers).
      "sdpa"  -- torch.nn.functional.scaled_dot_product_attention with the cache
                 pre-laid-out as [B,nh,T,dh] (the standard production KV-cache layout,
                 no per-step permute cost), to check whether "naive" was an unfairly
                 slow stand-in for Vanilla.

    Methodology: `warmup` steps run FIRST and are excluded from both timing and the
    peak-memory measurement (reset_peak_memory_stats happens AFTER warmup, so JIT/
    cuBLAS-algo-selection/allocator-growth transients from the first call(s) don't
    inflate the reported peak or bias the timed median). `seed` seeds the cache/query
    content for this call so repeated invocations can use different seeds (checking
    run-to-run variance is not an artifact of identical, cacheable inputs).

    Returns (peak_MB, ms/step, ms/token) or (None,None,None) on OOM.
    """
    import statistics
    import time
    _, layers, _, _ = model
    dt, nl = layers[0]["W_Q"].dtype, len(layers)
    torch.manual_seed(seed)
    torch.cuda.empty_cache()

    try:
        sdpa_full = (mode == "full" and vanilla_impl == "sdpa")
        # production KV-cache layout for ALL paths: [B, nh, T-or-w, dh], contiguous.
        # (B, T, nh, dh) -- the layout used before this fix -- puts the heads dim
        # between T and dh, which is NOT the layout torch.einsum's implied batched
        # matmul needs; PyTorch silently inserts a full permute+contiguous COPY of
        # the whole K/V cache to fix it up. At B=256/T=16384 that's a ~4.3GB hidden
        # copy PER LAYER PER STEP -- this was the dominant cost inflating naive-
        # vanilla's activation memory ~10x above DAR-abs's, and pushed the reported
        # whole-step peak ratio above the architecture-only cache ceiling. Fixed by
        # allocating (and computing against) the transpose-free layout everywhere.
        if mode == "full" and not sdpa_full:
            Kc = [torch.randn(B, N_HEADS, T, D_HEAD, device=device, dtype=dt) for _ in range(nl)]
            Vc = [torch.randn(B, N_HEADS, T, D_HEAD, device=device, dtype=dt) for _ in range(nl)]
        elif sdpa_full:
            Kc = [torch.randn(B, N_HEADS, T, D_HEAD, device=device, dtype=dt) for _ in range(nl)]
            Vc = [torch.randn(B, N_HEADS, T, D_HEAD, device=device, dtype=dt) for _ in range(nl)]
        else:
            w = min(window, T)
            nd = max(T - w, 0)
            Kc = [torch.randn(B, N_HEADS, w, D_HEAD, device=device, dtype=dt) for _ in range(nl)]
            Vc = [torch.randn(B, N_HEADS, w, D_HEAD, device=device, dtype=dt) for _ in range(nl)]
            # h^D/k_R are cached for EVERY token (size T), not just the T-w already-distant
            # ones: they're computed from x_j and cached at that token's FIRST forward pass
            # (x_j is not retained afterward, so this can't be deferred to "when the token
            # exits the window"). The distant-score compute below still only reads the
            # oldest nd=T-w entries (Hc[:, :nd]/Rc[:, :nd]) -- the newest w entries sit
            # cached-but-unread this step, exactly mirroring the real steady-state cache.
            Hc = [torch.randn(B, T, D_DOWN, device=device, dtype=dt) for _ in range(nl)]
            Rc = [torch.randn(B, T, D_R, device=device, dtype=dt) for _ in range(nl)]
            AB = [absorb(l) for l in layers]

        def step(x):
            for li, lay in enumerate(layers):
                q = torch.einsum("bh,ndh->bnd", x, lay["W_Q"]) + lay["b_Q"]        # [B,nh,dh]
                if mode == "full":
                    if sdpa_full:
                        qsd = q.unsqueeze(2)                                        # [B,nh,1,dh]
                        ctx = torch.nn.functional.scaled_dot_product_attention(
                            qsd, Kc[li], Vc[li], is_causal=False).squeeze(2)        # [B,nh,dh]
                    else:
                        sc = torch.einsum("bnd,bntd->bnt", q, Kc[li]) / NORM_FACTOR
                        p = torch.softmax(sc.float(), -1).to(dt)
                        ctx = torch.einsum("bnt,bntd->bnd", p, Vc[li])
                else:
                    WKp, cK, WVp, cV = AB[li]
                    # Hc/Rc are allocated at full length T (h^D/k_R cached for every token,
                    # per-design -- see cache_bytes_per_token docstring); the distant-path
                    # SCORE/CONTEXT compute only ever reads the oldest nd=T-w entries (the
                    # newest w are the current window and go through the Kc/Vc path instead;
                    # their Hc/Rc entries sit cached-but-unread this step, as in steady state).
                    Hd, Rd = Hc[li][:, :nd], Rc[li][:, :nd]
                    scw = torch.einsum("bnd,bntd->bnt", q, Kc[li]) / NORM_FACTOR
                    qp = torch.einsum("bnd,ndc->bnc", q, WKp)
                    q_R = (x @ lay["Wqr"].t()).view(B, N_HEADS, D_R)
                    scd = (torch.einsum("bnc,btc->bnt", qp, Hd)
                           + torch.einsum("bnd,nd->bn", q, cK)[:, :, None]
                           + torch.einsum("bnd,btd->bnt", q_R, Rd)) / NORM_FACTOR
                    sc = torch.cat([scw, scd], -1)
                    p = torch.softmax(sc.float(), -1).to(dt)
                    nwin = Kc[li].shape[2]
                    pw, pd = p[:, :, :nwin], p[:, :, nwin:]
                    sph = torch.einsum("bnt,btc->bnc", pd, Hd)
                    ctx = (torch.einsum("bnt,bntd->bnd", pw, Vc[li])
                           + torch.einsum("bnc,ndc->bnd", sph, WVp)
                           + pd.sum(-1)[:, :, None] * cV)
                x = ctx.reshape(B, HIDDEN) @ lay["dense_w"].t() + lay["dense_b"]
            return x

        for _ in range(warmup):
            x = torch.randn(B, HIDDEN, device=device, dtype=dt)
            step(x)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)   # AFTER warmup: peak reflects steady state

        times = []
        for _ in range(n_steps):
            x = torch.randn(B, HIDDEN, device=device, dtype=dt)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            step(x)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated(device) / 1e6
        ms_step = statistics.median(times) * 1000.0
        return peak, ms_step, ms_step / B            # ms/token = ms/step / B tokens
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None


def theoretical_floor(window, T, n_heads=N_HEADS, d_head=D_HEAD, d_down=D_DOWN, d_r=D_R):
    """Per-DISTANT-token compute and bandwidth ratios (absorbed / vanilla), derived
    from op/byte counts, independent of any measurement. At large T the window's
    fixed-size contribution -> 0, so this is the asymptotic ceiling the measured
    speedup should approach from below (never exceed, in a bandwidth-bound regime).

    K/V are PER-HEAD (n_heads separate copies); h^D and the shared k_R are a SINGLE
    vector reused across heads via per-head projections (W_K'/W_V'/q_R) -- so n_heads
    multiplies the vanilla side of both ratios but NOT the absorbed side.
    (An earlier version of this helper omitted n_heads on the bandwidth side and
    printed an inverted 0.8x "ceiling" -- caught by cross-checking against the
    2048/320=6.4 bytes-per-token figure computed by hand earlier in the session.)
    """
    compute_full = n_heads * 2 * d_head                         # per head: score (q.K) + context (p.V)
    compute_abs = n_heads * (2 * d_down + d_r)                  # per head: score(q'.h^D)+context(p.h^D)+RoPE score(q_R.k_R)
    compute_ratio = compute_abs / compute_full                  # absorbed does MORE compute per distant token (n_heads cancels)
    bytes_full = n_heads * 2 * d_head                           # K+V, all heads
    bytes_abs = d_down + d_r                                    # h^D + shared k_R (NOT multiplied by n_heads)
    bandwidth_ratio = bytes_full / bytes_abs                    # absorbed reads FEWER bytes per distant token
    return compute_ratio, bandwidth_ratio


def measure(ckpt, dtype, device, window, seqs, steps, batches, vanilla_impls, warmup, seed):
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    model = load_ckpt(ckpt, dtype, device)
    nl = len(model[1])
    print(f"== cache-cost model ({nl} layers, {dtype} = {dtype_bytes}B, window={window}) ==")
    print(f"   {'seqlen':>7} | {'vanilla KV/seq':>14} | {'DAR-abs cache/seq':>17} | {'reduction':>9}")
    for T in seqs:
        vb = cache_bytes_per_token("vanilla", window, T, dtype_bytes, nl)
        db = cache_bytes_per_token("dar_abs", window, T, dtype_bytes, nl)
        print(f"   {T:>7} | {vb/1e6:>11.1f} MB | {db/1e6:>14.1f} MB | {vb/db:>8.2f}x")

    cr, br = theoretical_floor(window, max(seqs))
    print(f"\n== theoretical floor (per distant token, T->inf so window's fixed cost -> negligible) ==")
    print(f"   absorbed COMPUTE = {cr:.3f}x vanilla (more, real cost)")
    print(f"   absorbed reads   = {1/br:.3f}x vanilla BYTES  (i.e. {br:.3f}x less bandwidth)")
    print(f"   -> bandwidth-bound ceiling: absorbed speedup should APPROACH but not EXCEED ~{br:.2f}x as batch/context grow")
    print(f"      (exceeding {br:.2f}x would indicate a bug, not genuine bandwidth-bound behavior)")

    print(f"\n== on-GPU decode microbench (warmup={warmup} steps excluded; median of {steps} timed steps; seed={seed}) ==")
    print(f"   {'batch':>5} {'seqlen':>7} | {'method':>12} | {'peak mem':>10} | {'ms/step':>9} | {'ms/tok':>8} | {'tok/s':>9} | {'abs speedup':>11}")
    for B in batches:
        for T in seqs:
            res = {}
            for impl in vanilla_impls:
                res[f"full-{impl}"] = _decode_bench(model, T, steps, window, "full", device, B,
                                                     vanilla_impl=impl, warmup=warmup, seed=seed)
            res["absorbed"] = _decode_bench(model, T, steps, window, "absorbed", device, B,
                                            warmup=warmup, seed=seed)
            best_full = min((res[f"full-{impl}"][2] for impl in vanilla_impls if res[f"full-{impl}"][0] is not None),
                            default=None)
            for key, label in [(f"full-{i}", f"full-KV({i})") for i in vanilla_impls] + [("absorbed", "absorbed")]:
                peak, mss, mstok = res[key]
                if peak is None:
                    print(f"   {B:>5} {T:>7} | {label:>12} | {'OOM':>10} | {'--':>9} | {'--':>8} | {'--':>9} | {'--':>11}")
                    continue
                sp = "--"
                if key == "absorbed" and best_full is not None:
                    sp = f"{best_full / mstok:.2f}x (vs best vanilla)"
                print(f"   {B:>5} {T:>7} | {label:>12} | {peak:>7.1f} MB | {mss:>7.2f} ms | {mstok:>6.3f} ms | {1000.0/mstok:>9.0f} | {sp:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["core", "attn", "logits", "batched", "logits_batch", "all", "measure"])
    ap.add_argument("--batch-check", type=int, nargs="+", default=[1, 8, 32, 64, 256],
                    help="batch sizes for the batched/logits_batch correctness gates")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=128, help="lskv_window_size (128 DAR, 0 Uniform)")
    ap.add_argument("--seqs", type=int, nargs="+", default=[8192, 16384])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, nargs="+", default=[1], help="decode batch size(s); each step = B tokens")
    ap.add_argument("--vanilla-impl", nargs="+", default=["naive"], choices=["naive", "sdpa"],
                    help="vanilla full-KV decode implementation(s) to time (naive einsum and/or SDPA)")
    ap.add_argument("--warmup", type=int, default=5, help="untimed warmup steps before the timed median")
    ap.add_argument("--seed", type=int, default=0, help="seed for cache/query content (vary across repeats)")
    args = ap.parse_args()
    dtype = {"fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    device = args.device if torch.cuda.is_available() else "cpu"
    tol = 1e-3 if dtype == torch.float32 else 5e-1
    print(f"ckpt={args.ckpt}  dtype={args.dtype}  device={device}  window={args.window}  tol={tol}\n")

    if args.gate == "measure":
        measure(args.ckpt, dtype, device, args.window, args.seqs, args.steps, args.batch,
               args.vanilla_impl, args.warmup, args.seed)
        return

    ok = True
    if args.gate in ("core", "all"):
        wk, wv = check_core(args.ckpt, dtype, device)
        ok &= max(wk, wv) < tol
    if args.gate in ("attn", "all"):
        w = check_attention(args.ckpt, dtype, device, args.window)
        ok &= w < tol
    if args.gate in ("logits", "all"):
        d, agree = check_logits(args.ckpt, dtype, device, args.window)
        ok &= (d < tol) and (agree > 0.999)
    if args.gate in ("batched", "all"):
        ok &= check_batched(args.ckpt, dtype, device, args.window, args.batch_check)
    if args.gate in ("logits_batch", "all"):
        ok &= check_logits_batch(args.ckpt, dtype, device, args.window, args.batch_check)
    print(f"\n{'PASS' if ok else 'FAIL'} (tol={tol}, dtype={args.dtype})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
