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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["core", "attn", "logits", "all"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=128, help="lskv_window_size (128 DAR, 0 Uniform)")
    args = ap.parse_args()
    dtype = {"fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    device = args.device if torch.cuda.is_available() else "cpu"
    tol = 1e-3 if dtype == torch.float32 else 5e-1
    print(f"ckpt={args.ckpt}  dtype={args.dtype}  device={device}  window={args.window}  tol={tol}\n")

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
    print(f"\n{'PASS' if ok else 'FAIL'} (tol={tol}, dtype={args.dtype})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
