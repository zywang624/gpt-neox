"""Exact-math fast path for LSKV attention with st_window_size == 1.

With window 1 the short-term (ST) branch is the diagonal only, so the
ST score matmul, the second context matmul and the ST/LT mask build reduce
to elementwise ops on the diagonal. Same math as the reference
score/context path in transformer_ddim.py / transformer_lskv.py, one fewer
full [sq, sk] batched matmul per call.

Shapes (after the caller's view): query/key/value/lt_key/lt_value are
[s, bnp, hn] with sq == sk. Returns context as [bnp, sq, hn].
"""

import torch


def lskv_attention_w1_reference(query, key, value, lt_key, lt_value, norm_factor, softmax_fn):
    """Current implementation, kept here as the numerical reference."""
    sq, bnp, hn = query.shape
    sk = key.shape[0]
    q = query.transpose(0, 1)
    st = torch.baddbmm(
        torch.empty(bnp, sq, sk, dtype=q.dtype, device=q.device),
        q, key.transpose(0, 1).transpose(1, 2), beta=0.0, alpha=1.0 / norm_factor,
    )
    lt = torch.baddbmm(
        torch.empty(bnp, sq, sk, dtype=q.dtype, device=q.device),
        q, lt_key.transpose(0, 1).transpose(1, 2), beta=0.0, alpha=1.0 / norm_factor,
    )
    eye = torch.eye(sq, dtype=torch.bool, device=q.device).unsqueeze(0)
    scores = torch.where(eye, st, lt)
    probs = softmax_fn(scores)
    st_f = eye.to(probs.dtype)
    return torch.bmm(probs * st_f, value.transpose(0, 1)) + torch.bmm(
        probs * (1.0 - st_f), lt_value.transpose(0, 1)
    )


def lskv_attention_w1_fast(query, key, value, lt_key, lt_value, norm_factor, softmax_fn):
    """Same result with one full batched matmul instead of two."""
    sq, bnp, hn = query.shape
    q = query.transpose(0, 1)
    lt = torch.baddbmm(
        torch.empty(bnp, sq, sq, dtype=q.dtype, device=q.device),
        q, lt_key.transpose(0, 1).transpose(1, 2), beta=0.0, alpha=1.0 / norm_factor,
    )
    diag_st = ((query.float() * key.float()).sum(-1) * (1.0 / norm_factor)).to(q.dtype).transpose(0, 1)  # [bnp, sq]
    lt.diagonal(dim1=1, dim2=2).copy_(diag_st)
    probs = softmax_fn(lt)
    diag_p = torch.diagonal(probs, dim1=1, dim2=2).unsqueeze(-1)  # [bnp, sq, 1]
    lt_v = lt_value.transpose(0, 1)
    eye = torch.eye(sq, dtype=torch.bool, device=q.device).unsqueeze(0)
    probs_off = probs.masked_fill(eye, 0.0)
    ctx_diag = (diag_p.float() * value.transpose(0, 1).float()).to(probs.dtype)
    return torch.bmm(probs_off, lt_v) + ctx_diag
