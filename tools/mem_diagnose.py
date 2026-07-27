"""Diagnose exactly what torch.cuda.max_memory_allocated() captures in the
_decode_bench benchmark: pure cache bytes vs whole-step peak (cache + per-step
activations), at B=256, T=16384, both full-KV(naive) and absorbed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from dar_abs_absorb import (load_ckpt, absorb, N_HEADS, D_HEAD, D_DOWN, D_R, HIDDEN,
                            NORM_FACTOR, cache_bytes_per_token)

CK = "/data/zhiyuanwang/code/ls_kv/ppl/train_pythia/spheron_seed_checkpoints/pythia-70m-deduped_lskv_d128_w128_dar_abs_fullbias/global_step19073"
device = "cuda"
dtype = torch.float16
B, T, window = 256, 16384, 128

model = load_ckpt(CK, dtype, device)
_, layers, _, _ = model
nl = len(layers)
dtype_bytes = torch.tensor([], dtype=dtype).element_size()

def mb(x): return x / 1e6

print(f"=== setup: B={B} T={T} window={window} n_layers={nl} dtype={dtype} ===\n")

# ---------------- FULL-KV (naive) ----------------
print("########## full-KV (naive einsum) ##########")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
mem_before_cache = torch.cuda.memory_allocated(device)

Kc = [torch.randn(B, T, N_HEADS, D_HEAD, device=device, dtype=dtype) for _ in range(nl)]
Vc = [torch.randn(B, T, N_HEADS, D_HEAD, device=device, dtype=dtype) for _ in range(nl)]
torch.cuda.synchronize(device)
mem_after_cache_alloc = torch.cuda.memory_allocated(device)
peak_after_cache_alloc = torch.cuda.max_memory_allocated(device)

pure_cache_measured = mem_after_cache_alloc - mem_before_cache
pure_cache_formula = cache_bytes_per_token("vanilla", window, T, dtype_bytes, nl) * B

print(f"  measured pure-cache bytes (memory_allocated delta):  {mb(pure_cache_measured):>10.1f} MB")
print(f"  formula  pure-cache bytes (cache_bytes_per_token*B): {mb(pure_cache_formula):>10.1f} MB")
print(f"  match: {'YES' if abs(pure_cache_measured-pure_cache_formula) < 1e6 else 'NO -- investigate'}")

# now run one decode step and track peak, WITH fine-grained checkpoints inside
torch.cuda.reset_peak_memory_stats(device)  # reset again, so this peak reflects ONLY the step (cache already resident, not counted as "new")
x = torch.randn(B, HIDDEN, device=device, dtype=dtype)
checkpoints = []
with torch.no_grad():
    for li, lay in enumerate(layers):
        q = torch.einsum("bh,ndh->bnd", x, lay["W_Q"]) + lay["b_Q"]
        torch.cuda.synchronize(device)
        checkpoints.append((f"L{li} after q", torch.cuda.memory_allocated(device)))
        sc = torch.einsum("bnd,btnd->bnt", q, Kc[li]) / NORM_FACTOR
        torch.cuda.synchronize(device)
        checkpoints.append((f"L{li} after sc (fp16 raw score)", torch.cuda.memory_allocated(device)))
        scf = sc.float()
        torch.cuda.synchronize(device)
        checkpoints.append((f"L{li} after sc.float() (fp32 copy)", torch.cuda.memory_allocated(device)))
        sm = torch.softmax(scf, -1)
        torch.cuda.synchronize(device)
        checkpoints.append((f"L{li} after softmax (fp32 out)", torch.cuda.memory_allocated(device)))
        p = sm.to(dtype)
        torch.cuda.synchronize(device)
        checkpoints.append((f"L{li} after p=.to(dtype)", torch.cuda.memory_allocated(device)))
        ctx = torch.einsum("bnt,btnd->bnd", p, Vc[li])
        torch.cuda.synchronize(device)
        checkpoints.append((f"L{li} after ctx", torch.cuda.memory_allocated(device)))
        x = ctx.reshape(B, HIDDEN) @ lay["dense_w"].t() + lay["dense_b"]
        torch.cuda.synchronize(device)
        checkpoints.append((f"L{li} after dense", torch.cuda.memory_allocated(device)))
        if li >= 1:  # only need to see the pattern repeat once
            break
peak_during_step = torch.cuda.max_memory_allocated(device)
overhead_during_step = peak_during_step - mem_after_cache_alloc

print(f"\n  fine-grained trace (layer 0-1 only, shows where the transient peak comes from):")
base = mem_after_cache_alloc
for name, val in checkpoints:
    print(f"    {name:38s}: allocated={mb(val):>10.1f} MB   (delta over resident cache: {mb(val-base):>+9.1f} MB)")

print(f"\n  peak DURING decode step (max_memory_allocated, whole-step): {mb(peak_during_step):>10.1f} MB")
print(f"  overhead beyond resident cache (this peak - cache-alloc level): {mb(overhead_during_step):>10.1f} MB")

del Kc, Vc, x, q, sc, scf, sm, p, ctx
torch.cuda.empty_cache()

# ---------------- ABSORBED ----------------
print("\n\n########## absorbed ##########")
torch.cuda.reset_peak_memory_stats(device)
mem_before_cache = torch.cuda.memory_allocated(device)

w = min(window, T); nd = max(T - w, 0)
Kc = [torch.randn(B, w, N_HEADS, D_HEAD, device=device, dtype=dtype) for _ in range(nl)]
Vc = [torch.randn(B, w, N_HEADS, D_HEAD, device=device, dtype=dtype) for _ in range(nl)]
Hc = [torch.randn(B, nd, D_DOWN, device=device, dtype=dtype) for _ in range(nl)]
Rc = [torch.randn(B, nd, D_R, device=device, dtype=dtype) for _ in range(nl)]
AB = [absorb(l) for l in layers]
torch.cuda.synchronize(device)
mem_after_cache_alloc = torch.cuda.memory_allocated(device)

pure_cache_measured = mem_after_cache_alloc - mem_before_cache
pure_cache_formula = cache_bytes_per_token("dar_abs", window, T, dtype_bytes, nl) * B
# AB (absorbed weight matrices W_K', c_K, W_V', c_V) are NOT part of the "cache" definition
# (they're per-layer weight-derived, batch-independent, tiny) -- report separately
ab_bytes = sum(t.numel()*t.element_size() for layer_ab in AB for t in layer_ab)
print(f"  measured pure-cache bytes (memory_allocated delta, incl. AB weights): {mb(pure_cache_measured):>10.1f} MB")
print(f"  formula  pure-cache bytes (cache_bytes_per_token*B):                 {mb(pure_cache_formula):>10.1f} MB")
print(f"  AB (absorbed W_K'/c_K/W_V'/c_V) weight bytes (batch-independent):     {mb(ab_bytes):>10.1f} MB")
print(f"  measured - formula - AB = {mb(pure_cache_measured - pure_cache_formula - ab_bytes):>10.1f} MB (should be ~0)")

torch.cuda.reset_peak_memory_stats(device)
x = torch.randn(B, HIDDEN, device=device, dtype=dtype)
with torch.no_grad():
    for li, lay in enumerate(layers):
        WKp, cK, WVp, cV = AB[li]
        q = torch.einsum("bh,ndh->bnd", x, lay["W_Q"]) + lay["b_Q"]
        scw = torch.einsum("bnd,btnd->bnt", q, Kc[li]) / NORM_FACTOR
        qp = torch.einsum("bnd,ndc->bnc", q, WKp)
        q_R = (x @ lay["Wqr"].t()).view(B, N_HEADS, D_R)
        scd = (torch.einsum("bnc,btc->bnt", qp, Hc[li])
               + torch.einsum("bnd,nd->bn", q, cK)[:, :, None]
               + torch.einsum("bnd,btd->bnt", q_R, Rc[li])) / NORM_FACTOR
        sc = torch.cat([scw, scd], -1)
        p = torch.softmax(sc.float(), -1).to(dtype)
        nwin = Kc[li].shape[1]
        pw, pd = p[:, :, :nwin], p[:, :, nwin:]
        sph = torch.einsum("bnt,btc->bnc", pd, Hc[li])
        ctx = (torch.einsum("bnt,btnd->bnd", pw, Vc[li])
               + torch.einsum("bnc,ndc->bnd", sph, WVp)
               + pd.sum(-1)[:, :, None] * cV)
        x = ctx.reshape(B, HIDDEN) @ lay["dense_w"].t() + lay["dense_b"]
peak_during_step = torch.cuda.max_memory_allocated(device)
overhead_during_step = peak_during_step - mem_after_cache_alloc
print(f"\n  peak DURING decode step (max_memory_allocated, whole-step): {mb(peak_during_step):>10.1f} MB")
print(f"  overhead beyond resident cache (this peak - cache-alloc level): {mb(overhead_during_step):>10.1f} MB")

print("\n\n########## summary: ratio using pure-cache vs whole-step-peak ##########")
