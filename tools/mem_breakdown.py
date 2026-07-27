"""Precise, additive memory breakdown at B=256, T=16384: weights vs cache vs
activations, for both Vanilla (naive) and DAR-abs (absorbed), using the EXACT
faithful decode-step code (chained expressions, no named-variable retention
distortion), matching _decode_bench exactly post cache-formula-fix."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from dar_abs_absorb import (load_ckpt, absorb, N_HEADS, D_HEAD, D_DOWN, D_R, HIDDEN,
                            NORM_FACTOR, cache_bytes_per_token)

CK = "/data/zhiyuanwang/code/ls_kv/ppl/train_pythia/spheron_seed_checkpoints/pythia-70m-deduped_lskv_d128_w128_dar_abs_fullbias/global_step19073"
device, dtype = "cuda", torch.float16
B, T, window = 256, 16384, 128

def mb(x): return x / 1e6

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
mem_start = torch.cuda.memory_allocated(device)

model = load_ckpt(CK, dtype, device)
_, layers, _, _ = model
nl = len(layers)
torch.cuda.synchronize(device)
mem_after_weights = torch.cuda.memory_allocated(device)
weights_bytes = mem_after_weights - mem_start

param_count = sum(t.numel() for lay in layers for t in lay.values()) \
    + model[0].numel() + model[2][0].numel() + model[2][1].numel() + model[3].numel()
print(f"=== weights ===")
print(f"  measured (memory_allocated delta after load_ckpt): {mb(weights_bytes):>10.1f} MB")
print(f"  parameter count (all layers+embed+unembed+norm)  : {param_count:>10,}  x {dtype} ({torch.tensor([],dtype=dtype).element_size()}B) "
      f"-> {mb(param_count*torch.tensor([],dtype=dtype).element_size()):>10.1f} MB (theoretical)")
print(f"  SAME for vanilla and absorbed: identical model object is reused for both modes\n")

results = {}
for label, mode, vanilla_impl in [("VANILLA (naive)", "full", "naive"), ("DAR-ABS (absorbed)", "absorbed", None)]:
    print(f"########## {label} ##########")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mem_before_cache = torch.cuda.memory_allocated(device)  # == mem_after_weights, re-measured for safety

    if mode == "full":
        Kc = [torch.randn(B, T, N_HEADS, D_HEAD, device=device, dtype=dtype) for _ in range(nl)]
        Vc = [torch.randn(B, T, N_HEADS, D_HEAD, device=device, dtype=dtype) for _ in range(nl)]
    else:
        w = min(window, T); nd = max(T - w, 0)
        Kc = [torch.randn(B, w, N_HEADS, D_HEAD, device=device, dtype=dtype) for _ in range(nl)]
        Vc = [torch.randn(B, w, N_HEADS, D_HEAD, device=device, dtype=dtype) for _ in range(nl)]
        Hc = [torch.randn(B, T, D_DOWN, device=device, dtype=dtype) for _ in range(nl)]
        Rc = [torch.randn(B, T, D_R, device=device, dtype=dtype) for _ in range(nl)]
        AB = [absorb(l) for l in layers]
    torch.cuda.synchronize(device)
    mem_after_cache = torch.cuda.memory_allocated(device)
    cache_bytes = mem_after_cache - mem_before_cache

    formula_bytes = cache_bytes_per_token("vanilla" if mode == "full" else "dar_abs", window, T,
                                          torch.tensor([], dtype=dtype).element_size(), nl) * B
    ab_bytes = sum(t.numel()*t.element_size() for layer_ab in AB for t in layer_ab) if mode != "full" else 0

    def step(x):
        for li, lay in enumerate(layers):
            q = torch.einsum("bh,ndh->bnd", x, lay["W_Q"]) + lay["b_Q"]
            if mode == "full":
                sc = torch.einsum("bnd,btnd->bnt", q, Kc[li]) / NORM_FACTOR
                p = torch.softmax(sc.float(), -1).to(dtype)
                ctx = torch.einsum("bnt,btnd->bnd", p, Vc[li])
            else:
                WKp, cK, WVp, cV = AB[li]
                Hd, Rd = Hc[li][:, :nd], Rc[li][:, :nd]
                scw = torch.einsum("bnd,btnd->bnt", q, Kc[li]) / NORM_FACTOR
                qp = torch.einsum("bnd,ndc->bnc", q, WKp)
                q_R = (x @ lay["Wqr"].t()).view(B, N_HEADS, D_R)
                scd = (torch.einsum("bnc,btc->bnt", qp, Hd)
                       + torch.einsum("bnd,nd->bn", q, cK)[:, :, None]
                       + torch.einsum("bnd,btd->bnt", q_R, Rd)) / NORM_FACTOR
                sc = torch.cat([scw, scd], -1)
                p = torch.softmax(sc.float(), -1).to(dtype)
                nwin = Kc[li].shape[1]
                pw, pd = p[:, :, :nwin], p[:, :, nwin:]
                sph = torch.einsum("bnt,btc->bnc", pd, Hd)
                ctx = (torch.einsum("bnt,btnd->bnd", pw, Vc[li])
                       + torch.einsum("bnc,ndc->bnd", sph, WVp)
                       + pd.sum(-1)[:, :, None] * cV)
            x = ctx.reshape(B, HIDDEN) @ lay["dense_w"].t() + lay["dense_b"]
        return x

    with torch.no_grad():
        for _ in range(5):
            step(torch.randn(B, HIDDEN, device=device, dtype=dtype))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(32):
            step(torch.randn(B, HIDDEN, device=device, dtype=dtype))
            torch.cuda.synchronize(device)
    peak_total = torch.cuda.max_memory_allocated(device)
    activations_bytes = peak_total - mem_after_cache

    print(f"  weights                : {mb(weights_bytes):>10.1f} MB")
    print(f"  cache (measured delta)  : {mb(cache_bytes):>10.1f} MB   (formula: {mb(formula_bytes):>10.1f} MB"
          + (f", of which AB weights: {mb(ab_bytes):>6.1f} MB" if mode != "full" else "") + ")")
    print(f"  activations (peak-cache): {mb(activations_bytes):>10.1f} MB")
    print(f"  SUM (weights+cache+act) : {mb(weights_bytes+cache_bytes+activations_bytes):>10.1f} MB")
    print(f"  measured peak (raw)     : {mb(peak_total):>10.1f} MB")
    print(f"  gap (sum - peak)        : {mb(weights_bytes+cache_bytes+activations_bytes-peak_total):>10.4f} MB (should be ~0, exact by construction)\n")

    results[label] = dict(weights=weights_bytes, cache=cache_bytes, activations=activations_bytes, peak=peak_total)

    for name in ["Kc", "Vc"] + (["Hc", "Rc", "AB"] if mode != "full" else []):
        if name in dir():
            del locals()[name]
    torch.cuda.empty_cache()

print("########## summary ##########")
for label, r in results.items():
    print(f"  {label}: weights={mb(r['weights']):.1f}MB  cache={mb(r['cache']):.1f}MB  "
          f"activations={mb(r['activations']):.1f}MB  peak={mb(r['peak']):.1f}MB")
v, a = results["VANILLA (naive)"], results["DAR-ABS (absorbed)"]
print(f"\n  activation ratio (absorbed/vanilla): {a['activations']/v['activations']:.3f}x")
print(f"  cache ratio (vanilla/absorbed)      : {v['cache']/a['cache']:.3f}x")
print(f"  peak ratio (vanilla/absorbed)       : {v['peak']/a['peak']:.3f}x")
