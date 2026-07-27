"""End-to-end memory/throughput measurement on the REAL HF model's generate()-style
incremental decode (prefill + N single-token steps), comparing lskv_absorbed_inference
False (full-width k_C/lt_V cached, existing code path) vs True (compressed h^D/k_R
cached, the newly-implemented real cache logic) -- NOT the synthetic _decode_bench
harness. Same methodology as before: 5 untimed warmup steps, reset_peak_memory_stats
AFTER warmup, median of N timed steps, synchronize around each timed step.
"""
import argparse
import statistics
import time

import torch
from dar_abs_modeling import GPTNeoXConfig, GPTNeoXForCausalLM


def mb(x):
    return x / 1e6


def _extend_cache_to_length(past, model, T, real_prefill_len, device, dtype):
    """Real short prefill gives a structurally-correct (real Cache subclass, real
    dtypes/shapes) past_key_values object; extend every layer's cached tensors with
    random content (via the SAME torch.cat the real Cache.update() itself uses) up to
    length T, so we test genuine decode-step cost/memory against a length-T cache
    WITHOUT paying for an O(T) or O(T^2) real prefill forward pass. Only the CONTENT
    of the extra positions is synthetic; the cache's structure/dtype/shape are 100%
    real (same classes, same tensor layout the real model produces and reads)."""
    extra = T - real_prefill_len
    if extra <= 0:
        return past
    n_layers = model.config.num_hidden_layers
    for li in range(n_layers):
        wlayer = past.layers[li]
        extra_k = torch.randn(*wlayer.keys.shape[:-2], extra, wlayer.keys.shape[-1], device=device, dtype=dtype)
        extra_v = torch.randn(*wlayer.values.shape[:-2], extra, wlayer.values.shape[-1], device=device, dtype=dtype)
        wlayer.keys = torch.cat([wlayer.keys, extra_k], dim=-2)
        wlayer.values = torch.cat([wlayer.values, extra_v], dim=-2)

        ltlayer = past.long_term_components_cache.layers[li]
        extra_ltk = torch.randn(*ltlayer.keys.shape[:-2], extra, ltlayer.keys.shape[-1], device=device, dtype=dtype)
        extra_ltv = torch.randn(*ltlayer.values.shape[:-2], extra, ltlayer.values.shape[-1], device=device, dtype=dtype)
        ltlayer.keys = torch.cat([ltlayer.keys, extra_ltk], dim=-2)
        ltlayer.values = torch.cat([ltlayer.values, extra_ltv], dim=-2)

        if hasattr(past, "decoupled_kr_cache") and len(past.decoupled_kr_cache.layers) > li \
           and past.decoupled_kr_cache.layers[li].keys is not None:
            krlayer = past.decoupled_kr_cache.layers[li]
            extra_kr = torch.randn(*krlayer.keys.shape[:-2], extra, krlayer.keys.shape[-1], device=device, dtype=dtype)
            krlayer.keys = torch.cat([krlayer.keys, extra_kr], dim=-2)
            krlayer.values = torch.cat([krlayer.values, extra_kr], dim=-2)
    return past


def bench(ckpt, absorbed, B, T, warmup, steps, dtype, device, seed, real_prefill_len=64):
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig.from_pretrained(ckpt)
    cfg.lskv_absorbed_inference = absorbed
    model = GPTNeoXForCausalLM.from_pretrained(ckpt, config=cfg, dtype=dtype, attn_implementation="eager").to(device).eval()
    vocab = cfg.vocab_size

    torch.cuda.empty_cache()
    # short REAL prefill (cheap) to get a structurally-correct cache object, then
    # synthetically extend it to the target length T (see docstring above)
    prefill_tokens = torch.randint(0, vocab, (B, min(real_prefill_len, T)), device=device)
    with torch.no_grad():
        out = model(prefill_tokens, use_cache=True)
    past = _extend_cache_to_length(out.past_key_values, model, T, min(real_prefill_len, T), device, dtype)

    torch.cuda.reset_peak_memory_stats(device)
    cache_pos = torch.tensor([T], device=device)

    def decode_step(tok, past, cache_pos):
        with torch.no_grad():
            out = model(tok, use_cache=True, past_key_values=past, cache_position=cache_pos)
        return out.past_key_values

    for _ in range(warmup):
        tok = torch.randint(0, vocab, (B, 1), device=device)
        past = decode_step(tok, past, cache_pos)
        cache_pos = cache_pos + 1
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(steps):
        tok = torch.randint(0, vocab, (B, 1), device=device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        past = decode_step(tok, past, cache_pos)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
        cache_pos = cache_pos + 1

    peak = torch.cuda.max_memory_allocated(device)
    ms_step = statistics.median(times) * 1000.0
    del model, past
    torch.cuda.empty_cache()
    return mb(peak), ms_step, ms_step / B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqs", type=int, nargs="+", default=[2048])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda"

    print(f"ckpt={args.ckpt}  dtype={args.dtype}  warmup={args.warmup}  steps={args.steps}\n")
    print(f"{'batch':>5} {'seqlen':>7} | {'mode':>10} | {'peak mem':>10} | {'ms/step':>9} | {'ms/tok':>8}")
    for B in args.batch:
        for T in args.seqs:
            for absorbed, label in [(False, "non-absorbed"), (True, "absorbed")]:
                try:
                    peak, ms_step, ms_tok = bench(args.ckpt, absorbed, B, T, args.warmup, args.steps, dtype, device, args.seed)
                    print(f"{B:>5} {T:>7} | {label:>10} | {peak:>7.1f} MB | {ms_step:>7.3f} ms | {ms_tok:>6.4f} ms")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{B:>5} {T:>7} | {label:>10} | {'OOM':>10} | {'--':>9} | {'--':>8}")


if __name__ == "__main__":
    main()
