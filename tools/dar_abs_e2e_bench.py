"""End-to-end memory/throughput measurement on REAL HF models' generate()-style
incremental decode (prefill + N single-token steps): TRUE VANILLA (standard,
unmodified transformers GPTNeoXForCausalLM, standard uncapped DynamicCache) vs
CORRECTED ABSORBED (dar_abs_modeling, lskv_absorbed_inference=True, window cache now
actually evicted/capped at w -- see LSKVCache.update() -- so this is the honest
architecture-level comparison, not non-absorbed-vs-absorbed). Same methodology as
before: 5 untimed warmup steps, reset_peak_memory_stats AFTER warmup, median of N
timed steps, synchronize around each timed step.
"""
import argparse
import statistics
import time

import torch
from transformers import GPTNeoXForCausalLM as VanillaGPTNeoXForCausalLM
from transformers import GPTNeoXConfig as VanillaGPTNeoXConfig
from transformers.cache_utils import DynamicCache

from dar_abs_modeling import GPTNeoXConfig, GPTNeoXForCausalLM


def mb(x):
    return x / 1e6


def _extend_vanilla_cache(past, model, T, real_prefill_len, device, dtype):
    """Standard DynamicCache: every layer's K/V grows uncapped to length T (this IS
    true vanilla's real behavior -- no window/distant split, nothing evicted)."""
    extra = T - real_prefill_len
    if extra <= 0:
        return past
    n_layers = model.config.num_hidden_layers
    for li in range(n_layers):
        layer = past.layers[li]
        extra_k = torch.randn(*layer.keys.shape[:-2], extra, layer.keys.shape[-1], device=device, dtype=dtype)
        extra_v = torch.randn(*layer.values.shape[:-2], extra, layer.values.shape[-1], device=device, dtype=dtype)
        layer.keys = torch.cat([layer.keys, extra_k], dim=-2)
        layer.values = torch.cat([layer.values, extra_v], dim=-2)
    return past


def _extend_absorbed_cache(past, model, T, real_prefill_len, window_size, device, dtype):
    """Real short prefill gives a structurally-correct (real LSKVCache, real
    dtypes/shapes) past_key_values object; extend every layer's cached tensors with
    random content (via the SAME torch.cat the real Cache.update() itself uses) to
    simulate having reached T total tokens, WITHOUT paying for an O(T) or O(T^2) real
    prefill forward pass. Only the CONTENT of the extra positions is synthetic; the
    cache's structure/dtype/shape are 100% real.

    Respects the window-eviction invariant now enforced by LSKVCache.update(): the
    window (this class's own, base-DynamicCache) layer is extended only up to
    min(window_size, T) entries (NOT T) -- extending it to T would defeat the very
    eviction fix being measured here. The distant (long_term_components_cache) and
    decoupled_kr_cache layers are never truncated and extend fully to T, matching
    real behavior.
    """
    n_layers = model.config.num_hidden_layers
    target_window_len = min(window_size, T)
    for li in range(n_layers):
        wlayer = past.layers[li]
        extra_win = target_window_len - wlayer.keys.shape[-2]
        if extra_win > 0:
            extra_k = torch.randn(*wlayer.keys.shape[:-2], extra_win, wlayer.keys.shape[-1], device=device, dtype=dtype)
            extra_v = torch.randn(*wlayer.values.shape[:-2], extra_win, wlayer.values.shape[-1], device=device, dtype=dtype)
            wlayer.keys = torch.cat([wlayer.keys, extra_k], dim=-2)
            wlayer.values = torch.cat([wlayer.values, extra_v], dim=-2)
        elif extra_win < 0:
            # real_prefill_len already exceeded window_size -- cap down, mirroring
            # what LSKVCache.update()'s own eviction would have done.
            wlayer.keys = wlayer.keys[..., -target_window_len:, :].clone()
            wlayer.values = wlayer.values[..., -target_window_len:, :].clone()

        extra_d = T - real_prefill_len
        ltlayer = past.long_term_components_cache.layers[li]
        extra_ltk = torch.randn(*ltlayer.keys.shape[:-2], extra_d, ltlayer.keys.shape[-1], device=device, dtype=dtype)
        extra_ltv = torch.randn(*ltlayer.values.shape[:-2], extra_d, ltlayer.values.shape[-1], device=device, dtype=dtype)
        ltlayer.keys = torch.cat([ltlayer.keys, extra_ltk], dim=-2)
        ltlayer.values = torch.cat([ltlayer.values, extra_ltv], dim=-2)

        if hasattr(past, "decoupled_kr_cache") and len(past.decoupled_kr_cache.layers) > li \
           and past.decoupled_kr_cache.layers[li].keys is not None:
            krlayer = past.decoupled_kr_cache.layers[li]
            extra_kr = torch.randn(*krlayer.keys.shape[:-2], extra_d, krlayer.keys.shape[-1], device=device, dtype=dtype)
            krlayer.keys = torch.cat([krlayer.keys, extra_kr], dim=-2)
            krlayer.values = torch.cat([krlayer.values, extra_kr], dim=-2)
    return past


def _run_decode_bench(model, past, T, B, warmup, steps, vocab, device):
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
    return mb(peak), ms_step, ms_step / B


def bench_vanilla(ckpt, B, T, warmup, steps, dtype, device, seed, real_prefill_len=64):
    torch.manual_seed(seed)
    cfg = VanillaGPTNeoXConfig.from_pretrained(ckpt)
    model = VanillaGPTNeoXForCausalLM.from_pretrained(ckpt, config=cfg, dtype=dtype, attn_implementation="eager").to(device).eval()
    vocab = cfg.vocab_size

    torch.cuda.empty_cache()
    prefill_tokens = torch.randint(0, vocab, (B, min(real_prefill_len, T)), device=device)
    with torch.no_grad():
        out = model(prefill_tokens, use_cache=True)
    past = _extend_vanilla_cache(out.past_key_values, model, T, min(real_prefill_len, T), device, dtype)

    peak, ms_step, ms_tok = _run_decode_bench(model, past, T, B, warmup, steps, vocab, device)
    del model, past
    torch.cuda.empty_cache()
    return peak, ms_step, ms_tok


def bench_absorbed(ckpt, B, T, warmup, steps, dtype, device, seed, real_prefill_len=64):
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig.from_pretrained(ckpt)
    cfg.lskv_absorbed_inference = True
    model = GPTNeoXForCausalLM.from_pretrained(ckpt, config=cfg, dtype=dtype, attn_implementation="eager").to(device).eval()
    vocab = cfg.vocab_size
    window_size = int(cfg.lskv_st_window_size)

    torch.cuda.empty_cache()
    prefill_tokens = torch.randint(0, vocab, (B, min(real_prefill_len, T)), device=device)
    with torch.no_grad():
        out = model(prefill_tokens, use_cache=True)
    past = _extend_absorbed_cache(out.past_key_values, model, T, min(real_prefill_len, T), window_size, device, dtype)

    peak, ms_step, ms_tok = _run_decode_bench(model, past, T, B, warmup, steps, vocab, device)
    del model, past
    torch.cuda.empty_cache()
    return peak, ms_step, ms_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-ckpt", default="/data/zhiyuanwang/code/ls_kv/checkpoints/official_pythia_70m_vanilla")
    ap.add_argument("--absorbed-ckpt", default="/data/zhiyuanwang/code/ls_kv/ppl/train_pythia/dar_abs_converted_checkpoints/official_pythia_70m_lskv_d128_w128_dar_abs_fullbias")
    ap.add_argument("--seqs", type=int, nargs="+", default=[2048])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda"

    print(f"vanilla_ckpt={args.vanilla_ckpt}\nabsorbed_ckpt={args.absorbed_ckpt}\ndtype={args.dtype}  warmup={args.warmup}  steps={args.steps}\n")
    print(f"{'batch':>5} {'seqlen':>7} | {'mode':>10} | {'peak mem':>10} | {'ms/step':>9} | {'ms/tok':>8}")
    results = {}
    for B in args.batch:
        for T in args.seqs:
            for mode, fn, ckpt in [("vanilla", bench_vanilla, args.vanilla_ckpt), ("absorbed", bench_absorbed, args.absorbed_ckpt)]:
                try:
                    peak, ms_step, ms_tok = fn(ckpt, B, T, args.warmup, args.steps, dtype, device, args.seed)
                    print(f"{B:>5} {T:>7} | {mode:>10} | {peak:>7.1f} MB | {ms_step:>7.3f} ms | {ms_tok:>6.4f} ms")
                    results[(B, T, mode)] = (peak, ms_step, ms_tok)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"{B:>5} {T:>7} | {mode:>10} | {'OOM':>10} | {'--':>9} | {'--':>8}")

    print(f"\n{'batch':>5} {'seqlen':>7} | {'mem ratio (van/abs)':>20} | {'speed ratio (van/abs)':>22}")
    for B in args.batch:
        for T in args.seqs:
            v = results.get((B, T, "vanilla"))
            a = results.get((B, T, "absorbed"))
            if v and a:
                mem_ratio = v[0] / a[0]
                speed_ratio = v[1] / a[1]
                print(f"{B:>5} {T:>7} | {mem_ratio:>19.3f}x | {speed_ratio:>21.3f}x")


if __name__ == "__main__":
    main()
