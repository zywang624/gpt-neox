"""Check lskv_w1_fast against the reference path: forward and gradients."""
import sys, math, torch
sys.path.insert(0, ".")
from megatron.model.lskv_w1_fast import lskv_attention_w1_reference as ref, lskv_attention_w1_fast as fast

def run(dtype, device, s=64, b=2, np_=4, hn=32):
    torch.manual_seed(0)
    bnp = b * np_
    norm = math.sqrt(hn)
    causal = torch.triu(torch.ones(s, s, dtype=torch.bool, device=device), 1)
    def softmax_fn(scores):
        scores = scores.float().masked_fill(causal, float("-inf"))
        return torch.softmax(scores, -1).to(dtype)
    outs = {}
    for name, fn in [("ref", ref), ("fast", fast)]:
        torch.manual_seed(1)
        xs = [torch.randn(s, bnp, hn, device=device, dtype=dtype, requires_grad=True) for _ in range(5)]
        out = fn(*xs, norm, softmax_fn)
        out.float().pow(2).sum().backward()
        outs[name] = (out.detach().float(), [x.grad.detach().float() for x in xs])
    o_r, g_r = outs["ref"]; o_f, g_f = outs["fast"]
    print(f"{str(dtype):14s} {device}: fwd max|diff|={ (o_r-o_f).abs().max().item():.3e} (scale {o_r.abs().max().item():.2f})")
    for i, (a, c) in enumerate(zip(g_r, g_f)):
        print(f"   grad[{i}] max|diff|={(a-c).abs().max().item():.3e} (scale {a.abs().max().item():.2f})")

run(torch.float32, "cpu")
if torch.cuda.is_available():
    run(torch.float16, "cuda")
    run(torch.float32, "cuda")
