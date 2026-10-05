"""Compare per-step lm_loss and step time between two NeoX logs (flag off vs on)."""
import re, sys, glob, statistics as st

def parse(path):
    loss, ms = {}, {}
    pat = re.compile(r"iteration\s+(\d+)/\s*\d+ \| elapsed time per iteration \(ms\): ([\d.]+) \| .*?lm_loss: ([\d.Ee+-]+)")
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            it = int(m.group(1)); loss[it] = float(m.group(3)); ms[it] = float(m.group(2))
    return loss, ms

off_path = sorted(glob.glob("fastcheck_off_*.txt"))[-1]
on_path = sorted(glob.glob("fastcheck_on_*.txt"))[-1]
lo, mo = parse(off_path); ln, mn = parse(on_path)
common = sorted(set(lo) & set(ln))
print(f"off={off_path} on={on_path} common_steps={len(common)}")
if common:
    d = [abs(lo[i] - ln[i]) for i in common]
    print(f"loss |diff|: max={max(d):.3e} mean={st.mean(d):.3e}  at step1 off={lo[common[0]]:.6f} on={ln[common[0]]:.6f}")
    late = [i for i in common if i > 10]
    if late:
        print(f"step>10 loss |diff| max={max(abs(lo[i]-ln[i]) for i in late):.3e}")
tail = [i for i in common if i > 20]
if tail:
    print(f"median step time ms (steps>20): off={st.median(mo[i] for i in tail if i in mo):.1f} on={st.median(mn[i] for i in tail if i in mn):.1f}")
