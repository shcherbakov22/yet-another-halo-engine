#!/usr/bin/env python3
"""The decode sampler (gen_decode_misc.gen_sample) against a numpy model of it, and the model against the exact top-p
distribution the host sampler drew from.

usage: sample_check.py <model.gguf> <workdir> [draws]

The model repeats the kernel's steps (nucleus threshold by binary search, the counter hash, Gumbel-max) in float64, so
the GPU must return its token for every (T, top_p, seed, pos) case (the kernel's exp / log are approximate, so a
near-tie could differ; none should with these logits). Over draws seeds the model's token frequencies must
match softmax(logits / T) cut to its top-p nucleus (total variation distance below a sampling-noise bound).
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bfp16_check as B  # noqa: E402
import gen_decode_misc as DM  # noqa: E402

V = 248320
M32 = 0xFFFFFFFF


def fmix(x):
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & M32
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & M32
    return x ^ (x >> 16)


def fmix_np(x):
    x = x ^ (x >> np.uint64(16))
    x = (x * np.uint64(0x85EBCA6B)) & np.uint64(M32)
    x = x ^ (x >> np.uint64(13))
    x = (x * np.uint64(0xC2B2AE35)) & np.uint64(M32)
    return x ^ (x >> np.uint64(16))


def threshold(logits, t, top_p):
    """The kernel's nucleus threshold: its float32 mid points, sums in float64."""
    mx = np.float32(logits.max())
    cut = np.float32(mx - np.float32(20.723265836946411) * np.float32(t))
    e = np.exp((logits.astype(np.float64) - mx) / t)

    def mass(lam):
        return e[logits >= lam].sum()
    target = mass(cut) * top_p
    lo, hi = cut, mx
    for _ in range(DM.SAMPLE_ITERS):
        mid = np.float32(np.float32(lo + hi) * np.float32(0.5))
        lo, hi = (mid, hi) if mass(mid) >= target else (lo, mid)
    return lo


def draw(logits, lam, t, seed, pos):
    """The kernel's token for this threshold: Gumbel-max over the kept tokens, the lowest index on ties."""
    kept = np.nonzero(logits >= lam)[0]
    rng = fmix(fmix(((pos * 0x9E3779B1) & M32) ^ (seed >> 32)) ^ (seed & M32))
    h = fmix_np(((kept.astype(np.uint64) * np.uint64(0x9E3779B1)) & np.uint64(M32)) ^ np.uint64(rng))
    u = ((h >> np.uint64(8)).astype(np.float64) + 0.5) * 2.0 ** -24
    score = (logits[kept].astype(np.float64) - logits.max()) / t - np.log(-np.log(u))
    return int(kept[np.argmax(score)])


def nucleus(logits, t, top_p):
    """The host sampler's distribution: softmax(logits / T) over the top_p nucleus (sorted, as engine.hpp Sample)."""
    p = np.exp((logits.astype(np.float64) - logits.max()) / t)
    order = np.argsort(-p, kind="stable")
    c = np.cumsum(p[order])
    k = int(np.searchsorted(c, top_p * c[-1])) + 1
    q = np.zeros(V)
    q[order[:k]] = p[order[:k]] / c[k - 1]
    return q


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model_path, work = sys.argv[1], sys.argv[2]
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "sample.loom")
    open(src, "w").write(DM.gen_sample(V))
    r = subprocess.run([sys.executable, B.EMIT, src, os.path.join(work, "sample"), "nop=0"], capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(f"emit failed\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    hal = r.stdout.strip().splitlines()[0]
    # logits shaped like a model's (a long tail and a handful of strong candidates), and flat ones (a nucleus of tens of
    # thousands, the threshold among many near-equal logits)
    g = np.random.default_rng(1)
    peaked = g.normal(-4.0, 2.0, V).astype(np.float32)
    peaked[g.choice(V, 12, replace=False)] = np.float32(12.0) + g.normal(0, 1.5, 12).astype(np.float32)
    flat = g.normal(0.0, 1.0, V).astype(np.float32)
    lf = os.path.join(work, "logits.f32")

    ok = n = 0
    for logits, t, top_p in ((peaked, 0.6, 0.95), (peaked, 1.0, 1.0), (peaked, 1.3, 0.5), (flat, 1.0, 0.9),
                             (flat, 0.6, 0.5)):
        logits.tofile(lf)
        lam = threshold(logits, t, top_p)
        print(f"  T {t} top_p {top_p}: {int((logits >= lam).sum())} tokens kept")
        for seed, pos in ((1, 0), (0x123456789ABCDEF, 17), (42, 4095), (7, 70000), (2**64 - 1, 3)):
            pf, sf, of = (os.path.join(work, x) for x in ("params.bin", "pos.bin", "out.bin"))
            np.array([t, top_p], np.float32).tofile(pf)
            with open(pf, "ab") as f:
                np.array([seed & M32, seed >> 32], np.uint32).tofile(f)
            np.array([pos], np.int32).tofile(sf)
            cmd = [B.GPURUN, "sample", "--", B.HALRUN, model_path, hal, "1", "1024", f"{V * 4},16,4,4", f"f:{lf}",
                   f"f:{pf}", f"f:{sf}", f"o:4:{of}"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if "hal_run: ok" not in r.stdout:
                sys.exit(f"hal_run failed\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
            got = int(np.fromfile(of, np.int32)[0])
            want = draw(logits, lam, t, seed, pos)
            n += 1
            ok += got == want
            if got != want:
                print(f"  T {t} top_p {top_p} seed {seed:#x} pos {pos}: GPU {got} (logit {logits[got]:.4f}), "
                      f"model {want} (logit {logits[want]:.4f}), threshold {lam:.6f}")
    print(f"sample: GPU = model on {ok} / {n} cases")

    worst = 0.0
    logits = peaked
    for t, top_p in ((0.6, 0.95), (1.3, 0.5)):
        q = nucleus(logits, t, top_p)
        lam = threshold(logits, t, top_p)
        counts = np.zeros(V)
        for s in range(draws):
            counts[draw(logits, lam, t, 0xABCDEF00 + s, s % 4096)] += 1
        tv = 0.5 * np.abs(counts / draws - q).sum()
        bound = 1.5 * np.sqrt((q > 0).sum() / draws)   # ~E[TV] of draws samples is <= sqrt(k / draws) / 2
        print(f"sample model: T {t} top_p {top_p}: {int((q > 0).sum())} tokens in the nucleus, total variation "
              f"{tv:.4f} over {draws} draws (bound {bound:.4f})")
        worst = max(worst, tv / bound)
    if ok != n or worst > 1.0:
        sys.exit("sample: FAIL")
    print("sample: ok")


if __name__ == "__main__":
    main()
