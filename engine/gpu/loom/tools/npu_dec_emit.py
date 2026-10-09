"""Shared pieces of the NPU weight decoders (gen_npu_dec): the low-asm emitter with independent op streams, register
type names, bf16 / bfp16 reference conversions and the llama.cpp block field parsers the oracles use."""
import numpy as np

V1, V2, V4 = "reg<aie2p.vec256>", "reg<aie2p.vec256 x2>", "reg<aie2p.vec256 x4>"
M1, M2, M4 = "reg<aie2p.mbms>", "reg<aie2p.mbms x2>", "reg<aie2p.mbms x4>"
# IQ4_XS non-linear values (llama.cpp kvalues_iq4nl)
KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


class Emit:
    """Low asm text with SSA temporaries; stream() makes an independent op list (own cursors) that zip() interleaves
    round-robin, so the locked in-order schedule can co-issue adjacent independent ops. Constants go to the root."""

    def __init__(self, root=None, sid=0):
        self.L = []
        self.root = root or self
        self.sid = sid
        self.cursors = {}
        if root is None:
            self.n = 0
            self.consts = {}
            self.pre = []
            self.nstreams = 0

    def __call__(self, s):
        self.L.append("  " + s)

    def t(self, p="t"):
        self.root.n += 1
        return f"%{p}_{self.root.n}"

    def const(self, v):
        nm = f"%k{v}".replace("-", "m")
        if nm not in self.root.consts:
            self.root.consts[nm] = True
            self.root.pre.append(f"  {nm} = {'mova.i32' if -1024 <= v <= 1023 else 'mov.i32'} {v}")
        return nm

    def fixed(self, base, at, cursor):
        """A pointer to base + at (from this stream's cursor) for streams to share: (reg, at)."""
        reg, imm = self.addr(base, at, 0, 64, cursor)
        assert imm == 0
        return reg, at

    def stream(self):
        self.root.nstreams += 1
        return Emit(self.root, self.root.nstreams)

    def zip_tails(self, streams, lag):
        """Staggered merge: stream i starts at step lag * i, one op per stream per step, but an op waits until every
        SSA value it reads from the streams has been emitted; x.tail runs after the body and after the previous tail."""
        import re as _re
        streamdefs = set()
        for x in streams:
            for op in x.L + getattr(x, "tail", []):
                if "=" in op:
                    streamdefs.update(_re.findall(r"%[\w.]+", op.split("=", 1)[0]))
        done = set()
        ops = [list(x.L) for x in streams]
        tails = [list(getattr(x, "tail", [])) for x in streams]
        pos = [0] * len(streams)
        tpos = [0] * len(streams)

        def ready(op):
            rhs = op.split("=", 1)[1] if "=" in op.split("(")[0] or " = " in op else op
            return all(u in done or u not in streamdefs for u in _re.findall(r"%[\w.]+", rhs))

        def emit(op):
            self.L.append(op)
            if "=" in op:
                done.update(_re.findall(r"%[\w.]+", op.split("=", 1)[0]))

        t = 0
        while any(pos[i] < len(ops[i]) or tpos[i] < len(tails[i]) for i in range(len(streams))):
            progressed = False
            for i in range(len(streams)):
                if t < lag * i:
                    break
                if pos[i] < len(ops[i]):
                    if ready(ops[i][pos[i]]):
                        emit(ops[i][pos[i]])
                        pos[i] += 1
                        progressed = True
                elif tpos[i] < len(tails[i]) and (i == 0 or tpos[i - 1] == len(tails[i - 1])):
                    if ready(tails[i][tpos[i]]):
                        emit(tails[i][tpos[i]])
                        tpos[i] += 1
                        progressed = True
            t += 1
            if not progressed and t > lag * len(streams) + 100000:
                raise RuntimeError("zip_tails deadlock")

    def zip(self, streams, lag=0):
        """Round-robin merge; stream i starts lag * i steps late."""
        lists = [x.L for x in streams]
        n = max((len(x) + lag * i for i, x in enumerate(lists)), default=0)
        for t in range(n):
            for i, x in enumerate(lists):
                if 0 <= t - lag * i < len(x):
                    self.L.append(x[t - lag * i])

    def _advance(self, cur, delta):
        left = delta - delta % 64
        while left:
            st = max(-512, min(448, left))
            nxt = self.t("pb")
            self(f"{nxt} = padda {cur}, {st}")
            cur, left = nxt, left - st
        if delta % 64:
            md, nxt = self.t("md"), self.t("pb")
            self(f"{md} = mov.modifier {self.const(delta % 64)}")
            self(f"{nxt} = padda.modifier {cur}, {md}")
            cur = nxt
        return cur

    def addr(self, base, off, hi=448, step=64, cursor=None):
        """(pointer, immediate) for base + off; immediates 0..hi in steps (and down to -(hi + step)).
        Each cursor is one pointer per region that only advances (consuming the previous one)."""
        name = (cursor or base) + f"#{self.sid}"
        if name not in self.cursors:
            if -(hi + step) <= off <= hi and cursor is None:
                return base, off
            cur = self.t("pc")
            self(f"{cur} = copy {base} : reg<aie2p.ep> -> reg<aie2p.ep>")
            self.cursors[name] = (cur, 0)
        cur, at = self.cursors[name]
        rel = off - at
        if -(hi + step) <= rel <= hi and rel % step == 0:
            return cur, rel
        tgt = off - (off % step) if rel > hi else off - (off % step)
        new = self._advance(cur, tgt - at)
        self.cursors[name] = (new, tgt)
        return new, off - tgt


def bfp_hw(w):
    """w [n][8 rows][8] f32 -> [n][72] bytes: E = exp(max|x|), +1 if a rounded mantissa leaves int8; subnormals flush."""
    w = np.where(np.abs(w) < np.float32(2.0 ** -126), np.float32(0), w)
    amax = np.abs(w).max(2)
    E = ((amax.view(np.uint32) >> 23) & 255).astype(np.int64)
    m0 = np.round(w.astype(np.float64) * np.exp2(133 - E[..., None]))
    E = E + np.any((m0 > 127) | (m0 < -128), axis=2)
    m = np.clip(np.round(w.astype(np.float64) * np.exp2(133 - E[..., None])), -128, 127).astype(np.int8)
    out = np.zeros(w.shape[:2] + (9,), np.uint8)
    out[..., 0] = E
    out[..., 1:] = m.view(np.uint8)
    return out.reshape(w.shape[0], 72)


def bf16_rne(x):
    """f32 -> nearest-even bf16, as f32."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def bf16_bits(v):
    return int(np.float32(v).view(np.uint32)) >> 16


def q4k_fields(blk):
    """blk [..., 144] u8 -> d, dmin (f64), sc, m [..., 8], q [..., 256] (llama.cpp layout)."""
    d = blk[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
    dm = blk[..., 2:4].copy().view(np.float16)[..., 0].astype(np.float64)
    s = blk[..., 4:16].astype(np.int64)
    sc = np.zeros(blk.shape[:-1] + (8,), np.int64)
    mn = np.zeros_like(sc)
    for j in range(4):
        sc[..., j] = s[..., j] & 63
        mn[..., j] = s[..., j + 4] & 63
    for j in range(4, 8):
        sc[..., j] = (s[..., j + 4] & 15) | ((s[..., j - 4] >> 6) << 4)
        mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
    qs = blk[..., 16:144].astype(np.int64).reshape(blk.shape[:-1] + (4, 32))
    q = np.concatenate([np.stack([qs[..., j, :] & 15, qs[..., j, :] >> 4], -2) for j in range(4)], -2)
    return d, dm, sc, mn, q.reshape(blk.shape[:-1] + (256,))


def iq4xs_fields(blk):
    """blk [..., 136] u8 -> d (f64), ls [..., 8], q [..., 256] (llama.cpp block_iq4_xs)."""
    d = blk[..., 0:2].copy().view(np.float16)[..., 0].astype(np.float64)
    sh = blk[..., 2].astype(np.int64) | (blk[..., 3].astype(np.int64) << 8)
    sl = blk[..., 4:8].astype(np.int64)
    ls = np.stack([((sl[..., j // 2] >> (4 * (j % 2))) & 15) | (((sh >> (2 * j)) & 3) << 4) for j in range(8)], -1)
    qs = blk[..., 8:136].astype(np.int64).reshape(blk.shape[:-1] + (8, 16))
    q = np.concatenate([qs & 15, qs >> 4], -1).reshape(blk.shape[:-1] + (256,))
    return d, ls, q
