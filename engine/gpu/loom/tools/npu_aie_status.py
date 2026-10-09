#!/usr/bin/env python3
"""The NPU firmware's tile status for the columns of live hardware contexts (amdxdna DRM_AMDXDNA_QUERY_AIE_STATUS),
decoded per tile: each DMA channel's state (running / stalled on a lock / starved of stream data, its current BD),
the core's status bits and the nonzero lock values. Meant for a stalled NPU job: YAH_NPU_CPU=1 runs it at a stall
through YAH_NPU_CPU_STALL_CMD, while the context lives (docs/build-and-run.md). Comparing the dump with one of the same
job parked healthy at its gate (YAH_NPU_CPU_HOLD) shows the column that differs.

npu_aie_status.py [--save raw.bin] | --load raw.bin

Only query while NPU work runs: GET_INFO wakes a suspended NPU (btop fork notes). Layout (XRT info_aie2): per column
the core tiles (rows 2..5: per DMA channel s2mm / mm2s status, core- and memory-mode events, core status, PC, SP, LR,
lock bytes), the memory tile and the shim (DMA status pairs, events, lock bytes)."""
import ctypes, fcntl, struct, sys

GET_INFO = 0xC0106447   # DRM_IOCTL_AMDXDNA_GET_INFO
QUERY_AIE_STATUS, QUERY_AIE_METADATA = 0, 1
CORE = {0: "en", 1: "rst", 2: "memS", 3: "memW", 4: "memN", 5: "memE", 6: "lkS", 7: "lkW", 8: "lkN", 9: "lkE",
        10: "ssIn", 12: "msOut", 14: "cascIn", 15: "cascOut", 16: "dbg", 17: "ecc", 19: "errHalt", 20: "DONE",
        21: "busStall"}   # AIE2P Core_Status


def get_info(fd, param, buf):
    req = struct.pack("<IIQ", param, ctypes.sizeof(buf), ctypes.addressof(buf))
    fcntl.ioctl(fd, GET_INFO, bytearray(req))


def query(dev="/dev/accel/accel0"):
    """(metadata dict, filled-column bitmap, raw status bytes)."""
    with open(dev, "rb+", buffering=0) as f:
        md = ctypes.create_string_buffer(64)
        get_info(f.fileno(), QUERY_AIE_METADATA, md)
        col_size, cols, rows = struct.unpack_from("<IHH", md.raw)
        tiles = [struct.unpack_from("<HHHHH", md.raw, 16 + 16 * i) for i in range(3)]   # core, mem, shim
        raw = ctypes.create_string_buffer(col_size * cols)
        st = ctypes.create_string_buffer(struct.pack("<QII", ctypes.addressof(raw), col_size * cols, 0))
        get_info(f.fileno(), QUERY_AIE_STATUS, st)
        filled = struct.unpack_from("<QII", st.raw)[2]
    meta = {"col_size": col_size, "cols": cols, "rows": rows,
            **{k: dict(zip(("rows", "start", "dma", "locks", "events"), t)) for k, t in zip(("core", "mem", "shim"), tiles)}}
    return meta, filled, raw.raw


def dma(v):
    if v == 0:
        return "idle"
    flags = [n for b, n in ((2, "LOCKACQ"), (3, "LOCKREL"), (4, "STARVE"), (5, "TCTFULL"), (8, "ELOCK"), (9, "EDM"),
                            (10, "EBDU"), (11, "EBDI"), (12, "EFOT"), (18, "QOVF")) if v >> b & 1]
    return (["idle", "start", "run", "?"][v & 3] + ("+" + "+".join(flags) if flags else "") +
            f" bd{v >> 24 & 63}" + (" q%d" % (v >> 20 & 7) if v >> 20 & 7 else ""))


def decode(meta, filled, raw):
    core, mem, shim = meta["core"], meta["mem"], meta["shim"]
    out = []
    for i in range(bin(filled).count("1")):
        w = raw[i * meta["col_size"]:(i + 1) * meta["col_size"]]
        o = 0

        def words(n):
            nonlocal o
            v = struct.unpack_from(f"<{n}I", w, o)
            o += 4 * n
            return v

        def locks(n):
            nonlocal o
            v = w[o:o + n]
            o += n
            return " ".join(f"{k}={x}" for k, x in enumerate(v) if x)

        def chans(n):
            v = words(2 * n)
            return " | ".join(f"{d}{c} {dma(v[2 * c + (d == 'mm2s')])}" for c in range(n) for d in ("s2mm", "mm2s")
                              if v[2 * c + (d == "mm2s")])

        for r in range(core["rows"]):
            ch = chans(core["dma"])
            words(2 * core["events"])
            cs, pc, sp, lr = words(4)
            flags = "|".join(n for k, n in CORE.items() if cs >> k & 1)
            out.append(f"c{i} r{core['start'] + r}: core {cs:08x} {flags:20s} {ch}  locks {locks(core['locks'])}")
        for name, t in (("mem", mem), ("shim", shim)):
            ch = chans(t["dma"])
            words(t["events"])
            out.append(f"c{i} {name}: {ch}  locks {locks(t['locks'])}")
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--load":
        b = open(sys.argv[2], "rb").read()
        n = struct.unpack_from("<I", b)[0]
        import json
        meta = json.loads(b[4:4 + n])
        filled, raw = meta.pop("filled"), b[4 + n:]
    else:
        meta, filled, raw = query()
        if len(sys.argv) == 3 and sys.argv[1] == "--save":
            import json
            j = json.dumps(dict(meta, filled=filled)).encode()
            open(sys.argv[2], "wb").write(struct.pack("<I", len(j)) + j + raw)
    print(f"columns filled 0x{filled:x}")
    print("\n".join(decode(meta, filled, raw)))
