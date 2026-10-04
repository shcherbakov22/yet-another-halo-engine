#!/usr/bin/env python3
"""Persistent kres with a deferred residual (rewrites a gen_gemm_tile afrag kres kernel; same values, bit-identical).

The kres epilogue (hidden2 = resid + acc, f32) waits on DRAM round trips for the residual with 128 accumulators live, so
the prefetch depth is capped and each slab group waits most of a round trip. Here each workgroup runs `ntiles` token
tiles of its row block in sequence (grid y = 1; tile i = token tile i):

  tile 0:          K loop, then the kstore epilogue: acc stored raw into the output (no residual)
  tiles 1 .. n-1:  the K loop in two parts: phases [0, 32) also finish the previous tile in 32 slices (each lane: one
                   16-byte vector of the residual and of the stored acc, loaded at the phase top and added / stored
                   before the decode -> MMA barrier, so the DRAM latency hides behind the decode), phases [32, kphases)
                   as before; then the kstore epilogue for this tile
  after the loop:  the last tile's 32 slices, 8 at a time (no accumulators live: deep prefetch)

Each output is still resid + acc in one f32 add (vector.addf %resid, %acc as in the kres epilogue); acc goes through
memory unchanged. Slice v = j * lanes + tid of a tile is token v / 32, row group 4 (v % 32): each wave covers one token's
128 contiguous rows (512 bytes) per access."""
import re

# The partial accumulators are global stores of other waves of the workgroup: make them visible before they are read
# (release: wait until the stores are acknowledged; acquire: drop the WGP caches), with the rendezvous between.
SYNC = ["  buffer.fence scope(device) ordering(release)",
        "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)",
        "  buffer.fence scope(device) ordering(acquire)"]

SLICES = 32
NAME = re.compile(r"%[A-Za-z_][A-Za-z0-9_]*")


def _block_end(L, i):
    d = 0
    for j in range(i, len(L)):
        d += L[j].count("{") - L[j].count("}")
        if d == 0:
            return j
    raise ValueError("unbalanced region")


def _defs(lines):
    """Names defined in these lines: op results, scf.for induction variables and iter args."""
    out = set()
    for l in lines:
        m = re.match(r"\s*((?:%[\w]+)(?:\s*,\s*%[\w]+)*)\s*=\s", l)
        if m:
            out.update(n.strip() for n in m.group(1).split(","))
        m = re.search(r"scf\.for (%[\w]+) = \[[^\]]*\](.*)$", l)
        if m:
            out.add(m.group(1))
            out.update(re.findall(r"(%[\w]+) = ", m.group(2)))
    return out


def _rename(lines, names, suffix):
    def sub(m):
        n = m.group(0)
        return n + suffix if n in names else n
    return [NAME.sub(sub, l) for l in lines]


def persist(kres_text, kstore_text, ntiles, lanes=512, bn=512, bm=128):
    L = kres_text.rstrip("\n").split("\n")
    S = kstore_text.rstrip("\n").split("\n")
    iwx = next(i for i, l in enumerate(L) if l.strip().startswith("%wg_x = kernel.workgroup.id<x>"))
    iwr = next(i for i, l in enumerate(L) if l.strip().startswith("%wg_row = index.mul %wg_x"))
    ik = next(i for i, l in enumerate(L) if "= scf.for %kp = [%c0 to %kphases step %c1]" in l)
    ike = _block_end(L, ik)
    iret = max(i for i, l in enumerate(L) if l.strip() == "kernel.return")
    sk = next(i for i, l in enumerate(S) if "= scf.for %kp = [%c0 to %kphases step %c1]" in l)
    ske = _block_end(S, sk)
    sret = max(i for i, l in enumerate(S) if l.strip() == "kernel.return")
    assert L[ik] == S[sk], "kres / kstore K loops differ"
    head, stag, setup, kloop = L[:iwx + 2], L[iwx + 2:iwr], L[iwr:ik], L[ik:ike + 1]
    epi_store = S[ske + 1:sret]
    assert not any("%resid" in l or "%res_flat" in l for l in epi_store)
    assert sum("kernel.barrier" in l for l in kloop) >= 2

    out = list(head) + list(stag)
    out += [f"  %ps_out = buffer.view %output_na[%base] : buffer -> view<[%out_total]xf32>",
            f"  %ps_res = buffer.view %resid_na[%base] : buffer -> view<[%out_total]xf32>",
            f"  %ps_c32 = index.constant 32 : index",
            f"  %ps_c4 = index.constant 4 : index",
            f"  %ps_lanes = index.constant {lanes} : index",
            f"  %ps_bn = index.constant {bn} : index",
            f"  %ps_cs = index.constant {SLICES} : index",
            f"  %ps_tid = kernel.workitem.id<x> : index",
            f"  %ps_bm = index.constant {bm} : index",
            f"  %ps_wgrow = index.mul %wg_x, %ps_bm : index"]

    def slice_addr(p, j, ty, row="%wg_row"):
        """Offset of slice j (index value) of token tile ty (index value) for this lane: lines + the offset name."""
        return [f"  %{p}v0 = index.mul {j}, %ps_lanes : index",
                f"  %{p}v = index.add %{p}v0, %ps_tid : index",
                f"  %{p}tok = index.div %{p}v, %ps_c32 : index",
                f"  %{p}c = index.rem %{p}v, %ps_c32 : index",
                f"  %{p}tb = index.mul {ty}, %ps_bn : index",
                f"  %{p}t = index.add %{p}tb, %{p}tok : index",
                f"  %{p}a0 = index.mul %{p}t, %m_rows : index",
                f"  %{p}a1 = index.add %{p}a0, {row} : index",
                f"  %{p}c4 = index.mul %{p}c, %ps_c4 : index",
                f"  %{p}off = index.add %{p}a1, %{p}c4 : index"], f"%{p}off"

    def tile(ty, kfirst, prev_ty, suffix):
        """One tile's region: setup, K loop (split when prev_ty), kstore epilogue; names renamed with suffix."""
        reg = [l.replace("%wg_y", ty) for l in setup]
        if prev_ty is None:
            reg += kloop
        else:
            hdr = kloop[0]
            body = kloop[1:-1]
            close = kloop[-1]
            res_names = [n.strip() for n in hdr.split("=", 1)[0].split(",")]
            # part A: phases [0, 32) with the previous tile's slices
            a_res = [n + "_pa" for n in res_names]
            hdr_a = ", ".join(a_res) + " =" + hdr.split("=", 1)[1].replace("[%c0 to %kphases step %c1]", "[%c0 to %ps_cs step %c1]", 1)
            bars = [i for i, l in enumerate(body) if "kernel.barrier" in l]
            b0, b1 = bars[0], bars[1]
            al, off = slice_addr("psa_", "%kp", prev_ty)
            loads = al + [f"  %psa_r = vector.load %ps_res[{off}] : view<[%out_total]xf32> -> vector<4xf32>",
                          f"  %psa_p = vector.load %ps_out[{off}] : view<[%out_total]xf32> -> vector<4xf32>"]
            fin = [f"  %psa_s = vector.addf %psa_r, %psa_p : vector<4xf32>",
                   f"  vector.store %psa_s, %ps_out[{off}] : vector<4xf32>, view<[%out_total]xf32>"]
            body_a = body[:b0 + 1] + ["  " + l for l in loads] + body[b0 + 1:b1] + ["  " + l for l in fin] + body[b1:]
            part_a = [hdr_a] + body_a + [close]
            # part B: phases [32, kphases), iter args from part A
            iters = re.findall(r"(%[\w]+) = ([^,()]+?) : ", hdr.split("](", 1)[1])
            hdr_b = hdr.replace("[%c0 to %kphases step %c1]", "[%ps_cs to %kphases step %c1]", 1)
            for (arg, init), a in zip(iters, a_res):
                hdr_b = hdr_b.replace(f"{arg} = {init} :", f"{arg} = {a} :", 1)
            part_b = _rename([hdr_b] + body + [close], _defs(body) | {"%kp"} | {a for a, _ in iters}, "_pb")
            reg += _rename(part_a, _defs(body) | {"%kp"} | {a for a, _ in iters}, "_pa2") + part_b
        reg += epi_store
        return _rename(reg, _defs(reg), suffix)

    out += tile("%c0", True, None, "_t0")
    # tiles 1 .. ntiles - 1
    out += [f"  %ps_nt = index.constant {ntiles} : index",
            "  scf.for %ps_ti = [%c1 to %ps_nt step %c1] {",
            "    %ps_tp = index.sub %ps_ti, %c1 : index"] + ["  " + l for l in SYNC]
    out += ["  " + l for l in tile("%ps_ti", False, "%ps_tp", "_tl")]
    out += ["  }"]
    # the last tile's slices: 8 at a time
    out += SYNC + [f"  %ps_last = index.constant {ntiles - 1} : index"]
    for g in range(SLICES // 8):
        blk = []
        for s in range(8):
            j = g * 8 + s
            blk += [f"  %psf_j{j} = index.constant {j} : index"]
            al, off = slice_addr(f"psf{j}_", f"%psf_j{j}", "%ps_last", "%ps_wgrow")
            blk += al + [f"  %psf_r{j} = vector.load %ps_res[{off}] : view<[%out_total]xf32> -> vector<4xf32>",
                         f"  %psf_p{j} = vector.load %ps_out[{off}] : view<[%out_total]xf32> -> vector<4xf32>"]
        for s in range(8):
            j = g * 8 + s
            blk += [f"  %psf_s{j} = vector.addf %psf_r{j}, %psf_p{j} : vector<4xf32>",
                    f"  vector.store %psf_s{j}, %ps_out[%psf{j}_off] : vector<4xf32>, view<[%out_total]xf32>"]
        out += blk + ["  scf.schedule.fence"]
    out += ["  kernel.return", "}", ""]
    return "\n".join(out)
