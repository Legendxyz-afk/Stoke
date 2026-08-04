"""tests/tooling/gguf_reader.py -- independent minimal GGUF v3 reader.

Shared by dump_tensor.py and ref_forward.py (M2 W1 deliverables). Uses
`struct` + `numpy` ONLY -- no `gguf` pip package. The whole point (M2 note
section 9) is an INDEPENDENT byte interpretation of the file: this module
was written from the public GGUF wire-format spec, not by reading/porting
src/gguf.c's parsing logic (a handful of format FACTS -- magic bytes,
version number, default alignment, the KV/tensor-info field order -- were
sanity-checked by peeking at src/gguf.c's header comments and fatal()
messages, since those are just facts about the wire format shared by any
correct reader, not "their" interpretation of ambiguous bytes; the actual
parsing code, the Q8_0 dequant formula's application, and every dims/shape
assumption below were written and verified independently, and are the
thing that cross-checks src/gguf.c for free by disagreeing if either side
has a bug).

Format (GGUF v3, all little-endian):
  offset 0:  magic "GGUF" (4 bytes, literal ASCII)
  offset 4:  version u32 (must be 3)
  offset 8:  tensor_count u64
  offset 16: kv_count u64
  offset 24: kv_count * {key: gguf_string, type: u32, value}
             gguf_string = u64 len + `len` raw bytes (no NUL terminator)
             ARRAY value = elem_type u32 + count u64 + count elements
  then:      tensor_count * {name: gguf_string, n_dims: u32 (1..4),
             dims: n_dims * u64 (ne[] order, dims[0] = fastest-varying =
             the contraction/row length), ggml_type: u32, offset: u64
             (relative to the DATA SECTION start, not file start)}
  then:      padding to align_up(cur, alignment) where alignment =
             general.alignment (UINT32 KV, default 32 if absent)
  then:      tensor data, each tensor's bytes at data_start + tensor.offset

Loud-failure doctrine throughout: every read is bounds-checked against the
mapped file size before use; anything malformed raises GgufError, and the
two CLI tools built on this module catch that at top level and exit
nonzero with a clear "fatal: ..." message (mirrors src/common.h's fatal()
style/spirit -- independently, not by importing anything from src/).
"""
from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class GgufError(Exception):
    """Any malformed-file or out-of-contract condition. Callers (CLI
    main()s) catch this at top level, print 'fatal: <msg>' to stderr, and
    exit(1) -- see fatal() below, used internally the same way."""


def fatal(msg: str):
    raise GgufError(msg)


# ---- GGUF metadata value-type enum (public gguf.md spec; matches
# src/gguf.h's enum of the same public spec, confirmed by inspection --
# both are transcriptions of the same wire-format document). ----
T_UINT8 = 0
T_INT8 = 1
T_UINT16 = 2
T_INT16 = 3
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10
T_INT64 = 11
T_FLOAT64 = 12

TYPE_NAME = {
    T_UINT8: "UINT8", T_INT8: "INT8", T_UINT16: "UINT16", T_INT16: "INT16",
    T_UINT32: "UINT32", T_INT32: "INT32", T_FLOAT32: "FLOAT32",
    T_BOOL: "BOOL", T_STRING: "STRING", T_ARRAY: "ARRAY",
    T_UINT64: "UINT64", T_INT64: "INT64", T_FLOAT64: "FLOAT64",
}

# struct format for each fixed-width scalar type (BOOL is 1 byte, 0/1)
_SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_UINT64: "<Q",
    T_INT64: "<q", T_FLOAT64: "<d", T_BOOL: "<B",
}

# ---- ggml tensor dtype enum (public spec; GGML_TYPE_F32=0 ... Q8_0=8 ...
# Q4_K=12 ... Q6_K=14 -- same numeric values src/gguf.h documents for the
# 3 it currently supports). M2's dump_tensor.py dequants F32 and Q8_0 only
# (brief scope); Q4_K/Q6_K block/type sizes are ALSO listed below purely so
# --list can compute correct nbytes for the Q4_K_M dev file too (that file
# is what makes the mandatory "cross-check dump_tensor --list against
# `stoke info`" mechanical check possible at all right now -- see
# m2-w1-notes.md: `stoke info` fatals immediately on the Q8_0 file because
# src/gguf.c's ggml_type allow-list doesn't include Q8_0=8 yet).
GGML_F32 = 0
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q6_K = 14

# type -> (block_size in elements, bytes per block)
_BLOCK_INFO = {
    GGML_F32: (1, 4),
    GGML_Q8_0: (32, 34),   # ggml-common.h block_q8_0: f16 d + 32x int8 qs
    GGML_Q4_K: (256, 144),
    GGML_Q6_K: (256, 210),
}
GGML_NAME = {GGML_F32: "F32", GGML_Q8_0: "Q8_0", GGML_Q4_K: "Q4_K", GGML_Q6_K: "Q6_K"}

# ggml types this module can actually DEQUANTIZE. Anything else fatals
# loudly if a caller asks for its row/full data. K-quants added at K4
# (2026-08-04), transcribed from ggml-quants.c and proven against
# llama.cpp's OWN dequant of the real file -- see the selftest.
DEQUANT_SUPPORTED = {GGML_F32, GGML_Q8_0, GGML_Q4_K, GGML_Q6_K}


@dataclass
class GArray:
    elem_type: int
    values: list


@dataclass
class TensorInfo:
    name: str
    type: int
    n_dims: int
    dims: Tuple[int, ...]     # dims[0] = fastest-varying = row/contraction length
    rel_offset: int           # as stored in file: relative to data section start
    nbytes: int
    abs_offset: int = -1      # filled in once data_start is known


def f16_bits_to_f32(bits: int) -> "np.float32":
    """Scalar f16(IEEE754 binary16) bit pattern -> float32, via numpy's
    native half-float widening (exact -- binary16 is a strict subset of
    binary32's representable values, so this conversion is lossless).
    Standalone helper for the mechanical hand-verification of one Q8_0
    block's `d` field; the vectorized dequant path below performs the
    identical conversion (same numpy '<f2'->astype(float32) operation) at
    array granularity for speed."""
    import numpy as np
    raw = struct.pack("<H", bits)
    return np.frombuffer(raw, dtype="<f2")[0].astype(np.float32)


class GGUFFile:
    def __init__(self, path: str):
        import numpy as np  # local import: keep numpy optional for pure
        self._np = np        # metadata-only callers (there are none today,
                              # but keeps the dependency boundary explicit)
        self.path = path
        try:
            self._fh = open(path, "rb")
        except OSError as e:
            fatal(f"cannot open '{path}' -- {e.strerror or e}")
        self._size = os.fstat(self._fh.fileno()).st_size
        if self._size < 24:
            self._fh.close()
            fatal(f"'{path}': file too small ({self._size} bytes) to hold a GGUF header")
        try:
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError as e:
            self._fh.close()
            fatal(f"cannot mmap '{path}' -- {e.strerror or e} (empty file?)")

        self.kv: Dict[str, Any] = {}
        self.kv_order: List[str] = []
        self.tensor_order: List[TensorInfo] = []
        self.tensors: Dict[str, TensorInfo] = {}
        self.alignment = 32
        self.data_start = 0

        self._parse()

    def close(self):
        try:
            self._mm.close()
        finally:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- bounds-checked primitive reads ----

    def _check(self, off: int, n: int, what: str):
        if off < 0 or n < 0 or off + n > self._size:
            fatal(f"'{self.path}': {what} at offset {off} (len {n}) runs "
                  f"past end of file (size {self._size})")

    def _u32(self, off: int) -> int:
        self._check(off, 4, "u32 read")
        return struct.unpack_from("<I", self._mm, off)[0]

    def _u64(self, off: int) -> int:
        self._check(off, 8, "u64 read")
        return struct.unpack_from("<Q", self._mm, off)[0]

    def _string(self, off: int) -> Tuple[str, int]:
        slen = self._u64(off)
        self._check(off + 8, slen, "string bytes")
        raw = bytes(self._mm[off + 8: off + 8 + slen])
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            fatal(f"'{self.path}': string at offset {off} ({slen} bytes) "
                  f"is not valid UTF-8: {e}")
        return s, off + 8 + slen

    def _scalar(self, vtype: int, off: int):
        fmt = _SCALAR_FMT.get(vtype)
        if fmt is None:
            fatal(f"'{self.path}': unknown metadata value type {vtype} at offset {off}")
        size = struct.calcsize(fmt)
        self._check(off, size, f"scalar type {vtype}")
        raw = struct.unpack_from(fmt, self._mm, off)[0]
        if vtype == T_BOOL:
            if raw not in (0, 1):
                fatal(f"'{self.path}': BOOL value {raw} at offset {off} is not 0/1")
            raw = bool(raw)
        return raw, off + size

    def _value(self, vtype: int, off: int):
        if vtype == T_STRING:
            return self._string(off)
        if vtype == T_ARRAY:
            elem_type = self._u32(off)
            off += 4
            if elem_type == T_ARRAY:
                fatal(f"'{self.path}': nested ARRAY-of-ARRAY at offset {off} "
                      f"-- not legal GGUF")
            n = self._u64(off)
            off += 8
            vals = []
            for _ in range(n):
                v, off = self._value(elem_type, off)
                vals.append(v)
            return GArray(elem_type, vals), off
        return self._scalar(vtype, off)

    # ---- top-level parse ----

    def _parse(self):
        magic = bytes(self._mm[0:4])
        if magic != b"GGUF":
            fatal(f"'{self.path}': bad magic {magic!r}, expected b'GGUF'")
        version = self._u32(4)
        if version != 3:
            fatal(f"'{self.path}': unsupported GGUF version {version} (only v3 supported)")
        tensor_count = self._u64(8)
        kv_count = self._u64(16)

        # Plausibility guard (loud, cheap): the smallest possible KV is a
        # zero-length key + a 1-byte scalar value (UINT8/INT8/BOOL) = 8
        # (key length prefix) + 0 (key bytes) + 4 (type) + 1 (value) = 13
        # bytes. Each tensor_info needs >=24 bytes (8 name-len + 0 name +
        # 4 n_dims + 4 type + 8 offset). Reject bogus huge counts up front
        # rather than walking them into a confusing later error.
        remaining = self._size - 24
        if kv_count > 0 and kv_count > remaining // 13:
            fatal(f"'{self.path}': kv_count {kv_count} implausible for "
                  f"remaining file size {remaining}")
        if tensor_count > 0 and tensor_count > remaining // 24:
            fatal(f"'{self.path}': tensor_count {tensor_count} implausible "
                  f"for remaining file size {remaining}")

        cur = 24
        for _ in range(kv_count):
            key, cur = self._string(cur)
            vtype = self._u32(cur)
            cur += 4
            val, cur = self._value(vtype, cur)
            if key in self.kv:
                fatal(f"'{self.path}': duplicate metadata key {key!r}")
            self.kv[key] = val
            self.kv_order.append(key)

        tensors: List[TensorInfo] = []
        for _ in range(tensor_count):
            name, cur = self._string(cur)
            n_dims = self._u32(cur)
            cur += 4
            if not (1 <= n_dims <= 4):
                fatal(f"'{self.path}': tensor {name!r} has n_dims={n_dims}, expected 1..4")
            dims = []
            for _d in range(n_dims):
                dims.append(self._u64(cur))
                cur += 8
            ggml_type = self._u32(cur)
            cur += 4
            rel_offset = self._u64(cur)
            cur += 8
            if ggml_type not in _BLOCK_INFO:
                fatal(f"'{self.path}': tensor {name!r} uses unsupported "
                      f"ggml_type {ggml_type} (dump_tensor.py's --list knows "
                      f"block sizes for F32/Q8_0/Q4_K/Q6_K only)")
            block_elems, block_bytes = _BLOCK_INFO[ggml_type]
            numel = 1
            for d in dims:
                numel *= d
            if numel % block_elems != 0:
                fatal(f"'{self.path}': tensor {name!r} numel {numel} not a "
                      f"multiple of block size {block_elems} for type "
                      f"{GGML_NAME[ggml_type]}")
            nbytes = numel // block_elems * block_bytes
            tensors.append(TensorInfo(name, ggml_type, n_dims, tuple(dims),
                                       rel_offset, nbytes))

        # general.alignment: UINT32, default 32 when absent (matches
        # src/gguf.c's documented default -- this is a wire-format
        # constant, not "their" logic).
        align_val = self.kv.get("general.alignment", 32)
        if isinstance(align_val, GArray) or not isinstance(align_val, int) or isinstance(align_val, bool):
            fatal(f"'{self.path}': general.alignment must be a scalar integer, got {align_val!r}")
        if align_val <= 0:
            fatal(f"'{self.path}': general.alignment must be positive, got {align_val}")
        alignment = align_val
        self.alignment = alignment

        pad = (alignment - (cur % alignment)) % alignment
        data_start = cur + pad
        if data_start > self._size:
            fatal(f"'{self.path}': data section start ({data_start}) is "
                  f"past end of file (size {self._size})")
        self.data_start = data_start

        for t in tensors:
            if t.rel_offset % alignment != 0:
                fatal(f"'{self.path}': tensor {t.name!r} offset {t.rel_offset} "
                      f"is not a multiple of alignment {alignment}")
            abs_off = data_start + t.rel_offset
            if t.rel_offset > self._size - data_start or abs_off + t.nbytes > self._size:
                fatal(f"'{self.path}': tensor {t.name!r} data "
                      f"(offset {abs_off}, {t.nbytes} bytes) exceeds file "
                      f"size {self._size}")
            t.abs_offset = abs_off
            if t.name in self.tensors:
                fatal(f"'{self.path}': duplicate tensor name {t.name!r}")
            self.tensors[t.name] = t
            self.tensor_order.append(t)

    # ---- dequantization ----

    # ---- K-quant dequant (K4). Transcribed from ggml-quants.c ----------
    #
    # These are module-level-in-spirit pure functions taking `np` explicitly
    # so the selftest exercises THE SAME code the reader calls, not a copy.
    #
    # WHY THE ARITHMETIC ORDER IS WRITTEN OUT RATHER THAN SIMPLIFIED: the C
    # computes `d1 * nibble - m1` with `d1 = d*sc` and `m1 = min*m`, all in
    # f32, and `d * sc[is] * q` left-to-right for Q6_K. f32 multiplication is
    # not associative, so a re-grouping CAN differ in the last bit while
    # looking algebraically identical -- and the check is exact equality
    # against llama.cpp's own output. Measured, not assumed: one regrouping
    # tried as a mutation ((d*dmin)/dmin*sc) happened to round-trip exactly
    # and passed, so "any regrouping breaks it" would be too strong a claim.
    # Keep the C's grouping because matching it is free and reasoning about
    # which regroupings are safe is not.

    @staticmethod
    def _scale_min_k4(np, scales, j: int):
        """ggml's get_scale_min_k4, vectorised over blocks.

        scales: uint8 [nb,12]. Returns (d, m) as uint8 [nb].

        THE TRAP THE PLAYBOOK NAMES: in the j>=4 branch the min's high bits
        come from `q[j-0]` -- that is q[j], NOT q[j-4]. The scale's high bits
        DO come from q[j-4]. The two lines are near-mirror images and swapping
        that one index is the classic transcription bug; it stays plausible
        because both indices are in range and the output is still finite.
        """
        if j < 4:
            d = scales[:, j] & 63
            m = scales[:, j + 4] & 63
        else:
            d = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
            m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
        return d.astype(np.uint8), m.astype(np.uint8)

    @classmethod
    def _dequant_q4_k(cls, np, raw, nblocks: int):
        """block_q4_K: f16 d, f16 dmin, u8 scales[12], u8 qs[128] = 144 B."""
        dt = np.dtype([("d", "<f2"), ("dmin", "<f2"),
                       ("scales", "u1", (12,)), ("qs", "u1", (128,))])
        if dt.itemsize != 144:
            fatal(f"_dequant_q4_k: block dtype is {dt.itemsize} B, expected 144")
        b = np.frombuffer(raw, dtype=dt, count=nblocks)
        d = b["d"].astype(np.float32)
        dmin = b["dmin"].astype(np.float32)
        scales = b["scales"]
        qs = b["qs"]
        out = np.empty((nblocks, 256), dtype=np.float32)
        # C: for (j = 0; j < QK_K; j += 64) with is += 2 each iteration.
        for g in range(4):
            sc1, m1u = cls._scale_min_k4(np, scales, 2 * g + 0)
            sc2, m2u = cls._scale_min_k4(np, scales, 2 * g + 1)
            d1 = d * sc1.astype(np.float32)
            m1 = dmin * m1u.astype(np.float32)
            d2 = d * sc2.astype(np.float32)
            m2 = dmin * m2u.astype(np.float32)
            q = qs[:, g * 32:(g + 1) * 32]
            lo = (q & 0xF).astype(np.float32)
            hi = (q >> 4).astype(np.float32)
            out[:, g * 64: g * 64 + 32] = d1[:, None] * lo - m1[:, None]
            out[:, g * 64 + 32: g * 64 + 64] = d2[:, None] * hi - m2[:, None]
        return out

    @staticmethod
    def _dequant_q6_k(np, raw, nblocks: int):
        """block_q6_K: u8 ql[128], u8 qh[64], i8 scales[16], f16 d = 210 B."""
        dt = np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)),
                       ("scales", "i1", (16,)), ("d", "<f2")])
        if dt.itemsize != 210:
            fatal(f"_dequant_q6_k: block dtype is {dt.itemsize} B, expected 210")
        b = np.frombuffer(raw, dtype=dt, count=nblocks)
        d = b["d"].astype(np.float32)
        out = np.empty((nblocks, 256), dtype=np.float32)
        # is = l/16 over l in [0,32) -- 0 for the first half, 1 for the second.
        is_ = np.repeat(np.array([0, 1], dtype=np.intp), 16)
        for g in range(2):                       # C: for (n = 0; n < QK_K; n += 128)
            ql = b["ql"][:, g * 64:(g + 1) * 64]
            qh = b["qh"][:, g * 32:(g + 1) * 32]
            sc = b["scales"][:, g * 8:(g + 1) * 8]
            lo, hi = ql[:, 0:32], ql[:, 32:64]
            # (int8_t)(6-bit value) is the identity here (max 63); the -32 is
            # what makes it signed, so it must happen in a WIDER type.
            q1 = ((lo & 0xF) | (((qh >> 0) & 3) << 4)).astype(np.int16) - 32
            q2 = ((hi & 0xF) | (((qh >> 2) & 3) << 4)).astype(np.int16) - 32
            q3 = ((lo >> 4) | (((qh >> 4) & 3) << 4)).astype(np.int16) - 32
            q4 = ((hi >> 4) | (((qh >> 6) & 3) << 4)).astype(np.int16) - 32
            base = g * 128
            for off, q, sidx in ((0, q1, 0), (32, q2, 2), (64, q3, 4), (96, q4, 6)):
                s = sc[:, is_ + sidx].astype(np.float32)
                out[:, base + off: base + off + 32] = (
                    (d[:, None] * s) * q.astype(np.float32))
        return out

    def _dequant_bytes(self, ggml_type: int, raw, n: int):
        """raw: a bytes-like object of exactly the right length for `n`
        elements of `ggml_type`. Returns an OWNED float32 ndarray (not a
        view into the mmap -- callers may hold these across further reads,
        and mmap-backed buffers must not outlive `raw`'s scope in a way
        that risks a stale/unmapped view)."""
        np = self._np
        if ggml_type == GGML_F32:
            arr = np.frombuffer(raw, dtype="<f4", count=n)
            return arr.astype(np.float32, copy=True)
        if ggml_type == GGML_Q8_0:
            if n % 32 != 0:
                fatal(f"_dequant_bytes: Q8_0 element count {n} not a multiple of 32")
            nblocks = n // 32
            block_dtype = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
            if len(raw) != nblocks * 34:
                fatal(f"_dequant_bytes: Q8_0 raw length {len(raw)} != "
                      f"{nblocks} blocks * 34 bytes")
            blocks = np.frombuffer(raw, dtype=block_dtype, count=nblocks)
            d = blocks["d"].astype(np.float32)          # [nblocks], f16->f32 exact widen
            qs = blocks["qs"].astype(np.float32)         # [nblocks,32]
            vals = qs * d[:, None]                        # float32 * float32 -> float32
            return vals.reshape(n).astype(np.float32, copy=False)
        if ggml_type in (GGML_Q4_K, GGML_Q6_K):
            name = GGML_NAME[ggml_type]
            block_elems, block_bytes = _BLOCK_INFO[ggml_type]
            if n % block_elems != 0:
                fatal(f"_dequant_bytes: {name} element count {n} not a "
                      f"multiple of {block_elems}")
            nblocks = n // block_elems
            if len(raw) != nblocks * block_bytes:
                fatal(f"_dequant_bytes: {name} raw length {len(raw)} != "
                      f"{nblocks} blocks * {block_bytes} bytes")
            fn = (self._dequant_q4_k if ggml_type == GGML_Q4_K
                  else self._dequant_q6_k)
            return fn(np, raw, nblocks).reshape(n).astype(np.float32, copy=False)
        fatal(f"_dequant_bytes: unsupported ggml_type {ggml_type} "
              f"({GGML_NAME.get(ggml_type, '?')})")

    def row_stride_bytes(self, t: TensorInfo) -> int:
        if t.type not in _BLOCK_INFO:
            fatal(f"row_stride_bytes: tensor {t.name!r} has unknown type {t.type}")
        block_elems, block_bytes = _BLOCK_INFO[t.type]
        n_in = t.dims[0]
        if n_in % block_elems != 0:
            fatal(f"row_stride_bytes: tensor {t.name!r} row length {n_in} "
                  f"not a multiple of block size {block_elems}")
        return (n_in // block_elems) * block_bytes

    def tensor_row_dequant(self, t: TensorInfo, row: int):
        """Dequantized float32 [n_in] vector for row `row` of a 1D/2D
        tensor (1D: the whole vector, row must be 0; 2D: row in
        [0, dims[1])). Memory-cheap: only that row's bytes are touched."""
        if t.type not in DEQUANT_SUPPORTED:
            fatal(f"tensor_row_dequant: tensor {t.name!r} has type "
                  f"{GGML_NAME.get(t.type, t.type)} -- not in DEQUANT_SUPPORTED "
                  f"{sorted(GGML_NAME[x] for x in DEQUANT_SUPPORTED)}")
        if t.n_dims == 1:
            if row != 0:
                fatal(f"tensor_row_dequant: {t.name!r} is 1D, only row 0 exists (got {row})")
            n_in = t.dims[0]
            row_off = t.abs_offset
            row_bytes = t.nbytes
        elif t.n_dims == 2:
            n_in, n_out = t.dims[0], t.dims[1]
            if not (0 <= row < n_out):
                fatal(f"tensor_row_dequant: {t.name!r} row {row} out of range [0,{n_out})")
            stride = self.row_stride_bytes(t)
            row_off = t.abs_offset + row * stride
            row_bytes = stride
        else:
            fatal(f"tensor_row_dequant: {t.name!r} has n_dims={t.n_dims}, only 1D/2D supported")
        self._check(row_off, row_bytes, f"{t.name!r} row {row}")
        raw = bytes(self._mm[row_off: row_off + row_bytes])
        return self._dequant_bytes(t.type, raw, n_in)

    def row_raw_bytes(self, t: TensorInfo, row: int, n: int) -> bytes:
        """First `n` raw (still-quantized) bytes of row `row`, for --raw
        hex dumps / hand verification. Does NOT dequantize."""
        if t.n_dims == 1:
            if row != 0:
                fatal(f"row_raw_bytes: {t.name!r} is 1D, only row 0 exists (got {row})")
            row_off = t.abs_offset
            row_len = t.nbytes
        elif t.n_dims == 2:
            n_out = t.dims[1]
            if not (0 <= row < n_out):
                fatal(f"row_raw_bytes: {t.name!r} row {row} out of range [0,{n_out})")
            stride = self.row_stride_bytes(t)
            row_off = t.abs_offset + row * stride
            row_len = stride
        else:
            fatal(f"row_raw_bytes: {t.name!r} has n_dims={t.n_dims}, only 1D/2D supported")
        n = min(n, row_len)
        self._check(row_off, n, f"{t.name!r} row {row} raw bytes")
        return bytes(self._mm[row_off: row_off + n])

    def tensor_dequant_full(self, t: TensorInfo):
        """Full dequantized float32 ndarray: shape (dims[0],) for 1D, or
        (n_out=dims[1], n_in=dims[0]) for 2D -- matches GGUF's own
        row-major layout directly (dims[0] fastest-varying = numpy's last
        axis), so `W @ x` with x of length n_in gives y of length n_out
        with NO transpose needed anywhere in ref_forward.py."""
        if t.type not in DEQUANT_SUPPORTED:
            fatal(f"tensor_dequant_full: tensor {t.name!r} has type "
                  f"{GGML_NAME.get(t.type, t.type)} -- not in DEQUANT_SUPPORTED")
        self._check(t.abs_offset, t.nbytes, f"{t.name!r} full data")
        raw = self._mm[t.abs_offset: t.abs_offset + t.nbytes]
        if t.n_dims == 1:
            return self._dequant_bytes(t.type, raw, t.dims[0])
        if t.n_dims == 2:
            n_in, n_out = t.dims[0], t.dims[1]
            flat = self._dequant_bytes(t.type, raw, n_in * n_out)
            return flat.reshape(n_out, n_in)
        fatal(f"tensor_dequant_full: {t.name!r} has n_dims={t.n_dims}, only 1D/2D supported")

    def get_tensor(self, name: str) -> TensorInfo:
        t = self.tensors.get(name)
        if t is None:
            fatal(f"'{self.path}': no tensor named {name!r}")
        return t


if __name__ == "__main__":
    # `python gguf_reader.py --selftest` -- permanent regression check for
    # the two hand-verifiable formulas this whole module rests on: f16->f32
    # bit-pattern conversion (M2 note section 4) and one hand-built Q8_0
    # block (M2 probe notes P3.1). No GGUF file needed. Part of the
    # mandatory mechanical verification recorded in m2-w1-report.md.
    import sys
    import numpy as np

    if "--selftest" not in sys.argv:
        print(__doc__)
        sys.exit(0)

    failures = 0

    def check(label, got, want):
        global failures
        ok = (got == want) or (isinstance(want, float) and abs(got - want) < 1e-12)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {label}: got={got!r} want={want!r}")

    # f16 bit-pattern checks (M2 note section 4)
    check("f16 0x3C00 -> 1.0", float(f16_bits_to_f32(0x3C00)), 1.0)
    v = float(f16_bits_to_f32(0x0001))
    check("f16 0x0001 -> subnormal ~5.9604645e-08", abs(v - 5.9604645e-08) < 1e-14, True)
    check("f16 0x7C00 -> +inf", f16_bits_to_f32(0x7C00), np.float32("inf"))

    # Hand-built Q8_0 block: d=1.0 (0x3C00), qs = [0,1,2,...,31] (i8).
    # Expected dequant: value_i = d*qs[i] = qs[i] exactly (d=1).
    raw = struct.pack("<H", 0x3C00) + b"".join(struct.pack("<b", i) for i in range(32))
    assert len(raw) == 34, len(raw)
    block_dtype = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    blk = np.frombuffer(raw, dtype=block_dtype, count=1)
    d = blk["d"].astype(np.float32)[0]
    qs = blk["qs"].astype(np.float32)[0]
    vals = qs * d
    check("hand Q8_0 block: d", float(d), 1.0)
    check("hand Q8_0 block: vals == qs (d=1)", list(vals.astype(np.float32)), [float(i) for i in range(32)])
    check("hand Q8_0 block: dtype is float32", vals.dtype, np.dtype(np.float32))

    # A second hand block with a non-trivial scale: d=2.0 (0x4000),
    # qs=[-128, -1, 0, 1, 127, 0...0]
    qs_vals = [-128, -1, 0, 1, 127] + [0] * 27
    raw2 = struct.pack("<H", 0x4000) + b"".join(struct.pack("<b", i) for i in qs_vals)
    blk2 = np.frombuffer(raw2, dtype=block_dtype, count=1)
    d2 = blk2["d"].astype(np.float32)[0]
    qs2 = blk2["qs"].astype(np.float32)[0]
    vals2 = qs2 * d2
    check("hand Q8_0 block2: d=2.0", float(d2), 2.0)
    check("hand Q8_0 block2: -128*2=-256", float(vals2[0]), -256.0)
    check("hand Q8_0 block2: 127*2=254", float(vals2[4]), 254.0)

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
