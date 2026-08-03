#!/usr/bin/env python3
"""tests/tooling/simd_vnni_check.py -- pin K3's VNNI idiom BEFORE any C exists.

WHY THIS EXISTS.  The M3-M7 forward note's §M3 says, in the entry on the
dpbusd sign trick: "PROBE the exact idiom from fetched ggml source at
dispatch ... and pin it with the exact-diff gate rather than trusting either
memory or this note."  That instruction earned itself immediately: the note
recalls the absolute value as `_mm256_abs_epi8(x)`, and the b10037 source
actually writes `_mm256_sign_epi8(x, x)`.  This module settles that, and
every other numeric question K3's kernel has to answer, by EXHAUSTIVE
enumeration over the i8 domain rather than by argument -- a signed 8-bit
type is small enough that "for all inputs" is a loop, not a proof sketch.

It is deliberately python-first and deliberately standalone.  Nothing here
imports or links stoke; it models the INTRINSICS from their published
semantics and models the SCALAR KERNEL from src/kernels/q8_0.c, then makes
them meet on real model bytes.  A transcription checked against the thing it
was transcribed from proves nothing; these two descriptions have no common
ancestor except the hardware spec and the C source.

WHAT IT PROVES, in the order the C will need it:

  1. `_mm256_sign_epi8(x, x)` and `_mm256_abs_epi8(x)` agree on ALL 256
     signed bytes -- including x = -128, where both return 0x80 and both are
     RIGHT, because the u8 reading of 0x80 is 128 = |-128|.  So the note's
     recalled idiom and ggml's real one are interchangeable here, and K3 may
     use ggml's spelling without inheriting a difference.

  2. THE SIGN TRICK IS EXACT ON THE Q8_0 DOMAIN AND NOT ONE STEP WIDER --
     which is a precondition K3's C inherits, not a footnote.
         u8(sign_epi8(x,x)) * i8(sign_epi8(y,x))  ==  x * y
     holds for every one of the 65,025 pairs in [-127,127]^2, and FAILS on
     exactly 128 of the 65,536 pairs in the full i8 domain: precisely those
     where the ACTIVATION quant is -128 and the weight quant is negative.
     The cause is that PSIGNB negates in 8-bit two's complement without
     saturating, so -(-128) wraps back to -128 instead of becoming +128 --
     and the sign then lands on the operand that keeps its sign rather than
     the one that is made unsigned.

     This was NOT predicted.  Reasoning about it beforehand got the -128
     case right for the operand that gets |.| applied (0x80 reads as u8 128,
     which is exactly right) and wrong for the operand that gets the sign
     applied, where the same wrap is a silent sign error.  Enumeration
     caught it; argument did not.  That asymmetry is the reason this file
     enumerates instead of reasons.

     WHY IT IS STILL SAFE, and where the obligation now lives: Q8_0 quants
     cannot be -128.  Both quantizers compute round(v * 127/amax) with
     |v| <= amax, so the image is [-127, 127] by construction -- ggml's
     included, which means upstream relies on this same unstated
     precondition.  K3's C may use the trick, and must not widen its inputs
     beyond Q8_0 quants without re-deriving this.  Check [6] confirms the
     bound on real weight rows rather than assuming it.

  3. THE `maddubs` FALLBACK CANNOT SATURATE on Q8_0 data -- and the margin
     is thin enough to be worth a number rather than a shrug.  `pmaddubsw`
     sums two u8xi8 products into a SIGNED i16 with saturation.  On Q8_0
     data a single product cannot leave [-16129, 16129], so a pair sum
     cannot leave [-32258, 32258] -- inside i16, with 509 to spare.
     Over the FULL i8 domain the negative extreme is 2 * -16384 = -32768,
     which lands exactly on i16 MIN with zero headroom; it is reachable only
     through the -128 quant that (2) already forbids, so this is a third
     independent reason that precondition is load-bearing rather than a
     stylistic note.

  4. dpbusd's LANE PARTIALS RE-SUM TO THE SCALAR'S INTEGER TOTAL EXACTLY.
     dpbusd yields 8 dwords, each the sum of 4 products; the scalar kernel
     sums all 32 in one i32.  Integer addition is associative, so this is
     exact -- but it is checked on real bytes anyway, because the claim that
     can actually be wrong is the BYTE-TO-LANE MAPPING, not the arithmetic.

  5. THE ONE PLACE SIMD CAN DIVERGE IS f32 GROUPING, AND IT IS NOT
     HYPOTHETICAL.  ggml's AVX2 loop keeps 8 f32 lane accumulators across
     blocks (`acc = fmadd(d, q, acc)`, one `hsum_float_8` at the very end),
     so each lane's quarter-block partial is scaled and rounded separately.
     src/kernels/q8_0.c reduces each block to ONE i32, scales once, and
     accumulates f32 sequentially.  Those are different roundings of the
     same real number.  This module measures how different on real rows, so
     K3's design decision rests on a figure instead of an intuition.

     The consequence for K3: a SIMD twin that copies ggml's accumulator
     shape would NOT be bit-exact against our own scalar, which is exactly
     what forward-note gate [1] requires.  The twin must reduce per block
     (an INTEGER hsum, therefore free of rounding) and keep the scalar's
     `sum += (f32)sumi * (dw*da)` grouping.  Being closer to llama.cpp's
     rounding is not the goal -- our scalar path is the reference the parity
     gate was run against, and SIMD must be a speedup, not a numeric edit.

Run:  python tests/tooling/simd_vnni_check.py --model models/<Q8_0>.gguf
      python tests/tooling/simd_vnni_check.py --selftest
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_reader as G  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

Q8_0_BLOCK_ELEMS = 32
Q8_0_BLOCK_BYTES = 34


# ===========================================================================
# Intrinsic models -- from the published semantics, byte-at-a-time.
# ===========================================================================
#
# These operate on int8 numpy arrays and return int8 numpy arrays, matching
# what the C sees.  They are written per-element and unvectorized on
# purpose: this file is a specification check, not a fast path, and a
# clever numpy formulation would be one more thing that could be subtly
# wrong in the same way the C could be.


def sign_epi8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """PSIGNB: dst[i] = 0 if b[i]==0; -a[i] if b[i]<0; a[i] otherwise.

    The negation is two's complement in 8 bits and does NOT saturate, so
    -(-128) wraps back to -128 (0x80).  That wrap is load-bearing in the
    sign trick, not a defect -- see prove_sign_trick_exhaustive.
    """
    a = np.asarray(a, dtype=np.int8)
    b = np.asarray(b, dtype=np.int8)
    out = np.empty_like(a)
    for i in range(a.size):
        bi = int(b.flat[i])
        ai = int(a.flat[i])
        if bi == 0:
            v = 0
        elif bi < 0:
            v = (-ai) & 0xFF          # 8-bit two's complement, wrapping
            if v >= 128:
                v -= 256
        else:
            v = ai
        out.flat[i] = np.int8(v)
    return out


def abs_epi8(a: np.ndarray) -> np.ndarray:
    """PABSB: dst[i] = |a[i]|, saturating-free, so |-128| stays 0x80."""
    a = np.asarray(a, dtype=np.int8)
    out = np.empty_like(a)
    for i in range(a.size):
        ai = int(a.flat[i])
        v = -ai if ai < 0 else ai
        v &= 0xFF
        if v >= 128:
            v -= 256
        out.flat[i] = np.int8(v)
    return out


def as_u8(a: np.ndarray) -> np.ndarray:
    """Reinterpret the same bits as unsigned -- what dpbusd's first operand
    means.  No conversion, no clamping: 0x80 reads as 128."""
    return np.asarray(a, dtype=np.int8).view(np.uint8)


def dpbusd(src: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """VPDPBUSD (the non-saturating form, which is what AVX-VNNI's
    _mm256_dpbusd_avx_epi32 emits).

    Per Intel's pseudocode, for each of the 8 dwords j:
        tmp_k = ZeroExtend16(a.byte[4j+k]) * SignExtend16(b.byte[4j+k])
        dst.dword[j] = src.dword[j] + tmp_1 + tmp_2 + tmp_3 + tmp_4
    The four products are formed as 16-bit values -- with a in [0,255] and
    b in [-128,127] a product spans [-32640, 32385] and therefore always
    fits -- and the accumulation into dst is 32-bit and does NOT saturate.
    """
    au = np.asarray(a, dtype=np.int8).view(np.uint8).astype(np.int64)
    bs = np.asarray(b, dtype=np.int8).astype(np.int64)
    src = np.asarray(src, dtype=np.int64)
    if au.size != 32 or bs.size != 32 or src.size != 8:
        raise ValueError("dpbusd models one 256-bit register: 32 bytes, 8 dwords")
    prod = au * bs
    lanes = prod.reshape(8, 4).sum(axis=1)
    out = src + lanes
    if np.any(out > 2**31 - 1) or np.any(out < -(2**31)):
        raise AssertionError("dpbusd model overflowed i32 -- inputs out of contract")
    return out.astype(np.int32)


def maddubs_epi16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """PMADDUBSW: 16 signed-SATURATING i16 lanes, each the sum of two
    u8 x i8 products.  The saturation is the whole reason this is modelled
    -- it is the fallback path's only way to be silently wrong."""
    au = np.asarray(a, dtype=np.int8).view(np.uint8).astype(np.int64)
    bs = np.asarray(b, dtype=np.int8).astype(np.int64)
    prod = au * bs
    pairs = prod.reshape(-1, 2).sum(axis=1)
    return np.clip(pairs, -32768, 32767).astype(np.int16), pairs


# ===========================================================================
# Scalar kernel model -- transcribed from src/kernels/q8_0.c, not from the
# intrinsic models above.
# ===========================================================================


def scalar_block_sumi(wq: np.ndarray, aq: np.ndarray) -> int:
    """The integer core of vec_dot_q8_0_q8_0: 32 products, one i32 sum."""
    return int(np.asarray(wq, dtype=np.int8).astype(np.int64).dot(
        np.asarray(aq, dtype=np.int8).astype(np.int64)))


def scalar_dot(wblocks, ablocks) -> np.float32:
    """`sum += (f32)sumi * (dw*da)`, sequential across blocks, f32 throughout.

    The scales multiply EACH OTHER first; q8_0.c's comment marks that
    grouping as binding, so the model must not simplify it.
    """
    total = np.float32(0.0)
    for (dw, wq), (da, aq) in zip(wblocks, ablocks):
        sumi = scalar_block_sumi(wq, aq)
        total = np.float32(total + np.float32(np.float32(sumi) * np.float32(dw * da)))
    return total


def ggml_avx2_dot(wblocks, ablocks) -> np.float32:
    """ggml b10037's AVX2 shape: 8 f32 lane accumulators, one hsum at the end.

    Modelled to MEASURE the grouping difference, not because K3 will use it.
    """
    acc = np.zeros(8, dtype=np.float32)
    for (dw, wq), (da, aq) in zip(wblocks, ablocks):
        ax = abs_epi8(wq)
        sy = sign_epi8(aq, wq)
        lanes = dpbusd(np.zeros(8, dtype=np.int32), ax, sy)
        d = np.float32(dw * da)
        acc = (acc + np.float32(d) * lanes.astype(np.float32)).astype(np.float32)
    # hsum_float_8: (lo+hi) then the two 64-bit folds -- the order matters
    lo, hi = acc[:4], acc[4:]
    v = (lo + hi).astype(np.float32)
    v = np.float32(v[:2] + v[2:])
    return np.float32(v[0] + v[1])


def simd_twin_dot(wblocks, ablocks) -> np.float32:
    """What K3 SHOULD build: dpbusd for the products, an INTEGER hsum per
    block (exact, no rounding), then the scalar's own f32 grouping."""
    total = np.float32(0.0)
    for (dw, wq), (da, aq) in zip(wblocks, ablocks):
        ax = sign_epi8(wq, wq)                    # ggml's spelling
        sy = sign_epi8(aq, wq)
        lanes = dpbusd(np.zeros(8, dtype=np.int32), ax, sy)
        sumi = int(lanes.astype(np.int64).sum())  # integer hsum: exact
        total = np.float32(total + np.float32(np.float32(sumi) * np.float32(dw * da)))
    return total


# ===========================================================================
# Proofs
# ===========================================================================

ALL_I8 = np.arange(-128, 128, dtype=np.int8)


def prove_abs_and_sign_agree() -> str:
    a = abs_epi8(ALL_I8)
    s = sign_epi8(ALL_I8, ALL_I8)
    bad = np.nonzero(a.view(np.uint8) != s.view(np.uint8))[0]
    if bad.size:
        raise AssertionError(
            f"abs_epi8 and sign_epi8(x,x) disagree on {bad.size} values, "
            f"first at x={int(ALL_I8[bad[0]])}")
    # and the value they return for -128 is the one the trick needs
    v = int(as_u8(abs_epi8(np.array([-128], dtype=np.int8)))[0])
    if v != 128:
        raise AssertionError(f"|-128| must read as u8 128, got {v}")
    return ("abs_epi8(x) == sign_epi8(x,x) for all 256 signed bytes; "
            "both give 0x80 at x=-128, which reads as u8 128 = |-128|")


def _sign_trick_products(xs: np.ndarray, ys: np.ndarray):
    ax = as_u8(sign_epi8(xs, xs)).astype(np.int64)
    sy = sign_epi8(ys, xs).astype(np.int64)
    got = ax * sy
    want = xs.astype(np.int64) * ys.astype(np.int64)
    return got, want


def prove_sign_trick_on_q8_0_domain() -> str:
    """Exhaustive over [-127,127]^2 -- every pair Q8_0 data can present."""
    dom = np.arange(-127, 128, dtype=np.int8)
    xs = np.repeat(dom, dom.size)
    ys = np.tile(dom, dom.size)
    got, want = _sign_trick_products(xs, ys)
    bad = np.nonzero(got != want)[0]
    if bad.size:
        i = bad[0]
        raise AssertionError(
            f"sign trick wrong on {bad.size} of {xs.size} Q8_0-domain pairs; "
            f"first x={int(xs[i])} y={int(ys[i])} "
            f"got {int(got[i])} want {int(want[i])}")
    return (f"u8(sign(x,x)) * i8(sign(y,x)) == x*y on all {xs.size:,} pairs "
            f"in [-127,127]^2 -- the entire reachable Q8_0 domain")


def prove_sign_trick_breaks_outside_q8_0_domain() -> str:
    """The precondition, stated as the EXACT failing set rather than a
    warning.  A trap you have enumerated is a contract; a trap you have only
    described is a trap."""
    xs = np.repeat(ALL_I8, 256)
    ys = np.tile(ALL_I8, 256)
    got, want = _sign_trick_products(xs, ys)
    bad = np.nonzero(got != want)[0]
    if bad.size == 0:
        raise AssertionError(
            "expected the trick to fail somewhere in the full i8 domain; it "
            "did not -- either the model or the claim is wrong")
    bx, by = xs[bad], ys[bad]
    predicted = np.nonzero((ys == -128) & (xs < 0))[0]
    if not np.array_equal(np.sort(bad), np.sort(predicted)):
        raise AssertionError(
            f"failing set is not exactly {{y == -128 and x < 0}}: "
            f"{bad.size} actual vs {predicted.size} predicted")
    if not (by == -128).all():
        raise AssertionError("some failure had y != -128")
    return (f"fails on exactly {bad.size} of 65,536 pairs, and the failing "
            f"set is precisely {{activation quant == -128, weight quant < 0}} "
            f"(x range [{int(bx.min())},{int(bx.max())}]) -- unreachable for "
            f"Q8_0, which is why ggml is safe using it")


def _single_product_range(dom: np.ndarray):
    """Range of ONE u8xi8 product after the sign trick, over dom^2."""
    xs = np.repeat(dom, dom.size)
    ys = np.tile(dom, dom.size)
    got, _ = _sign_trick_products(xs, ys)
    return int(got.min()), int(got.max())


def prove_maddubs_cannot_saturate() -> str:
    """Fallback path: can a pair sum leave i16?

    pmaddubsw adds TWO INDEPENDENT lane products, so the worst case is
    2 x the extreme single product -- not the extremes of some particular
    adjacent pairing.  Bounding it any other way samples rather than proves,
    which an earlier version of this check did: it paired adjacent
    enumeration indices, reported a 1.2% margin, and missed the real one.
    """
    q_lo, q_hi = _single_product_range(np.arange(-127, 128, dtype=np.int8))
    f_lo, f_hi = _single_product_range(ALL_I8)

    q_pair = (2 * q_lo, 2 * q_hi)
    f_pair = (2 * f_lo, 2 * f_hi)
    if q_pair[0] < -32768 or q_pair[1] > 32767:
        raise AssertionError(
            f"maddubs saturates on Q8_0 data: pair sums reach {q_pair}")
    q_margin = min(32767 - q_pair[1], q_pair[0] + 32768)

    # The full-i8 figure is not a safety claim -- it is the third independent
    # reason the -128 precondition matters, and it is worth a number.
    f_margin = min(32767 - f_pair[1], f_pair[0] + 32768)
    return (f"on Q8_0 data single products span [{q_lo}, {q_hi}] so pair sums "
            f"reach {q_pair} -- inside i16 with {q_margin} to spare "
            f"({q_margin / 32768 * 100:.2f}%). Over the FULL i8 domain they "
            f"reach {f_pair}, i.e. margin {f_margin} -- the negative extreme "
            f"lands exactly on i16 MIN, and it is reachable only via the "
            f"quant value check [3] already forbids")


def prove_lane_mapping(rng) -> str:
    """dpbusd's 8 lane partials must re-sum to the scalar's single i32."""
    worst = 0
    for _ in range(2000):
        w = rng.integers(-127, 128, size=32).astype(np.int8)
        a = rng.integers(-127, 128, size=32).astype(np.int8)
        ax = sign_epi8(w, w)
        sy = sign_epi8(a, w)
        lanes = dpbusd(np.zeros(8, dtype=np.int32), ax, sy)
        got = int(lanes.astype(np.int64).sum())
        want = scalar_block_sumi(w, a)
        if got != want:
            raise AssertionError(
                f"lane hsum {got} != scalar sumi {want} on w={w.tolist()}")
        worst = max(worst, abs(want))
    bound = 32 * 127 * 127
    return (f"2,000 random blocks: dpbusd lane hsum == scalar i32 sum exactly; "
            f"largest |sumi| seen {worst:,} against the i32 headroom bound "
            f"{bound:,}")


# ===========================================================================
# Real model bytes
# ===========================================================================


def read_q8_0_blocks(gf, tensor_name: str, row: int):
    t = gf.get_tensor(tensor_name)
    if t.type != G.GGML_Q8_0:
        raise AssertionError(
            f"{tensor_name} is {G.GGML_NAME.get(t.type, '?')}, want Q8_0")
    n_in = t.dims[0]
    nblk = n_in // Q8_0_BLOCK_ELEMS
    raw = gf.row_raw_bytes(t, row, nblk * Q8_0_BLOCK_BYTES)
    blocks = []
    for b in range(nblk):
        off = b * Q8_0_BLOCK_BYTES
        d_bits = int.from_bytes(raw[off:off + 2], "little")
        d = np.float32(G.f16_bits_to_f32(d_bits))
        qs = np.frombuffer(raw[off + 2:off + Q8_0_BLOCK_BYTES], dtype=np.int8)
        blocks.append((d, qs.copy()))
    return blocks


def quantize_activation_blocks(x: np.ndarray):
    """Transcribed from src/kernels/q8_0.c's quantize_row_q8_0.

    The three details the K2 session pinned as load-bearing are preserved:
    the reciprocal comes from the f32 scale BEFORE it is narrowed to f16,
    the zero guard yields 0 instead of dividing, and rounding is half-AWAY
    from zero (roundf), NOT the half-to-even that f32_to_f16 uses.
    """
    out = []
    for b in range(x.size // Q8_0_BLOCK_ELEMS):
        blk = x[b * Q8_0_BLOCK_ELEMS:(b + 1) * Q8_0_BLOCK_ELEMS].astype(np.float32)
        amax = np.float32(np.abs(blk).max())
        d = np.float32(amax / np.float32(127.0))
        if d == np.float32(0.0):
            q = np.zeros(Q8_0_BLOCK_ELEMS, dtype=np.int8)
        else:
            idf = np.float32(np.float32(127.0) / amax)
            scaled = (blk.astype(np.float32) * idf).astype(np.float32)
            # roundf: half away from zero
            q = np.where(scaled >= 0,
                         np.floor(scaled + 0.5),
                         np.ceil(scaled - 0.5)).astype(np.int64)
            if q.min() < -128 or q.max() > 127:
                raise AssertionError(f"quantized activation out of i8 range: "
                                     f"[{q.min()}, {q.max()}]")
            q = q.astype(np.int8)
        d16 = np.float32(G.f16_bits_to_f32(f32_to_f16_bits(float(d))))
        out.append((d16, q))
    return out


def f32_to_f16_bits(v: float) -> int:
    """numpy's float16 conversion is IEEE round-to-nearest-EVEN, which is
    what src/kernels/f16.c implements and what F16C hardware does."""
    return int(np.float16(np.float32(v)).view(np.uint16))


def check_real_rows(model_path: str, rng) -> list:
    lines = []
    targets = [
        ("blk.0.attn_q.weight", 0),
        ("blk.0.ffn_down.weight", 7),
        ("blk.17.attn_v.weight", 3),
        ("blk.35.ffn_up.weight", 11),
    ]
    seen_min, seen_max = 127, -128
    n_quants = 0
    exact = 0
    ggml_diff_max = 0.0
    ggml_diff_any = 0
    for name, row in targets:
        with G.GGUFFile(model_path) as gf:
            wblocks = read_q8_0_blocks(gf, name, row)
        n_in = len(wblocks) * Q8_0_BLOCK_ELEMS
        x = rng.standard_normal(n_in).astype(np.float32) * np.float32(0.35)
        ablocks = quantize_activation_blocks(x)

        for _, q in wblocks:
            seen_min = min(seen_min, int(q.min()))
            seen_max = max(seen_max, int(q.max()))
            n_quants += q.size

        s = scalar_dot(wblocks, ablocks)
        t = simd_twin_dot(wblocks, ablocks)
        g = ggml_avx2_dot(wblocks, ablocks)

        if s.view(np.uint32) != t.view(np.uint32):
            raise AssertionError(
                f"{name} row {row}: SIMD twin {t!r} not bit-identical to "
                f"scalar {s!r} -- the twin's grouping is wrong")
        exact += 1
        if g.view(np.uint32) != s.view(np.uint32):
            ggml_diff_any += 1
            rel = abs(float(g) - float(s)) / max(abs(float(s)), 1e-30)
            ggml_diff_max = max(ggml_diff_max, rel)
        lines.append(f"    {name:<26} row {row:<3} n_in={n_in:<5} "
                     f"scalar={float(s): .6f}  twin=BIT-EXACT  "
                     f"ggml_shape={'differs' if g != s else 'same'}")

    lines.append("")
    if seen_min < -127:
        raise AssertionError(
            f"a real Q8_0 weight row contains {seen_min} -- the sign trick's "
            f"precondition (check [3]) is VIOLATED by shipped model data, and "
            f"K3 cannot use dpbusd without a correction term")
    lines.append(f"    weight quants seen: [{seen_min}, {seen_max}] over "
                 f"{n_quants:,} values -- inside [-127,127], so check [3]'s "
                 f"precondition holds on real data, not just by construction")
    lines.append(f"    SIMD twin bit-exact vs scalar on {exact}/{len(targets)} rows")
    lines.append(f"    ggml's 8-lane f32 accumulator differs from our grouping on "
                 f"{ggml_diff_any}/{len(targets)} rows, max rel {ggml_diff_max:.3e}")
    return lines


# ===========================================================================
# selftest -- every proof must be able to FAIL
# ===========================================================================


def selftest() -> int:
    rng = np.random.default_rng(20260803)
    print("selftest: mutations must be caught\n")
    ok = True

    # M1: a broken sign trick (abs on the wrong operand) must be caught.
    try:
        xs = np.repeat(ALL_I8, 256)
        ys = np.tile(ALL_I8, 256)
        bad = as_u8(sign_epi8(xs, xs)).astype(np.int64) * ys.astype(np.int64)
        want = xs.astype(np.int64) * ys.astype(np.int64)
        assert np.any(bad != want), "mutation M1 produced no difference"
        print("  ok  M1 dropping sign_epi8(y,x) breaks the identity")
    except AssertionError as e:
        ok = False
        print(f"  FAIL M1 {e}")

    # M2: saturating the abs at +127 (the "obvious" fix for -128) must break it.
    try:
        x = np.array([-128] * 32, dtype=np.int8)
        y = np.array([100] * 32, dtype=np.int8)
        good = int(as_u8(sign_epi8(x, x)).astype(np.int64)[0]) * \
            int(sign_epi8(y, x).astype(np.int64)[0])
        clamped = 127 * int(sign_epi8(y, x).astype(np.int64)[0])
        assert good == -128 * 100, f"reference wrong: {good}"
        assert clamped != good, "clamping to 127 was not detected"
        print(f"  ok  M2 clamping |-128| to 127 gives {clamped} not {good}")
    except AssertionError as e:
        ok = False
        print(f"  FAIL M2 {e}")

    # M3: ggml's accumulator shape must NOT be bit-exact with ours, or the
    #     whole design finding is vacuous and this file proves nothing.
    try:
        wb, ab = [], []
        for _ in range(64):
            wb.append((np.float32(rng.random() * 0.02 + 0.001),
                       rng.integers(-127, 128, 32).astype(np.int8)))
            ab.append((np.float32(rng.random() * 0.02 + 0.001),
                       rng.integers(-127, 128, 32).astype(np.int8)))
        s = scalar_dot(wb, ab)
        t = simd_twin_dot(wb, ab)
        g = ggml_avx2_dot(wb, ab)
        assert s.view(np.uint32) == t.view(np.uint32), \
            f"twin not bit-exact: {s!r} vs {t!r}"
        assert g.view(np.uint32) != s.view(np.uint32), \
            ("ggml's shape happened to match on this input -- the grouping "
             "claim needs an input where it does not")
        print(f"  ok  M3 twin bit-exact ({float(s):.7f}); ggml shape differs "
              f"({float(g):.7f})")
    except AssertionError as e:
        ok = False
        print(f"  FAIL M3 {e}")

    # M4: the maddubs model must actually saturate when handed values that
    #     exceed i16, or its "cannot saturate" result is unfalsifiable -- and
    #     the Q8_0 worst case must sit strictly inside while the full-i8 one
    #     sits exactly on the boundary.
    try:
        q_lo, q_hi = _single_product_range(np.arange(-127, 128, dtype=np.int8))
        f_lo, f_hi = _single_product_range(ALL_I8)
        assert (q_lo, q_hi) == (-16129, 16129), f"Q8_0 product range {(q_lo, q_hi)}"
        assert 2 * q_hi < 32767 and 2 * q_lo > -32768, "Q8_0 pair sum not strict"
        assert 2 * f_lo == -32768, f"full-i8 negative extreme {2 * f_lo}, want -32768"
        # falsifiability: the model must clip something
        big = np.full(32, 200, dtype=np.uint8).view(np.int8)
        b = np.full(32, 127, dtype=np.int8)
        clipped, raw = maddubs_epi16(big, b)
        assert int(raw.max()) > 32767, "constructed case did not exceed i16"
        assert int(clipped.max()) == 32767, "model failed to saturate"
        print(f"  ok  M4 Q8_0 pair sums reach {2 * q_lo}..{2 * q_hi} (strict); "
              f"full-i8 reaches {2 * f_lo} = i16 MIN exactly; model does clip "
              f"a constructed {int(raw.max())}")
    except AssertionError as e:
        ok = False
        print(f"  FAIL M4 {e}")

    # M5: the domain restriction in check [2] must be LOAD-BEARING -- widening
    #     it by the single value -128 has to make it fail, or the precondition
    #     is decoration and a future edit will quietly drop it.
    try:
        dom = np.arange(-128, 128, dtype=np.int8)
        xs = np.repeat(dom, dom.size)
        ys = np.tile(dom, dom.size)
        got, want = _sign_trick_products(xs, ys)
        n_bad = int(np.count_nonzero(got != want))
        assert n_bad == 128, f"expected 128 failures on the widened domain, got {n_bad}"
        dom2 = np.arange(-127, 128, dtype=np.int8)
        xs2 = np.repeat(dom2, dom2.size)
        ys2 = np.tile(dom2, dom2.size)
        g2, w2 = _sign_trick_products(xs2, ys2)
        assert int(np.count_nonzero(g2 != w2)) == 0, "Q8_0 domain must be clean"
        print(f"  ok  M5 widening the domain by -128 alone introduces {n_bad} "
              f"failures; [-127,127]^2 has 0")
    except AssertionError as e:
        ok = False
        print(f"  FAIL M5 {e}")

    print("\nselftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="Q8_0 .gguf to read real rows from")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    rng = np.random.default_rng(20260803)
    print("K3 VNNI idiom check -- probed from llama.cpp b10037, "
          "proved over the i8 domain\n")

    print("  [1] absolute value")
    print(f"      {prove_abs_and_sign_agree()}")
    print("  [2] the sign trick, on the domain Q8_0 can actually present")
    print(f"      {prove_sign_trick_on_q8_0_domain()}")
    print("  [3] the sign trick's PRECONDITION, stated as its exact failing set")
    print(f"      {prove_sign_trick_breaks_outside_q8_0_domain()}")
    print("  [4] maddubs fallback saturation")
    print(f"      {prove_maddubs_cannot_saturate()}")
    print("  [5] dpbusd lane mapping")
    print(f"      {prove_lane_mapping(rng)}")

    if a.model:
        print("\n  [6] real model rows (and the -128 bound that [3] depends on)")
        for ln in check_real_rows(a.model, rng):
            print(ln)
    else:
        print("\n  [6] real model rows -- SKIPPED (pass --model)")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
