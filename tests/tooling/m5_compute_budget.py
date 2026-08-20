#!/usr/bin/env python3
"""M5 COMPUTE budget -- the ceiling the traffic budget says it does not price.

WHY THIS EXISTS, AND WHAT IT CORRECTS.

`m5_traffic_budget.py` ends its ceiling table with the words "upper bounds:
traffic only, no compute, no scheduler". That caveat was honest and it was
never cashed. This script cashes it, and the answer changes which constraint
M5's bars are actually against:

  * A decode step at N rows moves ~2.5 GB of weights ONCE, so its traffic time
    grows slowly with N.
  * It does N x (2 x n_params) FLOPs, so its ARITHMETIC time grows LINEARLY
    with N and amortises not at all.

Two lines with different slopes cross. Past the crossing the traffic ceiling is
a number about a machine that is no longer the one being described -- exactly
the failure the 08-11 crossover finding recorded, one level up: there the model
did not know what was being READ, here it did not know what was being COMPUTED.

THE FOURTH INSTANCE OF THE SAME LESSON in this milestone. The crossover that
priced f16 KV read once; E2's slope, a comparable share of which is scalar
arithmetic and not traffic at all; `token_embd.weight`, 319 MB streamed per
row; and now the arithmetic term itself. A budget is only as good as its
inventory of what the machine does.

MEASURED INPUTS, with provenance, and they are the only ones:
  195 GFLOP/s  -- stoke's batched matvec at saturation, derived from E3's
                  measured ~41 ms per prefilled token at chunk 128
                  (2026-08-19 E3 prefill A/B; the flatness across 87/554/1500
                  is what says it is a ceiling and not a ramp).
  Effective rates for any other engine are passed with --gflops so a referent
  measured in the same sitting can be used instead of a remembered one.

Usage:
  python tests/tooling/m5_compute_budget.py [--ctx 704] [--gflops 195]
         [--referent-pp 91.3] [--model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf]
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

W_BYTES_PER_S = 43.6e9      # same basis as m5_traffic_budget.py, same sitting
STOKE_GFLOPS = 195.0        # E3, 2026-08-19
BARS = {8: (17.7, 10.8), 16: (14.2, 16.8)}


def forward_flops_per_token(f, arch, ctx):
    """FLOPs one token costs in one forward pass, counted from the FILE.

    Two terms, and the second is the one that grows with context:

      MATVEC. Every 2-D weight tensor is a matrix applied once per token, so it
      contributes 2 * dims[0] * dims[1] (one multiply and one add per element).
      Counting from the tensor table rather than from a parameter count means a
      tied output projection, a missing bias, or an unexpected tensor cannot be
      silently mis-priced -- and `token_embd.weight` IS in this sum, because
      since `15d5ab1` the output projection reads it every step.

      ATTENTION. Per layer, Q.K^T and A.V each cost n_head * head_dim * ctx
      multiply-adds, so 4 * n_head * head_dim * ctx FLOPs per layer per token.
      Small next to the matvecs at short context and not small at long.
    """
    n_layer = f.kv[f"{arch}.block_count"]
    n_head = f.kv[f"{arch}.attention.head_count"]
    head_dim = f.kv[f"{arch}.attention.key_length"]

    matvec = 0
    counted = []
    for name, t in f.tensors.items():
        if t.n_dims != 2:
            continue          # norms and 1-D scales: not matrix work
        matvec += 2 * t.dims[0] * t.dims[1]
        counted.append(name)
    attn = 4 * n_layer * n_head * head_dim * ctx
    return matvec, attn, len(counted)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    ap.add_argument("--ctx", type=int, default=704)
    ap.add_argument("--gflops", type=float, default=STOKE_GFLOPS,
                    help="stoke's measured effective batched-matvec rate")
    ap.add_argument("--referent-pp", type=float, default=None,
                    help="llama-bench pp512 tok/s from the SAME sitting, to "
                         "convert into an effective GFLOP/s for comparison")
    ap.add_argument("--stoke-pp", type=float, default=32.229,
                    help="stoke's own prefill tok/s from the same sitting. The "
                         "referent gap must be prefill-against-PREFILL: "
                         "comparing llama's prefill to stoke's DECODE published "
                         "3.96x where the like-for-like figure is 2.83x")
    a = ap.parse_args()

    f = GGUFFile(a.model)
    arch = f.kv["general.architecture"]
    n_layer = f.kv[f"{arch}.block_count"]
    n_head = f.kv[f"{arch}.attention.head_count"]
    n_head_kv = f.kv[f"{arch}.attention.head_count_kv"]
    head_dim = f.kv[f"{arch}.attention.key_length"]
    weights = os.path.getsize(a.model)

    matvec, attn, n_counted = forward_flops_per_token(f, arch, a.ctx)
    flops = matvec + attn

    print(f"geometry from {a.model}")
    print(f"  {n_counted} two-dimensional weight tensors, {weights:,} B on disk")
    print(f"  matvec    {matvec/1e9:8.3f} GFLOP/token")
    print(f"  attention {attn/1e9:8.3f} GFLOP/token  (at ctx {a.ctx})")
    print(f"  TOTAL     {flops/1e9:8.3f} GFLOP/token\n")

    print(f"stoke measured effective rate: {a.gflops:.0f} GFLOP/s")
    print(f"  => {flops/(a.gflops*1e9)*1e3:.1f} ms per token of arithmetic, "
          f"and it does NOT amortise across rows")
    print(f"  => a COMPUTE ceiling of {a.gflops*1e9/flops:.1f} tok/s aggregate\n")

    # ▶ THIS CEILING IS NOT FLAT IN N, AND SAYING SO COST A FALSIFIER.
    # The default 195 GFLOP/s comes from E3's prefill at 128-row chunks: a
    # FULLY SATURATED 16-wide kernel tile and a FULLY SATURATED 8-thread pool.
    # At small N neither is saturated, so the effective rate is a function of N
    # and this line is an ASYMPTOTE, not a constant. Measured 2026-08-20
    # (`gate-reports/2026-08-20-m5-flatness-report.md`, 4 counterbalanced
    # passes, ctx 192, on a thermally-plateaued box):
    #
    #     N        1     2     4     8    16
    #     GFLOP/s 120   127   152   167   172      <- rises 1.43x
    #     cores  5.80  6.26  6.39  6.60  6.70      <- and 1.155x of that is
    #                                                 just thread occupancy
    #
    # A prediction of "flat from N=2" was falsified by N=8 landing 32% over
    # N=2. What the data does support: the curve SATURATES, and it is already
    # saturated where the gate measures -- N=8 to N=16 buys 2.9% against the
    # 1.88x that traffic-only reasoning predicts for the same step.
    # >> THE CURVE WAS MEASURED AT ctx 192, SO ITS TRAFFIC COLUMN IS COMPUTED AT
    # ctx 192 -- NOT at --ctx. Pricing a measured rate against a traffic ceiling
    # from a different context length is a mixed basis, which is the error this
    # project's tooling exists to prevent.
    OCC_CTX = 192
    OCC = {1: 120.0, 2: 127.0, 4: 152.0, 8: 167.0, 16: 172.0}
    occ_flops = matvec + 4 * n_layer * n_head * head_dim * OCC_CTX
    kv_ctx_tok = n_layer * n_head_kv * (head_dim * 4) * 2
    print(f"OCCUPANCY-CORRECTED, from the measured curve (plateaued box, ctx {OCC_CTX}):")
    print("   N   measured GFLOP/s   tok/s measured   traffic ceiling   ratio")
    for N in (1, 2, 4, 8, 16):
        g = OCC[N]
        t_traffic = (weights + N * OCC_CTX * kv_ctx_tok) / W_BYTES_PER_S
        meas, traf = g * 1e9 / occ_flops, N / t_traffic
        print(f"  {N:2d}   {g:14.0f}   {meas:14.1f}   {traf:15.1f}   {traf/meas:5.1f}x")
    lo, hi = OCC[1] / occ_flops, OCC[16] / occ_flops
    t_lo = 1 / ((weights + 1 * OCC_CTX * kv_ctx_tok) / W_BYTES_PER_S)
    t_hi = 16 / ((weights + 16 * OCC_CTX * kv_ctx_tok) / W_BYTES_PER_S)
    print(f"  ^ across N=1..16 the MEASURED gain is {hi/lo:.2f}x; traffic-only")
    print(f"    reasoning predicts {t_hi/t_lo:.1f}x. The curve saturates, and it is")
    print("    already saturated where the gate measures: N=8 -> N=16 buys 2.9%.")
    print("    The levels are not clock-corrected -- read the SHAPES.\n")

    if a.referent_pp:
        ref = a.referent_pp * flops / 1e9
        mine = a.stoke_pp * flops / 1e9
        print(f"referent, PREFILL against PREFILL, same sitting:")
        print(f"  llama-bench pp512 {a.referent_pp:6.2f} tok/s => {ref:5.0f} GFLOP/s")
        print(f"  stoke prefill     {a.stoke_pp:6.2f} tok/s => {mine:5.0f} GFLOP/s")
        print(f"  gap {ref/mine:.2f}x   (uniform speedup needed for the 17.7 bar: 2.75x)\n")

    # KV read per context token, kv-head-outer f32 -- post-E2, the shipping path
    kv_per_ctx_tok = n_layer * n_head_kv * (head_dim * 4) * 2

    print(f"WHICH CEILING BINDS, at ctx {a.ctx}  (f32 KV, post-E2)")
    print("   N   traffic ms   compute ms   binds     ceiling tok/s   bar   verdict")
    for N in (1, 2, 4, 8, 16):
        t_traffic = (weights + N * a.ctx * kv_per_ctx_tok) / W_BYTES_PER_S * 1e3
        t_compute = N * flops / (a.gflops * 1e9) * 1e3
        binds = "COMPUTE" if t_compute > t_traffic else "traffic"
        step = max(t_traffic, t_compute)
        ceil = N / (step / 1e3)
        bar = max(BARS[N]) if N in BARS else None
        if bar:
            verdict = f"{bar:5.1f}  {'clears' if ceil >= bar else 'BELOW THE BAR'}"
        else:
            verdict = "    -  -"
        print(f"  {N:2d}   {t_traffic:9.1f}   {t_compute:10.1f}   {binds:8s}  "
              f"{ceil:12.1f}   {verdict}")

    # The crossing, solved rather than eyeballed.
    #   weights/W + N*ctx*kv/W = N*flops/G   =>   N = (weights/W) / (flops/G - ctx*kv/W)
    lhs = weights / W_BYTES_PER_S
    per_row = flops / (a.gflops * 1e9) - a.ctx * kv_per_ctx_tok / W_BYTES_PER_S
    if per_row > 0:
        print(f"\n  traffic and compute cross at N = {lhs/per_row:.2f} "
              f"-- above that, arithmetic is the wall")
    else:
        print("\n  compute never overtakes traffic at this ctx and rate")

    print("\nWHAT THIS DOES NOT SAY. It does not say E2 was wrong: E2's measured")
    print("1.12x at ctx 1500 is a real, verified gain on the term it targets.")
    print("It says the AGGREGATE ceiling those bars were checked against was")
    print("the wrong ceiling from N=2 up, so the slack it reported was not")
    print("there to spend.")


if __name__ == "__main__":
    main()
