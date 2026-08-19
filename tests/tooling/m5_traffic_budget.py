#!/usr/bin/env python3
"""M5 traffic budget -- regenerates every derived number in the M5 entry note.

WHY THIS IS A SCRIPT AND NOT A TABLE IN A NOTE. This project has recorded the
same defect three times in two days: a quantity CARRIED instead of RECOMPUTED
(two gain percentages written from estimate, a KB/KiB conflation, a "~1-in-5,000"
that was wrong by 9x -- and that last one made its own argument STRONGER, which
is exactly when an error is least likely to be caught). The M5 entry note and the
ROADMAP's M5 row now carry a dozen derived figures. This regenerates all of them
from two inputs -- the GGUF's own geometry and the measured curve -- so the next
session can re-derive rather than trust, and a geometry change cannot leave a
stale number sitting in prose.

GEOMETRY IS READ FROM THE MODEL FILE, NOT TYPED. Typing 32 and 8 here would
reproduce the very coupling the script exists to remove: the amplification factor
IS n_head/n_head_kv, so a hand-entered head count would let a wrong ratio look
self-consistent.

MEASURED INPUTS, and they are the only two:
  106.75 ms/token at ctx 1500 and slope 0.0289 ms/context-token -- L4's 4-arm
  A/B (gate-reports/2026-08-18-m45-l4-measured.md), arm B/L, t=6 pinned.
  W = 43.6 GB/s -- Q4_K_M at 6T, re-measured 2026-08-11 under the quiesced
  protocol (the file tops out at 44.8 at 10T; 43.6 is the configuration the
  measurements were taken in, and mixing the two would be a mixed basis).

Usage:
  python tests/tooling/m5_traffic_budget.py \
      [--model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf]
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

# --- measured inputs (see module docstring for provenance) -------------------
W_BYTES_PER_S = 43.6e9
MEAS_MS_AT_1500 = 106.75
SLOPE_MS_PER_CTX_TOKEN = 0.0289
CTX_OF_MEASUREMENT = 1500

Q8_0_ELEMS, Q8_0_BYTES = 32, 34   # ggml block_q8_0: f16 scale + 32 int8


def kv_read_bytes_per_ctx_token(n_layer, heads, head_dim, vec_bytes):
    """Bytes of K+V pulled per context position, per token, across all layers.

    `heads` is n_head when attention is q-head-outer (each q-head re-reads its
    kv-head's vectors) and n_head_kv when it is kv-head-outer. That single
    substitution IS rung E2, and it is why this function takes the head count
    rather than reading it from the geometry.
    """
    return n_layer * heads * vec_bytes * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    a = ap.parse_args()

    f = GGUFFile(a.model)
    arch = f.kv["general.architecture"]
    n_layer = f.kv[f"{arch}.block_count"]
    n_head = f.kv[f"{arch}.attention.head_count"]
    n_head_kv = f.kv[f"{arch}.attention.head_count_kv"]
    head_dim = f.kv[f"{arch}.attention.key_length"]
    weights = os.path.getsize(a.model)
    gqa = n_head // n_head_kv

    print(f"geometry from {a.model}")
    print(f"  n_layer {n_layer}  n_head {n_head}  n_head_kv {n_head_kv} "
          f"(GQA {gqa}:1)  head_dim {head_dim}")
    print(f"  weights {weights:,} B    W {W_BYTES_PER_S/1e9:.1f} GB/s\n")

    f32v = head_dim * 4
    f16v = head_dim * 2
    q8v = (head_dim // Q8_0_ELEMS) * Q8_0_BYTES

    variants = [
        ("today (f32, q-head-outer)", n_head, f32v),
        ("after E2 (f32, kv-head-outer)", n_head_kv, f32v),
        ("the 08-11 budget's engine (f16, no amplification)", n_head_kv, f16v),
        ("after E2 + --kv q8", n_head_kv, q8v),
    ]
    per = {}
    print("KV READ per token-position, and the crossover where it equals weights")
    for name, heads, vec in variants:
        b = kv_read_bytes_per_ctx_token(n_layer, heads, head_dim, vec)
        per[name] = b
        print(f"  {name:52s} {b:>9,} B   N*L = {weights/b:8.1f}")

    today = per["today (f32, q-head-outer)"]
    after = per["after E2 (f32, kv-head-outer)"]

    # --- the measured curve, split -------------------------------------------
    icept = MEAS_MS_AT_1500 - SLOPE_MS_PER_CTX_TOKEN * CTX_OF_MEASUREMENT
    w_ms = weights / W_BYTES_PER_S * 1e3
    slope_pred = today / W_BYTES_PER_S * 1e3
    eff = slope_pred / SLOPE_MS_PER_CTX_TOKEN

    print(f"\nTHE MEASURED CURVE, SPLIT  (decode = {icept:.2f} + "
          f"{SLOPE_MS_PER_CTX_TOKEN} * ctx ms)")
    print(f"  intercept {icept:7.2f} ms   traffic says {w_ms:6.2f} "
          f"-> {w_ms/icept*100:.1f}% of ceiling")
    print(f"  slope     {SLOPE_MS_PER_CTX_TOKEN:7.4f} ms   traffic says "
          f"{slope_pred:6.5f} -> {eff*100:.1f}% of ceiling")
    print("  ^ both halves bandwidth-bound INDEPENDENTLY, from different byte")
    print("    counts -- totals could agree by two errors cancelling; these"
          " could not.")

    # E2 divides the slope's byte count by gqa and leaves the intercept alone,
    # carrying the MEASURED efficiency forward rather than assuming ceiling.
    new_slope = (after / W_BYTES_PER_S * 1e3) / eff
    print(f"\nE2 PREDICTION (slope {SLOPE_MS_PER_CTX_TOKEN} -> {new_slope:.5f}, "
          f"intercept untouched)")
    print("   ctx    before     after   argued   priced at half")
    for ctx in (87, 554, CTX_OF_MEASUREMENT):
        b = icept + SLOPE_MS_PER_CTX_TOKEN * ctx
        n = icept + new_slope * ctx
        print(f"  {ctx:5d} {b:8.2f}  {n:8.2f}   {b/n:6.3f}x   "
              f"{1 + (b/n - 1) / 2:6.3f}x")
    print("  ^ the SHAPE is the test: E2 removes a term proportional to ctx and")
    print("    touches nothing constant, so a flat gain refutes the mechanism.")

    # --- what it does to the bars --------------------------------------------
    BARS = {8: (17.7, 10.8), 16: (14.2, 16.8)}
    print(f"\nAGGREGATE TRAFFIC CEILINGS at ctx {CTX_OF_MEASUREMENT} "
          "(upper bounds: traffic only, no compute, no scheduler)")
    for label, key in (("today", "today (f32, q-head-outer)"),
                       ("after E2", "after E2 (f32, kv-head-outer)"),
                       ("E2 + q8", "after E2 + --kv q8")):
        for N, (bar_a, bar_b) in BARS.items():
            step = (weights + N * CTX_OF_MEASUREMENT * per[key]) / W_BYTES_PER_S
            ceil = N / step
            worst = max(bar_a, bar_b)
            print(f"  {label:9s} N={N:2d}  {ceil:6.1f} tok/s   vs bars "
                  f"{bar_a}/{bar_b}  -> hardest bar is {worst/ceil*100:4.1f}% "
                  f"of ceiling ({ceil/worst:.2f}x slack)")
    print("\n  A bar at 74% of a pure-DRAM ceiling is not a bar, it is a physics")
    print("  claim -- which is what made E2 structural rather than optional.")


if __name__ == "__main__":
    main()
