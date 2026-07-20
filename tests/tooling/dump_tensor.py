#!/usr/bin/env python3
"""tests/tooling/dump_tensor.py -- independent minimal GGUF v3 dumper.

M2 W1 deliverable #1. Reads the header/KV/tensor index via gguf_reader.py
(struct + numpy only, no `gguf` pip package -- see that module's docstring
for why this is an independent byte interpretation, not a port of
src/gguf.c). This is the L1 tool of the M2 parity-debug protocol (design
note section 9): dequants any tensor row so it can be diffed byte-for-byte
against the C's own dequant_row_q8_0 output.

Usage:
  python dump_tensor.py <gguf> --list
      Prints name, type, dims, offset, nbytes for every tensor, file order.
  python dump_tensor.py <gguf> --kv
      Prints every metadata KV (key, type, value) -- not in the brief's
      literal 2-flag spec, but a GGUF reader that parses KVs (it must, to
      find where the tensor index starts) and can't show them would be an
      odd omission; this is what P1's config-key diff runs against, since
      `stoke info` currently fatals on the Q8_0 file (see m2-w1-notes.md).
  python dump_tensor.py <gguf> --row <tensor_name> <row_index> [--raw]
      Dequantized float32 row to stdout, one value per line. --raw also
      hex-dumps the row's first 68 bytes (= 2 Q8_0 blocks) before the
      dequantized values, for hand verification.

Exit code: 0 on success, 1 with a "fatal: ..." stderr message on any
malformed input or bad usage (loud-failure doctrine, no silent fallback).
"""
from __future__ import annotations

import argparse
import sys

from gguf_reader import GgufError, GArray, GGML_NAME, TYPE_NAME, GGUFFile, T_ARRAY


def fmt_scalar_kv(val) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return repr(val)
    return str(val)


def cmd_list(gf: GGUFFile) -> None:
    print(f"== tensors ({len(gf.tensor_order)}, file order) ==")
    for i, t in enumerate(gf.tensor_order):
        dims_str = ",".join(str(d) for d in t.dims)
        type_name = GGML_NAME.get(t.type, f"?{t.type}")
        print(f"[{i:3d}] {t.name} {type_name} n_dims={t.n_dims} "
              f"dims=[{dims_str}] offset={t.abs_offset} nbytes={t.nbytes}")


def cmd_kv(gf: GGUFFile) -> None:
    print(f"== metadata ({len(gf.kv_order)} kv) ==")
    for i, key in enumerate(gf.kv_order):
        val = gf.kv[key]
        if isinstance(val, GArray):
            elem_name = TYPE_NAME.get(val.elem_type, f"?{val.elem_type}")
            print(f"[{i:3d}] {key} (ARRAY) = [{elem_name} x {len(val.values)}]")
        else:
            # Recover the declared scalar type name for display purposes
            # by re-deriving it from the python type (bool before int:
            # bool is a subclass of int in python).
            if isinstance(val, bool):
                tname = "BOOL"
            elif isinstance(val, float):
                tname = "FLOAT32/64"
            elif isinstance(val, int):
                tname = "INT/UINT"
            elif isinstance(val, str):
                tname = "STRING"
            else:
                tname = "?"
            print(f"[{i:3d}] {key} ({tname}) = {fmt_scalar_kv(val)}")


def hexdump(raw: bytes, base_label: str = "") -> None:
    for off in range(0, len(raw), 16):
        chunk = raw[off:off + 16]
        hex_bytes = " ".join(f"{b:02x}" for b in chunk)
        printable = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {base_label}{off:04d}: {hex_bytes:<47}  {printable}")


def cmd_row(gf: GGUFFile, name: str, row: int, raw: bool) -> None:
    t = gf.get_tensor(name)
    if raw:
        raw_bytes = gf.row_raw_bytes(t, row, 68)
        print(f"-- raw first {len(raw_bytes)} bytes of {name!r} row {row} "
              f"({GGML_NAME.get(t.type, t.type)}) --")
        hexdump(raw_bytes)
    vec = gf.tensor_row_dequant(t, row)
    print(f"-- dequantized float32 row (n={vec.shape[0]}) --")
    for x in vec:
        print(f"{float(x):.9g}")


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("gguf")
    p.add_argument("--list", action="store_true")
    p.add_argument("--kv", action="store_true")
    p.add_argument("--row", nargs=2, metavar=("TENSOR_NAME", "ROW_INDEX"))
    p.add_argument("--raw", action="store_true")
    args = p.parse_args()

    n_modes = sum([args.list, args.kv, args.row is not None])
    if n_modes != 1:
        print("fatal: exactly one of --list, --kv, --row is required", file=sys.stderr)
        return 1

    try:
        gf = GGUFFile(args.gguf)
        try:
            if args.list:
                cmd_list(gf)
            elif args.kv:
                cmd_kv(gf)
            else:
                name, row_s = args.row
                try:
                    row = int(row_s)
                except ValueError:
                    print(f"fatal: --row index {row_s!r} is not an integer", file=sys.stderr)
                    return 1
                cmd_row(gf, name, row, args.raw)
        finally:
            gf.close()
    except GgufError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
