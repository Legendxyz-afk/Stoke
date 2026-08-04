/* tests/test_kquant.c -- the K4 gate: src/kernels/kquant.c must reproduce
 * llama.cpp's own Q4_K/Q6_K dequant BIT FOR BIT.
 *
 * WHERE THE EXPECTATIONS COME FROM, and why it matters that it is not here.
 * Nothing in this file was computed by hand or by us. tests/data/
 * k4_dequant.golden holds real super-blocks lifted from the shipping
 * Qwen3-4B Q4_K_M file, each paired with the f32 values llama.cpp ITSELF
 * produced for those exact bytes:
 *
 *     llama-quantize --allow-requantize <Q4_K_M>.gguf <out>.gguf F32 8
 *
 * The python side (tests/tooling/gguf_reader.py) was verified against the
 * full 15 GB of that output -- 741,888 values, 0 differing -- and the golden
 * is a frozen slice of that passing comparison. So this test compares C
 * against llama.cpp, with python only as the courier. That is K2's rule
 * again: the strongest check on a transcription is a fixed point of the
 * original system's own output, because it requires no agreement about
 * intent.
 *
 * WHY EXACT BITS AND NOT A TOLERANCE -- this is the whole design, and it is
 * measured rather than assumed. The failure this gate exists to catch is the
 * get_scale_min_k4 slip named in the milestone plan, and when injected as a
 * mutation it perturbs only ~17%% of a row's values, by at most ~0.0138. It
 * is SMALL and SPARSE. Any epsilon loose enough to deserve the name
 * "tolerance" passes it. Dequantization is a fixed f32 sequence with no
 * reduction, so correct implementations must agree exactly -- and only
 * exactness discriminates.
 *
 * One golden record is ONE FULL 256-element super-block, which is the
 * natural unit: it spans all eight sub-blocks, so it exercises both branches
 * of get_scale_min_k4 (j<4 and j>=4) rather than only the easy one.
 */
#include "common.h"

#include "kernels/kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#define GOLDEN_PATH "tests/data/k4_dequant.golden"
#define MAX_ELEMS 256u
#define MAX_RAW 256u

static int hex_nib(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Decodes `2*n_bytes` hex chars into out. Returns 0 on success. */
static int unhex(const char* s, u8* out, size_t n_bytes) {
    for (size_t i = 0; i < n_bytes; i++) {
        int hi = hex_nib((unsigned char)s[2 * i]);
        int lo = hex_nib((unsigned char)s[2 * i + 1]);
        if (hi < 0 || lo < 0) return 1;
        out[i] = (u8)((hi << 4) | lo);
    }
    return 0;
}

static u32 f32_to_bits(f32 v) {
    u32 b;
    memcpy(&b, &v, sizeof(b));
    return b;
}

static f32 bits_to_f32(u32 b) {
    f32 v;
    memcpy(&v, &b, sizeof(v));
    return v;
}

/* Reads one line of arbitrary length into a growable buffer. The value lines
 * are 256 * 9 chars, comfortably past any fixed stack buffer worth having. */
static char* read_line(FILE* f) {
    size_t cap = 4096, len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return NULL;
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '\r') continue;
        if (c == '\n') break;
        if (len + 1 >= cap) {
            cap *= 2;
            char* nb = (char*)realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        buf[len++] = (char)c;
    }
    if (c == EOF && len == 0) { free(buf); return NULL; }
    buf[len] = '\0';
    return buf;
}

TEST(kquant_dequant_matches_llama_cpp_bit_for_bit) {
    FILE* f = fopen(GOLDEN_PATH, "rb");
    if (!f) {
        printf("  cannot open %s -- run from the repo root\n", GOLDEN_PATH);
        ASSERT_EQ_U32(1u, 0u);
        return;
    }

    u32 records = 0, q4k = 0, q6k = 0, bad_values = 0, bad_records = 0;
    u32 values = 0;
    char* line;
    while ((line = read_line(f)) != NULL) {
        if (line[0] == '#' || line[0] == '\0') { free(line); continue; }

        /* "<type> <tensor> <row> <raw-hex>" */
        char type[16], name[128];
        unsigned row = 0;
        int consumed = 0;
        if (sscanf(line, "%15s %127s %u %n", type, name, &row, &consumed) != 3) {
            printf("  malformed header line\n");
            bad_records++;
            free(line);
            continue;
        }
        const char* rawhex = line + consumed;
        size_t raw_bytes = strlen(rawhex) / 2;

        int is_q4 = (strcmp(type, "Q4_K") == 0);
        int is_q6 = (strcmp(type, "Q6_K") == 0);
        if (!is_q4 && !is_q6) {
            printf("  %s: unknown type %s\n", name, type);
            bad_records++;
            free(line);
            continue;
        }
        size_t want_bytes = is_q4 ? KERNELS_Q4_K_BLOCK_BYTES
                                  : KERNELS_Q6_K_BLOCK_BYTES;
        if (raw_bytes != want_bytes || raw_bytes > MAX_RAW) {
            printf("  %s: raw block is %u bytes, expected %u\n",
                   name, (unsigned)raw_bytes, (unsigned)want_bytes);
            bad_records++;
            free(line);
            continue;
        }
        u8 raw[MAX_RAW];
        if (unhex(rawhex, raw, raw_bytes) != 0) {
            printf("  %s: bad hex in raw block\n", name);
            bad_records++;
            free(line);
            continue;
        }
        free(line);

        char* vals = read_line(f);
        if (!vals) {
            printf("  %s: header line with no values line after it\n", name);
            bad_records++;
            break;
        }

        f32 want[MAX_ELEMS];
        u32 n = 0;
        const char* p = vals;
        while (*p && n < MAX_ELEMS) {
            while (*p == ' ') p++;
            if (!*p) break;
            unsigned long bits = strtoul(p, (char**)&p, 16);
            want[n++] = bits_to_f32((u32)bits);
        }
        free(vals);
        if (n != MAX_ELEMS) {
            printf("  %s: %u values, expected %u\n", name, n, MAX_ELEMS);
            bad_records++;
            continue;
        }

        f32 got[MAX_ELEMS];
        if (is_q4) {
            dequant_row_q4_k(raw, MAX_ELEMS, got);
            q4k++;
        } else {
            dequant_row_q6_k(raw, MAX_ELEMS, got);
            q6k++;
        }
        records++;

        u32 diff = 0;
        for (u32 i = 0; i < MAX_ELEMS; i++) {
            values++;
            if (f32_to_bits(got[i]) != f32_to_bits(want[i])) {
                if (diff == 0) {
                    printf("  %s (%s) first diff at [%u]: got %.9g "
                           "(0x%08X), want %.9g (0x%08X)\n",
                           name, type, i, (double)got[i], f32_to_bits(got[i]),
                           (double)want[i], f32_to_bits(want[i]));
                }
                diff++;
            }
        }
        if (diff) {
            printf("  %s (%s): %u/%u values differ\n", name, type, diff,
                   MAX_ELEMS);
            bad_values += diff;
            bad_records++;
        }
    }
    fclose(f);

    printf("  %u records (%u Q4_K, %u Q6_K), %u values, %u differing\n",
           records, q4k, q6k, values, bad_values);

    /* The golden must actually have covered BOTH types. A gate that silently
     * degrades to "0 records, 0 failures" is the shape of a passing test that
     * proves nothing -- this project has already been bitten by exactly that
     * (a mutation whose anchor failed to apply produced a 22/22 PASS). */
    ASSERT_EQ_U32(bad_values, 0u);
    ASSERT_EQ_U32(bad_records, 0u);
    ASSERT_EQ_U32(q4k > 0u ? 1u : 0u, 1u);
    ASSERT_EQ_U32(q6k > 0u ? 1u : 0u, 1u);
}

/* Geometry is asserted rather than assumed: these four constants are what
 * vec.c's row-stride arithmetic will use, and a wrong one addresses memory
 * rather than merely computing a wrong number. The byte counts are also
 * corroborated independently by the real file (its Q4_K tensors have
 * nbytes == (elems/256)*144). */
TEST(kquant_block_geometry_is_the_ggml_layout) {
    ASSERT_EQ_U32(KERNELS_Q4_K_BLOCK_ELEMS, 256u);
    ASSERT_EQ_U32(KERNELS_Q4_K_BLOCK_BYTES, 144u);   /* 2+2+12+128 */
    ASSERT_EQ_U32(KERNELS_Q6_K_BLOCK_ELEMS, 256u);
    ASSERT_EQ_U32(KERNELS_Q6_K_BLOCK_BYTES, 210u);   /* 128+64+16+2 */
    /* Both are super-blocks of eight 32-element sub-blocks. */
    ASSERT_EQ_U32(KERNELS_Q4_K_BLOCK_ELEMS / KERNELS_Q8_0_BLOCK_ELEMS, 8u);
    ASSERT_EQ_U32(KERNELS_Q6_K_BLOCK_ELEMS / KERNELS_Q8_0_BLOCK_ELEMS, 8u);
}
