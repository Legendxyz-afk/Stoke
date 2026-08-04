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

#include <math.h>
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

/* ---- the integer core -------------------------------------------------
 *
 * These reuse the SAME real blocks the bit-exactness gate above uses, read
 * from the same golden, rather than synthetic bytes: a scale/min packing is
 * only interesting on values that actually occur, and hand-built blocks tend
 * to be built from the same misunderstanding as the code under test.
 */

/* Loads every golden record of `want_type` into caller storage. Returns the
 * count. Deliberately tolerant of a short read -- the gate above is what
 * reports a malformed golden; here a short read just means fewer cases, and
 * the callers assert they got some. */
static u32 load_golden_blocks(const char* want_type, u8* raw_out,
                              size_t stride, u32 max_recs) {
    FILE* f = fopen(GOLDEN_PATH, "rb");
    if (!f) return 0;
    u32 got = 0;
    char* line;
    while ((line = read_line(f)) != NULL && got < max_recs) {
        if (line[0] == '#' || line[0] == '\0') { free(line); continue; }
        char type[16], name[128];
        unsigned row = 0;
        int consumed = 0;
        if (sscanf(line, "%15s %127s %u %n", type, name, &row, &consumed) != 3) {
            free(line);
            continue;
        }
        const char* rawhex = line + consumed;
        size_t raw_bytes = strlen(rawhex) / 2;
        if (strcmp(type, want_type) == 0 && raw_bytes == stride) {
            if (unhex(rawhex, raw_out + (size_t)got * stride, stride) == 0) {
                got++;
            }
        }
        free(line);
        char* vals = read_line(f);   /* skip the values line */
        free(vals);
    }
    fclose(f);
    return got;
}

/* THE CROSS-CHECK PROMISED IN kquant.c's COMMENTS. ggml unpacks Q4_K's
 * 6-bit scale/min pairs two different ways -- one pair at a time in the
 * dequant path, all sixteen at once with a uint32 mask/shift dance in the
 * dot path -- and stoke transcribes both because both are used. They must
 * agree. Two independent transcriptions of the same packing agreeing is
 * evidence about BOTH; reviewing either alone is evidence about neither. */
TEST(q4_k_the_two_scale_unpackings_agree_on_real_blocks) {
    enum { MAX_R = 32 };
    static u8 blocks[MAX_R * KERNELS_Q4_K_BLOCK_BYTES];
    u32 n = load_golden_blocks("Q4_K", blocks, KERNELS_Q4_K_BLOCK_BYTES, MAX_R);
    ASSERT_EQ_U32(n > 0u ? 1u : 0u, 1u);

    u32 mismatches = 0, pairs = 0;
    for (u32 r = 0; r < n; r++) {
        const u8* packed = blocks + (size_t)r * KERNELS_Q4_K_BLOCK_BYTES + 4;
        u8 scales8[8], mins8[8];
        kernels_q4_k_unpack_scales(packed, scales8, mins8);
        for (int j = 0; j < 8; j++) {
            u8 sc = 0, mn = 0;
            kernels_q4_k_scale_min_pair(packed, j, &sc, &mn);
            pairs++;
            if (sc != scales8[j] || mn != mins8[j]) {
                if (mismatches == 0) {
                    printf("  record %u j=%d: pairwise (%u,%u) vs bulk (%u,%u)\n",
                           r, j, sc, mn, scales8[j], mins8[j]);
                }
                mismatches++;
            }
        }
        /* Six-bit fields: anything above 63 means a shift or mask is wrong
         * in whichever path produced it. */
        for (int j = 0; j < 8; j++) {
            if (scales8[j] > 63u || mins8[j] > 63u) mismatches++;
        }
    }
    printf("  %u records, %u scale/min pairs compared\n", n, pairs);
    ASSERT_EQ_U32(mismatches, 0u);
    ASSERT_EQ_U32(pairs, n * 8u);
}

/* The integer dot must track the f32 dot of the SAME dequantized weights.
 * Only the activation is quantized differently between the two sides, so the
 * gap is Q8_K activation error alone -- the same shape of check, and the
 * same 2%% bar, that vec_dot_q8_0_q8_0 is held to in test_kernels.c. */
static void check_integer_dot_tracks_f32(const char* type_name,
                                         size_t block_bytes,
                                         void (*dequant)(const void*, u64, f32*),
                                         f32 (*dot)(const void*, const void*, u64)) {
    enum { MAX_R = 32, N = 256 };
    static u8 blocks[MAX_R * 256];
    u32 n = load_golden_blocks(type_name, blocks, block_bytes, MAX_R);
    ASSERT_EQ_U32(n > 0u ? 1u : 0u, 1u);

    f32 a[N], w[N];
    for (int i = 0; i < N; i++) {
        a[i] = cosf((f32)i * 0.11f) * 1.5f;
    }
    static u8 aq[(N / 256) * KERNELS_Q8_K_BLOCK_BYTES];
    quantize_row_q8_k(a, N, aq);

    u32 bad = 0;
    f32 worst = 0.0f;
    for (u32 r = 0; r < n; r++) {
        const u8* blk = blocks + (size_t)r * block_bytes;
        dequant(blk, N, w);
        f32 ref = 0.0f;
        for (int i = 0; i < N; i++) ref += w[i] * a[i];
        f32 got = dot(blk, aq, N);
        f32 denom = fabsf(ref) > 1.0f ? fabsf(ref) : 1.0f;
        f32 rel = fabsf(got - ref) / denom;
        if (rel > worst) worst = rel;
        if (!(rel < 0.02f)) {
            printf("  %s record %u: got %.6g, f32 ref %.6g, rel %.4g\n",
                   type_name, r, (double)got, (double)ref, (double)rel);
            bad++;
        }
    }
    printf("  %s: %u records, worst relative gap %.4g\n", type_name, n,
           (double)worst);
    ASSERT_EQ_U32(bad, 0u);
    /* A dot that returned 0 for everything would sail through a relative
     * bound against a 0 reference. Require the references to be real. */
    ASSERT_EQ_U32(worst > 0.0f ? 1u : 0u, 1u);
}

TEST(q4_k_integer_dot_tracks_the_f32_dot) {
    check_integer_dot_tracks_f32("Q4_K", KERNELS_Q4_K_BLOCK_BYTES,
                                 dequant_row_q4_k, vec_dot_q4_k_q8_k);
}

TEST(q6_k_integer_dot_tracks_the_f32_dot) {
    check_integer_dot_tracks_f32("Q6_K", KERNELS_Q6_K_BLOCK_BYTES,
                                 dequant_row_q6_k, vec_dot_q6_k_q8_k);
}

/* Q8_K's rounding is ggml's magic-constant nearest_int, which is
 * round-to-nearest-EVEN. roundf() is half-away-from-zero. They differ on
 * exact .5, and a block engineered to land on .5 is where that shows.
 * Pinned because switching to roundf would look like a harmless cleanup and
 * would silently move stoke out of llama.cpp's numeric class. */
TEST(q8_k_rounds_half_to_even_not_away_from_zero) {
    enum { N = 256 };
    f32 x[N];
    for (int i = 0; i < N; i++) x[i] = 0.0f;
    /* max = -254 => iscale = -127/-254 = +0.5, so x*iscale is exactly x/2:
     * every odd input lands on an exact .5 tie. */
    x[0] = -254.0f;
    x[1] = 3.0f;    /* 1.5  -> even: 2 */
    x[2] = 5.0f;    /* 2.5  -> even: 2 */
    x[3] = 7.0f;    /* 3.5  -> even: 4 */
    x[4] = 9.0f;    /* 4.5  -> even: 4 */
    static u8 q[(N / 256) * KERNELS_Q8_K_BLOCK_BYTES];
    quantize_row_q8_k(x, N, q);
    const i8* qs = (const i8*)(q + 4);
    ASSERT_EQ_U32((u32)(int)qs[0], (u32)(int)(i8)-127);
    ASSERT_EQ_U32((u32)(int)qs[1], 2u);
    ASSERT_EQ_U32((u32)(int)qs[2], 2u);
    ASSERT_EQ_U32((u32)(int)qs[3], 4u);
    ASSERT_EQ_U32((u32)(int)qs[4], 4u);
}

/* bsums is what the Q4_K min term rides on, and it is the field most easily
 * left stale in a reused scratch buffer. Assert it is the actual sum of each
 * group of 16 quants. */
TEST(q8_k_bsums_are_the_sums_of_their_sixteen_quants) {
    enum { N = 256 };
    f32 x[N];
    for (int i = 0; i < N; i++) x[i] = sinf((f32)i * 0.37f) * 2.0f;
    static u8 q[(N / 256) * KERNELS_Q8_K_BLOCK_BYTES];
    quantize_row_q8_k(x, N, q);
    const i8* qs = (const i8*)(q + 4);
    const u8* bs = q + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
    u32 bad = 0;
    for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; j++) {
        int sum = 0;
        for (u32 k = 0; k < 16u; k++) sum += qs[j * 16u + k];
        i16 stored = (i16)(u16)((u16)bs[2 * j] | ((u16)bs[2 * j + 1] << 8));
        if ((int)stored != sum) bad++;
    }
    ASSERT_EQ_U32(bad, 0u);
}

/* An all-zero block must not produce a NaN. ggml divides by `max` to build
 * iscale, so the zero guard is what stands between this and 0/0 -- the same
 * guard K2 pinned for Q8_0. Also pins that the zeroed bsums really are
 * zeroed, since this path is the one that skips the sum loop. */
TEST(q8_k_all_zero_block_is_zero_not_nan) {
    enum { N = 256 };
    f32 x[N];
    for (int i = 0; i < N; i++) x[i] = 0.0f;
    static u8 q[(N / 256) * KERNELS_Q8_K_BLOCK_BYTES];
    memset(q, 0xAB, sizeof(q));   /* poison: a reused scratch is not clean */
    quantize_row_q8_k(x, N, q);
    f32 d;
    u32 bits = (u32)q[0] | ((u32)q[1] << 8) | ((u32)q[2] << 16) | ((u32)q[3] << 24);
    memcpy(&d, &bits, sizeof(d));
    ASSERT_TRUE(d == 0.0f);
    u32 nonzero = 0;
    for (u32 i = 4; i < KERNELS_Q8_K_BLOCK_BYTES; i++) {
        if (q[i] != 0) nonzero++;
    }
    ASSERT_EQ_U32(nonzero, 0u);
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
