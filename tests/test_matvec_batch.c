/* tests/test_matvec_batch.c -- batch-composition independence at the kernel.
 *
 * THE BAR (ROADMAP section M4): a sequence's result must not depend on who
 * else is in the batch, EXACTLY -- bit for bit, not within a tolerance. This
 * is where that is earned, below the model and below the KV cache: for every
 * s, matvec_batch must write precisely what matvec would have written for
 * x[s] alone.
 *
 * A tolerance here would be worse than useless. The failure mode of a
 * mis-fused batched matvec is a handful of wrong values in one sequence,
 * which is exactly what an epsilon hides -- and downstream it becomes a
 * plausible token, not a crash. Bits or nothing, the same standard
 * test_pool.c holds the thread pool to.
 *
 * ▶ WHY EVERY N IS TESTED, NOT JUST ONE. The bug this guards against is an
 * inner loop that works for the first sequence and not the rest, or one that
 * works at N=2 and not N=3. Checking a single batch size would leave both
 * alive. Position within the batch is varied for the same reason: "first in
 * the batch" is what an off-by-one leaves working.
 *
 * ▶ THE QUANT PATHS CURRENTLY DELEGATE (one matvec call per sequence) while
 * only F32 is fused. They are tested anyway and deliberately: the test has
 * to be in place and green BEFORE each path is fused, or it gets written
 * against code already believed to work and shaped to pass rather than bite.
 */
#include "common.h"

#include "kernels/kernels.h"
#include "pool.h"

#include <stdbool.h>
#include <string.h>

#define N_IN   256u
#define N_OUT   96u
#define MAX_SEQ  5u

static int checks = 0, failures = 0;

static u64 g_rng = 0xD1B54A32D192ED03ull;
static f32 frnd(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (f32)((i64)(g_rng >> 11) % 2001 - 1000) / 1000.0f;
}

/* Same idiom as test_pool.c / test_kernels.c: a 2D tensor over caller bytes,
 * so no GGUF file is needed. */
static void fake_tensor(GgufTensor* w, const char* name, u32 type,
                        const void* data, u64 nbytes) {
    memset(w, 0, sizeof *w);
    w->name = name;
    w->type = type;
    w->n_dims = 2;
    w->dims[0] = N_IN;
    w->dims[1] = N_OUT;
    w->data = (void*)(uintptr_t)data;
    w->nbytes = nbytes;
}

static void check_type(const char* label, u32 type, u64 block_elems,
                       u64 block_bytes) {
    u64 row_bytes = (N_IN / block_elems) * block_bytes;
    u64 wbytes = row_bytes * N_OUT;
    u8* wdata = (u8*)xmalloc((size_t)wbytes);
    for (u64 i = 0; i < wbytes; i++) wdata[i] = (u8)(i * 31u + 7u);
    /* Keep every f16 block scale finite -- a raw byte pattern decodes to
     * inf/nan and both implementations fatal on it, which is a different
     * test's business. Same fixup test_pool.c applies. */
    if (type == GGUF_GGML_TYPE_Q8_0) {
        for (u64 b = 0; b < wbytes / block_bytes; b++) {
            u16 h = f32_to_f16(0.0125f);
            memcpy(wdata + b * block_bytes, &h, 2);
        }
    } else if (type == GGUF_GGML_TYPE_Q4_K) {
        for (u64 b = 0; b < wbytes / block_bytes; b++) {
            u16 d = f32_to_f16(0.0125f), dm = f32_to_f16(0.003f);
            memcpy(wdata + b * block_bytes, &d, 2);
            memcpy(wdata + b * block_bytes + 2, &dm, 2);
        }
    } else if (type == GGUF_GGML_TYPE_Q6_K) {
        for (u64 b = 0; b < wbytes / block_bytes; b++) {
            u16 d = f32_to_f16(0.0125f);
            memcpy(wdata + b * block_bytes + 208, &d, 2);
        }
    }

    GgufTensor w;
    fake_tensor(&w, label, type, wdata, wbytes);

    f32* x[MAX_SEQ];
    f32* y_solo[MAX_SEQ];
    f32* y_batch[MAX_SEQ];
    void* aq[MAX_SEQ];
    for (u32 s = 0; s < MAX_SEQ; s++) {
        x[s] = (f32*)xmalloc(N_IN * sizeof(f32));
        for (u32 i = 0; i < N_IN; i++) x[s][i] = frnd();
        y_solo[s]  = (f32*)xmalloc(N_OUT * sizeof(f32));
        y_batch[s] = (f32*)xmalloc(N_OUT * sizeof(f32));
        /* Generously sized: one Q8_0 block header + payload per 32 elems. */
        aq[s] = xmalloc((size_t)(N_IN / KERNELS_Q8_K_BLOCK_ELEMS) * KERNELS_Q8_K_BLOCK_BYTES + (size_t)(N_IN / KERNELS_Q8_0_BLOCK_ELEMS) * KERNELS_Q8_0_BLOCK_BYTES + 64u);
    }

    /* The referent: each sequence alone. */
    for (u32 s = 0; s < MAX_SEQ; s++)
        matvec(&w, x[s], N_IN, y_solo[s], N_OUT, aq[s]);

    /* Every batch size, and every position within it. */
    for (u32 n = 1; n <= MAX_SEQ; n++) {
        for (u32 rot = 0; rot < n; rot++) {
            const f32* xin[MAX_SEQ];
            f32* yout[MAX_SEQ];
            void* aqin[MAX_SEQ];
            u32 which[MAX_SEQ];
            for (u32 k = 0; k < n; k++) {
                which[k] = (k + rot) % n;
                xin[k]  = x[which[k]];
                yout[k] = y_batch[which[k]];
                aqin[k] = aq[which[k]];
                memset(yout[k], 0xA5, N_OUT * sizeof(f32));
            }

            matvec_batch(&w, xin, N_IN, yout, N_OUT, aqin, n);

            bool same = true;
            for (u32 k = 0; k < n; k++) {
                if (memcmp(y_batch[which[k]], y_solo[which[k]],
                           N_OUT * sizeof(f32)) != 0) same = false;
            }
            if (!same) {
                printf("  FAIL: %s n=%u rot=%u not bit-identical to solo\n",
                       label, n, rot);
                failures++;
            }
            checks++;
        }
    }

    for (u32 s = 0; s < MAX_SEQ; s++) {
        free(x[s]); free(y_solo[s]); free(y_batch[s]); free(aq[s]);
    }
    free(wdata);
}

int main(void) {
    /* Run at more than one thread count: the fused path hands rows to the
     * pool exactly as matvec does, and a batched job that got the row range
     * wrong would show up only when rows are actually split. */
    for (u32 threads = 1; threads <= 4; threads *= 2) {
        pool_start(threads);
        check_type("f32 (FUSED)",   GGUF_GGML_TYPE_F32,  1u,  4u);
        check_type("q8_0 (FUSED)",   GGUF_GGML_TYPE_Q8_0, 32u, 34u);
        check_type("q4_K",          GGUF_GGML_TYPE_Q4_K, 256u, 144u);
        check_type("q6_K",          GGUF_GGML_TYPE_Q6_K, 256u, 210u);
        pool_stop();
    }

    printf("test_matvec_batch: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
