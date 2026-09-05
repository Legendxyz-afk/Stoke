/* src/kernels/ops.c -- normalization, rotary embedding, softmax,
 * activation, and argmax kernels. See kernels.h for the full contract of
 * each.
 */
#include "common.h"

#include "kernels.h"

#include <math.h>

void rmsnorm(const f32* x, const f32* w, u64 n, f32 eps, f32* y) {
    f32 sum_sq = 0.0f;
    for (u64 i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    f32 mean_sq = sum_sq / (f32)n;
    f32 denom = sqrtf(mean_sq + eps);
    /* Second pass reads x[i] before writing y[i] at that same index (and
     * never revisits index i afterwards), so y == x (in-place) is safe. */
    for (u64 i = 0; i < n; i++) {
        y[i] = (x[i] / denom) * w[i];
    }
}

void rope_neox(f32* vec, u64 head_dim, u64 pos, f32 freq_base,
               const f32* freq_factors) {
    u64 half = head_dim / 2;
    f32 theta_scale = powf(freq_base, -2.0f / (f32)head_dim);
    f32 theta = (f32)pos;

    /* The NULL branch is a SEPARATE LOOP, not a `ff ? x/ff : x` inside one:
     * a model with no factors must execute what it executed before this
     * parameter existed, since every Qwen parity artefact in this project
     * (the id-stream goldens, the m3 gate archive, the bit-exact kernel
     * A/Bs) depends on rope being unmoved. */
    if (!freq_factors) {
        for (u64 k = 0; k < half; k++) {
            f32 cos_theta = cosf(theta);
            f32 sin_theta = sinf(theta);
            f32 x0 = vec[k];
            f32 x1 = vec[k + half];
            vec[k]        = x0 * cos_theta - x1 * sin_theta;
            vec[k + half] = x0 * sin_theta + x1 * cos_theta;
            theta *= theta_scale; /* prepares theta_{k+1}; matches the
                                      oracle's cache-init accumulation
                                      (m2-probe-notes.md P3.3) rather than a
                                      fresh powf per k */
        }
        return;
    }

    /* llama3 RoPE scaling. The DIVISION IS AT USE and the accumulator is
     * untouched -- ggml_rope_cache_init does `rope_yarn(theta/ff, ...)`
     * then `theta *= theta_scale` (ggml-cpu/ops.cpp:5846-5860). Dividing
     * the accumulator instead would compound the factor across k. */
    for (u64 k = 0; k < half; k++) {
        f32 theta_k = theta / freq_factors[k];
        f32 cos_theta = cosf(theta_k);
        f32 sin_theta = sinf(theta_k);
        f32 x0 = vec[k];
        f32 x1 = vec[k + half];
        vec[k]        = x0 * cos_theta - x1 * sin_theta;
        vec[k + half] = x0 * sin_theta + x1 * cos_theta;
        theta *= theta_scale;
    }
}

void rope_norm(f32* vec, u64 head_dim, u64 pos, f32 freq_base,
               const f32* freq_factors) {
    u64 half = head_dim / 2;
    f32 theta_scale = powf(freq_base, -2.0f / (f32)head_dim);
    f32 theta = (f32)pos;

    /* Same two-loop shape as rope_neox, and for the same reason: no model
       without factors may pay for the branch or the divide. */
    if (!freq_factors) {
        for (u64 k = 0; k < half; k++) {
            f32 cos_theta = cosf(theta);
            f32 sin_theta = sinf(theta);
            f32 x0 = vec[2 * k];
            f32 x1 = vec[2 * k + 1];
            vec[2 * k]     = x0 * cos_theta - x1 * sin_theta;
            vec[2 * k + 1] = x0 * sin_theta + x1 * cos_theta;
            theta *= theta_scale;
        }
        return;
    }

    for (u64 k = 0; k < half; k++) {
        f32 theta_k = theta / freq_factors[k];
        f32 cos_theta = cosf(theta_k);
        f32 sin_theta = sinf(theta_k);
        f32 x0 = vec[2 * k];
        f32 x1 = vec[2 * k + 1];
        vec[2 * k]     = x0 * cos_theta - x1 * sin_theta;
        vec[2 * k + 1] = x0 * sin_theta + x1 * cos_theta;
        theta *= theta_scale;
    }
}

void softmax(f32* x, u64 n) {
    /* Starting at -infinity (rather than x[0]) means this never
     * dereferences x unconditionally, so n == 0 is safe (every loop below
     * simply does not execute). */
    f32 max_val = -INFINITY;
    for (u64 i = 0; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    f32 sum = 0.0f;
    for (u64 i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (u64 i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

f32 silu(f32 z) {
    return z / (1.0f + expf(-z));
}

void swiglu(const f32* g, const f32* u, u64 n, f32* h) {
    for (u64 i = 0; i < n; i++) {
        h[i] = silu(g[i]) * u[i];
    }
}

u32 argmax(const f32* logits, u64 n) {
    u64 best = 0;
    for (u64 i = 1; i < n; i++) {
        if (logits[i] > logits[best]) { /* strict: equal values never
                                            displace an earlier index */
            best = i;
        }
    }
    return (u32)best;
}
