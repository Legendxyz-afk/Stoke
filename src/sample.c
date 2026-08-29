/* src/sample.c -- token selection from logits. See sample.h for the contract
 * and for why every function here is pure.
 */
#include "common.h"

#include "sample.h"

#include <math.h>
#include <string.h>

/* splitmix64. Chosen because a draw must be a PURE function of (seed, pos) --
 * see sample.h -- and splitmix64 turns a counter into a well-distributed 64-bit
 * value in four operations with no state to carry. Mixing the position INTO the
 * seed rather than iterating from it is what makes position 900 as cheap and as
 * reproducible as position 1, which matters because a replayed sequence asks
 * for its positions in whatever order the scheduler rebuilds them. */
u64 sample_rng(u64 seed, u32 pos) {
    u64 z = seed + 0x9E3779B97F4A7C15ULL * ((u64)pos + 1u);
    z     = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z     = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static u32 greedy(const f32* logits, u32 n) {
    u32 best = 0;
    for (u32 i = 1; i < n; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

/* The repeat penalty, applied to LOGITS before anything else reads them.
 *
 * ⚠ THE SIGN RULE IS NOT COSMETIC. llama.cpp divides a positive logit by the
 * penalty and MULTIPLIES a negative one; halving a logit of -8 to -4 would
 * otherwise make a disfavoured token MORE likely, which is the opposite of a
 * penalty and shows up as the model repeating exactly the words you asked it to
 * avoid. */
static void apply_repeat_penalty(f32* v, u32 n, const u32* history,
                                 u32 n_history, f32 penalty) {
    if (penalty <= 1.0f || !history || n_history == 0) return;
    for (u32 i = 0; i < n_history; i++) {
        u32 id = history[i];
        if (id >= n) continue;
        v[id] = (v[id] > 0.0f) ? v[id] / penalty : v[id] * penalty;
    }
}

/* Partial selection: leaves the `k` largest values of `v` in `idx[0..k)`,
 * descending. O(n*k) and k is small in practice; a full sort of a 152k-entry
 * vocabulary per token would cost more than the truncation saves. */
static void top_k_indices(const f32* v, u32 n, u32 k, u32* idx) {
    for (u32 i = 0; i < k; i++) idx[i] = 0;
    u32 filled = 0;
    for (u32 i = 0; i < n; i++) {
        if (filled < k) {
            u32 j = filled++;
            while (j > 0 && v[idx[j - 1]] < v[i]) { idx[j] = idx[j - 1]; j--; }
            idx[j] = i;
        } else if (v[i] > v[idx[k - 1]]) {
            u32 j = k - 1;
            while (j > 0 && v[idx[j - 1]] < v[i]) { idx[j] = idx[j - 1]; j--; }
            idx[j] = i;
        }
    }
}

u32 sample_token(const SampleParams* p, u32 pos, const f32* logits, u32 n_vocab,
                 const u32* history, u32 n_history, f32* scratch) {
    if (!p || !logits || n_vocab == 0) return 0;

    /* ▶ GREEDY IS THE FIRST BRANCH AND IT TOUCHES NOTHING ELSE. temperature 0
     * is what the M0 harness sends on every request, so it is the hot path, and
     * it must be bit-identical to the argmax it replaces -- no penalty pass, no
     * copy, no softmax, no rounding introduced on the way past. */
    if (p->temperature <= 0.0f) {
        if (p->repeat_penalty > 1.0f && history && n_history > 0 && scratch) {
            memcpy(scratch, logits, (size_t)n_vocab * sizeof(f32));
            apply_repeat_penalty(scratch, n_vocab, history, n_history,
                                 p->repeat_penalty);
            return greedy(scratch, n_vocab);
        }
        return greedy(logits, n_vocab);
    }
    if (!scratch) return greedy(logits, n_vocab);

    memcpy(scratch, logits, (size_t)n_vocab * sizeof(f32));
    apply_repeat_penalty(scratch, n_vocab, history, n_history,
                         p->repeat_penalty);

    /* The candidate set, by rank. Capped so the softmax below is bounded work
     * whatever the vocabulary: past a few hundred candidates the tail carries
     * no mass a sampler can distinguish, and an uncapped top_p would mean
     * sorting 152k entries per token. */
    u32 k = p->top_k ? p->top_k : 512u;
    if (k > n_vocab) k = n_vocab;
    if (k > 512u) k = 512u;

    u32 idx[512];
    top_k_indices(scratch, n_vocab, k, idx);

    /* Softmax over the candidates, shifted by the max for numerical stability
     * -- exp(8/temperature) overflows f32 for a small temperature and would
     * turn the whole distribution into NaN, which then samples as index 0. */
    f32 maxv = scratch[idx[0]];
    f32 sum  = 0.0f;
    f32 prob[512];
    for (u32 i = 0; i < k; i++) {
        prob[i] = expf((scratch[idx[i]] - maxv) / p->temperature);
        sum += prob[i];
    }
    if (!(sum > 0.0f)) return idx[0];
    for (u32 i = 0; i < k; i++) prob[i] /= sum;

    /* top_p: keep the shortest prefix whose mass reaches p, and ALWAYS keep at
     * least one -- a threshold below the winner's own share must collapse to
     * the winner rather than to an empty set. */
    /* ⚠ `top_p <= 0` MEANS UNSET, NOT "KEEP NOTHING". A zeroed params
     * struct has top_p 0, and reading that literally makes the nucleus one
     * token -- so every caller who zeroed a struct and set only a
     * temperature would get greedy output back and no error. The tests
     * caught exactly that: 400 draws at temperature 8 with top_k 2 never
     * once produced the runner-up. Treating it as unset is also the only
     * sane reading of a client literally asking to keep zero mass. */
    u32 keep = k;
    if (p->top_p > 0.0f && p->top_p < 1.0f) {
        f32 acc = 0.0f;
        keep    = 0;
        for (u32 i = 0; i < k; i++) {
            acc += prob[i];
            keep = i + 1;
            if (acc >= p->top_p) break;
        }
        if (keep == 0) keep = 1;
    }

    f32 kept = 0.0f;
    for (u32 i = 0; i < keep; i++) kept += prob[i];
    if (!(kept > 0.0f)) return idx[0];

    /* The draw. 53 bits of the 64 is more than an f32 accumulation can
     * distinguish, and taking the HIGH bits avoids the weak low bits of any
     * multiplicative generator. */
    f32 r   = (f32)((sample_rng(p->seed, pos) >> 11) * (1.0 / 9007199254740992.0));
    f32 t   = r * kept;
    f32 acc = 0.0f;
    for (u32 i = 0; i < keep; i++) {
        acc += prob[i];
        if (t <= acc) return idx[i];
    }
    return idx[keep - 1]; /* only reachable through f32 rounding at the very top */
}
