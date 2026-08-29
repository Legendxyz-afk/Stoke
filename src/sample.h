/* src/sample.h -- token selection from logits. M6 rung A6.
 *
 * ▶▶▶ PURE, AND THAT IS A CORRECTNESS REQUIREMENT RATHER THAN A STYLE. The M5
 * scheduler PREEMPTS a sequence and later REPLAYS it: the same positions are
 * generated a second time. A sampler holding a running RNG would hand back a
 * different token on the replay, and nothing would notice -- the stream stays
 * fluent, the counts still add up, and the "preempt-replay is lossless"
 * property the whole M5 design rests on quietly stops being true. So the
 * randomness is keyed on the POSITION: a draw is a pure function of
 * (seed, pos), and a replay reproduces it by construction rather than by
 * anyone remembering to.
 *
 * Everything here is a pure function of its arguments. No state, no allocation,
 * no clock -- which is also what makes it testable without a model.
 */
#ifndef STOKE_SAMPLE_H
#define STOKE_SAMPLE_H

#include "common.h"

/* A ZEROED STRUCT IS EXACTLY GREEDY, and each field's disabled value is
 * whatever makes that true -- which is NOT the same as "its identity".
 * top_p's identity is 1, but a zeroed struct holds 0, so 0 has to mean
 * unset as well; reading it literally would make the nucleus one token and
 * hand every caller who set only a temperature silent greedy output.
 *
 * That property is what the server leans on: `temperature: 0`, `top_p: 1` and
 * the penalties at their identities all need no special case, which is why the
 * request the M0 harness sends on every call needs no branch anywhere. */
typedef struct {
    f32 temperature;    /* <= 0 => greedy argmax; the whole struct is ignored */
    u32 top_k;          /* 0    => no truncation by rank                      */
    f32 top_p;          /* >= 1 OR <= 0 (unset) => no truncation by mass      */
    f32 repeat_penalty; /* <= 1 => no penalty                                 */
    u64 seed;
} SampleParams;

/* Chooses one id.
 *
 * `history` is the tokens already produced for THIS request, indexed by their
 * own position, and only the first `n_history` are read. It is used for the
 * repeat penalty and for nothing else.
 *
 * ⚠ THE HISTORY IS POSITION-INDEXED, NOT APPENDED, for the same replay reason
 * as the seed: a replayed position OVERWRITES its own slot instead of pushing a
 * duplicate, so the penalty a replay computes is the penalty the original
 * computed.
 *
 * `scratch` must hold at least `n_vocab` floats and is written arbitrarily; it
 * exists so a caller inside a token loop does not allocate.
 */
u32 sample_token(const SampleParams* p, u32 pos, const f32* logits, u32 n_vocab,
                 const u32* history, u32 n_history, f32* scratch);

/* splitmix64, exposed because the tests pin it: a seeded stream that changed
 * shape between releases would silently invalidate every reproduction anyone
 * had recorded. */
u64 sample_rng(u64 seed, u32 pos);

#endif /* STOKE_SAMPLE_H */
