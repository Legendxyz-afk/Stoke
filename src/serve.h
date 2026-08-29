/* src/serve.h -- M5: the REAL SchedHost, backed by the model.
 *
 * The scheduler is written against two callbacks (sched.h's SchedHost): a step
 * and a clock. The simulation supplies a hash and a virtual clock; this file
 * supplies `model_forward_batch` + `model_compute_logits_batch` + argmax, and
 * `now_ms`. Nothing else differs between the two, which is the whole point of
 * the seam -- docs/superpowers/design-notes/2026-08-19-m5-s0-sim-architecture.md.
 *
 * ▶ WHY IT IS A SEPARATE FILE AND NOT A FUNCTION IN main.c. It owns buffers
 * that must be sized to the widest batch the scheduler ever hands it and reused
 * across every step: a host that allocated per step would put a malloc in the
 * token loop, which M2's op-spec forbids and which would also make the ITL
 * percentiles measure the allocator.
 */
#ifndef STOKE_SERVE_H
#define STOKE_SERVE_H

#include "common.h"

#include "model.h"
#include "sched.h"

typedef struct ServeHost ServeHost;

/* ---- optional sampling (M6 rung A6) ------------------------------------
 *
 * ▶▶▶ THIS IS THE ONE CHANGE THIS FILE TAKES FOR THE SAMPLER, AND IT IS SHAPED
 * BY WHAT THIS FILE IS. Every M5 gate number was measured through serve_step,
 * so the milestone's standing rule is that touching it costs a re-measure. The
 * token is chosen here and nowhere else -- sched.c never calls the model -- so
 * "do not touch it" was never available. What IS available is making the change
 * as small as it can be: ONE nullable function pointer, and not a line of
 * sampling arithmetic in this file.
 *
 * With no sampler installed the step does `argmax` exactly as before, behind a
 * single predictable null check per yielding row. `bench --serve` installs
 * none, so the M5 measurement path keeps its shape.
 *
 * ⚠ THE POSITION IS PASSED, AND IT IS WHAT MAKES A SEEDED STREAM SURVIVE
 * PREEMPTION. The scheduler preempts and REPLAYS sequences, so a sampler that
 * advanced a per-request RNG on each call would produce a different token the
 * second time through and silently break the replay-losslessness the M5 design
 * rests on. Keyed on the position instead, a sample is a PURE FUNCTION of
 * (seed, position) and a replay reproduces it by construction. */
typedef u32 (*ServeSampleFn)(void* ctx, u64 req_id, u32 pos,
                             const f32* logits, u32 n_vocab);

/* NULL `fn` (the default) restores plain argmax. */
void serve_host_set_sampler(ServeHost* h, ServeSampleFn fn, void* ctx);

/* `m` is BORROWED and must outlive the host. */
ServeHost* serve_host_open(Model* m);
void       serve_host_close(ServeHost* h);

/* The SchedHost to hand sched_open. Valid while `h` is open. */
SchedHost serve_host(ServeHost* h);

/* Rows stepped and forwards issued, for the bench driver and for the same
 * anti-vacuity reason every other counter here exists: "the scheduler ran" is
 * also true of a scheduler that ran no rows. */
u64 serve_host_rows(const ServeHost* h);
u64 serve_host_steps(const ServeHost* h);

#endif /* STOKE_SERVE_H */
