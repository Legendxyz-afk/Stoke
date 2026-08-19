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
