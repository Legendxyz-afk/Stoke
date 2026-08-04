/* src/pool.h -- K5: the worker pool, and the one property it must not break.
 *
 * WHAT IT IS. A fixed set of persistent worker threads plus a parallel-for
 * over ROW CHUNKS, claimed through a single atomic counter. That is the
 * forward note's locked design ("work-stealing 64-row chunks, atomic
 * next-chunk counter"), chosen over a static even split because this box is
 * 6 P-cores + 4 E-cores: an even split leaves the P-cores idling at the
 * barrier waiting for the E-cores, and a proportional split has to be
 * re-tuned for every machine. A counter is self-balancing and needs to know
 * nothing about the topology.
 *
 * ▶▶ THE PROPERTY THAT MAKES THIS SAFE, AND IT IS NOT "THE LOCKS ARE RIGHT".
 * The rows of a matvec are INDEPENDENT dot products -- nothing reduces
 * across them. Each y[r] is computed by exactly the same sequential kernel,
 * over exactly the same bytes, in exactly the same order, whichever thread
 * happens to run it. So the output is BIT-IDENTICAL for any thread count and
 * any interleaving, and that is asserted directly (tests/test_pool.c runs
 * the same matvec at 1 and N threads and compares bit patterns).
 *
 * This is the same argument the x4 SIMD twin rests on, one level up: the
 * parallelism is over a dimension that carries no arithmetic dependency, so
 * it costs no gate. If a future op needs a reduction ACROSS chunks, that op
 * does NOT get to use this pool without its own numeric argument -- a
 * summation order that depends on thread scheduling is not reproducible, and
 * this project gates on bit-exactness.
 *
 * THREADING SUBSTRATE: pthreads + C11 atomics, PROBED on both legs
 * (2026-08-04) rather than assumed. Windows MinGW has no <threads.h> at all
 * but winpthreads works, and WSL gcc obviously does, so one implementation
 * covers the matrix with no #ifdef split. TSan runs on this file in the WSL
 * leg on every commit from here to M7.
 */
#ifndef STOKE_POOL_H
#define STOKE_POOL_H

#include "common.h"

/* Rows per claimed chunk. 64 is the forward note's locked value: large
 * enough that the atomic is noise against the work, small enough that the
 * tail cannot leave a fast core idle for long. */
#define POOL_CHUNK_ROWS 64u

/* Starts the pool with `n_threads` TOTAL workers including the calling
 * thread, so n_threads=1 means "no helper threads" and the pool is inert.
 * n_threads=0 asks for the measured default.
 *
 * Idempotent only in the trivial sense: calling it while already started
 * with a different count stops and restarts. Not thread-safe against
 * itself -- it is a process-setup call, made once from main() before any
 * work, never from inside a parallel region.
 *
 * THE DEFAULT IS 8, AND IT IS OUR OWN MEASUREMENT -- NOT llama's.
 *
 * The K3-entry sweep (2026-08-03, GATE-grade) put llama at 12.645 tok/s on
 * 6 pinned P-cores against 10.802 on 6P+4E: for llama, ten threads were 15%%
 * WORSE than six, so this pool's first default was 6. Measuring stoke itself
 * (2026-08-04, Q4_K_M, two passes) inverted that: 4T 7.6/7.0, 6T 9.9/8.7,
 * 8T 10.6/9.1, 10T 10.6/9.3, 12T 9.5/9.5, 16T 9.1/9.0 -- a broad plateau
 * from 8 to 12 with 6 measurably below it.
 *
 * The two are consistent once you look at WHY. llama's K-quant kernels are
 * near the memory ceiling, so extra threads buy nothing and cost contention;
 * ours retain only about half the size ratio, i.e. they are still
 * COMPUTE-bound, and a compute-bound kernel keeps scaling onto the E-cores
 * that a bandwidth-bound one is only obstructed by. 8 sits inside the
 * plateau in both passes rather than at the edge where the falloff starts.
 *
 * DURABLE: an optimal thread count is a property of THE KERNEL's
 * bottleneck, not of the machine -- inheriting another engine's tuned
 * constant imports its bottleneck along with its number. Re-measure this
 * when the kernels get faster; if they ever reach the memory ceiling, the
 * optimum should move back down toward 6.
 *
 * Not gate-grade: alternating passes on an unquiesced box, no sidecar. The
 * pass-to-pass drift (~10%%) is the same size as the 6-vs-8 gap, which is
 * why the default is set to the middle of a plateau rather than to whichever
 * single number came out highest. */
void pool_start(u32 n_threads);

/* Stops all workers and joins them. Safe to call when not started. */
void pool_stop(void);

/* Total workers including the caller; 1 when the pool is not started. */
u32 pool_threads(void);

/* Runs `fn(ctx, row_begin, row_end)` over every row in [0, n_rows),
 * partitioned into chunks of POOL_CHUNK_ROWS claimed atomically. Returns
 * only when all chunks are done. The CALLING THREAD participates -- it is
 * not a dispatcher that waits.
 *
 * `fn` must touch only its own [row_begin, row_end) slice of any output and
 * must not itself call pool_for_rows. Runs `fn` inline on the caller when
 * the pool is not started or n_rows is small enough that dispatch would
 * cost more than the work. */
void pool_for_rows(u64 n_rows, void (*fn)(void*, u64, u64), void* ctx);

#endif /* STOKE_POOL_H */
