/* src/common.c — the only non-inline part of common.h.
 *
 * common.h is header-only by design and says so at the top. This file exists
 * for exactly two objects, and for one reason: `fatal()` is `static inline`,
 * so every translation unit that includes common.h compiles its OWN copy. If
 * the trap chain and the message buffer were `static` beside that inline
 * function, each copy would consult a different chain — and whether a
 * load-time failure was CAUGHT by `stoke_open` or KILLED THE HOST PROCESS
 * would depend on which .c file happened to raise it. Correct in the TU where
 * the trap was armed, fatal everywhere else, and invisible to inspection
 * because every call site reads identically.
 *
 * So: external linkage, defined exactly once, here. Nothing else belongs in
 * this file — a helper added here stops being header-only for no reason.
 *
 * `_Thread_local` because two threads may open two models concurrently, and a
 * shared trap would let one thread's failure longjmp into another's stack
 * frame. Zero-initialised by definition, so an un-armed thread has a NULL top
 * and takes the unchanged exit(1) path.
 */
#include "common.h"

_Thread_local StokeTrap* stoke_trap_top = NULL;
_Thread_local char       stoke_errmsg[STOKE_ERRMSG_CAP];
