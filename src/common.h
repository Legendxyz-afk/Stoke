/* src/common.h — fixed-width types, fatal(), xmalloc family, timing.
 *
 * Header-only by design: every helper below is `static inline`, so any .c
 * file that includes this gets its own private copy per translation unit —
 * no separate common.c to build, no linkage/ODR concerns.
 *
 * ▶▶ ONE EXCEPTION, ADDED AT M6 ENTRY, AND IT IS THE EXCEPTION THAT PROVES
 * THE RULE MATTERED. The M6 ABI traps load-time failures: `stoke_open` arms a
 * jmp_buf, `fatal()` longjmps back to it instead of exiting, and the message
 * is read afterwards via `stoke_last_error()`. That state — the trap chain and
 * the message buffer — is the one thing here that CANNOT be per-translation-
 * unit. `fatal()` is `static inline`, so every .c gets its own copy; if the
 * trap state were a `static` beside it, each copy would consult a DIFFERENT
 * chain, and whether an error was caught or killed the host would depend on
 * which TU happened to raise it. That is silent, load-bearing and untestable
 * by inspection. So the two objects have EXTERNAL linkage and are defined
 * exactly once, in src/common.c. The inline helpers stay inline.
 *
 * MUST be the first #include in any .c file that uses it. Reason: under
 * strict -std=c11, glibc hides clock_gettime()/CLOCK_MONOTONIC (POSIX, not
 * ISO C) unless a feature-test macro raises visibility, and that macro has
 * to be defined before the *first* libc header lands in the translation
 * unit — same lesson ../loom/src/spawn.c documents for setenv/unsetenv on
 * this exact toolchain. Defining it here, first, covers every consumer.
 */
#ifndef STOKE_COMMON_H
#define STOKE_COMMON_H

#ifndef _WIN32
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

/* ---- fixed-width typedefs ---- */

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef float    f32;
typedef double   f64;

/* ---- the fatal() trap: how a library turns an abort into a return ----
 *
 * WHY THIS EXISTS. There are 322 `fatal()` call sites (counted, not estimated
 * — the M6 forward note's "~90" was wrong by 3.5x). A library cannot exit its
 * host's process because a model file was malformed, but rewriting 322 sites
 * into error returns is a different project, not a milestone. So NO CALL SITE
 * CHANGES: the single choke point below learns to longjmp when a trap is
 * armed, and behaves exactly as it always did when one is not. Every CLI path
 * arms nothing and is bit-for-bit unaffected.
 *
 * HOW TO USE IT. `setjmp` may not be called in a function that has returned by
 * the time the `longjmp` happens, and C11 7.13.1.1 restricts where the call may
 * syntactically appear — a comma expression is NOT one of the permitted forms,
 * so the push cannot be folded into the guard. The shape is fixed:
 *
 *     StokeTrap t;
 *     stoke_trap_push(&t);
 *     if (STOKE_TRAP_ARMED(t)) {
 *         ... work that may call fatal() ...
 *         stoke_trap_pop(&t);
 *     } else {
 *         // fatal() landed here; the trap is ALREADY popped.
 *         // stoke_errmsg holds the message. Clean up and report.
 *     }
 *
 * ⚠ A local variable modified between the `setjmp` and the `longjmp` has an
 * indeterminate value afterwards unless it is `volatile`. Anything the else-
 * branch must read to clean up — a partially built handle, an fd — has to be
 * declared `volatile`, or the teardown reads garbage on the path that matters
 * most and never on the path that gets tested.
 *
 * ⚠ fatal() pops the trap BEFORE jumping, on purpose: the handler runs
 * disarmed, so a fatal() raised while cleaning up after a fatal() cannot
 * longjmp back into its own handler forever. It exits, loudly, which is the
 * correct outcome for a failure inside failure handling.
 */
#define STOKE_ERRMSG_CAP 512

typedef struct StokeTrap {
    jmp_buf           jb;
    struct StokeTrap* prev;
} StokeTrap;

/* EXTERNAL linkage, defined once in src/common.c -- see the header comment.
 * Thread-local: two threads opening two models must not share a trap. */
extern _Thread_local StokeTrap* stoke_trap_top;
extern _Thread_local char       stoke_errmsg[STOKE_ERRMSG_CAP];

static inline void stoke_trap_push(StokeTrap* t) {
    t->prev        = stoke_trap_top;
    stoke_trap_top = t;
}

/* Pops only if `t` is still the top, so an unbalanced pop after fatal() has
 * already popped is a no-op rather than corruption of an outer frame's chain. */
static inline void stoke_trap_pop(StokeTrap* t) {
    if (stoke_trap_top == t) {
        stoke_trap_top = t->prev;
    }
}

/* The whole thing is the controlling expression of an `if`, which is one of
 * C11 7.13.1.1's permitted forms for a setjmp call. Do not wrap it in anything. */
#define STOKE_TRAP_ARMED(t_) (setjmp((t_).jb) == 0)

/* ---- fatal: loud failure, never returns ----
 * Every access in this codebase is checked; a failure that can't be handled
 * calls this instead of falling back silently. With no trap armed it prints
 * "fatal: <message>\n" to stderr and exit(1)s, exactly as it always has.
 */
static inline _Noreturn void fatal(const char* fmt, ...) {
    va_list ap;
    StokeTrap* t = stoke_trap_top;
    if (t) {
        va_start(ap, fmt);
        vsnprintf(stoke_errmsg, sizeof stoke_errmsg, fmt, ap);
        va_end(ap);
        stoke_trap_top = t->prev; /* disarm before jumping -- see above */
        longjmp(t->jb, 1);
    }
    fprintf(stderr, "fatal: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(1);
}

/* ---- checked allocation: fatal on OOM/overflow, never returns NULL for a
 * nonzero request ---- */

static inline void* xmalloc(size_t size) {
    void* p = malloc(size);
    if (!p && size != 0) {
        fatal("xmalloc: out of memory (%zu bytes)", size);
    }
    return p;
}

static inline void* xcalloc(size_t n, size_t size) {
    if (n != 0 && size > SIZE_MAX / n) {
        fatal("xcalloc: overflow computing %zu * %zu bytes", n, size);
    }
    void* p = calloc(n, size);
    if (!p && n != 0 && size != 0) {
        fatal("xcalloc: out of memory (%zu * %zu bytes)", n, size);
    }
    return p;
}

static inline void* xrealloc(void* ptr, size_t size) {
    void* p = realloc(ptr, size);
    if (!p && size != 0) {
        fatal("xrealloc: out of memory (%zu bytes)", size);
    }
    return p;
}

/* ---- now_ms: monotonic wall clock, milliseconds ---- */

#ifdef _WIN32
static inline double now_ms(void) {
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    if (!have_freq) {
        QueryPerformanceFrequency(&freq);
        have_freq = 1;
    }
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif

#endif /* STOKE_COMMON_H */
