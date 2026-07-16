/* src/common.h — fixed-width types, fatal(), xmalloc family, timing.
 *
 * Header-only by design: every helper below is `static inline`, so any .c
 * file that includes this gets its own private copy per translation unit —
 * no separate common.c to build, no linkage/ODR concerns.
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

/* ---- fatal: loud failure, never returns ----
 * Every access in this codebase is checked; a failure that can't be handled
 * calls this instead of falling back silently. Prints "fatal: <message>\n"
 * to stderr and exit(1)s.
 */
static inline _Noreturn void fatal(const char* fmt, ...) {
    va_list ap;
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
