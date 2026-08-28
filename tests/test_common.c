/* tests/test_common.c — exercises src/common.h: typedefs, xmalloc family,
 * now_ms(), and fatal() (spawn-self; see the last test for why).
 */
#include "common.h"

#include "gguf.h" /* for the cross-translation-unit trap test only */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

/* --fatal-child: re-invoking this same exe with this flag makes it call
 * fatal() directly (instead of running the test suite) so the parent test
 * below can observe, from the outside, what fatal() actually does: nonzero
 * exit code + "fatal: <msg>" on stderr. Wired to test.h's child-mode escape
 * hatch via TEST_CHILD_MAIN below.
 */
static int fatal_child_main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--fatal-child") == 0) {
        fatal("boom %d", 42);
    }
    fprintf(stderr, "test_common: unrecognized child mode '%s'\n",
            argc > 1 ? argv[1] : "(none)");
    return 2;
}
#define TEST_CHILD_MAIN fatal_child_main
#include "test.h"

/* ---- the fatal() trap (M6) ----
 *
 * The spawn-self test at the bottom of this file is the other half of these
 * and it is the one that must never be deleted: it proves that with NO trap
 * armed fatal() still exits(1) with "fatal: " on stderr. Everything below
 * proves the armed path, and the pair is what says the trap is additive.
 */
static void raise_via_fatal(void) { fatal("trapped %d", 7); }

TEST(trap_catches_fatal_and_captures_the_message) {
    StokeTrap t;
    int caught = 0;
    stoke_trap_push(&t);
    if (STOKE_TRAP_ARMED(t)) {
        raise_via_fatal();
        ASSERT_TRUE(0 && "fatal() returned to its caller");
        stoke_trap_pop(&t);
    } else {
        caught = 1;
    }
    ASSERT_EQ_I64(caught, 1);
    ASSERT_EQ_STR(stoke_errmsg, "trapped 7");
    /* fatal() disarms BEFORE jumping, so nothing is left armed on this thread. */
    ASSERT_TRUE(stoke_trap_top == NULL);
}

TEST(trap_pop_on_the_success_path_leaves_nothing_armed) {
    StokeTrap t;
    int ran = 0;
    stoke_trap_push(&t);
    if (STOKE_TRAP_ARMED(t)) {
        ran = 1;
        stoke_trap_pop(&t);
    }
    ASSERT_EQ_I64(ran, 1);
    ASSERT_TRUE(stoke_trap_top == NULL);
}

/* NESTING IS NOT DECORATION: stoke_open will arm a trap and call code that
 * arms its own. If the inner catch left the chain wrong, the OUTER trap would
 * either be lost (next failure kills the host) or double-popped. */
TEST(inner_trap_catches_and_the_outer_one_survives) {
    StokeTrap outer, inner;
    int inner_caught = 0, outer_intact = 0;
    stoke_trap_push(&outer);
    if (STOKE_TRAP_ARMED(outer)) {
        stoke_trap_push(&inner);
        if (STOKE_TRAP_ARMED(inner)) {
            raise_via_fatal();
        } else {
            inner_caught = 1;
        }
        outer_intact = (stoke_trap_top == &outer);
        stoke_trap_pop(&outer);
    }
    ASSERT_EQ_I64(inner_caught, 1);
    ASSERT_EQ_I64(outer_intact, 1);
    ASSERT_TRUE(stoke_trap_top == NULL);
}

/* THE ALLOCATORS MUST ROUTE THROUGH THE TRAP TOO, and this is the cheap
 * deterministic proof: xcalloc's OVERFLOW branch calls fatal() without
 * attempting a huge allocation, so the test costs nothing and does not depend
 * on the box being out of memory. If the x* family ever stops going through
 * the fatal() choke point, stoke_open's OOM path silently becomes an abort. */
TEST(xcalloc_overflow_is_trappable_not_fatal) {
    StokeTrap t;
    int caught = 0;
    stoke_trap_push(&t);
    if (STOKE_TRAP_ARMED(t)) {
        void* p = xcalloc(SIZE_MAX, 2);
        ASSERT_TRUE(p == NULL && "xcalloc overflow returned instead of failing");
        stoke_trap_pop(&t);
    } else {
        caught = 1;
    }
    ASSERT_EQ_I64(caught, 1);
    ASSERT_TRUE(strstr(stoke_errmsg, "xcalloc") != NULL);
}

/* ⚠⚠ THE ONE ABOVE ALL PASS EVEN IF THE TRAP STATE IS WRONG, AND THIS IS THE
 * ONE THAT DOES NOT. Every test so far arms the trap and raises the fatal() in
 * THIS translation unit, so they cannot distinguish the shipped design
 * (external linkage, one chain per thread) from the bug it was written to
 * avoid (a `static` beside the inline fatal(), giving every .c its own chain).
 * In one TU both spellings behave identically and the whole suite stays green
 * while `stoke_open` silently aborts its host for any failure raised inside
 * gguf.c, model.c or tok.c -- which is every failure that matters.
 *
 * So this one arms the trap HERE and raises it THERE: gguf_open() on a path
 * that does not exist fatals inside gguf.c, a different object file. It is the
 * only test in this file whose result depends on src/common.c existing.
 *
 * VERIFIED BY MUTATION, and the mutant's signature is worth knowing because it
 * is not a clean FAIL line. Replacing the extern declarations with per-TU
 * statics: 10 tests pass -- including the same-TU trap test
 * `volatile_local_survives_the_longjmp`, which is the whole point -- and then
 * the binary DIES here, exit 1, with
 *   `fatal: gguf: cannot open './definitely-not-a-real-model-file.gguf' ...`
 * on stderr and no "tests passed" line at all, because gguf.c's own copy of
 * fatal() found a NULL chain and called exit(). The remaining trap tests are
 * never reached. A harness that only greps for "FAIL" would call that a pass;
 * run_tests.sh catches it on the exit code. */
TEST(a_fatal_raised_in_another_translation_unit_is_trapped) {
    StokeTrap t;
    int caught = 0;
    stoke_trap_push(&t);
    if (STOKE_TRAP_ARMED(t)) {
        GgufFile* f = gguf_open("./definitely-not-a-real-model-file.gguf");
        ASSERT_TRUE(f == NULL && "gguf_open returned for a missing file");
        stoke_trap_pop(&t);
    } else {
        caught = 1;
    }
    ASSERT_EQ_I64(caught, 1);
    ASSERT_TRUE(strstr(stoke_errmsg, "gguf") != NULL);
    ASSERT_TRUE(stoke_trap_top == NULL);
}

/* A volatile local survives the longjmp; this is the shape stoke_open's
 * partial-construction cleanup depends on, asserted so the requirement is
 * executable rather than a comment in common.h. */
TEST(volatile_local_survives_the_longjmp) {
    StokeTrap t;
    volatile int stage = 0;
    stoke_trap_push(&t);
    if (STOKE_TRAP_ARMED(t)) {
        stage = 1;
        raise_via_fatal();
        stoke_trap_pop(&t);
    }
    ASSERT_EQ_I64(stage, 1);
}

TEST(typedefs_have_expected_sizes) {
    ASSERT_TRUE(sizeof(u8) == 1);
    ASSERT_TRUE(sizeof(u16) == 2);
    ASSERT_TRUE(sizeof(u32) == 4);
    ASSERT_TRUE(sizeof(u64) == 8);
    ASSERT_TRUE(sizeof(i8) == 1);
    ASSERT_TRUE(sizeof(i16) == 2);
    ASSERT_TRUE(sizeof(i32) == 4);
    ASSERT_TRUE(sizeof(i64) == 8);
    ASSERT_TRUE(sizeof(f32) == 4);
    ASSERT_TRUE(sizeof(f64) == 8);
}

TEST(xmalloc_returns_writable_memory) {
    size_t n = 1024 * 1024; /* 1 MiB */
    u8* buf = (u8*)xmalloc(n);
    ASSERT_TRUE(buf != NULL);
    for (size_t i = 0; i < n; i++) {
        buf[i] = (u8)(i * 37u + 11u);
    }
    int ok = 1;
    for (size_t i = 0; i < n; i++) {
        if (buf[i] != (u8)(i * 37u + 11u)) {
            ok = 0;
            break;
        }
    }
    ASSERT_TRUE(ok);
    free(buf);
}

TEST(xcalloc_zeroes_memory) {
    size_t n = 4096, sz = 256; /* 1 MiB, exact */
    u8* buf = (u8*)xcalloc(n, sz);
    ASSERT_TRUE(buf != NULL);
    ASSERT_TRUE(buf[0] == 0);
    ASSERT_TRUE(buf[n * sz - 1] == 0);
    buf[n * sz - 1] = 0xAB;
    ASSERT_TRUE(buf[n * sz - 1] == 0xAB);
    free(buf);
}

TEST(xrealloc_grows_and_preserves_contents) {
    u8* buf = (u8*)xmalloc(16);
    for (int i = 0; i < 16; i++) {
        buf[i] = (u8)i;
    }
    buf = (u8*)xrealloc(buf, 4096);
    ASSERT_TRUE(buf != NULL);
    int ok = 1;
    for (int i = 0; i < 16; i++) {
        if (buf[i] != (u8)i) ok = 0;
    }
    ASSERT_TRUE(ok);
    free(buf);
}

TEST(now_ms_is_monotonic_and_positive) {
    double t0 = now_ms();
    volatile long spin = 0;
    for (long i = 0; i < 2000000; i++) {
        spin += i;
    }
    double t1 = now_ms();
    ASSERT_TRUE(t0 > 0.0);
    ASSERT_TRUE(t1 >= t0);
}

TEST(assert_eq_i64_macro_compiles_and_passes) {
    i64 a = -12345;
    i64 b = -12345;
    ASSERT_EQ_I64(a, b);
    ASSERT_EQ_I64(7, 7);
}

TEST(assert_eq_u32_macro_compiles_and_passes) {
    u32 a = 4000000000u;
    u32 b = 4000000000u;
    ASSERT_EQ_U32(a, b);
    ASSERT_EQ_U32(0, 0);
}

TEST(assert_eq_str_macro_compiles_and_passes) {
    ASSERT_EQ_STR("hello", "hello");
    const char* s = "stoke";
    ASSERT_EQ_STR(s, "stoke");
}

/* fatal(): spawn THIS SAME exe with --fatal-child, capture its stderr to a
 * file next to it, then assert: nonzero exit code + stderr starts with
 * "fatal: boom 42".
 *
 * Windows system()-quoting proof (this is the mechanical deliverable the
 * plan asked for, exercised every `make test`, not just asserted in a
 * comment): cmd.exe's /C handling strips the command line's outer quote
 * pair UNLESS the whole line is exactly one quoted executable name and
 * nothing else (`cmd /?`). Our line is `"<exe>" --fatal-child 2>"<err>"` —
 * 4 quote characters, not the exempted 2 — so that strip fires and eats the
 * FIRST and LAST quote of the whole line, corrupting both the exe path and
 * the redirection target. That breaks the instant either path contains a
 * space, which every path here does: this repo lives under "...\New
 * Drive\...". Fix: wrap the entire line in one more pair of quotes. cmd
 * then strips THOSE two (first/last char of the now-longer line) and hands
 * CreateProcess the untouched, correctly-quoted original line. Proved by
 * actually running this test from M:\New Drive\...\stoke (see task report).
 */
TEST(fatal_exits_nonzero_with_stderr_prefix) {
#ifdef _WIN32
    char exe_path_buf[4096];
    DWORD full_len = GetFullPathNameA(g_test_argv0, (DWORD)sizeof exe_path_buf,
                                       exe_path_buf, NULL);
    ASSERT_TRUE(full_len > 0 && full_len < sizeof exe_path_buf);
    const char* exe_path = exe_path_buf;
#else
    const char* exe_path = g_test_argv0;
#endif

    char errfile[4160]; /* exe_path + ".fatal_child.stderr" + margin */
    char inner[8500];   /* "exe_path" + " --fatal-child 2>" + "errfile" + margin */
    char cmd[8700];     /* inner, extra-quoted on Windows, + margin */

    snprintf(errfile, sizeof errfile, "%s.fatal_child.stderr", exe_path);
    snprintf(inner, sizeof inner, "\"%s\" --fatal-child 2>\"%s\"", exe_path,
             errfile);

#ifdef _WIN32
    /* extra outer quote pair -- see the comment above this test */
    snprintf(cmd, sizeof cmd, "\"%s\"", inner);
#else
    snprintf(cmd, sizeof cmd, "%s", inner);
#endif

    int rc = system(cmd);
    ASSERT_TRUE(rc != 0);

    FILE* f = fopen(errfile, "rb");
    ASSERT_TRUE(f != NULL);
    if (f) {
        char buf[256];
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        buf[n] = '\0';
        fclose(f);
        const char* want = "fatal: boom 42";
        ASSERT_TRUE(strncmp(buf, want, strlen(want)) == 0);
        remove(errfile);
    }
}
