/* tests/test.h — minimal header-only test framework shared by every
 * tests/test_*.c binary.
 *
 * The Makefile builds ONE standalone executable per tests/test_*.c file
 * (never one combined test binary), so each test_*.c is its own translation
 * unit with exactly one #include of this header — defining file-scope
 * statics and `main` right here carries no ODR/multiple-definition risk.
 *
 * Registration choice: tests self-register via GCC's
 * __attribute__((constructor)) rather than an explicit per-file test list.
 * Both are equally portable across this project's two toolchains (MSYS2
 * ucrt64 gcc 15.2, WSL gcc) and both stay clean under -Wall -Wextra -Werror
 * on both (verified) — constructor registration won only because it needs
 * no hand-maintained list as tests are added or removed.
 *
 * Child-mode escape hatch: a test that needs to re-exec its OWN binary as a
 * subprocess (e.g. test_common.c's fatal() test, which must observe
 * fatal()'s exit code and stderr from the outside) defines TEST_CHILD_MAIN
 * to the name of an `int fn(int argc, char** argv)` function BEFORE
 * including this header. When invoked with extra argv, main() dispatches
 * straight to it instead of running the registered tests.
 */
#ifndef STOKE_TEST_H
#define STOKE_TEST_H

#include <stdio.h>
#include <string.h>

#define TEST_MAX 256

typedef void (*test_fn)(void);

typedef struct {
    const char* name;
    test_fn     fn;
} TestCase;

static TestCase    g_test_cases[TEST_MAX];
static int         g_test_case_count  = 0;
static int         g_test_case_failed = 0; /* current test's failure flag */
static int         g_test_any_failed  = 0;
static const char* g_test_argv0       = NULL;

#define TEST(name_) \
    static void stoke_test_##name_(void); \
    __attribute__((constructor)) static void stoke_register_##name_(void) { \
        if (g_test_case_count < TEST_MAX) { \
            g_test_cases[g_test_case_count].name = #name_; \
            g_test_cases[g_test_case_count].fn   = stoke_test_##name_; \
            g_test_case_count++; \
        } \
    } \
    static void stoke_test_##name_(void)

#define ASSERT_TRUE(expr_) \
    do { \
        if (!(expr_)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr_); \
            g_test_case_failed = 1; \
        } \
    } while (0)

#define ASSERT_EQ_I64(a_, b_) \
    do { \
        long long stoke_lhs_ = (long long)(a_); \
        long long stoke_rhs_ = (long long)(b_); \
        if (stoke_lhs_ != stoke_rhs_) { \
            fprintf(stderr, "FAIL %s:%d: %s == %s (%lld != %lld)\n", \
                    __FILE__, __LINE__, #a_, #b_, stoke_lhs_, stoke_rhs_); \
            g_test_case_failed = 1; \
        } \
    } while (0)

#define ASSERT_EQ_U32(a_, b_) \
    do { \
        unsigned long stoke_lhs_ = (unsigned long)(a_); \
        unsigned long stoke_rhs_ = (unsigned long)(b_); \
        if (stoke_lhs_ != stoke_rhs_) { \
            fprintf(stderr, "FAIL %s:%d: %s == %s (%lu != %lu)\n", \
                    __FILE__, __LINE__, #a_, #b_, stoke_lhs_, stoke_rhs_); \
            g_test_case_failed = 1; \
        } \
    } while (0)

#define ASSERT_EQ_STR(a_, b_) \
    do { \
        const char* stoke_lhs_ = (a_); \
        const char* stoke_rhs_ = (b_); \
        if (strcmp(stoke_lhs_, stoke_rhs_) != 0) { \
            fprintf(stderr, "FAIL %s:%d: %s == %s (\"%s\" != \"%s\")\n", \
                    __FILE__, __LINE__, #a_, #b_, stoke_lhs_, stoke_rhs_); \
            g_test_case_failed = 1; \
        } \
    } while (0)

int main(int argc, char** argv) {
    g_test_argv0 = argv[0];
    (void)g_test_argv0; /* only spawn-self tests read this again; silence
                            -Wunused-but-set-variable (-Wextra) otherwise */

#ifdef TEST_CHILD_MAIN
    if (argc > 1) {
        return TEST_CHILD_MAIN(argc, argv);
    }
#else
    (void)argc;
#endif

    int passed = 0;
    for (int i = 0; i < g_test_case_count; i++) {
        g_test_case_failed = 0;
        g_test_cases[i].fn();
        if (g_test_case_failed) {
            g_test_any_failed = 1;
        } else {
            printf("ok %s\n", g_test_cases[i].name);
            passed++;
        }
    }
    printf("%d/%d tests passed\n", passed, g_test_case_count);
    return g_test_any_failed ? 1 : 0;
}

#endif /* STOKE_TEST_H */
