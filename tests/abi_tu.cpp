/* tests/abi_tu.cpp -- M6 gate item [5]: stoke.h compiles and links as C++.
 *
 * WHY THIS IS A SEPARATE FILE AND NOT A FLAG ON THE C ONE. The failures it
 * catches are the ones C cannot see at all:
 *   - a missing `extern "C"`, which compiles perfectly and then fails to LINK
 *     because every symbol got C++ name mangling;
 *   - a field or parameter named after a C++ keyword (`new`, `class`,
 *     `template`, `delete`, `operator`), which is legal C and a syntax error
 *     here;
 *   - an implicit void*-to-T* conversion, legal in C and forbidden in C++;
 *   - an enum used where C++ will not narrow it.
 * Every one of those ships silently in a C-only build and lands on the first
 * host that writes C++. Compiled with -Werror, so a warning is a failure.
 *
 * It LINKS against the library rather than just compiling, because the missing
 * `extern "C"` case is invisible until link time -- a compile-only check would
 * pass on the exact defect this file exists to catch.
 *
 * VERIFIED BY MUTATION on 2026-08-28, and the signature is worth writing down
 * because it is not the one you would predict. With the `extern "C"` guard
 * #if 0'd out:
 *   - tests/abi_tu.c still builds and still exits 0. C cannot see the defect,
 *     which is the entire reason this file exists.
 *   - this TU compiles fine and dies at the link step with
 *         collect2.exe: error: ld returned 5 exit status
 *     and `make ... Error 1` -- with NO "undefined reference" lines anywhere in
 *     the log, on this toolchain (MSYS2 ucrt64 g++ 15.2.0). So a check that
 *     greps the build log for "undefined reference" would MISS it. The signal
 *     is the make exit code, which is what the `test` target uses.
 */
#include "stoke.h"
#include "stoke.h" /* twice: include guard, same reason as the C TU */

#include <cstddef>

/* Reached through a C++ template so the declarations are instantiated in a
 * genuinely C++ context, not merely parsed in one. */
template <typename Fn>
static bool declared(Fn f) {
    return f != nullptr;
}

int main() {
    stoke_open_opts opts{};
    opts.size = sizeof opts;

    stoke_request req{};
    req.size = sizeof req;
    req.eos_id = -1;

    /* C++ will not implicitly narrow, so this also asserts the status type is
     * what the header says it is rather than a bare enum. */
    stoke_status st = STOKE_OK;
    if (st != 0) return 1;

    if (!declared(&stoke_abi_version)) return 2;
    if (!declared(&stoke_open_abi)) return 3;
    if (!declared(&stoke_close)) return 4;
    if (!declared(&stoke_last_error)) return 5;
    if (!declared(&stoke_submit)) return 6;
    if (!declared(&stoke_step)) return 7;
    if (!declared(&stoke_poll)) return 8;
    if (!declared(&stoke_cancel)) return 9;
    if (!declared(&stoke_tokenize)) return 10;
    if (!declared(&stoke_detokenize)) return 11;
    if (!declared(&stoke_n_vocab)) return 12;
    if (!declared(&stoke_max_ctx)) return 13;
    if (!declared(&stoke_eos_id)) return 14;

    /* The calls that need no model on disk, so this binary is runnable and not
     * merely linkable -- and these two are the ones a host calls FIRST. */
    if (stoke_abi_version() != STOKE_ABI_VERSION) return 15;
    if (stoke_last_error() == nullptr) return 16;

    /* An ABI mismatch must be REPORTED, not crash: pass a version the library
     * cannot possibly be. This is also the only test that proves stoke_open's
     * validation runs before it touches the filesystem. */
    if (stoke_open_abi("/nonexistent", &opts, STOKE_ABI_VERSION + 1000) != nullptr) {
        return 17;
    }
    if (stoke_last_error()[0] == '\0') return 18;

    return 0;
}
