/* tests/abi_tu.c -- the C half of M6 gate item [5]: stoke.h must compile
 * STANDALONE.
 *
 * "Standalone" is the whole assertion, so this file includes stoke.h FIRST and
 * includes nothing else from src/. If stoke.h ever starts depending on
 * common.h -- for u32, for fatal(), for anything -- this stops compiling, which
 * is the point: a public header that only works when an internal header was
 * included first is not a public header, and the failure would otherwise
 * surface as a stranger's build error rather than ours.
 *
 * Included TWICE to prove the include guard works. That sounds trivial until a
 * host puts stoke.h in a precompiled header and their whole tree stops building.
 */
#include "stoke.h"
#include "stoke.h"

#include <stddef.h>

/* Taking the address of every function proves each one is DECLARED with a type
 * a caller can form -- a stricter check than calling them, and it needs no
 * model on disk. If a declaration is missing or malformed, this fails to
 * compile; if a symbol is missing from the library, the LINK fails. Both are
 * the errors a host would hit, hit here instead. */
static void* const g_abi_surface[] = {
    (void*)(size_t)&stoke_abi_version, (void*)(size_t)&stoke_open_abi,
    (void*)(size_t)&stoke_close,       (void*)(size_t)&stoke_last_error,
    (void*)(size_t)&stoke_n_vocab,     (void*)(size_t)&stoke_max_ctx,
    (void*)(size_t)&stoke_eos_id,      (void*)(size_t)&stoke_submit,
    (void*)(size_t)&stoke_step,        (void*)(size_t)&stoke_poll,
    (void*)(size_t)&stoke_cancel,      (void*)(size_t)&stoke_tokenize,
    (void*)(size_t)&stoke_detokenize,
};

int main(void) {
    /* The `size` field is load-bearing for forward compatibility, so assert the
     * shape a caller is told to write actually compiles as written. */
    stoke_open_opts opts;
    opts.size = sizeof opts;
    opts.max_ctx = 0;
    opts.max_seqs = 0;
    opts.n_threads = 0;
    opts.kv_q8 = 0;
    opts.dedup_prefix = 0;

    stoke_request req;
    req.size = sizeof req;
    req.id = 1;
    req.prompt_ids = NULL;
    req.n_prompt = 0;
    req.max_tokens = 0;
    req.eos_id = -1;

    if (stoke_abi_version() != STOKE_ABI_VERSION) return 1;
    if (stoke_last_error() == NULL) return 2; /* documented never-NULL */
    if (opts.size == 0 || req.size == 0) return 3;
    for (size_t i = 0; i < sizeof g_abi_surface / sizeof g_abi_surface[0]; i++) {
        if (g_abi_surface[i] == NULL) return 4;
    }
    return 0;
}
