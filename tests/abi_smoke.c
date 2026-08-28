/* tests/abi_smoke.c -- M6 rung A1: drive a real generation THROUGH stoke.h.
 *
 * ▶▶ WHY THIS EXISTS SEPARATELY FROM abi_tu.c. That one proves the header
 * compiles and links. This one proves the ABI WORKS -- and the gap between
 * those is the whole failure mode the M6 entry note is built to avoid: an ABI
 * that was never called by anything before it was frozen. Every function here
 * is reached the way a host would reach it, in the order a host would, using
 * ONLY stoke.h.
 *
 * ⚠ IT INCLUDES NOTHING FROM src/ EXCEPT stoke.h -- not common.h, not model.h.
 * If this file ever needs an internal header to do something a host would
 * plausibly want, that is a finding ABOUT THE HEADER and the fix belongs in the
 * header. Reaching around it here would hide exactly the defect worth finding.
 *
 * Not named test_*.c on purpose: it needs the 2.5 GB model and a path argument,
 * so it is driven by tests/abi_smoke.sh, which skips cleanly when the model is
 * absent -- the same policy as the golden gates.
 *
 * Usage: abi_smoke <model.gguf>
 */
#include "stoke.h"

#include <stdio.h>
#include <string.h>

#define PROMPT   "The capital of France is"
#define MAX_TOK  16
#define MAX_IDS  512

static int fail(const char* what) {
    fprintf(stderr, "abi_smoke: FAIL %s: %s\n", what, stoke_last_error());
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: abi_smoke <model.gguf>\n");
        return 2;
    }

    /* ---- 1. the failure path FIRST, while nothing is open ----
     * A library that cannot survive a bad path is not a library, and this is
     * the trapped fatal() from common.h reaching the ABI boundary. If the trap
     * regressed, THIS process dies here -- which is a far louder failure than
     * an assertion, and exactly the outcome a host would suffer. */
    if (stoke_open("./no-such-model-here.gguf", NULL) != NULL) {
        fprintf(stderr, "abi_smoke: FAIL open of a missing file returned non-NULL\n");
        return 1;
    }
    if (stoke_last_error()[0] == '\0') {
        fprintf(stderr, "abi_smoke: FAIL open failed but left no message\n");
        return 1;
    }
    printf("abi_smoke: bad path trapped, not fatal: %s\n", stoke_last_error());

    /* ---- 2. open for real ---- */
    stoke_open_opts opts;
    memset(&opts, 0, sizeof opts);
    opts.size     = sizeof opts;
    opts.max_ctx  = 512;
    opts.max_seqs = 2;

    stoke_engine* e = stoke_open(argv[1], &opts);
    if (!e) return fail("stoke_open");

    printf("abi_smoke: opened  n_vocab=%u max_ctx=%u eos=%u\n",
           stoke_n_vocab(e), stoke_max_ctx(e), stoke_eos_id(e));

    /* ---- 3. tokenize ---- */
    uint32_t ids[MAX_IDS];
    int32_t  n = stoke_tokenize(e, PROMPT, -1, 0, ids, MAX_IDS);
    if (n <= 0) { stoke_close(e); return fail("stoke_tokenize"); }
    if (n > MAX_IDS) {
        fprintf(stderr, "abi_smoke: FAIL prompt needs %d ids, buffer is %d\n",
                (int)n, MAX_IDS);
        stoke_close(e);
        return 1;
    }
    printf("abi_smoke: tokenized %d ids\n", (int)n);

    /* ---- 4. submit ----
     * `ids` is BORROWED by the engine and outlives the request because it is a
     * stack array in this frame that does not return until after the finish.
     * Written that way on purpose: it is the shape the header tells hosts to
     * use, so the smoke exercises the documented contract rather than a safer
     * one nobody was told about. */
    stoke_request req;
    memset(&req, 0, sizeof req);
    req.size       = sizeof req;
    req.id         = 1;
    req.prompt_ids = ids;
    req.n_prompt   = (uint32_t)n;
    req.max_tokens = MAX_TOK;
    req.eos_id     = (int64_t)stoke_eos_id(e);

    if (stoke_submit(e, &req) != STOKE_OK) { stoke_close(e); return fail("stoke_submit"); }

    /* A second submit of the SAME id must be refused, and with the specific
     * code -- not merely "false". An ABI that collapses distinguishable errors
     * is worse than the internal call it wraps. */
    if (stoke_submit(e, &req) != STOKE_ERR_DUPLICATE_ID) {
        fprintf(stderr, "abi_smoke: FAIL duplicate id was not refused as such\n");
        stoke_close(e);
        return 1;
    }

    /* ---- 5. step and poll to completion ---- */
    uint32_t got[MAX_TOK];
    int32_t  n_got = 0;
    int32_t  finish = STOKE_FINISH_NONE;
    int      steps = 0;

    while (finish == STOKE_FINISH_NONE && steps < 4096) {
        uint32_t rows = 0;
        if (stoke_step(e, &rows) != STOKE_OK) { stoke_close(e); return fail("stoke_step"); }
        steps++;
        int32_t k = stoke_poll(e, req.id, got + n_got,
                               MAX_TOK - n_got, &finish);
        if (k < 0) { stoke_close(e); return fail("stoke_poll"); }
        n_got += k;
        if (rows == 0 && finish == STOKE_FINISH_NONE) {
            fprintf(stderr, "abi_smoke: FAIL step moved 0 rows with work outstanding\n");
            stoke_close(e);
            return 1;
        }
    }

    /* ANTI-VACUITY. "It ran" is also true of a run that produced nothing, and
     * every gate in this project has been bitten by that at least once. */
    if (finish == STOKE_FINISH_NONE) {
        fprintf(stderr, "abi_smoke: FAIL never finished in %d steps\n", steps);
        stoke_close(e);
        return 1;
    }
    if (n_got == 0) {
        fprintf(stderr, "abi_smoke: FAIL finished (%d) with zero tokens\n", (int)finish);
        stoke_close(e);
        return 1;
    }

    /* ---- 6. detokenize ---- */
    char text[1024];
    size_t used = 0;
    for (int32_t i = 0; i < n_got; i++) {
        char piece[256];
        int32_t m = stoke_detokenize(e, got[i], piece, (int32_t)sizeof piece);
        if (m < 0) { stoke_close(e); return fail("stoke_detokenize"); }
        if ((size_t)m < sizeof piece && used + (size_t)m < sizeof text - 1) {
            memcpy(text + used, piece, (size_t)m);
            used += (size_t)m;
        }
    }
    text[used] = '\0';

    printf("abi_smoke: %d steps, %d tokens, finish=%d\n", steps, (int)n_got,
           (int)finish);
    printf("abi_smoke: \"%s%s\"\n", PROMPT, text);

    /* ---- 7. an unknown id is an error, not a crash ---- */
    if (stoke_cancel(e, 999999) != STOKE_ERR_UNKNOWN_ID) {
        fprintf(stderr, "abi_smoke: FAIL cancel of an unknown id was not refused\n");
        stoke_close(e);
        return 1;
    }

    stoke_close(e);
    stoke_close(NULL); /* documented as safe */
    printf("abi_smoke: OK\n");
    return 0;
}
