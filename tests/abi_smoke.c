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
#define MAX_IDS 2048
/* Sized for stoke_chat_template_bound's FIXED part, which grew from 64 to
 * 512 when the Llama-3 header template landed: that template prepends BOS
 * and an UNCONDITIONAL system block with a ~62-byte preamble, none of
 * which ChatML has. The bound is an UPPER bound over both templates, so it
 * pays the larger fixed cost even for a ChatML model. */

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

    /* ---- 7. the chat template, added at rung A3 ----
     *
     * ▶▶ THE FIRST ASSERTION BELOW IS THE ONE THAT MATTERS, and it is not about
     * chat at all: a `cap` under the bound must RETURN AN ERROR. The internal
     * encoder answers that condition with fatal(), which is right for a CLI and
     * is a remote process kill behind an HTTP endpoint -- any client could send
     * a long enough prompt. If this ever regresses, THIS PROCESS DIES HERE
     * rather than printing a failure, which is the loudest possible signal and
     * exactly what a host would suffer. */
    if (!stoke_has_chat_template(e)) {
        fprintf(stderr, "abi_smoke: FAIL model reports no chat template\n");
        stoke_close(e);
        return 1;
    }

    stoke_chat_message msgs[2];
    msgs[0].role        = "system";
    msgs[0].role_len    = 6;
    msgs[0].content     = "You are terse.";
    msgs[0].content_len = 14;
    msgs[1].role        = "user";
    msgs[1].role_len    = 4;
    msgs[1].content     = "Name one primary colour.";
    msgs[1].content_len = 24;

    int64_t bound = stoke_chat_template_bound(msgs, 2);
    if (bound <= 0) {
        fprintf(stderr, "abi_smoke: FAIL bound returned %lld\n", (long long)bound);
        stoke_close(e);
        return 1;
    }

    uint32_t chat_ids[MAX_IDS];
    if (bound > MAX_IDS) {
        fprintf(stderr, "abi_smoke: FAIL bound %lld exceeds the test buffer\n",
                (long long)bound);
        stoke_close(e);
        return 1;
    }

    /* Under the bound by one: refused, and we are still running. */
    int32_t too_small = stoke_apply_chat_template(e, msgs, 2, chat_ids,
                                                  (int32_t)bound - 1);
    if (too_small != STOKE_ERR_TOO_LONG) {
        fprintf(stderr, "abi_smoke: FAIL short cap gave %d, want %d\n",
                (int)too_small, (int)STOKE_ERR_TOO_LONG);
        stoke_close(e);
        return 1;
    }

    /* And misuse is enumerated, not a crash. */
    if (stoke_apply_chat_template(e, NULL, 2, chat_ids, MAX_IDS)
            != STOKE_ERR_INVALID_ARG ||
        stoke_apply_chat_template(e, msgs, 0, chat_ids, MAX_IDS)
            != STOKE_ERR_INVALID_ARG ||
        stoke_chat_template_bound(NULL, 1) >= 0) {
        fprintf(stderr, "abi_smoke: FAIL misuse was not refused as INVALID_ARG\n");
        stoke_close(e);
        return 1;
    }

    int32_t n_chat = stoke_apply_chat_template(e, msgs, 2, chat_ids,
                                               (int32_t)bound);
    if (n_chat <= 0) { stoke_close(e); return fail("stoke_apply_chat_template"); }
    if (n_chat > bound) {
        fprintf(stderr, "abi_smoke: FAIL wrote %d ids past a bound of %lld\n",
                (int)n_chat, (long long)bound);
        stoke_close(e);
        return 1;
    }

    /* The template must actually TEMPLATE: a plain encode of the same text
     * would be shorter and would carry no turn markers, so a stub returning
     * tok_encode's answer cannot pass this. */
    uint32_t plain_ids[MAX_IDS];
    int32_t  n_plain = stoke_tokenize(
        e, "You are terse.Name one primary colour.", -1, 0, plain_ids, MAX_IDS);
    if (n_chat <= n_plain) {
        fprintf(stderr, "abi_smoke: FAIL templated %d ids <= plain %d\n",
                (int)n_chat, (int)n_plain);
        stoke_close(e);
        return 1;
    }

    stoke_request creq;
    memset(&creq, 0, sizeof creq);
    creq.size       = sizeof creq;
    creq.id         = 2;
    creq.prompt_ids = chat_ids;
    creq.n_prompt   = (uint32_t)n_chat;
    creq.max_tokens = MAX_TOK;
    creq.eos_id     = (int64_t)stoke_eos_id(e);
    if (stoke_submit(e, &creq) != STOKE_OK) {
        stoke_close(e);
        return fail("stoke_submit(chat)");
    }

    uint32_t cgot[MAX_TOK];
    int32_t  n_cgot  = 0;
    int32_t  cfinish = STOKE_FINISH_NONE;
    int      csteps  = 0;
    while (cfinish == STOKE_FINISH_NONE && csteps < 4096) {
        uint32_t rows = 0;
        if (stoke_step(e, &rows) != STOKE_OK) {
            stoke_close(e);
            return fail("stoke_step(chat)");
        }
        csteps++;
        int32_t k = stoke_poll(e, creq.id, cgot + n_cgot, MAX_TOK - n_cgot,
                               &cfinish);
        if (k < 0) { stoke_close(e); return fail("stoke_poll(chat)"); }
        n_cgot += k;
    }
    if (cfinish == STOKE_FINISH_NONE || n_cgot == 0) {
        fprintf(stderr, "abi_smoke: FAIL chat produced %d tokens, finish=%d\n",
                (int)n_cgot, (int)cfinish);
        stoke_close(e);
        return 1;
    }

    char ctext[1024];
    size_t cused = 0;
    for (int32_t i = 0; i < n_cgot; i++) {
        char    piece[256];
        int32_t m = stoke_detokenize(e, cgot[i], piece, (int32_t)sizeof piece);
        if (m < 0) { stoke_close(e); return fail("stoke_detokenize(chat)"); }
        if ((size_t)m < sizeof piece && cused + (size_t)m < sizeof ctext - 1) {
            memcpy(ctext + cused, piece, (size_t)m);
            cused += (size_t)m;
        }
    }
    ctext[cused] = '\0';
    printf("abi_smoke: chat  %d prompt ids (bound %lld), %d tokens\n",
           (int)n_chat, (long long)bound, (int)n_cgot);
    printf("abi_smoke: chat  \"%s\"\n", ctext);

    /* ---- 8. an unknown id is an error, not a crash ---- */
    if (stoke_cancel(e, 999999) != STOKE_ERR_UNKNOWN_ID) {
        fprintf(stderr, "abi_smoke: FAIL cancel of an unknown id was not refused\n");
        stoke_close(e);
        return 1;
    }

    /* ---- 9. AND POLL SAYS SO TOO, which it did not until the A7 freeze
     * review. `stoke_poll` returned 0 with finish NONE for an id the scheduler
     * had never seen -- byte for byte what a LIVE request that has not produced
     * yet looks like. The documented loop is "poll until finish != NONE", so a
     * caller who ignored a failed stoke_submit (DUPLICATE_ID, TOO_LONG, a typo
     * in an id) waited forever on a request that did not exist. Cancel reported
     * it; poll did not; the asymmetry was invisible from the header.
     *
     * ⚠ THE SECOND HALF IS THE CONTROL AND IT IS NOT OPTIONAL. "Return
     * UNKNOWN_ID whenever nothing was drained" would pass the first check and
     * break every legitimate caller, because a request that is SUBMITTED but
     * not yet admitted has produced nothing either -- it lives on the submit
     * queue, which find() deliberately does not walk. So the pair is: an id
     * never seen is an ERROR, and an id seen but silent is 0/NONE. ---- */
    uint32_t pbuf[4];
    int32_t  pfin = -123;
    int32_t  prc  = stoke_poll(e, 888888u, pbuf, 4, &pfin);
    if (prc != STOKE_ERR_UNKNOWN_ID) {
        fprintf(stderr, "abi_smoke: FAIL poll of an id never submitted returned "
                        "%d, not STOKE_ERR_UNKNOWN_ID (%d) -- a caller that "
                        "ignored a failed submit would wait forever\n",
                (int)prc, (int)STOKE_ERR_UNKNOWN_ID);
        stoke_close(e);
        return 1;
    }

    stoke_request qreq;
    memset(&qreq, 0, sizeof qreq);
    qreq.size       = sizeof qreq;
    qreq.id         = 777u;
    qreq.prompt_ids = ids;
    qreq.n_prompt   = (uint32_t)n;
    qreq.max_tokens = 4;
    qreq.eos_id     = (int64_t)stoke_eos_id(e);
    if (stoke_submit(e, &qreq) != STOKE_OK) {
        stoke_close(e);
        return fail("stoke_submit(quiet)");
    }
    /* Polled with NO step in between: admitted nowhere, generated nothing. */
    pfin = -123;
    prc  = stoke_poll(e, 777u, pbuf, 4, &pfin);
    if (prc != 0 || pfin != STOKE_FINISH_NONE) {
        fprintf(stderr, "abi_smoke: FAIL poll of a submitted-but-unstepped id "
                        "returned %d finish %d, expected 0 / NONE -- reporting "
                        "UNKNOWN_ID here would break every correct caller\n",
                (int)prc, (int)pfin);
        stoke_close(e);
        return 1;
    }
    if (stoke_cancel(e, 777u) != STOKE_OK) {
        stoke_close(e);
        return fail("stoke_cancel(quiet)");
    }
    printf("abi_smoke: poll  unknown id -> UNKNOWN_ID, submitted-but-silent id -> 0/NONE\n");

    stoke_close(e);
    stoke_close(NULL); /* documented as safe */
    printf("abi_smoke: OK\n");
    return 0;
}
