/* tests/abi_sample.c -- M6 rung A6: sampling, driven THROUGH stoke.h.
 *
 * ▶▶ WHY THIS EXISTS SEPARATELY FROM tests/test_sample.c. That one tests the
 * CHOOSER: given logits, is the right id returned. It cannot see any of the
 * things that actually go wrong once a chooser is wired into an engine --
 * whether the params reached it, whether one request's params leaked into
 * another's, whether a finished request's slot ever came back. Those are
 * properties of the WIRING, and every one of them produces fluent text when
 * broken, which is why none of them can be caught by reading the output.
 *
 * ⚠ THE HARD PART OF TESTING A SAMPLER IS THAT ANY ANSWER LOOKS RIGHT. So
 * nothing below asserts that a token is good. Each case pins something a
 * BROKEN wiring must violate:
 *
 *   [A] greedy baseline -- and the anti-vacuity check that it repeats itself,
 *       because case [B] is only able to distinguish anything if it does.
 *   [B] repeat_penalty 100 at temperature 0: NO TOKEN MAY APPEAR TWICE. A
 *       deterministic assertion that a sampler which never received the params
 *       cannot satisfy.
 *   [C] two requests, same seed, RUNNING AT THE SAME TIME: identical streams.
 *       This is the (seed, pos) purity property from sample.h, end to end.
 *   [D] a different seed differs -- without which the seed is decorative and
 *       every "reproducible" claim resting on it is empty.
 *   [E] a greedy request running BESIDE a sampled one still gets [A] exactly.
 *       Catches params bleeding between slots.
 *   [F] eight sampled requests in a row, all matching [C]. The engine has only
 *       max_seqs sampling slots; if a finished request never gives its slot
 *       back, the later ones silently fall back to greedy. Silently is the
 *       whole problem -- the text stays fluent and nothing returns an error.
 *   [G] the same penalised request either side of a SHORTER one: identical. A
 *       slot handed on with its last occupant's history still in it changes the
 *       answer only at positions the new request never writes, which is why the
 *       shorter prompt is load-bearing rather than decorative.
 *
 * ⚠ IT INCLUDES NOTHING FROM src/ EXCEPT stoke.h, for the same reason
 * abi_smoke.c does: reaching around the header here would hide a defect IN the
 * header.
 *
 * Usage: abi_sample <model.gguf>
 */
#include "stoke.h"

#include <stdio.h>
#include <string.h>

/* Long enough that a greedy continuation repeats itself -- which case [A]
 * checks rather than assumes. */
#define PROMPT   "The capital of France is a city that"
#define MAX_TOK  24
#define MAX_IDS  512
#define MAX_RUN  4

/* max_seqs, and therefore the number of sampling slots. Small on purpose: case
 * [F] runs more sampled requests than there are slots, so a slot that is never
 * released turns into a wrong answer instead of into a bigger array.
 *
 * ⚠ THE VALUE IS ARITHMETIC, NOT A ROUND NUMBER, AND THE MUTATION LEG PROVED
 * IT. Cases [B]..[E] submit five sampled requests (1 + 2 + 1 + 1). At 4 this
 * file still caught a slot that is never reclaimed -- but it caught it in [E],
 * so [F], the case written for exactly that defect, never ran and stayed
 * unproven. One more than five puts the wall inside [F] where it belongs. A
 * future case that samples must raise this by the same amount, and
 * tests/tooling/run_m6_sample_mutants.sh fails loudly if it does not. */
#define MAX_SEQS 6

static stoke_engine* g_e;
static uint32_t      g_ids[MAX_IDS];
static int32_t       g_n_ids;

static int fail(const char* what) {
    fprintf(stderr, "abi_sample: FAIL %s: %s\n", what, stoke_last_error());
    return 1;
}

typedef struct {
    uint64_t id;
    uint32_t n_prompt;   /* 0 => the whole prompt. Case [G] needs two lengths. */
    float    temperature;
    uint32_t top_k;
    float    top_p;
    float    repeat_penalty;
    uint64_t seed;

    uint32_t got[MAX_TOK];
    int32_t  n_got;
    int32_t  finish;
} Run;

/* Submits every run, then steps ONCE for all of them -- so the concurrent cases
 * really are concurrent rather than a sequence dressed up as one. */
static int drive(Run* r, int n) {
    for (int i = 0; i < n; i++) {
        stoke_request q;
        memset(&q, 0, sizeof q);
        q.size           = sizeof q;
        q.id             = r[i].id;
        q.prompt_ids     = g_ids;
        q.n_prompt       = r[i].n_prompt ? r[i].n_prompt : (uint32_t)g_n_ids;
        q.max_tokens     = MAX_TOK;
        /* ▶ EOS DISABLED. A hot temperature can draw the end token, and a run
         * that stopped after two tokens would compare equal to another that
         * also stopped after two -- a pass carrying almost no information. Every
         * run here reaches MAX_TOK, so every comparison is over 24 tokens. */
        q.eos_id         = -1;
        q.temperature    = r[i].temperature;
        q.top_k          = r[i].top_k;
        q.top_p          = r[i].top_p;
        q.repeat_penalty = r[i].repeat_penalty;
        q.seed           = r[i].seed;

        r[i].n_got  = 0;
        r[i].finish = STOKE_FINISH_NONE;
        if (stoke_submit(g_e, &q) != STOKE_OK) {
            fprintf(stderr, "abi_sample: submit(id=%llu): %s\n",
                    (unsigned long long)r[i].id, stoke_last_error());
            return -1;
        }
    }

    for (int steps = 0; steps < 4096; steps++) {
        int outstanding = 0;
        for (int i = 0; i < n; i++) {
            if (r[i].finish == STOKE_FINISH_NONE) outstanding = 1;
        }
        if (!outstanding) return 0;

        uint32_t rows = 0;
        if (stoke_step(g_e, &rows) != STOKE_OK) return -1;
        for (int i = 0; i < n; i++) {
            if (r[i].finish != STOKE_FINISH_NONE) continue;
            int32_t k = stoke_poll(g_e, r[i].id, r[i].got + r[i].n_got,
                                   MAX_TOK - r[i].n_got, &r[i].finish);
            if (k < 0) return -1;
            r[i].n_got += k;
        }
    }
    fprintf(stderr, "abi_sample: %d run(s) did not finish in 4096 steps\n", n);
    return -1;
}

static int same(const Run* a, const Run* b) {
    if (a->n_got != b->n_got) return 0;
    for (int32_t i = 0; i < a->n_got; i++) {
        if (a->got[i] != b->got[i]) return 0;
    }
    return 1;
}

/* Index of the first token that appears twice, or -1. */
static int32_t first_repeat(const Run* r) {
    for (int32_t i = 0; i < r->n_got; i++) {
        for (int32_t j = 0; j < i; j++) {
            if (r->got[i] == r->got[j]) return i;
        }
    }
    return -1;
}

static void show(const char* tag, const Run* r) {
    char   text[1024];
    size_t used = 0;
    for (int32_t i = 0; i < r->n_got; i++) {
        char    piece[256];
        int32_t m = stoke_detokenize(g_e, r->got[i], piece, (int32_t)sizeof piece);
        if (m > 0 && (size_t)m < sizeof piece && used + (size_t)m < sizeof text - 1) {
            memcpy(text + used, piece, (size_t)m);
            used += (size_t)m;
        }
    }
    text[used] = '\0';
    printf("abi_sample: %-9s %2d tok  \"%s\"\n", tag, (int)r->n_got, text);
}

/* Every case fills this in so a full run reports what it actually checked. A
 * gate that cannot say which of its cases ran is one skipped case away from
 * being green for the wrong reason. */
static int g_cases;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: abi_sample <model.gguf>\n");
        return 2;
    }

    stoke_open_opts opts;
    memset(&opts, 0, sizeof opts);
    opts.size     = sizeof opts;
    opts.max_ctx  = 512;
    opts.max_seqs = MAX_SEQS;

    g_e = stoke_open(argv[1], &opts);
    if (!g_e) return fail("stoke_open");

    g_n_ids = stoke_tokenize(g_e, PROMPT, -1, 0, g_ids, MAX_IDS);
    if (g_n_ids <= 0) { stoke_close(g_e); return fail("stoke_tokenize"); }

    /* ---- [A] the greedy baseline ---------------------------------------- */
    Run greedy;
    memset(&greedy, 0, sizeof greedy);
    greedy.id = 1;
    if (drive(&greedy, 1) != 0) { stoke_close(g_e); return fail("drive(greedy)"); }
    if (greedy.n_got != MAX_TOK) {
        fprintf(stderr, "abi_sample: FAIL [A] greedy produced %d of %d tokens\n",
                (int)greedy.n_got, MAX_TOK);
        stoke_close(g_e);
        return 1;
    }
    show("greedy", &greedy);

    /* ⚠ ANTI-VACUITY FOR [B], AND IT IS NOT A CHECK ON THE ENGINE. Case [B]
     * asserts that a penalised run repeats no token. If the greedy run happens
     * not to repeat one either, [B] would pass against a sampler that ignored
     * the penalty entirely -- the fixture, not the code, would be what made it
     * green. So the fixture proves it can tell the two apart before [B] runs. */
    if (first_repeat(&greedy) < 0) {
        fprintf(stderr,
                "abi_sample: FAIL [A] the greedy continuation repeats no token "
                "in %d, so case [B] could not distinguish a penalty that works "
                "from one that was never applied. This is a FIXTURE problem: "
                "lengthen MAX_TOK or change PROMPT.\n", MAX_TOK);
        stoke_close(g_e);
        return 1;
    }
    printf("abi_sample: [A] greedy repeats a token at index %d "
           "(so [B] can distinguish)\n", (int)first_repeat(&greedy));
    g_cases++;

    /* ---- [B] repeat_penalty must make repetition impossible -------------- */
    Run pen;
    memset(&pen, 0, sizeof pen);
    pen.id             = 2;
    pen.repeat_penalty = 100.0f; /* temperature stays 0: greedy + penalty, so
                                  * the effect is deterministic, not a draw */
    if (drive(&pen, 1) != 0) { stoke_close(g_e); return fail("drive(penalty)"); }
    show("penalty", &pen);
    if (pen.n_got != MAX_TOK) {
        fprintf(stderr, "abi_sample: FAIL [B] produced %d of %d tokens\n",
                (int)pen.n_got, MAX_TOK);
        stoke_close(g_e);
        return 1;
    }
    int32_t rep = first_repeat(&pen);
    if (rep >= 0) {
        fprintf(stderr,
                "abi_sample: FAIL [B] token %u repeats at index %d under a "
                "repeat_penalty of 100 -- the penalty did not reach the "
                "sampler, or the history is not being recorded\n",
                pen.got[rep], (int)rep);
        stoke_close(g_e);
        return 1;
    }
    if (same(&pen, &greedy)) {
        fprintf(stderr, "abi_sample: FAIL [B] penalised run is identical to "
                        "greedy\n");
        stoke_close(g_e);
        return 1;
    }
    printf("abi_sample: [B] 24 penalised tokens, none repeated, != greedy\n");
    g_cases++;

    /* ---- [C] same seed, at the same time, identical --------------------- */
    Run pair[2];
    memset(pair, 0, sizeof pair);
    pair[0].id = 3; pair[0].temperature = 2.5f; pair[0].seed = 20260829u;
    pair[1].id = 4; pair[1].temperature = 2.5f; pair[1].seed = 20260829u;
    if (drive(pair, 2) != 0) { stoke_close(g_e); return fail("drive(pair)"); }
    show("seeded", &pair[0]);
    if (pair[0].n_got != MAX_TOK || pair[1].n_got != MAX_TOK) {
        fprintf(stderr, "abi_sample: FAIL [C] got %d and %d tokens\n",
                (int)pair[0].n_got, (int)pair[1].n_got);
        stoke_close(g_e);
        return 1;
    }
    if (!same(&pair[0], &pair[1])) {
        fprintf(stderr,
                "abi_sample: FAIL [C] two requests with seed %llu running "
                "together produced different streams -- the draw is not a pure "
                "function of (seed, position)\n",
                (unsigned long long)pair[0].seed);
        stoke_close(g_e);
        return 1;
    }
    printf("abi_sample: [C] two concurrent requests, seed %llu, identical\n",
           (unsigned long long)pair[0].seed);
    g_cases++;

    /* ---- [D] a different seed must actually differ ---------------------- */
    Run other;
    memset(&other, 0, sizeof other);
    other.id = 5; other.temperature = 2.5f; other.seed = 20260830u;
    if (drive(&other, 1) != 0) { stoke_close(g_e); return fail("drive(other)"); }
    if (same(&other, &pair[0])) {
        fprintf(stderr, "abi_sample: FAIL [D] seed %llu gave the same 24 tokens "
                        "as seed %llu -- the seed is not reaching the sampler\n",
                (unsigned long long)other.seed,
                (unsigned long long)pair[0].seed);
        stoke_close(g_e);
        return 1;
    }
    if (same(&other, &greedy)) {
        fprintf(stderr, "abi_sample: FAIL [D] a temperature-2.5 run is "
                        "byte-identical to greedy\n");
        stoke_close(g_e);
        return 1;
    }
    printf("abi_sample: [D] a different seed gives a different stream\n");
    g_cases++;

    /* ---- [E] a greedy request beside a sampled one is untouched ---------- */
    Run mix[2];
    memset(mix, 0, sizeof mix);
    mix[0].id = 6;                                    /* greedy */
    mix[1].id = 7; mix[1].temperature = 2.5f; mix[1].seed = 20260829u;
    if (drive(mix, 2) != 0) { stoke_close(g_e); return fail("drive(mix)"); }
    if (!same(&mix[0], &greedy)) {
        fprintf(stderr,
                "abi_sample: FAIL [E] a greedy request running beside a sampled "
                "one did not reproduce the greedy baseline -- sampling params "
                "are leaking across requests\n");
        stoke_close(g_e);
        return 1;
    }
    if (!same(&mix[1], &pair[0])) {
        fprintf(stderr,
                "abi_sample: FAIL [E] the sampled half of the mixed batch did "
                "not reproduce [C]'s stream for the same seed\n");
        stoke_close(g_e);
        return 1;
    }
    printf("abi_sample: [E] greedy and sampled side by side, both unchanged\n");
    g_cases++;

    /* ---- [F] slots come back -------------------------------------------- *
     * ▶▶ THE CASE THIS FILE WAS WRITTEN FOR. There are MAX_SEQS sampling
     * slots and eleven sampled requests have now run. If a finished request's
     * slot is never reclaimed, the fifth onwards finds none free and runs
     * GREEDY -- no error, no warning, fluent text, wrong answer. Each of these
     * carries [C]'s seed, so a fallback to greedy is exactly what is caught. */
    for (int i = 0; i < 8; i++) {
        Run again;
        memset(&again, 0, sizeof again);
        again.id          = 100 + (uint64_t)i;
        again.temperature = 2.5f;
        again.seed        = 20260829u;
        if (drive(&again, 1) != 0) { stoke_close(g_e); return fail("drive(F)"); }
        if (!same(&again, &pair[0])) {
            fprintf(stderr,
                    "abi_sample: FAIL [F] sampled request %d of 8 (id %llu) did "
                    "not reproduce [C]'s stream%s\n",
                    i + 1, (unsigned long long)again.id,
                    same(&again, &greedy)
                        ? " -- and it IS the greedy stream, so it ran with no "
                          "sampling slot"
                        : "");
            stoke_close(g_e);
            return 1;
        }
    }
    printf("abi_sample: [F] 8 more sampled requests, all reproduced [C]\n");
    g_cases++;

    /* ---- [G] a reused slot carries nothing from its last occupant -------- *
     *
     * ▶▶ WHY THIS NEEDS TWO PROMPT LENGTHS, which is the whole subtlety. The
     * history is indexed by ABSOLUTE position and a request overwrites every
     * position it generates at -- so a second request with the SAME prompt
     * covers its predecessor's entries before it ever reads them, and the two
     * are indistinguishable however dirty the slot is. The only positions a
     * request reads but never writes are the ones BELOW its own first generated
     * position. So a shorter prompt has to run in between to put something
     * there. Without that, this case grades nothing at all -- which is exactly
     * the shape of fixture this project has been bitten by before.
     *
     * The penalty is what makes those entries observable: temperature alone
     * never looks at the history. */
    Run g1, g2, g3;
    memset(&g1, 0, sizeof g1);
    g1.id = 200; g1.repeat_penalty = 100.0f;
    if (drive(&g1, 1) != 0) { stoke_close(g_e); return fail("drive(G1)"); }

    memset(&g2, 0, sizeof g2);
    g2.id = 201; g2.repeat_penalty = 100.0f;
    g2.n_prompt = (uint32_t)g_n_ids / 2u;   /* dirties the low positions */
    if (g2.n_prompt < 2) {
        fprintf(stderr, "abi_sample: FAIL [G] the prompt is too short to halve; "
                        "this case would grade nothing\n");
        stoke_close(g_e);
        return 1;
    }
    if (drive(&g2, 1) != 0) { stoke_close(g_e); return fail("drive(G2)"); }

    memset(&g3, 0, sizeof g3);
    g3.id = 202; g3.repeat_penalty = 100.0f;
    if (drive(&g3, 1) != 0) { stoke_close(g_e); return fail("drive(G3)"); }

    if (!same(&g1, &g3)) {
        fprintf(stderr,
                "abi_sample: FAIL [G] the same penalised request gave different "
                "answers either side of a shorter one -- the slot kept the "
                "previous occupant's history below position %u\n",
                g2.n_prompt);
        stoke_close(g_e);
        return 1;
    }
    if (!same(&g1, &pen)) {
        fprintf(stderr, "abi_sample: FAIL [G] the penalised baseline itself "
                        "changed since [B]\n");
        stoke_close(g_e);
        return 1;
    }
    printf("abi_sample: [G] a slot reused after a %u-token prompt is clean\n",
           g2.n_prompt);
    g_cases++;

    if (g_cases != 7) {
        fprintf(stderr, "abi_sample: FAIL only %d of 6 cases ran\n", g_cases);
        stoke_close(g_e);
        return 1;
    }

    stoke_close(g_e);
    printf("abi_sample: OK (%d cases)\n", g_cases);
    return 0;
}
