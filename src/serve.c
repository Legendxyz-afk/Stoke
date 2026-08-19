/* src/serve.c -- the model-backed SchedHost. See serve.h. */
#include "common.h"

#include "serve.h"

#include "kernels/kernels.h"

struct ServeHost {
    Model* m;
    u32    n_embd, n_vocab;

    /* Sized to the widest batch seen, grown on demand, never shrunk -- the step
     * must not allocate. */
    f32*  xbuf;  /* [cap][n_embd]  hidden state out of the forward   */
    f32** xp;    /* [cap]                                            */
    f32** yp;    /* [cap]  the subset of rows that yield an id       */
    f32*  lbuf;  /* [cap][n_vocab] logits                            */
    f32** lp;    /* [cap]                                            */
    u32   cap;

    u64 rows, steps;
};

static void ensure(ServeHost* h, u32 n) {
    if (n <= h->cap) return;
    h->xbuf = (f32*)xrealloc(h->xbuf, (size_t)n * h->n_embd * sizeof(f32));
    h->lbuf = (f32*)xrealloc(h->lbuf, (size_t)n * h->n_vocab * sizeof(f32));
    h->xp = (f32**)xrealloc(h->xp, (size_t)n * sizeof(f32*));
    h->yp = (f32**)xrealloc(h->yp, (size_t)n * sizeof(f32*));
    h->lp = (f32**)xrealloc(h->lp, (size_t)n * sizeof(f32*));
    h->cap = n;
    /* EVERY pointer is rebuilt, not just the new ones: the reallocs above may
     * have MOVED both buffers, which leaves every previously stored row pointer
     * dangling. Same hazard, and same answer, as model.c's scratch_n_ensure --
     * reachable only by a batch that grows, so a fixed-width test never sees
     * it. */
    for (u32 i = 0; i < n; i++) {
        h->xp[i] = h->xbuf + (size_t)i * h->n_embd;
        h->lp[i] = h->lbuf + (size_t)i * h->n_vocab;
    }
}

/* ▶▶ WHICH ROWS YIELD AN ID, AND WHY IT IS NOT "ALL OF THEM". A step carries
 * every runnable decode (rows [0, n_decode)) plus at most one prefill chunk
 * (the rest). A decode row always yields the next token. A prefill row yields
 * nothing EXCEPT the chunk's last row, and only when that chunk completes the
 * prompt -- the interior of a prompt is being ingested, not predicted. The
 * scheduler states both facts in SchedStep rather than leaving this file to
 * infer them from row counts, because inferring them is how a host ends up
 * sampling a token from the middle of somebody's prompt. */
static void serve_step(void* ctx, const SchedStep* in, u32* out) {
    ServeHost* h = (ServeHost*)ctx;
    if (in->n_rows == 0) return;
    ensure(h, in->n_rows);

    model_forward_batch(h->m, in->kv, in->token_ids, in->positions, in->n_rows,
                        h->xp);

    u32 n_yield = in->n_decode;
    for (u32 i = 0; i < in->n_decode; i++) h->yp[i] = h->xp[i];
    if (in->prefill_completes) h->yp[n_yield++] = h->xp[in->n_rows - 1u];

    if (n_yield > 0) {
        /* ONE projection for the whole yielding set. token_embd is 319 MB on
         * the 4B model, so per-row here would cost more than the rest of the
         * forward -- see model_compute_logits_batch. */
        model_compute_logits_batch(h->m, (const f32* const*)h->yp, n_yield,
                                   h->lp);
        for (u32 i = 0; i < n_yield; i++)
            out[i] = argmax(h->lp[i], h->n_vocab);
    }

    h->rows += in->n_rows;
    h->steps++;
}

static double serve_now(void* ctx) {
    (void)ctx;
    return now_ms();
}

ServeHost* serve_host_open(Model* m) {
    if (!m) fatal("serve: serve_host_open -- model must be non-NULL");
    ServeHost* h = (ServeHost*)xcalloc(1, sizeof(ServeHost));
    h->m = m;
    h->n_embd = model_n_embd(m);
    h->n_vocab = model_n_vocab(m);
    return h;
}

void serve_host_close(ServeHost* h) {
    if (!h) return;
    free(h->xbuf); free(h->lbuf); free(h->xp); free(h->yp); free(h->lp);
    free(h);
}

SchedHost serve_host(ServeHost* h) {
    SchedHost sh;
    sh.step = serve_step;
    sh.now_ms = serve_now;
    sh.ctx = h;
    return sh;
}

u64 serve_host_rows(const ServeHost* h) { return h->rows; }
u64 serve_host_steps(const ServeHost* h) { return h->steps; }
