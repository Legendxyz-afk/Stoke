/* tests/tooling/llama_logits.c -- a one-shot logit oracle on the pinned
 * llama.cpp build: load the GGUF through llama.dll, teacher-force an id
 * stream in one batch, print full-precision logits at chosen positions.
 *
 * WHY. The 08-05 ruling demotes llama.cpp to an optional SURVEY, and the first
 * survey (2026-09-15, B6 report section 8) needed exact logits at a handful of
 * positions. `llama-perplexity --save-all-logits` is the only stock one-shot
 * that emits logits, and it quantises them into a 16-logit window below the
 * max -- so a margin of 26 reads as "16, clamped", which is exactly the
 * quantity the survey was after. llama-server would give them exactly but is a
 * long-lived server on a 16 GB box (ask-first rule). This program loads,
 * computes, prints and exits.
 *
 * WHAT IT DOES NOT DO. Tokenize (the ids come from a file, one per line -- the
 * same `*.stream.ids` the reference is teacher-forced on, so all three arms see
 * the SAME ids), sample, or touch any llama_context_params field beyond the
 * leading integers. The header (llama.cpp-master snapshot) is slightly NEWER
 * than the b10037 DLL (four functions the DLL lacks), so struct layouts past
 * the leading fields are not trusted: llama_model_params is passed back
 * unmodified, and the context params are checked at runtime by dumping the
 * DLL's own defaults (the run of -1 enums must start where the header says).
 *
 * Usage:
 *   llama_logits <model.gguf> <ids-file> <threads> <pos>[:id,id,...] [<pos>[:ids] ...]
 * Output, one line per requested position:
 *   pos=<p> top=<id> logit=<f> | <id>:<logit> ... (top 10 by logit, descending)
 *   pos=<p> ask <id>:<logit> ...  (the explicitly requested ids, full precision)
 * The batch requests logits at exactly the given positions; position p's
 * logits are the distribution after consuming ids[p] (same convention as the
 * grid's abs_position).
 *
 * Build (mingw, from the repo root; the DLLs are the vendor build's own):
 *   gcc -O1 -o build/b6/survey/llama_logits.exe tests/tooling/llama_logits.c \
 *       -I ../llama.cpp-master/include -I ../llama.cpp-master/ggml/include \
 *       -L bench/vendor/llama-b10037 -lllama -lggml -lggml-base
 * Run with bench/vendor/llama-b10037 on PATH and GGML_BACKEND_DIR pointing at it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "llama.h"
#include "ggml-backend.h"

#define MAX_POS 256
#define MAX_ASK 64
#define TOPN 10

static int cmp_desc(const void* a, const void* b) {
    const float fa = ((const float*)a)[1], fb = ((const float*)b)[1];
    return (fa < fb) - (fa > fb);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <ids-file> <threads> <pos>[:id,...] ...\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    const char* ids_path = argv[2];
    int n_threads = atoi(argv[3]);

    /* ---- the ids ---- */
    FILE* f = fopen(ids_path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", ids_path); return 2; }
    int cap = 1 << 16, n_ids = 0;
    llama_token* ids = (llama_token*)malloc(sizeof(llama_token) * cap);
    long long v;
    while (fscanf(f, "%lld", &v) == 1) {
        if (n_ids == cap) { cap *= 2; ids = (llama_token*)realloc(ids, sizeof(llama_token) * cap); }
        ids[n_ids++] = (llama_token)v;
    }
    fclose(f);
    if (n_ids == 0) { fprintf(stderr, "no ids read\n"); return 2; }

    /* ---- the requested positions and their asked ids ---- */
    int n_pos = 0, pos[MAX_POS], n_ask[MAX_POS], ask[MAX_POS][MAX_ASK];
    for (int a = 4; a < argc && n_pos < MAX_POS; a++) {
        char* s = argv[a];
        char* colon = strchr(s, ':');
        pos[n_pos] = atoi(s);
        n_ask[n_pos] = 0;
        if (colon) {
            char* t = colon + 1;
            while (*t && n_ask[n_pos] < MAX_ASK) {
                ask[n_pos][n_ask[n_pos]++] = atoi(t);
                char* comma = strchr(t, ',');
                if (!comma) break;
                t = comma + 1;
            }
        }
        if (pos[n_pos] < 0 || pos[n_pos] >= n_ids) {
            fprintf(stderr, "position %d outside 0..%d\n", pos[n_pos], n_ids - 1);
            return 2;
        }
        n_pos++;
    }

    /* ---- backend + model: model params passed back UNTOUCHED (layout not trusted) ----
     * The CPU backend is a separate ggml-cpu-*.dll picked at load time; load_all()
     * looks beside THIS executable, which is not the vendor dir, so the vendor
     * dir is named explicitly (GGML_BACKEND_DIR) and the load is checked. */
    const char* bdir = getenv("GGML_BACKEND_DIR");
    if (bdir && *bdir) ggml_backend_load_all_from_path(bdir); else ggml_backend_load_all();
    if (ggml_backend_dev_count() == 0) {
        fprintf(stderr, "no ggml backend loaded (set GGML_BACKEND_DIR to the vendor dir)\n");
        return 3;
    }
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 3; }
    const struct llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    /* ---- context params: only the leading integer fields are written, and the
     * DLL's own defaults are dumped so a layout drift shows instead of hiding ---- */
    struct llama_context_params cp = llama_context_default_params();
    {
        const uint32_t* w = (const uint32_t*)&cp;
        fprintf(stderr, "ctx defaults (leading words): ");
        for (int i = 0; i < 14; i++) fprintf(stderr, "%d ", (int)w[i]);
        fprintf(stderr, "\n");
        /* header: n_ctx n_batch n_ubatch n_seq_max n_rs_seq n_outputs_max n_threads n_threads_batch
         *         ctx_type rope_scaling pooling attention flash_attn ...
         * so words 6,7 are the thread counts and word 8 must be ctx_type=0 followed by four -1s. */
        if (!((int)w[8] == 0 && (int)w[9] == -1 && (int)w[10] == -1 && (int)w[11] == -1 && (int)w[12] == -1)) {
            fprintf(stderr, "LAYOUT CHECK FAILED: the DLL's llama_context_params does not match the header "
                            "past the thread counts -- refusing to run\n");
            return 4;
        }
    }
    cp.n_ctx = (uint32_t)n_ids;
    cp.n_batch = (uint32_t)n_ids;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    struct llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 3; }
    fprintf(stderr, "n_ctx=%u n_batch=%u n_ubatch=%u threads=%d n_vocab=%d n_ids=%d\n",
            llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx), n_threads, n_vocab, n_ids);
    if (llama_n_ctx(ctx) < (uint32_t)n_ids) { fprintf(stderr, "n_ctx did not take\n"); return 4; }

    /* ---- one batch, logits only where asked ---- */
    struct llama_batch batch = llama_batch_init(n_ids, 0, 1);
    for (int i = 0; i < n_ids; i++) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    for (int k = 0; k < n_pos; k++) batch.logits[pos[k]] = 1;
    batch.n_tokens = n_ids;
    int rc = llama_decode(ctx, batch);
    if (rc != 0) { fprintf(stderr, "llama_decode returned %d\n", rc); return 5; }

    /* ---- print ---- */
    float* top = (float*)malloc(sizeof(float) * 2 * n_vocab);
    for (int k = 0; k < n_pos; k++) {
        const float* logits = llama_get_logits_ith(ctx, pos[k]);
        if (!logits) { fprintf(stderr, "no logits at %d\n", pos[k]); return 5; }
        for (int i = 0; i < n_vocab; i++) { top[2 * i] = (float)i; top[2 * i + 1] = logits[i]; }
        qsort(top, n_vocab, 2 * sizeof(float), cmp_desc);
        printf("pos=%d top=%d logit=%.6f |", pos[k], (int)top[0], (double)top[1]);
        for (int i = 0; i < TOPN; i++) printf(" %d:%.6f", (int)top[2 * i], (double)top[2 * i + 1]);
        printf("\n");
        if (n_ask[k]) {
            printf("pos=%d ask", pos[k]);
            for (int j = 0; j < n_ask[k]; j++) {
                int id = ask[k][j];
                if (id < 0 || id >= n_vocab) printf(" %d:NA", id);
                else printf(" %d:%.6f", id, (double)logits[id]);
            }
            printf("\n");
        }
    }
    free(top);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(ids);
    return 0;
}
