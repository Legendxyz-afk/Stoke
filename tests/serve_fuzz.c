/* tests/serve_fuzz.c -- the randomized serve smoke. M6 GATE ITEM [7].
 *
 * ▶▶▶ WHAT THIS IS FOR, AND WHY IT IS NOT A CORRECTNESS TEST. Every other serve
 * gate asserts what the server SAYS: the golden pins bytes, the openai suite
 * pins shapes, the corpus pins verdicts. This one asserts almost nothing about
 * the answers. It exists to keep the server BUSY and VARIED for ten minutes
 * while ASan watches it from the inside -- the failures it is hunting are a
 * use-after-free on a path only a client disconnect reaches, a read past the
 * end of a request buffer, a leak per rejected connection. Those do not show up
 * as a wrong answer. They show up as nothing at all, until they show up as a
 * crash in front of someone else.
 *
 * So the only things it treats as its OWN failures are: the server stopped
 * accepting connections, or it never answered anything. A 400 is a pass. A 404
 * is a pass. A refused connection is not.
 *
 * ▶▶ REPRODUCIBLE BY CONSTRUCTION. Every decision comes from a splitmix64
 * stream seeded per client from one master seed, which is PRINTED on the first
 * line. A failure at minute nine of a randomized run is worthless if you cannot
 * run it again; with the seed it is `--seed <n>` and the same traffic.
 *
 * ▶ THE TRAFFIC IS DELIBERATELY HOSTILE, because the polite paths are already
 * covered by three other gates. It sends: valid chat and completions (streamed
 * and not, with and without stop strings and usage), malformed JSON, the A2
 * garbage corpus verbatim, requests truncated mid-header, pipelined pairs,
 * bodies whose Content-Length lies, clients that vanish mid-stream, and clients
 * that connect and say nothing at all.
 *
 * Usage: serve_fuzz <host> <port> <seconds> <corpus-manifest> [clients] [seed]
 */
#include "common.h"

#include "net.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CLIENTS   16
#define MAX_CORPUS    128
#define CORPUS_MAX_LEN 65536

typedef struct {
    char*  data;
    size_t len;
} Blob;

static Blob     g_corpus[MAX_CORPUS];
static int      g_n_corpus;
static char     g_host[128];
static u16      g_port;
static double   g_deadline_ms;
static uint64_t g_seed;

/* How long a client waits for an answer, and how many tokens it asks for. Both
 * are set by the CALLER because the same driver runs against a -O3 Windows
 * build and against an ASan build where every forward pass is an order of
 * magnitude slower. At the first ASan pre-flight, 3 clients managed ONE
 * answered exchange out of 35 actions: every read hit its 5-second deadline
 * while the server was still inside a step. A soak that answers nothing
 * produces a clean sanitizer log and proves nothing -- which this driver
 * already refuses to call a pass, so these knobs exist to keep that refusal
 * from being the normal outcome on the slow leg. */
static double   g_read_ms = 5000.0;
static uint32_t g_max_tok = 12;

/* splitmix64 -- a PURE function of its state, which is what makes a replay a
 * replay. Same reason the forward note wants it for seeded sampling. */
static uint64_t sm64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static uint32_t pick(uint64_t* s, uint32_t n) { return (uint32_t)(sm64(s) % n); }

typedef struct {
    int      id;
    uint64_t rng;
    long     exchanges; /* completed request/response pairs                  */
    long     refused;   /* connections the server would not accept -- FATAL  */
    long     actions;
    /* ⚠ WHAT THE TRAFFIC ACTUALLY CONTAINED, REPORTED RATHER THAN ASSUMED. The
     * sampling half of this driver exists to give a sanitized soak some
     * coverage of the sampler, and "the parameters are in the format string" is
     * not evidence that any request carried one: a mis-built body is answered
     * with a 400, which this driver counts as a perfectly good outcome. The
     * summary prints these so the coverage claim is checkable from the run's
     * own output, on the leg where it actually ran. */
    long     sampled;   /* generating requests that asked for a sampler       */
    long     penalized; /* ...and the subset that asked for a repeat penalty  */
} Client;

/* Sends `req` and reads until the peer closes or a short deadline passes. The
 * reply is DISCARDED: what it says is not this gate's business. */
static void exchange(Client* c, const char* req, size_t len, int read_some) {
    NetSock s = net_connect(g_host, g_port);
    if (s == NET_INVALID_SOCK) {
        c->refused++;
        return;
    }
    if (net_send_all(s, req, len) != 0) {
        net_close(s);
        return;
    }

    if (read_some) {
        net_set_nonblocking(s);
        char   buf[8192];
        double t0 = now_ms();
        int    got = 0;
        /* ⚠ FIVE SECONDS, NOT TWENTY, AND THE REQUESTS ASK FOR A CLOSE. The
         * first version waited out its whole deadline on every SUCCESS: the
         * server keeps the connection alive after answering, so there is
         * nothing further to read and nothing to signal that. Three clients
         * managed 17 actions in 30 seconds, which is not a soak. Asking for
         * `Connection: close` on most requests turns the end of a response into
         * an EOF the client can actually see -- and exercises the close path
         * while it is at it. */
        while (now_ms() - t0 < g_read_ms) {
            NetWait w;
            memset(&w, 0, sizeof w);
            w.s         = s;
            w.want_read = 1;
            if (net_wait(&w, 1, 100) == NET_ERR) break;
            if (!w.can_read) continue;
            i64 n = net_recv(s, buf, sizeof buf);
            if (n == NET_WOULD_BLOCK) continue;
            if (n <= 0) break;
            got = 1;
            /* A client that vanishes mid-stream is the interesting case: the
             * server must cancel the request and free its blocks on a path
             * nothing else in the matrix reaches. */
            if (pick(&c->rng, 8) == 0) break;
        }
        if (got) c->exchanges++;
    }
    net_close(s);
}

static void build_chat(Client* c, char* out, size_t cap, int completions) {
    static const char* prompts[] = {
        "Say hello.", "Name one color.", "Count to three.",
        "What is 2+2?", "Describe water in five words.",
        "\\u00e9\\u00e8 caf\\u00e9 na\\u00efve",  /* escapes the parser must resolve */
        "line one\\nline two\\ttabbed",
    };
    static const char* stops[] = {"\".\"", "\"!\"", "\" the\"", "\"z\""};

    /* ⚠ THE SAMPLING TABLES EXIST BECAUSE THIS FUZZER SENT `"temperature":0` ON
     * EVERY REQUEST UNTIL 2026-08-30, so a ten-minute sanitized soak -- gate
     * item [7]'s entire memory-safety evidence -- never once entered
     * sample_token, the slot table, or the prologue's drain. The newest code in
     * the milestone was the least covered code in it. Values are all VALID on
     * purpose: an invalid one is answered with a 400, which this fuzzer counts
     * as a fine outcome and which reaches none of the machinery. Malformed
     * sampling input is the A2 corpus's job. The boundaries are here (top_k at
     * the sampler's 512 cap, repeat_penalty at 100) because a boundary that is
     * accepted is exactly where an off-by-one lives. */
    static const char* temps[] = {"0.7", "1.0", "1.4", "2.0"};
    static const char* topks[] = {"1", "8", "40", "512"};
    static const char* topps[] = {"0.1", "0.5", "0.9", "1"};
    static const char* pens[]  = {"1.05", "1.3", "1.8", "100"};

    uint32_t max_tok = 1 + pick(&c->rng, g_max_tok);
    int      stream  = (int)pick(&c->rng, 2);
    int      usage   = stream && pick(&c->rng, 2);
    int      use_stop = (int)pick(&c->rng, 3) == 0;
    const char* p    = prompts[pick(&c->rng, sizeof prompts / sizeof prompts[0])];

    /* ⚠⚠ EVERY DRAW BELOW HAPPENS WHATEVER THE KIND SELECTS, for the same
     * reason the stop string is drawn unconditionally: the RNG stream must
     * advance identically down every branch or "reproducible from the printed
     * seed" stops being true the moment a branch is taken differently. */
    uint32_t samp_kind = pick(&c->rng, 4);
    const char* t_s    = temps[pick(&c->rng, 4)];
    const char* k_s    = topks[pick(&c->rng, 4)];
    const char* p_s    = topps[pick(&c->rng, 4)];
    const char* pen_s  = pens[pick(&c->rng, 4)];
    uint32_t    seed_v = pick(&c->rng, 1000000);

    if (samp_kind != 0) c->sampled++;
    if (samp_kind == 3) c->penalized++;

    char samp[128];
    switch (samp_kind) {
    case 0: /* greedy, spelled out -- the explicit-zero parse path */
        snprintf(samp, sizeof samp, ",\"temperature\":0");
        break;
    case 1: /* a temperature alone: the sampler with its default candidate cap */
        snprintf(samp, sizeof samp, ",\"temperature\":%s,\"seed\":%u", t_s, seed_v);
        break;
    case 2: /* both truncations, including the ones that collapse to the argmax */
        snprintf(samp, sizeof samp,
                 ",\"temperature\":%s,\"top_k\":%s,\"top_p\":%s,\"seed\":%u",
                 t_s, k_s, p_s, seed_v);
        break;
    default: /* the penalty on the GREEDY path, which copies the whole logit row */
        snprintf(samp, sizeof samp, ",\"temperature\":0,\"repeat_penalty\":%s", pen_s);
        break;
    }

    /* ⚠ DRAWN BEFORE THE snprintf, NOT INSIDE ITS ARGUMENT LIST. C leaves the
     * evaluation ORDER of function arguments unspecified, and every pick()
     * advances the client's RNG -- so two draws in one argument list make the
     * generated traffic depend on the compiler, and "reproducible from the
     * seed" would hold only per-binary. The whole value of printing the seed is
     * that a failure at minute nine can be run again, including on the other
     * leg of the matrix. Every draw is hoisted into a variable. */
    const char* stop_s = use_stop ? stops[pick(&c->rng, 4)] : "";

    char body[1024];
    int  bn;
    if (completions) {
        /* ⚠ THE SAMPLING CLAUSE GOES BEFORE THE STOP CLAUSE, not after. The
         * fix-up below rewrites the LAST character of this body to close the
         * array the format string opened, so anything appended after `stop`
         * would be what it overwrote. */
        bn = snprintf(body, sizeof body,
                      "{\"model\":\"stoke\",\"prompt\":\"%s\",\"max_tokens\":%u,"
                      "\"stream\":%s%s%s%s}",
                      p, max_tok, stream ? "true" : "false", samp,
                      use_stop ? ",\"stop\":[" : "", stop_s);
        if (use_stop) {
            /* close the array the branch above opened */
            bn = (int)strlen(body);
            if (bn > 0 && (size_t)bn + 2 < sizeof body) {
                body[bn - 1] = ']';
                body[bn]     = '}';
                body[bn + 1] = '\0';
                bn           = (int)strlen(body);
            }
        }
    } else {
        bn = snprintf(body, sizeof body,
                      "{\"model\":\"stoke\",\"messages\":[{\"role\":\"user\","
                      "\"content\":\"%s\"}],\"max_tokens\":%u,\"stream\":%s%s%s}",
                      p, max_tok, stream ? "true" : "false", samp,
                      usage ? ",\"stream_options\":{\"include_usage\":true}" : "");
    }
    if (bn < 0 || (size_t)bn >= sizeof body) {
        out[0] = '\0';
        return;
    }
    /* Three quarters ask the server to close, so the client sees an EOF at the
     * end of the answer; the rest stay keep-alive so that path is exercised
     * too and the read simply times out, which is also a fine outcome. */
    int close_it = pick(&c->rng, 4) != 0;
    snprintf(out, cap,
             "POST /v1/%s HTTP/1.1\r\nHost: f\r\nContent-Type: application/json\r\n"
             "Connection: %s\r\nContent-Length: %d\r\n\r\n%s",
             completions ? "completions" : "chat/completions",
             close_it ? "close" : "keep-alive", bn, body);
}

static void* client_main(void* arg) {
    Client* c = (Client*)arg;

    while (now_ms() < g_deadline_ms) {
        char req[4096];
        c->actions++;

        switch (pick(&c->rng, 10)) {
            case 0:
            case 1: /* the ordinary path, weighted up so the engine stays busy */
                build_chat(c, req, sizeof req, 0);
                if (req[0]) exchange(c, req, strlen(req), 1);
                break;

            case 2:
                build_chat(c, req, sizeof req, 1);
                if (req[0]) exchange(c, req, strlen(req), 1);
                break;

            case 3: { /* a cheap endpoint, to interleave short requests */
                static const char* paths[] = {"/health", "/v1/models", "/nope",
                                              "/v1/chat/completions"};
                snprintf(req, sizeof req, "GET %s HTTP/1.1\r\nHost: f\r\n\r\n",
                         paths[pick(&c->rng, 4)]);
                exchange(c, req, strlen(req), 1);
                break;
            }

            case 4: { /* the A2 garbage corpus, verbatim */
                if (g_n_corpus == 0) break;
                Blob* b = &g_corpus[pick(&c->rng, (uint32_t)g_n_corpus)];
                exchange(c, b->data, b->len, 1);
                break;
            }

            case 5: { /* malformed JSON in an otherwise valid request */
                static const char* bad[] = {
                    "not json", "{", "{\"messages\":", "[]", "null",
                    "{\"messages\":[{\"role\":1,\"content\":2}]}",
                    "{\"messages\":[],\"max_tokens\":-5}",
                    "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":0.9}",
                };
                const char* b = bad[pick(&c->rng, sizeof bad / sizeof bad[0])];
                snprintf(req, sizeof req,
                         "POST /v1/chat/completions HTTP/1.1\r\nHost: f\r\n"
                         "Content-Length: %d\r\n\r\n%s",
                         (int)strlen(b), b);
                exchange(c, req, strlen(req), 1);
                break;
            }

            case 6: { /* truncated: a head that never ends */
                snprintf(req, sizeof req,
                         "POST /v1/chat/completions HTTP/1.1\r\nHost: f\r\nX-A: ");
                exchange(c, req, strlen(req), 0);
                break;
            }

            case 7: { /* Content-Length lies -- more promised than sent */
                const char* b = "{\"messages\":[]}";
                snprintf(req, sizeof req,
                         "POST /v1/chat/completions HTTP/1.1\r\nHost: f\r\n"
                         "Content-Length: %d\r\n\r\n%s",
                         (int)strlen(b) + 500, b);
                exchange(c, req, strlen(req), 0);
                break;
            }

            case 8: { /* two requests in one write */
                int n = snprintf(req, sizeof req,
                                 "GET /health HTTP/1.1\r\nHost: f\r\n\r\n"
                                 "GET /v1/models HTTP/1.1\r\nHost: f\r\n\r\n");
                exchange(c, req, (size_t)n, 1);
                break;
            }

            case 9: { /* connect and say nothing, or say garbage */
                if (pick(&c->rng, 2)) {
                    NetSock s = net_connect(g_host, g_port);
                    if (s == NET_INVALID_SOCK) {
                        c->refused++;
                    } else {
                        net_close(s);
                    }
                } else {
                    size_t n = 1 + (size_t)pick(&c->rng, 200);
                    for (size_t i = 0; i < n; i++) {
                        req[i] = (char)(pick(&c->rng, 256));
                    }
                    exchange(c, req, n, 0);
                }
                break;
            }
            default: break;
        }
    }
    return NULL;
}

static int load_corpus(const char* manifest) {
    FILE* m = fopen(manifest, "rb");
    if (!m) return -1;
    char line[1024];
    while (g_n_corpus < MAX_CORPUS && fgets(line, sizeof line, m)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (n == 0) continue;
        FILE* f = fopen(line, "rb");
        if (!f) continue;
        char*  buf = (char*)malloc(CORPUS_MAX_LEN);
        if (!buf) { fclose(f); break; }
        size_t got = fread(buf, 1, CORPUS_MAX_LEN, f);
        fclose(f);
        g_corpus[g_n_corpus].data = buf;
        g_corpus[g_n_corpus].len  = got;
        g_n_corpus++;
    }
    fclose(m);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: serve_fuzz <host> <port> <seconds> "
                        "<corpus-manifest> [clients] [seed]\n");
        return 2;
    }
    snprintf(g_host, sizeof g_host, "%s", argv[1]);
    g_port         = (u16)strtoul(argv[2], NULL, 10);
    double secs    = strtod(argv[3], NULL);
    int    clients = argc > 5 ? atoi(argv[5]) : 4;
    g_seed         = argc > 6 ? strtoull(argv[6], NULL, 10) : 20260829ULL;
    if (argc > 7) g_read_ms = strtod(argv[7], NULL);
    if (argc > 8) g_max_tok = (uint32_t)strtoul(argv[8], NULL, 10);
    if (g_max_tok < 1) g_max_tok = 1;

    if (clients < 1) clients = 1;
    if (clients > MAX_CLIENTS) clients = MAX_CLIENTS;

    if (net_startup() != 0) {
        fprintf(stderr, "serve_fuzz: FAIL net_startup: %s\n", net_error());
        return 1;
    }
    if (load_corpus(argv[4]) != 0) {
        fprintf(stderr, "serve_fuzz: FAIL -- cannot read manifest %s\n", argv[4]);
        return 1;
    }

    printf("serve_fuzz: %s:%u for %.0fs, %d clients, %d corpus files, seed %llu,"
           " read %.0fms, max_tok<=%u\n",
           g_host, (unsigned)g_port, secs, clients, g_n_corpus,
           (unsigned long long)g_seed, g_read_ms, g_max_tok);
    fflush(stdout);

    g_deadline_ms = now_ms() + secs * 1000.0;

    Client    cs[MAX_CLIENTS];
    pthread_t th[MAX_CLIENTS];
    for (int i = 0; i < clients; i++) {
        memset(&cs[i], 0, sizeof cs[i]);
        cs[i].id  = i;
        cs[i].rng = g_seed + (uint64_t)i * 0x1000193ULL;
        if (pthread_create(&th[i], NULL, client_main, &cs[i]) != 0) {
            fprintf(stderr, "serve_fuzz: FAIL -- cannot start client %d\n", i);
            return 1;
        }
    }

    long exchanges = 0, refused = 0, actions = 0, sampled = 0, penalized = 0;
    for (int i = 0; i < clients; i++) {
        pthread_join(th[i], NULL);
        exchanges += cs[i].exchanges;
        refused += cs[i].refused;
        actions += cs[i].actions;
        sampled += cs[i].sampled;
        penalized += cs[i].penalized;
    }
    for (int i = 0; i < g_n_corpus; i++) free(g_corpus[i].data);

    printf("serve_fuzz: %ld actions, %ld answered exchanges, %ld refused\n",
           actions, exchanges, refused);
    printf("serve_fuzz: %ld requests asked for a sampler (%ld of those for a "
           "repeat penalty)\n", sampled, penalized);

    /* ⚠ THE ONLY TWO FAILURES THIS DRIVER OWNS. Everything else -- 400s, 404s,
     * dropped streams -- is the server doing its job under abuse, and ASan is
     * the one judging what happened inside it. But a REFUSED connection means
     * the process stopped listening, and zero answered exchanges means the
     * traffic never reached it, which would make a clean ASan log meaningless
     * rather than reassuring. */
    if (refused > 0) {
        fprintf(stderr, "serve_fuzz: FAIL -- %ld connections refused; the "
                        "server stopped accepting\n", refused);
        return 1;
    }
    if (exchanges == 0) {
        fprintf(stderr, "serve_fuzz: FAIL -- not one exchange was answered; a "
                        "clean sanitizer log over no traffic proves nothing\n");
        return 1;
    }
    printf("serve_fuzz: OK\n");
    return 0;
}
