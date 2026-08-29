/* tests/serve_probe.c -- the server behaviours curl structurally cannot reach.
 * M6 rungs A3 (pipelining) and A4 (heartbeat, slow-client policy).
 *
 * ▶▶ WHY A CLIENT OF OUR OWN. tests/serve_golden.sh is gate item [2] and it is
 * driven by curl, which is the right tool for it -- it is what the compat
 * surface will actually meet. But three of this server's promises are outside
 * anything curl can express:
 *
 *   pipeline  curl dropped HTTP/1.1 pipelining in 7.62 and no flag brings it
 *             back, so nothing else can put two requests in ONE write.
 *   ping      the heartbeat is a fifteen-second liveness comment; a gate that
 *             has to finish cannot wait for one, and its arrival time is not
 *             byte-stable, so it cannot live in a golden either.
 *   slowread  proving the server disconnects a reader that stopped reading
 *             means BEING a reader that stopped reading, which curl will not
 *             do -- it always drains.
 *
 * Each mode is a separate process run: one server, three verdicts, no shared
 * state to leak between them.
 *
 * Not named test_*.c on purpose: it needs a server on a port, so it is driven
 * by tests/serve_raw.sh -- the same policy as abi_smoke.c.
 *
 * Usage: serve_probe <pipeline|ping|slowread> <host> <port>
 */
#include "common.h"

#include "net.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEALTH_REQ \
    "GET /health HTTP/1.1\r\nHost: stoke\r\nConnection: keep-alive\r\n\r\n"

/* ⚠ VERY generous, and it was 30 s until one intermittent failure taught me
 * better. Every probe here exits the moment it has its answer, so a large
 * deadline costs nothing on the passing path -- it is paid only by a genuine
 * hang. A tight one, by contrast, is a second clock racing the box: on a
 * loaded machine the ping probe once gave up 4,428 bytes into a stream the
 * server went on to finish correctly, and the resulting red said "no [DONE]",
 * which is indistinguishable from the server never sending one. A gate that
 * fails intermittently is worse than one that fails always, and a test-side
 * timeout tighter than the work is the commonest way to build one. */
#define DEADLINE_MS_DEFAULT 120000.0

/* Settable by the caller, because the same probe runs against a -O3 build
 * and against a sanitized one where a forward pass is an order of magnitude
 * slower. Under ASan the ping probe's 438-token prefill outran a 120-second
 * deadline: 3 heartbeats arrived and the stream had not finished, which the
 * failure message correctly reported as the PROBE's deadline rather than a
 * missing sentinel. A deadline tighter than the work manufactures reds. */
static double g_deadline_ms = DEADLINE_MS_DEFAULT;
#define DEADLINE_MS g_deadline_ms

static int count_of(const char* hay, size_t n, const char* needle) {
    size_t m = strlen(needle);
    int    c = 0;
    if (m == 0 || n < m) return 0;
    for (size_t i = 0; i + m <= n; i++) {
        if (memcmp(hay + i, needle, m) == 0) c++;
    }
    return c;
}

static int fail(const char* mode, const char* what) {
    fprintf(stderr, "serve_probe[%s]: FAIL -- %s: %s\n", mode, what,
            net_error());
    return 1;
}

/* A portable pause built only out of net.h.
 *
 * ⚠ IT LOOKS INDIRECT AND THE ALTERNATIVES ARE WORSE. The slow-read probe must
 * let time pass while deliberately NOT reading its socket; select() on that
 * socket returns instantly forever, because the unread bytes are exactly what
 * makes it readable, so waiting on it would spin a core. A listener nobody
 * will connect to is never readable, so waiting on it is a real sleep -- and
 * it needs no platform header here, which keeps src/net.c the only file in
 * this project that knows which operating system it is on. */
static void nap_ms(int ms) {
    NetSock l = net_listen("127.0.0.1", 0, 1);
    if (l == NET_INVALID_SOCK) return;
    NetWait w;
    memset(&w, 0, sizeof w);
    w.s         = l;
    w.want_read = 1;
    net_wait(&w, 1, ms);
    net_close(l);
}

/* Reads until `stop` has been seen, the peer closes, or the deadline passes.
 * Returns the byte count, or -1 on a socket error. `*closed` reports whether
 * the PEER ended the connection, which for a stream is the whole verdict. */
static i64 drain(NetSock s, char* buf, size_t cap, const char* stop,
                 int* closed) {
    size_t len   = 0;
    double start = now_ms();
    *closed      = 0;

    while (now_ms() - start < DEADLINE_MS && len < cap) {
        NetWait w;
        memset(&w, 0, sizeof w);
        w.s         = s;
        w.want_read = 1;
        if (net_wait(&w, 1, 200) == NET_ERR) return -1;
        if (!w.can_read) continue;

        i64 n = net_recv(s, buf + len, cap - len);
        if (n == NET_WOULD_BLOCK) continue;
        if (n < 0) {
            /* A reset is how Windows reports "the peer gave up on you", which
             * is a legitimate way for the slow-read policy to land. */
            *closed = 1;
            return (i64)len;
        }
        if (n == 0) {
            *closed = 1;
            return (i64)len;
        }
        len += (size_t)n;
        if (stop && count_of(buf, len, stop) > 0) return (i64)len;
    }
    return (i64)len;
}

/* Returns 0, or -1 if anything was truncated.
 *
 * ⚠⚠ THE TRUNCATION CHECK IS THE POINT, and it is here because its absence
 * cost a run. snprintf returns the length it WOULD have written, so a body
 * buffer one size too small produced a 511-byte body announced as
 * `Content-Length: 2100` -- a perfectly well-formed request for a body that
 * was never going to arrive. The server did exactly the right thing and waited
 * for it, the probe read zero bytes for thirty seconds, and the failure looked
 * like a broken heartbeat in the server rather than a broken request from the
 * test. A silent truncation in a test client is indistinguishable from a bug
 * in the thing under test. */
static int chat_request(char* out, size_t cap, const char* prompt,
                        const char* model, int max_tokens) {
    char body[4096];
    int  bn = snprintf(body, sizeof body,
                       "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\","
                       "\"content\":\"%s\"}],\"max_tokens\":%d,\"stream\":true}",
                       model, prompt, max_tokens);
    if (bn < 0 || (size_t)bn >= sizeof body) return -1;

    int rn = snprintf(out, cap,
                      "POST /v1/chat/completions HTTP/1.1\r\nHost: stoke\r\n"
                      "Content-Type: application/json\r\nContent-Length: %d\r\n"
                      "\r\n%s",
                      bn, body);
    if (rn < 0 || (size_t)rn >= cap) return -1;
    return 0;
}

/* ---- pipeline ----------------------------------------------------------
 *
 * ▶ THE SINGLE send() IS THE WHOLE TEST. Both requests go out in one call, so
 * they land in the server's receive buffer together and ONE recv() takes both.
 * The server answers the first and must then parse the second out of what it
 * already holds -- with no further bytes arriving, select() will never call
 * that socket readable again, so a server that waits for a read event answers
 * request one and leaves request two to rot until the idle reaper closes the
 * connection a minute later. That is what this caught on 2026-08-29. */
static int mode_pipeline(const char* host, u16 port) {
    NetSock s = net_connect(host, port);
    if (s == NET_INVALID_SOCK) return fail("pipeline", "connect");

    char   both[2 * sizeof HEALTH_REQ];
    size_t one = sizeof HEALTH_REQ - 1;
    memcpy(both, HEALTH_REQ, one);
    memcpy(both + one, HEALTH_REQ, one);
    if (net_send_all(s, both, one * 2) != 0) {
        net_close(s);
        return fail("pipeline", "send");
    }
    net_set_nonblocking(s);

    char buf[16384];
    int  closed = 0;
    i64  len    = drain(s, buf, sizeof buf, NULL, &closed);
    net_close(s);
    if (len < 0) return fail("pipeline", "recv");

    int got    = count_of(buf, (size_t)len, "HTTP/1.1 200 OK\r\n");
    int bodies = count_of(buf, (size_t)len, "{\"status\":\"ok\"}");
    if (got != 2 || bodies != 2) {
        fprintf(stderr,
                "serve_probe[pipeline]: FAIL -- sent 2 requests in one write, "
                "got %d status lines and %d bodies in %d bytes (a server that "
                "only answers what a NEW read event announces answers exactly "
                "the first one)\n",
                got, bodies, (int)len);
        fprintf(stderr, "---- received ----\n%.*s\n----\n", (int)len, buf);
        return 1;
    }
    printf("serve_probe[pipeline]: OK (2 requests in one write, 2 responses)\n");
    return 0;
}

/* ---- ping ---------------------------------------------------------------
 *
 * ⚠⚠ THE PROMPT IS LONG ON PURPOSE, and the first version of this probe was
 * wrong because it was not. A short prompt gives the heartbeat nowhere to
 * fire: the check runs once per loop turn, each turn's forward pass emits a
 * token, and emitting resets the timer -- so the elapsed time at the check is
 * always ~0 no matter how small the interval is set. Measured: 0 pings at
 * --sse-ping-ms 5. That is the heartbeat behaving CORRECTLY (it exists for
 * gaps, and steady generation has none; the tokens are the heartbeat), and it
 * means the gap has to be manufactured out of something real. Prefill is that
 * gap: a few hundred prompt tokens are consumed in chunks, none of which
 * produces output, which is exactly the silence a proxy would time out on.
 *
 * Two claims, and the second is the one worth having: the comment appears at
 * all, AND the stream still terminates correctly with it interleaved -- a
 * heartbeat written into the middle of a chunked body is precisely where
 * framing goes wrong. */
static int mode_ping(const char* host, u16 port) {
    NetSock s = net_connect(host, port);
    if (s == NET_INVALID_SOCK) return fail("ping", "connect");

    /* ~250 words, so a few hundred prompt tokens and a prefill long enough to
     * be silent for longer than the interval the gate configures. */
    char prompt[2048];
    size_t pl = 0;
    prompt[0] = '\0';
    while (pl + 64 < sizeof prompt) {
        int k = snprintf(prompt + pl, sizeof prompt - pl,
                         "the quick brown fox jumps over the lazy dog again. ");
        if (k <= 0) break;
        pl += (size_t)k;
    }

    char req[8192];
    if (chat_request(req, sizeof req, prompt, "stoke", 24) != 0) {
        net_close(s);
        fprintf(stderr, "serve_probe[ping]: FAIL -- request did not fit\n");
        return 1;
    }
    if (net_send_all(s, req, strlen(req)) != 0) {
        net_close(s);
        return fail("ping", "send");
    }
    net_set_nonblocking(s);

    char buf[65536];
    int  closed = 0;
    i64  len    = drain(s, buf, sizeof buf, "data: [DONE]", &closed);
    net_close(s);
    if (len < 0) return fail("ping", "recv");

    int pings = count_of(buf, (size_t)len, ": ping\n\n");
    int done  = count_of(buf, (size_t)len, "data: [DONE]");
    if (pings < 1 || done != 1) {
        /* Saying WHICH of the three ways this ended is the difference between
         * a diagnosis and a shrug: the server closed on us, the deadline ran
         * out with the stream still arriving, or the stream really did end
         * without a sentinel. */
        fprintf(stderr,
                "serve_probe[ping]: FAIL -- expected at least one `: ping` "
                "comment and exactly one [DONE]; got %d pings and %d [DONE] "
                "in %d bytes (%s)\n",
                pings, done, (int)len,
                closed ? "the server closed the connection"
                       : "the probe's own deadline expired -- the stream may "
                         "still have been arriving");
        fprintf(stderr, "---- received ----\n%.*s\n----\n", (int)len, buf);
        return 1;
    }
    printf("serve_probe[ping]: OK (%d heartbeat comments, stream still ended "
           "cleanly)\n",
           pings);
    return 0;
}

/* ---- slowread -----------------------------------------------------------
 *
 * ▶▶ THE POLICY UNDER TEST: a client that stops reading must be disconnected
 * and ITS request cancelled, and no other client may notice. The failure it
 * prevents is a stream that makes this process buy memory on a stranger's
 * behalf for as long as the stranger likes -- and, worse, one that could make
 * the scheduler wait, because every other client's tokens come off the same
 * loop.
 *
 * ⚠ THE SMALL RECEIVE BUFFER IS WHAT MAKES THE TEST FINISH IN SECONDS. Without
 * it the two kernels absorb tens of kilobytes between them before the server's
 * own buffer grows at all, and reaching even a small cap would take hundreds of
 * generated tokens. It is advisory -- the kernel may clamp it -- so the gate
 * ALSO runs the server with a tiny --sse-max-pending; neither alone is relied
 * on, and the failure message says which way to lean if the stream ever fits
 * inside the cap. */
static int mode_slowread(const char* host, u16 port) {
    NetSock s = net_connect_recvbuf(host, port, 2048);
    if (s == NET_INVALID_SOCK) return fail("slowread", "connect");

    char req[8192];
    if (chat_request(req, sizeof req,
                     "Count slowly from one to three hundred.", "stoke",
                     900) != 0) {
        net_close(s);
        fprintf(stderr, "serve_probe[slowread]: FAIL -- request did not fit\n");
        return 1;
    }
    if (net_send_all(s, req, strlen(req)) != 0) {
        net_close(s);
        return fail("slowread", "send");
    }
    net_set_nonblocking(s);

    /* ▶▶▶ NEVER READ. DETECT BY WRITING. Three versions of this probe got the
     * detection wrong before this one, each in a way worth writing down:
     *
     *   1. Read nothing for a fixed nap, then drain. The drain empties the
     *      kernel buffers, so the server stops accumulating the instant the
     *      measurement begins, and the verdict becomes "was the nap long
     *      enough", which nobody can answer in advance.
     *   2. Read one byte every 250 ms. This keeps the pipe filling and the
     *      server DID drop the client -- its log said so -- but the probe
     *      could not see it: end-of-stream sits behind every buffered byte,
     *      and at four bytes a second that is most of an hour away.
     *   3. Nap ten seconds, then drain fast. Correct in principle and still
     *      timing-dependent: how long the buffers take to fill is a function
     *      of how fast the box generates, which on a loaded box moved the
     *      answer across the ten-second line.
     *
     * Writing has none of these problems. A byte sent into a connection the
     * server has closed fails with a reset; a byte sent into a live one
     * succeeds and consumes nothing the server is holding. So the probe stays
     * perfectly silent as a reader -- which is the condition under test -- and
     * still notices the drop within one polling interval, whatever the box is
     * doing. The stray bytes land in a receive buffer this server never reads
     * while streaming, and go away with the connection. */
    double start   = now_ms();
    int    dropped = 0;
    while (now_ms() - start < DEADLINE_MS) {
        nap_ms(500);
        i64 w = net_send(s, "x", 1);
        if (w == NET_WOULD_BLOCK) continue;
        if (w < 0) {
            dropped = 1;
            break;
        }
    }
    double waited = now_ms() - start;

    /* Only now, and only to say what the client had been given: a stream that
     * reached [DONE] finished normally and was never dropped at all. */
    net_set_nonblocking(s);
    char  scratch[262144];
    int   closed   = 0;
    i64   n        = drain(s, scratch, sizeof scratch, NULL, &closed);
    i64   got      = n > 0 ? n : 0;
    int   saw_done = (n > 0) ? count_of(scratch, (size_t)n, "data: [DONE]") : 0;
    if (saw_done) dropped = 0;

    net_close(s);

    if (!dropped) {
        /* ⚠ THE FAILURE PATH REPORTS THE ONE NUMBER THAT DECIDES WHAT WENT
         * WRONG. Whether the policy is broken or merely out of reach turns on
         * how much the two kernels will hold before the server's own buffer
         * grows at all -- which nobody can guess and which changes with a
         * socket option -- so the bytes in flight are printed rather than left
         * for the next person to re-derive from a bare "did not drop". */
        fprintf(stderr,
                "serve_probe[slowread]: FAIL -- a client that read nothing for "
                "%.1f s was not disconnected; %d bytes were then drained%s. If that is "
                "close to everything the request could generate, the kernels "
                "between the two ends are absorbing the whole stream (raise "
                "the wait, or lower --sse-send-buffer). If the stream simply "
                "ENDED, raise max_tokens.\n",
                waited / 1000.0, (int)got,
                saw_done ? " and the stream reached [DONE]" : "");
        return 1;
    }
    printf("serve_probe[slowread]: OK (server dropped a silent client after "
           "%.1f s; %d bytes had been buffered, and no [DONE] arrived)\n",
           waited / 1000.0, (int)got);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: serve_probe <pipeline|ping|slowread> <host> <port>\n");
        return 2;
    }
    const char* mode = argv[1];
    const char* host = argv[2];
    u16         port = (u16)strtoul(argv[3], NULL, 10);
    if (argc > 4) g_deadline_ms = strtod(argv[4], NULL);

    if (net_startup() != 0) {
        fprintf(stderr, "serve_probe: FAIL net_startup: %s\n", net_error());
        return 1;
    }

    if (strcmp(mode, "pipeline") == 0) return mode_pipeline(host, port);
    if (strcmp(mode, "ping") == 0) return mode_ping(host, port);
    if (strcmp(mode, "slowread") == 0) return mode_slowread(host, port);

    fprintf(stderr, "serve_probe: unknown mode '%s'\n", mode);
    return 2;
}
