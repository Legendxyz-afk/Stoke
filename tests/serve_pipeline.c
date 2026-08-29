/* tests/serve_pipeline.c -- M6 rung A3: TWO REQUESTS IN ONE WRITE.
 *
 * ▶▶ WHY curl CANNOT DO THIS, WHICH IS WHY THIS FILE EXISTS. HTTP/1.1
 * pipelining means the client sends request 2 without waiting for response 1.
 * curl dropped pipelining in 7.62 and there is no flag that brings it back, so
 * the one gate that drives this server (tests/serve_golden.sh) cannot reach the
 * path at all. src/server.c claims to handle it -- service_write() re-enters
 * service_read() precisely because "a pipelining client may already have sent
 * the next request" -- and a claim no test can reach is the kind this project
 * has been burned by. So: a client of our own, over the same socket shim the
 * server uses.
 *
 * ▶ THE SINGLE send() IS THE WHOLE TEST. Both requests go out in one call, so
 * they land in the server's receive buffer together and ONE recv() takes both.
 * The server answers the first and must then parse the second out of what it
 * already holds -- with no further bytes arriving, select() will never call
 * that socket readable again, so a server that waits for a read event answers
 * request 1 and leaves request 2 to rot until the idle reaper closes the
 * connection a minute later. That is what this caught on 2026-08-29.
 *
 * Not named test_*.c on purpose: it needs a server on a port, so it is driven
 * by tests/serve_pipeline.sh -- the same policy as abi_smoke.c.
 *
 * Usage: serve_pipeline <host> <port>
 */
#include "common.h"

#include "net.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQ "GET /health HTTP/1.1\r\nHost: stoke\r\nConnection: keep-alive\r\n\r\n"

/* Generous: the server may be mid-forward-pass when this arrives, and the
 * failure being tested is "never", not "slow". */
#define DEADLINE_MS 20000.0

static int count_of(const char* hay, size_t n, const char* needle) {
    size_t m = strlen(needle);
    int    c = 0;
    if (m == 0 || n < m) return 0;
    for (size_t i = 0; i + m <= n; i++) {
        if (memcmp(hay + i, needle, m) == 0) c++;
    }
    return c;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: serve_pipeline <host> <port>\n");
        return 2;
    }
    const char* host = argv[1];
    u16         port = (u16)strtoul(argv[2], NULL, 10);

    if (net_startup() != 0) {
        fprintf(stderr, "serve_pipeline: FAIL net_startup: %s\n", net_error());
        return 1;
    }

    NetSock s = net_connect(host, port);
    if (s == NET_INVALID_SOCK) {
        fprintf(stderr, "serve_pipeline: FAIL net_connect %s:%u: %s\n", host,
                (unsigned)port, net_error());
        return 1;
    }

    char   both[2 * sizeof(REQ)];
    size_t one = sizeof(REQ) - 1;
    memcpy(both, REQ, one);
    memcpy(both + one, REQ, one);
    if (net_send_all(s, both, one * 2) != 0) {
        fprintf(stderr, "serve_pipeline: FAIL send: %s\n", net_error());
        net_close(s);
        return 1;
    }

    net_set_nonblocking(s);

    char   buf[16384];
    size_t len   = 0;
    double start = now_ms();
    int    got   = 0;

    while (now_ms() - start < DEADLINE_MS) {
        NetWait w;
        memset(&w, 0, sizeof w);
        w.s         = s;
        w.want_read = 1;
        if (net_wait(&w, 1, 200) == NET_ERR) {
            fprintf(stderr, "serve_pipeline: FAIL net_wait: %s\n", net_error());
            net_close(s);
            return 1;
        }
        if (!w.can_read) continue;

        i64 n = net_recv(s, buf + len, sizeof buf - len);
        if (n == NET_WOULD_BLOCK) continue;
        if (n < 0) {
            fprintf(stderr, "serve_pipeline: FAIL recv: %s\n", net_error());
            net_close(s);
            return 1;
        }
        if (n == 0) break; /* the server closed; whatever arrived is all of it */
        len += (size_t)n;

        got = count_of(buf, len, "HTTP/1.1 200 OK\r\n");
        if (got >= 2) break;
        if (len == sizeof buf) break;
    }
    net_close(s);

    got = count_of(buf, len, "HTTP/1.1 200 OK\r\n");
    /* Both must be complete answers, not two status lines and half a body. */
    int bodies = count_of(buf, len, "{\"status\":\"ok\"}");

    if (got != 2 || bodies != 2) {
        fprintf(stderr,
                "serve_pipeline: FAIL -- sent 2 pipelined requests in one "
                "write, got %d status lines and %d bodies in %zu bytes "
                "(a server that only answers what a NEW read event announces "
                "answers exactly the first one)\n",
                got, bodies, len);
        fprintf(stderr, "---- received ----\n%.*s\n------------------\n",
                (int)len, buf);
        return 1;
    }

    printf("serve_pipeline: OK (2 pipelined requests, 2 responses, %zu bytes)\n",
           len);
    return 0;
}
