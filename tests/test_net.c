/* tests/test_net.c -- exercises src/net.c, the socket shim. M6 rung A2.
 *
 * ▶▶ LOOPBACK ONLY, AND NO MODEL. Every test below binds an EPHEMERAL port on
 * 127.0.0.1, talks to itself and closes. Nothing listens on a fixed port, so two
 * copies of the suite can run at once and a developer's own server is never
 * disturbed -- which is also why net_listen() takes port 0 and reports back what
 * it got, rather than the caller picking a number and hoping.
 *
 * ▶ THE CLIENT SIDE GOES THROUGH THE SHIM TOO. It would be easier to open the
 * test's own socket with raw winsock/POSIX calls, and it would test half of
 * what this file claims to: the shim exists so exactly one place in the tree
 * knows which platform this is, and a test that reached around it would be
 * asserting against the platform rather than against the shim.
 *
 * Blocking is safe here without a timeout in every case but one: a connect() to
 * a listening socket completes into the backlog whether or not anyone has called
 * accept() yet, so listen -> connect -> accept runs in one thread. The single
 * exception is the large-send test, where the sender WILL fill the socket buffer
 * and must have a reader draining concurrently -- that one spawns a thread, and
 * says so.
 */
#include "common.h"

#include "net.h"

#include <pthread.h>
#include <string.h>

#include "test.h"

/* A listener plus a connected pair, the setup nearly every test wants. */
typedef struct {
    NetSock lis;
    NetSock cli; /* the client end   */
    NetSock srv; /* the accepted end */
} Pair;

static int pair_open(Pair* p) {
    p->lis = p->cli = p->srv = NET_INVALID_SOCK;
    p->lis              = net_listen(NULL, 0, 8);
    if (p->lis == NET_INVALID_SOCK) return 0;
    u16 port = net_port_of(p->lis);
    if (port == 0) return 0;
    p->cli = net_connect("127.0.0.1", port);
    if (p->cli == NET_INVALID_SOCK) return 0;
    p->srv = net_accept(p->lis);
    return p->srv != NET_INVALID_SOCK;
}

static void pair_close(Pair* p) {
    net_close(p->srv);
    net_close(p->cli);
    net_close(p->lis);
}

TEST(listen_on_port_zero_reports_the_port_it_got) {
    ASSERT_EQ_I64(net_startup(), 0);
    NetSock s = net_listen(NULL, 0, 8);
    ASSERT_TRUE(s != NET_INVALID_SOCK);
    if (s == NET_INVALID_SOCK) {
        fprintf(stderr, "  net_listen: %s\n", net_error());
        return;
    }
    u16 port = net_port_of(s);
    /* An ephemeral port is never 0 and never in the well-known range -- a shim
     * that returned the number it was ASKED for would pass a `!= 0` check while
     * reporting nothing. */
    ASSERT_TRUE(port > 1024);
    net_close(s);
}

TEST(loopback_carries_bytes_in_both_directions) {
    Pair p;
    ASSERT_TRUE(pair_open(&p));
    if (p.srv == NET_INVALID_SOCK) {
        fprintf(stderr, "  pair_open: %s\n", net_error());
        pair_close(&p);
        return;
    }

    ASSERT_EQ_I64(net_send_all(p.cli, "GET /health", 11), 0);
    char buf[32];
    memset(buf, 0, sizeof buf);
    i64 got = net_recv(p.srv, buf, sizeof buf);
    ASSERT_EQ_I64(got, 11);
    ASSERT_TRUE(memcmp(buf, "GET /health", 11) == 0);

    /* And back, so a shim that crossed its send and recv paths cannot pass. */
    ASSERT_EQ_I64(net_send_all(p.srv, "HTTP/1.1 200", 12), 0);
    memset(buf, 0, sizeof buf);
    got = net_recv(p.cli, buf, sizeof buf);
    ASSERT_EQ_I64(got, 12);
    ASSERT_TRUE(memcmp(buf, "HTTP/1.1 200", 12) == 0);

    pair_close(&p);
}

/* Peer close must be distinguishable from "no data yet" and from an error, or
 * the server either spins on a dead socket or drops a live one. */
TEST(recv_reports_peer_close_as_closed_not_as_error) {
    Pair p;
    ASSERT_TRUE(pair_open(&p));
    if (p.srv == NET_INVALID_SOCK) {
        pair_close(&p);
        return;
    }

    net_close(p.cli);
    p.cli = NET_INVALID_SOCK;

    char buf[16];
    ASSERT_EQ_I64(net_recv(p.srv, buf, sizeof buf), NET_CLOSED);
    pair_close(&p);
}

/* A4 forbids blocking the scheduler on a socket, so this is the property the
 * whole server loop rests on: a read with nothing to read returns immediately
 * and says so. If this test ever hangs, that is the failure. */
TEST(nonblocking_recv_reports_would_block_rather_than_blocking) {
    Pair p;
    ASSERT_TRUE(pair_open(&p));
    if (p.srv == NET_INVALID_SOCK) {
        pair_close(&p);
        return;
    }

    ASSERT_EQ_I64(net_set_nonblocking(p.srv), 0);
    char buf[16];
    ASSERT_EQ_I64(net_recv(p.srv, buf, sizeof buf), NET_WOULD_BLOCK);

    /* And it still delivers when there IS something -- a shim that returned
     * WOULD_BLOCK unconditionally would pass the assertion above. */
    ASSERT_EQ_I64(net_send_all(p.cli, "x", 1), 0);
    i64 got = -99;
    for (int i = 0; i < 1000 && got != 1; i++) {
        got = net_recv(p.srv, buf, sizeof buf);
        if (got == NET_WOULD_BLOCK) continue; /* the byte is still in flight */
        break;
    }
    ASSERT_EQ_I64(got, 1);
    pair_close(&p);
}

/* ---------------------------------------------------------------------------
 * The one test that needs a thread. A send larger than the socket buffer CANNOT
 * complete unless someone is draining the other end, so this is also the only
 * test that exercises net_send_all's loop at all: every send above fits in one
 * write and would pass against a send_all that ignored a partial result.
 * ---------------------------------------------------------------------------
 */

#define BIG (1u << 20)

typedef struct {
    NetSock s;
    size_t  want;
    size_t  got;
    int     mismatch;
} Drain;

static void* drain_thread(void* arg) {
    Drain* d = (Drain*)arg;
    char*  b = (char*)malloc(65536);
    if (!b) return NULL;
    while (d->got < d->want) {
        i64 n = net_recv(d->s, b, 65536);
        if (n <= 0) break;
        /* The payload is a byte counter, so a shim that dropped or duplicated a
         * chunk is caught by CONTENT, not merely by the total. */
        for (i64 i = 0; i < n; i++) {
            if ((unsigned char)b[i] != (unsigned char)((d->got + (size_t)i) & 0xFF)) {
                d->mismatch = 1;
            }
        }
        d->got += (size_t)n;
    }
    free(b);
    return NULL;
}

TEST(send_all_completes_a_buffer_larger_than_the_socket_buffer) {
    Pair p;
    ASSERT_TRUE(pair_open(&p));
    if (p.srv == NET_INVALID_SOCK) {
        pair_close(&p);
        return;
    }

    char* payload = (char*)malloc(BIG);
    ASSERT_TRUE(payload != NULL);
    if (!payload) {
        pair_close(&p);
        return;
    }
    for (u32 i = 0; i < BIG; i++) payload[i] = (char)(i & 0xFF);

    Drain d = {p.srv, BIG, 0, 0};
    pthread_t th;
    ASSERT_EQ_I64(pthread_create(&th, NULL, drain_thread, &d), 0);

    ASSERT_EQ_I64(net_send_all(p.cli, payload, BIG), 0);
    net_close(p.cli);
    p.cli = NET_INVALID_SOCK;

    pthread_join(th, NULL);
    ASSERT_EQ_I64(d.got, BIG);
    ASSERT_EQ_I64(d.mismatch, 0);

    free(payload);
    pair_close(&p);
}

TEST(connect_to_a_port_nobody_is_listening_on_fails_cleanly) {
    /* Bind and immediately release, so the port is real and almost certainly
     * still free -- picking a constant would be a race against whatever else is
     * on the box. */
    NetSock s = net_listen(NULL, 0, 8);
    ASSERT_TRUE(s != NET_INVALID_SOCK);
    u16 port = net_port_of(s);
    net_close(s);

    NetSock c = net_connect("127.0.0.1", port);
    ASSERT_EQ_I64(c, NET_INVALID_SOCK);
    ASSERT_TRUE(net_error()[0] != '\0'); /* and it says why */
    net_close(c);                        /* must be safe on the invalid handle */
}

TEST(close_of_an_invalid_socket_is_safe) {
    net_close(NET_INVALID_SOCK);
    net_close(NET_INVALID_SOCK);
}
