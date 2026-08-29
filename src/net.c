/* src/net.c -- the socket shim. See net.h for the contract.
 *
 * THIS IS THE ONLY FILE IN THE TREE THAT KNOWS WHICH PLATFORM'S SOCKETS THESE
 * ARE. If an #ifdef for winsock appears anywhere else, something above it is
 * reaching around this file.
 */
#include "common.h"

#include "net.h"

#include <pthread.h>
#include <string.h>

#ifdef _WIN32
/* common.h already included <windows.h> under WIN32_LEAN_AND_MEAN, which is
 * what keeps the winsock 1.1 declarations out of the way -- without it these
 * two headers collide with it rather than extending it. */
#include <winsock2.h>
#include <ws2tcpip.h>
#define SOCKH(s_) ((SOCKET)(s_))
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
/* ⚠ select() AND ITS fd_set LIVE HERE ON POSIX, AND ONLY HERE. On Windows
 * <winsock2.h> hands over select, fd_set, FD_ZERO/SET/ISSET and struct timeval
 * together, so net_wait() compiled on the Windows leg from the day it was
 * written (M6 rung A3) and did not compile on Linux ONCE -- for six commits,
 * because every build in that stretch was a Windows build. The whole point of
 * this file is to be the one place that knows which platform this is; a
 * portability hole in it is worth more than a comment, so: sys/select.h for
 * select and fd_set, sys/time.h for struct timeval. */
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>
#define SOCKH(s_) ((int)(s_))
#endif

/* Per thread, for the same reason stoke_last_error() is: two connections
 * failing on two threads must not overwrite each other's diagnosis. */
static _Thread_local char g_err[256];

static void set_err(const char* what) {
#ifdef _WIN32
    snprintf(g_err, sizeof g_err, "%s: winsock error %d", what,
             WSAGetLastError());
#else
    snprintf(g_err, sizeof g_err, "%s: %s", what, strerror(errno));
#endif
}

static void set_err_str(const char* what, const char* why) {
    snprintf(g_err, sizeof g_err, "%s: %s", what, why);
}

const char* net_error(void) { return g_err; }

/* ---- startup ------------------------------------------------------------ */

static pthread_once_t g_once       = PTHREAD_ONCE_INIT;
static int            g_startup_rc = 0;

static void do_startup(void) {
#ifdef _WIN32
    WSADATA d;
    if (WSAStartup(MAKEWORD(2, 2), &d) != 0) g_startup_rc = -1;
#endif
}

int net_startup(void) {
    pthread_once(&g_once, do_startup);
    if (g_startup_rc != 0) set_err_str("net_startup", "WSAStartup failed");
    return g_startup_rc;
}

/* ---- addresses ---------------------------------------------------------- */

/* getaddrinfo rather than inet_pton so a host may be a name. It is also the
 * only spelling that is identical on both platforms. */
static NetSock connect_to(const char* host, u16 port, int recvbuf) {
    char portstr[8];
    snprintf(portstr, sizeof portstr, "%u", (unsigned)port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof hints);
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    struct addrinfo* res = NULL;
    if (getaddrinfo(host, portstr, &hints, &res) != 0 || !res) {
        set_err_str("getaddrinfo", host);
        return NET_INVALID_SOCK;
    }

    NetSock s = NET_INVALID_SOCK;
    for (struct addrinfo* ai = res; ai; ai = ai->ai_next) {
        NetSock t = (NetSock)socket(ai->ai_family, ai->ai_socktype,
                                    ai->ai_protocol);
        if (t == NET_INVALID_SOCK) {
            set_err("socket");
            continue;
        }
        /* BEFORE connect(), which is the whole reason this is a parameter and
         * not a call the caller makes afterwards. Windows auto-tunes the
         * receive window and only stops once SO_RCVBUF has been set
         * explicitly, and the window is first advertised in the handshake --
         * so a value set after connect() may simply never take effect. */
        if (recvbuf > 0) net_set_recv_buffer(t, recvbuf);
        if (connect(SOCKH(t), ai->ai_addr, (int)ai->ai_addrlen) == 0) {
            s = t;
            break;
        }
        set_err("connect");
        net_close(t);
    }
    freeaddrinfo(res);
    return s;
}

NetSock net_connect(const char* host, u16 port) {
    if (net_startup() != 0) return NET_INVALID_SOCK;
    return connect_to(host ? host : "127.0.0.1", port, 0);
}

NetSock net_connect_recvbuf(const char* host, u16 port, int recvbuf) {
    if (net_startup() != 0) return NET_INVALID_SOCK;
    return connect_to(host ? host : "127.0.0.1", port, recvbuf);
}

NetSock net_listen(const char* bind_addr, u16 port, int backlog) {
    if (net_startup() != 0) return NET_INVALID_SOCK;

    NetSock s = (NetSock)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == NET_INVALID_SOCK) {
        set_err("socket");
        return NET_INVALID_SOCK;
    }

#ifndef _WIN32
    /* ⚠ POSIX ONLY, AND THE ASYMMETRY IS DELIBERATE. On POSIX SO_REUSEADDR
     * lets a restarting server rebind a port whose old connections are still
     * in TIME_WAIT, which is what it is for. On WINDOWS the same option means
     * something else entirely: it lets a DIFFERENT process bind an address and
     * port already in use and steal the traffic. Windows already permits the
     * restart case without any option, so setting it there would buy nothing
     * and hand away the port. */
    int on = 1;
    setsockopt(SOCKH(s), SOL_SOCKET, SO_REUSEADDR, (const void*)&on, sizeof on);
#endif

    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port   = htons(port);

    const char* addr = bind_addr ? bind_addr : "127.0.0.1";
    if (inet_pton(AF_INET, addr, &a.sin_addr) != 1) {
        set_err_str("net_listen", "bind address is not a dotted-quad IPv4");
        net_close(s);
        return NET_INVALID_SOCK;
    }

    if (bind(SOCKH(s), (struct sockaddr*)&a, sizeof a) != 0) {
        set_err("bind");
        net_close(s);
        return NET_INVALID_SOCK;
    }
    if (listen(SOCKH(s), backlog > 0 ? backlog : 16) != 0) {
        set_err("listen");
        net_close(s);
        return NET_INVALID_SOCK;
    }
    return s;
}

NetSock net_accept(NetSock listener) {
    NetSock c = (NetSock)accept(SOCKH(listener), NULL, NULL);
    if (c == NET_INVALID_SOCK) set_err("accept");
    return c;
}

u16 net_port_of(NetSock s) {
    struct sockaddr_in a;
    socklen_t          n = sizeof a;
    memset(&a, 0, sizeof a);
    if (getsockname(SOCKH(s), (struct sockaddr*)&a, &n) != 0) {
        set_err("getsockname");
        return 0;
    }
    return ntohs(a.sin_port);
}

/* ---- transfer ----------------------------------------------------------- */

/* recv/send take an int length on Windows. Clamping rather than truncating
 * matters: a cast would turn a 4 GiB request into a small one or a negative
 * one, and the caller would see a short read it had no way to explain. A short
 * read is already part of this contract, so clamping is invisible. */
#define NET_CHUNK_MAX (1u << 24)

static int clamp_len(size_t n) {
    return (int)(n > NET_CHUNK_MAX ? NET_CHUNK_MAX : n);
}

i64 net_recv(NetSock s, void* buf, size_t cap) {
    if (cap == 0) {
        /* recv() with a zero length returns 0, which this API spells CLOSED --
         * a caller that passed an empty buffer would be told the peer hung up. */
        set_err_str("net_recv", "zero capacity");
        return NET_ERR;
    }
    for (;;) {
        int n = recv(SOCKH(s), (char*)buf, clamp_len(cap), 0);
        if (n > 0) return n;
        if (n == 0) return NET_CLOSED;
#ifdef _WIN32
        int e = WSAGetLastError();
        if (e == WSAEINTR) continue;
        if (e == WSAEWOULDBLOCK) return NET_WOULD_BLOCK;
#else
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return NET_WOULD_BLOCK;
#endif
        set_err("recv");
        return NET_ERR;
    }
}

i64 net_send(NetSock s, const void* buf, size_t len) {
    if (len == 0) return 0;
    for (;;) {
        int n = send(SOCKH(s), (const char*)buf, clamp_len(len), 0);
        if (n > 0) return n;
#ifdef _WIN32
        int e = WSAGetLastError();
        if (e == WSAEINTR) continue;
        if (e == WSAEWOULDBLOCK) return NET_WOULD_BLOCK;
#else
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return NET_WOULD_BLOCK;
#endif
        set_err("send");
        return NET_ERR;
    }
}

int net_send_all(NetSock s, const void* buf, size_t len) {
    const char* p    = (const char*)buf;
    size_t      left = len;
    while (left > 0) {
        i64 n = net_send(s, p, left);
        if (n == NET_WOULD_BLOCK) {
            /* Refused rather than spun on. This function's contract is a
             * BLOCKING socket; retrying here would burn a core holding a
             * response nobody is reading, which is precisely the slow-client
             * case A4 handles with a bounded per-connection buffer instead. */
            set_err_str("net_send_all", "socket is non-blocking; use net_send");
            return NET_ERR;
        }
        if (n < 0) return NET_ERR;
        p += n;
        left -= (size_t)n;
    }
    return 0;
}

/* ---- options ------------------------------------------------------------ */

/* select(), not poll/WSAPoll, because it is the one spelling that is identical
 * on both platforms and this server's connection count is small by design. The
 * cost is the FD_SETSIZE ceiling, which is why net.h caps n and why exceeding
 * it is an ERROR here rather than a silent truncation -- select would otherwise
 * quietly ignore the sockets past the limit and the loop would stop servicing
 * them with nothing to show for it. */
int net_wait(NetWait* w, int n, int timeout_ms) {
    if (n < 0 || n > NET_WAIT_MAX) {
        set_err_str("net_wait", "socket count is outside NET_WAIT_MAX");
        return NET_ERR;
    }
    for (int i = 0; i < n; i++) {
        w[i].can_read  = 0;
        w[i].can_write = 0;
    }
    if (n == 0) return 0;

    fd_set rd, wr;
    FD_ZERO(&rd);
    FD_ZERO(&wr);

    int nfds = 0;
    int any  = 0;
    for (int i = 0; i < n; i++) {
        if (w[i].s == NET_INVALID_SOCK) continue;
        if (w[i].want_read) {
            FD_SET(SOCKH(w[i].s), &rd);
            any = 1;
        }
        if (w[i].want_write) {
            FD_SET(SOCKH(w[i].s), &wr);
            any = 1;
        }
        if ((int)w[i].s + 1 > nfds) nfds = (int)w[i].s + 1;
    }
    /* Nothing is being watched. Returning 0 at once rather than sleeping keeps
     * the decision with the caller -- select() with three empty sets is a
     * portable sleep on POSIX and WSAEINVAL on Windows, so it is not a
     * behaviour worth papering over here. */
    if (!any) return 0;

    struct timeval  tv;
    struct timeval* ptv = NULL;
    if (timeout_ms >= 0) {
        tv.tv_sec  = (long)(timeout_ms / 1000);
        tv.tv_usec = (long)((timeout_ms % 1000) * 1000);
        ptv        = &tv;
    }

#ifdef _WIN32
    (void)nfds; /* winsock ignores it; the fd_set carries its own count */
    int rc = select(0, &rd, &wr, NULL, ptv);
#else
    int rc = select(nfds, &rd, &wr, NULL, ptv);
#endif
    if (rc < 0) {
#ifdef _WIN32
        if (WSAGetLastError() == WSAEINTR) return 0;
#else
        if (errno == EINTR) return 0;
#endif
        set_err("select");
        return NET_ERR;
    }
    if (rc == 0) return 0;

    int ready = 0;
    for (int i = 0; i < n; i++) {
        if (w[i].s == NET_INVALID_SOCK) continue;
        /* Reported only for what was ASKED. select fills the sets it was given,
         * and echoing readability to a caller that only asked about writing
         * would have it read a socket it is not ready to handle. */
        if (w[i].want_read && FD_ISSET(SOCKH(w[i].s), &rd)) w[i].can_read = 1;
        if (w[i].want_write && FD_ISSET(SOCKH(w[i].s), &wr)) w[i].can_write = 1;
        if (w[i].can_read || w[i].can_write) ready++;
    }
    return ready;
}

int net_set_recv_buffer(NetSock s, int bytes) {
    if (setsockopt(SOCKH(s), SOL_SOCKET, SO_RCVBUF, (const char*)&bytes,
                   (socklen_t)sizeof bytes) != 0) {
        set_err("setsockopt(SO_RCVBUF)");
        return NET_ERR;
    }
    return 0;
}

int net_set_send_buffer(NetSock s, int bytes) {
    if (setsockopt(SOCKH(s), SOL_SOCKET, SO_SNDBUF, (const char*)&bytes,
                   (socklen_t)sizeof bytes) != 0) {
        set_err("setsockopt(SO_SNDBUF)");
        return NET_ERR;
    }
    return 0;
}

int net_set_nonblocking(NetSock s) {
#ifdef _WIN32
    u_long mode = 1;
    if (ioctlsocket(SOCKH(s), FIONBIO, &mode) != 0) {
        set_err("ioctlsocket");
        return -1;
    }
#else
    int fl = fcntl(SOCKH(s), F_GETFL, 0);
    if (fl < 0 || fcntl(SOCKH(s), F_SETFL, fl | O_NONBLOCK) < 0) {
        set_err("fcntl");
        return -1;
    }
#endif
    return 0;
}

int net_set_nodelay(NetSock s) {
    int on = 1;
    if (setsockopt(SOCKH(s), IPPROTO_TCP, TCP_NODELAY, (const void*)&on,
                   sizeof on) != 0) {
        set_err("setsockopt TCP_NODELAY");
        return -1;
    }
    return 0;
}

void net_close(NetSock s) {
    if (s == NET_INVALID_SOCK) return;
#ifdef _WIN32
    closesocket(SOCKH(s));
#else
    close(SOCKH(s));
#endif
}
