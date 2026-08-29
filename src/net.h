/* src/net.h -- the socket shim. M6 rung A2.
 *
 * ▶▶ WHAT IT IS FOR. Exactly one file in this tree knows whether this is
 * Winsock or POSIX sockets, and it is src/net.c. `server.c` includes this
 * header and contains no #ifdef of its own. That is the whole purpose: the
 * platform difference is small but it is everywhere -- a different handle type,
 * a different close, a different error channel, a mandatory startup call on one
 * side and not the other -- and left unabstracted it appears in every function
 * that touches a socket.
 *
 * ▶▶ NO PLATFORM HEADER IS INCLUDED HERE. <winsock2.h> must precede
 * <windows.h> or it collides with the winsock 1.1 declarations windows.h
 * carries, and common.h already includes windows.h. Keeping the platform
 * headers inside net.c means no consumer can be ordered wrongly, and `NetSock`
 * is `intptr_t` rather than `SOCKET` or `int` for the same reason -- it holds
 * either exactly, and (NetSock)-1 is the invalid value on both (Windows'
 * INVALID_SOCKET is (UINT_PTR)~0, which is -1 read as a signed word).
 *
 * ▶ ERRORS. Creators return NET_INVALID_SOCK; net_error() then describes the
 * failure for the calling thread. Nothing here calls fatal(): a refused
 * connection or a reset peer is ordinary weather for a server, not one of the
 * M6 failure classes.
 */
#ifndef STOKE_NET_H
#define STOKE_NET_H

#include "common.h"

typedef intptr_t NetSock;

#define NET_INVALID_SOCK ((NetSock)-1)

/* net_recv's non-byte-count returns. All negative or zero, so `n > 0` is
 * always "these many bytes" and nothing else. */
#define NET_CLOSED      0  /* the peer closed cleanly; there will be no more */
#define NET_WOULD_BLOCK (-1) /* non-blocking and nothing ready; try later    */
#define NET_ERR         (-2) /* the connection is finished; see net_error()  */

/* Idempotent. WSAStartup on Windows, nothing on POSIX -- which is exactly why
 * it exists as a named call rather than as an #ifdef at the top of main().
 * The socket creators below call it themselves if you have not, because a host
 * that forgot would otherwise fail with WSANOTINITIALISED, an error whose text
 * points nowhere near the mistake. Call it early anyway if you would rather
 * learn about a broken network stack at startup than at first connection. */
int net_startup(void);

/* Binds and listens. `bind_addr` NULL means 127.0.0.1.
 *
 * ⚠ LOOPBACK IS THE DEFAULT ON PURPOSE. This process holds a language model
 * and answers whoever asks; binding 0.0.0.0 by default would put it on every
 * interface the box has because a caller passed NULL. Reaching the network is
 * a decision someone types out.
 *
 * `port` 0 asks the OS for an ephemeral one -- use net_port_of() to learn
 * which. Returns NET_INVALID_SOCK on failure. */
NetSock net_listen(const char* bind_addr, u16 port, int backlog);

/* Blocks until a client arrives. NET_INVALID_SOCK on failure. */
NetSock net_accept(NetSock listener);

/* Connects to `host` (dotted-quad or a name) on `port`. Present because the
 * shim's own tests need a client: building one out of raw platform calls would
 * leave half of this file untested by the file that exists to test it. */
NetSock net_connect(const char* host, u16 port);

/* The local port a socket is bound to, or 0 if it cannot be determined. The
 * only way to learn what port 0 turned into. */
u16 net_port_of(NetSock s);

/* > 0: that many bytes. Otherwise NET_CLOSED / NET_WOULD_BLOCK / NET_ERR.
 * An interrupted call (EINTR) is retried here rather than surfaced, because a
 * signal is not news to a server loop. */
i64 net_recv(NetSock s, void* buf, size_t cap);

/* Writes as much as the socket will take right now. Same return convention;
 * a partial write is a positive count, not an error. */
i64 net_send(NetSock s, const void* buf, size_t len);

/* Loops until every byte is gone. Returns 0, or NET_ERR.
 *
 * ⚠ It BLOCKS if the socket does, which is correct for the small writes a
 * response header is and wrong for a large body on a slow client -- that case
 * is A4's bounded per-connection buffer, not this function. */
int net_send_all(NetSock s, const void* buf, size_t len);

/* 0 on success. After this, net_recv returns NET_WOULD_BLOCK instead of
 * waiting -- the property the server loop rests on, since a scheduler blocked
 * on one client's socket has stopped generating tokens for every other. */
int net_set_nonblocking(NetSock s);

/* Disables Nagle. A streamed token is a small write whose value is entirely in
 * arriving now; Nagle would hold it back waiting for company. */
int net_set_nodelay(NetSock s);

/* Asks the kernel for a receive buffer of `bytes` on this socket. Advisory --
 * the kernel may round it, clamp it, or ignore it entirely, so nothing may
 * depend on the exact number taking effect.
 *
 * ▶ ADDED AT RUNG A4 FOR A TEST CLIENT, and stated plainly because that is
 * unusual: proving a server disconnects a client that has stopped reading
 * means being a client that has stopped reading, and the kernel will happily
 * absorb tens of kilobytes on that client's behalf before the server's own
 * buffer ever grows. Shrinking the window is how the test reaches the policy
 * in seconds instead of in a minute of generated tokens. It is a real socket
 * option a real host might want; it is simply the test that needed it first. */
int net_set_recv_buffer(NetSock s, int bytes);

/* The sending half of the same option. Also advisory.
 *
 * ▶ A SERVER HAS A REASON TO WANT THIS, and it is not only the test. A server
 * that promises "at most N unsent bytes per connection" while the kernel holds
 * an unbounded amount on its behalf is describing only half of the memory that
 * connection costs. Measured here on 2026-08-29: a client reading essentially
 * nothing for thirty seconds had 71,417 bytes in flight and climbing, none of
 * which had reached the server's own buffer yet. */
int net_set_send_buffer(NetSock s, int bytes);

/* net_connect, with SO_RCVBUF applied to the socket BEFORE the connect.
 *
 * ⚠ THE ORDERING IS THE ENTIRE VALUE OF THIS FUNCTION, and it was learned by
 * watching the other order do nothing. Windows auto-tunes the receive window
 * and stops only once SO_RCVBUF has been set explicitly, and the window is
 * first advertised during the handshake -- so a value set afterwards may never
 * take effect. Measured: a client that set 2 KiB after connecting still let
 * roughly 100 KiB accumulate in flight. */
NetSock net_connect_recvbuf(const char* host, u16 port, int recvbuf);

/* ---- readiness ----------------------------------------------------------
 *
 * ▶ ADDED AT RUNG A3, because the server needs it and A2 had no caller for it.
 * A loop that owns both sockets AND a model cannot block on either: blocking in
 * recv() stops token generation for every other client, and blocking in send()
 * on one slow reader does the same. So the loop asks which sockets are ready,
 * with a timeout it chooses -- zero when the engine has work to do, longer when
 * it is idle and there is nothing to burn a core on.
 *
 * ⚠ AT MOST NET_WAIT_MAX SOCKETS. This is select(), whose Windows fd_set is a
 * fixed array of FD_SETSIZE (64) entries -- passing more silently drops the
 * excess rather than failing, so the cap is enforced here instead. */
#define NET_WAIT_MAX 60

typedef struct {
    NetSock s;
    int     want_read, want_write; /* what the caller is interested in */
    int     can_read, can_write;   /* what net_wait found; set on return */
} NetWait;

/* Waits up to `timeout_ms` (0 polls, negative blocks) for any of `n` sockets to
 * become ready. Returns how many entries had can_read or can_write set, 0 on
 * timeout, or NET_ERR. Every entry's can_* fields are cleared first, so a
 * caller never reads a previous round's answer. */
int net_wait(NetWait* w, int n, int timeout_ms);

/* A no-op on NET_INVALID_SOCK, which is what lets a teardown path close
 * whatever it happens to hold without testing each handle first.
 *
 * ⚠ Closing a LIVE socket twice is NOT safe and this does not make it so: the
 * descriptor may already have been reused by another connection, and the second
 * close would take that one down. Set your handle to NET_INVALID_SOCK after
 * closing it. */
void net_close(NetSock s);

/* The calling thread's most recent failure, or "" if there has been none.
 * Never NULL; owned by the library and valid until this thread fails again. */
const char* net_error(void);

#endif /* STOKE_NET_H */
