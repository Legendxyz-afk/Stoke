/* src/server.h -- the HTTP server. M6 rung A3.
 *
 * ▶▶ THIS IS NOT src/serve.h. `ServeHost` there is the in-process driver behind
 * `bench --serve`, and every M5 gate number was measured through it; it stays
 * exactly as it is, because the moment it changes those numbers stop being
 * reproducible. This file is a new consumer that speaks HTTP and reaches the
 * engine only through the public ABI.
 *
 * The entry point is deliberately one function taking plain values: everything
 * a host would configure is an argument, there is no server object to
 * mis-sequence, and `stoke serve` in main.c is a thin argument parse over it.
 */
#ifndef STOKE_SERVER_H
#define STOKE_SERVER_H

#include "stoke.h"

typedef struct {
    const char* bind_addr;   /* NULL = 127.0.0.1 -- see net.h on the default */
    uint16_t    port;        /* 0 = an ephemeral port, reported on startup    */
    int         verbose;     /* log one line per request                     */

    /* Serve this many requests, then return. 0 = forever. A request counts
     * once its response has been FULLY WRITTEN to the socket -- every
     * response, errors included, and not one moment earlier.
     *
     * ▶ IT EXISTS FOR THE GATE, AND IT EARNS ITS KEEP THERE. The curl goldens
     * need a server that starts, answers a known number of requests and EXITS
     * on its own. Killing a process from a script is the alternative, and it
     * races: the kill can land before the last response is flushed, and the
     * golden then records a truncated body as if that were the answer.
     *
     * ⚠ THE "FULLY WRITTEN" IN THAT DEFINITION IS THE WHOLE POINT, and it was
     * learned the hard way -- counting at response-BUILD time reproduced the
     * very race this option exists to remove, from the inside. */
    int max_requests;

    /* Serve for this many seconds, then return. 0 = no limit.
     *
     * ▶▶ IT EXISTS BECAUSE A SANITIZER RUN MUST END IN A CLEAN EXIT. Gate item
     * [7] is a ten-minute randomized soak under ASan, and the traffic is
     * random, so `max_requests` cannot express "stop at the end" -- nobody
     * knows the count in advance. The obvious alternative is to kill the
     * process, and that DEFEATS THE RUN: LeakSanitizer reports at process exit,
     * so a killed server is a server whose leaks were never checked, and the
     * gate would go green on exactly the failure it exists to find. Same shape
     * of argument as max_requests above -- a server that ends itself is worth
     * more than a script that ends it. */
    int max_seconds;

    /* Close a connection that has been SILENT this long. 0 = 60 s.
     *
     * ⚠ IT APPLIES ONLY TO A CONNECTION WE ARE WAITING ON, never to one
     * that is waiting on US -- see reap_idle. Exposed because the gate
     * sets it to a couple of seconds: a request whose GENERATION outlasts
     * this value is exactly the case that used to have its finished
     * response thrown away, and a regression guard for it has to be able
     * to reach that timeout in a gate that finishes. */
    int idle_timeout_secs;

    /* ---- SSE tuning. Both are real host knobs with sane defaults, and both
     * are also what makes the two behaviours below testable at all: a
     * fifteen-second heartbeat and a 256 KiB backlog cannot be reached by a
     * gate that has to finish. 0 means "the default". ---- */

    /* How long a stream may say nothing before a `: ping` comment goes out.
     * Default 15000 ms -- inside the 60 s idle timeout common in proxies. */
    int sse_ping_ms;

    /* How many unsent bytes one streaming connection may accumulate before it
     * is disconnected and its request cancelled. Default 256 KiB. This is the
     * slow-client policy: backpressure stops at the offending connection and
     * never reaches the scheduler, because every other client's tokens come
     * off the same loop. */
    size_t sse_max_pending;

    /* SO_SNDBUF for accepted connections; 0 leaves the OS default alone.
     *
     * ⚠ THIS IS THE OTHER HALF OF sse_max_pending AND THE MEASUREMENT SAYS SO.
     * The cap above bounds what THIS PROCESS holds for a connection; the
     * kernel independently holds however much its send buffer allows, and on
     * this box that turned out to be more than 71 KB and still climbing after
     * thirty seconds of a client reading nothing. So a stated per-connection
     * bound of 256 KiB is really 256 KiB plus whatever the kernel felt like.
     * Setting this makes the stated number close to the true one -- and it is
     * also what lets a gate observe the policy in seconds rather than in a
     * minute of generated tokens. */
    int sse_send_buffer;
} ServerOpts;

/* Opens the model, binds, and serves until `max_requests` is reached (or
 * forever). Returns 0 on a clean exit, non-zero on a startup failure, having
 * printed the reason. */
int server_run(const char* gguf_path, const stoke_open_opts* engine_opts,
               const ServerOpts* srv_opts);

#endif /* STOKE_SERVER_H */
