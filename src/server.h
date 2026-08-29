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
} ServerOpts;

/* Opens the model, binds, and serves until `max_requests` is reached (or
 * forever). Returns 0 on a clean exit, non-zero on a startup failure, having
 * printed the reason. */
int server_run(const char* gguf_path, const stoke_open_opts* engine_opts,
               const ServerOpts* srv_opts);

#endif /* STOKE_SERVER_H */
