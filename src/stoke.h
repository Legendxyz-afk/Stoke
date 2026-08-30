/* stoke.h -- the public C ABI. THIS HEADER IS THE API DOCUMENTATION.
 *
 * ⚠ v0 IS NOT FROZEN. M6 freezes it at rung A7, after `server.c` has been built
 * on it and has had the chance to prove it wrong. Until that tag, any symbol
 * here may change. After it, the rules in the M6 forward note apply: error
 * codes are never renumbered, fields are only appended, and nothing is removed.
 *
 * ▶▶ WHAT THIS IS. stoke runs one model and schedules many concurrent requests
 * against a shared paged KV pool. The engine does not own a thread: YOU call
 * `stoke_step()` in a loop, and submit/poll/cancel from wherever you like. That
 * is deliberate -- a library that starts threads imposes its threading policy
 * on every host, and the one thing every host already has is a loop.
 *
 * ▶▶ THE SHAPE OF EVERY BUFFER HERE IS CALLER-OWNS-WITH-A-CAP. You pass a
 * pointer and a capacity; the callee never writes past `cap`. This is `tok.h`'s
 * pattern, used here for the same reason: it needs no allocator agreement
 * between a library and its host, and it works identically from C, Python
 * ctypes, Go cgo or Rust.
 *
 * ⚠ THE RETURN VALUE MEANS TWO DIFFERENT THINGS AND THE DIFFERENCE IS REAL, SO
 * IT IS STATED ON EVERY FUNCTION RATHER THAN ASSUMED FROM THIS PARAGRAPH:
 *   - `stoke_tokenize` / `stoke_detokenize` return the count they WOULD have
 *     written. A return greater than `cap` means "truncated, size a buffer this
 *     big and call again", and is not an error. The total is knowable up front,
 *     so telling you it is strictly better than making you guess.
 *   - `stoke_poll` returns the count it DID write, never more than `cap`. It
 *     drains a live ring whose contents are still arriving; "how many are
 *     available" is not a stable number to report, and a caller that resized a
 *     buffer to match it would be chasing a value that moved. Poll again.
 *
 * ▶▶ ERRORS. Anything returning `stoke_status` returns STOKE_OK or a negative
 * code. Anything returning a count returns a NEGATIVE `stoke_status` on
 * failure, which is why counts are signed. `stoke_last_error()` returns a
 * human-readable message for the calling thread's most recent failure.
 *
 * There are three failure classes and they behave differently on purpose:
 *   - LOAD-TIME failures (bad path, malformed GGUF, unsupported arch, OOM
 *     while opening) are TRAPPED: `stoke_open` returns NULL, the message is in
 *     `stoke_last_error()`, and your process survives.
 *   - API MISUSE (NULL handle, unknown id, zero cap, ABI mismatch) returns an
 *     enumerated error. No trap is involved; these are your bugs and they are
 *     recoverable.
 *   - INTERNAL INVARIANT VIOLATIONS are OUR bugs and still abort the process,
 *     loudly, after printing. Aborting your process because our block
 *     accounting broke is defensible; silently returning you wrong tokens is
 *     not.
 *
 * ▶▶ THREAD SAFETY, per function, no exceptions and no "probably". EVERY
 * public function appears in this list, and tests/abi_checklist.sh fails the
 * build if one does not -- eight of the fifteen were missing when that check
 * was first run, which is what a list nobody can forget to update is for.
 *   stoke_submit / stoke_poll / stoke_cancel   SAFE from any thread.
 *   stoke_step                                 ONE thread at a time. It runs
 *                                              the forward pass; two callers
 *                                              would step the same batch twice.
 *   stoke_open_abi / stoke_close               EXTERNAL synchronization. Do not
 *              (stoke_open is the macro)       close an engine another thread
 *                                              is still stepping.
 *   stoke_tokenize / stoke_detokenize          SAFE from any thread (read-only
 *                                              against an immutable vocab).
 *   stoke_apply_chat_template                  SAFE from any thread, same
 *   stoke_has_chat_template                    reason: the vocab is immutable
 *                                              once the engine is open.
 *   stoke_chat_template_bound                  SAFE from any thread, and it
 *                                              takes no engine at all -- it is
 *                                              arithmetic on your own lengths.
 *   stoke_n_vocab / stoke_max_ctx              SAFE from any thread. They read
 *   stoke_eos_id                               facts fixed when the engine was
 *                                              opened, so the only unsafe use
 *                                              is racing them against a close.
 *   stoke_abi_version                          SAFE from any thread; it returns
 *                                              a compile-time constant.
 *   stoke_last_error                           Thread-local; you read your own
 *                                              thread's message, never another's.
 *
 * ▶▶ TEXT IS UTF-8 everywhere, in and out. Ids are opaque `uint32_t` vocabulary
 * indices and are only meaningful to the model that produced them.
 */
#ifndef STOKE_H
#define STOKE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- versioning ------------------------------------------------------- */

/* Bumped on any change that a caller compiled against the old header could
 * notice. `stoke_open` is a macro that passes this, so a header/library
 * mismatch is caught AT OPEN with a legible message instead of surfacing later
 * as a struct read at the wrong offset. Dynamic loaders that cannot use the
 * macro must call `stoke_open_abi` and pass STOKE_ABI_VERSION themselves. */
#define STOKE_ABI_VERSION 0

/* The version the LIBRARY was built with, whatever the caller's header says. */
int32_t stoke_abi_version(void);

/* ---- status codes ------------------------------------------------------
 *
 * A fixed-width int, not a bare enum: an enum's underlying type is
 * implementation-defined, which is a poor thing to put in an ABI. The names
 * below are an unnamed enum purely so they are visible to a debugger.
 *
 * ⚠ AFTER THE M6 FREEZE THESE NUMBERS ARE PERMANENT. New codes append. */
typedef int32_t stoke_status;
enum {
    STOKE_OK                = 0,
    STOKE_ERR_INVALID_ARG   = -1, /* NULL, zero cap, malformed request       */
    STOKE_ERR_ABI_MISMATCH  = -2, /* caller's STOKE_ABI_VERSION != library's */
    STOKE_ERR_OPEN_FAILED   = -3, /* trapped load-time failure; see last_error */
    STOKE_ERR_UNKNOWN_ID    = -4, /* no live request with that id            */
    STOKE_ERR_DUPLICATE_ID  = -5, /* an id already live was submitted again  */
    STOKE_ERR_TOO_LONG      = -6, /* prompt + max_tokens exceeds max_ctx     */
    STOKE_ERR_NO_TEMPLATE   = -7  /* this model's vocab has no chat template  */
};

/* Why a request stopped. Reported through `stoke_poll`'s `finish` out-param. */
enum {
    STOKE_FINISH_NONE       = 0, /* still running                            */
    STOKE_FINISH_EOS        = 1,
    STOKE_FINISH_MAX_TOKENS = 2,
    STOKE_FINISH_CANCELLED  = 3
};

/* ---- opening ----------------------------------------------------------- */

typedef struct stoke_engine stoke_engine;

/* ⚠ `size` MUST be set to sizeof(stoke_open_opts) by the caller. That is what
 * lets this struct gain fields later without breaking a binary compiled
 * against today's header: the library sees a smaller size and defaults the
 * rest. Pass NULL for opts to take every default.
 *
 * Fixed-width fields only, and no enums -- an enum inside a struct whose
 * layout crosses an ABI boundary has an implementation-defined size. */
typedef struct {
    size_t   size;        /* = sizeof(stoke_open_opts)                       */
    uint32_t max_ctx;     /* 0 = the model's own maximum                     */
    uint32_t max_seqs;    /* concurrent sequences to size the pool for; 0 = 8 */
    uint32_t n_threads;   /* 0 = the engine's default                        */
    int32_t  kv_q8;       /* 0 = f32 KV (default), 1 = q8_0 KV               */
    int32_t  dedup_prefix;/* 1 = share identical leading blocks between reqs  */
} stoke_open_opts;

/* Opens `gguf_path`. Returns NULL on failure WITHOUT killing your process --
 * check `stoke_last_error()`. This is the trapped path described at the top.
 *
 * Use the `stoke_open` macro unless you are resolving symbols dynamically. */
stoke_engine* stoke_open_abi(const char* gguf_path,
                             const stoke_open_opts* opts,
                             int32_t caller_abi_version);
#define stoke_open(path_, opts_) \
    stoke_open_abi((path_), (opts_), STOKE_ABI_VERSION)

/* Safe on NULL. Do not call while another thread is inside stoke_step. */
void stoke_close(stoke_engine* e);

/* The calling thread's most recent error message, or "" if there is none.
 * The pointer is owned by the library and stays valid until this thread's next
 * failure -- copy it if you need to keep it. Never NULL. */
const char* stoke_last_error(void);

/* ---- model facts ------------------------------------------------------- */

uint32_t stoke_n_vocab(const stoke_engine* e);  /* 0 if `e` is NULL */
uint32_t stoke_max_ctx(const stoke_engine* e);  /* the ctx this engine opened with */
uint32_t stoke_eos_id(const stoke_engine* e);

/* ---- requests ---------------------------------------------------------- */

/* ⚠⚠ `prompt_ids` IS BORROWED, NOT COPIED, AND MUST STAY ALIVE AND UNCHANGED
 * UNTIL THE REQUEST FINISHES OR IS CANCELLED. This is the single sharpest edge
 * in this API and it is not an oversight: a preempted request re-prefills from
 * its prompt, so copying would double exactly the memory the KV pool is
 * rationing. If your binding cannot guarantee the lifetime, copy the ids into
 * storage you control and free them when `stoke_poll` reports a finish reason
 * other than STOKE_FINISH_NONE.
 *
 * `size` works as it does in stoke_open_opts. */
typedef struct {
    size_t          size;        /* = sizeof(stoke_request)                  */
    uint64_t        id;          /* yours to choose; unique among LIVE reqs  */
    const uint32_t* prompt_ids;  /* BORROWED -- see above                    */
    uint32_t        n_prompt;    /* > 0                                      */
    uint32_t        max_tokens;  /* > 0                                      */
    int64_t         eos_id;      /* -1 for none; otherwise a vocab id        */

    /* ---- sampling. A ZEROED TAIL IS GREEDY, which is why these could be
     * added without breaking a caller: `size` tells the library how much of
     * this struct exists, and a request compiled against the older header
     * simply reports a smaller one and gets the greedy defaults it already
     * expected. That is the whole reason `size` is here. ---- */

    /* <= 0 is GREEDY and is the default. Greedy is not "temperature nearly
     * zero" -- it takes a separate path that is bit-identical to the argmax it
     * replaces, with no softmax and no rounding introduced on the way. */
    float    temperature;
    uint32_t top_k;          /* 0 = no truncation by rank                    */
    float    top_p;          /* >= 1, or <= 0 (unset) = no truncation        */
    float    repeat_penalty; /* <= 1 = none                                  */

    /* ⚠ A SEED IS A PROMISE THIS ENGINE CAN ACTUALLY KEEP, and that is not
     * automatic for a continuous batcher. The scheduler PREEMPTS a sequence
     * under KV pressure and later REPLAYS it, so a sampler with a running RNG
     * would produce a different token the second time through -- silently,
     * because the stream stays fluent. Draws here are a pure function of
     * (seed, position), so the same seed gives the same text whatever the
     * scheduler did to get there, and whatever else was running alongside. */
    uint64_t seed;
} stoke_request;

/* Queues a request. Nothing is queued on failure.
 *
 * Thread safety: safe from ANY thread, including while another thread is inside
 * stoke_step.
 *
 * ⚠ ONE CONSEQUENCE OF THAT, STATED RATHER THAN LEFT TO BE DISCOVERED: the
 * SAMPLING FIELDS ABOVE TAKE EFFECT FROM THE NEXT STEP THAT STARTS. A step
 * installs the sampling state for newly queued requests before it admits
 * anything, so a submit that lands in the middle of a step -- after that
 * install and before the admission -- can have the request admitted, prefilled
 * and its FIRST token chosen by the same step, greedily, with the rest of the
 * generation sampled as asked. One token, in a stream that stays fluent.
 *
 * This is an ordering, not a race, and no lock removes it: the sampling state
 * is deliberately queued only AFTER the scheduler has accepted the request, so
 * that a rejected submit leaves nothing behind for a later request reusing the
 * id. Submitting from the same thread that calls stoke_step -- which is what a
 * single-threaded host such as `stoke serve` does -- cannot hit it at all. If
 * you submit from another thread and need the first token sampled too, submit
 * between steps. */
stoke_status stoke_submit(stoke_engine* e, const stoke_request* req);

/* Runs one scheduler step: admissions, one prefill chunk, one decode batch.
 * `out_rows` (optional) receives the number of sequence-rows the step actually
 * moved -- 0 means there was nothing to do, which is how you know the queue has
 * drained rather than stalled. ONE THREAD AT A TIME. */
stoke_status stoke_step(stoke_engine* e, uint32_t* out_rows);

/* Drains tokens generated for `id` since your last poll into `out`.
 *
 * Returns the number of ids WRITTEN, 0..cap -- never more (see the note at the
 * top of this file about why poll differs from tokenize here). A full return
 * means there may be more waiting; poll again. Returns a negative stoke_status
 * on error.
 *
 * `finish` (optional) receives a STOKE_FINISH_* value. It becomes non-NONE only
 * once the request has finished AND its ring is drained, so a caller that stops
 * polling at the first non-NONE CANNOT lose a trailing token -- that ordering
 * is a guarantee, not an implementation detail.
 *
 * A non-NONE finish is also when the `prompt_ids` buffer you lent us at submit
 * becomes yours to free. */
int32_t stoke_poll(stoke_engine* e, uint64_t id, uint32_t* out, int32_t cap,
                   int32_t* finish);

/* Asks for `id` to stop. It finishes with STOKE_FINISH_CANCELLED; tokens
 * already generated remain pollable. Not instantaneous -- a step in flight
 * completes first. */
stoke_status stoke_cancel(stoke_engine* e, uint64_t id);

/* ---- text -------------------------------------------------------------- */

/* Encodes UTF-8 `text` (`len` bytes; pass -1 for NUL-terminated) into ids.
 * `add_special` non-zero prepends BOS if the tokenizer defines one.
 * Returns the id count, which may exceed `cap` -- see the caller-owns rule. */
int32_t stoke_tokenize(const stoke_engine* e, const char* text, int64_t len,
                       int32_t add_special, uint32_t* out, int32_t cap);

/* Writes the UTF-8 bytes for ONE id into `buf`. Returns the byte count, which
 * may exceed `cap`. Not NUL-terminated -- a token may legitimately contain any
 * byte, so the length is the only reliable delimiter. */
int32_t stoke_detokenize(const stoke_engine* e, uint32_t id, char* buf,
                         int32_t cap);

/* ---- chat ---------------------------------------------------------------
 *
 * ▶▶ ADDED AT RUNG A3, AND IT IS EXACTLY THE KIND OF ADDITION THE RUNG ORDER
 * EXISTS TO PRODUCE. `server.c` includes this header and nothing else from
 * src/, and the first thing it needed was a way to turn a list of chat messages
 * into ids. Without this it would have had to reach into tok.h -- and the ABI
 * would have shipped, and been frozen, unable to express the one request shape
 * this engine exists to serve. That is the failure the M6 note calls "an ABI
 * used by nothing", found by building the server ON the header rather than
 * beside it.
 *
 * Roles are "system" / "user" / "assistant" / "tool". Content is arbitrary
 * UTF-8; it is NEVER rendered to one string and re-parsed, so a message whose
 * text contains the literal bytes of a turn marker cannot forge a turn. */
typedef struct {
    const char* role;
    size_t      role_len;
    const char* content;
    size_t      content_len;
} stoke_chat_message;

/* A provable UPPER BOUND on the ids `stoke_apply_chat_template` can produce for
 * these messages, or a negative stoke_status. Pure arithmetic on the lengths --
 * it loads nothing and looks at no vocabulary.
 *
 * ⚠⚠ CALL THIS FIRST AND SIZE YOUR BUFFER TO IT. It is not a suggestion: the
 * template call REFUSES a `cap` below this bound rather than risk discovering
 * the overflow halfway through. The bound is roughly 3x the message bytes,
 * because a worst-case byte-level fallback can emit one id per byte and the
 * encoder's own contract is stated that way; a typical prompt uses a small
 * fraction of it, and the ACTUAL count comes back from the call. */
int64_t stoke_chat_template_bound(const stoke_chat_message* msgs, int32_t n);

/* Renders `msgs` as this model's chat template and encodes it, always ending
 * with the generation prompt so the result is ready to submit. Returns the id
 * count WRITTEN, or a negative stoke_status:
 *
 *   STOKE_ERR_NO_TEMPLATE  this model's vocab has no chat template. It is a
 *                          fact about the MODEL, identical for every request,
 *                          so check it once at startup and answer the chat endpoint
 *                          with a clear refusal rather than per-request.
 *   STOKE_ERR_TOO_LONG     `cap` is below stoke_chat_template_bound().
 *
 * SAFE from any thread (read-only against an immutable vocab). */
int32_t stoke_apply_chat_template(const stoke_engine* e,
                                  const stoke_chat_message* msgs, int32_t n,
                                  uint32_t* out, int32_t cap);

/* Whether this model can be chat-templated at all. A server should ask once at
 * startup, not once per request. 0 if `e` is NULL. */
int32_t stoke_has_chat_template(const stoke_engine* e);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* STOKE_H */
