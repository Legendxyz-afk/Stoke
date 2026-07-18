/* src/gguf.h -- GGUF v3 parser: mmap the file, walk metadata KVs and tensor
 * infos once at open time, and expose typed, bounds-checked accessors.
 *
 * Contract:
 *   - gguf_open() never returns NULL; any malformed input is a fatal().
 *   - Typed getters fatal on wrong-type or missing-key (except gguf_meta_has,
 *     which is the presence check and never fatals, and the _or variant,
 *     which falls back to a default ONLY when the key is entirely absent --
 *     a present-but-wrong-type key still fatals through the getter it
 *     wraps).
 *   - Every byte access is bounds-checked against the mapped file length
 *     before dereference -- no trusting counts or lengths read from the
 *     file.
 *
 * Two additions beyond the plan's literal interface block, both needed to
 * make `stoke info` implementable at all:
 *   1. Enumeration primitives (gguf_meta_count/gguf_meta_key/gguf_meta_type/
 *      gguf_meta_arr_elem_type): the by-key getters alone cannot dump all
 *      metadata KVs in file order without main.c hardcoding all 36 key
 *      names in advance.
 *   2. GgufTensor.offset: printing each tensor's file-relative offset
 *      requires exposing it independently of `data`/`nbytes` (this task's
 *      own valid_multi_tensor.gguf fixture has a deliberate gap between two
 *      tensors, so offset is not always derivable from cumulative nbytes)
 *      and the data-section base address isn't exposed anywhere else in
 *      the API to recover it from `data` by subtraction.
 */
#ifndef STOKE_GGUF_H
#define STOKE_GGUF_H

#include "common.h"

#include <stdbool.h>

typedef struct GgufFile GgufFile;   /* opaque; owns the mmap and all parsed
                                        state */

/* Opens and fully parses `path`. Every metadata KV and tensor_info is walked
 * and bounds-checked before this returns; a malformed file is a fatal(), so
 * a successful return means the header/KV/tensor_info structure (not tensor
 * *contents*, which are opaque bytes) is sound. Never returns NULL. */
GgufFile* gguf_open(const char* path);

/* Unmaps the file and frees every allocation owned by `f`. `f` may be NULL
 * (no-op), matching free()'s convention. */
void gguf_close(GgufFile* f);

/* ---- metadata: typed getters, by key ----
 * All fatal if `key` is absent, EXCEPT gguf_meta_has (presence-only, never
 * fatal) and gguf_meta_i64_or (falls back to `dflt` when `key` is absent).
 */

bool gguf_meta_has(const GgufFile* f, const char* key);

/* Accepts UINT8/INT8/UINT16/INT16/UINT32/INT32/UINT64/INT64/BOOL, widening
 * to i64; fatal on STRING/ARRAY/FLOAT32/FLOAT64 or a missing key. A UINT64
 * whose value exceeds INT64_MAX is rejected (fatal), never wrapped. */
i64 gguf_meta_i64(const GgufFile* f, const char* key);

/* Like gguf_meta_i64, but returns `dflt` if `key` is absent instead of
 * fataling. A present-but-wrong-type key still fatals (see file header
 * comment). Concrete consumer: general.alignment (absent -> default 32).
 * The only `_or` variant this task implements. */
i64 gguf_meta_i64_or(const GgufFile* f, const char* key, i64 dflt);

/* Fatal on any key whose declared type is not FLOAT32 (incl. missing). */
f32 gguf_meta_f32(const GgufFile* f, const char* key);

/* Fatal if `key` is absent or not STRING. Returned pointer is a
 * NUL-terminated copy owned by `f`; valid until gguf_close(f). */
const char* gguf_meta_str(const GgufFile* f, const char* key);

/* ---- metadata: array getters (tokenizer tables) ----
 * All fatal if `key` is absent or not an ARRAY.
 */

/* Element count of the array. O(1) -- does not walk/materialize elements. */
u64 gguf_meta_arr_len(const GgufFile* f, const char* key);

/* Fatal if the array's element type isn't STRING, or if i >= length.
 * Returned pointer is a NUL-terminated copy owned by `f`, valid until
 * gguf_close(f). First call for a given key walks+caches the whole array
 * (O(n) once, since STRING elements aren't fixed-width); later calls for
 * the same key are O(1). This lazy cache is NOT thread-safe: calling this
 * for the same key on the same GgufFile from multiple threads without
 * external synchronization races if more than one of those calls is a
 * key's first (the cache-building assignments are unsynchronized writes
 * through a const-cast -- see gguf.c's ensure_str_array_cached). */
const char* gguf_meta_arr_str(const GgufFile* f, const char* key, u64 i);

/* Widens any integer array-element type to i64 (UINT8/INT8/UINT16/INT16/
 * UINT32/INT32/UINT64/INT64); fatal on BOOL/STRING/FLOAT32/FLOAT64/ARRAY
 * elements or i >= length. O(1) always (fixed element width).
 *
 * Note: unlike the scalar gguf_meta_i64 (which does widen a scalar BOOL),
 * this getter does NOT widen BOOL array elements -- BOOL-array widening
 * was never specified as included here, and the target model has no BOOL
 * array KV to verify real behavior against, so this getter stays
 * conservative (fatal) rather than guessing at untested semantics. Add it
 * (with real fixture + golden coverage) the day a model that has one
 * arrives. */
i64 gguf_meta_arr_i64(const GgufFile* f, const char* key, u64 i);

/* gguf_meta_arr_f32 is intentionally NOT implemented: the target model
 * has no FLOAT32 array KV, so this would be dead, untestable code. Add it
 * (with real fixture + golden coverage) the day a model that has one
 * arrives. */

/* ---- metadata enumeration (implementer addition; see file header) ---- */

/* Number of top-level metadata KVs, file order. */
u64 gguf_meta_count(const GgufFile* f);

/* Key of KV `i` (file order). Fatal if i >= gguf_meta_count(f). Returned
 * pointer is owned by `f`, valid until gguf_close(f). */
const char* gguf_meta_key(const GgufFile* f, u64 i);

/* Declared GGUF_TYPE_* of `key` (GGUF_TYPE_ARRAY for arrays -- use
 * gguf_meta_arr_elem_type for the element type). Fatal if key absent. */
u32 gguf_meta_type(const GgufFile* f, const char* key);

/* Declared GGUF_TYPE_* of `key`'s array elements. Fatal if key absent or
 * not an ARRAY. */
u32 gguf_meta_arr_elem_type(const GgufFile* f, const char* key);

/* ---- tensors ---- */

typedef struct {
    const char* name;    /* NUL-terminated, owned by the GgufFile */
    u32         type;    /* GGUF_GGML_TYPE_* */
    u32         n_dims;  /* 1..4 */
    u64         dims[4];  /* dims[0..n_dims-1] valid, fastest-moving-first
                             (ggml ne[] order) exactly as stored in the file;
                             dims[n_dims..3] are zero */
    const void* data;    /* points into the mmap; nbytes long; read-only.
                             BORROWED from the mapping -- invalid the
                             instant gguf_close(f) runs, unlike Tok
                             (src/tok.h), which copies what it needs and
                             stays valid after its GgufFile is closed; a
                             future consumer of this field (e.g. a model
                             loader) must either keep `f` open for as long
                             as it reads tensor data, or copy the bytes out
                             before closing it. */
    u64         nbytes;
    u64         offset;  /* implementer addition -- see file header comment
                             point 2. As stored in the file: relative to the
                             data section start, NOT to the file start and
                             NOT the same as (data - mmap base). */
} GgufTensor;

u64 gguf_tensor_count(const GgufFile* f);

/* Fatal if i >= gguf_tensor_count(f). Returned pointer is owned by `f`,
 * valid until gguf_close(f). */
const GgufTensor* gguf_tensor(const GgufFile* f, u64 i);

/* Fatal if no tensor named `name` exists. */
const GgufTensor* gguf_tensor_by_name(const GgufFile* f, const char* name);

/* ---- ggml tensor-dtype enum --
 * Only the 3 values verified against the real file's bytes are named.
 * gguf_open() fatals on any tensor using a ggml_type outside this set --
 * later milestones add types only as each is probed against real bytes
 * (project "probe, don't guess" protocol). */
enum {
    GGUF_GGML_TYPE_F32  = 0,
    GGUF_GGML_TYPE_Q4_K = 12,
    GGUF_GGML_TYPE_Q6_K = 14,
};

/* ---- GGUF metadata value-type enum (from the official ggml gguf.md
 * spec) -- exposed so callers (stoke info) can interpret
 * gguf_meta_type()/gguf_meta_arr_elem_type() results. */
enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

#endif /* STOKE_GGUF_H */
