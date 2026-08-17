/* src/gguf.c -- GGUF v3 parser + mmap. See src/gguf.h for the public
 * contract. Every constant here (magic, value-type enum, ggml type-size
 * table, default alignment) was taken verbatim from the GGUF wire format,
 * probed against and independently byte-level re-verified against the
 * real model file this engine targets.
 *
 * Design summary:
 *   - One pass over the file at gguf_open() time parses every metadata KV
 *     and tensor_info, bounds-checking each read against the mapped length
 *     BEFORE dereferencing it. A malformed file fatal()s during this pass;
 *     gguf_open() never returns on bad input.
 *   - Small, bounded data (keys, tensor names, scalar STRING values -- at
 *     most a few hundred items in the real file) is eagerly duplicated into
 *     owned, NUL-terminated buffers, since the raw file bytes are NOT
 *     NUL-terminated at each field's end.
 *   - ARRAY<STRING> values (tokenizer.ggml.tokens/merges: 150K+ elements in
 *     the real file) are only ever WALKED (bounds-checked, no allocation)
 *     during the initial parse, to find where the array ends. The actual
 *     per-element strings are lazily indexed+duplicated on first
 *     gguf_meta_arr_str() call for that key, and cached -- so `stoke info`
 *     (which only ever calls gguf_meta_arr_len()) never pays that cost,
 *     while a caller that does want the strings (the tokenizer) gets O(1)
 *     amortized access after the first call.
 *   - ARRAY<integer> elements are fixed-width, so gguf_meta_arr_i64() reads
 *     directly by computed offset -- no walking or caching needed.
 */
#include "common.h"

#include "gguf.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h> /* getenv -- see map_pin */
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ===========================================================================
 * Platform mmap helpers
 * ===========================================================================
 */

typedef struct {
    const u8* base; /* NULL only when size==0 */
    u64       size;
#ifdef _WIN32
    HANDLE file_h;
    HANDLE map_h;
#else
    int fd;
#endif
} MappedFile;

/* ---------------------------------------------------------------------------
 * map_pin -- defend the mapping against the OS working-set manager.
 *
 * WHY THIS EXISTS, measured 2026-08-17. Decode streams every weight once per
 * token, so the model IS the working set; there is no locality to exploit and
 * nothing to gain from paging any of it out. Windows does not know that. Under
 * memory pressure its working-set manager trimmed a running decode from a
 * 2448 MB working set down to 183-441 MB *mid-run*, and the six worker threads
 * then re-faulted the pages back through one per-process working-set lock:
 * 2.2-2.7 MILLION soft faults per 128-token run, and 15.4 tok/s became 2.2.
 *
 * The signature is worth recording because it fooled two hypotheses first.
 * There is NO disk I/O (hard faults ~0 -- these are soft faults against pages
 * still in RAM) and NO compression (the Memory Compression process sat at 0),
 * so every disk- and residency-shaped explanation is wrong. It is also not
 * core placement: it reproduces under an affinity mask that makes hyperthread
 * stacking impossible. And it is invisible in an A/B against llama.cpp,
 * because llama.cpp calls PrefetchVirtualMemory on its mapping and therefore
 * never enters this state -- which is why llama measured rock-steady in the
 * same interleave where stoke swung 5.6x.
 *
 * Two defences, both BEST EFFORT -- a failure here costs speed and can never
 * cost correctness, so nothing in this function is allowed to fatal():
 *   (1) a HARD minimum working set covering the mapping, which is what
 *       actually stops the trimming; and
 *   (2) one PrefetchVirtualMemory call to fault the mapping in as a batch
 *       rather than as ~600K individual faults.
 *
 * Set STOKE_NO_PIN=1 to disable both. That escape is deliberately an
 * environment variable and not a flag: it makes the A/B a property of the RUN
 * rather than of the BINARY, so the two arms can be interleaved without a
 * rebuild and without two executables to keep straight.
 * ---------------------------------------------------------------------------
 */
/* Presence alone must NOT mean "disabled": STOKE_NO_PIN=0 reading as "no pin"
 * is the kind of surprise that costs an afternoon of A/B runs. Empty and "0"
 * both mean leave pinning ON. */
static bool stoke_no_pin(void) {
    const char* v = getenv("STOKE_NO_PIN");
    return v != NULL && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

#ifdef _WIN32
#ifndef QUOTA_LIMITS_HARDWS_MIN_ENABLE
#define QUOTA_LIMITS_HARDWS_MIN_ENABLE 0x00000001
#endif

/* Declared ourselves rather than relying on _WIN32_WINNT being high enough in
 * whichever mingw headers are in play; the layout is fixed ABI. */
typedef struct {
    PVOID  VirtualAddress;
    SIZE_T NumberOfBytes;
} StokeMemRange;
typedef BOOL(WINAPI* PrefetchVirtualMemoryFn)(HANDLE, ULONG_PTR, StokeMemRange*,
                                              ULONG);

/* SE_INC_WORKING_SET_NAME is held by ordinary users but DISABLED in the token,
 * and SetProcessWorkingSetSizeEx refuses to RAISE a minimum without it. The
 * first cut of map_pin() skipped this and ignored the return value, so the
 * request failed silently and the fix looked like it did not work -- which is
 * why the outcome is now reported rather than assumed. */
static bool enable_privilege(const char* name) {
    HANDLE tok = NULL;
    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        return false;
    }
    bool ok = false;
    LUID luid;
    if (LookupPrivilegeValueA(NULL, name, &luid)) {
        TOKEN_PRIVILEGES tp;
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Luid = luid;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        if (AdjustTokenPrivileges(tok, FALSE, &tp, sizeof(tp), NULL, NULL)) {
            /* AdjustTokenPrivileges reports success even when it changed
             * nothing; only GetLastError distinguishes the two. */
            ok = (GetLastError() == ERROR_SUCCESS);
        }
    }
    CloseHandle(tok);
    return ok;
}

static void map_pin(MappedFile* mf) {
    if (!mf->base || mf->size == 0) return;
    if (stoke_no_pin()) return;

    HANDLE self = GetCurrentProcess();
    bool   priv = enable_privilege("SeIncreaseWorkingSetPrivilege");
    bool   hard_ws = false;
    DWORD  ws_err = 0;

    /* (1) The hard floor. Ask for the mapping plus slack for everything else
     * the process touches (activations, KV pool, tokenizer tables). If the
     * request is refused -- too little RAM, policy, whatever -- we keep the old
     * behaviour, which is the pre-2026-08-17 status quo. */
    SIZE_T lo = 0, hi = 0;
    DWORD  flags = 0;
    /* ▶ THE SLACK IS NOT COSMETIC. The floor has to cover the mapping AND
     * everything else the process keeps hot -- KV pool, activation scratch,
     * tokenizer tables, code. Set it to the mapping alone and the process sits
     * pinned AT its floor under memory pressure, so every page it needs beyond
     * the model evicts another: the working set reads perfectly STABLE while
     * pages churn through it, which is precisely the trace measured on
     * 2026-08-17 (WS flat at 2430 MB, 1.5M transition faults, 6 threads
     * delivering 2.6 cores of work). Tunable so the floor can be swept without
     * a rebuild, since the right value is a property of the box, not the code. */
    SIZE_T slack_mb = 512;
    {
        const char* s = getenv("STOKE_PIN_SLACK_MB");
        if (s && s[0]) {
            long v = strtol(s, NULL, 10);
            if (v >= 0 && v <= 8192) slack_mb = (SIZE_T)v;
        }
    }
    if (GetProcessWorkingSetSizeEx(self, &lo, &hi, &flags)) {
        SIZE_T want = (SIZE_T)mf->size + (slack_mb << 20);
        if (want > lo) {
            SIZE_T want_hi = want + (SIZE_T)(256u << 20);
            if (hi > want_hi) want_hi = hi;
            hard_ws = SetProcessWorkingSetSizeEx(self, want, want_hi,
                                                 QUOTA_LIMITS_HARDWS_MIN_ENABLE)
                          ? true
                          : false;
            if (!hard_ws) ws_err = GetLastError();
        } else {
            hard_ws = true; /* already covered */
        }
    }
    if (getenv("STOKE_PIN_DEBUG")) {
        fprintf(stderr,
                "pin: privilege=%s hard_ws=%s%s (mapping %.2f GiB)\n",
                priv ? "yes" : "no", hard_ws ? "yes" : "NO",
                hard_ws ? "" : " <-- the model can be trimmed mid-run",
                (double)mf->size / (1024.0 * 1024.0 * 1024.0));
        if (!hard_ws && ws_err) {
            fprintf(stderr, "pin: SetProcessWorkingSetSizeEx error %lu\n",
                    (unsigned long)ws_err);
        }
    }

    /* (2) One batched fault-in. Resolved dynamically: it is Windows 8+, and a
     * missing entry point must degrade to "the old way", not to a link error
     * on an older toolchain. */
    HANDLE k32 = GetModuleHandleA("kernel32.dll");
    if (k32) {
        PrefetchVirtualMemoryFn prefetch =
            (PrefetchVirtualMemoryFn)(void (*)(void))GetProcAddress(
                (HMODULE)k32, "PrefetchVirtualMemory");
        if (prefetch) {
            StokeMemRange r;
            r.VirtualAddress = (PVOID)(void*)mf->base;
            r.NumberOfBytes = (SIZE_T)mf->size;
            (void)prefetch(self, 1, &r, 0);
        }
    }
}
#else
static void map_pin(MappedFile* mf) {
    if (!mf->base || mf->size == 0) return;
    if (stoke_no_pin()) return;
    /* posix_madvise, not madvise: the latter is not POSIX and _POSIX_C_SOURCE
     * (common.h) hides it under -std=c11. WILLNEED is advisory -- Linux reads
     * the pages in but will still reclaim under pressure; mlock() is the hard
     * equivalent of the Windows path above and is deliberately NOT taken here,
     * since on Linux it needs a raised RLIMIT_MEMLOCK and would fail loudly on
     * exactly the constrained boxes this engine targets. Revisit with a
     * measurement if Linux ever shows the same trimming signature. */
    (void)posix_madvise((void*)(uintptr_t)mf->base, (size_t)mf->size,
                        POSIX_MADV_WILLNEED);
}
#endif

#ifdef _WIN32
static void map_open(const char* path, MappedFile* mf) {
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fatal("gguf: cannot open '%s' -- CreateFileA failed (error %lu) -- "
              "does the file exist?",
              path, (unsigned long)GetLastError());
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) {
        DWORD err = GetLastError();
        CloseHandle(fh);
        fatal("gguf: cannot stat '%s' -- GetFileSizeEx failed (error %lu)",
              path, (unsigned long)err);
    }
    if (sz.QuadPart < 0) {
        CloseHandle(fh);
        fatal("gguf: '%s' -- GetFileSizeEx returned a negative size", path);
    }
    u64 size = (u64)sz.QuadPart;

    HANDLE mh = NULL;
    const u8* base = NULL;
    if (size > 0) {
        mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
        if (mh == NULL) {
            DWORD err = GetLastError();
            CloseHandle(fh);
            fatal("gguf: '%s' -- CreateFileMappingA failed (error %lu)", path,
                  (unsigned long)err);
        }
        base = (const u8*)MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
        if (base == NULL) {
            DWORD err = GetLastError();
            CloseHandle(mh);
            CloseHandle(fh);
            fatal("gguf: '%s' -- MapViewOfFile failed (error %lu)", path,
                  (unsigned long)err);
        }
    }
    mf->base = base;
    mf->size = size;
    mf->file_h = fh;
    mf->map_h = mh;
    map_pin(mf);
}

static void map_close(MappedFile* mf) {
    if (mf->base) UnmapViewOfFile((LPCVOID)mf->base);
    if (mf->map_h) CloseHandle(mf->map_h);
    if (mf->file_h != INVALID_HANDLE_VALUE) CloseHandle(mf->file_h);
}
#else
static void map_open(const char* path, MappedFile* mf) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fatal("gguf: cannot open '%s' -- %s", path, strerror(errno));
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        int e = errno;
        close(fd);
        fatal("gguf: cannot stat '%s' -- %s", path, strerror(e));
    }
    if (st.st_size < 0) {
        close(fd);
        fatal("gguf: '%s' -- fstat returned a negative size", path);
    }
    u64 size = (u64)st.st_size;

    const u8* base = NULL;
    if (size > 0) {
        void* p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            int e = errno;
            close(fd);
            fatal("gguf: '%s' -- mmap failed -- %s", path, strerror(e));
        }
        base = (const u8*)p;
    }
    mf->base = base;
    mf->size = size;
    mf->fd = fd;
    map_pin(mf);
}

static void map_close(MappedFile* mf) {
    if (mf->base) munmap((void*)mf->base, mf->size);
    if (mf->fd >= 0) close(mf->fd);
}
#endif

/* ===========================================================================
 * Bounds-checked, explicit-little-endian byte readers (no dereference
 * without a preceding check; the "len > size - off" form is
 * required specifically because it cannot overflow, unlike "off + len >
 * size").
 * ===========================================================================
 */

static void require_fits(u64 size, u64 off, u64 len, const char* what) {
    if (off > size) {
        fatal("gguf: %s -- offset %llu is past end of file (size %llu)",
              what, (unsigned long long)off, (unsigned long long)size);
    }
    if (len > size - off) {
        fatal("gguf: %s -- needs %llu bytes at offset %llu but only %llu "
              "remain (file size %llu) -- truncated or corrupt file",
              what, (unsigned long long)len, (unsigned long long)off,
              (unsigned long long)(size - off), (unsigned long long)size);
    }
}

/* count is attacker-controlled (a u64 read straight from the file); this
 * rejects any count that cannot possibly fit in the remaining bytes given
 * each entry's cheapest possible encoding, WITHOUT computing count*min_bytes
 * (which could itself overflow) -- division is overflow-free. */
static void require_plausible_count(u64 count, u64 remaining, u64 min_bytes,
                                     const char* what) {
    if (count > remaining / min_bytes) {
        fatal("gguf: %s -- count %llu is implausible for %llu remaining "
              "bytes (needs >= %llu bytes/entry) -- corrupt or hostile file",
              what, (unsigned long long)count, (unsigned long long)remaining,
              (unsigned long long)min_bytes);
    }
}

static u64 checked_mul_u64(u64 a, u64 b, const char* what) {
    if (a != 0 && b > UINT64_MAX / a) {
        fatal("gguf: %s -- multiplication overflow (%llu * %llu)", what,
              (unsigned long long)a, (unsigned long long)b);
    }
    return a * b;
}

static u8 rd_u8(const u8* base, u64 size, u64 off) {
    require_fits(size, off, 1, "u8 read");
    return base[off];
}
static u16 rd_u16(const u8* base, u64 size, u64 off) {
    require_fits(size, off, 2, "u16 read");
    return (u16)base[off] | ((u16)base[off + 1] << 8);
}
static u32 rd_u32(const u8* base, u64 size, u64 off) {
    require_fits(size, off, 4, "u32 read");
    return (u32)base[off] | ((u32)base[off + 1] << 8) |
           ((u32)base[off + 2] << 16) | ((u32)base[off + 3] << 24);
}
static u64 rd_u64(const u8* base, u64 size, u64 off) {
    require_fits(size, off, 8, "u64 read");
    u64 lo = (u64)base[off] | ((u64)base[off + 1] << 8) |
             ((u64)base[off + 2] << 16) | ((u64)base[off + 3] << 24);
    u64 hi = (u64)base[off + 4] | ((u64)base[off + 5] << 8) |
             ((u64)base[off + 6] << 16) | ((u64)base[off + 7] << 24);
    return lo | (hi << 32);
}
static f32 rd_f32(const u8* base, u64 size, u64 off) {
    u32 bits = rd_u32(base, size, off);
    f32 v;
    memcpy(&v, &bits, sizeof v); /* well-defined reinterpretation, no
                                    aliasing/alignment UB */
    return v;
}
static f64 rd_f64(const u8* base, u64 size, u64 off) {
    u64 bits = rd_u64(base, size, off);
    f64 v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

static char* dup_bytes_nul(const u8* base, u64 off, u64 len) {
    char* s = (char*)xmalloc((size_t)len + 1);
    memcpy(s, base + off, (size_t)len);
    s[len] = '\0';
    return s;
}

/* Fixed byte width of a scalar/array-element GGUF_TYPE_*; 0 for STRING and
 * ARRAY, which are not fixed width. */
static u64 fixed_width(u32 type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static bool is_integer_type(u32 type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
            return true;
        default:
            return false;
    }
}

/* ggml tensor dtype size table. ONLY these four values were verified
 * against real bytes; gguf_open() fatals on anything else (see
 * parse_tensor_info). Do not extend this from memory. Q8_0: 32 elems /
 * 34 bytes per block ({f16 d; i8 qs[32]}), probed from b10037 source
 * (ggml-common.h block_q8_0) and hand-annotated byte-for-byte against a
 * real block of the Q8_0 parity GGUF. */
typedef struct {
    u32 type;
    u64 blck_size;
    u64 type_size;
} GgmlTypeInfo;

static const GgmlTypeInfo GGML_TYPES[] = {
    {GGUF_GGML_TYPE_F32, 1, 4},
    {GGUF_GGML_TYPE_Q8_0, 32, 34},
    {GGUF_GGML_TYPE_Q4_K, 256, 144},
    {GGUF_GGML_TYPE_Q6_K, 256, 210},
};

static bool ggml_type_info(u32 type, u64* blck_size, u64* type_size) {
    for (size_t i = 0; i < sizeof(GGML_TYPES) / sizeof(GGML_TYPES[0]); i++) {
        if (GGML_TYPES[i].type == type) {
            *blck_size = GGML_TYPES[i].blck_size;
            *type_size = GGML_TYPES[i].type_size;
            return true;
        }
    }
    return false;
}

/* ===========================================================================
 * Internal metadata KV representation
 * ===========================================================================
 */

typedef struct {
    char* key;
    u32   type; /* GGUF_TYPE_* */

    /* valid when type is a fixed-width scalar (not STRING, not ARRAY) */
    union {
        u8  u8v;
        i8  i8v;
        u16 u16v;
        i16 i16v;
        u32 u32v;
        i32 i32v;
        u64 u64v;
        i64 i64v;
        f32 f32v;
        f64 f64v;
        u8  boolv;
    } scalar;

    char* str_val; /* owned, NUL-terminated; valid when type==STRING */

    /* valid when type==ARRAY */
    u32 arr_elem_type;
    u64 arr_count;
    u64 arr_elements_off; /* mmap byte offset of element 0's encoding */

    /* lazy cache for ARRAY<STRING>, built on first gguf_meta_arr_str() for
     * this key (see file header comment); NULL until then. */
    char** arr_str_cache; /* arr_count owned pointers into arr_str_blob */
    char*  arr_str_blob;  /* single owned buffer backing arr_str_cache */
} GgufKv;

struct GgufFile {
    MappedFile mf;

    GgufKv* kvs;
    u64     kv_count;

    GgufTensor* tensors;
    u64         tensor_count;
};

/* ===========================================================================
 * Parse: one metadata KV
 * ===========================================================================
 */

static void parse_kv(const u8* base, u64 size, u64* cur, GgufKv* out) {
    out->str_val = NULL;
    out->arr_str_cache = NULL;
    out->arr_str_blob = NULL;

    u64 key_len = rd_u64(base, size, *cur);
    *cur += 8;
    require_fits(size, *cur, key_len, "metadata key bytes");
    out->key = dup_bytes_nul(base, *cur, key_len);
    *cur += key_len;

    u32 vtype = rd_u32(base, size, *cur);
    *cur += 4;
    if (vtype > GGUF_TYPE_FLOAT64) {
        fatal("gguf: metadata key '%s' has invalid value_type %u (must be "
              "0..12)",
              out->key, vtype);
    }
    out->type = vtype;

    if (vtype == GGUF_TYPE_ARRAY) {
        u32 elem_type = rd_u32(base, size, *cur);
        *cur += 4;
        if (elem_type == GGUF_TYPE_ARRAY) {
            fatal("gguf: metadata key '%s' is an ARRAY of ARRAY -- "
                  "unsupported (nested arrays are unverified against real "
                  "GGUF files)",
                  out->key);
        }
        if (elem_type > GGUF_TYPE_FLOAT64) {
            fatal("gguf: metadata key '%s' array element type %u is "
                  "invalid (must be 0..12, non-ARRAY)",
                  out->key, elem_type);
        }
        u64 count = rd_u64(base, size, *cur);
        *cur += 8;
        out->arr_elem_type = elem_type;
        out->arr_count = count;
        out->arr_elements_off = *cur;

        u64 width = fixed_width(elem_type);
        if (width != 0) {
            u64 total = checked_mul_u64(count, width, "array element span");
            require_fits(size, *cur, total, "array elements");
            *cur += total;
        } else {
            /* STRING elements: not fixed stride -- must walk one by
             * one to find where the array ends. Bounds-checked, no
             * allocation; see file header comment for why this is
             * unconditional while duplication is lazy. */
            for (u64 i = 0; i < count; i++) {
                u64 elen = rd_u64(base, size, *cur);
                *cur += 8;
                require_fits(size, *cur, elen, "array string element bytes");
                *cur += elen;
            }
        }
        return;
    }

    if (vtype == GGUF_TYPE_STRING) {
        u64 slen = rd_u64(base, size, *cur);
        *cur += 8;
        require_fits(size, *cur, slen, "metadata string value bytes");
        out->str_val = dup_bytes_nul(base, *cur, slen);
        *cur += slen;
        return;
    }

    u64 width = fixed_width(vtype);
    if (width == 0) {
        fatal("gguf: metadata key '%s' -- internal: no fixed width for "
              "type %u",
              out->key, vtype);
    }
    require_fits(size, *cur, width, "metadata scalar value bytes");
    switch (vtype) {
        case GGUF_TYPE_UINT8:
            out->scalar.u8v = rd_u8(base, size, *cur);
            break;
        case GGUF_TYPE_INT8:
            out->scalar.i8v = (i8)rd_u8(base, size, *cur);
            break;
        case GGUF_TYPE_BOOL: {
            u8 b = rd_u8(base, size, *cur);
            if (b != 0 && b != 1) {
                fatal("gguf: metadata key '%s' -- BOOL value byte is %u, "
                      "must be 0 or 1",
                      out->key, b);
            }
            out->scalar.boolv = b;
            break;
        }
        case GGUF_TYPE_UINT16:
            out->scalar.u16v = rd_u16(base, size, *cur);
            break;
        case GGUF_TYPE_INT16:
            out->scalar.i16v = (i16)rd_u16(base, size, *cur);
            break;
        case GGUF_TYPE_UINT32:
            out->scalar.u32v = rd_u32(base, size, *cur);
            break;
        case GGUF_TYPE_INT32:
            out->scalar.i32v = (i32)rd_u32(base, size, *cur);
            break;
        case GGUF_TYPE_FLOAT32:
            out->scalar.f32v = rd_f32(base, size, *cur);
            break;
        case GGUF_TYPE_UINT64:
            out->scalar.u64v = rd_u64(base, size, *cur);
            break;
        case GGUF_TYPE_INT64:
            out->scalar.i64v = (i64)rd_u64(base, size, *cur);
            break;
        case GGUF_TYPE_FLOAT64:
            out->scalar.f64v = rd_f64(base, size, *cur);
            break;
        default:
            fatal("gguf: metadata key '%s' -- internal: unhandled scalar "
                  "type %u",
                  out->key, vtype);
    }
    *cur += width;
}

/* ===========================================================================
 * Parse: one tensor_info. `data` is left NULL; gguf_open() fills it in once
 * the data section start (which depends on general.alignment, itself a
 * metadata KV) is known, by combining that start with out->offset (the
 * file-encoded offset relative to the data section, set here).
 * ===========================================================================
 */

static void parse_tensor_info(const u8* base, u64 size, u64* cur,
                               GgufTensor* out) {
    u64 name_len = rd_u64(base, size, *cur);
    *cur += 8;
    require_fits(size, *cur, name_len, "tensor name bytes");
    out->name = dup_bytes_nul(base, *cur, name_len);
    *cur += name_len;

    u32 n_dims = rd_u32(base, size, *cur);
    *cur += 4;
    if (n_dims > 4) {
        fatal("gguf: tensor '%s' has n_dims=%u (must be <= 4)", out->name,
              n_dims);
    }
    out->n_dims = n_dims;
    for (u32 d = 0; d < 4; d++) out->dims[d] = 0;
    for (u32 d = 0; d < n_dims; d++) {
        out->dims[d] = rd_u64(base, size, *cur);
        *cur += 8;
    }

    u32 ggml_type = rd_u32(base, size, *cur);
    *cur += 4;

    out->offset = rd_u64(base, size, *cur);
    *cur += 8;

    u64 blck_size, type_size;
    if (!ggml_type_info(ggml_type, &blck_size, &type_size)) {
        fatal("gguf: tensor '%s' uses unsupported ggml_type %u (only "
              "F32=0, Q8_0=8, Q4_K=12, Q6_K=14 are implemented; later "
              "milestones add types only once probed against real bytes)",
              out->name, ggml_type);
    }
    out->type = ggml_type;

    u64 nelements = 1;
    for (u32 d = 0; d < n_dims; d++) {
        nelements = checked_mul_u64(nelements, out->dims[d],
                                     "tensor element count");
    }
    if (nelements % blck_size != 0) {
        fatal("gguf: tensor '%s' -- element count %llu is not a multiple "
              "of block size %llu for its type",
              out->name, (unsigned long long)nelements,
              (unsigned long long)blck_size);
    }
    out->nbytes = checked_mul_u64(nelements / blck_size, type_size,
                                   "tensor byte size");
    out->data = NULL;
}

/* ===========================================================================
 * gguf_open / gguf_close
 * ===========================================================================
 */

GgufFile* gguf_open(const char* path) {
    MappedFile mf;
    map_open(path, &mf);

    const u8* base = mf.base;
    u64       size = mf.size;

    if (size < 24) {
        map_close(&mf);
        fatal("gguf: '%s' -- file is %llu bytes, too small for a GGUF "
              "header (need >= 24) -- truncated or not a GGUF file",
              path, (unsigned long long)size);
    }
    if (memcmp(base, "GGUF", 4) != 0) {
        u8 m0 = base[0], m1 = base[1], m2 = base[2], m3 = base[3];
        map_close(&mf);
        fatal("gguf: '%s' -- bad magic (expected \"GGUF\", got bytes "
              "%02x %02x %02x %02x) -- not a GGUF file",
              path, m0, m1, m2, m3);
    }
    u32 version = rd_u32(base, size, 4);
    if (version != 3) {
        map_close(&mf);
        fatal("gguf: '%s' -- unsupported version %u (only v3 is "
              "supported)",
              path, version);
    }
    u64 tensor_count = rd_u64(base, size, 8);
    u64 kv_count = rd_u64(base, size, 16);

    u64 cur = 24;
    /* Conservative plausibility pre-checks against the whole remaining
     * file: the KV section and tensor_info section together can consume
     * at most size-24 bytes, so bounding each count against that whole
     * remainder never rejects a valid file, while rejecting any count
     * that's flatly impossible before we allocate or loop on it. Minimums:
     * a KV needs >= 8 (key_len) + 0 (key) + 4
     * (value_type) + 1 (smallest value, UINT8/INT8/BOOL) = 13 bytes; a
     * tensor_info needs >= 8 (name_len) + 0 (name) + 4 (n_dims, allowing
     * n_dims==0) + 4 (ggml_type) + 8 (offset) = 24 bytes. */
    require_plausible_count(kv_count, size - cur, 13, "metadata_kv_count");
    require_plausible_count(tensor_count, size - cur, 24, "tensor_count");

    GgufKv* kvs = (GgufKv*)xcalloc(kv_count ? kv_count : 1, sizeof(GgufKv));
    for (u64 i = 0; i < kv_count; i++) {
        parse_kv(base, size, &cur, &kvs[i]);
    }

    GgufTensor* tensors =
        (GgufTensor*)xcalloc(tensor_count ? tensor_count : 1, sizeof(GgufTensor));
    for (u64 i = 0; i < tensor_count; i++) {
        parse_tensor_info(base, size, &cur, &tensors[i]);
    }

    GgufFile* f = (GgufFile*)xcalloc(1, sizeof(GgufFile));
    f->mf = mf;
    f->kvs = kvs;
    f->kv_count = kv_count;
    f->tensors = tensors;
    f->tensor_count = tensor_count;

    /* general.alignment: UINT32 key, default 32 when absent. Looked up
     * through the public _or getter -- safe here because it only reads
     * f->kvs/f->kv_count, both already fully populated; f->tensors[*].data
     * (not yet set) is untouched by this call. A present-but-wrong-type
     * key still fatals through gguf_meta_i64, which is intentional (see
     * gguf.h). */
    i64 alignment_i64 = gguf_meta_i64_or(f, "general.alignment", 32);
    if (alignment_i64 <= 0) {
        gguf_close(f);
        fatal("gguf: '%s' -- general.alignment must be positive, got %lld",
              path, (long long)alignment_i64);
    }
    u64 alignment = (u64)alignment_i64;

    u64 pad = (alignment - (cur % alignment)) % alignment;
    u64 data_start = cur + pad; /* cur <= size already; pad < alignment, so
                                    this cannot overflow */
    if (data_start > size) {
        gguf_close(f);
        fatal("gguf: '%s' -- data section start (%llu) is past end of "
              "file (size %llu) after %llu-byte alignment padding",
              path, (unsigned long long)data_start, (unsigned long long)size,
              (unsigned long long)alignment);
    }

    for (u64 i = 0; i < tensor_count; i++) {
        GgufTensor* t = &tensors[i];
        u64         rel = t->offset;
        if (rel % alignment != 0) {
            /* Copy the tensor name to the stack BEFORE gguf_close(f): `t`
             * points into f->tensors, which gguf_close() frees, so using
             * t->name in the fatal() call AFTER closing would read freed
             * memory (caught by ASan: heap-use-after-free, gguf.c, during
             * the WSL sanitizer matrix). */
            char name_buf[256];
            snprintf(name_buf, sizeof name_buf, "%s", t->name);
            gguf_close(f);
            fatal("gguf: '%s' -- tensor '%s' offset %llu is not a "
                  "multiple of alignment %llu",
                  path, name_buf, (unsigned long long)rel,
                  (unsigned long long)alignment);
        }
        if (rel > size - data_start) {
            char name_buf[256];
            snprintf(name_buf, sizeof name_buf, "%s", t->name);
            gguf_close(f);
            fatal("gguf: '%s' -- tensor '%s' offset %llu + data section "
                  "start %llu exceeds file size %llu",
                  path, name_buf, (unsigned long long)rel,
                  (unsigned long long)data_start, (unsigned long long)size);
        }
        u64 abs_off = data_start + rel;
        char what_buf[300];
        snprintf(what_buf, sizeof what_buf, "tensor '%s' data", t->name);
        require_fits(size, abs_off, t->nbytes, what_buf);
        t->data = base + abs_off;
    }

    return f;
}

void gguf_close(GgufFile* f) {
    if (!f) return;
    for (u64 i = 0; i < f->kv_count; i++) {
        GgufKv* kv = &f->kvs[i];
        free(kv->key);
        free(kv->str_val);
        free(kv->arr_str_cache);
        free(kv->arr_str_blob);
    }
    free(f->kvs);
    for (u64 i = 0; i < f->tensor_count; i++) {
        free((void*)f->tensors[i].name);
    }
    free(f->tensors);
    map_close(&f->mf);
    free(f);
}

/* ===========================================================================
 * Metadata getters
 * ===========================================================================
 */

static const GgufKv* find_kv(const GgufFile* f, const char* key) {
    for (u64 i = 0; i < f->kv_count; i++) {
        if (strcmp(f->kvs[i].key, key) == 0) return &f->kvs[i];
    }
    return NULL;
}

static const GgufKv* find_kv_or_fatal(const GgufFile* f, const char* key,
                                       const char* who) {
    const GgufKv* kv = find_kv(f, key);
    if (!kv) {
        fatal("gguf: %s -- key '%s' not found", who, key);
    }
    return kv;
}

bool gguf_meta_has(const GgufFile* f, const char* key) {
    return find_kv(f, key) != NULL;
}

i64 gguf_meta_i64(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv_or_fatal(f, key, "gguf_meta_i64");
    switch (kv->type) {
        case GGUF_TYPE_UINT8:
            return (i64)kv->scalar.u8v;
        case GGUF_TYPE_INT8:
            return (i64)kv->scalar.i8v;
        case GGUF_TYPE_UINT16:
            return (i64)kv->scalar.u16v;
        case GGUF_TYPE_INT16:
            return (i64)kv->scalar.i16v;
        case GGUF_TYPE_UINT32:
            return (i64)kv->scalar.u32v;
        case GGUF_TYPE_INT32:
            return (i64)kv->scalar.i32v;
        case GGUF_TYPE_BOOL:
            return (i64)kv->scalar.boolv;
        case GGUF_TYPE_UINT64:
            if (kv->scalar.u64v > (u64)INT64_MAX) {
                fatal("gguf: gguf_meta_i64 -- key '%s' UINT64 value %llu "
                      "exceeds INT64_MAX, refusing to wrap",
                      key, (unsigned long long)kv->scalar.u64v);
            }
            return (i64)kv->scalar.u64v;
        case GGUF_TYPE_INT64:
            return kv->scalar.i64v;
        default:
            fatal("gguf: gguf_meta_i64 -- key '%s' has non-integer type %u",
                  key, kv->type);
    }
    return 0; /* unreachable: every case above returns or fatal()s */
}

i64 gguf_meta_i64_or(const GgufFile* f, const char* key, i64 dflt) {
    if (!gguf_meta_has(f, key)) return dflt;
    return gguf_meta_i64(f, key);
}

f32 gguf_meta_f32(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv_or_fatal(f, key, "gguf_meta_f32");
    if (kv->type != GGUF_TYPE_FLOAT32) {
        fatal("gguf: gguf_meta_f32 -- key '%s' has type %u, not FLOAT32",
              key, kv->type);
    }
    return kv->scalar.f32v;
}

const char* gguf_meta_str(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv_or_fatal(f, key, "gguf_meta_str");
    if (kv->type != GGUF_TYPE_STRING) {
        fatal("gguf: gguf_meta_str -- key '%s' has type %u, not STRING",
              key, kv->type);
    }
    return kv->str_val;
}

u64 gguf_meta_count(const GgufFile* f) { return f->kv_count; }

const char* gguf_meta_key(const GgufFile* f, u64 i) {
    if (i >= f->kv_count) {
        fatal("gguf: gguf_meta_key -- index %llu out of range (count %llu)",
              (unsigned long long)i, (unsigned long long)f->kv_count);
    }
    return f->kvs[i].key;
}

u32 gguf_meta_type(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv_or_fatal(f, key, "gguf_meta_type");
    return kv->type;
}

u32 gguf_meta_arr_elem_type(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv_or_fatal(f, key, "gguf_meta_arr_elem_type");
    if (kv->type != GGUF_TYPE_ARRAY) {
        fatal("gguf: gguf_meta_arr_elem_type -- key '%s' is not an ARRAY",
              key);
    }
    return kv->arr_elem_type;
}

u64 gguf_meta_arr_len(const GgufFile* f, const char* key) {
    const GgufKv* kv = find_kv_or_fatal(f, key, "gguf_meta_arr_len");
    if (kv->type != GGUF_TYPE_ARRAY) {
        fatal("gguf: gguf_meta_arr_len -- key '%s' is not an ARRAY", key);
    }
    return kv->arr_count;
}

i64 gguf_meta_arr_i64(const GgufFile* f, const char* key, u64 i) {
    const GgufKv* kv = find_kv_or_fatal(f, key, "gguf_meta_arr_i64");
    if (kv->type != GGUF_TYPE_ARRAY) {
        fatal("gguf: gguf_meta_arr_i64 -- key '%s' is not an ARRAY", key);
    }
    if (!is_integer_type(kv->arr_elem_type)) {
        fatal("gguf: gguf_meta_arr_i64 -- key '%s' element type %u is not "
              "integer (BOOL/STRING/FLOAT32/FLOAT64 array elements are not "
              "widened by this getter -- see gguf.h)",
              key, kv->arr_elem_type);
    }
    if (i >= kv->arr_count) {
        fatal("gguf: gguf_meta_arr_i64 -- key '%s' index %llu out of "
              "range (length %llu)",
              key, (unsigned long long)i, (unsigned long long)kv->arr_count);
    }
    u64 width = fixed_width(kv->arr_elem_type);
    u64 off = kv->arr_elements_off + i * width; /* cannot overflow: i <
        arr_count, and arr_count*width was already bounds-checked at parse
        time (require_fits in parse_kv), so any i<arr_count times width is
        smaller than that already-validated span */
    const u8* base = f->mf.base;
    u64       size = f->mf.size;
    switch (kv->arr_elem_type) {
        case GGUF_TYPE_UINT8:
            return (i64)rd_u8(base, size, off);
        case GGUF_TYPE_INT8:
            return (i64)(i8)rd_u8(base, size, off);
        case GGUF_TYPE_UINT16:
            return (i64)rd_u16(base, size, off);
        case GGUF_TYPE_INT16:
            return (i64)(i16)rd_u16(base, size, off);
        case GGUF_TYPE_UINT32:
            return (i64)rd_u32(base, size, off);
        case GGUF_TYPE_INT32:
            return (i64)(i32)rd_u32(base, size, off);
        case GGUF_TYPE_UINT64: {
            u64 v = rd_u64(base, size, off);
            if (v > (u64)INT64_MAX) {
                fatal("gguf: gguf_meta_arr_i64 -- key '%s'[%llu] UINT64 "
                      "value %llu exceeds INT64_MAX, refusing to wrap",
                      key, (unsigned long long)i, (unsigned long long)v);
            }
            return (i64)v;
        }
        case GGUF_TYPE_INT64:
            return (i64)rd_u64(base, size, off);
        default:
            fatal("gguf: gguf_meta_arr_i64 -- key '%s' -- internal: "
                  "unhandled integer type %u",
                  key, kv->arr_elem_type);
    }
    return 0; /* unreachable */
}

/* Lazily builds kv->arr_str_cache/arr_str_blob for an ARRAY<STRING> KV: one
 * pass to find each element's (offset,len) [reusing the same bounds-checked
 * reader as the initial parse -- the span was already validated then, so
 * these reads cannot fail], one pass to copy into a single owned,
 * NUL-separated blob. `kv` is mutated even though callers reach it through
 * a const GgufFile*: this cache is auxiliary memoization of data already
 * fully determined at gguf_open() time, not a change to any getter's
 * observable result, so the const-cast here is the standard C idiom for
 * lazy caching behind a const-correct API. */
static void ensure_str_array_cached(GgufKv* kv, const u8* base, u64 size) {
    if (kv->arr_str_cache != NULL) return;

    u64 n = kv->arr_count;
    u64 alloc_n = n ? n : 1;
    u64* offs = (u64*)xcalloc((size_t)alloc_n, sizeof(u64));
    u64* lens = (u64*)xcalloc((size_t)alloc_n, sizeof(u64));

    u64 cur = kv->arr_elements_off;
    u64 total_blob = 0;
    for (u64 i = 0; i < n; i++) {
        u64 len = rd_u64(base, size, cur);
        cur += 8;
        offs[i] = cur;
        lens[i] = len;
        total_blob += len + 1; /* +1 for this element's NUL */
        cur += len;
    }

    char*  blob = (char*)xmalloc((size_t)(total_blob ? total_blob : 1));
    char** ptrs = (char**)xcalloc((size_t)alloc_n, sizeof(char*));
    u64    w = 0;
    for (u64 i = 0; i < n; i++) {
        memcpy(blob + w, base + offs[i], (size_t)lens[i]);
        blob[w + lens[i]] = '\0';
        ptrs[i] = blob + w;
        w += lens[i] + 1;
    }

    free(offs);
    free(lens);
    kv->arr_str_blob = blob;
    kv->arr_str_cache = ptrs; /* set last: doubles as the "built" flag */
}

const char* gguf_meta_arr_str(const GgufFile* f, const char* key, u64 i) {
    const GgufKv* kv = find_kv_or_fatal(f, key, "gguf_meta_arr_str");
    if (kv->type != GGUF_TYPE_ARRAY) {
        fatal("gguf: gguf_meta_arr_str -- key '%s' is not an ARRAY", key);
    }
    if (kv->arr_elem_type != GGUF_TYPE_STRING) {
        fatal("gguf: gguf_meta_arr_str -- key '%s' element type %u is not "
              "STRING",
              key, kv->arr_elem_type);
    }
    if (i >= kv->arr_count) {
        fatal("gguf: gguf_meta_arr_str -- key '%s' index %llu out of "
              "range (length %llu)",
              key, (unsigned long long)i, (unsigned long long)kv->arr_count);
    }
    ensure_str_array_cached((GgufKv*)kv, f->mf.base, f->mf.size);
    return kv->arr_str_cache[i];
}

/* ===========================================================================
 * Tensor accessors
 * ===========================================================================
 */

u64 gguf_tensor_count(const GgufFile* f) { return f->tensor_count; }

const GgufTensor* gguf_tensor(const GgufFile* f, u64 i) {
    if (i >= f->tensor_count) {
        fatal("gguf: gguf_tensor -- index %llu out of range (count %llu)",
              (unsigned long long)i, (unsigned long long)f->tensor_count);
    }
    return &f->tensors[i];
}

const GgufTensor* gguf_tensor_by_name(const GgufFile* f, const char* name) {
    for (u64 i = 0; i < f->tensor_count; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    }
    fatal("gguf: gguf_tensor_by_name -- no tensor named '%s'", name);
    return NULL; /* unreachable */
}
