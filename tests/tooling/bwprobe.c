/* What IS this box's single-thread read bandwidth?
 *
 * bench.c's reference stream is `f32 acc = 0; for(..) acc += buf[i];`.
 * That is a SERIAL FLOATING-POINT REDUCTION: every add depends on the
 * previous one, and GCC may not reassociate it without -ffast-math, so it
 * cannot be vectorized at ANY -O level. Its rate is therefore f32-add
 * LATENCY, not memory bandwidth -- which is exactly why it reads the same
 * 7.6-7.9 GB/s at -O1 and -O3 while every real kernel sped up 3x.
 *
 * If that is right, then "%ref" is not "% of the memory ceiling" and no
 * conclusion about being bandwidth-bound can rest on it.
 *
 * Variants, same 256 MB buffer bench.c uses:
 *   serial_f32  reproduces the current reference stream
 *   split8_f32  eight independent f32 accumulators -- same arithmetic, the
 *               dependency chain broken; still not auto-vectorizable, but
 *               eight chains interleave
 *   sum_i64     integer accumulate -- integer add IS associative, so -O3 may
 *               vectorize it freely; this is the honest read-bandwidth probe
 *   xor_u64     XOR reduction, 1-cycle latency, nothing to saturate but the
 *               load ports
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef float f32;
typedef uint64_t u64;
typedef int64_t i64;

#define BYTES (256u * 1024u * 1024u)
#define RUNS 7

static volatile f32 sink_f;
static volatile u64 sink_u;

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void serial_f32(const f32 *b, u64 n) {
    f32 acc = 0.0f;
    for (u64 i = 0; i < n; i++) acc += b[i];
    sink_f = acc;
}

static void split8_f32(const f32 *b, u64 n) {
    f32 a0=0,a1=0,a2=0,a3=0,a4=0,a5=0,a6=0,a7=0;
    for (u64 i = 0; i + 8 <= n; i += 8) {
        a0+=b[i]; a1+=b[i+1]; a2+=b[i+2]; a3+=b[i+3];
        a4+=b[i+4]; a5+=b[i+5]; a6+=b[i+6]; a7+=b[i+7];
    }
    sink_f = ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7));
}

static void sum_i64(const f32 *b, u64 n) {
    const int32_t *p = (const int32_t *)b;   /* read the same bytes as ints */
    i64 acc = 0;
    for (u64 i = 0; i < n; i++) acc += p[i];
    sink_u = (u64)acc;
}

static void xor_u64(const f32 *b, u64 n) {
    const u64 *p = (const u64 *)b;
    u64 acc = 0;
    for (u64 i = 0; i < n / 2; i++) acc ^= p[i];
    sink_u = acc;
}

static double bw(void (*fn)(const f32 *, u64), const f32 *b, u64 n) {
    double s[RUNS];
    fn(b, n);                                 /* warm */
    for (int r = 0; r < RUNS; r++) {
        double t0 = now_ms();
        fn(b, n);
        s[r] = now_ms() - t0;
    }
    qsort(s, RUNS, sizeof(double), cmpd);
    return ((double)BYTES / 1e9) / (s[RUNS / 2] / 1e3);
}

int main(void) {
    u64  n   = BYTES / sizeof(f32);
    f32 *buf = (f32 *)malloc(BYTES);
    if (!buf) return 1;
    for (u64 i = 0; i < n; i++) buf[i] = 0.01f * (f32)((i64)(i % 97) - 48);

    printf("buffer %u MB, median of %d\n\n", BYTES / (1024u * 1024u), RUNS);
    printf("  %-12s %10s\n", "variant", "GB/s");
    printf("  %-12s %10.2f   <- bench.c's reference stream\n",
           "serial_f32", bw(serial_f32, buf, n));
    printf("  %-12s %10.2f\n", "split8_f32", bw(split8_f32, buf, n));
    printf("  %-12s %10.2f\n", "sum_i64", bw(sum_i64, buf, n));
    printf("  %-12s %10.2f\n", "xor_u64", bw(xor_u64, buf, n));
    free(buf);
    return 0;
}
