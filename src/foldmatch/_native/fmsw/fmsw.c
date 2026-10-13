/*
 * fmsw.c -- FoldMatch native Smith-Waterman kernel (ABI 1, DP id fmsw/1). API, semantics and build: fmsw.h.
 *
 * Provenance. Derived from the research kernel embcomp/throughput/build/swkern.c (md5 438d5a9811153f61e66cee845f1ed037)
 * and its R = 2 driver embcomp/evalue/score/swmulti.c (md5 b12fbc2c03638659440b12282a4d8670), through the P0.4 spike
 * rcsb-embedding-search src/scripts/esm_codes/spike/fmsw_spike.c. Kept from the research kernel: the rowcol
 * coefficient two-pass structure, the anti-diagonal strip DP (interleave_strip, ad_core, ad_score, ad_tb), traceback()
 * and the workspace sizing. Removed: the research-only sw_tb_f64_exact, sw_tb_f32, sw_score_f32, rowcol_z_fast_f32,
 * rowcol_z_f32 and swz_batch_f32, and the spike's fixed-order production similarity and block drivers.
 * Changes (PLAN section 7, P3.1):
 *   1. rowcol reductions use a fixed 4-lane order on every ISA (AVX2 builds use the same 4-lane reductions);
 *   2. query_prep in C in a fixed order (S5); the similarity is the caller's per-pair BLAS GEMM (S6, D2) into
 *      per-pair contiguous matrices at 64-byte-aligned offsets, read forward and through a row-reversed view;
 *      fmsw_sim_ref is the fixed-order test reference only;
 *   3. S7 degenerate rule (std < 1e-6 -> the row / column contributes exactly 0);
 *   4. generic vector-extension path (-DFMSW_GENERIC, and the #else of the ISA selection), vmax = a > b ? a : b;
 *   5. int64 pair offsets, int32 lengths, size_t workspace sizes;
 *   6. every entry point returns a status (0 ok, 1 workspace too small, 2 bad arguments);
 *   7. ABI and provenance exports (the source sha256 is injected by the build).
 * -0 results are canonicalised to +0 (S8).
 */
#include "fmsw.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define F_HY 1   /* H[i,j] = Y[i,j]            */
#define F_VX 2   /* V[i,j] = X[i,j]            */
#define F_XE 4   /* X[i,j] extends X[i-1,j]    */
#define F_YE 8   /* Y[i,j] extends Y[i,j-1]    */
#define F_ST 16  /* V[i,j] <= 0: start here    */
#define WS 16    /* rows per anti-diagonal strip */

#ifndef FMSW_SOURCE_SHA256
#define FMSW_SOURCE_SHA256 "unknown"
#endif

#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 12)
#error "fmsw needs GCC >= 12 or Clang >= 14 (__builtin_shufflevector on vector extensions)"
#endif

/* ------------------------------------------------------------------------------------------------ */
/* ISA selection (the DP vector width; the rowcol reductions always use 4-lane f4 vectors)            */
/* ------------------------------------------------------------------------------------------------ */
#if !defined(FMSW_GENERIC) && defined(__ARM_NEON)
#include <arm_neon.h>
#define VL 4
#define ISA_NAME "neon"
typedef float vf __attribute__((vector_size(16)));
typedef int32_t vi __attribute__((vector_size(16)));
typedef uint8_t vb __attribute__((vector_size(4)));
static inline vf vmax(vf a, vf b) { return (vf)vmaxq_f32((float32x4_t)a, (float32x4_t)b); }
#define SHIFTIN(p, c) __builtin_shufflevector((p), (c), 3, 4, 5, 6)          /* (p3, c0, c1, c2) */
#define LANES {0, 1, 2, 3}
#elif !defined(FMSW_GENERIC) && defined(__AVX2__)
#include <immintrin.h>
#define VL 8
#define ISA_NAME "avx2"
typedef float vf __attribute__((vector_size(32)));
typedef int32_t vi __attribute__((vector_size(32)));
typedef uint8_t vb __attribute__((vector_size(8)));
static inline vf vmax(vf a, vf b) { return (vf)_mm256_max_ps((__m256)a, (__m256)b); }
#define SHIFTIN(p, c) __builtin_shufflevector((p), (c), 7, 8, 9, 10, 11, 12, 13, 14)
#define LANES {0, 1, 2, 3, 4, 5, 6, 7}
#elif !defined(FMSW_GENERIC) && (defined(__SSE2__) || defined(__x86_64__))
#include <immintrin.h>
#define VL 4
#define ISA_NAME "sse2"
typedef float vf __attribute__((vector_size(16)));
typedef int32_t vi __attribute__((vector_size(16)));
typedef uint8_t vb __attribute__((vector_size(4)));
static inline vf vmax(vf a, vf b) { return (vf)_mm_max_ps((__m128)a, (__m128)b); }
#define SHIFTIN(p, c) __builtin_shufflevector((p), (c), 3, 4, 5, 6)
#define LANES {0, 1, 2, 3}
#else                                           /* -DFMSW_GENERIC, or no known vector ISA: vector extensions only */
#define VL 4
#define ISA_NAME "generic"
typedef float vf __attribute__((vector_size(16)));
typedef int32_t vi __attribute__((vector_size(16)));
typedef uint8_t vb __attribute__((vector_size(4)));
/* (a > b) ? a : b per lane: the operand order of _mm_max_ps(a, b) */
static inline vf vmax(vf a, vf b) { vi m = (vi)(a > b); return (vf)(((vi)a & m) | ((vi)b & ~m)); }
#define SHIFTIN(p, c) __builtin_shufflevector((p), (c), 3, 4, 5, 6)
#define LANES {0, 1, 2, 3}
#endif
#define NVEC (WS / VL)

typedef float f4 __attribute__((vector_size(16)));
static inline vf vload(const float *p) { vf v; memcpy(&v, p, sizeof(vf)); return v; }
static inline vf vbc(float x) { vf z = {0}; return z + x; }
static inline vi vsel_i(vi mask, vi a, vi b) { return (a & mask) | (b & ~mask); }
static inline f4 ld4(const float *p) { f4 v; memcpy(&v, p, 16); return v; }
static inline void st4(float *p, f4 v) { memcpy(p, &v, 16); }
static inline f4 bc4(float x) { f4 z = {0, 0, 0, 0}; return z + x; }
static inline float hsum4(f4 v) { float s = 0.0f; s += v[0]; s += v[1]; s += v[2]; s += v[3]; return s; }

/* ------------------------------------------------------------------------------------------------ */
/* provenance and dispatch                                                                           */
/* ------------------------------------------------------------------------------------------------ */
FMSW_API int fmsw_abi_version(void) { return FMSW_ABI_VERSION; }
FMSW_API const char *fmsw_dp_id(void) { return FMSW_DP_ID; }
FMSW_API const char *fmsw_isa(void) { return ISA_NAME; }
FMSW_API const char *fmsw_source_sha256(void) { return FMSW_SOURCE_SHA256; }
FMSW_API const char *fmsw_compiler(void)
{
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "unknown";
#endif
}

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
static unsigned int fmsw_xcr0_low(void)
{
    unsigned int eax, edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));    /* xgetbv(0) */
    (void)edx;
    return eax;
}
FMSW_API int fmsw_cpu_has_avx2(void)
{
    unsigned int a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    if (!((c >> 27) & 1u) || !((c >> 28) & 1u)) return 0;        /* OSXSAVE, AVX */
    if ((fmsw_xcr0_low() & 6u) != 6u) return 0;                  /* OS saves XMM and YMM state */
    if (__get_cpuid_max(0, NULL) < 7u) return 0;
    __cpuid_count(7, 0, a, b, c, d);
    return (int)((b >> 5) & 1u);                                  /* AVX2 */
}
#else
FMSW_API int fmsw_cpu_has_avx2(void) { return 0; }
#endif

/* ------------------------------------------------------------------------------------------------ */
/* S5 query_prep                                                                                     */
/* ------------------------------------------------------------------------------------------------ */
FMSW_API int fmsw_query_prep(const float *q, int n, float *qp)
{
    if (n < 0 || n > FMSW_MAX_LEN || (n > 0 && (!q || !qp))) return FMSW_ERR_ARGS;
    for (int i = 0; i < n; i++) {
        const float *r = q + (size_t)i * 64;
        float *o = qp + (size_t)i * 64;
        double ss = 0.0;
        for (int k = 0; k < 64; k++) {
            const double x = (double)r[k];
            if (!isfinite(x)) return FMSW_ERR_ARGS;
            ss += x * x;                        /* exact product (24 x 24 bits), one rounding per add, k ascending */
        }
        const double nrm = sqrt(ss);
        if (nrm < 1e-12) { memset(o, 0, 64 * sizeof(float)); continue; }
        const double den = 8.0 * nrm;
        for (int k = 0; k < 64; k++) o[k] = (float)((double)r[k] / den);
    }
    return FMSW_OK;
}

/* ------------------------------------------------------------------------------------------------ */
/* S6 test reference similarity: per cell, a float32 accumulator from +0 adds (bit ? qp_ik : -qp_ik)  */
/* for k = 0..63 in order. The loops run k outside and j inside so that independent cells can share  */
/* vector lanes; each cell's own sequence of additions is the fixed one above.                        */
/* ------------------------------------------------------------------------------------------------ */
FMSW_API int fmsw_sim_ref(const float *qp, int n, const uint8_t *codes, int m, float *sim)
{
    if (n < 0 || m < 0 || n > FMSW_MAX_LEN || m > FMSW_MAX_LEN) return FMSW_ERR_ARGS;
    if (n == 0 || m == 0) return FMSW_OK;
    if (!qp || !codes || !sim) return FMSW_ERR_ARGS;
    for (int i = 0; i < n; i++) {
        const float *qr = qp + (size_t)i * 64;
        float *s = sim + (size_t)i * (size_t)m;
        for (int j = 0; j < m; j++) s[j] = 0.0f;
        for (int k = 0; k < 64; k++) {
            const float pos = qr[k], neg = -qr[k];
            const int byte = k >> 3, shift = 7 - (k & 7);
            for (int j = 0; j < m; j++) s[j] += ((codes[(size_t)j * 8 + byte] >> shift) & 1) ? pos : neg;
        }
    }
    return FMSW_OK;
}

/* ------------------------------------------------------------------------------------------------ */
/* S7 rowcol z coefficients: research rowcol_coef_f32 with the 4-lane order on every ISA, int64       */
/* strides (negative = reversed view) and the degenerate rule.                                        */
/*   on return ra = a_i, rc = c_i, cb = b_j, cd = d_j with z_ij - offset = s_ij (a_i + b_j) - (c_i + d_j) */
/* ------------------------------------------------------------------------------------------------ */
#define DEGEN_SD 1e-6
static void rowcol_coef4(const float *sim, int n, int m, int64_t ld, float offset,
                         float *ra, float *rc, float *cb, float *cd)
{
    /* pass 1: means (ra <- row mean, cb <- column sum); 4 rows per sweep */
    for (int j = 0; j < m; j++) cb[j] = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float *s0 = sim + (ptrdiff_t)i * ld, *s1 = s0 + ld, *s2 = s1 + ld, *s3 = s2 + ld;
        f4 a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
        int j = 0;
        for (; j + 4 <= m; j += 4) {
            f4 v0 = ld4(s0 + j), v1 = ld4(s1 + j), v2 = ld4(s2 + j), v3 = ld4(s3 + j);
            a0 += v0; a1 += v1; a2 += v2; a3 += v3;
            st4(cb + j, ld4(cb + j) + ((v0 + v1) + (v2 + v3)));
        }
        float r0 = hsum4(a0), r1 = hsum4(a1), r2 = hsum4(a2), r3 = hsum4(a3);
        for (; j < m; j++) { r0 += s0[j]; r1 += s1[j]; r2 += s2[j]; r3 += s3[j]; cb[j] += (s0[j] + s1[j]) + (s2[j] + s3[j]); }
        ra[i] = (float)((double)r0 / m); ra[i + 1] = (float)((double)r1 / m);
        ra[i + 2] = (float)((double)r2 / m); ra[i + 3] = (float)((double)r3 / m);
    }
    for (; i < n; i++) {
        const float *s = sim + (ptrdiff_t)i * ld;
        f4 acc = {0};
        int j = 0;
        for (; j + 4 <= m; j += 4) { f4 v = ld4(s + j); acc += v; st4(cb + j, ld4(cb + j) + v); }
        float r = hsum4(acc);
        for (; j < m; j++) { r += s[j]; cb[j] += s[j]; }
        ra[i] = (float)((double)r / m);
    }
    const float inv_n = (float)(1.0 / n);
    for (int j = 0; j < m; j++) cb[j] *= inv_n;
    /* pass 2: squared deviations (rc <- row sum, cd <- column sum) */
    for (int j = 0; j < m; j++) cd[j] = 0.0f;
    i = 0;
    for (; i + 4 <= n; i += 4) {
        const float *s0 = sim + (ptrdiff_t)i * ld, *s1 = s0 + ld, *s2 = s1 + ld, *s3 = s2 + ld;
        const f4 m0 = bc4(ra[i]), m1 = bc4(ra[i + 1]), m2 = bc4(ra[i + 2]), m3 = bc4(ra[i + 3]);
        f4 a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
        int j = 0;
        for (; j + 4 <= m; j += 4) {
            f4 v0 = ld4(s0 + j), v1 = ld4(s1 + j), v2 = ld4(s2 + j), v3 = ld4(s3 + j);
            f4 d0 = v0 - m0, d1 = v1 - m1, d2 = v2 - m2, d3 = v3 - m3;
            a0 += d0 * d0; a1 += d1 * d1; a2 += d2 * d2; a3 += d3 * d3;
            f4 mc = ld4(cb + j);
            f4 e0 = v0 - mc, e1 = v1 - mc, e2 = v2 - mc, e3 = v3 - mc;
            st4(cd + j, ld4(cd + j) + ((e0 * e0 + e1 * e1) + (e2 * e2 + e3 * e3)));
        }
        float r0 = hsum4(a0), r1 = hsum4(a1), r2 = hsum4(a2), r3 = hsum4(a3);
        for (; j < m; j++) {
            float d0 = s0[j] - ra[i], d1 = s1[j] - ra[i + 1], d2 = s2[j] - ra[i + 2], d3 = s3[j] - ra[i + 3];
            r0 += d0 * d0; r1 += d1 * d1; r2 += d2 * d2; r3 += d3 * d3;
            float e0 = s0[j] - cb[j], e1 = s1[j] - cb[j], e2 = s2[j] - cb[j], e3 = s3[j] - cb[j];
            cd[j] += (e0 * e0 + e1 * e1) + (e2 * e2 + e3 * e3);
        }
        rc[i] = r0; rc[i + 1] = r1; rc[i + 2] = r2; rc[i + 3] = r3;
    }
    for (; i < n; i++) {
        const float *s = sim + (ptrdiff_t)i * ld;
        const float mr = ra[i];
        const f4 vmr = bc4(mr);
        f4 acc = {0};
        int j = 0;
        for (; j + 4 <= m; j += 4) {
            f4 v = ld4(s + j);
            f4 dr = v - vmr, dc = v - ld4(cb + j);
            acc += dr * dr;
            st4(cd + j, ld4(cd + j) + dc * dc);
        }
        float r = hsum4(acc);
        for (; j < m; j++) { float dr = s[j] - mr, dc = s[j] - cb[j]; r += dr * dr; cd[j] += dc * dc; }
        rc[i] = r;
    }
    /* coefficients; S7: a degenerate row (column) contributes exactly 0 */
    for (int k = 0; k < n; k++) {
        const double sd = sqrt((double)rc[k] / m);
        if (sd < DEGEN_SD) { ra[k] = 0.0f; rc[k] = 0.0f; continue; }
        const double a = 0.5 / (sd + 1e-8);
        rc[k] = (float)(a * ra[k]);
        ra[k] = (float)a;
    }
    for (int j = 0; j < m; j++) {
        const double sd = sqrt((double)cd[j] / n);
        if (sd < DEGEN_SD) { cb[j] = 0.0f; cd[j] = offset; continue; }
        const double b = 0.5 / (sd + 1e-8);
        cd[j] = (float)(b * cb[j] + offset);
        cb[j] = (float)b;
    }
}

/* ------------------------------------------------------------------------------------------------ */
/* traceback over the strip-major flag layout (research, unchanged): writes the path in FORWARD order */
/* ------------------------------------------------------------------------------------------------ */
static inline uint8_t flag_strip(const uint8_t *fl, int m, int i, int j)
{
    if (i <= 0 || j <= 0) return F_ST;          /* row 0 / column 0: H -> V -> start */
    const int T = m + WS;
    const int s = (i - 1) / WS, r = (i - 1) % WS;
    return fl[((size_t)s * T + (size_t)(j + r)) * WS + r];
}

static int traceback(const uint8_t *fl, int m, int bi, int bj, int32_t *path)
{
    int i = bi, j = bj, k = 0;
    int state = 0; /* 0 H, 1 V, 2 M, 3 X, 4 Y */
    for (;;) {
        uint8_t f = flag_strip(fl, m, i, j);
        if (state == 0) {
            state = (f & F_HY) ? 4 : 1;
        } else if (state == 1) {
            if (f & F_ST) break;
            state = (f & F_VX) ? 3 : 2;
        } else if (state == 2) {
            path[2 * k] = i - 1; path[2 * k + 1] = j - 1; k++;
            i--; j--; state = 0;
        } else if (state == 3) {
            path[2 * k] = i - 1; path[2 * k + 1] = -1; k++;
            state = (f & F_XE) ? 3 : 0;
            i--;
        } else {
            path[2 * k] = -1; path[2 * k + 1] = j - 1; k++;
            state = (f & F_YE) ? 4 : 1;
            j--;
        }
    }
    for (int a = 0, b = k - 1; a < b; a++, b--) {   /* built end -> start; reversed here, once */
        int32_t t0 = path[2 * a], t1 = path[2 * a + 1];
        path[2 * a] = path[2 * b]; path[2 * a + 1] = path[2 * b + 1];
        path[2 * b] = t0; path[2 * b + 1] = t1;
    }
    return k;
}

/* ------------------------------------------------------------------------------------------------ */
/* anti-diagonal strip DP (research; row stride int64, may be negative)                              */
/*   strip s covers rows i0 = 1 + 16 s .. i0 + 15; lane r (vector r / VL, element r % VL) of step t  */
/*   holds cell (i0 + r, t - r). up = lane r-1 of step t-1 (lane 0: row i0-1, Hrow/Xrow), diag =     */
/*   lane r-1 of step t-2 (= the previous step's 'up' vector), left = lane r of step t-1.             */
/*   No masks: virtual cells left of column 1 reproduce the boundary (H = V = 0); virtual cells right */
/*   of column m / below row n score BIG_NEG and can never beat the strict running best.              */
/*   Workspace (floats): Hrow, Xrow [m + WS + 1] each, Sv [(m + WS + 4) WS], P [WS PADW(m)].         */
/*   Flags (traceback): strip-major [nstrips][m + WS][WS] bytes.                                     */
/* ------------------------------------------------------------------------------------------------ */
#define BIG_NEG (-1e30f)
#define PADW(m) ((m) + 48)

static size_t ad_ws_floats(int64_t m)
{
    const size_t mm = (size_t)m;
    return 2 * (mm + WS + 1) + (mm + WS + 4) * WS + (size_t)WS * (mm + 48) + 64;
}

FMSW_API size_t fmsw_ws_floats(int n, int64_t mmax)
{
    if (n < 0) n = 0;
    if (mmax < 0) mmax = 0;
    return 2 * ((size_t)n + (size_t)mmax) + ad_ws_floats(mmax);
}

FMSW_API size_t fmsw_flag_bytes(int n, int64_t mmax)
{
    if (n < 0) n = 0;
    if (mmax < 0) mmax = 0;
    return (size_t)((n + WS - 1) / WS) * ((size_t)mmax + WS) * WS;
}

/* Sv[t WS + r] = score of cell (i0 + r, t - r), t = 0 .. T + 3 (T = m + WS): 0 for columns <= 0, BIG_NEG for
   columns > m or rows > n. Rows are staged with the z transform into padded buffers, then 4 rows x 4 steps are moved
   per 4x4 register transpose: row r's values for steps t0..t0+3 are the contiguous P[r][15 + t0 - r .. 18 + t0 - r]. */
static void interleave_strip(const float *S, int n, int m, int64_t ld, int i0,
                             const float *ra, const float *rc, const float *cb, const float *cd,
                             float *P, float *Sv)
{
    const int PW = PADW(m);
    const int T = m + WS;
    const f4 z4 = {0, 0, 0, 0};
    for (int r = 0; r < WS; r++) {
        float *pr = P + (size_t)r * PW;
        const int i = i0 + r;
        st4(pr, z4); st4(pr + 4, z4); st4(pr + 8, z4); st4(pr + 12, z4);
        float *d = pr + 16;
        int c = 0;
        if (i <= n) {
            const float *s = S + (ptrdiff_t)(i - 1) * ld;
            const float a = ra[i - 1], cc = rc[i - 1];
            const f4 a4 = {a, a, a, a}, c4 = {cc, cc, cc, cc};
            for (; c + 4 <= m; c += 4) st4(d + c, ld4(s + c) * (a4 + ld4(cb + c)) - (c4 + ld4(cd + c)));
            for (; c < m; c++) d[c] = s[c] * (a + cb[c]) - (cc + cd[c]);
        }
        for (; c < PW - 16; c++) d[c] = BIG_NEG;
    }
    for (int g = 0; g < WS / 4; g++) {
        const float *p0 = P + (size_t)(4 * g + 0) * PW + 15 - (4 * g + 0);
        const float *p1 = P + (size_t)(4 * g + 1) * PW + 15 - (4 * g + 1);
        const float *p2 = P + (size_t)(4 * g + 2) * PW + 15 - (4 * g + 2);
        const float *p3 = P + (size_t)(4 * g + 3) * PW + 15 - (4 * g + 3);
        float *out = Sv + 4 * g;
        for (int t0 = 0; t0 < T; t0 += 4) {
            const f4 R0 = ld4(p0 + t0), R1 = ld4(p1 + t0), R2 = ld4(p2 + t0), R3 = ld4(p3 + t0);
            const f4 a0 = __builtin_shufflevector(R0, R1, 0, 4, 1, 5), a1 = __builtin_shufflevector(R0, R1, 2, 6, 3, 7);
            const f4 a2 = __builtin_shufflevector(R2, R3, 0, 4, 1, 5), a3 = __builtin_shufflevector(R2, R3, 2, 6, 3, 7);
            st4(out + (size_t)(t0 + 0) * WS, __builtin_shufflevector(a0, a2, 0, 1, 4, 5));
            st4(out + (size_t)(t0 + 1) * WS, __builtin_shufflevector(a0, a2, 2, 3, 6, 7));
            st4(out + (size_t)(t0 + 2) * WS, __builtin_shufflevector(a1, a3, 0, 1, 4, 5));
            st4(out + (size_t)(t0 + 3) * WS, __builtin_shufflevector(a1, a3, 2, 3, 6, 7));
        }
    }
}

static inline __attribute__((always_inline))
float ad_core(const float *S, int n, int m, int64_t ld,
              const float *ra, const float *rc, const float *cb, const float *cd,
              float go, float ge, float *ws, uint8_t *fl, const int tb, int *bi_out, int *bj_out)
{
    const float NINF = -INFINITY;
    const int T = m + WS;                                   /* steps t = 0 .. T - 1 per strip (0 unused) */
    float *Hrow = ws, *Xrow = ws + (m + WS + 1);
    float *Sv = ws + 2 * ((size_t)m + WS + 1);
    float *P = Sv + ((size_t)m + WS + 4) * WS;
    for (int j = 0; j <= m + WS; j++) { Hrow[j] = 0.0f; Xrow[j] = NINF; }
    float best = 0.0f;
    int bi = 0, bj = 0;
    const vf vgo = vbc(go), vge = vbc(ge), zero = {0}, ninf = vbc(NINF);
    const vi lanes = LANES;
    const int nstrip = (n + WS - 1) / WS;
    for (int st = 0; st < nstrip; st++) {
        const int i0 = 1 + st * WS;
        interleave_strip(S, n, m, ld, i0, ra, rc, cb, cd, P, Sv);
        vf H1[NVEC], X1[NVEC], V1[NVEC], Y1[NVEC], HU[NVEC], lbest[NVEC];
        vi lbj[NVEC], tj[NVEC];
        for (int k = 0; k < NVEC; k++) {
            H1[k] = zero; X1[k] = ninf; V1[k] = zero; Y1[k] = ninf; HU[k] = zero;
            lbest[k] = vbc(best);
            lbj[k] = (vi){0} - 1;
            tj[k] = ((vi){0} + (1 - VL * k)) - lanes;          /* column of each lane at t = 1 */
        }
        uint8_t *fs = tb ? fl + (size_t)st * T * WS : NULL;
        for (int t = 1; t < T; t++) {
            const float *sv = Sv + (size_t)t * WS;
            const vf hu0 = vbc(Hrow[t]), xu0 = vbc(Xrow[t]);
            for (int k = NVEC - 1; k >= 0; k--) {
                const vf hup = SHIFTIN(k ? H1[k - 1] : hu0, H1[k]);
                const vf xup = SHIFTIN(k ? X1[k - 1] : xu0, X1[k]);
                const vf hdg = HU[k];
                HU[k] = hup;
                const vf xo = hup - vgo, xe = xup - vge;
                const vf x = vmax(xo, xe);
                const vf mm = hdg + vload(sv + VL * k);
                const vf v0 = vmax(x, mm);
                const vf v = vmax(v0, zero);
                const vf yo = V1[k] - vgo, ye = Y1[k] - vge;
                const vf y = vmax(yo, ye);
                const vf h = vmax(v, y);
                if (tb) {
                    vi f = ((xe > xo) & F_XE) | ((x > mm) & F_VX) | ((v0 <= zero) & F_ST) | ((ye > yo) & F_YE)
                           | ((y > v) & F_HY);
                    vb fb = __builtin_convertvector(f, vb);
                    memcpy(fs + (size_t)t * WS + VL * k, &fb, VL);
                }
                const vi gt = h > lbest[k];
                lbest[k] = vmax(lbest[k], h);
                lbj[k] = vsel_i(gt, tj[k], lbj[k]);
                tj[k] += 1;
                H1[k] = h; X1[k] = x; V1[k] = v; Y1[k] = y;
            }
            /* bottom row of the strip -> row above the next strip (index 0 is a dump slot) */
            int jl = t - (WS - 1);
            jl = jl > 0 ? jl : 0;
            Hrow[jl] = H1[NVEC - 1][VL - 1];
            Xrow[jl] = X1[NVEC - 1][VL - 1];
        }
        for (int r = 0; r < WS && i0 + r <= n; r++) {   /* rows in order; a later row needs a strictly larger H */
            const int k = r / VL, l = r % VL;
            if (lbj[k][l] >= 1 && lbest[k][l] > best) { best = lbest[k][l]; bi = i0 + r; bj = lbj[k][l]; }
        }
    }
    *bi_out = bi; *bj_out = bj;
    return best + 0.0f;                         /* -0 -> +0 (S8) */
}

static float ad_score(const float *S, int n, int m, int64_t ld, const float *ra, const float *rc, const float *cb,
                      const float *cd, float go, float ge, float *ws, int *bi, int *bj)
{
    return ad_core(S, n, m, ld, ra, rc, cb, cd, go, ge, ws, NULL, 0, bi, bj);
}

static float ad_tb(const float *S, int n, int m, int64_t ld, const float *ra, const float *rc, const float *cb,
                   const float *cd, float go, float ge, float *ws, uint8_t *fl, int *bi, int *bj)
{
    return ad_core(S, n, m, ld, ra, rc, cb, cd, go, ge, ws, fl, 1, bi, bj);
}

/* ------------------------------------------------------------------------------------------------ */
/* argument checks shared by the pair drivers                                                        */
/* ------------------------------------------------------------------------------------------------ */
static int check_pairs(const float *sim, int n, const int64_t *pair_off, const int32_t *m, int K, float offset,
                       float go, float ge, int64_t *mmax_out)
{
    if (n < 1 || n > FMSW_MAX_LEN || K < 0) return FMSW_ERR_ARGS;
    if (!isfinite(offset) || !isfinite(go) || !isfinite(ge) || !(go >= ge && ge >= 0.0f)) return FMSW_ERR_ARGS;
    if (K > 0 && (!sim || !pair_off || !m)) return FMSW_ERR_ARGS;
    int64_t mmax = 0;
    for (int k = 0; k < K; k++) {
        if (m[k] < 0 || m[k] > FMSW_MAX_LEN || pair_off[k] < 0) return FMSW_ERR_ARGS;
        if (m[k] > 0 && ((uintptr_t)(sim + pair_off[k]) & 63u) != 0) return FMSW_ERR_ARGS;
        if (m[k] > mmax) mmax = m[k];
    }
    *mmax_out = mmax;
    return FMSW_OK;
}

/* ------------------------------------------------------------------------------------------------ */
/* S9 score-only S_f and S_r; S_r reads the row-reversed view of the same per-pair matrix            */
/* ------------------------------------------------------------------------------------------------ */
FMSW_API int fmsw_score_fwd_rev(const float *sim, int n, const int64_t *pair_off, const int32_t *m, int K,
                                float offset, float go, float ge, float *s_fwd, float *s_rev, float *ws,
                                size_t ws_len)
{
    int64_t mmax = 0;
    int st = check_pairs(sim, n, pair_off, m, K, offset, go, ge, &mmax);
    if (st) return st;
    if (K > 0 && (!s_fwd || !s_rev || !ws)) return FMSW_ERR_ARGS;
    if (K > 0 && ws_len < fmsw_ws_floats(n, mmax)) return FMSW_ERR_WORKSPACE;
    float *ra = ws, *rc = ws + n, *cb = ws + 2 * (size_t)n, *cd = ws + 2 * (size_t)n + (size_t)mmax;
    float *dws = ws + 2 * ((size_t)n + (size_t)mmax);
    for (int k = 0; k < K; k++) {
        const int mk = m[k];
        int bi, bj;
        if (mk == 0) { s_fwd[k] = s_rev[k] = 0.0f; continue; }
        const float *fwd = sim + pair_off[k];
        const float *rev = fwd + (ptrdiff_t)(n - 1) * mk;          /* last row; stride -m */
        rowcol_coef4(fwd, n, mk, mk, offset, ra, rc, cb, cd);
        s_fwd[k] = ad_score(fwd, n, mk, mk, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        rowcol_coef4(rev, n, mk, -(int64_t)mk, offset, ra, rc, cb, cd);
        s_rev[k] = ad_score(rev, n, mk, -(int64_t)mk, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
    }
    return FMSW_OK;
}

/* ------------------------------------------------------------------------------------------------ */
/* forward DP, optional traceback (research swz_ad_batch_f32 with per-pair matrices)                 */
/* ------------------------------------------------------------------------------------------------ */
FMSW_API int fmsw_align(const float *sim, int n, const int64_t *pair_off, const int32_t *m, int K, float offset,
                        float go, float ge, int traceback_on, float *scores, int32_t *ends, int32_t *path_buf,
                        size_t path_cap, int64_t *path_off, float *ws, size_t ws_len, uint8_t *flags,
                        size_t flags_len)
{
    int64_t mmax = 0;
    int st = check_pairs(sim, n, pair_off, m, K, offset, go, ge, &mmax);
    if (st) return st;
    if (K > 0 && (!scores || !ends || !ws)) return FMSW_ERR_ARGS;
    if (traceback_on && (!path_off || (K > 0 && (!path_buf || !flags)))) return FMSW_ERR_ARGS;
    if (K > 0 && ws_len < fmsw_ws_floats(n, mmax)) return FMSW_ERR_WORKSPACE;
    if (traceback_on && K > 0 && flags_len < fmsw_flag_bytes(n, mmax)) return FMSW_ERR_WORKSPACE;
    float *ra = ws, *rc = ws + n, *cb = ws + 2 * (size_t)n, *cd = ws + 2 * (size_t)n + (size_t)mmax;
    float *dws = ws + 2 * ((size_t)n + (size_t)mmax);
    int64_t po = 0;
    if (path_off) path_off[0] = 0;
    for (int k = 0; k < K; k++) {
        const int mk = m[k];
        int bi = 0, bj = 0, plen = 0;
        float sc = 0.0f;
        if (mk > 0) {
            const float *fwd = sim + pair_off[k];
            if (traceback_on && (size_t)po + (size_t)n + (size_t)mk > path_cap) return FMSW_ERR_WORKSPACE;
            rowcol_coef4(fwd, n, mk, mk, offset, ra, rc, cb, cd);
            if (traceback_on) {
                sc = ad_tb(fwd, n, mk, mk, ra, rc, cb, cd, go, ge, dws, flags, &bi, &bj);
                plen = traceback(flags, mk, bi, bj, path_buf + 2 * po);
            } else {
                sc = ad_score(fwd, n, mk, mk, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
            }
        }
        scores[k] = sc;
        ends[2 * k] = bi; ends[2 * k + 1] = bj;
        po += plen;
        if (path_off) path_off[k + 1] = po;
    }
    return FMSW_OK;
}
