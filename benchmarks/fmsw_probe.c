/* Benchmark-only probe of the native kernel (benchmarks/stage2_esm_codes.py, mode 'components'; plan P7 profiling
 * gate): the per-pair loops of fmsw_score_fwd_rev and fmsw_align with a clock around each stage, so the worker time
 * the kernel call reports as one number splits into rowcol z, DP and traceback.
 *
 * The shipped kernel source is #included UNCHANGED and this file is compiled with the shipped build's flags for the
 * ISA (foldmatch._native.build.compile_command), so the stage functions are the shipped ones; the probe's scores,
 * best cells and paths are checked bit for bit against the shipped library by the benchmark. Never shipped.
 */
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#include "fmsw.c"

static double probe_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* fmsw_score_fwd_rev's loop; t[0] += rowcol z (fwd + rev), t[1] += score-only DP (fwd + rev) */
FMSW_API int probe_score_fwd_rev(const float *sim, int n, const int64_t *pair_off, const int32_t *m, int K,
                                 float offset, float go, float ge, float *s_fwd, float *s_rev, float *ws,
                                 size_t ws_len, double *t)
{
    int64_t mmax = 0;
    int st = check_pairs(sim, n, pair_off, m, K, offset, go, ge, &mmax);
    if (st) return st;
    if (K > 0 && (!s_fwd || !s_rev || !ws || !t)) return FMSW_ERR_ARGS;
    if (K > 0 && ws_len < fmsw_ws_floats(n, mmax)) return FMSW_ERR_WORKSPACE;
    float *ra = ws, *rc = ws + n, *cb = ws + 2 * (size_t)n, *cd = ws + 2 * (size_t)n + (size_t)mmax;
    float *dws = ws + 2 * ((size_t)n + (size_t)mmax);
    for (int k = 0; k < K; k++) {
        const int mk = m[k];
        int bi, bj;
        if (mk == 0) { s_fwd[k] = s_rev[k] = 0.0f; continue; }
        const float *fwd = sim + pair_off[k];
        const float *rev = fwd + (ptrdiff_t)(n - 1) * mk;
        double a = probe_now();
        rowcol_coef4(fwd, n, mk, mk, offset, ra, rc, cb, cd);
        double b = probe_now();
        s_fwd[k] = ad_score(fwd, n, mk, mk, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        double c = probe_now();
        rowcol_coef4(rev, n, mk, -(int64_t)mk, offset, ra, rc, cb, cd);
        double d = probe_now();
        s_rev[k] = ad_score(rev, n, mk, -(int64_t)mk, ra, rc, cb, cd, go, ge, dws, &bi, &bj);
        double e = probe_now();
        t[0] += (b - a) + (d - c);
        t[1] += (c - b) + (e - d);
    }
    return FMSW_OK;
}

/* fmsw_align's loop with traceback; t[0] += rowcol z, t[1] += DP with traceback flags, t[2] += traceback */
FMSW_API int probe_align_tb(const float *sim, int n, const int64_t *pair_off, const int32_t *m, int K, float offset,
                            float go, float ge, float *scores, int32_t *ends, int32_t *path_buf, size_t path_cap,
                            int64_t *path_off, float *ws, size_t ws_len, uint8_t *flags, size_t flags_len, double *t)
{
    int64_t mmax = 0;
    int st = check_pairs(sim, n, pair_off, m, K, offset, go, ge, &mmax);
    if (st) return st;
    if (K > 0 && (!scores || !ends || !ws || !path_off || !path_buf || !flags || !t)) return FMSW_ERR_ARGS;
    if (K > 0 && ws_len < fmsw_ws_floats(n, mmax)) return FMSW_ERR_WORKSPACE;
    if (K > 0 && flags_len < fmsw_flag_bytes(n, mmax)) return FMSW_ERR_WORKSPACE;
    float *ra = ws, *rc = ws + n, *cb = ws + 2 * (size_t)n, *cd = ws + 2 * (size_t)n + (size_t)mmax;
    float *dws = ws + 2 * ((size_t)n + (size_t)mmax);
    int64_t po = 0;
    path_off[0] = 0;
    for (int k = 0; k < K; k++) {
        const int mk = m[k];
        int bi = 0, bj = 0, plen = 0;
        float sc = 0.0f;
        if (mk > 0) {
            const float *fwd = sim + pair_off[k];
            if ((size_t)po + (size_t)n + (size_t)mk > path_cap) return FMSW_ERR_WORKSPACE;
            double a = probe_now();
            rowcol_coef4(fwd, n, mk, mk, offset, ra, rc, cb, cd);
            double b = probe_now();
            sc = ad_tb(fwd, n, mk, mk, ra, rc, cb, cd, go, ge, dws, flags, &bi, &bj);
            double c = probe_now();
            plen = traceback(flags, mk, bi, bj, path_buf + 2 * po);
            double d = probe_now();
            t[0] += b - a;
            t[1] += c - b;
            t[2] += d - c;
        }
        scores[k] = sc;
        ends[2 * k] = bi; ends[2 * k + 1] = bj;
        po += plen;
        path_off[k + 1] = po;
    }
    return FMSW_OK;
}
