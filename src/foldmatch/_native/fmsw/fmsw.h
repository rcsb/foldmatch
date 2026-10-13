/*
 * fmsw.h -- FoldMatch native Smith-Waterman kernel for the ESM3 residue-code Stage-2 scoring (system esm-codes-v1).
 *
 * ABI 1, DP id "fmsw/1". Normative method: docs/esm-codes-method.md (S5 query prep, S6 similarity, S7 rowcol z,
 * S8 Smith-Waterman setting D, S9 forward / reversed-view scores, S16 tolerance contract).
 *
 * ------------------------------------------------------------------------------------------------------------------
 * What the kernel computes (per pair: query of n rows x target of m residues)
 *   query prep (S5)   qp_i = fp32( fp64(q_i) / (8 sqrt(sum_{k=0..63} fp64(q_ik)^2)) ), k ascending in float64;
 *                     qp_i = 0 when the norm is < 1e-12; NaN / Inf in q -> status 2.
 *   similarity (S6)   NOT computed here in production: the caller passes per-pair fp32 matrices sim = qp @ D^T
 *                     (one BLAS GEMM per pair, D = 2 bit - 1). fmsw_sim_ref() is the fixed-order test reference
 *                     (float32 accumulator from +0, adding bit_jk ? qp_ik : -qp_ik for k = 0..63 ascending).
 *   rowcol z (S7)     z_ij = 0.5 ((s_ij - mu_i) / (sd_i + 1e-8) + (s_ij - nu_j) / (tau_j + 1e-8)), population std,
 *                     applied as the affine map z_ij - offset = s_ij (a_i + b_j) - (c_i + d_j). Row and column sums
 *                     are float32 in a FIXED 4-lane order on every ISA (lane l accumulates columns j = l mod 4 of a
 *                     row; lanes are then added 0 + 1 + 2 + 3; a scalar tail adds the last m mod 4 columns in order;
 *                     column sums add 4 rows at a time as ((r0 + r1) + (r2 + r3)) then single rows); two passes
 *                     (means, then squared deviations). Degenerate rule: a row (column) whose population std is
 *                     < 1e-6 contributes exactly 0 (a_i = c_i = 0; b_j = 0, d_j = offset).
 *   DP (S8)           local 3-state Gotoh on the cell score z_ij - offset (setting D: offset 1.0, go 5.0, ge 0.5):
 *                       X[i,j] = max(H[i-1,j] - go, X[i-1,j] - ge)         query residue vs gap (vertical)
 *                       M[i,j] = H[i-1,j-1] + (z_{i-1,j-1} - offset)       aligned pair
 *                       V[i,j] = max(M, X, 0)                              (V <= 0 before the clip: start here)
 *                       Y[i,j] = max(V[i,j-1] - go, Y[i,j-1] - ge)         gap vs target residue (horizontal)
 *                       H[i,j] = max(V, Y)
 *                     X <-> Y adjacency is allowed; a gap of length L costs go + (L - 1) ge.
 *                     Best cell: the FIRST cell in row-major order with the maximal H (strict '>' update).
 *                     Traceback tie rules: H -> Y only if Y > V; V -> X only if X > M; X extends only if
 *                     X[i-1,j] - ge > H[i-1,j] - go; Y extends only if Y[i,j-1] - ge > V[i,j-1] - go.
 *                     An empty alignment (no H > 0) scores +0 with no path and end cell (0, 0).
 *                     Scores are canonicalised: a -0 result is returned as +0.
 *   forward / reverse S_f = SW(z(sim)); S_r = SW(z(simr)) with simr[i, :] = sim[n-1-i, :] read through a
 *   (S9)              row-reversed VIEW of the same per-pair matrix (base = last row, row stride -m); z is recomputed
 *                     on the view in its own row order. No reversed copy is made.
 *
 * Implementation: anti-diagonal strips of 16 rows (4-lane vectors on SSE2 / NEON / generic, 8-lane on AVX2; the DP is
 * element-wise, so the lane width does not change any value). Given the same similarity, every build (sse2, avx2,
 * neon, generic; GCC or Clang) returns bit-identical scores, end cells and paths (one DP golden hash).
 *
 * ------------------------------------------------------------------------------------------------------------------
 * Similarity buffer layout (fmsw_score_fwd_rev, fmsw_align)
 *   sim is a float32 base pointer; pair k is the C-contiguous matrix [n, m[k]] (row stride m[k]) starting at the
 *   float offset pair_off[k] >= 0. Every pair matrix must start on a 64-byte boundary (status 2 otherwise): the S6
 *   GEMM output layout is fixed, and the kernel refuses buffers that break it. m[k] == 0 gives an empty alignment.
 *
 * Workspace
 *   ws    float32, at least fmsw_ws_floats(n, mmax) floats (mmax = max_k m[k]); no alignment requirement.
 *   flags uint8, at least fmsw_flag_bytes(n, mmax) bytes (traceback only).
 *   path  int32 [2 * path_cap]: path_cap counts (i, j) entries; pair k needs at most n + m[k].
 *   Undersized buffers return status 1 before anything is written for that pair.
 *
 * Status codes: 0 ok, 1 workspace / buffer too small, 2 bad arguments.
 *
 * ------------------------------------------------------------------------------------------------------------------
 * Floating-point semantics (documented per ISA; S5 guarantees finite inputs)
 *   vmax(a, b)  sse2 / avx2: _mm_max_ps / _mm256_max_ps(a, b) = (a > b) ? a : b, so the SECOND operand is returned
 *               when either is NaN or when a and b are zeros of either sign.
 *               generic: (a > b) ? a : b per lane, identical to sse2 / avx2.
 *               neon: FMAX: NaN if either operand is NaN; max(+0, -0) = +0.
 *   With finite similarities and go > 0, ge > 0 no NaN can arise, and the builds differ at most in the sign of a zero
 *   intermediate, which never changes a comparison, a flag, a later sum or a returned score (-0 is canonicalised).
 *   NaN in sim is not detected (the caller's S6 GEMM of finite inputs cannot produce one); NaN / Inf in q is (S5).
 *   All builds use -ffp-contract=off -fno-fast-math: no FMA contraction, no reassociation.
 *
 * Build (see foldmatch/_native/build.py): cc -O3 -std=c11 -fPIC -shared -ffp-contract=off -fno-fast-math
 *   -fvisibility=hidden [-march=x86-64 -mtune=generic on x86_64] [-mavx2 | -DFMSW_GENERIC]
 *   -DFMSW_SOURCE_SHA256="\"<sha256>\"" -o _fmsw_<isa>.<ext> fmsw.c -lm
 */
#ifndef FMSW_H
#define FMSW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FMSW_API __attribute__((visibility("default")))

#define FMSW_ABI_VERSION 1
#define FMSW_DP_ID "fmsw/1"

#define FMSW_OK 0
#define FMSW_ERR_WORKSPACE 1
#define FMSW_ERR_ARGS 2

#define FMSW_MAX_LEN (1 << 30)      /* n, m[k] limits (int arithmetic headroom); Stage 2 caps n m at 2^28 cells */

/* provenance and dispatch */
FMSW_API int fmsw_abi_version(void);                 /* FMSW_ABI_VERSION */
FMSW_API const char *fmsw_dp_id(void);               /* FMSW_DP_ID */
FMSW_API const char *fmsw_isa(void);                 /* "sse2" | "avx2" | "neon" | "generic" */
FMSW_API const char *fmsw_source_sha256(void);       /* injected at build time ("unknown" if not) */
FMSW_API const char *fmsw_compiler(void);
FMSW_API int fmsw_cpu_has_avx2(void);                /* x86: CPUID AVX2 + OS YMM state; 0 elsewhere */

/* S5: q [n, 64] float32 (exact fp32 of fp16) -> qp [n, 64]; status 2 on NaN / Inf (rows before it are written) */
FMSW_API int fmsw_query_prep(const float *q, int n, float *qp);

/* S6 test reference (NOT used in production): sim [n, m] contiguous, codes uint8 [m, 8] packbits MSB-first
   (bit k of residue j = codes[8 j + k / 8] >> (7 - k % 8) & 1) */
FMSW_API int fmsw_sim_ref(const float *qp, int n, const uint8_t *codes, int m, float *sim);

/* S9 score-only S_f (forward) and S_r (row-reversed view) for K pairs */
FMSW_API int fmsw_score_fwd_rev(const float *sim, int n, const int64_t *pair_off, const int32_t *m, int K,
                                float offset, float go, float ge, float *s_fwd, float *s_rev, float *ws,
                                size_t ws_len);

/* forward DP for K pairs: scores [K], ends [K, 2] (1-based best cell; (0, 0) = empty); traceback != 0 also writes
   the paths: path_buf [2 * path_cap] int32 (query i, target j) 0-based, -1 = gap, FORWARD order (first column first),
   pair k at entries path_off[k] .. path_off[k + 1] (path_off [K + 1] int64). With traceback == 0, path_buf, path_off
   and flags may be NULL (path_off, when given, is filled with zeros). */
FMSW_API int fmsw_align(const float *sim, int n, const int64_t *pair_off, const int32_t *m, int K, float offset,
                        float go, float ge, int traceback, float *scores, int32_t *ends, int32_t *path_buf,
                        size_t path_cap, int64_t *path_off, float *ws, size_t ws_len, uint8_t *flags,
                        size_t flags_len);

FMSW_API size_t fmsw_ws_floats(int n, int64_t mmax);
FMSW_API size_t fmsw_flag_bytes(int n, int64_t mmax);

#ifdef __cplusplus
}
#endif

#endif /* FMSW_H */
