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

/* n, m[k] limits (int arithmetic headroom). The esm-codes Stage 2 caps n m at 2^28 cells; the BLOSUM62 Stage 2
   (fmswi_align) has no cell cap: its flag buffer is n_pad x the longest target (review finding 6, left open). */
#define FMSW_MAX_LEN (1 << 30)

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

/*
 * ------------------------------------------------------------------------------------------------------------------
 * Integer local Smith-Waterman for the BLOSUM62 Stage 2 (DP id "fmswi/1"; same library and ABI, separate entry point)
 *
 * Output-equivalent to biotite 1.7.1 align_optimal(query, target, matrix, gap_penalty=(-open, -extend), local=True,
 * max_number=1): the same score, end cell and path (so every column FoldMatch derives from the path). With
 * s(i, j) = qprof[i - 1][t_{j-1}] (a position-specific query profile: the BLOSUM62 row of each query residue, plus its
 * composition-bias offset when that correction is on), go = open, ge = extend, the kernel computes:
 *   M[i,j]  = max(0, s(i, j) + max(M, G1, G2)[i-1, j-1])    aligned pair; M <= 0 before the clip is "dead"
 *   G1[i,j] = max(M[i, j-1] - go, G1[i, j-1] - ge)          target residue against a gap in the query ('D')
 *   G2[i,j] = max(M[i-1, j] - go, G2[i-1, j] - ge)          query residue against a gap in the target ('I')
 *   boundary row / column: M = 0, G1 = G2 = -go. Gaps open from M only: unlike Gotoh / parasail / SSW / MMseqs2 no
 *   insertion run is directly followed by a deletion run. A gap of length L costs go + (L - 1) ge.
 *   score    = max M; empty alignment (all zero outputs) when it is 0.
 *   end cell = the FIRST cell in row-major (query-major) order with M == score (numpy.where order).
 *   traceback from the end cell in state M: M -> predecessor M if M >= G1 and M >= G2, else G1 if G1 >= G2, else G2
 *   (diagonal first, then 'D', then 'I'); a G state extends only if G_prev - ge > M_prev - go (ties open, so gaps sit
 *   as early as possible); the path stops before a dead M. Paths start and end with an aligned pair.
 * These are the kernel's own tables, not biotite's literal ones: biotite starts from M = -inf on the boundary (except
 * M[0,0] = 0), G1 = 0 on row 0 and G2 = 0 on column 0 (-inf elsewhere), and leaves a G cell at -inf when its value is
 * <= 0. The table values differ; the score, end cell and path do not. That equivalence is checked against biotite
 * (tests/test_swint.py), not derived from its code.
 *
 * Arguments
 *   q        uint8 [n] query codes (< A): used for the identity count (nident compares q and t codes).
 *   qprof    int32 [n, A] row-major position-specific scores, |s| <= FMSWI_MAX_SCORE.
 *   t        uint8 target codes (< A), pair k at t + t_off[k], m[k] residues (m[k] == 0: empty alignment).
 *   gap_open >= 1 (the end-cell rule relies on it), gap_extend >= 0, both <= FMSWI_MAX_GAP.
 *   traceback 0: score, QEND, TEND and WIDE only (the other fields 0); 1: every field, one pass that stores a flag byte
 *            per cell; 2: every field, a score-only pass and then a flag pass over rows 1..QEND x columns 1..TEND only
 *            (exact: those cells' DP values depend on no other cell, so the second pass meets the same end cell; on
 *            the Fig. 2 sample that rectangle is 35% of the DP cells, cell-weighted). 1 and 2 give identical outputs.
 *   out      int32 [K, FMSWI_NOUT].
 *   ops      optional (traceback only): the paths as 'M' / 'D' / 'I' in forward order, pair k at
 *            ops[ops_off[k] .. ops_off[k + 1]); before pair k the kernel needs ops_off[k] + n + m[k] <= ops_cap.
 *   ops_off  optional int64 [K + 1] (filled with the running offsets; all zero without ops).
 * Implementation: Farrar striped profile with lazy F in 16-bit lanes (16 on AVX2, 8 on SSE2 / NEON / generic); a pair
 * whose score reaches 32767 - max(s) (or a profile / gap cost outside +-16384) is computed in 32-bit lanes (out WIDE
 * = 1). Values are exact integers on every ISA, so every build returns identical results. Workspace is allocated per
 * call: the profiles (A x n_pad x 2 or 4 bytes), 4 n_pad vectors and, with traceback, one flag byte per cell of the
 * longest target (n_pad x mmax bytes). Status: 0 ok, 1 allocation failure or ops buffer too small, 2 bad arguments
 * (or a score beyond int32), 3 internal check failed (traceback 2: the second pass did not meet the first pass's end
 * cell; never expected).
 */
#define FMSWI_DP_ID "fmswi/1"
#define FMSWI_MAX_SCORE (1 << 20)
#define FMSWI_MAX_GAP (1 << 20)
#define FMSWI_ERR_INTERNAL 3
#define FMSWI_NOUT 11
#define FMSWI_SCORE 0      /* raw score (>= 0) */
#define FMSWI_QSTART 1     /* 1-based first aligned query residue (0 = empty alignment or traceback off) */
#define FMSWI_QEND 2       /* 1-based end cell (query) */
#define FMSWI_TSTART 3
#define FMSWI_TEND 4
#define FMSWI_ALNLEN 5     /* path columns, gaps included */
#define FMSWI_NIDENT 6     /* aligned pairs with equal codes */
#define FMSWI_GAPOPEN 7    /* maximal gap runs, both sides */
#define FMSWI_NQGAP 8      /* 'D' columns: target residue against a gap in the query */
#define FMSWI_NTGAP 9      /* 'I' columns: query residue against a gap in the target */
#define FMSWI_WIDE 10      /* 1 when the pair was computed in 32-bit lanes */

FMSW_API const char *fmswi_dp_id(void);              /* FMSWI_DP_ID */
FMSW_API int fmswi_align(const uint8_t *q, const int32_t *qprof, int n, int A, const uint8_t *t,
                         const int64_t *t_off, const int32_t *m, int K, int gap_open, int gap_extend, int traceback,
                         int32_t *out, char *ops, size_t ops_cap, int64_t *ops_off);

#ifdef __cplusplus
}
#endif

#endif /* FMSW_H */
