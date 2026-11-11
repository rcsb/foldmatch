"""Test oracle for the esm-codes Stage-2 kernel (PLAN section 7, P3.6). numpy only; never used in production.

* ``normalize_scores`` and ``smith_waterman`` (with ``Alignment``, ``needleman_wunsch`` and ``_align``) are a VERBATIM
  copy of ``rcsb-embedding-search/src/scripts/esm3_needleman_wunsch.py`` (md5 7ec19b7e5ff89dc7b44d2512aed3997e), lines
  113-125 and 138-311: the float64 harness the research results were validated against. :func:`check_verbatim`
  proves the blocks below are unchanged (sha256 of each block, recorded when they were copied).
* The S5 zero-norm rule and the S7 degenerate rule are thin wrappers around it (:func:`reference_qprep64`,
  :func:`reference_rowcol_s7`), with the same expression order as the harness, so they are bit-identical to it
  whenever nothing is degenerate. This is exactly how the FoldMatch fixtures F1 / F-deg were computed
  (``make_foldmatch_fixtures.py`` ``ref_pair64``).
* :func:`sw_margins64` re-runs the same float64 recurrences keeping every matrix, to measure how close the oracle's
  best-cell choice and traceback decisions are to a tie (the S16 path clause).
* :func:`reference_encode` (S2/S3 with the research fp16 casts, float64 arithmetic otherwise) and
  :func:`reference_evalue` (S11) complete the reference chain.
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORACLE_SOURCE = "rcsb-embedding-search/src/scripts/esm3_needleman_wunsch.py"
ORACLE_SOURCE_MD5 = "7ec19b7e5ff89dc7b44d2512aed3997e"
VERBATIM_BLOCKS = {  # name: (first line, last line, sha256 of the block text including its final newline)
    "normalize_scores": (113, 125, "ad201b13c2e3b168901bb3816269f08ad8e3cf37e4b4ee60eab2b11de5078662"),
    "smith_waterman": (138, 311, "a49b1365509fe984544dd3a2cdac9886570fa41e9f3e7b0165736a333eb2c622"),
}
S7_EPS = 1e-6           # S7: a row / column with population std < 1e-6 contributes exactly 0
QPREP_EPS = 1e-12       # S5: |q_i| < 1e-12 -> zero row


# ---- BEGIN VERBATIM normalize_scores ----
def normalize_scores(sim: np.ndarray, mode: str = "rowcol") -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float64)
    eps = 1e-8
    if mode == "none":
        return sim
    if mode == "zscore":
        return (sim - sim.mean()) / (sim.std() + eps)
    if mode == "rowcol":
        z_row = (sim - sim.mean(axis=1, keepdims=True)) / (sim.std(axis=1, keepdims=True) + eps)
        z_col = (sim - sim.mean(axis=0, keepdims=True)) / (sim.std(axis=0, keepdims=True) + eps)
        return 0.5 * (z_row + z_col)
    raise ValueError(f"unknown normalization mode: {mode!r}")

# ---- END VERBATIM normalize_scores ----


# ---- BEGIN VERBATIM smith_waterman ----
# =============================================================================
# Needleman-Wunsch (global) and Smith-Waterman (local), affine gaps
# =============================================================================
@dataclass
class Alignment:
    score: float
    pairs: list[tuple[int | None, int | None]]  # (index in A, index in B), None = gap

    def gapped(self, seq_a: str, seq_b: str) -> tuple[str, str]:
        top = "".join("-" if i is None else seq_a[i] for i, _ in self.pairs)
        bottom = "".join("-" if j is None else seq_b[j] for _, j in self.pairs)
        return top, bottom

    def aligned_pairs(self) -> list[tuple[int, int]]:
        return [(i, j) for i, j in self.pairs if i is not None and j is not None]


def needleman_wunsch(
    scores: np.ndarray,
    gap_open: float,
    gap_extend: float | None = None,
    free_end_gaps: bool = False,
) -> Alignment:
    """Global alignment of the full sequences (Needleman & Wunsch 1970).

    scores[i, j]   score for aligning residue i of A with residue j of B (shape n x m)
    gap_open       penalty (positive) for the first position of a gap
    gap_extend     penalty for each further position; None -> gap_open (linear gaps)
    free_end_gaps  leading/trailing gaps cost nothing (semi-global)
    """
    return _align(scores, gap_open, gap_extend, local=False, free_end_gaps=free_end_gaps)


def smith_waterman(
    scores: np.ndarray,
    gap_open: float,
    gap_extend: float | None = None,
) -> Alignment:
    """Local alignment: the highest-scoring pair of segments (Smith & Waterman 1981).

    Only meaningful when unrelated residue pairs score below zero on average, so shift
    z-scored matrices down first (LOCAL_SCORE_OFFSET). Returns an empty alignment
    (no pairs, score 0) when no residue pair scores above zero.
    """
    return _align(scores, gap_open, gap_extend, local=True, free_end_gaps=False)


def _align(scores, gap_open, gap_extend, local: bool, free_end_gaps: bool) -> Alignment:
    """Dynamic programming shared by both algorithms (Gotoh 1982 affine-gap recurrences).

    States: M = residue pair, X = A residue vs gap, Y = gap vs B residue, H = max(M, X, Y);
    Smith-Waterman adds 0 as a fourth option ("start a new alignment here").
    A gap of length L costs gap_open + (L - 1) * gap_extend.
    Each DP row is filled with numpy: M and X need only the previous row, and the
    within-row Y recurrence is solved with a running maximum, so Python loops over rows only.
    """
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("scores must be a 2-D matrix")
    n, m = S.shape
    go = float(gap_open)
    ge = go if gap_extend is None else float(gap_extend)
    if not go >= ge >= 0:
        raise ValueError("need gap_open >= gap_extend >= 0")

    cols = np.arange(m + 1)

    def edge_score(k):  # first row / column: a leading gap of length k
        if local or free_end_gaps:
            return np.zeros_like(k, dtype=np.float64)
        return np.where(k > 0, -(go + (k - 1) * ge), 0.0)

    # Traceback flags
    h_from_y = np.zeros((n + 1, m + 1), dtype=bool)  # H[i,j] = Y[i,j]  (else V = max(M, X[, 0]))
    v_from_x = np.zeros((n + 1, m + 1), dtype=bool)  # V[i,j] = X[i,j]  (else M, diagonal)
    x_extend = np.zeros((n + 1, m + 1), dtype=bool)  # X[i,j] extends X[i-1,j] (else opens from H[i-1,j])
    y_extend = np.zeros((n + 1, m + 1), dtype=bool)  # Y[i,j] extends Y[i,j-1] (else opens from V[i,j-1])
    start = np.zeros((n + 1, m + 1), dtype=bool)     # the alignment starts here: traceback stops

    # Row 0
    H_prev = edge_score(cols)
    X_prev = np.full(m + 1, -np.inf)
    last_col = np.empty(n + 1)
    last_col[0] = H_prev[m]
    if local:
        start[0, :] = True
    else:
        start[0, 0] = True
        h_from_y[0, 1:] = True  # leading gap in A
        y_extend[0, 2:] = True
    best_score, best_i, best_j = 0.0, 0, 0  # Smith-Waterman: best cell so far

    for i in range(1, n + 1):
        # X: A[i-1] against a gap (vertical move)
        x_open = H_prev - go
        x_ext = X_prev - ge
        X = np.maximum(x_open, x_ext)
        x_extend[i] = x_ext > x_open

        # M: A[i-1] paired with B[j-1] (diagonal move)
        M = np.empty(m + 1)
        M[0] = -np.inf
        M[1:] = H_prev[:-1] + S[i - 1]

        V = np.maximum(M, X)
        v_from_x[i] = X > M
        if local:
            # Smith-Waterman: a prefix scoring <= 0 is dropped and a new alignment starts
            start[i] = V <= 0
            V = np.maximum(V, 0.0)
            X[0] = -np.inf
        else:
            # Column 0: leading gap in B
            V[0] = X[0] = edge_score(np.array(i))
            v_from_x[i, 0] = True
            x_extend[i, 0] = i > 1

        # Y: gap against B[j-1] (horizontal move)
        #   Y[j] = max(V[j-1] - go, Y[j-1] - ge) = max_{k<j}(V[k] + ge*k) - go - ge*(j-1)
        #   (opening from V rather than H is exact because gap_open >= gap_extend)
        Y = np.empty(m + 1)
        Y[0] = -np.inf
        if m:
            Y[1:] = np.maximum.accumulate(V[:-1] + ge * cols[:-1]) - go - ge * cols[:-1]
            y_extend[i, 1:] = (Y[:-1] - ge) > (V[:-1] - go)

        H = np.maximum(V, Y)
        h_from_y[i] = Y > V

        if local:
            j_max = int(np.argmax(H))
            if H[j_max] > best_score:
                best_score, best_i, best_j = float(H[j_max]), i, j_max
        H_prev, X_prev = H, X
        last_col[i] = H[m]

    # Where the traceback starts
    rev: list[tuple[int | None, int | None]] = []  # built in reverse
    if local:
        i, j, score = best_i, best_j, best_score
    elif free_end_gaps:
        j_end = m - int(np.argmax(H_prev[::-1]))  # best cell in the last row (ties -> j = m)
        i_end = n - int(np.argmax(last_col[::-1]))  # best cell in the last column (ties -> i = n)
        if H_prev[j_end] >= last_col[i_end]:
            i, j, score = n, j_end, float(H_prev[j_end])
        else:
            i, j, score = i_end, m, float(last_col[i_end])
        rev.extend((None, jj) for jj in range(m - 1, j - 1, -1))  # free trailing gap in A
        rev.extend((ii, None) for ii in range(n - 1, i - 1, -1))  # free trailing gap in B
    else:
        i, j, score = n, m, float(H_prev[m])

    # Traceback; ties prefer the diagonal
    state = "H"
    while True:
        if state == "H":
            state = "Y" if h_from_y[i, j] else "V"
        elif state == "V":
            if start[i, j]:
                break
            state = "X" if v_from_x[i, j] else "M"
        elif state == "M":
            rev.append((i - 1, j - 1))
            i, j, state = i - 1, j - 1, "H"
        elif state == "X":
            rev.append((i - 1, None))
            state = "X" if x_extend[i, j] else "H"
            i -= 1
        else:  # "Y"
            rev.append((None, j - 1))
            state = "Y" if y_extend[i, j] else "V"
            j -= 1
    rev.reverse()
    return Alignment(score=score, pairs=rev)
# ---- END VERBATIM smith_waterman ----


def _verbatim_text(name: str) -> str:
    text = Path(__file__).read_text()
    begin, end = f"# ---- BEGIN VERBATIM {name} ----\n", f"# ---- END VERBATIM {name} ----"
    return text[text.index(begin) + len(begin):text.index(end)]


def check_verbatim(source: str | os.PathLike | None = None) -> bool:
    """True when every verbatim block hashes to its recorded sha256 and, if the oracle source file is given (or
    found via $FOLDMATCH_ORACLE_SOURCE), that file has md5 7ec19b7e... and its line ranges equal the blocks."""
    for name, (_a, _b, sha) in VERBATIM_BLOCKS.items():
        if hashlib.sha256(_verbatim_text(name).encode()).hexdigest() != sha:
            return False
    source = source or os.environ.get("FOLDMATCH_ORACLE_SOURCE")
    if source:
        data = Path(source).read_bytes()
        if hashlib.md5(data).hexdigest() != ORACLE_SOURCE_MD5:
            return False
        lines = data.decode().split("\n")
        for name, (a, b, _sha) in VERBATIM_BLOCKS.items():
            if "\n".join(lines[a - 1:b]) + "\n" != _verbatim_text(name):
                return False
    return True


# =============================================================================
# S5 / S7 wrappers and the float64 pair oracle
# =============================================================================
def reference_qprep64(q16) -> np.ndarray:
    """harness query rows: fp64(q) / |q| (unit rows). S5 adds: |q_i| < 1e-12 -> zero row; NaN / Inf is an error."""
    Q = np.asarray(q16, dtype=np.float64)
    if not np.isfinite(Q).all():
        raise FloatingPointError("non-finite query encoding (S5: error)")
    nrm = np.linalg.norm(Q, axis=1, keepdims=True)
    zero = nrm < QPREP_EPS
    return np.where(zero, 0.0, Q / np.where(zero, 1.0, nrm))


def decode64(codes) -> np.ndarray:
    """t = (unpackbits(codes)[:, :64] * 2 - 1) / 8 (unit rows), float64."""
    return (np.unpackbits(np.asarray(codes, np.uint8), axis=1)[:, :64].astype(np.float64) * 2 - 1) / 8.0


def reference_rowcol_s7(sim):
    """normalize_scores(sim, 'rowcol') with S7; same expression order as the verbatim harness (bit-identical to it
    when nothing is degenerate). Returns (z, degenerate rows, degenerate columns)."""
    sim = np.asarray(sim, dtype=np.float64)
    eps = 1e-8
    mu, sd = sim.mean(axis=1, keepdims=True), sim.std(axis=1, keepdims=True)
    nu, tau = sim.mean(axis=0, keepdims=True), sim.std(axis=0, keepdims=True)
    z_row = (sim - mu) / (sd + eps)
    z_col = (sim - nu) / (tau + eps)
    dr, dc = sd < S7_EPS, tau < S7_EPS
    if dr.any():
        z_row = np.where(dr, 0.0, z_row)
    if dc.any():
        z_col = np.where(dc, 0.0, z_col)
    return 0.5 * (z_row + z_col), int(dr.sum()), int(dc.sum())


def aln_path(aln: Alignment) -> np.ndarray:
    """Alignment.pairs -> int32 [k, 2] (query i, target j), -1 = gap, forward order."""
    return np.array([(-1 if i is None else i, -1 if j is None else j) for i, j in aln.pairs], np.int32).reshape(-1, 2)


def reference_pair64(q16, codes, offset: float = 1.0, gap_open: float = 5.0, gap_extend: float = 0.5,
                     s7: bool = True, path: bool = True, margins: bool = False) -> dict:
    """float64 oracle for one pair: S_f (+ forward path), S_r = the same on the row-reversed query (S9).
    ``margins`` adds the best-cell and traceback margins of the forward run (:func:`sw_margins64`)."""
    Q, T = reference_qprep64(q16), decode64(codes)
    res = {}
    for d, Qd in (("f", Q), ("r", Q[::-1])):
        sim = Qd @ T.T
        if s7:
            z, nr, nc = reference_rowcol_s7(sim)
        else:
            z, nr, nc = normalize_scores(sim, "rowcol"), -1, -1
        a = smith_waterman(z - offset, gap_open, gap_extend)
        res[f"S_{d}"] = float(a.score)
        res[f"ndeg_rows_{d}"], res[f"ndeg_cols_{d}"] = nr, nc
        if d == "f":
            res["z"] = z
            if path:
                res["path"] = aln_path(a)
            if margins:
                mg = sw_margins64(z - offset, gap_open, gap_extend)
                res.update(best_margin=mg["best_margin"], tb_margin=mg["tb_margin"])
    return res


def _gap(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b)


def sw_margins64(scores, gap_open: float, gap_extend: float) -> dict:
    """The verbatim local recurrences (``_align``, local=True) keeping H, X, V (before the clip), M and Y, then the
    oracle traceback with each decision's distance from a tie. Returns score, best cell (1-based), path,
    ``best_margin`` (best H minus the best H of any other cell; 0 for an empty alignment) and ``tb_margin`` (the
    smallest decision gap along the traceback: Y vs V, the V <= 0 start test, X vs M, and the X / Y extend tests)."""
    S = np.asarray(scores, dtype=np.float64)
    n, m = S.shape
    go, ge = float(gap_open), float(gap_extend)
    cols = np.arange(m + 1)
    H = np.zeros((n + 1, m + 1)); X = np.full((n + 1, m + 1), -np.inf); V = np.zeros((n + 1, m + 1))
    Vpre = np.full((n + 1, m + 1), -np.inf); M = np.full((n + 1, m + 1), -np.inf); Y = np.full((n + 1, m + 1), -np.inf)
    H_prev, X_prev = np.zeros(m + 1), np.full(m + 1, -np.inf)
    best, bi, bj = 0.0, 0, 0
    for i in range(1, n + 1):
        Xi = np.maximum(H_prev - go, X_prev - ge)
        Mi = np.empty(m + 1)
        Mi[0] = -np.inf
        Mi[1:] = H_prev[:-1] + S[i - 1]
        Vi = np.maximum(Mi, Xi)
        Vpre[i] = Vi
        Vi = np.maximum(Vi, 0.0)
        Xi[0] = -np.inf
        Yi = np.empty(m + 1)
        Yi[0] = -np.inf
        if m:
            Yi[1:] = np.maximum.accumulate(Vi[:-1] + ge * cols[:-1]) - go - ge * cols[:-1]
        Hi = np.maximum(Vi, Yi)
        H[i], X[i], V[i], M[i], Y[i] = Hi, Xi, Vi, Mi, Yi
        j_max = int(np.argmax(Hi))
        if Hi[j_max] > best:
            best, bi, bj = float(Hi[j_max]), i, j_max
        H_prev, X_prev = Hi, Xi
    if best > 0.0:
        Hf = H.ravel().copy()
        Hf[bi * (m + 1) + bj] = -np.inf
        best_margin = best - float(Hf.max())
    else:
        best_margin = 0.0
    tb = math.inf
    i, j, state, rev = bi, bj, "H", []
    while best > 0.0:
        if state == "H":
            tb = min(tb, _gap(Y[i, j], V[i, j]))
            state = "Y" if Y[i, j] > V[i, j] else "V"
        elif state == "V":
            tb = min(tb, abs(Vpre[i, j]) if math.isfinite(Vpre[i, j]) else math.inf)
            if Vpre[i, j] <= 0 or i == 0 or j == 0:
                break
            tb = min(tb, _gap(X[i, j], M[i, j]))
            state = "X" if X[i, j] > M[i, j] else "M"
        elif state == "M":
            rev.append((i - 1, j - 1))
            i, j, state = i - 1, j - 1, "H"
        elif state == "X":
            rev.append((i - 1, -1))
            tb = min(tb, _gap(X[i - 1, j] - ge, H[i - 1, j] - go))
            state = "X" if (X[i - 1, j] - ge) > (H[i - 1, j] - go) else "H"
            i -= 1
        else:
            rev.append((-1, j - 1))
            tb = min(tb, _gap(Y[i, j - 1] - ge, V[i, j - 1] - go))
            state = "Y" if (Y[i, j - 1] - ge) > (V[i, j - 1] - go) else "V"
            j -= 1
    path = np.array(rev[::-1], np.int32).reshape(-1, 2)
    return dict(score=best, best=(bi, bj), path=path, best_margin=best_margin, tb_margin=tb)


# =============================================================================
# S2/S3 reference encoder and S11 E-values
# =============================================================================
def _gelu_exact(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf
    except ImportError:  # pragma: no cover
        erf = np.frompyfunc(math.erf, 1, 1)
    return 0.5 * x * (1.0 + np.asarray(erf(x / math.sqrt(2.0)), dtype=np.float64))


def _mlp64(xw: np.ndarray, w: dict) -> np.ndarray:
    """f(x) = lin(x) + fc2(GELU_exact(fc1(x))) in float64; w holds numpy 'lin.weight', 'lin.bias', 'fc1.*', 'fc2.*'
    (nn.Linear layout: weight [out, in])."""
    g = {k: np.asarray(v, np.float64) for k, v in w.items()}
    h = xw @ g["fc1.weight"].T + g["fc1.bias"]
    return xw @ g["lin.weight"].T + g["lin.bias"] + _gelu_exact(h) @ g["fc2.weight"].T + g["fc2.bias"]


def reference_encode(rows16, dims, offset, M, f_q: dict, f_t: dict):
    """Reference S2-S5 encoder for ESM3 rows ``rows16`` fp16 [L, 1536] (BOS / EOS already removed):
    xs = fp32(x16[:, dims]) - fp32(offset); xw = fp32(fp16(xs @ M)) (S2, product in float64 here);
    h = f(xw) in float64 (S3); codes = packbits(h_t >= 0) (S4); q = fp16(h_q) (S5).
    Returns (codes uint8 [L, 8], q float16 [L, 64], h_t float64 [L, 64]). Bits with |h_t| ~ 0 may differ from an
    fp32 encoder (see the fixtures' margin_t)."""
    x16 = np.asarray(rows16, np.float16)
    xs = x16[:, np.asarray(dims)].astype(np.float32) - np.asarray(offset, np.float32)
    xw = (xs.astype(np.float64) @ np.asarray(M, np.float64)).astype(np.float16).astype(np.float64)
    h_t, h_q = _mlp64(xw, f_t), _mlp64(xw, f_q)
    return np.packbits(h_t >= 0, axis=1), h_q.astype(np.float16), h_t


def reference_lam_lnk(m, centres, lam, lnK):
    """S11: lambda(m), lnK(m) linear in ln m between the centres, clamped at both ends (D15)."""
    x = np.log(np.asarray(m, np.float64))
    lc = np.log(np.asarray(centres, np.float64))
    return np.interp(x, lc, np.asarray(lam, np.float64)), np.interp(x, lc, np.asarray(lnK, np.float64))


def reference_evalue(S_prime, m, N, centres, lam, lnK):
    """S11: lnE = (lnK(m) + ln m + ln N) - lambda(m) S' and bits = (lambda(m) S' - lnK(m)) / ln 2 (float64)."""
    la, lk = reference_lam_lnk(m, centres, lam, lnK)
    Sp = np.asarray(S_prime, np.float64)
    lnE = (lk + np.log(np.asarray(m, np.float64)) + np.log(np.asarray(N, np.float64))) - la * Sp
    return lnE, (la * Sp - lk) / math.log(2.0)
