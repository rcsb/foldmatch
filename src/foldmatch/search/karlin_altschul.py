"""Calibrated Karlin-Altschul statistics for gapped local alignment.

Computes E-values, p-values and bit scores from a raw Smith-Waterman score using
published Karlin-Altschul constants and a finite-size (edge-effect) corrected
search space, so the values are exact and reproducible rather than estimated by
sampling.

The constants and the correction are those of NCBI's ALP library ("Sequence
Local Statistics", Sheetlin/Park/Spouge), taken as vendored and called by MMseqs2
via ``Sls::AlignmentEvaluer`` — which makes the output directly comparable to
``mmseqs search``. This module ports that call path:

    E = K * exp(-lambda * S) * area(S, query_len, db_residues)

where ``area`` is *not* a plain ``length - (a*S + b)`` edge correction but a
Gaussian-smoothed expression with a covariance cross-term (see :func:`area`).

Source map (soedinglab/MMseqs2, pinned to commit
5d152c612b6ad2a56f657b7a02c127eceaea2a75; line numbers drift with master):

    evalue()/bitScore()  lib/alp/sls_alignment_evaluer.hpp:135-162
    area()               lib/alp/sls_pvalues.cpp:366-545
    thresholds           lib/alp/sls_pvalues.cpp:46, 342-364
    normal CDF           lib/alp/sls_basic.hpp:195-198
    1-exp(y)             lib/alp/sls_basic.cpp:94-105
    parameter table      src/alignment/EvalueComputation.h:56-81
    caller               src/alignment/EvalueComputation.h:32-45

Validated against mmseqs 18-8cc5c on 33,192 real hits (200 SCOP queries vs a
200k-sequence, 64.5M-residue UniProt subset, E-values spanning 8e-32 to 1e4),
run with ``--comp-bias-corr 0 --mask 0``: worst relative E-value error 4.7e-4,
which is mmseqs's own 4-significant-digit output precision, and 400/400 sampled
pairs had identical raw Smith-Waterman scores under biotite.

mmseqs's default ``--comp-bias-corr 1`` changes the raw score these statistics
are computed from, not the statistics themselves. FoldMatch ports that
correction (``alignment.local_composition_bias``) but leaves it off by default;
enable it with ``fm-search``'s ``--comp-bias-corr`` flag when comparing with a
default mmseqs run.

Caution when diffing against mmseqs yourself: its ``raw`` output column is
reconstructed from the stored *integer* bit score and is lossy by +/-1. Since
E = K*exp(-lambda*S)*area, a 1-unit score error moves E by exp(+/-lambda) ~ 1.31x.
Recover the true score as the integer consistent with both the reported bit
score and the reported E-value.

Fidelity notes (deliberate, do not "fix"):

* **Gap convention.** MMseqs2's Smith-Waterman charges a length-L gap as
  ``open + (L-1)*extend`` (a length-1 gap costs 11 at nominal 11/1), which is
  exactly biotite's convention — so biotite's native ``(-11, -1)`` feeds these
  constants directly with no adjustment. ALP itself *documents* its penalty as
  ``open + L*extend`` and MMseqs2 passes 11/1 through unconverted, so the
  hardcoded lambda/K are calibrated to a gap model one extension unit away from
  what the aligner scores. Reproducing MMseqs2 means inheriting that offset.
  Corollary: MMseqs2's "11/1" equals NCBI BLAST's "10/1" in total gap cost, so
  these numbers are **not** interchangeable with BLAST's at any nominal setting.
* **No area floor.** ALP's ``Tmax(area, 1.0)`` is commented out upstream and the
  ``blast_`` fallback that would force ``area = 1.0`` is dead code. Areas in
  (0, 1) are therefore admitted and deflate E-values for short queries. This is
  reproduced faithfully.
* **p-value** uses ALP's piecewise ``1 - exp(y)`` (Taylor below |y| = 1e-3),
  not :func:`math.expm1`, which is more accurate but is not what MMseqs2 computes.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

# lib/alp/sls_pvalues.cpp:46
_NAT_CUT_OFF_IN_MAX = 2.0
# lib/alp/sls_basic.hpp:59 — computed, never a decimal literal, so that rounding
# matches the C++ bit for bit.
_CONST_VAL = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass
class AlpParams:
    """One ALP gapped/gapless parameter set (a row of MMseqs2's static table).

    Field order in the C++ aggregate is ``(lambda, k, a1, b1, a2, b2, alpha1,
    beta1, alpha2, beta2, sigma, tau)``; ``initParameters`` remaps subscript
    1 -> ``_J`` and 2 -> ``_I`` (sls_alignment_evaluer.cpp:679-723), which is
    what the ``_i``/``_j`` suffixes here reflect.
    """
    lam: float
    k: float
    a_i: float
    b_i: float
    a_j: float
    b_j: float
    alpha_i: float
    beta_i: float
    alpha_j: float
    beta_j: float
    sigma: float
    tau: float
    # Precomputed once, as in pvalues::compute_tmp_values (sls_pvalues.cpp:342-364).
    vi_y_thr: float = field(init=False)
    vj_y_thr: float = field(init=False)
    c_y_thr: float = field(init=False)

    def __post_init__(self):
        # lambda > 0 for every shipped set, so the else-branch that zeroes all
        # three thresholds never fires.
        self.vi_y_thr = max(_NAT_CUT_OFF_IN_MAX * self.alpha_i / self.lam, 0.0)
        self.vj_y_thr = max(_NAT_CUT_OFF_IN_MAX * self.alpha_j / self.lam, 0.0)
        self.c_y_thr = max(_NAT_CUT_OFF_IN_MAX * self.sigma / self.lam, 0.0)


# BLOSUM62, gapOpen=11, gapExtend=1, gapped — the only gapped BLOSUM62 row in
# MMseqs2's table (src/alignment/EvalueComputation.h:69-74) and the default for
# protein search (Parameters.cpp:2571-2572). The literals carry more significant
# digits than an IEEE-754 double holds; Python truncates them identically to the
# C++ compiler, so they are transcribed in full.
BLOSUM62_GAPPED_11_1 = AlpParams(
    lam=0.27359865037097330642,
    k=0.044620920658722244834,
    a_i=1.5938724404943873658,
    b_i=-19.959867650284412122,
    a_j=1.5938724404943873658,
    b_j=-19.959867650284412122,
    alpha_i=30.455610143099914211,
    beta_i=-622.28684628915891608,
    alpha_j=30.455610143099914211,
    beta_j=-622.28684628915891608,
    sigma=29.602444874818868215,
    tau=-601.81087985041381216,
)

# MMseqs2 hardcodes exactly four parameter sets and falls back to a wall-clock-
# budgeted Monte-Carlo fit (ALP initGapped) for anything else — a fit whose
# result depends on machine speed even with its fixed seed. We therefore support
# only the combination we can reproduce exactly, and refuse the rest rather than
# guessing or substituting NCBI's published constants (which genuinely differ:
# gapped 11/1 lambda 0.2736 here vs BLAST's 0.267).
_SUPPORTED: Dict[Tuple[str, int, int], AlpParams] = {
    ("BLOSUM62", 11, 1): BLOSUM62_GAPPED_11_1,
}

SUPPORTED_COMBINATIONS = "BLOSUM62 with gap_open=11, gap_extend=1"


def params_for(gap_open: int, gap_extend: int, matrix_name: str = "BLOSUM62") -> AlpParams:
    """Look up the ALP parameter set, or raise if we cannot reproduce MMseqs2."""
    key = (matrix_name.upper(), int(gap_open), int(gap_extend))
    try:
        return _SUPPORTED[key]
    except KeyError:
        raise ValueError(
            f"No calibrated statistics for {matrix_name} with gaps "
            f"{gap_open}/{gap_extend}; the only supported combination is "
            f"{SUPPORTED_COMBINATIONS}. Use significance_mode='sampled' for a "
            f"relative-only signal at these gap penalties."
        ) from None


def _normal_probability(x: float) -> float:
    """Standard normal CDF, as sls_basic.hpp:195-198 (the 1-arg overload).

    ALP's 2-arg trapezoidal overload is dead code in MMseqs2 and is *not* what
    ``area`` calls, so this must stay an ``erfc`` form.
    """
    return 0.5 * math.erfc(-math.sqrt(0.5) * x)


def _half_term(length: float, a: float, b: float, alpha: float, beta: float,
               v_thr: float, y: float) -> Tuple[float, float]:
    """One side of the area product: E[max(N(mu, v), 0)] and its Phi(F).

    Ports sls_pvalues.cpp:423-456 (the _I side) / 459-490 (the _J side).
    """
    li_y = length - (a * y + b)
    v_y = max(v_thr, alpha * y + beta)      # the only clamps in the whole path
    sqrt_v = math.sqrt(v_y)
    if sqrt_v == 0.0:
        # Degenerate branch (sls_pvalues.cpp:438-445): the C++ divides by a
        # 1e100 sentinel, which drives Phi -> 1 and phi -> 0. Unreachable for
        # BLOSUM62 11/1 (vi_y_thr ~= 222.6 > 0) but kept for structural fidelity.
        p_f = 1.0
        e_f = 0.0
    else:
        f = li_y / sqrt_v
        p_f = _normal_probability(f)
        e_f = -_CONST_VAL * math.exp(-0.5 * f * f)   # == -phi(F), negative
    # Minus sign with a negative e_f: algebraically li_y*Phi(F) + sqrt_v*phi(F).
    p = li_y * p_f - sqrt_v * e_f
    return p, p_f


def area(score: int, seqlen1: int, seqlen2: int, params: AlpParams) -> float:
    """ALP's finite-size-corrected search-space area.

    ``seqlen1`` is the query length and ``seqlen2`` the target extent, matching
    MMseqs2's ``computeEvalue(score, queryLength)`` call. Note the argument swap
    at sls_alignment_evaluer.cpp:1014-1016: ``seqlen2`` binds to the ``_I``
    parameter family and ``seqlen1`` to ``_J``. That is numerically a no-op for
    this (symmetric) parameter set, but is implemented as written so the code
    stays correct if asymmetric parameters are ever added.
    """
    if seqlen1 <= 0 or seqlen2 <= 0:
        raise ValueError("sequence lengths must be positive")
    y = float(score)
    m = float(seqlen2)   # -> _I
    n = float(seqlen1)   # -> _J

    p1, p_m_f = _half_term(m, params.a_i, params.b_i, params.alpha_i,
                           params.beta_i, params.vi_y_thr, y)
    p2, p_n_f = _half_term(n, params.a_j, params.b_j, params.alpha_j,
                           params.beta_j, params.vj_y_thr, y)
    c_y = max(params.c_y_thr, params.sigma * y + params.tau)
    # No floor: ALP's Tmax(area, 1.0) is commented out upstream (see module docstring).
    return p1 * p2 + c_y * p_m_f * p_n_f


def evalue(score: int, query_len: int, target_extent: int, params: AlpParams) -> float:
    """MMseqs2's E-value: ``K * exp(-lambda*S) * area``.

    ``target_extent`` is the total residue count searched — for a database
    E-value that is the concatenated size of the whole target DB. MMseqs2 has no
    per-sequence term and never multiplies by the number of DB sequences.
    """
    return (params.k * math.exp(-params.lam * float(score))
            * area(score, query_len, target_extent, params))


def bit_score(score: int, params: AlpParams) -> float:
    """Length-independent bit score ``(lambda*S - ln K)/ln 2``.

    MMseqs2 emits this rounded to an int (``int(bitScore + 0.5)``,
    Matcher.cpp:132); compare after applying the same rounding.
    """
    return (params.lam * float(score) - math.log(params.k)) / math.log(2.0)


def pvalue_from_evalue(e: float) -> float:
    """ALP's piecewise ``1 - exp(-E)`` (sls_basic.cpp:94-105).

    Uses the 5th-order Taylor form below |y| = 1e-3 exactly as MMseqs2 does;
    :func:`math.expm1` would be more accurate but would not agree digit for digit.
    """
    y = -e
    if abs(y) > 1e-3:
        return 1.0 - math.exp(y)
    return -(y * (120 + y * (60 + y * (20 + y * (5.0 + y)))) / 120.0)
