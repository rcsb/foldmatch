"""E-value statistics of esm-codes systems (method spec S10, S11; plan section 8 P4). numpy only, no torch.

For a query of length ``m`` against a database of ``N`` residues, with ``S = fp64(S_f) - fp64(S_r)`` and
``S' = S - Q90(query)``::

    lambda(m), lnK(m) = linear in ln m between the six bin centres; clamped below the first and above the last
    lnE  = lnK(m) + ln m + ln N - lambda(m) S'          (float64, left to right)
    bits = (lambda(m) S' - lnK(m)) / ln 2               (so E = m N 2^-bits), printed as floor(bits + 0.5)

The table and its rules come from the scoring system: every function takes a ``table``, any object with
``bin_centres``, ``lam``, ``lnK``, ``low_m_rule`` and ``high_m_rule`` (an
:class:`~foldmatch.search.scoring_system.EsmCodesSystem`, or the picklable worker view of a prepared one). Only the
``"clamp"`` rule is implemented (S11, D15); a table naming any other rule is refused, never approximated.

The interpolation is ``numpy.interp(ln m, ln centres, values)``, the research code's function, so the values at the
centres are the table's own numbers bit for bit.
"""
from __future__ import annotations

import math
import sys
from typing import Tuple

import numpy as np

IMPLEMENTED_M_RULES = ("clamp",)
LN2 = math.log(2.0)
LN10 = math.log(10.0)
_LOG_NORMAL_MIN = math.log(sys.float_info.min)     # below this exp() is subnormal (or 0) and loses digits
_LOG_MAX = math.log(sys.float_info.max)
#: |bits| above this is refused by render_bits: beyond 2^53 a float64 no longer holds every integer, and the array
#: path's int64 would saturate at 2^63. Real bit scores are a few thousand at most.
MAX_RENDERED_BITS = float(2 ** 53)


def _table(table) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for rule in ("low_m_rule", "high_m_rule"):
        value = getattr(table, rule)
        if value not in IMPLEMENTED_M_RULES:
            raise ValueError(f"{rule}={value!r} is not implemented by this FoldMatch (implemented: "
                             f"{', '.join(IMPLEMENTED_M_RULES)})")
    c = np.asarray(table.bin_centres, np.float64)
    lam = np.asarray(table.lam, np.float64)
    lnk = np.asarray(table.lnK, np.float64)
    if c.ndim != 1 or c.size < 2 or lam.shape != c.shape or lnk.shape != c.shape:
        raise ValueError("the E-value table needs matching 1-D bin_centres, lam and lnK of at least 2 entries")
    if not (np.all(c > 0) and np.all(np.diff(c) > 0)):
        raise ValueError("bin_centres must be positive and strictly increasing")
    return np.log(c), lam, lnk


def _positive(name: str, value) -> np.ndarray:
    a = np.asarray(value, np.float64)
    if not np.all(np.isfinite(a)) or not np.all(a > 0):
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return a


def _out(a):
    return float(a) if np.ndim(a) == 0 else a


def lam_lnk(m, table):
    """``(lambda(m), lnK(m))`` for query length(s) ``m`` > 0 (S11): linear in ln m between the centres, the first
    centre's values below it and the last centre's above it (both rules ``"clamp"``)."""
    lc, lam, lnk = _table(table)
    x = np.log(_positive("m (query length)", m))
    return (_out(np.interp(x, lc, lam, left=lam[0], right=lam[-1])),
            _out(np.interp(x, lc, lnk, left=lnk[0], right=lnk[-1])))


def log_evalue(s_prime, m, n_residues, table):
    """``lnE = lnK(m) + ln m + ln N - lambda(m) S'`` in float64 (S11). ``n_residues`` is N, the database residue
    count (manifest ``total_residues``); it must be > 0, since an empty database has no search space."""
    la, lk = lam_lnk(m, table)
    m = np.asarray(m, np.float64)
    n = _positive("n_residues (database residues)", n_residues)
    sp = np.asarray(s_prime, np.float64)
    if not np.all(np.isfinite(sp)):
        raise ValueError("S' must be finite")
    return _out((lk + np.log(m) + np.log(n)) - la * sp)


def bit_score(s_prime, m, table):
    """``bits = (lambda(m) S' - lnK(m)) / ln 2`` (S11), so that ``E = m N 2^-bits``."""
    la, lk = lam_lnk(m, table)
    sp = np.asarray(s_prime, np.float64)
    if not np.all(np.isfinite(sp)):
        raise ValueError("S' must be finite")
    return _out((la * sp - lk) / LN2)


def render_bits(bits):
    """The printed bit score, ``floor(bits + 0.5)`` (D9): -2.5 -> -2, -0.4 -> 0, -1.7 -> -2 (the BLOSUM path's
    ``int(bits + 0.5)`` truncates towards zero and would print -1). An int, or an int64 array. A bit score that is
    not finite or exceeds ``MAX_RENDERED_BITS`` (2^53) in magnitude is refused, on both paths alike."""
    a = np.asarray(bits, np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"bit score must be finite, got {bits!r}")
    if np.any(np.abs(a) > MAX_RENDERED_BITS):
        raise ValueError(f"bit score beyond +/-2^53 cannot be rendered exactly, got {bits!r}")
    r = np.floor(a + 0.5)
    return int(r) if r.ndim == 0 else r.astype(np.int64)


def format_evalue_from_log(log_e: float) -> str:
    """E rendered as ``%.3E`` from ln E, without under- or overflow: ``-1000 -> '5.076E-435'``.

    Inside the normal float range the text is exactly ``f"{math.exp(log_e):.3E}"``, the format of the BLOSUM62 path.
    Outside it the mantissa and exponent come from ``log10 E = ln E / ln 10``. ``-inf`` gives ``0.000E+00``."""
    x = float(log_e)
    if math.isnan(x):
        raise ValueError("ln E is NaN")
    if x == -math.inf:
        return "0.000E+00"
    if x == math.inf:
        return "INF"
    if _LOG_NORMAL_MIN <= x <= _LOG_MAX:
        return f"{math.exp(x):.3E}"
    return _format_from_log10(x / LN10)


def _format_from_log10(t: float) -> str:
    """``%.3E`` text of ``10 ** t`` for any finite ``t`` (no float of that size needed)."""
    exponent = math.floor(t)
    mantissa = f"{10.0 ** (t - exponent):.3f}"
    if mantissa == "10.000":                     # 9.9995... rounds up into the next decade
        mantissa, exponent = "1.000", exponent + 1
    return f"{mantissa}E{'-' if exponent < 0 else '+'}{abs(exponent):02d}"


def q90(panel_scores, quantile: float = 0.9, method: str = "linear") -> float:
    """S10: ``numpy.quantile(float64 S(query, panel members), quantile)`` with linear interpolation."""
    if method != "linear":
        raise ValueError(f"quantile method {method!r} is not implemented (implemented: linear)")
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")
    s = np.asarray(panel_scores, np.float64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("panel scores must be a non-empty 1-D array")
    if not np.all(np.isfinite(s)):
        raise ValueError("panel scores must be finite (a skipped or failed panel pair has no score)")
    return float(np.quantile(s, float(quantile), method="linear"))
