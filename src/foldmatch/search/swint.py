"""ctypes wrapper of ``fmswi_align``, the integer Smith-Waterman entry point of the native fmsw library (DP id
``fmswi/1``): the alignment engine of the BLOSUM62 Stage 2 (:mod:`foldmatch.search.alignment`).

The kernel is output-equivalent to biotite 1.7.1 ``align_optimal(..., local=True, max_number=1)`` with affine gaps:
the same raw score, end cell and path, hence the same alignment columns (model and tie rules in ``fmsw.h``; checked
against biotite by tests/test_swint.py). It aligns one query against a batch of targets per call, so a Stage-2 task
makes one native call for all its candidates.

numpy + ctypes only (never torch, never biotite). The library is the one :func:`foldmatch.search.swkern.load` selects
(ISA dispatch, ``FOLDMATCH_SW_ISA``, ABI and stale-source checks): a missing, stale or wrong-ABI library raises
:class:`~foldmatch.search.swkern.KernelError` with the rebuild command, as for esm-codes.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import logging
import struct
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from foldmatch import _native

from . import swkern

logger = logging.getLogger(__name__)

DP_ID = "fmswi/1"
NOUT = 11
(SCORE, QSTART, QEND, TSTART, TEND, ALNLEN, NIDENT, GAPOPEN, NQGAP, NTGAP, WIDE) = range(NOUT)
MAX_SCORE = 1 << 20                     # FMSWI_MAX_SCORE: |profile score| bound
MAX_GAP = 1 << 20                       # FMSWI_MAX_GAP: gap_open and gap_extend bound
STATUS_INTERNAL = 3                     # FMSWI_ERR_INTERNAL: the prefix traceback missed the end cell (never expected)

#: biotite's ProteinSequence alphabet in code order (the rows / columns of its BLOSUM62).
ALPHABET = "ACDEFGHIKLMNPQRSTVWYBZX*"

_SANITIZE = bytes(c if chr(c) in ALPHABET else ord("X") for c in range(256))
_TO_CODE = bytes(ALPHABET.index(chr(c)) if chr(c) in ALPHABET else 255 for c in range(256))

_vp = C.c_void_p

#: Traceback strategies of the kernel, with identical outputs: 'full' = one pass storing a flag byte per DP cell;
#: 'prefix' = a score-only pass, then the flag pass over rows 1..qend x columns 1..tend only. Stage 2 uses
#: :data:`DEFAULT_TRACE`: on the Stage 2 of the paper's Fig. 2 search (17.7M pairs, one 128-thread AVX2 node) both gave
#: byte-identical tables in the same wall time, prefix with less worker CPU (4,181 s against 6,103-6,700 s) and a
#: lower peak memory (35.9 against 42.4 GB: fewer flag pages touched). 'full' is kept for the tests.
TRACE_MODES = {"full": 1, "prefix": 2}
DEFAULT_TRACE = "prefix"


def trace_mode(name: str = DEFAULT_TRACE) -> int:
    """The kernel's traceback argument for a strategy of :data:`TRACE_MODES`."""
    try:
        return TRACE_MODES[name]
    except KeyError:
        raise ValueError(f"{name!r} is not a traceback strategy; expected one of {', '.join(TRACE_MODES)}") from None


class IntKernel:
    """``fmswi_*`` of a loaded fmsw library."""

    def __init__(self, kernel: swkern.Kernel):
        lib = kernel.lib
        try:
            lib.fmswi_dp_id.restype = C.c_char_p
            lib.fmswi_dp_id.argtypes = []
            lib.fmswi_align.restype = C.c_int
            lib.fmswi_align.argtypes = [_vp, _vp, C.c_int, C.c_int, _vp, _vp, _vp, C.c_int, C.c_int, C.c_int, C.c_int,
                                        _vp, _vp, C.c_size_t, _vp]
        except AttributeError as e:
            raise swkern.KernelError(f"{kernel.path} has no integer Smith-Waterman entry point ({e}); "
                                     f"{_native.rebuild_hint()}") from e
        dp = lib.fmswi_dp_id().decode()
        if dp != DP_ID:
            raise swkern.KernelError(f"{kernel.path} implements integer DP {dp!r}, expected {DP_ID!r}")
        self.kernel = kernel
        self.lib = lib
        self.isa = kernel.isa

    def __repr__(self):
        return f"IntKernel(isa={self.isa!r}, path={str(self.kernel.path)!r})"


_INT_KERNELS = {}


def load(isa: Optional[str] = None, **kw) -> IntKernel:
    """The integer kernel of :func:`swkern.load` (cached; same arguments). Raises
    :class:`~foldmatch.search.swkern.KernelError` (with the rebuild command) when no usable library is found."""
    k = swkern.load(isa, **kw)
    ik = _INT_KERNELS.get(id(k))
    if ik is None or ik.kernel is not k:
        ik = _INT_KERNELS[id(k)] = IntKernel(k)
    return ik


# =====================================================================================================================
# sequences
# =====================================================================================================================
def sanitize_bytes(sequence: str) -> Optional[bytes]:
    """``alignment.sanitize_sequence(sequence)`` as ASCII bytes, or None when the stripped, uppercased text is not ASCII
    (the caller then uses the generic function: Python's ``upper`` and ``strip`` act on non-ASCII text)."""
    text = sequence.strip().upper()
    if not text.isascii():
        return None
    return text.encode("ascii").translate(_SANITIZE)


def codes_of(clean: bytes) -> np.ndarray:
    """uint8 codes (biotite order) of sanitized ASCII bytes."""
    return np.frombuffer(clean.translate(_TO_CODE), dtype=np.uint8)


def query_profile(qcodes: np.ndarray, matrix: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """int32 [n, A] position-specific scores: row ``qcodes[i]`` of ``matrix`` plus ``bias[i]`` (composition bias)."""
    prof = np.asarray(matrix, dtype=np.int32)[np.asarray(qcodes, dtype=np.intp)]
    if bias is not None:
        prof = prof + np.asarray(bias, dtype=np.int32)[:, None]
    return np.ascontiguousarray(prof, dtype=np.int32)


# =====================================================================================================================
# the batch call
# =====================================================================================================================
@dataclass
class BatchResult:
    """One :func:`align_batch` call: ``out`` int32 [K, NOUT] (fields SCORE .. WIDE); with ``ops`` the paths as
    ``b'M' / b'D' / b'I'`` (forward order), pair k at ``ops[ops_off[k]:ops_off[k + 1]]``."""
    out: np.ndarray
    ops: Optional[bytes] = None
    ops_off: Optional[np.ndarray] = None

    def path(self, k: int) -> str:
        return self.ops[self.ops_off[k]:self.ops_off[k + 1]].decode("ascii")


def align_batch(qcodes: np.ndarray, qprof: np.ndarray, tcodes: np.ndarray, t_off: Sequence[int],
                t_len: Sequence[int], gap_open: int, gap_extend: int, *, traceback: bool = True, ops: bool = False,
                trace: str = DEFAULT_TRACE, kernel: Optional[IntKernel] = None) -> BatchResult:
    """Align one query (codes ``qcodes`` [n], profile ``qprof`` int32 [n, A]) against K targets, target k being
    ``tcodes[t_off[k]:t_off[k] + t_len[k]]``. ``gap_open`` >= 1 and ``gap_extend`` >= 0 are positive costs, both at
    most :data:`MAX_GAP` (a gap of length L costs open + (L - 1) extend). ``ops`` also returns the paths (traceback
    only). ``trace`` picks the traceback strategy (:data:`TRACE_MODES`; the outputs do not depend on it)."""
    k = kernel or load()
    q = np.ascontiguousarray(qcodes, dtype=np.uint8)
    prof = np.ascontiguousarray(qprof, dtype=np.int32)
    n = int(q.shape[0])
    if prof.ndim != 2 or prof.shape[0] != n:
        raise ValueError(f"qprof must be int32 [n={n}, A], got {prof.shape}")
    A = int(prof.shape[1])
    t = np.ascontiguousarray(tcodes, dtype=np.uint8)
    off = np.ascontiguousarray(t_off, dtype=np.int64)
    lens = np.ascontiguousarray(t_len, dtype=np.int32)
    K = int(lens.shape[0])
    if off.shape[0] != K:
        raise ValueError("t_off and t_len must have the same length")
    if K and (off.min() < 0 or int((off + lens).max()) > t.shape[0]):
        raise ValueError("a target range lies outside tcodes")
    out = np.zeros((K, NOUT), dtype=np.int32)
    want_ops = bool(ops and traceback)
    cap = int(K * n + int(lens.sum())) if want_ops else 0
    ops_buf = C.create_string_buffer(max(cap, 1)) if want_ops else None
    ops_off = np.zeros(K + 1, dtype=np.int64) if want_ops else None
    st = k.lib.fmswi_align(
        q.ctypes.data if n else None, prof.ctypes.data if n else None, n, A,
        t.ctypes.data if t.size else None, off.ctypes.data if K else None, lens.ctypes.data if K else None, K,
        int(gap_open), int(gap_extend), trace_mode(trace) if traceback else 0, out.ctypes.data if K else None,
        ops_buf, cap, ops_off.ctypes.data if want_ops else None)
    if st == swkern.STATUS_ARGS:
        raise ValueError("fmswi_align refused its arguments (code outside the alphabet, score or gap cost out of "
                         "range, gap_open < 1, or a score beyond int32)")
    if st == STATUS_INTERNAL:
        raise swkern.KernelError("fmswi_align: the prefix traceback pass did not meet the score pass's end cell")
    if st != swkern.STATUS_OK:
        raise MemoryError(f"fmswi_align could not allocate its workspace (status {st})")
    if want_ops:
        return BatchResult(out, ops_buf.raw[:int(ops_off[-1])], ops_off)
    return BatchResult(out)


def align_strings(query: str, targets: List[str], matrix: np.ndarray, gap_open: int, gap_extend: int, *,
                  bias: Optional[np.ndarray] = None, ops: bool = False, trace: str = DEFAULT_TRACE,
                  kernel: Optional[IntKernel] = None) -> BatchResult:
    """Convenience for tests: sanitized ASCII strings in, :func:`align_batch` out."""
    qb = query.encode("ascii").translate(_SANITIZE)
    qc = codes_of(qb)
    tb = [s.encode("ascii").translate(_SANITIZE) for s in targets]
    lens = np.fromiter((len(b) for b in tb), dtype=np.int32, count=len(tb))
    off = np.zeros(len(tb), dtype=np.int64)
    if len(tb) > 1:
        np.cumsum(lens[:-1], out=off[1:])
    return align_batch(qc, query_profile(qc, matrix, bias), codes_of(b"".join(tb)), off, lens, gap_open, gap_extend,
                       ops=ops, trace=trace, kernel=kernel)


def run_length_encode(ops: bytes) -> str:
    """CIGAR of a path (``b'MMMDDM'`` -> ``'3M2D1M'``)."""
    out: List[str] = []
    k, n = 0, len(ops)
    while k < n:
        c = ops[k]
        e = k + 1
        while e < n and ops[e] == c:
            e += 1
        out.append(f"{e - k}{chr(c)}")
        k = e
    return "".join(out)


def gapped_strings(ops: bytes, query: str, target: str, q_start: int, t_start: int) -> Tuple[str, str]:
    """The aligned query and target strings (``'-'`` for a gap) of a path starting at the 1-based (q_start, t_start)."""
    qa: List[str] = []
    ta: List[str] = []
    i, j = q_start - 1, t_start - 1
    for c in ops:
        if c == 77:          # 'M'
            qa.append(query[i]); ta.append(target[j]); i += 1; j += 1
        elif c == 68:        # 'D': gap in the query
            qa.append("-"); ta.append(target[j]); j += 1
        else:                # 'I': gap in the target
            qa.append(query[i]); ta.append("-"); i += 1
    return "".join(qa), "".join(ta)


# =====================================================================================================================
# golden digest (no biotite: run by swkern.selftest and tests/test_swkern.py on every build and ISA)
# =====================================================================================================================
GOLDEN_SEED = 20261031
_GOLDEN_GAPS = ((11, 1), (10, 1), (5, 2), (3, 3), (1, 1), (20, 5), (2, 0), (1, 4))
_GOLDEN_A = len(ALPHABET)


@dataclass(frozen=True)
class GoldenCase:
    """One :func:`align_batch` call of the golden set: a query (codes and profile) against its targets."""
    kind: str
    qcodes: np.ndarray                  # uint8 [n]
    qprof: np.ndarray                   # int32 [n, A]
    targets: Tuple[np.ndarray, ...]     # uint8 codes each
    gap_open: int
    gap_extend: int


def _golden_mutant(g, seg: np.ndarray) -> np.ndarray:
    """A related target: ``seg`` with a random share (0-59%) of its codes substituted, ~5% deleted, an insertion of
    0..14 codes and random flanks of 0..39 codes (never empty)."""
    rate = g.int(0, 60)
    sub = g.ints(0, 100, seg.shape) < rate
    seg = np.where(sub, g.ints(0, _GOLDEN_A, seg.shape), seg)
    seg = seg[g.ints(0, 20, seg.shape) != 0]
    pos = g.int(0, len(seg) + 1)
    t = np.concatenate([g.ints(0, _GOLDEN_A, (g.int(0, 40),)), seg[:pos], g.ints(0, _GOLDEN_A, (g.int(0, 15),)),
                        seg[pos:], g.ints(0, _GOLDEN_A, (g.int(0, 40),))])
    return (t if len(t) else g.ints(0, _GOLDEN_A, (1,))).astype(np.uint8)


def golden_cases(seed: int = GOLDEN_SEED) -> List[GoldenCase]:
    """The deterministic inputs of the golden digest, integer only (swkern's splitmix64 stream: the same bytes on every
    platform and numpy version). The profiles come from a synthetic symmetric 24 x 24 score table (diagonal 4..11,
    elsewhere -4..2), not from biotite's BLOSUM62, so the digest needs only numpy:

    * ``random``: 48 queries of 1..300 codes against 4 random targets each, the gap settings in turn;
    * ``related``: 48 queries against 4 mutated copies of one of their segments (substitutions, indels, flanks);
    * ``bias``: 16 related cases with per-position offsets -3..3 (the shape of the composition-bias correction);
    * ``longgap``: a 60-code insertion in the target, then in the query, at 11/1, 20/5 and 3/3;
    * ``wide``: scores beyond the 16-bit limit (a 1,200-code query with +25 offsets against itself and a mutated
      copy, at 11/1 and 1/4), a profile beyond +-16,384 and one with a single code below -32,768 (32-bit lanes from
      the start), scores just below and just above the limit, 32767 - max(s), and a column best that reaches it to
      the unit;
    * ``carry``: a lazy-F loop long enough to wrap 16 bits without the clamp of F at -open, at 1000/1000 (n = 1024
      for 8 lanes, 2048 for 16);
    * ``edge``: n = 1, m = 1, an empty target, and a profile with no positive score (empty alignments).
    """
    g = swkern._SplitMix64(seed)
    A = _GOLDEN_A
    upper = np.triu(g.ints(-4, 3, (A, A)), 1)
    table = (upper + upper.T + np.diag(g.ints(4, 12, (A,)))).astype(np.int32)

    def codes(n):
        return g.ints(0, A, (n,)).astype(np.uint8)

    def case(kind, q, targets, gap, offsets=None):
        prof = table[q.astype(np.intp)]
        if offsets is not None:
            prof = prof + np.asarray(offsets, dtype=np.int32)[:, None]
        return GoldenCase(kind, q, np.ascontiguousarray(prof, dtype=np.int32), tuple(targets), gap[0], gap[1])

    out: List[GoldenCase] = []
    for c in range(48):
        q = codes(g.int(1, 301))
        out.append(case("random", q, [codes(g.int(1, 301)) for _ in range(4)], _GOLDEN_GAPS[c % len(_GOLDEN_GAPS)]))
    for c in range(64):
        q = codes(g.int(20, 301))
        targets = []
        for _ in range(4):
            a, b = sorted((g.int(0, len(q)), g.int(0, len(q))))
            targets.append(_golden_mutant(g, q[a:b + 1]))
        if c < 48:
            out.append(case("related", q, targets, _GOLDEN_GAPS[c % len(_GOLDEN_GAPS)]))
        else:
            out.append(case("bias", q, targets, (11, 1), g.ints(-3, 4, (len(q),))))
    core, insert = codes(150), codes(60)
    gapped = np.concatenate([core[:75], insert, core[75:]])
    for gap in ((11, 1), (20, 5), (3, 3)):
        out.append(case("longgap", core, [gapped, np.concatenate([gapped[:120], gapped[150:]])], gap))
        out.append(case("longgap", gapped, [core], gap))
    q = codes(1200)
    mutant = _golden_mutant(g, q)
    for gap in ((11, 1), (1, 4)):
        out.append(case("wide", q, [q, mutant, q[:40]], gap, np.full(len(q), 25)))
    big = codes(200)
    out.append(GoldenCase("wide", big, np.ascontiguousarray(table[big.astype(np.intp)] * 2000, dtype=np.int32),
                          (big, _golden_mutant(g, big)), 11, 1))
    # one code scores -40,000 on every row and no score is above 11: 32-bit lanes from the start, although the score
    # (750, the query against itself, which holds that code 4 times) is far below the 16-bit limit; in 16-bit lanes
    # the -40,000 would wrap to +25,536
    floor = table[q[:120].astype(np.intp)]
    floor[:, q[60]] = -40000
    out.append(GoldenCase("wide", q[:120], floor, (q[:120],), 11, 1))
    # every cell scores 11, so every score is a multiple of 11 and none lands on 32767: 2,977 columns end at
    # 32,747, below the 16-bit limit 32767 - 11, and 2,978 at 32,758, above it
    flat = codes(3000)
    out.append(GoldenCase("wide", flat, np.full((3000, A), 11, dtype=np.int32), (flat, flat[:2978], flat[:2977]),
                          11, 1))
    # row 1 scores 10 and every other row 11, and the target starts with code 1, which scores only on row 1: column
    # j's best is 11 j - 1, so column 2,978's 32,757 reaches the limit 32767 - 11 one column before 32,768 would wrap
    lim = np.full((2979, A), 11, dtype=np.int32)
    lim[:, 1] = -100
    lim[0] = 10
    lim_t = np.zeros(2979, dtype=np.uint8)
    lim_t[0] = 1
    out.append(GoldenCase("wide", np.zeros(2979, np.uint8), lim, (lim_t,), 11, 1))
    # a +200 diagonal of 162 cells (32,400, codes 1..22 in turn) ends on row 384, the last row of lane 2 when a lane
    # holds 128 rows (8 lanes at n = 1024, 16 at n = 2048), and every row below it scores -16,000: at 1000/1000 the
    # lazy-F loop of that column runs for over 32 steps (the carry into lane 3 keeps improving G2). With F clamped at
    # -open no lane goes below -2,000; unclamped, lane 0's F falls by 1,000 a step from -1,000 and wraps to +32,536,
    # a false best in the last column (code 23, scoring 0)
    for n in (1024, 2048):
        carry = np.full((n, A), -200, dtype=np.int32)
        carry[384:] = -16000
        carry[:, 23] = 0
        carry_t = np.zeros(202, dtype=np.uint8)
        for d in range(162):
            carry_t[39 + d] = 1 + d % 22
            carry[222 + d, 1 + d % 22] = 200
        carry_t[201] = 23
        out.append(GoldenCase("carry", np.zeros(n, np.uint8), carry, (carry_t,), 1000, 1000))
    out.append(case("edge", codes(1), [codes(1), codes(50), np.zeros(0, np.uint8)], (11, 1)))
    out.append(case("edge", codes(50), [codes(1)], (5, 2)))
    neg = codes(40)
    out.append(GoldenCase("edge", neg, np.full((40, A), -1, dtype=np.int32), (codes(40),), 11, 1))
    return out


def golden_digest_update(h, case: GoldenCase, mode: int, out: np.ndarray, ops: Optional[bytes],
                         ops_off: Optional[np.ndarray]):
    """One call's record: the inputs (<i4 n, A, K, gap_open, gap_extend, traceback mode; query codes; profile <i4;
    per target <i4 length and codes), then ``out`` as <i4 and, with a traceback, ``ops_off`` as <i8 and the paths."""
    h.update(struct.pack("<6i", len(case.qcodes), case.qprof.shape[1], len(case.targets), case.gap_open,
                         case.gap_extend, mode))
    h.update(case.qcodes.tobytes())
    h.update(np.ascontiguousarray(case.qprof, dtype="<i4").tobytes())
    for t in case.targets:
        h.update(struct.pack("<i", len(t)) + t.tobytes())
    h.update(np.ascontiguousarray(out, dtype="<i4").tobytes())
    if mode:
        h.update(np.ascontiguousarray(ops_off, dtype="<i8").tobytes())
        h.update(ops)


def golden_digest_new():
    return hashlib.sha256(b"fmswi/1 golden v1\0")


def run_golden(case: GoldenCase, mode: int, kernel: Optional[IntKernel] = None) -> BatchResult:
    """One golden case with traceback ``mode`` (0 score only, 1 'full', 2 'prefix'); paths with a traceback."""
    lens = np.fromiter((len(t) for t in case.targets), dtype=np.int32, count=len(case.targets))
    off = np.zeros(len(lens), dtype=np.int64)
    if len(lens) > 1:
        np.cumsum(lens[:-1], out=off[1:])
    trace = {1: "full", 2: "prefix"}.get(mode, DEFAULT_TRACE)
    return align_batch(case.qcodes, case.qprof, np.concatenate(case.targets), off, lens, case.gap_open,
                       case.gap_extend, traceback=mode != 0, ops=mode != 0, trace=trace, kernel=kernel)


# sha256 of the golden records (golden_digest_update) of golden_cases() in traceback modes 0, 1 and 2, case by case.
# Computed 2026-10-31 on an Apple M1 Max: Apple clang 21 neon and generic and GCC 14.2 neon and generic (arm64), Apple
# clang sse2, avx2 and generic (x86_64 under Rosetta) all give this value; every conforming build reproduces it bit
# for bit (the kernel's values are exact integers).
GOLDEN_SHA256 = "77a7ccf5eac920023fd947758b44abff3feada5be33cb6e91dcf09f335540cab"


def selftest(kernel: Optional[swkern.Kernel] = None) -> Dict[str, object]:
    """The integer DP's part of :func:`swkern.selftest`: the golden digest of ``kernel``'s ``fmswi_align`` (default:
    the dispatched library), and the agreement of its traceback modes (1 and 2 identical; 0 the same score, end cell
    and WIDE, every other field 0). Raises :class:`~foldmatch.search.swkern.KernelError` on failure."""
    t0 = time.perf_counter()
    ik = load() if kernel is None else IntKernel(kernel)
    head = [SCORE, QEND, TEND, WIDE]
    h = golden_digest_new()
    cases = golden_cases()
    for case in cases:
        runs = [run_golden(case, mode, ik) for mode in (0, 1, 2)]
        for mode, r in enumerate(runs):
            golden_digest_update(h, case, mode, r.out, r.ops, r.ops_off)
        score_only, full, prefix = runs
        if not (np.array_equal(full.out, prefix.out) and full.ops == prefix.ops
                and np.array_equal(score_only.out[:, head], full.out[:, head])
                and not np.delete(score_only.out, head, axis=1).any()):
            raise swkern.KernelError(f"fmswi selftest: the traceback modes disagree on a {case.kind!r} case "
                                     f"(n={len(case.qcodes)}, gaps {case.gap_open}/{case.gap_extend}; {ik})")
    digest = h.hexdigest()
    if digest != GOLDEN_SHA256:
        raise swkern.KernelError(f"fmswi selftest: golden digest {digest} != {GOLDEN_SHA256} ({ik})")
    report = dict(golden_sha256=digest, cases=len(cases), pairs=sum(len(c.targets) for c in cases),
                  seconds=round(time.perf_counter() - t0, 3))
    logger.info("fmswi selftest passed: %s", report)
    return report
