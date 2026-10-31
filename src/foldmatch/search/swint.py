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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from foldmatch import _native

from . import swkern

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
