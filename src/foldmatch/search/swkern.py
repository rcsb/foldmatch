"""ctypes wrapper of the native Smith-Waterman kernel ``fmsw`` (ABI 1, DP id ``fmsw/1``) for esm-codes Stage 2.

numpy + ctypes only: this module never imports torch (S6: the similarity is an fp32 numpy GEMM on the CPU, never
torch, never a GPU, never TF32 / autocast). Method: ``docs/esm-codes-method.md`` (S5-S9, S16).

Pipeline for one query against K candidates (codes: uint8 [rows, 8], packbits MSB-first)::

    qp = query_prep(q16)                                   # S5, fixed order in C (float64 norm)
    res = score_fwd_rev(qp, codes, offsets, params)        # round 1: S_f, S_r (score only)
    aln = align(qp, codes, offsets, params)                # round 2: forward traceback (survivors)

Each candidate's similarity is its own GEMM ``numpy.matmul(qp, D_t.T, out=buf)`` (S6) into a 64-byte-aligned,
C-contiguous ``[n, m_t]`` slice of a block buffer, so its bits depend only on the pair and the BLAS build, never on
the blocking, the candidate order or the round. ``S_r`` reads the same matrix through a row-reversed view (S9).
With the fixed-order reference similarity (``reference_sim=True``, ``fmsw_sim_ref``) the whole chain from q fp16 is
bit-identical on every build: that is the DP golden hash checked by :func:`selftest` and the tests.

Dispatch (:func:`load`): x86_64 loads ``_fmsw_sse2`` and switches to ``_fmsw_avx2`` when ``fmsw_cpu_has_avx2()``;
arm64 loads ``_fmsw_neon``; anything else ``_fmsw_generic``. ``FOLDMATCH_SW_ISA`` overrides the choice and is
validated (``avx2`` on a CPU without AVX2 is refused rather than left to SIGILL). The library must report ABI 1 and
the source sha256 of the installed kernel source (a stale developer build is refused).

BLAS threads: multi-core speed comes from the Stage-2 worker pool; workers keep BLAS single-threaded
(``OMP/OPENBLAS/MKL/VECLIB_*_NUM_THREADS=1``), which keeps reruns bit-identical (S16). Any thread count stays
within S16.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import logging
import os
import re
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from foldmatch import _native

logger = logging.getLogger(__name__)

ABI_VERSION = 1
DP_ID = "fmsw/1"
ISA_ENV = "FOLDMATCH_SW_ISA"
DEFAULT_BLOCK_FLOATS = 1 << 17          # 512 KB of similarity per block (P0.4 sweep; tuned in P7)
DEFAULT_MAX_PAIR_CELLS = 1 << 28        # about 1 GiB of similarity: larger pairs are skipped with a warning (D15)
ALIGN_BYTES = 64
_ALIGN_FLOATS = ALIGN_BYTES // 4
DIM = 64                                # code bits / query encoding width
CODE_BYTES = 8

STATUS_OK, STATUS_WORKSPACE, STATUS_ARGS = 0, 1, 2
_STATUS_TEXT = {1: "workspace or buffer too small", 2: "bad arguments"}


class KernelError(RuntimeError):
    """The native kernel cannot be used (missing, stale, wrong ABI, refused ISA) or a call failed."""


# =====================================================================================================================
# parameters
# =====================================================================================================================
@dataclass(frozen=True)
class DPParams:
    """Smith-Waterman parameters: cell score ``z - offset``, a gap of length L costs ``gap_open + (L-1) gap_extend``.

    The kernel computes in float32, so every value must be exactly representable in float32 (no silent rounding)."""
    offset: float
    gap_open: float
    gap_extend: float

    def __post_init__(self):
        for name in ("offset", "gap_open", "gap_extend"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or float(np.float32(v)) != v:
                raise ValueError(f"DPParams.{name}={v!r} must be finite and exactly representable in float32")
            object.__setattr__(self, name, v)
        if not self.gap_open >= self.gap_extend >= 0.0:
            raise ValueError("DPParams needs gap_open >= gap_extend >= 0")


SETTING_D = DPParams(offset=1.0, gap_open=5.0, gap_extend=0.5)   # esm-codes-v1 (S8); also the golden-hash setting


# =====================================================================================================================
# library loading and dispatch
# =====================================================================================================================
_vp = C.c_void_p


class Kernel:
    """One loaded ISA library. Use :func:`load` (dispatch, caching) rather than constructing it directly."""

    def __init__(self, path: Path, isa: str):
        self.path = Path(path).resolve()
        try:
            lib = C.CDLL(str(self.path))
        except OSError as e:
            raise KernelError(f"cannot load the native kernel {self.path}: {e}; {_native.rebuild_hint()}") from e
        try:
            lib.fmsw_abi_version.restype = C.c_int
            lib.fmsw_abi_version.argtypes = []
            abi = int(lib.fmsw_abi_version())
        except AttributeError as e:
            raise KernelError(f"{self.path} is not an fmsw kernel library ({e}); {_native.rebuild_hint()}") from e
        if abi != ABI_VERSION:
            raise KernelError(f"native kernel {self.path} has ABI {abi}, this FoldMatch needs ABI {ABI_VERSION}; "
                              f"{_native.rebuild_hint()}")
        for name in ("fmsw_isa", "fmsw_dp_id", "fmsw_source_sha256", "fmsw_compiler"):
            getattr(lib, name).restype = C.c_char_p
            getattr(lib, name).argtypes = []
        lib.fmsw_cpu_has_avx2.restype = C.c_int
        lib.fmsw_cpu_has_avx2.argtypes = []
        lib.fmsw_query_prep.restype = C.c_int
        lib.fmsw_query_prep.argtypes = [_vp, C.c_int, _vp]
        lib.fmsw_sim_ref.restype = C.c_int
        lib.fmsw_sim_ref.argtypes = [_vp, C.c_int, _vp, C.c_int, _vp]
        lib.fmsw_score_fwd_rev.restype = C.c_int
        lib.fmsw_score_fwd_rev.argtypes = [_vp, C.c_int, _vp, _vp, C.c_int, C.c_float, C.c_float, C.c_float,
                                           _vp, _vp, _vp, C.c_size_t]
        lib.fmsw_align.restype = C.c_int
        lib.fmsw_align.argtypes = [_vp, C.c_int, _vp, _vp, C.c_int, C.c_float, C.c_float, C.c_float, C.c_int,
                                   _vp, _vp, _vp, C.c_size_t, _vp, _vp, C.c_size_t, _vp, C.c_size_t]
        lib.fmsw_ws_floats.restype = C.c_size_t
        lib.fmsw_ws_floats.argtypes = [C.c_int, C.c_int64]
        lib.fmsw_flag_bytes.restype = C.c_size_t
        lib.fmsw_flag_bytes.argtypes = [C.c_int, C.c_int64]
        self.lib = lib
        self.abi = abi
        self.isa = lib.fmsw_isa().decode()
        if self.isa != isa:
            raise KernelError(f"{self.path} reports ISA {self.isa!r}, expected {isa!r}; {_native.rebuild_hint()}")
        self.dp_id = lib.fmsw_dp_id().decode()
        if self.dp_id != DP_ID:
            raise KernelError(f"{self.path} implements DP {self.dp_id!r}, expected {DP_ID!r}")
        self.source_sha256 = lib.fmsw_source_sha256().decode()
        self.compiler = lib.fmsw_compiler().decode()

    def info(self) -> Dict[str, object]:
        return dict(isa=self.isa, abi=self.abi, source_sha256=self.source_sha256, path=str(self.path),
                    dp_id=self.dp_id, compiler=self.compiler)

    def cpu_has_avx2(self) -> bool:
        return bool(self.lib.fmsw_cpu_has_avx2())

    def ws_floats(self, n: int, mmax: int) -> int:
        return int(self.lib.fmsw_ws_floats(int(n), int(mmax)))

    def flag_bytes(self, n: int, mmax: int) -> int:
        return int(self.lib.fmsw_flag_bytes(int(n), int(mmax)))

    def __repr__(self):
        return f"Kernel(isa={self.isa!r}, path={str(self.path)!r})"


_KERNELS: Dict[tuple, Kernel] = {}
_LOAD_LOCK = threading.Lock()
_LOGGED_PATHS: set = set()


def _open(isa: str, search_dirs: Optional[Tuple[Path, ...]], check_source: bool) -> Kernel:
    path = _native.find_library(isa, search_dirs)
    if path is None:
        where = ", ".join(str(d) for d in (search_dirs or (_native.NATIVE_DIR,)))
        raise KernelError(f"native kernel library {_native.library_filename(isa)} not found in {where}; "
                          f"{_native.rebuild_hint(missing=True)}")
    kernel = Kernel(path, isa)
    if check_source:
        tree = _native.source_sha256()
        if tree is not None and kernel.source_sha256 != tree:
            raise KernelError(f"stale native kernel {path}: built from source sha256 {kernel.source_sha256}, the "
                              f"installed source is {tree}; {_native.rebuild_hint()}")
    return kernel


def _select(requested: Optional[str], machine: str, search_dirs, check_source: bool) -> Kernel:
    machine = _native.normalized_machine(machine)
    allowed = _native.machine_isas(machine)
    if requested is not None:
        isa = requested.strip().lower()
        if isa not in _native.ISAS:
            raise KernelError(f"{ISA_ENV}={requested!r} is not a kernel ISA; expected one of "
                              f"{', '.join(_native.ISAS)}")
        if isa not in allowed:
            raise KernelError(f"{ISA_ENV}={isa} cannot run on this {machine or 'unknown'} machine; available: "
                              f"{', '.join(allowed)}")
        if isa == "avx2":
            probe = _open("sse2", search_dirs, check_source)          # the probe runs in the baseline library
            if not probe.cpu_has_avx2():
                raise KernelError(f"{ISA_ENV}=avx2 refused: this CPU or OS does not support AVX2 (the avx2 library "
                                  "would stop with an illegal instruction); unset it to use sse2")
        return _open(isa, search_dirs, check_source)
    if machine == "x86_64":
        base = _open("sse2", search_dirs, check_source)
        return _open("avx2", search_dirs, check_source) if base.cpu_has_avx2() else base
    if machine == "arm64":
        return _open("neon", search_dirs, check_source)
    return _open("generic", search_dirs, check_source)


def load(isa: Optional[str] = None, *, search_dirs: Optional[Iterable[Path]] = None, machine: Optional[str] = None,
         check_source: bool = True) -> Kernel:
    """The kernel for this machine (cached). ``isa`` (else ``$FOLDMATCH_SW_ISA``) forces an ISA after validation.
    ``search_dirs`` (default: the installed package directory) and ``machine`` exist for tests.
    Raises :class:`KernelError` with the rebuild command when no usable library is found."""
    requested = isa if isa is not None else (os.environ.get(ISA_ENV) or "").strip() or None
    m = _native.normalized_machine(machine)
    dirs = tuple(Path(d) for d in search_dirs) if search_dirs is not None else None
    key = (requested, m, dirs, check_source)
    with _LOAD_LOCK:
        kernel = _KERNELS.get(key)
        if kernel is None:
            kernel = _select(requested, m, dirs, check_source)
            _KERNELS[key] = kernel
            if str(kernel.path) not in _LOGGED_PATHS:
                _LOGGED_PATHS.add(str(kernel.path))
                logger.info("fmsw kernel: %s", kernel.info())
    return kernel


def _check_status(status: int, what: str):
    if status != STATUS_OK:
        raise KernelError(f"{what} returned status {status} ({_STATUS_TEXT.get(status, 'unknown')})")


# =====================================================================================================================
# buffers
# =====================================================================================================================
def aligned_empty(n_floats: int) -> np.ndarray:
    """Uninitialised float32 [n_floats] whose data starts on a 64-byte boundary, also when ``n_floats`` is 0 (an
    empty numpy slice keeps its base's address, so the empty buffer is cut from an aligned one-float view)."""
    n_floats = int(n_floats)
    raw = np.empty(n_floats + _ALIGN_FLOATS, np.float32)
    off = (-raw.ctypes.data % ALIGN_BYTES) // 4
    return raw[off:off + max(n_floats, 1)][:n_floats]


def _pad(cells: int) -> int:
    return -(-int(cells) // _ALIGN_FLOATS) * _ALIGN_FLOATS


class Workspace:
    """Reusable buffers of one process (one per thread via :func:`default_workspace`); grown on demand, never shrunk.
    Not thread-safe: never share one between threads."""

    def __init__(self):
        self._ws = np.empty(0, np.float32)
        self._flags = np.empty(0, np.uint8)
        self._sim = aligned_empty(0)
        self._dec = np.empty(0, np.float32)
        self._path = np.empty(0, np.int32)
        self._poff = np.empty(0, np.int64)

    def kernel_floats(self, need: int) -> np.ndarray:
        if self._ws.size < need:
            self._ws = np.empty(int(need), np.float32)
        return self._ws

    def flags(self, need: int) -> np.ndarray:
        need = max(int(need), 1)
        if self._flags.size < need:
            self._flags = np.empty(need, np.uint8)
        return self._flags

    def sim(self, n_floats: int) -> np.ndarray:
        """64-byte-aligned float32 buffer of at least ``n_floats`` floats."""
        if self._sim.size < n_floats:
            self._sim = aligned_empty(max(int(n_floats), _ALIGN_FLOATS))
        return self._sim

    def decoded(self, rows: int) -> np.ndarray:
        """C-contiguous float32 [rows, 64] view of the decode buffer."""
        if self._dec.size < rows * DIM:
            self._dec = np.empty(int(rows) * DIM, np.float32)
        return self._dec[:rows * DIM].reshape(rows, DIM)

    def path_entries(self, entries: int) -> np.ndarray:
        """int32 buffer of at least ``2 x entries`` values (the traceback's (i, j) path entries)."""
        need = 2 * max(int(entries), 1)
        if self._path.size < need:
            self._path = np.empty(need, np.int32)
        return self._path

    def path_offsets(self, count: int) -> np.ndarray:
        """int64 buffer of at least ``count`` values (a kernel call's path offsets)."""
        if self._poff.size < count:
            self._poff = np.empty(int(count), np.int64)
        return self._poff

    def nbytes(self) -> int:
        return int(self._ws.nbytes + self._flags.nbytes + self._sim.nbytes + self._dec.nbytes + self._path.nbytes
                   + self._poff.nbytes)


_TLS = threading.local()


def default_workspace() -> Workspace:
    ws = getattr(_TLS, "workspace", None)
    if ws is None:
        ws = _TLS.workspace = Workspace()
    return ws


# =====================================================================================================================
# S5 query prep, codes, S6 similarity
# =====================================================================================================================
def query_prep(q16: np.ndarray, kernel: Optional[Kernel] = None) -> np.ndarray:
    """S5: q fp16 [n, 64] -> qp float32 [n, 64] (C, float64 norm in ascending k, zero rows stay zero).
    NaN or Inf raises ValueError."""
    q16 = np.asarray(q16)
    if q16.dtype != np.float16 or q16.ndim != 2 or q16.shape[1] != DIM:
        raise TypeError(f"query encodings must be float16 [n, {DIM}], got {q16.dtype} {q16.shape}")
    k = kernel or load()
    q = np.ascontiguousarray(q16, dtype=np.float32)         # exact: every fp16 is an fp32
    qp = np.empty_like(q)
    st = k.lib.fmsw_query_prep(q.ctypes.data, len(q), qp.ctypes.data)
    if st == STATUS_ARGS:
        raise ValueError("query encoding contains NaN or Inf (S5)")
    _check_status(st, "fmsw_query_prep")
    return qp


def _require_qp(qp: np.ndarray) -> np.ndarray:
    if not isinstance(qp, np.ndarray) or qp.dtype != np.float32 or qp.ndim != 2 or qp.shape[1] != DIM:
        raise TypeError(f"qp must be a float32 [n, {DIM}] numpy array (query_prep output)")
    if not qp.flags.c_contiguous:
        raise ValueError("qp must be C-contiguous (S6 operand layout)")
    if qp.shape[0] < 1:
        raise ValueError("empty query")
    return qp


def _as_codes(codes) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != CODE_BYTES:
        raise TypeError(f"codes must be uint8 [rows, {CODE_BYTES}], got {codes.dtype} {codes.shape}")
    return codes


# +-1 table: _LUT[byte] = 2 * unpackbits(byte) - 1 (MSB first), so _LUT[codes] is decode(codes) exactly
_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32) * 2 - 1


def decode_codes(codes: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """codes uint8 [m, 8] -> D float32 [m, 64] with D = 2 bit - 1 (bit k = packbits MSB-first), C-contiguous."""
    codes = _as_codes(codes)
    m = len(codes)
    if out is None:
        out = np.empty((m, DIM), np.float32)
    elif out.dtype != np.float32 or out.shape != (m, DIM) or not out.flags.c_contiguous:
        raise ValueError(f"decode buffer must be C-contiguous float32 [{m}, {DIM}]")
    np.take(_LUT, codes, axis=0, out=out.reshape(m, CODE_BYTES, 8), mode="clip")
    return out


def _require_sim_out(out: np.ndarray, n: int, m: int):
    if not isinstance(out, np.ndarray) or out.dtype != np.float32:
        raise TypeError("similarity output must be a float32 numpy array")
    if out.shape != (n, m) or not out.flags.c_contiguous:
        raise ValueError(f"similarity output must be C-contiguous [{n}, {m}] (ld = m), got {out.shape}")
    if out.ctypes.data % ALIGN_BYTES:
        raise ValueError("similarity output must start on a 64-byte boundary (S6)")


def similarity(qp: np.ndarray, codes_t: np.ndarray, out: np.ndarray, *,
               decoded: Optional[np.ndarray] = None) -> np.ndarray:
    """S6 for one pair: ``out = numpy.matmul(qp, D_t.T)`` as one fp32 GEMM (``D_t.T`` a transposed view, never
    materialised) into a 64-byte-aligned, C-contiguous [n, m] buffer. Never torch, never a GPU."""
    _require_qp(qp)
    D = decode_codes(codes_t, out=decoded)
    _require_sim_out(out, qp.shape[0], D.shape[0])
    if D.shape[0]:
        np.matmul(qp, D.T, out=out)
    return out


def sim_ref(qp: np.ndarray, codes_t: np.ndarray, out: Optional[np.ndarray] = None,
            kernel: Optional[Kernel] = None) -> np.ndarray:
    """The fixed-order test reference similarity ``fmsw_sim_ref`` (golden hash only; not the production S6)."""
    _require_qp(qp)
    codes_t = np.ascontiguousarray(_as_codes(codes_t))
    n, m = qp.shape[0], len(codes_t)
    if out is None:
        out = aligned_empty(n * m).reshape(n, m)
    _require_sim_out(out, n, m)
    k = kernel or load()
    _check_status(k.lib.fmsw_sim_ref(qp.ctypes.data, n, codes_t.ctypes.data, m, out.ctypes.data), "fmsw_sim_ref")
    return out


# =====================================================================================================================
# candidate lists, blocks and the score / align drivers
# =====================================================================================================================
_SKIPPED = {"pairs": 0, "warned": False}
_SKIP_LOCK = threading.Lock()


def skipped_pair_count() -> int:
    """Pairs skipped by ``max_pair_cells`` in this process so far."""
    return _SKIPPED["pairs"]


def reset_skipped_pairs():
    with _SKIP_LOCK:
        _SKIPPED.update(pairs=0, warned=False)


def _note_skipped(count: int, n: int, max_pair_cells: int):
    if not count:
        return
    with _SKIP_LOCK:
        _SKIPPED["pairs"] += int(count)
        warn = not _SKIPPED["warned"]
        _SKIPPED["warned"] = True
    if warn:
        logger.warning("esm-codes Stage 2: skipping pairs whose similarity exceeds max_pair_cells=%d cells "
                       "(first: query length %d); they get no score. Further skips are counted, not logged.",
                       max_pair_cells, n)


def _candidates(codes, offsets, lengths) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(codes, starts int64 [K], lengths int64 [K]). ``offsets`` is either K + 1 boundaries (candidate k =
    codes[offsets[k]:offsets[k+1]]) or, with ``lengths``, K start rows."""
    codes = _as_codes(codes)
    off = np.asarray(offsets, dtype=np.int64).ravel()
    if lengths is None:
        if off.size < 1:
            raise ValueError("offsets needs K + 1 >= 1 entries")
        starts, lens = off[:-1], np.diff(off)
    else:
        starts, lens = off, np.asarray(lengths, dtype=np.int64).ravel()
        if starts.shape != lens.shape:
            raise ValueError("starts and lengths differ in size")
    if lens.size and (lens.min() < 0 or starts.min() < 0 or int((starts + lens).max()) > len(codes)):
        raise ValueError("candidate rows out of range of codes")
    return codes, starts, lens


def plan_blocks(n: int, lengths: Sequence[int], skipped: Sequence[bool], block_floats: int) -> List[Tuple[int, int]]:
    """Candidate ranges [k0, k1) whose 64-byte-padded per-pair matrices fit in ``block_floats`` floats (greedy, in
    order). A candidate that alone exceeds the budget is its own block; a skipped candidate ends the current block.
    Every pair gets the same layout ([n, m] at a 64-byte boundary) in any block, so blocking never changes bits."""
    if block_floats < 1:
        raise ValueError("block_floats must be >= 1")
    blocks, k0, acc = [], None, 0
    for k, (m, skip) in enumerate(zip(lengths, skipped)):
        if skip:
            if k0 is not None:
                blocks.append((k0, k))
                k0 = None
            continue
        p = _pad(n * int(m))
        if k0 is not None and acc + p > block_floats:
            blocks.append((k0, k))
            k0 = None
        if k0 is None:
            k0, acc = k, 0
        acc += p
    if k0 is not None:
        blocks.append((k0, len(lengths)))
    return blocks


class _Layout:
    """Every block of one :func:`score_fwd_rev` / :func:`align` call, laid out once per call (vectorised) instead of
    block by block: the blocks of :func:`plan_blocks`; per pair its float offset from its block's start
    (``pair_off`` int64 [K]: the 64-byte-padded [n, m] matrices of a block one after another, exactly the layout of
    :class:`_Block`) and its length as int32 (``m32``); per block its buffer size (``floats``), its decoded rows
    (``rows``) and whether its targets are consecutive rows of the codes (``contiguous``); ``mmax`` the longest
    scored pair. Python lists of the per-pair values save numpy scalar access in the per-pair loops."""
    __slots__ = ("blocks", "pair_off", "m32", "floats", "rows", "contiguous", "mmax", "starts_l", "lens_l", "off_l")

    def __init__(self, n: int, starts: np.ndarray, lens: np.ndarray, skipped: np.ndarray, block_floats: int):
        K = len(lens)
        self.blocks = plan_blocks(n, lens.tolist(), skipped.tolist(), block_floats)
        self.m32 = lens.astype(np.int32)
        self.starts_l, self.lens_l = starts.tolist(), lens.tolist()
        if not self.blocks:
            self.pair_off = np.zeros(K, np.int64)
            self.floats = self.rows = self.contiguous = []
            self.off_l, self.mmax = self.pair_off.tolist(), 0
            return
        padded = -(-(n * lens) // _ALIGN_FLOATS) * _ALIGN_FLOATS
        csum = np.zeros(K + 1, np.int64)
        np.cumsum(padded, out=csum[1:])
        rsum = np.zeros(K + 1, np.int64)
        np.cumsum(lens, out=rsum[1:])
        b0 = np.fromiter((b[0] for b in self.blocks), np.int64, len(self.blocks))
        b1 = np.fromiter((b[1] for b in self.blocks), np.int64, len(self.blocks))
        first = np.zeros(K, np.int64)                  # each pair's block start (blocks are increasing ranges)
        first[b0] = b0
        np.maximum.accumulate(first, out=first)
        self.pair_off = csum[:-1] - csum[first]        # (skipped pairs belong to no block; their value is unused)
        self.off_l = self.pair_off.tolist()
        self.floats = (csum[b1] - csum[b0]).tolist()
        self.rows = (rsum[b1] - rsum[b0]).tolist()
        breaks = np.zeros(K, np.int64)
        if K > 1:
            np.cumsum(starts[1:] != starts[:-1] + lens[:-1], out=breaks[1:])
        self.contiguous = (breaks[b1 - 1] == breaks[b0]).tolist()
        self.mmax = int(lens[~skipped].max())


def _fill_block(layout: _Layout, b: int, qp: np.ndarray, codes: np.ndarray, D: np.ndarray, D3: np.ndarray,
                sim: np.ndarray) -> None:
    """Block ``b`` of ``layout``: its targets decoded into ``D`` (``D3`` the [rows, 8, 8] view of the same buffer)
    and each pair's similarity as its own fp32 GEMM (S6) into ``sim`` at the pair's offset -- the same decode, the
    same GEMM operands and the same output layout as :class:`_Block`."""
    k0, k1 = layout.blocks[b]
    n = qp.shape[0]
    starts_l, lens_l, off_l = layout.starts_l, layout.lens_l, layout.off_l
    r = 0
    if layout.contiguous[b]:
        rows = layout.rows[b]
        if rows:
            s0 = starts_l[k0]
            np.take(_LUT, codes[s0:s0 + rows], axis=0, out=D3[:rows], mode="clip")
        for k in range(k0, k1):
            m = lens_l[k]
            if m:
                o = off_l[k]
                np.matmul(qp, D[r:r + m].T, out=sim[o:o + n * m].reshape(n, m))      # S6: one fp32 GEMM per pair
                r += m
        return
    for k in range(k0, k1):
        m = lens_l[k]
        if m:
            s, o = starts_l[k], off_l[k]
            np.take(_LUT, codes[s:s + m], axis=0, out=D3[r:r + m], mode="clip")
            np.matmul(qp, D[r:r + m].T, out=sim[o:o + n * m].reshape(n, m))          # S6: one fp32 GEMM per pair
            r += m


class _Block:
    """One block's similarity matrices in the workspace buffer, filled by per-pair GEMMs (or ``fmsw_sim_ref``).
    Production calls lay out all their blocks at once (:class:`_Layout`, :func:`_fill_block`: the same matrices at
    the same offsets); this class remains for ``reference_sim`` and for the benchmark's per-stage timings."""
    __slots__ = ("sim", "pair_off", "m", "K")

    def __init__(self, kernel: Kernel, ws: Workspace, qp: np.ndarray, codes: np.ndarray, starts: np.ndarray,
                 lens: np.ndarray, k0: int, k1: int, reference_sim: bool):
        n = qp.shape[0]
        ms = lens[k0:k1]
        padded = -(-(n * ms) // _ALIGN_FLOATS) * _ALIGN_FLOATS
        pair_off = np.zeros(k1 - k0, np.int64)
        if k1 - k0 > 1:
            np.cumsum(padded[:-1], out=pair_off[1:])
        total = int(padded.sum())
        sim = ws.sim(total)
        if sim.ctypes.data % ALIGN_BYTES:                       # by construction; S6 layout guard
            raise KernelError("similarity block buffer is not 64-byte aligned")
        self.sim, self.pair_off, self.m, self.K = sim, pair_off, ms.astype(np.int32), k1 - k0
        st, ln = starts[k0:k1], ms
        if reference_sim:
            qptr = qp.ctypes.data
            for s, m, o in zip(st.tolist(), ln.tolist(), pair_off.tolist()):
                if m:
                    c = np.ascontiguousarray(codes[s:s + m])
                    _check_status(kernel.lib.fmsw_sim_ref(qptr, n, c.ctypes.data, m, sim[o:].ctypes.data),
                                  "fmsw_sim_ref")
            return
        rows = int(ln.sum())
        D = ws.decoded(rows)
        contiguous = bool(np.all(st[1:] == st[:-1] + ln[:-1])) if len(st) > 1 else True
        if contiguous and rows:
            s0 = int(st[0])
            np.take(_LUT, codes[s0:s0 + rows], axis=0, out=D.reshape(rows, CODE_BYTES, 8), mode="clip")
        r = 0
        for s, m, o in zip(st.tolist(), ln.tolist(), pair_off.tolist()):
            if not m:
                continue
            Dk = D[r:r + m]
            if not contiguous:
                np.take(_LUT, codes[s:s + m], axis=0, out=Dk.reshape(m, CODE_BYTES, 8), mode="clip")
            np.matmul(qp, Dk.T, out=sim[o:o + n * m].reshape(n, m))      # S6: one fp32 GEMM per pair
            r += m


@dataclass
class ScoreResult:
    """Round-1 scores: ``s_fwd`` (S_f) and ``s_rev`` (S_r) float32 [K]; skipped pairs are NaN in both."""
    s_fwd: np.ndarray
    s_rev: np.ndarray
    skipped: np.ndarray


@dataclass
class AlignResult:
    """Forward DP: ``scores`` float32 [K] (S_f), ``ends`` int32 [K, 2] (1-based best cell, (0, 0) = empty),
    ``path`` int32 [P, 2] (query i, target j; 0-based, -1 = gap; forward order as the kernel wrote it), candidate k
    at ``path[path_off[k]:path_off[k+1]]``; ``skipped`` pairs have a NaN score and no path."""
    scores: np.ndarray
    ends: np.ndarray
    path: np.ndarray
    path_off: np.ndarray
    skipped: np.ndarray

    def path_of(self, k: int) -> np.ndarray:
        return self.path[self.path_off[k]:self.path_off[k + 1]]


def _prepare(qp, codes, offsets, lengths, max_pair_cells):
    qp = _require_qp(qp)
    codes, starts, lens = _candidates(codes, offsets, lengths)
    n = qp.shape[0]
    skipped = (lens * n) > int(max_pair_cells)
    _note_skipped(int(skipped.sum()), n, max_pair_cells)
    return qp, codes, starts, lens, skipped


def _add_time(timings: Dict[str, float], name: str, seconds: float) -> None:
    timings[name] = timings.get(name, 0.0) + seconds


def score_fwd_rev(qp: np.ndarray, codes: np.ndarray, offsets, params: DPParams, *, lengths=None,
                  block_floats: int = DEFAULT_BLOCK_FLOATS, max_pair_cells: int = DEFAULT_MAX_PAIR_CELLS,
                  reference_sim: bool = False, kernel: Optional[Kernel] = None,
                  workspace: Optional[Workspace] = None, timings: Optional[Dict[str, float]] = None) -> ScoreResult:
    """Round 1 (S9): S_f and S_r of one query against K candidates, score only. One similarity per pair (its own
    GEMM); S_r runs on the row-reversed view of it. See :func:`plan_blocks` for ``block_floats``.

    ``timings`` (profiling only): a dict that receives the seconds spent in the similarity (``sim``: decode +
    per-pair GEMMs) and in the kernel call (``kernel``: rowcol z + DP of S_f and S_r), and the ``blocks`` count; the
    results are the same with or without it."""
    k = kernel or load()
    ws = workspace or default_workspace()
    qp, codes, starts, lens, skipped = _prepare(qp, codes, offsets, lengths, max_pair_cells)
    n, K = qp.shape[0], len(lens)
    s_f = np.full(K, np.nan, np.float32)
    s_r = np.full(K, np.nan, np.float32)
    if reference_sim:
        for k0, k1 in plan_blocks(n, lens.tolist(), skipped.tolist(), block_floats):
            b = _Block(k, ws, qp, codes, starts, lens, k0, k1, True)
            kw = ws.kernel_floats(k.ws_floats(n, int(b.m.max())))
            _check_status(k.lib.fmsw_score_fwd_rev(b.sim.ctypes.data, n, b.pair_off.ctypes.data, b.m.ctypes.data,
                                                   b.K, params.offset, params.gap_open, params.gap_extend,
                                                   s_f[k0:].ctypes.data, s_r[k0:].ctypes.data, kw.ctypes.data,
                                                   kw.size), "fmsw_score_fwd_rev")
        return ScoreResult(s_f, s_r, skipped)
    lay = _Layout(n, starts, lens, skipped, block_floats)
    if not lay.blocks:
        return ScoreResult(s_f, s_r, skipped)
    # one set of buffers and pointers for the call (block k's arguments are offsets into them)
    sim, kw = ws.sim(max(lay.floats)), ws.kernel_floats(k.ws_floats(n, lay.mmax))
    D = ws.decoded(max(lay.rows))
    D3 = D.reshape(-1, CODE_BYTES, 8)
    sim_p, po_p, m_p = sim.ctypes.data, lay.pair_off.ctypes.data, lay.m32.ctypes.data
    sf_p, sr_p, kw_p, kw_n = s_f.ctypes.data, s_r.ctypes.data, kw.ctypes.data, kw.size
    if sim_p % ALIGN_BYTES:                                             # by construction; S6 layout guard
        raise KernelError("similarity block buffer is not 64-byte aligned")
    fn, off, go, ge = k.lib.fmsw_score_fwd_rev, params.offset, params.gap_open, params.gap_extend
    for b, (k0, k1) in enumerate(lay.blocks):
        if timings is not None:
            t0 = time.perf_counter()
        _fill_block(lay, b, qp, codes, D, D3, sim)
        if timings is not None:
            t1 = time.perf_counter()
            _add_time(timings, "sim", t1 - t0)
        st = fn(sim_p, n, po_p + 8 * k0, m_p + 4 * k0, k1 - k0, off, go, ge, sf_p + 4 * k0, sr_p + 4 * k0, kw_p,
                kw_n)
        if timings is not None:
            _add_time(timings, "kernel", time.perf_counter() - t1)
            _add_time(timings, "blocks", 1)
        if st:
            _check_status(st, "fmsw_score_fwd_rev")
    return ScoreResult(s_f, s_r, skipped)


def align(qp: np.ndarray, codes: np.ndarray, offsets, params: DPParams, *, lengths=None, traceback: bool = True,
          block_floats: int = DEFAULT_BLOCK_FLOATS, max_pair_cells: int = DEFAULT_MAX_PAIR_CELLS,
          reference_sim: bool = False, kernel: Optional[Kernel] = None,
          workspace: Optional[Workspace] = None, timings: Optional[Dict[str, float]] = None) -> AlignResult:
    """Round 2: forward DP of one query against K candidates; with ``traceback`` also the paths (forward order,
    exactly as the kernel wrote them; never reversed here). The scores equal :func:`score_fwd_rev`'s S_f bit for
    bit (same per-pair GEMM, same DP).

    ``timings`` (profiling only): as for :func:`score_fwd_rev` (``kernel`` is rowcol z + DP + traceback here), plus
    ``paths``, the seconds spent gathering the paths of the blocks; the results are the same with or without it."""
    k = kernel or load()
    ws = workspace or default_workspace()
    qp, codes, starts, lens, skipped = _prepare(qp, codes, offsets, lengths, max_pair_cells)
    n, K = qp.shape[0], len(lens)
    scores = np.full(K, np.nan, np.float32)
    ends = np.zeros((K, 2), np.int32)
    if reference_sim:
        return _align_blockwise(k, ws, qp, codes, starts, lens, skipped, params, traceback, block_floats, scores,
                                ends, timings)
    counts = np.zeros(K, np.int64)
    lay = _Layout(n, starts, lens, skipped, block_floats)
    path = np.zeros((0, 2), np.int32)
    if lay.blocks:
        # one set of buffers and pointers for the call; the blocks' paths go one after another into one buffer
        sim, kw = ws.sim(max(lay.floats)), ws.kernel_floats(k.ws_floats(n, lay.mmax))
        D = ws.decoded(max(lay.rows))
        D3 = D.reshape(-1, CODE_BYTES, 8)
        sim_p, po_p, m_p = sim.ctypes.data, lay.pair_off.ctypes.data, lay.m32.ctypes.data
        sc_p, en_p, kw_p, kw_n = scores.ctypes.data, ends.ctypes.data, kw.ctypes.data, kw.size
        if sim_p % ALIGN_BYTES:                                         # by construction; S6 layout guard
            raise KernelError("similarity block buffer is not 64-byte aligned")
        if traceback:
            fl = ws.flags(k.flag_bytes(n, lay.mmax))
            cap = int(n * int((~skipped).sum()) + int(lens[~skipped].sum()))      # each pair needs <= n + m entries
            pbuf = ws.path_entries(cap)
            poff = ws.path_offsets(max(k1 - k0 for k0, k1 in lay.blocks) + 1)
            fl_p, fl_n, pb_p, pf_p = fl.ctypes.data, fl.size, pbuf.ctypes.data, poff.ctypes.data
        else:
            cap, fl_p, fl_n, pb_p, pf_p = 0, None, 0, None, None
        fn, off, go, ge, tb = k.lib.fmsw_align, params.offset, params.gap_open, params.gap_extend, int(bool(traceback))
        used = 0                                                         # path entries written so far
        for b, (k0, k1) in enumerate(lay.blocks):
            if timings is not None:
                t0 = time.perf_counter()
            _fill_block(lay, b, qp, codes, D, D3, sim)
            if timings is not None:
                t2 = time.perf_counter()
                _add_time(timings, "sim", t2 - t0)
            st = fn(sim_p, n, po_p + 8 * k0, m_p + 4 * k0, k1 - k0, off, go, ge, tb, sc_p + 4 * k0, en_p + 8 * k0,
                    None if pb_p is None else pb_p + 8 * used, cap - used, pf_p, kw_p, kw_n, fl_p, fl_n)
            if timings is not None:
                _add_time(timings, "kernel", time.perf_counter() - t2)
                _add_time(timings, "blocks", 1)
            if st:
                _check_status(st, "fmsw_align")
            if traceback:
                pb = poff[:k1 - k0 + 1]
                counts[k0:k1] = np.diff(pb)
                used += int(pb[-1])
        if traceback:
            if timings is not None:
                t3 = time.perf_counter()
            path = pbuf[:2 * used].reshape(-1, 2).copy()        # compact, and the workspace buffer is reused
            if timings is not None:
                _add_time(timings, "paths", time.perf_counter() - t3)
    if timings is not None:
        timings.setdefault("paths", 0.0)
    path_off = np.zeros(K + 1, np.int64)
    np.cumsum(counts, out=path_off[1:])
    return AlignResult(scores, ends, path, path_off, skipped)


def _align_blockwise(k: Kernel, ws: Workspace, qp, codes, starts, lens, skipped, params: DPParams, traceback: bool,
                     block_floats: int, scores, ends, timings) -> AlignResult:
    """:func:`align` block by block through :class:`_Block` (the ``reference_sim`` path)."""
    n, K = qp.shape[0], len(lens)
    counts = np.zeros(K, np.int64)
    parts: List[Tuple[int, np.ndarray, np.ndarray]] = []
    reference_sim = True
    for k0, k1 in plan_blocks(n, lens.tolist(), skipped.tolist(), block_floats):
        if timings is not None:
            t0 = time.perf_counter()
        b = _Block(k, ws, qp, codes, starts, lens, k0, k1, reference_sim)
        if timings is not None:
            t1 = time.perf_counter()
            _add_time(timings, "sim", t1 - t0)
        mmax = int(b.m.max())
        kw = ws.kernel_floats(k.ws_floats(n, mmax))
        if traceback:
            fl = ws.flags(k.flag_bytes(n, mmax))
            cap = int(b.K * n + int(b.m.sum()))
            pbuf = np.empty(2 * max(cap, 1), np.int32)
            poff = np.empty(b.K + 1, np.int64)
            fl_ptr, fl_len, p_ptr, po_ptr = fl.ctypes.data, fl.size, pbuf.ctypes.data, poff.ctypes.data
        else:
            cap, fl_ptr, fl_len, p_ptr, po_ptr = 0, None, 0, None, None
        if timings is not None:
            t2 = time.perf_counter()
        st = k.lib.fmsw_align(b.sim.ctypes.data, n, b.pair_off.ctypes.data, b.m.ctypes.data, b.K,
                              params.offset, params.gap_open, params.gap_extend, int(bool(traceback)),
                              scores[k0:].ctypes.data, ends[k0:].ctypes.data, p_ptr, cap, po_ptr,
                              kw.ctypes.data, kw.size, fl_ptr, fl_len)
        if timings is not None:
            _add_time(timings, "kernel", time.perf_counter() - t2)
            _add_time(timings, "blocks", 1)
        _check_status(st, "fmsw_align")
        if traceback:
            counts[k0:k1] = np.diff(poff)
            parts.append((k0, pbuf[:2 * int(poff[-1])].reshape(-1, 2), poff))
    if timings is not None:
        t3 = time.perf_counter()
    path_off = np.zeros(K + 1, np.int64)
    np.cumsum(counts, out=path_off[1:])
    if parts:
        path = np.concatenate([p for _, p, _ in parts]) if len(parts) > 1 else parts[0][1].copy()
    else:
        path = np.zeros((0, 2), np.int32)
    if timings is not None:
        _add_time(timings, "paths", time.perf_counter() - t3)
    return AlignResult(scores, ends, path, path_off, skipped)


# =====================================================================================================================
# paths: CIGAR and metrics (numpy only; no biotite)
# =====================================================================================================================
_OP_M, _OP_I, _OP_D = ord("M"), ord("I"), ord("D")


def path_ops(path: np.ndarray) -> np.ndarray:
    """Column types of a path as uint8 'M' (both residues), 'I' (query residue vs target gap, j == -1) and 'D'
    (gap in the query, i == -1) -- the BLOSUM path's CIGAR convention."""
    p = np.asarray(path).reshape(-1, 2)
    ops = np.full(len(p), _OP_M, np.uint8)
    ops[p[:, 1] < 0] = _OP_I
    ops[p[:, 0] < 0] = _OP_D
    return ops


def cigar_from_path(path: np.ndarray) -> str:
    ops = path_ops(path)
    if not len(ops):
        return ""
    starts = np.flatnonzero(np.r_[True, ops[1:] != ops[:-1]])
    runs = np.diff(np.r_[starts, len(ops)])
    return "".join(f"{r}{chr(o)}" for r, o in zip(runs.tolist(), ops[starts].tolist()))


_CIGAR_RE = re.compile(r"(\d+)([MID])")


def expand_cigar(cigar: str) -> str:
    """'3M1D2M' -> 'MMMDMM' (refuses anything but M/I/D runs)."""
    parts = _CIGAR_RE.findall(cigar)
    if "".join(f"{c}{o}" for c, o in parts) != cigar:
        raise ValueError(f"not an M/I/D CIGAR: {cigar!r}")
    return "".join(o * int(c) for c, o in parts)


@dataclass(frozen=True)
class PathMetrics:
    """MMseqs2-style column tallies of one path. Bounds are 1-based (0 for an empty path); ``q_aligned`` /
    ``t_aligned`` count the columns with a query / target residue (coverage = aligned / length)."""
    aln_len: int
    n_ident: int
    mismatch: int
    gap_open: int
    q_aligned: int
    t_aligned: int
    q_start: int
    q_end: int
    t_start: int
    t_end: int
    cigar: Optional[str] = None
    q_aln: Optional[str] = None
    t_aln: Optional[str] = None


def _runs(mask: np.ndarray) -> int:
    return int(mask[0]) + int(np.count_nonzero(mask[1:] & ~mask[:-1])) if mask.size else 0


def path_metrics(path: np.ndarray, query_seq: str, target_seq: str, *, strings: bool = False) -> PathMetrics:
    """Identity, mismatches, gap opens, bounds (and with ``strings`` the CIGAR and gapped strings) of a path over
    the given sequences (already sanitised and upper-cased by the caller; compared byte for byte)."""
    p = np.asarray(path, dtype=np.int64).reshape(-1, 2)
    qi, tj = p[:, 0], p[:, 1]
    q_gap, t_gap = qi < 0, tj < 0
    both = ~q_gap & ~t_gap
    q_arr = np.frombuffer(query_seq.encode("ascii"), np.uint8)
    t_arr = np.frombuffer(target_seq.encode("ascii"), np.uint8)
    n_ident = int(np.count_nonzero(q_arr[qi[both]] == t_arr[tj[both]]))
    q_res, t_res = qi[~q_gap], tj[~t_gap]
    cigar = q_aln = t_aln = None
    if strings:
        cigar = cigar_from_path(p)
        qa = np.full(len(p), ord("-"), np.uint8)
        ta = qa.copy()
        qa[~q_gap] = q_arr[q_res]
        ta[~t_gap] = t_arr[t_res]
        q_aln, t_aln = qa.tobytes().decode("ascii"), ta.tobytes().decode("ascii")
    return PathMetrics(
        aln_len=len(p), n_ident=n_ident, mismatch=int(both.sum()) - n_ident, gap_open=_runs(q_gap) + _runs(t_gap),
        q_aligned=int(q_res.size), t_aligned=int(t_res.size),
        q_start=int(q_res.min()) + 1 if q_res.size else 0, q_end=int(q_res.max()) + 1 if q_res.size else 0,
        t_start=int(t_res.min()) + 1 if t_res.size else 0, t_end=int(t_res.max()) + 1 if t_res.size else 0,
        cigar=cigar, q_aln=q_aln, t_aln=t_aln)


# =====================================================================================================================
# float64 diagnostics (S16 path-tie adjudication; tests)
# =====================================================================================================================
def query_prep64(q16: np.ndarray) -> np.ndarray:
    """float64 S5: q / (8 |q|) per row, zero rows stay zero."""
    q = np.asarray(q16, dtype=np.float64)
    nrm = np.sqrt(np.sum(q * q, axis=1, keepdims=True))
    zero = nrm < 1e-12
    return np.where(zero, 0.0, q / np.where(zero, 1.0, 8.0 * nrm))


def rowcol_z64(sim: np.ndarray) -> np.ndarray:
    """float64 S7 rowcol z with the degenerate rule (a row / column with population std < 1e-6 contributes 0)."""
    s = np.asarray(sim, dtype=np.float64)
    sd, tau = s.std(axis=1, keepdims=True), s.std(axis=0, keepdims=True)
    zr = np.where(sd < 1e-6, 0.0, (s - s.mean(axis=1, keepdims=True)) / (sd + 1e-8))
    zc = np.where(tau < 1e-6, 0.0, (s - s.mean(axis=0, keepdims=True)) / (tau + 1e-8))
    return 0.5 * (zr + zc)


def z64_pair(q16: np.ndarray, codes_t: np.ndarray) -> np.ndarray:
    """float64 z of a pair (S5 -> S6 -> S7 in float64)."""
    return rowcol_z64(query_prep64(q16) @ decode_codes(codes_t).astype(np.float64).T)


def path_score(z: np.ndarray, path: np.ndarray, params: DPParams) -> float:
    """Score of a path on a z matrix: sum of (z_ij - offset) over aligned pairs minus go + (L - 1) ge per maximal
    gap run (X and Y runs counted separately, as in the DP)."""
    p = np.asarray(path, dtype=np.int64).reshape(-1, 2)
    if not len(p):
        return 0.0
    qi, tj = p[:, 0], p[:, 1]
    both = (qi >= 0) & (tj >= 0)
    s = float(np.sum(np.asarray(z, np.float64)[qi[both], tj[both]] - params.offset))
    for gap in (tj < 0, qi < 0):
        runs, total = _runs(gap), int(gap.sum())
        s -= runs * params.gap_open + (total - runs) * params.gap_extend
    return s


def s16_score_bound(ref) -> np.ndarray:
    """S16 bound on |S - S_ref| for S_f / S_r: 5e-5 max(|ref|, 1)."""
    return 5e-5 * np.maximum(np.abs(np.asarray(ref, np.float64)), 1.0)


def s16_path_margin(s_f) -> float:
    """S16 near-tie margin for paths: 1e-4 + 2e-6 |S_f|."""
    return 1e-4 + 2e-6 * abs(float(s_f))


# =====================================================================================================================
# DP golden hash and selftest
# =====================================================================================================================
_SM_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM_M1, _SM_M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


class _SplitMix64:
    """Portable integer stream (splitmix64 of a counter): the golden inputs never depend on numpy's RNG streams or
    on libm, so they are the same bytes on every platform and numpy version."""

    def __init__(self, seed: int):
        self._seed = np.uint64(seed)
        self._pos = 0

    def u64(self, size: int) -> np.ndarray:
        i = np.arange(self._pos + 1, self._pos + size + 1, dtype=np.uint64)
        self._pos += size
        z = self._seed + i * _SM_GAMMA
        z = (z ^ (z >> np.uint64(30))) * _SM_M1
        z = (z ^ (z >> np.uint64(27))) * _SM_M2
        return z ^ (z >> np.uint64(31))

    def ints(self, lo: int, hi: int, shape=()) -> np.ndarray:
        """Integers in [lo, hi) (modulo reduction; the bias is < 2^-50 for these ranges)."""
        size = int(np.prod(shape)) if shape != () else 1
        v = (self.u64(size) % np.uint64(hi - lo)).astype(np.int64) + lo
        return v.reshape(shape) if shape != () else v[0]

    def int(self, lo: int, hi: int) -> int:
        return int(self.ints(lo, hi))

    def qrows(self, rows: int) -> np.ndarray:
        """int64 [rows, 64] in [-2048, 2048): q = value / 256 is exact in fp16."""
        return (self.u64(rows * DIM) >> np.uint64(52)).astype(np.int64).reshape(rows, DIM) - 2048

    def code_rows(self, rows: int) -> np.ndarray:
        return self.u64(rows).astype("<u8").view(np.uint8).reshape(rows, CODE_BYTES).copy()


def _q16(qint: np.ndarray) -> np.ndarray:
    return (qint.astype(np.float64) / 256.0).astype(np.float16)


GOLDEN_SEED = 20261013


def golden_pairs(count: int = 1000, seed: int = GOLDEN_SEED) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Deterministic synthetic pairs (kind, q fp16 [n, 64], codes uint8 [m, 8]): half random (n, m in [1, 400]),
    40% related (target codes = sign bits of a noisy, gapped copy of a query segment with flanks), 10% edge cases
    (n = 1, m = 1, identical codes, identical query rows, zero query rows, 1 x 1). Integer arithmetic only."""
    g = _SplitMix64(seed)
    n_random, n_related = count // 2, (2 * count) // 5
    out = []
    for p in range(count):
        if p < n_random:
            n, m = g.int(1, 401), g.int(1, 401)
            out.append(("random", _q16(g.qrows(n)), g.code_rows(m)))
        elif p < n_random + n_related:
            n = g.int(20, 401)
            q = g.qrows(n)
            a, b = sorted((g.int(0, n), g.int(0, n)))
            seg = q[a:b + 1]
            seg = seg[g.ints(0, 10, (len(seg),)) != 0]                       # ~10% deletions
            ins = g.qrows(g.int(0, 10))
            pos = g.int(0, len(seg) + 1)
            t = np.concatenate([g.qrows(g.int(0, 60)), seg[:pos], ins, seg[pos:], g.qrows(g.int(0, 60))])
            if not len(t):
                t = g.qrows(1)
            amp = g.int(300, 1500)
            t = t + g.ints(-amp, amp + 1, t.shape)
            out.append(("related", _q16(q), np.packbits(t >= 0, axis=1)))
        else:
            e = (p - n_random - n_related) % 6
            n, m = g.int(1, 200), g.int(1, 200)
            if e == 0:
                q, c = g.qrows(1), g.code_rows(m)
            elif e == 1:
                q, c = g.qrows(n), g.code_rows(1)
            elif e == 2:
                q, c = g.qrows(n), np.repeat(g.code_rows(1), m, axis=0)
            elif e == 3:
                q, c = np.repeat(g.qrows(1), n, axis=0), g.code_rows(m)
            elif e == 4:
                q = g.qrows(n)
                q[g.ints(0, 10, (n,)) < 3] = 0
                c = g.code_rows(m)
            else:
                q, c = g.qrows(1), g.code_rows(1)
            out.append((f"edge{e}", _q16(q), c))
    return out


def dp_digest_update(h, n: int, m: int, s_f: float, s_r: float, path: np.ndarray):
    """One pair's record: <i4 n, <i4 m, <f4 S_f, <f4 S_r, <i4 path length, path as <i4 [k, 2]."""
    p = np.ascontiguousarray(np.asarray(path).reshape(-1, 2), dtype="<i4")
    h.update(struct.pack("<iiffi", int(n), int(m), float(s_f), float(s_r), len(p)))
    h.update(p.tobytes())


def dp_golden_new():
    return hashlib.sha256(b"fmsw/1 dp golden v1\0")


@dataclass
class PairRun:
    """One query against one target: round-1 scores, round-2 score and path."""
    s_f: float
    s_r: float
    s_tb: float
    end: Tuple[int, int]
    path: np.ndarray


def run_pair(q16: np.ndarray, codes: np.ndarray, params: DPParams = SETTING_D, *, reference_sim: bool,
             kernel: Optional[Kernel] = None) -> PairRun:
    """S5 -> S6 (BLAS, or the reference) -> S_f, S_r (score_fwd_rev) and the forward path (align) for one pair."""
    k = kernel or load()
    qp = query_prep(q16, k)
    off = np.array([0, len(codes)], np.int64)
    r1 = score_fwd_rev(qp, codes, off, params, reference_sim=reference_sim, kernel=k)
    r2 = align(qp, codes, off, params, reference_sim=reference_sim, kernel=k)
    return PairRun(float(r1.s_fwd[0]), float(r1.s_rev[0]), float(r2.scores[0]), tuple(int(x) for x in r2.ends[0]),
                   r2.path_of(0).copy())


# sha256 of the DP golden records (dp_digest_update) of golden_pairs(1000) through fmsw_sim_ref, setting D.
# Computed 2026-10-13 on an Apple M1 Max: Apple clang 21 neon and generic (arm64), sse2 and generic (x86_64 under
# Rosetta) all give this value; every conforming build reproduces it bit for bit. tests/test_swkern.py also holds
# the F1 + synthetic constant (F1 is not shipped in wheels, so selftest() checks the synthetic part).
DP_GOLDEN_SYNTH_SHA256 = "6b7974aae7081dc8de65a8c240c24094ab0c09ae1ba276a062684f26e647017e"


def blas_info() -> Dict[str, object]:
    """numpy's BLAS (np.show_config) and, when threadpoolctl is installed, the BLAS thread pools."""
    info: Dict[str, object] = {"numpy": np.__version__}
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        info["blas"] = {k: blas.get(k) for k in ("name", "version", "openblas configuration") if k in blas}
    except Exception as e:                                              # pragma: no cover - numpy < 1.26
        info["blas"] = f"unavailable ({e})"
    try:
        import threadpoolctl
        info["threadpools"] = [{k: d.get(k) for k in ("internal_api", "version", "num_threads", "architecture")}
                               for d in threadpoolctl.threadpool_info() if d.get("user_api") == "blas"]
    except ImportError:
        pass
    return info


def selftest(kernel: Optional[Kernel] = None, *, count: int = 1000) -> Dict[str, object]:
    """Check this install: the DP golden hash through ``fmsw_sim_ref`` (bit-identical on every conforming build),
    then the production BLAS path against the golden values within S16 (scores; paths identical unless the float64
    path scores tie within the S16 margin). Logs the kernel and numpy's BLAS. Raises KernelError on failure."""
    t0 = time.perf_counter()
    k = kernel or load()
    pairs = golden_pairs(count)
    h = dp_golden_new()
    ref: List[PairRun] = []
    for _kind, q16, codes in pairs:
        r = run_pair(q16, codes, reference_sim=True, kernel=k)
        if not (np.isfinite(r.s_f) and np.isfinite(r.s_r)) or r.s_tb != r.s_f:
            raise KernelError(f"selftest: non-finite or inconsistent golden scores ({r.s_f}, {r.s_r}, {r.s_tb})")
        dp_digest_update(h, len(q16), len(codes), r.s_f, r.s_r, r.path)
        ref.append(r)
    digest = h.hexdigest()
    report: Dict[str, object] = dict(kernel=k.info(), blas=blas_info(), pairs=count, dp_golden_sha256=digest)
    if count == 1000 and digest != DP_GOLDEN_SYNTH_SHA256:
        raise KernelError(f"selftest: DP golden hash {digest} != {DP_GOLDEN_SYNTH_SHA256} ({k.info()})")
    worst, path_diff, path_ties = 0.0, 0, 0
    for (_kind, q16, codes), g in zip(pairs, ref):
        b = run_pair(q16, codes, reference_sim=False, kernel=k)
        for got, want in ((b.s_f, g.s_f), (b.s_r, g.s_r)):
            ratio = abs(got - want) / float(s16_score_bound(want))
            worst = max(worst, ratio)
        if not np.array_equal(b.path, g.path):
            path_diff += 1
            z = z64_pair(q16, codes)
            if abs(path_score(z, b.path, SETTING_D) - path_score(z, g.path, SETTING_D)) >= s16_path_margin(g.s_f):
                raise KernelError(f"selftest: BLAS path differs from the golden path beyond the S16 tie margin "
                                  f"(n={len(q16)}, m={len(codes)})")
            path_ties += 1
    if worst > 1.0:
        raise KernelError(f"selftest: BLAS scores outside S16 (max |delta| / bound = {worst:.3g})")
    report.update(s16_max_score_ratio=worst, blas_path_differences=path_diff, blas_path_near_ties=path_ties,
                  seconds=round(time.perf_counter() - t0, 3))
    logger.info("fmsw selftest passed: %s", report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m foldmatch.search.swkern``: load the kernel (dispatch, or ``$FOLDMATCH_SW_ISA``), run
    :func:`selftest` and print its report as JSON. Exit status 1 when the kernel is unusable or the selftest fails.
    Used after installs (Docker, CI, wheels) to check that the packaged libraries work on this machine."""
    import argparse
    import json
    import sys
    ap = argparse.ArgumentParser(prog="python -m foldmatch.search.swkern",
                                 description="Run the native kernel selftest and print its report as JSON.")
    ap.parse_args(argv)
    try:
        report = selftest()
    except KernelError as e:
        print(f"fmsw selftest failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
