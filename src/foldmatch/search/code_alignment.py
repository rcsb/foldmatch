"""esm-codes Stage 2: align Stage-1 candidates with ESM3 residue codes (plan section 8 P6.4; method S5-S14, S16).

For each query (its S5 encoding in the per-run query memmap, :class:`~foldmatch.search.query_encodings.
QueryEncodingIndex`) and its Stage-1 candidates (their S4 codes in the database's
:class:`~foldmatch.search.residue_code_store.ResidueCodeStore`, looked up **by id**: the ``.bin`` order depends on the
number of build ranks), :func:`align_candidates_esm_codes` runs::

    parent    drop fully degenerate queries (S13b: n = 1 or all q rows bitwise identical; no Stage-2 work, no hits)
    round 1   score only, per pair: S_f and S_r (S9; one similarity GEMM per pair, S_r on its row-reversed view)
              over the panel members (D11: only when E-values are needed) and the candidates, in chunked tasks
    parent    S = fp64(S_f) - fp64(S_r); Q90 = quantile(panel S, 0.9) (S10); S' = S - Q90, ln E, bits (S11);
              survivors: ln E <= ln(max_evalue), or every candidate when there is no E filter
    round 2   survivors only: forward traceback of S_f (S12); its score must equal round 1's S_f bit for bit
              (:class:`~foldmatch.errors.InternalError` otherwise); whether the target has enough distinct codes (S13)
    parent    vectorised path metrics, ecal (S13), filters (alnlen > 0, identity, min(qcov, tcov)), sort (S14)

**Tasks.** A task is ``(task id, source, query memmap offset, query length, [(offset, length), ...])``: the rows of
its targets in the panel codes or in the code store. Workers hold everything else (:func:`_init_worker`: the kernel,
the scoring npz through :meth:`~foldmatch.search.scoring_system.WorkerSpec.load_panel`, the code-store memmap, the
query memmap and a :class:`~foldmatch.search.swkern.Workspace`) and return numpy arrays. The task size adapts to the
batch: at most ``task_cells`` similarity cells, and small enough for at least 4 tasks per worker (but not below
2^20 cells), so that one interactive query still uses every core. Workers import neither torch nor biotite.

**Dataflow** (:class:`_Dataflow`). A batch's round-1 tasks are queued longest first (LPT, by ``n x sum(m)``); each
query moves on as soon as its own tasks are back: statistics and survivors when its round 1 is complete (its round-2
tasks are queued ahead of round 1), hits when its round 2 is complete (its paths are then released). The next batch is
looked up and queued while the workers still have this one's tasks (at most two batches open). Tasks are submitted one
at a time, at most twice the pool's workers in flight (:class:`_Session`), and their results consumed in completion
order: the parent holds the round-2 payload of the queries in flight only, and builds hits while the workers compute.

**Reproducibility (S16).** Every pair's similarity is its own fp32 GEMM into a 64-byte-aligned buffer, so a pair's
bits depend only on the pair, the BLAS build and the BLAS thread count (multi-threaded OpenBLAS changes last bits,
within S16): with BLAS single-threaded the output is bit-identical for any ``task_cells``, ``block_floats``,
``query_batch``, worker count (in-process or pool) and candidate order. The pool's workers are started with
``OMP/OPENBLAS/MKL/VECLIB_*_NUM_THREADS=1`` (and limited with threadpoolctl when it is installed); in-process tasks run
under a threadpoolctl limit of 1 thread, so without threadpoolctl a pool that was asked for workers never runs tasks
in-process, and an explicitly serial run (0 or 1 workers) needs ``*_NUM_THREADS=1`` in the environment to give the
pool's bits.

**Pool.** :class:`CodeAlignmentPool` is the persistent ``spawn`` pool of one run (a
:class:`concurrent.futures.ProcessPoolExecutor`), started lazily at the first batch that needs it and reused for both
rounds and every batch; without a worker count it starts only as many workers as the batch's work justifies (one per
:data:`MIN_WORKER_CELLS` similarity cells, up to every CPU); ``None``, ``0`` or ``1`` as the aligner's ``pool`` runs
the same task functions in-process. A worker that ends without returning its task (killed, e.g. by the out-of-memory killer, or unable to start) stops the run
with :class:`WorkerPoolError` instead of leaving it waiting. Spawn workers re-import the main module, so a script
that runs this Stage 2 with a pool must guard its entry point with ``if __name__ == "__main__":`` (the ``fm-search``
command does); without the guard the workers cannot start and the run fails with :class:`WorkerPoolError`.
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from foldmatch.errors import InternalError
from foldmatch.search import code_statistics as cs
from foldmatch.search import stage2_profile
from foldmatch.search import swkern
from foldmatch.search.alignment import (
    _BIOTITE_ALPHABET, AlignmentMetrics, Hit, _available_cpus, _expand_cpu_affinity,
)
from foldmatch.utils.sequence import STANDARD_AMINO_ACIDS, normalize_sequence, seq_sha1

logger = logging.getLogger(__name__)

PANEL_POLICIES = ("auto", "always", "never")
#: Output fields whose values need the panel pass (Q90): with none of them requested and no finite ``max_evalue``,
#: ``panel_policy="auto"`` skips the panel (D11). ``q90`` and ``rawadj`` are esm-codes fields (P6.5).
PANEL_FIELDS = frozenset({"evalue", "bits", "q90", "rawadj"})
DEFAULT_TASK_CELLS = 1 << 24
TASK_CELLS_FLOOR = 1 << 20
TASKS_PER_WORKER = 4
#: A pool sized by default (``num_workers=None``) starts at most one worker per this many similarity cells of a batch
#: (about 40 ms of one EPYC core, against about 0.3 s to start a spawn worker), so a small query does not start a
#: worker on every CPU; a later batch with more work grows it (by restart, at least doubling). 2^23 gives a 300-aa
#: query with 100 candidates and the panel 14 workers.
MIN_WORKER_CELLS = 1 << 23
DEFAULT_QUERY_BATCH = 256
_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                     "BLIS_NUM_THREADS")
_PANEL, _CODES = 0, 1
_EXAMPLES = 10


class WorkerPoolError(RuntimeError):
    """A Stage-2 worker process ended without returning its task's result: it was killed (for example by the
    out-of-memory killer) or could not start (a script without an ``if __name__ == "__main__":`` guard). The pool
    is stopped; a persistent :class:`CodeAlignmentPool` starts a new one at its next batch."""


#: swkern's once-per-process warning about pairs above ``max_pair_cells``. Stage 2 drops it (in a pool worker it
#: would bypass the parent's logging setup, once per worker) and the parent logs the run's count once instead.
_SWKERN_SKIP_WARNING = "esm-codes Stage 2: skipping pairs whose similarity exceeds max_pair_cells"


class _DropSwkernSkipWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith(_SWKERN_SKIP_WARNING)


_DROP_SKIP_WARNING = _DropSwkernSkipWarning()


# =====================================================================================================================
# output needs
# =====================================================================================================================
@dataclass(frozen=True)
class OutputNeeds:
    """What the requested output columns need beyond the always-computed scalars: the heavy strings, and whether any
    column needs the panel pass (``significance``: evalue, bits, q90 or rawadj)."""
    cigar: bool = False
    qaln: bool = False
    taln: bool = False
    qseq: bool = False
    tseq: bool = False
    significance: bool = False

    @classmethod
    def from_fields(cls, fields: Iterable[str]) -> "OutputNeeds":
        fs = set(fields)
        return cls(cigar="cigar" in fs, qaln="qaln" in fs, taln="taln" in fs, qseq="qseq" in fs,
                   tseq="tseq" in fs, significance=bool(fs & PANEL_FIELDS))

    @classmethod
    def coerce(cls, value) -> "OutputNeeds":
        """An :class:`OutputNeeds`, or the ``--format-output`` field names to derive it from."""
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            raise TypeError(f"output_needs must be an OutputNeeds or a sequence of output field names, got {value!r}")
        return cls.from_fields(value)


# =====================================================================================================================
# task planning
# =====================================================================================================================
def target_task_cells(total_cells: int, workers: int, task_cells: int) -> int:
    """Similarity cells per task for a batch of ``total_cells``: at most ``task_cells``, and small enough to give at
    least ``TASKS_PER_WORKER`` tasks per worker, but not below ``TASK_CELLS_FLOOR`` (2^20) unless ``task_cells`` is."""
    adaptive = max(TASK_CELLS_FLOOR, -(-int(total_cells) // (TASKS_PER_WORKER * max(1, int(workers)))))
    return max(1, min(int(task_cells), adaptive))


def plan_chunks(n: int, lengths: Sequence[int], target_cells: int) -> List[Tuple[int, int]]:
    """Ranges ``[k0, k1)`` of consecutive targets whose ``n x m`` cells add up to at most ``target_cells`` (greedy,
    in order); a target that alone exceeds it is its own chunk. Chunking never changes a pair's bits."""
    out: List[Tuple[int, int]] = []
    k0, acc = 0, 0
    for k, m in enumerate(lengths):
        cells = n * int(m)
        if k > k0 and acc + cells > target_cells:
            out.append((k0, k))
            k0, acc = k, 0
        acc += cells
    if k0 < len(lengths):
        out.append((k0, len(lengths)))
    return out


def _lpt(tasks: list, costs: list) -> list:
    """Longest-processing-time-first order (stable: equal costs keep their task id order)."""
    order = sorted(range(len(tasks)), key=lambda i: -costs[i])
    return [tasks[i] for i in order]


# =====================================================================================================================
# worker state and task functions (also run in-process by the serial path)
# =====================================================================================================================
@dataclass(frozen=True)
class _WorkerInit:
    """Picklable worker setup: the kernel ISA, the scoring system's worker view, the code store and query memmaps as
    ``(path, rows, width)``, and the kernel blocking limits."""
    isa: str
    spec: Any                         # scoring_system.WorkerSpec
    codes: Tuple[str, int, int]       # code store .bin, N_res, code_bytes
    queries: Tuple[str, int, int]     # query encodings, rows, width
    block_floats: int
    max_pair_cells: int
    profile: bool = False             # profiling mode (stage2_profile): tasks also return their timing record


class _WorkerState:
    __slots__ = ("init", "kernel", "params", "panel_codes", "codes", "queries", "workspace", "blas_limit", "error",
                 "init_record")

    def __init__(self, init: _WorkerInit, error: Optional[BaseException] = None):
        self.init = init
        self.error = error
        self.kernel = self.params = self.panel_codes = self.codes = self.queries = self.workspace = None
        self.blas_limit = None
        self.init_record = None       # profiling: this process's start-up, sent with its first task


_STATE: Optional[_WorkerState] = None


def _open_rows(path: str, rows: int, width: int, dtype) -> np.ndarray:
    """A raw ``[rows, width]`` file mapped read-only (an empty array for 0 rows, which np.memmap cannot map)."""
    if rows == 0:
        return np.empty((0, width), dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(int(rows), int(width)))


def _blas_limit_available() -> bool:
    """Whether this process can limit its BLAS to one thread at run time (threadpoolctl is installed)."""
    try:
        import threadpoolctl  # noqa: F401
    except ImportError:
        return False
    return True


def _limit_blas_threads():
    """threadpoolctl's limiter of the BLAS pools to 1 thread, or None when threadpoolctl is not installed."""
    try:
        import threadpoolctl
    except ImportError:
        return None
    return threadpoolctl.threadpool_limits(limits=1, user_api="blas")


_BLAS_CONTROL_NOTED = False


def _blas_controllable() -> bool:
    """Whether threadpoolctl finds a BLAS it can limit in this process. It cannot see every BLAS: Apple Accelerate
    (numpy's BLAS on macOS) is not one it supports, so a limit there has no effect."""
    try:
        import threadpoolctl
    except ImportError:
        return False
    return any(info.get("user_api") == "blas" for info in threadpoolctl.threadpool_info())


def _note_uncontrolled_blas() -> None:
    """Once per process, when in-process tasks start: say so if threadpoolctl cannot limit this BLAS (they then run
    with its own thread count, like a run without threadpoolctl)."""
    global _BLAS_CONTROL_NOTED
    if _BLAS_CONTROL_NOTED:
        return
    _BLAS_CONTROL_NOTED = True
    if not _blas_controllable():
        logger.info("esm-codes Stage 2 runs in-process and threadpoolctl cannot limit this BLAS (for example Apple "
                    "Accelerate): it keeps its own thread count (set OMP/OPENBLAS/MKL/VECLIB_*_NUM_THREADS=1 for "
                    "the bits of a worker pool; S16 holds either way)")


def _init_worker(init: _WorkerInit, raise_errors: bool = False) -> None:
    """Load what every task needs, once per process. In a pool worker an error is kept and raised by every task
    instead (a failing initializer would only report a broken pool)."""
    global _STATE
    _STATE = None
    if init.profile:
        t_unix, t0 = time.time(), time.perf_counter()
    try:
        st = _WorkerState(init)
        st.kernel = swkern.load(init.isa)
        st.params = init.spec.dp_params()
        st.panel_codes, _ = init.spec.load_panel()
        st.codes = _open_rows(*init.codes, dtype=np.uint8)
        st.queries = _open_rows(*init.queries, dtype=np.float16)
        st.workspace = swkern.Workspace()
        if not raise_errors:
            st.blas_limit = _limit_blas_threads()
        if init.profile:
            st.init_record = stage2_profile.init_record(t_unix, time.perf_counter() - t0)
        _STATE = st
    except BaseException as e:  # noqa: BLE001  (re-raised by every task of this worker)
        if raise_errors:
            raise
        _STATE = _WorkerState(init, error=e)


def _init_pool_worker(init: _WorkerInit) -> None:
    """The pool's initializer: swkern's oversize-pair warning dropped for the life of the worker process (the parent
    reports the run's count), then :func:`_init_worker` with its errors kept for the tasks."""
    logging.getLogger(swkern.__name__).addFilter(_DROP_SKIP_WARNING)
    _init_worker(init)


def _reset_worker_state() -> None:
    global _STATE
    _STATE = None


def _state() -> _WorkerState:
    st = _STATE
    if st is None:
        raise InternalError("esm-codes Stage-2 task run in a process whose worker state was not initialised")
    if st.error is not None:
        raise st.error
    return st


def _query_prep(st: _WorkerState, q_off: int, n: int) -> np.ndarray:
    return swkern.query_prep(np.asarray(st.queries[q_off:q_off + n]), st.kernel)


def _task_timer(st: _WorkerState) -> Optional[stage2_profile.TaskTimer]:
    return stage2_profile.TaskTimer() if st.init.profile else None


def _task_record(st: _WorkerState, timer: stage2_profile.TaskTimer, kind: str, n: int, pairs, result: tuple) -> dict:
    init, st.init_record = st.init_record, None
    return timer.record(kind, n, pairs, result, init)


def _round1_task(task):
    """Round 1: ``(task id, S_f float32 [k], S_r float32 [k], skipped bool [k])`` of one query against a chunk of
    panel members or candidates (profiling mode: plus the task's timing record)."""
    tid, source, q_off, n, pairs = task
    st = _state()
    timer = _task_timer(st)
    qp = _query_prep(st, q_off, n)
    if timer is not None:
        timer.lap("prep")
    codes = st.panel_codes if source == _PANEL else st.codes
    r = swkern.score_fwd_rev(qp, codes, pairs[:, 0], st.params, lengths=pairs[:, 1],
                             block_floats=st.init.block_floats, max_pair_cells=st.init.max_pair_cells,
                             kernel=st.kernel, workspace=st.workspace, timings=None if timer is None else timer.t)
    if timer is None:
        return tid, r.s_fwd, r.s_rev, r.skipped
    timer.lap("swkern")
    out = (tid, r.s_fwd, r.s_rev, r.skipped)
    return out + (_task_record(st, timer, "panel" if source == _PANEL else "candidates", n, pairs, out),)


def _distinct_codes(codes: np.ndarray, cap: int) -> int:
    """``min(distinct codes, cap)``: S13 only asks whether a target has at least ``cap``
    (``target_min_distinct_codes``) distinct codes, so the count stops there. For ``cap <= 2`` (esm-codes-v1: 2)
    that is one comparison of every code with the first instead of a sort."""
    if len(codes) == 0:
        return 0
    v = np.ascontiguousarray(codes).view(np.uint64).ravel()
    if cap <= 1:
        return 1
    if cap == 2:
        return 2 if bool((v != v[0]).any()) else 1
    return min(int(np.unique(v).size), int(cap))


def _round2_task(task):
    """Round 2: ``(task id, S_f float32 [k], ends int32 [k, 2], path int32 [P, 2], path_off int64 [k + 1],
    distinct codes int64 [k] (counted up to ``target_min_distinct_codes``), skipped bool [k])``: the forward
    traceback of survivors (profiling mode: plus the task's timing record)."""
    tid, q_off, n, pairs = task
    st = _state()
    timer = _task_timer(st)
    qp = _query_prep(st, q_off, n)
    if timer is not None:
        timer.lap("prep")
    a = swkern.align(qp, st.codes, pairs[:, 0], st.params, lengths=pairs[:, 1], traceback=True,
                     block_floats=st.init.block_floats, max_pair_cells=st.init.max_pair_cells, kernel=st.kernel,
                     workspace=st.workspace, timings=None if timer is None else timer.t)
    if timer is not None:
        timer.lap("swkern")
    cap = st.init.spec.calibrated_domain.target_min_distinct_codes
    distinct = np.array([_distinct_codes(st.codes[o:o + m], cap) for o, m in pairs.tolist()], np.int64)
    if timer is None:
        return tid, a.scores, a.ends, a.path, a.path_off, distinct, a.skipped
    timer.lap("distinct")
    out = (tid, a.scores, a.ends, a.path, a.path_off, distinct, a.skipped)
    return out + (_task_record(st, timer, "round2", n, pairs, out),)


# =====================================================================================================================
# the pool
# =====================================================================================================================
@contextmanager
def _single_threaded_blas_env():
    """``*_NUM_THREADS=1`` in this process's environment while the pool's workers are spawned (they inherit it)."""
    saved = {k: os.environ.get(k) for k in _BLAS_THREAD_VARS}
    os.environ.update(dict.fromkeys(_BLAS_THREAD_VARS, "1"))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _blas_one_thread_here():
    limiter = _limit_blas_threads()
    if limiter is not None:
        _note_uncontrolled_blas()
    try:
        yield
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


@contextmanager
def _swkern_skip_warning_dropped():
    """In-process tasks: swkern's oversize-pair warning dropped, as in a pool worker (the parent logs the count)."""
    log = logging.getLogger(swkern.__name__)
    added = _DROP_SKIP_WARNING not in log.filters
    log.addFilter(_DROP_SKIP_WARNING)
    try:
        yield
    finally:
        if added:
            log.removeFilter(_DROP_SKIP_WARNING)


def _stop_workers(executor: ProcessPoolExecutor) -> None:
    """Terminate an executor's worker processes now, without waiting for their tasks (``terminate_workers`` from
    Python 3.14; before it, through the executor's process table), then release the executor."""
    terminate = getattr(executor, "terminate_workers", None)
    if callable(terminate):
        try:
            terminate()
        except Exception:  # noqa: BLE001  (an executor that is already broken or shut down)
            pass
    else:
        for proc in list((getattr(executor, "_processes", None) or {}).values()):
            if proc.is_alive():
                proc.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


_LOST_WORKER = ("esm-codes Stage 2: a worker process ended without returning its task. It was killed (for example "
                "by the out-of-memory killer: a worker holds the similarity matrix of its largest pair, up to "
                "max_pair_cells float32 cells, so fewer --align-workers need less memory) or it could not start (a "
                "script that runs this Stage 2 with a worker pool must guard its entry point with 'if __name__ == "
                "\"__main__\":', because spawn workers re-import the main module). The worker pool was stopped.")


class CodeAlignmentPool:
    """The persistent Stage-2 worker pool of one run (plan P7 design): ``spawn`` workers of a
    :class:`concurrent.futures.ProcessPoolExecutor`, created lazily at the first batch with more than one task (after
    ESM3 and the aggregator are freed) and started as its tasks are submitted, reused for both rounds and every
    batch, restarted only when the worker setup (database, query memmap, limits) changes.

    ``num_workers``: ``None`` uses up to every CPU (the CPU affinity mask is widened first, as for BLOSUM62), but
    starts only as many workers as the work justifies (one per ``min_worker_cells`` similarity cells of the batch,
    :data:`MIN_WORKER_CELLS`; a later, larger batch restarts the pool with more); ``0`` or ``1`` runs the task
    functions in-process; ``N > 1`` uses N workers, started as tasks are submitted (a small first batch starts fewer,
    and later batches start the rest). While no pool is running, a batch with a single task, or (``None``) with less
    work than one worker's share, runs in-process, but only when BLAS can be limited to one thread here
    (threadpoolctl); otherwise a pool of one worker is started, so the bits never depend on the batch.

    A worker that ends without returning its task (killed, or unable to start) raises :class:`WorkerPoolError` in
    the parent instead of a wait that never ends; the pool is stopped and the next batch starts a new one. Spawn
    workers re-import the main module: a script that uses a pool needs an ``if __name__ == "__main__":`` guard.
    Use it as a context manager, or call :meth:`close`.
    """

    def __init__(self, num_workers: Optional[int] = None, *, start_method: str = "spawn",
                 min_worker_cells: int = MIN_WORKER_CELLS):
        if num_workers is not None and (isinstance(num_workers, bool) or not isinstance(num_workers, int)
                                        or num_workers < 0):
            raise ValueError(f"num_workers must be None or an int >= 0, got {num_workers!r}")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"start method {start_method!r} is not available here ({mp.get_all_start_methods()})")
        self.num_workers = num_workers
        self.start_method = start_method
        self.min_worker_cells = _positive_int("min_worker_cells", min_worker_cells)
        self._pool: Optional[ProcessPoolExecutor] = None
        self._init: Optional[_WorkerInit] = None
        self._requested: Optional[int] = None
        self._in_process: Optional[_WorkerInit] = None
        self.workers = 0          # worker processes the running pool may start (0 = in-process or not started)
        self.starts = 0           # process pools started so far

    def __enter__(self) -> "CodeAlignmentPool":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(terminate=exc_type is not None)

    def __repr__(self):
        return (f"CodeAlignmentPool(num_workers={self.num_workers!r}, start_method={self.start_method!r}, "
                f"workers={self.workers})")

    @property
    def serial(self) -> bool:
        return self.num_workers is not None and self.num_workers <= 1

    def requested_workers(self) -> int:
        """The worker budget: 1 when serial, else ``num_workers`` or every usable CPU (after widening affinity)."""
        if self.serial:
            return 1
        if self._requested is None:
            _expand_cpu_affinity()
            self._requested = max(1, _available_cpus() if self.num_workers is None else int(self.num_workers))
        return self._requested

    def plan_workers(self, cells: int) -> int:
        """The workers a batch of ``cells`` similarity cells is planned for (task sizes) and runs on: 1 when serial,
        ``num_workers`` when given, else (``None``) one per ``min_worker_cells`` cells, at least the running pool's
        workers and at most the budget (:meth:`requested_workers`)."""
        budget = self.requested_workers()
        if self.serial or self.num_workers is not None:
            return budget
        work = max(1, -(-max(int(cells), 0) // self.min_worker_cells))
        return min(budget, max(work, self.workers))

    @property
    def processes(self) -> int:
        """Worker processes the running pool has started so far (the executor starts them as tasks are
        submitted)."""
        return len(getattr(self._pool, "_processes", None) or {}) if self._pool is not None else 0

    def run(self, fn, tasks: list, init: _WorkerInit, profile: Optional[stage2_profile.Stage2Profile] = None,
            workers: Optional[int] = None) -> Iterator:
        """The results of ``fn`` over ``tasks``, in completion order, on ``workers`` workers (default: the budget;
        the aligner passes :meth:`plan_workers`). ``profile`` (profiling mode) receives the executor start-up and
        each task's submit time."""
        if not tasks:
            return
        if self._pool is not None and self._init != init:
            logger.info("esm-codes Stage 2: the worker setup changed; restarting the worker pool")
            self.close()
        want = self.requested_workers() if workers is None else max(1, min(int(workers), self.requested_workers()))
        if self._pool is None:
            if self.serial or (min(want, len(tasks)) <= 1 and _blas_limit_available()):
                yield from self._run_here(fn, tasks, init)
                return
            self._start(init, want, profile)
        elif want > self.workers:
            logger.info(f"esm-codes Stage 2: this batch needs {want} workers; restarting the pool of {self.workers}")
            self.close()
            self._start(init, want, profile)
        yield from self._run_in_pool(fn, tasks, profile)

    def _run_here(self, fn, tasks, init):
        if _STATE is None or _STATE.init != init or _STATE.error is not None:
            _init_worker(init, raise_errors=True)
        if self._in_process is None and not _blas_limit_available():
            logger.info("esm-codes Stage 2 runs in-process without threadpoolctl: BLAS keeps this process's thread "
                        "count (set OMP/OPENBLAS/MKL/VECLIB_*_NUM_THREADS=1 for the bits of a worker pool; S16 holds "
                        "either way)")
        self._in_process = init
        with _blas_one_thread_here(), _swkern_skip_warning_dropped():
            for task in tasks:
                yield fn(task)

    def _run_in_pool(self, fn, tasks, profile=None):
        """Submit every task (longest first: the caller's order), then yield the results as they complete. Tasks not
        started yet are cancelled when the caller stops early (an error); a lost worker raises
        :class:`WorkerPoolError`."""
        futures = []
        try:
            try:
                # The executor spawns its workers as tasks are submitted: they inherit *_NUM_THREADS=1 from here.
                with _single_threaded_blas_env():
                    for task in tasks:
                        if profile is not None:
                            profile.submitted(task)
                        futures.append(self._pool.submit(fn, task))
                for future in as_completed(futures):
                    yield future.result()
            except BrokenProcessPool as exc:
                self.close(terminate=True)
                raise WorkerPoolError(_LOST_WORKER) from exc
        finally:
            for future in futures:
                future.cancel()

    def session(self, init: _WorkerInit, cells: int, tasks: int,
                profile: Optional[stage2_profile.Stage2Profile] = None) -> "_Session":
        """A :class:`_Session` for one aligner call whose first batch has ``cells`` similarity cells in ``tasks``
        round-1 tasks: in-process (serial, or no pool running and the batch qualifies, as in :meth:`run`), else on
        this pool, started or grown to :meth:`plan_workers` (a pool with another worker setup is restarted)."""
        if self._pool is not None and self._init != init:
            logger.info("esm-codes Stage 2: the worker setup changed; restarting the worker pool")
            self.close()
        want = self.plan_workers(cells)
        if self._pool is None:
            if self.serial or (min(want, tasks) <= 1 and _blas_limit_available()):
                return _Session(self, init, True, profile)
            self._start(init, want, profile)
        elif want > self.workers:
            logger.info(f"esm-codes Stage 2: this batch needs {want} workers; restarting the pool of {self.workers}")
            self.close()
            self._start(init, want, profile)
        return _Session(self, init, False, profile)

    def _start(self, init: _WorkerInit, workers: int, profile=None):
        t0 = time.perf_counter()
        ctx = mp.get_context(self.start_method)
        # max_workers is the pool's size for the rest of the run: the executor starts the workers as tasks are
        # submitted, so a small first batch starts fewer and a larger batch later starts the rest
        self._pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_pool_worker,
                                         initargs=(init,))
        self._init, self.workers = init, workers
        self.starts += 1
        if profile is not None:
            profile.executor_created(time.perf_counter() - t0, workers)
        logger.info(f"esm-codes Stage 2: a pool of up to {workers} {self.start_method} workers")

    def close(self, terminate: bool = False) -> None:
        """Stop the workers (``terminate``: at once, without waiting for queued or running tasks) and release the
        in-process state."""
        pool, self._pool = self._pool, None
        if pool is not None:
            self._init, self.workers = None, 0
            if terminate:
                _stop_workers(pool)
            else:
                pool.shutdown(wait=True)
        if self._in_process is not None and _STATE is not None and _STATE.init == self._in_process:
            _reset_worker_state()
        self._in_process = None


class _Session:
    """The tasks of one aligner call: submitted one at a time and their results taken in completion order, on the
    pool's executor (at most :attr:`window` in flight: twice the pool's workers, so every worker has its next task
    queued while the parent bounds what it holds) or in-process (each task runs when it is submitted, one in flight).
    ``with`` it: leaving cancels the tasks not started yet. A worker that ends without returning its task raises
    :class:`WorkerPoolError` (the pool is stopped)."""

    def __init__(self, pool: CodeAlignmentPool, init: _WorkerInit, in_process: bool,
                 profile: Optional[stage2_profile.Stage2Profile]):
        self.pool, self.init, self.in_process, self.profile = pool, init, in_process, profile
        self.done: "queue.SimpleQueue" = queue.SimpleQueue()
        self.pending: set = set()
        self.local: deque = deque()
        self.inflight = 0
        self._exit = None
        if in_process:
            if _STATE is None or _STATE.init != init or _STATE.error is not None:
                _init_worker(init, raise_errors=True)
            if pool._in_process is None and not _blas_limit_available():
                logger.info("esm-codes Stage 2 runs in-process without threadpoolctl: BLAS keeps this process's "
                            "thread count (set OMP/OPENBLAS/MKL/VECLIB_*_NUM_THREADS=1 for the bits of a worker pool; "
                            "S16 holds either way)")
            pool._in_process = init
            self.workers, self.window = 1, 1
        else:
            self.workers = pool.workers
            self.window = max(2 * pool.workers, 2)

    def __enter__(self) -> "_Session":
        if self.in_process:
            blas, warn = _blas_one_thread_here(), _swkern_skip_warning_dropped()
            blas.__enter__()
            warn.__enter__()
            self._exit = (warn, blas)
        return self

    def __exit__(self, exc_type, exc, tb):
        for future in self.pending:
            future.cancel()
        self.pending.clear()
        if self._exit is not None:
            for cm in self._exit:
                cm.__exit__(None, None, None)
            self._exit = None

    @property
    def full(self) -> bool:
        return (len(self.local) if self.in_process else self.inflight) >= self.window

    def submit(self, fn, task, tag, rnd: Optional[int] = None) -> None:
        """Run ``fn(task)``; its result comes back from :meth:`next` with ``tag`` (``rnd``: the profile's round)."""
        if self.in_process:
            self.local.append((tag, fn(task)))
            return
        prof, executor = self.profile, self.pool._pool
        if prof is not None:
            prof.submitted(task, rnd)
        try:
            if self.pool.processes < self.pool.workers:
                # the executor may spawn a worker for this task: it inherits *_NUM_THREADS=1 from here
                with _single_threaded_blas_env():
                    future = executor.submit(fn, task)
            else:
                future = executor.submit(fn, task)
        except BrokenProcessPool as exc:
            self.pool.close(terminate=True)
            raise WorkerPoolError(_LOST_WORKER) from exc
        self.pending.add(future)
        self.inflight += 1
        future.add_done_callback(lambda f, tag=tag: self.done.put((tag, f)))

    def next(self):
        """``(tag, result)`` of a finished task (the oldest in-process one, else the next to complete)."""
        if self.local:
            return self.local.popleft()
        tag, future = self.done.get()
        self.pending.discard(future)
        self.inflight -= 1
        try:
            return tag, future.result()
        except BrokenProcessPool as exc:
            self.pool.close(terminate=True)
            raise WorkerPoolError(_LOST_WORKER) from exc

    @property
    def idle(self) -> bool:
        return not self.local and self.inflight == 0


def _as_pool(pool) -> Tuple[CodeAlignmentPool, bool]:
    """``(pool, owned)``: a :class:`CodeAlignmentPool` as given, or a new one (closed by the caller) for ``None``
    (in-process) or a worker count."""
    if isinstance(pool, CodeAlignmentPool):
        return pool, False
    if pool is None:
        return CodeAlignmentPool(0), True
    if isinstance(pool, int) and not isinstance(pool, bool):
        return CodeAlignmentPool(pool), True
    raise TypeError(f"pool must be a CodeAlignmentPool, a worker count or None, got {type(pool).__name__}")


# =====================================================================================================================
# sequences and path metrics (numpy only)
# =====================================================================================================================
def _sanitize_table() -> bytes:
    table = bytearray(b"X" * 256)
    for b in range(128):
        ch = chr(b).upper()
        if ch in _BIOTITE_ALPHABET:
            table[b] = ord(ch)
    return bytes(table)


_SANITIZE = _sanitize_table()
_STANDARD_BYTES = STANDARD_AMINO_ACIDS.encode("ascii")


def sanitized_bytes(sequence: str) -> bytes:
    """``alignment.sanitize_sequence(sequence)`` as ASCII bytes, for the normalised sequences Stage 2 sees (no
    surrounding whitespace): every residue upper-cased and anything outside the biotite protein alphabet made ``X``,
    one byte per residue (a non-ASCII character becomes ``X``)."""
    return sequence.encode("ascii", "replace").translate(_SANITIZE)


def nonstandard_count(sequence: str) -> int:
    """Residues outside the 20 standard amino acids (S13, S15)."""
    return len(sequence.encode("ascii", "replace").translate(None, _STANDARD_BYTES))


def path_columns(path: np.ndarray, path_off: np.ndarray, q_bytes: np.ndarray, t_bytes: np.ndarray,
                 t_starts: np.ndarray) -> Dict[str, np.ndarray]:
    """S12 column tallies of K forward paths at once (vectorised over every column of every path).

    ``path`` int [P, 2] (query i, target j; 0-based, -1 = gap) with path k at ``path[path_off[k]:path_off[k + 1]]``;
    ``q_bytes`` the sanitised query as uint8; ``t_bytes`` the sanitised targets concatenated, target k starting at
    ``t_starts[k]``. Returns int64 [K] arrays ``aln_len``, ``n_ident``, ``n_aligned`` (pairs), ``mismatch``,
    ``gap_open`` (maximal gap runs, each side counted separately), ``q_aligned`` / ``t_aligned`` (columns with a
    query / target residue) and the 1-based bounds ``q_start``, ``q_end``, ``t_start``, ``t_end`` (0 for an empty
    path). Equal, path by path, to :func:`foldmatch.search.swkern.path_metrics`."""
    path = np.asarray(path, np.int64).reshape(-1, 2)
    path_off = np.asarray(path_off, np.int64)
    k = len(path_off) - 1
    lens = np.diff(path_off)
    seg = np.repeat(np.arange(k), lens)
    qi, tj = path[:, 0], path[:, 1]
    q_gap, t_gap = qi < 0, tj < 0
    both = ~q_gap & ~t_gap
    ident = np.zeros(len(path), bool)
    if both.any():
        ident[both] = q_bytes[qi[both]] == t_bytes[np.asarray(t_starts, np.int64)[seg[both]] + tj[both]]

    def count(mask):
        return np.bincount(seg, weights=mask, minlength=k).astype(np.int64)

    nonempty = lens > 0
    head = np.zeros(len(path), bool)
    head[path_off[:-1][nonempty]] = True
    q_open = q_gap & (head | ~np.r_[False, q_gap[:-1]])
    t_open = t_gap & (head | ~np.r_[False, t_gap[:-1]])
    out = dict(aln_len=lens.astype(np.int64), n_ident=count(ident), n_aligned=count(both),
               gap_open=count(q_open) + count(t_open), q_aligned=count(~q_gap), t_aligned=count(~t_gap))
    out["mismatch"] = out["n_aligned"] - out["n_ident"]
    big = np.int64(1) << 62
    starts = path_off[:-1][nonempty]
    for name, idx, gap, aligned in (("q", qi, q_gap, out["q_aligned"]), ("t", tj, t_gap, out["t_aligned"])):
        lo, hi = np.zeros(k, np.int64), np.zeros(k, np.int64)
        if starts.size:
            lo[nonempty] = np.minimum.reduceat(np.where(gap, big, idx), starts)
            hi[nonempty] = np.maximum.reduceat(np.where(gap, -1, idx), starts)
        has = aligned > 0
        out[f"{name}_start"] = np.where(has, lo + 1, 0)
        out[f"{name}_end"] = np.where(has, hi + 1, 0)
    return out


def _gapped(idx: np.ndarray, seq: np.ndarray) -> str:
    out = np.full(len(idx), ord("-"), np.uint8)
    res = idx >= 0
    out[res] = seq[idx[res]]
    return out.tobytes().decode("ascii")


class _PathStrings:
    """The CIGAR and gapped-string columns of many paths at once (``path`` int [P, 2] with path k at
    ``path[path_off[k]:path_off[k + 1]]``): per path the same strings as :func:`swkern.cigar_from_path` and
    :func:`_gapped` of its segment, from one vectorised pass over all the columns. ``q_bytes`` the sanitised query,
    ``t_bytes`` the sanitised targets concatenated (target k from ``t_starts[k]``)."""

    def __init__(self, path, path_off, q_bytes, t_bytes, t_starts, *, cigar: bool, qaln: bool, taln: bool):
        path = np.asarray(path).reshape(-1, 2)
        self.path_off = np.asarray(path_off, np.int64)
        self.cigar_tokens = self.run_starts = self.qaln = self.taln = None
        if cigar:
            ops = swkern.path_ops(path)
            boundary = np.ones(len(ops), bool)
            if len(ops) > 1:
                boundary[1:] = ops[1:] != ops[:-1]
            heads = self.path_off[:-1][np.diff(self.path_off) > 0]
            boundary[heads] = True                          # every path starts a run
            starts = np.flatnonzero(boundary)
            runs = np.diff(np.append(starts, len(ops)))
            self.run_starts = starts
            self.cigar_tokens = [f"{r}{chr(o)}" for r, o in zip(runs.tolist(), ops[starts].tolist())]
        if qaln or taln:
            qi, tj = path[:, 0], path[:, 1]
            if qaln:
                g = np.full(len(path), ord("-"), np.uint8)
                res = qi >= 0
                g[res] = q_bytes[qi[res]]
                self.qaln = g.tobytes()
            if taln:
                g = np.full(len(path), ord("-"), np.uint8)
                res = tj >= 0
                seg = np.repeat(np.arange(len(self.path_off) - 1), np.diff(self.path_off))
                g[res] = t_bytes[np.asarray(t_starts, np.int64)[seg[res]] + tj[res]]
                self.taln = g.tobytes()

    def cigar(self, lo: int, hi: int) -> str:
        if hi <= lo:
            return ""
        a, b = np.searchsorted(self.run_starts, (lo, hi)).tolist()
        return "".join(self.cigar_tokens[a:b])


# =====================================================================================================================
# the aligner
# =====================================================================================================================
class _Query:
    """One query of a batch: its encoding rows, candidates and the results of both rounds."""
    __slots__ = ("qid", "off", "n", "seq", "cand_ids", "cand_emb", "cand_off", "cand_len", "s_f", "s_r", "skipped",
                 "panel_f", "panel_r", "q90", "s", "s_prime", "lne", "bits", "surv", "parts", "inside", "r1_left",
                 "r2_left", "sequences")

    def __init__(self, qid, off, n, seq, cand_ids, cand_emb, inside: bool):
        self.qid, self.off, self.n, self.seq = qid, off, n, seq
        self.cand_ids, self.cand_emb = cand_ids, cand_emb
        self.inside = inside           # the query side of S13
        self.q90 = None
        self.parts: list = []
        self.r1_left = self.r2_left = 0
        self.sequences = None


def _log_threshold(max_evalue) -> Optional[float]:
    """``ln(max_evalue)``, ``-inf`` for 0, or None for no E filter (None or +inf)."""
    if max_evalue is None:
        return None
    try:
        value = float(max_evalue)
    except (TypeError, ValueError):
        raise ValueError(f"max_evalue must be a number or None, got {max_evalue!r}") from None
    if math.isnan(value) or value < 0:
        raise ValueError(f"max_evalue must be >= 0 (or None / inf for no E-value filter), got {max_evalue!r}")
    if value == math.inf:
        return None
    return -math.inf if value == 0 else math.log(value)


def _check_fraction(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}") from None
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return v


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


def align_candidates_esm_codes(
        query_index,
        prefilter_results: Mapping[str, Tuple[Sequence[str], Sequence[float]]],
        code_store,
        fetch_subject_sequences,
        prepared,
        db_residues: int,
        pool,
        *,
        query_sequences: Mapping[str, str],
        min_seq_identity: float = 0.0,
        min_coverage: float = 0.0,
        max_evalue: Optional[float] = 1e-3,
        output_needs,
        block_floats: int = swkern.DEFAULT_BLOCK_FLOATS,
        task_cells: int = DEFAULT_TASK_CELLS,
        max_pair_cells: int = swkern.DEFAULT_MAX_PAIR_CELLS,
        query_batch: int = DEFAULT_QUERY_BATCH,
        panel_policy: str = "auto",
        stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Hit]]:
    """esm-codes Stage 2 (module docstring): ``{query_id: [Hit, ...]}`` in ``prefilter_results`` order, each list
    sorted by S descending, then the embedding score descending, then the target id (S14).

    Args:
        query_index: the run's :class:`~foldmatch.search.query_encodings.QueryEncodingIndex` (``entries``,
            ``memmap()``, ``memmap_args()``). Every query in ``prefilter_results`` must have an encoding whose length
            and ``seq_sha1`` match ``query_sequences``; queries are processed in memmap order.
        prefilter_results: Stage-1 output ``{query_id: ([target_ids], [embedding scores])}``.
        code_store: the database's :class:`~foldmatch.search.residue_code_store.ResidueCodeStore` (complete;
            ``lookup_rows()`` by id, its ``.bin`` mapped by the workers).
        fetch_subject_sequences: ``ids -> {id: sequence}`` (``SequenceStore.fetch``), called once per query with
            survivors, for its survivors only; every survivor's sequence must be there, with as many residues as its
            codes.
        prepared: the :class:`~foldmatch.search.scoring_system.PreparedSystem` of the database's system.
        db_residues: N, the database's residue count (manifest ``total_residues``); E-values need N > 0.
        pool: a :class:`CodeAlignmentPool` (reused, not closed), a worker count (a pool for this call only) or
            ``None`` (in-process). Its spawn workers re-import the main module: a script that calls this function
            with a pool must guard its entry point with ``if __name__ == "__main__":``. A worker that ends without
            returning its task (killed, or unable to start) raises :class:`WorkerPoolError`.
        query_sequences: ``{query_id: sequence}`` (normalised here, S15), for identity, the strings and S13.
        min_seq_identity / min_coverage: keep a hit only if ``fident >= min_seq_identity`` and
            ``min(qcov, tcov) >= min_coverage`` (D8: both 0 under esm-codes).
        max_evalue: survivors are ``ln E <= ln(max_evalue)``; ``None`` or ``inf`` traces every candidate.
        output_needs: an :class:`OutputNeeds` or the ``--format-output`` field names: the strings are built only for
            the requested columns, and the panel policy looks at the E-value columns.
        block_floats / max_pair_cells: kernel blocking (``swkern``); a pair above ``max_pair_cells`` cells is skipped
            (no score, no hit) and counted.
        task_cells / query_batch: task-size cap and queries per batch (scheduling only; the output does not depend
            on them).
        panel_policy: ``"auto"`` (D11) skips the panel pass when there is no finite ``max_evalue`` and no
            evalue / bits / q90 / rawadj column: hits are still ranked by S, and those columns render empty;
            ``"always"`` runs it; ``"never"`` skips it and refuses a finite ``max_evalue``.
        stats: a dict to receive the run's counters (also logged once at INFO): pairs, survivors, tasks, workers,
            the per-query Q90, the fully degenerate queries (S13b) and the calibrated-domain counts (S13); in the
            profiling mode (``FOLDMATCH_STAGE2_PROFILE=1``, :mod:`~foldmatch.search.stage2_profile`) also
            ``stats["profile"]``, the time breakdown of the parent and the workers (the hits do not change).
    """
    t_start = time.perf_counter()
    prof = stage2_profile.Stage2Profile() if stage2_profile.profiling_enabled() else None
    needs = OutputNeeds.coerce(output_needs)
    min_seq_identity = _check_fraction("min_seq_identity", min_seq_identity)
    min_coverage = _check_fraction("min_coverage", min_coverage)
    ln_max = _log_threshold(max_evalue)
    block_floats = _positive_int("block_floats", block_floats)
    task_cells = _positive_int("task_cells", task_cells)
    max_pair_cells = _positive_int("max_pair_cells", max_pair_cells)
    query_batch = _positive_int("query_batch", query_batch)
    if isinstance(db_residues, bool) or not isinstance(db_residues, (int, np.integer)) or db_residues < 0:
        raise ValueError(f"db_residues must be an int >= 0 (the database's total residues), got {db_residues!r}")
    db_residues = int(db_residues)
    if panel_policy not in PANEL_POLICIES:
        raise ValueError(f"panel_policy must be one of {PANEL_POLICIES}, got {panel_policy!r}")
    if panel_policy == "never" and ln_max is not None:
        raise ValueError("panel_policy='never' cannot filter by E-value: pass max_evalue=None (or inf)")
    need_panel = panel_policy == "always" or (panel_policy == "auto" and (ln_max is not None or needs.significance))

    system = prepared.system
    spec = prepared.for_worker()
    kernel = swkern.load()
    run = _Run(spec, needs, need_panel, ln_max, min_seq_identity, min_coverage, db_residues, task_cells)
    run.prof = prof
    results: Dict[str, List[Hit]] = {qid: [] for qid in prefilter_results}
    if prof is not None:
        t_phase = time.perf_counter()
        prof.add("setup", t_phase - t_start)
    queries = _collect_queries(query_index, prefilter_results, query_sequences, run)
    if prof is not None:
        prof.add("collect", time.perf_counter() - t_phase)
    if queries and need_panel and db_residues <= 0:
        raise ValueError("E-values need a non-empty database (db_residues > 0), but the database has "
                         f"{db_residues} residues; pass max_evalue=None and panel_policy='never' to rank by S only")
    logger.info(
        f"esm-codes Stage 2: system {system.name} (digest {system.digest()}, calibration {system.calibration_version}, "
        f"bundle {system.bundle_repo}@{system.bundle_revision}), kernel {kernel.info()}, N = {db_residues} database "
        f"residues, {len(prefilter_results)} queries ({len(queries)} to align), panel "
        f"{'on' if need_panel else 'skipped'} (policy {panel_policy})")
    pool, owned = _as_pool(pool)
    try:
        if queries:
            _check_code_store(code_store)
            manifest = code_store.manifest()
            init = _WorkerInit(isa=kernel.isa, spec=spec,
                               codes=(str(code_store.bin_path), int(manifest["total_residues"]),
                                      int(manifest["code_bytes"])),
                               queries=tuple(query_index.memmap_args()), block_floats=block_floats,
                               max_pair_cells=max_pair_cells, profile=prof is not None)
            if need_panel:
                _codes, panel_offsets = spec.load_panel()
                run.set_panel(panel_offsets)
            _Dataflow(queries, run, pool, init, code_store, fetch_subject_sequences, results, query_batch).execute()
    except BaseException:
        if owned:
            pool.close(terminate=True)
        raise
    if owned:
        if prof is not None:
            t_phase = time.perf_counter()
        pool.close()
        if prof is not None:
            prof.add("pool_close", time.perf_counter() - t_phase)
    if prof is not None:
        t_phase = time.perf_counter()
    run.finish(results, time.perf_counter() - t_start, max_pair_cells, stats)
    if prof is not None:
        prof.add("finish", time.perf_counter() - t_phase)
        summary = prof.summary(run.workers)
        logger.info(stage2_profile.Stage2Profile.log_line(summary))
        if stats is not None:
            stats["profile"] = summary
    return results


def _check_code_store(code_store) -> None:
    """The workers map the ``.bin`` with the manifest's shape; :meth:`open_memmap` checks its size first."""
    codes = code_store.open_memmap()
    del codes


class _Run:
    """Counters and settings of one :func:`align_candidates_esm_codes` call."""

    def __init__(self, spec, needs, need_panel, ln_max, min_id, min_cov, db_residues, task_cells):
        self.spec, self.needs, self.need_panel, self.ln_max = spec, needs, need_panel, ln_max
        self.min_id, self.min_cov, self.db_residues, self.task_cells = min_id, min_cov, db_residues, task_cells
        self.domain = spec.calibrated_domain
        self.panel_pairs = None
        self.panel_lengths: List[int] = []
        self.panel_residues = 0
        self.queries = self.no_candidates = 0
        self.degenerate: List[str] = []
        self.batches = self.round1_tasks = self.round2_tasks = self.workers = 0
        self.candidate_pairs = self.panel_pair_count = self.skipped_pairs = self.survivors = 0
        self.largest_skipped = (0, 0)  # (query length, target length) of the largest skipped pair
        self.q90: Dict[str, float] = {}
        self.query_domain = dict(aligned=0, short=0, long=0, nonstandard=0)
        self.hit_domain = dict(hits=0, ecal0=0, query_outside=0, target_long=0, target_nonstandard=0,
                               target_low_distinct=0)
        self.prof: Optional[stage2_profile.Stage2Profile] = None     # profiling mode only

    def query_inside(self, seq: str) -> bool:
        """The query side of S13 (length, non-standard fraction), counted for the once-per-run warning."""
        d, qd, n = self.domain, self.query_domain, len(seq)
        short, long = n < d.query_min_length, n > d.query_max_length
        nonstandard = nonstandard_count(seq) / n > d.max_nonstandard_fraction
        qd["aligned"] += 1
        qd["short"] += int(short)
        qd["long"] += int(long)
        qd["nonstandard"] += int(nonstandard)
        return not (short or long or nonstandard)

    def set_panel(self, offsets: np.ndarray):
        offsets = np.asarray(offsets, np.int64)
        lengths = np.diff(offsets)
        self.panel_pairs = np.column_stack((offsets[:-1], lengths)).astype(np.int64)
        self.panel_lengths = lengths.tolist()
        self.panel_residues = int(lengths.sum())

    def finish(self, results, seconds: float, max_pair_cells: int, stats: Optional[dict]):
        qd, hd = self.query_domain, self.hit_domain
        logger.info(
            f"esm-codes Stage 2 done in {seconds:.2f}s: {self.candidate_pairs} candidate pairs"
            f"{f' and {self.panel_pair_count} panel pairs' if self.need_panel else ''} scored in "
            f"{self.round1_tasks} round-1 tasks, {self.survivors} survivors traced back in {self.round2_tasks} "
            f"round-2 tasks ({self.batches} batches, {self.workers or 'no'} worker processes); {hd['hits']} hits, "
            f"{hd['hits'] - hd['ecal0']} inside the calibrated domain (S13); queries: {qd['aligned']} aligned, "
            f"{self.no_candidates} without candidates, {len(self.degenerate)} fully degenerate (S13b, no hits); "
            f"{self.skipped_pairs} pairs skipped above max_pair_cells={max_pair_cells}")
        if self.skipped_pairs:
            n, m = self.largest_skipped
            logger.warning(
                f"esm-codes Stage 2: {self.skipped_pairs} candidate pair(s) skipped because their similarity matrix "
                f"(query length x target length) exceeds max_pair_cells={max_pair_cells} cells (the largest: {n} x "
                f"{m}); they get no score and no hit")
        outside = qd["short"] + qd["long"] + qd["nonstandard"]
        if outside or hd["ecal0"] or self.degenerate:
            d = self.domain
            logger.warning(
                "esm-codes: E-values are calibrated only inside the calibrated domain (S13: queries of "
                f"{d.query_min_length}-{d.query_max_length} aa, targets up to {d.target_max_length} aa, at most "
                f"{d.max_nonstandard_fraction:.0%} non-standard residues, targets with at least "
                f"{d.target_min_distinct_codes} distinct codes); outside it hits are reported with ecal=0. This run: "
                f"{qd['short']} queries shorter than {d.query_min_length} aa, {qd['long']} longer than "
                f"{d.query_max_length} aa, {qd['nonstandard']} with too many non-standard residues; {hd['ecal0']} of "
                f"{hd['hits']} hits with ecal=0 ({hd['query_outside']} for the query, {hd['target_long']} target too "
                f"long, {hd['target_nonstandard']} target non-standard, {hd['target_low_distinct']} target with too "
                f"few distinct codes); {len(self.degenerate)} fully degenerate queries (one residue, or all encoding "
                f"rows identical) report no hits (S13b)"
                + (f": {', '.join(self.degenerate[:_EXAMPLES])}" if self.degenerate else ""))
        if stats is not None:
            stats.update(
                queries=self.queries, queries_without_candidates=self.no_candidates,
                degenerate_queries=list(self.degenerate), panel=self.need_panel, batches=self.batches,
                round1_tasks=self.round1_tasks, round2_tasks=self.round2_tasks, workers=self.workers,
                candidate_pairs=self.candidate_pairs, panel_pairs=self.panel_pair_count,
                skipped_pairs=self.skipped_pairs, survivors=self.survivors, q90=dict(self.q90),
                query_domain=dict(qd), hit_domain=dict(hd), hits=sum(len(h) for h in results.values()))


def _is_degenerate(q16: np.ndarray) -> bool:
    """S13b: one residue, or every q fp16 row bitwise identical (then S_f = S_r for every target)."""
    if len(q16) < 2:
        return True
    rows = np.ascontiguousarray(q16).view(np.uint16)
    return bool(np.all(rows == rows[0]))


def _collect_queries(query_index, prefilter_results, query_sequences, run: _Run) -> List[_Query]:
    """The queries to align, in memmap order: checked against their encodings; those without candidates and the
    fully degenerate ones (S13b) left out (and counted)."""
    entries = query_index.entries
    memmap = None
    out: List[_Query] = []
    for qid, value in prefilter_results.items():
        run.queries += 1
        subject_ids, emb_scores = value
        subject_ids, emb_scores = list(subject_ids), [float(s) for s in emb_scores]
        if len(subject_ids) != len(emb_scores):
            raise ValueError(f"query {qid!r}: {len(subject_ids)} candidate ids but {len(emb_scores)} scores")
        entry = entries.get(qid)
        if entry is None:
            raise ValueError(f"query {qid!r} has no query encoding in {query_index.path} (the ESM3 pass did not "
                             "encode it)")
        seq = query_sequences.get(qid)
        if seq is None:
            raise ValueError(f"query {qid!r} has no sequence")
        try:
            seq = normalize_sequence(seq)                  # S15 (a no-op for the run's normalised sequences)
        except ValueError as e:
            raise ValueError(f"query {qid!r}: {e} (S15)") from e
        if len(seq) != entry.length:
            raise ValueError(f"query {qid!r}: its sequence has {len(seq)} residues, its encoding {entry.length} rows")
        if seq_sha1(seq) != bytes(entry.seq_sha1):
            raise ValueError(f"query {qid!r}: its sequence is not the one its encoding was computed from (seq_sha1 "
                             "differs)")
        if not subject_ids:
            run.no_candidates += 1
            continue
        if memmap is None:
            memmap = query_index.memmap()
        q16 = memmap[entry.offset:entry.offset + entry.length]
        if not np.all(np.isfinite(q16)):
            raise ValueError(f"query {qid!r}: its encoding contains NaN or Inf (S5)")
        if _is_degenerate(q16):
            run.degenerate.append(qid)
            continue
        out.append(_Query(qid, int(entry.offset), int(entry.length), seq, subject_ids, emb_scores,
                          run.query_inside(seq)))
    out.sort(key=lambda q: q.off)
    return out


class _Batch:
    """One batch of queries in the dataflow: its round-1 task size, the queries not finished yet, its profile
    rounds."""
    __slots__ = ("index", "queries", "target", "open", "r1_round", "r2_round", "r1_left", "r2_left")

    def __init__(self, index: int, queries: List[_Query]):
        self.index, self.queries = index, queries
        self.target = 0
        self.open = len(queries)
        self.r1_round = self.r2_round = None
        self.r1_left = self.r2_left = 0


class _Dataflow:
    """Stage 2 of one call as a dataflow on a :class:`_Session` instead of two barriers per batch:

    * a batch's round-1 tasks (panel and candidate chunks of every query, longest first) are queued when the batch
      is opened (code lookup, task plan); the next batch is opened, at most two at a time, while the workers still
      have this one's tasks, so its lookup and plan overlap their work;
    * a query whose round-1 tasks are all back gets its statistics (S, Q90, ln E, survivors) at once, its survivors'
      sequences, and its round-2 tasks, queued ahead of any round-1 task;
    * a query whose round-2 tasks are all back gets its hits at once (``_hits``), and its paths are released: the
      parent holds the round-2 payload of the queries in flight only (at most the session's window of tasks), not of
      a whole batch, and builds hits while the workers compute;
    * a batch that needs more workers than the session has (the pool sized by default: more work than any batch
      before; or an in-process session and a batch with more than one worker's share) is opened once the workers are
      idle, on a session of the grown pool (at least twice the workers, up to the budget; :meth:`_grow`), so a
      small first batch does not keep the whole call in-process or on a few workers.

    Scheduling only: every pair is the same task function on the same worker setup, so the output is the same, bit for
    bit, as with any other order, task size or worker count (S16)."""

    def __init__(self, queries: List[_Query], run: _Run, pool: CodeAlignmentPool, init: _WorkerInit, code_store,
                 fetch_subject_sequences, results: Dict[str, List[Hit]], query_batch: int):
        self.run, self.pool, self.init = run, pool, init
        self.code_store, self.fetch, self.results = code_store, fetch_subject_sequences, results
        self.pending = [queries[b0:b0 + query_batch] for b0 in range(0, len(queries), query_batch)]
        self.pending.reverse()                       # pop() gives the next batch
        self.round1: deque = deque()                 # (fn, task, tag, rnd) to submit, low priority
        self.round2: deque = deque()                 # high priority
        self.open_batches: List[_Batch] = []
        self.tid = 0
        self.session: Optional[_Session] = None

    def _cells(self, batch_queries) -> int:
        run = self.run
        return sum(q.n * ((run.panel_residues if run.need_panel else 0) + int(q.cand_len.sum()))
                   for q in batch_queries)

    def execute(self):
        run, prof = self.run, self.run.prof
        first = self._lookup(self.pending.pop())
        cells = self._cells(first)
        self._enter_session(cells, self._round1_tasks_estimate(first, cells))
        staged = None            # (batch, cells) of a batch waiting for the workers to be idle, to grow the pool
        try:
            self._open(first, cells)
            while True:
                session = self.session
                if (staged is None and self.pending and len(self.open_batches) < 2
                        and len(self.round1) < 4 * session.window):
                    q = self._lookup(self.pending.pop())
                    staged = self._open_or_stage(q, self._cells(q))
                while not session.full and (self.round2 or self.round1):
                    fn, task, tag, rnd = (self.round2 or self.round1).popleft()
                    if prof is not None:
                        prof.round_started(rnd)
                    session.submit(fn, task, tag, rnd)
                if session.idle:
                    if self.open_batches:                   # every open query has a task in flight or queued
                        raise InternalError(f"esm-codes Stage 2: {sum(b.open for b in self.open_batches)} queries "
                                            "left unfinished with no task in flight")
                    if staged is None:
                        if not self.pending:
                            break
                        q = self._lookup(self.pending.pop())
                        staged = self._open_or_stage(q, self._cells(q))
                    if staged is not None:
                        self._grow(*staged)
                        self._open(*staged)
                        staged = None
                    continue
                tag, res = session.next()
                if prof is not None:
                    res = prof.received(res, tag[1].r1_round if tag[0] == 1 else tag[1].r2_round)
                if tag[0] == 1:
                    self._round1_result(tag, res)
                else:
                    self._round2_result(tag, res)
        except BaseException as exc:
            self._exit_session(exc)
            raise
        self._exit_session(None)

    # ---------------------------------------------------------------------------------------------------- sessions
    def _enter_session(self, cells: int, tasks: int) -> None:
        session = self.pool.session(self.init, cells, tasks, self.run.prof)
        session.__enter__()
        self.session = session

    def _exit_session(self, exc: Optional[BaseException]) -> None:
        session, self.session = self.session, None
        if session is not None:
            self.run.workers = max(self.run.workers, self.pool.processes)
            session.__exit__(None if exc is None else type(exc), exc, None)

    def _open_or_stage(self, batch: List[_Query], cells: int):
        """Open ``batch`` on the current session, or return it as ``(batch, cells)`` when it needs a larger pool
        (:meth:`_needs_growth`): it is then opened by :meth:`_grow` once the session's tasks are done."""
        if self._needs_growth(batch, cells):
            return batch, cells
        self._open(batch, cells)
        return None

    def _needs_growth(self, batch: List[_Query], cells: int) -> bool:
        """Whether ``batch`` needs more workers than the session has: an in-process session whose pool would start
        workers for this batch (as :meth:`CodeAlignmentPool.session` decides at the first batch), or a pool sized by
        default that plans more workers for this batch than it has."""
        pool, session = self.pool, self.session
        if pool.serial:
            return False
        want = pool.plan_workers(cells)
        if session.in_process:
            return min(want, self._round1_tasks_estimate(batch, cells)) > 1
        return want > session.workers

    def _grow(self, batch: List[_Query], cells: int) -> None:
        """A new session for ``batch`` once the current one is idle, on a pool grown to at least twice the
        session's workers (at most the budget; geometric, so a call restarts its pool only a few times)."""
        workers = self.session.workers
        self._exit_session(None)
        self._enter_session(max(int(cells), 2 * workers * self.pool.min_worker_cells),
                            max(2, self._round1_tasks_estimate(batch, cells)))

    # ---------------------------------------------------------------------------------------------------- batches
    def _lookup(self, batch: List[_Query]) -> List[_Query]:
        """The candidates' code rows by id (the .bin order depends on the build's ranks)."""
        prof = self.run.prof
        if prof is not None:
            t0 = time.perf_counter()
        offsets, lengths = self.code_store.lookup_rows([tid for q in batch for tid in q.cand_ids])
        start = 0
        for q in batch:
            end = start + len(q.cand_ids)
            q.cand_off, q.cand_len = offsets[start:end], lengths[start:end]
            start = end
        if prof is not None:
            prof.add("lookup", time.perf_counter() - t0)
        return batch

    def _round1_tasks_estimate(self, batch: List[_Query], cells: int) -> int:
        """How many round-1 tasks the first batch will have (for the in-process decision of a one-task batch)."""
        if len(batch) > 1 or self.run.need_panel:
            return 2
        target = target_task_cells(cells, self.pool.plan_workers(cells), self.run.task_cells)
        return len(plan_chunks(batch[0].n, batch[0].cand_len.tolist(), target))

    def _open(self, batch_queries: List[_Query], cells: int):
        run, prof, session = self.run, self.run.prof, self.session
        if prof is not None:
            t0 = time.perf_counter()
        b = _Batch(run.batches, batch_queries)
        run.batches += 1
        workers = session.workers if session.in_process else min(self.pool.plan_workers(cells), session.workers)
        b.target = target = target_task_cells(cells, workers, run.task_cells)
        tasks, costs = [], []
        panel_chunks: Dict[int, List[Tuple[int, int]]] = {}
        for qi, q in enumerate(batch_queries):
            k = len(q.cand_ids)
            q.s_f = np.full(k, np.nan, np.float32)
            q.s_r = np.full(k, np.nan, np.float32)
            q.skipped = np.zeros(k, bool)
            q.r1_left = 0
            if run.need_panel:
                p = len(run.panel_lengths)
                q.panel_f = np.full(p, np.nan, np.float32)
                q.panel_r = np.full(p, np.nan, np.float32)
                chunks = panel_chunks.get(q.n)
                if chunks is None:
                    chunks = panel_chunks[q.n] = plan_chunks(q.n, run.panel_lengths, target)
                for k0, k1 in chunks:
                    pairs = run.panel_pairs[k0:k1]
                    tasks.append(((self._next_tid(), _PANEL, q.off, q.n, pairs), (1, b, qi, _PANEL, k0, k1)))
                    costs.append(q.n * int(pairs[:, 1].sum()))
                    q.r1_left += 1
                run.panel_pair_count += p
            for k0, k1 in plan_chunks(q.n, q.cand_len.tolist(), target):
                pairs = np.column_stack((q.cand_off[k0:k1], q.cand_len[k0:k1]))
                tasks.append(((self._next_tid(), _CODES, q.off, q.n, pairs), (1, b, qi, _CODES, k0, k1)))
                costs.append(q.n * int(pairs[:, 1].sum()))
                q.r1_left += 1
            run.candidate_pairs += k
        run.round1_tasks += len(tasks)
        b.r1_left = len(tasks)
        if prof is not None:
            prof.add("plan_r1", time.perf_counter() - t0)
            b.r1_round = prof.begin_round("round1", min(workers, len(tasks)))
        for task, tag in _lpt(tasks, costs):
            self.round1.append((_round1_task, task, tag, b.r1_round))
        self.open_batches.append(b)
        for qi, q in enumerate(batch_queries):        # a query without round-1 tasks (no candidates: never here)
            if q.r1_left == 0:
                self._statistics_and_round2(b, qi)

    def _next_tid(self) -> int:
        self.tid += 1
        return self.tid - 1

    def _close_query(self, b: _Batch):
        b.open -= 1
        if b.open == 0:
            self.open_batches.remove(b)
            prof = self.run.prof
            if prof is not None and b.r2_round is not None:
                prof.add("round2", prof.end_round(None, self._round_workers(), b.r2_round))

    def _round_workers(self) -> int:
        return 1 if self.session.in_process else self.session.workers

    # ---------------------------------------------------------------------------------------------------- round 1
    def _round1_result(self, tag, res):
        prof = self.run.prof
        if prof is not None:
            tc = time.perf_counter()
        _kind, b, qi, source, k0, k1 = tag
        tid, s_f, s_r, skipped = res
        q = b.queries[qi]
        if source == _PANEL:
            q.panel_f[k0:k1], q.panel_r[k0:k1] = s_f, s_r
            if np.any(skipped):
                raise ValueError(f"query {q.qid!r} ({q.n} residues) is too long for the panel pass: its similarity "
                                 f"with a panel member exceeds max_pair_cells={self.init.max_pair_cells} cells")
        else:
            q.s_f[k0:k1], q.s_r[k0:k1], q.skipped[k0:k1] = s_f, s_r, skipped
        q.r1_left -= 1
        b.r1_left -= 1
        if prof is not None:
            prof.add("consume_r1", time.perf_counter() - tc)
            if b.r1_left == 0:
                prof.add("round1", prof.end_round(None, self._round_workers(), b.r1_round))
        if q.r1_left == 0:
            self._statistics_and_round2(b, qi)

    def _statistics_and_round2(self, b: _Batch, qi: int):
        run, prof = self.run, self.run.prof
        q = b.queries[qi]
        if prof is not None:
            t0 = time.perf_counter()
        _statistics(q, run)
        q.panel_f = q.panel_r = None
        if prof is not None:
            prof.add("statistics", time.perf_counter() - t0)
        if not q.surv.size:
            self._close_query(b)
            return
        if prof is not None:
            t0 = time.perf_counter()
        wanted = {q.cand_ids[k] for k in q.surv.tolist()}
        sequences = self.fetch(wanted)
        missing = sorted(t for t in wanted if t not in sequences)
        if missing:
            raise ValueError(f"{len(missing)} target(s) have residue codes but no sequence in the sequence store "
                             f"(first: {', '.join(missing[:_EXAMPLES])}): the code store and the database disagree")
        q.sequences = sequences
        if prof is not None:
            t1 = time.perf_counter()
            prof.add("fetch", t1 - t0)
        # round-2 tasks: the survivors in chunks of the batch's task size, longest first, ahead of round 1
        q.parts = []
        off, ln = q.cand_off[q.surv], q.cand_len[q.surv]
        tasks, costs = [], []
        for k0, k1 in plan_chunks(q.n, ln.tolist(), b.target):
            pairs = np.column_stack((off[k0:k1], ln[k0:k1]))
            tasks.append(((self._next_tid(), q.off, q.n, pairs), (2, b, qi, k0, k1)))
            costs.append(q.n * int(pairs[:, 1].sum()))
        q.r2_left = len(tasks)
        b.r2_left += len(tasks)
        run.round2_tasks += len(tasks)
        if prof is not None:
            prof.add("plan_r2", time.perf_counter() - t1)
            if b.r2_round is None:
                b.r2_round = prof.begin_round("round2", self._round_workers())
        for task, tag in _lpt(tasks, costs):
            self.round2.append((_round2_task, task, tag, b.r2_round))

    # ---------------------------------------------------------------------------------------------------- round 2
    def _round2_result(self, tag, res):
        run, prof = self.run, self.run.prof
        if prof is not None:
            tc = time.perf_counter()
        _kind, b, qi, k0, k1 = tag
        tid, scores, ends, path, path_off, distinct, skipped = res
        q = b.queries[qi]
        want = q.s_f[q.surv[k0:k1]]
        if np.any(skipped) or not np.array_equal(np.asarray(scores, np.float32).view(np.uint32),
                                                 want.view(np.uint32)):
            bad = np.flatnonzero(np.asarray(scores, np.float32).view(np.uint32) != want.view(np.uint32))
            raise InternalError(
                f"esm-codes Stage 2: the round-2 traceback score of query {q.qid!r} differs from its round-1 S_f "
                f"(targets {[q.cand_ids[int(q.surv[k0 + i])] for i in bad[:_EXAMPLES].tolist()]}: "
                f"{np.asarray(scores)[bad[:_EXAMPLES]].tolist()} vs {want[bad[:_EXAMPLES]].tolist()}); the reported "
                "raw score would not be the one behind the E-value")
        part = (k0, np.asarray(ends), np.asarray(path), np.asarray(path_off, np.int64), np.asarray(distinct, np.int64))
        q.parts.append(part)
        q.r2_left -= 1
        b.r2_left -= 1
        if prof is not None:
            prof.round2_payload(sum(int(a.nbytes) for a in part[1:]), len(part[4]))
            prof.add("consume_r2", time.perf_counter() - tc)
        if q.r2_left == 0:
            if prof is not None:
                held = sum(int(a.nbytes) for p in q.parts for a in p[1:])
            self.results[q.qid] = _hits(q, run, q.sequences)
            q.sequences = None
            if prof is not None:
                prof.round2_payload(-held, -int(q.surv.size))
            self._close_query(b)


def _statistics(q: _Query, run: _Run):
    """S9-S11 in the parent: S, Q90, S', ln E, bits; then the survivors."""
    spec = run.spec
    valid = ~q.skipped
    if q.skipped.any():
        run.skipped_pairs += int(q.skipped.sum())
        m = int(q.cand_len[q.skipped].max())
        if q.n * m > run.largest_skipped[0] * run.largest_skipped[1]:
            run.largest_skipped = (q.n, m)
    q.s = q.s_f.astype(np.float64) - q.s_r.astype(np.float64)
    q.s_prime = q.lne = q.bits = None
    if run.need_panel:
        q.q90 = cs.q90(q.panel_f.astype(np.float64) - q.panel_r.astype(np.float64), spec.panel_quantile,
                       spec.quantile_method)
        run.q90[q.qid] = q.q90
        q.s_prime = q.s - q.q90
        q.lne = np.full(len(q.s), np.nan)
        q.bits = np.full(len(q.s), np.nan)
        if valid.any():
            q.lne[valid] = cs.log_evalue(q.s_prime[valid], q.n, run.db_residues, spec)
            q.bits[valid] = cs.bit_score(q.s_prime[valid], q.n, spec)
    if run.ln_max is None:
        surv = valid
    else:
        surv = valid & (np.nan_to_num(q.lne, nan=np.inf) <= run.ln_max)
    q.surv = np.flatnonzero(surv)
    run.survivors += int(q.surv.size)


def _assemble(q: _Query):
    """The round-2 parts of a query in survivor order: ends [S, 2], path [P, 2], path_off [S + 1], distinct [S]."""
    parts = sorted(q.parts, key=lambda p: p[0])
    s = int(q.surv.size)
    if sum(len(p[4]) for p in parts) != s:
        raise InternalError(f"esm-codes Stage 2: round 2 returned {sum(len(p[4]) for p in parts)} of {s} survivors "
                            f"of query {q.qid!r}")
    ends = np.concatenate([p[1] for p in parts]).reshape(-1, 2)
    path = np.concatenate([p[2].reshape(-1, 2) for p in parts]) if parts else np.zeros((0, 2), np.int32)
    lens = np.concatenate([np.diff(p[3]) for p in parts])
    path_off = np.zeros(s + 1, np.int64)
    np.cumsum(lens, out=path_off[1:])
    distinct = np.concatenate([p[4] for p in parts])
    q.parts = []
    return ends, path, path_off, distinct


def _hits(q: _Query, run: _Run, sequences: Mapping[str, str]) -> List[Hit]:
    """Metrics, ecal, filters and S14 order of one query's survivors. The per-hit values come from the vectorised
    arrays as Python lists (``tolist`` gives the same ints and floats as ``int()`` / ``float()`` of each element), so
    the loop only builds the objects."""
    needs, d = run.needs, run.domain
    prof = run.prof
    if prof is not None:
        t0 = time.perf_counter()
    ends, path, path_off, distinct = _assemble(q)
    if prof is not None:
        t1 = time.perf_counter()
        prof.add("hits_assemble", t1 - t0)
    surv = q.surv
    surv_l = surv.tolist()
    q_bytes = np.frombuffer(sanitized_bytes(q.seq), np.uint8)
    t_seqs = [sequences[q.cand_ids[k]] for k in surv_l]
    t_clean = [sanitized_bytes(s) for s in t_seqs]
    t_len = q.cand_len[surv]
    t_len_l = t_len.tolist()
    bad = [q.cand_ids[surv_l[i]] for i, b in enumerate(t_clean) if len(b) != t_len_l[i]]
    if bad:
        raise ValueError(f"{len(bad)} target(s) have residue codes of another length than their stored sequence "
                         f"(first: {', '.join(bad[:_EXAMPLES])}): the code store and the database disagree")
    t_starts = np.zeros(len(t_clean), np.int64)
    if len(t_clean) > 1:
        np.cumsum(t_len[:-1], out=t_starts[1:])
    t_bytes = np.frombuffer(b"".join(t_clean), np.uint8)
    cols = path_columns(path, path_off, q_bytes, t_bytes, t_starts)
    aln_len = cols["aln_len"]
    with np.errstate(divide="ignore", invalid="ignore"):
        identity = np.where(aln_len > 0, cols["n_ident"] / np.maximum(aln_len, 1), 0.0)
        qcov = cols["q_aligned"] / q.n
        tcov = np.where(t_len > 0, cols["t_aligned"] / np.maximum(t_len, 1), 0.0)
    kept = (aln_len > 0) & (identity >= run.min_id) & (np.minimum(qcov, tcov) >= run.min_cov)
    kidx = np.flatnonzero(kept)
    hits: List[Hit] = []
    if kidx.size:
        # every kept path (aln_len > 0) must end at the kernel's best cell
        seg_lo, seg_hi = path_off[kidx], path_off[kidx + 1]
        off_end = ~np.all(ends[kidx] == path[seg_hi - 1] + 1, axis=1)
        if off_end.any():
            i = int(kidx[np.flatnonzero(off_end)[0]])
            raise InternalError(f"esm-codes Stage 2: the path of query {q.qid!r} against {q.cand_ids[surv_l[i]]!r} "
                                f"does not end at the kernel's best cell {ends[i].tolist()}")
        ks = surv[kidx]
        kidx_l, ks_l, m_l = kidx.tolist(), ks.tolist(), t_len[kidx].tolist()
        lo_l, hi_l = seg_lo.tolist(), seg_hi.tolist()
        ident_l, qcov_l, tcov_l = identity[kidx].tolist(), qcov[kidx].tolist(), tcov[kidx].tolist()
        aln_l, nid_l, mis_l, gap_l = (cols[c][kidx].tolist() for c in ("aln_len", "n_ident", "mismatch", "gap_open"))
        qs_l, qe_l, ts_l, te_l = (cols[c][kidx].tolist() for c in ("q_start", "q_end", "t_start", "t_end"))
        sf_l, sr_l, dist_l = q.s_f[ks].tolist(), q.s_r[ks].tolist(), distinct[kidx].tolist()
        if run.need_panel:
            lne_l, bits_l, sp_l = q.lne[ks].tolist(), q.bits[ks].tolist(), q.s_prime[ks].tolist()
    if prof is not None:
        t2 = time.perf_counter()
        prof.add("hits_metrics", t2 - t1)

    q_inside, n = q.inside, q.n
    hd = run.hit_domain
    q_str = q_bytes.tobytes().decode("ascii") if needs.qseq else None
    t_max, ns_max, cap = d.target_max_length, d.max_nonstandard_fraction, d.target_min_distinct_codes
    strings = None
    if kidx.size and (needs.cigar or needs.qaln or needs.taln):
        strings = _PathStrings(path, path_off, q_bytes, t_bytes, t_starts, cigar=needs.cigar, qaln=needs.qaln,
                               taln=needs.taln)
    cand_ids, cand_emb, q90 = q.cand_ids, q.cand_emb, q.q90
    lne = evalue = bit_score = s_prime = None
    with np.errstate(over="ignore", under="ignore"):
        for j, i in enumerate(kidx_l if kidx.size else ()):
            k, m = ks_l[j], m_l[j]
            t_nonstd = nonstandard_count(t_seqs[i]) / m if m else 1.0
            t_long, t_nonstandard = m > t_max, t_nonstd > ns_max
            t_low = dist_l[j] < cap
            ecal = q_inside and not (t_long or t_nonstandard or t_low)
            hd["hits"] += 1
            if not ecal:
                hd["ecal0"] += 1
                hd["query_outside"] += int(not q_inside)
                hd["target_long"] += int(t_long)
                hd["target_nonstandard"] += int(t_nonstandard)
                hd["target_low_distinct"] += int(t_low)
            cigar = q_aln = t_aln = None
            if strings is not None:
                lo, hi = lo_l[j], hi_l[j]
                if needs.cigar:
                    cigar = strings.cigar(lo, hi)
                if needs.qaln:
                    q_aln = strings.qaln[lo:hi].decode("ascii")
                if needs.taln:
                    t_aln = strings.taln[lo:hi].decode("ascii")
            if run.need_panel:
                lne = lne_l[j]
                evalue = float(np.exp(lne))
                bit_score, s_prime = bits_l[j], sp_l[j]
            n_ident = nid_l[j]
            metrics = AlignmentMetrics(
                identity_aln=ident_l[j], identity_shorter=n_ident / min(n, m),
                query_coverage=qcov_l[j], subject_coverage=tcov_l[j], aln_len=aln_l[j],
                score=sf_l[j], bit_score=bit_score, pvalue=None, evalue=evalue, query_len=n, subject_len=m,
                n_ident=n_ident, mismatch=mis_l[j], gap_open=gap_l[j],
                q_start=qs_l[j], q_end=qe_l[j], t_start=ts_l[j], t_end=te_l[j], cigar=cigar, q_aln=q_aln,
                t_aln=t_aln, q_seq=q_str, t_seq=t_clean[i].decode("ascii") if needs.tseq else None,
                score_rev=sr_l[j], q90=q90 if run.need_panel else None, score_adj=s_prime, log_evalue=lne,
                ecal=bool(ecal))
            hits.append(Hit(subject_id=cand_ids[k], emb_score=cand_emb[k], metrics=metrics))
    if prof is not None:
        t3 = time.perf_counter()
        prof.add("hits_objects", t3 - t2)
    # S14: S = fp64(S_f) - fp64(S_r) descending (score and score_rev hold the exact float32 values), then the
    # embedding score descending, then the target id ascending
    hits.sort(key=lambda h: (-(h.metrics.score - h.metrics.score_rev), -h.emb_score, h.subject_id))
    if prof is not None:
        prof.add("hits_sort", time.perf_counter() - t3)
    return hits
