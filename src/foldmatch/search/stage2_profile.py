"""Profiling mode of esm-codes Stage 2 (plan section 8 P7, profiling gate): where the time of
:func:`~foldmatch.search.code_alignment.align_candidates_esm_codes` goes, in the parent and in the workers.

Off by default. ``FOLDMATCH_STAGE2_PROFILE=1`` (anything but empty, ``0``, ``false``, ``no`` or ``off``) turns it on
for the next Stage-2 call; the pool's workers learn it from the worker setup. Profiling only reads clocks, sizes
and the peak RSS: the hits are the same, bit for bit, with it on or off; when it is off the aligner and the kernel
wrapper only test ``is not None`` once per batch, query, task or block.

When it is on, the call logs one ``esm-codes Stage 2 profile`` INFO line and, given a ``stats`` dict, stores the
breakdown in ``stats["profile"]`` (:meth:`Stage2Profile.summary`):

``parent``  seconds per parent phase: ``setup``, ``collect`` (query checks), per batch ``lookup`` (SQLite lookup of
            the candidates' codes), ``plan_r1`` / ``plan_r2`` (task lists), ``round1`` / ``round2`` (wall time from the
            first submit to the last result, including the pool start-up of the first batch), ``consume_r1`` /
            ``consume_r2`` (the parent's work on each result), ``statistics`` (S, Q90, ln E, bits, survivors),
            ``fetch`` (survivor sequences), ``hits_assemble`` / ``hits_metrics`` (vectorised S12 metrics) /
            ``hits_objects`` (one ``Hit`` per kept survivor, CIGAR and strings) / ``hits_sort``, ``finish``,
            ``pool_close`` (a pool owned by the call), ``total``; ``profile_overhead`` is the time profiling itself
            spent in the parent (pickled task sizes), included in the phases above.
``workers`` per task kind (``panel`` and ``candidates`` in round 1, ``round2``): tasks, pairs, similarity cells,
            worker seconds and their split: ``prep`` (S5 query prep), ``sim`` (decode + per-pair GEMMs), ``kernel``
            (one native call per block: rowcol z + DP, and the traceback in round 2), ``paths`` (round 2: gathering
            the paths), ``distinct`` (round 2: distinct-code counts), ``driver`` (the rest of the kernel wrapper:
            block planning, buffers, ctypes) and ``other`` (the rest of the task); ns per cell for ``sim`` and
            ``kernel``.
``ipc``     pickled bytes of the tasks (parent) and results (worker) with the time a ``pickle.dumps`` of them took
            (estimates of the pickling cost the executor pays once more), the wait from submit to task start
            (queueing included), the latency from task end to the parent's receipt, the idle gaps of each worker
            between tasks, and per round the worker utilisation ``busy / (workers x wall)``.
``pool``    executor creation, workers, and per worker the delay from the first submit to its module import, its
            initializer seconds and when it was ready.
``memory``  the largest worker and parent peak RSS (:func:`max_rss_bytes`) and the largest round-2 path payload held
            by the parent for one batch (bytes, and bytes per survivor).
``tasks``   the per-task records (kind, pid, n, pairs, cells, start / end, components), for the benchmark.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

PROFILE_ENV = "FOLDMATCH_STAGE2_PROFILE"
#: When this module was first imported in this process (in a spawn worker: right after interpreter start-up and the
#: re-import of the main module, before the initializer runs).
IMPORT_UNIX = time.time()

KINDS = ("panel", "candidates", "round2")
_COMPONENTS = ("prep", "sim", "kernel", "paths", "distinct", "driver", "other")


def profiling_enabled() -> bool:
    """Whether ``FOLDMATCH_STAGE2_PROFILE`` asks for the profiling mode."""
    return os.environ.get(PROFILE_ENV, "").strip().lower() not in ("", "0", "false", "no", "off")


def max_rss_bytes() -> int:
    """This process's peak resident set size in bytes: Linux ``VmHWM`` (of the current address space, so a spawn
    worker does not inherit its parent's peak, as ``ru_maxrss`` does across fork + exec), else ``ru_maxrss``
    (bytes on macOS)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import resource
    except ImportError:  # pragma: no cover (Windows)
        return 0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def process_start_unix() -> Optional[float]:
    """This process's start time (Linux ``/proc``), seconds since the epoch; None elsewhere."""
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        with open("/proc/stat") as f:
            btime = next(int(line.split()[1]) for line in f if line.startswith("btime"))
        return btime + int(fields[19]) / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001  (informational only)
        return None


def init_record(start_unix: float, seconds: float) -> Dict[str, Any]:
    """A worker's start-up record, sent back with its first task."""
    return dict(pid=os.getpid(), process_start_unix=process_start_unix(), import_unix=IMPORT_UNIX,
                init_start_unix=start_unix, init_seconds=seconds)


class TaskTimer:
    """The phase timers of one task in a worker (or in-process): :meth:`lap` adds the time since the previous lap to
    a component; the kernel wrapper adds ``sim`` / ``kernel`` / ``paths`` / ``blocks`` to :attr:`t` itself."""
    __slots__ = ("t", "start", "last", "start_unix")

    def __init__(self):
        self.start_unix = time.time()
        self.start = self.last = time.perf_counter()
        self.t: Dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.t[name] = self.t.get(name, 0.0) + (now - self.last)
        self.last = now

    def record(self, kind: str, n: int, pairs: np.ndarray, result: tuple, init: Optional[dict]) -> Dict[str, Any]:
        """The task's record (sent back as the last element of its result tuple)."""
        end = time.perf_counter()
        rec = dict(kind=kind, pid=os.getpid(), n=int(n), pairs=int(len(pairs)),
                   cells=int(n) * int(np.asarray(pairs)[:, 1].sum()) if len(pairs) else 0,
                   start_unix=self.start_unix, end_unix=time.time(), seconds=end - self.start, t=dict(self.t))
        p0 = time.perf_counter()
        rec["result_bytes"] = len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
        rec["result_pickle_s"] = time.perf_counter() - p0
        rec["rss_max_bytes"] = max_rss_bytes()
        if init is not None:
            rec["init"] = init
        return rec


def _stats(values) -> Dict[str, float]:
    v = np.asarray(list(values), np.float64)
    if v.size == 0:
        return dict(n=0, sum=0.0, p50=0.0, p95=0.0, max=0.0)
    return dict(n=int(v.size), sum=float(v.sum()), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                max=float(v.max()))


class Stage2Profile:
    """The parent side of one profiled Stage-2 call (see the module docstring)."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.parent: Dict[str, float] = {}
        self.tasks: List[Dict[str, Any]] = []
        self.inits: List[Dict[str, Any]] = []
        self.rounds: List[Dict[str, Any]] = []
        self._submit: Dict[int, float] = {}
        self._round: Optional[Dict[str, Any]] = None
        self.task_bytes = 0
        self.task_pickle_s = 0.0
        self.executor_create_s = 0.0
        self.executors = 0
        self.workers = 0
        self.r2_parts_peak_bytes = 0
        self.r2_parts_peak_survivors = 0

    # ---------------------------------------------------------------------------------------------------- parent
    def add(self, phase: str, seconds: float) -> None:
        self.parent[phase] = self.parent.get(phase, 0.0) + seconds

    def begin_round(self, name: str, workers: int) -> None:
        self._submit = {}
        self._round = dict(name=name, workers=int(workers), start_unix=time.time(), first_submit_unix=None,
                           tasks=0)
        self.rounds.append(self._round)

    def end_round(self, wall: float, workers: int) -> None:
        if self._round is not None:
            self._round.update(wall=wall, end_unix=time.time(), workers=max(int(workers), self._round["workers"]))
        self._round = None

    def submitted(self, task) -> None:
        """Called by the pool for each task as it is submitted: its time, and its pickled size (an estimate of the
        pickling the executor does; the time this takes is counted as profiling overhead)."""
        now = time.time()
        p0 = time.perf_counter()
        self.task_bytes += len(pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL))
        dt = time.perf_counter() - p0
        self.task_pickle_s += dt
        self.add("profile_overhead", dt)
        self._submit[int(task[0])] = now
        if self._round is not None:
            self._round["tasks"] += 1
            if self._round["first_submit_unix"] is None:
                self._round["first_submit_unix"] = now

    def received(self, result: tuple) -> tuple:
        """A task result with its trailing record: the record is kept (with the submit and receipt times) and the
        plain result returned."""
        rec = result[-1]
        rec["receive_unix"] = time.time()
        rec["submit_unix"] = self._submit.pop(int(result[0]), None)
        rec["round"] = len(self.rounds) - 1
        init = rec.pop("init", None)
        if init is not None:
            init["first_submit_unix"] = self.rounds[-1]["first_submit_unix"] if self.rounds else None
            self.inits.append(init)
        self.tasks.append(rec)
        return result[:-1]

    def executor_created(self, seconds: float, workers: int) -> None:
        self.executor_create_s += seconds
        self.executors += 1
        self.workers = max(self.workers, int(workers))

    def round2_parts(self, queries) -> None:
        """The round-2 payload (ends, paths, offsets, distinct counts) the parent holds for one batch."""
        nbytes = sum(int(a.nbytes) for q in queries for part in q.parts for a in part[1:])
        if nbytes > self.r2_parts_peak_bytes:
            self.r2_parts_peak_bytes = nbytes
            self.r2_parts_peak_survivors = sum(int(q.surv.size) for q in queries)

    # ---------------------------------------------------------------------------------------------------- summary
    def summary(self, workers: int) -> Dict[str, Any]:
        self.parent["total"] = time.perf_counter() - self.t0
        out: Dict[str, Any] = dict(parent={k: float(v) for k, v in self.parent.items()})
        kinds: Dict[str, Dict[str, float]] = {}
        for rec in self.tasks:
            k = kinds.setdefault(rec["kind"], dict(tasks=0, pairs=0, cells=0, seconds=0.0,
                                                   **dict.fromkeys(_COMPONENTS, 0.0)))
            t = rec["t"]
            k["tasks"] += 1
            k["pairs"] += rec["pairs"]
            k["cells"] += rec["cells"]
            k["seconds"] += rec["seconds"]
            swk = t.get("swkern", 0.0)
            for c in ("prep", "sim", "kernel", "paths", "distinct"):
                k[c] += t.get(c, 0.0)
            k["driver"] += swk - t.get("sim", 0.0) - t.get("kernel", 0.0) - t.get("paths", 0.0)
            k["other"] += rec["seconds"] - swk - t.get("prep", 0.0) - t.get("distinct", 0.0)
        for k in kinds.values():
            cells = max(k["cells"], 1)
            k["sim_ns_per_cell"] = 1e9 * k["sim"] / cells
            k["kernel_ns_per_cell"] = 1e9 * k["kernel"] / cells
        out["workers"] = kinds
        # IPC: queueing, return latency, per-worker idle gaps, utilisation per round
        waits = [r["start_unix"] - r["submit_unix"] for r in self.tasks if r.get("submit_unix") is not None]
        returns = [r["receive_unix"] - r["end_unix"] for r in self.tasks]
        rounds = []
        gaps_all: List[float] = []
        for i, rd in enumerate(self.rounds):
            recs = [r for r in self.tasks if r["round"] == i]
            busy = sum(r["seconds"] for r in recs)
            wall = rd.get("wall", 0.0) or 0.0
            w = max(rd.get("workers", 1), 1)
            by_pid: Dict[int, List[Dict[str, Any]]] = {}
            for r in recs:
                by_pid.setdefault(r["pid"], []).append(r)
            gaps, first_start, tail = [], [], []
            for lst in by_pid.values():
                lst.sort(key=lambda r: r["start_unix"])
                gaps += [b["start_unix"] - a["end_unix"] for a, b in zip(lst, lst[1:])]
                if rd.get("first_submit_unix") is not None:
                    first_start.append(lst[0]["start_unix"] - rd["first_submit_unix"])
                if rd.get("end_unix") is not None:
                    tail.append(rd["end_unix"] - lst[-1]["end_unix"])
            gaps_all += gaps
            rounds.append(dict(name=rd["name"], tasks=len(recs), workers=w, processes=len(by_pid), wall=wall,
                               busy=busy, utilisation=busy / (w * wall) if wall > 0 else None,
                               first_task_start=_stats(first_start), idle_between_tasks=_stats(gaps),
                               tail_idle=_stats(tail)))
        out["ipc"] = dict(task_bytes=int(self.task_bytes), task_pickle_s=float(self.task_pickle_s),
                          result_bytes=int(sum(r["result_bytes"] for r in self.tasks)),
                          result_pickle_s=float(sum(r["result_pickle_s"] for r in self.tasks)),
                          submit_to_start=_stats(waits), end_to_receipt=_stats(returns),
                          idle_between_tasks=_stats(gaps_all), rounds=rounds)
        inits = []
        for rec in self.inits:
            fs = rec.get("first_submit_unix")
            inits.append(dict(pid=rec["pid"], init_seconds=rec["init_seconds"],
                              import_after_submit=(rec["import_unix"] - fs) if fs is not None else None,
                              ready_after_submit=(rec["init_start_unix"] + rec["init_seconds"] - fs)
                              if fs is not None else None,
                              process_start_after_submit=(rec["process_start_unix"] - fs)
                              if fs is not None and rec.get("process_start_unix") is not None else None))
        out["pool"] = dict(executors=self.executors, executor_create_s=float(self.executor_create_s),
                           workers=max(self.workers, int(workers)), worker_starts=inits,
                           init_seconds=_stats(i["init_seconds"] for i in inits),
                           ready_after_submit=_stats(i["ready_after_submit"] for i in inits
                                                     if i["ready_after_submit"] is not None))
        out["memory"] = dict(worker_rss_max_bytes=max((r["rss_max_bytes"] for r in self.tasks
                                                       if r["pid"] != os.getpid()), default=0),
                             in_process_rss_max_bytes=max((r["rss_max_bytes"] for r in self.tasks
                                                           if r["pid"] == os.getpid()), default=0),
                             parent_rss_max_bytes=max_rss_bytes(),
                             round2_parts_peak_bytes=int(self.r2_parts_peak_bytes),
                             round2_parts_peak_survivors=int(self.r2_parts_peak_survivors),
                             round2_parts_bytes_per_survivor=(self.r2_parts_peak_bytes
                                                              / self.r2_parts_peak_survivors
                                                              if self.r2_parts_peak_survivors else None))
        out["tasks"] = [dict(kind=r["kind"], pid=r["pid"], round=r["round"], n=r["n"], pairs=r["pairs"],
                             cells=r["cells"], start_unix=r["start_unix"], end_unix=r["end_unix"],
                             submit_unix=r.get("submit_unix"), receive_unix=r["receive_unix"], seconds=r["seconds"],
                             result_bytes=r["result_bytes"], **{k: float(v) for k, v in r["t"].items()})
                        for r in self.tasks]
        return out

    @staticmethod
    def log_line(summary: Dict[str, Any]) -> str:
        p, w, ipc, mem = summary["parent"], summary["workers"], summary["ipc"], summary["memory"]
        parts = [f"{k} {v:.3f}" for k, v in p.items()]
        kinds = []
        for name in KINDS:
            k = w.get(name)
            if k:
                kinds.append(f"{name}: {k['tasks']} tasks {k['seconds']:.3f} worker-s (sim {k['sim']:.3f}, kernel "
                             f"{k['kernel']:.3f}, prep {k['prep']:.3f}, driver {k['driver']:.3f}, other "
                             f"{k['other']:.3f})")
        util = ", ".join(f"{r['name']} {r['utilisation']:.2f}" for r in ipc["rounds"]
                         if r["utilisation"] is not None)
        return (f"esm-codes Stage 2 profile: parent s: {', '.join(parts)}; workers: {'; '.join(kinds)}; "
                f"IPC {ipc['task_bytes']} B out, {ipc['result_bytes']} B back; utilisation {util or 'n/a'}; "
                f"worker RSS max {mem['worker_rss_max_bytes'] / 2**20:.1f} MiB, round-2 parts peak "
                f"{mem['round2_parts_peak_bytes'] / 2**20:.2f} MiB")
