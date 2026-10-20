"""P6.4: the esm-codes Stage-2 aligner (``foldmatch.search.code_alignment``; plan section 8 P6, method S5-S14, S16).

No ESM3: the encodings and codes are the fixtures' (F1 ``pairs_v1.npz``, F3 ``q90_queries_v1.npz``, F-deg
``degenerate_v1.npz``); the databases are built in the test with ``SequenceStore`` and ``ResidueCodeStore``, the query
encodings written as the per-run memmap of a ``QueryEncodingIndex``. The panel and the E-value table come from the
registry bundle (``esm_codes_bundle``: skipped only when offline and not cached).

Tolerances are S16: S_f / S_r within 5e-5 max(|ref|, 1); paths identical unless the float64 path-score margin is below
1e-4 + 2e-6 |S_f|; Q90 and ln E within 1e-4 (F1, at N = 1e9); F3 Q90 within 1e-3 (the plan's bound). On one machine
the output is bit-identical for any task size, block size, query batch and worker count, and round 2 reproduces
round 1's S_f bit for bit.
"""
import contextlib
import dataclasses
import functools
import logging
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import foldmatch
from foldmatch.errors import InternalError
from foldmatch.search import alignment, output, swkern
from foldmatch.search import code_statistics as cs
from foldmatch.search import code_alignment as ca
from foldmatch.search import scoring_system as ss
from foldmatch.search import stage2_profile
from foldmatch.search.query_encodings import QueryEncodingEntry, QueryEncodingIndex
from foldmatch.search.residue_code_store import ResidueCodeStore, ResidueCodeStoreError
from foldmatch.search.sequence_store import SequenceStore
from foldmatch.utils.sequence import STANDARD_AMINO_ACIDS, seq_sha1

_RES = Path(__file__).parent / "resources" / "esm_codes"
SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)
ESM_REV = "636acb420daf674df37ab0220f56061a5de2f169"   # passed explicitly: the code store needs no torch
N_F1 = 10 ** 9                                        # the fixtures' database size for ln E
ALL_FIELDS = list(output.SUPPORTED_OUTPUT_FIELDS)
_AA = np.frombuffer(STANDARD_AMINO_ACIDS.encode(), np.uint8)


# =====================================================================================================================
# data
# =====================================================================================================================
@functools.lru_cache(maxsize=None)
def _prepared():
    import esm_codes_bundle
    esm_codes_bundle.registry_files()                # unittest.SkipTest only when offline and not cached
    return ss.prepare(SYSTEM)


@functools.lru_cache(maxsize=None)
def _f1():
    z = np.load(_RES / "pairs_v1.npz")
    qo, to = z["q_offsets"], z["t_offsets"]
    pairs = []
    for p in range(len(z["pair_q"])):
        a, b = int(z["pair_q"][p]), int(z["pair_t"][p])
        pairs.append(dict(
            qid=str(z["q_ids"][a]), q16=z["q"][qo[a]:qo[a + 1]], qseq=str(z["q_sequences"][a]),
            tid=str(z["t_ids"][b]), codes=z["codes"][to[b]:to[b + 1]], tseq=str(z["t_sequences"][b]),
            S_f=float(z["S_f"][p]), S_r=float(z["S_r"][p]), q90=float(z["q90"][p]), lnE=float(z["lnE_N1e9"][p]),
            path=z["path_flat"][z["path_off"][p]:z["path_off"][p + 1]], n_ident=int(z["n_ident"][p]),
            aln_cols=int(z["aln_cols"][p])))
    return pairs


def _random_seq(rng, n):
    return _AA[rng.integers(0, len(_AA), n)].tobytes().decode("ascii")


@contextlib.contextmanager
def _deadline(seconds: float):
    """TimeoutError in this (the main) thread after ``seconds``: a test of a wait that must end fails instead."""
    def expired(*_):
        raise TimeoutError(f"still waiting after {seconds} s")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _run_script(source: str, folder: Path, timeout: float = 120.0):
    """Run ``source`` as a script file (spawn workers re-import a main *file*) in its own session, with this tree on
    PYTHONPATH; on timeout the session (the script and its workers) is killed. Returns ``(returncode, output)``,
    returncode None on a timeout."""
    script = folder / "script.py"
    script.write_text(textwrap.dedent(source))
    src = str(Path(foldmatch.__file__).resolve().parents[1])
    paths = [src, str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in paths if p))
    proc = subprocess.Popen([sys.executable, str(script)], cwd=folder, env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        return None, out


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class World:
    """A database (sequence store + residue-code store) and a run's query-encoding memmap in a temporary folder."""

    def __init__(self, folder: Path, targets, queries, name="db"):
        """``targets``: ``[(id, sequence, codes uint8 [m, 8])]``; ``queries``: ``[(id, sequence, q fp16 [n, 64])]``."""
        self.folder = Path(folder)
        fasta = self.folder / f"{name}.fasta"
        with open(fasta, "w") as f:
            for tid, seq, _ in targets:
                f.write(f">{tid}\n{seq}\n")
        self.seq_store = SequenceStore(self.folder, name)
        self.seq_store.create(fasta)
        self.store = ResidueCodeStore(self.folder, name)
        self.store.create([([t[0] for t in targets], [np.asarray(t[2], np.uint8) for t in targets],
                            [seq_sha1(t[1]) for t in targets])], SYSTEM, esm_revision=ESM_REV)
        self.store.validate(self.seq_store, len(targets), preflight=None)
        self.residues = self.store.manifest()["total_residues"]
        self.query_index = self.make_query_index(queries, f"{name}.queries.f16")
        self.query_sequences = {qid: seq for qid, seq, _ in queries}

    def make_query_index(self, queries, filename):
        path = self.folder / filename
        entries, rows = {}, 0
        with open(path, "wb") as f:
            for qid, seq, q16 in queries:
                f.write(np.ascontiguousarray(q16, "<f2").tobytes())
                entries[qid] = QueryEncodingEntry(rows, len(q16), seq_sha1(seq))
                rows += len(q16)
        return QueryEncodingIndex(path, rows, swkern.DIM, entries, SYSTEM.name)

    def align(self, prefilter, pool=None, *, db_residues=N_F1, output_needs=None, query_index=None, **kwargs):
        stats = kwargs.pop("stats", {})
        res = ca.align_candidates_esm_codes(
            query_index or self.query_index, prefilter, self.store, self.seq_store.fetch, _prepared(), db_residues,
            pool, query_sequences=kwargs.pop("query_sequences", self.query_sequences),
            output_needs=ALL_FIELDS if output_needs is None else output_needs, stats=stats, **kwargs)
        return res, stats


class _WorldCase(unittest.TestCase):
    """A test class with its own temporary folder; ``f1_world()`` holds the 50 F1 targets and queries."""

    @classmethod
    def setUpClass(cls):
        _prepared()
        swkern.load()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls._worlds = {}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def f1_world(cls) -> World:
        if "f1" not in cls._worlds:
            f1 = _f1()
            cls._worlds["f1"] = World(cls.tmp, [(p["tid"], p["tseq"], p["codes"]) for p in f1],
                                      [(p["qid"], p["qseq"], p["q16"]) for p in f1], name="f1")
        return cls._worlds["f1"]


def _f1_own_target():
    """Each F1 query against its own target (the fixture pairs)."""
    return {p["qid"]: ([p["tid"]], [0.5]) for p in _f1()}


def _f1_mixed(queries=None, per_query=8, stride=7):
    """Each F1 query against its own target and ``per_query - 1`` others, with decreasing embedding scores."""
    f1 = _f1()
    picks = range(len(f1)) if queries is None else queries
    out = {}
    for a in picks:
        tids = [f1[(a + j * stride) % len(f1)]["tid"] for j in range(per_query)]
        out[f1[a]["qid"]] = (tids, [1.0 - 0.05 * j for j in range(per_query)])
    return out


def _signature(results):
    """Everything the output can show, bit for bit: every metrics field (floats as hex) and the rendered row."""
    def plain(v):
        return float(v).hex() if isinstance(v, float) else v
    sig = {}
    for qid, hits in results.items():
        sig[qid] = [(h.subject_id, float(h.emb_score).hex(),
                     tuple(plain(getattr(h.metrics, f.name)) for f in dataclasses.fields(h.metrics)),
                     output.format_row(qid, h, ALL_FIELDS)) for h in hits]
    return sig


def _path_from_hit(m) -> np.ndarray:
    """The forward path a hit's CIGAR and start positions describe."""
    i, j, out = m.q_start - 1, m.t_start - 1, []
    for op in swkern.expand_cigar(m.cigar):
        if op == "M":
            out.append((i, j))
            i, j = i + 1, j + 1
        elif op == "I":
            out.append((i, -1))
            i += 1
        else:
            out.append((-1, j))
            j += 1
    return np.array(out, np.int32).reshape(-1, 2)


def _assert_path_by_margin(tc, got, want, q16, codes, s_f, what):
    """S16: the path equals the reference unless the float64 path scores tie within 1e-4 + 2e-6 |S_f|."""
    if np.array_equal(got, want):
        return
    z = swkern.z64_pair(q16, codes)
    gap = abs(swkern.path_score(z, got, swkern.SETTING_D) - swkern.path_score(z, want, swkern.SETTING_D))
    tc.assertLess(gap, swkern.s16_path_margin(s_f), f"{what}: path differs beyond the S16 near-tie margin")


# =====================================================================================================================
# differential: F1 and F3
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class F1DifferentialTest(_WorldCase):

    def _check_pair(self, p, m):
        self.assertLessEqual(abs(m.score - p["S_f"]), 5e-5 * max(abs(p["S_f"]), 1.0), p["qid"])
        self.assertLessEqual(abs(m.score_rev - p["S_r"]), 5e-5 * max(abs(p["S_r"]), 1.0), p["qid"])
        self.assertLessEqual(abs(m.q90 - p["q90"]), 1e-4, p["qid"])
        self.assertLessEqual(abs(m.log_evalue - p["lnE"]), 1e-4, p["qid"])
        self.assertEqual(m.score_adj, (m.score - m.score_rev) - m.q90)
        _assert_path_by_margin(self, _path_from_hit(m), p["path"], p["q16"], p["codes"], p["S_f"], p["qid"])
        self.assertEqual((m.aln_len, m.n_ident), (p["aln_cols"], p["n_ident"]))
        self.assertTrue(m.ecal)

    def test_within_s16_and_every_candidate_traced_without_an_e_filter(self):
        world = self.f1_world()
        res, stats = world.align(_f1_own_target(), max_evalue=None)
        self.assertTrue(stats["panel"])
        self.assertEqual((stats["candidate_pairs"], stats["survivors"]), (50, 50))   # inf: every candidate traced
        for p in _f1():
            hits = res[p["qid"]]
            self.assertEqual([h.subject_id for h in hits], [p["tid"]])
            self._check_pair(p, hits[0].metrics)
        self.assertEqual(stats["hits"], 50)

    def test_only_three_pairs_survive_at_1e_3(self):
        world = self.f1_world()
        res, stats = world.align(_f1_own_target(), max_evalue=1e-3)
        want = {p["qid"] for p in _f1() if p["lnE"] <= math.log(1e-3)}
        self.assertEqual(len(want), 3)
        self.assertEqual(stats["survivors"], 3)
        self.assertEqual(stats["round2_tasks"], 3)
        self.assertEqual({q for q, h in res.items() if h}, want)
        for p in _f1():
            if p["qid"] in want:
                self._check_pair(p, res[p["qid"]][0].metrics)
                self.assertLessEqual(res[p["qid"]][0].metrics.log_evalue, math.log(1e-3))
        self.assertEqual(list(res), [p["qid"] for p in _f1()])          # every query, in prefilter order

    def test_reported_scores_are_round_1_s_f_bit_for_bit(self):
        """raw is the S_f behind the E-value: the same bits as a direct score_fwd_rev of the pair."""
        world = self.f1_world()
        res, _ = world.align(_f1_mixed(range(0, 50, 5)), max_evalue=None, panel_policy="never")
        for qid, hits in res.items():
            p = next(x for x in _f1() if x["qid"] == qid)
            for h in hits:
                t = next(x for x in _f1() if x["tid"] == h.subject_id)
                with ca._blas_one_thread_here():                 # the aligner's BLAS thread count (OpenBLAS: 1)
                    r = swkern.score_fwd_rev(swkern.query_prep(p["q16"]), t["codes"], [0, len(t["codes"])],
                                             swkern.SETTING_D)
                self.assertEqual(np.float32(h.metrics.score).view(np.uint32), r.s_fwd[0].view(np.uint32))
                self.assertEqual(np.float32(h.metrics.score_rev).view(np.uint32), r.s_rev[0].view(np.uint32))

    def test_round_2_score_mismatch_is_an_internal_error(self):
        world = self.f1_world()
        real = ca._round2_task

        def perturbed(task):
            out = list(real(task))
            out[1] = out[1].copy()
            out[1][0] = np.nextafter(out[1][0], np.float32(np.inf))
            return tuple(out)

        with mock.patch.object(ca, "_round2_task", perturbed):
            with self.assertRaisesRegex(InternalError, "round-2 traceback score"):
                world.align(_f1_own_target(), max_evalue=None, panel_policy="never")


@pytest.mark.usefixtures("fmsw_libs")
class F3Q90Test(_WorldCase):

    def test_q90_of_the_benchmark_queries(self):
        z = np.load(_RES / "q90_queries_v1.npz")
        o = z["offsets"]
        queries = [(str(z["ids"][k]), str(z["sequences"][k]), z["q"][o[k]:o[k + 1]]) for k in range(len(o) - 1)]
        world = self.f1_world()
        qi = world.make_query_index(queries, "f3.queries.f16")
        target = _f1()[0]["tid"]
        res, stats = world.align({q[0]: ([target], [0.5]) for q in queries}, query_index=qi, max_evalue=None,
                                 query_sequences={q[0]: q[1] for q in queries})
        got = np.array([stats["q90"][q[0]] for q in queries])
        self.assertEqual(len(got), 20)
        self.assertLessEqual(float(np.max(np.abs(got - z["q90"]))), 1e-3)
        self.assertEqual(stats["query_domain"], dict(aligned=20, short=0, long=0, nonstandard=0))


# =====================================================================================================================
# invariance: serial == spawn pool, task / block / batch sizes
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class InvarianceTest(_WorldCase):
    """Output bytes (every metrics field bit for bit, every rendered column) do not depend on the scheduling."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # A: the panel on, an E filter that some candidates pass (F1 queries 4 and 7 have significant own pairs; 48
        # and 10 are the shortest); B: no panel, every candidate traced back
        cls.workload_a = dict(prefilter=_f1_mixed([4, 7, 9, 48, 10], per_query=12), max_evalue=10.0)
        cls.workload_b = dict(prefilter=_f1_mixed(per_query=8), max_evalue=None, panel_policy="never")
        world = cls.f1_world()
        cls.ref_a = world.align(cls.workload_a["prefilter"], max_evalue=10.0)
        cls.ref_b = world.align(cls.workload_b["prefilter"], max_evalue=None, panel_policy="never")

    def _same(self, ref, got):
        self.assertEqual(_signature(ref[0]), _signature(got[0]))

    def _run(self, workload, pool=None, **kwargs):
        w = dict(workload)
        return self.f1_world().align(w.pop("prefilter"), pool, **w, **kwargs)

    def test_references_exercise_both_rounds(self):
        _, sa = self.ref_a
        self.assertTrue(sa["panel"])
        self.assertGreater(sa["survivors"], 0)
        self.assertLess(sa["survivors"], sa["candidate_pairs"])
        _, sb = self.ref_b
        self.assertFalse(sb["panel"])
        self.assertEqual(sb["survivors"], sb["candidate_pairs"])
        self.assertGreater(sb["hits"], 50)

    def test_task_cells_block_floats_and_query_batch(self):
        for kwargs in (dict(task_cells=1 << 12), dict(task_cells=1 << 16, block_floats=1 << 12),
                       dict(block_floats=1 << 20), dict(query_batch=1), dict(query_batch=7, task_cells=5000)):
            with self.subTest(**kwargs):
                got_a = self._run(self.workload_a, **kwargs)
                got_b = self._run(self.workload_b, **kwargs)
                self._same(self.ref_a, got_a)
                self._same(self.ref_b, got_b)
                if "task_cells" in kwargs:
                    self.assertGreater(got_a[1]["round1_tasks"], self.ref_a[1]["round1_tasks"])

    def test_serial_equals_spawn_pool_bit_for_bit(self):
        parent_env = tuple(os.environ.get(k) for k in ca._BLAS_THREAD_VARS)
        with ca.CodeAlignmentPool(3) as pool:
            got_a = self._run(self.workload_a, pool, task_cells=1 << 14)
            got_b = self._run(self.workload_b, pool, task_cells=1 << 14)
            self.assertEqual(pool.start_method, "spawn")
            self.assertEqual((pool.starts, pool.workers), (1, 3))          # one persistent pool, both rounds, 2 calls
            heavy = list(pool._pool.map(eval, ["sorted(m for m in __import__('sys').modules "
                                               "if m.split('.')[0] in ('torch', 'biotite'))"] * 12))
            # the workers were spawned with BLAS on one thread (S6), the parent's environment is left as it was
            blas_env = list(pool._pool.map(eval, [f"tuple(__import__('os').environ.get(k) "
                                                  f"for k in {ca._BLAS_THREAD_VARS!r})"] * 12))
        self.assertEqual(set(blas_env), {("1",) * len(ca._BLAS_THREAD_VARS)})
        self.assertEqual(tuple(os.environ.get(k) for k in ca._BLAS_THREAD_VARS), parent_env)
        self.assertEqual(got_a[1]["workers"], 3)
        self._same(self.ref_a, got_a)
        self._same(self.ref_b, got_b)
        self.assertEqual([m for ms in heavy for m in ms], [], "a Stage-2 worker imported torch or biotite")

    def test_worker_count_does_not_change_the_output(self):
        got = self._run(self.workload_b, 2, query_batch=13, task_cells=1 << 14)   # a pool for this call only
        self.assertEqual(got[1]["workers"], 2)
        self._same(self.ref_b, got)

    def test_pool_restarts_when_the_worker_setup_changes(self):
        world = self.f1_world()
        f1 = _f1()
        other = world.make_query_index([(p["qid"], p["qseq"], p["q16"]) for p in f1[:10]], "other.queries.f16")
        pre = _f1_mixed(range(10), per_query=6)
        with ca.CodeAlignmentPool(2) as pool:
            a, _ = world.align(pre, pool, max_evalue=None, panel_policy="never", task_cells=1 << 12)
            b, _ = world.align(pre, pool, max_evalue=None, panel_policy="never", task_cells=1 << 12,
                               query_index=other)
            self.assertEqual(pool.starts, 2)
        self.assertEqual(_signature(a), _signature(b))


# =====================================================================================================================
# profiling mode (P7.1)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class ProfileTest(_WorldCase):
    """FOLDMATCH_STAGE2_PROFILE: the time breakdown is recorded, every task is accounted for, and the hits are the same
    bit for bit as without it (in-process and with a spawn pool)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.prefilter = _f1_mixed([4, 7, 9, 48, 10], per_query=12)
        with mock.patch.dict(os.environ, {stage2_profile.PROFILE_ENV: ""}):
            cls.ref = cls.f1_world().align(cls.prefilter, max_evalue=10.0)

    def _profiled(self, pool=None, **kwargs):
        with mock.patch.dict(os.environ, {stage2_profile.PROFILE_ENV: "1"}):
            return self.f1_world().align(self.prefilter, pool, max_evalue=10.0, **kwargs)

    def _check_accounting(self, stats):
        p = stats["profile"]
        w = p["workers"]
        self.assertEqual(w["panel"]["tasks"] + w["candidates"]["tasks"], stats["round1_tasks"])
        self.assertEqual(w["round2"]["tasks"], stats["round2_tasks"])
        self.assertEqual(w["panel"]["pairs"], stats["panel_pairs"])
        self.assertEqual(w["candidates"]["pairs"], stats["candidate_pairs"])
        self.assertEqual(w["round2"]["pairs"], stats["survivors"])
        self.assertEqual(len(p["tasks"]), stats["round1_tasks"] + stats["round2_tasks"])
        for kind in ("panel", "candidates", "round2"):
            k = w[kind]
            parts = sum(k[c] for c in ("prep", "sim", "kernel", "paths", "distinct", "driver", "other"))
            self.assertAlmostEqual(parts, k["seconds"], delta=1e-9 + 1e-9 * k["seconds"], msg=kind)
            self.assertGreater(k["sim"], 0.0, kind)
            self.assertGreater(k["kernel"], 0.0, kind)
        self.assertGreater(w["round2"]["distinct"], 0.0)
        for phase in ("setup", "collect", "lookup", "plan_r1", "round1", "consume_r1", "statistics", "fetch",
                      "plan_r2", "round2", "consume_r2", "hits_assemble", "hits_metrics", "hits_objects",
                      "hits_sort", "finish", "total"):
            self.assertIn(phase, p["parent"])
        self.assertGreater(p["memory"]["round2_parts_peak_bytes"], 0)
        self.assertEqual(p["memory"]["round2_parts_peak_survivors"], stats["survivors"])
        self.assertEqual([r["name"] for r in p["ipc"]["rounds"]], ["round1", "round2"])
        return p

    def test_the_mode_is_off_unless_asked_for(self):
        self.assertNotIn("profile", self.ref[1])
        for value in ("", "0", "false", "No", " off "):
            with mock.patch.dict(os.environ, {stage2_profile.PROFILE_ENV: value}):
                self.assertFalse(stage2_profile.profiling_enabled(), value)
        for value in ("1", "yes", "true"):
            with mock.patch.dict(os.environ, {stage2_profile.PROFILE_ENV: value}):
                self.assertTrue(stage2_profile.profiling_enabled(), value)

    def test_in_process_profile_keeps_the_hits_and_accounts_for_every_task(self):
        res, stats = self._profiled(task_cells=1 << 14)
        self.assertEqual(_signature(res), _signature(self.ref[0]))
        p = self._check_accounting(stats)
        self.assertEqual(p["pool"]["executors"], 0)
        self.assertGreater(p["memory"]["in_process_rss_max_bytes"], 0)
        self.assertEqual(p["ipc"]["task_bytes"], 0)                     # nothing is pickled in-process

    def test_pool_profile_keeps_the_hits_and_records_the_worker_start_up(self):
        with ca.CodeAlignmentPool(2) as pool:
            res, stats = self._profiled(pool, task_cells=1 << 14)
            again, stats2 = self._profiled(pool, task_cells=1 << 14)
            self.assertEqual(pool.starts, 1)
        self.assertEqual(_signature(res), _signature(self.ref[0]))
        self.assertEqual(_signature(again), _signature(self.ref[0]))
        p = self._check_accounting(stats)
        self.assertEqual((p["pool"]["executors"], p["pool"]["workers"]), (1, 2))
        starts = p["pool"]["worker_starts"]
        self.assertTrue(1 <= len(starts) <= 2)
        for s in starts:
            self.assertGreater(s["init_seconds"], 0.0)
            self.assertGreaterEqual(s["ready_after_submit"], 0.0)
        self.assertGreater(p["ipc"]["task_bytes"], 0)
        self.assertGreater(p["ipc"]["result_bytes"], 0)
        self.assertEqual(p["ipc"]["submit_to_start"]["n"], stats["round1_tasks"] + stats["round2_tasks"])
        self.assertGreater(p["memory"]["worker_rss_max_bytes"], 0)
        for r in p["ipc"]["rounds"]:
            self.assertGreater(r["utilisation"], 0.0)
            self.assertLessEqual(r["utilisation"], 1.0 + 1e-6)
        # each worker process reports its start-up once (with its first task), and the warm pool starts nothing
        pids = [s["pid"] for s in starts + stats2["profile"]["pool"]["worker_starts"]]
        self.assertEqual(len(pids), len(set(pids)))
        self.assertLessEqual(len(pids), 2)
        self.assertEqual(stats2["profile"]["pool"]["executors"], 0)

    def test_switching_the_mode_restarts_the_pool_and_the_hits_stay_the_same(self):
        with ca.CodeAlignmentPool(2) as pool:
            on, _ = self._profiled(pool, task_cells=1 << 14)
            with mock.patch.dict(os.environ, {stage2_profile.PROFILE_ENV: ""}):
                off, stats = self.f1_world().align(self.prefilter, pool, max_evalue=10.0, task_cells=1 << 14)
            self.assertEqual(pool.starts, 2)
        self.assertNotIn("profile", stats)
        self.assertEqual(_signature(on), _signature(off))

    def test_kernel_wrapper_timings_do_not_change_the_results(self):
        f1 = _f1()
        q16 = f1[3]["q16"]
        codes = np.concatenate([p["codes"] for p in f1[:20]])
        offsets = np.concatenate([[0], np.cumsum([len(p["codes"]) for p in f1[:20]])])
        qp = swkern.query_prep(q16)
        a = swkern.score_fwd_rev(qp, codes, offsets, swkern.SETTING_D, block_floats=1 << 13)
        t1 = {}
        b = swkern.score_fwd_rev(qp, codes, offsets, swkern.SETTING_D, block_floats=1 << 13, timings=t1)
        for x, y in ((a.s_fwd, b.s_fwd), (a.s_rev, b.s_rev)):
            np.testing.assert_array_equal(x.view(np.uint32), y.view(np.uint32))
        self.assertEqual(set(t1), {"sim", "kernel", "blocks"})
        self.assertGreater(t1["blocks"], 1)
        c = swkern.align(qp, codes, offsets, swkern.SETTING_D, block_floats=1 << 13)
        t2 = {}
        d = swkern.align(qp, codes, offsets, swkern.SETTING_D, block_floats=1 << 13, timings=t2)
        np.testing.assert_array_equal(c.scores.view(np.uint32), d.scores.view(np.uint32))
        for x, y in ((c.ends, d.ends), (c.path, d.path), (c.path_off, d.path_off)):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(set(t2), {"sim", "kernel", "blocks", "paths"})
        self.assertEqual(t2["blocks"], t1["blocks"])


# =====================================================================================================================
# panel policy (D11), ranking (S14), filters
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class PanelPolicyTest(_WorldCase):

    def test_auto_skips_the_panel_without_e_values_and_the_ranking_is_unchanged(self):
        world = self.f1_world()
        pre = _f1_mixed(range(0, 50, 3), per_query=10)
        with_panel, s1 = world.align(pre, max_evalue=None)                             # evalue / bits requested
        without, s2 = world.align(pre, max_evalue=None, output_needs=["query", "target", "raw", "fident"])
        self.assertTrue(s1["panel"])
        self.assertFalse(s2["panel"])
        self.assertEqual(s2["panel_pairs"], 0)
        self.assertEqual(s2["q90"], {})
        for qid in pre:
            self.assertEqual([h.subject_id for h in with_panel[qid]], [h.subject_id for h in without[qid]])
            for a, b in zip(with_panel[qid], without[qid]):
                self.assertEqual((a.metrics.score, a.metrics.score_rev), (b.metrics.score, b.metrics.score_rev))
                self.assertIsNotNone(a.metrics.log_evalue)
                for name in ("q90", "score_adj", "log_evalue", "evalue", "bit_score"):
                    self.assertIsNone(getattr(b.metrics, name), name)
                self.assertIsNotNone(b.metrics.ecal)
            s = [h.metrics.score - h.metrics.score_rev for h in without[qid]]
            self.assertEqual(s, sorted(s, reverse=True))

    def test_policies(self):
        world = self.f1_world()
        pre = _f1_mixed(range(3), per_query=3)
        _, st = world.align(pre, max_evalue=None, output_needs=["query", "target"], panel_policy="always")
        self.assertTrue(st["panel"])
        _, st = world.align(pre, max_evalue=None, output_needs=["query", "q90"])           # q90 needs the panel
        self.assertTrue(st["panel"])
        _, st = world.align(pre, max_evalue=float("inf"), output_needs=["query", "target"])
        self.assertFalse(st["panel"])
        _, st = world.align(pre, max_evalue=1.0, output_needs=["query", "target"])         # an E filter needs it
        self.assertTrue(st["panel"])
        with self.assertRaisesRegex(ValueError, "cannot filter by E-value"):
            world.align(pre, max_evalue=1e-3, panel_policy="never")
        with self.assertRaisesRegex(ValueError, "panel_policy"):
            world.align(pre, panel_policy="sometimes")

    def test_ties_are_broken_by_embedding_score_then_target_id(self):
        p = _f1()[0]
        targets = [(tid, p["tseq"], p["codes"]) for tid in ("tC", "tA", "tB", "tD")]
        (self.tmp / "ties").mkdir()
        world = World(self.tmp / "ties", targets, [(p["qid"], p["qseq"], p["q16"])], name="ties")
        res, _ = world.align({p["qid"]: (["tC", "tA", "tB", "tD"], [0.5, 0.5, 0.9, 0.1])}, max_evalue=None)
        self.assertEqual([h.subject_id for h in res[p["qid"]]], ["tB", "tA", "tC", "tD"])

    def test_identity_and_coverage_filters(self):
        world = self.f1_world()
        pre = _f1_mixed(range(0, 50, 4), per_query=6)
        everything, _ = world.align(pre, max_evalue=None, panel_policy="never")
        filtered, _ = world.align(pre, max_evalue=None, panel_policy="never", min_seq_identity=0.2,
                                  min_coverage=0.5)
        for qid in pre:
            want = [h.subject_id for h in everything[qid]
                    if h.metrics.identity_aln >= 0.2 and min(h.metrics.query_coverage,
                                                             h.metrics.subject_coverage) >= 0.5]
            self.assertEqual([h.subject_id for h in filtered[qid]], want)
            self.assertTrue(all(h.metrics.aln_len > 0 for h in everything[qid]))
        self.assertGreater(sum(map(len, everything.values())), sum(map(len, filtered.values())))


# =====================================================================================================================
# identity, metrics and strings
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class PathMetricsTest(unittest.TestCase):

    def test_identity_along_a_hand_checked_path(self):
        #        query ACDEFGHIKL vs target ACDXFGHKL:
        #        A C D E F G H I K L
        #        A C D X F G H - K L     -> 8 identical, 1 mismatch (E/X), 1 gap run (I = target gap), 10 columns
        q, t = "ACDEFGHIKL", "ACDXFGHKL"
        path = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, -1), (8, 7), (9, 8)], np.int32)
        cols = ca.path_columns(path, [0, len(path)], np.frombuffer(q.encode(), np.uint8),
                               np.frombuffer(t.encode(), np.uint8), [0])
        got = {k: int(v[0]) for k, v in cols.items()}
        self.assertEqual(got, dict(aln_len=10, n_ident=8, n_aligned=9, mismatch=1, gap_open=1, q_aligned=10,
                                   t_aligned=9, q_start=1, q_end=10, t_start=1, t_end=9))
        self.assertEqual(swkern.cigar_from_path(path), "7M1I2M")

    def test_vectorised_metrics_equal_the_per_path_helper(self):
        """path_columns over many paths at once == swkern.path_metrics path by path (F1 paths, 300 random
        alignments, and empty paths), with targets concatenated."""
        rng = np.random.default_rng(6)
        cases = [(p["path"], p["qseq"], p["tseq"]) for p in _f1()]
        cases.append((np.zeros((0, 2), np.int32), _random_seq(rng, 30), _random_seq(rng, 40)))
        for _kind, q16, codes in swkern.golden_pairs(300, seed=99):
            a = swkern.align(swkern.query_prep(q16), codes, [0, len(codes)], swkern.SETTING_D)
            cases.append((a.path_of(0), _random_seq(rng, len(q16)), _random_seq(rng, len(codes))))
        by_query = {}
        for path, qseq, tseq in cases:
            by_query.setdefault(qseq, []).append((path, tseq))
        for qseq, items in by_query.items():
            path = np.concatenate([np.asarray(p, np.int32).reshape(-1, 2) for p, _ in items])
            path_off = np.concatenate([[0], np.cumsum([len(p) for p, _ in items])])
            t_starts = np.concatenate([[0], np.cumsum([len(t) for _, t in items])[:-1]])
            cols = ca.path_columns(path, path_off, np.frombuffer(qseq.encode(), np.uint8),
                                   np.frombuffer("".join(t for _, t in items).encode(), np.uint8), t_starts)
            for k, (p, tseq) in enumerate(items):
                pm = swkern.path_metrics(p, qseq, tseq)
                self.assertEqual(
                    tuple(int(cols[f][k]) for f in ("aln_len", "n_ident", "mismatch", "gap_open", "q_aligned",
                                                      "t_aligned", "q_start", "q_end", "t_start", "t_end")),
                    (pm.aln_len, pm.n_ident, pm.mismatch, pm.gap_open, pm.q_aligned, pm.t_aligned, pm.q_start,
                     pm.q_end, pm.t_start, pm.t_end))
        self.assertTrue(any(len(p) == 0 for p, _, _ in cases))

    def test_sanitised_bytes_and_nonstandard_count(self):
        for b in range(128):
            ch = chr(b)
            if ch.isspace():
                continue
            self.assertEqual(ca.sanitized_bytes("A" + ch + "C").decode(), alignment.sanitize_sequence("A" + ch + "C"))
        self.assertEqual(ca.nonstandard_count("ACDXBZ*_-"), 6)
        self.assertEqual(ca.nonstandard_count(STANDARD_AMINO_ACIDS), 0)

    def test_vectorised_strings_equal_the_per_path_builders(self):
        """_PathStrings over many paths at once == cigar_from_path / _gapped path by path (empty paths included)."""
        rng = np.random.default_rng(8)
        paths, t_seqs = [], []
        q16 = _f1()[2]["q16"]
        qseq = _random_seq(rng, len(q16))
        for p in _f1()[:30]:
            a = swkern.align(swkern.query_prep(q16), p["codes"], [0, len(p["codes"])], swkern.SETTING_D)
            paths.append(a.path_of(0))
            t_seqs.append(_random_seq(rng, len(p["codes"])))
        paths.insert(3, np.zeros((0, 2), np.int32))
        t_seqs.insert(3, _random_seq(rng, 12))
        paths.append(np.zeros((0, 2), np.int32))
        t_seqs.append(_random_seq(rng, 5))
        path = np.concatenate(paths)
        path_off = np.concatenate([[0], np.cumsum([len(p) for p in paths])])
        t_starts = np.concatenate([[0], np.cumsum([len(t) for t in t_seqs])[:-1]])
        q_bytes = np.frombuffer(qseq.encode(), np.uint8)
        t_bytes = np.frombuffer("".join(t_seqs).encode(), np.uint8)
        strings = ca._PathStrings(path, path_off, q_bytes, t_bytes, t_starts, cigar=True, qaln=True, taln=True)
        for k, (p, tseq) in enumerate(zip(paths, t_seqs)):
            lo, hi = int(path_off[k]), int(path_off[k + 1])
            self.assertEqual(strings.cigar(lo, hi), swkern.cigar_from_path(p), k)
            self.assertEqual(strings.qaln[lo:hi].decode(), ca._gapped(p[:, 0], q_bytes), k)
            self.assertEqual(strings.taln[lo:hi].decode(), ca._gapped(p[:, 1], np.frombuffer(tseq.encode(), np.uint8)),
                             k)
        self.assertTrue(any("I" in swkern.cigar_from_path(p) or "D" in swkern.cigar_from_path(p) for p in paths))

    def test_distinct_codes_are_counted_up_to_the_s13_minimum(self):
        """The round-2 distinct-code count is min(distinct codes, cap), so 'fewer than cap' is decided exactly."""
        rng = np.random.default_rng(12)
        pool = rng.integers(0, 256, (5, 8), dtype=np.uint8)
        cases = [np.zeros((0, 8), np.uint8), pool[:1], np.repeat(pool[:1], 9, axis=0), _f1()[0]["codes"]]
        for size in (2, 3, 7, 40):
            cases.append(pool[rng.integers(0, len(pool), size)])
            cases.append(np.repeat(pool[rng.integers(0, len(pool), 1)], size, axis=0))
        for codes in cases:
            exact = len(np.unique(codes.view(np.uint64))) if len(codes) else 0
            for cap in (1, 2, 3, 5):
                got = ca._distinct_codes(codes, cap)
                self.assertEqual(got, min(exact, cap), (len(codes), exact, cap))
                self.assertEqual(got < cap, exact < cap)


@pytest.mark.usefixtures("fmsw_libs")
class HitFieldsTest(_WorldCase):

    def test_fields_strings_and_identity_on_f1(self):
        world = self.f1_world()
        res, _ = world.align(_f1_mixed(range(0, 50, 7), per_query=5), max_evalue=None)
        seen = 0
        for qid, hits in res.items():
            for h in hits:
                m = h.metrics
                seen += 1
                p = next(x for x in _f1() if x["qid"] == qid)
                t = next(x for x in _f1() if x["tid"] == h.subject_id)
                path = _path_from_hit(m)
                # identity hand-checked along the path: naive loop over the sanitised sequences
                ident = sum(1 for i, j in path.tolist() if i >= 0 and j >= 0 and p["qseq"][i] == t["tseq"][j])
                self.assertEqual(m.n_ident, ident)
                self.assertEqual(m.identity_aln, ident / len(path))
                self.assertEqual(m.identity_shorter, ident / min(len(p["qseq"]), len(t["tseq"])))
                self.assertEqual((m.query_len, m.subject_len), (len(p["qseq"]), len(t["tseq"])))
                self.assertEqual(m.q_aln.replace("-", ""), p["qseq"][m.q_start - 1:m.q_end])
                self.assertEqual(m.t_aln.replace("-", ""), t["tseq"][m.t_start - 1:m.t_end])
                self.assertEqual((m.q_seq, m.t_seq), (p["qseq"], t["tseq"]))
                self.assertEqual(m.query_coverage, (m.q_end - m.q_start + 1) / m.query_len)
                self.assertEqual(m.subject_coverage, (m.t_end - m.t_start + 1) / m.subject_len)
                self.assertEqual(m.aln_len, m.n_ident + m.mismatch + m.q_aln.count("-") + m.t_aln.count("-"))
                spec = _prepared().for_worker()
                self.assertEqual(m.log_evalue, cs.log_evalue(m.score_adj, m.query_len, N_F1, spec))
                self.assertEqual(m.bit_score, cs.bit_score(m.score_adj, m.query_len, spec))
                self.assertEqual(m.evalue, math.exp(m.log_evalue))
                self.assertIsInstance(m.score, float)
                self.assertIsNone(m.pvalue)
        self.assertGreater(seen, 10)

    def test_strings_are_built_only_when_requested(self):
        world = self.f1_world()
        res, _ = world.align(_f1_own_target(), max_evalue=None, output_needs=list(output.DEFAULT_OUTPUT_FIELDS))
        for hits in res.values():
            for h in hits:
                for name in ("cigar", "q_aln", "t_aln", "q_seq", "t_seq"):
                    self.assertIsNone(getattr(h.metrics, name), name)
        res, _ = world.align(_f1_own_target(), max_evalue=None, output_needs=ca.OutputNeeds(cigar=True))
        self.assertTrue(all(h.metrics.cigar and h.metrics.q_aln is None for hs in res.values() for h in hs))


# =====================================================================================================================
# degenerate and tiny inputs (F-deg; D14, S13, S13b)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class DegenerateTest(_WorldCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        z = cls.z = np.load(_RES / "degenerate_v1.npz")
        ro = z["real_offsets"]
        f1 = _f1()[0]
        names = [str(x) for x in z["real_ids"]] + ["base"]
        seqs = [str(x) for x in z["real_sequences"]] + [None]
        assert np.array_equal(z["base_q"], f1["q16"]) and np.array_equal(z["base_codes"], f1["codes"])
        cls.items = [dict(name=names[i], seq=seqs[i], q16=z["real_q"][ro[i]:ro[i + 1]],
                          codes=z["real_codes"][ro[i]:ro[i + 1]]) for i in range(len(ro) - 1)]
        cls.items.append(dict(name="base", seq=f1["qseq"], tseq=f1["tseq"], q16=f1["q16"], codes=f1["codes"]))
        for it in cls.items[:-1]:
            it["tseq"] = it["seq"]
        # a synthetic query of 40 bitwise-identical rows (S13b)
        so = z["syn_q_offsets"]
        k = list(z["syn_ids"].astype(str)).index("identical_q_rows")
        cls.ident_q = z["syn_q"][so[k]:so[k + 1]]
        rng = np.random.default_rng(3)
        queries = [(f"q_{it['name']}", it["seq"], it["q16"]) for it in cls.items]
        queries.append(("q_identical_rows", _random_seq(rng, len(cls.ident_q)), cls.ident_q))
        cls.world = World(cls.tmp, [(f"t_{it['name']}", it["tseq"], it["codes"]) for it in cls.items], queries,
                          name="fdeg")
        targets = [f"t_{it['name']}" for it in cls.items]
        cls.prefilter = {q[0]: (targets, [0.5] * len(targets)) for q in queries}
        logs = _ListHandler()
        log = logging.getLogger("foldmatch.search.code_alignment")
        log.addHandler(logs)
        try:
            cls.res, cls.stats = cls.world.align(cls.prefilter, max_evalue=None)
        finally:
            log.removeHandler(logs)
        cls.logs = [r.getMessage() for r in logs.records if r.levelno >= logging.WARNING]

    def test_single_residue_and_identical_row_queries_report_no_hits(self):
        self.assertEqual(sorted(self.stats["degenerate_queries"]), ["q_Met1", "q_identical_rows"])
        self.assertEqual(self.res["q_Met1"], [])
        self.assertEqual(self.res["q_identical_rows"], [])
        warning = "\n".join(self.logs)
        self.assertIn("2 fully degenerate queries", warning)
        self.assertIn("q_Met1", warning)

    def test_pairs_follow_the_fixture_and_d14(self):
        z = self.z
        names = [it["name"] for it in self.items]
        checked = 0
        for p, (a, b) in enumerate(zip(z["pair_q"], z["pair_t"])):
            qname, tname = f"q_{names[a]}", f"t_{names[b]}"
            if qname == "q_Met1":
                continue
            hits = {h.subject_id: h.metrics for h in self.res[qname]}
            if tname not in hits:
                self.assertEqual(z["pair_path_off"][p + 1] - z["pair_path_off"][p], 0, (qname, tname))
                self.assertEqual(z["pair_S_f"][p], 0.0)
                continue
            m = hits[tname]
            for v in (m.score, m.score_rev, m.q90, m.score_adj, m.log_evalue, m.bit_score):
                self.assertTrue(math.isfinite(v))
            self.assertLessEqual(abs(m.score - z["pair_S_f"][p]), 5e-5 * max(abs(z["pair_S_f"][p]), 1.0))
            self.assertLessEqual(abs(m.score_rev - z["pair_S_r"][p]), 5e-5 * max(abs(z["pair_S_r"][p]), 1.0))
            self.assertLessEqual(abs(m.q90 - z["pair_q90"][p]), 1e-4, qname)
            self.assertLessEqual(abs(m.log_evalue - z["pair_lnE_N1e9"][p]), 1e-4, (qname, tname))
            self.assertEqual(m.ecal, bool(z["pair_ecal_provisional"][p]), (qname, tname))
            checked += 1
        self.assertGreater(checked, 20)

    def test_tiny_and_low_complexity_items_are_aligned_with_ecal_0(self):
        for qname in ("q_allX30", "q_polyA50", "q_His6", "q_GS", "q_angiotensinI"):
            self.assertTrue(self.res[qname], qname)
        for qname in ("q_allX30", "q_His6", "q_GS", "q_angiotensinI"):          # non-standard or short queries
            self.assertTrue(all(not h.metrics.ecal for h in self.res[qname]), qname)
        for qname, hits in self.res.items():
            for h in hits:
                if h.subject_id in ("t_Met1", "t_allX30"):                        # 1 distinct code; non-standard
                    self.assertFalse(h.metrics.ecal, (qname, h.subject_id))
        self.assertTrue(any(h.metrics.ecal for h in self.res["q_polyA50"]))
        self.assertTrue(any(h.metrics.ecal for h in self.res["q_base"]))
        qd = self.stats["query_domain"]
        self.assertEqual((qd["aligned"], qd["short"], qd["nonstandard"]), (6, 3, 1))

    def test_synthetic_cases(self):
        z = self.z
        qo, to = z["syn_q_offsets"], z["syn_t_offsets"]
        rng = np.random.default_rng(11)
        d = SYSTEM.calibrated_domain
        for i, name in enumerate(z["syn_ids"].astype(str)):
            with self.subTest(case=name):
                q16, codes = z["syn_q"][qo[i]:qo[i + 1]], z["syn_codes"][to[i]:to[i + 1]]
                qseq, tseq = _random_seq(rng, len(q16)), _random_seq(rng, len(codes))
                folder = self.tmp / f"syn_{name}"
                folder.mkdir()
                world = World(folder, [("t", tseq, codes)], [("q", qseq, q16)], name="syn")
                if z["syn_expect_error"][i]:
                    with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                        world.align({"q": (["t"], [0.5])}, max_evalue=None)
                    continue
                res, stats = world.align({"q": (["t"], [0.5])}, max_evalue=None)
                degenerate = len(q16) == 1 or bool(np.all(q16.view(np.uint16) == q16[:1].view(np.uint16)))
                self.assertEqual(stats["degenerate_queries"], ["q"] if degenerate else [])
                if degenerate:
                    self.assertEqual(res["q"], [])
                    continue
                if not res["q"]:
                    self.assertEqual(z["syn_S_f"][i], 0.0)
                    continue
                m = res["q"][0].metrics
                self.assertLessEqual(abs(m.score - z["syn_S_f"][i]), 1e-5 * max(abs(z["syn_S_f"][i]), 1.0))
                self.assertLessEqual(abs(m.score_rev - z["syn_S_r"][i]), 1e-5 * max(abs(z["syn_S_r"][i]), 1.0))
                distinct = len(np.unique(codes.view(np.uint64)))
                want_ecal = (d.query_min_length <= len(q16) <= d.query_max_length
                             and len(codes) <= d.target_max_length and distinct >= d.target_min_distinct_codes)
                self.assertEqual(m.ecal, want_ecal)
                if name in ("m1", "identical_codes"):
                    self.assertFalse(m.ecal)


@pytest.mark.usefixtures("fmsw_libs")
class CalibratedDomainTest(_WorldCase):
    """S13 at its limits: queries of 966 and 967 residues, targets of 3,000 and 3,001, and 100-residue sequences with
    exactly 1% (inside) and 2% (outside) non-standard residues on either side. The encodings and codes are an F1
    pair's (query 3: 247 x 275, S_f 415.7) tiled to the lengths, so every pair aligns; ecal must be 1 exactly for
    the pairs whose query and target are both inside, and the once-per-run warning gives the counts."""

    QUERIES = {"q966": (966, 0), "q967": (967, 0), "q100": (100, 0), "q100_x1": (100, 1), "q100_x2": (100, 2)}
    TARGETS = {"t3000": (3000, 0), "t3001": (3001, 0), "t100": (100, 0), "t100_x1": (100, 1), "t100_x2": (100, 2)}
    Q_INSIDE = {"q966", "q100", "q100_x1"}
    T_INSIDE = {"t3000", "t100", "t100_x1"}

    @staticmethod
    def _tile(rows, n):
        return np.ascontiguousarray(np.concatenate([rows] * -(-n // len(rows)))[:n])

    @staticmethod
    def _sequence(rng, n, nonstandard):
        seq = list(_random_seq(rng, n))
        for k in range(nonstandard):
            seq[10 + 37 * k] = "X"
        return "".join(seq)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        p = _f1()[3]
        rng = np.random.default_rng(966)
        targets = [(tid, cls._sequence(rng, m, x), cls._tile(p["codes"], m)) for tid, (m, x) in cls.TARGETS.items()]
        queries = [(qid, cls._sequence(rng, n, x), cls._tile(p["q16"], n)) for qid, (n, x) in cls.QUERIES.items()]
        cls.world = World(cls.tmp, targets, queries, name="domain")
        prefilter = {qid: (list(cls.TARGETS), [0.5] * len(cls.TARGETS)) for qid in cls.QUERIES}
        logs = _ListHandler()
        log = logging.getLogger("foldmatch.search.code_alignment")
        log.addHandler(logs)
        try:
            cls.res, cls.stats = cls.world.align(prefilter, max_evalue=None, panel_policy="never",
                                                 output_needs=["query", "target", "ecal"])
        finally:
            log.removeHandler(logs)
        cls.warnings = [r.getMessage() for r in logs.records if r.levelno == logging.WARNING]

    def test_the_inputs_sit_on_the_limits(self):
        d = SYSTEM.calibrated_domain
        self.assertEqual((d.query_min_length, d.query_max_length, d.target_max_length, d.max_nonstandard_fraction),
                         (21, 966, 3000, 0.01))
        for qid, seq in self.world.query_sequences.items():
            self.assertEqual((len(seq), ca.nonstandard_count(seq)), self.QUERIES[qid])

    def test_ecal_of_every_pair(self):
        for qid in self.QUERIES:
            hits = {h.subject_id: h.metrics.ecal for h in self.res[qid]}
            self.assertEqual(set(hits), set(self.TARGETS), qid)                  # every pair aligned
            for tid, ecal in hits.items():
                with self.subTest(query=qid, target=tid):
                    self.assertIs(ecal, qid in self.Q_INSIDE and tid in self.T_INSIDE)

    def test_counts_and_the_warning(self):
        self.assertEqual(self.stats["query_domain"], dict(aligned=5, short=0, long=1, nonstandard=1))
        self.assertEqual(self.stats["hit_domain"], dict(hits=25, ecal0=16, query_outside=10, target_long=5,
                                                        target_nonstandard=5, target_low_distinct=0))
        self.assertEqual(len(self.warnings), 1)
        for text in ("0 queries shorter than 21 aa, 1 longer than 966 aa, 1 with too many non-standard residues",
                     "16 of 25 hits with ecal=0 (10 for the query, 5 target too long, 5 target non-standard, 0 "
                     "target with too few distinct codes)"):
            self.assertIn(text, self.warnings[0])


# =====================================================================================================================
# refusals and edge cases
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class EdgeCaseTest(_WorldCase):

    def test_empty_database_gives_no_hits(self):
        folder = self.tmp / "empty"
        folder.mkdir()
        p = _f1()[0]
        world = World(folder, [], [(p["qid"], p["qseq"], p["q16"])], name="empty")
        self.assertEqual(world.residues, 0)
        res, stats = world.align({p["qid"]: ([], [])}, db_residues=0, max_evalue=1e-3)
        self.assertEqual(res, {p["qid"]: []})
        self.assertEqual((stats["queries_without_candidates"], stats["candidate_pairs"]), (1, 0))

    def test_queries_without_candidates_are_skipped_before_the_panel(self):
        world = self.f1_world()
        f1 = _f1()
        pre = {f1[0]["qid"]: ([], []), f1[1]["qid"]: ([f1[1]["tid"]], [0.4])}
        res, stats = world.align(pre, max_evalue=None)
        self.assertEqual(res[f1[0]["qid"]], [])
        self.assertEqual(len(res[f1[1]["qid"]]), 1)
        self.assertEqual((stats["queries_without_candidates"], stats["panel_pairs"]), (1, SYSTEM.panel_size))
        self.assertEqual(list(stats["q90"]), [f1[1]["qid"]])

    def test_oversize_pairs_are_skipped_and_counted(self):
        world = self.f1_world()
        f1 = _f1()
        short = min(f1, key=lambda p: len(p["codes"]))
        p = f1[0]
        limit = len(p["q16"]) * (len(short["codes"]) + 1)
        self.assertGreater(len(p["q16"]) * len(p["codes"]), limit)
        handlers = {name: _ListHandler() for name in ("foldmatch.search.code_alignment", "foldmatch.search.swkern")}
        swkern.reset_skipped_pairs()                # swkern warns once per process: make this the first skip
        for name, handler in handlers.items():
            logging.getLogger(name).addHandler(handler)
        try:
            res, stats = world.align({p["qid"]: ([p["tid"], short["tid"]], [0.9, 0.8])}, max_evalue=None,
                                     panel_policy="never", max_pair_cells=limit)
        finally:
            for name, handler in handlers.items():
                logging.getLogger(name).removeHandler(handler)
        self.assertEqual(stats["skipped_pairs"], 1)
        self.assertEqual([h.subject_id for h in res[p["qid"]]], [short["tid"]])
        # one warning, from the parent, with the count (swkern's own once-per-process warning is dropped)
        warnings = {name: [r.getMessage() for r in h.records if r.levelno == logging.WARNING]
                    for name, h in handlers.items()}
        self.assertEqual(warnings["foldmatch.search.swkern"], [])
        self.assertEqual(warnings["foldmatch.search.code_alignment"],
                         [f"esm-codes Stage 2: 1 candidate pair(s) skipped because their similarity matrix (query "
                          f"length x target length) exceeds max_pair_cells={limit} cells (the largest: "
                          f"{len(p['q16'])} x {len(p['codes'])}); they get no score and no hit"])
        self.assertNotIn(ca._DROP_SKIP_WARNING, logging.getLogger("foldmatch.search.swkern").filters)
        with self.assertRaisesRegex(ValueError, "too long for the panel pass"):
            world.align({p["qid"]: ([short["tid"]], [0.8])}, max_evalue=1.0, max_pair_cells=limit)

    def test_refusals(self):
        world = self.f1_world()
        p = _f1()[0]
        pre = {p["qid"]: ([p["tid"]], [0.5])}
        with self.assertRaisesRegex(ValueError, "no query encoding"):
            world.align({"nope": ([p["tid"]], [0.5])})
        with self.assertRaisesRegex(ValueError, "seq_sha1 differs"):
            world.align(pre, query_sequences={p["qid"]: "W" * len(p["qseq"])})
        with self.assertRaisesRegex(ValueError, "residues, its encoding"):
            world.align(pre, query_sequences={p["qid"]: p["qseq"] + "A"})
        with self.assertRaisesRegex(ValueError, "no sequence"):
            world.align(pre, query_sequences={})
        with self.assertRaisesRegex(ValueError, "non-empty database"):
            world.align(pre, db_residues=0)
        with self.assertRaises(ResidueCodeStoreError):
            world.align({p["qid"]: (["not-in-the-store"], [0.5])})
        with self.assertRaisesRegex(ValueError, "no sequence in the sequence store"):
            ca.align_candidates_esm_codes(world.query_index, pre, world.store, lambda ids: {}, _prepared(), N_F1,
                                          None, query_sequences=world.query_sequences, output_needs=ALL_FIELDS,
                                          max_evalue=None)
        with self.assertRaisesRegex(ValueError, "another length"):
            ca.align_candidates_esm_codes(world.query_index, pre, world.store, lambda ids: {i: "ACD" for i in ids},
                                          _prepared(), N_F1, None, query_sequences=world.query_sequences,
                                          output_needs=ALL_FIELDS, max_evalue=None)
        for bad in (dict(max_evalue=-1.0), dict(max_evalue=float("nan")), dict(min_seq_identity=1.5),
                    dict(task_cells=0), dict(query_batch=0), dict(db_residues=-1)):
            with self.subTest(**bad), self.assertRaises(ValueError):
                world.align(pre, **bad)
        with self.assertRaises(TypeError):
            world.align(pre, output_needs="evalue")
        with self.assertRaises(TypeError):
            world.align(pre, pool="four")

    def test_max_evalue_zero_keeps_nothing(self):
        res, stats = self.f1_world().align(_f1_own_target(), max_evalue=0.0)
        self.assertEqual((stats["survivors"], stats["hits"]), (0, 0))
        self.assertTrue(all(h == [] for h in res.values()))


# =====================================================================================================================
# task planning, pool and worker isolation
# =====================================================================================================================
class TaskPlanningTest(unittest.TestCase):

    def test_task_size_adapts_to_the_workers(self):
        self.assertEqual(ca.target_task_cells(10 ** 6, 8, 1 << 24), ca.TASK_CELLS_FLOOR)        # floor 2^20
        self.assertEqual(ca.target_task_cells(10 ** 12, 8, 1 << 24), 1 << 24)                 # cap task_cells
        self.assertEqual(ca.target_task_cells(1 << 30, 8, 1 << 26), (1 << 30) // 32)          # 4 per worker
        self.assertEqual(ca.target_task_cells(10 ** 9, 8, 5000), 5000)                         # a smaller cap wins

    def test_one_interactive_query_uses_every_core(self):
        """A 300-aa query's panel pass alone (107 M cells): at least 4 tasks per worker on 8 workers; on 64 the
        2^20-cell floor allows about 100 tasks, still more than one per worker."""
        lengths = np.diff(_panel_offsets()).tolist()
        for workers in (8, 64):
            total = 300 * sum(lengths)
            chunks = ca.plan_chunks(300, lengths, ca.target_task_cells(total, workers, ca.DEFAULT_TASK_CELLS))
            self.assertGreaterEqual(len(chunks), min(4 * workers, total // ca.TASK_CELLS_FLOOR))
            self.assertGreater(len(chunks), workers)
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(chunks[-1][1], len(lengths))

    def test_chunks_respect_the_target_and_keep_oversize_pairs_alone(self):
        rng = np.random.default_rng(1)
        lengths = rng.integers(1, 3000, 500).tolist()
        for target in (1, 1000, 50_000, 10 ** 9):
            chunks = ca.plan_chunks(10, lengths, target)
            self.assertEqual([k for a, b in chunks for k in range(a, b)], list(range(len(lengths))))
            for a, b in chunks:
                self.assertTrue(b - a == 1 or 10 * sum(lengths[a:b]) <= target)
        self.assertEqual(ca.plan_chunks(5, [], 100), [])

    def test_lpt_order_is_longest_first_and_stable(self):
        self.assertEqual(ca._lpt(["a", "b", "c", "d"], [3, 7, 3, 9]), ["d", "b", "a", "c"])


@functools.lru_cache(maxsize=None)
def _panel_offsets():
    return _prepared().for_worker().load_panel()[1]


@pytest.mark.usefixtures("fmsw_libs")
class PoolAndWorkerTest(_WorldCase):

    def _init(self, world, **kwargs):
        m = world.store.manifest()
        return ca._WorkerInit(isa=swkern.load().isa, spec=_prepared().for_worker(),
                              codes=(str(world.store.bin_path), m["total_residues"], m["code_bytes"]),
                              queries=world.query_index.memmap_args(), block_floats=kwargs.get("block_floats", 1 << 17),
                              max_pair_cells=kwargs.get("max_pair_cells", 1 << 28))

    def _tasks(self, world):
        f1 = _f1()
        p = f1[2]
        entries = world.store.lookup([x["tid"] for x in f1[:6]])
        pairs = np.array([(entries[x["tid"]].offset, entries[x["tid"]].length) for x in f1[:6]], np.int64)
        qe = world.query_index.entries[p["qid"]]
        panel = np.column_stack((_panel_offsets()[:-1], np.diff(_panel_offsets())))[:40]
        return ((0, ca._CODES, qe.offset, qe.length, pairs), (1, ca._PANEL, qe.offset, qe.length, panel),
                (2, qe.offset, qe.length, pairs))

    def test_workers_import_neither_torch_nor_biotite(self):
        """A subprocess with torch and biotite made unimportable initialises a worker and runs a round-1 task
        (candidates and panel) and a round-2 task; the results equal the in-process ones bit for bit."""
        world = self.f1_world()
        init = self._init(world)
        t1, tp, t2 = self._tasks(world)
        ca._init_worker(init, raise_errors=True)
        try:
            with ca._blas_one_thread_here():                     # as in a worker: BLAS on one thread
                want = (ca._round1_task(t1), ca._round1_task(tp), ca._round2_task(t2))
        finally:
            ca._reset_worker_state()
        src = str(Path(foldmatch.__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d) / "in.pkl", Path(d) / "out.pkl"
            inp.write_bytes(pickle.dumps((init, t1, tp, t2)))
            code = (
                "import pickle, sys\n"
                "sys.modules['torch'] = None\n"
                "sys.modules['biotite'] = None\n"
                "from foldmatch.search import code_alignment as ca\n"
                f"init, t1, tp, t2 = pickle.loads(open({str(inp)!r}, 'rb').read())\n"
                "ca._init_worker(init)\n"
                "res = (ca._round1_task(t1), ca._round1_task(tp), ca._round2_task(t2))\n"
                "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'biotite')\n"
                "               and sys.modules[m] is not None)\n"
                f"open({str(out)!r}, 'wb').write(pickle.dumps((res, heavy)))\n")
            env = dict(__import__("os").environ, PYTHONPATH=src)
            proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            got, heavy = pickle.loads(out.read_bytes())
        self.assertEqual(heavy, [])
        for g, w in zip(got, want):
            self.assertEqual(len(g), len(w))
            for a, b in zip(g, w):
                if isinstance(a, np.ndarray):
                    self.assertEqual(a.dtype, b.dtype)
                    self.assertEqual(a.tobytes(), b.tobytes())
                else:
                    self.assertEqual(a, b)
        self.assertGreater(len(got[2][3]), 0)                       # round 2 returned paths

    def test_worker_initialisation_errors_are_raised_by_tasks(self):
        world = self.f1_world()
        init = self._init(world)
        bad = dataclasses.replace(init, spec=dataclasses.replace(init.spec, scoring_sha256="0" * 64))
        ca._init_worker(bad)
        self.assertNotIn(ca._DROP_SKIP_WARNING, logging.getLogger("foldmatch.search.swkern").filters)  # pool only
        try:
            with self.assertRaisesRegex(ss.BundleMismatchError, "sha256"):
                ca._round1_task(self._tasks(world)[0])
        finally:
            ca._reset_worker_state()
        with self.assertRaises(InternalError):
            ca._round1_task(self._tasks(world)[0])
        with self.assertRaises(ss.BundleMismatchError):                     # in-process: raised at once
            ca._init_worker(bad, raise_errors=True)
        ca._reset_worker_state()

    def test_pool_arguments(self):
        self.assertTrue(ca.CodeAlignmentPool(0).serial)
        self.assertTrue(ca.CodeAlignmentPool(1).serial)
        self.assertFalse(ca.CodeAlignmentPool().serial)
        self.assertEqual(ca.CodeAlignmentPool(1).requested_workers(), 1)
        self.assertEqual(ca.CodeAlignmentPool(5).requested_workers(), 5)
        for bad in (-1, 2.0, True):
            with self.assertRaises(ValueError):
                ca.CodeAlignmentPool(bad)
        with self.assertRaises(ValueError):
            ca.CodeAlignmentPool(2, start_method="nope")
        self.assertIn("num_workers=3", repr(ca.CodeAlignmentPool(3)))

    def test_a_worker_that_dies_stops_the_run(self):
        """A worker killed in its task (SIGKILL, as the out-of-memory killer sends) raises WorkerPoolError at once,
        instead of a wait that never ends; the pool is stopped, and the next batch starts a new one."""
        world = self.f1_world()
        init = self._init(world)
        with ca.CodeAlignmentPool(2) as pool:
            t0 = time.perf_counter()
            with _deadline(60), self.assertRaises(ca.WorkerPoolError) as ctx:
                list(pool.run(signal.raise_signal, [int(signal.SIGKILL)] * 3, init))
            self.assertLess(time.perf_counter() - t0, 30)
            self.assertIsInstance(ctx.exception.__cause__, BrokenProcessPool)
            for text in ("out-of-memory killer", "if __name__ == \"__main__\":", "The worker pool was stopped"):
                self.assertIn(text, str(ctx.exception))
            self.assertEqual((pool.workers, pool._pool, pool.starts), (0, None, 1))
            t1, tp, _ = self._tasks(world)
            with _deadline(60):
                got = list(pool.run(ca._round1_task, [t1, tp], init))
            self.assertEqual(sorted(r[0] for r in got), [0, 1])
            self.assertEqual((pool.starts, pool.workers), (2, 2))

    def test_a_script_without_a_main_guard_fails_instead_of_hanging(self):
        """Spawn workers re-import the main script: without an ``if __name__ == "__main__":`` guard they cannot
        start. The run fails with WorkerPoolError (a multiprocessing.Pool respawned them forever); with the guard the
        same script runs."""
        body = """
            from foldmatch.search import code_alignment as ca
            init = ca._WorkerInit(isa="x", spec=None, codes=("", 0, 8), queries=("", 0, 64), block_floats=1,
                                  max_pair_cells=1)
            {guard}
                with ca.CodeAlignmentPool(2) as pool:
                    print("RESULTS", sorted(pool.run(abs, [-1, -2, -3], init)))
            """
        with tempfile.TemporaryDirectory() as d:
            rc, out = _run_script(body.format(guard="if True:"), Path(d))
            self.assertIsNotNone(rc, f"the script without a main guard hung:\n{out[-2000:]}")
            self.assertNotEqual(rc, 0, out[-2000:])
            self.assertIn("WorkerPoolError", out)
            self.assertIn("if __name__ == \"__main__\":", out)
            rc, out = _run_script(body.format(guard="if __name__ == '__main__':"), Path(d))
            self.assertEqual(rc, 0, out[-2000:])
            self.assertIn("RESULTS [1, 2, 3]", out)

    def test_oversize_pairs_are_reported_once_with_a_pool(self):
        """swkern warns once per process about pairs above max_pair_cells; in pool workers that bypassed the
        parent's logging (logging.lastResort), once per worker. Workers drop it and the parent warns once."""
        source = """
            import logging, sys, tempfile
            from pathlib import Path
            if __name__ == "__main__":
                logging.basicConfig(level=logging.WARNING, format="PARENT %(name)s %(levelname)s: %(message)s")
                import test_code_alignment as T
                from foldmatch.search import code_alignment as ca
                f1 = T._f1()
                p = max(f1, key=lambda x: len(x["q16"]))
                targets = [(f"t{k}", x["tseq"], x["codes"]) for k, x in enumerate(f1)]
                with tempfile.TemporaryDirectory() as d:
                    world = T.World(Path(d), targets, [(p["qid"], p["qseq"], p["q16"])], name="skip")
                    with ca.CodeAlignmentPool(3) as pool:
                        _, stats = world.align({p["qid"]: ([t[0] for t in targets], [0.5] * len(targets))}, pool,
                                               max_evalue=None, panel_policy="never", output_needs=["target"],
                                               max_pair_cells=len(p["q16"]) * 60, task_cells=len(p["q16"]) * 200)
                print("SKIPPED", stats["skipped_pairs"], "TASKS", stats["round1_tasks"], "WORKERS", stats["workers"])
            """
        with tempfile.TemporaryDirectory() as d:
            rc, out = _run_script(source, Path(d))
        self.assertEqual(rc, 0, out[-3000:])
        skipped = sum(len(x["codes"]) > 60 for x in _f1())
        self.assertIn(f"SKIPPED {skipped} TASKS", out)
        self.assertGreater(int(out.split("TASKS ")[1].split()[0]), 3)            # every worker got tasks
        self.assertIn("WORKERS 3", out)
        self.assertNotIn(ca._SWKERN_SKIP_WARNING, out)
        lines = [line for line in out.splitlines() if "skipped because their similarity matrix" in line]
        self.assertEqual(len(lines), 1, out[-3000:])
        self.assertTrue(lines[0].startswith(f"PARENT foldmatch.search.code_alignment WARNING: esm-codes Stage 2: "
                                            f"{skipped} candidate pair(s) skipped"), lines[0])

    def test_single_task_batches_stay_out_of_process_without_threadpoolctl(self):
        """A pool asked for workers runs a one-task batch in-process only when BLAS can be limited to one thread
        there; without threadpoolctl it starts one worker instead (same bits as any other batch)."""
        world = self.f1_world()
        pre = {_f1()[0]["qid"]: ([_f1()[1]["tid"]], [0.5])}
        with ca.CodeAlignmentPool(4) as pool:
            ref, st = world.align(pre, pool, max_evalue=None, panel_policy="never")
            self.assertEqual(st["round1_tasks"], 1)
            self.assertEqual(pool.starts, 0 if ca._blas_limit_available() else 1)
        with mock.patch.object(ca, "_blas_limit_available", return_value=False), ca.CodeAlignmentPool(4) as pool:
            got, st = world.align(pre, pool, max_evalue=None, panel_policy="never")
            self.assertEqual((pool.starts, pool.workers, st["workers"]), (1, 1, 1))
        self.assertEqual(_signature(ref), _signature(got))

    def test_serial_pool_releases_its_in_process_state(self):
        world = self.f1_world()
        pool = ca.CodeAlignmentPool(0)
        world.align(_f1_own_target(), pool, max_evalue=None, panel_policy="never")
        self.assertIsNotNone(ca._STATE)                 # kept between calls of a persistent pool
        pool.close()
        self.assertIsNone(ca._STATE)
        world.align(_f1_own_target(), None, max_evalue=None, panel_policy="never")
        self.assertIsNone(ca._STATE)                    # a pool owned by the call is closed after it



# =====================================================================================================================
# P6.5: output fields
# =====================================================================================================================
def _metrics(**kw):
    base = dict(identity_aln=0.5, identity_shorter=0.5, query_coverage=1.0, subject_coverage=1.0, aln_len=10,
                score=12.25, query_len=10, subject_len=10, n_ident=5, mismatch=5)
    base.update(kw)
    return alignment.AlignmentMetrics(**base)


def _row(m, fields):
    return output.format_row("q", alignment.Hit("t", 0.5, m), fields).split(output.DELIMITER)


class OutputRenderingTest(unittest.TestCase):

    def test_field_registry(self):
        self.assertEqual(output.DEFAULT_OUTPUT_FIELDS,
                         ("query", "target", "embscore", "fident", "alnlen", "mismatch", "gapopen", "qstart", "qend",
                          "tstart", "tend", "evalue", "bits"))
        self.assertEqual(output.ESM_CODES_OUTPUT_FIELDS, ("rawrev", "q90", "rawadj", "ecal"))
        self.assertEqual(output.SUPPORTED_OUTPUT_FIELDS[-4:], output.ESM_CODES_OUTPUT_FIELDS)
        self.assertLessEqual(ca.PANEL_FIELDS, set(output.SUPPORTED_OUTPUT_FIELDS))   # the panel gate's fields exist
        self.assertEqual(output.parse_format_output("query,rawrev,q90,rawadj,ecal"),
                         ["query", "rawrev", "q90", "rawadj", "ecal"])

    def test_bits_round_half_up_including_negative_scores(self):
        """esm-codes (a float score): floor(bits + 0.5) (D9, S11); MMseqs2's int(bits + 0.5) would print -1 for
        -1.7 and 0 for -0.6."""
        for bits, want in ((-2.5, "-2"), (-0.4, "0"), (-0.5, "0"), (-0.6, "-1"), (-1.7, "-2"), (0.5, "1"),
                           (2.4999, "2"), (41.5, "42")):
            self.assertEqual(_row(_metrics(bit_score=bits), ["bits"]), [want], bits)
        self.assertEqual(_row(_metrics(), ["bits"]), [""])

    def test_blosum62_bits_keep_mmseqs2_rounding(self):
        """BLOSUM62 (an integer score) keeps int(bits + 0.5), negative bit scores included, so its output is
        byte-identical to before P6 (principle 5)."""
        for bits, want in ((-7.6, "-7"), (-2.5, "-2"), (-0.6, "0"), (-1.7, "-1"), (-0.4, "0"), (0.5, "1"),
                           (2.4999, "2"), (41.5, "42")):
            self.assertEqual(_row(_metrics(score=37, bit_score=bits), ["bits"]), [want], bits)
        # --significance-mode sampled with low gap costs gives negative bit scores (here -3.2 and -7.5)
        q = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKA"
             "LRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVF"
             "LVGIGGKLSDGHRHDVRAPDYDDWUAIGLLKEDGDLTQQALGEAKLDD")
        subjects = {"s_close": q[5:200], "s_far": "GSHMLEDPVDAFQKAIGLLQEHGKEVVHVVGHSMGGRWAKKLSEEGVPVAALVLFAPSKHGLTEPEA",
                    "s_rand": "PLWQERTYIPASDFGHKLCVNM" * 6}
        res = alignment.align_candidates(
            query_sequences={"q1": q},
            prefilter_results={"q1": (list(subjects), [0.9, 0.8, 0.7])},
            fetch_subject_sequences=lambda ids: {i: subjects[i] for i in ids},
            min_seq_identity=0.0, min_coverage=0.0, max_evalue=None, gap_open=3, gap_extend=1, num_workers=0,
            subject_db_size=10_000_000, compute_significance=True,
            significance_mode=alignment.SignificanceMode.sampled, output_fields=["bits"])
        negative = 0
        for h in res["q1"]:
            self.assertIsInstance(h.metrics.score, int)
            self.assertEqual(output.format_row("q1", h, ["bits"]), str(int(h.metrics.bit_score + 0.5)))
            if h.metrics.bit_score < 0 and int(h.metrics.bit_score + 0.5) != math.floor(h.metrics.bit_score + 0.5):
                negative += 1
        self.assertGreater(negative, 0, [h.metrics.bit_score for h in res["q1"]])

    def test_evalue_from_ln_e_never_underflows(self):
        self.assertEqual(_row(_metrics(log_evalue=-1000.0, evalue=0.0), ["evalue"]), ["5.076E-435"])
        self.assertEqual(_row(_metrics(log_evalue=0.0, evalue=1.0), ["evalue"]), ["1.000E+00"])
        self.assertEqual(_row(_metrics(log_evalue=math.log(2.5e-7), evalue=2.5e-7), ["evalue"]), ["2.500E-07"])
        self.assertEqual(_row(_metrics(evalue=3.2e-5), ["evalue"]), ["3.200E-05"])          # BLOSUM62, as before
        self.assertEqual(_row(_metrics(), ["evalue"]), [""])

    def test_esm_codes_scores_and_ecal(self):
        m = _metrics(score=171.25, score_rev=12.5, q90=20.0625, score_adj=138.6875, ecal=True)
        self.assertEqual(_row(m, ["raw", "rawrev", "q90", "rawadj", "ecal"]),
                         ["171.250", "12.500", "20.062", "138.688", "1"])
        self.assertEqual(_row(_metrics(score=0.5, score_adj=-1e-9, q90=-0.0004, ecal=False),
                              ["raw", "rawadj", "q90", "ecal"]), ["0.500", "0.000", "0.000", "0"])
        self.assertEqual(_row(_metrics(score_adj=-2.25), ["rawadj"]), ["-2.250"])

    def test_blosum62_metrics_keep_their_columns_and_leave_the_new_ones_empty(self):
        m = _metrics(score=37, bit_score=19.78, evalue=1.5e-3)
        self.assertEqual(_row(m, ["raw", "bits", "evalue", "rawrev", "q90", "rawadj", "ecal"]),
                         ["37", "20", "1.500E-03", "", "", "", ""])
        res = alignment.align_candidates(
            query_sequences={"q1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
            prefilter_results={"q1": (["s1"], [0.9])},
            fetch_subject_sequences=lambda ids: {"s1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVA"},
            min_seq_identity=0.0, min_coverage=0.0, max_evalue=None, num_workers=1, subject_db_size=1000,
            output_fields=ALL_FIELDS)
        h = res["q1"][0]
        row = dict(zip(ALL_FIELDS, output.format_row("q1", h, ALL_FIELDS).split(output.DELIMITER)))
        self.assertEqual(row["raw"], str(h.metrics.score))
        self.assertIsInstance(h.metrics.score, int)
        self.assertEqual(row["bits"], str(int(h.metrics.bit_score + 0.5)))            # bits >= 0: unchanged
        self.assertEqual(row["evalue"], f"{h.metrics.evalue:.3E}")
        self.assertEqual([row[f] for f in output.ESM_CODES_OUTPUT_FIELDS], ["", "", "", ""])

    def test_tsv_header_names_the_new_fields(self):
        fields = ["query", "target", "raw", "rawrev", "q90", "rawadj", "ecal"]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "hits.tsv"
            m = _metrics(score=3.0, score_rev=1.0, q90=0.5, score_adj=1.5, ecal=True)
            output.write_aligned_results({"q": [alignment.Hit("t", 0.5, m)]}, fields, str(out),
                                         output.FormatMode.tsv)
            lines = out.read_text().splitlines()
        self.assertEqual(lines, ["\t".join(fields), "q\tt\t3.000\t1.000\t0.500\t1.500\t1"])


@pytest.mark.usefixtures("fmsw_libs")
class EsmCodesOutputTest(_WorldCase):

    def test_columns_of_f1_hits(self):
        res, _ = self.f1_world().align(_f1_own_target(), max_evalue=None)
        fields = ["raw", "rawrev", "q90", "rawadj", "ecal", "evalue", "bits"]
        for hits in res.values():
            m = hits[0].metrics
            row = _row(m, fields)
            self.assertEqual(row[:4], [f"{v:.3f}" for v in (m.score, m.score_rev, m.q90, m.score_adj)])
            self.assertEqual(row[4], "1")
            self.assertEqual(row[5], f"{math.exp(m.log_evalue):.3E}")
            self.assertEqual(row[6], str(math.floor(m.bit_score + 0.5)))

    def test_every_field_materialises_when_requested_alone(self):
        """Under esm-codes every supported column renders, also when requested on its own (the E-value columns
        then turn the panel on)."""
        world = self.f1_world()
        p = _f1()[4]
        for field in ALL_FIELDS:
            with self.subTest(field=field):
                res, stats = world.align({p["qid"]: ([p["tid"]], [0.5])}, max_evalue=None, output_needs=[field])
                self.assertEqual(stats["panel"], field in ca.PANEL_FIELDS)
                self.assertNotEqual(output.format_row(p["qid"], res[p["qid"]][0], [field]), "", field)

    def test_skipped_panel_leaves_the_e_value_columns_empty(self):
        fields = ["query", "target", "raw", "rawrev", "ecal", "fident"]
        res, stats = self.f1_world().align(_f1_mixed(range(0, 50, 10), per_query=4), max_evalue=None,
                                           output_needs=fields)
        self.assertFalse(stats["panel"])
        for qid, hits in res.items():
            for h in hits:
                self.assertEqual(output.format_row(qid, h, ["evalue", "bits", "q90", "rawadj"]), "\t\t\t")
                self.assertTrue(all(output.format_row(qid, h, fields).split("\t")))

    def test_degenerate_hits_render_ecal_0(self):
        z = np.load(_RES / "degenerate_v1.npz")
        ro = z["real_offsets"]
        names = list(z["real_ids"].astype(str))
        a, b = names.index("allX30"), names.index("polyA50")
        folder = self.tmp / "ecal0"
        folder.mkdir()
        world = World(folder, [("t", str(z["real_sequences"][b]), z["real_codes"][ro[b]:ro[b + 1]])],
                      [("q", str(z["real_sequences"][a]), z["real_q"][ro[a]:ro[a + 1]])], name="ecal0")
        res, _ = world.align({"q": (["t"], [0.5])}, max_evalue=None)
        self.assertEqual(output.format_row("q", res["q"][0], ["ecal"]), "0")


if __name__ == "__main__":
    unittest.main()
