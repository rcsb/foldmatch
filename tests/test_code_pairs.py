"""The in-memory esm-codes Stage 2 (``foldmatch.search.code_alignment.align_query_codes``) and the scoring-system
resolvers it uses (``scoring_system.resolve`` / ``resolve_system``).

No ESM3: the arrays are the F1 fixtures' (``pairs_v1.npz``) and the F-deg ones, as in ``test_code_alignment``, whose
helpers (``World``, ``_signature``, ``_path_from_hit``, the invariance workloads) this file reuses. The reference is
fm-search's own Stage 2, ``align_candidates_esm_codes``, on the same arrays.
"""
import dataclasses
import hashlib
import logging
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import foldmatch
from foldmatch.errors import RefusalError
from foldmatch.search import code_alignment as ca
from foldmatch.search import output, swkern
from foldmatch.search import scoring_system as ss
from foldmatch.search.code_alignment import (
    AlignedResidues, CodeQuery, CodeTarget, PanelQ90, Provenance, QueryCodeAlignment, align_query_codes, panel_q90,
)
from foldmatch.utils.sequence import seq_sha1
from test_code_alignment import (
    ALL_FIELDS, N_F1, World, _RES, _WorldCase, _assert_path_by_margin, _f1, _f1_mixed, _path_from_hit, _prepared,
    _random_seq, _run_script, _signature,
)

SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)
NO_PANEL_FIELDS = [f for f in ALL_FIELDS if f not in ca.PANEL_FIELDS]   # every column but the panel's
_TESTS = Path(__file__).resolve().parent
OTHER_STATISTICS = dataclasses.replace(SYSTEM, name="esm-codes-test-stats", calibration_version="test")


def _fake_prepared(system=SYSTEM):
    return ss.PreparedSystem(system=system, files={}, bundle={}, encoders_file="e", scoring_file="s")


# =====================================================================================================================
# C1: scoring_system.resolve_system / resolve
# =====================================================================================================================
class ResolveSystemTest(unittest.TestCase):
    """The scoring-system argument without I/O (``resolve_system``) and prepared (``resolve``)."""

    def test_every_accepted_form_gives_the_registry_entry(self):
        prepared = _fake_prepared()
        with mock.patch.object(ss, "prepare", side_effect=AssertionError("resolve_system did I/O")):
            self.assertIs(ss.resolve_system(SYSTEM.name), SYSTEM)
            self.assertIs(ss.resolve_system(SYSTEM), SYSTEM)
            self.assertIs(ss.resolve_system(prepared), SYSTEM)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "scoring_system is required: pass the name, EsmCodesSystem or "
                                                "PreparedSystem"):
            ss.resolve_system(None)
        with self.assertRaises(TypeError):
            ss.resolve_system(3)
        with self.assertRaises(ss.UnknownScoringSystemError):
            ss.resolve_system("esm-codes-v0")
        edited = dataclasses.replace(SYSTEM, calibration_version="edited")
        with self.assertRaisesRegex(ss.UnknownScoringSystemError, "calibration_version"):
            ss.resolve_system(edited)
        with self.assertRaises(ss.UnknownScoringSystemError):
            ss.resolve_system(_fake_prepared(OTHER_STATISTICS))
        with self.assertRaises(ss.UnknownScoringSystemError):
            ss.resolve(_fake_prepared(OTHER_STATISTICS))

    def test_resolve_prepares_once_and_keeps_a_prepared_system(self):
        prepared = _fake_prepared()
        with mock.patch.object(ss, "prepare", return_value=prepared) as prepare:
            self.assertIs(ss.resolve(prepared), prepared)
            prepare.assert_not_called()
            self.assertIs(ss.resolve(SYSTEM.name), prepared)
            prepare.assert_called_once_with(SYSTEM, local_files_only=False)
            ss.resolve(SYSTEM, local_files_only=True)
            prepare.assert_called_with(SYSTEM, local_files_only=True)
        with self.assertRaises(ValueError):
            ss.resolve(None)


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _by_id():
    f1 = _f1()
    return {p["qid"]: p for p in f1}, {p["tid"]: p for p in f1}


def _query(qid, **kw) -> CodeQuery:
    p = _by_id()[0][qid]
    return CodeQuery(p["qseq"], p["q16"], **kw)


def _targets(tids, **kw):
    t = _by_id()[1]
    return {tid: CodeTarget(t[tid]["tseq"], t[tid]["codes"], **kw) for tid in tids}


def _api(qid, tids, emb=None, **kwargs):
    """align_query_codes on the F1 arrays of query ``qid`` and targets ``tids`` (emb scores in that order)."""
    kwargs.setdefault("scoring_system", _prepared())
    kwargs.setdefault("db_residues", N_F1)
    return align_query_codes(_query(qid), _targets(tids), emb_scores=None if emb is None else dict(zip(tids, emb)),
                             **kwargs)


def _api_results(prefilter, **kwargs):
    """``{qid: hits}`` and ``{qid: result}`` of the API on every query of a fm-search prefilter dict."""
    out, res = {}, {}
    for qid, (tids, emb) in prefilter.items():
        r = _api(qid, list(tids), list(emb), **kwargs)
        out[qid], res[qid] = r.hits, r
    return out, res


def _fm_kwargs(api_kwargs: dict) -> dict:
    """fm-search's arguments for the API's (output_needs None = the default columns)."""
    kw = dict(api_kwargs)
    if kw.get("output_needs", "unset") is None:
        kw["output_needs"] = list(output.DEFAULT_OUTPUT_FIELDS)
    return kw


def _s14(h):
    return -(h.metrics.score - h.metrics.score_rev), -h.emb_score, h.subject_id


def _tile(rows, n):
    return np.ascontiguousarray(np.concatenate([rows] * -(-n // len(rows)))[:n])


# =====================================================================================================================
# T1: equivalence with fm-search's Stage 2 on the same arrays
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class EquivalenceTest(_WorldCase):
    """The API's hits are fm-search's ``results[qid]``, bit for bit (``_signature``: every metrics field as hex and
    the rendered row of every column), on fm-search's own invariance workloads and the F1 grid."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test_code_alignment.InvarianceTest's workloads: A the panel on and some survivors, B no panel, all traced
        cls.workload_a = dict(prefilter=_f1_mixed([4, 7, 9, 48, 10], per_query=12), max_evalue=10.0)
        cls.workload_b = dict(prefilter=_f1_mixed(per_query=8), max_evalue=None)

    def _check(self, prefilter, fm_kwargs, api_kwargs, ref=None):
        ref = ref if ref is not None else self.f1_world().align(prefilter, **fm_kwargs)[0]
        got, res = _api_results(prefilter, **api_kwargs)
        self.assertEqual(_signature(ref), _signature(got))
        for qid, r in res.items():
            self.assertEqual(list(r.paths), [h.subject_id for h in r.hits])
        return ref, res

    def test_fm_search_workloads_and_its_scheduling_settings(self):
        world = self.f1_world()
        a, b = dict(self.workload_a), dict(self.workload_b)
        pre_a, pre_b = a.pop("prefilter"), b.pop("prefilter")
        ref_a, sa = world.align(pre_a, **a)
        ref_b, sb = world.align(pre_b, **b, panel_policy="never")
        self.assertTrue(sa["panel"])
        self.assertLess(0, sa["survivors"])
        self.assertLess(sa["survivors"], sa["candidate_pairs"])
        self.assertFalse(sb["panel"])
        self.assertEqual(sb["survivors"], sb["candidate_pairs"])
        # B: fm-search's panel_policy="never" is the API's max_evalue=None with no panel column
        self._check(pre_a, {}, dict(a, output_needs=ALL_FIELDS), ref=ref_a)
        self._check(pre_b, {}, dict(b, output_needs=NO_PANEL_FIELDS), ref=ref_b)
        for kwargs in (dict(task_cells=1 << 12), dict(task_cells=1 << 16, block_floats=1 << 12),
                       dict(block_floats=1 << 20), dict(query_batch=1), dict(query_batch=7, task_cells=5000)):
            with self.subTest(**kwargs):
                fa = world.align(pre_a, **a, **kwargs)[0]
                fb = world.align(pre_b, **b, panel_policy="never", **kwargs)[0]
                self.assertEqual(_signature(fa), _signature(ref_a))
                self.assertEqual(_signature(fb), _signature(ref_b))
                with mock.patch.object(ca, "_API_TASK_CELLS", kwargs.get("task_cells", ca._API_TASK_CELLS)):
                    self._check(pre_a, {}, dict(a, output_needs=ALL_FIELDS), ref=fa)
                    self._check(pre_b, {}, dict(b, output_needs=NO_PANEL_FIELDS), ref=fb)

    def test_f1_grid(self):
        pre = _f1_mixed(range(0, 50, 5), per_query=6)
        for max_evalue in (1e-3, None, float("inf")):
            for fields in (None, ALL_FIELDS):
                for min_id, min_cov in ((0.0, 0.0), (0.2, 0.5)):
                    kw = dict(max_evalue=max_evalue, output_needs=fields, min_seq_identity=min_id,
                              min_coverage=min_cov)
                    with self.subTest(**kw):
                        self._check(pre, _fm_kwargs(kw), kw)

    def test_panel_policies_map_to_fields(self):
        pre = _f1_mixed(range(3), per_query=3)
        # "always": the API asks for the q90 column
        ref, st = self.f1_world().align(pre, max_evalue=None, output_needs=["query", "target", "raw"],
                                        panel_policy="always")
        self.assertTrue(st["panel"])
        _, res = self._check(pre, {}, dict(max_evalue=None, output_needs=["query", "target", "raw", "q90"]), ref=ref)
        self.assertTrue(all(r.q90 is not None for r in res.values()))
        # "never": max_evalue None and no panel column
        ref, st = self.f1_world().align(pre, max_evalue=None, output_needs=["query", "target"], panel_policy="never")
        self.assertFalse(st["panel"])
        _, res = self._check(pre, {}, dict(max_evalue=None, output_needs=["query", "target"]), ref=ref)
        self.assertTrue(all(r.q90 is None for r in res.values()))

    def test_every_form_of_the_scoring_system(self):
        pre = _f1_mixed([4], per_query=4)
        ref = self.f1_world().align(pre, max_evalue=None)[0]
        for system in (SYSTEM.name, SYSTEM, _prepared()):
            with self.subTest(system=type(system).__name__):
                self._check(pre, {}, dict(max_evalue=None, output_needs=ALL_FIELDS, scoring_system=system), ref=ref)

    def test_a_worker_pool_run_gives_the_same_bits(self):
        """fm-search with a spawn pool of 2 workers (BLAS at one thread by the environment) against the API
        in-process. Where threadpoolctl cannot limit this BLAS (Apple Accelerate), the comparison runs in a
        subprocess started with *_NUM_THREADS=1, so the in-process API has the workers' BLAS regime."""
        if ca._blas_controllable():
            pre, kw = self.workload_a["prefilter"], dict(max_evalue=10.0)
            with ca.CodeAlignmentPool(2) as pool:
                ref, stats = self.f1_world().align(pre, pool, task_cells=1 << 14, **kw)
            self.assertEqual(stats["workers"], 2)
            self._check(pre, {}, dict(kw, output_needs=ALL_FIELDS), ref=ref)
            return
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, dict.fromkeys(ca._BLAS_THREAD_VARS, "1")):
            code, out = _run_script("""
                import sys
                from pathlib import Path
                from foldmatch.search import code_alignment as ca
                import test_code_alignment as T
                import test_code_pairs as P

                if __name__ == "__main__":
                    world = T.World(Path("."), [(p["tid"], p["tseq"], p["codes"]) for p in T._f1()],
                                    [(p["qid"], p["qseq"], p["q16"]) for p in T._f1()], name="f1")
                    pre = T._f1_mixed([4, 7, 9, 48, 10], per_query=12)
                    with ca.CodeAlignmentPool(2) as pool:
                        ref, stats = world.align(pre, pool, task_cells=1 << 14, max_evalue=10.0)
                    assert stats["workers"] == 2, stats["workers"]
                    got, _ = P._api_results(pre, max_evalue=10.0, output_needs=T.ALL_FIELDS)
                    assert T._signature(ref) == T._signature(got), "pool and API differ"
                    print("SAME", sum(len(h) for h in ref.values()))
            """, Path(d), timeout=300)
        self.assertEqual(code, 0, out)
        self.assertIn("SAME", out)


# =====================================================================================================================
# T3: invariance (order, splitting, chunking)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class InvarianceTest(_WorldCase):

    QID_INDEX = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        f1 = _f1()
        cls.qid = f1[cls.QID_INDEX]["qid"]
        cls.tids, cls.emb = _f1_mixed([cls.QID_INDEX], per_query=20, stride=3)[cls.qid]
        cls.kw = dict(max_evalue=None, output_needs=ALL_FIELDS)
        cls.ref = _api(cls.qid, cls.tids, cls.emb, **cls.kw)

    def test_target_order_does_not_matter(self):
        order = list(range(len(self.tids)))
        random.Random(3).shuffle(order)
        tids, emb = [self.tids[i] for i in order], [self.emb[i] for i in order]
        got = _api(self.qid, tids, emb, **self.kw)
        self.assertEqual(_signature({"q": got.hits}), _signature({"q": self.ref.hits}))
        self.assertEqual(dict(got.paths), dict(self.ref.paths))

    def test_targets_split_over_calls_merge_to_the_same_hits(self):
        half = len(self.tids) // 2
        a = _api(self.qid, self.tids[:half], self.emb[:half], **self.kw)
        b = _api(self.qid, self.tids[half:], self.emb[half:], q90=a.q90, **self.kw)
        merged = sorted(a.hits + b.hits, key=_s14)
        self.assertEqual(_signature({"q": merged}), _signature({"q": self.ref.hits}))
        paths = {**a.paths, **b.paths}
        self.assertEqual({t: paths[t] for t in self.ref.paths}, dict(self.ref.paths))

    def test_one_call_per_pair_gives_the_same_metrics(self):
        pq = panel_q90(_query(self.qid), scoring_system=_prepared())
        want = {h.subject_id: h for h in self.ref.hits}
        for tid, e in zip(self.tids, self.emb):
            got = _api(self.qid, [tid], [e], q90=pq, **self.kw)
            if tid not in want:
                self.assertEqual(got.hits, [])
                continue
            self.assertEqual(_signature({"q": got.hits}), _signature({"q": [want[tid]]}))
            self.assertEqual(got.paths[tid], self.ref.paths[tid])

    def test_small_chunks_in_both_rounds(self):
        calls = {"score": 0, "trace": 0}
        real_score, real_trace = ca._score_rows, ca._trace_rows

        def score(*a, **k):
            calls["score"] += 1
            return real_score(*a, **k)

        def trace(*a, **k):
            calls["trace"] += 1
            return real_trace(*a, **k)

        with mock.patch.object(ca, "_API_TASK_CELLS", 1 << 12), mock.patch.object(ca, "_score_rows", score), \
                mock.patch.object(ca, "_trace_rows", trace):
            got = _api(self.qid, self.tids, self.emb, **self.kw)
        self.assertGreater(calls["trace"], 2)
        self.assertGreater(calls["score"], len(self.tids))         # the panel's chunks too
        self.assertEqual(_signature({"q": got.hits}), _signature({"q": self.ref.hits}))
        self.assertEqual(dict(got.paths), dict(self.ref.paths))


# =====================================================================================================================
# T4: Q90 reuse
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class Q90ReuseTest(_WorldCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        f1 = _f1()
        cls.qid = f1[7]["qid"]
        cls.tids, cls.emb = _f1_mixed([7], per_query=6)[cls.qid]
        cls.pq = panel_q90(_query(cls.qid), scoring_system=_prepared())

    def test_reuse_equals_the_inline_panel_pass_and_fm_search(self):
        inline = _api(self.qid, self.tids, self.emb, max_evalue=None, output_needs=ALL_FIELDS)
        reused = _api(self.qid, self.tids, self.emb, max_evalue=None, output_needs=ALL_FIELDS, q90=self.pq)
        self.assertEqual(_signature({"q": inline.hits}), _signature({"q": reused.hits}))
        self.assertEqual(inline.q90, self.pq)
        self.assertIs(reused.q90, self.pq)
        self.assertEqual(self.pq.value.hex(), inline.q90.value.hex())
        self.assertEqual(self.pq.encoding_sha256, _query(self.qid).encoding_sha256)
        _, stats = self.f1_world().align({self.qid: (self.tids, self.emb)}, max_evalue=None)
        self.assertEqual(float(stats["q90"][self.qid]).hex(), self.pq.value.hex())

    def test_another_query_or_setup_is_refused(self):
        other = PanelQ90(self.pq.value, hashlib.sha256(b"x").hexdigest(), self.pq.provenance)
        with self.assertRaisesRegex(RefusalError, "another query encoding"):
            _api(self.qid, self.tids, self.emb, max_evalue=None, q90=other)
        other_q = panel_q90(_query(_f1()[3]["qid"]), scoring_system=_prepared())
        with self.assertRaisesRegex(RefusalError, "another query encoding"):
            _api(self.qid, self.tids, self.emb, max_evalue=None, q90=other_q)
        for field, value in (("numpy_version", "0.0"), ("blas", (("openblas", "0.3.0"),)), ("blas_limited", True),
                             ("kernel", "isa=generic"), ("foldmatch_version", "0.0.1")):
            if getattr(self.pq.provenance, field) == value:
                continue
            moved = dataclasses.replace(self.pq, provenance=dataclasses.replace(self.pq.provenance, **{field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(RefusalError, f"another setup.*{field}"):
                _api(self.qid, self.tids, self.emb, max_evalue=None, q90=moved)

    def test_a_pickled_q90_in_another_process(self):
        inline = _api(self.qid, self.tids, self.emb, max_evalue=None, output_needs=ALL_FIELDS)
        with tempfile.TemporaryDirectory() as d:
            inp, out = Path(d) / "q90.pkl", Path(d) / "sig.pkl"
            inp.write_bytes(pickle.dumps((self.pq, self.qid, self.tids, self.emb)))
            code, log = _run_script(f"""
                import pickle
                import test_code_pairs as P
                import test_code_alignment as T
                pq, qid, tids, emb = pickle.loads(open({str(inp)!r}, "rb").read())
                res = P._api(qid, tids, emb, max_evalue=None, output_needs=T.ALL_FIELDS, q90=pq)
                open({str(out)!r}, "wb").write(pickle.dumps((T._signature({{"q": res.hits}}), res.provenance)))
            """, Path(d))
            self.assertEqual(code, 0, log)
            sig, prov = pickle.loads(out.read_bytes())
        self.assertEqual(prov, self.pq.provenance)
        self.assertEqual(sig, _signature({"q": inline.hits}))

    def test_q90_only_where_the_panel_is_needed(self):
        with self.assertRaisesRegex(ValueError, "needs no panel"):
            _api(self.qid, self.tids, self.emb, max_evalue=None, output_needs=["query", "target"], q90=self.pq)
        res = _api(self.qid, self.tids, self.emb, max_evalue=None, output_needs=["query", "target", "raw"])
        self.assertIsNone(res.q90)
        self.assertTrue(all(h.metrics.q90 is None for h in res.hits))
        flat = CodeQuery("A" * 40, np.ones((40, 64), np.float16))
        self.assertIsNone(panel_q90(flat, scoring_system=_prepared()))
        one = CodeQuery("A", np.ones((1, 64), np.float16))
        self.assertIsNone(panel_q90(one, scoring_system=_prepared()))
        with self.assertRaises(TypeError):
            panel_q90(_f1()[0]["q16"], scoring_system=_prepared())
        with self.assertRaises(TypeError):
            PanelQ90(1.0, "x", None)
        with self.assertRaises(ValueError):
            PanelQ90(float("nan"), "x", self.pq.provenance)


# =====================================================================================================================
# T5: golden values through the public API
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class GoldenTest(unittest.TestCase):

    def test_f1_pairs_within_s16(self):
        prepared = _prepared()
        for p in _f1():
            res = align_query_codes(CodeQuery(p["qseq"], p["q16"]), {p["tid"]: CodeTarget(p["tseq"], p["codes"])},
                                    scoring_system=prepared, db_residues=N_F1, max_evalue=None)
            self.assertEqual([h.subject_id for h in res.hits], [p["tid"]])
            m = res.hits[0].metrics
            self.assertLessEqual(abs(m.score - p["S_f"]), 5e-5 * max(abs(p["S_f"]), 1.0), p["qid"])
            self.assertLessEqual(abs(m.score_rev - p["S_r"]), 5e-5 * max(abs(p["S_r"]), 1.0), p["qid"])
            self.assertLessEqual(abs(res.q90.value - p["q90"]), 1e-4, p["qid"])
            self.assertLessEqual(abs(m.log_evalue - p["lnE"]), 1e-4, p["qid"])
            _assert_path_by_margin(self, res.paths[p["tid"]].rows, p["path"], p["q16"], p["codes"], p["S_f"],
                                   p["qid"])
            self.assertEqual((m.aln_len, m.n_ident), (p["aln_cols"], p["n_ident"]))
            self.assertTrue(m.ecal)


# =====================================================================================================================
# T6: paths
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class PathTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        f1 = _f1()
        cls.p = f1[4]
        cls.tids, cls.emb = _f1_mixed([4], per_query=10)[cls.p["qid"]]
        t = _by_id()[1]
        cls.qkeys = [("A", i, "") for i in range(1, len(cls.p["qseq"]) + 1)]
        cls.tkeys = {tid: [f"{tid}:{100 + i}" for i in range(len(t[tid]["tseq"]))] for tid in cls.tids}
        q = CodeQuery(cls.p["qseq"], cls.p["q16"], residue_keys=cls.qkeys)
        targets = {tid: CodeTarget(t[tid]["tseq"], t[tid]["codes"], residue_keys=cls.tkeys[tid]) for tid in cls.tids}
        cls.res = align_query_codes(q, targets, scoring_system=_prepared(), db_residues=N_F1, max_evalue=None,
                                    emb_scores=dict(zip(cls.tids, cls.emb)), output_needs=ALL_FIELDS)
        cls.bare = _api(cls.p["qid"], cls.tids, cls.emb, max_evalue=None, output_needs=["query", "target"])

    def test_rows_are_the_kernel_path(self):
        t = _by_id()[1]
        self.assertGreater(len(self.res.hits), 3)
        self.assertEqual(list(self.res.paths), [h.subject_id for h in self.res.hits])
        for h in self.res.hits:
            rows = self.res.paths[h.subject_id].rows
            self.assertEqual((rows.dtype, rows.ndim, rows.shape[1]), (np.int32, 2, 2))
            np.testing.assert_array_equal(rows, _path_from_hit(h.metrics))
            with ca._blas_one_thread_here():
                run = swkern.run_pair(self.p["q16"], t[h.subject_id]["codes"], reference_sim=False)
            np.testing.assert_array_equal(rows, run.path)
            self.assertEqual((int(rows[-1, 0]) + 1, int(rows[-1, 1]) + 1), run.end)
            m = h.metrics
            q, tt = rows[rows[:, 0] >= 0, 0], rows[rows[:, 1] >= 0, 1]
            self.assertEqual((m.q_start, m.q_end, m.t_start, m.t_end),
                             (q.min() + 1, q.max() + 1, tt.min() + 1, tt.max() + 1))
            self.assertEqual(m.aln_len, len(rows))
            self.assertFalse(rows.flags.writeable)
            with self.assertRaises(ValueError):
                rows[0, 0] = 5
            # the same path whether or not the strings were asked for
            self.assertEqual(self.bare.paths[h.subject_id].rows.tobytes(), rows.tobytes())

    def test_keys_and_hashes(self):
        for h in self.res.hits:
            path = self.res.paths[h.subject_id]
            rows = path.rows
            qk, tk = path.query_keys(), path.target_keys()
            self.assertEqual(len(qk), len(rows))
            for c, (i, j) in enumerate(rows.tolist()):
                self.assertEqual(qk[c], None if i < 0 else self.qkeys[i])
                self.assertEqual(tk[c], None if j < 0 else self.tkeys[h.subject_id][j])
            matched = path.matched()
            self.assertTrue(np.all(matched >= 0))
            self.assertEqual(len(matched), h.metrics.n_ident + h.metrics.mismatch)
            self.assertEqual(path.matched_keys(),
                             [(self.qkeys[i], self.tkeys[h.subject_id][j]) for i, j in matched.tolist()])
            self.assertEqual(path.query_seq_sha1, seq_sha1(self.p["qseq"]))
            self.assertEqual(path.target_seq_sha1, seq_sha1(_by_id()[1][h.subject_id]["tseq"]))
            bare = self.bare.paths[h.subject_id]
            self.assertIsNone(bare.query_keys())
            self.assertIsNone(bare.target_keys())
            self.assertEqual((bare.query_seq_sha1, bare.target_seq_sha1), (path.query_seq_sha1, path.target_seq_sha1))
            with self.assertRaisesRegex(ValueError, "residue_keys on both"):
                bare.matched_keys()
            self.assertNotEqual(bare, path)                   # the keys differ
            self.assertEqual(bare, AlignedResidues(bare.rows.copy(), bare.query_seq_sha1, bare.target_seq_sha1))


# =====================================================================================================================
# T7: indexing and records
# =====================================================================================================================
class _BadArray:
    """Stands in for a CUDA tensor or one that requires grad: numpy conversion raises."""

    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")


class RecordTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = _f1()[0]

    def test_rows_must_be_the_normalised_sequence(self):
        p = self.p
        n, m = len(p["qseq"]), len(p["tseq"])
        with_bos = np.concatenate([p["q16"][:1], p["q16"], p["q16"][:1]])         # an unsliced [L + 2, 64] table
        with self.assertRaisesRegex(RefusalError, f"its sequence has {n} residues, its encoding {n + 2} rows"):
            CodeQuery(p["qseq"], with_bos)
        codes_bos = np.concatenate([p["codes"][:1], p["codes"], p["codes"][:1]])
        with self.assertRaisesRegex(RefusalError, f"its sequence has {m} residues, its codes {m + 2} rows"):
            CodeTarget(p["tseq"], codes_bos)
        with self.assertRaisesRegex(RefusalError, "0 rows"):
            CodeQuery(p["qseq"], np.zeros((0, 64), np.float16))
        with self.assertRaisesRegex(RefusalError, "MissingResidueCodesError"):
            CodeTarget(p["tseq"], np.zeros((0, 8), np.uint8))
        with self.assertRaisesRegex(RefusalError, "0 rows"):
            CodeTarget("", np.zeros((0, 8), np.uint8))

    def test_dtypes_shapes_and_tensors(self):
        p = self.p
        for bad in (p["q16"].astype(">f2"), p["q16"].astype(np.float32), p["q16"][:, :32], p["q16"][0],
                    p["q16"].tolist()):
            with self.subTest(dtype=np.asarray(bad).dtype.str, shape=np.shape(bad)), self.assertRaises(TypeError):
                CodeQuery(p["qseq"], bad)
        for bad in (p["codes"].astype(np.int8), p["codes"][:, :7], p["codes"].astype(np.uint16)):
            with self.subTest(dtype=bad.dtype.str, shape=bad.shape), self.assertRaises(TypeError):
                CodeTarget(p["tseq"], bad)
        with self.assertRaisesRegex(TypeError, r"\.detach\(\)\.cpu\(\)"):
            CodeQuery(p["qseq"], _BadArray())
        with self.assertRaisesRegex(TypeError, r"\.detach\(\)\.cpu\(\)"):
            CodeTarget(p["tseq"], _BadArray())
        with self.assertRaises(TypeError):
            CodeQuery(list(p["qseq"]), p["q16"])
        with self.assertRaises(TypeError):
            CodeQuery(p["qseq"], p["q16"], seq_sha1=seq_sha1(p["qseq"]).hex())
        with self.assertRaises(TypeError):
            CodeQuery(p["qseq"], p["q16"], 3)                     # fields after the first two are keyword-only

    def test_sequence_normalisation_and_sha1(self):
        enc = np.random.default_rng(1).normal(size=(6, 64)).astype(np.float16)
        q = CodeQuery("mkT AYw\n", enc)
        self.assertEqual(q.sequence, "MKTAYW")
        self.assertEqual(q.seq_sha1, seq_sha1("MKTAYW"))
        self.assertEqual(CodeQuery("mkT AYw", enc, seq_sha1=seq_sha1("MKTAYW")).seq_sha1, seq_sha1("MKTAYW"))
        with self.assertRaisesRegex(RefusalError, "seq_sha1 differs"):
            CodeQuery("MKTAYW", enc, seq_sha1=seq_sha1("MKTAYF"))
        with self.assertRaisesRegex(RefusalError, "non-ASCII") as cm:
            CodeQuery("MKTAYé", enc)
        self.assertIsInstance(cm.exception.__cause__, ValueError)
        bad = enc.copy()
        bad[2, 3] = np.inf
        with self.assertRaisesRegex(RefusalError, "NaN or Inf"):
            CodeQuery("MKTAYW", bad)

    def test_keys(self):
        p = self.p
        n = len(p["qseq"])
        q = CodeQuery(p["qseq"], p["q16"], residue_keys=(f"r{i}" for i in range(n)))   # any iterable
        self.assertEqual(q.residue_keys, tuple(f"r{i}" for i in range(n)))
        with self.assertRaisesRegex(RefusalError, "residue_keys for"):
            CodeQuery(p["qseq"], p["q16"], residue_keys=range(n - 1))
        with self.assertRaisesRegex(RefusalError, "unique; 0 names"):
            CodeQuery(p["qseq"], p["q16"], residue_keys=[0] + list(range(n - 1)))
        with self.assertRaisesRegex(RefusalError, "hashable"):
            CodeQuery(p["qseq"], p["q16"], residue_keys=[[i] for i in range(n)])
        with self.assertRaises(TypeError):
            CodeQuery(p["qseq"], p["q16"], residue_keys=5)
        with self.assertRaisesRegex(TypeError, "not a str"):                    # one key per character
            CodeQuery(p["qseq"], p["q16"], residue_keys="x" * n)
        with self.assertRaisesRegex(TypeError, "not a bytes"):
            CodeTarget(p["tseq"], p["codes"], residue_keys=b"x" * len(p["tseq"]))
        with self.assertRaisesRegex(RefusalError, "may not contain None"):      # None marks a gap
            CodeQuery(p["qseq"], p["q16"], residue_keys=[None] + list(range(1, n)))
        with self.assertRaisesRegex(RefusalError, "may not contain None"):
            CodeTarget(p["tseq"], p["codes"], residue_keys=[None] * len(p["tseq"]))

    def test_records_copy_and_are_read_only(self):
        p = self.p
        q16, codes = p["q16"].copy(), p["codes"].copy()
        q, t = CodeQuery(p["qseq"], q16), CodeTarget(p["tseq"], codes)
        q16[:] = 0
        codes[:] = 0
        np.testing.assert_array_equal(q.encoding, p["q16"])
        np.testing.assert_array_equal(t.codes, p["codes"])
        self.assertFalse(q.encoding.flags.writeable or t.codes.flags.writeable)
        self.assertTrue(q.encoding.flags.c_contiguous and t.codes.flags.c_contiguous)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            q.sequence = "A"
        strided = np.asfortranarray(p["q16"])
        self.assertTrue(CodeQuery(p["qseq"], strided).encoding.flags.c_contiguous)
        self.assertEqual(q.encoding_sha256, hashlib.sha256(p["q16"].astype("<f2").tobytes()).hexdigest())
        self.assertEqual((len(q), len(t)), (len(p["qseq"]), len(p["tseq"])))
        self.assertIn("n=", repr(q))

    def test_the_worked_example_of_the_indexing_convention(self):
        """SPEC/README §2.6: query "mkT AYw" (n = 6), target "KTYW" (m = 4), the path [[1,0],[2,1],[3,-1],[4,2],
        [5,3]] (illustrative): CIGAR 2M1I2M, bounds 2-6 / 1-4, 5 columns, 4 identities, 1 gap open."""
        rng = np.random.default_rng(2)
        q = CodeQuery("mkT AYw", rng.normal(size=(6, 64)).astype(np.float16),
                      residue_keys=[("A", i) for i in range(1, 7)])
        t = CodeTarget("KTYW", rng.integers(0, 256, (4, 8), dtype=np.uint8),
                       residue_keys=[f"P12345:{i}" for i in range(101, 105)])
        rows = np.array([[1, 0], [2, 1], [3, -1], [4, 2], [5, 3]], np.int32)
        path = AlignedResidues(rows, q.seq_sha1, t.seq_sha1, query_residue_keys=q.residue_keys,
                               target_residue_keys=t.residue_keys)
        self.assertEqual(swkern.cigar_from_path(rows), "2M1I2M")
        cols = ca.path_columns(rows, np.array([0, 5]), np.frombuffer(ca.sanitized_bytes(q.sequence), np.uint8),
                               np.frombuffer(ca.sanitized_bytes(t.sequence), np.uint8), np.array([0]))
        self.assertEqual({k: int(v[0]) for k, v in cols.items()},
                         dict(aln_len=5, n_ident=4, n_aligned=4, mismatch=0, gap_open=1, q_aligned=5, t_aligned=4,
                              q_start=2, q_end=6, t_start=1, t_end=4))
        self.assertEqual(path.matched().tolist(), [[1, 0], [2, 1], [4, 2], [5, 3]])
        self.assertEqual(path.query_keys(), (("A", 2), ("A", 3), ("A", 4), ("A", 5), ("A", 6)))
        self.assertEqual(path.target_keys(), ("P12345:101", "P12345:102", None, "P12345:103", "P12345:104"))
        self.assertEqual(path.matched_keys(), [(("A", 2), "P12345:101"), (("A", 3), "P12345:102"),
                                               (("A", 5), "P12345:103"), (("A", 6), "P12345:104")])


# =====================================================================================================================
# T8: refusals before any work
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class RefusalOrderTest(unittest.TestCase):
    """Every refusal of the call's arguments fires before the system is prepared, and the panel pre-check before
    any kernel work."""

    @classmethod
    def setUpClass(cls):
        f1 = _f1()
        cls.qid, cls.tids = f1[0]["qid"], [f1[0]["tid"], f1[1]["tid"]]
        cls.q, cls.t = _query(cls.qid), _targets(cls.tids)
        cls.pq = panel_q90(cls.q, scoring_system=_prepared())

    def _refused(self, exc, regex=None, query=None, targets=None, **kwargs):
        kw = dict(scoring_system=SYSTEM.name, db_residues=N_F1)
        kw.update(kwargs)
        with mock.patch.object(ss, "prepare", side_effect=AssertionError("prepare() was reached")), \
                mock.patch.object(swkern, "score_fwd_rev", side_effect=AssertionError("kernel work")), \
                mock.patch.object(swkern, "align", side_effect=AssertionError("kernel work")):
            with self.assertRaisesRegex(exc, regex or ""):
                align_query_codes(self.q if query is None else query, self.t if targets is None else targets, **kw)

    def test_argument_refusals(self):
        t0 = self.tids[0]
        self._refused(TypeError, "CodeQuery", query=_f1()[0]["q16"])
        self._refused(TypeError, "Mapping", targets=list(self.t.values()))
        self._refused(TypeError, "str target ids", targets={1: self.t[t0]})
        self._refused(TypeError, "CodeTarget", targets={t0: _f1()[0]["codes"]})
        self._refused(TypeError, "emb_scores keys", emb_scores={1: 0.5, t0: 0.5})
        self._refused(TypeError, "emb_scores must be a Mapping", emb_scores=[0.5, 0.5])
        self._refused(ValueError, "exactly the targets' ids", emb_scores={t0: 0.5})
        self._refused(ValueError, "finite", emb_scores={t: float("nan") for t in self.tids})
        self._refused(TypeError, "output_needs", output_needs="evalue")
        self._refused(ValueError, r"unknown output_needs field\(s\): 'bit'\. Supported fields: query,",
                      output_needs=["query", "target", "bit"])                  # 'bits' misspelt
        self._refused(ValueError, "unknown output_needs field", max_evalue=None, output_needs=("query", 3))
        self._refused(ValueError, "min_seq_identity", min_seq_identity=1.5)
        self._refused(ValueError, "min_coverage", min_coverage=-0.1)
        self._refused(ValueError, "max_evalue", max_evalue=-1.0)
        for bad in (0, -1, True, 1.5, "10"):
            with self.subTest(db_residues=bad):
                self._refused(ValueError, "pass None for no E-values", db_residues=bad)
        self._refused(ValueError, "needs db_residues", db_residues=None)                   # max_evalue 1e-3
        self._refused(TypeError, "PanelQ90", q90=self.pq.value)
        self._refused(ValueError, "needs no panel", q90=self.pq, max_evalue=None, output_needs=["query", "raw"])
        other = PanelQ90(self.pq.value, "0" * 64, self.pq.provenance)
        self._refused(RefusalError, "another query encoding", q90=other)
        self._refused(ValueError, "scoring_system is required", scoring_system=None)
        self._refused(TypeError, "EsmCodesSystem", scoring_system=3)
        self._refused(ss.UnknownScoringSystemError, "esm-codes-v0", scoring_system="esm-codes-v0")
        moved = dataclasses.replace(self.pq, provenance=dataclasses.replace(self.pq.provenance, numpy_version="0"))
        self._refused(RefusalError, "another setup", q90=moved)

    def test_the_panel_pre_check_comes_before_any_kernel_work(self):
        n = len(self.q.encoding)
        with mock.patch.object(swkern, "score_fwd_rev", side_effect=AssertionError("kernel work")), \
                mock.patch.object(swkern, "align", side_effect=AssertionError("kernel work")), \
                mock.patch.object(ca, "_API_MAX_PAIR_CELLS", n * 997):
            with self.assertRaisesRegex(RefusalError, "too long for the panel pass.*longest panel member \\(998"):
                align_query_codes(self.q, self.t, scoring_system=_prepared(), db_residues=N_F1)
            with self.assertRaisesRegex(RefusalError, "too long for the panel pass"):
                panel_q90(self.q, scoring_system=_prepared())
        # with a reused Q90 there is no panel pass, so no pre-check
        with mock.patch.object(ca, "_API_MAX_PAIR_CELLS", n * 997):
            res = align_query_codes(self.q, self.t, scoring_system=_prepared(), db_residues=N_F1, q90=self.pq,
                                    max_evalue=None)
        self.assertTrue(res.hits)

    def test_the_esm_codes_v1_panel_threshold(self):
        """n > 268,973 at 2^28 cells: the longest esm-codes-v1 panel member has 998 residues."""
        _, _, lengths, longest = ca._api_panel(_prepared(), _prepared().for_worker())
        self.assertEqual((len(lengths), sum(lengths), min(lengths), longest), (1000, 355626, 31, 998))
        self.assertEqual(ca._API_MAX_PAIR_CELLS // longest, 268973)
        ca._check_panel_fits(268973, longest, ca._API_MAX_PAIR_CELLS)
        with self.assertRaises(RefusalError):
            ca._check_panel_fits(268974, longest, ca._API_MAX_PAIR_CELLS)


# =====================================================================================================================
# T9: edge cases
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class EdgeCaseTest(_WorldCase):

    def test_degenerate_queries(self):
        t = _targets([_f1()[0]["tid"]])
        for name, q in (("one row", CodeQuery("M", _f1()[0]["q16"][:1])),
                        ("identical rows", CodeQuery("A" * 40, np.tile(_f1()[0]["q16"][3], (40, 1))))):
            with self.subTest(name):
                res = align_query_codes(q, t, scoring_system=_prepared(), db_residues=N_F1)
                self.assertTrue(res.degenerate)
                self.assertEqual((res.hits, res.paths, res.q90, res.skipped), ([], {}, None, ()))

    def test_a_degenerate_query_too_long_for_the_panel_is_degenerate_as_in_fm_search(self):
        rows = np.tile(_f1()[0]["q16"][3], (40, 1))
        seq = "A" * 40
        p = _f1()[0]
        folder = self.tmp / "deglong"
        folder.mkdir()
        world = World(folder, [(p["tid"], p["tseq"], p["codes"])], [("q", seq, rows)], name="deglong")
        res, stats = world.align({"q": ([p["tid"]], [0.5])}, max_evalue=1.0, max_pair_cells=10)
        self.assertEqual((res["q"], stats["degenerate_queries"]), ([], ["q"]))
        with mock.patch.object(ca, "_API_MAX_PAIR_CELLS", 10), \
                mock.patch.object(ss, "prepare", side_effect=AssertionError("prepare() was reached")):
            got = align_query_codes(CodeQuery(seq, rows), _targets([p["tid"]]), scoring_system=SYSTEM.name,
                                    db_residues=N_F1, max_evalue=1.0)
        self.assertTrue(got.degenerate)

    def test_no_targets(self):
        with mock.patch.object(ss, "prepare", side_effect=AssertionError("prepare() was reached")):
            res = align_query_codes(_query(_f1()[0]["qid"]), {}, scoring_system=SYSTEM.name, db_residues=None,
                                    max_evalue=None, output_needs=["query", "target", "evalue", "bits"])
        self.assertEqual((res.hits, res.paths, res.q90, res.degenerate, res.skipped, res.db_residues),
                         ([], {}, None, False, (), None))
        self.assertIsInstance(res, QueryCodeAlignment)
        self.assertEqual(res.provenance.system_name, SYSTEM.name)
        q = _query(_f1()[0]["qid"])
        pq = panel_q90(q, scoring_system=_prepared())
        with mock.patch.object(ss, "prepare", side_effect=AssertionError("prepare() was reached")):
            res = align_query_codes(q, {}, scoring_system=SYSTEM.name, db_residues=N_F1, q90=pq)   # panel needed
        self.assertEqual((res.hits, res.q90, res.degenerate), ([], None, False))   # no Q90, even the one given

    def test_a_one_residue_target(self):
        """m = 1: aligned like any target, outside the calibrated domain (one distinct code), as in fm-search."""
        z = np.load(_RES / "degenerate_v1.npz")
        ro = z["real_offsets"]
        names = list(z["real_ids"].astype(str))
        k = names.index("Met1")
        tseq, codes = str(z["real_sequences"][k]), z["real_codes"][ro[k]:ro[k + 1]]
        self.assertEqual(len(codes), 1)
        folder = self.tmp / "m1"
        folder.mkdir()
        f1 = _f1()
        queries = [(p["qid"], p["qseq"], p["q16"]) for p in f1[:20]]
        world = World(folder, [("t_Met1", tseq, codes)], queries, name="m1")
        ref, _ = world.align({q[0]: (["t_Met1"], [0.5]) for q in queries}, max_evalue=None)
        hits = 0
        for qid, qseq, q16 in queries:
            res = align_query_codes(CodeQuery(qseq, q16), {"t_Met1": CodeTarget(tseq, codes)},
                                    scoring_system=_prepared(), db_residues=N_F1, max_evalue=None,
                                    emb_scores={"t_Met1": 0.5}, output_needs=ALL_FIELDS)
            self.assertEqual(_signature({qid: res.hits}), _signature({qid: ref[qid]}))
            for h in res.hits:
                hits += 1
                self.assertFalse(h.metrics.ecal)
                self.assertEqual(h.metrics.aln_len, 1)
                self.assertEqual(res.paths["t_Met1"].rows.shape, (1, 2))
        self.assertGreater(hits, 0)

    def test_oversize_pairs_are_skipped(self):
        f1 = _f1()
        short = min(f1, key=lambda p: len(p["codes"]))
        p = f1[0]
        limit = len(p["q16"]) * (len(short["codes"]) + 1)
        ref, stats = self.f1_world().align({p["qid"]: ([p["tid"], short["tid"]], [0.9, 0.8])}, max_evalue=None,
                                           panel_policy="never", max_pair_cells=limit)
        self.assertEqual(stats["skipped_pairs"], 1)
        with mock.patch.object(ca, "_API_MAX_PAIR_CELLS", limit):
            res = _api(p["qid"], [p["tid"], short["tid"]], [0.9, 0.8], max_evalue=None, output_needs=NO_PANEL_FIELDS)
        self.assertEqual(res.skipped, (p["tid"],))
        self.assertEqual(_signature({"q": res.hits}), _signature({"q": ref[p["qid"]]}))
        self.assertEqual(list(res.paths), [short["tid"]])

    def test_without_emb_scores_ties_go_by_id(self):
        p = _f1()[0]
        targets = {tid: CodeTarget(p["tseq"], p["codes"]) for tid in ("tC", "tA", "tB")}
        res = align_query_codes(_query(p["qid"]), targets, scoring_system=_prepared(), db_residues=N_F1,
                                max_evalue=None)
        self.assertEqual([h.subject_id for h in res.hits], ["tA", "tB", "tC"])
        self.assertEqual({h.emb_score for h in res.hits}, {0.0})

    def test_without_n_the_bit_score_and_q90_remain(self):
        pre = _f1_mixed([4, 7], per_query=5)
        ref, _ = self.f1_world().align(pre, max_evalue=None)
        fields = ["query", "target", "evalue", "bits", "q90", "rawadj"]
        for qid, (tids, emb) in pre.items():
            res = _api(qid, list(tids), list(emb), db_residues=None, max_evalue=None, output_needs=fields)
            self.assertIsNone(res.db_residues)
            self.assertEqual([h.subject_id for h in res.hits], [h.subject_id for h in ref[qid]])
            for a, b in zip(res.hits, ref[qid]):
                self.assertIsNone(a.metrics.log_evalue)
                self.assertIsNone(a.metrics.evalue)
                for name in ("bit_score", "q90", "score_adj", "score", "score_rev"):
                    self.assertEqual(float(getattr(a.metrics, name)).hex(), float(getattr(b.metrics, name)).hex())
                row = output.format_row(qid, a, fields).split("\t")
                self.assertEqual(row[2], "")                          # evalue renders empty
                self.assertEqual(row[3], output.format_row(qid, b, ["bits"]))
        with self.assertRaisesRegex(ValueError, "needs db_residues"):
            _api(next(iter(pre)), [_f1()[0]["tid"]], db_residues=None, max_evalue=1.0)

    def test_numpy_integer_n(self):
        pre = _f1_mixed([4], per_query=4)
        qid, (tids, emb) = next(iter(pre.items()))
        a = _api(qid, list(tids), list(emb), max_evalue=None, db_residues=np.int64(N_F1))
        b = _api(qid, list(tids), list(emb), max_evalue=None, db_residues=N_F1)
        self.assertEqual(_signature({"q": a.hits}), _signature({"q": b.hits}))
        self.assertIs(type(a.db_residues), int)


# =====================================================================================================================
# T10: no heavy imports
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class NoHeavyImportsTest(unittest.TestCase):

    def test_the_api_runs_without_torch_and_biotite(self):
        p = _f1()[3]
        tids, emb = _f1_mixed([3], per_query=6)[p["qid"]]
        want = _api(p["qid"], tids, emb, max_evalue=None, output_needs=ALL_FIELDS)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.pkl"
            code = (
                "import sys\n"
                "sys.modules['torch'] = None\n"
                "sys.modules['biotite'] = None\n"
                "import pickle\n"
                "from foldmatch.search import code_alignment as ca\n"
                "from foldmatch.search import scoring_system as ss\n"
                "import test_code_pairs as P, test_code_alignment as T\n"
                f"q = P._query({p['qid']!r})\n"
                "pq = ca.panel_q90(q, scoring_system=ss.DEFAULT_ESM_CODES)\n"
                f"res = ca.align_query_codes(q, P._targets({tids!r}), scoring_system=ss.DEFAULT_ESM_CODES,\n"
                f"    db_residues=T.N_F1, emb_scores=dict(zip({tids!r}, {emb!r})), max_evalue=None,\n"
                "    output_needs=T.ALL_FIELDS, q90=pq)\n"
                "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'biotite')\n"
                "               and sys.modules[m] is not None)\n"
                f"open({str(out)!r}, 'wb').write(pickle.dumps((T._signature({{'q': res.hits}}), heavy)))\n")
            src = str(Path(foldmatch.__file__).resolve().parents[1])
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, str(_TESTS)]))
            proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            sig, heavy = pickle.loads(out.read_bytes())
        self.assertEqual(heavy, [])
        self.assertEqual(sig, _signature({"q": want.hits}))


# =====================================================================================================================
# T12: the BLAS regime; T13: operand alignment
# =====================================================================================================================
def _blas_info():
    try:
        import threadpoolctl
    except ImportError:
        return "threadpoolctl not installed"
    return [{k: i.get(k) for k in ("user_api", "internal_api", "version", "num_threads", "filepath")}
            for i in threadpoolctl.threadpool_info()]


@pytest.mark.usefixtures("fmsw_libs")
class BlasRegimeTest(_WorldCase):

    def test_every_blas_pool_is_at_one_thread_during_the_kernel_work(self):
        if not ca._blas_controllable():
            self.skipTest(f"threadpoolctl finds no BLAS it can limit here (numpy BLAS {ca._numpy_blas()}, "
                          f"threadpool_info {_blas_info()}): the one-thread limit has no effect on this platform")
        import threadpoolctl
        seen = []
        real_score, real_align = swkern.score_fwd_rev, swkern.align

        def spy(fn):
            def wrapped(*a, **k):
                seen.append([i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"])
                return fn(*a, **k)
            return wrapped

        before = [i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"]
        with mock.patch.object(swkern, "score_fwd_rev", spy(real_score)), \
                mock.patch.object(swkern, "align", spy(real_align)):
            res = _api(_f1()[4]["qid"], _f1_mixed([4], per_query=6)[_f1()[4]["qid"]][0], max_evalue=None)
        self.assertTrue(res.provenance.blas_limited)
        self.assertGreater(len(seen), 2)
        self.assertTrue(all(t == [1] * len(t) and t for t in seen), seen)
        self.assertEqual([i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"],
                         before)

    def test_a_multi_threaded_blas_limit_does_not_change_the_bits(self):
        """Under threadpool_limits(cpu_count), a pair big enough for a multi-threaded SGEMM (900 x 3,000) through the
        API equals a 2-worker fm-search pool run bit for bit."""
        if not ca._blas_controllable():
            self.skipTest(f"threadpoolctl finds no BLAS it can limit here (numpy BLAS {ca._numpy_blas()}, "
                          f"threadpool_info {_blas_info()}): neither the limit under test nor the API's one-thread "
                          "limit takes effect")
        import threadpoolctl
        p = _f1()[3]
        rng = np.random.default_rng(900)
        q16, codes = _tile(p["q16"], 900), _tile(p["codes"], 3000)
        qseq, tseq = _random_seq(rng, 900), _random_seq(rng, 3000)
        folder = self.tmp / "big"
        folder.mkdir()
        world = World(folder, [("t", tseq, codes), ("u", p["tseq"], p["codes"])], [("q", qseq, q16)], name="big")
        with ca.CodeAlignmentPool(2) as pool:
            ref, stats = world.align({"q": (["t", "u"], [0.5, 0.4])}, pool, max_evalue=None, task_cells=1 << 20)
        self.assertEqual(stats["workers"], 2)
        with threadpoolctl.threadpool_limits(limits=os.cpu_count(), user_api="blas"):
            got = align_query_codes(CodeQuery(qseq, q16), {"t": CodeTarget(tseq, codes),
                                                           "u": CodeTarget(p["tseq"], p["codes"])},
                                    scoring_system=_prepared(), db_residues=N_F1, max_evalue=None,
                                    emb_scores={"t": 0.5, "u": 0.4}, output_needs=ALL_FIELDS)
        self.assertEqual(_signature({"q": got.hits}), _signature(ref))


@pytest.mark.usefixtures("fmsw_libs")
class OperandAlignmentTest(unittest.TestCase):
    """The S6 GEMM's bits do not depend on where its operands start: qp at float offsets 1-15 from any alignment,
    and targets as consecutive rows (one decode) or scattered rows (one decode per target)."""

    def test_offsets_and_scattered_rows(self):
        f1 = _f1()
        rng = np.random.default_rng(13)
        q16 = f1[5]["q16"]
        targets = [p["codes"] for p in f1[:12]]
        lengths = np.array([len(c) for c in targets], np.int64)
        contiguous = np.concatenate(targets)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        filler = [rng.integers(0, 256, (int(rng.integers(1, 9)), 8), dtype=np.uint8) for _ in targets]
        scattered_parts, scattered_starts, at = [], [], 0
        for c, f in zip(targets, filler):
            scattered_parts += [f, c]
            scattered_starts.append(at + len(f))
            at += len(f) + len(c)
        scattered = np.concatenate(scattered_parts)
        with ca._blas_one_thread_here():
            qp = swkern.query_prep(q16)
            ref = swkern.score_fwd_rev(qp, contiguous, starts, swkern.SETTING_D, lengths=lengths)
            ref_a = swkern.align(qp, contiguous, starts, swkern.SETTING_D, lengths=lengths)
            for off in range(0, 16):
                buf = np.empty(qp.size + 16, np.float32)
                moved = buf[off:off + qp.size].reshape(qp.shape)
                moved[:] = qp
                for codes, st in ((contiguous, starts), (scattered, np.array(scattered_starts, np.int64))):
                    with self.subTest(offset=off, scattered=codes is scattered):
                        r = swkern.score_fwd_rev(moved, codes, st, swkern.SETTING_D, lengths=lengths)
                        a = swkern.align(moved, codes, st, swkern.SETTING_D, lengths=lengths)
                        np.testing.assert_array_equal(r.s_fwd.view(np.uint32), ref.s_fwd.view(np.uint32))
                        np.testing.assert_array_equal(r.s_rev.view(np.uint32), ref.s_rev.view(np.uint32))
                        np.testing.assert_array_equal(a.scores.view(np.uint32), ref_a.scores.view(np.uint32))
                        np.testing.assert_array_equal(a.path, ref_a.path)
                        np.testing.assert_array_equal(a.path_off, ref_a.path_off)


# =====================================================================================================================
# T14: threads
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class _FakeBlasLimiter:
    """threadpoolctl's limiter of an OpenMP BLAS, as :func:`ca._blas_one_thread_here` describes it: the limit sets
    the OpenMP runtime's count (``set_threads``) to 1, and the restore sets the count recorded at entry, here a
    stale one (``stale``; None: the count at entry)."""

    def __init__(self, get_threads, set_threads, stale=None):
        self.set_threads = set_threads
        self.original = get_threads() if stale is None else stale
        set_threads(1)

    def restore_original_limits(self):
        self.set_threads(self.original)


class ThreadTest(unittest.TestCase):

    def test_thread_counts_come_back_with_faiss_loaded(self):
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss is not installed")
        if not ca._blas_controllable():
            self.skipTest(f"threadpoolctl finds no BLAS it can limit here, so nothing changes a thread count "
                          f"(threadpool_info {_blas_info()}); test_a_stale_openmp_restore_is_set_back covers the "
                          f"restore")
        qid = _f1()[4]["qid"]
        tids = _f1_mixed([4], per_query=4)[qid][0]
        before_faiss = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(3)
        try:
            before = (_blas_info(), [t for _, t in ca._openmp_thread_counts()])
            _api(qid, tids, max_evalue=None)
            after = (_blas_info(), [t for _, t in ca._openmp_thread_counts()])
            self.assertEqual(faiss.omp_get_max_threads(), 3)
            self.assertEqual(after, before)
        finally:
            faiss.omp_set_num_threads(before_faiss)

    def test_a_stale_openmp_restore_is_set_back(self):
        """The OpenMP-BLAS case on any machine: the limit puts FAISS's OpenMP runtime at 1 during the kernel work,
        threadpoolctl's restore leaves it at a stale count, and the call sets it back to FAISS's 3."""
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss is not installed")
        qid = _f1()[4]["qid"]
        tids = _f1_mixed([4], per_query=4)[qid][0]
        before_faiss = faiss.omp_get_max_threads()
        during, score = [], ca._api_score

        def spy(*a, **kw):
            during.append(faiss.omp_get_max_threads())
            return score(*a, **kw)

        faiss.omp_set_num_threads(3)
        try:
            if 3 not in [t for _, t in ca._openmp_thread_counts()]:
                self.skipTest(f"threadpoolctl does not list FAISS's OpenMP runtime here ({_blas_info()})")
            limiter = lambda: _FakeBlasLimiter(faiss.omp_get_max_threads, faiss.omp_set_num_threads, stale=128)
            with mock.patch.object(ca, "_limit_blas_threads", limiter), mock.patch.object(ca, "_api_score", spy):
                _api(qid, tids, max_evalue=None)
            self.assertTrue(during)
            self.assertEqual(set(during), {1})
            self.assertEqual(faiss.omp_get_max_threads(), 3)
        finally:
            faiss.omp_set_num_threads(before_faiss)

    def test_an_api_call_overlapping_an_in_process_fm_search_run(self):
        """An API call and fm-search's in-process session overlap in two threads (the API enters first and leaves
        first): BLAS stays at one thread and swkern's warning filter stays on until the session leaves too, and the
        thread count at the API's entry comes back after both."""
        blas = {"threads": 8}
        log = logging.getLogger(swkern.__name__)
        filt = ca._DROP_SKIP_WARNING
        self.assertNotIn(filt, log.filters)
        inside, release = threading.Event(), threading.Event()
        score = ca._api_score

        def paused(*a, **kw):
            if not inside.is_set():
                inside.set()
                self.assertTrue(release.wait(60))
            return score(*a, **kw)

        def get():
            return blas["threads"]

        def put(n):
            blas["threads"] = n

        qid = _f1()[4]["qid"]
        tids = _f1_mixed([4], per_query=4)[qid][0]
        session = ca._Session.__new__(ca._Session)                 # fm-search's in-process session, its enter/exit
        session.in_process, session._exit, session.pending = True, None, set()
        with mock.patch.object(ca, "_limit_blas_threads", lambda: _FakeBlasLimiter(get, put)), \
                mock.patch.object(ca, "_openmp_thread_counts", return_value=[]), \
                mock.patch.object(ca, "_api_score", paused), ThreadPoolExecutor(1) as ex:
            api = ex.submit(_api, qid, tids, max_evalue=None)
            self.assertTrue(inside.wait(60))
            self.assertEqual((blas["threads"], filt in log.filters), (1, True))
            session.__enter__()
            try:
                release.set()
                api.result(timeout=120)
                self.assertEqual((blas["threads"], filt in log.filters), (1, True))    # fm-search still runs
            finally:
                session.__exit__(None, None, None)
        self.assertEqual((blas["threads"], filt in log.filters), (8, False))
        self.assertEqual(ca._REGIME_USERS, 0)

    def test_concurrent_calls_equal_serial_calls(self):
        work = [(p["qid"], _f1_mixed([k], per_query=8)[p["qid"]]) for k, p in enumerate(_f1()[:6])]
        serial = [_signature({"q": _api(qid, list(t), list(e), max_evalue=None, output_needs=ALL_FIELDS).hits})
                  for qid, (t, e) in work]
        barrier = threading.Barrier(2)

        def run(item):
            qid, (t, e) = item
            barrier.wait(timeout=60)
            return _signature({"q": _api(qid, list(t), list(e), max_evalue=None, output_needs=ALL_FIELDS).hits})

        for k in range(0, len(work), 2):
            with ThreadPoolExecutor(2) as ex:
                got = list(ex.map(run, work[k:k + 2]))
            self.assertEqual(got, serial[k:k + 2])


# =====================================================================================================================
# the module's public names and the provenance record
# =====================================================================================================================
class PublicNamesTest(unittest.TestCase):

    def test_all(self):
        for name in ca.__all__:
            self.assertTrue(hasattr(ca, name), name)
        for name in ("CodeQuery", "CodeTarget", "Provenance", "PanelQ90", "AlignedResidues", "QueryCodeAlignment",
                     "align_query_codes", "panel_q90", "align_candidates_esm_codes", "CodeAlignmentPool",
                     "OutputNeeds", "plan_chunks", "WorkerPoolError"):
            self.assertIn(name, ca.__all__)

    @pytest.mark.usefixtures("fmsw_libs")
    def test_provenance(self):
        prov = ca._provenance(SYSTEM, swkern.load())
        self.assertIsInstance(prov, Provenance)
        k = swkern.load()
        self.assertEqual((prov.system_name, prov.system_digest, prov.calibration_version),
                         (SYSTEM.name, SYSTEM.digest(), SYSTEM.calibration_version))
        self.assertIn(k.source_sha256, prov.kernel)
        self.assertIn(f"isa={k.isa}", prov.kernel)
        self.assertEqual((prov.numpy_version, prov.foldmatch_version), (np.__version__, foldmatch.__version__))
        self.assertEqual(prov.blas_limited, ca._numpy_blas_limited())
        self.assertEqual(prov, pickle.loads(pickle.dumps(prov)))

    @pytest.mark.usefixtures("fmsw_libs")
    def test_blas_limited_is_about_numpys_blas(self):
        """``blas_limited`` is True only when threadpoolctl lists numpy's own BLAS: another library's does not
        count (here: Accelerate for numpy, an OpenBLAS of FAISS's)."""
        threadpoolctl = pytest.importorskip("threadpoolctl")
        np_dir = os.path.dirname(os.path.realpath(np.__file__))
        foreign = dict(user_api="blas", internal_api="openblas", num_threads=8, version="0.3.27",
                       filepath="/opt/elsewhere/faiss/lib/libopenblas.so.0")
        wheel = dict(foreign, filepath=os.path.join(np_dir + ".libs", "libscipy_openblas64_-abc.so"))
        omp = dict(user_api="openmp", internal_api="openmp", num_threads=3, version=None, filepath="/x/libomp.dylib")
        cases = [
            ((("accelerate", "unknown"),), [omp, foreign], False),
            ((("accelerate", "unknown"),), [omp], False),
            ((("scipy-openblas", "0.3.27"),), [omp, foreign], True),             # numpy's family
            ((("mkl-sdl", "2024"),), [foreign], False),
            ((("unknown", "unknown"),), [wheel], True),                          # numpy's wheel's file
            ((), [foreign], False),
        ]
        ca._numpy_blas_limited.cache_clear()
        self.addCleanup(ca._numpy_blas_limited.cache_clear)
        for blas, infos, want in cases:
            with self.subTest(blas=blas, libs=[i["filepath"] for i in infos]), \
                    mock.patch.object(ca, "_numpy_blas", return_value=blas), \
                    mock.patch.object(threadpoolctl, "threadpool_info", return_value=infos):
                ca._numpy_blas_limited.cache_clear()
                self.assertIs(ca._numpy_blas_limited(), want)
                self.assertIs(ca._provenance(SYSTEM, swkern.load()).blas_limited, want)
        with mock.patch.dict(sys.modules, {"threadpoolctl": None}):
            ca._numpy_blas_limited.cache_clear()
            self.assertIs(ca._numpy_blas_limited(), False)
