"""Native Smith-Waterman kernel (fmsw, ABI 1) and its ctypes wrapper (PLAN section 7, P3.5 tests 1-9).

The libraries come from the session fixture ``fmsw_libs`` (conftest.py: developer build in place when missing or
stale). Tolerances are the S16 contract: S_f / S_r within 5e-5 max(|ref|, 1); paths identical unless the float64
oracle's best-cell or traceback margin is below 1e-4 + 2e-6 |S_f|; Q90 and ln E within 1e-4.
"""
import functools
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import foldmatch
from foldmatch import _native
from foldmatch._native import build as native_build
from foldmatch.search import swkern

import esm_codes_reference as ref

_RES = Path(__file__).parent / "resources" / "esm_codes"
D = swkern.SETTING_D

# sha256 of the DP golden records (swkern.dp_digest_update) of F1 (50 pairs, pairs_v1.npz order) followed by
# swkern.golden_pairs(1000), from q fp16 through fmsw_sim_ref, setting D. Computed 2026-10-13 on an Apple M1 Max:
# Apple clang 21 neon / generic (arm64), sse2 / generic (x86_64 under Rosetta) and GCC 14 neon / generic all agree.
GOLDEN_F1_SYNTH_SHA256 = "30ec40c6d57f9b7d28c2502b7a40b5ec8e97d46b83ce69d97fbcd44b2fe281e3"
# sha256 of the golden_pairs(1000) input bytes (kind, q fp16, codes): the generator is integer-only and portable
GOLDEN_INPUTS_SHA256 = "6725eda9e247c975f722e85fdefd1d71dbd74c0d2b28ee4c016c2027b634955d"

_AA = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", np.uint8)


# =====================================================================================================================
# data
# =====================================================================================================================
@functools.lru_cache(maxsize=None)
def _f1():
    z = np.load(_RES / "pairs_v1.npz")
    qo, to = z["q_offsets"], z["t_offsets"]
    pairs = []
    for p in range(len(z["pair_q"])):
        a, b = int(z["pair_q"][p]), int(z["pair_t"][p])
        pairs.append(dict(q16=z["q"][qo[a]:qo[a + 1]], codes=z["codes"][to[b]:to[b + 1]],
                          qseq=str(z["q_sequences"][a]), tseq=str(z["t_sequences"][b]),
                          S_f=float(z["S_f"][p]), S_r=float(z["S_r"][p]), S_f_f64=float(z["S_f_f64"][p]),
                          S_r_f64=float(z["S_r_f64"][p]), end=tuple(int(x) for x in z["end_1based"][p]),
                          path=z["path_flat"][z["path_off"][p]:z["path_off"][p + 1]],
                          n_aligned=int(z["n_aligned"][p]), aln_cols=int(z["aln_cols"][p]),
                          n_ident=int(z["n_ident"][p])))
    return pairs


@functools.lru_cache(maxsize=None)
def _golden_pairs():
    return swkern.golden_pairs()


@functools.lru_cache(maxsize=None)
def _evalue_table():
    t = json.loads((_RES / "evalue_grid_v1.json").read_text())["table"]
    return t["center_len"], t["lam"], t["lnK"]


def _random_seq(rng, n):
    return _AA[rng.integers(0, len(_AA), n)].tobytes().decode("ascii")


def _related_codes(rng, q16):
    """Target codes = sign bits of a noisy, gapped copy of a query segment with random flanks."""
    q = q16.astype(np.float32)
    n = len(q)
    a, b = sorted(int(x) for x in rng.integers(0, n, 2))
    seg = q[a:b + 1]
    seg = seg[rng.random(len(seg)) > 0.1]
    ins = rng.standard_normal((int(rng.integers(0, 10)), 64)).astype(np.float32)
    pos = int(rng.integers(0, len(seg) + 1))
    t = np.concatenate([rng.standard_normal((int(rng.integers(0, 60)), 64)).astype(np.float32), seg[:pos], ins,
                        seg[pos:], rng.standard_normal((int(rng.integers(0, 60)), 64)).astype(np.float32)])
    if not len(t):
        t = rng.standard_normal((1, 64)).astype(np.float32)
    t = t + float(rng.uniform(0.3, 1.5)) * rng.standard_normal(t.shape).astype(np.float32)
    return np.packbits(t >= 0, axis=1)


@functools.lru_cache(maxsize=None)
def _property_cases(seed=20261013, queries=50, per_query=10):
    """500 seeded cases: 50 queries (n in [1, 400]) x 10 candidates (m in [1, 400] random, or related)."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(queries):
        n = int(rng.integers(1, 401))
        q16 = rng.standard_normal((n, 64)).astype(np.float16)
        cands = []
        for _ in range(per_query):
            if rng.random() < 0.5:
                codes = rng.integers(0, 256, (int(rng.integers(1, 401)), 8), dtype=np.uint8)
            else:
                codes = _related_codes(rng, q16)
            cands.append(codes)
        lens = [len(c) for c in cands]
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        cases.append(dict(q16=q16, codes=np.concatenate(cands), offsets=offsets, qseq=_random_seq(rng, n),
                          tseqs=[_random_seq(rng, m) for m in lens]))
    return cases


def _kernels():
    """Every kernel build of this machine that can run here (avx2 only on an AVX2 CPU)."""
    out = []
    for isa in _native.machine_isas():
        if _native.find_library(isa) is None:
            continue
        try:
            out.append(swkern.load(isa))
        except swkern.KernelError:
            if isa != "avx2":
                raise
    return out


def _subprocess_env():
    env = dict(os.environ)
    src = str(Path(foldmatch.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


# =====================================================================================================================
# shared checks
# =====================================================================================================================
def _assert_s16_score(tc, got, want, what=""):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    bound = swkern.s16_score_bound(want)
    bad = np.flatnonzero(~(np.abs(got - want) <= bound))
    tc.assertEqual(len(bad), 0, f"{what}: outside S16 at {bad[:5]}: got {got[bad[:5]]} want {want[bad[:5]]}")


def _assert_path_invariants(tc, path, end, score, n, m):
    path = np.asarray(path)
    if len(path) == 0:
        tc.assertEqual(score, 0.0)
        tc.assertEqual(tuple(int(x) for x in end), (0, 0))
        return
    tc.assertGreater(score, 0.0)
    tc.assertTrue(np.all(path[0] >= 0) and np.all(path[-1] >= 0), "both path ends must be aligned pairs")
    tc.assertEqual((int(path[-1, 0]) + 1, int(path[-1, 1]) + 1), tuple(int(x) for x in end))
    tc.assertFalse(np.any((path[:, 0] < 0) & (path[:, 1] < 0)), "double-gap column")
    qi, tj = path[:, 0][path[:, 0] >= 0], path[:, 1][path[:, 1] >= 0]
    tc.assertTrue(np.all(np.diff(qi) == 1) and np.all(np.diff(tj) == 1), "coordinates must be monotone, step 1")
    tc.assertTrue(qi.max() < n and tj.max() < m)


def _assert_cigar_round_trip(tc, path, qseq, tseq):
    pm = swkern.path_metrics(path, qseq, tseq, strings=True)
    ops = swkern.path_ops(path).tobytes().decode("ascii")
    tc.assertEqual(swkern.expand_cigar(pm.cigar), ops)
    nM, nI, nD = ops.count("M"), ops.count("I"), ops.count("D")
    tc.assertEqual(nM + nI + nD, pm.aln_len)
    tc.assertEqual(pm.aln_len, len(path))
    naive_ident = sum(1 for i, j in np.asarray(path).tolist() if i >= 0 and j >= 0 and qseq[i] == tseq[j])
    tc.assertEqual(pm.n_ident, naive_ident)
    tc.assertEqual(pm.mismatch, nM - naive_ident)
    if not len(path):
        tc.assertEqual((pm.cigar, pm.q_aln, pm.t_aln, pm.q_start, pm.q_end, pm.t_start, pm.t_end),
                       ("", "", "", 0, 0, 0, 0))
        return
    tc.assertEqual(pm.q_aln.replace("-", ""), qseq[pm.q_start - 1:pm.q_end])
    tc.assertEqual(pm.t_aln.replace("-", ""), tseq[pm.t_start - 1:pm.t_end])
    tc.assertEqual(pm.q_end - pm.q_start + 1, nM + nI)
    tc.assertEqual(pm.t_end - pm.t_start + 1, nM + nD)
    tc.assertEqual((pm.q_aligned, pm.t_aligned), (nM + nI, nM + nD))
    runs = sum(1 for a, b in zip(" " + ops, ops) if b in "ID" and a != b)
    tc.assertEqual(pm.gap_open, runs)


def _golden_runs(kernel, reference_sim):
    """(F1 + synthetic) PairRun list in golden order."""
    pairs = [(p["q16"], p["codes"]) for p in _f1()] + [(q, c) for _, q, c in _golden_pairs()]
    return [swkern.run_pair(q, c, reference_sim=reference_sim, kernel=kernel) for q, c in pairs], pairs


def _digests(runs, pairs):
    hall, hsyn = swkern.dp_golden_new(), swkern.dp_golden_new()
    for i, (r, (q, c)) in enumerate(zip(runs, pairs)):
        swkern.dp_digest_update(hall, len(q), len(c), r.s_f, r.s_r, r.path)
        if i >= len(_f1()):
            swkern.dp_digest_update(hsyn, len(q), len(c), r.s_f, r.s_r, r.path)
    return hall.hexdigest(), hsyn.hexdigest()


# =====================================================================================================================
# 1. differential F1 (and the P3.6 oracle)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class DifferentialF1Test(unittest.TestCase):

    def test_production_path_matches_research_within_s16(self):
        for k in _kernels():
            sf, sr, paths = [], [], 0
            for p in _f1():
                r = swkern.run_pair(p["q16"], p["codes"], reference_sim=False, kernel=k)
                sf.append(r.s_f)
                sr.append(r.s_r)
                paths += int(np.array_equal(r.path, p["path"]))
                self.assertEqual(r.end, p["end"])
                self.assertEqual(r.s_tb, r.s_f)
            self.assertEqual(paths, 50, k.isa)
            _assert_s16_score(self, sf, [p["S_f"] for p in _f1()], f"{k.isa} S_f vs research")
            _assert_s16_score(self, sr, [p["S_r"] for p in _f1()], f"{k.isa} S_r vs research")
            _assert_s16_score(self, sf, [p["S_f_f64"] for p in _f1()], f"{k.isa} S_f vs float64 oracle")
            _assert_s16_score(self, sr, [p["S_r_f64"] for p in _f1()], f"{k.isa} S_r vs float64 oracle")

    def test_batched_f1_targets_equal_single_pairs(self):
        """One query against all 50 F1 targets in one call: the pair (k, k) equals the single-pair run bit for bit."""
        f1 = _f1()
        codes = np.concatenate([p["codes"] for p in f1])
        offsets = np.concatenate([[0], np.cumsum([len(p["codes"]) for p in f1])]).astype(np.int64)
        for a in (0, 17, 49):
            qp = swkern.query_prep(f1[a]["q16"])
            res = swkern.score_fwd_rev(qp, codes, offsets, D)
            single = swkern.run_pair(f1[a]["q16"], f1[a]["codes"], reference_sim=False)
            self.assertEqual(res.s_fwd[a], np.float32(single.s_f))
            self.assertEqual(res.s_rev[a], np.float32(single.s_r))

    def test_cigar_and_metrics_match_fixture(self):
        for p in _f1():
            r = swkern.run_pair(p["q16"], p["codes"], reference_sim=False)
            pm = swkern.path_metrics(r.path, p["qseq"], p["tseq"], strings=True)
            self.assertEqual((pm.aln_len, pm.n_ident, pm.mismatch + pm.n_ident),
                             (p["aln_cols"], p["n_ident"], p["n_aligned"]))
            _assert_cigar_round_trip(self, r.path, p["qseq"], p["tseq"])


@pytest.mark.usefixtures("fmsw_libs")
class OracleTest(unittest.TestCase):

    def test_verbatim_copy(self):
        self.assertTrue(ref.check_verbatim())
        self.assertEqual(ref.ORACLE_SOURCE_MD5, "7ec19b7e5ff89dc7b44d2512aed3997e")

    def test_s7_wrapper_changes_nothing_on_f1(self):
        for p in _f1():
            Q, T = ref.reference_qprep64(p["q16"]), ref.decode64(p["codes"])
            for Qd in (Q, Q[::-1]):
                sim = Qd @ T.T
                z7, nr, nc = ref.reference_rowcol_s7(sim)
                self.assertEqual((nr, nc), (0, 0))
                self.assertTrue(np.array_equal(z7, ref.normalize_scores(sim, "rowcol")))

    def test_oracle_reproduces_fixture_and_margin_helper_matches_oracle(self):
        for p in _f1()[::7]:
            r = ref.reference_pair64(p["q16"], p["codes"], margins=True)
            self.assertAlmostEqual(r["S_f"], p["S_f_f64"], delta=1e-9 * max(abs(p["S_f_f64"]), 1.0))
            self.assertTrue(np.array_equal(r["path"], p["path"]))
            mg = ref.sw_margins64(r["z"] - D.offset, D.gap_open, D.gap_extend)
            self.assertEqual(mg["score"], r["S_f"])
            self.assertTrue(np.array_equal(mg["path"], r["path"]))
            self.assertGreater(min(mg["best_margin"], mg["tb_margin"]), 0.0)

    def test_reference_evalue_reproduces_f4(self):
        rows = json.loads((_RES / "evalue_grid_v1.json").read_text())["rows"]
        centres, lam, lnK = _evalue_table()
        for r in rows:
            lnE, bits = ref.reference_evalue(r["S_prime"], r["m"], r["N"], centres, lam, lnK)
            self.assertLessEqual(abs(lnE - r["lnE"]), 1e-12 * max(abs(r["lnE"]), 1.0))
            self.assertLessEqual(abs(bits - r["bits"]), 1e-12 * max(abs(r["bits"]), 1.0))

    def test_path_score_helper_reproduces_the_oracle_score(self):
        for p in _f1()[::5]:
            r = ref.reference_pair64(p["q16"], p["codes"])
            self.assertAlmostEqual(swkern.path_score(r["z"], r["path"], D), r["S_f"], delta=1e-9 * abs(r["S_f"]))


# =====================================================================================================================
# 2. properties (500 seeded cases)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class PropertyTest(unittest.TestCase):

    def test_scores_paths_invariants_and_round_trips(self):
        near_ties = compared = 0
        for c in _property_cases():
            qp = swkern.query_prep(c["q16"])
            n = len(qp)
            res = swkern.score_fwd_rev(qp, c["codes"], c["offsets"], D)
            aln = swkern.align(qp, c["codes"], c["offsets"], D)
            aln_score_only = swkern.align(qp, c["codes"], c["offsets"], D, traceback=False)
            self.assertTrue(np.all(np.isfinite(res.s_fwd)) and np.all(np.isfinite(res.s_rev)))
            self.assertTrue(np.all(res.s_fwd >= 0) and np.all(res.s_rev >= 0))
            self.assertTrue(np.array_equal(aln.scores.view(np.int32), res.s_fwd.view(np.int32)))
            self.assertTrue(np.array_equal(aln_score_only.scores.view(np.int32), res.s_fwd.view(np.int32)))
            self.assertTrue(np.array_equal(aln_score_only.ends, aln.ends))
            self.assertFalse(np.any(np.signbit(res.s_fwd)) or np.any(np.signbit(res.s_rev)), "-0 not canonicalised")
            for k in range(len(c["offsets"]) - 1):
                codes_k = c["codes"][c["offsets"][k]:c["offsets"][k + 1]]
                path = aln.path_of(k)
                _assert_path_invariants(self, path, aln.ends[k], float(aln.scores[k]), n, len(codes_k))
                _assert_cigar_round_trip(self, path, c["qseq"], c["tseqs"][k])
                o = ref.reference_pair64(c["q16"], codes_k, margins=True)
                s_f = float(aln.scores[k])
                self.assertLessEqual(abs(swkern.path_score(o["z"], path, D) - s_f), 5e-5 * max(abs(s_f), 1.0))
                margin = min(o["best_margin"], o["tb_margin"])
                if margin > max(1e-3, swkern.s16_path_margin(o["S_f"])):
                    compared += 1
                    self.assertTrue(np.array_equal(path, o["path"]), f"path differs at margin {margin}")
                else:
                    near_ties += 1
        self.assertGreater(compared, 400)

    def test_batch_order_and_block_size_invariance(self):
        rng = np.random.default_rng(7)
        for c in _property_cases()[::5]:
            qp = swkern.query_prep(c["q16"])
            base = swkern.score_fwd_rev(qp, c["codes"], c["offsets"], D)
            base_aln = swkern.align(qp, c["codes"], c["offsets"], D)
            for bf in (1 << 12, 1 << 17, 1 << 20):
                r = swkern.score_fwd_rev(qp, c["codes"], c["offsets"], D, block_floats=bf)
                a = swkern.align(qp, c["codes"], c["offsets"], D, block_floats=bf)
                self.assertTrue(np.array_equal(r.s_fwd.view(np.int32), base.s_fwd.view(np.int32)), bf)
                self.assertTrue(np.array_equal(r.s_rev.view(np.int32), base.s_rev.view(np.int32)), bf)
                self.assertTrue(np.array_equal(a.path, base_aln.path) and np.array_equal(a.path_off, base_aln.path_off))
            K = len(c["offsets"]) - 1
            perm = rng.permutation(K)
            starts, lens = c["offsets"][:-1][perm], np.diff(c["offsets"])[perm]
            r = swkern.score_fwd_rev(qp, c["codes"], starts, D, lengths=lens)       # non-contiguous candidate rows
            a = swkern.align(qp, c["codes"], starts, D, lengths=lens)
            self.assertTrue(np.array_equal(r.s_fwd.view(np.int32), base.s_fwd[perm].view(np.int32)))
            self.assertTrue(np.array_equal(r.s_rev.view(np.int32), base.s_rev[perm].view(np.int32)))
            for i, k in enumerate(perm):
                self.assertTrue(np.array_equal(a.path_of(i), base_aln.path_of(k)))

    def test_reversed_view_equals_reversed_query(self):
        for c in _property_cases()[::3]:
            qp = swkern.query_prep(c["q16"])
            qr = np.ascontiguousarray(qp[::-1])
            ref_view = swkern.score_fwd_rev(qp, c["codes"], c["offsets"], D, reference_sim=True)
            ref_qrev = swkern.score_fwd_rev(qr, c["codes"], c["offsets"], D, reference_sim=True)
            self.assertTrue(np.array_equal(ref_view.s_rev.view(np.int32), ref_qrev.s_fwd.view(np.int32)))
            blas_view = swkern.score_fwd_rev(qp, c["codes"], c["offsets"], D)
            blas_qrev = swkern.score_fwd_rev(qr, c["codes"], c["offsets"], D)
            _assert_s16_score(self, blas_view.s_rev, blas_qrev.s_fwd, "BLAS reversed view")
            _assert_s16_score(self, blas_view.s_rev, ref_view.s_rev, "BLAS vs reference S_r")


# =====================================================================================================================
# 3. DP golden hash
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class GoldenHashTest(unittest.TestCase):

    def test_golden_inputs_are_stable(self):
        h = hashlib.sha256()
        for kind, q, c in _golden_pairs():
            h.update(kind.encode() + b"\0")
            h.update(np.ascontiguousarray(q, "<f2").tobytes() + b"\0")
            h.update(c.tobytes() + b"\0")
        self.assertEqual(h.hexdigest(), GOLDEN_INPUTS_SHA256)

    def test_dp_golden_hash_on_every_build(self):
        for k in _kernels():
            runs, pairs = _golden_runs(k, reference_sim=True)
            for r in runs:
                self.assertTrue(np.isfinite(r.s_f) and np.isfinite(r.s_r))
                self.assertEqual(r.s_tb, r.s_f)
            self.assertEqual(_digests(runs, pairs), (GOLDEN_F1_SYNTH_SHA256, swkern.DP_GOLDEN_SYNTH_SHA256), k.info())

    def test_selftest(self):
        report = swkern.selftest()
        self.assertEqual(report["dp_golden_sha256"], swkern.DP_GOLDEN_SYNTH_SHA256)
        self.assertLessEqual(report["s16_max_score_ratio"], 1.0)
        self.assertIn("blas", report["blas"])
        self.assertEqual(report["kernel"]["isa"], swkern.load().isa)


# =====================================================================================================================
# 4. BLAS conformance (S16)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class BlasConformanceTest(unittest.TestCase):

    def test_production_vs_golden_on_the_golden_pairs(self):
        k = swkern.load()
        gold, pairs = _golden_runs(k, reference_sim=True)
        prod, _ = _golden_runs(k, reference_sim=False)
        _assert_s16_score(self, [r.s_f for r in prod], [r.s_f for r in gold], "S_f")
        _assert_s16_score(self, [r.s_r for r in prod], [r.s_r for r in gold], "S_r")
        for (q, c), g, b in zip(pairs, gold, prod):
            if not np.array_equal(g.path, b.path):
                z = swkern.z64_pair(q, c)
                gap = abs(swkern.path_score(z, b.path, D) - swkern.path_score(z, g.path, D))
                self.assertLess(gap, swkern.s16_path_margin(g.s_f), "path change outside an S16 near-tie")

    def test_q90_lne_bits_and_decisions_within_s16(self):
        f1 = _f1()
        centres, lam, lnK = _evalue_table()
        codes = np.concatenate([p["codes"] for p in f1])
        offsets = np.concatenate([[0], np.cumsum([len(p["codes"]) for p in f1])]).astype(np.int64)
        out = {}
        for mode in (True, False):
            q90, lne, bits = [], [], []
            for a, p in enumerate(f1):
                res = swkern.score_fwd_rev(swkern.query_prep(p["q16"]), codes, offsets, D, reference_sim=mode)
                S = res.s_fwd.astype(np.float64) - res.s_rev.astype(np.float64)
                q = float(np.quantile(S, 0.9))
                e, b = ref.reference_evalue(S[a] - q, len(p["q16"]), 1e9, centres, lam, lnK)
                q90.append(q)
                lne.append(float(e))
                bits.append(float(b))
            out[mode] = (np.array(q90), np.array(lne), np.array(bits))
        (gq, ge, gb), (bq, be, bb) = out[True], out[False]
        self.assertLessEqual(float(np.max(np.abs(bq - gq))), 1e-4)
        self.assertLessEqual(float(np.max(np.abs(be - ge))), 1e-4)
        for t in (1e-3, 1.0, 10.0):
            same = (ge <= np.log(t)) == (be <= np.log(t))
            boundary = np.abs(ge - np.log(t)) < 1e-4
            self.assertTrue(np.all(same | boundary), t)
        rb_same = np.floor(gb + 0.5) == np.floor(bb + 0.5)
        near = np.abs((gb + 0.5) - np.round(gb + 0.5)) < 2e-4
        self.assertTrue(np.all(rb_same | near))

    def test_gemm_operand_and_output_layout_is_enforced(self):
        p = _f1()[0]
        qp = swkern.query_prep(p["q16"])
        n, m = len(qp), len(p["codes"])
        out = swkern.aligned_empty(n * m).reshape(n, m)
        np.testing.assert_array_equal(swkern.similarity(qp, p["codes"], out), qp @ swkern.decode_codes(p["codes"]).T)
        self.assertEqual(out.ctypes.data % 64, 0)
        with self.assertRaises(TypeError):
            swkern.similarity(qp.astype(np.float64), p["codes"], out)
        with self.assertRaises(ValueError):
            swkern.similarity(np.asfortranarray(qp), p["codes"], out)
        raw = swkern.aligned_empty(n * m + 1)
        with self.assertRaises(ValueError):
            swkern.similarity(qp, p["codes"], raw[1:].reshape(n, m))                     # 4-byte misaligned
        with self.assertRaises(ValueError):
            swkern.similarity(qp, p["codes"], np.empty((m, n), np.float32).T)             # not C-contiguous
        with self.assertRaises(TypeError):
            swkern.similarity(qp, p["codes"].astype(np.int16), out)
        # the kernel itself refuses a pair matrix off the 64-byte grid
        k = swkern.load()
        sim = swkern.aligned_empty(n * m + 16)
        sim[:] = 0.1
        ws = np.empty(k.ws_floats(n, m), np.float32)
        sf, sr = np.empty(1, np.float32), np.empty(1, np.float32)
        mm = np.array([m], np.int32)
        for off, want in ((0, 0), (16, 0), (1, 2), (4, 2)):
            po = np.array([off], np.int64)
            st = k.lib.fmsw_score_fwd_rev(sim.ctypes.data, n, po.ctypes.data, mm.ctypes.data, 1, 1.0, 5.0, 0.5,
                                          sf.ctypes.data, sr.ctypes.data, ws.ctypes.data, ws.size)
            self.assertEqual(st, want, off)

    def test_block_plan_keeps_every_pair_on_the_64_byte_grid(self):
        n = 37
        lens = [0, 1, 5, 300, 7, 4000, 2, 0, 33]
        for bf in (1 << 12, 1 << 17):
            blocks = swkern.plan_blocks(n, lens, [False] * len(lens), bf)
            self.assertEqual([k for b in blocks for k in range(*b)], list(range(len(lens))))
            for k0, k1 in blocks:
                sizes = [-(-n * m // 16) * 16 for m in lens[k0:k1]]
                self.assertTrue(k1 - k0 == 1 or sum(sizes) <= bf)
        self.assertEqual(swkern.plan_blocks(n, [5, 5, 5], [False, True, False], 1 << 20), [(0, 1), (2, 3)])
        ws = swkern.Workspace()
        for size in (1, 17, 1000, 123457):
            self.assertEqual(ws.sim(size).ctypes.data % 64, 0)

    def test_workers_blas_threads_one(self):
        try:
            import threadpoolctl
        except ImportError:
            self.skipTest("threadpoolctl not installed")
        if not [d for d in threadpoolctl.threadpool_info() if d.get("user_api") == "blas"]:
            self.skipTest("threadpoolctl cannot introspect this BLAS (e.g. Accelerate)")
        env = _subprocess_env()
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[v] = "1"
        code = ("import json; from foldmatch.search import swkern; "
                "print(json.dumps(swkern.blas_info()['threadpools']))")
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        pools = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertTrue(pools)
        self.assertTrue(all(p["num_threads"] == 1 for p in pools), pools)

    def test_no_torch_or_biotite_in_the_kernel_path(self):
        p = _f1()[3]
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d) / "pair.npz", q16=p["q16"], codes=p["codes"])
            code = (
                "import sys; sys.modules['torch'] = None; sys.modules['biotite'] = None\n"
                "import numpy as np\n"
                "from foldmatch.search import swkern\n"
                f"z = np.load({str(Path(d) / 'pair.npz')!r})\n"
                "qp = swkern.query_prep(z['q16'])\n"
                "off = np.array([0, len(z['codes'])], np.int64)\n"
                "r = swkern.score_fwd_rev(qp, z['codes'], off, swkern.SETTING_D)\n"
                "a = swkern.align(qp, z['codes'], off, swkern.SETTING_D)\n"
                "m = swkern.path_metrics(a.path_of(0), 'A' * len(qp), 'A' * len(z['codes']), strings=True)\n"
                "print(float(r.s_fwd[0]), float(a.scores[0]), m.cigar[:3])\n")
            out = subprocess.run([sys.executable, "-c", code], env=_subprocess_env(), capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        sf, sa, _ = out.stdout.split()
        self.assertEqual(float(sf), float(sa))
        self.assertEqual(np.float32(float(sf)), np.float32(swkern.run_pair(p["q16"], p["codes"],
                                                                           reference_sim=False).s_f))


# =====================================================================================================================
# 5. degenerate inputs (F-deg)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class DegenerateTest(unittest.TestCase):

    def test_synthetic_cases_match_the_s7_reference(self):
        z = np.load(_RES / "degenerate_v1.npz")
        qo, to = z["syn_q_offsets"], z["syn_t_offsets"]
        for i, name in enumerate(z["syn_ids"].astype(str)):
            q16, codes = z["syn_q"][qo[i]:qo[i + 1]], z["syn_codes"][to[i]:to[i + 1]]
            if z["syn_expect_error"][i]:
                with self.assertRaises(ValueError, msg=name):
                    swkern.query_prep(q16)
                continue
            for mode in (False, True):
                r = swkern.run_pair(q16, codes, reference_sim=mode)
                self.assertTrue(np.isfinite(r.s_f) and np.isfinite(r.s_r), name)
                # abs <= 1e-5 on the degenerate scores (all <= 2.1); the full-size controls zero_row_q and
                # subnormal_row_q (S_f ~ 322, float32 ulp 3e-5) get the same bound relative: 1e-5 max(|ref|, 1)
                self.assertLessEqual(abs(r.s_f - z["syn_S_f"][i]), 1e-5 * max(abs(z["syn_S_f"][i]), 1.0), (name, mode))
                self.assertLessEqual(abs(r.s_r - z["syn_S_r"][i]), 1e-5 * max(abs(z["syn_S_r"][i]), 1.0), (name, mode))
                self.assertEqual(r.s_tb, r.s_f)
                if name in ("identical_codes", "identical_q_rows", "identical_both", "n1"):
                    self.assertLessEqual(abs(r.s_f - r.s_r), 1e-5, name)
                if name in ("identical_both", "all_zero_q"):
                    self.assertEqual((r.s_f, len(r.path)), (0.0, 0), name)

    def test_real_esm3_tiny_and_low_complexity_pairs(self):
        z = np.load(_RES / "degenerate_v1.npz")
        ro = z["real_offsets"]
        items = [(z["real_q"][ro[i]:ro[i + 1]], z["real_codes"][ro[i]:ro[i + 1]]) for i in range(len(ro) - 1)]
        items.append((z["base_q"], z["base_codes"]))
        got_f, got_r = [], []
        for a, b in zip(z["pair_q"], z["pair_t"]):
            r = swkern.run_pair(items[a][0], items[b][1], reference_sim=False)
            self.assertTrue(np.isfinite(r.s_f) and np.isfinite(r.s_r))
            got_f.append(r.s_f)
            got_r.append(r.s_r)
        _assert_s16_score(self, got_f, z["pair_S_f"], "F-deg real S_f")
        _assert_s16_score(self, got_r, z["pair_S_r"], "F-deg real S_r")

    def test_s7_threshold_on_hand_built_similarities(self):
        """S7 pins the cutoff at population std < 1e-6, on both sides: a row (column) of std 5e-7 contributes exactly
        0 and one of std 2e-6 does not. Neither the golden pairs nor F1 / F-deg have a row or column with std in
        [1e-7, 1e-5), so the kernel is fed hand-built similarities directly; the float64 oracle with the cutoff at
        1e-7 and at 1e-5 gives clearly different scores, so a moved cutoff cannot pass."""
        for kind in ("rows", "cols"):
            sim = _s7_similarity(kind)
            want = {eps: _s7_oracle(sim, eps) for eps in (1e-7, 1e-6, 1e-5)}
            n_deg = want[1e-6]["ndeg"]
            self.assertEqual(n_deg, (1, 0) if kind == "rows" else (0, 1), kind)
            self.assertEqual(want[1e-5]["ndeg"], (2, 0) if kind == "rows" else (0, 2), kind)
            for eps in (1e-7, 1e-5):
                self.assertGreater(abs(want[eps]["S_f"] - want[1e-6]["S_f"]), 0.5, (kind, eps))
            for k in _kernels():
                s_f, s_r = _score_sim(k, sim)
                _assert_s16_score(self, [s_f, s_r], [want[1e-6]["S_f"], want[1e-6]["S_r"]], f"S7 {kind} {k.isa}")

    def test_nan_and_inf_query_raise(self):
        q = _f1()[0]["q16"].copy()
        for bad in (np.nan, np.inf, -np.inf):
            q2 = q.copy()
            q2[3, 7] = bad
            with self.assertRaises(ValueError):
                swkern.query_prep(q2)
        with self.assertRaises(TypeError):
            swkern.query_prep(q.astype(np.float32))


def _s7_similarity(kind, n=20, seed=20261014):
    """[n, n] fp32 similarity with a strong diagonal (the forward alignment runs through every row and column) and
    random values elsewhere, except two noise-only rows (``kind='rows'``) or columns (``'cols'``) of mean 0 and
    population std 5e-7 (degenerate under S7) and 2e-6 (not degenerate), largest on their diagonal cell."""
    if kind == "cols":
        return np.ascontiguousarray(_s7_similarity("rows", n, seed + 1).T)
    rng = np.random.default_rng(seed)
    s = 0.1 * rng.standard_normal((n, n))
    s[np.arange(n), np.arange(n)] += 1.0
    for r, sd in ((n // 3, 5e-7), (2 * n // 3, 2e-6)):
        v = rng.standard_normal(n)
        v[r] = np.abs(v).max() + 1.0
        s[r] = sd * (v - v.mean()) / v.std()
    return s.astype(np.float32)


def _s7_oracle(sim, eps):
    """float64 oracle S_f / S_r (setting D) of a given similarity, with the S7 cutoff at ``eps``."""
    out = {}
    with mock.patch.object(ref, "S7_EPS", eps):
        for d, x in (("f", sim), ("r", sim[::-1])):
            z, nr, nc = ref.reference_rowcol_s7(np.asarray(x, np.float64))
            out[f"S_{d}"] = float(ref.smith_waterman(z - D.offset, D.gap_open, D.gap_extend).score)
            out["ndeg"] = (nr, nc)
    return out


def _score_sim(kernel, sim):
    """fmsw_score_fwd_rev on one given [n, m] similarity (64-byte aligned copy): (S_f, S_r)."""
    n, m = sim.shape
    buf = swkern.aligned_empty(n * m)
    buf[:] = np.asarray(sim, np.float32).ravel()
    ws = np.empty(kernel.ws_floats(n, m), np.float32)
    sf, sr = np.empty(1, np.float32), np.empty(1, np.float32)
    po, mm = np.zeros(1, np.int64), np.array([m], np.int32)
    st = kernel.lib.fmsw_score_fwd_rev(buf.ctypes.data, n, po.ctypes.data, mm.ctypes.data, 1, D.offset, D.gap_open,
                                       D.gap_extend, sf.ctypes.data, sr.ctypes.data, ws.ctypes.data, ws.size)
    assert st == 0, st
    return float(sf[0]), float(sr[0])


# =====================================================================================================================
# 6. long pair and max_pair_cells
# =====================================================================================================================
def _rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


@pytest.mark.usefixtures("fmsw_libs")
class LongPairTest(unittest.TestCase):

    def test_oversize_pair_is_skipped_with_one_warning(self):
        p = _f1()[0]
        qp = swkern.query_prep(p["q16"])
        codes = np.concatenate([p["codes"], _f1()[1]["codes"], p["codes"]])
        m0, m1 = len(p["codes"]), len(_f1()[1]["codes"])
        offsets = np.array([0, m0, m0 + m1, 2 * m0 + m1], np.int64)
        limit = len(qp) * m0 - 1                         # pairs 0 and 2 are too large, pair 1 fits
        self.assertLess(len(qp) * m1, limit)
        swkern.reset_skipped_pairs()
        with self.assertLogs("foldmatch.search.swkern", level="WARNING") as logs:
            r = swkern.score_fwd_rev(qp, codes, offsets, D, max_pair_cells=limit)
            a = swkern.align(qp, codes, offsets, D, max_pair_cells=limit)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(swkern.skipped_pair_count(), 4)
        self.assertEqual(r.skipped.tolist(), [True, False, True])
        self.assertTrue(np.isnan(r.s_fwd[0]) and np.isnan(r.s_rev[2]) and np.isnan(a.scores[0]))
        self.assertEqual(len(a.path_of(0)), 0)
        single = swkern.run_pair(p["q16"], _f1()[1]["codes"], reference_sim=False)
        self.assertEqual(r.s_fwd[1], np.float32(single.s_f))
        self.assertTrue(np.array_equal(a.path_of(1), single.path))
        swkern.reset_skipped_pairs()

    @unittest.skipUnless(os.environ.get("FOLDMATCH_SLOW_TESTS") == "1",
                         "slow: 5,000 x 5,000 pair; set FOLDMATCH_SLOW_TESTS=1 to run")
    def test_5000_by_5000(self):
        rng = np.random.default_rng(5000)
        q16 = rng.standard_normal((5000, 64)).astype(np.float16)
        t = q16.astype(np.float32)[200:4900] + 0.8 * rng.standard_normal((4700, 64)).astype(np.float32)
        t = np.concatenate([rng.standard_normal((150, 64)).astype(np.float32), t,
                            rng.standard_normal((150, 64)).astype(np.float32)])
        codes = np.packbits(t >= 0, axis=1)
        self.assertEqual(codes.shape, (5000, 8))
        off = np.array([0, 5000], np.int64)
        ws = swkern.Workspace()
        rss0 = _rss_bytes()
        qp = swkern.query_prep(q16)
        r = swkern.score_fwd_rev(qp, codes, off, D, workspace=ws)
        a = swkern.align(qp, codes, off, D, workspace=ws)
        grown = _rss_bytes() - rss0
        self.assertEqual(a.scores[0], r.s_fwd[0])
        self.assertGreater(float(r.s_fwd[0]), 1000.0)
        path = a.path_of(0)
        _assert_path_invariants(self, path, a.ends[0], float(a.scores[0]), 5000, 5000)
        # 25 M cells: similarity 100 MB + flags 25 MB + workspace and path; allow 2x headroom
        self.assertLess(ws.nbytes(), 160 * 2 ** 20)
        self.assertLess(grown, 400 * 2 ** 20)
        z = swkern.z64_pair(q16, codes)
        self.assertLessEqual(abs(swkern.path_score(z, path, D) - float(a.scores[0])),
                             5e-5 * float(a.scores[0]))
        with self.assertLogs("foldmatch.search.swkern", level="WARNING"):
            swkern.reset_skipped_pairs()
            s = swkern.score_fwd_rev(qp, codes, off, D, max_pair_cells=5000 * 5000 - 1, workspace=ws)
        self.assertTrue(s.skipped[0] and np.isnan(s.s_fwd[0]))
        swkern.reset_skipped_pairs()


# =====================================================================================================================
# 7. ABI, dispatch, missing and stale libraries
# =====================================================================================================================
class _FakeKernel:
    def __init__(self, isa, avx2):
        self.isa, self._avx2 = isa, avx2

    def cpu_has_avx2(self):
        return self._avx2


@pytest.mark.usefixtures("fmsw_libs")
class AbiDispatchTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fmsw_test_"))
        self.isa = _native.machine_isas()[0]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build_variant(self, edit):
        src = self.tmp / "src"
        src.mkdir()
        for name in _native.SOURCE_FILES:
            shutil.copy(_native.SOURCE_DIR / name, src / name)
        edit(src)
        out = self.tmp / "lib"
        native_build.compile_library(self.isa, out, source_dir=src, quiet=True)
        return out

    def test_default_dispatch_and_info(self):
        k = swkern.load()
        info = k.info()
        self.assertEqual(set(info) >= {"isa", "abi", "source_sha256", "path"}, True)
        self.assertEqual(info["abi"], swkern.ABI_VERSION)
        self.assertEqual(info["source_sha256"], _native.source_sha256())
        self.assertEqual(info["dp_id"], "fmsw/1")
        machine = _native.normalized_machine()
        if machine == "arm64":
            self.assertEqual(k.isa, "neon")
        elif machine == "x86_64":
            self.assertEqual(k.isa, "avx2" if swkern.load("sse2").cpu_has_avx2() else "sse2")
        else:
            self.assertEqual(k.isa, "generic")

    def test_isa_override(self):
        with mock.patch.dict(os.environ, {swkern.ISA_ENV: "generic"}):
            self.assertEqual(swkern.load().isa, "generic")
        with mock.patch.dict(os.environ, {swkern.ISA_ENV: "GENERIC "}):
            self.assertEqual(swkern.load().isa, "generic")
        with mock.patch.dict(os.environ, {swkern.ISA_ENV: "fast"}):
            with self.assertRaisesRegex(swkern.KernelError, "not a kernel ISA"):
                swkern.load()
        foreign = "sse2" if _native.normalized_machine() != "x86_64" else "neon"
        with self.assertRaisesRegex(swkern.KernelError, "cannot run on this"):
            swkern.load(foreign)

    def test_avx2_override_refused_without_avx2(self):
        def fake_open(isa, dirs, check):
            return _FakeKernel(isa, avx2=False)
        with mock.patch.object(swkern, "_open", side_effect=fake_open):
            with self.assertRaisesRegex(swkern.KernelError, "avx2 refused"):
                swkern._select("avx2", "x86_64", None, True)
            self.assertEqual(swkern._select(None, "x86_64", None, True).isa, "sse2")
        with mock.patch.object(swkern, "_open", side_effect=lambda isa, d, c: _FakeKernel(isa, avx2=True)):
            self.assertEqual(swkern._select("avx2", "x86_64", None, True).isa, "avx2")
            self.assertEqual(swkern._select(None, "x86_64", None, True).isa, "avx2")
            self.assertEqual(swkern._select(None, "aarch64", None, True).isa, "neon")
            self.assertEqual(swkern._select(None, "riscv64", None, True).isa, "generic")
            with self.assertRaisesRegex(swkern.KernelError, "cannot run on this"):
                swkern._select("neon", "x86_64", None, True)

    def test_missing_library_names_the_rebuild_command(self):
        with self.assertRaises(swkern.KernelError) as cm:
            swkern.load(self.isa, search_dirs=[self.tmp])
        self.assertIn(_native.REBUILD_COMMAND, str(cm.exception))
        self.assertIn(_native.REINSTALL_COMMAND, str(cm.exception))

    def test_abi_mismatch_refused(self):
        def edit(src):
            h = src / "fmsw.h"
            h.write_text(h.read_text().replace("#define FMSW_ABI_VERSION 1", "#define FMSW_ABI_VERSION 2"))
        lib = self._build_variant(edit)
        with self.assertRaisesRegex(swkern.KernelError, "ABI 2"):
            swkern.load(self.isa, search_dirs=[lib], check_source=False)

    def test_stale_library_refused(self):
        lib = self._build_variant(lambda src: (src / "fmsw.c").write_text((src / "fmsw.c").read_text() + "\n/**/\n"))
        self.assertFalse(native_build.library_is_current(lib / _native.library_filename(self.isa)))
        self.assertTrue(native_build.library_is_current(_native.find_library(self.isa)))
        with self.assertRaisesRegex(swkern.KernelError, "stale native kernel"):
            swkern.load(self.isa, search_dirs=[lib])
        k = swkern.load(self.isa, search_dirs=[lib], check_source=False)
        self.assertNotEqual(k.source_sha256, _native.source_sha256())

    def test_not_a_kernel_library(self):
        (self.tmp / _native.library_filename(self.isa)).write_bytes(b"not a library")
        with self.assertRaises(swkern.KernelError):
            swkern.load(self.isa, search_dirs=[self.tmp])

    def test_x86_libraries_are_built_for_the_x86_64_baseline(self):
        """Every x86_64 library gets -march=x86-64 -mtune=generic before its ISA flag, whatever the compiler's default
        -march (a Cray wrapper's -march=znver3 made the sse2 library an AVX2 build); arm64 libraries get none."""
        def cmd(isa, machine=None):
            return native_build.compile_command(isa, self.tmp / "lib.so", cc=sys.executable, machine=machine)
        base = ["-march=x86-64", "-mtune=generic"]
        for isa, machine in (("sse2", None), ("avx2", None), ("generic", "x86_64"), ("generic", "AMD64")):
            c = cmd(isa, machine)
            i = c.index("-march=x86-64")
            self.assertEqual(c[i:i + 2], base, (isa, machine))
            if isa == "avx2":
                self.assertGreater(c.index("-mavx2"), i + 1)
        for isa, machine in (("neon", None), ("generic", "arm64"), ("generic", "aarch64"), ("generic", "riscv64")):
            self.assertFalse([f for f in cmd(isa, machine) if f.startswith(("-march", "-mtune"))], (isa, machine))
        self.assertNotIn("-mfma", cmd("avx2"))
        self.assertFalse([f for f in cmd("avx2") if f == "-march=native"])

    def test_dp_params_validation(self):
        with self.assertRaises(ValueError):
            swkern.DPParams(0.1, 5.0, 0.5)                     # not exact in float32
        with self.assertRaises(ValueError):
            swkern.DPParams(1.0, 0.5, 5.0)                     # gap_open < gap_extend
        with self.assertRaises(ValueError):
            swkern.DPParams(float("nan"), 5.0, 0.5)
        self.assertEqual(swkern.SETTING_D, swkern.DPParams(1.0, 5.0, 0.5))


# =====================================================================================================================
# 8. workspace and status codes
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class WorkspaceStatusTest(unittest.TestCase):

    def setUp(self):
        self.k = swkern.load()
        rng = np.random.default_rng(3)
        self.n, self.m = 37, 53
        self.sim = swkern.aligned_empty(self.n * self.m)
        self.sim[:] = rng.standard_normal(self.n * self.m).astype(np.float32) * 0.1
        self.po = np.zeros(1, np.int64)
        self.mm = np.array([self.m], np.int32)

    def _score(self, ws_len, n=None, po=None, mm=None, go=5.0, ge=0.5, offset=1.0, K=1):
        n = self.n if n is None else n
        ws = np.empty(max(ws_len, 1), np.float32)
        sf, sr = np.empty(1, np.float32), np.empty(1, np.float32)
        po = self.po if po is None else po
        mm = self.mm if mm is None else mm
        return self.k.lib.fmsw_score_fwd_rev(self.sim.ctypes.data, n, po.ctypes.data, mm.ctypes.data, K, offset, go,
                                             ge, sf.ctypes.data, sr.ctypes.data, ws.ctypes.data, ws_len)

    def _align(self, ws_len, fl_len, cap, tb=1):
        ws = np.empty(max(ws_len, 1), np.float32)
        fl = np.empty(max(fl_len, 1), np.uint8)
        pb = np.empty(2 * max(cap, 1), np.int32)
        poff = np.empty(2, np.int64)
        sc, ends = np.empty(1, np.float32), np.empty(2, np.int32)
        return self.k.lib.fmsw_align(self.sim.ctypes.data, self.n, self.po.ctypes.data, self.mm.ctypes.data, 1, 1.0,
                                     5.0, 0.5, tb, sc.ctypes.data, ends.ctypes.data, pb.ctypes.data, cap,
                                     poff.ctypes.data, ws.ctypes.data, ws_len, fl.ctypes.data, fl_len)

    def test_undersized_buffers_return_status_1(self):
        need_ws, need_fl = self.k.ws_floats(self.n, self.m), self.k.flag_bytes(self.n, self.m)
        cap = self.n + self.m
        self.assertEqual(self._score(need_ws), 0)
        self.assertEqual(self._score(need_ws - 1), 1)
        self.assertEqual(self._align(need_ws, need_fl, cap), 0)
        self.assertEqual(self._align(need_ws - 1, need_fl, cap), 1)
        self.assertEqual(self._align(need_ws, need_fl - 1, cap), 1)
        self.assertEqual(self._align(need_ws, need_fl, cap - 1), 1)
        self.assertEqual(self._align(need_ws - 1, 0, 0, tb=0), 1)
        self.assertEqual(self._align(need_ws, 0, 0, tb=0), 0)

    def test_bad_arguments_return_status_2(self):
        need = self.k.ws_floats(self.n, self.m)
        self.assertEqual(self._score(need, n=0), 2)
        self.assertEqual(self._score(need, mm=np.array([-1], np.int32)), 2)
        self.assertEqual(self._score(need, po=np.array([-16], np.int64)), 2)
        self.assertEqual(self._score(need, go=0.4, ge=0.5), 2)
        self.assertEqual(self._score(need, ge=-0.5, go=-0.5), 2)
        self.assertEqual(self._score(need, offset=float("nan")), 2)
        self.assertEqual(self._score(need, K=-1), 2)
        self.assertEqual(self._score(need, K=0), 0)
        q = np.ones((2, 64), np.float32)
        self.assertEqual(self.k.lib.fmsw_query_prep(q.ctypes.data, -1, q.ctypes.data), 2)

    def test_empty_target_scores_zero(self):
        self.assertEqual(self._score(self.k.ws_floats(self.n, 0), mm=np.array([0], np.int32)), 0)
        qp = swkern.query_prep(_f1()[0]["q16"])
        codes = _f1()[0]["codes"]
        res = swkern.align(qp, codes, np.array([0, 0, len(codes), len(codes)], np.int64), D)
        self.assertEqual(res.scores[0], 0.0)
        self.assertEqual(res.scores[2], 0.0)
        self.assertEqual(res.path_off.tolist()[:2], [0, 0])
        self.assertGreater(res.scores[1], 0.0)

    def test_only_empty_targets_in_a_fresh_workspace(self):
        """A call (or a first block) whose targets are all empty, in fresh workspaces: scores 0, no paths, never the
        64-byte guard. An empty numpy slice keeps its base's address, so aligned_empty(0) took the alignment of a
        fresh allocation (off in about 3 of 4); the allocations here are made to start 4 bytes past a 64-byte
        boundary, so the check does not depend on the allocator."""
        real_empty = np.empty

        def shifted(shape, dtype=float, *args, **kwargs):
            """np.empty whose data starts 4 bytes past a 64-byte boundary."""
            dtype = np.dtype(dtype)
            count = int(np.prod(shape))
            raw = real_empty(count * dtype.itemsize + 2 * swkern.ALIGN_BYTES, np.uint8)
            start = -raw.ctypes.data % swkern.ALIGN_BYTES + 4
            return raw[start:start + count * dtype.itemsize].view(dtype).reshape(shape)

        with mock.patch.object(np, "empty", shifted):
            self.assertEqual(np.empty(3, np.float32).ctypes.data % swkern.ALIGN_BYTES, 4)
            bufs = {n: swkern.aligned_empty(n) for n in (0, 1, 15, 16, 17)}
            fresh = [swkern.Workspace() for _ in range(8)]
        for n, buf in bufs.items():
            self.assertEqual((buf.size, buf.ctypes.data % swkern.ALIGN_BYTES), (n, 0))
        qp = swkern.query_prep(_f1()[0]["q16"])
        codes = _f1()[0]["codes"]
        m = len(codes)
        for lens in ([0], [0, 0, 0], [0, 0, m], [0, m, 0]):
            offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            with self.subTest(lens=lens):
                with mock.patch.object(np, "empty", shifted):
                    ws, ws2 = swkern.Workspace(), swkern.Workspace()     # their first buffers: shifted allocations
                r = swkern.score_fwd_rev(qp, codes, offsets, D, workspace=ws, block_floats=1)
                a = swkern.align(qp, codes, offsets, D, workspace=ws2, block_floats=1)
                empty = np.asarray(lens) == 0
                self.assertTrue(np.all(r.s_fwd[empty] == 0) and np.all(r.s_rev[empty] == 0))
                self.assertTrue(np.all(a.scores[empty] == 0))
                self.assertTrue(np.all(np.diff(a.path_off)[empty] == 0))
                self.assertTrue(np.array_equal(a.scores.view(np.uint32), r.s_fwd.view(np.uint32)))
        for ws in fresh:
            self.assertEqual(ws.sim(0).ctypes.data % swkern.ALIGN_BYTES, 0)

    def test_workspace_grows_and_is_per_thread(self):
        ws = swkern.Workspace()
        a = ws.kernel_floats(10)
        self.assertGreaterEqual(a.size, 10)
        self.assertIs(ws.kernel_floats(5), a)
        self.assertGreaterEqual(ws.kernel_floats(100).size, 100)
        seen = []
        t = threading.Thread(target=lambda: seen.append(swkern.default_workspace()))
        t.start()
        t.join()
        self.assertIsNot(seen[0], swkern.default_workspace())
        self.assertIs(swkern.default_workspace(), swkern.default_workspace())


# =====================================================================================================================
# 9. regression: align() output is in forward order (the research codepath.py:261 double reversal is not ported)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class ForwardPathRegressionTest(unittest.TestCase):

    def test_align_path_is_forward_and_not_reversed(self):
        p = max(_f1(), key=lambda x: len(x["path"]))
        r = swkern.run_pair(p["q16"], p["codes"], reference_sim=False)
        path = r.path
        self.assertTrue(np.array_equal(path, p["path"]))
        self.assertLess(path[0, 0], path[-1, 0])
        self.assertLess(path[0, 1], path[-1, 1])
        self.assertEqual((path[-1, 0] + 1, path[-1, 1] + 1), r.end)
        self.assertFalse(np.array_equal(path, path[::-1]))
        qp = swkern.query_prep(p["q16"])
        a = swkern.align(qp, p["codes"], np.array([0, len(p["codes"])], np.int64), D)
        self.assertTrue(np.array_equal(a.path, path))


# =====================================================================================================================
# CIGAR round trip on every F1 alignment plus the 1,000 golden alignments (P3.5 acceptance d)
# =====================================================================================================================
@pytest.mark.usefixtures("fmsw_libs")
class CigarRoundTripTest(unittest.TestCase):

    def test_round_trip_on_1050_alignments(self):
        rng = np.random.default_rng(1050)
        done = 0
        for p in _f1():
            r = swkern.run_pair(p["q16"], p["codes"], reference_sim=False)
            _assert_cigar_round_trip(self, r.path, p["qseq"], p["tseq"])
            done += 1
        for _kind, q16, codes in _golden_pairs():
            r = swkern.run_pair(q16, codes, reference_sim=False)
            qs, ts = _random_seq(rng, len(q16)), _random_seq(rng, len(codes))
            if len(r.path) and rng.random() < 0.5:                 # force identities along half of the paths
                ts = list(ts)
                for i, j in r.path.tolist():
                    if i >= 0 and j >= 0 and rng.random() < 0.5:
                        ts[j] = qs[i]
                ts = "".join(ts)
            _assert_cigar_round_trip(self, r.path, qs, ts)
            done += 1
        self.assertEqual(done, 1050)

    def test_cigar_helpers(self):
        path = np.array([[0, 0], [1, -1], [2, -1], [-1, 1], [3, 2], [4, 3]], np.int32)
        self.assertEqual(swkern.cigar_from_path(path), "1M2I1D2M")
        self.assertEqual(swkern.expand_cigar("1M2I1D2M"), "MIIDMM")
        self.assertEqual(swkern.cigar_from_path(np.zeros((0, 2), np.int32)), "")
        with self.assertRaises(ValueError):
            swkern.expand_cigar("3M2X")
        pm = swkern.path_metrics(path, "ACDEF", "AQEF", strings=True)
        self.assertEqual((pm.q_aln, pm.t_aln), ("ACD-EF", "A--QEF"))
        self.assertEqual((pm.n_ident, pm.mismatch, pm.gap_open, pm.q_start, pm.q_end, pm.t_start, pm.t_end),
                         (3, 0, 2, 1, 5, 1, 4))


if __name__ == "__main__":
    unittest.main()
