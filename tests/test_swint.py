"""The BLOSUM62 Stage 2's native integer kernel (fmswi_align, :mod:`foldmatch.search.swint`) against biotite.

The kernel must be output-equivalent to biotite 1.7.1 ``align_optimal(..., local=True, max_number=1)``: the same
score, end cell and path, so every output column of the Stage 2 is the one the biotite path (tests/blosum62_reference.py)
gave. biotite is the oracle here; the biotite-free golden digest of the kernel is in test_swkern.py. The libraries come
from the session fixture ``fmsw_libs``.
"""
import dataclasses
import random
import unittest

import numpy as np
import pytest

import blosum62_reference as ref
from foldmatch import _native
from foldmatch.search import alignment, karlin_altschul, swint, swkern

AA = "ACDEFGHIKLMNPQRSTVWY"
ALL_NEEDS = alignment._FieldNeeds(cigar=True, qaln=True, taln=True, qseq=True, tseq=True)


def _trace_from_ops(ops: str, q_start: int, t_start: int) -> np.ndarray:
    i, j, out = q_start - 1, t_start - 1, []
    for c in ops:
        if c == "M":
            out.append((i, j)); i += 1; j += 1
        elif c == "D":
            out.append((-1, j)); j += 1
        else:
            out.append((i, -1)); i += 1
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _kernels():
    """The integer kernel of every ISA library of this machine that can run here (avx2 only on an AVX2 CPU)."""
    out = []
    for isa in _native.machine_isas():
        if _native.find_library(isa) is None:
            continue
        try:
            out.append(swint.load(isa))
        except swkern.KernelError:
            if isa != "avx2":
                raise
    return out


def _long_gap_pairs(rng):
    """Pairs whose best alignment has one long gap (60 residues, across many stripe segments), in each direction."""
    core = "".join(rng.choice(AA) for _ in range(150))
    insert = "".join(rng.choice(AA) for _ in range(60))
    gapped = core[:75] + insert + core[75:]
    return [(core, [gapped, gapped[:120] + gapped[150:]]), (gapped, [core])]


@pytest.mark.usefixtures("fmsw_libs")
class TestKernelAgainstBiotite(unittest.TestCase):
    def setUp(self):
        matrix = ref.blosum62()
        self.assertEqual("".join(matrix.get_alphabet1().get_symbols()), swint.ALPHABET)
        self.mat = matrix.score_matrix()

    def check(self, q, targets, gap=(11, 1), bias=None, kernel=None):
        res = swint.align_strings(q, targets, self.mat, gap[0], gap[1], bias=bias, ops=True, kernel=kernel,
                                  trace="full")
        two = swint.align_strings(q, targets, self.mat, gap[0], gap[1], bias=bias, ops=True, kernel=kernel,
                                  trace="prefix")
        np.testing.assert_array_equal(two.out, res.out)             # the traceback strategy changes nothing
        self.assertEqual(two.ops, res.ops)
        for idx, t in enumerate(targets):
            want = ref.biotite_alignment(q, t, gap, bias)
            o = res.out[idx]
            msg = f"{q!r} vs {t!r} gap {gap}"
            self.assertEqual(int(o[swint.SCORE]), int(want.score), msg)
            self.assertEqual(int(o[swint.ALNLEN]), len(want.trace), msg)
            if len(want.trace):
                got = _trace_from_ops(res.path(idx), int(o[swint.QSTART]), int(o[swint.TSTART]))
                np.testing.assert_array_equal(got, want.trace, err_msg=msg)
                self.assertEqual(int(o[swint.QEND]), int(want.trace[:, 0].max()) + 1, msg)
                self.assertEqual(int(o[swint.TEND]), int(want.trace[:, 1].max()) + 1, msg)
        return res

    def test_constructed_tie_cases(self):
        # gaps go as early as possible, prefixes summing to zero are cut, first maximal cell in row-major order
        r = self.check("MWKAAAAKWM", ["MWKAAAKWM"])
        self.assertEqual(r.path(0), "MMMIMMMMMM")                   # MWKAAAAKWM / MWK-AAAKWM
        r = self.check("APWW", ["AWWW"])
        self.assertEqual((int(r.out[0, swint.QSTART]), int(r.out[0, swint.TSTART])), (3, 2))
        self.check("AWA", ["AWAAWA", "W", "WAW"])
        core = "MKTAYIAKQRQISFVKSHFSRQ"
        self.check(core + "W" * 12, [core + "D" * 12])             # no insertion next to a deletion (biotite)
        self.check("WW", ["PP"])                                    # score 0: empty alignment
        self.check("BZX*", ["BZX*", "XXXX", "**", "B", "Z"])

    def test_random_pairs_every_gap_setting(self):
        rng = random.Random(3)
        alphabets = ["AG", "AGW", "ACDW", "WD", "AAAAGP", AA, AA + "BZX*"]
        for gap in [(11, 1), (10, 1), (5, 2), (3, 3), (1, 1), (20, 5), (2, 0), (1, 4)]:
            for _ in range(25):
                q = "".join(rng.choice(rng.choice(alphabets)) for _ in range(rng.randint(1, 90)))
                ts = ["".join(rng.choice(rng.choice(alphabets)) for _ in range(rng.randint(1, 90))) for _ in range(4)]
                self.check(q, ts, gap)

    def test_long_gaps(self):
        rng = random.Random(4)
        for gap in [(11, 1), (20, 5), (3, 3)]:
            for q, ts in _long_gap_pairs(rng):
                r = self.check(q, ts, gap)
                if gap == (11, 1):
                    self.assertIn("D" * 60 if len(q) == 150 else "I" * 60, r.path(0), gap)

    def test_position_specific_bias(self):
        rng = random.Random(5)
        for _ in range(60):
            q = "".join(rng.choice(AA) for _ in range(rng.randint(1, 120)))
            bias = np.array([rng.randint(-3, 3) for _ in q], dtype=np.int64)
            ts = ["".join(rng.choice(AA) for _ in range(rng.randint(1, 120))) for _ in range(3)]
            self.check(q, ts, (11, 1), bias=bias)
        q = "Q" * 30 + "MKTAYIAKQRQISFVKSHFSRQ" + "GGGGS" * 8          # the MMseqs2 correction itself
        self.check(q, [q, q[10:70], "MKTAYIAKQRQISFVKSHFSRQ"], bias=alignment.local_composition_bias(q))

    def test_int32_redo(self):
        q = "W" * 3100                                              # 34,100 > 32767 - 11
        r = self.check(q, [q, q[:3000] + "P" + q[3000:]])
        self.assertEqual(int(r.out[0, swint.SCORE]), 34100)
        self.assertEqual(int(r.out[0, swint.WIDE]), 1)
        small = swint.align_strings("WWWW", ["WWWW"], self.mat, 11, 1)
        self.assertEqual(int(small.out[0, swint.WIDE]), 0)

    def test_every_isa_library_agrees(self):
        """Every ISA library of the machine (not only the dispatched one) gives the same outputs and paths, in both
        traceback strategies: random pairs, a 32-bit pair, a biased pair and long gaps, at two gap settings (the
        dispatched library is checked against biotite above)."""
        rng = random.Random(9)
        q = "".join(rng.choice(AA) for _ in range(300))
        cases = [(q, ["".join(rng.choice(AA) for _ in range(rng.randint(1, 400))) for _ in range(20)] + [q], None)]
        wide = "W" * 3100
        cases.append((wide, [wide, wide[:3000] + "P" + wide[3000:], "W" * 40], None))
        biased = "Q" * 30 + q[:120] + "GGGGS" * 8
        cases.append((biased, [biased, q, biased[25:140]], alignment.local_composition_bias(biased)))
        cases += [(lq, lts, None) for lq, lts in _long_gap_pairs(rng)]
        kernels = _kernels()
        self.assertIn(swint.load().isa, [k.isa for k in kernels])
        for gap in [(11, 1), (20, 5)]:
            for query, targets, bias in cases:
                base = swint.align_strings(query, targets, self.mat, *gap, bias=bias, ops=True, trace="full")
                if query is wide and gap == (11, 1):
                    self.assertEqual(base.out[:, swint.WIDE].tolist(), [1, 1, 0])
                for k in kernels:
                    for trace in swint.TRACE_MODES:
                        msg = f"{k.isa} {trace} gap {gap} query of {len(query)}"
                        other = swint.align_strings(query, targets, self.mat, *gap, bias=bias, ops=True, kernel=k,
                                                    trace=trace)
                        np.testing.assert_array_equal(other.out, base.out, err_msg=msg)
                        self.assertEqual(other.ops, base.ops, msg)

    def test_the_golden_set_against_biotite(self):
        """The biotite-free golden digest (swint.GOLDEN_SHA256, checked on every build by test_swkern.py) pins
        biotite's alignments: every pair of swint.golden_cases (synthetic profiles, offsets, long gaps, gap_extend >
        gap_open, the 32-bit path, the 16-bit limit to the unit, the long lazy-F loop, empty alignments and targets)
        aligns as biotite aligns it with the same position-specific scores, every output field and the path."""
        import biotite.sequence as seq
        import biotite.sequence.align as align
        wide = 0
        for case in swint.golden_cases():
            res = swint.run_golden(case, 1)
            n = len(case.qcodes)
            alphabet = seq.Alphabet(list(range(n)))
            positional = seq.GeneralSequence(alphabet, list(range(n)))
            matrix = align.SubstitutionMatrix(alphabet, seq.ProteinSequence.alphabet, case.qprof)
            for k, t in enumerate(case.targets):
                o = res.out[k]
                msg = f"{case.kind} n={n} m={len(t)} gaps {case.gap_open}/{case.gap_extend}"
                wide += int(o[swint.WIDE])
                if len(t) == 0:
                    self.assertFalse(o.any(), msg)
                    continue
                target = seq.ProteinSequence()
                target.code = t
                want = align.align_optimal(positional, target, matrix, local=True, max_number=1,
                                           gap_penalty=(-case.gap_open, -case.gap_extend))[0]
                self.assertEqual(int(o[swint.SCORE]), int(want.score), msg)
                self.assertEqual(int(o[swint.ALNLEN]), len(want.trace), msg)
                if not len(want.trace):
                    continue
                trace = want.trace
                np.testing.assert_array_equal(_trace_from_ops(res.path(k), int(o[swint.QSTART]),
                                                              int(o[swint.TSTART])), trace, err_msg=msg)
                gaps = trace < 0                                 # [L, 2]: 'D' (query side), 'I' (target side)
                both = ~gaps.any(axis=1)
                self.assertEqual((int(o[swint.QEND]), int(o[swint.TEND])), tuple(trace.max(axis=0) + 1), msg)
                self.assertEqual(int(o[swint.NIDENT]), int((case.qcodes[trace[both, 0]] == t[trace[both, 1]]).sum()),
                                 msg)
                self.assertEqual(int(o[swint.GAPOPEN]), int(gaps[0].sum() + (gaps[1:] & ~gaps[:-1]).sum()), msg)
                self.assertEqual((int(o[swint.NQGAP]), int(o[swint.NTGAP])), tuple(gaps.sum(axis=0)), msg)
        self.assertEqual(wide, 10)

    def test_score_only_and_arguments(self):
        res = swint.align_batch(swint.codes_of(b"MKV"), swint.query_profile(swint.codes_of(b"MKV"), self.mat),
                                swint.codes_of(b"MKVMKV"), [0, 3], [3, 3], 11, 1, traceback=False)
        self.assertEqual(res.out[:, swint.SCORE].tolist(), [14, 14])       # M 5 + K 5 + V 4
        self.assertEqual(res.out[:, swint.ALNLEN].tolist(), [0, 0])
        empty = swint.align_strings("MKV", ["", "MKV"], self.mat, 11, 1)
        self.assertEqual(empty.out[0].tolist(), [0] * swint.NOUT)
        for go, ge in ((0, 1), (11, -1), (swint.MAX_GAP + 1, 1), (11, swint.MAX_GAP + 1)):
            with self.subTest(gap=(go, ge)), self.assertRaises(ValueError):
                swint.align_strings("MKV", ["MKV"], self.mat, go, ge)
        swint.align_strings("MKV", ["MKV"], self.mat, swint.MAX_GAP, swint.MAX_GAP)
        with self.assertRaises(ValueError):
            swint.align_strings("MKV", ["MKV"], self.mat, 11, 1, trace="banded")
        with self.assertRaises(ValueError):
            swint.align_batch(np.array([30], np.uint8), np.zeros((1, 24), np.int32), np.zeros(1, np.uint8), [0],
                              [1], 11, 1)
        self.assertEqual(swint.DEFAULT_TRACE, "prefix")

    def test_sanitize_fast_path(self):
        chars = [chr(c) for c in range(128)] + ["ß", "é", "Ω", "ﬁ", "µ"]
        rng = random.Random(1)
        for _ in range(500):
            s = "".join(rng.choice(chars) for _ in range(rng.randint(0, 40)))
            fast = alignment._sanitized_bytes(s)
            self.assertEqual(fast.decode("ascii"), alignment.sanitize_sequence(s), repr(s))


_STORE = {
    "s1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "s2": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVA",
    "s3": "MKTAYIAKQRQISFVKSHFSRQ",
    "s4": "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",
    "s5": "PLYISNDACEFHIKLMNPQRSTVWYACDEFGHIK",
    "s6": "mktayiakqrqisfvkshfsrqleerlglieee",
    "s7": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQUOJ*",
    "s8": "",
    "s9": "W",
    "s10": " MKTAYIAKQRQ ISFVKSHFSRQ\n",
}
_QUERIES = {"q1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "q2": "ACDEFGHIKLMNPQRSTVWY", "q3": "GGGGWGGGG",
            "q4": "", "q5": "BZX*MKTAYIAKQ", "q6": "mktayJakqrqisfvkshfsrqUO"}
_FIELDS = ["query", "target", "cigar", "qaln", "taln", "qseq", "tseq", "evalue", "bits"]


def _expected(gap=(11, 1), comp_bias_scale=None, alp=None, lam=None, k=None, db_residues=None):
    """align_candidates' result built from the biotite reference, pair by pair: the empty alignments dropped, sorted
    by bits (else raw score), then embedding score, descending (filters off)."""
    out = {}
    for q, qseq in _QUERIES.items():
        hits = []
        for i, (sid, sseq) in enumerate(_STORE.items()):
            m = ref.reference_metrics(qseq, sseq, gap=gap, comp_bias_scale=comp_bias_scale, needs=ALL_NEEDS,
                                      lam=lam, k=k, alp=alp, db_residues=db_residues)
            if m is not None and m.aln_len > 0:
                hits.append((sid, 1.0 - 0.01 * i, m))
        hits.sort(key=lambda h: (h[2].bit_score if h[2].bit_score is not None else h[2].score, h[1]), reverse=True)
        out[q] = [(sid, emb, dataclasses.astuple(m)) for sid, emb, m in hits]
    return out


def _run(**kw):
    pre = {q: (list(_STORE), [1.0 - 0.01 * i for i in range(len(_STORE))]) for q in _QUERIES}
    res = alignment.align_candidates(
        _QUERIES, pre, lambda ids: {i: _STORE[i] for i in ids}, min_seq_identity=0.0, min_coverage=0.0,
        max_evalue=None, subject_db_size=10_000_000, output_fields=_FIELDS, **kw)
    return {q: [(h.subject_id, h.emb_score, dataclasses.astuple(h.metrics)) for h in hits] for q, hits in res.items()}


@pytest.mark.usefixtures("fmsw_libs")
class TestStage2AgainstBiotite(unittest.TestCase):
    """The Stage 2 on the kernel against the biotite path it replaced (tests/blosum62_reference.py): every
    AlignmentMetrics field, every heavy string, the significance columns, the order."""

    def tearDown(self):
        alignment._worker_init(11, 1, None, None, None)

    def test_align_candidates_equals_the_biotite_path(self):
        alp = karlin_altschul.params_for(11, 1)
        for workers in (1, 2):
            with self.subTest(workers=workers):
                self.assertEqual(_run(num_workers=workers), _expected(alp=alp, db_residues=10_000_000))
                self.assertEqual(_run(num_workers=workers, comp_bias_corr=True),
                                 _expected(comp_bias_scale=1.0, alp=alp, db_residues=10_000_000))
                self.assertEqual(_run(num_workers=workers, comp_bias_corr=True, comp_bias_corr_scale=0.5,
                                      compute_significance=False),
                                 _expected(comp_bias_scale=0.5))
                self.assertEqual(_run(num_workers=workers, gap_open=-11, gap_extend=-1, compute_significance=False),
                                 _expected())

    def test_sampled_significance_equals_the_biotite_path(self):
        kw = dict(gap_open=5, gap_extend=2, significance_mode="sampled", significance_sample_size=20,
                  significance_sample_length=50)
        lam, k = alignment._estimate_lambda_k(5, 2, 20, 50, 0)
        self.assertEqual(_run(num_workers=1, **kw), _expected(gap=(5, 2), lam=lam, k=k, db_residues=10_000_000))

    def test_worker_metrics_equal_the_biotite_path(self):
        """``_align_query`` (one kernel call for a task) against the biotite path pair by pair, on random and related
        pairs, at several gap settings, with and without the composition bias and significance."""
        rng = random.Random(11)
        query = "".join(rng.choice(AA) for _ in range(150))
        cands = [(f"t{i}", "".join(rng.choice(AA + "BZX*") for _ in range(rng.randint(1, 300))), 0.5)
                 for i in range(30)]
        cands += [("self", query, 0.9), ("part", query[40:120], 0.8), ("mut", query[:60] + "W" * 7 + query[60:], 0.7),
                  ("empty", "", 0.1)]
        alp = karlin_altschul.params_for(11, 1)
        settings = [dict(gap=(11, 1), alp=alp, db_residues=10_169_347_728),
                    dict(gap=(11, 1), comp_bias_scale=1.0, alp=alp, db_residues=10_169_347_728),
                    dict(gap=(11, 1), comp_bias_scale=0.5),
                    dict(gap=(5, 2), lam=0.3, k=0.05, db_residues=1_000_000),
                    dict(gap=(1, 1)), dict(gap=(3, 3)), dict(gap=(20, 5)), dict(gap=(2, 0))]
        for s in settings:
            with self.subTest(**{k: v for k, v in s.items() if k != "alp"}):
                alignment._worker_init(s["gap"][0], s["gap"][1], s.get("lam"), s.get("k"), s.get("db_residues"),
                                       s.get("alp"), ALL_NEEDS, s.get("comp_bias_scale"))
                qid, hits = alignment._align_query(("q", query, cands))
                want = [ref.reference_metrics(query, seq, needs=ALL_NEEDS, **s) for _, seq, _ in cands]
                self.assertEqual(qid, "q")
                self.assertEqual([h.subject_id for h in hits], [c[0] for c, w in zip(cands, want) if w is not None])
                self.assertEqual([h.metrics for h in hits], [w for w in want if w is not None])
