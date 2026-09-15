import csv
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from foldmatch.search import alignment, output
from foldmatch.search.alignment import _chunk_candidate_tasks
from foldmatch.types.api_types import FormatMode
from foldmatch.search.output import (
    DEFAULT_OUTPUT_FIELDS,
    SUPPORTED_OUTPUT_FIELDS,
    parse_format_output,
)


# A small, fixed set of short protein sequences. Absolute identity values don't
# matter for these tests — only that serial and parallel paths agree and that
# every candidate is accounted for — so any deterministic sequences will do.
_STORE = {
    "s1": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "s2": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVA",   # 1 substitution vs s1
    "s3": "MKTAYIAKQRQISFVKSHFSRQ",              # prefix of s1
    "s4": "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG",   # unrelated
    "s5": "PLYISNDACEFHIKLMNPQRSTVWYACDEFGHIK",
    "s6": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEEE",
    "s7": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ",
    "s8": "MKTAYIAKQRQISFVKSHFSRQLEERLG",
}


def _fetch(ids):
    return {i: _STORE[i] for i in ids if i in _STORE}


def _signature(results):
    """A hashable, order-sensitive view of the aligned results for comparison."""
    out = {}
    for query_id, hits in results.items():
        out[query_id] = [
            (
                h.subject_id,
                round(h.metrics.identity_aln, 6),
                round(h.metrics.query_coverage, 6),
                round(h.metrics.subject_coverage, 6),
                h.metrics.score,
            )
            for h in hits
        ]
    return out


class TestChunkCandidateTasks(unittest.TestCase):
    def test_preserves_all_pairs_in_order(self):
        q1 = [(f"a{i}", "SEQ", float(i)) for i in range(10)]
        q2 = [(f"b{i}", "SEQ", float(i)) for i in range(3)]
        tasks = [("q1", "AAAA", q1), ("q2", "BBBB", q2)]

        chunks = _chunk_candidate_tasks(tasks, worker_budget=2)  # -> chunk_size 2

        # No chunk mixes queries and none exceeds the computed chunk size.
        self.assertTrue(all(len(c) <= 2 for _, _, c in chunks))
        # Concatenating each query's chunks reproduces its candidate list in order.
        regrouped = {}
        for qid, _, cand in chunks:
            regrouped.setdefault(qid, []).extend(cand)
        self.assertEqual(regrouped["q1"], q1)
        self.assertEqual(regrouped["q2"], q2)

    def test_single_query_is_split_across_many_chunks(self):
        cand = [(f"s{i}", "SEQ", float(i)) for i in range(64)]
        tasks = [("q", "AAAA", cand)]
        chunks = _chunk_candidate_tasks(tasks, worker_budget=8)
        # A lone query must fan out to more than one chunk so the pool can spread
        # it — this is the whole point of the change.
        self.assertGreater(len(chunks), 1)

    def test_empty_and_no_pairs(self):
        self.assertEqual(_chunk_candidate_tasks([], worker_budget=4), [])
        tasks = [("q", "AAAA", [])]
        # No pairs to align: returned as-is (caller re-seeds it to []).
        self.assertEqual(_chunk_candidate_tasks(tasks, worker_budget=4), tasks)


class TestAlignCandidatesParallelism(unittest.TestCase):
    def _run(self, prefilter, queries, num_workers):
        return alignment.align_candidates(
            query_sequences=queries,
            prefilter_results=prefilter,
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,   # keep every hit so completeness is checkable
            min_coverage=0.0,
            max_evalue=None,
            num_workers=num_workers,
            compute_significance=False,  # skip the slow lambda/K sampling pass
        )

    def test_serial_and_parallel_agree_single_query(self):
        # One query with many candidates: the case that previously pinned to one
        # core. Serial (workers=1) and pooled (workers=2, forces chunking) must
        # return byte-for-byte identical metrics and ordering.
        queries = {"q1": _STORE["s1"]}
        sids = [f"s{i}" for i in range(1, 9)]
        prefilter = {"q1": (sids, [1.0 / i for i in range(1, 9)])}

        serial = self._run(prefilter, queries, num_workers=1)
        parallel = self._run(prefilter, queries, num_workers=2)

        self.assertEqual(_signature(serial), _signature(parallel))
        # All eight candidates survive the (disabled) filters.
        self.assertEqual(len(parallel["q1"]), 8)

    def test_serial_and_parallel_agree_multi_query(self):
        queries = {"q1": _STORE["s1"], "q2": _STORE["s5"], "q3": _STORE["s7"]}
        prefilter = {
            "q1": (["s1", "s2", "s3", "s6", "s8"], [0.9, 0.8, 0.7, 0.6, 0.5]),
            "q2": (["s5", "s7", "s4"], [0.95, 0.55, 0.2]),
            "q3": (["s7", "s5"], [0.99, 0.4]),
        }
        serial = self._run(prefilter, queries, num_workers=1)
        parallel = self._run(prefilter, queries, num_workers=4)
        self.assertEqual(_signature(serial), _signature(parallel))
        # Result ordering (query order) is preserved.
        self.assertEqual(list(serial.keys()), list(parallel.keys()))

    def test_query_with_no_candidates_maps_to_empty(self):
        queries = {"q1": _STORE["s1"], "q2": _STORE["s5"]}
        prefilter = {
            "q1": (["s1", "s2"], [0.9, 0.8]),
            "q2": ([], []),          # nothing survived the prefilter
        }
        res = self._run(prefilter, queries, num_workers=2)
        self.assertIn("q2", res)
        self.assertEqual(res["q2"], [])
        self.assertEqual(len(res["q1"]), 2)


class TestParseFormatOutput(unittest.TestCase):
    def test_default_when_none(self):
        self.assertEqual(parse_format_output(None), list(DEFAULT_OUTPUT_FIELDS))

    def test_preserves_order_and_repeats(self):
        self.assertEqual(
            parse_format_output("target,query,target"),
            ["target", "query", "target"],
        )

    def test_strips_whitespace_and_blanks(self):
        self.assertEqual(parse_format_output(" query , target ,"), ["query", "target"])

    def test_unknown_field_raises_with_supported_list(self):
        with self.assertRaises(ValueError) as ctx:
            parse_format_output("query,bogus,target")
        msg = str(ctx.exception)
        self.assertIn("bogus", msg)
        # The error advertises the full supported set so the user can self-correct.
        self.assertIn("qaln", msg)

    def test_empty_spec_raises(self):
        with self.assertRaises(ValueError):
            parse_format_output("  ,  ")

    def test_every_supported_field_is_renderable(self):
        # Registry and renderer table must stay in lock-step.
        self.assertEqual(
            set(SUPPORTED_OUTPUT_FIELDS),
            set(output._FIELD_RENDERERS),
        )

    def test_every_supported_field_materializes_when_requested_alone(self):
        """Guard the registry (output.py) against the compute gate (alignment.py).

        The expensive per-hit strings are only built when the requested columns
        ask for them, and that gate — ``alignment._needs_from_fields`` — lives on
        the other side of the module boundary from the field registry. A column
        added to the registry and renderer but not wired into the gate would
        never be materialized and would silently render as an empty string in
        every output file. Requesting each field *on its own* is what exercises
        the gate; asking for them all at once would mask a missing entry.
        """
        for field in SUPPORTED_OUTPUT_FIELDS:
            with self.subTest(field=field):
                res = alignment.align_candidates(
                    query_sequences={"q1": _STORE["s1"]},
                    prefilter_results={"q1": (["s1"], [0.9])},
                    fetch_subject_sequences=_fetch,
                    min_seq_identity=0.0,
                    min_coverage=0.0,
                    max_evalue=None,
                    num_workers=1,
                    subject_db_size=sum(len(v) for v in _STORE.values()),
                    compute_significance=True,
                    output_fields=[field],
                )
                hit = res["q1"][0]
                self.assertNotEqual(
                    output.format_row("q1", hit, [field]), "",
                    f"column '{field}' rendered empty when requested alone — "
                    f"is it wired into alignment._needs_from_fields?",
                )


def _cigar_counts(cigar):
    """Sum of each op's run lengths in a CIGAR string, e.g. '7M1D3I' -> {M:7,D:1,I:3}."""
    counts = {"M": 0, "D": 0, "I": 0}
    for num, op in re.findall(r"(\d+)([MDI])", cigar):
        counts[op] += int(num)
    return counts


def _gap_runs(gapped):
    """Number of maximal '-' runs in an aligned string."""
    return len(re.findall(r"-+", gapped))


class TestOutputFieldComputation(unittest.TestCase):
    """The mmseqs-style columns must satisfy their defining invariants."""

    def _all_fields_hits(self, num_workers):
        queries = {"q1": _STORE["s1"], "q2": _STORE["s5"]}
        prefilter = {
            "q1": (["s1", "s2", "s3", "s6", "s8"], [0.9, 0.8, 0.7, 0.6, 0.5]),
            "q2": (["s5", "s7", "s4"], [0.95, 0.55, 0.2]),
        }
        return alignment.align_candidates(
            query_sequences=queries,
            prefilter_results=prefilter,
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            min_coverage=0.0,
            max_evalue=None,
            num_workers=num_workers,
            subject_db_size=sum(len(v) for v in _STORE.values()),
            compute_significance=False,
            output_fields=list(SUPPORTED_OUTPUT_FIELDS),
        )

    def test_field_invariants(self):
        results = self._all_fields_hits(num_workers=1)
        seen = 0
        for query_id, hits in results.items():
            for hit in hits:
                seen += 1
                m = hit.metrics
                # Sequences reported are exactly what was aligned.
                self.assertIsNotNone(m.q_seq)
                self.assertIsNotNone(m.t_seq)
                self.assertEqual(m.query_len, len(m.q_seq))
                self.assertEqual(m.subject_len, len(m.t_seq))

                # Aligned strings line up with the alignment length.
                self.assertEqual(len(m.q_aln), m.aln_len)
                self.assertEqual(len(m.t_aln), m.aln_len)

                # Column bookkeeping: alnlen = matches + mismatches + gap columns.
                q_gaps = m.q_aln.count("-")
                t_gaps = m.t_aln.count("-")
                self.assertEqual(m.aln_len, m.n_ident + m.mismatch + q_gaps + t_gaps)

                # fident is nident/alnlen.
                self.assertAlmostEqual(m.identity_aln, m.n_ident / m.aln_len, places=9)

                # CIGAR is consistent with the aligned strings.
                cc = _cigar_counts(m.cigar)
                self.assertEqual(sum(cc.values()), m.aln_len)
                self.assertEqual(cc["M"], m.n_ident + m.mismatch)
                self.assertEqual(cc["D"], q_gaps)   # D = gap in query
                self.assertEqual(cc["I"], t_gaps)   # I = gap in target

                # gapopen counts runs, not characters.
                self.assertEqual(m.gap_open, _gap_runs(m.q_aln) + _gap_runs(m.t_aln))

                # 1-indexed bounds within the sequences, and coverage matches them.
                self.assertTrue(1 <= m.q_start <= m.q_end <= m.query_len)
                self.assertTrue(1 <= m.t_start <= m.t_end <= m.subject_len)
                self.assertEqual(m.q_end - m.q_start + 1, m.aln_len - q_gaps)
                self.assertEqual(m.t_end - m.t_start + 1, m.aln_len - t_gaps)
                self.assertAlmostEqual(
                    m.query_coverage, (m.aln_len - q_gaps) / m.query_len, places=9
                )
                self.assertAlmostEqual(
                    m.subject_coverage, (m.aln_len - t_gaps) / m.subject_len, places=9
                )

                # The non-gap columns of qaln/taln are substrings of the sequences.
                self.assertEqual(m.q_aln.replace("-", ""), m.q_seq[m.q_start - 1:m.q_end])
                self.assertEqual(m.t_aln.replace("-", ""), m.t_seq[m.t_start - 1:m.t_end])
        self.assertGreater(seen, 0)

    def test_serial_and_parallel_agree_on_all_fields(self):
        def sig(results):
            return {
                qid: [output.format_row(qid, h, list(SUPPORTED_OUTPUT_FIELDS), "|") for h in hits]
                for qid, hits in results.items()
            }
        self.assertEqual(
            sig(self._all_fields_hits(1)),
            sig(self._all_fields_hits(2)),
        )


class TestDefaultFormatGating(unittest.TestCase):
    def test_heavy_fields_not_materialized_by_default(self):
        res = alignment.align_candidates(
            query_sequences={"q1": _STORE["s1"]},
            prefilter_results={"q1": (["s1", "s2"], [0.9, 0.8])},
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            min_coverage=0.0,
            max_evalue=None,
            num_workers=1,
            subject_db_size=sum(len(v) for v in _STORE.values()),
            compute_significance=False,
            # default output_fields (None) -> no heavy strings requested
        )
        m = res["q1"][0].metrics
        for attr in ("cigar", "q_aln", "t_aln", "q_seq", "t_seq"):
            self.assertIsNone(getattr(m, attr), f"{attr} should be gated off by default")
        # Scalars are always available regardless of format.
        self.assertGreater(m.n_ident, 0)
        self.assertEqual(m.q_start, 1)


class TestNoAlignmentCandidateDropped(unittest.TestCase):
    def test_empty_alignment_is_not_reported_even_at_zero_thresholds(self):
        # Two sequences that share no positive-scoring local alignment produce an
        # empty biotite trace (aln_len == 0). Such a candidate is not a hit and
        # must be dropped rather than emitted as a malformed position-0 row —
        # even with the identity/coverage thresholds relaxed all the way to 0.
        store = {"far": "PPPPPPPPPP"}
        res = alignment.align_candidates(
            query_sequences={"q": "WWWWWWWWWW"},
            prefilter_results={"q": (["far"], [0.99])},
            fetch_subject_sequences=lambda ids: {i: store[i] for i in ids if i in store},
            min_seq_identity=0.0,
            min_coverage=0.0,
            max_evalue=None,
            num_workers=1,
            subject_db_size=10,
            output_fields=list(SUPPORTED_OUTPUT_FIELDS),
        )
        self.assertEqual(res["q"], [])


class TestMaxEvalueFilter(unittest.TestCase):
    """The E-value threshold drops weak hits and self-enables significance."""

    def _run(self, max_evalue, **kwargs):
        return alignment.align_candidates(
            query_sequences={"q1": _STORE["s1"]},
            # s1/s2/s3 are near-identical to the query; s4 is unrelated noise.
            prefilter_results={"q1": (["s1", "s2", "s3", "s4"], [0.9, 0.8, 0.7, 0.6])},
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            min_coverage=0.0,
            max_evalue=max_evalue,
            num_workers=1,
            subject_db_size=sum(len(v) for v in _STORE.values()),
            **kwargs,
        )

    def test_threshold_drops_insignificant_hits(self):
        unfiltered = self._run(None, compute_significance=True)["q1"]
        filtered = self._run(1e-3)["q1"]
        self.assertGreater(len(unfiltered), len(filtered))
        # Only hits at or below the threshold survive, and the strong ones do.
        self.assertTrue(all(h.metrics.evalue <= 1e-3 for h in filtered))
        self.assertIn("s1", [h.subject_id for h in filtered])

    def test_significance_is_forced_on_for_the_filter(self):
        # compute_significance=False would leave every evalue None; the filter
        # must turn the pass back on rather than drop the whole result set.
        hits = self._run(1e-3, compute_significance=False)["q1"]
        self.assertTrue(hits)
        self.assertTrue(all(h.metrics.evalue is not None for h in hits))

    def test_none_and_inf_disable_the_filter(self):
        baseline = self._run(None, compute_significance=True)["q1"]
        self.assertEqual(len(self._run(float("inf"))["q1"]), len(baseline))

    def test_requires_a_search_space(self):
        with self.assertRaises(ValueError):
            alignment.align_candidates(
                query_sequences={"q1": _STORE["s1"]},
                prefilter_results={"q1": (["s1"], [0.9])},
                fetch_subject_sequences=_fetch,
                min_seq_identity=0.0,
                max_evalue=1e-3,
                num_workers=1,
                subject_db_size=None,
            )


class TestThresholdDefaultsMatchTheCli(unittest.TestCase):
    """The library's Stage-2 thresholds are the CLI's documented defaults.

    ``align_candidates`` is the public Python entry point for the same search
    the ``fm-search query`` commands run, so a user calling it directly must get
    the filtering the CLI's help text and README promise. The two live in
    different modules, so nothing but this test keeps them in step.
    """

    SHARED_DEFAULTS = (
        "min_seq_identity", "min_coverage", "max_evalue",
        "comp_bias_corr", "comp_bias_corr_scale",
    )

    def test_defaults_are_in_step(self):
        import inspect

        from foldmatch.cli import search as cli

        lib = inspect.signature(alignment.align_candidates).parameters
        for command in (cli.query_database_from_fasta, cli.query_database_from_database):
            cmd = inspect.signature(command).parameters
            for name in self.SHARED_DEFAULTS:
                with self.subTest(command=command.__name__, threshold=name):
                    self.assertEqual(
                        lib[name].default, cmd[name].default,
                        f"align_candidates {name}={lib[name].default!r} but "
                        f"--{name.replace('_', '-')} defaults to "
                        f"{cmd[name].default!r}; update both (and the README).",
                    )


_RESOURCES = Path(__file__).parent / "resources"
# Residue count of the target DB the mmseqs fixture was computed against.
_MMSEQS_FIXTURE_DB_RESIDUES = 9_805_184_914


def _reference_composition_bias(sequence, scale=1.0):
    """Loop-for-loop transcription of MMseqs2's calcLocalAaBiasCorrection + ssw_init rounding.

    Independent of the vectorized implementation; numpy scalars stand in for the
    C++ float/double types.
    """
    matrix, background = alignment._mmseqs_blosum62()
    lookup = {c: i for i, c in enumerate(alignment._MMSEQS_ALPHABET)}
    codes = [lookup.get(alignment._MMSEQS_LETTER_MAP.get(c, c), 20) for c in sequence.strip().upper()]
    n = len(codes)
    out = []
    for i in range(n):
        lo, hi = max(0, i - 20), min(n, i + 20)
        sub_mat = matrix[codes[i]]
        total = sum(int(sub_mat[codes[j]]) for j in range(lo, hi)) - int(sub_mat[codes[i]])
        delta = np.float32(total)
        delta = np.float32(np.float64(delta) / (-1.0 * np.float64(np.float32(hi - lo))))
        for a in range(21):
            delta = np.float32(np.float64(delta) + background[a] * np.float64(np.float32(sub_mat[a])))
        value = float(np.float32(scale) * delta)
        out.append(int(value - 0.5) if value < 0.0 else int(value + 0.5))
    return out


class TestLocalCompositionBias(unittest.TestCase):
    SEQUENCES = [
        _STORE["s1"], _STORE["s5"], _STORE["s7"], _STORE["s4"],
        "Q" * 60, "GGGGS" * 12, "PLYISNDACEFHIKLMNPQRSTVWY" * 3,
        "M", "MK", "mktayiakqrqisfv",
        "MKTBZJUOX*AYIAKQRQ",   # letters MMseqs2 folds (B->D, Z->E, J->L, U/O->X) or maps to X
    ]

    def test_matches_reference_transcription(self):
        for sequence in self.SEQUENCES:
            for scale in (1.0, 0.5, 0.3):
                with self.subTest(sequence=sequence[:15], scale=scale):
                    self.assertEqual(
                        alignment.local_composition_bias(sequence, scale).tolist(),
                        _reference_composition_bias(sequence, scale),
                    )

    def test_poly_q_interior_value(self):
        # From the MMseqs2 source: inside a poly-Q run, Q-Q (+5) is corrected to -1.
        bias = alignment.local_composition_bias("Q" * 60)
        self.assertTrue((bias[20:41] == -6).all(), bias.tolist())

    # Expected offsets from a C transcription of the MMseqs2 18-8cc5c source (integer
    # matrix generated from data/blosum62.out, setupLetterMapping,
    # calcLocalAaBiasCorrection and the ssw_init rounding), so they share no
    # constants with alignment.py. The first sequence depends on MMseqs2's flat -1 X
    # row/column; the second on its B->D, Z->E, J->L folding.
    MMSEQS_SOURCE_ORACLE = {
        ("XTAXXTXAXAAXSTSTAXAAXXSXTTXSXXXXXXTXTXAXXXXXSXTXTSXST", 1.0):
            [0, -1, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1, 0, 0, -1, 0, -1, -1, 0,
             -1, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, 0, 0, -1],
        ("XTAXXTXAXAAXSTSTAXAAXXSXTTXSXXXXXXTXTXAXXXXXSXTXTSXST", 0.5):
            [0, -1, -1, 0, 0, -1, 0, -1, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        ("JJEEJBJELBEBDLDEDLDDBLEBBJJZZBDBZBZDZEBZBEJZJDDZDELZJL", 1.0):
            [0, 0, -1, -2, 0, -3, 0, -1, 0, -3, -2, -3, -3, 0, -3, -2, -3, 1, -3, -3, -3, 1, -3, -4, -4, 1, 1,
             -3, -3, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, 1, -3, 1, -3, -3, -3, -3, -3, 1, -3, 1, 1],
        ("JJEEJBJELBEBDLDEDLDDBLEBBJJZZBDBZBZDZEBZBEJZJDDZDELZJL", 0.5):
            [0, 0, -1, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, 0, -2, -1, -2, 0, -2, -2, -2, 1, -1, -2, -2, 1, 1,
             -1, -1, -2, -2, -2, -1, -2, -1, -2, -1, -1, -2, -1, -2, -1, 1, -1, 0, -2, -2, -2, -2, -1, 1, -1, 0, 0],
    }

    def test_matches_mmseqs_source_oracle(self):
        for (sequence, scale), expected in self.MMSEQS_SOURCE_ORACLE.items():
            with self.subTest(sequence=sequence[:12], scale=scale):
                self.assertEqual(alignment.local_composition_bias(sequence, scale).tolist(), expected)

    def test_shape_and_zero_scale(self):
        self.assertEqual(len(alignment.local_composition_bias("  mktay\n")), 5)
        self.assertEqual(alignment.local_composition_bias("").tolist(), [])
        self.assertFalse(alignment.local_composition_bias(_STORE["s1"], 0.0).any())


class TestCompBiasMatchesMmseqs(unittest.TestCase):
    """Stage-2 scores against real mmseqs 18-8cc5c on the same pairs.

    resources/alignment/mmseqs_comp_bias_pairs.tsv holds 22 (query, target) pairs
    with the exact integer scores mmseqs assigned (recovered from its printed
    E-values) under --comp-bias-corr 0, --comp-bias-corr 1, and 1 with
    --comp-bias-corr-scale 0.5, plus the E-value mmseqs printed at its defaults
    for a 9,805,184,914-residue target DB. The pairs cover large score drops,
    corrections that raise the score, corrections that change the alignment
    path, and a 30-residue query.
    """

    @classmethod
    def setUpClass(cls):
        with open(_RESOURCES / "alignment" / "mmseqs_comp_bias_pairs.tsv") as f:
            cls.rows = list(csv.DictReader(f, delimiter="\t"))

    def _hits(self, num_workers=1, **kwargs):
        queries = {r["query"]: r["query_sequence"] for r in self.rows}
        targets = {r["target"]: r["target_sequence"] for r in self.rows}
        prefilter = {}
        for r in self.rows:
            ids, scores = prefilter.setdefault(r["query"], ([], []))
            ids.append(r["target"])
            scores.append(0.0)
        res = alignment.align_candidates(
            query_sequences=queries,
            prefilter_results=prefilter,
            fetch_subject_sequences=lambda ids: {i: targets[i] for i in ids if i in targets},
            min_seq_identity=0.0,
            min_coverage=0.0,
            max_evalue=None,
            num_workers=num_workers,
            subject_db_size=_MMSEQS_FIXTURE_DB_RESIDUES,
            **kwargs,
        )
        return {(q, h.subject_id): h for q, hits in res.items() for h in hits}

    def _assert_scores(self, column, **kwargs):
        hits = self._hits(**kwargs)
        self.assertEqual(len(hits), len(self.rows))
        for r in self.rows:
            with self.subTest(query=r["query"], target=r["target"]):
                self.assertEqual(hits[(r["query"], r["target"])].metrics.score, int(r[column]))

    def test_fixture_exercises_the_correction(self):
        changed = sum(r["raw_cbc0"] != r["raw_cbc1"] for r in self.rows)
        raised = sum(int(r["raw_cbc1"]) > int(r["raw_cbc0"]) for r in self.rows)
        self.assertGreaterEqual(changed, 15)
        self.assertGreaterEqual(raised, 3)

    def test_default_is_uncorrected(self):
        self._assert_scores("raw_cbc0")

    def test_on_matches_mmseqs_default(self):
        self._assert_scores("raw_cbc1", comp_bias_corr=True)

    def test_on_evalues_match_mmseqs(self):
        hits = self._hits(comp_bias_corr=True)
        for r in self.rows:
            with self.subTest(query=r["query"], target=r["target"]):
                expected = float(r["evalue_cbc1"])
                got = hits[(r["query"], r["target"])].metrics.evalue
                self.assertAlmostEqual(math.log(got), math.log(expected), delta=1e-3)

    def test_off_matches_comp_bias_corr_0(self):
        self._assert_scores("raw_cbc0", comp_bias_corr=False)

    def test_scale_half_matches_mmseqs(self):
        self._assert_scores("raw_cbc1_scale05", comp_bias_corr=True, comp_bias_corr_scale=0.5)

    def test_scale_zero_is_off(self):
        self._assert_scores("raw_cbc0", comp_bias_corr=True, comp_bias_corr_scale=0.0)

    def test_scale_out_of_range_is_rejected(self):
        for scale in (-0.1, 1.5):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self._hits(comp_bias_corr_scale=scale)

    def test_serial_and_parallel_agree_with_correction(self):
        # Workers get the correction through the pool initializer, not a global
        # set in the parent; both paths must score and place alignments identically.
        def signature(hits):
            return {k: (h.metrics.score, h.metrics.q_start, h.metrics.q_end, h.metrics.t_start, h.metrics.t_end)
                    for k, h in hits.items()}
        self.assertEqual(
            signature(self._hits(num_workers=1, comp_bias_corr=True)),
            signature(self._hits(num_workers=2, comp_bias_corr=True)),
        )


class TestWriteAlignedResults(unittest.TestCase):
    def test_tsv_no_header(self):
        fields = ["query", "target", "fident", "alnlen", "cigar"]
        res = alignment.align_candidates(
            query_sequences={"q1": _STORE["s1"]},
            prefilter_results={"q1": (["s1", "s2"], [0.9, 0.8])},
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            min_coverage=0.0,
            max_evalue=None,
            num_workers=1,
            subject_db_size=sum(len(v) for v in _STORE.values()),
            compute_significance=False,
            output_fields=fields,
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "hits.tsv"
            output.write_aligned_results(res, fields, str(out))
            lines = out.read_text().splitlines()

        self.assertEqual(len(lines), 2)  # two hits, no header row
        for line in lines:
            cols = line.split("\t")
            self.assertEqual(len(cols), len(fields))
            self.assertEqual(cols[0], "q1")     # query id
        # First hit is the identical self-match: fident 1.000, cigar full-length M.
        self.assertEqual(lines[0].split("\t")[2], "1.000")
        self.assertRegex(lines[0].split("\t")[4], r"^\d+M$")

    def test_embedding_score_column_carries_prefilter_score(self):
        """The embscore column reports the Stage-1 score for that hit."""
        fields = ["query", "target", "embscore"]
        res = alignment.align_candidates(
            query_sequences={"q1": _STORE["s1"]},
            prefilter_results={"q1": (["s1", "s2"], [0.9, 0.75])},
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            min_coverage=0.0,
            max_evalue=None,
            num_workers=1,
            compute_significance=False,
            output_fields=fields,
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "hits.tsv"
            output.write_aligned_results(res, fields, str(out))
            rows = [ln.split("\t") for ln in out.read_text().splitlines()]

        by_target = {r[1]: r[2] for r in rows}
        # Formatted identically to the embedding-only output writer.
        self.assertEqual(by_target["s1"], output.format_embedding_score(0.9))
        self.assertEqual(by_target["s2"], output.format_embedding_score(0.75))
        self.assertEqual(by_target["s1"], "0.900000")

    def test_default_format_leads_with_ids_then_embedding_score(self):
        self.assertEqual(
            output.DEFAULT_OUTPUT_FIELDS[:3],
            ("query", "target", "embscore"),
        )


class TestFormatMode(unittest.TestCase):
    """--format-mode selects the file layout; the column set is chosen elsewhere."""

    def _hits(self, fields):
        return alignment.align_candidates(
            query_sequences={"q1": _STORE["s1"]},
            prefilter_results={"q1": (["s1", "s2"], [0.9, 0.8])},
            fetch_subject_sequences=_fetch,
            min_seq_identity=0.0,
            min_coverage=0.0,
            max_evalue=None,
            num_workers=1,
            compute_significance=False,
            output_fields=fields,
        )

    def test_headless_is_the_default_and_emits_no_header(self):
        fields = ["query", "target", "fident"]
        res = self._hits(fields)
        with tempfile.TemporaryDirectory() as d:
            implicit, explicit = Path(d) / "a.tsv", Path(d) / "b.tsv"
            output.write_aligned_results(res, fields, str(implicit))
            output.write_aligned_results(res, fields, str(explicit),
                                         FormatMode.headless_tsv)
            a, b = implicit.read_text(), explicit.read_text()
        # Omitting the argument must equal asking for headless explicitly.
        self.assertEqual(a, b)
        self.assertEqual(a.splitlines()[0].split("\t")[0], "q1")  # data, not a header

    def test_tsv_mode_adds_exactly_one_header_row(self):
        fields = ["query", "target", "fident", "alnlen"]
        res = self._hits(fields)
        with tempfile.TemporaryDirectory() as d:
            headless, headed = Path(d) / "a.tsv", Path(d) / "b.tsv"
            output.write_aligned_results(res, fields, str(headless))
            output.write_aligned_results(res, fields, str(headed), FormatMode.tsv)
            bare, hdr = headless.read_text().splitlines(), headed.read_text().splitlines()
        self.assertEqual(hdr[0].split("\t"), fields)   # header names the columns
        self.assertEqual(hdr[1:], bare)                # data rows are untouched
        self.assertEqual(len(hdr), len(bare) + 1)      # exactly one extra line

    def test_header_follows_the_requested_column_order(self):
        fields = ["bits", "query", "cigar", "target"]  # deliberately not canonical
        res = self._hits(fields)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "a.tsv"
            output.write_aligned_results(res, fields, str(out), FormatMode.tsv)
            header = out.read_text().splitlines()[0]
        self.assertEqual(header.split("\t"), fields)

    def test_cluster_header(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "c.tsv"
            n = output.write_cluster_results(
                [("1acb.A", 0, 2), ("1acb.B", 0, 2)], str(out), FormatMode.tsv)
            lines = out.read_text().splitlines()
        self.assertEqual(lines[0].split("\t"), list(output.CLUSTER_OUTPUT_FIELDS))
        self.assertEqual(n, 2, "returned count must exclude the header row")

    def test_plain_string_mode_is_accepted(self):
        # Typer hands over the enum, but the API should tolerate its value too.
        self.assertEqual(output.format_spec("tsv"), output.format_spec(FormatMode.tsv))

    def test_unknown_mode_raises_with_supported_list(self):
        with self.assertRaises(ValueError) as ctx:
            output.format_spec("json")
        self.assertIn("headless_tsv", str(ctx.exception))

    def test_every_mode_has_a_spec(self):
        # A FormatMode member added without a spec would fail only at write time.
        for mode in FormatMode:
            self.assertIsNotNone(output.format_spec(mode))


class TestWriteEmbeddingResults(unittest.TestCase):
    """The embedding-only (Stage-1) output shares the Stage-2 file convention."""

    def test_tsv_no_header_three_columns(self):
        results = {
            "q1": (["s1", "s2"], [0.91, 0.4]),
            "q2": (["s3"], [0.55]),
        }
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "emb.tsv"
            output.write_embedding_results(results, str(out))
            lines = out.read_text().splitlines()

        self.assertEqual(len(lines), 3)  # three hits, no header row
        self.assertEqual(lines[0].split("\t"), ["q1", "s1", "0.910000"])
        self.assertEqual(lines[1].split("\t"), ["q1", "s2", "0.400000"])
        self.assertEqual(lines[2].split("\t"), ["q2", "s3", "0.550000"])
        for line in lines:
            self.assertEqual(len(line.split("\t")), len(output.EMBEDDING_OUTPUT_FIELDS))

    def test_query_with_no_hits_emits_no_rows(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "emb.tsv"
            output.write_embedding_results({"q1": ([], [])}, str(out))
            self.assertEqual(out.read_text(), "")

    def test_header_mode_prepends_column_names(self):
        results = {"q1": (["s1"], [0.91])}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "emb.tsv"
            output.write_embedding_results(results, str(out), FormatMode.tsv)
            lines = out.read_text().splitlines()
        self.assertEqual(lines[0].split("\t"), list(output.EMBEDDING_OUTPUT_FIELDS))
        self.assertEqual(lines[1].split("\t"), ["q1", "s1", "0.910000"])

    def test_columns_match_stage2_field_names(self):
        # Every embedding-only column name is also a Stage-2 field, so the two
        # files can be parsed with one schema.
        for name in output.EMBEDDING_OUTPUT_FIELDS:
            self.assertIn(name, SUPPORTED_OUTPUT_FIELDS)


if __name__ == "__main__":
    unittest.main()
