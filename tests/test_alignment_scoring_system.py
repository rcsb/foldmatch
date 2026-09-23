"""Stage-2 alignment driven by a ScoringSystem.

The failures pinned here are the ones that produce a plausible result file rather than an
error: identity measured on paired symbols (which empties every result through the 0.3
filter), gaps taken from the CLI ints instead of the system's scaled ones (which makes
gaps nearly free), significance taken from the wrong constants, and a query silently
falling back to BLOSUM62 when its profile is missing (which mixes two scoring systems,
and two lambdas, in one file).
"""
import unittest
from dataclasses import replace

import numpy as np

from foldmatch.search import alignment
from foldmatch.search import scoring_system as ss
from foldmatch.types.api_types import SignificanceMode

SEQUENCE = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGCEKAVQVKVKALPDAQFEVVHSLAKWKR"
            "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")


def three_di_string(length, seed=20260922):
    rng = np.random.default_rng(seed)
    states = []
    while len(states) < length:
        states.extend([rng.integers(ss.N_STATES)] * int(rng.integers(1, 5)))
    return "".join(ss.THREE_DI[s] for s in states[:length])


def one_hot(states):
    profile = np.zeros((len(states), ss.N_STATES), dtype=np.float32)
    profile[np.arange(len(states)), states] = 1.0
    return np.clip(profile, 1e-6, None)


class PairedAlignmentTest(unittest.TestCase):
    """A self-hit under the 3Di system, which is where the identity trap shows up."""

    def setUp(self):
        self.states = ss.three_di_codes(three_di_string(len(SEQUENCE)))
        self.profile = one_hot(self.states)
        self.system = ss.get("3di-v1-exact")

    def run_search(self, system=None, **overrides):
        system = system or self.system
        kwargs = dict(
            query_sequences={"q": SEQUENCE},
            prefilter_results={"q": (["t"], [0.9])},
            fetch_subject_sequences=lambda ids: {"t": SEQUENCE},
            scoring=system,
            query_profiles={"q": self.profile},
            subject_codes={"t": self.states},
            significance_mode=SignificanceMode.scoring_system,
            subject_db_size=len(SEQUENCE),
            min_seq_identity=0.3,
            min_coverage=0.8,
            max_evalue=None,
            num_workers=1,
        )
        kwargs.update(overrides)
        return alignment.align_candidates(**kwargs)

    def test_self_hit_has_identity_one(self):
        """Identity must be measured on residues, not on paired symbols.

        Reading identity off the paired alphabet gives ~0.006 rather than 1.0, so the
        default min_seq_identity=0.3 would discard every hit while the scores and
        E-values still looked perfect.
        """
        hits = self.run_search()["q"]
        self.assertEqual(len(hits), 1, "the self-hit must survive the identity filter")
        self.assertAlmostEqual(hits[0].metrics.identity_aln, 1.0, places=6)
        self.assertAlmostEqual(hits[0].metrics.query_coverage, 1.0, places=6)

    def test_scores_beat_the_amino_only_system(self):
        """The 3Di term must actually reach the score."""
        paired = self.run_search()["q"][0].metrics.score
        plain = alignment.align_candidates(
            query_sequences={"q": SEQUENCE},
            prefilter_results={"q": (["t"], [0.9])},
            fetch_subject_sequences=lambda ids: {"t": SEQUENCE},
            compute_significance=False, max_evalue=None, num_workers=1,
        )["q"][0].metrics.score
        self.assertGreater(paired, plain * 10,
                           "a scale-100 3Di+amino score must dwarf the plain BLOSUM62 one")

    def test_gaps_come_from_the_system(self):
        """Deployed gaps are the system's, scaled; the nominal CLI ints must not leak."""
        captured = {}
        original = alignment._worker_init

        def spy(*args, **kwargs):
            captured["gaps"] = (args[0], args[1])
            return original(*args, **kwargs)

        alignment._worker_init = spy
        try:
            self.run_search(gap_open=11, gap_extend=1)
        finally:
            alignment._worker_init = original
        self.assertEqual(captured["gaps"], (3000, 300),
                         "gap penalties must be scaled with the matrix")

    def test_significance_uses_the_systems_constants(self):
        hits = self.run_search(max_evalue=None)["q"]
        metrics = hits[0].metrics
        system = self.system
        self.assertAlmostEqual(metrics.bit_score, system.bit_score(metrics.score), places=6)
        self.assertAlmostEqual(
            metrics.evalue,
            system.evalue(metrics.score, metrics.query_len, len(SEQUENCE)),
            delta=abs(metrics.evalue) * 1e-9 + 1e-300)

    def test_wrong_significance_mode_is_refused(self):
        """The catastrophic combination: a scale-100 score under BLOSUM62's constants.

        'default' would apply lambda 0.2736 to a score of ~300,000 and report bits in the
        hundred-thousands with E=0.0; 'sampled' would estimate lambda by sampling BLOSUM62
        while the aligner runs a 3Di matrix. Neither used to raise.
        """
        for mode in (SignificanceMode.default, SignificanceMode.sampled):
            with self.assertRaisesRegex(ValueError, "significance_mode must be"):
                self.run_search(significance_mode=mode)

    def test_amino_system_keeps_its_own_mode(self):
        """blosum62-11-1 takes its statistics from the ALP tables, so 'default' is right."""
        hits = alignment.align_candidates(
            query_sequences={"q": SEQUENCE},
            prefilter_results={"q": (["t"], [0.9])},
            fetch_subject_sequences=lambda ids: {"t": SEQUENCE},
            scoring=ss.get("blosum62-11-1"),
            significance_mode=SignificanceMode.default,
            subject_db_size=1000, max_evalue=None, num_workers=1,
        )["q"]
        self.assertEqual(len(hits), 1)
        with self.assertRaisesRegex(ValueError, "significance_mode must be"):
            alignment.align_candidates(
                query_sequences={"q": SEQUENCE},
                prefilter_results={"q": (["t"], [0.9])},
                fetch_subject_sequences=lambda ids: {"t": SEQUENCE},
                scoring=ss.get("blosum62-11-1"),
                significance_mode=SignificanceMode.scoring_system,
                subject_db_size=1000, max_evalue=None, num_workers=1)

    def test_amino_system_that_is_not_blosum_is_refused(self):
        """Its matrix would be ignored in favour of BLOSUM62, silently."""
        odd = replace(ss.get("blosum62-11-1"), name="odd", weight_amino=2.0, score_scale=10)
        with self.assertRaises(NotImplementedError):
            alignment.align_candidates(
                query_sequences={"q": SEQUENCE},
                prefilter_results={"q": (["t"], [0.9])},
                fetch_subject_sequences=lambda ids: {"t": SEQUENCE},
                scoring=odd, significance_mode=SignificanceMode.default,
                subject_db_size=1000, max_evalue=None, num_workers=1)

    def test_uncalibrated_system_is_refused(self):
        bare = replace(self.system, name="bare", lam=None, k=None, reference_set=None)
        with self.assertRaisesRegex(ValueError, "no calibrated"):
            self.run_search(system=bare)

    def test_missing_profile_skips_the_query(self):
        """Never fall back to BLOSUM62: that would mix two lambdas in one result file."""
        results = self.run_search(query_profiles={"other": self.profile})
        self.assertEqual(results["q"], [])

    def test_length_mismatched_profile_skips_the_query(self):
        results = self.run_search(query_profiles={"q": self.profile[:-3]})
        self.assertEqual(results["q"], [])

    def test_missing_subject_codes_skips_the_subject(self):
        results = self.run_search(subject_codes={})
        self.assertEqual(results["q"], [])

    def test_required_inputs_are_checked(self):
        with self.assertRaisesRegex(ValueError, "query_profiles"):
            self.run_search(query_profiles=None)
        with self.assertRaisesRegex(ValueError, "subject_codes"):
            self.run_search(subject_codes=None)

    def test_comp_bias_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            self.run_search(comp_bias_corr=True, comp_bias_corr_scale=1.0)

    def test_parallel_matches_serial(self):
        """Worker globals must survive the pool: spawn inherits nothing."""
        serial = self.run_search(num_workers=1)["q"][0].metrics
        parallel = self.run_search(num_workers=2)["q"][0].metrics
        self.assertEqual(serial.score, parallel.score)
        self.assertAlmostEqual(serial.identity_aln, parallel.identity_aln, places=9)


class IdentityIsAminoAcidAlongThe3DiPathTest(unittest.TestCase):
    """Reported identity is always amino-acid identity, measured on the 3Di-chosen path.

    3Di scoring decides WHICH columns align; identity then compares the amino acids in
    those columns. It must be neither identity over 3Di symbols nor identity from a
    separate amino-acid-only alignment. An unrelated amino-acid sequence carrying the same
    3Di states separates the three: 3Di aligns it end to end, only chance amino acids
    agree along that path, and BLOSUM62 alone would pick a different, shorter path.
    """

    def test_identity_is_measured_on_the_3di_alignment_with_amino_acids(self):
        rng = np.random.default_rng(0)
        target = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), len(SEQUENCE)))
        states = ss.three_di_codes(three_di_string(len(SEQUENCE), seed=1))
        fields = ["query", "target", "fident", "alnlen", "qaln", "taln"]
        hit = alignment.align_candidates(
            query_sequences={"q": SEQUENCE}, prefilter_results={"q": (["t"], [0.9])},
            fetch_subject_sequences=lambda ids: {"t": target},
            scoring=ss.get("3di-v1-exact"), query_profiles={"q": one_hot(states)},
            subject_codes={"t": states}, significance_mode=SignificanceMode.scoring_system,
            subject_db_size=len(target), min_seq_identity=0.0, min_coverage=0.0,
            max_evalue=None, num_workers=1, output_fields=fields)["q"][0].metrics

        # 3Di drove a full-length path through unrelated amino acids...
        self.assertEqual(len(hit.q_aln), len(SEQUENCE))
        self.assertTrue(set(hit.q_aln + hit.t_aln) <= set("ACDEFGHIKLMNPQRSTVWY-"),
                        "aligned strings must be amino acids, not paired symbols")
        # ...and identity counts identical AMINO ACIDS in exactly those columns.
        by_hand = sum(a == b and a != "-" for a, b in zip(hit.q_aln, hit.t_aln)) / len(hit.q_aln)
        self.assertAlmostEqual(hit.identity_aln, by_hand, places=12)
        self.assertLess(hit.identity_aln, 0.15, "identity over 3Di symbols would be ~1.0")

        plain = alignment.align_candidates(
            query_sequences={"q": SEQUENCE}, prefilter_results={"q": (["t"], [0.9])},
            fetch_subject_sequences=lambda ids: {"t": target}, compute_significance=False,
            min_seq_identity=0.0, min_coverage=0.0, max_evalue=None, num_workers=1,
            output_fields=fields)["q"]
        if plain:
            self.assertNotEqual(len(plain[0].metrics.q_aln), len(hit.q_aln),
                                "identity must come from the 3Di path, not a BLOSUM62 one")


class CodedTargetTest(unittest.TestCase):
    """The sequence-only database path, where the target's 3Di is a predicted byte."""

    def test_coded_system_aligns_and_scores(self):
        system = ss.get("3di-v1-coded")
        states = ss.three_di_codes(three_di_string(len(SEQUENCE)))
        # Codes are (predicted state, confidence bucket): place every residue in a
        # mid-confidence bucket of its true state.
        codes = states * 12 + 6
        results = alignment.align_candidates(
            query_sequences={"q": SEQUENCE},
            prefilter_results={"q": (["t"], [0.9])},
            fetch_subject_sequences=lambda ids: {"t": SEQUENCE},
            scoring=system,
            query_profiles={"q": one_hot(states)},
            subject_codes={"t": codes},
            significance_mode=SignificanceMode.scoring_system,
            subject_db_size=len(SEQUENCE),
            min_coverage=0.8, max_evalue=None, num_workers=1,
        )
        hits = results["q"]
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].metrics.identity_aln, 1.0, places=6)
        self.assertGreater(hits[0].metrics.score, 0)

    def test_out_of_range_codes_are_rejected(self):
        system = ss.get("3di-v1-coded")
        states = ss.three_di_codes(three_di_string(len(SEQUENCE)))
        with self.assertRaises(ValueError):
            ss.prepare(system).target_sequence(
                np.zeros(len(states), dtype=int), np.full(len(states), 240))


class EmptyDatabaseTest(unittest.TestCase):
    """An empty or unbuilt sequence store must say so, not raise a TypeError.

    ``SequenceStore.total_residues()`` is ``COALESCE(SUM(length), 0)``, so an empty store
    arrives here as ``0`` rather than ``None``. That slipped past the ``is None`` guard,
    left the database E-value unset, and surfaced as ``'<=' not supported between
    instances of 'NoneType' and 'float'`` from inside the result loop.
    """

    COMMON = dict(
        query_sequences={"q": SEQUENCE},
        prefilter_results={"q": (["t"], [0.9])},
        fetch_subject_sequences=lambda ids: {"t": SEQUENCE},
        num_workers=1,
    )

    def test_empty_database_with_evalue_filter_is_explained(self):
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError) as caught:
                    alignment.align_candidates(max_evalue=1e-3, subject_db_size=size,
                                               **self.COMMON)
                message = str(caught.exception)
                self.assertIn("non-empty database", message)
                self.assertIn("max_evalue=None", message)
                self.assertNotIsInstance(caught.exception, TypeError)

    def test_missing_database_size_still_explained(self):
        with self.assertRaisesRegex(ValueError, "requires subject_db_size"):
            alignment.align_candidates(max_evalue=1e-3, subject_db_size=None, **self.COMMON)

    def test_empty_database_without_evalue_filter_still_works(self):
        """Ranking without E-values is legitimate on an empty search space."""
        hits = alignment.align_candidates(max_evalue=None, subject_db_size=0,
                                          compute_significance=True, **self.COMMON)["q"]
        self.assertEqual(len(hits), 1)
        self.assertIsNone(hits[0].metrics.evalue)
        self.assertIsNotNone(hits[0].metrics.bit_score)

    def test_normal_database_size_is_unaffected(self):
        hits = alignment.align_candidates(max_evalue=1e-3, subject_db_size=len(SEQUENCE),
                                          **self.COMMON)["q"]
        self.assertEqual(len(hits), 1)
        self.assertIsNotNone(hits[0].metrics.evalue)


class BlosumPathUnchangedTest(unittest.TestCase):
    """The default path must be exactly what it was before a scoring system existed."""

    def test_no_scoring_system_is_the_old_behaviour(self):
        common = dict(
            query_sequences={"q": SEQUENCE},
            prefilter_results={"q": (["t"], [0.9])},
            fetch_subject_sequences=lambda ids: {"t": SEQUENCE[20:] + SEQUENCE[:20]},
            compute_significance=True,
            significance_mode=SignificanceMode.default,
            subject_db_size=1000, max_evalue=None, num_workers=1,
        )
        hits = alignment.align_candidates(**common)["q"]
        self.assertEqual(len(hits), 1)
        # Same numbers as the amino-only ScoringSystem, which is the same scoring system
        # expressed through the new machinery.
        paired = alignment.align_candidates(
            scoring=ss.get("blosum62-11-1"), **common)["q"]
        self.assertEqual(hits[0].metrics.score, paired[0].metrics.score)
        self.assertAlmostEqual(hits[0].metrics.identity_aln,
                               paired[0].metrics.identity_aln, places=9)


if __name__ == "__main__":
    unittest.main()
