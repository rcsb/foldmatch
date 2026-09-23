"""The query half: a sidecar on disk actually reaching the aligner.

The build half writes codes; this is where they are read back and scored. The failures
guarded here are the quiet ones — a subject whose codes are missing vanishes from the
results with only a log line, and a database searched under the wrong scoring system
still produces alignments and confident E-values.
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from foldmatch.search import alignment
from foldmatch.search import scoring_system as ss
from foldmatch.search.three_di_store import ThreeDiStore, ThreeDiStoreError
from foldmatch.types.api_types import SignificanceMode

SEQUENCE = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGCEKAVQVKVKALPDAQFEVVHSLAKWKR"
            "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")


def states(seed=7):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < len(SEQUENCE):
        out.extend([int(rng.integers(ss.N_STATES))] * int(rng.integers(1, 5)))
    return np.array(out[:len(SEQUENCE)])


def one_hot(state_ids):
    profile = np.full((len(state_ids), ss.N_STATES), 1e-6, dtype=np.float32)
    profile[np.arange(len(state_ids)), state_ids] = 1.0
    return profile / profile.sum(1, keepdims=True)


class SidecarFeedsTheAlignerTest(unittest.TestCase):
    """The join: codes on disk -> ThreeDiStore.fetch -> align_candidates."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.system = ss.get("3di-v1-coded")
        self.states = states()
        self.codes = ss.prepare(self.system).encode_target(one_hot(self.states))
        self.store = ThreeDiStore(self.dir, "db")
        self.store.create([("t", self.codes), ("other", self.codes)])
        self.store.write_manifest(self.system, model_revision="a" * 40,
                                  esm_revision="b" * 40, entry_count=2)

    def search(self, **overrides):
        kwargs = dict(
            query_sequences={"q": SEQUENCE},
            prefilter_results={"q": (["t"], [0.9])},
            fetch_subject_sequences=lambda ids: {"t": SEQUENCE},
            scoring=self.system,
            query_profiles={"q": one_hot(self.states)},
            # A callable, exactly as the CLI passes it.
            subject_codes=self.store.fetch,
            significance_mode=SignificanceMode.scoring_system,
            subject_db_size=len(SEQUENCE),
            min_seq_identity=0.0, min_coverage=0.0, max_evalue=None, num_workers=1,
        )
        kwargs.update(overrides)
        return alignment.align_candidates(**kwargs)

    def test_codes_from_the_store_are_scored(self):
        hits = self.search()["q"]
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].metrics.identity_aln, 1.0, places=6)
        self.assertGreater(hits[0].metrics.score, 0)

    def test_callable_and_dict_forms_agree(self):
        """The callable form must not change a single number."""
        from_callable = self.search()["q"][0].metrics
        from_dict = self.search(subject_codes=self.store.fetch(["t"]))["q"][0].metrics
        self.assertEqual(from_callable.score, from_dict.score)
        self.assertEqual(from_callable.evalue, from_dict.evalue)

    def test_the_callable_receives_the_whole_candidate_union(self):
        """Building the dict caller-side would mean recomputing this union by hand."""
        seen = {}

        def spy(ids):
            seen["ids"] = set(ids)
            return self.store.fetch(ids)

        self.search(prefilter_results={"q": (["t", "other"], [0.9, 0.8])},
                    fetch_subject_sequences=lambda ids: {"t": SEQUENCE, "other": SEQUENCE},
                    subject_codes=spy)
        self.assertEqual(seen["ids"], {"t", "other"})

    def test_subject_without_codes_is_skipped_not_scored_as_amino_acids(self):
        hits = self.search(prefilter_results={"q": (["absent"], [0.9])},
                           fetch_subject_sequences=lambda ids: {"absent": SEQUENCE})["q"]
        self.assertEqual(hits, [], "a subject with no codes must not fall back silently")


class ManifestGuardsTest(unittest.TestCase):
    """A database must not be searchable under a system it was not built with."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.system = ss.get("3di-v1-coded")
        self.store = ThreeDiStore(self.dir, "db")
        self.store.create([("t", np.zeros(4, np.uint8))])
        self.store.write_manifest(self.system, model_revision="a" * 40,
                                  esm_revision="b" * 40, entry_count=1)

    def test_partial_build_is_not_usable(self):
        store = ThreeDiStore(self.dir, "partial")
        store.create([("t", np.zeros(4, np.uint8))])
        self.assertFalse(store.exists())
        self.assertTrue(store.is_partial())

    def test_wrong_temperature_refuses(self):
        with self.assertRaises(ThreeDiStoreError):
            self.store.check_compatible(replace(self.system, profile_temperature=1.0))

    def test_wrong_esm_revision_refuses(self):
        with self.assertRaises(ThreeDiStoreError):
            self.store.check_compatible(self.system, esm_revision="0" * 40)

    def test_matching_everything_passes(self):
        self.store.check_compatible(self.system, model_revision="a" * 40,
                                    esm_revision="b" * 40)


class CliSurfaceTest(unittest.TestCase):
    """The switch is amino-acid-only (default) versus amino-acid + 3Di."""

    def test_query_sequences_offers_scoring_and_defaults_to_amino_only(self):
        import inspect
        from foldmatch.cli import search
        parameters = inspect.signature(search.query_database_from_fasta).parameters
        self.assertIn("scoring", parameters)
        from foldmatch.types.api_types import ScoringChoice
        self.assertIs(parameters["scoring"].default, ScoringChoice.blosum62,
                      "amino-acid-only must stay the default")

    def test_query_db_refuses_scoring_for_now(self):
        import inspect
        from foldmatch.cli import search
        source = inspect.getsource(search.query_database_from_database)
        self.assertIn("not yet supported for a database-vs-database", source)

    def test_build_offers_three_di(self):
        import inspect
        from foldmatch.cli import search
        parameter = inspect.signature(search.build_database_from_fasta).parameters["three_di"]
        self.assertIs(parameter.default, False, "3Di must be opt-in at build time")

    def test_significance_mode_is_derived_not_asked(self):
        """The CLI default would make a 3Di search raise; the funnel must override it."""
        import inspect
        from foldmatch.cli import search
        source = inspect.getsource(search._stage2_align_and_report)
        self.assertIn("SignificanceMode.scoring_system", source)


class ScoringChoicesTest(unittest.TestCase):
    """Users pick 'blosum62' or 'three_di'; which 3Di system that means is the database's.

    Friendly names cost nothing in reproducibility because 'three_di' is resolved from the
    subject database's manifest, never from "whatever is newest": a database built under
    one encoding keeps being searched under that encoding after a newer one ships.
    """

    def _database_with_three_di(self, name="3di-v1-coded"):
        import json
        folder = Path(tempfile.mkdtemp())
        store = ThreeDiStore(folder, "db")
        store.create([("t", np.zeros(4, np.uint8))])
        store.write_manifest(ss.get("3di-v1-coded"), model_revision="a" * 40,
                             esm_revision="b" * 40, entry_count=1)
        if name != "3di-v1-coded":
            manifest = store.read_manifest()
            manifest["scoring_system"]["name"] = name
            store.manifest_path.write_text(json.dumps(manifest))
        return folder

    def test_choices_are_the_two_friendly_names(self):
        from foldmatch.types.api_types import ScoringChoice
        self.assertEqual({m.value for m in ScoringChoice}, {"blosum62", "three_di"})

    def test_the_build_default_is_a_coded_registry_system(self):
        """--three-di writes this encoding; it must be one a database can store."""
        system = ss.get(ss.DEFAULT_THREE_DI)
        self.assertEqual(system.target_repr, "coded",
                         "a sequence database can only store coded (predicted) 3Di")

    def test_three_di_resolves_from_the_database_manifest(self):
        from foldmatch.cli.search import _resolve_three_di_system
        folder = self._database_with_three_di()
        self.assertEqual(_resolve_three_di_system(str(folder / "db")), "3di-v1-coded")

    def test_resolution_follows_the_database_not_the_newest_encoding(self):
        """A database records its own system; the search must not substitute another."""
        from foldmatch.cli.search import _resolve_three_di_system
        folder = self._database_with_three_di()
        original = ss.DEFAULT_THREE_DI
        try:
            ss.DEFAULT_THREE_DI = "some-future-encoding"
            self.assertEqual(_resolve_three_di_system(str(folder / "db")), "3di-v1-coded")
        finally:
            ss.DEFAULT_THREE_DI = original

    def test_database_without_three_di_is_refused_clearly(self):
        from foldmatch.cli.search import _resolve_three_di_system
        folder = Path(tempfile.mkdtemp())
        with self.assertRaisesRegex(ValueError, "built without --three-di"):
            _resolve_three_di_system(str(folder / "db"))

    def test_interrupted_build_is_named_as_such(self):
        from foldmatch.cli.search import _resolve_three_di_system
        folder = Path(tempfile.mkdtemp())
        ThreeDiStore(folder, "db").create([("t", np.zeros(4, np.uint8))])
        with self.assertRaisesRegex(ValueError, "did not finish"):
            _resolve_three_di_system(str(folder / "db"))

    def test_unknown_system_in_the_manifest_is_refused(self):
        """A database from a newer FoldMatch must not be read as whatever this one has."""
        from foldmatch.cli.search import _resolve_three_di_system
        folder = self._database_with_three_di(name="3di-v9-coded")
        with self.assertRaisesRegex(ValueError, "does not know"):
            _resolve_three_di_system(str(folder / "db"))

    def test_invalid_values_are_rejected_before_anything_runs(self):
        """Typer refuses them at parse time: exit code 2, no ESM3 download, no traceback."""
        from typer.testing import CliRunner
        from foldmatch.cli.search import app
        runner = CliRunner()
        for argv in (["query", "sequences", "--scoring", "3di-v1-coded"],
                     ["query", "sequences", "--scoring", "nonsense"],
                     ["query", "db", "--scoring", "nonsense"]):
            with self.subTest(" ".join(argv)):
                result = runner.invoke(app, argv)
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("Invalid value", result.output)

    def test_amino_system_needs_no_sidecar(self):
        """Naming the amino-acid system to the funnel must still run the unchanged path."""
        import types
        from foldmatch.cli import search
        from foldmatch.search.sequence_store import SequenceStore
        folder = Path(tempfile.mkdtemp())
        fasta = folder / "s.fa"
        fasta.write_text(f">t\n{SEQUENCE}\n")
        SequenceStore(folder, "db").create(fasta_file=fasta)
        database = types.SimpleNamespace(db_folder=folder, index_name="db",
                                         db=types.SimpleNamespace(chain_ids=["t"]))
        search._stage2_align_and_report(
            embedding_db=database, prefilter_results={"q": (["t"], [0.9])},
            query_sequences={"q": SEQUENCE}, min_seq_identity=0.0, min_coverage=0.0,
            max_evalue=None, gap_open=11, gap_extend=1, comp_bias_corr=False,
            comp_bias_corr_scale=1.0, num_workers=1, output_file=str(folder / "o.tsv"),
            scoring_name="blosum62-11-1")
        self.assertTrue((folder / "o.tsv").read_text().strip(), "no hit was written")


if __name__ == "__main__":
    unittest.main()
