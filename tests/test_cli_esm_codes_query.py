"""P6.1-P6.3: ``fm-search query sequences --scoring blosum62|esm-codes`` (plan section 8 P6; D7, D8, D11, D14, D20).

Most tests replace the ESM3 pass by a fake that writes the shards the real pass writes (same dataset, same writer),
with the F2c reference codes (builds) and query encodings (queries) of the fixture FASTA: the command-line
orchestration, the refusals, the query-encoding memmap and the real esm-codes Stage 2 (kernel, registry bundle) run
without ESM3. ``RealEsm3SmokeTest`` is the end-to-end smoke of the plan (CPU, ESM3 from the cache) and
``RealEsm3DegenerateTest`` runs the real-ESM3 F-deg items through the command line (D14, S13, S13b).
"""
import dataclasses
import gc
import io
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
import weakref
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

import esm_codes_bundle
from foldmatch.cli import search
from foldmatch.errors import RefusalError
from foldmatch.search import code_alignment, output, swkern
from foldmatch.search import scoring_system as ss
from foldmatch.search.residue_code_store import (
    IncompatibleCodeStoreError, ResidueCodeStore, ResidueCodeStoreError,
)
from foldmatch.types.api_types import ScoringChoice, SignificanceMode
from foldmatch.utils.sequence import STANDARD_AMINO_ACIDS
from test_cli_residue_codes import _build_args, _esm3_spy, _f2c, _sha1, _sha256, _write, fake_chain_predict, \
    invoke, launch_env

_TEST_PATH = Path(__file__).parent
_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
_F2C = _TEST_PATH / "resources" / "esm_codes" / "fasta_rows_codes_v1.npz"
_FDEG = _TEST_PATH / "resources" / "esm_codes" / "degenerate_v1.npz"
SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)
DOMAIN = SYSTEM.calibrated_domain
ALL_FIELDS = ",".join(output.SUPPORTED_OUTPUT_FIELDS)
DEFAULT_FIELDS = output.DEFAULT_OUTPUT_FIELDS
#: Numeric output columns (finite whenever they are filled).
NUMERIC_FIELDS = ("embscore", "evalue", "gapopen", "pident", "fident", "nident", "qstart", "qend", "qlen", "tstart",
                  "tend", "tlen", "alnlen", "raw", "bits", "mismatch", "qcov", "tcov", "rawrev", "q90", "rawadj",
                  "ecal")
_DB_SUFFIXES = (".index", ".metadata", ".sequences")


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _f2c_q() -> dict:
    """F2c: ``{sequence: reference query encoding fp16 [L, 64]}`` of the fixture FASTA."""
    z = np.load(_F2C, allow_pickle=False)
    offsets = z["offsets"]
    return {str(s): z["q"][offsets[k]:offsets[k + 1]] for k, s in enumerate(z["sequences"])}


def _fake_q(name: str, sequence: str) -> np.ndarray:
    """The F2c query encodings of the fixture's sequences; pseudo-random fp16 rows (fixed per id) for any other."""
    reference = _f2c_q().get(sequence)
    if reference is not None:
        return reference
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return rng.normal(size=(len(sequence), 64)).astype(np.float16)


def fake_query_pass(calls=None, payload=_fake_q, seq_sha1=None):
    """Stand-in for ``chain_complete_inference.predict`` in a query: :func:`fake_chain_predict` with the query
    encodings as payloads; ``seq_sha1(name, sequence)``, if given, replaces the seq_sha1 the shard records."""
    if seq_sha1 is None:
        return fake_chain_predict(calls, payload=payload)

    def predict(**kwargs):
        from foldmatch.writer import batch_writer
        real = batch_writer.ParquetBatchWriter.write_on_batch_end

        def write(self, trainer, module, prediction, *args):
            embeddings, names, payloads, _ = prediction
            sequences = {name: seq for name, seq in _fasta_records(kwargs["src_stream"].path)}
            sha1s = tuple(seq_sha1(name, sequences[name]) for name in names)
            return real(self, trainer, module, (embeddings, names, payloads, sha1s), *args)

        with mock.patch.object(batch_writer.ParquetBatchWriter, "write_on_batch_end", write):
            fake_chain_predict(calls, payload=payload)(**kwargs)
    return predict


def _fasta_records(path):
    from foldmatch.utils.fasta import iter_fasta
    from foldmatch.utils.sequence import normalize_sequence
    return [(name, normalize_sequence(seq)) for name, seq in iter_fasta(path)]


def _query_args(db, fasta, shards, out, *extra):
    return ("query", "sequences", "--db-path", db, "--fasta-file", fasta, "--tmp-embedding-folder", shards,
            "--output-file", out, "--accelerator", "cpu", "--devices", "1", *extra)


def _esm(*extra):
    return ("--scoring", "esm-codes", *extra)


def _rows(path, fields=DEFAULT_FIELDS):
    """The rows of a headless TSV output as dicts over ``fields``."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line:
            cols = line.split("\t")
            assert len(cols) == len(fields), (line, fields)
            out.append(dict(zip(fields, cols)))
    return out


def _no_pass():
    """The ESM3 pass must not be reached."""
    return mock.patch("foldmatch.search.embedding_computer.chain_predict",
                      mock.Mock(side_effect=AssertionError("the ESM3 pass was reached")))


def _clear(folder: Path):
    for path in folder.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


class _Case(unittest.TestCase):
    """A temporary folder per test (``self.tmp``, query shards in ``self.qshards``), no launch variables."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.qshards = self.tmp / "qshards"
        self.qshards.mkdir()
        self.enterContext(launch_env())

    def refused(self, result, cause=None):
        """Exit status 1 with a RefusalError (D20); returns its message."""
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIsInstance(result.exception, RefusalError, repr(result.exception))
        if cause is not None:
            self.assertIsInstance(result.exception.__cause__, cause, repr(result.exception.__cause__))
        return str(result.exception)

    def usage_error(self, result, *texts):
        """Exit status 2 (typer.BadParameter) naming every one of ``texts`` (typer's error panel borders dropped)."""
        self.assertEqual(result.exit_code, 2, result.output)
        flat = " ".join(result.output.translate({ord(c): " " for c in "│╭╮╰╯─"}).split())
        for text in texts:
            self.assertIn(text, flat)
        return flat


@pytest.mark.usefixtures("fmsw_libs")
class _DbCase(_Case):
    """Two databases of the fixture FASTA built once per class with the fake pass: ``codes`` (``--residue-codes``,
    F2c codes) and ``plain``. ``database(kind)`` copies one into the test's folder."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        esm_codes_bundle.registry_files()  # unittest.SkipTest only when offline and not cached
        swkern.load()
        cls._dbs = tempfile.TemporaryDirectory()
        cls.dbs = Path(cls._dbs.name)
        for kind, extra in (("codes", ("--residue-codes",)), ("plain", ())):
            (cls.dbs / kind).mkdir()
            shards = cls.dbs / f"{kind}-shards"
            shards.mkdir()
            with launch_env(FOLDMATCH_RUN_ID=f"build-{kind}"), \
                    mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
                result = invoke(*_build_args(_FASTA, cls.dbs / kind / "db", shards, *extra))
            assert result.exit_code == 0, (result.output, result.exception)

    @classmethod
    def tearDownClass(cls):
        cls._dbs.cleanup()
        super().tearDownClass()

    def database(self, kind="codes", name="db") -> Path:
        """A copy of the class's ``kind`` database in ``self.tmp/kind``; returns its ``--db-path``."""
        folder = self.tmp / kind
        if not folder.exists():
            shutil.copytree(self.dbs / kind, folder)
        return folder / name

    def query(self, db, *extra, fasta=_FASTA, out=None, run_id="q", predict=None, calls=None):
        """``query sequences`` with the fake query pass; returns ``(result, output path)``."""
        out = out or self.tmp / "out.tsv"
        with launch_env(FOLDMATCH_RUN_ID=run_id), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", predict or fake_query_pass(calls)):
            result = invoke(*_query_args(db, fasta, self.qshards, out, *extra))
        return result, out


def _edit_manifest(db: Path, edit):
    path = Path(f"{db}.rescodes.json")
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


# =====================================================================================================================
# P6.1: the options
# =====================================================================================================================
class OptionTest(unittest.TestCase):

    def _command(self, *path):
        import typer
        command = typer.main.get_command(search.app)
        for name in path:
            command = command.commands[name]
        return command

    def _param(self, path, name):
        return next(p for p in self._command(*path).params if p.name == name)

    def test_scoring_option(self):
        param = self._param(("query", "sequences"), "scoring")
        self.assertEqual(param.opts, ["--scoring"])
        self.assertEqual(list(param.type.choices), ["blosum62", "esm-codes"])
        self.assertIs(param.default, ScoringChoice.blosum62)
        for path in (("query", "db"), ("query", "structure"), ("query", "embedding")):
            with self.subTest(command=" ".join(path)):
                self.assertNotIn("scoring", [p.name for p in self._command(*path).params])

    def test_scoring_dependent_defaults_are_resolved_per_scoring(self):
        """The typer defaults are None (resolved per scoring) and the help states both scorings' defaults."""
        for name in ("min_seq_identity", "min_coverage", "max_evalue", "gap_open", "gap_extend", "comp_bias_corr",
                     "significance_mode"):
            with self.subTest(option=name):
                param = self._param(("query", "sequences"), name)
                self.assertIsNone(param.default)
        for name, text in (("min_seq_identity", "Default: 0.3 under --scoring blosum62, 0 under esm-codes"),
                           ("min_coverage", "Default: 0.8 under --scoring blosum62, 0 under esm-codes"),
                           ("max_evalue", "(default 1e-3)"), ("gap_open", "refused with --scoring esm-codes"),
                           ("gap_extend", "refused with --scoring esm-codes"),
                           ("comp_bias_corr", "refused with --scoring esm-codes"),
                           ("significance_mode", "refused with --scoring esm-codes")):
            with self.subTest(option=name):
                self.assertIn(text, self._param(("query", "sequences"), name).help)
        # query db keeps its options and defaults.
        for name, default in (("min_seq_identity", 0.3), ("min_coverage", 0.8), ("max_evalue", 1e-3),
                              ("gap_open", 11), ("gap_extend", 1), ("comp_bias_corr", False)):
            with self.subTest(command="query db", option=name):
                self.assertEqual(self._param(("query", "db"), name).default, default)

    def test_output_fields_in_the_help(self):
        """--format-output lists the esm-codes fields in query sequences and query db alike (P6.5, CHANGELOG)."""
        for path in (("query", "sequences"), ("query", "db")):
            with self.subTest(command=" ".join(path)):
                text = self._param(path, "format_output").help
                self.assertIn(", ".join(output.SUPPORTED_OUTPUT_FIELDS), text)
                self.assertIn("Only 'query sequences --scoring esm-codes' fills rawrev, q90, rawadj, ecal", text)
                help_text = invoke(*path, "--help").output
                for field in output.ESM_CODES_OUTPUT_FIELDS:
                    self.assertIn(field, help_text)
        self.assertEqual(output.DEFAULT_OUTPUT_FIELDS[-2:], ("evalue", "bits"))  # unchanged default columns


class ResolveOptionsTest(unittest.TestCase):
    """:func:`search.resolve_query_stage2_options` (P6.1, D8, D20)."""

    def test_blosum62_defaults_are_the_old_ones(self):
        o = search.resolve_query_stage2_options("blosum62")
        self.assertEqual((o.scoring, o.min_seq_identity, o.min_coverage, o.max_evalue, o.gap_open, o.gap_extend,
                          o.comp_bias_corr, o.significance_mode),
                         (ScoringChoice.blosum62, 0.3, 0.8, 1e-3, 11, 1, False, SignificanceMode.default))
        self.assertEqual(search.resolve_query_stage2_options(), o)

    def test_esm_codes_defaults(self):
        o = search.resolve_query_stage2_options("esm-codes")
        self.assertEqual((o.scoring, o.min_seq_identity, o.min_coverage, o.max_evalue),
                         (ScoringChoice.esm_codes, 0.0, 0.0, 1e-3))
        self.assertEqual((o.gap_open, o.gap_extend, o.comp_bias_corr, o.significance_mode), (None,) * 4)

    def test_explicit_values_are_honoured(self):
        for scoring in ("blosum62", "esm-codes"):
            with self.subTest(scoring=scoring):
                o = search.resolve_query_stage2_options(scoring, min_seq_identity=0.5, min_coverage=0.25,
                                                        max_evalue=float("inf"), seq_identity=True)
                self.assertEqual((o.min_seq_identity, o.min_coverage, o.max_evalue), (0.5, 0.25, float("inf")))
        o = search.resolve_query_stage2_options("blosum62", gap_open=10, gap_extend=2, comp_bias_corr=True,
                                                significance_mode="sampled", min_seq_identity=0.0)
        self.assertEqual((o.gap_open, o.gap_extend, o.comp_bias_corr, o.significance_mode, o.min_seq_identity),
                         (10, 2, True, SignificanceMode.sampled, 0.0))

    def test_blosum62_options_are_refused_under_esm_codes(self):
        import typer
        cases = {"--gap-open": dict(gap_open=11), "--gap-extend": dict(gap_extend=1),
                 "--comp-bias-corr/--no-comp-bias-corr": dict(comp_bias_corr=False),
                 "--significance-mode": dict(significance_mode=SignificanceMode.default),
                 "--no-seq-identity": dict(seq_identity=False)}
        for flag, kwargs in cases.items():
            with self.subTest(flag=flag), self.assertRaises(typer.BadParameter) as ctx:
                search.resolve_query_stage2_options("esm-codes", **kwargs)
            self.assertIn(flag, str(ctx.exception))
        with self.assertRaises(typer.BadParameter) as ctx:
            search.resolve_query_stage2_options("esm-codes", gap_open=5, significance_mode="sampled",
                                                seq_identity=False)
        self.assertIn("--gap-open, --significance-mode are BLOSUM62 options", str(ctx.exception))
        self.assertIn("--no-seq-identity", str(ctx.exception))

    def test_esm_codes_filter_values_are_checked(self):
        import typer
        for kwargs in (dict(min_seq_identity=1.5), dict(min_coverage=-0.1), dict(min_coverage=float("nan")),
                       dict(max_evalue=-1.0), dict(max_evalue=float("nan"))):
            with self.subTest(**kwargs), self.assertRaises(typer.BadParameter):
                search.resolve_query_stage2_options("esm-codes", **kwargs)
        # blosum62 keeps today's behaviour: no check here.
        search.resolve_query_stage2_options("blosum62", min_seq_identity=1.5, max_evalue=-1.0)


class RefusedOptionsCliTest(_Case):
    """Each refused option exits 2 before anything is read: the database path does not even exist."""

    def run_esm(self, *extra):
        with _esm3_spy() as esm3, _no_pass() as predict, \
                mock.patch.object(search, "resolve_esm_codes_database",
                                  side_effect=AssertionError("the database was read")) as resolve:
            result = invoke(*_query_args(self.tmp / "missing" / "db", _FASTA, self.qshards, self.tmp / "o.tsv",
                                         *_esm(*extra)))
        esm3.assert_not_called()
        predict.assert_not_called()
        resolve.assert_not_called()
        return result

    def test_each_refused_option(self):
        for extra, flag in ((("--gap-open", "11"), "--gap-open"), (("--gap-extend", "2"), "--gap-extend"),
                            (("--comp-bias-corr",), "--comp-bias-corr"), (("--no-comp-bias-corr",), "--comp-bias-corr"),
                            (("--significance-mode", "sampled"), "--significance-mode"),
                            (("--no-seq-identity",), "--no-seq-identity")):
            with self.subTest(extra=extra):
                self.usage_error(self.run_esm(*extra), flag, "--scoring")

    def test_option_values(self):
        for extra, text in ((("--min-seq-identity", "2"), "--min-seq-identity"),
                            (("--min-coverage", "-0.5"), "--min-coverage"), (("--max-evalue", "-1"), "--max-evalue"),
                            (("--align-workers", "-1"), "--align-workers"),
                            (("--format-output", "query,target,not_a_field"), "not_a_field")):
            with self.subTest(extra=extra):
                self.usage_error(self.run_esm(*extra), text)

    def test_seq_identity_on_is_accepted(self):
        """--seq-identity (Stage 2 on) is what esm-codes does anyway: it reaches the database resolution."""
        with mock.patch.object(search, "resolve_esm_codes_database",
                               side_effect=RefusalError("stop here")) as resolve, _esm3_spy() as esm3:
            result = invoke(*_query_args(self.tmp / "db", _FASTA, self.qshards, self.tmp / "o.tsv",
                                         *_esm("--seq-identity")))
        self.assertEqual(result.exit_code, 1)
        resolve.assert_called_once()
        esm3.assert_not_called()


# =====================================================================================================================
# P6.2: resolution before ESM3, every refusal on every rank alike
# =====================================================================================================================
class RefusalTest(_DbCase):

    def run_esm(self, db, *extra, fasta=_FASTA, rank_env=None, prepare_ok=True):
        """An esm-codes query that must be refused before ESM3 (and before the pass); ``prepare_ok=False``: before
        the bundle is prepared (step 5) too."""
        prepare = mock.patch.object(ss, "prepare", wraps=ss.prepare) if prepare_ok else \
            mock.patch.object(ss, "prepare", side_effect=AssertionError("bundle prepared"))
        with launch_env(**(rank_env or {})), _esm3_spy() as esm3, _no_pass() as predict, prepare as prep:
            result = invoke(*_query_args(db, fasta, self.qshards, self.tmp / "o.tsv", *_esm(*extra)))
        esm3.assert_not_called()
        predict.assert_not_called()
        if not prepare_ok:
            prep.assert_not_called()
        self.assertEqual(list(self.qshards.iterdir()), [])
        self.assertFalse((self.tmp / "o.tsv").exists())
        return result

    def _ranks(self):
        return ({}, {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"})

    def test_no_sequence_store(self):
        db = self.database("plain")
        Path(f"{db}.sequences").unlink()
        for rank_env in self._ranks():
            with self.subTest(rank_env=rank_env):
                message = self.refused(self.run_esm(db, rank_env=rank_env, prepare_ok=False))
                self.assertIn("has no sequence store", message)
                self.assertIn("build sequences --residue-codes", message)
        self.assertFalse(Path(f"{db}.sequences").exists())  # nothing was created by the check

    def test_built_without_residue_codes(self):
        db = self.database("plain")
        for rank_env in self._ranks():
            with self.subTest(rank_env=rank_env):
                message = self.refused(self.run_esm(db, rank_env=rank_env, prepare_ok=False), FileNotFoundError)
                self.assertIn("was built without --residue-codes", message)
                self.assertIn("--scoring blosum62 still works", message)
                self.assertNotIn("partial", message)

    def test_partial_build(self):
        db = self.database("codes")
        Path(f"{db}.rescodes.json").unlink()
        for rank_env in self._ranks():
            with self.subTest(rank_env=rank_env):
                message = self.refused(self.run_esm(db, rank_env=rank_env, prepare_ok=False), ResidueCodeStoreError)
                self.assertIn("are a partial build", message)
                self.assertNotIn("built without", message)
                self.assertIn("--from-shards", message)

    def test_unreadable_manifest(self):
        db = self.database("codes")
        Path(f"{db}.rescodes.json").write_text("{not json")
        message = self.refused(self.run_esm(db, prepare_ok=False), IncompatibleCodeStoreError)
        self.assertIn("not readable JSON", message)

    def test_unknown_system(self):
        db = self.database("codes")
        _edit_manifest(db, lambda m: m.update(system_name="esm-codes-v99"))
        for rank_env in self._ranks():
            with self.subTest(rank_env=rank_env):
                message = self.refused(self.run_esm(db, rank_env=rank_env, prepare_ok=False),
                                       ss.UnknownScoringSystemError)
                self.assertIn("'esm-codes-v99'", message)
                self.assertIn("database was built by a newer FoldMatch; upgrade", message)

    def test_a_finite_max_evalue_needs_residues(self):
        """Step 6: a finite --max-evalue needs a database with residues (N of the E-values): refused on an empty
        database, which the commands cannot build (written here through the stores' API); --max-evalue inf passes."""
        from foldmatch.search.sequence_store import SequenceStore
        folder = self.tmp / "empty"
        folder.mkdir()
        sequence_store = SequenceStore(folder, "db")
        sequence_store.create(_write(folder / "empty.fasta", ""))
        store = ResidueCodeStore(folder, "db")
        store.create([([], [], [])], SYSTEM)
        store.validate(sequence_store, 0, preflight=None)
        db = folder / "db"
        self.assertEqual(store.manifest()["total_residues"], 0)
        message = self.refused(self.run_esm(db), ValueError)
        self.assertIn("needs a non-empty database", message)
        self.assertEqual(search.resolve_esm_codes_database(str(db), float("inf")).total_residues, 0)

    def test_incompatible_stores(self):
        """check_compatible against the system and the sequence store meta (step 3), each field on its own."""
        cases = {
            "manifest version": (lambda m: m.update(manifest_version=2), "unknown manifest_version 2"),
            "encoding digest": (lambda m: m.update(encoding_digest="0" * 16), "encoding_digest"),
            "ESM3 weights": (lambda m: m["esm"].update(weights_sha256="0" * 64), "built with ESM3 weights " + "0" * 64),
            "entries": (lambda m: m.update(entries=3), "3 entries, the sequence store 2"),
            "residues": (lambda m: m.update(total_residues=304), "304 residues, the sequence store 303"),
        }
        for case, (edit, text) in cases.items():
            with self.subTest(case=case):
                shutil.rmtree(self.tmp / "codes", ignore_errors=True)
                db = self.database("codes")
                _edit_manifest(db, edit)
                message = self.refused(self.run_esm(db, prepare_ok=False), IncompatibleCodeStoreError)
                self.assertIn(text, message)

    def test_missing_kernel(self):
        db = self.database("codes")
        for rank_env in self._ranks():
            with self.subTest(rank_env=rank_env), mock.patch.dict(swkern._KERNELS, clear=True), \
                    mock.patch("foldmatch._native.find_library", return_value=None):
                message = self.refused(self.run_esm(db, rank_env=rank_env, prepare_ok=False), swkern.KernelError)
                self.assertIn("native Smith-Waterman kernel", message)
                self.assertIn("python -m foldmatch._native.build", message)  # the rebuild command

    def test_a_bundle_problem(self):
        db = self.database("codes")
        with _esm3_spy() as esm3, _no_pass(), mock.patch.object(
                ss, "prepare", side_effect=ss.BundleUnavailableError("not cached")) as prepare:
            result = invoke(*_query_args(db, _FASTA, self.qshards, self.tmp / "o.tsv", *_esm()))
        self.assertIn("not cached", self.refused(result, ss.BundleUnavailableError))
        prepare.assert_called_once()
        esm3.assert_not_called()

    def test_one_fetch_then_cached_only(self):
        """The process that starts the run may download the bundle; a rank Lightning starts, or a multi-node run,
        finds it cached; the pass gets the prepared registry system and writes query encodings."""
        db = self.database("codes")
        stop = RuntimeError("stop at the pass")
        for rank_env, extra, local_only in (({}, (), False), ({"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}, (), True),
                                            ({}, ("--num-nodes", "2"), True)):
            with self.subTest(rank_env=rank_env, extra=extra), launch_env(**rank_env), \
                    mock.patch.object(ss, "prepare", wraps=ss.prepare) as prepare, \
                    mock.patch("foldmatch.search.embedding_computer.chain_predict", side_effect=stop) as predict:
                result = invoke(*_query_args(db, _FASTA, self.qshards, self.tmp / "o.tsv", *_esm(*extra)))
                self.assertIs(result.exception.__cause__ or result.exception, stop)
                prepare.assert_called_once_with(SYSTEM, local_files_only=local_only)
                kwargs = predict.call_args.kwargs
                self.assertEqual(kwargs["residue_output"], "query_encoding")
                self.assertIs(kwargs["scoring_system"].system, SYSTEM)

    def test_query_fasta_the_method_cannot_take(self):
        """Duplicate ids, '<' / '>' and non-ASCII sequences: refused by the strict query pre-flight, before the
        database is read; a blosum62 query of the same file is not strict."""
        db = self.database("codes")
        for text, what in ((">a\nACDE\n>b\nKLM\n>a\nACDE\n", "repeat an earlier id"),
                           (">a\nAC<mask>DE\n", "'<' or '>'"), (">a\nACDE>\n", "'<' or '>'"),
                           (">a\nACDÉ\n", "non-ASCII")):
            for rank_env in self._ranks():
                with self.subTest(what=what, rank_env=rank_env), \
                        mock.patch.object(search, "resolve_esm_codes_database",
                                          side_effect=AssertionError("the database was read")):
                    fasta = _write(self.tmp / "bad.fasta", text)
                    self.assertIn(what, self.refused(self.run_esm(db, fasta=fasta, rank_env=rank_env,
                                                                  prepare_ok=False), ValueError))
        fasta = _write(self.tmp / "dup.fasta", ">a\nACDE\n>a\nACDE\n")
        with _esm3_spy() as esm3:
            result = invoke(*_query_args(db, fasta, self.qshards, self.tmp / "o.tsv"))
        self.assertIsInstance(result.exception, AssertionError)  # the blosum62 pass was reached
        esm3.assert_called_once()

    def test_query_encodings_of_an_earlier_run_in_the_way(self):
        db = self.database("codes")
        with launch_env(FOLDMATCH_RUN_ID="same"):
            (self.qshards / "query_encodings_same.f16").touch()
            with _esm3_spy() as esm3, _no_pass():
                result = invoke(*_query_args(db, _FASTA, self.qshards, self.tmp / "o.tsv", *_esm()))
        self.assertIn("query_encodings_same.f16", self.refused(result, FileExistsError))
        esm3.assert_not_called()
        self.assertEqual([p.name for p in self.qshards.iterdir()], ["query_encodings_same.f16"])

    def test_shards_of_the_same_run_id(self):
        db = self.database("codes")
        with launch_env(FOLDMATCH_RUN_ID="r"):
            (self.qshards / "emb_r-0.parquet").touch()
            with _esm3_spy() as esm3:
                result = invoke(*_query_args(db, _FASTA, self.qshards, self.tmp / "o.tsv", *_esm()))
        self.assertIn("same run id", self.refused(result, FileExistsError))
        esm3.assert_not_called()
        self.assertEqual([p.name for p in self.qshards.iterdir()], ["emb_r-0.parquet"])  # not this run's: kept

    def test_the_faiss_count_after_the_index_loads(self):
        """Rank 0 compares the live FAISS count with the code store once the index is loaded (after the pass); the
        run's shards and query encodings are removed all the same."""
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        load = FaissEmbeddingDatabase.load_database

        def one_more(self_, *args, **kwargs):
            out = load(self_, *args, **kwargs)
            self_.chain_ids = list(self_.chain_ids) + ["phantom"]
            return out

        db = self.database("codes")
        with mock.patch.object(FaissEmbeddingDatabase, "load_database", one_more):
            result, out = self.query(db, *_esm())
        message = self.refused(result, IncompatibleCodeStoreError)
        self.assertIn("has 2 entries, the FAISS index 3", message)
        self.assertEqual(list(self.qshards.iterdir()), [])
        self.assertFalse(out.exists())

    def test_exit_status_of_a_refusal(self):
        """fm-search prints a refusal on one line and exits 1 (D20)."""
        db = self.database("plain")
        argv = ["fm-search", *map(str, _query_args(db, _FASTA, self.qshards, self.tmp / "o.tsv", *_esm()))]
        stderr = self.enterContext(mock.patch("sys.stderr", new_callable=io.StringIO))
        root = logging.getLogger()
        handlers = root.handlers[:]
        try:
            with mock.patch.object(sys, "argv", argv), _esm3_spy(), self.assertRaises(SystemExit) as ctx:
                search.main()
        finally:
            root.handlers[:] = handlers
        self.assertEqual(ctx.exception.code, 1)
        errors = [line for line in stderr.getvalue().splitlines() if line.startswith("Error: ")]
        self.assertEqual(len(errors), 1, stderr.getvalue())
        self.assertIn("was built without --residue-codes", errors[0])


# =====================================================================================================================
# P6.1-P6.3: the query (fake ESM3 pass, real bundle and kernel)
# =====================================================================================================================
class QueryTest(_DbCase):

    def assert_self_hits(self, rows):
        lengths = {name: len(seq) for name, seq in _fasta_records(_FASTA)}
        by_query = {}
        for row in rows:
            by_query.setdefault(row["query"], []).append(row)
        self.assertEqual(set(by_query), set(lengths))
        for qid, hits in by_query.items():
            top = hits[0]
            self.assertEqual(top["target"], qid)
            self.assertEqual((top["qstart"], top["qend"]), (top["tstart"], top["tend"]))   # main diagonal
            self.assertEqual(top["gapopen"], "0")
            self.assertEqual(int(top["alnlen"]), int(top["qend"]) - int(top["qstart"]) + 1)
            self.assertGreaterEqual(float(top["fident"]), 0.99)
            self.assertGreaterEqual(int(top["alnlen"]) / lengths[qid], 0.9)                 # qcov = tcov here
            self.assertLessEqual(float(top["evalue"]), 1e-3)

    def test_esm_codes_query(self):
        """Default columns, no empty cell; each query's top hit is itself on the diagonal; Stage 2 got the defaults
        of D8, N = the manifest's 303 residues, the run's query encodings; BLOSUM62 was not used."""
        db = self.database("codes")
        calls = []
        with mock.patch.object(code_alignment, "align_candidates_esm_codes",
                               wraps=code_alignment.align_candidates_esm_codes) as esm, \
                mock.patch("foldmatch.search.alignment.align_candidates",
                           side_effect=AssertionError("BLOSUM62 Stage 2")) as blosum:
            result, out = self.query(db, *_esm("--align-workers", "0"), calls=calls)
        self.assertEqual(result.exit_code, 0, (result.output, result.exception))
        blosum.assert_not_called()
        esm.assert_called_once()
        args, kwargs = esm.call_args
        query_index, prefilter, code_store, _fetch, prepared, db_residues, _pool = args
        self.assertEqual(db_residues, 303)
        self.assertEqual((code_store.base, prepared.system), (db, SYSTEM))
        self.assertEqual((kwargs["min_seq_identity"], kwargs["min_coverage"], kwargs["max_evalue"]), (0.0, 0.0, 1e-3))
        self.assertEqual(kwargs["output_needs"], list(DEFAULT_FIELDS))
        self.assertEqual(kwargs["query_sequences"], dict(_fasta_records(_FASTA)))
        self.assertEqual(sorted(query_index.entries), ["1acb_E", "2uzi_A"])
        self.assertEqual(query_index.path.name, "query_encodings_q.f16")
        self.assertEqual(calls[0]["residue_output"], "query_encoding")
        rows = _rows(out)
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(all(row.values()), row)
        self.assert_self_hits(rows)
        # The run's shards and its query encodings are gone.
        self.assertEqual(list(self.qshards.iterdir()), [])

    def test_blosum62_is_unchanged(self):
        """Without --scoring: the BLOSUM62 Stage 2 with its old defaults; on a database with codes it writes the
        same bytes as on one without; the esm-codes fields render empty."""
        outs = {}
        for kind in ("codes", "plain"):
            db = self.database(kind)
            with mock.patch("foldmatch.search.alignment.align_candidates",
                            wraps=__import__("foldmatch.search.alignment", fromlist=["x"]).align_candidates) as blosum, \
                    mock.patch.object(code_alignment, "align_candidates_esm_codes",
                                      side_effect=AssertionError("esm-codes Stage 2")):
                result, outs[kind] = self.query(db, out=self.tmp / f"{kind}.tsv", run_id=f"b{kind}")
            self.assertEqual(result.exit_code, 0, (result.output, result.exception))
            kwargs = blosum.call_args.kwargs
            self.assertEqual({k: kwargs[k] for k in ("min_seq_identity", "min_coverage", "max_evalue", "gap_open",
                                                       "gap_extend", "comp_bias_corr", "significance_mode")},
                             dict(min_seq_identity=0.3, min_coverage=0.8, max_evalue=1e-3, gap_open=11, gap_extend=1,
                                  comp_bias_corr=False, significance_mode=SignificanceMode.default))
        self.assertEqual(_sha256(outs["codes"]), _sha256(outs["plain"]))
        self.assertTrue(_rows(outs["codes"]))
        result, out = self.query(self.database("codes"), "--format-output", ALL_FIELDS, run_id="ball")
        self.assertEqual(result.exit_code, 0, result.output)
        rows = _rows(out, output.SUPPORTED_OUTPUT_FIELDS)
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual([row[f] for f in output.ESM_CODES_OUTPUT_FIELDS], [""] * 4)
            self.assertNotIn(".", row["raw"])   # the integer BLOSUM62 score

    def test_explicit_values_are_honoured(self):
        db = self.database("codes")
        with mock.patch.object(code_alignment, "align_candidates_esm_codes",
                               wraps=code_alignment.align_candidates_esm_codes) as esm:
            result, _ = self.query(db, *_esm("--min-seq-identity", "0.5", "--min-coverage", "0.7", "--max-evalue",
                                             "10", "--align-workers", "0"))
        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = esm.call_args.kwargs
        self.assertEqual((kwargs["min_seq_identity"], kwargs["min_coverage"], kwargs["max_evalue"]), (0.5, 0.7, 10.0))

    def test_every_field_and_the_header(self):
        db = self.database("codes")
        result, out = self.query(db, *_esm("--format-output", ALL_FIELDS, "--format-mode", "tsv", "--max-evalue",
                                           "inf", "--threshold", "-2", "--align-workers", "0"))
        self.assertEqual(result.exit_code, 0, result.output)
        lines = out.read_text().splitlines()
        self.assertEqual(lines[0], ALL_FIELDS.replace(",", "\t"))
        rows = [dict(zip(output.SUPPORTED_OUTPUT_FIELDS, line.split("\t"))) for line in lines[1:]]
        self.assertEqual(len(rows), 4)  # every candidate traced back, each pair aligned
        for row in rows:
            self.assertTrue(all(row.values()), row)
            for field in NUMERIC_FIELDS:
                self.assertTrue(math.isfinite(float(row[field])), (field, row[field]))
            self.assertEqual(row["ecal"], "1")
            raw, rawrev, q90, rawadj = (float(row[f]) for f in ("raw", "rawrev", "q90", "rawadj"))
            self.assertAlmostEqual(rawadj, raw - rawrev - q90, delta=2e-3)
            self.assertEqual(len(row["qaln"]), int(row["alnlen"]))

    def test_serial_and_pool_give_the_same_bytes(self):
        """S16 on one machine: the default worker pool (spawn) and the in-process run write identical files."""
        db = self.database("codes")
        outs = []
        for k, extra in enumerate((("--align-workers", "0"), (), ("--align-workers", "3"))):
            result, out = self.query(db, *_esm("--format-output", ALL_FIELDS, "--max-evalue", "inf", *extra),
                                     out=self.tmp / f"o{k}.tsv", run_id=f"p{k}")
            self.assertEqual(result.exit_code, 0, result.output)
            outs.append(_sha256(out))
        self.assertEqual(len(set(outs)), 1)

    def test_queries_without_candidates_are_skipped_before_the_panel(self):
        db = self.database("codes")
        stats = {}
        real = code_alignment.align_candidates_esm_codes

        def align(*args, **kwargs):
            return real(*args, stats=stats, **kwargs)

        with mock.patch.object(code_alignment, "align_candidates_esm_codes", align):
            result, out = self.query(db, *_esm("--threshold", "2", "--align-workers", "0"))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual((stats["queries"], stats["queries_without_candidates"], stats["panel_pairs"],
                          stats["round1_tasks"]), (2, 2, 0, 0))
        self.assertEqual(out.read_text(), "")

    def test_queries_outside_the_calibrated_domain(self):
        """A 9-aa query: aligned, its hits reported with ecal=0, and one warning gives the counts (S13)."""
        db = self.database("codes")
        fasta = _write(self.tmp / "short.fasta", _FASTA.read_text() + ">short\nACDEFGHIK\n")
        with self.assertLogs("foldmatch.search.code_alignment", "WARNING") as logs:
            result, out = self.query(db, *_esm("--format-output", "query,target,ecal,evalue", "--max-evalue",
                                               "inf", "--threshold", "-2", "--align-workers", "0"), fasta=fasta)
        self.assertEqual(result.exit_code, 0, result.output)
        rows = _rows(out, ("query", "target", "ecal", "evalue"))
        short = [r for r in rows if r["query"] == "short"]
        self.assertTrue(short)
        self.assertEqual({r["ecal"] for r in short}, {"0"})
        self.assertEqual({r["ecal"] for r in rows if r["query"] != "short"}, {"1"})
        warnings = [r.getMessage() for r in logs.records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 queries shorter than 21 aa", warnings[0])

    def test_an_encoding_that_does_not_match_its_query_is_an_error(self):
        """A missing encoding (empty payload), a length or a seq_sha1 that differs from the query's: an error after
        the pass (exit 1, not a refusal), with the run's files removed."""
        db = self.database("codes")

        def short_payload(name, sequence):
            q = _fake_q(name, sequence)
            return q[:-1] if name == "2uzi_A" else q

        def empty_payload(name, sequence):
            return np.empty((0, 64), np.float16) if name == "2uzi_A" else _fake_q(name, sequence)

        cases = {
            "length": (fake_query_pass(payload=short_payload), "its sequence has 58 residues, its encoding 57 rows"),
            "empty": (fake_query_pass(payload=empty_payload), "without an encoding"),
            "seq_sha1": (fake_query_pass(seq_sha1=lambda name, s: _sha1(s + "A") if name == "1acb_E" else _sha1(s)),
                         "seq_sha1 differs"),
        }
        for case, (predict, text) in cases.items():
            with self.subTest(case=case):
                result, out = self.query(db, *_esm("--align-workers", "0"), predict=predict, run_id=f"e{case}")
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertNotIsInstance(result.exception, RefusalError)
                self.assertIsInstance(result.exception, ValueError)
                self.assertIn(text, str(result.exception))
                self.assertEqual(list(self.qshards.iterdir()), [])
                self.assertFalse(out.exists())

    def test_the_run_files_are_removed_after_a_stage_2_failure(self):
        db = self.database("codes")
        with mock.patch.object(code_alignment, "align_candidates_esm_codes",
                               side_effect=RuntimeError("stage 2 failed")):
            result, _ = self.query(db, *_esm())
        self.assertEqual(str(result.exception), "stage 2 failed")
        self.assertEqual(list(self.qshards.iterdir()), [])

    def test_a_failed_pass_deletes_the_shard_it_wrote(self):
        """The query shard holds the query encodings (128 B per residue): a process whose pass fails deletes the shard
        it wrote, and the error goes on (not a refusal)."""
        db = self.database("codes")
        write = fake_query_pass()

        def failing(**kwargs):
            write(**kwargs)
            raise RuntimeError("the ESM3 pass failed after writing its shard")

        result, out = self.query(db, *_esm(), predict=failing, run_id="f")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertEqual(str(result.exception), "the ESM3 pass failed after writing its shard")
        self.assertNotIsInstance(result.exception, RefusalError)
        self.assertEqual(list(self.qshards.iterdir()), [])
        self.assertFalse(out.exists())

    def test_a_rank_other_than_0_leaves_after_the_pass(self):
        """Only rank 0 searches, aligns and writes: a rank > 0 (LOCAL_RANK=1; its process group simulated through
        _is_rank_zero) returns after the pass without loading the index, running Stage 2, writing the output or
        deleting the run's shards, which rank 0 reads."""
        db = self.database("codes")
        out = self.tmp / "out.tsv"
        from foldmatch.search.embedding_database import EmbeddingDatabase
        with launch_env(LOCAL_RANK=1, FOLDMATCH_RUN_ID="r1"), \
                mock.patch.object(search, "_is_rank_zero", return_value=False), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_query_pass()), \
                mock.patch.object(EmbeddingDatabase, "load", side_effect=AssertionError("index loaded")) as load, \
                mock.patch.object(code_alignment, "align_candidates_esm_codes",
                                  side_effect=AssertionError("Stage 2 ran")) as esm:
            result = invoke(*_query_args(db, _FASTA, self.qshards, out, *_esm()))
        self.assertEqual(result.exit_code, 0, (result.output, result.exception))
        load.assert_not_called()
        esm.assert_not_called()
        self.assertFalse(out.exists())
        self.assertEqual([p.name for p in self.qshards.iterdir()], ["emb_r1-0.parquet"])


# =====================================================================================================================
# real ESM3 (CPU): the end-to-end smoke and the F-deg items through the command line
# =====================================================================================================================
def _track_models():
    """Patches recording a weak reference to every ESM3 model and aggregator the pass builds, and whether they are
    still alive when Stage 2 starts. Returns ``(patches, refs, alive_at_stage_2)``."""
    from foldmatch.inference import chain_complete_inference as cci
    refs, alive = [], []

    def track(fn):
        def build(*args, **kwargs):
            model = fn(*args, **kwargs)
            refs.append(weakref.ref(model))
            return model
        return build

    real_align = code_alignment.align_candidates_esm_codes

    def align(*args, **kwargs):
        alive.append([ref() is not None for ref in refs])
        return real_align(*args, **kwargs)

    patches = (mock.patch.object(cci, "get_residue_model", track(cci.get_residue_model)),
               mock.patch.object(cci, "get_aggregator_model", track(cci.get_aggregator_model)),
               mock.patch.object(code_alignment, "align_candidates_esm_codes", align))
    return patches, refs, alive


@pytest.mark.usefixtures("fmsw_libs")
class RealEsm3SmokeTest(unittest.TestCase):
    """The plan's end-to-end smoke (CPU, the 2-sequence fixture, ESM3 from the cache): build with --residue-codes, then
    query the same FASTA with --scoring esm-codes --threshold 0; a blosum62 query on that database writes the same
    bytes as on a database built without codes. The commands' wall times are printed (acceptance: the smoke within
    2x the current query sequences test)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        esm_codes_bundle.registry_files()
        swkern.load()
        gc.collect()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.results, cls.seconds = {}, {}
        patches, cls.model_refs, cls.alive = _track_models()
        steps = (
            ("build codes", _build_args(_FASTA, cls.tmp / "codes" / "db", cls.tmp / "bshards", "--residue-codes")),
            ("query esm-codes", _query_args(cls.tmp / "codes" / "db", _FASTA, cls.tmp / "qshards",
                                            cls.tmp / "esm.tsv", *_esm("--threshold", "0"))),
            ("query blosum62 codes db", _query_args(cls.tmp / "codes" / "db", _FASTA, cls.tmp / "qshards",
                                                    cls.tmp / "blosum_codes.tsv")),
            ("build plain", _build_args(_FASTA, cls.tmp / "plain" / "db", cls.tmp / "bshards")),
            ("query blosum62 plain db", _query_args(cls.tmp / "plain" / "db", _FASTA, cls.tmp / "qshards",
                                                    cls.tmp / "blosum_plain.tsv")),
        )
        for folder in ("codes", "plain", "bshards", "qshards"):
            (cls.tmp / folder).mkdir()
        with launch_env(), patches[0], patches[1], patches[2]:
            for k, (name, args) in enumerate(steps):
                os.environ["FOLDMATCH_RUN_ID"] = f"smoke{k}"
                t0 = time.perf_counter()
                cls.results[name] = invoke(*args)
                cls.seconds[name] = time.perf_counter() - t0
                gc.collect()
        print("\nP6 smoke wall times (s): " + ", ".join(f"{k} {v:.1f}" for k, v in cls.seconds.items()),
              file=sys.stderr)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()

    def setUp(self):
        for name, result in self.results.items():
            self.assertEqual(result.exit_code, 0, f"{name}: {result.output[-4000:]} {result.exception!r}")

    def test_each_query_finds_itself(self):
        rows = _rows(self.tmp / "esm.tsv")
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(all(row.values()), row)          # the default 13 columns, no empty cell
        QueryTest.assert_self_hits(self, rows)
        manifest = ResidueCodeStore(self.tmp / "codes", "db").manifest()
        self.assertEqual((manifest["entries"], manifest["total_residues"]), (2, 303))   # N = 303

    def test_blosum62_on_a_database_with_codes_is_byte_identical(self):
        self.assertEqual(_sha256(self.tmp / "blosum_codes.tsv"), _sha256(self.tmp / "blosum_plain.tsv"))
        self.assertTrue(_rows(self.tmp / "blosum_codes.tsv"))
        for suffix in _DB_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertEqual(_sha256(self.tmp / "codes" / f"db{suffix}"), _sha256(self.tmp / "plain" / f"db{suffix}"))

    def test_the_models_are_freed_before_stage_2(self):
        self.assertEqual(len(self.alive), 1)
        self.assertEqual(len(self.alive[0]), 4)   # ESM3 and the aggregator of the build and of the query pass
        self.assertEqual(self.alive[0], [False] * 4)

    def test_query_files_are_removed(self):
        self.assertEqual(list((self.tmp / "qshards").iterdir()), [])


def _fdeg_items():
    z = np.load(_FDEG, allow_pickle=False)
    return [(str(i), str(s)) for i, s in zip(z["real_ids"], z["real_sequences"])]


#: Two longer chains (F3 queries of 439 and 609 aa) whose best cell against the length-1 target M is clearly positive
#: (S_f 0.27 / 0.26 on the Mac CPU, where 1acb_E's is only 0.016), so the length-1 target is aligned and reported.
_F3_CHAINS = ("Q8X1D8_1", "P17169_1")


def _f3_items(ids):
    z = np.load(_TEST_PATH / "resources" / "esm_codes" / "q90_queries_v1.npz", allow_pickle=False)
    sequences = {str(i): str(s) for i, s in zip(z["ids"], z["sequences"])}
    return [(i, sequences[i]) for i in ids]


@pytest.mark.usefixtures("fmsw_libs")
class RealEsm3DegenerateTest(unittest.TestCase):
    """The real-ESM3 F-deg items through the command line (plan section 8 P6 Tests; D14, S13, S13b): a database of the
    six items, the fixture's two chains and two longer F3 chains, built with --residue-codes, queried with the same
    FASTA under --max-evalue inf. Every output is finite and every hit's ecal is S13's; all-X, poly-A, HHHHHH, GS and
    the 10-aa peptide are aligned and reported; the single-residue query M returns no hits and is counted in the
    warning, as is a synthetic identical-row query (poly-A with its encoding rows made identical); the length-1
    target M is aligned by the long chains and gets ecal=0."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        esm_codes_bundle.registry_files()
        swkern.load()
        gc.collect()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.sequences = dict(_fdeg_items() + _fasta_records(_FASTA) + _f3_items(_F3_CHAINS))
        cls.fasta = cls.tmp / "fdeg.fasta"
        cls.fasta.write_text("".join(f">{name}\n{seq}\n" for name, seq in cls.sequences.items()))
        for folder in ("db", "bshards", "qshards"):
            (cls.tmp / folder).mkdir()
        cls.db = cls.tmp / "db" / "fdeg"
        cls.results, cls.warnings = {}, {}
        from foldmatch.search.embedding_computer import EmbeddingComputer
        real_build = EmbeddingComputer.build_query_encodings

        def identical_rows(self_, dst, system):   # poly-A's encoding rows made identical (a synthetic S13b query)
            index = real_build(self_, dst, system)
            entry = index.entries["polyA50"]
            rows = np.memmap(index.path, dtype=np.float16, mode="r+", shape=(index.rows, index.width))
            rows[entry.offset:entry.offset + entry.length] = rows[entry.offset]
            rows.flush()
            del rows
            return index

        query = ("--scoring", "esm-codes", "--max-evalue", "inf", "--threshold", "-2", "--format-output", ALL_FIELDS)
        with launch_env():
            os.environ["FOLDMATCH_RUN_ID"] = "fdeg-build"
            cls.results["build"] = invoke(*_build_args(cls.fasta, cls.db, cls.tmp / "bshards", "--residue-codes"))
            gc.collect()
            for name, patch in (("real", None), ("identical", identical_rows)):
                os.environ["FOLDMATCH_RUN_ID"] = f"fdeg-{name}"
                handler = _Records()
                logger = logging.getLogger("foldmatch.search.code_alignment")
                logger.addHandler(handler)
                try:
                    with (mock.patch.object(EmbeddingComputer, "build_query_encodings", patch) if patch
                          else mock.patch.dict(os.environ)):
                        cls.results[name] = invoke(*_query_args(cls.db, cls.fasta, cls.tmp / "qshards",
                                                                cls.tmp / f"{name}.tsv", *query))
                finally:
                    logger.removeHandler(handler)
                cls.warnings[name] = [r.getMessage() for r in handler.records if r.levelno == logging.WARNING]
                gc.collect()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()

    def setUp(self):
        for name, result in self.results.items():
            self.assertEqual(result.exit_code, 0, f"{name}: {result.output[-4000:]} {result.exception!r}")

    def rows(self, name):
        return _rows(self.tmp / f"{name}.tsv", output.SUPPORTED_OUTPUT_FIELDS)

    def distinct_codes(self) -> dict:
        store = ResidueCodeStore(self.tmp / "db", "fdeg")
        codes = np.asarray(store.open_memmap())
        return {name: len(np.unique(codes[e.offset:e.offset + e.length], axis=0))
                for name, e in store.lookup(self.sequences).items()}

    def expected_ecal(self, qid, tid, distinct):
        def nonstandard(seq):
            return sum(c not in STANDARD_AMINO_ACIDS for c in seq) / len(seq) > DOMAIN.max_nonstandard_fraction
        q, t = self.sequences[qid], self.sequences[tid]
        inside = (DOMAIN.query_min_length <= len(q) <= DOMAIN.query_max_length and not nonstandard(q)
                  and len(t) <= DOMAIN.target_max_length and not nonstandard(t)
                  and distinct[tid] >= DOMAIN.target_min_distinct_codes)
        return "1" if inside else "0"

    def test_outputs_are_finite_and_ecal_follows_s13(self):
        distinct = self.distinct_codes()
        self.assertEqual(distinct["Met1"], 1)
        for name in ("real", "identical"):
            rows = self.rows(name)
            self.assertTrue(rows)
            for row in rows:
                with self.subTest(run=name, query=row["query"], target=row["target"]):
                    self.assertTrue(all(row.values()), row)
                    for field in NUMERIC_FIELDS:
                        self.assertTrue(math.isfinite(float(row[field])), (field, row[field]))
                    self.assertGreater(int(row["alnlen"]), 0)
                    self.assertEqual(row["ecal"], self.expected_ecal(row["query"], row["target"], distinct))

    def test_d14_items_are_aligned_and_reported(self):
        rows = self.rows("real")
        queries = {row["query"] for row in rows}
        for qid in ("allX30", "polyA50", "His6", "GS", "angiotensinI"):
            self.assertIn(qid, queries)
        for qid in ("allX30", "His6", "GS", "angiotensinI"):   # outside S13 (non-standard, or shorter than 21 aa)
            self.assertEqual({row["ecal"] for row in rows if row["query"] == qid}, {"0"}, qid)
        self.assertIn("1", {row["ecal"] for row in rows if row["query"] == "polyA50"})

    def test_a_length_1_target_gets_ecal_0(self):
        on_met1 = [row for row in self.rows("real") if row["target"] == "Met1"]
        self.assertTrue({row["query"] for row in on_met1} & set(_F3_CHAINS), "no query aligned to the length-1 target")
        self.assertEqual({row["ecal"] for row in on_met1}, {"0"})

    def test_fully_degenerate_queries_report_no_hits(self):
        """S13b, even with --max-evalue inf: M (one residue) in both runs, poly-A when its rows are identical; each
        counted in the run's one warning."""
        self.assertNotIn("Met1", {row["query"] for row in self.rows("real")})
        identical = {row["query"] for row in self.rows("identical")}
        self.assertNotIn("Met1", identical)
        self.assertNotIn("polyA50", identical)
        for name, want in (("real", "1 fully degenerate queries"), ("identical", "2 fully degenerate queries")):
            with self.subTest(run=name):
                self.assertEqual(len(self.warnings[name]), 1, self.warnings[name])
                self.assertIn(want, self.warnings[name][0])
                self.assertIn("Met1", self.warnings[name][0])
        self.assertIn("polyA50", self.warnings["identical"][0])
        # The other queries' hits are those of the real run.
        real = [row for row in self.rows("real") if row["query"] != "polyA50"]
        self.assertEqual([row for row in self.rows("identical")], real)


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


if __name__ == "__main__":
    unittest.main()
