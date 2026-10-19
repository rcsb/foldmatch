"""P6.1-P6.2: ``fm-search query sequences --scoring blosum62|esm-codes`` (plan section 8 P6; D7, D8, D11, D14, D20).

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
        finds it cached."""
        db = self.database("codes")
        for rank_env, extra, local_only in (({}, (), False), ({"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}, (), True),
                                            ({}, ("--num-nodes", "2"), True)):
            with self.subTest(rank_env=rank_env, extra=extra), launch_env(**rank_env), \
                    mock.patch.object(ss, "prepare", wraps=ss.prepare) as prepare, _esm3_spy() as esm3:
                result = invoke(*_query_args(db, _FASTA, self.qshards, self.tmp / "o.tsv", *_esm(*extra)))
                self.assertIn("not implemented yet (plan P6.3)", self.refused(result))
                prepare.assert_called_once_with(SYSTEM, local_files_only=local_only)
                esm3.assert_not_called()

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


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


if __name__ == "__main__":
    unittest.main()
