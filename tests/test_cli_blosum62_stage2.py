"""P9.S6: the BLOSUM62 Stage 2's checks before Stage 1, in ``fm-search query sequences`` and ``fm-search query db``.

Whenever Stage 2 runs, both commands refuse, before the ESM3 pass (``query sequences``) or the FAISS search (``query
db``):

* gap costs the native kernel cannot take (``--gap-open`` below 1, either above 2^20; the sign is ignored), and,
  under ``--significance-mode default``, gap costs other than 11/1 when the run needs significance (an ``evalue`` or
  ``bits`` column, or a finite ``--max-evalue``): usage errors (exit status 2, D20);
* under ``--significance-mode sampled``, a ``--significance-sample-size`` below 2 when the run needs significance
  (typer's ``min=2`` on the command line): a usage error;
* a native kernel that cannot be used: a refusal (exit status 1) with the rebuild command.

The ESM3 pass is the fake one of test_cli_residue_codes (the shards the real pass writes); the database is built from
the fixture FASTA with it once per class. Tests that run Stage 2 use the fixture ``fmsw_libs``; the refusals do not
need the kernel.
"""
import hashlib
import itertools
import math
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import typer

from foldmatch.cli import search
from foldmatch.errors import RefusalError
from foldmatch.search import alignment, output, swint, swkern
from foldmatch.types.api_types import SignificanceMode
from test_cli_residue_codes import _build_args, _esm3_spy, fake_chain_predict, invoke, launch_env

_TEST_PATH = Path(__file__).parent
_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
MAX_GAP = swint.MAX_GAP


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _no_pass():
    """The ESM3 pass must not be reached."""
    return mock.patch("foldmatch.search.embedding_computer.chain_predict",
                      mock.Mock(side_effect=AssertionError("the ESM3 pass was reached")))


def _no_search():
    """FAISS must not load the database (the first step of both commands' search)."""
    return mock.patch("foldmatch.search.embedding_database.EmbeddingDatabase.load",
                      mock.Mock(side_effect=AssertionError("the database was loaded")))


@contextmanager
def _no_kernel():
    """No kernel library can be found, as in an install built without a C compiler."""
    with mock.patch.dict(swkern._KERNELS, clear=True), mock.patch("foldmatch._native.find_library", return_value=None):
        yield


class _Blosum62Case(unittest.TestCase):
    """A database of the fixture FASTA with a sequence store (no residue codes), built once per class with the fake
    pass; a temporary folder per test (``self.tmp``, query shards in ``self.qshards``), no launch variables."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._dbs = tempfile.TemporaryDirectory()
        dbs = Path(cls._dbs.name)
        (dbs / "shards").mkdir()
        with launch_env(FOLDMATCH_RUN_ID="build"), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
            result = invoke(*_build_args(_FASTA, dbs / "db", dbs / "shards"))
        assert result.exit_code == 0, (result.output, result.exception)
        cls.db = dbs / "db"

    @classmethod
    def tearDownClass(cls):
        cls._dbs.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.qshards = self.tmp / "qshards"
        self.qshards.mkdir()
        self.enterContext(launch_env())

    def args(self, command, *extra, out=None):
        out = out or self.tmp / "out.tsv"
        if command == "sequences":
            return ("query", "sequences", "--db-path", self.db, "--fasta-file", _FASTA, "--tmp-embedding-folder",
                    self.qshards, "--output-file", out, "--accelerator", "cpu", "--devices", "1", *extra)
        return ("query", "db", "--query-db-path", self.db, "--subject-db-path", self.db, "--output-file", out, *extra)

    def run_query(self, command, *extra, out=None, run_id="q"):
        """``query sequences`` (fake pass) or ``query db``, to the end; returns ``(result, output path)``."""
        out = out or self.tmp / "out.tsv"
        with launch_env(FOLDMATCH_RUN_ID=run_id), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
            return invoke(*self.args(command, *extra, out=out)), out

    def before_stage1(self, command, *extra, rank_env=None):
        """``command`` with ``extra``, which must stop before Stage 1: neither ESM3 nor the pass is reached, FAISS
        does not load the database, and nothing is written."""
        with launch_env(**(rank_env or {})), _esm3_spy() as esm3, _no_pass() as predict, _no_search() as load:
            result = invoke(*self.args(command, *extra))
        esm3.assert_not_called()
        predict.assert_not_called()
        load.assert_not_called()
        self.assertEqual(list(self.qshards.iterdir()), [])
        self.assertFalse((self.tmp / "out.tsv").exists())
        return result

    def usage_error(self, result, *texts):
        """Exit status 2 (typer.BadParameter) naming every one of ``texts`` (typer's error panel borders dropped)."""
        self.assertEqual(result.exit_code, 2, result.output)
        flat = " ".join(result.output.translate({ord(c): " " for c in "│╭╮╰╯─"}).split())
        for text in texts:
            self.assertIn(text, flat)
        return flat

    def refused(self, result, cause=None):
        """Exit status 1 with a RefusalError (D20); returns its message."""
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIsInstance(result.exception, RefusalError, repr(result.exception))
        if cause is not None:
            self.assertIsInstance(result.exception.__cause__, cause, repr(result.exception.__cause__))
        return str(result.exception)


class RefusedBeforeStage1Test(_Blosum62Case):
    """Each refusal stops the command before Stage 1, under both commands (and on any rank of ``query sequences``)."""

    COMMANDS = ("sequences", "db")

    def test_gap_costs_outside_the_kernel_range(self):
        cases = [  # (options, flag, text)
            (("--gap-open", "0"), "--gap-open", "must be between 1 and 1048576 (the sign is ignored)"),
            (("--gap-open", "0", "--significance-mode", "sampled", "--max-evalue", "inf"), "--gap-open", "got 0"),
            (("--gap-open", str(MAX_GAP + 1), "--significance-mode", "sampled"), "--gap-open",
             f"got {MAX_GAP + 1}"),
            (("--gap-open", str(-MAX_GAP - 1), "--significance-mode", "sampled"), "--gap-open",
             f"got {-MAX_GAP - 1}"),
            (("--gap-extend", str(MAX_GAP + 1), "--significance-mode", "sampled"), "--gap-extend",
             "must be between 0 and 1048576"),
            (("--gap-extend", str(-MAX_GAP - 1), "--max-evalue", "inf", "--format-output", "query,target"),
             "--gap-extend", "The default is 1."),
        ]
        for command in self.COMMANDS:
            for extra, flag, text in cases:
                with self.subTest(command=command, extra=extra):
                    self.usage_error(self.before_stage1(command, *extra), flag, text, "Smith-Waterman kernel")

    def test_calibrated_significance_needs_11_1(self):
        """Under --significance-mode default, any run that needs significance: the message names what needs it and
        the ways out that apply."""
        use = "Use --gap-open 11 --gap-extend 1; or --significance-mode sampled"
        cases = [  # (options, texts)
            (("--gap-open", "10"),       # the default columns hold evalue and bits; the default --max-evalue is 1e-3
             ("the evalue and bits columns of --format-output and the E-value filter --max-evalue 0.001",
              "gap costs given are 10/1", use, "or --max-evalue inf and a --format-output without evalue and bits")),
            (("--gap-open", "10", "--max-evalue", "inf"),
             ("for the evalue and bits columns of --format-output;", use,
              "or a --format-output without evalue and bits, to run")),
            (("--gap-extend", "2", "--format-output", "query,target,fident"),
             ("for the E-value filter --max-evalue 0.001;", "given are 11/2", use, "or --max-evalue inf, to run")),
            (("--gap-open", "5", "--gap-extend", "2", "--max-evalue", "inf", "--format-output", "query,bits"),
             ("for the bits column of --format-output;", "given are 5/2")),
            (("--gap-open", "-10", "--gap-extend", "-1", "--significance-mode", "default"), ("given are 10/1",)),
        ]
        for command in self.COMMANDS:
            for extra, texts in cases:
                with self.subTest(command=command, extra=extra):
                    self.usage_error(self.before_stage1(command, *extra), "'--gap-open' / '--gap-extend'",
                                     "--significance-mode default has calibrated statistics only for the gap costs "
                                     "11/1", *texts)

    def test_a_significance_sample_below_2(self):
        """--significance-sample-size of 1 or less gave infinite or NaN sampled statistics, or failed after Stage 1
        (a negative size); the option now has min=2."""
        for command in self.COMMANDS:
            for size in ("1", "0", "-5"):
                with self.subTest(command=command, size=size):
                    self.usage_error(self.before_stage1(command, "--significance-mode", "sampled",
                                                        "--significance-sample-size", size),
                                     "--significance-sample-size", f"{size} is not in the range x>=2")

    def test_every_rank_refuses_alike(self):
        for rank_env in ({}, {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}):
            with self.subTest(rank_env=rank_env):
                self.usage_error(self.before_stage1("sequences", "--gap-open", "0", rank_env=rank_env), "--gap-open")
                self.usage_error(self.before_stage1("sequences", "--gap-open", "10", rank_env=rank_env),
                                 "--significance-mode default")

    def test_a_kernel_that_cannot_be_used(self):
        """Stage 2 has no other aligner: an install without the kernel is refused before Stage 1, with the rebuild
        command, as --scoring esm-codes is."""
        for command, rank_env in (("sequences", {}), ("sequences", {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}),
                                  ("db", {})):
            with self.subTest(command=command, rank_env=rank_env):
                with _no_kernel():
                    message = self.refused(self.before_stage1(command, rank_env=rank_env), swkern.KernelError)
                self.assertIn("Stage 2 (sequence identity) needs the native Smith-Waterman kernel", message)
                self.assertIn("python -m foldmatch._native.build", message)        # the rebuild command
                self.assertIn("--no-seq-identity runs an embedding-only search", message)

    def test_the_option_checks_come_first(self):
        """A usage error (exit status 2) is reported before a refusal of the installation (exit status 1)."""
        for command in self.COMMANDS:
            with self.subTest(command=command), _no_kernel():
                self.usage_error(self.before_stage1(command, "--gap-open", "0"), "--gap-open")

    def test_query_db_checks_format_output_before_the_search(self):
        """query db now parses --format-output before its search whenever Stage 2 runs, as query sequences does."""
        for extra, text in ((("--format-output", "query,target,not_a_field"), "not_a_field"),
                            (("--format-output", ","), "Empty --format-output")):
            with self.subTest(extra=extra):
                self.usage_error(self.before_stage1("db", *extra), text, "--format-output")


@pytest.mark.usefixtures("fmsw_libs")
class AcceptedTest(_Blosum62Case):
    """The combinations the checks let through run Stage 2 to the end, with the options given."""

    def run_stage2(self, command, *extra, out=None, run_id="q"):
        with mock.patch.object(alignment, "align_candidates", wraps=alignment.align_candidates) as blosum:
            result, out = self.run_query(command, *extra, out=out, run_id=run_id)
        self.assertEqual(result.exit_code, 0, (result.output, result.exception))
        blosum.assert_called_once()
        self.assertTrue(out.read_text().splitlines())
        return blosum.call_args.kwargs, out

    def test_other_gap_costs_without_calibrated_significance(self):
        cases = [  # (options, (gap_open, gap_extend, significance_mode) given to align_candidates)
            (("--gap-open", "5", "--gap-extend", "2", "--significance-mode", "sampled",
              "--significance-sample-size", "20"), (5, 2, SignificanceMode.sampled)),
            (("--gap-open", "5", "--gap-extend", "2", "--max-evalue", "inf", "--format-output",
              "query,target,fident,raw"), (5, 2, SignificanceMode.default)),
            (("--gap-open", "1", "--gap-extend", "0", "--max-evalue", "inf", "--format-output", "query,target,raw"),
             (1, 0, SignificanceMode.default)),
            (("--gap-open", str(MAX_GAP), "--gap-extend", str(MAX_GAP), "--max-evalue", "inf", "--format-output",
              "query,target,raw"), (MAX_GAP, MAX_GAP, SignificanceMode.default)),
        ]
        for command in ("sequences", "db"):
            for k, (extra, want) in enumerate(cases):
                with self.subTest(command=command, extra=extra):
                    kwargs, _ = self.run_stage2(command, *extra, out=self.tmp / f"{command}{k}.tsv", run_id=f"q{k}")
                    self.assertEqual((kwargs["gap_open"], kwargs["gap_extend"], kwargs["significance_mode"]), want)

    def test_the_sign_is_ignored(self):
        """Negative gap costs are their absolute values, as before; -11/-1 is 11/1, also for calibrated significance
        (P9.S3: it used to be refused there)."""
        for command in ("sequences", "db"):
            with self.subTest(command=command):
                _, plain = self.run_stage2(command, out=self.tmp / f"{command}-plain.tsv", run_id="p")
                kwargs, negative = self.run_stage2(command, "--gap-open", "-11", "--gap-extend", "-1",
                                                   out=self.tmp / f"{command}-neg.tsv", run_id="n")
                self.assertEqual((kwargs["gap_open"], kwargs["gap_extend"]), (-11, -1))
                self.assertEqual(_sha256(negative), _sha256(plain))
                _, a = self.run_stage2(command, "--gap-open", "3", "--gap-extend", "2", "--max-evalue", "inf",
                                       "--format-output", "query,target,raw,cigar", out=self.tmp / f"{command}-a.tsv",
                                       run_id="a")
                _, b = self.run_stage2(command, "--gap-open", "-3", "--gap-extend", "-2", "--max-evalue", "inf",
                                       "--format-output", "query,target,raw,cigar", out=self.tmp / f"{command}-b.tsv",
                                       run_id="b")
                self.assertEqual(_sha256(a), _sha256(b))

    def test_an_embedding_only_search_needs_neither(self):
        """Without Stage 2 the gap options are unused, as before, and the kernel is not needed."""
        for command in ("sequences", "db"):
            with self.subTest(command=command), _no_kernel(), \
                    mock.patch.object(alignment, "align_candidates", side_effect=AssertionError("Stage 2")):
                result, out = self.run_query(command, "--no-seq-identity", "--gap-open", "0", "--gap-extend",
                                             str(MAX_GAP + 1), out=self.tmp / f"{command}.tsv")
                self.assertEqual(result.exit_code, 0, (result.output, result.exception))
                self.assertTrue(out.read_text().splitlines())


class PythonFunctionTest(_Blosum62Case):
    """``query_database_from_database`` called from Python, where typer parses nothing: the same refusals before the
    search (typer.BadParameter), and a whole float gap cost runs as its integer, as it did before P9.S3."""

    def call(self, name, **kw):
        out = self.tmp / f"{name}.tsv"
        search.query_database_from_database(query_db_path=str(self.db), subject_db_path=str(self.db),
                                            output_file=str(out), **kw)
        return out

    def test_refusals_before_the_search(self):
        both = ["--gap-open", "--gap-extend"]
        cases = [  # (options, param_hint, text)
            (dict(gap_open=10.5), "--gap-open", "must be a whole number, got 10.5"),
            (dict(gap_extend=True), "--gap-extend", "must be a whole number, got True"),
            (dict(gap_open=0.0), "--gap-open", "must be between 1 and 1048576"),
            (dict(significance_mode="bogus"), "--significance-mode", "'bogus' is not a significance mode"),
            (dict(gap_open=10.0), both, "the gap costs given are 10/1"),
            (dict(gap_open=np.int64(-10), significance_mode="default"), both, "the gap costs given are 10/1"),
            (dict(significance_mode="sampled", significance_sample_size=-5), "--significance-sample-size",
             "must be at least 2 with --significance-mode sampled, got -5"),
            (dict(significance_mode=SignificanceMode.sampled, significance_sample_size=1, max_evalue=math.inf),
             "--significance-sample-size", "got 1"),                     # the default columns hold evalue and bits
        ]
        for kw, param, text in cases:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}), _no_search() as load:
                with self.assertRaises(typer.BadParameter) as ctx:
                    self.call("refused", **kw)
                load.assert_not_called()
                self.assertEqual(ctx.exception.param_hint, param)
                self.assertIn(text, ctx.exception.format_message())

    @pytest.mark.usefixtures("fmsw_libs")
    def test_whole_floats_and_unused_sample_sizes_run(self):
        """gap_open=11.0 (the release took it, through biotite) is 11 again, with calibrated significance; a sample
        size the run does not use is not checked."""
        ints = self.call("ints", gap_open=11, gap_extend=1)
        self.assertTrue(ints.read_text().splitlines())
        floats = self.call("floats", gap_open=11.0, gap_extend=np.float64(1.0))
        self.assertEqual(_sha256(floats), _sha256(ints))
        unused = self.call("unused", significance_sample_size=0)                  # --significance-mode default
        self.assertEqual(_sha256(unused), _sha256(ints))


class GapCostRuleTest(unittest.TestCase):
    """The command line refuses exactly the combinations ``align_candidates`` refuses (check_gap_costs), so nothing
    it lets through fails after Stage 1."""

    def test_the_command_line_and_align_candidates_agree(self):
        """Over the kernel range, its bounds and both signs, whole and other floats and a bool (what the Python
        functions may be given), both modes, the significance columns and E-value filters."""
        gaps = (-MAX_GAP - 1, -MAX_GAP, -11, -10, -2, -1, 0, 1, 2, 10, 11, 12, MAX_GAP, MAX_GAP + 1, 11.0, -1.0, 0.0,
                10.5, np.int64(11), True)
        fields = (list(output.DEFAULT_OUTPUT_FIELDS), ["query", "target", "fident"], ["query", "bits"], ["evalue"],
                  ["evalue", "query", "evalue"])
        max_evalues = (None, math.inf, math.nan, 1e-3, 0.0, 10.0)
        refused = 0
        for go, ge, mode, cols, max_evalue in itertools.product(gaps, gaps, SignificanceMode, fields, max_evalues):
            try:
                alignment.check_gap_costs(go, ge, mode, output.needs_significance(cols), max_evalue)
                library = True
            except (ValueError, TypeError):
                library = False
            try:
                search._check_blosum62_gap_costs(go, ge, mode, cols, max_evalue)
                cli = True
            except typer.BadParameter:
                cli = False
            if cli != library:
                self.fail(f"gaps {go}/{ge}, {mode.value}, {cols}, max_evalue={max_evalue}: command line "
                          f"{'accepts' if cli else 'refuses'}, align_candidates {'accepts' if library else 'refuses'}")
            refused += not cli
        self.assertGreater(refused, 0)

    def test_each_refusal_names_its_own_cause(self):
        """Only the 11/1 rule gets the 11/1 message (a float gap cost, a bool or an unknown mode used to get it, with
        empty slots); a refusal names one option."""
        check = search._check_blosum62_gap_costs
        for args in ((11.0, 1.0, SignificanceMode.sampled, ["query"], 1e-3),
                     (11.0, 1, SignificanceMode.default, ["query", "target"], math.inf),
                     (11.0, 1, SignificanceMode.default, ["query"], None), (-11.0, -1.0, "default", ["evalue"], 1e-3)):
            with self.subTest(args=args):
                check(*args)                                                   # accepted: 11/1, or no significance
        cases = [  # (args, param_hint, text)
            ((11.5, 1, SignificanceMode.sampled, ["query"], 1e-3), "--gap-open", "must be a whole number, got 11.5"),
            ((True, 1, SignificanceMode.default, ["query"], None), "--gap-open", "must be a whole number, got True"),
            ((11, "1", SignificanceMode.default, ["query"], None), "--gap-extend", "must be a whole number, got '1'"),
            ((10, 1, "bogus", ["query"], None), "--significance-mode", "'bogus' is not a significance mode"),
            ((10.0, 1, "default", ["query"], 1e-3), ["--gap-open", "--gap-extend"],
             "this run needs them for the E-value filter --max-evalue 0.001; the gap costs given are 10/1"),
        ]
        for args, param, text in cases:
            with self.subTest(args=args), self.assertRaises(typer.BadParameter) as ctx:
                check(*args)
            self.assertEqual(ctx.exception.param_hint, param)
            self.assertIn(text, ctx.exception.format_message())
            self.assertNotIn("needs them for ;", ctx.exception.format_message())

    def test_the_sample_size_check(self):
        check = search._check_significance_sample_size
        for args in ((1, SignificanceMode.sampled, ["evalue"], None), (0, "sampled", ["query"], 1e-3),
                     (-5, SignificanceMode.sampled, ["bits"], math.inf)):
            with self.subTest(args=args), self.assertRaisesRegex(typer.BadParameter, "must be at least 2"):
                check(*args)
        for args in ((2, SignificanceMode.sampled, ["evalue"], 1e-3), (1, SignificanceMode.default, ["evalue"], 1e-3),
                     (0, SignificanceMode.sampled, ["query"], None),
                     (-5, SignificanceMode.sampled, ["query"], math.inf)):
            with self.subTest(args=args):
                check(*args)                                                   # 2 or more, or not sampled

    def test_a_repeated_column_is_named_once(self):
        with self.assertRaises(typer.BadParameter) as ctx:
            search._check_blosum62_gap_costs(10, 1, SignificanceMode.default, ["evalue", "query", "evalue"], None)
        self.assertIn("for the evalue column of --format-output;", ctx.exception.format_message())


if __name__ == "__main__":
    unittest.main()
