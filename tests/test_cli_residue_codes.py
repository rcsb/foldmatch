"""P5.6: ``fm-search build sequences --residue-codes``.

Most tests replace the ESM3 pass by a fake that writes the shard the real pass writes (same dataset, same writer)
with the F2c reference codes for the fixture FASTA, so the command-line orchestration, the refusals and the code
store run without ESM3. ``RealEsm3CliTest`` runs the real CPU pass through the command line and compares the codes
with F2c (gate A4-xcpu). The bundle comes from the scoring-system registry (``esm_codes_bundle``: skipped only when
offline and not cached).
"""
import gc
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from typer.testing import CliRunner

import esm_codes_bundle
import foldmatch
from foldmatch.cli import search
from foldmatch.errors import RefusalError
from foldmatch.search import scoring_system as ss
from foldmatch.search.residue_code_store import CodeStoreValidationError, ResidueCodeStore, entries_digest

_TEST_PATH = Path(__file__).parent
_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
_F2C = _TEST_PATH / "resources" / "esm_codes" / "fasta_rows_codes_v1.npz"
_SRC = str(Path(foldmatch.__file__).resolve().parent.parent)
_LAUNCH_ENV = ("FOLDMATCH_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_RESTART_COUNT", "SLURM_PROCID",
               "SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "RANK", "LOCAL_RANK", "NODE_RANK", "WORLD_SIZE")

SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)

#: Plan section 8, P5.6: the help text of --residue-codes, verbatim.
PLAN_HELP = ("Also compute 8-byte per-residue ESM3 codes in the same pass, enabling `query sequences --scoring "
             "esm-codes`. Adds 8 B per residue plus up to about 100 B per entry for the index, and a one-time ~26 MB "
             "download. Sequence (FASTA) databases only.")
#: A4_THRESHOLDS.json (sha256 7eabf36e...), gate A4-xcpu: CPU Mac vs CPU Linux code bits, also the F2c threshold.
A4_XCPU_BITS = 0.999
_DB_SUFFIXES = (".index", ".metadata", ".sequences")


@contextmanager
def launch_env(**values):
    """Exactly ``values`` among the launch variables; the rest unset (only those variables are restored)."""
    saved = {name: os.environ.get(name) for name in _LAUNCH_ENV}
    try:
        for name in _LAUNCH_ENV:
            os.environ.pop(name, None)
        os.environ.update({k: str(v) for k, v in values.items()})
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _f2c():
    """F2c: ``{sequence: reference codes [L, 8]}``, the ids and sequences in file order, all codes ``[303, 8]``."""
    z = np.load(_F2C, allow_pickle=False)
    offsets = z["offsets"]
    ids = [str(i) for i in z["ids"]]
    sequences = [str(s) for s in z["sequences"]]
    by_sequence = {s: z["codes"][offsets[k]:offsets[k + 1]] for k, s in enumerate(sequences)}
    return by_sequence, ids, sequences, z["codes"]


def _sha1(sequence: str) -> bytes:
    return hashlib.sha1(sequence.encode("ascii")).digest()


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bits_equal(a, b):
    return int((np.unpackbits(a, axis=1) == np.unpackbits(b, axis=1)).sum()), int(a.size * 8)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _fake_codes(name: str, sequence: str) -> np.ndarray:
    """The F2c reference codes of the fixture's sequences; pseudo-random codes (fixed per id) for any other."""
    reference = _f2c()[0].get(sequence)
    if reference is not None:
        return reference
    return np.random.default_rng(zlib.crc32(name.encode())).integers(0, 256, (len(sequence), 8), dtype=np.uint8)


def fake_chain_predict(calls=None, payload=_fake_codes, system=None, ranks=1, device="cpu"):
    """Stand-in for ``chain_complete_inference.predict``: the shards the pass writes, from the same dataset and
    writer (with the writer's start-of-pass hook, which records the pass), with an 8-d chain embedding fixed per id
    and, for a residue pass, ``payload(name, sequence)`` as the residue codes (``system`` overrides the scoring system
    the shards are written for). With ``ranks`` > 1 the entries go round-robin to that many rank shards, as DDP's
    sampler deals them; ``device`` is the device type the module 'runs' on."""
    def predict(**kwargs):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        from foldmatch.inference.chain_complete_inference import residue_shard_writer
        from foldmatch.writer.batch_writer import ParquetBatchWriter
        if calls is not None:
            calls.append(kwargs)
        dataset = EsmProtFromFasta(kwargs["src_stream"], min_res_n=kwargs["min_res_n"])
        residue_output = kwargs.get("residue_output")
        for rank in range(ranks):
            items = [dataset[i] for i in range(rank, len(dataset), ranks)]
            names = tuple(name for _, name in items)
            embeddings = [torch.from_numpy(np.random.default_rng(zlib.crc32(name.encode())).normal(size=8)
                                           .astype(np.float32)) for name in names]
            trainer = SimpleNamespace(global_rank=rank, world_size=ranks)
            if residue_output is None:
                writer = ParquetBatchWriter(kwargs["out_folder"], kwargs["out_name"])
                prediction = (embeddings, names)
            else:
                shard_system = system or kwargs["scoring_system"].system
                writer = residue_shard_writer(kwargs["out_folder"], kwargs["out_name"], residue_output, shard_system,
                                              min_res_n=kwargs["min_res_n"])
                writer.on_predict_start(trainer, SimpleNamespace(residue_output=residue_output,
                                                                 device=torch.device(device)))
                payloads = tuple(np.ascontiguousarray(payload(name, prot.sequence)).tobytes() for prot, name in items)
                prediction = (embeddings, names, payloads, tuple(_sha1(prot.sequence) for prot, _ in items))
            writer.write_on_batch_end(trainer, None, prediction, None, None, 0, 0)
            writer.on_predict_end(trainer, None)
    return predict


def _esm3_spy():
    """Patch the ESM3 loader of the pass; any call means ESM3 was reached."""
    return mock.patch("foldmatch.inference.chain_complete_inference.get_residue_model",
                      mock.Mock(side_effect=AssertionError("ESM3 was loaded")))


def invoke(*args):
    """``fm-search`` in-process (exit codes as in a shell, the exception kept), the logging setup restored after."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        return CliRunner().invoke(search.app, [str(a) for a in args])
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def _build_args(fasta, db, shards, *extra):
    return ("build", "sequences", "--fasta-file", fasta, "--output-db", db, "--tmp-embedding-folder", shards,
            "--accelerator", "cpu", "--devices", "1", *extra)


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.shards = self.tmp / "shards"
        self.shards.mkdir()
        self.enterContext(launch_env())

    def refused(self, result, cause=None):
        """The command exited 1 with a RefusalError (D20); returns its message."""
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIsInstance(result.exception, RefusalError, repr(result.exception))
        if cause is not None:
            self.assertIsInstance(result.exception.__cause__, cause, repr(result.exception.__cause__))
        return str(result.exception)

    def usage_error(self, result, text):
        """The command exited 2 (typer.BadParameter or a click usage error) naming ``text`` (in typer's error panel,
        whose borders are dropped)."""
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn(text, " ".join(result.output.translate({ord(c): " " for c in "│╭╮╰╯─"}).split()))


class _WithBundle(_Tmp):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        esm_codes_bundle.registry_files()  # unittest.SkipTest only when offline and not cached


# =====================================================================================================================
# P5.6: the option
# =====================================================================================================================
class OptionTest(unittest.TestCase):

    def _command(self, *path):
        import typer
        command = typer.main.get_command(search.app)
        for name in path:
            command = command.commands[name]
        return command

    def test_help_text_and_default(self):
        param = next(p for p in self._command("build", "sequences").params if p.name == "residue_codes")
        self.assertEqual(param.help, PLAN_HELP)
        self.assertEqual(param.opts, ["--residue-codes"])
        self.assertEqual(param.secondary_opts, ["--no-residue-codes"])
        self.assertIs(param.default, False)
        output = CliRunner().invoke(search.app, ["build", "sequences", "--help"], env={"COLUMNS": "400"}).output
        help_text = " ".join(output.translate({ord(c): " " for c in "│╭╮╰╯─"}).split())
        self.assertIn(f"--residue-codes --no-residue-codes {PLAN_HELP} [default: no-residue-codes]", help_text)


# =====================================================================================================================
# P5.6: refusals, every one before ESM3 and on every rank alike
# =====================================================================================================================
class BuildRefusalTest(_Tmp):

    def build(self, fasta=_FASTA, *extra):
        with _esm3_spy() as esm3, mock.patch.object(ss, "prepare",
                                                    side_effect=AssertionError("bundle fetched")) as prepare:
            result = invoke(*_build_args(fasta, self.tmp / "db", self.shards, "--residue-codes", *extra))
        esm3.assert_not_called()
        prepare.assert_not_called()
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db*")), [])
        self.assertEqual(list(self.shards.iterdir()), [])
        return result

    def test_fasta_the_method_cannot_take(self):
        for text, what in ((">a\nACDE\n>b\nKLM\n>a\nACDE\n", "repeat an earlier id"),
                           (">a\nAC<mask>DE\n", "'<' or '>'"), (">a\nACDE>\n", "'<' or '>'"),
                           (">a\nACDÉ\n", "non-ASCII")):
            for rank_env in ({}, {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}):
                with self.subTest(what=what, text=text, rank_env=rank_env), launch_env(**rank_env):
                    fasta = _write(self.tmp / "bad.fasta", text)
                    self.assertIn(what, self.refused(self.build(fasta), ValueError))

    def test_a_header_without_an_id(self):
        """'>' alone: refused by the pre-flight before ESM3, with the file named (it used to be an IndexError); a
        plain build refuses it the same way, as a ValueError."""
        fasta = _write(self.tmp / "h.fasta", ">\nACDE\n>b\nKLMN\n")
        message = self.refused(self.build(fasta), ValueError)
        self.assertIn(f"{fasta}: the header line at byte 0 is empty", message)
        with _esm3_spy() as esm3:
            result = invoke(*_build_args(fasta, self.tmp / "db", self.shards))
        self.assertEqual(result.exit_code, 1)
        self.assertIs(type(result.exception), ValueError)
        self.assertIn("is empty ('>' without an id)", str(result.exception))
        esm3.assert_not_called()

    def test_a_plain_build_of_the_same_fasta_is_not_strict(self):
        fasta = _write(self.tmp / "dup.fasta", ">a\nACDE\n>a\nACDE\n")
        with _esm3_spy() as esm3:
            result = invoke(*_build_args(fasta, self.tmp / "db", self.shards))
        self.assertIsInstance(result.exception, AssertionError)  # the pass was reached
        esm3.assert_called_once()

    def test_existing_outputs(self):
        for suffix in (".index", ".metadata", ".invlists", ".sequences", ".rescodes.bin", ".rescodes.idx",
                       ".rescodes.json", ".rescodes.json.tmp"):
            for rank_env in ({}, {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}):
                with self.subTest(suffix=suffix, rank_env=rank_env), launch_env(**rank_env):
                    existing = self.tmp / f"db{suffix}"
                    existing.touch()
                    try:
                        with _esm3_spy() as esm3, mock.patch.object(ss, "prepare") as prepare:
                            result = invoke(*_build_args(_FASTA, self.tmp / "db", self.shards, "--residue-codes"))
                    finally:
                        existing.unlink()
                    self.assertIn(str(existing), self.refused(result, FileExistsError))
                    esm3.assert_not_called()
                    prepare.assert_not_called()

    def test_shards_of_the_same_run_id(self):
        with launch_env(FOLDMATCH_RUN_ID="r"):
            (self.shards / "emb_r-0.parquet").touch()
            with _esm3_spy() as esm3, mock.patch.object(ss, "prepare") as prepare:
                result = invoke(*_build_args(_FASTA, self.tmp / "db", self.shards, "--residue-codes"))
        self.assertIn("same run id", self.refused(result, FileExistsError))
        esm3.assert_not_called()
        prepare.assert_not_called()

    def test_a_bundle_problem(self):
        with _esm3_spy() as esm3, mock.patch.object(
                ss, "prepare", side_effect=ss.BundleUnavailableError("not cached")) as prepare:
            result = invoke(*_build_args(_FASTA, self.tmp / "db", self.shards, "--residue-codes"))
        self.assertIn("not cached", self.refused(result, ss.BundleUnavailableError))
        prepare.assert_called_once()
        esm3.assert_not_called()
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db*")), [])

    def test_one_fetch_then_cached_only(self):
        """The process that starts the run may download the bundle; a rank Lightning starts, or a multi-node run,
        finds it cached (P4 outcome: prepare(system, local_files_only=True) as the every-rank pre-flight)."""
        stop = RuntimeError("stop before the pass")
        for rank_env, extra, local_only in (({}, (), False), ({"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}, (), True),
                                            ({}, ("--num-nodes", "2"), True)):
            with self.subTest(rank_env=rank_env, extra=extra), launch_env(**rank_env), \
                    mock.patch.object(ss, "prepare") as prepare, \
                    mock.patch("foldmatch.search.embedding_computer.chain_predict", side_effect=stop) as predict:
                result = invoke(*_build_args(_FASTA, self.tmp / "db", self.shards, "--residue-codes", *extra))
                self.assertIs(result.exception, stop)
                prepare.assert_called_once_with(SYSTEM, local_files_only=local_only)
                kwargs = predict.call_args.kwargs
                self.assertEqual(kwargs["residue_output"], "residue_codes")
                self.assertIs(kwargs["scoring_system"], prepare.return_value)

    def test_exit_status_of_a_refusal(self):
        """fm-search prints a refusal on one line and exits 1 (D20)."""
        fasta = _write(self.tmp / "dup.fasta", ">a\nACDE\n>a\nACDE\n")
        argv = ["fm-search", *map(str, _build_args(fasta, self.tmp / "db", self.shards, "--residue-codes"))]
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
        self.assertIn("repeat an earlier id", errors[0])


# =====================================================================================================================
# P5.6: the build (fake ESM3 pass, real bundle)
# =====================================================================================================================
class BuildTest(_WithBundle):

    def build(self, *extra, fasta=_FASTA, db=None, calls=None, run_id="p56", **fake):
        db = db or self.tmp / "db"
        with launch_env(FOLDMATCH_RUN_ID=run_id), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict(calls, **fake)):
            return invoke(*_build_args(fasta, db, self.shards, *extra))

    def test_the_store_is_built_and_validated(self):
        calls = []
        result = self.build("--residue-codes", calls=calls)
        self.assertEqual(result.exit_code, 0, result.output)
        _, ids, sequences, codes = _f2c()
        store = ResidueCodeStore(self.tmp, "db")
        self.assertTrue(store.exists())
        manifest = json.loads(store.manifest_path.read_text())
        self.assertEqual((manifest["entries"], manifest["total_residues"], manifest["min_res_n"]), (2, 303, 0))
        self.assertEqual(manifest, store.check_compatible())
        self.assertEqual(store.bin_path.stat().st_size, 8 * 303)
        np.testing.assert_array_equal(store.open_memmap(), codes)
        self.assertEqual(manifest["bin_sha256"], hashlib.sha256(codes.tobytes()).hexdigest())
        self.assertEqual(manifest["entries_digest"], entries_digest([(i, _sha1(s)) for i, s in zip(ids, sequences)]))
        self.assertEqual({i: e[:2] for i, e in store.lookup(ids).items()}, {"1acb_E": (0, 245), "2uzi_A": (245, 58)})
        self.assertEqual(manifest["cross_checks"], {
            "preflight_entries": 2, "preflight_residues": 303, "sequence_store_entries": 2,
            "sequence_store_residues": 303, "faiss_entries": 2, "seq_sha1_mismatches": 0, "length_mismatches": 0})
        build = manifest["build"]
        self.assertEqual((build["run_id"], build["from_shards"], build["accelerator"], build["devices"],
                          build["esm_precision"], build["encoder_precision"]),
                         ("p56", False, "cpu", 1, "fp32", "fp32-no-tf32"))
        from foldmatch.search.residue_shards import read_shard_info
        record = read_shard_info(self.shards / "emb_p56-0.parquet").pass_record
        for key in ("foldmatch_version", "torch", "esm", "platform"):  # the pass's, from the shards
            self.assertEqual(build[key], record[key], key)
        self.assertEqual(build["foldmatch_version"], foldmatch.__version__)
        self.assertEqual((record["min_res_n"], record["esm_revision"]), (0, manifest["esm"]["revision"]))
        self.assertEqual(manifest["system_name"], SYSTEM.name)
        self.assertEqual(manifest["encoding_digest"], SYSTEM.encoding_digest())
        # The pass got the prepared registry system; the shards are kept (D13).
        self.assertEqual(calls[0]["residue_output"], "residue_codes")
        self.assertIs(calls[0]["scoring_system"].system, SYSTEM)
        self.assertEqual([p.name for p in self.shards.iterdir()], ["emb_p56-0.parquet"])
        for suffix in _DB_SUFFIXES:
            self.assertTrue((self.tmp / f"db{suffix}").exists())

    def test_a_plain_build_is_unchanged(self):
        """Without the option: the pass is called without residue keywords, no bundle is fetched, no code store is
        written, and the database files are byte-identical to those of a --residue-codes build."""
        calls = []
        with mock.patch.object(ss, "prepare", side_effect=AssertionError("bundle fetched")):
            result = self.build(calls=calls, db=self.tmp / "plain" / "db")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("residue_output", calls[0])
        self.assertNotIn("scoring_system", calls[0])
        self.assertEqual(sorted(p.name for p in (self.tmp / "plain").iterdir()),
                         ["db.index", "db.metadata", "db.sequences"])
        result = self.build("--residue-codes", db=self.tmp / "codes" / "db", run_id="p56b")
        self.assertEqual(result.exit_code, 0, result.output)
        for suffix in _DB_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertEqual(_sha256(self.tmp / "plain" / f"db{suffix}"),
                                 _sha256(self.tmp / "codes" / f"db{suffix}"))
        result = self.build("--no-residue-codes", db=self.tmp / "off" / "db", run_id="p56c")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(sorted(p.name for p in (self.tmp / "off").iterdir()),
                         ["db.index", "db.metadata", "db.sequences"])

    def test_min_res_n(self):
        result = self.build("--residue-codes", "--min-res-n", "100")
        self.assertEqual(result.exit_code, 0, result.output)
        manifest = ResidueCodeStore(self.tmp, "db").check_compatible()
        self.assertEqual((manifest["entries"], manifest["total_residues"], manifest["min_res_n"]), (1, 245, 100))

    def test_the_build_block_describes_the_pass(self):
        """The device type, its ESM3 precision and the ranks come from the pass (the shards' record): two ranks on
        CUDA give accelerator 'cuda', devices 2 and bf16 autocast."""
        result = self.build("--residue-codes", ranks=2, device="cuda")
        self.assertEqual(result.exit_code, 0, result.output)
        build = ResidueCodeStore(self.tmp, "db").check_compatible()["build"]
        self.assertEqual((build["accelerator"], build["devices"], build["esm_precision"], build["encoder_precision"]),
                         ("cuda", 2, "bf16-autocast", "fp32-no-tf32"))
        self.assertEqual(sorted(p.name for p in self.shards.iterdir()), ["emb_p56-0.parquet", "emb_p56-1.parquet"])

    def test_the_live_faiss_count_is_validated(self):
        """The direct build gives validate() the live FAISS index's count: one too many fails check (e) and only
        the code store is removed."""
        from foldmatch.search.embedding_database import EmbeddingDatabase
        create_db = EmbeddingDatabase.create_db

        def one_more(self_, *args, **kwargs):
            out = create_db(self_, *args, **kwargs)
            self_.db.chain_ids.append("phantom")
            return out

        with mock.patch.object(EmbeddingDatabase, "create_db", one_more):
            result = self.build("--residue-codes")
        message = self.refused(result, CodeStoreValidationError)
        self.assertEqual(set(result.exception.__cause__.failures), {"e"})
        self.assertIn("the FAISS index has 3 entries, the code store 2", message)
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db*")), ["db.index", "db.metadata", "db.sequences"])

    def test_every_refusal_of_a_residue_codes_build_is_a_refusal_error(self):
        """D20: a refused step of a --residue-codes build, in the pass or on rank 0 after it, exits 1 with a
        RefusalError (the step's error is its cause); a plain build raises what it always raised."""
        def missing_rank(**kwargs):  # rank 1's shard is lost
            fake_chain_predict(ranks=2)(**kwargs)
            next(Path(kwargs["out_folder"]).glob("*-1.parquet")).unlink()

        cases = {
            "the pass": (mock.Mock(side_effect=ValueError("the pass refused")), "the pass refused"),
            "the shard count": (missing_rank, "a rank's shard is missing"),
        }
        for case, (predict, text) in cases.items():
            for extra in (("--residue-codes",), ()):
                with self.subTest(case=case, extra=extra):
                    for path in [*self.shards.iterdir(), *self.tmp.glob("db*")]:
                        path.unlink()
                    with launch_env(FOLDMATCH_RUN_ID="d20"), \
                            mock.patch("foldmatch.search.embedding_computer.chain_predict", predict):
                        result = invoke(*_build_args(_FASTA, self.tmp / "db", self.shards, *extra))
                    self.assertEqual(result.exit_code, 1, result.output)
                    self.assertIn(text, str(result.exception))
                    if extra:
                        self.refused(result, ValueError)
                    else:
                        self.assertIs(type(result.exception), ValueError)

    def test_cleanup_embeddings_comes_after_the_store(self):
        result = self.build("--residue-codes", "--cleanup-embeddings")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(ResidueCodeStore(self.tmp, "db").exists())
        self.assertEqual(list(self.shards.iterdir()), [])

    def test_a_validation_failure_keeps_the_database_and_the_shards(self):
        """An entry the pass could not encode (an empty payload, P5.1) fails check (c) after the pass: the code store
        is removed, the database still serves BLOSUM62 and the shards stay, even with --cleanup-embeddings."""
        def payload(name, sequence):
            return np.empty((0, 8), np.uint8) if name == "2uzi_A" else _fake_codes(name, sequence)

        result = self.build("--residue-codes", "--cleanup-embeddings", payload=payload)
        message = self.refused(result, CodeStoreValidationError)
        self.assertEqual(result.exception.__cause__.failures["c"], (1, ("2uzi_A",)))
        self.assertIn("empty payload", message)
        self.assertIn(f"kept in {self.shards} (run id p56)", message)
        # The pass wrote the empty payload, so the advice is a fresh build, not --from-shards.
        self.assertIn("rebuilding from the same shards would fail the same way", message)
        self.assertNotIn("--from-shards", message)
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db*")), ["db.index", "db.metadata", "db.sequences"])
        self.assertEqual([p.name for p in self.shards.iterdir()], ["emb_p56-0.parquet"])



# =====================================================================================================================
# real ESM3 through the command line (CPU)
# =====================================================================================================================
class RealEsm3CliTest(unittest.TestCase):
    """``fm-search build sequences --residue-codes`` on ``test_sequences.fasta`` with the real CPU pass: the manifest,
    the codes against F2c (A4-xcpu), and the plain build's files."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        esm_codes_bundle.registry_files()
        gc.collect()  # earlier tests' ESM3 models, if only reference cycles still hold them
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.shards = cls.tmp / "shards"
        cls.shards.mkdir()
        cls.results = {}
        for kind, extra in (("codes", ("--residue-codes",)), ("plain", ())):
            with launch_env(FOLDMATCH_RUN_ID=f"real-{kind}"):
                cls.results[kind] = invoke(*_build_args(_FASTA, cls.tmp / kind / "db", cls.shards, "--batch-size",
                                                        "2", *extra))
            gc.collect()  # the pass's ESM3 (5.6 GB) is only reachable through reference cycles

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()

    def setUp(self):
        for kind, result in self.results.items():
            self.assertEqual(result.exit_code, 0, f"{kind}: {result.output[-4000:]} {result.exception!r}")

    def test_manifest_and_f2c_codes(self):
        _, ids, sequences, codes_ref = _f2c()
        store = ResidueCodeStore(self.tmp / "codes", "db")
        manifest = store.check_compatible()
        self.assertEqual((manifest["entries"], manifest["total_residues"]), (2, 303))
        self.assertEqual(store.bin_path.stat().st_size, 8 * 303)
        entries = store.lookup(ids)
        self.assertEqual([entries[i][:2] for i in ids], [(0, 245), (245, 58)])
        self.assertEqual([entries[i].seq_sha1 for i in ids], [_sha1(s) for s in sequences])
        codes = np.asarray(store.open_memmap())
        equal, n = _bits_equal(codes, codes_ref)
        print(f"\nCLI build F2c CPU code bits: {equal}/{n} = {equal / n:.6f}", file=sys.stderr)
        self.assertGreaterEqual(equal / n, A4_XCPU_BITS, f"F2c code bits {equal}/{n}")

    def test_a_build_without_codes_writes_the_same_database_files(self):
        for suffix in _DB_SUFFIXES:
            with self.subTest(suffix=suffix):
                self.assertEqual(_sha256(self.tmp / "plain" / f"db{suffix}"),
                                 _sha256(self.tmp / "codes" / f"db{suffix}"))
        self.assertEqual(sorted(p.name for p in (self.tmp / "plain").iterdir()),
                         ["db.index", "db.metadata", "db.sequences"])



_TWO_RANK_CHILD_ENV = "FOLDMATCH_TEST_PREPARE_LOG"
# Rendezvous state this process may hold from an earlier in-process DDP pass (Lightning sets MASTER_ADDR/MASTER_PORT):
# a child that inherited MASTER_PORT would try to join that process group, or find its port taken.
_DDP_ENV_PREFIXES = ("MASTER_", "PL_", "TORCHELASTIC_")
_DDP_ENV = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK")
_SITECUSTOMIZE = r"""
import json, os
from foldmatch.search import scoring_system as ss
_prepare = ss.prepare
def _spy(system, **kwargs):
    rank = max(int(os.environ.get(k) or 0) for k in ("RANK", "LOCAL_RANK", "NODE_RANK"))
    with open(os.path.join(os.environ["FOLDMATCH_TEST_PREPARE_LOG"], f"prepare-{rank}.json"), "w") as f:
        json.dump(kwargs, f)
    return _prepare(system, **kwargs)
ss.prepare = _spy
"""


@unittest.skipUnless(os.environ.get("FOLDMATCH_SLOW_TESTS") == "1",
                     "slow: two ESM3 ranks on CPU; set FOLDMATCH_SLOW_TESTS=1 to run")
class TwoRankCliSmoke(unittest.TestCase):
    """``fm-search build sequences --residue-codes --devices 2`` as a command (CPU, gloo): Lightning re-runs the
    command for rank 1, both ranks pass the pre-flight (rank 1 checks the bundle cached-only), and rank 0 builds a
    store whose codes are F2c's."""

    def test_two_rank_build(self):
        _, ids, _, codes_ref = _f2c()
        esm_codes_bundle.registry_files()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "shards").mkdir()
            (tmp / "site").mkdir()
            (tmp / "log").mkdir()
            (tmp / "site" / "sitecustomize.py").write_text(_SITECUSTOMIZE)
            env = {k: v for k, v in os.environ.items()
                   if k not in _LAUNCH_ENV and k not in _DDP_ENV and not k.startswith(_DDP_ENV_PREFIXES)}
            env["PYTHONPATH"] = os.pathsep.join([str(tmp / "site"), _SRC] +
                                                [p for p in [os.environ.get("PYTHONPATH")] if p])
            env[_TWO_RANK_CHILD_ENV] = str(tmp / "log")
            command = [sys.executable, "-m", "foldmatch.cli.search",
                       *map(str, _build_args(_FASTA, tmp / "db" / "idx", tmp / "shards", "--residue-codes"))]
            command[command.index("--devices") + 1] = "2"
            run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=1800)
            self.assertEqual(run.returncode, 0, run.stderr[-4000:])
            calls = {p.name: json.loads(p.read_text()) for p in (tmp / "log").iterdir()}
            store = ResidueCodeStore(tmp / "db", "idx")
            manifest = store.check_compatible()
            codes = np.asarray(store.open_memmap()).copy()
            entries = store.lookup(ids)
            shards = sorted(p.name for p in (tmp / "shards").iterdir())
        self.assertEqual(calls, {"prepare-0.json": {"local_files_only": False},
                                 "prepare-1.json": {"local_files_only": True}})
        self.assertEqual((manifest["entries"], manifest["total_residues"], manifest["build"]["devices"]), (2, 303, 2))
        self.assertEqual(len(shards), 2)
        self.assertEqual(manifest["build"]["run_id"], shards[0][len("emb_"):-len("-0.parquet")])
        z = np.load(_F2C, allow_pickle=False)
        for k, name in enumerate(ids):
            offset, length, _ = entries[name]
            got = codes[offset:offset + length]
            equal, n = _bits_equal(got, codes_ref[z["offsets"][k]:z["offsets"][k + 1]])
            self.assertGreaterEqual(equal / n, A4_XCPU_BITS, f"{name} code bits {equal}/{n}")


if __name__ == "__main__":
    unittest.main()
