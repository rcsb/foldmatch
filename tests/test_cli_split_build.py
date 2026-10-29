"""P8b: the split sequence build (``build sequences-embed``, ``build sequences-index``, ``build sequences-seal``) and
the phases of ``build sequences`` it shares with the one-shot build.

No ESM3 except in ``RealEsm3SplitTest`` and ``TwoRankSplitSmoke`` (so a run of this module loads ESM3 on CPU): elsewhere
the pass is ``fake_chain_predict`` and the spies are those of ``test_cli_residue_codes``. ``OneShotPhasesTest``
(P8b.1) checks that ``build sequences`` runs its phases in the old order with the old arguments, the run binding and
the pre-flight of ``EmbeddingComputer``, the index-phase log lines and the CPU-affinity message. ``EmbedOptionTest``,
``EmbedStepTest`` and ``EmbedRefusalTest`` (P8b.3) check step 1, ``build sequences-embed``: its options, the shards and
the run record it leaves, and each of its refusals E1-E12 at its place. Two ranks are the fake pass's two shards plus a
patched ``_gather_rank_facts`` that returns two ranks' facts. ``IndexOptionTest``, ``IndexStepTest`` and
``IndexRefusalTest`` (P8b.4) check step 2, ``build sequences-index``: its options, that the two steps build the
one-shot's database (``assert_same_database``), its threads, and each of its refusals I1-I24 at its place; step 2 runs
under the ESM3 spy with ``scoring_system.prepare`` refusing. ``SealOptionTest``, ``SealTest`` and ``SealRefusalTest``
(P8b.5) check the recovery, ``build sequences-seal``, under the same spies: its options, that the shards a failed
one-shot kept or a step 1 left without a record are sealed and then built into the one-shot's database, and each of its
refusals S1-S11, which write nothing. ``LaunchChecksBeforeTorchTest`` (P9.I1) runs step 2 and the seal as commands
with a hook that ends the process at any import of torch or Lightning: I2 and I3 refuse before it; ``CommandLineTest``
(P9.I2) checks the command line their advice ends with. ``RealEsm3SplitTest`` runs both builds with the real CPU pass,
and ``TwoRankSplitSmoke`` (slow) both builds on two real ranks as commands.
"""
import dataclasses
import errno
import gc
import glob
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack, chdir, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

import esm_codes_bundle
import foldmatch
import foldmatch.search.code_alignment as code_alignment
from foldmatch.cli import search
from foldmatch.search import alignment, split_run
from foldmatch.search import scoring_system as ss
from foldmatch.search.embedding_computer import EmbeddingComputer
from foldmatch.search.embedding_database import EmbeddingDatabase
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.search.residue_code_store import CodeStoreValidationError, ResidueCodeStore
from foldmatch.search.residue_shards import META_PASS, pass_record, read_shard_info
from foldmatch.search.sequence_store import SequenceStore
from foldmatch.types.api_types import Accelerator, IndexConfig, IndexType, LogLevel, Strategy
from foldmatch.utils.esm.loaders import ESM_REVISION
from foldmatch.utils.sequence import preflight_fasta
from test_cli_residue_codes import A4_XCPU_BITS, OTHER_ENCODING, PLAN_HELP, _DDP_ENV, _DDP_ENV_PREFIXES, \
    _LAUNCH_ENV, _SITECUSTOMIZE, _SRC, _TWO_RANK_CHILD_ENV, _bits_equal, _esm3_spy, _f2c, _fake_codes, _Tmp, \
    _WithBundle, _build_args, _write, fake_chain_predict, invoke, launch_env

_TEST_PATH = Path(__file__).parent
_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)
#: Non-default pass and index options, so that the spies see each one threaded through.
_OPTIONS = ("--min-res-n", "100", "--batch-size", "2", "--ivf-nlist", "256", "--load-batch-size", "1000",
            "--cleanup-embeddings")


def _faiss_libs() -> str:
    """The faiss-cpu wheel's library folder, beside the installed faiss package (the macOS wheels have none)."""
    import faiss
    return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(faiss.__file__))), "faiss_cpu.libs")


# =====================================================================================================================
# P8b.1: the phases of build sequences
# =====================================================================================================================
class OneShotPhasesTest(_WithBundle):

    def build(self, *extra, run_id="p8b1", db="db"):
        with launch_env(FOLDMATCH_RUN_ID=run_id), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
            return invoke(*_build_args(_FASTA, self.tmp / db, self.shards, *extra))

    def test_the_phases_run_in_order_with_the_old_arguments(self):
        """Pre-flight, pass, index phase, cleanup: each once, in this order, with the command's arguments; the pass,
        create_db and the sequence store get exactly the arguments they got before the phases were functions, and
        the index phase gets no run record and no hook."""
        for codes in (False, True):
            with self.subTest(codes=codes):
                for path in [*self.shards.iterdir(), *self.tmp.glob("db*")]:
                    path.unlink()
                extra = _OPTIONS + (("--residue-codes",) if codes else ())
                manager = mock.Mock()
                spies = {
                    "preflight": mock.patch.object(search, "_sequence_build_preflight",
                                                   wraps=search._sequence_build_preflight),
                    "run_pass": mock.patch.object(search, "_run_sequence_pass", wraps=search._run_sequence_pass),
                    "index": mock.patch.object(search, "_build_sequence_database",
                                               wraps=search._build_sequence_database),
                    "cleanup": mock.patch.object(EmbeddingComputer, "remove_shards", autospec=True,
                                                 side_effect=EmbeddingComputer.remove_shards),
                    "compute": mock.patch.object(EmbeddingComputer, "compute_from_fasta", autospec=True,
                                                 side_effect=EmbeddingComputer.compute_from_fasta),
                    "create_db": mock.patch.object(EmbeddingDatabase, "create_db", autospec=True,
                                                   side_effect=EmbeddingDatabase.create_db),
                    "store": mock.patch.object(SequenceStore, "create", autospec=True,
                                               side_effect=SequenceStore.create),
                }
                prepare = (mock.patch.object(ss, "prepare", wraps=ss.prepare) if codes else
                           mock.patch.object(ss, "prepare", side_effect=AssertionError("bundle fetched")))
                with ExitStack() as stack, prepare:
                    for name, patcher in spies.items():
                        manager.attach_mock(stack.enter_context(patcher), name)
                    result = self.build(*extra)
                self.assertEqual(result.exit_code, 0, result.output)

                self.assertEqual([c[0] for c in manager.mock_calls],
                                 ["preflight", "run_pass", "compute", "index", "create_db", "store", "cleanup"])
                calls = {c[0]: c for c in manager.mock_calls}
                self.assertEqual(calls["preflight"], mock.call.preflight(
                    _FASTA.resolve(), 100, str(self.tmp / "db"), self.shards.resolve(), 1, codes))
                run = calls["run_pass"].args[0]
                self.assertIsInstance(run, search._SequencePass)
                self.assertEqual(run.residue_codes, codes)
                self.assertEqual(calls["run_pass"], mock.call.run_pass(
                    run, fasta_file=_FASTA.resolve(), min_res_n=100, batch_size=2, num_workers=0, num_nodes=1,
                    accelerator=Accelerator.cpu, devices=["1"], strategy=Strategy.auto))
                residue = {"residue_output": "residue_codes", "scoring_system": run.prepared} if codes else {}
                self.assertEqual(calls["compute"], mock.call.compute(
                    run.embedding_computer, fasta_file=_FASTA.resolve(), min_res_n=100, batch_size=2,
                    num_workers=0, num_nodes=1, accelerator=Accelerator.cpu, devices=1, strategy=Strategy.auto,
                    preflight=run.preflight, **residue))
                self.assertEqual(calls["index"], mock.call.index(
                    run.embedding_computer, run.preflight, fasta_file=_FASTA.resolve(), output_db=str(self.tmp / "db"),
                    min_res_n=100, system=SYSTEM if codes else None,
                    index=search._IndexOptions(False, IndexType.auto, 256, None, 1000), refusals=run.refusals,
                    omp_env_at_start=os.environ.get("OMP_NUM_THREADS")))
                self.assertNotIn("split", calls["index"].kwargs)
                self.assertNotIn("after_sequence_store", calls["index"].kwargs)
                create_db = calls["create_db"].kwargs
                self.assertEqual((create_db["use_gpu_index"], create_db["index_type"], create_db["index_config"]),
                                 (False, IndexType.auto, IndexConfig(nlist=256, nprobe=None)))
                self.assertEqual(calls["store"].kwargs, {"fasta_file": _FASTA.resolve(), "min_res_n": 100})
                self.assertEqual(calls["cleanup"], mock.call.cleanup(run.embedding_computer))
                self.assertEqual(run.preflight.entries_kept, 1)
                self.assertEqual(ResidueCodeStore(self.tmp, "db").exists(), codes)
                self.assertEqual(list(self.shards.iterdir()), [])

    def test_the_index_phase_hook(self):
        """``after_sequence_store`` runs once the sequence store has passed its check and before the code store, inside
        the refusal wrapper it is given; a refused store check never reaches it. Without ``system`` the phase builds
        no code store and returns None."""
        prepared = ss.prepare(SYSTEM, local_files_only=True)
        with launch_env(FOLDMATCH_RUN_ID="hook"), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
            computer = EmbeddingComputer(self.shards)
            computer.compute_from_fasta(_FASTA, accelerator=Accelerator.cpu, residue_output="residue_codes",
                                        scoring_system=prepared)
        seen = []

        def build(db, hook, refusals=search._refusals, system=SYSTEM):
            return search._build_sequence_database(
                computer, computer.preflight, fasta_file=_FASTA, output_db=str(self.tmp / db), min_res_n=0,
                system=system, index=search._IndexOptions(False, IndexType.flat, 256, None, 1000), refusals=refusals,
                after_sequence_store=hook)

        def files(db):
            return sorted(p.name for p in self.tmp.glob(f"{db}.*"))

        manifest = build("db", lambda: seen.append(files("db")))
        self.assertEqual(seen, [["db.index", "db.metadata", "db.sequences"]])
        self.assertEqual(manifest["entries"], 2)
        self.assertTrue(ResidueCodeStore(self.tmp, "db").exists())
        self.assertIsNone(build("plain", lambda: seen.append(files("plain")), system=None))
        self.assertEqual(seen[-1], files("plain"))
        self.assertEqual(files("plain"), ["plain.index", "plain.metadata", "plain.sequences"])
        with mock.patch.object(search, "_check_store_matches_preflight", side_effect=ValueError("store mismatch")), \
                self.assertRaises(search.RefusalError):
            build("db2", lambda: seen.append("reached"))
        self.assertNotIn("reached", seen)

        def stat_fails():
            raise FileNotFoundError("the FASTA is gone")

        for refusals, raised in ((search._refusals, search.RefusalError), (search.nullcontext, FileNotFoundError)):
            with self.subTest(refusals=refusals.__name__):
                for path in self.tmp.glob("db3.*"):
                    path.unlink()
                with self.assertRaises(raised) as ctx:
                    build("db3", stat_fails, refusals=refusals)
                self.assertIn("the FASTA is gone", str(ctx.exception))
                self.assertFalse(ResidueCodeStore(self.tmp, "db3").exists())

    def test_the_mode_of_a_pass(self):
        """``_SequencePass``: the residue keywords and the refusal wrapper of each mode (D20)."""
        plain = search._SequencePass(None, None, None, False)
        self.assertEqual(plain.residue, {})
        with self.assertRaises(ValueError) as ctx, plain.refusals():
            raise ValueError("plain")
        self.assertIs(type(ctx.exception), ValueError)
        prepared = object()
        codes = search._SequencePass(None, None, prepared, True)
        self.assertEqual(codes.residue, {"residue_output": "residue_codes", "scoring_system": prepared})
        with self.assertRaises(search.RefusalError) as ctx, codes.refusals():
            raise FileExistsError("codes")
        self.assertIsInstance(ctx.exception.__cause__, FileExistsError)

    def test_a_fresh_index_config_per_build(self):
        """An ivf_pq build sets its config's nprobe: each build gets a new one."""
        options = search._IndexOptions(False, IndexType.ivf_pq, 256, None, 1000)
        first = options.index_config()
        first.nprobe = 4
        self.assertIsNot(options.index_config(), first)
        self.assertEqual(options.index_config(), IndexConfig(nlist=256, nprobe=None))

    def test_a_computer_bound_to_a_run(self):
        """``run`` binds the computer to that run's shards without drawing an id; a malformed id is refused."""
        (self.shards / "emb_x-0.parquet").touch()
        (self.shards / "emb_x-1-0.parquet").touch()
        with mock.patch("foldmatch.search.embedding_computer.run_id",
                        side_effect=AssertionError("run_id() called")), \
                launch_env(LOCAL_RANK=1):  # a rank > 0 without an id: run_id() would refuse
            computer = EmbeddingComputer(self.shards, run="x")
            self.assertEqual((computer.run_id, computer.out_name), ("x", "emb_x"))
            self.assertEqual(computer.shard_files(), [self.shards / "emb_x-0.parquet"])
            for bad in ("a b", "", "-x", "a/b", "a*"):
                with self.subTest(run=bad), self.assertRaises(ValueError) as ctx:
                    EmbeddingComputer(self.shards, run=bad)
                self.assertIn("is not a run id", str(ctx.exception))
        with launch_env(FOLDMATCH_RUN_ID="drawn"):
            self.assertEqual(EmbeddingComputer(self.shards).run_id, "drawn")

    def test_attach_preflight_sets_what_the_pass_set(self):
        with launch_env(FOLDMATCH_RUN_ID="a"), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
            passed = EmbeddingComputer(self.shards)
            passed.compute_from_fasta(_FASTA, min_res_n=100, accelerator=Accelerator.cpu)
        bound = EmbeddingComputer(self.shards, run="a")
        preflight = bound.attach_preflight(_FASTA, 100)
        self.assertIs(bound.preflight, preflight)
        self.assertEqual((bound.expected_count, bound.preflight), (passed.expected_count, passed.preflight))
        self.assertEqual(bound.preflight.offsets.tolist(), passed.preflight.offsets.tolist())
        self.assertEqual(bound.expected_count, 1)
        bound.check_embedding_count()  # the pass's shard, as rank 0 checks it
        self.assertEqual(bound.input_ids(), ["1acb_E"])
        # A given pre-flight is taken as it is, after its path and min_res_n are matched.
        other = EmbeddingComputer(self.shards, run="a")
        self.assertIs(other.attach_preflight(str(_FASTA), 100, passed.preflight), passed.preflight)
        copy = Path(shutil.copyfile(_FASTA, self.tmp / "copy.fasta"))
        for fasta, min_res_n in ((_FASTA, 0), (copy, 100)):
            with self.subTest(fasta=fasta, min_res_n=min_res_n):
                fresh = EmbeddingComputer(self.shards, run="a")
                with self.assertRaises(ValueError) as ctx:
                    fresh.attach_preflight(fasta, min_res_n, passed.preflight)
                self.assertIn("does not match", str(ctx.exception))
                self.assertEqual((fresh.expected_count, fresh.preflight), (None, None))
        # strict refuses what a plain pre-flight only logs; a residue pass is strict.
        dup = _write(self.tmp / "dup.fasta", ">a\nACDE\n>a\nACDE\n")
        self.assertEqual(EmbeddingComputer(self.shards, run="b").attach_preflight(dup, 0).duplicates, 1)
        with self.assertRaises(ValueError) as ctx:
            EmbeddingComputer(self.shards, run="b").attach_preflight(dup, 0, strict=True)
        self.assertIn("repeat an earlier id", str(ctx.exception))
        with launch_env(FOLDMATCH_RUN_ID="c"), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict",
                           side_effect=AssertionError("the pass ran")), self.assertRaises(ValueError) as ctx:
            EmbeddingComputer(self.shards).compute_from_fasta(dup, residue_output="residue_codes")
        self.assertIn("repeat an earlier id", str(ctx.exception))

    def logged_build(self, *extra, **kwargs):
        """The build's root-logger records (``set_log_level`` would replace assertLogs' handler)."""
        with mock.patch.object(search, "set_log_level"), self.assertLogs(level="INFO") as logs:
            result = self.build(*extra, **kwargs)
        self.assertEqual(result.exit_code, 0, result.output)
        return logs

    def test_index_phase_log_lines(self):
        logs = self.logged_build("--residue-codes")
        messages = [record.getMessage() for record in logs.records]
        phase = [m for m in messages if m.startswith("Index phase: ")]
        self.assertEqual(len(phase), 1, messages)
        import faiss
        omp = os.environ.get("OMP_NUM_THREADS", "unset")
        self.assertRegex(phase[0], rf"^Index phase: FAISS {re.escape(faiss.__version__)}, "
                                   rf"{faiss.omp_get_max_threads()} OpenMP thread\(s\); \d+ of {os.cpu_count()} CPUs "
                                   rf"allowed \({code_alignment._physical_cpus()} physical cores\); "
                                   rf"OMP_NUM_THREADS={omp}$")
        for what in ("FAISS index", "Sequence store", "Residue-code store"):
            with self.subTest(what=what):
                self.assertEqual(len([m for m in messages if re.fullmatch(rf"{what} built in \d+\.\d s", m)]), 1,
                                 messages)
        order = [next(k for k, m in enumerate(messages) if m.startswith(prefix))
                 for prefix in ("Index phase: ", "FAISS index built", "Sequence store built", "Residue-code store "
                                "written", "Residue-code store built")]
        self.assertEqual(order, sorted(order))
        self.assertFalse([m for m in messages if "stay allocated" in m])
        plain = [record.getMessage() for record in self.logged_build(run_id="plain", db="plain").records]
        self.assertFalse([m for m in plain if m.startswith("Residue-code store built")])

    def test_the_multi_node_hint(self):
        messages = [record.getMessage() for record in self.logged_build("--num-nodes", "2").records]
        hint = [m for m in messages if "stay allocated" in m]
        self.assertEqual(len(hint), 1, messages)
        self.assertIn("the other 1 node(s) stay allocated", hint[0])
        self.assertIn("'build sequences-embed' and then 'build sequences-index'", hint[0])
        self.assertLess(messages.index(hint[0]),
                        next(k for k, m in enumerate(messages) if m.startswith("Index phase")))
        # Ranks > 0 return after the pass: no index phase, no hint.
        with mock.patch.object(search, "_is_rank_zero", return_value=False), \
                mock.patch.object(search, "set_log_level"), self.assertLogs(level="INFO") as logs:
            result = self.build("--num-nodes", "2", run_id="rank1", db="rank1")
        self.assertEqual(result.exit_code, 0, result.output)
        messages = [record.getMessage() for record in logs.records]
        self.assertFalse([m for m in messages if "stay allocated" in m or m.startswith("Index phase")], messages)
        self.assertEqual(sorted(p.name for p in self.tmp.glob("rank1*")), [])

    def test_the_thread_warning(self):
        """A WARNING when FAISS has fewer OpenMP threads than the usable physical cores, naming the cause; never on
        macOS, where foldmatch pins OMP_NUM_THREADS=1 against the second libomp. OMP_NUM_THREADS unset when the
        command started (``omp_env_at_start=None``) and set at the index phase was set by the pass: Lightning's DDP
        launcher, which the INFO line and the WARNING name, with the remedy. With the variable unset, the cause is a
        limit set inside the process (OpenMP's default, one thread per allowed CPU, is never below the physical
        cores)."""
        pools = [{"user_api": "blas", "internal_api": "openblas", "num_threads": 4,
                  "filepath": "/opt/lib/libscipy_openblas64_-ff651d7f.so"}]
        not_given = search._OMP_ENV_NOT_RECORDED
        lightning = ("OMP_NUM_THREADS=2, which Lightning's DDP launcher set during the ESM3 pass (it was unset at the "
                     "start; Lightning sets it to the allowed CPUs // devices per node): the index phase may be slower "
                     "than this node allows. To choose FAISS's threads, export OMP_NUM_THREADS before the run (it also "
                     "sets each rank's torch threads in the pass), or build in two steps ('build sequences-embed', "
                     "then 'build sequences-index' as its own job).")
        inside = ("an OpenMP thread limit set inside this process or by another OpenMP variable (OMP_NUM_THREADS is "
                  "unset)")
        cases = (("linux", "2", not_given, "OMP_NUM_THREADS=2: the index phase may be slower than this node allows."),
                 ("linux", "2", "2", "OMP_NUM_THREADS=2: the index phase may be slower than this node allows."),
                 ("linux", "2", None, lightning),
                 ("linux", None, not_given, inside), ("linux", None, None, inside),
                 ("darwin", "2", None, None), ("darwin", "2", not_given, None), ("darwin", None, not_given, None))
        for platform, omp_env, at_start, cause in cases:
            env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
            if omp_env is not None:
                env["OMP_NUM_THREADS"] = omp_env
            args = () if at_start is not_given else (at_start,)
            with self.subTest(platform=platform, omp_env=omp_env, at_start=args), \
                    mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, "platform", platform), \
                    mock.patch("faiss.omp_get_max_threads", return_value=2), \
                    mock.patch.object(code_alignment, "_physical_cpus", return_value=8), \
                    mock.patch("threadpoolctl.threadpool_info", return_value=pools), \
                    self.assertLogs(level="INFO") as logs:
                search._log_index_phase(*args)
            messages = [(record.levelname, record.getMessage()) for record in logs.records]
            self.assertIn(("INFO", "Thread pool: openblas, 4 thread(s), libscipy_openblas64_-ff651d7f.so"), messages)
            self.assertIn("2 OpenMP thread(s)", messages[0][1])
            self.assertIn("(8 physical cores)", messages[0][1])
            set_by_pass = at_start is None and omp_env is not None
            suffix = (" (set during the ESM3 pass by Lightning: the allowed CPUs // devices per node; unset at the "
                      "start)")
            self.assertTrue(messages[0][1].endswith(f"OMP_NUM_THREADS={omp_env or 'unset'}"
                                                    + (suffix if set_by_pass else "")), messages[0][1])
            warnings = [m for level, m in messages if level == "WARNING"]
            if cause is not None:
                self.assertEqual(len(warnings), 1, messages)
                self.assertEqual(warnings[0], f"FAISS has 2 OpenMP thread(s) for 8 usable physical cores, because of "
                                              f"{cause}" + ("" if cause.endswith(".") else
                                                            ": the index phase may be slower than this node allows."))
                self.assertEqual("Lightning" in warnings[0], set_by_pass)
            else:
                self.assertEqual(warnings, [])
        with mock.patch.object(sys, "platform", "linux"), mock.patch("faiss.omp_get_max_threads", return_value=8), \
                mock.patch.object(code_alignment, "_physical_cpus", return_value=8), \
                mock.patch("threadpoolctl.threadpool_info", return_value=pools), \
                self.assertLogs(level="INFO") as logs:
            search._log_index_phase(None)
        self.assertEqual([r for r in logs.records if r.levelno >= logging.WARNING], [])

    def test_omp_num_threads_as_the_command_found_it(self):
        """The one-shot reads OMP_NUM_THREADS before its pass and gives that value to the index phase: a value the pass
        set (as Lightning's DDP launcher does when the variable is unset) is logged as set during the pass; one set
        before the command is not."""
        real = search._run_sequence_pass

        def setting(*args, **kwargs):  # what Lightning's launcher does during a multi-device pass
            real(*args, **kwargs)
            os.environ.setdefault("OMP_NUM_THREADS", "3")

        for at_start in (None, "3"):
            with self.subTest(at_start=at_start), mock.patch.dict(os.environ), \
                    mock.patch.object(search, "_run_sequence_pass", side_effect=setting), \
                    mock.patch.object(search, "_log_index_phase", wraps=search._log_index_phase) as phase:
                os.environ.pop("OMP_NUM_THREADS", None)
                if at_start is not None:
                    os.environ["OMP_NUM_THREADS"] = at_start
                messages = [record.getMessage() for record in self.logged_build(
                    run_id=f"omp{at_start}", db=f"omp{at_start}").records]
            self.assertEqual(phase.call_args, mock.call(at_start))
            line = next(m for m in messages if m.startswith("Index phase: "))
            self.assertEqual(line.endswith("OMP_NUM_THREADS=3 (set during the ESM3 pass by Lightning: the allowed CPUs "
                                           "// devices per node; unset at the start)"), at_start is None, line)
            self.assertTrue(line.endswith("OMP_NUM_THREADS=3") or at_start is None, line)

    def test_threadpoolctl_failures_do_not_fail_a_build(self):
        with mock.patch("threadpoolctl.threadpool_info", side_effect=RuntimeError("cannot inspect")), \
                self.assertLogs(level="INFO") as logs:
            search._log_index_phase()
        self.assertFalse([r for r in logs.records if r.getMessage().startswith("Thread pool: ")])

    def test_a_single_threaded_faiss_blas(self):
        """One line when FAISS's BLAS runs on one thread by build, a WARNING (INFO on macOS) naming the library and the
        pinned faiss-cpu: the faiss-cpu 1.13.2 wheel's OpenBLAS in faiss_cpu.libs (threading layer disabled), FAISS's
        own BLAS whose threading layer threadpoolctl does not report, any BLAS built without threads; the first one
        only. None for a threaded BLAS (MKL at 128 threads, numpy's OpenBLAS limited to one thread, the faiss-cpu
        1.15.1 wheel's OpenMP OpenBLAS at one thread or more), and none when threadpoolctl finds no BLAS."""
        import faiss
        package = os.path.dirname(os.path.realpath(faiss.__file__))
        libs = os.path.join(os.path.dirname(package), "faiss_cpu.libs")

        def blas(path, layer, threads, api="openblas"):
            runtime = {"user_api": "blas", "internal_api": api, "num_threads": threads, "filepath": path,
                       "version": "0.3.15"}
            if layer is not None:
                runtime["threading_layer"] = layer
            return runtime

        gomp = {"user_api": "openmp", "internal_api": "openmp", "num_threads": 8, "version": None,
                "filepath": os.path.join(libs, "libgomp-e985bcbb.so.1.0.0")}
        wheel = blas(os.path.join(libs, "libopenblas-r0-11edc3fa.3.15.so"), "disabled", 1)
        serial = blas("/opt/conda/lib/libopenblasp-r0.3.21.so", "disabled", 1)
        pinned = os.path.join(libs, "libopenblaso-r0-d77a1985.3.15.so")
        cases = {
            "the faiss-cpu 1.13.2 wheel": ([gomp, wheel], "libopenblas-r0-11edc3fa.3.15.so"),
            "in the package, no threading layer": (
                [blas(os.path.join(package, ".dylibs", "libopenblas.0.dylib"), None, 1)], "libopenblas.0.dylib"),
            "in faiss_cpu.libs, an unknown layer": ([blas(os.path.join(libs, "libopenblas.so"), "unknown", 1)],
                                                    "libopenblas.so"),
            "in faiss_cpu.libs, no threading layer, 8 threads": (
                [blas(os.path.join(libs, "libopenblas.so"), None, 8)], None),
            "built without threads elsewhere": ([serial], "libopenblasp-r0.3.21.so"),
            "two of them": ([wheel, serial], "libopenblas-r0-11edc3fa.3.15.so"),
            "MKL at 128 threads": ([gomp, blas("/envs/faisscf/lib/libmkl_rt.so.2", "intel", 128, api="mkl")], None),
            "numpy's OpenBLAS limited to one thread": (
                [blas("/sp/numpy.libs/libscipy_openblas64_-ff651d7f.so", "pthreads", 1)], None),
            "the faiss-cpu 1.15.1 wheel's OpenMP OpenBLAS at one thread": ([blas(pinned, "openmp", 1)], None),
            "the faiss-cpu 1.15.1 wheel's OpenMP OpenBLAS": ([gomp, blas(pinned, "openmp", 64)], None),
            "no BLAS": ([gomp], None),
            "nothing found": ([], None),
        }
        text = ("FAISS's BLAS, {}, is single-threaded, so the ivf_pq training and adds run on one core whatever "
                "FAISS's thread count: install the faiss-cpu 1.15.1 FoldMatch pins (pip install faiss-cpu==1.15.1), "
                "whose wheel bundles an OpenMP OpenBLAS, or a FAISS built on MKL (README, 'FAISS threads').")
        for case, (pools, name) in cases.items():
            for platform in ("linux", "darwin"):
                with self.subTest(case=case, platform=platform), mock.patch.object(sys, "platform", platform), \
                        mock.patch("faiss.omp_get_max_threads", return_value=8), \
                        mock.patch.object(code_alignment, "_physical_cpus", return_value=8), \
                        mock.patch("threadpoolctl.threadpool_info", return_value=pools), \
                        self.assertLogs(level="INFO") as logs:
                    search._log_index_phase(None)
                lines = [(record.levelname, record.getMessage()) for record in logs.records
                         if "single-threaded" in record.getMessage()]
                self.assertEqual(lines, [] if name is None else
                                 [("INFO" if platform == "darwin" else "WARNING", text.format(name))])
                self.assertEqual(len([record for record in logs.records if record.getMessage().startswith(
                    "Thread pool: ")]), len(pools))
                self.assertEqual([record.getMessage() for record in logs.records if record.levelno >= logging.WARNING
                                  and "single-threaded" not in record.getMessage()], [])  # FAISS has its 8 threads

    def widen(self, *args):
        """``_expand_cpu_affinity(*args)`` widening 1 CPU to 4 (affinity calls mocked); its log lines."""
        with mock.patch.object(os, "cpu_count", return_value=4), \
                mock.patch.object(os, "sched_setaffinity", create=True) as setaffinity, \
                mock.patch.object(os, "sched_getaffinity", create=True, side_effect=[{0}, {0, 1, 2, 3}]), \
                self.assertLogs("foldmatch.search.alignment", level="INFO") as logs:
            self.assertEqual(alignment._expand_cpu_affinity(*args), 4)
        setaffinity.assert_called_once_with(0, range(4))
        return logs.output

    def test_the_default_affinity_message_is_unchanged(self):
        self.assertEqual(self.widen(),
                         ["INFO:foldmatch.search.alignment:Widened CPU affinity for alignment: 1 -> 4 CPUs"])

    def test_the_affinity_message_names_its_purpose(self):
        self.assertEqual(self.widen("the index phase"),
                         ["INFO:foldmatch.search.alignment:Widened CPU affinity for the index phase: 1 -> 4 CPUs"])


# =====================================================================================================================
# P8b.3: build sequences-embed
# =====================================================================================================================
#: Plan P8b (CLI), verbatim: the command help, and the help texts step 1 does not take from 'build sequences'. The
#: command help's sbatch example also gives --kill-on-invalid-dep=yes (work/p8b/deviations.md, P8b.3).
EMBED_HELP = (
    "Step 1 of a split 'build sequences' for multi-node GPU runs: the pre-flight and the ESM3 pass (chain embeddings, "
    "and residue codes with --residue-codes) on every rank, writing the embedding shards and, last, a run record into "
    "--shards-folder. No database file is written. Then build the database with 'build sequences-index' as its own "
    "one-process job on a CPU node (e.g. sbatch --dependency=afterok:<this job> --kill-on-invalid-dep=yes), so that "
    "these nodes are released."
)
EMBED_OUTPUT_DB_HELP = (
    "Path of the database 'build sequences-index' will build (as --output-db of 'build sequences'). Refused if any of "
    "its files exist; recorded in the run record."
)
SHARDS_FOLDER_HELP = (
    "Folder for this step's output: the embedding shards (emb_<run id>-<rank>.parquet) and, written last, the run "
    "record (emb_<run id>.run.json) that 'build sequences-index --from-shards' reads. It must be on storage every node "
    "can write and that outlives the job (e.g. $SCRATCH), with room for about 6 KiB per entry plus 8 B per residue "
    "with --residue-codes. Use one folder per build."
)
#: Five entries: two and three ranks get different deals.
_FIVE = ">e0\nACDEFGHIKL\n>e1 second\nKLMNPQRSTV\n>e2\nVWYACDEFGH\n>e3\nIKLMNPQRST\n>e4\nACDEFGHIKLMN\n"
#: A rank that starts the run, and a rank Lightning or srun starts (it shares the run id it is given).
_RANK_ENVS = ({}, {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"})


def _embed_args(fasta, db, shards, *extra):
    return ("build", "sequences-embed", "--fasta-file", fasta, "--output-db", db, "--shards-folder", shards,
            "--accelerator", "cpu", "--devices", "1", *extra)


def _ranks(n, change=None):
    """A ``_gather_rank_facts`` of an ``n``-rank world: rank 0's facts, and a copy for every other rank, rank 1's
    updated with ``change(facts)``."""
    def gather(facts):
        others = [dict(facts) for _ in range(n - 1)]
        if change is not None:
            others[0].update(change(facts))
        return [facts, *others]
    return gather


def _flat(output: str) -> str:
    """Typer's help output with the panel borders dropped and the whitespace collapsed."""
    return " ".join(output.translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


def _command(*path):
    import typer
    command = typer.main.get_command(search.app)
    for name in path:
        command = command.commands[name]
    return command


def _rewrite(path, change):
    """Rewrite the shard at ``path`` as ``change(table)``."""
    pq.write_table(change(pq.read_table(path)), path)


def _with_pass(**changes):
    """A table change: the shard's pass record (``foldmatch.pass``) with ``changes``."""
    def change(table):
        meta = dict(table.schema.metadata)
        record = {**json.loads(meta[META_PASS]), **changes}
        meta[META_PASS] = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii")
        return table.replace_schema_metadata(meta)
    return change


class EmbedOptionTest(_Tmp):

    def test_the_command_help(self):
        self.assertEqual(_command("build", "sequences-embed").help, EMBED_HELP)
        output = CliRunner().invoke(search.app, ["build", "sequences-embed", "--help"], env={"COLUMNS": "400"}).output
        for text in (EMBED_HELP, EMBED_OUTPUT_DB_HELP, SHARDS_FOLDER_HELP, PLAN_HELP):
            with self.subTest(text=text[:40]):
                self.assertIn(text, _flat(output))

    def test_the_options(self):
        """Exactly the step-1 options; each one the type, default and help of 'build sequences' unless the plan gives
        its own help (--output-db, --shards-folder, which is --tmp-embedding-folder renamed). --min-res-n refuses a
        negative value (a record holds no negative count)."""
        embed = {param.name: param for param in _command("build", "sequences-embed").params}
        oneshot = {param.name: param for param in _command("build", "sequences").params}
        self.assertEqual(sorted(embed), sorted([
            "fasta_file", "output_db", "shards_folder", "min_res_n", "accelerator", "num_nodes", "devices", "strategy",
            "batch_size", "num_workers", "residue_codes", "log_level"]))
        own_help = {"output_db": EMBED_OUTPUT_DB_HELP, "shards_folder": SHARDS_FOLDER_HELP}

        def info(param, *drop):
            out = param.to_info_dict()
            for key in ("name", "opts", "help", *drop):
                out.pop(key)
            return out

        for name, param in embed.items():
            with self.subTest(name=name):
                same = oneshot["tmp_embedding_folder" if name == "shards_folder" else name]
                self.assertEqual(param.help, own_help.get(name, same.help))
                if name == "min_res_n":
                    self.assertEqual(info(param, "type"), info(same, "type"))
                    self.assertEqual((param.type.min, param.type.max, param.default), (0, None, 0))
                else:
                    self.assertEqual(info(param), info(same))
        self.assertEqual(embed["shards_folder"].opts, ["--shards-folder"])
        self.assertTrue(embed["shards_folder"].required)
        self.assertEqual(embed["residue_codes"].help, PLAN_HELP)
        codes = embed["residue_codes"]
        self.assertEqual((codes.opts, codes.secondary_opts, codes.default),
                         (["--residue-codes"], ["--no-residue-codes"], False))

    def test_options_another_command_owns(self):
        """The index phase's options (step 2 takes them), --cleanup-embeddings (the shards are this step's output),
        --tmp-embedding-folder (renamed --shards-folder) and step 2's run options: exit 2, nothing read or written."""
        for option in (("--use-gpu-index",), ("--index-type", "flat"), ("--ivf-nlist", "256"), ("--ivf-nprobe", "4"),
                       ("--load-batch-size", "1000"), ("--cleanup-embeddings",),
                       ("--tmp-embedding-folder", self.shards), ("--from-shards", self.shards), ("--run-id", "x")):
            with self.subTest(option=option[0]), _esm3_spy() as esm3:
                result = invoke(*_embed_args(_FASTA, self.tmp / "db", self.shards, *option))
                self.usage_error(result, f"No such option: {option[0]}")
                esm3.assert_not_called()
        self.usage_error(invoke(*_embed_args(_FASTA, self.tmp / "db", self.shards, "--min-res-n", "-1")),
                         "--min-res-n")
        self.usage_error(invoke("build", "sequences-embed", "--fasta-file", _FASTA, "--output-db", self.tmp / "db"),
                         "--shards-folder")
        self.assertEqual(list(self.shards.iterdir()), [])
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db*")), [])


class EmbedStepTest(_WithBundle):

    def embed(self, *extra, ranks=1, run_id="p8b3", fasta=_FASTA, db=None, shards=None, gather=None, **fake):
        """Step 1 with the fake pass on ``ranks`` ranks (``gather``, default :func:`_ranks`, stands in for the
        all-gather of more than one rank)."""
        db = db or self.tmp / "db"
        with launch_env(FOLDMATCH_RUN_ID=run_id), ExitStack() as stack:
            stack.enter_context(mock.patch("foldmatch.search.embedding_computer.chain_predict",
                                           fake_chain_predict(ranks=ranks, **fake)))
            if gather is not None or ranks > 1:
                stack.enter_context(mock.patch.object(search, "_gather_rank_facts",
                                                      side_effect=gather or _ranks(ranks)))
            return invoke(*_embed_args(fasta, db, shards or self.shards, *extra))

    def logged_embed(self, *extra, **kwargs):
        """The step's root-logger messages (``set_log_level`` would replace assertLogs' handler)."""
        with mock.patch.object(search, "set_log_level"), self.assertLogs(level="INFO") as logs:
            result = self.embed(*extra, **kwargs)
        self.assertEqual(result.exit_code, 0, result.output)
        return [record.getMessage() for record in logs.records]

    def test_the_run_record(self):
        """Plain and codes, on 1 and 2 ranks, with --min-res-n 0 and 100: the shards and the record are the only files
        (no database file), and every field of the record holds what the run did."""
        stat = split_run.fasta_stat(_FASTA)
        sha256 = hashlib.sha256(_FASTA.read_bytes()).hexdigest()
        system = {"name": SYSTEM.name, "digest": SYSTEM.digest(), "encoding_digest": SYSTEM.encoding_digest()}
        for codes in (False, True):
            for ranks in (1, 2):
                for min_res_n in (0, 100):
                    name = f"{'codes' if codes else 'plain'}{ranks}-{min_res_n}"
                    with self.subTest(codes=codes, ranks=ranks, min_res_n=min_res_n):
                        shards = self.tmp / name
                        shards.mkdir()
                        db = self.tmp / f"db-{name}"
                        result = self.embed("--min-res-n", min_res_n, *(("--residue-codes",) if codes else ()),
                                            ranks=ranks, run_id=name, db=db, shards=shards)
                        self.assertEqual(result.exit_code, 0, result.output)
                        kept = 2 if min_res_n == 0 else 1
                        deal = split_run.dealt_rows(kept, ranks)
                        names = [f"emb_{name}-{rank}.parquet" for rank in sorted(deal)]
                        self.assertEqual(sorted(p.name for p in shards.iterdir()), names + [f"emb_{name}.run.json"])
                        self.assertEqual(sorted(p.name for p in self.tmp.glob(f"db-{name}*")), [])
                        path = shards / f"emb_{name}.run.json"
                        record = json.loads(path.read_text())
                        self.assertEqual(split_run.read_run_record(path), record)
                        created = datetime.strptime(record.pop("created_utc"), "%Y-%m-%dT%H:%M:%SZ")
                        self.assertLess(abs(datetime.now(timezone.utc).replace(tzinfo=None) - created),
                                        timedelta(minutes=5))
                        run_pass = record.pop("pass")
                        self.assertEqual(record, {
                            "kind": "foldmatch-sequence-run", "record_version": 1, "run_id": name,
                            "command": "build sequences-embed", "sealed": False,
                            "foldmatch_version": foldmatch.__version__,
                            "fasta": {"path": str(_FASTA.resolve()), "bytes": stat.bytes, "mtime_ns": stat.mtime_ns,
                                      "sha256": sha256},
                            "min_res_n": min_res_n, "strict": codes,
                            "preflight": split_run.preflight_counts(preflight_fasta(_FASTA, min_res_n)),
                            "output_db": os.path.abspath(db),
                            "residue_output": "residue_codes" if codes else None,
                            "system": system if codes else None,
                            "world_size": ranks,
                            "options": {"accelerator": "cpu", "devices": ["1"], "num_nodes": 1, "strategy": "auto",
                                        "batch_size": 1, "num_workers": 0},
                            "shards": [{"name": n, "rank": rank, "rows": deal[rank],
                                        "bytes": (shards / n).stat().st_size} for rank, n in zip(sorted(deal), names)],
                            "rows": kept,
                        })
                        counts = record["preflight"]
                        self.assertEqual((counts["entries"], counts["entries_kept"], counts["residues"]),
                                         (2, 2, 303) if min_res_n == 0 else (2, 1, 245))
                        if codes:
                            self.assertEqual(run_pass, read_shard_info(shards / names[0]).pass_record)
                            self.assertEqual((run_pass["accelerator"], run_pass["world_size"], run_pass["min_res_n"],
                                              run_pass["esm_revision"]), ("cpu", ranks, min_res_n, ESM_REVISION))
                        else:
                            self.assertEqual(run_pass, pass_record(min_res_n=min_res_n, esm_revision=ESM_REVISION,
                                                                   world_size=ranks))
                            self.assertIsNone(run_pass["accelerator"])

    def test_the_pass_of_a_cuda_run(self):
        """A codes run records its shards' pass (the device type and its ESM3 precision); a plain one records no
        device."""
        for codes in (True, False):
            with self.subTest(codes=codes):
                shards = self.tmp / f"cuda-{codes}"
                shards.mkdir()
                result = self.embed(*(("--residue-codes",) if codes else ()), ranks=2, device="cuda", shards=shards,
                                    run_id=f"cuda-{codes}", db=self.tmp / f"cuda-{codes}-db")
                self.assertEqual(result.exit_code, 0, result.output)
                run_pass = split_run.read_run_record(shards / f"emb_cuda-{codes}.run.json")["pass"]
                if codes:
                    self.assertEqual((run_pass["accelerator"], run_pass["esm_precision"], run_pass["world_size"]),
                                     ("cuda", "bf16-autocast", 2))
                else:
                    self.assertEqual((run_pass["accelerator"], run_pass["esm_precision"], run_pass["world_size"]),
                                     (None, None, 2))

    def test_every_rank_gathers_its_facts_after_the_pass(self):
        """After the pass and before the rank-0 check, every rank gathers what it saw: its FASTA stat before the
        pre-flight and after the pass, no stat error, the size, its pre-flight counts and the pre-flight's seconds."""
        manager = mock.Mock()
        with mock.patch.object(search, "_run_sequence_pass", wraps=search._run_sequence_pass) as run_pass, \
                mock.patch.object(search, "_gather_rank_facts", wraps=search._gather_rank_facts) as gather, \
                mock.patch.object(search, "_is_rank_zero", wraps=search._is_rank_zero) as rank_zero:
            for name, spy in (("run_pass", run_pass), ("gather", gather), ("rank_zero", rank_zero)):
                manager.attach_mock(spy, name)
            result = self.embed("--residue-codes")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual([c[0] for c in manager.mock_calls], ["run_pass", "gather", "rank_zero"])
        facts = gather.call_args.args[0]
        self.assertEqual(sorted(facts), ["after", "before", "error", "preflight", "seconds", "size"])
        stat = split_run.fasta_stat(_FASTA)
        self.assertEqual((facts["before"], facts["after"], facts["error"], facts["size"]),
                         (stat, stat, None, stat.bytes))
        self.assertEqual(facts["preflight"], split_run.preflight_counts(preflight_fasta(_FASTA, 0)))
        self.assertGreaterEqual(facts["seconds"], 0)

    def test_gather_rank_facts(self):
        """``[facts]`` without a process group; ``all_gather_object`` over the group's ranks with one."""
        self.assertEqual(search._gather_rank_facts({"rank": 0}), [{"rank": 0}])
        import torch.distributed as dist

        def all_gather_object(out, obj):
            out[:] = [obj, {"rank": 1}]

        with mock.patch.object(dist, "is_available", return_value=True), \
                mock.patch.object(dist, "is_initialized", return_value=True), \
                mock.patch.object(dist, "get_world_size", return_value=2), \
                mock.patch.object(dist, "all_gather_object", side_effect=all_gather_object) as gathered:
            self.assertEqual(search._gather_rank_facts({"rank": 0}), [{"rank": 0}, {"rank": 1}])
        self.assertEqual(gathered.call_args.args, ([{"rank": 0}, {"rank": 1}], {"rank": 0}))

    def test_ranks_above_zero_write_no_record(self):
        """A rank above 0 gathers its facts and exits 0 once the pass is over: no record, and nothing new logged."""
        with mock.patch.object(search, "_is_rank_zero", return_value=False), \
                mock.patch.object(search, "_gather_rank_facts", wraps=search._gather_rank_facts) as gather, \
                mock.patch.object(search, "set_log_level"), self.assertLogs(level="INFO") as logs:
            result = self.embed("--residue-codes")
        self.assertEqual(result.exit_code, 0, result.output)
        gather.assert_called_once()
        self.assertEqual([p.name for p in self.shards.iterdir()], ["emb_p8b3-0.parquet"])
        messages = [record.getMessage() for record in logs.records]
        self.assertFalse([m for m in messages if m.startswith(("Pre-flight on", "FASTA sha256", "Run p8b3",
                                                               "Build the database", "Add --index-type"))], messages)

    def test_the_log_lines(self):
        """Rank 0 logs the ranks' pre-flight times, the FASTA hash, the run, the next command and the index hint, in
        this order."""
        size = _FASTA.stat().st_size
        cases = ((("--residue-codes",), 1, f"2 entries in 1 shard(s) of 1 rank(s), residue codes of {SYSTEM.name}"),
                 ((), 2, "2 entries in 2 shard(s) of 2 rank(s)"))
        for extra, ranks, run in cases:
            with self.subTest(ranks=ranks):
                for path in self.shards.iterdir():
                    path.unlink()
                messages = self.logged_embed(*extra, ranks=ranks)
                prefixes = ("Pre-flight on ", "FASTA sha256 of ", "Run p8b3 complete: ", "Build the database with: ",
                            "Add --index-type")
                lines = [[m for m in messages if m.startswith(prefix)] for prefix in prefixes]
                self.assertEqual([len(found) for found in lines], [1] * 5, messages)
                lines = [found[0] for found in lines]
                self.assertRegex(lines[0], rf"^Pre-flight on {ranks} rank\(s\): min \d+\.\d s, max \d+\.\d s$")
                self.assertRegex(lines[1], rf"^FASTA sha256 of {re.escape(str(_FASTA.resolve()))} \({size} bytes\) "
                                           rf"in \d+\.\d s$")
                self.assertEqual(lines[2], f"Run p8b3 complete: {run}; record "
                                           f"{self.shards.resolve() / 'emb_p8b3.run.json'}")
                self.assertIn(f"auto picks hnsw at {FaissEmbeddingDatabase.N_EMBEDDINGS_DB_THR:,} entries or more",
                              lines[4])
                self.assertIn("about 12 KiB per entry at peak; ivf_pq is bounded", lines[4])
                order = [messages.index(line) for line in lines]
                self.assertEqual(order, sorted(order))
                self.assertFalse([m for m in messages if "stay allocated" in m or m.startswith("Index phase")])

    def test_the_next_command(self):
        """The logged next command, split as a shell splits it, is step 2 on this run: the FASTA, the database path the
        record holds (absolute, also for a relative --output-db), the shards folder and the run id, with paths that
        hold spaces. (Running it needs 'build sequences-index', P8b.4.)"""
        folder = self.tmp / "with space"
        folder.mkdir()
        fasta = Path(shutil.copyfile(_FASTA, folder / "my seqs.fasta"))
        shards = folder / "shard s"
        shards.mkdir()
        for db, run_id in ((folder / "my db", "abs"), (Path("rel dir") / "my db", "rel")):
            with self.subTest(db=str(db)), chdir(folder):
                messages = self.logged_embed(fasta=fasta, db=db, shards=shards, run_id=run_id)
                line = next(m for m in messages if m.startswith("Build the database with: "))
                argv = shlex.split(line[len("Build the database with: "):])
                record = split_run.read_run_record(shards / f"emb_{run_id}.run.json")
                self.assertEqual(record["output_db"], str(folder.resolve() / db) if run_id == "rel" else str(db))
                self.assertEqual(argv, ["fm-search", "build", "sequences-index", "--fasta-file", str(fasta.resolve()),
                                        "--output-db", record["output_db"], "--from-shards", str(shards.resolve()),
                                        "--run-id", run_id])


class EmbedRefusalTest(_WithBundle):

    def refused_before_esm3(self, *extra, fasta=_FASTA, codes=False, prepare=None, cause=None):
        """Step 1 with the real pass under the ESM3 spy, refused before ESM3 (exit 1, a RefusalError), the bundle
        untouched unless ``prepare`` (a patch, or ``nullcontext()`` for the real one) says otherwise: no file written,
        the shards folder unchanged. Returns the message."""
        before = {p.name: p.read_bytes() for p in self.shards.iterdir()}
        args = _embed_args(fasta, self.tmp / "db", self.shards, *extra, *(("--residue-codes",) if codes else ()))
        with _esm3_spy() as esm3, prepare or mock.patch.object(ss, "prepare",
                                                               side_effect=AssertionError("bundle fetched")):
            result = invoke(*args)
        esm3.assert_not_called()
        message = self.refused(result, cause)
        self.assertEqual({p.name: p.read_bytes() for p in self.shards.iterdir()}, before)
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db*")), [])
        return message

    @staticmethod
    def every_rank(modes=(False, True), envs=_RANK_ENVS):
        """``(codes, rank env)`` for each mode, on rank 0 and on a rank started with the run id (by default)."""
        return [(codes, rank_env) for codes in modes for rank_env in envs]

    def test_e1_fasta_the_pre_flight_refuses(self):
        """A header without an id in either mode; duplicate ids, '<'/'>' and non-ASCII under codes. A plain run takes
        those, as 'build sequences' does (the pass is reached)."""
        cases = [(">\nACDE\n>b\nKLMN\n", "is empty", False), (">\nACDE\n>b\nKLMN\n", "is empty", True),
                 (">a\nACDE\n>b\nKLM\n>a\nACDE\n", "repeat an earlier id", True),
                 (">a\nAC<mask>DE\n", "'<' or '>'", True), (">a\nACDÉ\n", "non-ASCII", True)]
        for text, what, codes in cases:
            fasta = _write(self.tmp / "bad.fasta", text)
            for _, rank_env in self.every_rank(modes=(codes,)):
                with self.subTest(what=what, codes=codes, rank_env=rank_env), launch_env(**rank_env):
                    self.assertIn(what, self.refused_before_esm3(fasta=fasta, codes=codes, cause=ValueError))
        with launch_env(FOLDMATCH_RUN_ID="dup"), _esm3_spy() as esm3:
            result = invoke(*_embed_args(_write(self.tmp / "dup.fasta", ">a\nACDE\n>a\nACDE\n"), self.tmp / "db",
                                         self.shards))
        self.assertIsInstance(result.exception, AssertionError)  # the pass was reached
        esm3.assert_called_once()

    def test_e2_an_output_exists(self):
        for suffix in (".index", ".metadata", ".invlists", ".sequences", ".rescodes.bin", ".rescodes.idx",
                       ".rescodes.json", ".rescodes.json.tmp"):
            existing = self.tmp / f"db{suffix}"
            for codes, rank_env in self.every_rank():
                with self.subTest(suffix=suffix, codes=codes, rank_env=rank_env), launch_env(**rank_env):
                    existing.touch()
                    try:
                        with _esm3_spy() as esm3, mock.patch.object(ss, "prepare") as prepare:
                            result = invoke(*_embed_args(_FASTA, self.tmp / "db", self.shards,
                                                         *(("--residue-codes",) if codes else ())))
                    finally:
                        existing.unlink()
                    message = self.refused(result, FileExistsError)
                    self.assertIn(f"Database already exists at {self.tmp / 'db'} (found: {existing})", message)
                    self.assertIn("Choose a different --output-db", message)
                    esm3.assert_not_called()
                    prepare.assert_not_called()
                    self.assertEqual(list(self.shards.iterdir()), [])

    def test_e3_nothing_to_build(self):
        for codes, rank_env in self.every_rank():
            with self.subTest(codes=codes, rank_env=rank_env), launch_env(**rank_env):
                self.assertIn("there is nothing to build",
                              self.refused_before_esm3("--min-res-n", "1000", codes=codes, cause=ValueError))

    def test_e4_no_run_id(self):
        """A malformed FOLDMATCH_RUN_ID; a rank above 0 with no run id to share."""
        cases = [*self.every_rank(envs=({"FOLDMATCH_RUN_ID": "a b"}, {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "a b"})),
                 *self.every_rank(envs=({"LOCAL_RANK": 1},))]
        for codes, rank_env in cases:
            with self.subTest(codes=codes, rank_env=rank_env), launch_env(**rank_env):
                what = "is not a valid run id" if "FOLDMATCH_RUN_ID" in rank_env else "has no run id to share"
                self.assertIn(what, self.refused_before_esm3(codes=codes, cause=ValueError))

    def test_e5_shards_of_the_same_run_id(self):
        (self.shards / "emb_r-0.parquet").write_bytes(b"an earlier run's shard")
        for codes, rank_env in self.every_rank(envs=({"FOLDMATCH_RUN_ID": "r"}, _RANK_ENVS[1])):
            with self.subTest(codes=codes, rank_env=rank_env), launch_env(**rank_env):
                self.assertIn("same run id", self.refused_before_esm3(codes=codes, cause=FileExistsError))

    def test_e6_a_record_of_the_same_run_id(self):
        """A record, or a staged .tmp, of an earlier run with this id: refused after the bundle check, before ESM3."""
        for name, what in (("emb_r.run.json", "a run record of an earlier run with the same run id"),
                           ("emb_r.run.json.tmp", "stopped while it wrote its run record")):
            (self.shards / name).write_text("{}\n")
            for codes, rank_env in self.every_rank(envs=({"FOLDMATCH_RUN_ID": "r"}, _RANK_ENVS[1])):
                with self.subTest(name=name, codes=codes, rank_env=rank_env), launch_env(**rank_env):
                    message = self.refused_before_esm3(codes=codes, prepare=nullcontext())
                    self.assertIn(what, message)
                    self.assertIn(str(self.shards.resolve() / name), message)
            (self.shards / name).unlink()

    def test_e7_the_bundle(self):
        """A bundle that cannot be fetched; a bundle missing from the cache of a rank that must find it cached (a rank
        above 0, every rank of a multi-node run): the refusal names download-models."""
        for _, rank_env in self.every_rank(modes=(True,)):
            with self.subTest(rank_env=rank_env), launch_env(**rank_env):
                unavailable = mock.patch.object(ss, "prepare", side_effect=ss.BundleUnavailableError("not cached"))
                self.assertIn("not cached", self.refused_before_esm3(codes=True, prepare=unavailable,
                                                                     cause=ss.BundleUnavailableError))
        prepare = ss.prepare
        empty = self.tmp / "empty-hf-cache"
        empty.mkdir()

        def empty_cache(system, *, local_files_only):
            self.assertTrue(local_files_only)
            return prepare(system, cache_dir=empty, local_files_only=local_files_only)

        for extra, rank_env in (((), _RANK_ENVS[1]), (("--num-nodes", "2"), {})):
            with self.subTest(extra=extra, rank_env=rank_env), launch_env(**rank_env):
                message = self.refused_before_esm3(*extra, codes=True, cause=ss.BundleUnavailableError,
                                                   prepare=mock.patch.object(ss, "prepare", side_effect=empty_cache))
                self.assertIn("download-models", message)

    def test_e8_the_residue_pass(self):
        """The pass's own refusal before ESM3 loads (check_residue_pass): a prepared system that is not this
        FoldMatch's registry entry."""
        foreign = ss.PreparedSystem(system=OTHER_ENCODING, files={}, bundle={}, encoders_file="e", scoring_file="s")
        for _, rank_env in self.every_rank(modes=(True,)):
            with self.subTest(rank_env=rank_env), launch_env(**rank_env):
                message = self.refused_before_esm3(codes=True, cause=ss.UnknownScoringSystemError,
                                                   prepare=mock.patch.object(ss, "prepare", return_value=foreign))
                self.assertIn(OTHER_ENCODING.name, message)

    # -- rank 0, after the pass ---------------------------------------------------------------------------------------
    def post_pass(self, *extra, codes=True, ranks=1, gather=None, predict=None, fasta=_FASTA, foreign=(), **fake):
        """Step 1 with the fake pass (``predict`` wraps or replaces it), refused by rank 0 after the pass: exit 1, a
        RefusalError saying that no record was written and where the shards are kept; no record and no .tmp but the
        ``foreign`` files the pass put there; the shards as the pass left them, byte for byte; no database file.
        Returns the message."""
        for path in self.shards.iterdir():
            path.unlink()
        written = {}
        base = predict or fake_chain_predict(ranks=ranks, **fake)

        def recording(**kwargs):
            base(**kwargs)
            written.update({p.name: p.read_bytes() for p in Path(kwargs["out_folder"]).glob("*.parquet")})

        with launch_env(FOLDMATCH_RUN_ID="e"), ExitStack() as stack:
            stack.enter_context(mock.patch("foldmatch.search.embedding_computer.chain_predict", recording))
            if gather is not None or ranks > 1:
                stack.enter_context(mock.patch.object(search, "_gather_rank_facts",
                                                      side_effect=gather or _ranks(ranks)))
            result = invoke(*_embed_args(fasta, self.tmp / "db", self.shards, *extra,
                                         *(("--residue-codes",) if codes else ())))
        message = self.refused(result)
        self.assertTrue(written)
        self.assertEqual({p.name: p.read_bytes() for p in self.shards.glob("*.parquet")}, written)
        self.assertEqual(sorted(p.name for p in self.shards.iterdir() if p.suffix != ".parquet"), sorted(foreign))
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db*")), [])
        self.assertIn(f"No run record was written; the shards are kept in {self.shards.resolve()} as run e.", message)
        return message

    def assert_seal_hint(self, message, fasta=_FASTA, present=True):
        hint = (f"Once the cause is fixed, 'build sequences-seal' can record them: " + shlex.join([
            "fm-search", "build", "sequences-seal", "--fasta-file", str(Path(fasta).resolve()), "--from-shards",
            str(self.shards.resolve()), "--run-id", "e", "--min-res-n", "0"]))
        if present:
            self.assertTrue(message.endswith(hint), message)
        else:
            self.assertNotIn("sequences-seal", message)

    def test_e9_the_shards_are_not_the_deal(self):
        """A lost shard (the embedding count), a rank of the world with no shard (W = 3 gathered for 2 shards), and
        rows that are not the deal (the right count): refused before any footer is read."""
        five = _write(self.tmp / "five.fasta", _FIVE)

        def lost(**kwargs):
            fake_chain_predict(ranks=2)(**kwargs)
            (Path(kwargs["out_folder"]) / f"{kwargs['out_name']}-1.parquet").unlink()

        def blocks(**kwargs):  # rank 0 holds entries 0, 2; rank 1 entries 4, 1, 3: five rows, not the deal
            fake_chain_predict(ranks=2)(**kwargs)
            first, second = (Path(kwargs["out_folder"]) / f"{kwargs['out_name']}-{rank}.parquet" for rank in (0, 1))
            t0, t1 = pq.read_table(first), pq.read_table(second)
            pq.write_table(t0.slice(0, 2), first)
            pq.write_table(pa.concat_tables([t0.slice(2), t1.replace_schema_metadata(t0.schema.metadata)]), second)

        for codes in (False, True):
            with self.subTest(case="a lost shard", codes=codes), \
                    mock.patch.object(split_run, "check_plain_footers", side_effect=AssertionError("footers read")), \
                    mock.patch.object(split_run, "check_codes_footers", side_effect=AssertionError("footers read")):
                message = self.post_pass(codes=codes, ranks=2, predict=lost, fasta=five)
                self.assertIn("Found 3 embeddings in 1 shard(s)", message)
                self.assertIn("a rank's shard is missing", message)
                self.assert_seal_hint(message, present=False)
            with self.subTest(case="a dropped rank", codes=codes):
                message = self.post_pass(codes=codes, ranks=2, gather=_ranks(3), fasta=five)
                self.assertIn("are not the deal of its 5 entries over the 3 rank(s) of the pass (no shard of rank(s) "
                              "2; rank 0 has 3 row(s), not 2): the shards cannot be indexed.", message)
                self.assert_seal_hint(message, present=False)
            with self.subTest(case="rows not the deal", codes=codes):
                message = self.post_pass(codes=codes, ranks=2, predict=blocks, fasta=five)
                self.assertIn("(rank 0 has 2 row(s), not 3; rank 1 has 3 row(s), not 2)", message)

    def test_e10_the_footers(self):
        """Codes: a shard of another system's encoding, a plain shard among codes shards, a pass record with another
        --min-res-n or world size; plain: a shard that is not plain. Each names the shard."""
        def pass_then(change, rank=0, ranks=1):
            def predict(**kwargs):
                fake_chain_predict(ranks=ranks)(**kwargs)
                _rewrite(Path(kwargs["out_folder"]) / f"{kwargs['out_name']}-{rank}.parquet", change)
            return predict

        def plain(table):
            return table.select(["id", "embedding"]).replace_schema_metadata(None)

        def tagged(table):
            return table.replace_schema_metadata({b"foldmatch.residue_output": b"residue_codes"})

        cases = {
            "a stale system": ({"system": OTHER_ENCODING}, ("stale", "encoding_digest")),
            "a plain shard mixed in": ({"predict": pass_then(plain, rank=1, ranks=2), "ranks": 2},
                                       ("mixed embedding shards", "emb_e-1.parquet")),
            "pass.min_res_n": ({"predict": pass_then(_with_pass(min_res_n=5))},
                               ("do not record a pass with --min-res-n 0 on 1 rank(s)",
                                "emb_e-0.parquet (min_res_n 5, world_size 1)")),
            "pass.world_size": ({"predict": pass_then(_with_pass(world_size=3))},
                                ("emb_e-0.parquet (min_res_n 0, world_size 3)",)),
            "a plain run's shard that is not plain": ({"codes": False, "predict": pass_then(tagged)},
                                                       ("not plain embedding shards", "emb_e-0.parquet")),
        }
        for case, (kwargs, texts) in cases.items():
            with self.subTest(case=case):
                message = self.post_pass(**kwargs)
                for text in texts:
                    self.assertIn(text, message)
                self.assert_seal_hint(message, present=False)

    def test_e11_a_fasta_modified_during_the_pass(self):
        fasta = Path(shutil.copyfile(_FASTA, self.tmp / "copy.fasta"))
        size = fasta.stat().st_size

        def appending(**kwargs):
            fake_chain_predict()(**kwargs)
            with open(fasta, "a") as f:
                f.write(">late\nACDE\n")

        for codes in (False, True):
            with self.subTest(codes=codes):
                shutil.copyfile(_FASTA, fasta)
                message = self.post_pass(codes=codes, predict=appending, fasta=fasta)
                self.assertIn(f"The FASTA was modified during the pass (rank 0: before {size} bytes, mtime_ns", message)
                self.assertIn(f", after {size + 11} bytes, mtime_ns", message)
                self.assert_seal_hint(message, fasta=fasta)

    def test_e11_another_rank_saw_another_fasta(self):
        """Rank 1's stat changed during its pass (inode, mtime or size), its FASTA has another size, its pre-flight
        counted otherwise, or its stat failed."""
        def changed(**fields):
            return lambda facts: {"after": facts["after"]._replace(**{k: getattr(facts["after"], k) + v
                                                                     for k, v in fields.items()})}

        def other_size(facts):
            grown = facts["before"]._replace(bytes=facts["before"].bytes + 1)
            return {"before": grown, "after": grown, "size": grown.bytes}

        size = _FASTA.stat().st_size
        cases = {
            "an inode": (changed(ino=1), "The FASTA was modified during the pass (rank 1: before"),
            "an mtime": (changed(mtime_ns=1), "The FASTA was modified during the pass (rank 1: before"),
            "a size": (changed(bytes=1), f", after {size + 1} bytes"),
            "another size": (other_size, f"The ranks did not read the same FASTA (rank 1: {size + 1} bytes, rank 0: "
                                         f"{size})"),
            "other counts": (lambda facts: {"preflight": {**facts["preflight"], "residues": 7}},
                             "(rank 1: residues 7, rank 0: residues 303)"),
            "a stat error": (lambda facts: {"before": None, "size": None,
                                            "error": "stat before the pre-flight: [Errno 2] No such file"},
                             "could not be stat'ed on every rank (rank 1: stat before the pre-flight: [Errno 2] No "
                             "such file)"),
        }
        for case, (change, text) in cases.items():
            for codes in (False, True):
                with self.subTest(case=case, codes=codes):
                    message = self.post_pass(codes=codes, ranks=2, gather=_ranks(2, change))
                    self.assertIn(text, message)
                    self.assert_seal_hint(message)

    def test_e11_rank_zero_cannot_stat_or_hash(self):
        stat = split_run.fasta_stat
        calls = []

        def second_fails(path):
            calls.append(path)
            if len(calls) == 2:
                raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(path))
            return stat(path)

        with mock.patch.object(split_run, "fasta_stat", side_effect=second_fails):
            message = self.post_pass()
        self.assertEqual(len(calls), 3)  # before the pre-flight, after the pass (failed), after the hash
        self.assertIn("(rank 0: stat after the pass: [Errno 2] No such file or directory", message)
        self.assert_seal_hint(message)
        with mock.patch.object(split_run, "fasta_sha256", side_effect=OSError(errno.EIO, "Input/output error")):
            message = self.post_pass()
        self.assertIn("Input/output error", message)
        self.assert_seal_hint(message)

    def test_e11_a_write_during_rank_zeros_hash(self):
        """Rank 0's bracket closes after the hash: a write while rank 0 hashes the FASTA (its mtime moved once the hash
        is done) is refused, so the record never pairs a sha256 with a stat taken before it."""
        fasta = Path(shutil.copyfile(_FASTA, self.tmp / "copy.fasta"))
        real = split_run.fasta_sha256

        def written_while_hashed(path):
            out = real(path)
            stat = os.stat(path)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
            return out

        for codes in (False, True):
            with self.subTest(codes=codes), mock.patch.object(split_run, "fasta_sha256",
                                                              side_effect=written_while_hashed):
                message = self.post_pass(codes=codes, fasta=fasta)
            self.assertIn("The FASTA was modified during the pass (rank 0: before", message)
            self.assert_seal_hint(message, fasta=fasta)

    def test_e11_node_local_copies_are_allowed(self):
        """Another inode on rank 1, the same before and after its pass: its own copy of the FASTA, accepted."""
        def copy(facts):
            return {key: facts[key]._replace(ino=facts[key].ino + 1) for key in ("before", "after")}

        with launch_env(FOLDMATCH_RUN_ID="copy"), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict(ranks=2)), \
                mock.patch.object(search, "_gather_rank_facts", side_effect=_ranks(2, copy)):
            result = invoke(*_embed_args(_FASTA, self.tmp / "db", self.shards, "--residue-codes"))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(split_run.read_run_record(self.shards / "emb_copy.run.json")["world_size"], 2)

    def test_e12_the_record_cannot_be_written(self):
        """A full disk while the record is staged: its .tmp is removed, the shards kept."""
        with mock.patch.object(split_run, "_write_all",
                               side_effect=OSError(errno.ENOSPC, "No space left on device")):
            message = self.post_pass()
        self.assertIn("No space left on device", message)
        self.assertIn(f"The run record {self.shards.resolve() / 'emb_e.run.json'} cannot be written", message)
        self.assert_seal_hint(message)

    def test_e12_a_record_or_tmp_that_appeared_during_the_pass(self):
        """Another process's .tmp or record of this run id, written during the pass: refused, and left as it is."""
        for name, text in (("emb_e.run.json.tmp", "cannot be staged"), ("emb_e.run.json", "File exists")):
            def predict(**kwargs):
                fake_chain_predict()(**kwargs)
                (Path(kwargs["out_folder"]) / name).write_text("another process\n")

            with self.subTest(name=name):
                message = self.post_pass(predict=predict, foreign=(name,))
                self.assertIn(text, message)
                self.assertIn(name, message)
                self.assertEqual((self.shards / name).read_text(), "another process\n")
                self.assert_seal_hint(message)


# =====================================================================================================================
# P8b.4: build sequences-index
# =====================================================================================================================
#: Plan P8b (CLI), verbatim: step 2's command help and the help texts it does not take from 'build sequences'.
INDEX_HELP = (
    "Step 2 of a split 'build sequences': build the database from the shards of one 'build sequences-embed' run, "
    "after checking against its run record that they are complete and were computed from this FASTA: the FAISS "
    "index, the sequence store and, when the run computed residue codes, the residue-code store, as 'build "
    "sequences' builds them after its ESM3 pass. No ESM3, GPU or download. One process: run it as its own job on a "
    "CPU node, not inside the GPU job; FAISS uses the node's physical cores unless OMP_NUM_THREADS is set."
)
INDEX_OWN_HELP = {
    "fasta_file": "The FASTA file the run embedded. Checked against the run record (size, sha256, pre-flight counts) "
                  "and against the id of every shard row; it may have moved since.",
    "output_db": "Path to save the FAISS database (as for 'build sequences'). Usually the run's --output-db, which "
                 "the record holds; another path is allowed.",
    "from_shards": "Folder holding the shards and the run record of a 'build sequences-embed' run (its "
                   "--shards-folder).",
    "run_id": "Run id of the run to build from (its record is emb_<run id>.run.json; 'build sequences-embed' logs "
              "it). Needed when the folder holds more than one run, unfinished ones included.",
    "residue_codes": "Build the residue-code store from the run's codes (enabling `query sequences --scoring "
                     "esm-codes`). Default: exactly when the run computed them ('build sequences-embed "
                     "--residue-codes'). --residue-codes refuses a run without codes; --no-residue-codes leaves the "
                     "store out (add it later with 'build residue-codes --from-shards').",
    "cleanup_embeddings": "Delete the run's record, then its shards, from --from-shards once the database is built. "
                          "Use it only on the last build from this run. Refused together with --no-residue-codes on a "
                          "run with residue codes. By default they are kept.",
}
#: SLURM job variables that launch_env leaves alone and step 2's launch checks read.
_JOB_ENV = ("SLURM_JOB_NUM_NODES", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")
#: The build-block keys in which a split-built code store may differ from the one-shot's (plan P8b, Identical outputs).
_EXEMPT = ("created_utc", "run_id", "split")


def _index_args(fasta, db, shards, *extra):
    return ("build", "sequences-index", "--fasta-file", fasta, "--output-db", db, "--from-shards", shards, *extra)


#: ``sys.argv[0]`` of the ``fm-search`` console script, as step 2 and the seal see it in ``_SplitBuild`` (P9.I2).
_FM_SEARCH = "/opt/foldmatch/bin/fm-search"


def _as_fm_search(args):
    """``sys.argv`` as the ``fm-search`` console script sets it for ``args``."""
    return mock.patch.object(sys, "argv", [_FM_SEARCH, *map(str, args)])


def _command_line(args) -> str:
    """The command line that the launch checks' advice gives for ``args`` run through the console script."""
    return shlex.join(["fm-search", *map(str, args)])


#: What the launch checks' advice says the one-node job's SCRIPT runs next time (P9.I2, review finding 2).
_NEXT_BUILD = ("with the paths and run id of that job's build (export FOLDMATCH_RUN_ID=ID before the sbatch, and pass "
               "--run-id \"$FOLDMATCH_RUN_ID\")")


def _tree(*folders) -> dict:
    """``{path: sha256}`` of every file under ``folders`` (``None`` for a folder), and whether each folder exists."""
    out = {}
    for folder in map(Path, folders):
        out[str(folder)] = folder.exists()
        for path in sorted(folder.rglob("*")):
            out[str(path)] = None if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _database_files(db) -> dict:
    """``{suffix: path}`` of the files of the database at ``db`` (``--output-db``)."""
    db = Path(db)
    return {path.name[len(db.name):]: path for path in db.parent.glob(f"{glob.escape(db.name)}.*")}


def _code_store_index(path):
    """``(entries rows, meta rows)`` of a ``.rescodes.idx``."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return (con.execute("SELECT * FROM entries ORDER BY id").fetchall(),
                dict(con.execute("SELECT key, value FROM meta").fetchall()))
    finally:
        con.close()


def assert_same_database(a, b, exempt=_EXEMPT):
    """Plan P8b (Identical outputs): the databases at ``a`` and ``b`` (``--output-db`` paths) have the same files, each
    byte-identical except the residue-code store's index and manifest, which are equal in content: the ``entries``
    rows equal, the ``meta`` rows equal except ``build_json``, which decodes to its own manifest's ``build`` block,
    and the manifests equal except the ``build`` keys in ``exempt`` (present on one side or not)."""
    files = _database_files(a), _database_files(b)
    if sorted(files[0]) != sorted(files[1]):
        raise AssertionError(f"{a} and {b} have other files: {sorted(files[0])} != {sorted(files[1])}")
    for suffix in sorted(files[0]):
        if suffix not in (".rescodes.idx", ".rescodes.json") and \
                files[0][suffix].read_bytes() != files[1][suffix].read_bytes():
            raise AssertionError(f"{files[0][suffix]} and {files[1][suffix]} differ")
    if ".rescodes.json" not in files[0]:
        return
    stores = []
    for side in files:
        manifest = json.loads(side[".rescodes.json"].read_text())
        entries, meta = _code_store_index(side[".rescodes.idx"])
        build = json.loads(meta.pop("build_json"))
        if build != manifest["build"]:
            raise AssertionError(f"{side['.rescodes.idx']}: build_json {build} is not its manifest's build block "
                                 f"{manifest['build']}")
        manifest["build"] = {key: value for key, value in build.items() if key not in exempt}
        stores.append((entries, meta, manifest))
    for k, what in enumerate(("entries rows", "meta rows", "manifest")):
        if stores[0][k] != stores[1][k]:
            raise AssertionError(f"the code stores of {a} and {b} differ in their {what}: {stores[0][k]} != "
                                 f"{stores[1][k]}")


def _resync(record_path):
    """Write the record's shard sizes and rows as the shards now have them (a shard rewritten on purpose)."""
    record = json.loads(record_path.read_text())
    for shard in record["shards"]:
        path = record_path.parent / shard["name"]
        shard.update(bytes=path.stat().st_size, rows=pq.read_metadata(path).num_rows)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _edit_record(record_path, change):
    """Rewrite the record as ``change(record)`` (a dict changed in place)."""
    record = json.loads(record_path.read_text())
    change(record)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _one_residue_changed(path):
    """Rewrite the FASTA at ``path`` with the first residue of its first sequence line replaced by another standard
    residue: the same size, ids and pre-flight counts, another sha256. Its mtime is moved on by 1 s (as well as by the
    write), so that a clock coarser than the test cannot hide the change from a stat bracket."""
    path = Path(path)
    stat = path.stat()
    data = bytearray(path.read_bytes())
    position = data.index(b"\n", data.index(b">")) + 1
    data[position] = ord("W") if data[position] != ord("W") else ord("G")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(bytes(data))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    assert path.stat().st_size == stat.st_size
    assert hashlib.sha256(path.read_bytes()).hexdigest() != before
    return path


class _SplitBuild(_WithBundle):
    """Step 1 with the fake pass, step 2 under the ESM3 spy with ``scoring_system.prepare`` refusing (no bundle), and
    the one-shot build to compare them with. The database goes into ``out``; the SLURM job variables are unset."""

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(os.environ))
        for name in _JOB_ENV:
            os.environ.pop(name, None)
        # On Linux the real widening would change this test process's CPU mask for the rest of the session; tests of
        # the widening patch it themselves.
        self.enterContext(mock.patch.object(alignment, "_expand_cpu_affinity", return_value=os.cpu_count() or 1))
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.db = self.out / "db"

    def embed(self, *extra, ranks=1, run_id="run", fasta=_FASTA, db=None, shards=None, gather=None, **fake):
        """Step 1 (as ``EmbedStepTest.embed``), which must succeed; returns its record's path."""
        shards = shards or self.shards
        with launch_env(FOLDMATCH_RUN_ID=run_id), ExitStack() as stack:
            stack.enter_context(mock.patch("foldmatch.search.embedding_computer.chain_predict",
                                           fake_chain_predict(ranks=ranks, **fake)))
            if gather is not None or ranks > 1:
                stack.enter_context(mock.patch.object(search, "_gather_rank_facts",
                                                      side_effect=gather or _ranks(ranks)))
            result = invoke(*_embed_args(fasta, db or self.db, shards, *extra))
        self.assertEqual(result.exit_code, 0, result.output)
        return Path(shards) / f"emb_{run_id}.run.json"

    def index(self, *extra, fasta=_FASTA, db=None, shards=None, env=None):
        """Step 2 under the ESM3 spy, the bundle refused, with ``env`` set and ``sys.argv`` as the console script
        sets it; returns the result."""
        args = _index_args(fasta, db or self.db, shards or self.shards, *extra)
        with _esm3_spy() as esm3, mock.patch.object(ss, "prepare", side_effect=AssertionError("bundle fetched")), \
                mock.patch.dict(os.environ, {name: str(value) for name, value in (env or {}).items()}), \
                _as_fm_search(args):
            result = invoke(*args)
        esm3.assert_not_called()
        return result

    def built(self, *extra, **kwargs):
        """Step 2, which must succeed and leave no build lock."""
        result = self.index(*extra, **kwargs)
        self.assertEqual(result.exit_code, 0, f"{result.output} {result.exception!r}")
        db = kwargs.get("db") or self.db
        self.assertFalse(Path(f"{db}.build.lock").exists())
        return result

    def oneshot(self, *extra, run_id="oneshot", fasta=_FASTA, ranks=1, **fake):
        """``build sequences`` on the same input, into its own folders; returns its database path."""
        db = self.tmp / f"{run_id}-db" / "db"
        shards = self.tmp / f"{run_id}-shards"
        shards.mkdir()
        with launch_env(FOLDMATCH_RUN_ID=run_id), \
                mock.patch("foldmatch.search.embedding_computer.chain_predict",
                           fake_chain_predict(ranks=ranks, **fake)):
            result = invoke(*_build_args(fasta, db, shards, *extra))
        self.assertEqual(result.exit_code, 0, result.output)
        return db

    def logged(self, call):
        """``call()``'s root-logger records as ``(level, message)`` (``set_log_level`` would replace assertLogs'
        handler)."""
        with mock.patch.object(search, "set_log_level"), self.assertLogs(level="INFO") as logs:
            call()
        return [(record.levelname, record.getMessage()) for record in logs.records]

    def setup_and_warnings(self, call):
        """``call()``'s ``set_log_level`` calls, as ``("set_log_level", level)``, and root-logger WARNINGs, as
        ``("WARNING", message)``, in the order they came."""
        events = []

        class Recorder(logging.Handler):
            def emit(self, record):
                events.append((record.levelname, record.getMessage()))

        root, recorder = logging.getLogger(), Recorder(level=logging.WARNING)
        root.addHandler(recorder)
        try:
            with mock.patch.object(search, "set_log_level",
                                   side_effect=lambda level, **_: events.append(("set_log_level", level))):
                call()
        finally:
            root.removeHandler(recorder)
        return events


class IndexOptionTest(_Tmp):

    def test_the_command_help(self):
        self.assertEqual(_command("build", "sequences-index").help, INDEX_HELP)
        output = CliRunner().invoke(search.app, ["build", "sequences-index", "--help"], env={"COLUMNS": "400"}).output
        for text in (INDEX_HELP, *INDEX_OWN_HELP.values()):
            with self.subTest(text=text[:40]):
                self.assertIn(text, _flat(output))

    def test_the_options(self):
        """Exactly step 2's options; the index options and --log-level are the one-shot's (type, default, help), the
        others have the plan's help; --residue-codes is a tri-state (default None, D24)."""
        index = {param.name: param for param in _command("build", "sequences-index").params}
        oneshot = {param.name: param for param in _command("build", "sequences").params}
        self.assertEqual(sorted(index), sorted([
            "fasta_file", "output_db", "from_shards", "run_id", "residue_codes", "cleanup_embeddings",
            "use_gpu_index", "index_type", "ivf_nlist", "ivf_nprobe", "load_batch_size", "log_level"]))

        def info(param, *drop):
            out = param.to_info_dict()
            for key in ("name", "opts", "help", *drop):
                out.pop(key)
            return out

        for name, param in index.items():
            with self.subTest(name=name):
                self.assertEqual(param.help, INDEX_OWN_HELP.get(name, oneshot.get(name) and oneshot[name].help))
                if name in oneshot and name != "residue_codes":
                    self.assertEqual(info(param), info(oneshot[name]))
        from_shards = index["from_shards"]
        self.assertEqual((from_shards.opts, from_shards.required, from_shards.type.exists, from_shards.type.dir_okay,
                          from_shards.type.file_okay), (["--from-shards"], True, True, True, False))
        self.assertEqual((index["run_id"].default, index["run_id"].required, index["run_id"].type.name),
                         (None, False, "text"))
        codes = index["residue_codes"]
        self.assertEqual((codes.opts, codes.secondary_opts, codes.default, codes.type.name),
                         (["--residue-codes"], ["--no-residue-codes"], None, "boolean"))
        self.assertEqual(index["cleanup_embeddings"].default, False)

    def test_options_another_command_owns(self):
        """The pass's options (there is no pass; --min-res-n comes from the record), --shards-folder and
        --tmp-embedding-folder (use --from-shards): exit 2, nothing read or written. A malformed --run-id is exit 2 too
        (I1)."""
        for option in (("--min-res-n", "0"), ("--accelerator", "cpu"), ("--num-nodes", "1"), ("--devices", "1"),
                       ("--strategy", "auto"), ("--batch-size", "1"), ("--num-workers", "0"),
                       ("--shards-folder", self.shards), ("--tmp-embedding-folder", self.shards), ("--force",)):
            with self.subTest(option=option[0]), _esm3_spy() as esm3:
                result = invoke(*_index_args(_FASTA, self.tmp / "db", self.shards, *option))
                self.usage_error(result, f"No such option: {option[0]}")
                esm3.assert_not_called()
        for bad in ("a b", "a/b", "a*"):
            with self.subTest(run_id=bad), mock.patch.object(split_run, "find_run") as find_run:
                self.usage_error(invoke(*_index_args(_FASTA, self.tmp / "db", self.shards, "--run-id", bad)),
                                 "is not a run id")
                find_run.assert_not_called()
        self.usage_error(invoke("build", "sequences-index", "--fasta-file", _FASTA, "--output-db", self.tmp / "db"),
                         "--from-shards")
        self.assertEqual(list(self.shards.iterdir()), [])
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db*")), [])

    def test_the_from_shards_help_names_step_1(self):
        """'build residue-codes' names step 1's shards in its --from-shards help and in its command help."""
        param = next(p for p in _command("build", "residue-codes").params if p.name == "from_shards")
        self.assertIn("or those of a 'build sequences-embed --residue-codes' run (its --shards-folder).", param.help)
        self.assertIn("from the embedding shards its 'build sequences --residue-codes' run kept, or those of the "
                      "'build sequences-embed --residue-codes' run it was built from, without running ESM3 again",
                      _command("build", "residue-codes").help)

    def test_the_load_batch_size_help(self):
        """--load-batch-size has one help in every build command: FAISS adds IVF-PQ vectors in chunks of 32768, so a
        larger batch changes only the memory (the older text claimed that larger batches use more cores)."""
        expected = ("Number of vectors per batch streamed into the FAISS index; larger batches use more memory (FAISS "
                    "adds IVF-PQ vectors in chunks of 32768 whatever this value).")
        for name in ("structures", "embeddings", "sequences", "sequences-index"):
            with self.subTest(command=name):
                param = next(p for p in _command("build", name).params if p.name == "load_batch_size")
                self.assertEqual(param.help, expected)


class IndexStepTest(_SplitBuild):

    def test_two_steps_equal_the_one_shot(self):
        """Plain and codes, on 1 and 2 ranks, with --min-res-n 0 and 100: step 1 plus step 2 with no extra option build
        the one-shot's database (assert_same_database); the code store's manifest says the build was split."""
        for codes in (False, True):
            for ranks in (1, 2):
                for min_res_n in (0, 100):
                    name = f"{'codes' if codes else 'plain'}{ranks}-{min_res_n}"
                    with self.subTest(codes=codes, ranks=ranks, min_res_n=min_res_n):
                        extra = ("--min-res-n", min_res_n, *(("--residue-codes",) if codes else ()))
                        shards = self.tmp / f"shards-{name}"
                        shards.mkdir()
                        db = self.tmp / f"split-{name}" / "db"
                        self.embed(*extra, ranks=ranks, run_id=f"split-{name}", shards=shards, db=db)
                        self.built(shards=shards, db=db)
                        oneshot = self.oneshot(*extra, run_id=f"one-{name}", ranks=ranks)
                        assert_same_database(oneshot, db)
                        self.assertEqual(sorted(_database_files(db)),
                                         sorted([".index", ".metadata", ".sequences",
                                                 *((".rescodes.bin", ".rescodes.idx", ".rescodes.json") if codes
                                                   else ())]))
                        if codes:
                            build = ResidueCodeStore(db.parent, "db").check_compatible()["build"]
                            self.assertEqual((build["split"], build["from_shards"], build["run_id"],
                                              build["devices"]), (True, False, f"split-{name}", ranks))
                            self.assertNotIn("split",
                                             ResidueCodeStore(oneshot.parent, "db").check_compatible()["build"])
                        self.assertEqual(len(list(shards.glob("*.parquet"))), len(split_run.dealt_rows(
                            2 if min_res_n == 0 else 1, ranks)))  # the shards are kept

    def test_hnsw_at_one_thread(self):
        """--index-type hnsw: byte-identical too, at one FAISS thread (set explicitly, with OMP_NUM_THREADS=1 so that
        step 2 leaves it; setting the variable alone after FAISS started would not change its threads). HNSW over the
        fixture's two vectors is the same at any thread count, so this checks the equality, not thread determinism."""
        import faiss
        self.embed()
        previous = faiss.omp_get_max_threads()
        self.addCleanup(faiss.omp_set_num_threads, previous)
        faiss.omp_set_num_threads(1)
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}):
            self.built("--index-type", "hnsw")
            assert_same_database(self.oneshot("--index-type", "hnsw"), self.db)
        self.assertEqual(faiss.omp_get_max_threads(), 1)

    def test_assert_same_database_sees_a_difference(self):
        """The helper's own controls: another index, a missing file, another manifest field, another build key (in the
        manifest and the index's build_json), a build_json that is not its manifest's, another entry row."""
        self.embed("--residue-codes")
        self.built()
        oneshot = self.oneshot("--residue-codes")
        assert_same_database(oneshot, self.db)
        copy = self.tmp / "copy"
        cases = {
            "another index": (lambda db: Path(f"{db}.index").write_bytes(Path(f"{db}.index").read_bytes() + b"\0"),
                              "db.index differ"),
            "a missing file": (lambda db: Path(f"{db}.sequences").unlink(), "have other files"),
            "another manifest field": (lambda db: _edit_record(
                Path(f"{db}.rescodes.json"), lambda m: m.update(nonstandard_entries=99)), "differ in their manifest"),
            "another build key": (lambda db: (
                _edit_record(Path(f"{db}.rescodes.json"), lambda m: m["build"].update(devices=7)),
                _execute(Path(f"{db}.rescodes.idx"),
                         "UPDATE meta SET value = json_set(value, '$.devices', 7) WHERE key = 'build_json'")),
                "differ in their manifest"),
            "a build_json that is not its manifest's": (lambda db: _edit_record(
                Path(f"{db}.rescodes.json"), lambda m: m["build"].update(devices=7)),
                "is not its manifest's build block"),
            "another entry": (lambda db: _execute(Path(f"{db}.rescodes.idx"),
                                                  "UPDATE entries SET length = length + 1"), "differ in their entries"),
        }
        for case, (change, text) in cases.items():
            with self.subTest(case=case):
                shutil.rmtree(copy, ignore_errors=True)
                shutil.copytree(self.out, copy)
                change(copy / "db")
                with self.assertRaises(AssertionError) as ctx:
                    assert_same_database(oneshot, copy / "db")
                self.assertIn(text, str(ctx.exception))
        assert_same_database(oneshot, self.db, exempt=_EXEMPT)
        with self.assertRaises(AssertionError) as ctx:
            assert_same_database(oneshot, self.db, exempt=("created_utc", "run_id"))
        self.assertIn("'split': True", str(ctx.exception))

    def test_the_tri_state(self):
        """The default follows the run: a code store for a codes run, none for a plain run; --residue-codes on a codes
        run builds what the default builds."""
        self.embed("--residue-codes", run_id="codes")
        self.built("--run-id", "codes")
        self.assertTrue(ResidueCodeStore(self.out, "db").exists())
        self.built("--residue-codes", db=self.tmp / "explicit" / "db")
        assert_same_database(self.db, self.tmp / "explicit" / "db")
        plain = self.tmp / "plain"
        plain.mkdir()
        self.embed(run_id="plain", shards=plain, db=self.tmp / "p" / "db")
        self.built(shards=plain, db=self.tmp / "p" / "db")
        self.assertEqual(sorted(_database_files(self.tmp / "p" / "db")), [".index", ".metadata", ".sequences"])

    def test_no_residue_codes_on_a_codes_run(self):
        """--no-residue-codes on a codes run builds the plain one-shot's database; a later 'build residue-codes
        --from-shards --run-id' adds the one-shot's code store (bin_sha256, entries_digest)."""
        self.embed("--residue-codes")
        self.built("--no-residue-codes")
        assert_same_database(self.oneshot(run_id="plain"), self.db)
        result = invoke("build", "residue-codes", self.out, "db", "--from-shards", self.shards, "--run-id", "run")
        self.assertEqual(result.exit_code, 0, result.output)
        direct = ResidueCodeStore(self.oneshot("--residue-codes", run_id="codes").parent, "db").manifest()
        added = ResidueCodeStore(self.out, "db").check_compatible()
        self.assertEqual((added["bin_sha256"], added["entries_digest"]),
                         (direct["bin_sha256"], direct["entries_digest"]))
        self.assertEqual(sorted(p.name for p in self.shards.iterdir()), ["emb_run-0.parquet", "emb_run.run.json"])

    def test_no_residue_codes_with_a_system_unknown_here(self):
        """A codes run whose system this FoldMatch does not register: the default is refused (I10), and
        --no-residue-codes builds the plain database from the shards' own consistency."""
        self.embed("--residue-codes")
        with mock.patch.object(ss, "REGISTRY", MappingProxyType({})):
            self.assertIn("unknown esm-codes scoring system", self.refused(self.index()))
            self.built("--no-residue-codes")
        assert_same_database(self.oneshot(run_id="plain"), self.db)

    def test_two_builds_from_one_run(self):
        """Two step-2 builds from one run into two folders: the same database, and the shards and record unchanged."""
        self.embed("--residue-codes", ranks=2)
        before = _tree(self.shards)
        self.built()
        self.built(db=self.tmp / "second" / "db")  # a folder that does not exist yet
        self.assertEqual(_tree(self.shards), before)
        assert_same_database(self.db, self.tmp / "second" / "db")

    def test_a_copied_folder_and_a_moved_fasta(self):
        """A copy of the shards folder builds the database (the record names its shards by file name), and so does a
        copy of the FASTA elsewhere with the same bytes (logged at INFO)."""
        self.embed("--residue-codes")
        copied = self.tmp / "copied"
        shutil.copytree(self.shards, copied)
        shutil.rmtree(self.shards)
        moved = self.tmp / "elsewhere" / "seqs.fasta"
        moved.parent.mkdir()
        shutil.copyfile(_FASTA, moved)
        messages = self.logged(lambda: self.built(shards=copied, fasta=moved))
        self.assertIn(("INFO", f"{moved.resolve()} has the size and sha256 of the FASTA run run embedded, with another "
                               f"path and modification time (recorded: {_FASTA.resolve()}, mtime_ns "
                               f"{_FASTA.stat().st_mtime_ns})"), messages)
        assert_same_database(self.oneshot("--residue-codes"), self.db)

    def test_cleanup_embeddings(self):
        """--cleanup-embeddings deletes the record, then the shards, after the code store; no lock is left."""
        for codes in (True, False):
            with self.subTest(codes=codes):
                for path in [*self.shards.iterdir(), *self.out.iterdir()]:
                    path.unlink()
                record = self.embed(*(("--residue-codes",) if codes else ()))
                seen = []
                remove_shards = EmbeddingComputer.remove_shards

                def removing(computer):
                    seen.append((record.exists(), ResidueCodeStore(self.out, "db").exists(),
                                 sorted(p.name for p in self.shards.iterdir())))
                    remove_shards(computer)

                with mock.patch.object(EmbeddingComputer, "remove_shards", autospec=True, side_effect=removing):
                    self.built("--cleanup-embeddings")
                self.assertEqual(seen, [(False, codes, ["emb_run-0.parquet"])])
                self.assertEqual(list(self.shards.iterdir()), [])
                self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                                 sorted(f"db{s}" for s in (".index", ".metadata", ".sequences",
                                                           *((".rescodes.bin", ".rescodes.idx", ".rescodes.json")
                                                             if codes else ()))))

    def test_no_residue_codes_and_cleanup_on_a_plain_run(self):
        """--no-residue-codes --cleanup-embeddings is refused only on a run with codes (I9): a plain run builds and
        deletes its record and shards."""
        self.embed()
        self.built("--no-residue-codes", "--cleanup-embeddings")
        self.assertEqual(list(self.shards.iterdir()), [])
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["db.index", "db.metadata", "db.sequences"])

    def test_the_index_options_reach_the_index_phase(self):
        """Step 2's index options go to the index phase as given (the phase mocked: --use-gpu-index needs faiss-gpu),
        with the pre-flight bound to the computer, so that get_embedding_batches counts the shard rows again."""
        self.embed()
        seen = []

        def phase(computer, preflight, **kwargs):
            seen.append((kwargs["index"], computer.expected_count, computer.preflight is preflight,
                         preflight.entries_kept))

        with mock.patch.object(search, "_build_sequence_database", side_effect=phase):
            self.built("--index-type", "ivf_pq", "--ivf-nlist", "256", "--ivf-nprobe", "4", "--load-batch-size", "1000",
                       "--use-gpu-index")
        self.assertEqual(seen, [(search._IndexOptions(True, IndexType.ivf_pq, 256, 4, 1000), 2, True, 2)])

    def test_the_pre_flight_is_the_records(self):
        """Step 2's pre-flight runs with the record's min_res_n and strictness: strict for a codes run."""
        for codes in (True, False):
            with self.subTest(codes=codes):
                for path in [*self.shards.iterdir(), *self.out.iterdir()]:
                    path.unlink()
                self.embed("--min-res-n", "100", *(("--residue-codes",) if codes else ()))
                with mock.patch.object(search, "_fasta_preflight", wraps=search._fasta_preflight) as preflight:
                    self.built()
                self.assertEqual([(c.args[1], c.kwargs["strict"]) for c in preflight.call_args_list], [(100, codes)])

    def test_the_manifest_of_a_split_build(self):
        """build.split is set and from_shards false; the ESM3 revision is the shards' pass record's, not this
        installation's pin; the other build keys are the one-shot's (they describe the pass)."""
        self.embed("--residue-codes")
        with mock.patch("foldmatch.search.residue_code_store._installed_esm_revision",
                        return_value="another-installation"):
            self.built()
        split = ResidueCodeStore(self.out, "db").check_compatible()
        self.assertEqual(split["esm"]["revision"], ESM_REVISION)
        self.assertEqual(read_shard_info(self.shards / "emb_run-0.parquet").pass_record["esm_revision"], ESM_REVISION)
        direct = ResidueCodeStore(self.oneshot("--residue-codes").parent, "db").manifest()
        self.assertEqual({k: v for k, v in split["build"].items() if k not in _EXEMPT},
                         {k: v for k, v in direct["build"].items() if k not in _EXEMPT})
        self.assertEqual((split["build"]["split"], split["build"]["from_shards"], split["min_res_n"]), (True, False, 0))

    def test_the_log_lines(self):
        """The FASTA hash and id-check times, the run it builds from, the threads, the end line; a WARNING for another
        FoldMatch's run and for a sealed record, an INFO line for another --output-db."""
        record = self.embed("--residue-codes", ranks=2)
        messages = self.logged(lambda: self.built())
        db, shards = self.db, self.shards.resolve()
        size = _FASTA.stat().st_size
        found = [m for _, m in messages]
        prefixes = (f"FASTA sha256 of {_FASTA.resolve()} ({size} bytes) in ", "Ids of the 2 shard rows checked "
                    "against ", f"Building {db} from run run in {shards}: ", "Index phase: ",
                    f"Database {db} built from run run in ")
        lines = [[m for m in found if m.startswith(prefix)] for prefix in prefixes]
        self.assertEqual([len(line) for line in lines], [1] * 5, found)
        created = split_run.read_run_record(record)["created_utc"]
        self.assertEqual(lines[2][0], f"Building {db} from run run in {shards}: 2 entries in 2 shard(s), embedded "
                                      f"{created} by FoldMatch {foldmatch.__version__} on 2 rank(s) (cpu), residue "
                                      f"codes of {SYSTEM.name}")
        self.assertRegex(lines[4][0], r" in \d+\.\d s$")
        order = [found.index(line[0]) for line in lines]
        self.assertEqual(order, sorted(order))
        self.assertFalse([m for level, m in messages if level == "WARNING" and "FoldMatch" in m])
        self.assertFalse([m for m in found if "not the --output-db the run recorded" in m])

        # Another FoldMatch's step 1: D30.
        _edit_record(record, lambda r: r.update(foldmatch_version="0.0.1"))
        messages = self.logged(lambda: self.built(db=self.tmp / "older" / "db"))
        warnings = [m for level, m in messages if level == "WARNING"]
        self.assertTrue([m for m in warnings if m.startswith(f"Run run was embedded by FoldMatch 0.0.1, and this is "
                                                             f"FoldMatch {foldmatch.__version__}")], warnings)
        self.assertFalse([m for m in warnings if "sequences-seal" in m], warnings)
        start = next(m for _, m in messages if m.startswith("Building ") and " from run run " in m)
        self.assertIn(f"embedded {created} by FoldMatch 0.0.1 on 2 rank(s) (cpu), residue codes", start)

        # A sealed record holds the seal's FoldMatch; a codes run's pass record holds the pass's, which D30 compares.
        def sealed_by(seal_version, pass_version):
            def change(record_):
                record_.update(foldmatch_version=seal_version, command=split_run.SEAL_COMMAND, sealed=True,
                               output_db=None, options={})
                record_["pass"]["foldmatch_version"] = pass_version
            return change

        for k, (seal_version, pass_version) in enumerate((("0.0.2", "0.0.1"), ("0.0.2", foldmatch.__version__))):
            with self.subTest(seal=seal_version, run_pass=pass_version):
                _edit_record(record, sealed_by(seal_version, pass_version))
                other = self.tmp / f"sealed{k}" / "db"
                messages = self.logged(lambda: self.built("--no-residue-codes", db=other))
                warnings = [m for level, m in messages if level == "WARNING"]
                d30 = [m for m in warnings if m.startswith("Run run was embedded by FoldMatch ")]
                self.assertEqual(d30, [] if pass_version == foldmatch.__version__ else [
                    f"Run run was embedded by FoldMatch 0.0.1, and this is FoldMatch {foldmatch.__version__}: its "
                    f"record, shard schemas and digests, pre-flight counts and ids were checked, and the database is "
                    f"built by this FoldMatch."])
                self.assertIn(f"The record of run run was written by 'build sequences-seal' (FoldMatch 0.0.2): it "
                              f"binds the shards to the FASTA by their entry count, rows and ids (and, with residue "
                              f"codes, their seq_sha1), not by the pass that wrote them.", warnings)
                start = next(m for _, m in messages if m.startswith(f"Building {other} from run"))
                self.assertEqual(start, f"Building {other} from run run in {shards}: 2 entries in 2 shard(s), sealed "
                                        f"{created} by FoldMatch 0.0.2; embedded by FoldMatch {pass_version} on 2 "
                                        f"rank(s) (cpu), its residue codes of {SYSTEM.name} left out "
                                        f"(--no-residue-codes); sealed record")
                self.assertFalse([m for _, m in messages if "not the --output-db" in m])  # a sealed record names none
        _edit_record(record, lambda r: r.update(output_db="/elsewhere/db"))
        messages = self.logged(lambda: self.built("--no-residue-codes", db=self.tmp / "third" / "db"))
        self.assertIn(("INFO", f"Building {self.tmp / 'third' / 'db'}, not the --output-db the run recorded "
                               f"(/elsewhere/db)"), messages)
        plain = self.tmp / "plain"
        plain.mkdir()
        plain_record = self.embed(run_id="plain", shards=plain, db=self.tmp / "p" / "db")
        messages = self.logged(lambda: self.built(shards=plain, db=self.tmp / "p" / "db"))
        start = next(m for _, m in messages if m.startswith("Building ") and " from run plain " in m)
        self.assertTrue(start.endswith("on 1 rank(s) (device not recorded), no residue codes"), start)
        # A sealed plain run: its pass record is the seal's, so the pass's FoldMatch is unknown (no D30).
        _edit_record(plain_record, lambda r: r.update(foldmatch_version="0.0.2", command=split_run.SEAL_COMMAND,
                                                      sealed=True, output_db=None, options={}))
        plain_created = split_run.read_run_record(plain_record)["created_utc"]
        messages = self.logged(lambda: self.built(shards=plain, db=self.tmp / "p2" / "db"))
        warnings = [m for level, m in messages if level == "WARNING"]
        self.assertFalse([m for m in warnings if "was embedded by FoldMatch" in m], warnings)
        self.assertIn("The record of run plain was written by 'build sequences-seal' (FoldMatch 0.0.2): it binds the "
                      "shards to the FASTA by their entry count, rows and ids (and, with residue codes, their "
                      "seq_sha1), not by the pass that wrote them, whose FoldMatch version is not recorded (shards "
                      "without residue codes do not record it).", warnings)
        start = next(m for _, m in messages if m.startswith("Building ") and " from run plain " in m)
        self.assertTrue(start.endswith(f"sealed {plain_created} by FoldMatch 0.0.2; embedded by a FoldMatch it does "
                                       f"not record on 1 rank(s) (device not recorded), no residue codes; sealed "
                                       f"record"), start)

    def test_the_logged_next_command_runs_verbatim(self):
        """Step 1's next command, split as a shell splits it (paths with spaces, a relative --output-db recorded
        absolute), builds the database step 1 recorded."""
        folder = self.tmp / "with space"
        folder.mkdir()
        fasta = Path(shutil.copyfile(_FASTA, folder / "my seqs.fasta"))
        shards = folder / "shard s"
        shards.mkdir()
        with chdir(folder):
            messages = self.logged(lambda: self.embed("--residue-codes", fasta=fasta, shards=shards,
                                                      db=Path("rel dir") / "my db"))
        line = next(m for _, m in messages if m.startswith("Build the database with: "))
        argv = shlex.split(line[len("Build the database with: "):])
        self.assertEqual(argv[:3], ["fm-search", "build", "sequences-index"])
        with _esm3_spy() as esm3, mock.patch.object(ss, "prepare", side_effect=AssertionError("bundle fetched")):
            result = invoke(*argv[1:])
        esm3.assert_not_called()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(ResidueCodeStore(folder / "rel dir", "my db").check_compatible())

    def test_launch_warnings(self):
        """A one-task srun step of a multi-node job, and a one-node job with GPUs: built, with a WARNING (D34)."""
        self.embed()
        cases = {
            "an srun step": ({"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 2, "SLURM_STEP_ID": 3,
                              "SLURM_STEP_NUM_TASKS": 1, "SLURM_NTASKS": 8},
                             "'build sequences-index' runs in a one-task srun step of SLURM job 42, which holds 2 "
                             "nodes: the others stay allocated until it ends. Next time, run it as its own one-node "
                             "CPU job: submit it from the GPU job's script before the job's work starts: sbatch -C cpu "
                             "-N 1 --dependency=afterok:$SLURM_JOB_ID --kill-on-invalid-dep=yes ... SCRIPT. SCRIPT "
                             "runs this command, without srun, " + _NEXT_BUILD + ":\n    {command}"),
            "GPUs": ({"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 1, "SLURM_GPUS_ON_NODE": 4},
                     "SLURM job 42 has GPUs (SLURM_GPUS_ON_NODE=4), which 'build sequences-index' does not use: run it "
                     "as its own job on a CPU node (sbatch -C cpu -N 1 ...); a project with GPU hours only may run it "
                     "on one GPU node, where ivf_pq fits (flat and hnsw may not)."),
        }
        for k, (case, (env, text)) in enumerate(cases.items()):
            with self.subTest(case=case):
                db = self.tmp / f"w{k}" / "db"
                messages = self.logged(lambda: self.built(env=env, db=db))
                warnings = [m for level, m in messages if level == "WARNING"]
                self.assertIn(text.format(command=_command_line(_index_args(_FASTA, db, self.shards))), warnings)
                self.assertFalse([m for m in warnings if "afterok:42" in m], warnings)  # the build runs in job 42
        messages = self.logged(lambda: self.built(env={"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 1},
                                                  db=self.tmp / "w" / "db"))
        self.assertFalse([m for level, m in messages if level == "WARNING" and "SLURM job" in m])

    def test_launch_warnings_after_the_logging_setup(self):
        """P9.I1: the launch checks run before set_log_level, and their WARNING is logged once it has set up the
        logging, so in its format."""
        self.embed()
        for k, env in enumerate(({"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 2, "SLURM_STEP_ID": 3,
                                  "SLURM_STEP_NUM_TASKS": 1},
                                 {"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 1, "SLURM_GPUS_ON_NODE": 4})):
            with self.subTest(env=env):
                events = self.setup_and_warnings(lambda: self.built(env=env, db=self.tmp / f"w{k}" / "db"))
                launch = [event for event in events if "SLURM job 42" in event[1]]
                self.assertEqual(len(launch), 1, events)
                self.assertEqual(events[:2], [("set_log_level", LogLevel.info), launch[0]])

    def test_offline_with_an_empty_hf_cache(self):
        """Step 2 as a command, with HF_HUB_OFFLINE=1 and an empty Hugging Face cache: the code store is built from
        the registry entry alone."""
        self.embed("--residue-codes")
        empty = self.tmp / "empty-hf-cache"
        empty.mkdir()
        env = {k: v for k, v in os.environ.items()
               if k not in _LAUNCH_ENV and k not in _DDP_ENV and not k.startswith(_DDP_ENV_PREFIXES)}
        env.update(HF_HUB_OFFLINE="1", HF_HUB_CACHE=str(empty), HF_HOME=str(empty),
                   PYTHONPATH=os.pathsep.join([_SRC] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        run = subprocess.run([sys.executable, "-m", "foldmatch.cli.search", *map(str, _index_args(
            _FASTA, self.db, self.shards))], env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-4000:])
        self.assertTrue(ResidueCodeStore(self.out, "db").check_compatible())
        self.assertEqual(list(empty.rglob("models--*")), [])  # nothing fetched (transformers writes a version.txt)

    # -- D29: the threads of the index phase (FAISS's thread calls and threadpoolctl's limit mocked) ------------------
    #: threadpoolctl's runtimes as a Linux node with the faiss-cpu 1.15.1 wheel might list them: numpy's and scipy's
    #: OpenBLAS (one of them twice), FAISS's OpenMP OpenBLAS in faiss_cpu.libs, and an OpenMP.
    POOLS = [{"user_api": "blas", "internal_api": "openblas", "num_threads": 6,
              "filepath": "/site-packages/numpy.libs/libscipy_openblas64_-ff651d7f.so"},
             {"user_api": "blas", "internal_api": "openblas", "num_threads": 6,
              "filepath": "/site-packages/scipy.libs/libscipy_openblas-68440149.so"},
             {"user_api": "blas", "internal_api": "openblas", "num_threads": 6,
              "filepath": "/elsewhere/libscipy_openblas-68440149.so"},
             {"user_api": "blas", "internal_api": "openblas", "num_threads": 6, "threading_layer": "openmp",
              "filepath": os.path.join(_faiss_libs(), "libopenblaso-r0-d77a1985.3.15.so")},
             {"user_api": "openmp", "internal_api": "openmp", "num_threads": 6,
              "filepath": os.path.join(_faiss_libs(), "libgomp-a34b3233.so.1.0.0")}]
    #: The same without FAISS's BLAS (the faiss-cpu 1.13.2 wheel's is listed at 1 thread, an MKL build's not at all).
    POOLS_NO_FAISS_BLAS = [pool for pool in POOLS if "libopenblaso" not in pool["filepath"]]

    def threads(self, platform="linux", pools=POOLS, info_error=None, **env):
        """``_whole_node_threads`` with ``env`` (the thread variables otherwise unset) on ``platform``, threadpoolctl
        listing ``pools`` (or raising ``info_error``): ``(FAISS thread calls inside the block, after it,
        threadpool_limits calls, (limit left inside, after), log)``."""
        clean = {k: v for k, v in os.environ.items() if k not in search._BLAS_THREAD_VARS}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(sys, "platform", platform), \
                mock.patch.object(code_alignment, "_physical_cpus", return_value=6), \
                mock.patch("faiss.omp_get_max_threads", return_value=3), \
                mock.patch("faiss.omp_set_num_threads") as set_threads, \
                mock.patch("threadpoolctl.threadpool_info", return_value=pools, side_effect=info_error), \
                mock.patch("threadpoolctl.threadpool_limits") as limits, self.assertLogs(level="INFO") as logs:
            with search._whole_node_threads():
                inside = list(set_threads.call_args_list)
                exited = limits.return_value.__exit__.called
            after = list(set_threads.call_args_list)
        return inside, after, list(limits.call_args_list), (exited, limits.return_value.__exit__.called), \
            [record.getMessage() for record in logs.records]

    def test_d29_threads_with_the_variables_unset(self):
        """FAISS gets the physical cores (the previous count restored after the block), and so does BLAS; the INFO line
        names the BLAS libraries threadpoolctl found (the limit reaches no other; the faiss-cpu 1.15.1 wheel's OpenMP
        OpenBLAS is one of them), or says it found none."""
        inside, after, limits, exited, log = self.threads()
        self.assertEqual(inside, [mock.call(6)])
        self.assertEqual(after, [mock.call(6), mock.call(3)])
        self.assertEqual(limits, [mock.call(limits=6, user_api="blas")])
        self.assertEqual(exited, (False, True))
        self.assertEqual(len(log), 1)
        self.assertRegex(log[0], r"^FAISS/BLAS threads set to 6 \(physical cores of the \d+ allowed CPUs\); BLAS "
                                 r"libraries limited: libopenblaso-r0-d77a1985\.3\.15\.so, "
                                 r"libscipy_openblas-68440149\.so, libscipy_openblas64_-ff651d7f\.so$")
        *_, log = self.threads(pools=self.POOLS_NO_FAISS_BLAS)
        self.assertRegex(log[0], r"^FAISS/BLAS threads set to 6 \(physical cores of the \d+ allowed CPUs\); BLAS "
                                 r"libraries limited: libscipy_openblas-68440149\.so, "
                                 r"libscipy_openblas64_-ff651d7f\.so$")
        *_, log = self.threads(pools=[pool for pool in self.POOLS if pool["user_api"] != "blas"])
        self.assertRegex(log[0], r"^FAISS/BLAS threads set to 6 \(physical cores of the \d+ allowed CPUs\); "
                                 r"threadpoolctl found no BLAS library to limit$")

    def test_d29_omp_num_threads_set(self):
        inside, after, limits, _, log = self.threads(OMP_NUM_THREADS="2")
        self.assertEqual((inside, after, limits), ([], [], []))
        self.assertEqual(log, ["FAISS threads left to OMP_NUM_THREADS=2; BLAS threads left to OMP_NUM_THREADS=2"])

    def test_d29_a_blas_variable_set(self):
        """BLAS is left to the variable, and the INFO line says that FAISS's OpenMP OpenBLAS (the faiss-cpu 1.15.1
        wheel's), which the variable does not govern, runs FAISS's threads; with OMP_NUM_THREADS also set, FAISS is
        left to it too and nothing is added."""
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            for pools, follows in ((self.POOLS, ", except FAISS's OpenMP OpenBLAS, libopenblaso-r0-d77a1985.3.15.so, "
                                                "which runs FAISS's OpenMP threads"),
                                   (self.POOLS_NO_FAISS_BLAS, ""), ([], "")):
                with self.subTest(name=name, pools=len(pools)):
                    inside, after, limits, _, log = self.threads(pools=pools, **{name: "1"})
                    self.assertEqual((inside, limits), ([mock.call(6)], []))
                    self.assertRegex(log[0], rf"^FAISS threads set to 6 \(physical cores of the \d+ allowed CPUs\); "
                                             rf"BLAS threads left to {name}=1{re.escape(follows)}$")
        inside, after, limits, _, log = self.threads(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        self.assertEqual((inside, after, limits), ([], [], []))
        self.assertEqual(log, ["FAISS threads left to OMP_NUM_THREADS=2; BLAS threads left to OMP_NUM_THREADS=2, "
                               "OPENBLAS_NUM_THREADS=1"])
        *_, log = self.threads(info_error=RuntimeError("cannot inspect"), OPENBLAS_NUM_THREADS="1")
        self.assertRegex(log[0], r"BLAS threads left to OPENBLAS_NUM_THREADS=1$")

    def test_d29_never_on_macos(self):
        for env in ({}, {"OMP_NUM_THREADS": "1"}):
            with self.subTest(env=env):
                inside, after, limits, _, log = self.threads(platform="darwin", **env)
                self.assertEqual((inside, after, limits), ([], [], []))
                self.assertEqual(log, ["FAISS/BLAS threads left as they are on macOS (foldmatch pins "
                                       "OMP_NUM_THREADS)"])

    def test_d29_restored_on_an_exception(self):
        clean = {k: v for k, v in os.environ.items() if k not in search._BLAS_THREAD_VARS}
        with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(code_alignment, "_physical_cpus", return_value=6), \
                mock.patch("faiss.omp_get_max_threads", return_value=3), \
                mock.patch("faiss.omp_set_num_threads") as set_threads, \
                mock.patch("threadpoolctl.threadpool_info", return_value=self.POOLS), \
                mock.patch("threadpoolctl.threadpool_limits") as limits, self.assertRaises(RuntimeError):
            with search._whole_node_threads():
                raise RuntimeError("the index phase failed")
        self.assertEqual(set_threads.call_args_list, [mock.call(6), mock.call(3)])
        self.assertTrue(limits.return_value.__exit__.called)

    def test_d29_step_2_sets_the_threads(self):
        """Through the command: the affinity is widened for 'the index phase' before the run is read; FAISS and BLAS
        get the physical cores for the index phase, which runs inside the limit, and FAISS's count is restored."""
        self.embed()
        clean = {k: v for k, v in os.environ.items() if k not in search._BLAS_THREAD_VARS}
        manager = mock.Mock()
        with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(code_alignment, "_physical_cpus", return_value=6), \
                mock.patch("faiss.omp_set_num_threads") as set_threads, \
                mock.patch("threadpoolctl.threadpool_limits") as limits, \
                mock.patch("threadpoolctl.threadpool_info", return_value=[]), \
                mock.patch.object(alignment, "_expand_cpu_affinity", return_value=8) as widen, \
                mock.patch.object(split_run, "find_run", wraps=split_run.find_run) as find_run, \
                mock.patch.object(search, "_build_sequence_database", wraps=search._build_sequence_database) as phase:
            for name, spy in (("widen", widen), ("find_run", find_run), ("set_threads", set_threads),
                              ("limits", limits), ("phase", phase)):
                manager.attach_mock(spy, name)
            self.built()
        calls = [c[0] for c in manager.mock_calls if not c[0].startswith("limits().")]
        self.assertEqual(calls, ["widen", "find_run", "set_threads", "limits", "phase", "set_threads"])
        self.assertEqual(widen.call_args, mock.call(purpose="the index phase"))
        self.assertEqual(set_threads.call_args_list[0], mock.call(6))
        self.assertEqual(limits.call_args, mock.call(limits=6, user_api="blas"))
        self.assertEqual(phase.call_args.kwargs["split"], split_run.read_run_record(self.shards / "emb_run.run.json"))
        self.assertIs(phase.call_args.kwargs["refusals"], search._refusals)


#: A fresh process that loads torch first, as the CLI does, then runs an unmocked ``_whole_node_threads`` block and
#: prints FAISS's OpenMP count and FAISS's OpenBLAS's threads before, inside and after it (FaissWheelTest).
_D29_CHILD = """
import json, logging, os, sys
import torch
import faiss, threadpoolctl
from foldmatch.cli import search
from foldmatch.search.code_alignment import _physical_cpus
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
def faiss_blas():
    return [r["num_threads"] for r in threadpoolctl.threadpool_info() if r.get("user_api") == "blas"
            and os.path.basename(r["filepath"]).startswith("libopenblaso")]
before = (faiss.omp_get_max_threads(), faiss_blas())
with search._whole_node_threads():
    inside = (faiss.omp_get_max_threads(), faiss_blas())
after = (faiss.omp_get_max_threads(), faiss_blas())
print("D29 " + json.dumps({"n": _physical_cpus(), "before": before, "inside": inside, "after": after}))
"""

# The esm-codes in-process BLAS limit with the wheel: FAISS's count set apart from its OpenBLAS's (no BLAS call since,
# so the OpenBLAS still reports its own count), as after faiss.omp_set_num_threads in the Python API.
_ESM_CODES_LIMIT_CHILD = """
import json, os
import torch
import faiss, threadpoolctl
from foldmatch.search import code_alignment
def faiss_blas():
    return [r["num_threads"] for r in threadpoolctl.threadpool_info() if r.get("user_api") == "blas"
            and os.path.basename(r["filepath"]).startswith("libopenblaso")]
reported = faiss_blas()
k = 2 if reported != [2] else 3
faiss.omp_set_num_threads(k)
before = (faiss.omp_get_max_threads(), torch.get_num_threads(), faiss_blas())
with code_alignment._blas_one_thread_here():
    inside = faiss.omp_get_max_threads()
after = (faiss.omp_get_max_threads(), torch.get_num_threads())
print("ESM " + json.dumps({"k": k, "before": before, "inside": inside, "after": after}))
"""


class FaissWheelTest(unittest.TestCase):
    """P9: the pinned faiss-cpu wheel, unmocked. Linux only (the macOS wheels use Accelerate, which threadpoolctl does
    not list), and only when the installed FAISS is that wheel (not a conda build, not another version)."""

    def setUp(self):
        import faiss
        if not sys.platform.startswith("linux"):
            self.skipTest("the faiss-cpu wheels bundle an OpenBLAS on Linux only")
        if faiss.__version__ != search._FAISS_CPU_PIN or not os.path.isdir(_faiss_libs()):
            self.skipTest(f"FAISS {faiss.__version__} at {faiss.__file__} is not the faiss-cpu "
                          f"{search._FAISS_CPU_PIN} wheel")

    def test_the_wheel_blas_is_threaded(self):
        """threadpoolctl lists the wheel's BLAS in faiss_cpu.libs as an OpenMP OpenBLAS, and the index phase logs it
        without the single-threaded line (a pin to a wheel with a single-threaded BLAS fails here)."""
        import threadpoolctl
        runtimes = threadpoolctl.threadpool_info()
        own = [r for r in runtimes if r.get("user_api") == "blas"
               and os.path.dirname(os.path.realpath(r["filepath"])) == os.path.realpath(_faiss_libs())]
        self.assertEqual([(r["internal_api"], r.get("threading_layer")) for r in own], [("openblas", "openmp")],
                         runtimes)
        self.assertEqual(search._faiss_openmp_blas(runtimes), os.path.basename(os.path.realpath(own[0]["filepath"])))
        with self.assertLogs(level="INFO") as logs:
            search._log_index_phase(None)
        messages = [record.getMessage() for record in logs.records]
        self.assertFalse([m for m in messages if "single-threaded" in m], messages)
        self.assertEqual(len([m for m in messages if m.startswith("Thread pool: openblas, ")
                              and m.endswith(os.path.basename(own[0]["filepath"]))]), 1, messages)

    def test_d29_with_the_wheel(self):
        """D29 unmocked, in a fresh process with torch loaded first: the INFO line names FAISS's OpenMP OpenBLAS
        among the libraries limited; inside the block FAISS and that OpenBLAS run the physical cores (at most the
        wheel's 128); after it FAISS has its previous count again."""
        env = {k: v for k, v in os.environ.items() if k not in search._BLAS_THREAD_VARS
               and k not in _LAUNCH_ENV and k not in _DDP_ENV and not k.startswith(_DDP_ENV_PREFIXES)}
        env["PYTHONPATH"] = os.pathsep.join([_SRC] + [p for p in [os.environ.get("PYTHONPATH")] if p])
        run = subprocess.run([sys.executable, "-c", _D29_CHILD], env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-4000:])
        lines = run.stdout.splitlines()
        result = json.loads(next(line for line in lines if line.startswith("D29 "))[4:])
        n = result["n"]
        want = min(n, 128)
        limited = [line for line in lines if line.startswith("FAISS/BLAS threads set to ")]
        self.assertEqual(len(limited), 1, run.stdout)
        self.assertRegex(limited[0], rf"^FAISS/BLAS threads set to {n} \(physical cores of the \d+ allowed CPUs\); "
                                     rf"BLAS libraries limited: (.+, )?libopenblaso-[^,]+\.so(, .+)?$")
        self.assertEqual(result["inside"], [want, [want]], result)
        self.assertEqual(result["after"][0], result["before"][0], result)

    def test_the_esm_codes_in_process_limit_with_the_wheel(self):
        """The esm-codes Stage 2's in-process BLAS limit, unmocked, in a fresh process with torch loaded first: it
        reaches FAISS's OpenMP count (1 inside), and afterwards FAISS and torch have their counts of before, although
        FAISS's had been set apart from the count its OpenBLAS reports (threadpoolctl's restore alone gives them that
        stale count)."""
        env = {k: v for k, v in os.environ.items() if k not in search._BLAS_THREAD_VARS
               and k not in _LAUNCH_ENV and k not in _DDP_ENV and not k.startswith(_DDP_ENV_PREFIXES)}
        env["PYTHONPATH"] = os.pathsep.join([_SRC] + [p for p in [os.environ.get("PYTHONPATH")] if p])
        run = subprocess.run([sys.executable, "-c", _ESM_CODES_LIMIT_CHILD], env=env, capture_output=True, text=True,
                             timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-4000:])
        result = json.loads(next(line for line in run.stdout.splitlines() if line.startswith("ESM "))[4:])
        k, (faiss_before, torch_before, blas) = result["k"], result["before"]
        self.assertEqual(faiss_before, k, result)
        self.assertEqual(len(blas), 1, result)
        self.assertNotEqual(blas[0], k, result)              # the OpenBLAS reports another count: the case at issue
        self.assertEqual(result["inside"], 1, result)
        self.assertEqual(result["after"], [faiss_before, torch_before], result)


def _execute(path, sql):
    con = sqlite3.connect(path)
    try:
        con.execute(sql)
        con.commit()
    finally:
        con.close()


class IndexRefusalTest(_SplitBuild):

    def setUp(self):
        super().setUp()
        self.inputs = self.tmp / "in"
        self.inputs.mkdir()
        self.fasta = Path(shutil.copyfile(_FASTA, self.inputs / "seqs.fasta"))

    def refused_index(self, *extra, cause=None, env=None, shards=None):
        """Step 2 refused (exit 1, a RefusalError), before ESM3, leaving the shards folder and the database folder as
        they were (no lock, no database file, nothing changed; I22/I23 remove what they wrote). Returns the
        message."""
        before = _tree(self.out, shards or self.shards)
        result = self.index(*extra, fasta=self.fasta, shards=shards, env=env)
        message = self.refused(result, cause)
        self.assertEqual(_tree(self.out, shards or self.shards), before)
        return message

    def embed_run(self, *extra, **kwargs):
        return self.embed(*extra, fasta=self.fasta, **kwargs)

    # -- before the build lock ----------------------------------------------------------------------------------------
    def test_i1_a_malformed_run_id(self):
        self.embed_run()
        self.usage_error(self.index("--run-id", "a b", fasta=self.fasta), "is not a run id")

    def test_i2_one_of_several_ranks(self):
        self.embed_run()
        for env, launch in (({"WORLD_SIZE": 2}, "WORLD_SIZE=2"), ({"LOCAL_RANK": 1}, "LOCAL_RANK=1"),
                            ({"SLURM_JOB_ID": 7, "SLURM_STEP_ID": 0, "SLURM_STEP_NUM_TASKS": 2},
                             "SLURM_STEP_NUM_TASKS=2")):
            with self.subTest(env=env), mock.patch.object(split_run, "find_run") as find_run, \
                    mock.patch.object(alignment, "_expand_cpu_affinity") as widen:
                message = self.refused_index(env=env)
                find_run.assert_not_called()
                widen.assert_not_called()
            self.assertEqual(message, f"'build sequences-index' runs as one process ({launch}): launch it once, "
                                      f"directly in the batch script or with srun -n 1. The command:\n    "
                                      f"{_command_line(_index_args(self.fasta, self.db, self.shards))}")

    def test_i3_the_batch_step_of_a_multi_node_job(self):
        self.embed_run()
        with mock.patch.object(split_run, "find_run") as find_run:
            message = self.refused_index(env={"SLURM_JOB_ID": 4711, "SLURM_JOB_NUM_NODES": 2, "SLURM_NTASKS": 8})
        find_run.assert_not_called()
        self.assertIn("SLURM job 4711 holds 2 nodes", message)
        # No dependency on the job being refused: its afterok would never be met.
        self.assertTrue(message.endswith(
            "Run it as its own one-node CPU job instead: once this job has ended, sbatch -C cpu -N 1 ... SCRIPT (no "
            "dependency is needed then); next time, submit it from the GPU job's script before the job's work starts: "
            "sbatch -C cpu -N 1 --dependency=afterok:$SLURM_JOB_ID --kill-on-invalid-dep=yes ... SCRIPT. SCRIPT runs "
            "this command, without srun, next time " + _NEXT_BUILD + ":\n    "
            + _command_line(_index_args(self.fasta, self.db, self.shards))), message)
        self.assertNotIn("afterok:4711", message)

    def test_i2_i3_before_the_logging_setup(self):
        """P9.I1: I2 and I3 refuse before set_log_level (which, until P9.L2, imported Lightning and so torch)."""
        self.embed_run()
        for env in ({"WORLD_SIZE": 2}, {"SLURM_JOB_ID": 4711, "SLURM_JOB_NUM_NODES": 2, "SLURM_NTASKS": 8}):
            with self.subTest(env=env), mock.patch.object(search, "set_log_level",
                                                          side_effect=AssertionError("set_log_level")) as setup:
                self.refused_index(env=env)
                setup.assert_not_called()

    def test_i4_no_run_with_a_record(self):
        record = self.embed_run()
        empty = self.tmp / "empty"
        empty.mkdir()
        self.assertIn("No run in", self.refused_index(shards=empty))
        record.unlink()
        message = self.refused_index()
        self.assertIn("Run run in", message)
        self.assertIn("has no run record (1 shard(s), no run record): it did not finish, or its shards were kept by "
                      "'build sequences'", message)
        self.assertIn("'fm-search build sequences-seal --fasta-file FASTA --from-shards", message)
        for path in self.shards.iterdir():
            path.unlink()
        (self.shards / "emb_run.run.json.tmp").write_text("{}")
        message = self.refused_index()
        self.assertIn("an unfinished run record (.tmp)", message)
        self.assertIn(f"delete {self.shards.resolve() / 'emb_run.run.json.tmp'} first", message)

    def test_i5_several_runs(self):
        self.embed_run()
        self.embed_run(run_id="run2", db=self.tmp / "other")
        (self.shards / "emb_run3-0.parquet").write_bytes(b"a run without a record")
        message = self.refused_index()
        self.assertIn("holds 3 runs: run (run record, 1 shard(s)); run2 (run record, 1 shard(s)); run3 (1 shard(s), "
                      "no run record)", message)
        self.assertIn("Pass --run-id", message)
        self.built("--run-id", "run2", fasta=self.fasta)

    def test_i6_a_run_id_without_a_record(self):
        self.embed_run()
        message = self.refused_index("--run-id", "nope")
        self.assertIn("Run nope has no run record (emb_nope.run.json)", message)
        self.assertIn("Runs with a record here: run.", message)

    def test_i7_an_invalid_record(self):
        record = self.embed_run()
        original = record.read_text()
        cases = {
            "another kind": (lambda r: r.update(kind="other"), "kind is 'other'"),
            "a newer version": (lambda r: r.update(record_version=2), "upgrade FoldMatch"),
            "a missing key": (lambda r: r.pop("rows"), "the key rows is missing"),
            "another run id": (lambda r: r.update(run_id="other"), "but the file is named for run 'run'"),
        }
        for case, (change, text) in cases.items():
            with self.subTest(case=case):
                record.write_text(original)
                _edit_record(record, change)
                self.assertIn(text, self.refused_index())
        record.write_text("not json")
        self.assertIn("it is not JSON", self.refused_index())

    def test_i8_residue_codes_on_a_plain_run(self):
        self.embed_run()
        message = self.refused_index("--residue-codes")
        self.assertIn("Run run computed no residue codes", message)
        self.assertIn("run step 1 with --residue-codes", message)

    def test_i9_no_residue_codes_and_cleanup_on_a_codes_run(self):
        self.embed_run("--residue-codes")
        message = self.refused_index("--no-residue-codes", "--cleanup-embeddings")
        self.assertIn("would delete the only copy of run run's residue codes", message)

    def test_i10_the_run_system(self):
        record = self.embed_run("--residue-codes")
        with mock.patch.object(ss, "REGISTRY", MappingProxyType({})):
            message = self.refused_index(cause=ss.UnknownScoringSystemError)
        self.assertIn(f"unknown esm-codes scoring system '{SYSTEM.name}'", message)
        self.assertIn("--no-residue-codes builds the database without its residue codes.", message)
        _edit_record(record, lambda r: r["system"].update(digest="0" * 64))
        message = self.refused_index()
        self.assertIn(f"with digest {'0' * 64} and encoding_digest {SYSTEM.encoding_digest()}, but this "
                      f"FoldMatch's {SYSTEM.name} has digest {SYSTEM.digest()}", message)

    # -- inside the build lock, before any database file --------------------------------------------------------------
    def test_i11_a_build_lock(self):
        self.embed_run("--residue-codes")
        lock = Path(f"{self.db}.build.lock")
        lock.write_text('{"pid": 1, "host": "elsewhere"}\n')
        Path(f"{self.db}.index").touch()
        message = self.refused_index()
        self.assertIn(f"{lock} exists ({{\"pid\": 1, \"host\": \"elsewhere\"}})", message)
        self.assertIn("files present: db.index.", message)
        self.assertIn("delete the lock and these files, then rerun", message)
        self.assertNotIn("build residue-codes", message)
        Path(f"{self.db}.metadata").touch()
        SequenceStore(self.out, "db").create(self.fasta)
        message = self.refused_index()
        self.assertIn(f"'fm-search build residue-codes {self.out} db --from-shards {self.shards.resolve()} --run-id "
                      f"run' adds it (delete the lock first).", message)

    def test_i12_an_output_exists(self):
        """Refused inside the lock before the shards are read or the FASTA hashed (patched to fail), so not by the
        pre-flight's own outputs check, which comes after both."""
        self.embed_run("--residue-codes")
        def early():
            stack = ExitStack()
            stack.enter_context(mock.patch.object(split_run, "check_run_shards",
                                                  side_effect=AssertionError("shards read")))
            stack.enter_context(mock.patch.object(split_run, "fasta_sha256",
                                                  side_effect=AssertionError("FASTA hashed")))
            return stack

        for suffix in (".index", ".metadata", ".invlists", ".sequences", ".rescodes.bin", ".rescodes.idx",
                       ".rescodes.json", ".rescodes.json.tmp"):
            with self.subTest(suffix=suffix):
                existing = Path(f"{self.db}{suffix}")
                existing.touch()
                try:
                    with early():
                        message = self.refused_index(cause=FileExistsError)
                finally:
                    existing.unlink()
                self.assertIn(f"Database already exists at {self.db} (found: {existing})", message)
                self.assertNotIn("build residue-codes", message)
        for suffix in (".index", ".metadata"):
            Path(f"{self.db}{suffix}").touch()
        SequenceStore(self.out, "db").create(self.fasta)
        with early():
            message = self.refused_index()
        self.assertTrue(message.endswith(
            f"The database is complete but for its residue-code store: 'fm-search build residue-codes {self.out} db "
            f"--from-shards {self.shards.resolve()} --run-id run' adds it."), message)

    def test_i13_the_recorded_shards(self):
        self.embed_run(ranks=2)
        shard = self.shards / "emb_run-1.parquet"
        data = shard.read_bytes()
        shard.unlink()
        self.assertIn("missing emb_run-1.parquet", self.refused_index())
        shard.write_bytes(data)
        shutil.copyfile(shard, self.shards / "emb_run-7.parquet")
        self.assertIn("not in the record emb_run-7.parquet", self.refused_index())

    def test_i14_a_changed_shard(self):
        self.embed_run()
        shard = self.shards / "emb_run-0.parquet"
        data = shard.read_bytes()
        pq.write_table(pq.read_table(shard), shard, compression="none")
        message = self.refused_index()
        self.assertIn("1 shard(s) of run run in", message)
        self.assertIn("changed since the run: emb_run-0.parquet (", message)
        shard.write_bytes(b"\0" * len(data))  # the recorded size, no footer
        self.assertIn("the footer of a shard of run run cannot be read", self.refused_index())

    def test_i15_the_footers(self):
        """Codes: a pass record with another --min-res-n (the shard's size and rows written into the record); under
        --no-residue-codes, shards of another system than the record's; plain: a shard that is not plain."""
        def rewritten(record, change):
            _rewrite(self.shards / "emb_run-0.parquet", change)
            _resync(record)

        record = self.embed_run("--residue-codes")
        rewritten(record, _with_pass(min_res_n=5))
        self.assertIn("emb_run-0.parquet (min_res_n 5, world_size 1)", self.refused_index())
        self.assertIn("emb_run-0.parquet (min_res_n 5, world_size 1)", self.refused_index("--no-residue-codes"))
        for path in self.shards.iterdir():
            path.unlink()
        record = self.embed_run("--residue-codes")
        _edit_record(record, lambda r: r["system"].update(name="esm-codes-test-other"))
        message = self.refused_index("--no-residue-codes")
        self.assertIn(f"hold the residue codes of {SYSTEM.name}", message)
        self.assertIn("not those of esm-codes-test-other", message)
        for path in self.shards.iterdir():
            path.unlink()
        record = self.embed_run()
        rewritten(record, lambda table: table.replace_schema_metadata({b"foldmatch.residue_output": b"x"}))
        self.assertIn("are not plain embedding shards", self.refused_index())

    def test_i16_another_size(self):
        self.embed_run()
        size = self.fasta.stat().st_size
        with open(self.fasta, "a") as f:
            f.write(">late\nACDE\n")
        with mock.patch.object(split_run, "fasta_sha256", side_effect=AssertionError("hashed")):
            message = self.refused_index()
        self.assertIn(f"{self.fasta.resolve()} has {size + 11} bytes, but the FASTA run run embedded "
                      f"({self.fasta.resolve()}) had {size}", message)

    def test_the_folders_a_refused_build_created_are_removed(self):
        """A database folder that did not exist is created for the build lock and removed again by a refusal (here
        I16); one that existed stays."""
        self.embed_run()
        with open(self.fasta, "a") as f:
            f.write(">late\nACDE\n")
        new = self.tmp / "new"
        result = self.index(fasta=self.fasta, db=new / "deeper" / "db")
        self.assertIn("bytes, but the FASTA run run embedded", self.refused(result))
        self.assertFalse(new.exists())
        (new / "deeper").mkdir(parents=True)
        self.refused(self.index(fasta=self.fasta, db=new / "deeper" / "db"))
        self.assertEqual(list((new / "deeper").iterdir()), [])

    def test_i17_another_sha256(self):
        self.embed_run()
        data = bytearray(self.fasta.read_bytes())
        position = data.index(b"\n", data.index(b">")) + 1
        data[position] = ord("A") if data[position] != ord("A") else ord("C")
        self.fasta.write_bytes(bytes(data))
        message = self.refused_index()
        self.assertIn("is not the FASTA run run embedded", message)
        self.assertIn("though it has its size", message)

    def test_i17_the_given_fasta_is_hashed(self):
        """--fasta-file is what step 2 hashes: a same-size copy with one residue changed, given while the recorded
        FASTA is still in place, is refused; the recorded one is untouched."""
        self.embed_run()
        given = self.tmp / "other" / "seqs.fasta"
        given.parent.mkdir()
        _one_residue_changed(shutil.copyfile(self.fasta, given))
        recorded = self.fasta.read_bytes()
        before = _tree(self.out, self.shards)
        message = self.refused(self.index(fasta=given))
        self.assertIn(f"{given.resolve()} (sha256 ", message)
        self.assertIn(f"is not the FASTA run run embedded ({self.fasta.resolve()}, sha256", message)
        self.assertEqual((self.fasta.read_bytes(), _tree(self.out, self.shards)), (recorded, before))

    def test_i18_the_pre_flight(self):
        """The pre-flight's own refusal and nothing to build (mocked: the FASTA has the recorded bytes)."""
        self.embed_run()
        from foldmatch.utils import sequence
        with mock.patch.object(sequence.FastaPreflight, "check", side_effect=ValueError("a pre-flight refusal")):
            self.assertIn("a pre-flight refusal", self.refused_index(cause=ValueError))
        real = sequence.preflight_fasta

        def nothing(path, min_res_n):
            return dataclasses.replace(real(path, min_res_n), entries_kept=0)

        with mock.patch.object(sequence, "preflight_fasta", side_effect=nothing):
            self.assertIn("there is nothing to build", self.refused_index(cause=ValueError))

    def test_i19_other_counts(self):
        self.embed_run()
        from foldmatch.utils import sequence
        real = sequence.preflight_fasta

        def counting_otherwise(path, min_res_n):
            found = real(path, min_res_n)
            return dataclasses.replace(found, residues=found.residues + 1)

        with mock.patch.object(sequence, "preflight_fasta", side_effect=counting_otherwise):
            message = self.refused_index()
        self.assertIn("counts residues 304, but run run's record holds residues 303", message)
        self.assertIn("this FoldMatch's pre-flight counts differently", message)

    def test_i20_ids_not_at_their_dealt_positions(self):
        record = self.embed_run()
        shard = self.shards / "emb_run-0.parquet"
        _rewrite(shard, lambda table: table.set_column(0, table.schema.field("id"),
                                                       pa.array(table.column("id").to_pylist()[::-1],
                                                                table.schema.field("id").type)))
        _resync(record)
        message = self.refused_index()
        self.assertIn("2 shard row(s) of run run do not hold the id of the entry", message)
        self.assertIn("emb_run-0.parquet row 0 holds '2uzi_A', entry 0 is '1acb_E'", message)

    # -- the index phase ----------------------------------------------------------------------------------------------
    def test_i21_a_faiss_refusal(self):
        """ivf_pq on the fake 8-d embeddings (PQ m = 64 does not divide 8): nothing written, rerun with other
        options."""
        self.embed_run("--residue-codes")
        message = self.refused_index("--index-type", "ivf_pq", cause=ValueError)
        self.assertIn("must divide embedding dim D=8", message)
        self.assertIn("'build sequences-index' can be rerun with other index options, without a new ESM3 pass.",
                      message)
        self.built("--index-type", "flat", fasta=self.fasta)

    def test_i22_the_store_does_not_match_the_pre_flight(self):
        """The FASTA grows after FAISS, before the sequence store reads it: the database files are removed."""
        self.embed_run()
        create_db = EmbeddingDatabase.create_db

        def then_grows(db, *args, **kwargs):
            create_db(db, *args, **kwargs)
            with open(self.fasta, "a") as f:
                f.write(">late\nACDEFGHIKL\n")

        with mock.patch.object(EmbeddingDatabase, "create_db", then_grows):
            message = self.refused_index(cause=ValueError)
        self.assertIn("does not match its pre-flight", message)
        self.assertIn("The database files it wrote were removed (db.index, db.metadata, db.sequences)", message)

    def test_i23_the_fasta_changed_during_the_build(self):
        """The FASTA rewritten with one residue changed (the same size and counts, another sha256) while the sequence
        store is built: refused by the stat bracket, the database files removed (also when the FASTA is gone)."""
        self.embed_run("--residue-codes")
        create = SequenceStore.create

        def rewrite(store, *args, **kwargs):
            create(store, *args, **kwargs)
            _one_residue_changed(self.fasta)

        with mock.patch.object(SequenceStore, "create", autospec=True, side_effect=rewrite):
            message = self.refused_index()
        self.assertIn(f"The FASTA changed during this build ({self.fasta.resolve()}: ", message)
        self.assertIn("The database files it wrote were removed (db.index, db.metadata, db.sequences).", message)
        shutil.copyfile(_FASTA, self.fasta)
        os.utime(self.fasta, ns=(0, split_run.read_run_record(self.shards / "emb_run.run.json")["fasta"]["mtime_ns"]))

        def removed(store, *args, **kwargs):
            create(store, *args, **kwargs)
            self.fasta.rename(self.inputs / "gone.fasta")

        with mock.patch.object(SequenceStore, "create", autospec=True, side_effect=removed):
            message = self.refused_index()
        self.assertIn("cannot be stat'ed", message)
        self.assertIn("The database files it wrote were removed", message)

    def test_i23_the_bracket_opens_before_the_hash(self):
        """The bracket starts with the stat taken before the FASTA is hashed: a change during the id check (after the
        hash and the pre-flight, before the index phase) is refused at the sequence store, which read the new bytes;
        no database file is left."""
        self.embed_run()
        check = split_run.check_dealt_ids

        def then_changed(*args):
            seconds = check(*args)
            _one_residue_changed(self.fasta)
            return seconds

        with mock.patch.object(split_run, "check_dealt_ids", side_effect=then_changed):
            message = self.refused_index()
        self.assertIn(f"The FASTA changed during this build ({self.fasta.resolve()}: ", message)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_i24_the_code_store_fails_validation(self):
        """An entry with an empty payload: the code store is removed; the database, the record and the shards stay,
        also with --cleanup-embeddings; no lock is left."""
        def payload(name, sequence):
            return np.empty((0, 8), np.uint8) if name == "2uzi_A" else _fake_codes(name, sequence)

        self.embed_run("--residue-codes", payload=payload)
        before = _tree(self.shards)
        result = self.index("--cleanup-embeddings", fasta=self.fasta)
        message = self.refused(result, CodeStoreValidationError)
        self.assertIn("empty payload", message)
        self.assertIn(f"This build's shards are kept in {self.shards.resolve()} (run id run).", message)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["db.index", "db.metadata", "db.sequences"])
        self.assertEqual(_tree(self.shards), before)


# =====================================================================================================================
# P8b.5: build sequences-seal
# =====================================================================================================================
#: Plan P8b (CLI), verbatim: the seal's command help and the help texts it does not take from 'build sequences'.
SEAL_HELP = (
    "Recovery: write the run record of embedding shards that have none, so that 'build sequences-index' can build from "
    "them: a 'build sequences-embed' run whose rank 0 stopped after the pass (wall time, out of memory, a full disk), "
    "or the shards a 'build sequences' run kept when its index phase failed. Checks the shards against the FASTA "
    "(entry count, each rank's rows, the shard schema and the id of every row) without ESM3 or a GPU, in one process. "
    "The record says it was sealed: it binds the shards by these checks, not by the pass."
)
SEAL_OWN_HELP = {
    "fasta_file": "The FASTA file the run embedded.",
    "from_shards": "Folder holding the run's shards.",
    "run_id": "Run id of the shards to seal (emb_<run id>-<rank>.parquet).",
    "min_res_n": "The --min-res-n the run used. Required for a run without residue codes, whose shards do not record "
                 "it; for a run with codes it is read from the shards, and a different value is refused.",
}
#: The record fields a seal writes otherwise than step 1 (plan P8b.5, step 10).
_SEALED = {"sealed": True, "command": "build sequences-seal", "options": {}, "output_db": None}


def _seal_args(fasta, shards, run_id, *extra):
    return ("build", "sequences-seal", "--fasta-file", fasta, "--from-shards", shards, "--run-id", run_id, *extra)


def _codes_flag(codes):
    return ("--residue-codes",) if codes else ()


class _Sealing(_SplitBuild):
    """``_SplitBuild`` plus the seal, under the ESM3 spy with ``scoring_system.prepare`` refusing, and runs that have
    no record."""

    def seal(self, *extra, fasta=_FASTA, shards=None, run_id="run", env=None):
        """``build sequences-seal`` with ``env`` set and ``sys.argv`` as the console script sets it; returns the
        result."""
        args = _seal_args(fasta, shards or self.shards, run_id, *extra)
        with _esm3_spy() as esm3, mock.patch.object(ss, "prepare", side_effect=AssertionError("bundle fetched")), \
                mock.patch.dict(os.environ, {name: str(value) for name, value in (env or {}).items()}), \
                _as_fm_search(args):
            result = invoke(*args)
        esm3.assert_not_called()
        return result

    def sealed(self, *extra, shards=None, run_id="run", **kwargs):
        """The seal, which must succeed; returns its record's path."""
        result = self.seal(*extra, shards=shards, run_id=run_id, **kwargs)
        self.assertEqual(result.exit_code, 0, f"{result.output} {result.exception!r}")
        return Path(shards or self.shards) / f"emb_{run_id}.run.json"

    def unrecorded(self, *extra, **kwargs):
        """Step 1 whose rank 0 stopped after the pass: its record deleted. Returns that record."""
        path = self.embed(*extra, **kwargs)
        record = split_run.read_run_record(path)
        path.unlink()
        return record


class SealOptionTest(_Tmp):

    def test_the_command_help(self):
        self.assertEqual(_command("build", "sequences-seal").help, SEAL_HELP)
        output = CliRunner().invoke(search.app, ["build", "sequences-seal", "--help"], env={"COLUMNS": "400"}).output
        for text in (SEAL_HELP, *SEAL_OWN_HELP.values()):
            with self.subTest(text=text[:40]):
                self.assertIn(text, _flat(output))

    def test_the_options(self):
        """Exactly the seal's options, with the plan's help; --fasta-file and --from-shards are step 2's (required,
        must exist), --log-level the one-shot's; --run-id is required; --min-res-n is optional and at least 0."""
        seal = {param.name: param for param in _command("build", "sequences-seal").params}
        oneshot = {param.name: param for param in _command("build", "sequences").params}
        index = {param.name: param for param in _command("build", "sequences-index").params}
        self.assertEqual(sorted(seal), sorted(["fasta_file", "from_shards", "run_id", "min_res_n", "log_level"]))

        def info(param):
            out = param.to_info_dict()
            for key in ("name", "help"):
                out.pop(key)
            return out

        for name, param in seal.items():
            with self.subTest(name=name):
                self.assertEqual(param.help, SEAL_OWN_HELP[name] if name in SEAL_OWN_HELP else oneshot[name].help)
        for name, same in (("fasta_file", index), ("from_shards", index), ("log_level", oneshot)):
            with self.subTest(name=name):
                self.assertEqual(info(seal[name]), info(same[name]))
        run_id = seal["run_id"]
        self.assertEqual((run_id.opts, run_id.required, run_id.type.name), (["--run-id"], True, "text"))
        min_res_n = seal["min_res_n"]
        self.assertEqual((min_res_n.opts, min_res_n.required, min_res_n.default, min_res_n.type.min,
                          min_res_n.type.max), (["--min-res-n"], False, None, 0, None))

    def test_options_another_command_owns(self):
        """Every pass, index and output option, --residue-codes (the shards say whether they hold codes) and
        --cleanup-embeddings: exit 2, nothing read or written. S1: a malformed --run-id, a negative --min-res-n and
        no --run-id are exit 2 too."""
        for option in (("--output-db", self.tmp / "db"), ("--shards-folder", self.shards),
                       ("--tmp-embedding-folder", self.shards), ("--accelerator", "cpu"), ("--num-nodes", "1"),
                       ("--devices", "1"), ("--strategy", "auto"), ("--batch-size", "1"), ("--num-workers", "0"),
                       ("--use-gpu-index",), ("--index-type", "flat"), ("--ivf-nlist", "256"), ("--ivf-nprobe", "4"),
                       ("--load-batch-size", "1000"), ("--residue-codes",), ("--no-residue-codes",),
                       ("--cleanup-embeddings",), ("--force",)):
            with self.subTest(option=option[0]), _esm3_spy() as esm3:
                result = invoke(*_seal_args(_FASTA, self.shards, "run", *option))
                self.usage_error(result, f"No such option: {option[0]}")
                esm3.assert_not_called()
        for bad in ("a b", "a/b", "a*"):
            with self.subTest(run_id=bad), mock.patch.object(split_run, "refuse_existing_record") as refuse:
                self.usage_error(invoke(*_seal_args(_FASTA, self.shards, bad)), "is not a run id")
                refuse.assert_not_called()
        self.usage_error(invoke(*_seal_args(_FASTA, self.shards, "run", "--min-res-n", "-1")), "--min-res-n")
        self.usage_error(invoke("build", "sequences-seal", "--fasta-file", _FASTA, "--from-shards", self.shards),
                         "--run-id")
        self.assertEqual(list(self.shards.iterdir()), [])


class SealTest(_Sealing):

    def test_the_shards_a_failed_one_shot_kept(self):
        """A 'build sequences' whose index phase fails after the pass (ivf_pq on the fake 8-d embeddings), plain (the
        seal given --min-res-n) and codes (the seal reads it from the shards), on 1 and 2 ranks, with --min-res-n 0 and
        100: sealed, then step 2 with a flat index builds the database of a flat one-shot."""
        for codes in (False, True):
            for ranks in (1, 2):
                for min_res_n in (0, 100):
                    name = f"{'codes' if codes else 'plain'}{ranks}-{min_res_n}"
                    with self.subTest(codes=codes, ranks=ranks, min_res_n=min_res_n):
                        extra = ("--min-res-n", min_res_n, *_codes_flag(codes))
                        shards = self.tmp / f"kept-{name}"
                        shards.mkdir()
                        with launch_env(FOLDMATCH_RUN_ID=f"kept-{name}"), \
                                mock.patch("foldmatch.search.embedding_computer.chain_predict",
                                           fake_chain_predict(ranks=ranks)):
                            failed = invoke(*_build_args(_FASTA, self.tmp / f"failed-{name}" / "db", shards, *extra,
                                                         "--index-type", "ivf_pq"))
                        self.assertEqual(failed.exit_code, 1, failed.output)
                        self.assertIn("must divide embedding dim D=8", str(failed.exception))
                        path = self.sealed(*(() if codes else ("--min-res-n", min_res_n)), shards=shards,
                                           run_id=f"kept-{name}")
                        record = split_run.read_run_record(path)
                        kept = 2 if min_res_n == 0 else 1
                        self.assertEqual((record["sealed"], record["min_res_n"], record["rows"], record["world_size"]),
                                         (True, min_res_n, kept, ranks if codes else min(ranks, kept)))
                        db = self.tmp / f"split-{name}" / "db"
                        self.built("--index-type", "flat", shards=shards, db=db)
                        assert_same_database(self.oneshot(*extra, "--index-type", "flat", run_id=f"one-{name}",
                                                          ranks=ranks), db)

    def test_a_lost_record(self):
        """A step-1 run whose record was deleted, plain and codes, on 1 and 2 ranks, with --min-res-n 0 and 100 (one
        entry, so one shard): the sealed record is the lost one except sealed, command, options, output_db and
        created_utc and, plain, the pass's ESM3 revision (no plain shard records it); the record is the only new
        file."""
        for codes in (False, True):
            for ranks, min_res_n in ((1, 0), (2, 0), (1, 100)):
                name = f"{'codes' if codes else 'plain'}{ranks}-{min_res_n}"
                with self.subTest(codes=codes, ranks=ranks, min_res_n=min_res_n):
                    shards = self.tmp / f"shards-{name}"
                    shards.mkdir()
                    lost = self.unrecorded("--min-res-n", min_res_n, *_codes_flag(codes), ranks=ranks, run_id=name,
                                           shards=shards, db=self.tmp / f"db-{name}")
                    before = _tree(shards)
                    path = self.sealed(*(() if codes else ("--min-res-n", min_res_n)), shards=shards, run_id=name)
                    sealed = split_run.read_run_record(path)
                    self.assertEqual(json.loads(path.read_text()), sealed)
                    self.assertEqual({k: v for k, v in _tree(shards).items() if k != str(path)}, before)
                    created = datetime.strptime(sealed["created_utc"], "%Y-%m-%dT%H:%M:%SZ")
                    self.assertLess(abs(datetime.now(timezone.utc).replace(tzinfo=None) - created),
                                    timedelta(minutes=5))
                    expected = {**lost, **_SEALED, "created_utc": sealed["created_utc"]}
                    if not codes:
                        self.assertEqual(lost["pass"]["esm_revision"], ESM_REVISION)
                        expected["pass"] = {**lost["pass"], "esm_revision": None}
                    self.assertEqual(sealed, expected)

    def test_the_seal_command_step_1_names(self):
        """Step 1 on 2 ranks refused after it hashed the FASTA (E12: a full disk while the record is staged): the seal
        command its refusal names, run verbatim (paths with spaces), records the shards, and step 2 builds the
        one-shot's database. Plain and codes (the command gives --min-res-n in both modes)."""
        for codes in (False, True):
            name = "codes" if codes else "plain"
            with self.subTest(codes=codes):
                shards = self.tmp / f"shard s {name}"
                shards.mkdir()
                with launch_env(FOLDMATCH_RUN_ID=name), \
                        mock.patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict(ranks=2)), \
                        mock.patch.object(search, "_gather_rank_facts", side_effect=_ranks(2)), \
                        mock.patch.object(split_run, "_write_all",
                                          side_effect=OSError(errno.ENOSPC, "No space left on device")):
                    result = invoke(*_embed_args(_FASTA, self.tmp / f"db-{name}", shards, *_codes_flag(codes)))
                message = self.refused(result)
                hint = "Once the cause is fixed, 'build sequences-seal' can record them: "
                argv = shlex.split(message[message.index(hint) + len(hint):])
                self.assertEqual(argv[:3], ["fm-search", "build", "sequences-seal"])
                self.assertEqual(split_run.record_run_ids(shards), [])
                with _esm3_spy() as esm3, mock.patch.object(ss, "prepare",
                                                            side_effect=AssertionError("bundle fetched")):
                    sealed = invoke(*argv[1:])
                esm3.assert_not_called()
                self.assertEqual(sealed.exit_code, 0, f"{sealed.output} {sealed.exception!r}")
                self.assertEqual(split_run.read_run_record(shards / f"emb_{name}.run.json")["world_size"], 2)
                db = self.tmp / f"split-{name}" / "db"
                self.built(shards=shards, db=db)
                assert_same_database(self.oneshot(*_codes_flag(codes), run_id=f"one-{name}", ranks=2), db)

    def test_a_world_larger_than_its_entries(self):
        """A pass on 3 ranks over 2 entries writes 2 shards: a codes run's sealed record keeps its pass's world of 3, a
        plain run's world is its 2 shards (the same deal). Both build the one-shot's database."""
        for codes in (False, True):
            name = "codes" if codes else "plain"
            with self.subTest(codes=codes):
                shards = self.tmp / f"shards-{name}"
                shards.mkdir()
                lost = self.unrecorded(*_codes_flag(codes), ranks=3, run_id=name, shards=shards,
                                       db=self.tmp / f"x-{name}")
                self.assertEqual((lost["world_size"], len(lost["shards"])), (3, 2))
                sealed = split_run.read_run_record(self.sealed(*(() if codes else ("--min-res-n", "0")),
                                                               shards=shards, run_id=name))
                self.assertEqual((sealed["world_size"], sealed["pass"]["world_size"]), (3, 3) if codes else (2, 2))
                self.assertEqual(sealed["shards"], lost["shards"])
                db = self.tmp / f"split-{name}" / "db"
                self.built(shards=shards, db=db)
                assert_same_database(self.oneshot(*_codes_flag(codes), run_id=f"one-{name}", ranks=3), db)

    def test_the_log_lines_and_the_next_command(self):
        """The FASTA hash and the id check are timed; then the run sealed, the WARNING that the record binds the
        shards by these checks (and seq_sha1 with codes), and step 2's command, which builds the database once DB is
        given (paths with spaces, split as a shell splits them)."""
        folder = self.tmp / "with space"
        folder.mkdir()
        fasta = Path(shutil.copyfile(_FASTA, folder / "my seqs.fasta"))
        for codes in (True, False):
            with self.subTest(codes=codes):
                shards = folder / f"shard s {codes}"
                shards.mkdir()
                self.unrecorded(*_codes_flag(codes), ranks=2, fasta=fasta, shards=shards, db=self.tmp / f"db-{codes}")
                messages = self.logged(lambda: self.sealed(*(() if codes else ("--min-res-n", "0")), fasta=fasta,
                                                           shards=shards))
                found = [m for _, m in messages]
                size = fasta.stat().st_size
                prefixes = (f"FASTA sha256 of {fasta.resolve()} ({size} bytes) in ",
                            f"Ids{' and seq_sha1' if codes else ''} of the 2 shard rows checked against "
                            f"{fasta.resolve()} in ", "Run run sealed: ", "The record of run run is sealed: ",
                            "Build the database DB with: ")
                lines = [[m for m in found if m.startswith(prefix)] for prefix in prefixes]
                self.assertEqual([len(line) for line in lines], [1] * 5, found)
                lines = [line[0] for line in lines]
                order = [found.index(line) for line in lines]
                self.assertEqual(order, sorted(order))
                for line in lines[:2]:
                    self.assertRegex(line, r" in \d+\.\d s$")
                self.assertEqual(lines[2], f"Run run sealed: 2 entries in 2 shard(s) of 2 rank(s)"
                                           f"{f', residue codes of {SYSTEM.name}' if codes else ''}; record "
                                           f"{shards.resolve() / 'emb_run.run.json'}")
                self.assertIn(("WARNING", lines[3]), messages)
                self.assertIn(f"binds the shards to {fasta.resolve()} by their entry count, each rank's rows and the "
                              f"id of every row", lines[3])
                self.assertEqual("seq_sha1" in lines[3], codes)
                self.assertEqual(" and its seq_sha1, the sha1 of the entry's sequence, not by the pass" in lines[3],
                                 codes)
                self.assertTrue(lines[3].endswith(", not by the pass that wrote them, which 'build sequences-seal' did "
                                                  "not see."), lines[3])
                argv = shlex.split(lines[4][len("Build the database DB with: "):])
                self.assertEqual(argv, ["fm-search", "build", "sequences-index", "--fasta-file", str(fasta.resolve()),
                                        "--output-db", "DB", "--from-shards", str(shards.resolve()), "--run-id", "run"])
                db = folder / f"my db {codes}"
                argv[argv.index("DB")] = str(db)
                with _esm3_spy() as esm3, mock.patch.object(ss, "prepare",
                                                            side_effect=AssertionError("bundle fetched")):
                    result = invoke(*argv[1:])
                esm3.assert_not_called()
                self.assertEqual(result.exit_code, 0, result.output)
                assert_same_database(self.oneshot(*_codes_flag(codes), run_id=f"one-{codes}", fasta=fasta, ranks=2), db)

    def test_offline_with_an_empty_hf_cache(self):
        """The seal as a command, with HF_HUB_OFFLINE=1 and an empty Hugging Face cache: a codes run is sealed from its
        shards' footers, without the bundle."""
        self.unrecorded("--residue-codes")
        empty = self.tmp / "empty-hf-cache"
        empty.mkdir()
        env = {k: v for k, v in os.environ.items()
               if k not in _LAUNCH_ENV and k not in _DDP_ENV and not k.startswith(_DDP_ENV_PREFIXES)}
        env.update(HF_HUB_OFFLINE="1", HF_HUB_CACHE=str(empty), HF_HOME=str(empty),
                   PYTHONPATH=os.pathsep.join([_SRC] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        run = subprocess.run([sys.executable, "-m", "foldmatch.cli.search", *map(str, _seal_args(
            _FASTA, self.shards, "run"))], env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-4000:])
        self.assertEqual(split_run.read_run_record(self.shards / "emb_run.run.json")["system"]["name"], SYSTEM.name)
        self.assertEqual(list(empty.rglob("models--*")), [])

    def test_a_system_unknown_here(self):
        """The seal needs no registry entry: a codes run whose system this FoldMatch does not register is sealed;
        step 2 then refuses it (I10) or builds the plain database (--no-residue-codes)."""
        self.unrecorded("--residue-codes")
        with mock.patch.object(ss, "REGISTRY", MappingProxyType({})):
            self.sealed()
            self.assertIn("unknown esm-codes scoring system", self.refused(self.index()))
            self.built("--no-residue-codes")
        assert_same_database(self.oneshot(run_id="plain"), self.db)

    def test_the_foldmatch_of_a_sealed_pass(self):
        """Step 1 by another FoldMatch, its record lost, sealed by this one: the sealed record holds the seal's version,
        and step 2 takes the pass's from the codes shards' pass record (D30 WARNING, start line). A plain run's pass
        version is not recorded anywhere (its pass record is the seal's): no D30 WARNING, and the sealed WARNING says
        the version is unknown."""
        for codes in (True, False):
            with self.subTest(codes=codes):
                for path in [*self.shards.iterdir(), *self.out.iterdir()]:
                    path.unlink()
                with mock.patch.object(foldmatch, "__version__", "0.7.0-old"), \
                        mock.patch.object(search, "__version__", "0.7.0-old"):
                    lost = self.unrecorded(*_codes_flag(codes))
                self.assertEqual((lost["foldmatch_version"], lost["pass"]["foldmatch_version"]), ("0.7.0-old",) * 2)
                record = split_run.read_run_record(self.sealed(*(() if codes else ("--min-res-n", "0"))))
                self.assertEqual((record["foldmatch_version"], record["pass"]["foldmatch_version"]),
                                 (foldmatch.__version__, "0.7.0-old" if codes else foldmatch.__version__))
                messages = self.logged(lambda: self.built())
                warnings = [m for level, m in messages if level == "WARNING"]
                d30 = [m for m in warnings if m.startswith("Run run was embedded by FoldMatch ")]
                self.assertEqual([m.split(",")[0] for m in d30], ["Run run was embedded by FoldMatch 0.7.0-old"]
                                 if codes else [])
                sealed = next(m for m in warnings if "written by 'build sequences-seal'" in m)
                self.assertIn(f"(FoldMatch {foldmatch.__version__})", sealed)
                self.assertEqual(sealed.endswith("whose FoldMatch version is not recorded (shards without residue "
                                                 "codes do not record it)."), not codes)
                start = next(m for _, m in messages if m.startswith(f"Building {self.db} from run run "))
                self.assertIn(f"sealed {record['created_utc']} by FoldMatch {foldmatch.__version__}; embedded by "
                              + ("FoldMatch 0.7.0-old on" if codes else "a FoldMatch it does not record on"), start)

    def test_launch_warnings_after_the_logging_setup(self):
        """P9.I1: a one-task srun step of a multi-node job, and a one-node job with GPUs: sealed, with the WARNING of
        D34 logged once set_log_level has set up the logging."""
        cases = {
            "an srun step": ({"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 2, "SLURM_STEP_ID": 3,
                              "SLURM_STEP_NUM_TASKS": 1, "SLURM_NTASKS": 8},
                             "'build sequences-seal' runs in a one-task srun step of SLURM job 42, which holds 2 "
                             "nodes: the others stay allocated until it ends. Next time, run it as its own one-node "
                             "CPU job: submit it from the GPU job's script before the job's work starts: sbatch -C cpu "
                             "-N 1 --dependency=afterok:$SLURM_JOB_ID --kill-on-invalid-dep=yes ... SCRIPT. SCRIPT "
                             "runs this command, without srun, " + _NEXT_BUILD + ":\n    {command}"),
            "GPUs": ({"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 1, "SLURM_JOB_GPUS": "0,1"},
                     "SLURM job 42 has GPUs (SLURM_JOB_GPUS=0,1), which 'build sequences-seal' does not use: run it as "
                     "its own job on a CPU node (sbatch -C cpu -N 1 ...); a project with GPU hours only may run it on "
                     "one GPU node, where ivf_pq fits (flat and hnsw may not)."),
        }
        for k, (case, (env, text)) in enumerate(cases.items()):
            with self.subTest(case=case):
                shards = self.tmp / f"w{k}"
                shards.mkdir()
                self.unrecorded(shards=shards)
                events = self.setup_and_warnings(lambda: self.sealed("--min-res-n", 0, env=env, shards=shards))
                text = text.format(command=_command_line(_seal_args(_FASTA, shards, "run", "--min-res-n", 0)))
                self.assertEqual(events[:2], [("set_log_level", LogLevel.info), ("WARNING", text)], events)


class SealRefusalTest(_Sealing):

    def setUp(self):
        super().setUp()
        self.inputs = self.tmp / "in"
        self.inputs.mkdir()
        self.fasta = Path(shutil.copyfile(_FASTA, self.inputs / "seqs.fasta"))

    def refused_seal(self, *extra, cause=None, env=None, shards=None, run_id="run", fasta=None):
        """The seal refused (exit 1, a RefusalError), before ESM3, writing nothing: the shards folder and the database
        folder as they were. Returns the message."""
        before = _tree(self.out, shards or self.shards)
        result = self.seal(*extra, fasta=fasta or self.fasta, shards=shards, run_id=run_id, env=env)
        message = self.refused(result, cause)
        self.assertEqual(_tree(self.out, shards or self.shards), before)
        return message

    def unrecorded_run(self, *extra, **kwargs):
        return self.unrecorded(*extra, **{"fasta": self.fasta, **kwargs})

    def clear(self):
        for path in self.shards.iterdir():
            path.unlink()

    def shard(self, rank=0):
        return self.shards / f"emb_run-{rank}.parquet"

    def test_s1_the_flags(self):
        self.unrecorded_run()
        self.usage_error(self.seal("--min-res-n", "-1", fasta=self.fasta), "--min-res-n")
        self.usage_error(self.seal(fasta=self.fasta, run_id="a b"), "is not a run id")
        self.assertEqual(split_run.record_run_ids(self.shards), [])

    def test_s2_the_launch(self):
        """One of several ranks (I2), and the batch step of a multi-node job (I3): refused before the folder is
        read, and before set_log_level (P9.I1)."""
        self.unrecorded_run()
        command = _command_line(_seal_args(self.fasta, self.shards, "run"))
        i2 = f" The command:\n    {command}"
        i3 = f" SCRIPT. SCRIPT runs this command, without srun, next time {_NEXT_BUILD}:\n    {command}"
        for env, text, end in (({"WORLD_SIZE": 2}, "'build sequences-seal' runs as one process (WORLD_SIZE=2): launch "
                                                   "it once, directly in the batch script or with srun -n 1.", i2),
                               ({"LOCAL_RANK": 1}, "(LOCAL_RANK=1)", i2),
                               ({"SLURM_JOB_ID": 7, "SLURM_STEP_ID": 0, "SLURM_STEP_NUM_TASKS": 2},
                                "(SLURM_STEP_NUM_TASKS=2)", i2),
                               ({"SLURM_JOB_ID": 4711, "SLURM_JOB_NUM_NODES": 2, "SLURM_NTASKS": 8},
                                "SLURM job 4711 holds 2 nodes, and 'build sequences-seal' runs in one process", i3)):
            with self.subTest(env=env), mock.patch.object(split_run, "refuse_existing_record") as refuse, \
                    mock.patch.object(search, "set_log_level", side_effect=AssertionError("set_log_level")) as setup:
                message = self.refused_seal(env=env)
                self.assertIn(text, message)
                self.assertTrue(message.endswith(end), message)
                refuse.assert_not_called()
                setup.assert_not_called()  # P9.I1: before set_log_level (until P9.L2 it imported Lightning)

    def test_s3_a_record_or_tmp(self):
        """A run with a record: use it. A record whose writer stopped (.tmp only): delete it, then seal again."""
        record = self.embed(fasta=self.fasta)
        message = self.refused_seal()
        self.assertIn(f"{self.shards.resolve() / 'emb_run.run.json'} exists: run run in {self.shards.resolve()} has "
                      f"its run record already, so there is nothing to seal.", message)
        self.assertIn(f"Use it: 'build sequences-index --from-shards {self.shards.resolve()} --run-id run'", message)
        record.rename(self.shards / "emb_run.run.json.tmp")
        message = self.refused_seal()
        self.assertIn(f"{self.shards.resolve() / 'emb_run.run.json.tmp'} exists", message)
        self.assertIn("Delete the .tmp, then run 'build sequences-seal' again.", message)

    def test_s4_no_shards_of_the_run(self):
        self.unrecorded_run()
        self.embed(fasta=self.fasta, run_id="done", db=self.tmp / "other")
        message = self.refused_seal(run_id="nope")
        self.assertIn(f"No embedding shards of run nope (emb_nope-<rank>.parquet) in {self.shards.resolve()}, so there "
                      f"is nothing to seal. It holds run done (run record, 1 shard(s)); run run (1 shard(s), no run "
                      f"record).", message)
        empty = self.tmp / "empty"
        empty.mkdir()
        self.assertTrue(self.refused_seal(shards=empty).endswith("It holds no run."))

    def test_s5_the_footers(self):
        """Plain and codes shards mixed; codes shards of two encodings; a codes shard without a pass record; residue
        shards of another kind (query encodings); a plain shard that is not legacy; a shard that is no Parquet file.
        Each names the shards."""
        def metadata(change):
            def rewrite(table):
                meta = dict(table.schema.metadata)
                change(meta)
                return table.replace_schema_metadata(meta)
            return rewrite

        cases = {
            "plain and codes": (True, 1, lambda t: t.select(["id", "embedding"]).replace_schema_metadata(None),
                                ("The 2 shards of run run in", "are mixed: ", "no residue column (a plain pass) (e.g. "
                                 "emb_run-1.parquet)", "residue_codes of")),
            "two encodings": (True, 1, metadata(lambda m: m.update({b"foldmatch.encoding_digest": b"0" * 64})),
                              ("are mixed: ", "encoding_digest " + "0" * 64)),
            "no pass record": (True, 0, metadata(lambda m: m.pop(META_PASS)),
                               ("1 of the 2 shard(s) of run run in", "record no pass (foldmatch.pass): "
                                "emb_run-0.parquet.")),
            "query encodings": (True, None, metadata(lambda m: m.update({b"foldmatch.residue_output":
                                                                         b"query_encoding"})),
                                ("hold query_encoding of", f"not the residue codes of {SYSTEM.name}")),
            "not legacy": (False, None, lambda t: t.replace_schema_metadata({b"foldmatch.system": b"x"}),
                           ("are not plain embedding shards", "the metadata foldmatch.system")),
        }
        for case, (codes, rank, change, texts) in cases.items():
            with self.subTest(case=case):
                self.clear()
                self.unrecorded_run(*_codes_flag(codes), ranks=2)
                for r in ((0, 1) if rank is None else (rank,)):
                    _rewrite(self.shard(r), change)
                message = self.refused_seal(*(() if codes else ("--min-res-n", "0")))
                for text in texts:
                    self.assertIn(text, message)
        with self.subTest(case="no Parquet file"):
            self.clear()
            self.unrecorded_run("--residue-codes")
            self.shard().write_bytes(b"not a shard")
            self.assertIn("emb_run-0.parquet: not a readable Parquet embedding shard", self.refused_seal())

    def test_s6_the_pass(self):
        """A plain run without --min-res-n; a codes run with another --min-res-n than its shards'; codes shards that
        disagree on their pass's world size or --min-res-n, or record no valid one; a world smaller than the shard
        count. Each gives both values."""
        self.unrecorded_run()
        self.assertIn("hold no residue codes, so they do not record the --min-res-n of their pass: give the "
                      "--min-res-n the run used.", self.refused_seal())
        cases = {
            "another --min-res-n": ((), ("--min-res-n", "100"),
                                    ("--min-res-n is 100, but the shards of run run in", "record a pass with "
                                     "--min-res-n 0: leave --min-res-n out")),
            "world sizes disagree": (((1, _with_pass(world_size=3)),), (),
                                     ("disagree on their pass's world_size: 2 in emb_run-0.parquet, 3 in "
                                      "emb_run-1.parquet.",)),
            "min_res_n disagree": (((1, _with_pass(min_res_n=5)),), (),
                                   ("disagree on their pass's min_res_n: 0 in emb_run-0.parquet, 5 in "
                                    "emb_run-1.parquet.",)),
            "no valid min_res_n": (((0, _with_pass(min_res_n=None)), (1, _with_pass(min_res_n=None))), (),
                                   ("record a pass without a valid --min-res-n and world size (min_res_n None, "
                                    "world_size 2)",)),
            "a world smaller than the shards": (((0, _with_pass(world_size=1)), (1, _with_pass(world_size=1))), (),
                                                ("record a pass on 1 rank(s), but there are 2 of them",)),
        }
        for case, (changes, extra, texts) in cases.items():
            with self.subTest(case=case):
                self.clear()
                self.unrecorded_run("--residue-codes", ranks=2)
                for rank, change in changes:
                    _rewrite(self.shard(rank), change)
                message = self.refused_seal(*extra)
                for text in texts:
                    self.assertIn(text, message)

    def test_s7_the_pre_flight(self):
        """The pre-flight's own refusals (strict for a codes run, a header without an id in either mode) and nothing to
        build: as the one-shot."""
        self.unrecorded_run("--residue-codes")
        duplicate = _write(self.inputs / "dup.fasta", ">1acb_E\nACDE\n>1acb_E\nACDE\n")
        self.assertIn("repeat an earlier id", self.refused_seal(fasta=duplicate, cause=ValueError))
        self.clear()
        self.unrecorded_run()
        empty_id = _write(self.inputs / "empty-id.fasta", ">\nACDE\n>b\nKLMN\n")
        self.assertIn("is empty", self.refused_seal("--min-res-n", "0", fasta=empty_id, cause=ValueError))
        self.assertIn("there is nothing to build", self.refused_seal("--min-res-n", "1000", cause=ValueError))

    def test_s8_the_shards_are_incomplete(self):
        """Rows other than the kept entries (a lost shard, codes or plain; a plain run sealed with another --min-res-n)
        and rows that are not the deal."""
        message_end = "The shards are incomplete: run 'build sequences-embed' again"
        self.unrecorded_run("--residue-codes", ranks=2)
        self.shard(1).unlink()
        message = self.refused_seal()
        self.assertIn(f"The 1 shard(s) of run run in {self.shards.resolve()} hold 1 row(s), but "
                      f"{self.fasta.resolve()} has 2 entries with at least 0 residues.", message)
        self.assertTrue(message.endswith(f"{message_end}."), message)
        self.clear()
        self.unrecorded_run(ranks=2)
        self.shard(1).unlink()
        message = self.refused_seal("--min-res-n", "0")
        self.assertIn("hold 1 row(s), but", message)
        self.assertTrue(message.endswith(f"{message_end}, or give the --min-res-n the run used."), message)
        self.clear()
        self.unrecorded_run()
        self.assertIn("hold 2 row(s), but", self.refused_seal("--min-res-n", "100"))
        self.clear()
        five = _write(self.inputs / "five.fasta", _FIVE)
        self.unrecorded_run("--residue-codes", ranks=2, fasta=five)
        t0, t1 = pq.read_table(self.shard(0)), pq.read_table(self.shard(1))
        pq.write_table(t0.slice(0, 2), self.shard(0))  # rank 0: entries 0, 2; rank 1: 4, 1, 3
        pq.write_table(pa.concat_tables([t0.slice(2), t1.replace_schema_metadata(t0.schema.metadata)]), self.shard(1))
        message = self.refused_seal(fasta=five)
        self.assertIn("are not the deal of its 5 entries over the 2 rank(s) of the pass (rank 0 has 2 row(s), not 3; "
                      "rank 1 has 3 row(s), not 2): the shards cannot be indexed.", message)
        self.assertTrue(message.endswith(f"{message_end}."), message)

    def test_s9_ids_not_at_their_dealt_positions(self):
        for codes in (False, True):
            with self.subTest(codes=codes):
                self.clear()
                self.unrecorded_run(*_codes_flag(codes))
                _rewrite(self.shard(), lambda table: table.set_column(0, table.schema.field("id"), pa.array(
                    table.column("id").to_pylist()[::-1], table.schema.field("id").type)))
                message = self.refused_seal(*(() if codes else ("--min-res-n", "0")))
                self.assertIn("2 shard row(s) of run run do not hold the id of the entry", message)
                self.assertIn("emb_run-0.parquet row 0 holds '2uzi_A', entry 0 is '1acb_E'", message)

    def test_s9_sequences_that_are_not_the_shards(self):
        """A FASTA with the run's ids, order and lengths but another residue (the same size and pre-flight counts):
        a codes run's seq_sha1 refuses it; a plain run has no seq_sha1, so its seal binds by ids only (D31)."""
        for codes in (True, False):
            with self.subTest(codes=codes):
                self.clear()
                shutil.copyfile(_FASTA, self.fasta)
                self.unrecorded_run(*_codes_flag(codes))
                _one_residue_changed(self.fasta)
                if not codes:
                    self.sealed("--min-res-n", "0", fasta=self.fasta)
                    continue
                message = self.refused_seal()
                self.assertIn("1 shard row(s) of run run hold a seq_sha1 that is not the sha1 of the sequence of the "
                              f"entry of {self.fasta.resolve()} the pass dealt them, though they hold its id: "
                              "emb_run-0.parquet row 0 ('1acb_E', entry 0). These shards were not computed from this "
                              "FASTA's sequences.", message)

    def test_s10_a_write_during_the_hash(self):
        """The seal's bracket opens before the hash: a write while the FASTA is hashed (its mtime moved once the hash
        is done) is refused, so the record never pairs a sha256 with a stat taken after it."""
        self.unrecorded_run("--residue-codes")
        real = split_run.fasta_sha256

        def written_while_hashed(path):
            out = real(path)
            stat = os.stat(path)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
            return out

        with mock.patch.object(split_run, "fasta_sha256", side_effect=written_while_hashed):
            message = self.refused_seal()
        self.assertIn(f"The FASTA changed during the seal ({self.fasta.resolve()}: ", message)
        self.assertEqual(split_run.record_run_ids(self.shards), [])

    def test_s10_the_fasta_changed_during_the_seal(self):
        """The FASTA touched, or removed, after the id check: refused with both stats."""
        self.unrecorded_run("--residue-codes")
        check = split_run.check_dealt_ids
        stat = self.fasta.stat()

        def touching(*args, **kwargs):
            seconds = check(*args, **kwargs)
            os.utime(self.fasta, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
            return seconds

        with mock.patch.object(split_run, "check_dealt_ids", side_effect=touching):
            message = self.refused_seal()
        self.assertIn(f"The FASTA changed during the seal ({self.fasta.resolve()}: {stat.st_size} bytes, mtime_ns "
                      f"{stat.st_mtime_ns}, inode {stat.st_ino} before it was hashed, now {stat.st_size} bytes, "
                      f"mtime_ns {stat.st_mtime_ns + 10 ** 9}, inode {stat.st_ino})", message)

        def removing(*args, **kwargs):
            seconds = check(*args, **kwargs)
            self.fasta.rename(self.inputs / "gone.fasta")
            return seconds

        with mock.patch.object(split_run, "check_dealt_ids", side_effect=removing):
            self.assertIn("now cannot be stat'ed", self.refused_seal())

    def test_s11_the_record_cannot_be_written(self):
        """A full disk while the record is staged: the .tmp is removed. A record of the run written by another process
        during the seal: refused, and left as it is."""
        self.unrecorded_run("--residue-codes")
        with mock.patch.object(split_run, "_write_all", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            message = self.refused_seal()
        self.assertIn(f"The run record {self.shards.resolve() / 'emb_run.run.json'} cannot be written", message)
        self.assertIn("No space left on device", message)
        check = split_run.check_dealt_ids

        def racing(*args, **kwargs):
            seconds = check(*args, **kwargs)
            (self.shards / "emb_run.run.json").write_text("another process\n")
            return seconds

        with mock.patch.object(split_run, "check_dealt_ids", side_effect=racing):
            message = self.refused(self.seal(fasta=self.fasta))
        self.assertIn("File exists", message)
        self.assertEqual((self.shards / "emb_run.run.json").read_text(), "another process\n")
        self.assertEqual(sorted(p.name for p in self.shards.iterdir()), ["emb_run-0.parquet", "emb_run.run.json"])


# =====================================================================================================================
# P9.I1, P9.I2: the launch checks of step 2 and the seal, before torch, and the command line in their advice
# P9.L2: their later refusals, before torch too
# =====================================================================================================================
#: A sitecustomize for the commands that ``_HookedCommands`` runs: any import of torch, Lightning, ESM, FAISS,
#: transformers or huggingface_hub ends the process at once with status 97, naming the module and the innermost
#: FoldMatch function that imported it, so that no ``except`` can hide it; one of them already imported when the
#: interpreter starts (by a ``.pth`` file) ends it with status 98.
_NO_HEAVY_IMPORTS = r"""
import os, sys
_BLOCKED = ("torch", "lightning", "lightning_fabric", "pytorch_lightning", "esm", "faiss", "transformers",
            "huggingface_hub")
_loaded = sorted(name for name in sys.modules if name.partition(".")[0] in _BLOCKED)
if _loaded:
    sys.stderr.write(f"FORBIDDEN IMPORT at start-up: {', '.join(_loaded)}\n")
    sys.stderr.flush()
    os._exit(98)
class _NoHeavyImports:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in _BLOCKED:
            frame = sys._getframe(1)
            while frame is not None and not frame.f_globals.get("__name__", "").startswith("foldmatch"):
                frame = frame.f_back
            where = f" (from {frame.f_globals['__name__']}.{frame.f_code.co_name})" if frame is not None else ""
            sys.stderr.write(f"FORBIDDEN IMPORT: {name}{where}\n")
            sys.stderr.flush()
            os._exit(97)
        return None
sys.meta_path.insert(0, _NoHeavyImports())
"""
#: The console script that pip and uv install for a ``[project.scripts]`` entry point.
_CONSOLE_SCRIPT = """\
#!{python}
import sys
from {module} import {attr}
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit({attr}())
"""


def _fm_search_entry_point() -> str:
    """``fm-search``'s entry point, from ``pyproject.toml`` when the tests run in a source tree."""
    path = Path(_SRC).parent / "pyproject.toml"
    if not path.is_file():
        return "foldmatch.cli.search:main"
    import tomllib
    return tomllib.loads(path.read_text())["project"]["scripts"]["fm-search"]


class _HookedCommands:
    """``fm-search`` commands as subprocesses in ``self.tmp``, through an ``fm-search`` console script of the
    package's entry point or through ``python -m foldmatch.cli.search``, with ``_NO_HEAVY_IMPORTS`` unless ``hook``
    is false. ``hooked_setup`` needs ``self.tmp`` and the environment the commands get, set."""

    def hooked_setup(self):
        for name in ("site", "bin"):
            (self.tmp / name).mkdir()
        (self.tmp / "site" / "sitecustomize.py").write_text(_NO_HEAVY_IMPORTS)
        module, _, attr = _fm_search_entry_point().partition(":")
        self.fm_search = self.tmp / "bin" / "fm-search"
        self.fm_search.write_text(_CONSOLE_SCRIPT.format(python=sys.executable, module=module, attr=attr))
        self.env = {k: v for k, v in os.environ.items()
                    if k not in _LAUNCH_ENV and k not in _DDP_ENV and k not in _JOB_ENV
                    and not k.startswith(_DDP_ENV_PREFIXES)}
        self.env["HF_HUB_OFFLINE"] = "1"

    @staticmethod
    def program(entry) -> list:
        """The command line of ``entry`` (``"fm-search"`` or ``"python -m"``) before the arguments."""
        return [sys.executable, "-m", "foldmatch.cli.search"] if entry == "python -m" else ["fm-search"]

    def command_env(self, hook=True, **env) -> dict:
        """The environment of the commands, with ``env`` added, and with the hook unless ``hook`` is false."""
        path = ([str(self.tmp / "site")] if hook else []) + [_SRC] + \
            [p for p in [os.environ.get("PYTHONPATH")] if p]
        return {**self.env, "PYTHONPATH": os.pathsep.join(path), **env}

    def run_command(self, entry, args, hook=True, **env):
        """``args`` through ``entry`` (``"fm-search"`` or ``"python -m"``) in ``self.tmp``, with ``env`` added, and with
        the hook unless ``hook`` is false."""
        program = [sys.executable, str(self.fm_search)] if entry == "fm-search" else self.program(entry)
        return subprocess.run([*program, *map(str, args)], env=self.command_env(hook, **env), cwd=self.tmp,
                              capture_output=True, text=True, timeout=600)

    def in_this_folder(self, entry) -> list:
        """What the I3 advice's command starts with when run through ``entry`` in ``self.tmp`` with absolute paths: a
        cd to ``self.tmp``, except through the console script (P9.I2)."""
        return [] if entry == "fm-search" else ["cd", os.path.realpath(self.tmp), "&&"]

    def assert_refused_before_torch(self, result, start, logged=0, command=False):
        """Exit status 1 with ``logged`` log lines on stderr, then ``Error: `` and the refusal, which starts with
        ``start``: one line, or with ``command`` that line and the command line the launch checks' advice ends with, on
        a line of its own indented by four spaces (P9.M). Returns the refusal's lines, the command line without its
        indent."""
        self.assertNotIn("FORBIDDEN IMPORT", result.stderr)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        lines = result.stderr.split("\n")
        self.assertEqual(len(lines), logged + 2 + command, result.stderr)
        self.assertTrue(lines[logged].startswith(f"Error: {start}"), result.stderr)
        self.assertEqual(lines[-1], "", result.stderr)
        if not command:
            return [lines[logged]]
        self.assertRegex(lines[logged + 1], "^    [^ ]", result.stderr)
        return [lines[logged], lines[logged + 1][4:]]


class LaunchChecksBeforeTorchTest(_HookedCommands, unittest.TestCase):
    """``build sequences-index`` and ``build sequences-seal`` as commands, through an ``fm-search`` console script of
    the package's entry point and through ``python -m foldmatch.cli.search``, with ``_NO_HEAVY_IMPORTS``: I2 and I3
    refuse with exit status 1 and their whole message on stderr, before torch or Lightning is imported (P8b, N3 case
    E: I3 came 118 s after the launch, after ``set_log_level``'s Lightning import). Without a launch refusal they
    refuse the empty shards folder, still before torch (P9.L2). The WARNING of a one-task srun step of a multi-node
    job, without the hook, comes in the logging format ``set_log_level`` sets up."""

    I3_ENV = {"SLURM_JOB_ID": "4711", "SLURM_JOB_NUM_NODES": "2", "SLURM_NTASKS": "8"}
    SRUN_STEP_ENV = {"SLURM_JOB_ID": "42", "SLURM_JOB_NUM_NODES": "2", "SLURM_STEP_ID": "3",
                     "SLURM_STEP_NUM_TASKS": "1", "SLURM_NTASKS": "8"}

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.tmp / "shards").mkdir()
        self.hooked_setup()
        (self.tmp / "in put").mkdir()
        self.fasta = Path(shutil.copyfile(_FASTA, self.tmp / "in put" / "it's seqs.fasta"))
        self.db = self.tmp / "out" / "db"

    def commands(self) -> dict:
        return {"build sequences-index": _index_args(self.fasta, self.db, self.tmp / "shards"),
                "build sequences-seal": _seal_args(self.fasta, self.tmp / "shards", "run")}

    def assert_nothing_written(self):
        self.assertFalse(self.db.parent.exists())
        self.assertEqual(list((self.tmp / "shards").iterdir()), [])

    def test_i3_before_torch(self):
        for entry in ("fm-search", "python -m"):
            for command, args in self.commands().items():
                with self.subTest(entry=entry, command=command):
                    result = self.run_command(entry, args, **self.I3_ENV)
                    error, line = self.assert_refused_before_torch(
                        result, f"SLURM job 4711 holds 2 nodes, and '{command}' runs in one process on one of them: "
                                f"the others would stay allocated until it ends. Run it as its own one-node CPU job "
                                f"instead: ", command=True)
                    self.assertIn("--dependency=afterok:$SLURM_JOB_ID --kill-on-invalid-dep=yes", error)
                    self.assertNotIn("afterok:4711", result.stderr)
                    # P9.I2: the command to run in the one-node job, last, as it was invoked, shell-quoted; P9.M: on
                    # a line of its own.
                    self.assertTrue(error.endswith(f" SCRIPT runs this command, without srun, next time "
                                                   f"{_NEXT_BUILD}:"), error)
                    self.assertEqual(shlex.split(line),
                                     [*self.in_this_folder(entry), *self.program(entry), *map(str, args)])
        self.assert_nothing_written()

    def test_the_command_runs_from_another_folder(self):
        """P9.I2: with relative paths, the I3 refusal's command starts with a cd to the folder the command ran in, so
        that it runs from any folder (sbatch starts the one-node job in the folder it is submitted from). From another
        folder it passes the paths' checks and refuses the empty shards folder (P9.L2: still before torch); without
        the cd it is refused there with exit status 2. The I2 refusal's command, for the same batch script, has no
        cd."""
        self.fm_search.chmod(0o755)
        elsewhere = self.tmp / "jobs"
        elsewhere.mkdir()
        env = self.command_env(PATH=os.pathsep.join([str(self.tmp / "bin"), os.environ.get("PATH", "")]))
        relative = {"build sequences-index": _index_args("in put/it's seqs.fasta", "out/db", "shards"),
                    "build sequences-seal": _seal_args("in put/it's seqs.fasta", "shards", "run")}
        for command, args in relative.items():
            with self.subTest(command=command):
                result = self.run_command("fm-search", args, **self.I3_ENV)
                _, line = self.assert_refused_before_torch(
                    result, f"SLURM job 4711 holds 2 nodes, and '{command}' runs ", command=True)
                self.assertEqual(shlex.split(line), ["cd", os.path.realpath(self.tmp), "&&", "fm-search", *args])
                rerun = subprocess.run(["/bin/sh", "-c", line], cwd=elsewhere, env=env, capture_output=True, text=True,
                                       timeout=600)
                self.assert_refused_before_torch(rerun, "No run in " if "index" in command else
                                                 "No embedding shards of run run ")
                without_cd = subprocess.run(["/bin/sh", "-c", line.split(" && ", 1)[1]], cwd=elsewhere, env=env,
                                            capture_output=True, text=True, timeout=600)
                self.assertEqual(without_cd.returncode, 2, without_cd.stderr)
                self.assertIn("Invalid value for '--fasta-file'", without_cd.stderr)
                i2 = self.run_command("fm-search", args, WORLD_SIZE="2")
                _, line = self.assert_refused_before_torch(i2, f"'{command}' runs as one process (WORLD_SIZE=2): ",
                                                           command=True)
                self.assertEqual(shlex.split(line), ["fm-search", *args])
        self.assert_nothing_written()
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_i2_before_torch(self):
        for command, args in self.commands().items():
            for env, launch in (({"WORLD_SIZE": "2"}, "WORLD_SIZE=2"),
                                ({"SLURM_JOB_ID": "7", "SLURM_STEP_ID": "0", "SLURM_STEP_NUM_TASKS": "4",
                                  "SLURM_JOB_NUM_NODES": "2"}, "SLURM_STEP_NUM_TASKS=4")):
                with self.subTest(command=command, env=env):
                    result = self.run_command("fm-search", args, **env)
                    error, line = self.assert_refused_before_torch(result, "", command=True)
                    self.assertEqual(error, f"Error: '{command}' runs as one process ({launch}): launch it once, "
                                            f"directly in the batch script or with srun -n 1. The command:")
                    self.assertEqual(shlex.split(line), ["fm-search", *map(str, args)])
        self.assert_nothing_written()

    def test_without_a_launch_refusal(self):
        """Without a launch refusal, each command, through each entry, goes on and refuses the empty shards folder,
        still before torch or Lightning is imported (P9.L2; before it, set_log_level's Lightning import came first:
        ``LaterRefusalsBeforeTorchTest`` has the hook's control)."""
        starts = {"build sequences-index": f"No run in {os.path.realpath(self.tmp / 'shards')}: no run record",
                  "build sequences-seal": "No embedding shards of run run (emb_run-<rank>.parquet) in "}
        for entry in ("fm-search", "python -m"):
            for command, args in self.commands().items():
                with self.subTest(entry=entry, command=command):
                    self.assert_refused_before_torch(self.run_command(entry, args), starts[command])
        self.assert_nothing_written()

    def test_the_srun_step_warning_in_the_logging_format(self):
        """Without the hook: the WARNING of a one-task srun step of a multi-node job, in the format of --log-level info,
        warn (P9.L1: it had no format) and debug; then the empty shards folder is refused."""
        detailed = r"> \d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} - \[proc-0\] - WARNING - search\.py:\d+ - "
        formats = {"info": r"\[proc-0\] ", "warn": detailed, "debug": detailed}
        for command, args in self.commands().items():
            for level, prefix in formats.items():
                with self.subTest(command=command, level=level):
                    result = self.run_command("fm-search", [*args, "--log-level", level], hook=False,
                                              **self.SRUN_STEP_ENV)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    stderr = result.stderr.splitlines()
                    lines = [k for k, line in enumerate(stderr) if "one-task srun step" in line]
                    self.assertEqual(len(lines), 1, result.stderr)
                    warning = stderr[lines[0]]
                    self.assertRegex(warning, "^" + prefix + re.escape(
                        f"'{command}' runs in a one-task srun step of SLURM job 42, which holds 2 nodes: the others "
                        f"stay allocated until it ends. Next time, run it as its own one-node CPU job: "))
                    self.assertTrue(warning.endswith(f" SCRIPT runs this command, without srun, {_NEXT_BUILD}:"),
                                    warning)
                    # P9.M: the command on the next line, indented, without the log line's prefix.
                    self.assertEqual(stderr[lines[0] + 1],
                                     "    " + shlex.join(["fm-search", *map(str, args), "--log-level", level]))
                    self.assertEqual(len([line for line in result.stderr.splitlines() if line.startswith("Error: ")]),
                                     1, result.stderr)



class LaterRefusalsBeforeTorchTest(_HookedCommands, _Sealing):
    """P9.L2: the refusals of step 2 up to its index phase, and every refusal of the seal, come before torch or
    Lightning is imported, through the ``fm-search`` console script with ``_NO_HEAVY_IMPORTS`` (before, they came
    after ``set_log_level``'s Lightning import: minutes from a cold file system on a compute node). The control: with
    nothing to refuse, step 2 reaches its deferred Lightning import after its last check before the index phase; the
    seal seals without any of them."""

    def setUp(self):
        super().setUp()
        self.hooked_setup()
        self.inputs = self.tmp / "in"
        self.inputs.mkdir()
        self.fasta = Path(shutil.copyfile(_FASTA, self.inputs / "seqs.fasta"))

    def refused_hooked(self, args, text):
        """``args`` refused with exit status 1 before torch, Lightning, ESM, FAISS, transformers or huggingface_hub
        was imported: log lines, then the ``Error:`` line, which holds ``text``. Returns the log lines."""
        result = self.run_command("fm-search", args)
        self.assertNotIn("FORBIDDEN IMPORT", result.stderr)
        self.assertEqual((result.returncode, result.stdout), (1, ""), result.stderr)
        *logged, error, end = result.stderr.split("\n")
        self.assertEqual(end, "", result.stderr)
        self.assertTrue(error.startswith("Error: "), result.stderr)
        self.assertIn(text, error)
        for line in logged:
            self.assertTrue(line.startswith("[proc-0] "), result.stderr)
        return logged

    def test_step_2_refusals(self):
        """I4 (no run record: an empty folder, a run whose record is gone), I6, I8, I11, I12 (inside the build lock),
        I16 and I17 (a FASTA that is not the one the run embedded: another size, one residue changed)."""
        self.embed(fasta=self.fasta)
        unrecorded = self.tmp / "unrecorded"
        unrecorded.mkdir()
        self.embed(fasta=self.fasta, shards=unrecorded, db=self.tmp / "other").unlink()
        empty = self.tmp / "empty"
        empty.mkdir()
        larger = _write(self.inputs / "larger.fasta", self.fasta.read_text() + ">late\nACDE\n")
        changed = _one_residue_changed(shutil.copyfile(self.fasta, self.inputs / "changed.fasta"))
        lock = Path(f"{self.db}.build.lock")
        cases = [
            ("I4", _index_args(self.fasta, self.db, empty), "No run in ", None),
            ("I4", _index_args(self.fasta, self.db, unrecorded), "has no run record", None),
            ("I6", _index_args(self.fasta, self.db, self.shards, "--run-id", "nope"), "Run nope has no run record",
             None),
            ("I8", _index_args(self.fasta, self.db, self.shards, "--residue-codes"), "computed no residue codes", None),
            ("I11", _index_args(self.fasta, self.db, self.shards), f"{lock} exists", lock),
            ("I12", _index_args(self.fasta, self.db, self.shards), "Database already exists at",
             Path(f"{self.db}.index")),
            ("I16", _index_args(larger, self.db, self.shards), "bytes, but the FASTA run run embedded", None),
            ("I17", _index_args(changed, self.db, self.shards), "is not the FASTA run run embedded", None),
        ]
        for case, args, text, present in cases:
            with self.subTest(case=case, text=text):
                if present is not None:
                    present.write_text("{}\n")
                try:
                    self.refused_hooked(args, text)
                finally:
                    if present is not None:
                        present.unlink()
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), [])

    def test_seal_refusals(self):
        """S3 (a run with a record), S4 (no shards of the run), S7 (nothing to build) and S8 (a FASTA with more
        entries than the shards hold rows)."""
        self.embed(fasta=self.fasta)
        unrecorded = self.tmp / "unrecorded"
        unrecorded.mkdir()
        self.embed(fasta=self.fasta, shards=unrecorded, db=self.tmp / "other").unlink()
        larger = _write(self.inputs / "larger.fasta", self.fasta.read_text() + ">late\nACDEFGHIKL\n")
        cases = [
            ("S3", _seal_args(self.fasta, self.shards, "run"), "has its run record already"),
            ("S4", _seal_args(self.fasta, unrecorded, "nope"), "No embedding shards of run nope"),
            ("S7", _seal_args(self.fasta, unrecorded, "run", "--min-res-n", "1000"), "there is nothing to build"),
            ("S8", _seal_args(larger, unrecorded, "run", "--min-res-n", "0"), "hold 2 row(s), but"),
        ]
        for case, args, text in cases:
            with self.subTest(case=case):
                self.refused_hooked(args, text)
        self.assertEqual(sorted(p.name for p in unrecorded.iterdir()), ["emb_run-0.parquet"])

    def test_the_seal_imports_none_of_them(self):
        """A plain run and a residue-code run sealed with the hook: exit status 0, the record written."""
        for codes in (False, True):
            with self.subTest(codes=codes):
                shards = self.tmp / f"unrecorded-{codes}"
                shards.mkdir()
                self.embed(*_codes_flag(codes), fasta=self.fasta, shards=shards, db=self.tmp / f"db-{codes}").unlink()
                args = _seal_args(self.fasta, shards, "run", *(() if codes else ("--min-res-n", "0")))
                result = self.run_command("fm-search", args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("FORBIDDEN IMPORT", result.stderr)
                self.assertIn("[proc-0] Run run sealed: 2 entries in 1 shard(s) of 1 rank(s)", result.stderr)
                self.assertTrue(split_run.read_run_record(shards / "emb_run.run.json")["sealed"])

    def test_the_hook_stops_step_2_at_its_deferred_lightning_import(self):
        """The control: with nothing to refuse, step 2 logs its last line before the index phase, then imports
        Lightning in set_log_level's second call, which the hook stops; without the hook it builds the database."""
        self.embed(fasta=self.fasta)
        args = _index_args(self.fasta, self.db, self.shards)
        result = self.run_command("fm-search", args)
        self.assertEqual(result.returncode, 97, result.stderr)
        *_, building, forbidden, end = result.stderr.split("\n")
        self.assertTrue(building.startswith(f"[proc-0] Building {self.db} from run run in "), result.stderr)
        self.assertEqual((forbidden, end),
                         ("FORBIDDEN IMPORT: lightning (from foldmatch.cli.args_utils._import_lightning)", ""))
        Path(f"{self.db}.build.lock").unlink()  # the hook ended the process inside the lock
        result = self.run_command("fm-search", args, hook=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"[proc-0] Database {self.db} built from run run in ", result.stderr)

    def test_set_log_level_and_the_build_time(self):
        """In process: step 2 sets up logging without Lightning, then again before EmbeddingComputer, which imports
        it; the build's time leaves that import out, as before. The seal sets up logging once, without Lightning."""
        self.embed(fasta=self.fasta)
        calls = []

        def slow_import(level, defer_lightning=False):
            calls.append((level, defer_lightning))
            if not defer_lightning:
                time.sleep(1.5)

        with mock.patch.object(search, "set_log_level", side_effect=slow_import), \
                self.assertLogs(level="INFO") as logs:
            self.built(fasta=self.fasta)
        self.assertEqual(calls, [(LogLevel.info, True), (LogLevel.info, False)])
        [built] = [r.getMessage() for r in logs.records if " built from run run in " in r.getMessage()]
        self.assertLess(float(built.rsplit(" in ", 1)[1].removesuffix(" s")), 1.5, built)
        calls.clear()
        unrecorded = self.tmp / "unrecorded"
        unrecorded.mkdir()
        self.embed(fasta=self.fasta, shards=unrecorded, db=self.tmp / "other").unlink()
        with mock.patch.object(search, "set_log_level", side_effect=slow_import):
            self.sealed("--min-res-n", "0", fasta=self.fasta, shards=unrecorded)
        self.assertEqual(calls, [(LogLevel.info, True)])


class CommandLineTest(unittest.TestCase):
    """P9.I2: ``_command_line``, the command line that the launch checks' advice ends with, from ``sys.argv`` and
    ``sys.orig_argv``."""

    ARGS = ["build", "sequences-index", "--fasta-file", "/data/my seqs.fasta", "--output-db", "it's/db",
            "--from-shards=/scratch/$USER/shards", "--run-id", "60158326_2_0_ab12cd34", "--index-type", "ivf_pq"]

    def command_line(self, argv, orig_argv=(), command="build sequences-index", **options):
        with mock.patch.object(sys, "argv", list(argv)), mock.patch.object(sys, "orig_argv", list(orig_argv)):
            return search._command_line(command, **options)

    def test_the_console_script(self):
        """``fm-search``, whatever path the console script ran from; every argument as given, shell-quoted."""
        for argv0 in ("/env/bin/fm-search", "fm-search", "./fm-search"):
            with self.subTest(argv0=argv0):
                line = self.command_line([argv0, *self.ARGS], ["/env/bin/python3.12", argv0, *self.ARGS])
                self.assertEqual(line, "fm-search build sequences-index --fasta-file '/data/my seqs.fasta' "
                                       "--output-db 'it'\"'\"'s/db' '--from-shards=/scratch/$USER/shards' --run-id "
                                       "60158326_2_0_ab12cd34 --index-type ivf_pq")
                self.assertEqual(shlex.split(line), ["fm-search", *self.ARGS])
        seal = ["build", "sequences-seal", "--fasta-file", "f.fasta", "--from-shards", "s", "--run-id", "r"]
        self.assertEqual(self.command_line(["/env/bin/fm-search", *seal], command="build sequences-seal"),
                         "fm-search " + shlex.join(seal))

    def test_python_m(self):
        """``python -m foldmatch.cli.search``, with the interpreter and its options as they were typed."""
        program = ["python3", "-X", "importtime", "-m", "foldmatch.cli.search"]
        line = self.command_line([search.__file__, *self.ARGS], [*program, *self.ARGS])
        self.assertTrue(line.startswith("python3 -X importtime -m foldmatch.cli.search build sequences-index "), line)
        self.assertEqual(shlex.split(line), [*program, *self.ARGS])

    def test_other_invocations(self):
        """What ``sys.orig_argv`` holds before the arguments."""
        cases = {
            "the module as a script": ("src/foldmatch/cli/search.py",
                                       ["/usr/bin/python3", "src/foldmatch/cli/search.py"]),
            "python -c": ("-c", ["python", "-c", "from foldmatch.cli.search import main; main()"]),
            "a wrapper script": ("/home/me/run fm.py", ["python", "/home/me/run fm.py"]),
        }
        for case, (argv0, program) in cases.items():
            with self.subTest(case=case):
                line = self.command_line([argv0, *self.ARGS], [*program, *self.ARGS])
                self.assertEqual(shlex.split(line), [*program, *self.ARGS])

    def test_fallbacks(self):
        """``fm-search`` when ``sys.orig_argv`` does not end with the arguments (``sys.argv`` rewritten), and
        ``fm-search <command> ...`` when ``sys.argv`` does not run the command (the app called in-process)."""
        line = self.command_line(["/x/search.py", *self.ARGS], ["python", "-m", "foldmatch.cli.search", "build"])
        self.assertEqual(shlex.split(line), ["fm-search", *self.ARGS])
        for argv in (["/venv/bin/pytest", "-q", "tests"], ["/env/bin/fm-search", "build", "sequences-seal"],
                     ["/env/bin/fm-search"], []):
            with self.subTest(argv=argv):
                self.assertEqual(self.command_line(argv, argv), "fm-search build sequences-index ...")
        with mock.patch.object(sys, "argv", ["/venv/bin/pytest"]):
            del sys.argv
            self.assertEqual(search._command_line("build sequences-seal"), "fm-search build sequences-seal ...")

    def test_in_another_job(self):
        """For another job, which sbatch starts in the folder it is submitted from: a cd to this process's folder first
        when the line depends on it, a relative --fasta-file, --output-db or --from-shards (as ``--option PATH`` or
        ``--option=PATH``) or a program other than fm-search; not when that folder is gone."""
        absolute = ["build", "sequences-index", "--fasta-file", "/data/s.fasta", "--output-db", "/db/x",
                    "--from-shards=/scratch/shards", "--run-id", "r1", "--index-type", "ivf_pq"]
        fm_search = ["/env/bin/fm-search", *absolute]
        with mock.patch.object(os, "getcwd", return_value="/scratch/my run"):
            for options in ({}, {"in_another_job": True}):
                self.assertEqual(self.command_line(fm_search, **options), "fm-search " + shlex.join(absolute))
            for k, value in ((3, "s.fasta"), (3, "../s.fasta"), (5, "db/x"), (5, "x"), (5, ""), (6, "--from-shards=s")):
                argv = list(fm_search)
                argv[k + 1] = value
                with self.subTest(argv=argv):
                    line = self.command_line(argv, in_another_job=True)
                    self.assertEqual(shlex.split(line), ["cd", "/scratch/my run", "&&", "fm-search", *argv[1:]])
                    self.assertTrue(line.startswith("cd '/scratch/my run' && fm-search build "), line)
                    self.assertEqual(self.command_line(argv), shlex.join(["fm-search", *argv[1:]]))
            equals = ["/env/bin/fm-search", *absolute[:2], "--fasta-file=s.fasta", *absolute[4:]]
            self.assertEqual(shlex.split(self.command_line(equals, in_another_job=True))[:3],
                             ["cd", "/scratch/my run", "&&"])
            program = ["/usr/bin/python3", "-m", "foldmatch.cli.search"]
            self.assertEqual(shlex.split(self.command_line([search.__file__, *absolute], [*program, *absolute],
                                                           in_another_job=True)),
                             ["cd", "/scratch/my run", "&&", *program, *absolute])
            self.assertEqual(self.command_line(["/venv/bin/pytest"], in_another_job=True),
                             "fm-search build sequences-index ...")
        relative = ["/env/bin/fm-search", *absolute[:3], "s.fasta", *absolute[4:]]
        with mock.patch.object(os, "getcwd", side_effect=FileNotFoundError(errno.ENOENT, "No such file or directory")):
            self.assertEqual(self.command_line(relative, in_another_job=True), shlex.join(["fm-search", *relative[1:]]))

    def test_a_newline(self):
        """An argument holding a newline keeps it, inside its quotes: the line runs as given, but on two lines."""
        args = ["build", "sequences-index", "--output-db", "/db/a\nb"]
        line = self.command_line(["/env/bin/fm-search", *args])
        self.assertEqual(line.count("\n"), 1, line)
        self.assertEqual(shlex.split(line), ["fm-search", *args])

# =====================================================================================================================
# real ESM3 (CPU)
# =====================================================================================================================
class RealEsm3SplitTest(unittest.TestCase):
    """The one-shot ``build sequences`` and the two steps on ``test_sequences.fasta`` with the real CPU pass at
    --batch-size 2, plain and codes, with pinned run ids, one build after another."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        esm_codes_bundle.registry_files()
        gc.collect()  # earlier tests' ESM3 models, if only reference cycles still hold them
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.results = {}
        with mock.patch.dict(os.environ):
            for name in _JOB_ENV:
                os.environ.pop(name, None)
            for kind, extra in (("codes", ("--residue-codes",)), ("plain", ())):
                for step in ("one", "split"):
                    (cls.tmp / f"{step}-{kind}-shards").mkdir()
                with launch_env(FOLDMATCH_RUN_ID=f"one-{kind}"):
                    cls.results[f"one-{kind}"] = invoke(*_build_args(
                        _FASTA, cls.tmp / f"one-{kind}" / "db", cls.tmp / f"one-{kind}-shards", "--batch-size", "2",
                        *extra))
                gc.collect()  # the pass's ESM3 (5.6 GB) is only reachable through reference cycles
                with launch_env(FOLDMATCH_RUN_ID=f"split-{kind}"):
                    cls.results[f"embed-{kind}"] = invoke(*_embed_args(
                        _FASTA, cls.tmp / f"split-{kind}" / "db", cls.tmp / f"split-{kind}-shards", "--batch-size",
                        "2", *extra))
                gc.collect()
                with launch_env(), _esm3_spy(), \
                        mock.patch.object(ss, "prepare", side_effect=AssertionError("bundle fetched")):
                    cls.results[f"index-{kind}"] = invoke(*_index_args(
                        _FASTA, cls.tmp / f"split-{kind}" / "db", cls.tmp / f"split-{kind}-shards"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()

    def setUp(self):
        for name, result in self.results.items():
            self.assertEqual(result.exit_code, 0, f"{name}: {result.output[-4000:]} {result.exception!r}")

    def test_two_steps_equal_the_one_shot(self):
        _, ids, _, codes_ref = _f2c()
        for kind in ("codes", "plain"):
            with self.subTest(kind=kind):
                columns = ["id", "embedding", *(("residue_codes", "seq_sha1") if kind == "codes" else ())]
                tables = [pq.read_table(self.tmp / f"{step}-{kind}-shards" / f"emb_{step}-{kind}-0.parquet",
                                        columns=columns) for step in ("one", "split")]
                self.assertTrue(tables[0].equals(tables[1]),
                                "precondition: the one-shot's and step 1's passes wrote different shard tables")
                assert_same_database(self.tmp / f"one-{kind}" / "db", self.tmp / f"split-{kind}" / "db")
                if kind == "codes":
                    store = ResidueCodeStore(self.tmp / f"split-{kind}", "db")
                    manifest = store.check_compatible()
                    self.assertEqual((manifest["entries"], manifest["total_residues"], manifest["build"]["split"]),
                                     (2, 303, True))
                    equal, n = _bits_equal(np.asarray(store.open_memmap()), codes_ref)
                    print(f"\nsplit build F2c CPU code bits: {equal}/{n} = {equal / n:.6f}", file=sys.stderr)
                    self.assertGreaterEqual(equal / n, A4_XCPU_BITS, f"F2c code bits {equal}/{n}")
                record = split_run.read_run_record(self.tmp / f"split-{kind}-shards" / f"emb_split-{kind}.run.json")
                self.assertEqual((record["rows"], record["preflight"]["residues"], record["world_size"],
                                  record["pass"]["accelerator"]), (2, 303, 1, "cpu" if kind == "codes" else None))


# =====================================================================================================================
# two real ranks as commands (slow)
# =====================================================================================================================
@unittest.skipUnless(os.environ.get("FOLDMATCH_SLOW_TESTS") == "1",
                     "slow: two ESM3 ranks on CPU, twice; set FOLDMATCH_SLOW_TESTS=1 to run")
class TwoRankSplitSmoke(unittest.TestCase):
    """``build sequences-embed --residue-codes --devices 2`` as a command (CPU, gloo: Lightning re-runs the command for
    rank 1, and the real ``all_gather_object`` runs), then ``build sequences-index``, and a two-rank one-shot ``build
    sequences --residue-codes`` to compare with (pattern of ``TwoRankCliSmoke``)."""

    def test_two_rank_split_build(self):
        esm_codes_bundle.registry_files()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("shards", "oneshot-shards", "site", "log-embed", "log-oneshot", "empty-hf-cache"):
                (tmp / name).mkdir()
            (tmp / "site" / "sitecustomize.py").write_text(_SITECUSTOMIZE)
            env = {k: v for k, v in os.environ.items()
                   if k not in _LAUNCH_ENV and k not in _DDP_ENV and k not in _JOB_ENV
                   and not k.startswith(_DDP_ENV_PREFIXES)}
            env["PYTHONPATH"] = os.pathsep.join([str(tmp / "site"), _SRC] +
                                                [p for p in [os.environ.get("PYTHONPATH")] if p])
            if sys.platform == "darwin":
                env["OMP_NUM_THREADS"] = "1"

            def run(args, **extra_env):
                command = [sys.executable, "-m", "foldmatch.cli.search", *map(str, args)]
                if "--devices" in command:
                    command[command.index("--devices") + 1] = "2"
                return subprocess.run(command, env={**env, **extra_env}, capture_output=True, text=True,
                                      timeout=1800)

            split_db, oneshot_db = tmp / "split" / "db", tmp / "oneshot" / "db"
            embed = run(_embed_args(_FASTA, split_db, tmp / "shards", "--residue-codes"),
                        **{_TWO_RANK_CHILD_ENV: str(tmp / "log-embed")})
            self.assertEqual(embed.returncode, 0, embed.stderr[-4000:])
            records = sorted((tmp / "shards").glob("*.run.json"))
            self.assertEqual(len(records), 1, sorted(p.name for p in (tmp / "shards").iterdir()))
            record = split_run.read_run_record(records[0])
            calls = {p.name: json.loads(p.read_text()) for p in (tmp / "log-embed").iterdir()}

            refused = run(_index_args(_FASTA, split_db, tmp / "shards"), WORLD_SIZE="2")
            self.assertEqual(refused.returncode, 1, refused.stderr[-4000:])
            self.assertIn("runs as one process (WORLD_SIZE=2)", refused.stderr)
            self.assertFalse(split_db.parent.exists())

            index = run(_index_args(_FASTA, split_db, tmp / "shards"), HF_HUB_OFFLINE="1",
                        HF_HUB_CACHE=str(tmp / "empty-hf-cache"))
            self.assertEqual(index.returncode, 0, index.stderr[-4000:])
            oneshot = run(_build_args(_FASTA, oneshot_db, tmp / "oneshot-shards", "--residue-codes"),
                          **{_TWO_RANK_CHILD_ENV: str(tmp / "log-oneshot")})
            self.assertEqual(oneshot.returncode, 0, oneshot.stderr[-4000:])
            tables = [[pq.read_table(path).select(["id", "embedding", "residue_codes", "seq_sha1"])
                       for path in sorted(folder.glob("*.parquet"))] for folder in (tmp / "oneshot-shards",
                                                                                     tmp / "shards")]
            same_shards = len(tables[0]) == len(tables[1]) and all(a.equals(b) for a, b in zip(*tables))
            self.assertTrue(same_shards, "precondition: the two passes wrote different shard tables")
            assert_same_database(oneshot_db, split_db)
            manifest = ResidueCodeStore(split_db.parent, "db").check_compatible()
        self.assertEqual(calls, {"prepare-0.json": {"local_files_only": False},
                                 "prepare-1.json": {"local_files_only": True}})
        self.assertEqual((record["world_size"], len(record["shards"]), record["rows"]), (2, 2, 2))
        self.assertEqual((manifest["entries"], manifest["build"]["devices"], manifest["build"]["split"]), (2, 2, True))


if __name__ == "__main__":
    unittest.main()
