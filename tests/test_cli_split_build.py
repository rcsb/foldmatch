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
under the ESM3 spy with ``scoring_system.prepare`` refusing. ``RealEsm3SplitTest`` runs both builds with the real CPU
pass, and ``TwoRankSplitSmoke`` (slow) both builds on two real ranks as commands.
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
from foldmatch.types.api_types import Accelerator, IndexConfig, IndexType, Strategy
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
        """One line when FAISS's BLAS runs on one thread by build, a WARNING (INFO on macOS) naming the library: the
        faiss-cpu 1.13.2 wheel's OpenBLAS in faiss_cpu.libs (threading layer disabled), FAISS's own BLAS whose
        threading layer threadpoolctl does not report, any BLAS built without threads; the first one only. None for a
        threaded BLAS (MKL at 128 threads, numpy's OpenBLAS limited to one thread, a later wheel's OpenMP OpenBLAS at
        one thread or more), and none when threadpoolctl finds no BLAS."""
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
        later = os.path.join(libs, "libopenblaso-r0-d77a1985.3.15.so")
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
            "a later wheel's OpenMP OpenBLAS at one thread": ([blas(later, "openmp", 1)], None),
            "a later wheel's OpenMP OpenBLAS": ([gomp, blas(later, "openmp", 64)], None),
            "no BLAS": ([gomp], None),
            "nothing found": ([], None),
        }
        text = ("FAISS's BLAS, {}, is single-threaded, so the ivf_pq training and adds run on one core whatever "
                "FAISS's thread count: a FAISS built on a multi-threaded BLAS (for example the MKL build of the "
                "official 'pytorch' conda channel) builds a 30M-entry ivf_pq index about 10x faster.")
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
        """Step 2 under the ESM3 spy, the bundle refused, with ``env`` set; returns the result."""
        with _esm3_spy() as esm3, mock.patch.object(ss, "prepare", side_effect=AssertionError("bundle fetched")), \
                mock.patch.dict(os.environ, {name: str(value) for name, value in (env or {}).items()}):
            result = invoke(*_index_args(fasta, db or self.db, shards or self.shards, *extra))
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
                              f"binds the shards to the FASTA by their entry count, rows and ids (and seq_sha1 when "
                              f"the residue-code store is validated), not by the pass that wrote them.", warnings)
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
                      "shards to the FASTA by their entry count, rows and ids (and seq_sha1 when the residue-code "
                      "store is validated), not by the pass that wrote them, whose FoldMatch version is not recorded "
                      "(shards without residue codes do not record it).", warnings)
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
                             "-N 1 --dependency=afterok:$SLURM_JOB_ID --kill-on-invalid-dep=yes ..."),
            "GPUs": ({"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 1, "SLURM_GPUS_ON_NODE": 4},
                     "SLURM job 42 has GPUs (SLURM_GPUS_ON_NODE=4), which 'build sequences-index' does not use: run it "
                     "as its own job on a CPU node (sbatch -C cpu -N 1 ...); a project with GPU hours only may run it "
                     "on one GPU node, where ivf_pq fits (flat and hnsw may not)."),
        }
        for k, (case, (env, text)) in enumerate(cases.items()):
            with self.subTest(case=case):
                messages = self.logged(lambda: self.built(env=env, db=self.tmp / f"w{k}" / "db"))
                warnings = [m for level, m in messages if level == "WARNING"]
                self.assertIn(text, warnings)
                self.assertFalse([m for m in warnings if "afterok:42" in m], warnings)  # the build runs in job 42
        messages = self.logged(lambda: self.built(env={"SLURM_JOB_ID": 42, "SLURM_JOB_NUM_NODES": 1},
                                                  db=self.tmp / "w" / "db"))
        self.assertFalse([m for level, m in messages if level == "WARNING" and "SLURM job" in m])

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
    #: threadpoolctl's runtimes as a Linux node might list them: two BLAS libraries (one of them twice) and an OpenMP.
    POOLS = [{"user_api": "blas", "internal_api": "openblas", "num_threads": 6,
              "filepath": "/site-packages/numpy.libs/libscipy_openblas64_-ff651d7f.so"},
             {"user_api": "blas", "internal_api": "openblas", "num_threads": 6,
              "filepath": "/site-packages/scipy.libs/libscipy_openblas-68440149.so"},
             {"user_api": "blas", "internal_api": "openblas", "num_threads": 6,
              "filepath": "/elsewhere/libscipy_openblas-68440149.so"},
             {"user_api": "openmp", "internal_api": "openmp", "num_threads": 6,
              "filepath": "/site-packages/faiss/../faiss_cpu.libs/libgomp-a34b3233.so.1.0.0"}]

    def threads(self, platform="linux", pools=POOLS, **env):
        """``_whole_node_threads`` with ``env`` (the thread variables otherwise unset) on ``platform``, threadpoolctl
        listing ``pools``: ``(FAISS thread calls inside the block, after it, threadpool_limits calls, (limit left
        inside, after), log)``."""
        clean = {k: v for k, v in os.environ.items() if k not in search._BLAS_THREAD_VARS}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(sys, "platform", platform), \
                mock.patch.object(code_alignment, "_physical_cpus", return_value=6), \
                mock.patch("faiss.omp_get_max_threads", return_value=3), \
                mock.patch("faiss.omp_set_num_threads") as set_threads, \
                mock.patch("threadpoolctl.threadpool_info", return_value=pools), \
                mock.patch("threadpoolctl.threadpool_limits") as limits, self.assertLogs(level="INFO") as logs:
            with search._whole_node_threads():
                inside = list(set_threads.call_args_list)
                exited = limits.return_value.__exit__.called
            after = list(set_threads.call_args_list)
        return inside, after, list(limits.call_args_list), (exited, limits.return_value.__exit__.called), \
            [record.getMessage() for record in logs.records]

    def test_d29_threads_with_the_variables_unset(self):
        """FAISS gets the physical cores (the previous count restored after the block), and so does BLAS; the INFO line
        names the BLAS libraries threadpoolctl found (the limit reaches no other), or says it found none."""
        inside, after, limits, exited, log = self.threads()
        self.assertEqual(inside, [mock.call(6)])
        self.assertEqual(after, [mock.call(6), mock.call(3)])
        self.assertEqual(limits, [mock.call(limits=6, user_api="blas")])
        self.assertEqual(exited, (False, True))
        self.assertEqual(len(log), 1)
        self.assertRegex(log[0], r"^FAISS/BLAS threads set to 6 \(physical cores of the \d+ allowed CPUs\); BLAS "
                                 r"libraries limited: libscipy_openblas-68440149\.so, "
                                 r"libscipy_openblas64_-ff651d7f\.so$")
        *_, log = self.threads(pools=self.POOLS[3:])
        self.assertRegex(log[0], r"^FAISS/BLAS threads set to 6 \(physical cores of the \d+ allowed CPUs\); "
                                 r"threadpoolctl found no BLAS library to limit$")

    def test_d29_omp_num_threads_set(self):
        inside, after, limits, _, log = self.threads(OMP_NUM_THREADS="2")
        self.assertEqual((inside, after, limits), ([], [], []))
        self.assertEqual(log, ["FAISS threads left to OMP_NUM_THREADS=2; BLAS threads left to OMP_NUM_THREADS=2"])

    def test_d29_a_blas_variable_set(self):
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            with self.subTest(name=name):
                inside, after, limits, _, log = self.threads(**{name: "1"})
                self.assertEqual((inside, limits), ([mock.call(6)], []))
                self.assertRegex(log[0], rf"^FAISS threads set to 6 \(physical cores of the \d+ allowed CPUs\); "
                                         rf"BLAS threads left to {name}=1$")

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
                                      f"directly in the batch script or with srun -n 1.")

    def test_i3_the_batch_step_of_a_multi_node_job(self):
        self.embed_run()
        with mock.patch.object(split_run, "find_run") as find_run:
            message = self.refused_index(env={"SLURM_JOB_ID": 4711, "SLURM_JOB_NUM_NODES": 2, "SLURM_NTASKS": 8})
        find_run.assert_not_called()
        self.assertIn("SLURM job 4711 holds 2 nodes", message)
        # No dependency on the job being refused: its afterok would never be met.
        self.assertTrue(message.endswith(
            "Run it as its own one-node CPU job instead: once this job has ended, sbatch -C cpu -N 1 ... (no "
            "dependency is needed then); next time, submit it from the GPU job's script before the job's work starts: "
            "sbatch -C cpu -N 1 --dependency=afterok:$SLURM_JOB_ID --kill-on-invalid-dep=yes ..."), message)
        self.assertNotIn("afterok:4711", message)

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
