"""P8b: the split sequence build (``build sequences-embed``, ``build sequences-index``, ``build sequences-seal``) and
the phases of ``build sequences`` it shares with the one-shot build.

No ESM3: the pass is ``fake_chain_predict`` and the spies are those of ``test_cli_residue_codes``. ``OneShotPhasesTest``
(P8b.1) checks that ``build sequences`` runs its phases in the old order with the old arguments, the run binding and
the pre-flight of ``EmbeddingComputer``, the index-phase log lines and the CPU-affinity message. ``EmbedOptionTest``,
``EmbedStepTest`` and ``EmbedRefusalTest`` (P8b.3) check step 1, ``build sequences-embed``: its options, the shards and
the run record it leaves, and each of its refusals E1-E12 at its place. Two ranks are the fake pass's two shards plus a
patched ``_gather_rank_facts`` that returns two ranks' facts.
"""
import errno
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
import unittest
from contextlib import ExitStack, chdir, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

import foldmatch
import foldmatch.search.code_alignment as code_alignment
from foldmatch.cli import search
from foldmatch.search import alignment, split_run
from foldmatch.search import scoring_system as ss
from foldmatch.search.embedding_computer import EmbeddingComputer
from foldmatch.search.embedding_database import EmbeddingDatabase
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.search.residue_code_store import ResidueCodeStore
from foldmatch.search.residue_shards import META_PASS, pass_record, read_shard_info
from foldmatch.search.sequence_store import SequenceStore
from foldmatch.types.api_types import Accelerator, IndexConfig, IndexType, Strategy
from foldmatch.utils.esm.loaders import ESM_REVISION
from foldmatch.utils.sequence import preflight_fasta
from test_cli_residue_codes import OTHER_ENCODING, PLAN_HELP, _esm3_spy, _Tmp, _WithBundle, _build_args, _write, \
    fake_chain_predict, invoke, launch_env

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


if __name__ == "__main__":
    unittest.main()
