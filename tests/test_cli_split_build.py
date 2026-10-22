"""P8b: the split sequence build (``build sequences-embed``, ``build sequences-index``, ``build sequences-seal``) and
the phases of ``build sequences`` it shares with the one-shot build.

No ESM3: the pass is ``fake_chain_predict`` and the spies are those of ``test_cli_residue_codes``. ``OneShotPhasesTest``
(P8b.1) checks that ``build sequences`` runs its phases in the old order with the old arguments, the run binding and
the pre-flight of ``EmbeddingComputer``, the index-phase log lines and the CPU-affinity message.
"""
import logging
import os
import re
import shutil
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import foldmatch.search.code_alignment as code_alignment
from foldmatch.cli import search
from foldmatch.search import alignment
from foldmatch.search import scoring_system as ss
from foldmatch.search.embedding_computer import EmbeddingComputer
from foldmatch.search.embedding_database import EmbeddingDatabase
from foldmatch.search.residue_code_store import ResidueCodeStore
from foldmatch.search.sequence_store import SequenceStore
from foldmatch.types.api_types import Accelerator, IndexConfig, IndexType, Strategy
from test_cli_residue_codes import _WithBundle, _build_args, _write, fake_chain_predict, invoke, launch_env

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


if __name__ == "__main__":
    unittest.main()
