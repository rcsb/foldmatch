"""Shared run id and shard handling for multi-rank embedding passes (G7)."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest.mock
from unittest.mock import patch

import numpy as np
import torch

import foldmatch
from foldmatch.types.api_types import Accelerator
from foldmatch.utils.dist import RUN_ID_ENV, command_digest, exported_run_id, launch_rank, run_id

_TEST_PATH = Path(__file__).parent
_FIXTURE_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
_SRC = str(Path(foldmatch.__file__).resolve().parent.parent)

# Everything that decides the run id or the launch rank.
_LAUNCH_ENV = (RUN_ID_ENV, "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_RESTART_COUNT",
               "SLURM_PROCID", "RANK", "LOCAL_RANK", "NODE_RANK")


@contextmanager
def launch_env(**values):
    """Run with exactly ``values`` set among the launch variables (the rest unset).

    Only those variables are saved and restored. Restoring the whole environment
    would also undo what libraries set on first import (sklearn sets
    KMP_DUPLICATE_LIB_OK, without which a second OpenMP runtime aborts the process).
    """
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


# Rendezvous state a parent process may hold from its own (single-rank) DDP pass;
# a child that inherited MASTER_PORT would try to join that dead process group.
_DDP_ENV_PREFIXES = ("MASTER_", "PL_", "TORCHELASTIC_")
_DDP_ENV = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK")


def _child_env(**values):
    env = {k: v for k, v in os.environ.items()
           if k not in _LAUNCH_ENV and k not in _DDP_ENV and not k.startswith(_DDP_ENV_PREFIXES)}
    env.update({k: str(v) for k, v in values.items()})
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _embedding(name, dim=8):
    vec = np.random.default_rng(zlib.crc32(name.encode())).normal(size=dim).astype(np.float32)
    return torch.from_numpy(vec / np.linalg.norm(vec))


def _write_shard(folder, out_name, rank, names, keys=None):
    """Write one rank's shard exactly as the Lightning writer callback does; each row's embedding is fixed by its
    entry of ``keys`` (default: its id)."""
    from foldmatch.writer.batch_writer import ParquetBatchWriter
    writer = ParquetBatchWriter(folder, out_name)
    trainer = SimpleNamespace(global_rank=rank)
    writer.write_on_batch_end(trainer, None, ([_embedding(k) for k in (keys or names)], tuple(names)),
                              None, None, 0, 0)
    writer.on_predict_end(trainer, None)


def fake_chain_predict(world_size=1, drop=(), by_sequence=False):
    """Stand-in for chain_complete_inference.predict on FASTA input.

    Splits the entries over ``world_size`` simulated ranks the way the
    distributed sampler does and writes one shard per rank; ``drop`` loses
    entries, like a rank whose shard never reaches rank 0. With
    ``by_sequence`` a row's embedding is fixed by the sequence the dataset
    gives (as with ESM3) instead of by its id, so two records with one id
    have different embeddings.
    """
    def predict(**kwargs):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        dataset = EsmProtFromFasta(kwargs["src_stream"], min_res_n=kwargs["min_res_n"])
        items = [dataset[i] for i in range(len(dataset))]
        items = [(prot, n) for prot, n in items if n not in drop]
        for rank in range(world_size):
            dealt = items[rank::world_size]
            _write_shard(kwargs["out_folder"], kwargs["out_name"], rank, [n for _, n in dealt],
                         keys=[prot.sequence for prot, _ in dealt] if by_sequence else None)
    return predict


def _ids(computer):
    return sorted(i for ids, _ in computer.get_embedding_batches() for i in ids)


def _digest(argv):
    """What command_digest() gives for the command line ``argv``."""
    return hashlib.sha1("\0".join(argv[1:]).encode()).hexdigest()[:8]


# Two command lines, as two commands of one srun step would have them.
_BUILD_ARGV = ["fm-search", "build", "sequences", "--fasta-file", "a.fasta", "--output-db", "db"]
_QUERY_ARGV = ["fm-search", "query", "sequences", "--db-path", "db", "--fasta-file", "q.fasta"]


class RunIdTest(unittest.TestCase):

    def test_explicit_id_wins(self):
        with launch_env(FOLDMATCH_RUN_ID="my-run.1", SLURM_JOB_ID=7, LOCAL_RANK=1):
            self.assertEqual(run_id(), "my-run.1")

    def test_srun_step_id_includes_step_restart_count_and_command(self):
        digest = _digest(_BUILD_ARGV)
        with patch.object(sys, "argv", _BUILD_ARGV):
            self.assertEqual(command_digest(), digest)
            with launch_env(SLURM_JOB_ID=123, SLURM_STEP_ID=4, SLURM_RESTART_COUNT=2, SLURM_PROCID=3):
                self.assertEqual(run_id(), f"123_4_2_{digest}")
            with launch_env(SLURM_JOB_ID=123, SLURM_STEP_ID=4):
                self.assertEqual(run_id(), f"123_4_0_{digest}")
        # The program path is not part of it: every task runs the same arguments.
        with patch.object(sys, "argv", ["/other/path/search.py"] + _BUILD_ARGV[1:]):
            self.assertEqual(command_digest(), digest)
        with patch.object(sys, "argv", _QUERY_ARGV):
            self.assertNotEqual(command_digest(), digest)

    def test_sbatch_batch_step_draws_a_fresh_id_per_command(self):
        # The batch step has no SLURM_STEP_ID, or one of SLURM's special step values.
        for step in ({}, {"SLURM_STEP_ID": 4294967294}, {"SLURM_STEP_ID": 0xFFFFFFFB}):
            with self.subTest(step=step), patch.object(sys, "argv", _BUILD_ARGV):
                with launch_env(SLURM_JOB_ID=123, SLURM_PROCID=0, **step):
                    first, second = run_id(), run_id()
                self.assertRegex(first, r"^[0-9a-f]{16}$")
                self.assertNotEqual(first, second)
                # Only a Lightning child (which inherits the id) can be rank > 0 there.
                with launch_env(SLURM_JOB_ID=123, SLURM_PROCID=0, LOCAL_RANK=1, **step):
                    with self.assertRaises(ValueError):
                        run_id()

    def test_rank_zero_draws_a_fresh_id(self):
        with launch_env():
            first, second = run_id(), run_id()
        self.assertRegex(first, r"^[0-9a-f]{16}$")
        self.assertNotEqual(first, second)

    def test_rank_above_zero_without_an_id_refuses(self):
        for name in ("LOCAL_RANK", "SLURM_PROCID", "NODE_RANK", "RANK"):
            with self.subTest(name=name), launch_env(**{name: 1}):
                with self.assertRaises(ValueError) as ctx:
                    run_id()
                self.assertIn(RUN_ID_ENV, str(ctx.exception))

    def test_invalid_explicit_id_is_refused(self):
        for bad in ("../x", "a/b", "a*", "-lead", "a b"):
            with self.subTest(bad=bad), launch_env(FOLDMATCH_RUN_ID=bad):
                with self.assertRaises(ValueError):
                    run_id()

    def test_launch_rank(self):
        cases = [({}, 0), ({"LOCAL_RANK": 0, "NODE_RANK": 0}, 0), ({"LOCAL_RANK": 2}, 2),
                 ({"SLURM_PROCID": 5}, 5), ({"NODE_RANK": 1, "LOCAL_RANK": 0}, 1),
                 ({"RANK": 0, "LOCAL_RANK": 0, "SLURM_PROCID": 0}, 0), ({"RANK": 6, "LOCAL_RANK": 2}, 6),
                 # A Lightning child of a single srun task inherits SLURM_PROCID=0.
                 ({"SLURM_PROCID": 0, "LOCAL_RANK": 1}, 1)]
        for values, expected in cases:
            with self.subTest(values=values), launch_env(**values):
                self.assertEqual(launch_rank(), expected)

    def test_exported_run_id_restores_the_environment(self):
        with launch_env():
            with exported_run_id("abc"):
                self.assertEqual(os.environ[RUN_ID_ENV], "abc")
            self.assertNotIn(RUN_ID_ENV, os.environ)
        with launch_env(FOLDMATCH_RUN_ID="user"):
            with exported_run_id("user"):
                self.assertEqual(os.environ[RUN_ID_ENV], "user")
            self.assertEqual(os.environ[RUN_ID_ENV], "user")
        with launch_env():
            with self.assertRaises(RuntimeError):
                with exported_run_id("abc"):
                    raise RuntimeError("pass failed")
            self.assertNotIn(RUN_ID_ENV, os.environ)


class ShardIsolationTest(unittest.TestCase):
    """Runs sharing one folder only ever read their own shards."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _computer(self, **env):
        from foldmatch.search.embedding_computer import EmbeddingComputer
        with launch_env(**env):
            return EmbeddingComputer(self.tmp)

    def test_same_slurm_job_different_steps(self):
        with patch.object(sys, "argv", _BUILD_ARGV):
            step0 = self._computer(SLURM_JOB_ID=99, SLURM_STEP_ID=0)
            step1 = self._computer(SLURM_JOB_ID=99, SLURM_STEP_ID=1)
            requeued = self._computer(SLURM_JOB_ID=99, SLURM_STEP_ID=0, SLURM_RESTART_COUNT=1)
        self.assertEqual(len({step0.out_name, step1.out_name, requeued.out_name}), 3)
        _write_shard(self.tmp, step0.out_name, 0, ["s0_a"])
        _write_shard(self.tmp, step0.out_name, 1, ["s0_b"])
        _write_shard(self.tmp, step1.out_name, 0, ["s1_a"])
        _write_shard(self.tmp, requeued.out_name, 0, ["r_a"])
        self.assertEqual(_ids(step0), ["s0_a", "s0_b"])
        self.assertEqual(_ids(step1), ["s1_a"])
        self.assertEqual(_ids(requeued), ["r_a"])
        digest = _digest(_BUILD_ARGV)
        self.assertEqual([p.name for p in step0.shard_files()],
                         [f"emb_99_0_0_{digest}-0.parquet", f"emb_99_0_0_{digest}-1.parquet"])

    def test_different_commands_in_one_srun_step(self):
        with patch.object(sys, "argv", _BUILD_ARGV):
            build = self._computer(SLURM_JOB_ID=99, SLURM_STEP_ID=2, SLURM_PROCID=0)
            build_rank1 = self._computer(SLURM_JOB_ID=99, SLURM_STEP_ID=2, SLURM_PROCID=1)
        with patch.object(sys, "argv", _QUERY_ARGV):
            query = self._computer(SLURM_JOB_ID=99, SLURM_STEP_ID=2, SLURM_PROCID=0)
        self.assertEqual(build.out_name, build_rank1.out_name)
        self.assertNotEqual(build.out_name, query.out_name)
        _write_shard(self.tmp, build.out_name, 0, ["b0"])
        _write_shard(self.tmp, build_rank1.out_name, 1, ["b1"])
        _write_shard(self.tmp, query.out_name, 0, ["q0"])
        self.assertEqual(_ids(build), ["b0", "b1"])
        self.assertEqual(_ids(query), ["q0"])
        query.remove_shards()
        self.assertEqual(_ids(build), ["b0", "b1"])

    def test_commands_in_one_sbatch_batch_step(self):
        with patch.object(sys, "argv", _BUILD_ARGV):
            first = self._computer(SLURM_JOB_ID=99)
            second = self._computer(SLURM_JOB_ID=99)
        self.assertNotEqual(first.out_name, second.out_name)

    def test_existing_shards_with_this_id_are_refused_before_the_pass(self):
        computer = self._computer(FOLDMATCH_RUN_ID="reused")
        _write_shard(self.tmp, computer.out_name, 1, ["stale"])
        fasta = self.tmp / "one.fasta"
        fasta.write_text(">q0\nACDEFGHIKL\n")
        structures = self.tmp / "structures"
        structures.mkdir()
        (structures / "x.cif").write_text("")
        predict = unittest.mock.Mock()
        for method, path in ((computer.compute_from_fasta, fasta), (computer.compute_from_structures, structures)):
            with self.subTest(method=method.__name__), \
                    patch("foldmatch.search.embedding_computer.chain_predict", predict), \
                    patch("foldmatch.search.embedding_computer.assembly_predict", predict):
                with self.assertRaises(FileExistsError) as ctx:
                    method(path, accelerator=Accelerator.cpu)
                self.assertIn("same run id (reused)", str(ctx.exception))
                self.assertIn(RUN_ID_ENV, str(ctx.exception))
        predict.assert_not_called()
        # Another id in the same folder is not affected.
        other = self._computer(FOLDMATCH_RUN_ID="fresh")
        with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
            other.compute_from_fasta(fasta, accelerator=Accelerator.cpu)
        self.assertEqual(_ids(other), ["q0"])

    def test_id_prefix_does_not_leak(self):
        short = self._computer(FOLDMATCH_RUN_ID="abc")
        longer = self._computer(FOLDMATCH_RUN_ID="abc-1")
        _write_shard(self.tmp, short.out_name, 0, ["short"])
        _write_shard(self.tmp, longer.out_name, 0, ["longer"])
        self.assertEqual(_ids(short), ["short"])
        self.assertEqual(_ids(longer), ["longer"])

    def test_count_check(self):
        computer = self._computer(FOLDMATCH_RUN_ID="run")
        _write_shard(self.tmp, computer.out_name, 0, ["a", "b"])
        computer.expected_count = 2
        self.assertEqual(_ids(computer), ["a", "b"])
        # A stale shard from an earlier run with the same id is detected, not merged.
        _write_shard(self.tmp, computer.out_name, 1, ["stale"])
        with self.assertRaises(ValueError) as ctx:
            _ids(computer)
        self.assertIn("Found 3 embeddings", str(ctx.exception))
        self.assertIn("2 entries", str(ctx.exception))
        # So is a missing rank.
        computer.expected_count = 4
        with self.assertRaises(ValueError):
            _ids(computer)

    def test_remove_shards_only_removes_this_run(self):
        mine = self._computer(FOLDMATCH_RUN_ID="mine")
        other = self._computer(FOLDMATCH_RUN_ID="mine-2")
        _write_shard(self.tmp, mine.out_name, 0, ["a"])
        _write_shard(self.tmp, mine.out_name, 1, ["b"])
        _write_shard(self.tmp, other.out_name, 0, ["c"])
        mine.remove_shards()
        self.assertEqual(mine.shard_files(), [])
        self.assertEqual(_ids(other), ["c"])


class InputOrderTest(unittest.TestCase):
    """P7.0a: :meth:`EmbeddingComputer.in_input_order` puts a multi-rank run's search results (rank 0 reads the shards
    rank by rank) back in input order, with the values of a 1-rank run, a repeated id included; values are never
    changed. The fake pass fixes each row's embedding by its sequence, as ESM3 does, so two records with one id
    differ."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(launch_env(FOLDMATCH_RUN_ID="order"))

    def _pass(self, records, world_size, min_res_n=0):
        """A FASTA pass over ``records`` with the fake pass on ``world_size`` ranks; returns the computer and
        ``{row: embedding}`` as rank 0 reads it from the shards (the dict a search builds with ``rows=True``)."""
        from foldmatch.search.embedding_computer import EmbeddingComputer
        fasta = self.tmp / "q.fasta"
        fasta.write_text("".join(f">{name} some description\n{seq}\n\n" for name, seq in records))
        computer = EmbeddingComputer(self.tmp)
        with patch("foldmatch.search.embedding_computer.chain_predict",
                   fake_chain_predict(world_size=world_size, by_sequence=True)):
            computer.compute_from_fasta(fasta, min_res_n=min_res_n, accelerator=Accelerator.cpu)
        results = {}
        for rows, embeddings in computer.get_embedding_batches(rows=True):
            for row, embedding in zip(rows, embeddings):
                results[row] = embedding
        return computer, results

    def _finish(self, computer):
        computer.remove_shards()

    @staticmethod
    def _one_rank(records, min_res_n=0):
        """What a 1-rank search's dict holds: ``{id: embedding of its last kept record}``, each id at its first."""
        out = {}
        for name, seq in records:
            if len(seq) >= min_res_n:
                out[name] = _embedding(seq).numpy()
        return out

    def assertSameResults(self, got, want):
        self.assertEqual(list(got), list(want))
        for name in want:
            np.testing.assert_array_equal(got[name], want[name], err_msg=name)

    def test_two_ranks_are_put_back_in_input_order(self):
        records = [(f"q{k}", "ACDEFGHIKL" * (k + 1)) for k in range(7)]
        names = [name for name, _ in records]
        for world_size in (2, 3):
            with self.subTest(world_size=world_size):
                computer, results = self._pass(records, world_size)
                read = [i for ids, _ in computer.get_embedding_batches() for i in ids]
                self.assertNotEqual(read, names)            # rank order: rank 0's queries first
                ordered = computer.in_input_order(results)
                self.assertSameResults(ordered, self._one_rank(records))
                values = {id(v) for v in results.values()}
                self.assertTrue(all(id(v) in values for v in ordered.values()))   # the values, not copies
                self.assertEqual(computer.input_ids(), names)
                self._finish(computer)

    def test_one_rank_keeps_the_input_order(self):
        records = [(f"q{k}", "ACDEFGHIKL" * k) for k in (3, 1, 2)]
        computer, results = self._pass(records, 1)
        ordered = computer.in_input_order(results)
        self.assertEqual(list(ordered), ["q3", "q1", "q2"])
        for row, name in enumerate(["q3", "q1", "q2"]):
            self.assertIs(ordered[name], results[row])

    def test_only_the_kept_records_count(self):
        """``min_res_n`` drops records before the pass; the order is that of the kept ones, and a subset of the
        results (a filtered dict) keeps that order."""
        records = [("short", "ACD"), ("b", "ACDEFGHIKL"), ("a", "ACDEFGHIKLM"), ("c", "ACDEFGHIKLMN")]
        computer, results = self._pass(records, 2, min_res_n=5)
        self.assertEqual(computer.input_ids(), ["b", "a", "c"])
        self.assertEqual([i for ids, _ in computer.get_embedding_batches() for i in ids], ["b", "c", "a"])
        self.assertSameResults(computer.in_input_order(results), self._one_rank(records, 5))
        self.assertEqual(list(computer.in_input_order({1: "c", 0: "b"})), ["b", "c"])
        self.assertEqual(computer.input_sequences(), {"b": "ACDEFGHIKL", "a": "ACDEFGHIKLM", "c": "ACDEFGHIKLMN"})

    def test_a_repeated_id_takes_its_last_kept_record(self):
        """Duplicate ids (accepted by a plain pass): one key, at its first kept record, with the embedding of its
        last kept record, as in a 1-rank run, whatever the ranks; and the sequence Stage 2 aligns is that record's
        too. Two ranks deal x(1), z | y, x(2): rank 0's shard, read first, holds the earlier x."""
        x1, x2 = "ACDEFGHIKL", "KLMNPQRSTVWY"
        records = [("x", x1), ("y", "ACDEFGHIKLM"), ("z", "ACDEFGHIKLMN"), ("x", x2)]
        for world_size in (1, 2, 3):
            with self.subTest(world_size=world_size):
                computer, results = self._pass(records, world_size)
                self.assertEqual(computer.input_ids(), ["x", "y", "z", "x"])
                ordered = computer.in_input_order(results)
                self.assertSameResults(ordered, self._one_rank(records))
                np.testing.assert_array_equal(ordered["x"], _embedding(x2).numpy())
                self.assertEqual(computer.input_sequences()["x"], x2)
                self._finish(computer)
        # the last record is too short for the pass: the earlier one is the query, for Stage 1 and Stage 2 alike
        short = [("x", x1 * 2), ("y", "ACDEFGHIKLM"), ("x", "ACD")]
        for world_size in (1, 2):
            with self.subTest(world_size=world_size, min_res_n=5):
                computer, results = self._pass(short, world_size, min_res_n=5)
                ordered = computer.in_input_order(results)
                self.assertSameResults(ordered, self._one_rank(short, 5))
                self.assertEqual(computer.input_sequences(), {"x": x1 * 2, "y": "ACDEFGHIKLM"})
                self._finish(computer)

    def test_shards_not_dealt_round_robin(self):
        """Shards that are not the sampler's round robin (never from Lightning's predict): an id that occurs once is
        still placed exactly; a repeated id takes its positions in read order, with a warning."""
        from foldmatch.search.embedding_computer import EmbeddingComputer
        for records, warned in (([("a", "AC"), ("b", "DE"), ("c", "FG"), ("d", "HI")], False),
                                ([("x", "AC"), ("b", "DE"), ("x", "FG"), ("d", "HI")], True)):
            with self.subTest(warned=warned):
                fasta = self.tmp / "q.fasta"
                fasta.write_text("".join(f">{name}\n{seq}\n" for name, seq in records))

                def contiguous(**kwargs):
                    half = [records[:2], records[2:]]
                    for rank, part in enumerate(half):
                        _write_shard(kwargs["out_folder"], kwargs["out_name"], rank, [n for n, _ in part],
                                     keys=[seq for _, seq in part])

                computer = EmbeddingComputer(self.tmp)
                with patch("foldmatch.search.embedding_computer.chain_predict", contiguous):
                    computer.compute_from_fasta(fasta, accelerator=Accelerator.cpu)
                results = {row: e for rows, embs in computer.get_embedding_batches(rows=True)
                           for row, e in zip(rows, embs)}
                with self.assertLogs("foldmatch.search.embedding_computer", level="WARNING") if warned else \
                        _no_warning(self, "foldmatch.search.embedding_computer"):
                    ordered = computer.in_input_order(results)
                self.assertSameResults(ordered, self._one_rank(records))
                self._finish(computer)

    def test_an_id_not_in_the_input_is_an_error(self):
        computer, results = self._pass([("a", "ACDEFGHIKL"), ("b", "ACDEFGHIKLM")], 2)
        _write_shard(self.tmp, computer.out_name, 2, ["stale"])
        with self.assertRaises(ValueError) as ctx:
            computer.in_input_order(results)
        self.assertIn("'stale'", str(ctx.exception))
        self.assertIn("shards of another run", str(ctx.exception))
        (self.tmp / f"{computer.out_name}-2.parquet").unlink()
        self.assertEqual(list(computer.in_input_order(results)), ["a", "b"])
        with self.assertRaises(ValueError) as ctx:
            computer.in_input_order({99: None})
        self.assertIn("not a row", str(ctx.exception))

    def test_only_after_a_fasta_pass(self):
        from foldmatch.search.embedding_computer import EmbeddingComputer
        for method in ("in_input_order", "input_sequences"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                getattr(EmbeddingComputer(self.tmp), method)(*(({},) if method == "in_input_order" else ()))


@contextmanager
def _no_warning(case, logger_name):
    """``assertNoLogs`` at WARNING."""
    with case.assertNoLogs(logger_name, level="WARNING"):
        yield


# A simulated rank > 0: it re-runs the command, so it derives its own run id,
# then writes its shard the way the Lightning writer callback does.
_CHILD = """
import json
import sys
from types import SimpleNamespace
import torch
from foldmatch.search.embedding_computer import EmbeddingComputer
from foldmatch.writer.batch_writer import ParquetBatchWriter
folder, rank, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if len(sys.argv) > 4:
    sys.argv = json.loads(sys.argv[4])  # the command line srun gives every task
computer = EmbeddingComputer(folder)
writer = ParquetBatchWriter(folder, computer.out_name)
trainer = SimpleNamespace(global_rank=rank)
writer.write_on_batch_end(trainer, None, ([torch.ones(8)], (name,)), None, None, 0, 0)
writer.on_predict_end(trainer, None)
"""


class RunIdSharingTest(unittest.TestCase):
    """Simulated ranks in separate processes: rank 0 reads every rank's shard."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.fasta = self.tmp / "two.fasta"
        self.fasta.write_text(">q0\nACDEFGHIKL\n>q1\nMNPQRSTVWY\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _child(self, rank, name, argv=None, **env):
        return subprocess.run(
            [sys.executable, "-c", _CHILD, str(self.tmp), str(rank), name]
            + ([json.dumps(argv)] if argv is not None else []),
            env=_child_env(**env), capture_output=True, text=True,
        )

    def _run(self, child_env, parent_env=None, inherit_run_id=True, same_argv=False):
        """Rank 0 runs the pass; its predict launches rank 1 with ``child_env``
        (and, with ``same_argv``, rank 0's command line, as srun gives every task)."""
        from foldmatch.search.embedding_computer import EmbeddingComputer
        children = []

        def predict(**kwargs):
            _write_shard(kwargs["out_folder"], kwargs["out_name"], 0, ["q0"])
            env = dict(child_env)
            if inherit_run_id and RUN_ID_ENV in os.environ:
                env[RUN_ID_ENV] = os.environ[RUN_ID_ENV]  # what a Lightning child inherits
            children.append(self._child(1, "q1", argv=list(sys.argv) if same_argv else None, **env))

        with launch_env(**(parent_env or {})):
            computer = EmbeddingComputer(self.tmp)
            with patch("foldmatch.search.embedding_computer.chain_predict", predict):
                computer.compute_from_fasta(self.fasta, accelerator=Accelerator.cpu)
            self.assertNotIn(RUN_ID_ENV, os.environ)  # cleared after the pass
        return computer, children[0]

    def test_lightning_child_with_local_rank(self):
        computer, child = self._run({"LOCAL_RANK": 1})
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(_ids(computer), ["q0", "q1"])

    def test_child_with_slurm_procid(self):
        computer, child = self._run({"SLURM_PROCID": 1})
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(_ids(computer), ["q0", "q1"])

    def test_srun_tasks_share_the_slurm_id(self):
        slurm = {"SLURM_JOB_ID": 555, "SLURM_STEP_ID": 2}
        with patch.object(sys, "argv", _BUILD_ARGV):
            computer, child = self._run(dict(slurm, SLURM_PROCID=1), parent_env=dict(slurm, SLURM_PROCID=0),
                                        inherit_run_id=False, same_argv=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(computer.run_id, f"555_2_0_{_digest(_BUILD_ARGV)}")
        self.assertEqual(_ids(computer), ["q0", "q1"])

    def test_child_without_an_id_refuses(self):
        for rank_env in ({"LOCAL_RANK": 1}, {"SLURM_PROCID": 1}):
            with self.subTest(rank_env=rank_env):
                for path in self.tmp.glob("emb_*"):
                    path.unlink()
                computer, child = self._run(rank_env, inherit_run_id=False)
                self.assertNotEqual(child.returncode, 0)
                self.assertIn(RUN_ID_ENV, child.stderr)
                # Only rank 0's shard exists, and rank 0 says so instead of building from half the data.
                with self.assertRaises(ValueError):
                    _ids(computer)


class CliShardLifecycleTest(unittest.TestCase):
    """Build shards are kept unless --cleanup-embeddings; query shards are deleted;
    a lost rank fails the build before the index is written."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(launch_env())

    def _build(self, name, predict, **kwargs):
        from foldmatch.cli.search import build_database_from_fasta
        with patch("foldmatch.search.embedding_computer.chain_predict", predict):
            build_database_from_fasta(fasta_file=_FIXTURE_FASTA, output_db=str(self.tmp / name),
                                      tmp_embedding_folder=self.tmp, accelerator=Accelerator.cpu,
                                      **kwargs)

    def _shards(self):
        return sorted(p.name for p in self.tmp.glob("emb_*.parquet"))

    def test_two_rank_build_keeps_shards_and_query_deletes_its_own(self):
        from foldmatch.cli.search import query_database_from_fasta
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        self._build("db", fake_chain_predict(world_size=2))
        db = FaissEmbeddingDatabase(self.tmp, "db")
        db.load_database()
        self.assertEqual(sorted(db.chain_ids), ["1acb_E", "2uzi_A"])
        build_shards = self._shards()
        self.assertEqual(len(build_shards), 2)
        self.assertEqual(len({name.rsplit("-", 1)[0] for name in build_shards}), 1)

        output = self.tmp / "hits.tsv"
        with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict(world_size=2)):
            query_database_from_fasta(db_path=str(self.tmp / "db"), fasta_file=_FIXTURE_FASTA,
                                      tmp_embedding_folder=self.tmp, output_file=str(output),
                                      threshold=None, seq_identity=False, accelerator=Accelerator.cpu)
        self.assertEqual({line.split("\t")[0] for line in output.read_text().splitlines()},
                         {"1acb_E", "2uzi_A"})
        self.assertEqual(self._shards(), build_shards)

    def test_build_then_query_in_one_sbatch_batch_step(self):
        """Each command of a batch step gets its own id, so the query leaves the build's shards alone."""
        from foldmatch.cli.search import query_database_from_fasta
        with launch_env(SLURM_JOB_ID=4242, SLURM_PROCID=0):
            self._build("db", fake_chain_predict(world_size=2))
            build_shards = self._shards()
            self.assertEqual(len(build_shards), 2)
            with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict(world_size=1)):
                query_database_from_fasta(db_path=str(self.tmp / "db"), fasta_file=_FIXTURE_FASTA,
                                          tmp_embedding_folder=self.tmp, output_file=str(self.tmp / "hits.tsv"),
                                          threshold=None, seq_identity=False, accelerator=Accelerator.cpu)
            self._build("db2", fake_chain_predict(world_size=1))
        shards = self._shards()
        self.assertEqual(len(shards), 3)
        self.assertTrue(set(build_shards) <= set(shards))

    def test_same_command_twice_in_one_srun_step_is_refused_before_esm3(self):
        with launch_env(SLURM_JOB_ID=4242, SLURM_STEP_ID=3, SLURM_PROCID=0), patch.object(sys, "argv", _BUILD_ARGV):
            self._build("db", fake_chain_predict(world_size=2))
            build_shards = self._shards()
            for suffix in (".index", ".metadata", ".sequences"):
                (self.tmp / f"db{suffix}").unlink()
            predict = unittest.mock.Mock(side_effect=fake_chain_predict(world_size=1))
            with self.assertRaises(FileExistsError) as ctx:
                self._build("db", predict)
        predict.assert_not_called()
        self.assertIn("same run id", str(ctx.exception))
        self.assertEqual(self._shards(), build_shards)

    def test_cleanup_embeddings(self):
        self._build("db", fake_chain_predict(world_size=2), cleanup_embeddings=True)
        self.assertTrue((self.tmp / "db.index").exists())
        self.assertEqual(self._shards(), [])

    def test_lost_rank_fails_before_the_index_is_written(self):
        with self.assertRaises(ValueError) as ctx:
            self._build("db", fake_chain_predict(world_size=2, drop=("2uzi_A",)))
        self.assertIn("Found 1 embeddings", str(ctx.exception))
        self.assertFalse((self.tmp / "db.index").exists())
        self.assertFalse((self.tmp / "db.sequences").exists())


@unittest.skipUnless(os.environ.get("FOLDMATCH_SLOW_TESTS") == "1",
                     "slow: two ESM3 ranks on CPU; set FOLDMATCH_SLOW_TESTS=1 to run")
class TwoRankDDPSmoke(unittest.TestCase):
    """A real two-process DDP build (CPU, gloo) keeps both ranks' embeddings."""

    def test_two_rank_cpu_build_indexes_every_entry(self):
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        from foldmatch.search.sequence_store import SequenceStore
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result = subprocess.run(
                [sys.executable, "-m", "foldmatch.cli.search", "build", "sequences",
                 "--fasta-file", str(_FIXTURE_FASTA), "--output-db", str(tmp / "db"),
                 "--tmp-embedding-folder", str(tmp), "--accelerator", "cpu", "--devices", "2"],
                env=_child_env(), capture_output=True, text=True, timeout=1800,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-4000:])
            db = FaissEmbeddingDatabase(tmp, "db")
            db.load_database()
            self.assertEqual(sorted(db.chain_ids), ["1acb_E", "2uzi_A"])
            self.assertEqual(SequenceStore(tmp, "db").count(), 2)
            shards = sorted(p.name for p in tmp.glob("emb_*.parquet"))
            self.assertEqual(len(shards), 2, shards)
            self.assertEqual(shards[0].rsplit("-", 1)[0], shards[1].rsplit("-", 1)[0])


if __name__ == "__main__":
    unittest.main()
