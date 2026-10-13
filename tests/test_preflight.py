"""Read-only FASTA pre-flight on every rank before ESM3 (G9)."""
import functools
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
from unittest.mock import Mock, patch

import numpy as np
import torch

import foldmatch
from foldmatch.types.api_types import Accelerator, StructureFormat
from foldmatch.utils.sequence import FastaPreflight, preflight_fasta, scan_fasta

_TEST_PATH = Path(__file__).parent
_FIXTURE_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
_TOKENIZER_CASES = _TEST_PATH / "resources" / "esm_codes" / "tokenizer_cases_v1.json"
_SRC = str(Path(foldmatch.__file__).resolve().parent.parent)
_LAUNCH_ENV = ("FOLDMATCH_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_RESTART_COUNT",
               "SLURM_PROCID", "RANK", "LOCAL_RANK", "NODE_RANK")

# Records, in order: a(4) b(3, '<') a(5, dup) c(4, non-ASCII) d(10) a(2, dup) [empty e] b(2, dup)
_PROBLEM_FASTA = (">a x\nACDE\n>b\nmk<\n>a\nWWWWW\n>c\nACDÉ\n>d\nAAAAA AAAAA\n"
                  ">a\nCC\n>e\n\n>b\nZZ\n")


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


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def fake_chain_predict(on_pass=None):
    """Stand-in for chain_complete_inference.predict: one shard, built from the
    same dataset the real pass uses."""
    def predict(**kwargs):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        from foldmatch.writer.batch_writer import ParquetBatchWriter
        dataset = EsmProtFromFasta(kwargs["src_stream"], min_res_n=kwargs["min_res_n"])
        names = [dataset[i][1] for i in range(len(dataset))]
        embeddings = []
        for name in names:
            vec = np.random.default_rng(zlib.crc32(name.encode())).normal(size=8).astype(np.float32)
            embeddings.append(torch.from_numpy(vec))
        writer = ParquetBatchWriter(kwargs["out_folder"], kwargs["out_name"])
        trainer = SimpleNamespace(global_rank=0)
        writer.write_on_batch_end(trainer, None, (embeddings, tuple(names)), None, None, 0, 0)
        writer.on_predict_end(trainer, None)
        if on_pass is not None:
            on_pass()
    return predict


def _esm3_spy():
    """Patch the ESM3 loader the FASTA pass uses; any call means ESM3 was reached."""
    return patch("foldmatch.inference.chain_complete_inference.get_residue_model",
                 Mock(side_effect=AssertionError("ESM3 was loaded")))


class PreflightCountsTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.fasta = _write(self.tmp / "problems.fasta", _PROBLEM_FASTA)

    def test_counts(self):
        pf = preflight_fasta(self.fasta)
        self.assertEqual(
            (pf.entries, pf.entries_kept, pf.residues, pf.duplicates, pf.unique_entries,
             pf.unique_residues, pf.max_length),
            (7, 7, 30, 3, 4, 21, 10))
        self.assertEqual((pf.nonstandard_entries, pf.nonstandard_residues), (3, 4))
        self.assertEqual((pf.non_ascii_entries, pf.angle_bracket_entries), (1, 1))
        self.assertEqual(pf.duplicate_ids, ("a", "b"))
        self.assertEqual((pf.non_ascii_ids, pf.angle_bracket_ids), (("c",), ("b",)))
        self.assertEqual(pf.offsets.tolist(), [offset for offset, _, _ in scan_fasta(self.fasta)])

    def test_min_res_n_applies_to_every_count_but_entries(self):
        pf = preflight_fasta(self.fasta, min_res_n=4)
        self.assertEqual(
            (pf.entries, pf.entries_kept, pf.residues, pf.duplicates, pf.unique_residues,
             pf.angle_bracket_entries, pf.non_ascii_entries, pf.nonstandard_entries),
            (7, 4, 23, 1, 18, 0, 1, 1))
        kept = [offset for offset, _, length in scan_fasta(self.fasta) if length >= 4]
        self.assertEqual(pf.offsets.tolist(), kept)

    def test_fixture_and_empty_file(self):
        pf = preflight_fasta(_FIXTURE_FASTA)
        self.assertEqual((pf.entries_kept, pf.residues, pf.duplicates, pf.max_length), (2, 303, 0, 245))
        self.assertEqual(pf.problems(), [])
        empty = preflight_fasta(_write(self.tmp / "empty.fasta", ""))
        self.assertEqual((empty.entries, empty.entries_kept, empty.residues, len(empty.offsets)), (0, 0, 0, 0))

    def test_hash_collisions_are_resolved_exactly(self):
        # Every id hashing to the same value must still count only true repeats.
        with patch("foldmatch.utils.sequence._id_hash", return_value=42):
            pf = preflight_fasta(self.fasta)
        self.assertEqual((pf.duplicates, pf.unique_residues, pf.duplicate_ids), (3, 21, ("a", "b")))
        many = _write(self.tmp / "many.fasta", "".join(f">id{i}\nACDE\n" for i in range(200)) + ">id7\nAC\n")
        with patch("foldmatch.utils.sequence._id_hash", side_effect=lambda name: len(name)):
            pf = preflight_fasta(many)
        self.assertEqual((pf.duplicates, pf.unique_residues, pf.duplicate_ids), (1, 800, ("id7",)))

    def test_strict_refuses_each_problem_and_plain_warns(self):
        cases = {
            "duplicate": ">a\nACDE\n>a\nACDE\n",
            "angle": ">a\nAC<mask>DE\n",
            "non_ascii": ">a\nACDÉ\n",
        }
        for label, text in cases.items():
            with self.subTest(label=label):
                pf = preflight_fasta(_write(self.tmp / f"{label}.fasta", text))
                self.assertEqual(len(pf.problems()), 1)
                with self.assertRaises(ValueError) as ctx:
                    pf.check(strict=True)
                self.assertIn(str(self.tmp / f"{label}.fasta"), str(ctx.exception))
                with self.assertLogs("foldmatch.utils.sequence", level="WARNING"):
                    pf.check(strict=False)
        clean = preflight_fasta(_FIXTURE_FASTA)
        clean.check(strict=True)

    def test_tokenizer_cases(self):
        """F-tok: the pre-flight's per-record figures agree with the fixture."""
        with open(_TOKENIZER_CASES) as f:
            cases = json.load(f)["cases"]
        for case in cases:
            with self.subTest(case=case["id"]):
                pf = preflight_fasta(_write(self.tmp / "case.fasta", f">{case['id']}\n{case['raw']}\n"))
                self.assertEqual(pf.entries, 1)
                expected = case["strict_refusal"]
                self.assertEqual(pf.angle_bracket_entries, int(expected == "angle_bracket"))
                self.assertEqual(pf.non_ascii_entries, int(expected == "non_ascii"))
                if expected is not None:
                    with self.assertRaises(ValueError):
                        pf.check(strict=True)
                if not case["normalize_refuses"]:
                    self.assertEqual(pf.residues, case["len_normalized"])
                    self.assertEqual(pf.nonstandard_residues, case["n_nonstandard"])

    def test_dataset_takes_the_preflight_offsets(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        pf = preflight_fasta(self.fasta, min_res_n=4)
        with patch("foldmatch.utils.sequence._scan_records") as scan:
            dataset = EsmProtFromFasta(pf, min_res_n=4)
        scan.assert_not_called()
        by_path = EsmProtFromFasta(self.fasta, min_res_n=4)
        self.assertEqual(len(dataset), 4)
        self.assertEqual([dataset[i][1] for i in range(4)], [by_path[i][1] for i in range(4)])
        self.assertEqual([dataset[i][0].sequence for i in range(4)], ["ACDE", "WWWWW", "ACDÉ", "A" * 10])
        with self.assertRaises(ValueError):
            EsmProtFromFasta(pf, min_res_n=0)


_PREFLIGHT_CHILD = """
import sys
from foldmatch.cli.search import _fasta_preflight
try:
    _fasta_preflight(sys.argv[1], 0, strict=True)
except ValueError as exc:
    print(exc)
    sys.exit(3)
"""


class PreflightTest(unittest.TestCase):
    """Every rank refuses the same way, before ESM3 is ever loaded."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(launch_env())

    def test_every_rank_raises_the_same_refusal(self):
        fasta = _write(self.tmp / "problems.fasta", _PROBLEM_FASTA)
        outputs = set()
        for rank_env in ({}, {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"},
                         {"SLURM_JOB_ID": 5, "SLURM_PROCID": 3}):
            env = {k: v for k, v in os.environ.items() if k not in _LAUNCH_ENV}
            env.update({k: str(v) for k, v in rank_env.items()})
            env["PYTHONPATH"] = _SRC
            env["PYTHONHASHSEED"] = str(len(outputs) + 1)  # ids must not depend on hash()
            child = subprocess.run([sys.executable, "-c", _PREFLIGHT_CHILD, str(fasta)],
                                   env=env, capture_output=True, text=True)
            self.assertEqual(child.returncode, 3, child.stderr)
            outputs.add(child.stdout)
        self.assertEqual(len(outputs), 1)
        self.assertIn("3 record(s) repeat an earlier id", outputs.pop())

    def _build(self, output_db, fasta=_FIXTURE_FASTA, **kwargs):
        from foldmatch.cli.search import build_database_from_fasta
        build_database_from_fasta(fasta_file=fasta, output_db=str(output_db),
                                  tmp_embedding_folder=self.tmp, accelerator=Accelerator.cpu, **kwargs)

    def test_existing_output_is_refused_before_esm3(self):
        for suffix in (".index", ".metadata", ".invlists", ".sequences", ".rescodes.json", ".rescodes.bin"):
            for rank_env in ({}, {"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}):
                with self.subTest(suffix=suffix, rank_env=rank_env), launch_env(**rank_env), \
                        _esm3_spy() as esm3:
                    existing = self.tmp / f"db{suffix}"
                    existing.touch()
                    try:
                        with self.assertRaises(FileExistsError) as ctx:
                            self._build(self.tmp / "db")
                    finally:
                        existing.unlink()
                    self.assertIn(str(existing), str(ctx.exception))
                    esm3.assert_not_called()

    def test_existing_index_is_refused_before_esm3_for_structures(self):
        from foldmatch.cli.search import build_database_from_structures
        (self.tmp / "sdb.index").touch()
        with _esm3_spy() as esm3:
            with self.assertRaises(FileExistsError):
                build_database_from_structures(
                    structure_folder=_TEST_PATH / "resources" / "pdb", output_db=str(self.tmp / "sdb"),
                    tmp_embedding_folder=self.tmp, structure_format=StructureFormat.mmcif,
                    file_extension=".cif", accelerator=Accelerator.cpu)
        esm3.assert_not_called()

    def test_strict_refusal_comes_before_esm3(self):
        from foldmatch.cli import search
        fasta = _write(self.tmp / "dups.fasta", ">a\nACDE\n>a\nACDE\n")
        strict = functools.partial(search._fasta_preflight, strict=True)
        with patch("foldmatch.cli.search._fasta_preflight", strict), _esm3_spy() as esm3:
            with self.assertRaises(ValueError) as ctx:
                self._build(self.tmp / "db", fasta=fasta)
        self.assertIn("repeat an earlier id", str(ctx.exception))
        esm3.assert_not_called()
        self.assertEqual(list(self.tmp.glob("db*")), [])

    def test_plain_build_warns_and_keeps_todays_behaviour(self):
        from foldmatch.search.faiss_database import FaissEmbeddingDatabase
        from foldmatch.search.sequence_store import SequenceStore
        fasta = _write(self.tmp / "problems.fasta", _PROBLEM_FASTA)
        with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()), \
                self.assertLogs("foldmatch.utils.sequence", level="WARNING") as logs:
            self._build(self.tmp / "db", fasta=fasta)
        self.assertEqual(len([r for r in logs.records if r.levelname == "WARNING"]), 3)
        pf = preflight_fasta(fasta)
        db = FaissEmbeddingDatabase(self.tmp, "db")
        db.load_database()
        self.assertEqual(len(db.chain_ids), pf.entries_kept)
        store = SequenceStore(self.tmp, "db")
        self.assertEqual(store.meta(), {"entry_count": pf.unique_entries,
                                        "total_residues": pf.unique_residues,
                                        "duplicates_ignored": pf.duplicates})
        # First occurrence kept, normalised; '<' and non-ASCII kept as they are.
        self.assertEqual(store.fetch(["a", "b", "c", "d"]),
                         {"a": "ACDE", "b": "MK<", "c": "ACDÉ", "d": "A" * 10})

    def test_one_scan_feeds_the_preflight_and_the_dataset(self):
        import foldmatch.utils.sequence as sequence
        scans = []
        real_scan = sequence._scan_records

        def counting_scan(path):
            scans.append(path)
            return real_scan(path)

        with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()), \
                patch("foldmatch.utils.sequence._scan_records", counting_scan):
            self._build(self.tmp / "db")
        self.assertEqual(len(scans), 1)

    def test_store_meta_must_equal_the_preflight(self):
        fasta = _write(self.tmp / "grows.fasta", ">a\nACDE\n>b\nFGHIK\n")

        def append_record():
            # The FASTA changes after the pre-flight: the store would no longer match the index.
            with open(fasta, "a") as f:
                f.write(">c\nLMNPQ\n")

        with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict(append_record)):
            with self.assertRaises(ValueError) as ctx:
                self._build(self.tmp / "db", fasta=fasta)
        self.assertIn("pre-flight", str(ctx.exception))
        # The whole database goes, not just the store, so the same build can run again.
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db.*")), [])
        self.assertIn("db.index, db.metadata, db.sequences", str(ctx.exception))
        with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
            self._build(self.tmp / "db", fasta=fasta, cleanup_embeddings=True)
        self.assertTrue((self.tmp / "db.index").exists())

    def test_nothing_to_build_is_refused_before_esm3(self):
        for text, min_res_n in (("", 0), (">a\nACD\n>b\nEF GH\n", 5)):
            with self.subTest(text=text, min_res_n=min_res_n), _esm3_spy() as esm3:
                fasta = _write(self.tmp / "nothing.fasta", text)
                with self.assertRaisesRegex(ValueError, "nothing to build"):
                    self._build(self.tmp / "db", fasta=fasta, min_res_n=min_res_n)
                esm3.assert_not_called()
                self.assertEqual(list(self.tmp.glob("db*")), [])

    def _store_problem(self, problem):
        """Build a database from the fixture, then give its sequence store ``problem``."""
        import sqlite3
        with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
            self._build(self.tmp / "db", cleanup_embeddings=True)
        store = self.tmp / "db.sequences"
        if problem == "missing":
            store.unlink()
            return
        conn = sqlite3.connect(store)
        try:
            if problem == "empty":
                conn.execute("DELETE FROM sequences")
                conn.execute("UPDATE meta SET value = 0 WHERE key IN ('entry_count', 'total_residues')")
            elif problem == "incomplete":
                conn.execute("DELETE FROM meta")
            conn.commit()
        finally:
            conn.close()

    def test_stage2_store_problems_are_refused_before_esm3(self):
        from foldmatch.cli.search import query_database_from_fasta
        cases = [  # (store problem, query options, expected message)
            ("empty", {}, "non-empty database"),
            ("incomplete", {}, "is incomplete"),
            ("incomplete", {"max_evalue": float("inf")}, "is incomplete"),
            ("missing", {"seq_identity": True}, "has no sequence store"),
        ]
        for problem, options, message in cases:
            with self.subTest(problem=problem, options=options):
                for path in self.tmp.glob("db*"):
                    path.unlink()
                self._store_problem(problem)
                with _esm3_spy() as esm3:
                    with self.assertRaisesRegex(ValueError, message):
                        query_database_from_fasta(
                            db_path=str(self.tmp / "db"), fasta_file=_FIXTURE_FASTA,
                            tmp_embedding_folder=self.tmp, output_file=str(self.tmp / "hits.tsv"),
                            accelerator=Accelerator.cpu, **options)
                esm3.assert_not_called()
        # Without Stage 2, or with an E-value filter switched off, an empty store is fine.
        for options in ({"seq_identity": False}, {"max_evalue": float("inf")}):
            with self.subTest(problem="empty", options=options):
                for path in self.tmp.glob("db*"):
                    path.unlink()
                self._store_problem("empty")
                with patch("foldmatch.search.embedding_computer.chain_predict", fake_chain_predict()):
                    query_database_from_fasta(
                        db_path=str(self.tmp / "db"), fasta_file=_FIXTURE_FASTA,
                        tmp_embedding_folder=self.tmp, output_file=str(self.tmp / "hits.tsv"),
                        threshold=None, accelerator=Accelerator.cpu, **options)

    def test_stage2_store_problems_are_refused_before_the_db_search(self):
        from foldmatch.cli.search import query_database_from_database
        for problem, message in (("empty", "non-empty database"), ("incomplete", "is incomplete")):
            with self.subTest(problem=problem):
                for path in self.tmp.glob("db*"):
                    path.unlink()
                self._store_problem(problem)
                with patch("foldmatch.search.embedding_database.EmbeddingDatabase.load",
                           side_effect=AssertionError("database loaded")):
                    with self.assertRaisesRegex(ValueError, message):
                        query_database_from_database(
                            query_db_path=str(self.tmp / "db"), subject_db_path=str(self.tmp / "db"),
                            output_file=str(self.tmp / "hits.tsv"))

    def test_output_check_shares_the_database_path_rules_without_faiss(self):
        from foldmatch.search import db_path, embedding_database
        self.assertIs(embedding_database._parse_db_path, db_path.parse_db_path)
        child = subprocess.run(
            [sys.executable, "-c",
             "import sys\n"
             "from foldmatch.cli import search\n"
             "search._fasta_preflight(sys.argv[1], 0, output_db=sys.argv[2])\n"
             "print(sorted(m for m in sys.modules if m.startswith(('faiss', 'torch', 'esm'))))",
             str(_FIXTURE_FASTA), str(self.tmp / "db")],
            env=dict(os.environ, PYTHONPATH=_SRC), capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
