import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from foldmatch.search.sequence_store import META_KEYS, SequenceStore

_TEST_PATH = os.path.dirname(__file__)


def _write_fasta(path: Path, records):
    with open(path, "w") as f:
        for name, sequence in records:
            f.write(f">{name}\n{sequence}\n")


def _scan(store: SequenceStore):
    conn = sqlite3.connect(str(store.path))
    try:
        return conn.execute("SELECT COUNT(*), COALESCE(SUM(length), 0) FROM sequences").fetchone()
    finally:
        conn.close()


class SequenceStoreMetaTest(unittest.TestCase):
    """The store records entry_count, total_residues and duplicates_ignored at build
    time, and old stores without the meta table still answer from the rows."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _store(self, records, min_res_n=0, name="db"):
        fasta = self.tmp / f"{name}.fasta"
        _write_fasta(fasta, records)
        store = SequenceStore(self.tmp, name)
        stored = store.create(fasta, min_res_n=min_res_n)
        return store, stored

    def test_meta_counts_stored_entries_residues_and_duplicates(self):
        records = [
            ("a", "ACDEFGHIK"),     # 9
            ("b", "LMNPQ"),         # 5
            ("a", "WWWWWWWWWWWW"),  # duplicate id: the first occurrence is kept
            ("c", "RST"),           # 3, below min_res_n
            ("d", "VWYACDEFGH"),    # 10
            ("b", "KKKKKK"),        # duplicate id
        ]
        store, stored = self._store(records, min_res_n=4)
        meta = store.meta()
        self.assertEqual(meta, {"entry_count": 3, "total_residues": 24, "duplicates_ignored": 2})
        self.assertEqual(stored, 3)
        self.assertEqual(tuple(meta), META_KEYS)
        for key, value in meta.items():
            self.assertIsInstance(value, int, key)
        # The recorded totals equal a scan over the stored rows.
        self.assertEqual(_scan(store), (meta["entry_count"], meta["total_residues"]))
        self.assertEqual(store.count(), 3)
        self.assertEqual(store.total_residues(), 24)
        self.assertEqual(store.fetch(["a", "b"]), {"a": "ACDEFGHIK", "b": "LMNPQ"})

    def test_fixture_fasta(self):
        store = SequenceStore(self.tmp, "fixture")
        store.create(Path(_TEST_PATH) / "resources" / "fasta" / "test_sequences.fasta")
        self.assertEqual(store.meta(), {"entry_count": 2, "total_residues": 303, "duplicates_ignored": 0})

    def test_empty_store(self):
        store, stored = self._store([("short", "AC")], min_res_n=10)
        self.assertEqual(stored, 0)
        self.assertEqual(store.meta(), {"entry_count": 0, "total_residues": 0, "duplicates_ignored": 0})
        self.assertEqual(store.total_residues(), 0)

    def test_readers_use_the_meta_table(self):
        store, _ = self._store([("a", "ACDE"), ("b", "FGHIK")])
        conn = sqlite3.connect(str(store.path))
        try:
            conn.execute("UPDATE meta SET value = 1000 WHERE key = 'total_residues'")
            conn.execute("UPDATE meta SET value = 7 WHERE key = 'entry_count'")
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(store.total_residues(), 1000)
        self.assertEqual(store.count(), 7)

    def test_old_store_without_meta_falls_back_to_the_rows(self):
        # The layout written before the meta table existed.
        store = SequenceStore(self.tmp, "old")
        conn = sqlite3.connect(str(store.path))
        try:
            conn.execute(
                "CREATE TABLE sequences ("
                "id TEXT PRIMARY KEY, sequence TEXT NOT NULL, length INTEGER NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO sequences (id, sequence, length) VALUES (?, ?, ?)",
                [("a", "ACDE", 4), ("b", "FGHIKLM", 7)],
            )
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(store.meta(), {"entry_count": 2, "total_residues": 11, "duplicates_ignored": None})
        self.assertEqual(store.count(), 2)
        self.assertEqual(store.total_residues(), 11)

    def test_force_rebuild_rewrites_the_meta(self):
        store, _ = self._store([("a", "ACDE"), ("a", "ACDE")])
        self.assertEqual(store.meta()["duplicates_ignored"], 1)
        fasta = self.tmp / "rebuild.fasta"
        _write_fasta(fasta, [("x", "ACDEFG"), ("y", "HIK")])
        store.create(fasta, force=True)
        self.assertEqual(store.meta(), {"entry_count": 2, "total_residues": 9, "duplicates_ignored": 0})

    def test_interrupted_build_is_refused(self):
        """A build that stopped after some batches leaves rows but no totals; that is not a store."""
        from foldmatch.search import sequence_store
        fasta = self.tmp / "ten.fasta"
        _write_fasta(fasta, [(f"s{i}", "ACDEFGHIKL") for i in range(10)])
        real_iter_fasta = sequence_store.iter_fasta

        def killed_after_six(path):
            for i, record in enumerate(real_iter_fasta(path)):
                if i == 6:
                    raise KeyboardInterrupt("build killed")
                yield record

        store = SequenceStore(self.tmp, "interrupted")
        with mock.patch.object(sequence_store, "_INSERT_BATCH", 3), \
                mock.patch.object(sequence_store, "iter_fasta", killed_after_six):
            with self.assertRaises(KeyboardInterrupt):
                store.create(fasta)
        # Two batches of three were committed before the kill.
        self.assertTrue(store.exists())
        self.assertEqual(_scan(store), (6, 60))
        for method in (store.meta, store.count, store.total_residues):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(ValueError, "is incomplete"):
                    method()

    def test_missing_store_raises(self):
        store = SequenceStore(self.tmp, "absent")
        for method in (store.meta, store.count, store.total_residues):
            with self.subTest(method=method.__name__):
                with self.assertRaises(FileNotFoundError):
                    method()


if __name__ == "__main__":
    unittest.main()
