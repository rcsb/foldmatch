"""P5.5: the residue-code store ``P.rescodes.{bin,idx,json}`` (plan section 8 P5.5, Appendix B).

No ESM3: the codes are the F2c reference codes of ``test_sequences.fasta`` or random bytes. Most tests pass the ESM3
revision and weights sha256 explicitly, so only the tests of the ``_INSTALLED`` defaults import
``foldmatch.utils.esm.loaders`` (torch, never the weights).
"""
import dataclasses
import hashlib
import json
import os
import random
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa

from foldmatch.search import residue_code_store as rcs
from foldmatch.search import scoring_system as ss
from foldmatch.search.residue_code_store import (
    MANIFEST_KEYS, CodeStoreValidationError, IncompatibleCodeStoreError, ResidueCodeStore, ResidueCodeStoreError,
    entries_digest,
)
from foldmatch.search.sequence_store import SequenceStore
from foldmatch.utils.sequence import STANDARD_AMINO_ACIDS, preflight_fasta, seq_sha1

_TEST_PATH = Path(__file__).parent
_SRC = str(_TEST_PATH.parent / "src")
_F2C = _TEST_PATH / "resources" / "esm_codes" / "fasta_rows_codes_v1.npz"
_FIXTURE_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"

SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)
ESM_SHA = SYSTEM.esm_weights_sha256
ESM_REV = "636acb420daf674df37ab0220f56061a5de2f169"   # passed explicitly: these tests never need torch
CODE_BYTES = SYSTEM.code_bytes


def _write_fasta(path: Path, records):
    with open(path, "w") as f:
        for name, sequence in records:
            f.write(f">{name}\n{sequence}\n")
    return path


def _records(n, rng, min_len=3, max_len=40, prefix="P"):
    return [(f"{prefix}{i:05d}",
             "".join(rng.choice(STANDARD_AMINO_ACIDS) for _ in range(rng.randint(min_len, max_len))))
            for i in range(n)]


def _entries(records, seed=0):
    """``[(id, codes uint8 [L, 8], seq_sha1)]`` with random codes."""
    gen = np.random.default_rng(seed)
    return [(name, gen.integers(0, 256, size=(len(seq), CODE_BYTES), dtype=np.uint8), seq_sha1(seq))
            for name, seq in records]


def _batches(entries, size=None):
    size = size or max(len(entries), 1)
    out = []
    for start in range(0, len(entries), size):
        chunk = entries[start:start + size]
        out.append(([e[0] for e in chunk], [e[1] for e in chunk], [e[2] for e in chunk]))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _store_files(store):
    return {p.name: _sha256(p) for p in (store.bin_path, store.idx_path, store.manifest_path) if p.exists()}


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _database(self, records, name="db"):
        """A FASTA, its sequence store and its pre-flight (what a build has before the code store)."""
        fasta = _write_fasta(self.tmp / f"{name}.fasta", records)
        seq = SequenceStore(self.tmp, name)
        seq.create(fasta)
        return seq, preflight_fasta(fasta)

    def _create(self, entries, name="db", size=None, **kwargs):
        store = ResidueCodeStore(self.tmp, name)
        kwargs.setdefault("esm_revision", ESM_REV)
        summary = store.create(_batches(entries, size), kwargs.pop("system", SYSTEM), **kwargs)
        return store, summary

    def _build(self, records, entries=None, name="db", size=None, faiss=None, **kwargs):
        seq, preflight = self._database(records, name)
        entries = _entries(records) if entries is None else entries
        store, _ = self._create(entries, name, size, **kwargs)
        manifest = store.validate(seq, len(records) if faiss is None else faiss, preflight=preflight)
        return store, seq, manifest

    def _check(self, store, **kwargs):
        kwargs.setdefault("sequence_store", store.sequence_store())
        kwargs.setdefault("esm_weights_sha256", ESM_SHA)
        return store.check_compatible(kwargs.pop("system", SYSTEM), **kwargs)

    def _edit_manifest(self, store, edit):
        manifest = json.loads(store.manifest_path.read_text())
        edit(manifest)
        store.manifest_path.write_text(json.dumps(manifest))

    def assertNoStoreFiles(self, store):
        left = sorted(p.name for p in self.tmp.glob(f"{store.index_name}.rescodes*"))
        self.assertEqual(left, [])
        self.assertFalse(store.exists())
        self.assertFalse(store.is_partial())
        self.assertFalse(store.has_pending_build())

    def assertStoreUnchanged(self, store, before):
        """The complete store is exactly as it was (``before`` = :func:`_store_files`), usable, with no pending
        files next to it."""
        self.assertEqual(_store_files(store), before)
        self.assertEqual(sorted(p.name for p in self.tmp.glob(f"{store.index_name}.rescodes*")),
                         [f"{store.index_name}.rescodes.{x}" for x in ("bin", "idx", "json")])
        self.assertFalse(store.has_pending_build())
        self._check(store)


# =====================================================================================================================
class RoundTripTest(_StoreCase):
    def test_f2c_codes_round_trip(self):
        """The F2c reference codes of test_sequences.fasta: manifest entries 2 / total_residues 303 (Appendix B),
        the memmap equals the codes, lookup gives each entry's slice."""
        f2c = np.load(_F2C, allow_pickle=False)
        ids, offsets, codes = [str(i) for i in f2c["ids"]], f2c["offsets"], f2c["codes"]
        seq, preflight = SequenceStore(self.tmp, "db"), preflight_fasta(_FIXTURE_FASTA)
        seq.create(_FIXTURE_FASTA)
        sequences = seq.fetch(ids)
        self.assertEqual([sequences[i] for i in ids], [str(s) for s in f2c["sequences"]])
        store = ResidueCodeStore(self.tmp, "db")
        batches = [([i], [codes[offsets[k]:offsets[k + 1]]], [seq_sha1(sequences[i])]) for k, i in enumerate(ids)]
        build = {"run_id": "run-1", "accelerator": "cpu", "devices": 1, "esm_precision": "fp32",
                 "encoder_precision": "fp32-no-tf32"}
        summary = store.create(iter(batches), SYSTEM, build=build, esm_revision=ESM_REV)
        self.assertEqual((summary["entries"], summary["total_residues"]), (2, 303))
        manifest = store.validate(seq, 2, preflight=preflight)

        self.assertEqual(tuple(sorted(manifest)), tuple(sorted(MANIFEST_KEYS)))
        self.assertEqual(json.loads(store.manifest_path.read_text()), manifest)
        self.assertEqual((manifest["manifest_version"], manifest["kind"]), (1, "foldmatch-residue-codes"))
        self.assertEqual((manifest["entries"], manifest["total_residues"], manifest["min_res_n"]), (2, 303, 0))
        self.assertEqual(manifest["system_name"], "esm-codes-v1")
        self.assertEqual(manifest["system"], SYSTEM.to_dict())
        self.assertEqual(ss.EsmCodesSystem.from_dict(manifest["system"]), SYSTEM)
        self.assertEqual((manifest["system_digest"], manifest["encoding_digest"]),
                         (SYSTEM.digest(), SYSTEM.encoding_digest()))
        self.assertEqual((manifest["code_bytes"], manifest["bitorder"]), (8, "big"))
        self.assertEqual(manifest["esm"], {"repo": "rcsb/rcsb-esm", "revision": ESM_REV, "weights_sha256": ESM_SHA})
        self.assertEqual(manifest["bundle"], {"repo": SYSTEM.bundle_repo, "revision": SYSTEM.bundle_revision,
                                              "files": dict(SYSTEM.bundle_files)})
        self.assertEqual(manifest["bin_sha256"], hashlib.sha256(codes.tobytes()).hexdigest())
        self.assertEqual(manifest["entries_digest"], entries_digest([(i, seq_sha1(sequences[i])) for i in ids]))
        self.assertEqual((manifest["nonstandard_entries"], manifest["input_normalisation"]),
                         (0, "ascii_upper_nows_v1"))
        self.assertEqual(manifest["cross_checks"], {
            "preflight_entries": 2, "preflight_residues": 303, "sequence_store_entries": 2,
            "sequence_store_residues": 303, "faiss_entries": 2, "seq_sha1_mismatches": 0, "length_mismatches": 0})
        for key, value in build.items():
            self.assertEqual(manifest["build"][key], value)
        self.assertEqual(manifest["build"]["platform"].count("-"), 1)

        self.assertTrue(store.exists())
        self.assertFalse(store.is_partial())
        self.assertEqual(store.bin_path.stat().st_size, 8 * 303)
        memmap = store.open_memmap()
        self.assertIsInstance(memmap, np.memmap)
        self.assertEqual((memmap.shape, memmap.dtype), ((303, 8), np.uint8))
        np.testing.assert_array_equal(memmap, codes)
        self.assertFalse(memmap.flags.writeable)
        found = store.lookup(ids)
        for k, i in enumerate(ids):
            entry = found[i]
            self.assertEqual((entry.offset, entry.length), (offsets[k], offsets[k + 1] - offsets[k]))
            self.assertEqual(entry.seq_sha1, seq_sha1(sequences[i]))
            np.testing.assert_array_equal(memmap[entry.offset:entry.offset + entry.length],
                                          codes[offsets[k]:offsets[k + 1]])
        self.assertEqual(self._check(store), manifest)
        store.check_faiss_count(2)
        with self.assertRaisesRegex(IncompatibleCodeStoreError, "2 entries, the FAISS index 3"):
            store.check_faiss_count(3)

    def test_batch_forms_and_sizes_give_the_same_store(self):
        """Tuples of arrays or bytes, pyarrow batches and tables, any batch size and transactions of any size (here
        5 rows) write the same files."""
        records = _records(23, random.Random(1))
        entries = _entries(records)
        reference = None

        def arrow(chunk, table=False):
            data = {"id": pa.array([e[0] for e in chunk], pa.string()),
                    "residue_codes": pa.array([e[1].tobytes() for e in chunk], pa.large_binary()),
                    "seq_sha1": pa.array([e[2] for e in chunk], pa.binary(20))}
            return pa.table(data) if table else pa.record_batch(data)

        forms = {
            "arrays": lambda chunk: ([e[0] for e in chunk], [e[1] for e in chunk], [e[2] for e in chunk]),
            "bytes": lambda chunk: ([e[0] for e in chunk], [e[1].tobytes() for e in chunk],
                                    [bytearray(e[2]) for e in chunk]),
            "flat-arrays": lambda chunk: ([e[0] for e in chunk], [e[1].reshape(-1) for e in chunk],
                                          [memoryview(e[2]) for e in chunk]),
            "record-batch": arrow,
            "table": lambda chunk: arrow(chunk, table=True),
        }
        for (form, make), size in [(f, s) for f in forms.items() for s in (1, 7, 23)]:
            with self.subTest(form=form, size=size), mock.patch.object(rcs, "_INSERT_BATCH", 5):
                name = f"db_{form}_{size}"
                seq, preflight = self._database(records, name)
                store = ResidueCodeStore(self.tmp, name)
                batches = [make(entries[s:s + size]) for s in range(0, len(entries), size)]
                store.create(batches, SYSTEM, esm_revision=ESM_REV)
                manifest = store.validate(seq, len(records), preflight=preflight)
                conn = sqlite3.connect(store.idx_path)
                rows = conn.execute("SELECT id, offset, length, seq_sha1 FROM entries ORDER BY id").fetchall()
                conn.close()
                got = (manifest["bin_sha256"], manifest["entries_digest"], manifest["total_residues"], rows)
                if reference is None:
                    reference = got
                self.assertEqual(got, reference)

    def test_entry_order_changes_bin_sha256_not_entries_digest(self):
        records = _records(30, random.Random(2))
        entries = _entries(records)
        store, _, manifest = self._build(records, entries, name="a")
        shuffled = list(entries)
        random.Random(3).shuffle(shuffled)
        other, _, manifest2 = self._build(records, shuffled, name="b")
        self.assertEqual(manifest2["entries_digest"], manifest["entries_digest"])
        self.assertNotEqual(manifest2["bin_sha256"], manifest["bin_sha256"])
        memmap, memmap2 = store.open_memmap(), other.open_memmap()
        found, found2 = store.lookup([e[0] for e in entries]), other.lookup([e[0] for e in entries])
        for name, codes, _ in entries:
            a, b = found[name], found2[name]
            np.testing.assert_array_equal(memmap[a.offset:a.offset + a.length], codes)
            np.testing.assert_array_equal(memmap2[b.offset:b.offset + b.length], codes)


# =====================================================================================================================
class DigestTest(_StoreCase):
    def test_bin_sha256_is_the_file_hash_computed_while_writing(self):
        records = _records(40, random.Random(4))
        entries = _entries(records)
        with mock.patch.object(rcs, "_INSERT_BATCH", 7):
            store, summary = self._create(entries, size=6)
        self.assertEqual(summary["bin_sha256"], _sha256(store.tmp_bin_path))
        self.assertEqual(summary["bin_sha256"],
                         hashlib.sha256(b"".join(e[1].tobytes() for e in entries)).hexdigest())

    def test_entries_digest_definition(self):
        """sha256(tag, then per entry in UTF-8 byte order of the ids: uint32 LE id length, id, 20-byte seq_sha1).
        SQLite's ORDER BY id (BINARY) is that order, ASCII case and non-ASCII ids included."""
        names = ["b", "a", "Z", "a_1", "été", "A", "a1", "中"]
        records = [(n, "ACDE"[: 1 + k % 4]) for k, n in enumerate(names)]
        entries = _entries(records)
        store, summary = self._create(entries)
        h = hashlib.sha256(b"foldmatch-residue-codes-entries/1\n")
        for name, _, sha1 in sorted(entries, key=lambda e: e[0].encode("utf-8")):
            h.update(struct.pack("<I", len(name.encode("utf-8"))) + name.encode("utf-8") + sha1)
        self.assertEqual(summary["entries_digest"], h.hexdigest())
        self.assertEqual(entries_digest([(e[0], e[2]) for e in entries]), h.hexdigest())
        changed = [(e[0], e[2]) for e in entries]
        changed[3] = (changed[3][0], hashlib.sha1(b"other").digest())
        self.assertNotEqual(entries_digest(changed), h.hexdigest())
        renamed = [(e[0] + "x" if k == 0 else e[0], e[2]) for k, e in enumerate(entries)]
        self.assertNotEqual(entries_digest(renamed), h.hexdigest())


# =====================================================================================================================
class LookupTest(_StoreCase):
    def test_lookup_of_more_than_900_ids(self):
        """2,000 ids in one call, with SQLite's host-parameter limit lowered to 900: lookup chunks its IN lists."""
        records = _records(2000, random.Random(5), min_len=1, max_len=4)
        entries = _entries(records)
        store, _, _ = self._build(records, entries, size=333)
        real_connect = sqlite3.connect

        def limited(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 900)
            return conn

        ids = [e[0] for e in entries]
        with mock.patch.object(rcs.sqlite3, "connect", side_effect=limited) as connect:
            found = store.lookup(list(reversed(ids)) + ids[:50])
            conn = limited(str(store.idx_path))
            with self.assertRaises(sqlite3.OperationalError):  # the limit is real
                conn.execute(f"SELECT id FROM entries WHERE id IN ({','.join('?' * 901)})", ids[:901])
            conn.close()
        self.assertGreaterEqual(connect.call_count, 1)
        self.assertEqual(len(found), 2000)
        memmap = store.open_memmap()
        offset = 0
        for name, codes, sha1 in entries:
            entry = found[name]
            self.assertEqual((entry.offset, entry.length, entry.seq_sha1), (offset, len(codes), sha1))
            np.testing.assert_array_equal(memmap[entry.offset:entry.offset + entry.length], codes)
            offset += len(codes)

    def test_missing_id_raises(self):
        records = _records(5, random.Random(6))
        store, _, _ = self._build(records)
        self.assertEqual(store.lookup([]), {})
        with self.assertRaises(ResidueCodeStoreError) as ctx:
            store.lookup([records[0][0], "nope", records[1][0], "absent", "nope"])
        self.assertIn("2 id(s) have no residue codes", str(ctx.exception))
        self.assertIn("nope, absent", str(ctx.exception))
        with self.assertRaises(ResidueCodeStoreError) as ctx:
            store.lookup([f"m{i:02d}" for i in range(15)])
        self.assertIn("15 id(s)", str(ctx.exception))
        self.assertIn("m09)", str(ctx.exception))
        self.assertNotIn("m10", str(ctx.exception))

    def test_readers_refuse_a_store_without_manifest(self):
        records = _records(4, random.Random(7))
        self._database(records)
        store, _ = self._create(_entries(records))
        self.assertTrue(store.is_partial())
        for read in (lambda: store.lookup([records[0][0]]), store.open_memmap, store.manifest,
                     lambda: self._check(store), lambda: store.check_faiss_count(4)):
            with self.subTest(read=read), self.assertRaisesRegex(ResidueCodeStoreError, "incomplete"):
                read()
        none = ResidueCodeStore(self.tmp, "other")
        for read in (lambda: none.lookup(["x"]), none.open_memmap, lambda: self._check(none)):
            with self.subTest(read=read), self.assertRaisesRegex(FileNotFoundError, "without --residue-codes"):
                read()


# =====================================================================================================================
class ManifestProtocolTest(_StoreCase):
    def _traced_build(self, store, seq, preflight, entries, **create_kwargs):
        """create() + validate() with fsync (by inode), unlink and replace recorded in order."""
        events = []
        real_fsync, real_replace, real_unlink = os.fsync, os.replace, os.unlink

        def fsync(fd):
            events.append(("fsync", os.fstat(fd).st_ino))
            return real_fsync(fd)

        def replace(src, dst):
            if Path(dst) == store.manifest_path:
                self.assertEqual(json.loads(Path(src).read_text())["entries"], len(entries))  # whole before rename
            events.append(("replace", Path(src), Path(dst), os.stat(src).st_ino))
            return real_replace(src, dst)

        def unlink(path):
            events.append(("unlink", Path(path)))
            return real_unlink(path)

        with mock.patch.object(rcs.os, "fsync", side_effect=fsync), \
                mock.patch.object(rcs.os, "replace", side_effect=replace), \
                mock.patch.object(rcs.os, "unlink", side_effect=unlink):
            store.create(_batches(entries), SYSTEM, esm_revision=ESM_REV, **create_kwargs)
            self.assertTrue(store.tmp_bin_path.exists() and store.tmp_idx_path.exists())  # written aside
            self.assertTrue(store.has_pending_build())
            events.append(("created",))
            store.validate(seq, len(entries), preflight=preflight)
        return events

    def test_manifest_is_written_last_through_tmp_fsync_and_replace(self):
        records = _records(8, random.Random(8))
        seq, preflight = self._database(records)
        store = ResidueCodeStore(self.tmp, "db")
        events = self._traced_build(store, seq, preflight, _entries(records))
        replaces = [e[1:3] for e in events if e[0] == "replace"]
        self.assertEqual(replaces, [(store.tmp_bin_path, store.bin_path), (store.tmp_idx_path, store.idx_path),
                                    (store.tmp_manifest_path, store.manifest_path)])
        inode = {p: p.stat().st_ino for p in (store.bin_path, store.idx_path, store.manifest_path, self.tmp)}
        first = next(k for k, e in enumerate(events) if e[0] == "replace")
        last = next(k for k, e in enumerate(events) if e[0] == "replace" and e[2] == store.manifest_path)
        self.assertEqual(last, max(k for k, e in enumerate(events) if e[0] == "replace"))  # the manifest goes last
        for path in (store.bin_path, store.idx_path, store.manifest_path):  # all three whole on disk first
            self.assertLess(events.index(("fsync", inode[path])), first, path.name)
        self.assertIn(("fsync", inode[self.tmp]), events[first:last])  # the .bin/.idx renames are durable first
        self.assertIn(("fsync", inode[self.tmp]), events[last:])  # and so is the manifest's
        self.assertLess(events.index(("fsync", inode[store.bin_path])), events.index(("created",)))
        self.assertFalse(store.has_pending_build())
        self.assertTrue(store.exists())
        self.assertFalse(store.is_partial())

    def test_a_replacement_removes_the_old_manifest_before_its_files(self):
        """With force, the old store's manifest goes before the new files are renamed over its own, so no manifest
        ever describes a mix of the two."""
        records = _records(8, random.Random(80))
        store, seq, _ = self._build(records)
        preflight = preflight_fasta(self.tmp / "db.fasta")
        events = self._traced_build(store, seq, preflight, _entries(records, seed=9), force=True)
        created = events.index(("created",))
        self.assertNotIn(("unlink", store.manifest_path), events[:created])  # create() left the old store whole
        unlinked = events.index(("unlink", store.manifest_path))
        replaces = [k for k, e in enumerate(events) if e[0] == "replace"]
        self.assertLess(unlinked, replaces[0])
        self.assertEqual(store.manifest()["bin_sha256"], hashlib.sha256(
            b"".join(e[1].tobytes() for e in _entries(records, seed=9))).hexdigest())
        self._check(store)

    def test_failed_rename_leaves_no_manifest(self):
        records = _records(6, random.Random(9))
        seq, preflight = self._database(records)
        store, _ = self._create(_entries(records))
        with mock.patch.object(rcs.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                store.validate(seq, 6, preflight=preflight)
        self.assertNoStoreFiles(store)

    def test_failed_manifest_write_keeps_the_store_it_replaces(self):
        """No space for the new manifest: the pending build is discarded before the old store is touched."""
        records = _records(6, random.Random(90))
        store, seq, _ = self._build(records)
        before = _store_files(store)
        store.create(_batches(_entries(records, seed=3)), SYSTEM, force=True, esm_revision=ESM_REV)
        with mock.patch.object(rcs, "_write_synced", side_effect=OSError("No space left on device")):
            with self.assertRaisesRegex(OSError, "No space"):
                store.validate(seq, 6, preflight=None)
        self.assertStoreUnchanged(store, before)

    def test_interrupted_create_leaves_nothing(self):
        records = _records(10, random.Random(10))
        entries = _entries(records)

        def batches():
            yield from _batches(entries[:5])
            raise RuntimeError("shard unreadable")

        store = ResidueCodeStore(self.tmp, "db")
        with self.assertRaisesRegex(RuntimeError, "shard unreadable"):
            store.create(batches(), SYSTEM, esm_revision=ESM_REV)
        self.assertNoStoreFiles(store)

    def test_partial_store_is_cleared_by_create(self):
        records = _records(6, random.Random(11))
        seq, preflight = self._database(records)
        store, _ = self._create(_entries(records, seed=1))  # validate never ran: a crash after create()
        self.assertTrue(store.is_partial())
        self.assertFalse(store.exists())
        with self.assertLogs(rcs.logger, "WARNING") as logs:
            store, _ = self._create(_entries(records, seed=2))
        self.assertIn("partial", logs.output[0])
        manifest = store.validate(seq, 6, preflight=preflight)
        self.assertEqual(manifest["bin_sha256"], hashlib.sha256(
            b"".join(e[1].tobytes() for e in _entries(records, seed=2))).hexdigest())
        # The remains of an interrupted swap (store files without a manifest) are partial too.
        store.manifest_path.unlink()
        self.assertTrue(store.is_partial())
        with self.assertLogs(rcs.logger, "WARNING"):
            store, _ = self._create(_entries(records, seed=3))
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db.rescodes*")),
                         ["db.rescodes.bin.tmp", "db.rescodes.idx.tmp"])

    def test_clobber_refused_and_store_untouched(self):
        records = _records(6, random.Random(12))
        store, _, _ = self._build(records)
        before = _store_files(store)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self._create(_entries(records, seed=5))
        self.assertEqual(_store_files(store), before)
        self._check(store)

    def test_force_keeps_the_old_store_until_the_new_one_is_validated(self):
        """A forced rebuild writes aside: while it runs, after a crash in it and after a failed validation, the old
        store is whole and usable; only a validated rebuild replaces it."""
        records = _records(6, random.Random(13))
        store, seq, old = self._build(records)
        before = _store_files(store)
        seen = []

        def batches():
            seen.append((store.manifest_path.exists(), store.check_compatible(SYSTEM, esm_weights_sha256=ESM_SHA)))
            raise RuntimeError("crash during the rebuild")
            yield  # pragma: no cover

        with self.assertRaisesRegex(RuntimeError, "crash"):
            store.create(batches(), SYSTEM, force=True, esm_revision=ESM_REV)
        self.assertEqual(seen, [(True, old)])  # readable while the rebuild ran
        self.assertStoreUnchanged(store, before)

        other = _records(6, random.Random(130), prefix="Q")  # same count, other ids: (a), (b) fail
        cases = {
            "other ids": _entries(other),
            "a missing entry": _entries(records)[1:],
            "an empty payload": [(n, b"" if k == 2 else c, h) for k, (n, c, h) in enumerate(_entries(records))],
            "another seq_sha1": [(n, c, seq_sha1("W") if k == 0 else h)
                                 for k, (n, c, h) in enumerate(_entries(records))],
        }
        for case, entries in cases.items():
            with self.subTest(case=case):
                store.create(_batches(entries), SYSTEM, force=True, esm_revision=ESM_REV)
                with self.assertRaises(CodeStoreValidationError) as ctx:
                    store.validate(seq, 6, preflight=None)
                self.assertIn("were discarded", str(ctx.exception))
                self.assertIn(f"existing residue-code store {store.manifest_path} and the database are unchanged",
                              str(ctx.exception))
                self.assertStoreUnchanged(store, before)
        with self.assertRaises(CodeStoreValidationError):  # the FAISS count, check (e)
            store.create(_batches(_entries(records, seed=6)), SYSTEM, force=True, esm_revision=ESM_REV)
            store.validate(seq, 5, preflight=None)
        self.assertStoreUnchanged(store, before)

        store, _ = self._create(_entries(records, seed=6), force=True)
        self.assertEqual(_store_files(store), before)  # still the old files until validate() swaps
        manifest = store.validate(seq, 6, preflight=None)
        self.assertEqual(manifest["cross_checks"]["preflight_entries"], None)
        self.assertNotEqual(manifest["bin_sha256"], old["bin_sha256"])
        self.assertEqual(manifest["bin_sha256"], _sha256(store.bin_path))
        self.assertFalse(store.has_pending_build())
        self._check(store)

    def test_leftover_pending_files_beside_a_complete_store(self):
        """An interrupted forced rebuild may leave pending files next to the store: readers ignore them, a plain
        create() still refuses the store, and the next forced create() clears them."""
        records = _records(5, random.Random(131))
        store, seq, _ = self._build(records)
        before = _store_files(store)
        for path in (store.tmp_bin_path, store.tmp_idx_path, store.tmp_manifest_path):
            path.write_bytes(b"leftover")
        self.assertTrue(store.exists())
        self.assertFalse(store.is_partial())
        self.assertTrue(store.has_pending_build())
        self._check(store)
        self.assertEqual(len(store.lookup([records[0][0]])), 1)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            self._create(_entries(records))
        self.assertTrue(store.tmp_bin_path.exists())
        with self.assertLogs(rcs.logger, "WARNING") as logs:
            self._create(_entries(records, seed=4), force=True)
        self.assertIn("unvalidated", " ".join(logs.output))
        self.assertEqual(_store_files(store), before)
        store.validate(seq, 5, preflight=None)
        self.assertFalse(store.has_pending_build())

    def test_validate_needs_a_pending_build(self):
        records = _records(4, random.Random(132))
        store, seq, _ = self._build(records)
        before = _store_files(store)
        with self.assertRaisesRegex(ResidueCodeStoreError, "No pending residue-code build"):
            store.validate(seq, 4, preflight=None)
        self.assertStoreUnchanged(store, before)

    def test_validate_refuses_an_incomplete_index(self):
        records = _records(6, random.Random(14))
        seq, preflight = self._database(records)
        store, _ = self._create(_entries(records))
        conn = sqlite3.connect(store.tmp_idx_path)
        conn.execute("DELETE FROM meta WHERE key = 'entries_digest'")
        conn.commit()
        conn.close()
        with self.assertRaisesRegex(ResidueCodeStoreError, "incomplete.*entries_digest"):
            store.validate(seq, 6, preflight=preflight)
        self.assertNoStoreFiles(store)
        self.assertEqual(seq.meta()["entry_count"], 6)

    def test_remove_deletes_every_file_manifest_first(self):
        records = _records(4, random.Random(15))
        store, _, _ = self._build(records)
        store.tmp_manifest_path.write_text("{}")
        unlinked = []
        real_unlink = os.unlink
        with mock.patch.object(rcs.os, "unlink", side_effect=lambda p: (unlinked.append(Path(p)), real_unlink(p))):
            removed = store.remove()
        self.assertEqual(unlinked[0], store.manifest_path)
        self.assertEqual(set(removed), {store.manifest_path, store.tmp_manifest_path, store.bin_path,
                                        store.idx_path})
        self.assertNoStoreFiles(store)
        self.assertEqual(store.remove(), [])
        self.assertTrue(store.sequence_store().exists())


# =====================================================================================================================
class EmptyDatabaseTest(_StoreCase):
    def test_empty_database_is_a_valid_empty_store(self):
        seq, preflight = self._database([])
        self.assertEqual(seq.meta()["entry_count"], 0)
        store = ResidueCodeStore(self.tmp, "db")
        summary = store.create([([], [], [])], SYSTEM, esm_revision=ESM_REV)
        self.assertEqual((summary["entries"], summary["total_residues"]), (0, 0))
        manifest = store.validate(seq, 0, preflight=preflight)
        self.assertEqual((manifest["entries"], manifest["total_residues"]), (0, 0))
        self.assertEqual(manifest["bin_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(manifest["entries_digest"],
                         hashlib.sha256(b"foldmatch-residue-codes-entries/1\n").hexdigest())
        self.assertEqual(store.bin_path.stat().st_size, 0)
        codes = store.open_memmap()
        self.assertNotIsInstance(codes, np.memmap)
        self.assertEqual((codes.shape, codes.dtype), ((0, 8), np.uint8))
        self.assertEqual(store.lookup([]), {})
        with self.assertRaisesRegex(ResidueCodeStoreError, "1 id"):
            store.lookup(["x"])
        self.assertEqual(self._check(store)["entries"], 0)

    def test_no_batches_at_all(self):
        seq, preflight = self._database([], name="none")
        store = ResidueCodeStore(self.tmp, "none")
        store.create(iter(()), SYSTEM, esm_revision=ESM_REV)
        self.assertEqual(store.validate(seq, 0, preflight=preflight)["entries"], 0)


# =====================================================================================================================
class CreateInputTest(_StoreCase):
    def test_duplicate_id_raises_and_leaves_nothing(self):
        records = _records(9, random.Random(16))
        entries = _entries(records)
        cases = {
            "same batch": [entries[:4] + [entries[2]]],
            "next batch": [entries[:4], [entries[1]] + entries[4:]],
            "next transaction": [entries[:4], entries[4:], [entries[0]]],
        }
        for case, batches in cases.items():
            with self.subTest(case=case), mock.patch.object(rcs, "_INSERT_BATCH", 3):
                store = ResidueCodeStore(self.tmp, "db")
                with self.assertRaisesRegex(ResidueCodeStoreError, "duplicate entry id") as ctx:
                    store.create([_batches(b)[0] for b in batches], SYSTEM, esm_revision=ESM_REV)
                duplicate = {"same batch": entries[2], "next batch": entries[1], "next transaction": entries[0]}
                self.assertIn(duplicate[case][0], str(ctx.exception))
                self.assertNoStoreFiles(store)

    def test_malformed_input_raises_and_leaves_nothing(self):
        codes = np.zeros((3, 8), dtype=np.uint8)
        sha1 = hashlib.sha1(b"ACD").digest()
        bad = {
            "7 code bytes": (["a"], [b"\x00" * 7], [sha1]),
            "uint16 codes": (["a"], [codes.astype(np.uint16)], [sha1]),
            "4-byte rows": (["a"], [np.zeros((3, 4), dtype=np.uint8)], [sha1]),
            "3-d codes": (["a"], [np.zeros((1, 3, 8), dtype=np.uint8)], [sha1]),
            "no codes": (["a"], [None], [sha1]),
            "19-byte sha1": (["a"], [codes], [sha1[:19]]),
            "hex sha1": (["a"], [codes], [sha1.hex()]),
            "int id": ([7], [codes], [sha1]),
            "ragged batch": (["a", "b"], [codes], [sha1, sha1]),
            "two columns": (["a"], [codes]),
            "arrow without seq_sha1": pa.record_batch({"id": pa.array(["a"]),
                                                      "residue_codes": pa.array([codes.tobytes()], pa.large_binary())}),
        }
        for case, batch in bad.items():
            with self.subTest(case=case):
                store = ResidueCodeStore(self.tmp, "db")
                with self.assertRaises(ResidueCodeStoreError):
                    store.create([(["ok"], [codes], [sha1]), batch], SYSTEM, esm_revision=ESM_REV)
                self.assertNoStoreFiles(store)

    def test_bad_arguments_are_refused_before_any_file_is_touched(self):
        records = _records(4, random.Random(17))
        store, _, _ = self._build(records)
        before = _store_files(store)
        for kwargs in ({"system": SYSTEM.to_dict()}, {"min_res_n": -1}, {"min_res_n": 1.5}, {"build": ["x"]},
                       {"build": {"run_id": float("nan")}}, {"esm_revision": 5}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                self._create(_entries(records), force=True, **kwargs)
            self.assertEqual(_store_files(store), before)
        self._check(store)

    def test_min_res_n_and_build_are_recorded(self):
        records = _records(4, random.Random(18), min_len=10)
        _, _, manifest = self._build(records, min_res_n=10, build={"run_id": "abc", "extra": [1, 2]})
        self.assertEqual(manifest["min_res_n"], 10)
        self.assertEqual((manifest["build"]["run_id"], manifest["build"]["extra"]), ("abc", [1, 2]))
        self.assertEqual(manifest["build"]["accelerator"], None)
        import foldmatch
        self.assertEqual(manifest["build"]["foldmatch_version"], foldmatch.__version__)

    def test_default_esm_revision_is_this_installations_pin(self):
        from foldmatch.utils.esm import loaders
        records = _records(3, random.Random(19))
        seq, preflight = self._database(records)
        store = ResidueCodeStore(self.tmp, "db")
        store.create(_batches(_entries(records)), SYSTEM)
        self.assertEqual(store.validate(seq, 3, preflight=preflight)["esm"]["revision"], loaders.ESM_REVISION)
        other = dataclasses.replace(SYSTEM, esm_weights_sha256="0" * 64)
        store.create(_batches(_entries(records)), other, force=True)
        self.assertIsNone(store.validate(seq, 3, preflight=preflight)["esm"]["revision"])


# =====================================================================================================================
class CheckCompatibleTest(_StoreCase):
    def setUp(self):
        super().setUp()
        self.records = _records(7, random.Random(20))
        self.store, self.seq, self.manifest = self._build(self.records)

    def assertRefused(self, pattern, **kwargs):
        with self.assertRaisesRegex(IncompatibleCodeStoreError, pattern) as ctx:
            self._check(self.store, **kwargs)
        return str(ctx.exception)

    def test_compatible_store_returns_its_manifest(self):
        self.assertEqual(self._check(self.store), self.manifest)
        self.assertEqual(self._check(self.store, sequence_store=self.seq.meta()), self.manifest)

    def test_statistics_only_system_is_compatible(self):
        stats_only = dataclasses.replace(SYSTEM, name="esm-codes-stats-test", panel_quantile=0.95)
        self.assertNotEqual(stats_only.digest(), SYSTEM.digest())
        self.assertEqual(self._check(self.store, system=stats_only), self.manifest)

    def test_unknown_manifest_version(self):
        for version in (2, 0, "1", True, None):
            with self.subTest(version=version):
                self._edit_manifest(self.store, lambda m: m.update(manifest_version=version))
                self.assertRefused(f"unknown manifest_version {version!r}")
        self._edit_manifest(self.store, lambda m: m.pop("manifest_version"))
        self.assertRefused("unknown manifest_version None")

    def test_wrong_kind(self):
        self._edit_manifest(self.store, lambda m: m.update(kind="something-else"))
        self.assertRefused("kind 'something-else'")

    def test_each_missing_key(self):
        original = self.store.manifest_path.read_text()
        for key in MANIFEST_KEYS:
            if key in ("manifest_version", "kind"):
                continue
            with self.subTest(key=key):
                self.store.manifest_path.write_text(original)
                self._edit_manifest(self.store, lambda m: m.pop(key))
                self.assertRefused(f"lacks {key}\\b")
        for block, keys in rcs._NESTED_KEYS.items():
            for key in keys:
                with self.subTest(block=block, key=key):
                    self.store.manifest_path.write_text(original)
                    self._edit_manifest(self.store, lambda m: m[block].pop(key))
                    self.assertRefused(f"lacks {block}\\.{key}\\b")
            with self.subTest(block=block, value="not an object"):
                self.store.manifest_path.write_text(original)
                self._edit_manifest(self.store, lambda m: m.update({block: [1]}))
                self.assertRefused(f"lacks {block} \\(not an object\\)")
        self.store.manifest_path.write_text(original)
        self._check(self.store)

    def test_unreadable_manifest(self):
        self.store.manifest_path.write_text("{not json")
        self.assertRefused("not readable JSON")
        self.store.manifest_path.write_text("[1, 2]")
        self.assertRefused("not a JSON object")

    def test_encoding_digest_mismatch(self):
        self._edit_manifest(self.store, lambda m: m.update(encoding_digest="0" * 16))
        self.assertRefused("encoding_digest '0000000000000000'")
        self.store.manifest_path.write_text(json.dumps(self.manifest))
        other = dataclasses.replace(SYSTEM, sign_rule="bit_k = (h_k > 0)")
        message = self.assertRefused("encoding_digest", system=other)
        self.assertIn("sign_rule", message)

    def test_esm3_mismatch(self):
        self.assertRefused("this FoldMatch loads 0000", esm_weights_sha256="0" * 64)
        cases = {
            "esm.weights_sha256": lambda m: m["esm"].update(weights_sha256="1" * 64),
            "system.esm_weights_sha256": lambda m: m["system"].update(esm_weights_sha256="1" * 64),
        }
        for case, edit in cases.items():
            with self.subTest(case=case):
                self.store.manifest_path.write_text(json.dumps(self.manifest))
                self._edit_manifest(self.store, edit)
                message = self.assertRefused("ESM3 weights")
                self.assertNotIn("encoding_digest", message)  # the content hash is checked on its own
        self.store.manifest_path.write_text(json.dumps(self.manifest))
        other = dataclasses.replace(SYSTEM, esm_weights_sha256="2" * 64)
        self.assertRefused("built with ESM3 weights sha256 5ead.*has 2222", system=other)

    def test_target_encoder_and_transform_mismatch(self):
        for label, field in (("target encoder", "target_encoder_sha256"), ("transform", "transform_sha256")):
            with self.subTest(field=field, where="manifest"):
                self.store.manifest_path.write_text(json.dumps(self.manifest))
                self._edit_manifest(self.store, lambda m: m["system"].update({field: "9" * 64}))
                message = self.assertRefused(f"built with {label} sha256 9999")
                self.assertNotIn("encoding_digest", message)
            with self.subTest(field=field, where="system"):
                self.store.manifest_path.write_text(json.dumps(self.manifest))
                other = dataclasses.replace(SYSTEM, **{field: "8" * 64})
                message = self.assertRefused(f"{label} sha256 {getattr(SYSTEM, field)}, {SYSTEM.name} has 8888",
                                             system=other)
                self.assertIn("encoding_digest", message)

    def test_other_encoding_fields_and_code_layout(self):
        self._edit_manifest(self.store, lambda m: m["system"].update(input_cast="float32"))
        self.assertRefused("input_cast 'float32'")
        for edit, pattern in ((lambda m: m.update(code_bytes=4), "codes of 4 bytes"),
                              (lambda m: m.update(bitorder="little"), "bit order 'little'"),
                              (lambda m: m["system"].update(name="esm-codes-x"), "the recorded system is")):
            with self.subTest(pattern=pattern):
                self.store.manifest_path.write_text(json.dumps(self.manifest))
                self._edit_manifest(self.store, edit)
                self.assertRefused(pattern)

    def test_entries_and_total_residues_against_the_sequence_store_meta(self):
        meta = self.seq.meta()
        self.assertRefused(f"7 entries, the sequence store 8",
                           sequence_store=dict(meta, entry_count=8))
        self.assertRefused(f"{meta['total_residues']} residues, the sequence store {meta['total_residues'] + 1}",
                           sequence_store=dict(meta, total_residues=meta["total_residues"] + 1))
        self._edit_manifest(self.store, lambda m: m.update(entries=6))
        self.assertRefused("6 entries, the sequence store 7")
        with self.assertRaises(TypeError):
            self._check(self.store, sequence_store={"entry_count": 7})

    def test_missing_index_or_wrong_bin_size(self):
        with open(self.store.bin_path, "ab") as f:
            f.write(b"\x00" * 8)
        self.assertRefused("is .* bytes, the manifest needs")
        self.store.bin_path.unlink()
        self.assertRefused("rescodes.bin is missing")
        self.store.idx_path.unlink()
        self.assertRefused("rescodes.idx is missing")

    def test_none_never_skips_a_check(self):
        for name in ("system", "sequence_store", "esm_weights_sha256"):
            with self.subTest(argument=name), self.assertRaisesRegex(TypeError, f"{name}=None would skip"):
                self._check(self.store, **{name: None})
        with self.assertRaises(TypeError):
            self._check(self.store, system=SYSTEM.to_dict())

    def test_installed_defaults(self):
        """Each omitted argument is compared with this installation: the registry entry the manifest names, the
        database's own sequence store, and the ESM3 weights FoldMatch loads (torch is imported, not the weights)."""
        from foldmatch.utils.esm import loaders
        self.assertEqual(self.store.check_compatible(), self.manifest)
        with mock.patch.object(loaders, "ESM_WEIGHTS_SHA256", "3" * 64):
            with self.assertRaisesRegex(IncompatibleCodeStoreError, "this FoldMatch loads 3333"):
                self.store.check_compatible()
        conn = sqlite3.connect(self.seq.path)
        conn.execute("UPDATE meta SET value = value + 1 WHERE key = 'total_residues'")
        conn.commit()
        conn.close()
        with self.assertRaisesRegex(IncompatibleCodeStoreError, "residues, the sequence store"):
            self.store.check_compatible()
        conn = sqlite3.connect(self.seq.path)
        conn.execute("UPDATE meta SET value = value - 1 WHERE key = 'total_residues'")
        conn.commit()
        conn.close()
        self._edit_manifest(self.store, lambda m: m.update(system_name="esm-codes-v999"))
        with self.assertRaisesRegex(ss.UnknownScoringSystemError, "esm-codes-v999"):
            self.store.check_compatible()

    def test_read_only(self):
        files = [self.store.bin_path, self.store.idx_path, self.store.manifest_path, self.seq.path]
        before = {p: (_sha256(p), p.stat().st_mtime_ns) for p in files}
        for p in files:
            p.chmod(0o444)
        try:
            self._check(self.store)
            self.store.lookup([self.records[0][0]])
            self.assertEqual(self.store.open_memmap().shape[0], self.manifest["total_residues"])
        finally:
            for p in files:
                p.chmod(0o644)
        self.assertEqual({p: (_sha256(p), p.stat().st_mtime_ns) for p in files}, before)
        self.assertEqual(sorted(p.name for p in self.tmp.glob("db.rescodes*")),
                         ["db.rescodes.bin", "db.rescodes.idx", "db.rescodes.json"])


# =====================================================================================================================
class ValidationFaultTest(_StoreCase):
    """Each fault (a)-(e), injected: validate() raises naming the check and its first ids, removes the code store,
    and leaves the sequence store as it was (the database still serves BLOSUM62)."""

    def setUp(self):
        super().setUp()
        self.records = _records(14, random.Random(21), min_len=5)
        self.sequences = dict(self.records)
        self.entries = _entries(self.records)

    def _fault(self, entries, faiss=14, preflight=True, after_create=None):
        self._faults = getattr(self, "_faults", 0) + 1
        name = f"db{self._faults}"
        seq, pf = self._database(self.records, name)
        seq_before = _sha256(seq.path)
        store, _ = self._create(entries, name)
        if after_create is not None:
            after_create(store, seq)
            seq_before = _sha256(seq.path)
        with self.assertRaises(CodeStoreValidationError) as ctx:
            store.validate(seq, faiss, preflight=pf if preflight is True else preflight)
        self.assertNoStoreFiles(store)
        self.assertEqual(_sha256(seq.path), seq_before)
        self.assertEqual(seq.fetch([self.records[0][0]]), {self.records[0][0]: self.records[0][1]})
        message = str(ctx.exception)
        self.assertIn("still serves --scoring blosum62", message)
        for check in ctx.exception.failures:
            self.assertIn(f"({check})", message)
        return ctx.exception.failures, message

    def test_a_missing_codes(self):
        failures, message = self._fault(self.entries[:3] + self.entries[4:])
        self.assertEqual(failures["a"], (1, (self.records[3][0],)))
        self.assertEqual(set(failures), {"a", "e"})  # the totals disagree as well
        self.assertIn(f"(a) 1 sequence-store entr(ies) have no residue codes (first: {self.records[3][0]})", message)

    def test_a_reports_the_first_10_of_14(self):
        failures, message = self._fault([])
        n, ids = failures["a"]
        self.assertEqual((n, ids), (14, tuple(sorted(self.sequences))[:10]))
        self.assertNotIn(sorted(self.sequences)[10], message)

    def test_b_extra_codes(self):
        extra = ("zz_extra", np.zeros((5, 8), dtype=np.uint8), seq_sha1("ACDEF"))
        failures, _ = self._fault(self.entries + [extra])
        self.assertEqual(failures["b"], (1, ("zz_extra",)))
        self.assertEqual(set(failures), {"b", "e"})

    def test_c_length_mismatch_alone(self):
        """One entry one residue short and another one long: the totals agree, only (c) fails."""
        entries = list(self.entries)
        (n2, c2, s2), (n5, c5, s5) = entries[2], entries[5]
        entries[2] = (n2, c2[:-1], s2)
        entries[5] = (n5, np.vstack([c5, c5[:1]]), s5)
        failures, message = self._fault(entries)
        self.assertEqual(failures, {"c": (2, tuple(sorted((n2, n5))))})
        self.assertNotIn("empty payload", message)

    def test_c_empty_payload(self):
        entries = list(self.entries)
        name, _, sha1 = entries[6]
        entries[6] = (name, b"", sha1)
        failures, message = self._fault(entries)
        self.assertEqual(failures["c"], (1, (name,)))
        self.assertIn("1 of them with an empty payload", message)
        self.assertEqual(set(failures), {"c", "e"})
        # The pass wrote the empty payload: the same shards would fail again, so the hint is a fresh build.
        self.assertIn("rebuilding from the same shards would fail the same way", message)
        self.assertIn("'fm-search build sequences --residue-codes'", message)
        self.assertNotIn("--from-shards", message)
        _, message = self._fault(self.entries[1:])  # any other failure: the shards may serve
        self.assertIn("--from-shards", message)

    def test_d_seq_sha1_mismatch_alone(self):
        entries = list(self.entries)
        bad = []
        for k, sha1 in ((1, seq_sha1("WWWWW")), (8, hashlib.sha1(self.records[8][1].lower().encode()).digest())):
            name, codes, _ = entries[k]
            entries[k] = (name, codes, sha1)
            bad.append(name)
        failures, _ = self._fault(entries)
        self.assertEqual(failures, {"d": (2, tuple(sorted(bad)))})

    def test_e_faiss_count(self):
        failures, message = self._fault(self.entries, faiss=13)
        self.assertEqual(set(failures), {"e"})
        self.assertIn("the FAISS index has 13 entries, the code store 14", message)

    def test_e_preflight_counts(self):
        total = sum(len(s) for s in self.sequences.values())
        for preflight, pattern in ((SimpleNamespace(unique_entries=15, unique_residues=total),
                                    "the pre-flight has 15 entries"),
                                   (SimpleNamespace(unique_entries=14, unique_residues=total - 1),
                                    f"the pre-flight has {total - 1} residues")):
            with self.subTest(pattern=pattern):
                failures, message = self._fault(self.entries, preflight=preflight)
                self.assertEqual(set(failures), {"e"})
                self.assertIn(pattern, message)

    def test_e_sequence_store_meta(self):
        def tamper(store, seq):
            conn = sqlite3.connect(seq.path)
            conn.execute("UPDATE meta SET value = value + 3 WHERE key = 'total_residues'")
            conn.commit()
            conn.close()

        failures, message = self._fault(self.entries, after_create=tamper,
                                        preflight=None)
        self.assertEqual(set(failures), {"e"})
        self.assertIn("the sequence store meta has", message)

    def test_e_bin_size(self):
        def truncate(store, seq):
            with open(store.tmp_bin_path, "r+b") as f:
                f.truncate(store.tmp_bin_path.stat().st_size - 8)

        failures, message = self._fault(self.entries, after_create=truncate)
        self.assertEqual(set(failures), {"e"})
        self.assertIn("rescodes.bin.tmp is", message)

    def test_e_index_rows(self):
        def drop_row(store, seq):
            conn = sqlite3.connect(store.tmp_idx_path)
            conn.execute("DELETE FROM entries WHERE id = ?", (self.records[0][0],))
            conn.commit()
            conn.close()

        failures, message = self._fault(self.entries, after_create=drop_row)
        self.assertEqual(set(failures), {"a", "e"})
        self.assertIn("the code index has 13 entries, the code store 14", message)

    def test_missing_sequence_store_removes_the_codes(self):
        seq, pf = self._database(self.records)
        store, _ = self._create(self.entries)
        seq.path.unlink()
        with self.assertRaises(FileNotFoundError):
            store.validate(seq, 14, preflight=pf)
        self.assertNoStoreFiles(store)

    def test_bad_arguments(self):
        seq, pf = self._database(self.records)
        store, _ = self._create(self.entries)
        for args, kwargs in (((seq, 14), {}), ((seq, -1), {"preflight": pf}), ((seq, True), {"preflight": pf}),
                             ((seq, 14.0), {"preflight": pf}), ((seq.path, 14), {"preflight": pf}),
                             ((seq, 14), {"preflight": 14})):
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(TypeError):
                store.validate(*args, **kwargs)
        self.assertTrue(store.is_partial())  # a misuse is not a fault of the store
        self.assertEqual(store.validate(seq, np.int64(14), preflight=pf)["entries"], 14)

    def test_corrupt_files_are_named(self):
        """A sequence store or code index that is not an SQLite database: a store error naming the file (SQLite's
        own message names none), and validate() discards the pending build."""
        seq, pf = self._database(self.records)
        store, _ = self._create(self.entries)
        good = seq.path.read_bytes()
        seq.path.write_bytes(b"not a database" * 100)
        with self.assertRaises(ResidueCodeStoreError) as ctx:
            store.validate(seq, 14, preflight=pf)
        self.assertIn(f"{seq.path} is not a readable sequence store", str(ctx.exception))
        self.assertNoStoreFiles(store)
        seq.path.write_bytes(good)
        store, _ = self._create(self.entries)
        store.tmp_idx_path.write_bytes(b"not a database" * 100)
        with self.assertRaises(ResidueCodeStoreError) as ctx:
            store.validate(seq, 14, preflight=pf)
        self.assertIn(f"{store.tmp_idx_path}: not a readable", str(ctx.exception))
        self.assertNoStoreFiles(store)
        store, _ = self._create(self.entries)
        store.validate(seq, 14, preflight=pf)
        seq.path.write_bytes(b"not a database" * 100)
        with self.assertRaises(IncompatibleCodeStoreError) as ctx:
            self._check(store, sequence_store=seq)
        self.assertIn(f"{seq.path} is not a readable sequence store", str(ctx.exception))
        seq.path.write_bytes(good)
        store.idx_path.write_bytes(b"not a database" * 100)
        with self.assertRaises(ResidueCodeStoreError) as ctx:
            store.lookup([self.records[0][0]])
        self.assertIn(f"{store.idx_path}: not a readable", str(ctx.exception))

    def test_nonstandard_entries_are_counted(self):
        records = [("x1", "ACDXE"), ("x2", "ACDEF"), ("x3", "B*CD"), ("x4", "ACD-E")]
        _, _, manifest = self._build(records, name="ns")
        self.assertEqual(manifest["nonstandard_entries"], 3)


# =====================================================================================================================
class PathsTest(unittest.TestCase):
    def test_files_follow_the_database_path(self):
        store = ResidueCodeStore.from_db_path("/data/dbs/uniref50")
        self.assertEqual((store.db_folder, store.index_name), (Path("/data/dbs"), "uniref50"))
        self.assertEqual([store.bin_path, store.idx_path, store.manifest_path],
                         [Path("/data/dbs/uniref50.rescodes.bin"), Path("/data/dbs/uniref50.rescodes.idx"),
                          Path("/data/dbs/uniref50.rescodes.json")])
        self.assertEqual(store.sequence_store().path, Path("/data/dbs/uniref50.sequences"))
        self.assertEqual(ResidueCodeStore.from_db_path("bare").db_folder, Path.cwd())
        self.assertEqual([store.tmp_bin_path, store.tmp_idx_path, store.tmp_manifest_path],
                         [Path("/data/dbs/uniref50.rescodes.bin.tmp"), Path("/data/dbs/uniref50.rescodes.idx.tmp"),
                          Path("/data/dbs/uniref50.rescodes.json.tmp")])
        # every file is covered by the build pre-flight's outputs-exist glob '{index}.rescodes.*'
        for path in (store.bin_path, store.idx_path, store.manifest_path, *store._pending_paths()):
            self.assertTrue(path.name.startswith("uniref50.rescodes."), path)

    def test_module_does_not_import_torch(self):
        code = ("import sys; import foldmatch.search.residue_code_store; "
                "sys.exit(int('torch' in sys.modules))")
        env = dict(os.environ, PYTHONPATH=_SRC)
        self.assertEqual(subprocess.run([sys.executable, "-c", code], env=env).returncode, 0)


if __name__ == "__main__":
    unittest.main()
