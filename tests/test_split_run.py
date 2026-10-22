"""P8b.2: the run record of a split sequence build (``foldmatch.search.split_run``) and ``multi_rank_launch``.

The shards are written by the fake ESM3 pass of ``test_cli_residue_codes`` (the pass's dataset and writer), plain and
with residue codes, on one or more ranks; the records are assembled here as step 1 assembles them.
"""
import builtins
import errno
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pyarrow.parquet as pq

import foldmatch
from foldmatch.errors import RefusalError
from foldmatch.inference.chain_complete_inference import bundle_fetch_allowed
from foldmatch.search import split_run
from foldmatch.search.residue_shards import META_PASS, pass_record, run_shard_files
from foldmatch.search.sequence_store import SequenceStore
from foldmatch.utils.dist import multi_rank_launch
from foldmatch.utils.sequence import preflight_fasta
from test_cli_residue_codes import OTHER_ENCODING, SYSTEM, _write, fake_chain_predict, launch_env

_TEST_PATH = Path(__file__).parent
_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
_FIVE = ">e0\nACDEFGHIKL\n>e1 second\nKLMNPQRSTV\n>e2\nVWYACDEFGH\n>e3\nIKLMNPQRST\n>e4\nACDEFGHIKLMN\n"


def _system_block(system):
    return {"name": system.name, "digest": system.digest(), "encoding_digest": system.encoding_digest()}


class _Run(unittest.TestCase):
    """A temporary folder, the launch variables cleared, and helpers that write a run's shards and its record."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.shards = self.tmp / "shards"
        self.shards.mkdir()
        self.enterContext(launch_env())

    def write_shards(self, run_id="r", fasta=_FASTA, *, ranks=1, codes=False, min_res_n=0, folder=None, **fake):
        """The shards the pass writes for ``fasta`` on ``ranks`` ranks; returns ``(preflight, shard paths)``."""
        folder = folder or self.shards
        preflight = preflight_fasta(fasta, min_res_n)
        kwargs = {"src_stream": preflight, "min_res_n": min_res_n, "out_folder": folder, "out_name": f"emb_{run_id}"}
        if codes:
            kwargs.update(residue_output="residue_codes", scoring_system=SimpleNamespace(system=SYSTEM))
        fake_chain_predict(ranks=ranks, **fake)(**kwargs)
        return preflight, run_shard_files(folder, run_id)

    def make_record(self, run_id="r", fasta=_FASTA, *, ranks=1, codes=False, min_res_n=0, folder=None, **fake):
        """Shards plus the record step 1 would write for them; returns ``(record, preflight, shard paths)``."""
        folder = folder or self.shards
        preflight, paths = self.write_shards(run_id, fasta, ranks=ranks, codes=codes, min_res_n=min_res_n,
                                             folder=folder, **fake)
        sha256, _ = split_run.fasta_sha256(fasta)
        stat = split_run.fasta_stat(fasta)
        if codes:
            run_pass = split_run.check_codes_footers(paths, system=SYSTEM, min_res_n=min_res_n, world_size=ranks)
        else:
            run_pass = pass_record(min_res_n=min_res_n, esm_revision="test-revision", world_size=ranks)
        shards = split_run.shard_entries(paths)
        record = {
            "kind": split_run.RUN_RECORD_KIND, "record_version": split_run.RUN_RECORD_VERSION, "run_id": run_id,
            "command": split_run.EMBED_COMMAND, "sealed": False, "created_utc": "2026-10-22T12:00:00Z",
            "foldmatch_version": foldmatch.__version__,
            "fasta": {"path": str(Path(fasta).resolve()), "bytes": stat.bytes, "mtime_ns": stat.mtime_ns,
                      "sha256": sha256},
            "min_res_n": min_res_n, "strict": codes, "preflight": split_run.preflight_counts(preflight),
            "output_db": str(self.tmp / "db"), "residue_output": "residue_codes" if codes else None,
            "system": _system_block(SYSTEM) if codes else None, "world_size": ranks, "pass": run_pass,
            "options": {"accelerator": "cpu", "devices": ["1"], "num_nodes": 1, "strategy": "auto", "batch_size": 1,
                        "num_workers": 0},
            "shards": shards, "rows": sum(shard["rows"] for shard in shards),
        }
        return record, preflight, paths

    def refused(self, call, *texts):
        """``call()`` raised a RefusalError containing every one of ``texts``; returns its message."""
        with self.assertRaises(RefusalError) as ctx:
            call()
        message = str(ctx.exception)
        for text in texts:
            self.assertIn(text, message)
        return message

    def names(self, folder=None):
        return sorted(p.name for p in (folder or self.shards).iterdir())


# =====================================================================================================================
# record I/O
# =====================================================================================================================
class RunRecordIOTest(_Run):

    def test_round_trip(self):
        record, _, _ = self.make_record(codes=True)
        path = split_run.write_run_record(self.shards, record)
        self.assertEqual(path, self.shards / "emb_r.run.json")
        self.assertEqual(path, split_run.run_record_path(self.shards, "r"))
        self.assertEqual(path.read_text(), json.dumps(record, indent=2, sort_keys=True) + "\n")
        self.assertEqual(split_run.read_run_record(path), record)
        self.assertEqual(self.names(), ["emb_r-0.parquet", "emb_r.run.json"])
        self.assertEqual(split_run.record_run_ids(self.shards), ["r"])
        self.assertEqual(split_run.tmp_record_run_ids(self.shards), [])

    def test_a_record_is_no_shard(self):
        """The record is not taken for a shard of any run, nor for an embedding file of 'build embeddings'."""
        from foldmatch.search.residue_shards import shard_run_ids
        record, _, _ = self.make_record()
        split_run.write_run_record(self.shards, record)
        self.assertEqual(shard_run_ids(self.shards), ["r"])
        self.assertEqual(run_shard_files(self.shards, "r"), [self.shards / "emb_r-0.parquet"])
        self.assertFalse(split_run.RUN_RECORD_SUFFIX.endswith((".pt", ".csv", ".parquet")))

    def test_a_failure_leaves_no_file(self):
        """The dump, the write, the fsync or the link raising: neither the record nor the .tmp is left, and the
        refusal names the file."""
        record, _, _ = self.make_record()
        before = self.names()
        failures = {
            "the dump": ({**record, "rows_nan": float("nan")}, None, "cannot be written as JSON"),
            "the write": (record, mock.patch.object(split_run, "_write_all",
                                                    side_effect=OSError(errno.ENOSPC, "No space left on device")),
                          "emb_r.run.json"),
            "the fsync": (record, mock.patch.object(split_run.os, "fsync", side_effect=OSError(errno.EIO, "I/O")),
                          "emb_r.run.json"),
            "the link": (record, mock.patch.object(split_run.os, "link", side_effect=OSError(errno.EIO, "I/O")),
                         "emb_r.run.json"),
        }
        for case, (value, patch, text) in failures.items():
            with self.subTest(case=case):
                with patch or nullcontext():
                    self.refused(lambda: split_run.write_run_record(self.shards, value), text)
                self.assertEqual(self.names(), before)

    def test_the_staged_record_is_checked_before_it_is_committed(self):
        record, _, _ = self.make_record()
        record["shards"][0]["bytes"] += 1
        self.refused(lambda: split_run.write_run_record(self.shards, record), "emb_r.run.json.tmp failed its check",
                     "changed since the run")
        self.assertEqual(self.names(), ["emb_r-0.parquet"])
        record, _, _ = self.make_record("s")
        record["rows"] += 1
        self.refused(lambda: split_run.write_run_record(self.shards, record), "failed its check", "rows is 3")
        self.assertEqual(self.names(), ["emb_r-0.parquet", "emb_s-0.parquet"])

    def test_an_existing_record_or_tmp_is_refused(self):
        record, _, _ = self.make_record()
        tmp = split_run.run_record_tmp_path(self.shards, "r")
        tmp.write_text("another writer's")
        self.refused(lambda: split_run.write_run_record(self.shards, record), str(tmp))
        self.assertEqual(tmp.read_text(), "another writer's")  # not ours: left alone
        self.refused(lambda: split_run.refuse_existing_record(self.shards, "r"), str(tmp), "same run id (r)",
                     "Delete the .tmp", "FOLDMATCH_RUN_ID")
        tmp.unlink()
        final = split_run.write_run_record(self.shards, record)
        text = final.read_text()
        self.refused(lambda: split_run.write_run_record(self.shards, record), str(final))
        self.assertEqual((final.read_text(), self.names()), (text, ["emb_r-0.parquet", "emb_r.run.json"]))
        self.refused(lambda: split_run.refuse_existing_record(self.shards, "r"), str(final), "same run id (r)",
                     "Remove it and its shards", "another --shards-folder", "FOLDMATCH_RUN_ID")
        split_run.refuse_existing_record(self.shards, "other")

    def test_without_hard_links(self):
        """``os.link`` unsupported (ENOTSUP): the .tmp is renamed into place (INFO), unless a record exists."""
        record, _, _ = self.make_record()
        with mock.patch.object(split_run.os, "link", side_effect=OSError(errno.ENOTSUP, "Operation not supported")), \
                self.assertLogs("foldmatch.search.split_run", "INFO") as logs:
            path = split_run.write_run_record(self.shards, record)
        self.assertIn("does not support hard links", logs.output[0])
        self.assertEqual(split_run.read_run_record(path), record)
        self.assertEqual(self.names(), ["emb_r-0.parquet", "emb_r.run.json"])

        other, _, _ = self.make_record("s")
        final = split_run.run_record_path(self.shards, "s")

        def appears(src, dst):  # another writer commits between the check of the .tmp and the link
            final.write_text("theirs")
            raise OSError(errno.EOPNOTSUPP, "Operation not supported")

        with mock.patch.object(split_run.os, "link", side_effect=appears):
            self.refused(lambda: split_run.write_run_record(self.shards, other), str(final))
        self.assertEqual(final.read_text(), "theirs")
        self.assertFalse(split_run.run_record_tmp_path(self.shards, "s").exists())

    def test_each_errno_of_a_missing_hard_link(self):
        """EPERM (link(2) on a filesystem without hard links, as on Linux), ENOTSUP and EOPNOTSUPP (one errno on Linux,
        two on macOS): each renames the .tmp into place, and the folder holds only the shard and the record."""
        for code in sorted({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP}):
            folder = self.tmp / f"errno-{code}"
            folder.mkdir()
            with self.subTest(errno=errno.errorcode[code]):
                record, _, _ = self.make_record(folder=folder)
                with mock.patch.object(split_run.os, "link", side_effect=OSError(code, os.strerror(code))), \
                        self.assertLogs("foldmatch.search.split_run", "INFO") as logs:
                    path = split_run.write_run_record(folder, record)
                self.assertIn("does not support hard links", logs.output[0])
                self.assertEqual(split_run.read_run_record(path), record)
                self.assertEqual(self.names(folder), ["emb_r-0.parquet", "emb_r.run.json"])

    def test_discovery_ignores_names_that_are_no_run_id(self):
        for name in ("emb_a.run.json", "emb_b.run.json.tmp", "emb_copy[1].run.json", "emb_-x.run.json",
                     "emb_c.run.json.bak", "emb_.run.json", "x_d.run.json"):
            (self.shards / name).touch()
        self.assertEqual(split_run.record_run_ids(self.shards), ["a"])
        self.assertEqual(split_run.tmp_record_run_ids(self.shards), ["b"])
        self.assertEqual(split_run.run_record_path(self.shards, "a.b"), self.shards / "emb_a.b.run.json")
        self.refused(lambda: split_run.run_record_path(self.shards, "a b"), "is not a run id")


class ReadRunRecordTest(_Run):
    """Every I7 refusal of ``read_run_record``, each naming the file and the key."""

    def setUp(self):
        super().setUp()
        self.record, _, _ = self.make_record(ranks=2, codes=True)
        self.path = self.shards / "emb_r.run.json"

    def read(self, value, path=None):
        path = path or self.path
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return split_run.read_run_record(path)

    def bad(self, change, *texts):
        record = json.loads(json.dumps(self.record))
        change(record)
        return self.refused(lambda: self.read(record), str(self.path), *texts)

    def test_the_record_reads(self):
        self.assertEqual(self.read(self.record), self.record)
        self.assertEqual(self.read({**self.record, "written_by_a_later_version": [1]})["run_id"], "r")  # ignored

    def test_unreadable_or_not_an_object(self):
        self.refused(lambda: split_run.read_run_record(self.shards / "emb_none.run.json"), "cannot be read")
        self.path.write_bytes(b"\xff\xfe")
        self.refused(lambda: split_run.read_run_record(self.path), "cannot be read")
        self.refused(lambda: self.read("{not json"), "not JSON")
        self.refused(lambda: self.read([self.record]), "not a JSON object")

    def test_kind_and_version(self):
        self.bad(lambda r: r.update(kind="foldmatch-other"), "kind is 'foldmatch-other'")
        message = self.bad(lambda r: r.update(record_version=2), "newer FoldMatch", "record_version 2")
        self.assertIn("upgrade FoldMatch", message)
        for version in (0, "1", 1.0, True, None):
            with self.subTest(version=version):
                self.bad(lambda r: r.update(record_version=version), "record_version is")

    def test_every_key_is_required(self):
        for key in split_run._RECORD_TYPES:
            if key in ("kind", "record_version"):  # test_kind_and_version
                continue
            with self.subTest(key=key):
                self.bad(lambda r: r.pop(key), f"the key {key} is missing")
        for block, keys in (("fasta", split_run.FASTA_KEYS), ("preflight", split_run.PREFLIGHT_KEYS),
                            ("system", split_run.SYSTEM_KEYS), ("pass", tuple(self.record["pass"]))):
            for key in keys:
                with self.subTest(block=block, key=key):
                    self.bad(lambda r: r[block].pop(key), f"the key {block}.{key} is missing")
        for key in ("name", "rank", "rows", "bytes"):
            with self.subTest(shard_key=key):
                self.bad(lambda r: r["shards"][1].pop(key), f"the key shards[1].{key} is missing")

    def test_wrong_types(self):
        cases = {"rows": "2", "sealed": 0, "world_size": True, "output_db": 3, "residue_output": 1, "system": [],
                 "options": None, "shards": {}, "min_res_n": 1.5, "created_utc": 0, "pass": "cpu"}
        for key, value in cases.items():
            with self.subTest(key=key):
                self.bad(lambda r: r.update({key: value}), f"{key} is {value!r}")
        self.bad(lambda r: r["fasta"].update(bytes="303"), "fasta.bytes is '303', expected integer")
        self.bad(lambda r: r["preflight"].update(entries=None), "preflight.entries is None")
        self.bad(lambda r: r["pass"].update(world_size="2"), "pass.world_size is '2'")
        self.bad(lambda r: r["pass"].update(accelerator=1), "pass.accelerator is 1, expected string or null")
        self.bad(lambda r: r["shards"].__setitem__(0, "emb_r-0.parquet"), "shards[0] is 'emb_r-0.parquet'")
        self.bad(lambda r: r["shards"][0].update(rows=False), "shards[0].rows is False")

    def test_negative_counts(self):
        self.bad(lambda r: r["preflight"].update(residues=-1), "preflight.residues is negative")
        self.bad(lambda r: r.update(min_res_n=-1), "min_res_n is negative")
        self.bad(lambda r: r["fasta"].update(bytes=-5), "fasta.bytes is negative")
        self.bad(lambda r: r["shards"][0].update(bytes=-1), "shards[0].bytes is negative")
        self.bad(lambda r: r.update(world_size=0), "world_size is 0")

    def test_inconsistent_fields(self):
        self.refused(lambda: self.read(self.record, self.shards / "emb_s.run.json"), "run_id is 'r'",
                     "named for run 's'")
        self.refused(lambda: self.read(self.record, self.shards / "record.json"), "named for no run")
        self.bad(lambda r: r.update(run_id="a b"), "run_id 'a b' is not a run id")
        self.bad(lambda r: r.update(command="build sequences"), "command is 'build sequences'")
        self.bad(lambda r: r.update(sealed=True), "sealed is True for the command 'build sequences-embed'")
        self.bad(lambda r: r.update(command=split_run.SEAL_COMMAND), "sealed is False")
        self.bad(lambda r: r.update(created_utc="2026-10-22 12:00:00"), "created_utc")
        self.bad(lambda r: r["fasta"].update(sha256="ABC"), "fasta.sha256 'ABC' is not a sha256")
        self.bad(lambda r: r.update(residue_output="query_encoding"), "residue_output is 'query_encoding'")
        self.bad(lambda r: r.update(strict=False), "strict is False, but residue_output is 'residue_codes'")
        self.bad(lambda r: r.update(system=None), "system is null")
        self.bad(lambda r: r.update(residue_output=None, strict=False), "system is set, but residue_output is None")
        self.bad(lambda r: r["pass"].update(min_res_n=100), "pass.min_res_n is 100, but min_res_n is 0")
        self.bad(lambda r: r["pass"].update(world_size=1), "pass.world_size is 1, but world_size is 2")
        self.bad(lambda r: r["preflight"].update(unique_entries=1), "preflight.unique_entries is 1")
        self.bad(lambda r: r["preflight"].update(entries=1), "entries_kept (2) exceeds preflight.entries (1)")

    def test_rows_and_the_deal(self):
        self.bad(lambda r: r.update(rows=3), "rows is 3, but the shards hold 2")
        self.bad(lambda r: (r.update(rows=3), r["shards"][0].update(rows=2)), "rows is 3, but preflight")
        self.bad(lambda r: r["shards"][1].update(name="emb_r-2.parquet"), "shards[1] is named 'emb_r-2.parquet'")
        self.bad(lambda r: r["shards"].reverse(), "in file-name order")
        self.bad(lambda r: r["shards"].__setitem__(1, dict(r["shards"][0])), "listed once each")
        # Two rows on rank 0 and none on rank 1: the right total, not the deal of 2 over 2.
        message = self.bad(lambda r: (r["shards"].pop(), r["shards"][0].update(rows=2)), "not the deal of 2 entries "
                           "over 2 rank(s)")
        self.assertIn("rank 0: 2 row(s)", message)
        self.assertIn("rank 1: 1 row(s)", message)
        # A world of 3 for 2 entries: ranks 0 and 1, one row each (rank 2 writes no shard).
        record = json.loads(json.dumps(self.record))
        record.update(world_size=3)
        record["pass"].update(world_size=3)
        self.assertEqual(self.read(record)["world_size"], 3)
        record.update(world_size=1)
        record["pass"].update(world_size=1)
        self.refused(lambda: self.read(record), "not the deal of 2 entries over 1 rank(s)")


# =====================================================================================================================
# selection and checks
# =====================================================================================================================
class FindRunTest(_Run):

    def commit(self, run_id, **kwargs):
        record, _, _ = self.make_record(run_id, **kwargs)
        return split_run.write_run_record(self.shards, record)

    def test_no_run(self):
        self.refused(lambda: split_run.find_run(self.shards), "No run in", "--shards-folder")
        self.refused(lambda: split_run.find_run(self.shards, "r"), "Run r has no run record",
                     "neither a run record nor embedding shards", "Runs with a record here: none")

    def test_the_only_run(self):
        path = self.commit("r")
        self.assertEqual(split_run.find_run(self.shards), ("r", path))
        self.assertEqual(split_run.find_run(self.shards, "r"), ("r", path))

    def test_shards_without_a_record(self):
        self.write_shards("r", ranks=2)
        message = self.refused(lambda: split_run.find_run(self.shards), "Run r", "2 shard(s), no run record",
                               "did not finish", "kept by 'build sequences'", "build sequences-seal",
                               f"--from-shards {self.shards} --run-id r", "build sequences-embed' again")
        self.assertNotIn(".tmp", message)
        self.refused(lambda: split_run.find_run(self.shards, "r"), "Run r has no run record", "2 shard(s)",
                     "build sequences-seal", "Runs with a record here: none")

    def test_a_tmp_only(self):
        split_run.run_record_tmp_path(self.shards, "r").write_text("{}")
        self.refused(lambda: split_run.find_run(self.shards), "Run r", "unfinished run record (.tmp)",
                     f"delete {split_run.run_record_tmp_path(self.shards, 'r')} first")
        self.refused(lambda: split_run.find_run(self.shards, "r"), "Delete", "emb_r.run.json.tmp")

    def test_a_complete_run_beside_an_unfinished_one(self):
        """D33: two runs and no --run-id is refused, even when only one of them has a record."""
        path = self.commit("done")
        self.write_shards("half")
        self.refused(lambda: split_run.find_run(self.shards), "holds 2 runs", "done (run record, 1 shard(s))",
                     "half (1 shard(s), no run record)", "Pass --run-id")
        self.assertEqual(split_run.find_run(self.shards, "done"), ("done", path))
        self.refused(lambda: split_run.find_run(self.shards, "half"), "Run half has no run record",
                     "build sequences-seal", "Runs with a record here: done")

    def test_several_runs(self):
        paths = {rid: self.commit(rid) for rid in ("a", "b")}
        split_run.run_record_tmp_path(self.shards, "c").write_text("{}")
        self.refused(lambda: split_run.find_run(self.shards), "holds 3 runs", "a (run record, 1 shard(s))",
                     "b (run record, 1 shard(s))", "c (0 shard(s), an unfinished run record (.tmp))")
        for rid, path in paths.items():
            self.assertEqual(split_run.find_run(self.shards, rid), (rid, path))
        self.refused(lambda: split_run.find_run(self.shards, "z"), "Run z has no run record",
                     "neither a run record nor embedding shards", "Runs with a record here: a, b")

    def test_names_that_are_no_run_id_are_ignored(self):
        path = self.commit("r")
        shutil.copyfile(path, self.shards / "emb_copy[1].run.json")
        shutil.copyfile(self.shards / "emb_r-0.parquet", self.shards / "emb_copy[1]-0.parquet")
        (self.shards / "notes.run.json").touch()
        self.assertEqual(split_run.find_run(self.shards), ("r", path))


class CheckRunShardsTest(_Run):

    def setUp(self):
        super().setUp()
        self.record, _, self.paths = self.make_record(ranks=2, codes=True)

    def test_the_recorded_shards(self):
        self.assertEqual(split_run.check_run_shards(self.shards, self.record), self.paths)
        copy = self.tmp / "copy"
        shutil.copytree(self.shards, copy)
        self.assertEqual(split_run.check_run_shards(copy, self.record), run_shard_files(copy, "r"))

    def test_a_missing_or_an_extra_shard(self):
        self.paths[1].unlink()
        self.refused(lambda: split_run.check_run_shards(self.shards, self.record), "missing emb_r-1.parquet",
                     "changed since the run")
        shutil.copyfile(self.paths[0], self.paths[1])
        shutil.copyfile(self.paths[0], self.shards / "emb_r-2.parquet")
        self.refused(lambda: split_run.check_run_shards(self.shards, self.record),
                     "not in the record emb_r-2.parquet")

    def test_a_changed_shard(self):
        self.write_shards("x", _write(self.tmp / "other.fasta", ">o\nACDEFGHIKLMNPQRSTVWY\n"), codes=True,
                          folder=self.tmp)
        shutil.copyfile(self.tmp / "emb_x-0.parquet", self.paths[1])  # same name, one row, other bytes
        self.refused(lambda: split_run.check_run_shards(self.shards, self.record), "1 shard(s) of run r",
                     "changed since the run", "emb_r-1.parquet (")
        self.record["shards"][1].update(rows=1, bytes=self.paths[1].stat().st_size)
        self.assertEqual(split_run.check_run_shards(self.shards, self.record), self.paths)
        self.record["shards"][1].update(rows=2)
        self.refused(lambda: split_run.check_run_shards(self.shards, self.record), "1 rows; recorded",
                     "2 rows")
        self.paths[0].write_bytes(b"x" * self.record["shards"][0]["bytes"])  # same size, no footer
        self.refused(lambda: split_run.check_run_shards(self.shards, self.record), "emb_r-0.parquet",
                     "cannot be read")


class CheckDealtIdsTest(_Run):

    def rewrite(self, path, ids):
        """``path`` with its rows' ids replaced by ``ids`` (rows and schema kept)."""
        import pyarrow as pa
        table = pq.read_table(path)
        table = table.set_column(table.schema.get_field_index("id"), "id", pa.array(ids, pa.string()))
        pq.write_table(table, path)

    def test_the_deal(self):
        fasta = _write(self.tmp / "five.fasta", _FIVE)
        for ranks in (1, 2, 3, 5, 7):
            with self.subTest(ranks=ranks):
                record, preflight, paths = self.make_record(f"w{ranks}", fasta, ranks=ranks)
                self.assertEqual(len(paths), min(ranks, 5))
                self.assertIsInstance(split_run.check_dealt_ids(paths, record, preflight), float)

    def test_a_swapped_pair(self):
        record, preflight, paths = self.make_record()
        self.rewrite(paths[0], ["2uzi_A", "1acb_E"])
        message = self.refused(lambda: split_run.check_dealt_ids(paths, record, preflight),
                               "2 shard row(s) of run r", "emb_r-0.parquet row 0 holds '2uzi_A', entry 0 is '1acb_E'",
                               "emb_r-0.parquet row 1 holds '1acb_E', entry 1 is '2uzi_A'")
        self.assertIn(str(_FASTA), message)

    def test_a_repeated_id_at_the_wrong_position(self):
        """A plain FASTA may repeat an id: the multiset of ids is right, the positions are not."""
        fasta = _write(self.tmp / "dup.fasta", ">x\nACDEFGHIKL\n>y\nKLMNPQRSTV\n>x\nVWYACDEFGH\n>z\nIKLMNPQRST\n")
        record, preflight, paths = self.make_record("d", fasta, ranks=2)
        split_run.check_dealt_ids(paths, record, preflight)
        # rank 0 holds entries 0 and 2 (x, x), rank 1 entries 1 and 3 (y, z); move one x to rank 1.
        self.rewrite(paths[0], ["x", "y"])
        self.rewrite(paths[1], ["x", "z"])
        self.refused(lambda: split_run.check_dealt_ids(paths, record, preflight), "2 shard row(s)",
                     "emb_d-0.parquet row 1 holds 'y', entry 2 is 'x'",
                     "emb_d-1.parquet row 0 holds 'x', entry 1 is 'y'")

    def test_a_world_larger_than_the_input(self):
        record, preflight, paths = self.make_record(ranks=3)
        self.assertEqual([p.name for p in paths], ["emb_r-0.parquet", "emb_r-1.parquet"])
        split_run.check_dealt_ids(paths, record, preflight)
        shutil.copyfile(paths[1], self.shards / "emb_r-2.parquet")
        self.refused(lambda: split_run.check_dealt_ids(run_shard_files(self.shards, "r"), record, preflight),
                     "not the deal of the 2 entries", "rank 2: 1 row(s)")
        self.refused(lambda: split_run.check_dealt_ids(paths[:1], record, preflight), "rank 0: 1 row(s)) are not")

    def test_many_offsets(self):
        """More entries than one offset chunk (the chunk patched down to 2)."""
        fasta = _write(self.tmp / "five.fasta", _FIVE)
        record, preflight, paths = self.make_record("m", fasta, ranks=2)
        with mock.patch.object(split_run, "_OFFSET_CHUNK", 2):
            split_run.check_dealt_ids(paths, record, preflight)
            self.rewrite(paths[1], ["e1", "e4"])
            self.refused(lambda: split_run.check_dealt_ids(paths, record, preflight),
                         "emb_m-1.parquet row 1 holds 'e4', entry 3 is 'e3'")


class FooterTest(_Run):

    def test_plain_footers(self):
        _, paths = self.write_shards(ranks=2)
        split_run.check_plain_footers(paths)
        _, codes = self.write_shards("c", codes=True)
        message = self.refused(lambda: split_run.check_plain_footers(paths + codes), "1 of the 3 shard(s)",
                               "not plain embedding shards", "emb_c-0.parquet")
        self.assertIn("foldmatch.residue_output", message)
        table = pq.read_table(paths[0])
        pq.write_table(table.replace_schema_metadata({b"foldmatch.pass": b"{}"}), paths[0])
        self.refused(lambda: split_run.check_plain_footers(paths), "emb_r-0.parquet", "foldmatch.pass")

    def test_codes_footers_against_the_system(self):
        five = _write(self.tmp / "five.fasta", _FIVE)
        _, paths = self.write_shards(fasta=five, ranks=2, codes=True, min_res_n=10, device="cuda")
        self.assertEqual(len(paths), 2)
        run_pass = split_run.check_codes_footers(paths, system=SYSTEM, min_res_n=10, world_size=2)
        self.assertEqual((run_pass["min_res_n"], run_pass["world_size"], run_pass["accelerator"]), (10, 2, "cuda"))
        self.refused(lambda: split_run.check_codes_footers(paths, system=SYSTEM, min_res_n=0, world_size=2),
                     "2 of the 2 shard(s) do not record a pass with --min-res-n 0 on 2 rank(s)",
                     "emb_r-0.parquet (min_res_n 10, world_size 2)")
        self.refused(lambda: split_run.check_codes_footers(paths, system=SYSTEM, min_res_n=10, world_size=3),
                     "on 3 rank(s)")
        _, plain = self.write_shards("p")
        self.refused(lambda: split_run.check_codes_footers(paths + plain, system=SYSTEM, min_res_n=10,
                                                           world_size=2), "mixed embedding shards")
        _, stale = self.write_shards("o", codes=True, system=OTHER_ENCODING)
        self.refused(lambda: split_run.check_codes_footers(stale, system=SYSTEM, min_res_n=0, world_size=1),
                     "stale embedding shards")
        table = pq.read_table(paths[1])
        meta = {k: v for k, v in table.schema.metadata.items() if k != META_PASS}
        pq.write_table(table.replace_schema_metadata(meta), paths[1])
        self.refused(lambda: split_run.check_codes_footers(paths, system=SYSTEM, min_res_n=10, world_size=2),
                     "1 of the 2 shard(s)", "emb_r-1.parquet (no pass record)")
        self.refused(lambda: split_run.check_codes_footers([], system=SYSTEM, min_res_n=0, world_size=1),
                     "no shards")

    def test_codes_footers_against_a_record(self):
        """``expect`` (a record's system block): self-consistency, also for a system this FoldMatch does not know."""
        _, paths = self.write_shards(codes=True)
        split_run.check_codes_footers(paths, expect=_system_block(SYSTEM), min_res_n=0, world_size=1)
        _, other = self.write_shards("o", codes=True, system=OTHER_ENCODING)
        run_pass = split_run.check_codes_footers(other, expect=_system_block(OTHER_ENCODING), min_res_n=0,
                                                 world_size=1)
        self.assertEqual(run_pass["world_size"], 1)
        message = self.refused(lambda: split_run.check_codes_footers(other, expect=_system_block(SYSTEM), min_res_n=0,
                                                                     world_size=1),
                               f"residue codes of {OTHER_ENCODING.name}", f"not those of {SYSTEM.name}")
        self.assertIn(SYSTEM.encoding_digest(), message)
        self.assertIn(OTHER_ENCODING.encoding_digest(), message)
        self.refused(lambda: split_run.check_codes_footers(paths + other, expect=_system_block(SYSTEM), min_res_n=0,
                                                           world_size=1), "Mixed embedding shards")
        _, plain = self.write_shards("p")
        self.refused(lambda: split_run.check_codes_footers(plain, expect=_system_block(SYSTEM), min_res_n=0,
                                                           world_size=1), "no residue column",
                     f"not the residue codes of {SYSTEM.name}")
        with self.assertRaises(ValueError):
            split_run.check_codes_footers(paths, min_res_n=0, world_size=1)
        with self.assertRaises(ValueError):
            split_run.check_codes_footers(paths, system=SYSTEM, expect=_system_block(SYSTEM), min_res_n=0,
                                          world_size=1)


# =====================================================================================================================
# FASTA and shard facts
# =====================================================================================================================
class FactsTest(_Run):

    def test_fasta_sha256(self):
        reads = []
        real_open = builtins.open

        class Spy:
            def __init__(self, f):
                self.f = f

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.f.close()

            def read(self, size=-1):
                reads.append(size)
                return self.f.read(size)

        data = os.urandom(3 * 1000 + 7)
        path = self.tmp / "data.fasta"
        path.write_bytes(data)
        with mock.patch.object(split_run, "open", create=True, side_effect=lambda *a, **k: Spy(real_open(*a, **k))):
            digest, seconds = split_run.fasta_sha256(path)
        self.assertEqual(digest, hashlib.sha256(data).hexdigest())
        self.assertGreaterEqual(seconds, 0.0)
        self.assertEqual(reads, [16 << 20, 16 << 20])  # one read of the file, one at its end
        reads.clear()
        with mock.patch.object(split_run, "HASH_CHUNK", 1000), \
                mock.patch.object(split_run, "open", create=True, side_effect=lambda *a, **k: Spy(real_open(*a, **k))):
            self.assertEqual(split_run.fasta_sha256(path)[0], hashlib.sha256(data).hexdigest())
        self.assertEqual(reads, [1000] * 5)

    def test_fasta_stat(self):
        path = _write(self.tmp / "a.fasta", ">a\nACDE\n")
        st = os.stat(path)
        self.assertEqual(split_run.fasta_stat(path), split_run.FastaStat(st.st_size, st.st_mtime_ns, st.st_ino))
        self.assertEqual(split_run.fasta_stat(path).bytes, 8)
        with open(path, "a") as f:
            f.write(">b\nKLMN\n")
        self.assertEqual(split_run.fasta_stat(path).bytes, 16)
        with self.assertRaises(OSError):
            split_run.fasta_stat(self.tmp / "missing.fasta")

    def test_preflight_counts(self):
        preflight = preflight_fasta(_write(self.tmp / "d.fasta", ">a\nACDX\n>a\nACDE\n>b\nKL\n"), 3)
        counts = split_run.preflight_counts(preflight)
        self.assertEqual(tuple(counts), split_run.PREFLIGHT_KEYS)
        self.assertEqual(counts, {"entries": 3, "entries_kept": 2, "residues": 8, "duplicates": 1,
                                  "unique_entries": 1, "unique_residues": 4, "nonstandard_entries": 1,
                                  "nonstandard_residues": 1, "non_ascii_entries": 0, "angle_bracket_entries": 0,
                                  "max_length": 4})

    def test_dealt_rows(self):
        self.assertEqual(split_run.dealt_rows(5, 2), {0: 3, 1: 2})
        self.assertEqual(split_run.dealt_rows(2, 3), {0: 1, 1: 1})
        self.assertEqual(split_run.dealt_rows(4, 4), {0: 1, 1: 1, 2: 1, 3: 1})
        self.assertEqual(split_run.dealt_rows(0, 4), {})
        self.assertEqual(split_run.dealt_rows(7, 1), {0: 7})

    def test_shard_entries(self):
        fasta = _write(self.tmp / "five.fasta", _FIVE)
        _, paths = self.write_shards("r-1", fasta, ranks=3)
        self.assertEqual(split_run.shard_entries(paths), [
            {"name": f"emb_r-1-{rank}.parquet", "rank": rank, "rows": rows, "bytes": paths[rank].stat().st_size}
            for rank, rows in ((0, 2), (1, 2), (2, 1))])
        bad = self.tmp / "emb_r-x.parquet"
        bad.touch()
        self.refused(lambda: split_run.shard_entries([bad]), "is not an embedding shard")
        (self.tmp / "emb_r-0.parquet").write_bytes(b"")
        self.refused(lambda: split_run.shard_entries([self.tmp / "emb_r-0.parquet"]), "not a readable Parquet")


# =====================================================================================================================
# the build lock
# =====================================================================================================================
class BuildLockTest(_Run):

    def setUp(self):
        super().setUp()
        self.db = str(self.tmp / "db")
        self.record = {"run_id": "r", "residue_output": "residue_codes"}

    def enter(self, record=None, **kwargs):
        """Take the lock (a refusal is raised here; a lock taken is released at cleanup)."""
        self.enterContext(split_run.build_lock(self.db, record or self.record, **kwargs))

    def test_content_and_removal(self):
        lock = Path(f"{self.db}.build.lock")
        with launch_env(SLURM_JOB_ID=4242), split_run.build_lock(self.db, self.record) as held:
            self.assertEqual(held, lock)
            content = json.loads(lock.read_text())
        self.assertEqual(set(content), {"run_id", "host", "pid", "slurm_job_id", "started_utc"})
        self.assertEqual((content["run_id"], content["host"], content["pid"], content["slurm_job_id"]),
                         ("r", socket.gethostname(), os.getpid(), "4242"))
        self.assertRegex(content["started_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertFalse(lock.exists())
        with split_run.build_lock(self.db, self.record):
            self.assertIsNone(json.loads(lock.read_text())["slurm_job_id"])

    def test_removed_on_an_exception(self):
        lock = Path(f"{self.db}.build.lock")
        for exc in (RefusalError("refused"), RuntimeError("bug"), KeyboardInterrupt()):
            with self.subTest(exc=type(exc).__name__), self.assertRaises(type(exc)):
                with split_run.build_lock(self.db, self.record):
                    self.assertTrue(lock.exists())
                    raise exc
            self.assertFalse(lock.exists())

    def test_exclusive(self):
        (self.tmp / "db.index").touch()
        (self.tmp / "db.rescodes.bin.tmp").touch()
        with split_run.build_lock(self.db, self.record):
            message = self.refused(self.enter,
                                   f"{self.db}.build.lock exists", '"run_id": "r"', f'"pid": {os.getpid()}',
                                   "is running, or was killed", "files present: db.index, db.rescodes.bin.tmp",
                                   "delete the lock and these files, then rerun")
            self.assertNotIn("build residue-codes", message)
            self.assertTrue(Path(f"{self.db}.build.lock").exists())
        self.assertFalse(Path(f"{self.db}.build.lock").exists())

    def test_a_lock_that_is_no_longer_ours_stays(self):
        lock = Path(f"{self.db}.build.lock")
        with split_run.build_lock(self.db, self.record):
            lock.unlink()
            lock.write_text("another build's")
        self.assertEqual(lock.read_text(), "another build's")

    def test_the_hint_when_only_the_code_store_is_missing(self):
        lock = Path(f"{self.db}.build.lock")
        lock.write_text("{}")
        for suffix in (".index", ".metadata"):
            Path(f"{self.db}{suffix}").touch()
        store = SequenceStore(self.tmp, "db")
        store.create(_FASTA)
        Path(f"{self.db}.rescodes.bin.tmp").touch()
        message = self.refused(lambda: self.enter(shards_folder=self.shards),
                               "files present: db.index, db.metadata, db.sequences, db.rescodes.bin.tmp",
                               f"'fm-search build residue-codes {self.tmp} db --from-shards {self.shards} "
                               f"--run-id r'")
        self.assertIn("complete but for its residue-code store", message)
        self.assertNotIn("build residue-codes",
                         self.refused(lambda: self.enter(record={"run_id": "r", "residue_output": None})))
        Path(f"{self.db}.rescodes.json").touch()
        self.assertNotIn("build residue-codes", self.refused(self.enter))
        Path(f"{self.db}.rescodes.json").unlink()
        con = sqlite3.connect(store.path)
        con.execute("DELETE FROM meta")  # a store whose build stopped before its totals
        con.commit()
        con.close()
        self.assertNotIn("build residue-codes", self.refused(self.enter))
        self.assertEqual(lock.read_text(), "{}")


class TorchFreeTest(unittest.TestCase):

    def test_importing_split_run_loads_no_torch_or_faiss(self):
        """The module is pyarrow, numpy and the standard library only (plan P8b, Module API): a fresh interpreter that
        imports it has loaded neither torch nor FAISS. (This test process has both loaded already.)"""
        src = str(Path(split_run.__file__).resolve().parents[2])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([src] + [p for p in [os.environ.get("PYTHONPATH")] if p])}
        code = ("import sys, foldmatch.search.split_run; "
                "print(sorted(m for m in ('torch', 'faiss') if m in sys.modules))")
        with tempfile.TemporaryDirectory() as cwd:
            run = subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, text=True,
                                 timeout=300)
        self.assertEqual(run.returncode, 0, run.stderr[-4000:])
        self.assertEqual(run.stdout.strip(), "[]", run.stderr[-4000:])


# =====================================================================================================================
# utils.dist.multi_rank_launch
# =====================================================================================================================
class MultiRankLaunchTest(unittest.TestCase):

    def test_launch_environments(self):
        cases = [
            ({}, None),
            ({"SLURM_JOB_ID": 7, "SLURM_PROCID": 0, "SLURM_NTASKS": 32}, None),  # the batch step of a 32-task job
            ({"SLURM_JOB_ID": 7, "SLURM_STEP_ID": 4294967294, "SLURM_PROCID": 0, "SLURM_NTASKS": 32}, None),
            ({"SLURM_JOB_ID": 7, "SLURM_STEP_ID": 0, "SLURM_PROCID": 0, "SLURM_STEP_NUM_TASKS": 2, "SLURM_NTASKS": 2},
             "SLURM_STEP_NUM_TASKS=2"),
            ({"SLURM_JOB_ID": 7, "SLURM_STEP_ID": 0, "SLURM_PROCID": 1, "SLURM_STEP_NUM_TASKS": 2, "SLURM_NTASKS": 2},
             "SLURM_PROCID=1"),
            ({"SLURM_JOB_ID": 7, "SLURM_STEP_ID": 3, "SLURM_PROCID": 0, "SLURM_STEP_NUM_TASKS": 1,
              "SLURM_NTASKS": 32}, None),  # srun -n 1 in a 32-task job
            ({"SLURM_JOB_ID": 7, "SLURM_STEP_ID": 3, "SLURM_PROCID": 0, "SLURM_NTASKS": 4}, "SLURM_NTASKS=4"),
            ({"WORLD_SIZE": 2}, "WORLD_SIZE=2"),
            ({"WORLD_SIZE": 1, "RANK": 0, "LOCAL_RANK": 0}, None),
            ({"RANK": 3, "WORLD_SIZE": 4}, "RANK=3"),
            ({"LOCAL_RANK": 1, "FOLDMATCH_RUN_ID": "r"}, "LOCAL_RANK=1"),
            ({"NODE_RANK": 2}, "NODE_RANK=2"),
            ({"WORLD_SIZE": "x", "RANK": "", "SLURM_NTASKS": "2"}, None),  # not whole numbers; no srun step
        ]
        for env, expected in cases:
            with self.subTest(env=env), launch_env(**env):
                self.assertEqual(multi_rank_launch(), expected)
                # The rule of bundle_fetch_allowed: one process exactly where a single-node run may fetch.
                self.assertEqual(bundle_fetch_allowed(1), expected is None)


if __name__ == "__main__":
    unittest.main()
