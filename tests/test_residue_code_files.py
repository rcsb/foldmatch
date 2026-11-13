"""P9.X: the residue-code files of ``fm-embedding from-sequences residue|chain --output-codes-folder``: their writers
(one file per id in csv and pt, one per rank in parquet and json) and the readers that must refuse them (the shard readers of
builds and queries, and the embedding readers of fm-search).

No model: the predictions are made up, as a module with the ``residue_codes`` output returns them.
"""
import gzip
import hashlib
import json
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch

from foldmatch.cli import search
from foldmatch.errors import RefusalError
from foldmatch.search import residue_shards as rs
from foldmatch.search import scoring_system as ss
from foldmatch.writer.batch_writer import (
    CsvBatchWriter, CsvCodeWriter, JsonCodeStorage, ParquetBatchWriter, ParquetCodeWriter, TensorBatchWriter,
    TensorCodeWriter, residue_code_tables,
)

SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)


def _prediction(ids, lengths, seed=0):
    rng = np.random.default_rng(seed)
    tables = [rng.integers(0, 256, (n, 8), dtype=np.uint8) for n in lengths]
    return (([torch.zeros(1)] * len(ids)), tuple(ids), tuple(t.tobytes() for t in tables),
            tuple(hashlib.sha1(i.encode()).digest() for i in ids)), tables


_CODES_MODULE = SimpleNamespace(residue_output=rs.RESIDUE_CODES, device=torch.device("cpu"))


class WriterTest(unittest.TestCase):

    def setUp(self):
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_parquet_ranks_and_row_groups(self):
        for rank in (0, 3):
            trainer = SimpleNamespace(global_rank=rank, world_size=4)
            writer = ParquetCodeWriter(str(self.out), "c", SYSTEM, min_res_n=0, byte_budget=100)
            writer.on_predict_start(trainer, _CODES_MODULE)
            ids = [f"r{rank}-{k}" for k in range(4)]
            prediction, tables = _prediction(ids, [10, 0, 3, 7], seed=rank)
            writer.write_on_batch_end(trainer, None, prediction, None, None, 0, 0)
            writer.write_on_batch_end(trainer, None, _prediction(["z"], [20])[0], None, None, 1, 0)
            writer.on_predict_end(trainer, None)
            path = self.out / f"c-{rank}.parquet"
            self.assertEqual(pq.ParquetFile(path).metadata.num_row_groups, 2)
            table = pq.read_table(path)
            self.assertEqual(table.column("id").to_pylist(), ids + ["z"])
            for got, want in zip(table.column("residue_codes").to_pylist(), tables):
                np.testing.assert_array_equal(np.array(got, dtype=np.uint8).reshape(-1, 8), want)
            self.assertEqual(json.loads(pq.read_schema(path).metadata[b"foldmatch.pass"])["world_size"], 4)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["c-0.parquet", "c-3.parquet"])

    def test_no_rows_no_parquet_file(self):
        writer = ParquetCodeWriter(str(self.out), "c", SYSTEM)
        trainer = SimpleNamespace(global_rank=0, world_size=1)
        writer.on_predict_start(trainer, _CODES_MODULE)
        writer.on_predict_end(trainer, None)
        self.assertEqual(list(self.out.iterdir()), [])

    def test_json_ranks(self):
        for rank in (0, 1):
            trainer = SimpleNamespace(global_rank=rank)
            writer = JsonCodeStorage(str(self.out), "c")
            prediction, tables = _prediction(["a", "b"], [2, 0])
            writer.write_on_batch_end(trainer, None, prediction, None, None, 0, 0)
            writer.on_predict_end(trainer, None)
            records = json.loads(gzip.decompress((self.out / f"c-{rank}.json.gz").read_bytes()))
            self.assertEqual(records, [{"id": "a", "residue_codes": tables[0].tolist()},
                                       {"id": "b", "residue_codes": []}])

    def test_item_writers(self):
        """csv and pt: <codes folder>/<id>.<format>, as the embedding of the id is <output folder>/<id>.<format>; the
        layouts; a file that the folder takes for one written twice is refused, naming the residue codes."""
        trainer = SimpleNamespace(global_rank=0)
        prediction, tables = _prediction(["a", "b"], [3, 0])
        for writer, fmt in ((CsvCodeWriter, "csv"), (TensorCodeWriter, "pt")):
            with self.subTest(fmt=fmt):
                out = Path(tempfile.mkdtemp(dir=self.out))
                codes = writer(str(out))
                codes.on_predict_start(trainer, _CODES_MODULE)
                codes.write_on_batch_end(trainer, None, prediction, None, None, 0, 0)
                self.assertEqual(sorted(p.name for p in out.iterdir()), [f"a.{fmt}", f"b.{fmt}"])
                if fmt == "csv":
                    self.assertEqual((out / "a.csv").read_text(),
                                     "".join(",".join(map(str, row)) + "\n" for row in tables[0]))
                    self.assertEqual((out / "b.csv").read_bytes(), b"")
                else:
                    for dom_id, table in zip("ab", tables):
                        got = torch.load(out / f"{dom_id}.pt")
                        self.assertEqual((got.dtype, got.device.type), (torch.uint8, "cpu"))
                        np.testing.assert_array_equal(got.numpy(), table)
                with self.assertRaises(RefusalError) as refused:
                    codes.write_on_batch_end(trainer, None, _prediction(["a"], [2])[0], None, None, 1, 0)
                text = str(refused.exception)
                self.assertTrue(text.startswith(
                    f"{out}/a.{fmt} would be written twice by this run, for 'a' and then for 'a': the first is kept"),
                    text)
                self.assertIn("Each residue-code table of a run needs a file of its own: write them as parquet or "
                              "json", text)

    def test_a_code_file_name_that_cannot_be(self):
        trainer = SimpleNamespace(global_rank=0)
        long_id = "x" * 260
        with self.assertRaises(RefusalError) as refused:
            CsvCodeWriter(str(self.out)).write_on_batch_end(trainer, None, _prediction([long_id], [3])[0], None, None,
                                                            0, 0)
        text = str(refused.exception)
        self.assertTrue(text.startswith(f"The residue codes of {long_id!r} cannot be written to {self.out}"), text)
        self.assertTrue(text.endswith("or give it another name."), text)
        self.assertIn("Write the residue codes as parquet or json, which keep each under its id", text)
        # The embedding writers' refusals are those of before
        plain = CsvBatchWriter(str(self.out))
        with self.assertRaises(RefusalError) as refused:
            plain.write_on_batch_end(trainer, None, ([torch.ones(2)], ("y" * 260,)), None, None, 0, 0)
        self.assertTrue(str(refused.exception).startswith("The embedding 'yyy"), str(refused.exception))
        self.assertIn("Write the embeddings as parquet or json, which keep each under its id", str(refused.exception))
        self.assertTrue(str(refused.exception).endswith("or give it another name."), str(refused.exception))
        plain.write_on_batch_end(trainer, None, ([torch.ones(2)], ("e",)), None, None, 0, 0)
        with self.assertRaises(RefusalError) as refused:
            plain.write_on_batch_end(trainer, None, ([torch.ones(2)], ("e",)), None, None, 1, 0)
        self.assertTrue(str(refused.exception).endswith(
            "Each embedding of a run needs a file of its own: write them as parquet or json, which keep each under "
            "its id, or, for names that differ only in case, to a folder of a case-sensitive file system."),
            str(refused.exception))

    def test_writers_refuse_a_module_without_codes(self):
        trainer = SimpleNamespace(global_rank=0, world_size=1)
        for writer in (ParquetCodeWriter(str(self.out), "c", SYSTEM), JsonCodeStorage(str(self.out), "c"),
                       CsvCodeWriter(str(self.out)), TensorCodeWriter(str(self.out))):
            with self.subTest(writer=type(writer).__name__):
                with self.assertRaisesRegex(ValueError, "the module produces no residue output"):
                    writer.on_predict_start(trainer, SimpleNamespace(residue_output=None))
                with self.assertRaisesRegex(ValueError, "the module produces query_encoding"):
                    writer.on_predict_start(trainer, SimpleNamespace(residue_output=rs.QUERY_ENCODING))

    def test_residue_code_tables(self):
        with self.assertRaisesRegex(ValueError, "got a 2-tuple"):
            residue_code_tables(([], ()))
        with self.assertRaisesRegex(ValueError, "not a whole number of 8-byte codes"):
            residue_code_tables(([None], ("a",), (b"123",), (bytes(20),)))
        with self.assertRaisesRegex(ValueError, "seq_sha1 must be 20 bytes"):
            residue_code_tables(([None], ("a",), (bytes(8),), (b"x",)))
        ids, tables, _ = residue_code_tables(([None], ("a",), (bytes(range(16)),), (bytes(20),)))
        self.assertTrue(tables[0].flags.writeable)
        np.testing.assert_array_equal(tables[0], np.arange(16, dtype=np.uint8).reshape(2, 8))



def write_run(folder: Path, codes_name: str = "inference-codes", embeddings: bool = True):
    """The Parquet files of a chain run of ids a and b with --output-codes-folder, both in ``folder`` (as a user who
    moved them there would have them): inference-0.parquet (legacy chain embeddings, unless not ``embeddings``) and
    ``<codes_name>-0.parquet``."""
    trainer = SimpleNamespace(global_rank=0, world_size=1)
    prediction, tables = _prediction(["a", "b"], [3, 5])
    if embeddings:
        writer = ParquetBatchWriter(str(folder), "inference")
        writer.write_on_batch_end(trainer, None, ([torch.ones(4), torch.zeros(4)],) + prediction[1:], None, None, 0, 0)
        writer.on_predict_end(trainer, None)
    writer = ParquetCodeWriter(str(folder), codes_name, SYSTEM, min_res_n=0)
    writer.on_predict_start(trainer, _CODES_MODULE)
    writer.write_on_batch_end(trainer, None, prediction, None, None, 0, 0)
    writer.on_predict_end(trainer, None)
    return tables


# =====================================================================================================================
# The readers of shards and of embeddings refuse a code file
# =====================================================================================================================
class ReaderTest(unittest.TestCase):

    def setUp(self):
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_shard_readers_refuse_a_code_file(self):
        shards = self.out / "shards"
        shards.mkdir()
        write_run(shards, codes_name="emb_p9x-run", embeddings=False)
        shard = shards / "emb_p9x-run-0.parquet"
        info = rs.read_shard_info(shard)
        self.assertEqual(info.residue_output, rs.RESIDUE_CODE_TABLE)
        for kind in rs.RESIDUE_OUTPUTS:
            with self.subTest(kind=kind), self.assertRaisesRegex(rs.ResidueShardError, "not an embedding shard"):
                rs.check_shards([shard], kind, SYSTEM)
        with self.assertRaisesRegex(rs.ResidueShardError, "not an embedding shard"):
            rs.iter_residue_codes([shard], SYSTEM)
        from foldmatch.search.query_encodings import QueryEncodingIndex
        with self.assertRaisesRegex(rs.ResidueShardError, "not an embedding shard"):
            QueryEncodingIndex.build([shard], shards / "q.bin", system=SYSTEM)
        self.assertFalse((shards / "q.bin").exists())
        with self.assertRaisesRegex(RefusalError, "None of the embedding shards .* hold residue codes"):
            search._residue_code_shards(shards, None)
        from foldmatch.search import split_run
        with self.assertRaisesRegex(RefusalError, "not an embedding shard"):
            split_run.check_codes_footers([shard], system=SYSTEM, min_res_n=0, world_size=1)
        with self.assertRaisesRegex(RefusalError, "not the residue codes of"):
            split_run.check_codes_footers([shard], expect={"name": SYSTEM.name, "digest": SYSTEM.digest(),
                                                           "encoding_digest": SYSTEM.encoding_digest()},
                                          min_res_n=0, world_size=1)
        with self.assertRaisesRegex(RefusalError, "not an embedding shard|not plain|not the residue codes"):
            search._seal_footers([shard], folder=shards, run_id="p9x-run", min_res_n=None)
        with self.assertRaises(RefusalError):
            split_run.check_plain_footers([shard])

    def test_embedding_readers_refuse_a_code_file(self):
        from foldmatch.utils.data import refuse_repeated_query_ids, stream_embeddings
        out = self.out
        write_run(out)
        refuse_repeated_query_ids(out)   # the ids of the code file are not query ids
        with self.assertRaises(RefusalError) as refused:
            list(stream_embeddings(out))
        self.assertEqual(str(refused.exception), (
            f"{out / 'inference-codes-0.parquet'} holds residue codes, as 'fm-embedding from-sequences ... "
            f"--output-codes-folder' writes them (one table of 8-byte codes per sequence), not embeddings: residue "
            f"codes belong in a folder of their own (--output-codes-folder); move them out of the embedding folder."))
        self.assertEqual([ids for ids, _ in stream_embeddings(out / "inference-0.parquet")], [["a", "b"]])

    def test_embedding_readers_refuse_code_csv_and_pt_files(self):
        """fm-search's .pt/.csv embedding reader on code files: a .pt by its type, whatever L (a one-residue code
        table would otherwise be an 8-value vector); a .csv table with a word on residue codes. A one-residue .csv
        cannot be told from an 8-value vector (a question for the user)."""
        from foldmatch.utils.data import _file_vector
        trainer = SimpleNamespace(global_rank=0)
        prediction, _ = _prediction(["a", "b", "c"], [5, 1, 0])
        for writer, fmt in ((TensorCodeWriter, "pt"), (CsvCodeWriter, "csv")):
            out = Path(tempfile.mkdtemp(dir=self.out))
            writer(str(out)).write_on_batch_end(trainer, None, prediction, None, None, 0, 0)
            for dom_id, rows in (("a", 5), ("b", 1), ("c", 0)):
                with self.subTest(fmt=fmt, rows=rows):
                    path = out / f"{dom_id}.{fmt}"
                    if fmt == "csv" and rows == 1:
                        self.assertEqual(_file_vector(path).shape, (8,))   # not recognisable
                        continue
                    with self.assertRaises(RefusalError) as refused:
                        _file_vector(path)
                    text = str(refused.exception)
                    if fmt == "pt":
                        self.assertIn(f"holds residue codes (a uint8 tensor of {rows} x 8), as 'fm-embedding "
                                      f"from-sequences ... --output-codes-folder' writes them, not an embedding "
                                      f"vector: residue codes belong in a folder of their own "
                                      f"(--output-codes-folder); move them out of the embedding folder.", text)
                    elif rows == 5:
                        self.assertIn("holds a table of 5 x 8 values, not one embedding vector", text)
                        self.assertIn("A table of 8 columns of integers 0-255 may be residue codes, as "
                                      "'fm-embedding from-sequences ... --output-codes-folder' writes them (<id>.csv): "
                                      "they are not embeddings either; residue codes belong in a folder of their own "
                                      "(--output-codes-folder): move them out of the embedding folder.", text)
                    else:
                        self.assertIn("is not a CSV of numbers", text)
        # Embeddings are read as before: float tables are not called codes, float vectors and an 8-value uint8
        # vector are vectors
        out = Path(tempfile.mkdtemp(dir=self.out))
        torch.save(torch.rand(3, 8), out / "f.pt")
        with self.assertRaises(RefusalError) as refused:
            _file_vector(out / "f.pt")
        self.assertNotIn("residue codes", str(refused.exception))
        torch.save(torch.arange(8, dtype=torch.uint8), out / "v.pt")
        np.testing.assert_array_equal(_file_vector(out / "v.pt"), np.arange(8, dtype=np.float32))


# =====================================================================================================================
# What an output file holds (foldmatch.utils.file_kinds)
# =====================================================================================================================
class FileKindTest(unittest.TestCase):
    """Positive identification of each writer's files: embeddings of every writer and format, residue codes of every
    code writer; anything else (empty, cut short, corrupt, foreign) of no kind."""

    def setUp(self):
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def written(self) -> dict:
        """{file: kind} of every writer, a and b (3 and 0 residues) in each."""
        trainer = SimpleNamespace(global_rank=0, world_size=1)
        prediction, _ = _prediction(["a", "b"], [3, 0])
        chain = ([torch.rand(6), torch.rand(6)],) + prediction[1:]
        residue = ([torch.rand(5, 6), torch.rand(2, 6)],) + prediction[1:]
        files = {}

        def run(writer, pred, folder, kind, start=None):
            folder = self.out / folder
            folder.mkdir()
            writer.out_path = str(folder)
            if start is not None:
                writer.on_predict_start(trainer, start)
            writer.write_on_batch_end(trainer, None, pred, None, None, 0, 0)
            if hasattr(writer, "on_predict_end"):
                writer.on_predict_end(trainer, None)
            files.update({path: kind for path in folder.iterdir()})

        from foldmatch.utils.file_kinds import EMBEDDINGS, RESIDUE_CODES
        from foldmatch.writer.batch_writer import JsonStorage
        for name, pred in (("chain", chain), ("residue", residue)):
            run(CsvBatchWriter("."), pred, f"{name}-csv", EMBEDDINGS)
            run(TensorBatchWriter("."), pred, f"{name}-pt", EMBEDDINGS)
            run(JsonStorage(".", "e"), pred, f"{name}-json", EMBEDDINGS)
        run(ParquetBatchWriter(".", "e"), chain, "chain-parquet", EMBEDDINGS)
        run(ParquetBatchWriter(".", "e", residue_embeddings=True), residue, "residue-parquet", EMBEDDINGS)
        run(ParquetBatchWriter(".", "e", residue_output=rs.RESIDUE_CODES, system=SYSTEM, min_res_n=0),
            chain + (), "shard-parquet", EMBEDDINGS, _CODES_MODULE)
        run(CsvCodeWriter("."), prediction, "codes-csv", RESIDUE_CODES, _CODES_MODULE)
        run(TensorCodeWriter("."), prediction, "codes-pt", RESIDUE_CODES, _CODES_MODULE)
        run(JsonCodeStorage(".", "c"), prediction, "codes-json", RESIDUE_CODES, _CODES_MODULE)
        run(ParquetCodeWriter(".", "c", SYSTEM), prediction, "codes-parquet", RESIDUE_CODES, _CODES_MODULE)
        # b's code table has no residue: an empty csv is of no kind, an empty [0, 8] tensor is residue codes
        files[self.out / "codes-csv" / "b.csv"] = None
        return files

    def test_every_writer(self):
        from foldmatch.utils.file_kinds import file_kind
        files = self.written()
        self.assertEqual(len(files), 19)
        for path, kind in files.items():
            with self.subTest(file=f"{path.parent.name}/{path.name}"):
                self.assertEqual(file_kind(path), kind)

    def test_a_file_cut_short_is_never_of_the_other_kind(self):
        """A prefix of a file (a file that a writer is writing, or that was cut short) is of its kind or of none:
        never of the other kind; a Parquet or .pt file is of none until whole."""
        from foldmatch.utils import file_kinds
        for path, kind in self.written().items():
            data = path.read_bytes()
            cut = path.parent / f"cut-{path.name}"
            for size in sorted({0, 1, 2, 5, 10, 20, 31, 64, 100, len(data) // 2, len(data) - 1}):
                if size >= len(data):
                    continue
                with self.subTest(file=f"{path.parent.name}/{path.name}", size=size):
                    cut.write_bytes(data[:size])
                    got = file_kinds.file_kind(cut)
                    self.assertIn(got, (kind, None))
                    if path.suffix in (".parquet", ".pt"):
                        self.assertIsNone(got)

    def test_foreign_files_are_of_no_kind(self):
        from foldmatch.utils.file_kinds import file_kind
        import pyarrow as pa
        foreign = {
            "random.pt": bytes(range(256)) * 4, "random.parquet": b"PAR1" + bytes(100) + b"PAR1",
            "random.csv": bytes(range(256)), "random.json.gz": bytes(range(256)), "text.json.gz": b"[]",
            "header.csv": b"a,b,c\n0.5,1.5,2.5\n", "integers.csv": b"1,2,3\n4,5,6\n",
            "nine.csv": b"1,2,3,4,5,6,7,8,9\n", "big.csv": b"1,2,3,4,5,6,7,256\n", "word.csv": b"x\n",
            "spaces.csv": b" 0.5, 1.5\n", "codes-no-newline.csv": b"1,2,3,4,5,6,7,8",
            "other.txt": b"0.5\n",
        }
        for name, data in foreign.items():
            (self.out / name).write_bytes(data)
        (self.out / "empty.json.gz").write_bytes(gzip.compress(b"[]"))
        (self.out / "other-keys.json.gz").write_bytes(gzip.compress(b'[{"id":"a","x":[1]}]'))
        (self.out / "scalar.json.gz").write_bytes(gzip.compress(b'[{"id":"a","embedding":1}]'))
        torch.save({"a": torch.rand(3)}, self.out / "dict.pt")
        torch.save(torch.arange(6, dtype=torch.int64), self.out / "int.pt")
        torch.save(torch.zeros(3, 9, dtype=torch.uint8), self.out / "uint8-9.pt")
        torch.save(torch.rand(3), self.out / "legacy.pt", _use_new_zipfile_serialization=False)
        pq.write_table(pa.table({"id": ["a"], "embedding": [["x"]]}), self.out / "strings.parquet")
        pq.write_table(pa.table({"id": ["a"], "vector": [[0.5]]}), self.out / "vector.parquet")
        pq.write_table(pa.table({"id": ["a"], "embedding": [0.5]}), self.out / "scalar.parquet")
        for path in sorted(self.out.iterdir()):
            with self.subTest(file=path.name):
                self.assertIsNone(file_kind(path))
        # A folder named like a file, a missing file, a link to a file is the file
        (self.out / "folder.pt").mkdir()
        self.assertIsNone(file_kind(self.out / "folder.pt"))
        self.assertIsNone(file_kind(self.out / "missing.csv"))
        torch.save(torch.rand(3), self.out / "e.pt")
        (self.out / "link.pt").symlink_to(self.out / "e.pt")
        self.assertEqual(file_kind(self.out / "link.pt"), "embeddings")

    def test_the_suffix_in_any_case(self):
        """A case-insensitive folder takes X.PT for the file x.pt that a run would write: read as x.pt is."""
        from foldmatch.utils.file_kinds import EMBEDDINGS, RESIDUE_CODES, describe, file_kind
        for path, kind in self.written().items():
            if kind is None:
                continue
            upper = path.with_name(path.name.upper())
            with self.subTest(file=f"{path.parent.name}/{upper.name}"):
                shutil.copyfile(path, path.parent / f"x-{upper.name}")
                self.assertEqual(file_kind(path.parent / f"x-{upper.name}"), kind)
        self.assertEqual(describe("A.PT", EMBEDDINGS), "an embedding")
        self.assertEqual(describe("A.Csv", EMBEDDINGS), "an embedding")
        self.assertEqual(describe("A.PARQUET", EMBEDDINGS), "embeddings")
        self.assertEqual(describe("A.PT", RESIDUE_CODES), "residue codes")

    def test_a_file_is_read_once_from_one_check_to_the_next(self):
        """Within remembering() a file's kind is remembered while the file is unchanged; the next check, outside it,
        takes it from memory once and forgets it; nothing is remembered outside it, or after a refusal within it."""
        from foldmatch.utils import file_kinds
        path = self.out / "x.pt"
        torch.save(torch.rand(3), path)
        file_kinds._KNOWN.clear()
        with unittest.mock.patch.dict(file_kinds._KINDS, {".pt": unittest.mock.Mock(wraps=file_kinds.pt_kind)}):
            read = file_kinds._KINDS[".pt"]
            # Outside remembering(): each check reads, nothing is kept
            self.assertEqual(file_kinds.file_kind(path), file_kinds.EMBEDDINGS)
            self.assertEqual(file_kinds.file_kind(path), file_kinds.EMBEDDINGS)
            self.assertEqual((read.call_count, file_kinds._KNOWN), (2, {}))
            # The command's check, then the library's: one read
            with file_kinds.remembering():
                self.assertEqual(file_kinds.file_kind(path), file_kinds.EMBEDDINGS)
                self.assertEqual(file_kinds.file_kind(path), file_kinds.EMBEDDINGS)
            self.assertEqual(read.call_count, 3)
            self.assertEqual(file_kinds.file_kind(path), file_kinds.EMBEDDINGS)
            self.assertEqual((read.call_count, file_kinds._KNOWN), (3, {}))
            self.assertEqual(file_kinds.file_kind(path), file_kinds.EMBEDDINGS)
            self.assertEqual(read.call_count, 4)
            # A file changed since it was remembered is read again
            with file_kinds.remembering():
                self.assertEqual(file_kinds.file_kind(path), file_kinds.EMBEDDINGS)
            torch.save(torch.zeros(4, 8, dtype=torch.uint8), path)
            self.assertEqual(file_kinds.file_kind(path), file_kinds.RESIDUE_CODES)
            self.assertEqual(read.call_count, 6)
            # A check that refuses ends the run: what it read is forgotten
            file_kinds._KNOWN.clear()
            with self.assertRaises(RuntimeError), file_kinds.remembering():
                file_kinds.file_kind(path)
                raise RuntimeError("refused")
            self.assertEqual(file_kinds._KNOWN, {})


if __name__ == "__main__":
    unittest.main()
