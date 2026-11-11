"""P9.B4: a Parquet row holds one vector, and the embeddings read have one dimension.

stream_embeddings, the reader of 'fm-search build embeddings' and 'fm-search query embedding', reshaped each batch of a
Parquet file's 'embedding' column into one row per id. Residue embeddings flattened one table per row (rows of
different lengths) were cut into equal slices that mixed chains, or stopped on a numpy error that named no file, and a
null row was sliced the same way. Every row of a file must now hold as many values as its first row, across batches; a
row with no values, NaN or infinite values and an 'embedding' column that is not a list of floats are refused, naming
the file and the row or the type. A file that cannot be read as Parquet, or without the column id or embedding, is
refused, naming the file: it stopped on pyarrow's message, which named no file, or on a traceback. The embeddings read
have one dimension, the first one's at build and the database's at query: two dimensions stopped a build on numpy's
concatenate error, and a query on an empty FAISS AssertionError. A .pt and a .csv of one stem, two embeddings under one
id, are refused."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from foldmatch.cli import search
from foldmatch.errors import RefusalError
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.utils.data import stream_embeddings
from foldmatch.writer.batch_writer import CsvBatchWriter, ParquetBatchWriter, TensorBatchWriter

_RESOURCES = Path(__file__).parent / "resources"
_ROW_ADVICE = ("A row holds one embedding vector; rows of different lengths are usually residue embeddings flattened "
               "by an earlier FoldMatch: compute chain embeddings with 'fm-embedding from-structures chain' or "
               "'fm-embedding from-sequences chain'.")
_ONE_DIMENSION = "all embeddings of a database have one dimension."
_DATABASE_DIMENSION = "query it with embeddings of the kind it was built from."


def _vectors(dim: int, prefix: str = "c", n: int = 5, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {f"{prefix}{i}.A": torch.randn(dim, generator=generator) for i in range(n)}


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def _residues() -> dict:
    """The ESM3 residue embeddings of chains 1acb.A (243 x 1536) and 1acb.B (65 x 1536), BOS and EOS rows included."""
    return {name: _load(_RESOURCES / "embeddings" / f"{name}.pt") for name in ("1acb.A", "1acb.B")}


def _residues_64() -> dict:
    """Residue embeddings of 64 values per residue of chains r7 (7 residues, 448 values) and r5 (5 residues, 320)."""
    generator = torch.Generator().manual_seed(1)
    return {"r7": torch.randn(7, 64, generator=generator), "r5": torch.randn(5, 64, generator=generator)}


def _chains() -> dict:
    """The FoldMatch chain embeddings of the five test chains, 1-D [1536], in file name order."""
    return {path.stem: _load(path) for path in sorted((_RESOURCES / "chain_embeddings").glob("*.pt"))}


def _parquet(path: Path, rows: list, ids: list, kind=pa.list_(pa.float32())) -> Path:
    """A Parquet file of the columns id and embedding written with pyarrow, for rows that FoldMatch does not write."""
    pq.write_table(pa.table({"id": ids, "embedding": pa.array(rows, type=kind)}), path)
    return path


def _flat_residues(path: Path, tables: dict) -> Path:
    """Residue embeddings in Parquet as FoldMatch wrote them: one row per chain, its [L, D] table flattened into a row
    of list<float32> (no shape stored)."""
    return _parquet(path, [table.numpy().ravel() for table in tables.values()], list(tables))


def _read(path: Path, **kwargs) -> tuple:
    """The ids (in order), the [N, D] vectors and the batch shapes that stream_embeddings reads from ``path``."""
    ids, batches = [], []
    for batch_ids, batch in stream_embeddings(path, **kwargs):
        ids.extend(batch_ids)
        batches.append(batch)
    return ids, np.concatenate(batches), [batch.shape for batch in batches]


class ParquetEmbeddingsTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def folder(self, name: str) -> Path:
        folder = self.tmp / name
        folder.mkdir()
        return folder

    def shard(self, folder: Path, embeddings: dict, df_id: str = "inference", rank: int = 0) -> Path:
        """The shard {df_id}-{rank}.parquet that fm-embedding and inference write to Parquet, through their writer."""
        writer = ParquetBatchWriter(str(folder), df_id)
        trainer = SimpleNamespace(global_rank=rank)
        writer.write_on_batch_end(trainer, None, (list(embeddings.values()), list(embeddings)), None, None, 0, 0)
        writer.on_predict_end(trainer, None)
        return folder / f"{df_id}-{rank}.parquet"

    def files(self, writer, folder: Path, embeddings: dict) -> Path:
        """The .pt or .csv files that fm-embedding and inference write, through their writer."""
        files = writer(str(folder))
        for dom_id, embedding in embeddings.items():
            files._write_embedding(embedding, dom_id)
        return folder

    def refusal(self, path: Path, **kwargs) -> str:
        """The message of the RefusalError with which stream_embeddings refuses the file or folder ``path``."""
        with self.assertRaises(RefusalError) as refused:
            _read(path, **kwargs)
        return str(refused.exception)

    def test_chain_parquet_is_read_bit_for_bit(self):
        """Chain or assembly embeddings as ParquetBatchWriter writes them (D 1536 and 64) read bit for bit and in
        order, at the default batch size and at 2 and 1, which split the row group."""
        for dim in (1536, 64):
            vectors = _vectors(dim)
            folder = self.folder(f"d{dim}")
            self.shard(folder, vectors)
            for batch_size, rows in ((32768, [5]), (2, [2, 2, 1]), (1, [1] * 5)):
                with self.subTest(dim=dim, batch_size=batch_size):
                    ids, read, shapes = _read(folder, batch_size=batch_size)
                    self.assertEqual(ids, list(vectors))
                    self.assertEqual(shapes, [(n, dim) for n in rows])
                    self.assertEqual(read.dtype, np.float32)
                    np.testing.assert_array_equal(read, np.stack([v.numpy() for v in vectors.values()]))

    def test_other_lists_of_floats_are_read(self):
        """An 'embedding' column of large lists, of fixed-size lists or of doubles holds one vector per row too, read
        as float32 as before."""
        vectors = _vectors(64)
        rows = [v.numpy() for v in vectors.values()]
        folder = self.folder("kinds")
        for name, kind in (("large", pa.large_list(pa.float32())), ("fixed", pa.list_(pa.float32(), 64)),
                           ("double", pa.list_(pa.float64()))):
            with self.subTest(kind=str(kind)):
                ids, read, _ = _read(_parquet(folder / f"{name}.parquet", rows, list(vectors), kind))
                self.assertEqual(ids, list(vectors))
                np.testing.assert_array_equal(read, np.stack(rows))

    def test_residue_rows_are_refused(self):
        """Residue embeddings flattened one table per row are refused, naming the file and two rows of different
        lengths, instead of cut into equal slices (r7 and r5 made 2 vectors of 384 values): also with batch size 1,
        since the file's first row sets the length of all its batches. When each file holds one chain, the files
        differ in dimension. After a chain shard, the row lengths of the file's first batch are checked before its
        dimension, so the residue advice is given, unless that batch holds one row (batch size 1)."""
        tables = {"esm3": _residues(), "width-64": _residues_64()}
        for name, residues in tables.items():
            path = _flat_residues(self.folder(name) / "inference-0.parquet", residues)
            (first, a), (second, b) = residues.items()
            expected = f"{path}: row {first} has {a.numel()} values, but row {second} has {b.numel()}. {_ROW_ADVICE}"
            for batch_size in (32768, 1):
                with self.subTest(tables=name, batch_size=batch_size):
                    self.assertEqual(self.refusal(path.parent, batch_size=batch_size), expected)
        folder = self.folder("one-chain-per-file")
        files = [_flat_residues(folder / f"inference-{rank}.parquet", {name: table})
                 for rank, (name, table) in enumerate(_residues().items())]
        with self.subTest(tables="one chain per file"):
            self.assertEqual(self.refusal(folder), f"{files[1]} (row 1acb.B) has 99840 values, but {files[0]} (row "
                                                   f"1acb.A) has 373248: {_ONE_DIMENSION}")
        folder = self.folder("after-a-chain-shard")
        self.shard(folder, _vectors(64), df_id="a")
        path = _flat_residues(folder / "inference-0.parquet", tables["width-64"])
        with self.subTest(tables="after a chain shard"):
            self.assertEqual(self.refusal(folder), f"{path}: row r7 has 448 values, but row r5 has 320. {_ROW_ADVICE}")
            # at batch size 1 the first batch holds one row, whose length is refused as a dimension
            self.assertEqual(self.refusal(folder, batch_size=1), f"{path} (row r7) has 448 values, but "
                                                                 f"{folder / 'a-0.parquet'} (row c0.A) has 64: "
                                                                 f"{_ONE_DIMENSION}")

    def test_rows_without_values_or_with_nan_are_refused(self):
        """A row with no embedding (null, or an empty list) and a row with NaN or infinite values (a NaN chain
        embedding as ParquetBatchWriter writes it, infinity, a null inside the list) are refused, naming the file
        and the row: the batch was reshaped around a null row, and a NaN embedding is never found."""
        vectors = _vectors(64, n=3)
        rows = [v.numpy() for v in vectors.values()]
        folder = self.folder("rows")
        infinite = rows[1].copy()
        infinite[[3, 5]] = np.inf
        hole = rows[1].tolist()
        hole[7] = None
        never = "such an embedding is never found, and as a query it finds nothing."
        cases = {"null": (_parquet(folder / "null.parquet", [rows[0], None, rows[2]], list(vectors)),
                          "row c1.A has no embedding."),
                 "empty": (_parquet(folder / "empty.parquet", [rows[0], [], rows[2]], list(vectors)),
                           "row c1.A has no embedding."),
                 "nan": (self.shard(self.folder("nan"), dict(vectors, **{"c1.A": torch.full((64,), float("nan"))})),
                         f"row c1.A holds 64 NaN or infinite value(s) among its 64: {never}"),
                 "infinity": (_parquet(folder / "infinity.parquet", [rows[0], infinite, rows[2]], list(vectors)),
                              f"row c1.A holds 2 NaN or infinite value(s) among its 64: {never}"),
                 "null value": (_parquet(folder / "hole.parquet", [rows[0], hole, rows[2]], list(vectors)),
                                f"row c1.A holds 1 NaN or infinite value(s) among its 64: {never}")}
        for name, (path, text) in cases.items():
            with self.subTest(row=name):
                self.assertEqual(self.refusal(path), f"{path}: {text}")

    def test_a_column_of_another_type_is_refused(self):
        """An 'embedding' column that is not a list of floating-point numbers is refused, naming the file and the
        type."""
        folder = self.folder("types")
        for name, rows, kind, stored in (("text", ["0.1,0.2,0.3"], pa.string(), "string"),
                                         ("integers", [[1, 2, 3]], pa.list_(pa.int64()), "list<element: int64>")):
            with self.subTest(kind=stored):
                path = _parquet(folder / f"{name}.parquet", rows, ["c0.A"], kind)
                self.assertEqual(self.refusal(path), f"{path}: its 'embedding' column is of type {stored}, not a list "
                                                     f"of floating-point numbers: a row holds one embedding vector.")

    def test_a_file_that_cannot_be_read_or_lacks_a_column_is_refused(self):
        """A Parquet file that cannot be read and one without the column id or embedding are refused, naming the
        file: an empty file after a chain shard, a shard cut short (its first half, as an interrupted run leaves it:
        the footer is written last), a shard with a corrupted page (its first page header zeroed) and a shard's rows
        under the columns name and vector, or without id. They stopped on pyarrow's message, which names no file, or
        on a traceback: an OSError, or a KeyError (iter_batches drops a missing column without a word). A folder named
        *.parquet, such as the dataset folder that Spark writes, is refused as a folder, and a broken link with the
        system's reason, which pyarrow's message gives after the whole path."""
        shard = self.shard(self.folder("shard"), _vectors(64))
        data, table = shard.read_bytes(), pq.read_table(shard)
        empty = self.folder("empty")
        self.shard(empty, _vectors(64), df_id="a")
        (empty / "inference-0.parquet").write_bytes(b"")
        folder = self.folder("files")
        cut, corrupted = folder / "cut.parquet", folder / "corrupted.parquet"
        cut.write_bytes(data[:len(data) // 2])
        corrupted.write_bytes(data[:4] + bytes(16) + data[20:])  # the first page header follows the magic bytes PAR1
        renamed, no_id = folder / "renamed.parquet", folder / "no-id.parquet"
        pq.write_table(table.rename_columns(["name", "vector"]), renamed)
        pq.write_table(table.drop_columns(["id"]), no_id)
        dataset = self.folder("dataset")
        (dataset / "embeddings.parquet").mkdir()
        broken = self.folder("broken")
        (broken / "inference-0.parquet").symlink_to(broken / "moved.parquet")
        again = "Write it again: an interrupted run or a damaged copy leaves an unreadable file."
        columns = "a Parquet embedding file holds the columns id and embedding."
        cases = {"empty": (empty, f"{empty / 'inference-0.parquet'} could not be read as Parquet (Parquet file size is "
                                  f"0 bytes). {again}"),
                 "cut short": (cut, f"{cut} could not be read as Parquet (Parquet magic bytes not found in footer. "
                                    f"Either the file is corrupted or this is not a parquet file.). {again}"),
                 "corrupted page": (corrupted, f"{corrupted} could not be read as Parquet (Couldn't deserialize "
                                               f"thrift: TProtocolException: Invalid data). {again}"),
                 "name and vector": (renamed, f"{renamed} has no 'id' or 'embedding' column: {columns}"),
                 "no id": (no_id, f"{no_id} has no 'id' column: {columns}"),
                 "folder": (dataset, f"{dataset / 'embeddings.parquet'} is a folder, not a Parquet file."),
                 "broken link": (broken, f"{broken / 'inference-0.parquet'} could not be opened (No such file or "
                                         f"directory).")}
        for name, (path, text) in cases.items():
            with self.subTest(file=name):
                self.assertEqual(self.refusal(path), text)

    def test_two_dimensions_are_refused(self):
        """Embeddings of two dimensions are refused, naming both, a Parquet file with its first row (Parquet files
        are read first, then .pt, then .csv), and a build writes no database: it stopped on numpy's concatenate
        error, which names no file."""
        two_shards = self.folder("two-shards")
        a = self.shard(two_shards, _vectors(1536, "a"), df_id="a")
        b = self.shard(two_shards, _vectors(64, "b"), df_id="b")
        shard_and_pt = self.folder("shard-and-pt")
        c = self.shard(shard_and_pt, _vectors(1536, "c"), df_id="c")
        self.files(TensorBatchWriter, shard_and_pt, {"x.A": _vectors(64)["c0.A"]})
        pt_and_csv = self.folder("pt-and-csv")
        self.files(TensorBatchWriter, pt_and_csv, {"y.A": _vectors(1536)["c0.A"]})
        self.files(CsvBatchWriter, pt_and_csv, {"z.A": _vectors(64)["c0.A"]})
        cases = {"two shards": (two_shards, f"{b} (row b0.A) has 64 values, but {a} (row a0.A) has 1536"),
                 "shard and .pt": (shard_and_pt, f"{shard_and_pt / 'x.A.pt'} has 64 values, but {c} (row c0.A) has "
                                                 f"1536"),
                 ".pt and .csv": (pt_and_csv, f"{pt_and_csv / 'z.A.csv'} has 64 values, but {pt_and_csv / 'y.A.pt'} "
                                              f"has 1536")}
        for name, (folder, text) in cases.items():
            with self.subTest(files=name):
                self.assertEqual(self.refusal(folder), f"{text}: {_ONE_DIMENSION}")
        self.enterContext(mock.patch.object(search, "set_log_level"))
        db = self.folder("db") / "mixed"
        with self.assertRaises(RefusalError):
            search.build_database_from_embeddings(embedding_folder=two_shards, output_db=str(db), file_extension=None,
                                                  use_gpu_index=False)
        self.assertEqual(list(db.parent.iterdir()), [])

    def test_a_query_of_another_dimension_is_refused_before_faiss(self):
        """'fm-search query embedding' refuses query embeddings of another dimension than the database's, naming the
        file (or Parquet file and row), before FAISS (whose AssertionError had no message), and writes no output: 64
        values against 1536, 1536 against 64, and residue embeddings in Parquet whose rows all have one length (here
        one chain), which a build cannot tell from vectors. Residue rows of two lengths get the residue advice: the
        row lengths are checked before the database's dimension."""
        self.enterContext(mock.patch.object(search, "set_log_level"))
        dbs = {1536: self.folder("db1536") / "chains", 64: self.folder("db64") / "vectors"}
        sources = {1536: _RESOURCES / "chain_embeddings",
                   64: self.files(TensorBatchWriter, self.folder("v64"), _vectors(64))}
        for dim, db in dbs.items():
            search.build_database_from_embeddings(embedding_folder=sources[dim], output_db=str(db), file_extension=None,
                                                  use_gpu_index=False)
        csv = self.files(CsvBatchWriter, self.folder("q64"), _vectors(64, "q", n=2)) / "q0.A.csv"
        shard = self.shard(self.folder("q1536"), _vectors(1536, "q", n=2))
        residue = _flat_residues(self.folder("residue") / "inference-0.parquet", {"1acb.B": _residues()["1acb.B"]})
        residues = _flat_residues(self.folder("residues") / "inference-0.parquet", _residues_64())
        cases = {"64 against 1536": (1536, csv.parent, f"{csv} has 64 values, but the database has dimension 1536: "
                                                       f"{_DATABASE_DIMENSION}"),
                 "1536 against 64": (64, shard.parent, f"{shard} (row q0.A) has 1536 values, but the database has "
                                                       f"dimension 64: {_DATABASE_DIMENSION}"),
                 "residue rows": (1536, residue.parent, f"{residue} (row 1acb.B) has 99840 values, but the database "
                                                        f"has dimension 1536: {_DATABASE_DIMENSION}"),
                 "residue rows of two lengths": (64, residues.parent, f"{residues}: row r7 has 448 values, but row r5 "
                                                                      f"has 320. {_ROW_ADVICE}")}
        hits = self.tmp / "hits.tsv"
        for name, (dim, queries, text) in cases.items():
            with self.subTest(query=name):
                with mock.patch.object(FaissEmbeddingDatabase, "search_batch") as search_batch:
                    with self.assertRaises(RefusalError) as refused:
                        search.query_database_from_embedding(db_path=str(dbs[dim]), embedding_folder=queries,
                                                             output_file=str(hits), file_extension=None, top_k=5,
                                                             threshold=None, use_gpu_index=False)
                self.assertEqual(str(refused.exception), text)
                search_batch.assert_not_called()
                self.assertFalse(hits.exists())

    def test_build_and_query_from_chain_parquet(self):
        """Chain embeddings in Parquet build the database that the same vectors in .pt files build, and query it as
        before: the five chain fixtures, written by ParquetBatchWriter in the shards of two ranks, each find
        themselves first."""
        self.enterContext(mock.patch.object(search, "set_log_level"))
        chains = _chains()
        names = list(chains)
        shards = self.folder("shards")
        self.shard(shards, {name: chains[name] for name in names[:3]}, rank=0)
        self.shard(shards, {name: chains[name] for name in names[3:]}, rank=1)
        databases = {}
        for source, folder in (("parquet", shards), ("pt", _RESOURCES / "chain_embeddings")):
            db = self.folder(source) / "chains"
            search.build_database_from_embeddings(embedding_folder=folder, output_db=str(db), file_extension=None,
                                                  use_gpu_index=False)
            databases[source] = db
        built = FaissEmbeddingDatabase(databases["parquet"].parent, "chains")
        built.load_database()
        self.assertEqual((built.dimension, built.chain_ids), (1536, names))
        for suffix in (".index", ".metadata"):
            with self.subTest(file=suffix):
                self.assertEqual((databases["parquet"].parent / f"chains{suffix}").read_bytes(),
                                 (databases["pt"].parent / f"chains{suffix}").read_bytes())
        hits = self.tmp / "hits.tsv"
        search.query_database_from_embedding(db_path=str(databases["parquet"]), embedding_folder=shards,
                                             output_file=str(hits), file_extension=None, top_k=5, threshold=None,
                                             use_gpu_index=False)
        top_hits = {}
        for line in hits.read_text().splitlines():
            query, hit, score = line.split("\t")[:3]
            top_hits.setdefault(query, (hit, float(score)))
        self.assertEqual(sorted(top_hits), names)
        for query, (hit, score) in top_hits.items():
            with self.subTest(query=query):
                self.assertEqual(hit, query)
                self.assertGreater(score, 0.9999)

    def test_a_pt_and_a_csv_of_one_id_are_refused(self):
        """A .pt and a .csv of one stem give two embeddings one id: refused, naming one pair and the number of others,
        and --file-extension reads one kind. Residue and chain runs that wrote to one folder leave such pairs: the
        pair is refused before the residue table is read."""
        vectors = _vectors(64, n=3)
        both = self.folder("both")
        for writer in (TensorBatchWriter, CsvBatchWriter):
            self.files(writer, both, vectors)
        advice =("the id of a .pt or .csv embedding is its file name without the extension. Read one kind of file "
                  "with --file-extension .pt or --file-extension .csv (runs that write residue and chain embeddings "
                  "to one folder leave such pairs).")
        self.assertEqual(self.refusal(both), f"{both / 'c0.A.pt'} and {both / 'c0.A.csv'} give the same id, c0.A, as "
                                             f"do 2 more pair(s) of .pt and .csv files: {advice}")
        for extension in (".pt", ".csv"):
            with self.subTest(file_extension=extension):
                ids, read, _ = _read(both, file_extension=extension)
                self.assertEqual(ids, list(vectors))
                self.assertEqual(read.shape, (3, 64))
        mixed = self.folder("residue-and-chain")
        self.files(CsvBatchWriter, mixed, {"1acb.B": _residues()["1acb.B"]})
        self.files(TensorBatchWriter, mixed, {"1acb.B": _chains()["1acb.B"]})
        self.assertEqual(self.refusal(mixed), f"{mixed / '1acb.B.pt'} and {mixed / '1acb.B.csv'} give the same id, "
                                              f"1acb.B: {advice}")
        ids, read, _ = _read(mixed, file_extension=".pt")
        self.assertEqual(ids, ["1acb.B"])
        np.testing.assert_array_equal(read[0], _chains()["1acb.B"].numpy())


if __name__ == "__main__":
    unittest.main()
