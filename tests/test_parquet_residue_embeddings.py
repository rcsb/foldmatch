"""P9.B5: residue embeddings in Parquet keep their shape.

ParquetBatchWriter wrote the residue embeddings of 'fm-embedding ... residue' and 'inference residue-embedding' one row
per chain, its [L, D] table flattened into list<float32>, which stores no shape: ResidueEmbeddingFromParquet and
ResidueAssemblyEmbeddingFromParquet were given D (embedding_dim) to reshape each row, read as a Python list. With
residue_embeddings=True the writer stores one list of D values per residue, list<fixed_size_list<float32, D>>, in row
groups of about 256 MB, so D is in the type: the readers take it from there and refuse a flat file and one that cannot
be read, and stream_embeddings refuses the new files as residue embeddings. Without the flag the writer writes what it
wrote before, but refuses a table. The CHANGELOG's conversion of a flat file is run as it is written there."""
import re
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from foldmatch.cli import search
from foldmatch.dataset.residue_assembly_embedding_from_parquet import ResidueAssemblyEmbeddingFromParquet
from foldmatch.dataset.residue_embedding_from_parquet import ResidueEmbeddingFromParquet
from foldmatch.errors import RefusalError
from foldmatch.search import scoring_system
from foldmatch.types.api_types import SrcLocation
from foldmatch.utils.data import stream_embeddings
from foldmatch.writer.batch_writer import DEFAULT_BYTE_BUDGET, ParquetBatchWriter

_ROOT = Path(__file__).resolve().parents[1]
_RESOURCES = _ROOT / "tests" / "resources"
_TRAINER = SimpleNamespace(global_rank=0)
_FLAT = ("holds flat 'embedding' rows (one list of numbers per row): chain or assembly embeddings, or residue "
         "embeddings as earlier versions wrote them, each table flattened. Residue embeddings in Parquet now hold one "
         "list of D values per residue: recompute them, or convert an earlier file as the CHANGELOG shows.")
_RESIDUES = ("holds residue embeddings (one list of values per residue), not one embedding vector per row: compute chain "
             "embeddings with 'fm-embedding from-structures chain' or 'fm-embedding from-sequences chain', or in Python "
             "with foldmatch.inference.chain_inference.predict(src_stream=FILE, src_from=SrcTensorFrom.parquet, "
             "out_folder=OUT), which writes one chain embedding per .csv file to the existing folder OUT.")


def _esm3() -> dict:
    """The ESM3 residue embeddings of the five test chains, [L + 2, 1536]: the first and last rows are BOS and EOS."""
    return {path.stem: torch.load(path, map_location="cpu", weights_only=True)
            for path in sorted((_RESOURCES / "embeddings").glob("*.pt"))}


def _tables(dim: int, lengths=(9, 1, 5), prefix: str = "t", seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {f"{prefix}{i}.A": torch.randn(n, dim, generator=generator) for i, n in enumerate(lengths)}


def _write(folder: Path, tables: dict, df_id: str = "inference", batch: int = 2, **kwargs) -> Path:
    """The shard {df_id}-0.parquet that the residue commands write to Parquet, through their writer, ``batch`` tables
    per prediction."""
    writer = ParquetBatchWriter(str(folder), df_id, residue_embeddings=True, **kwargs)
    names = list(tables)
    for start in range(0, len(names), batch):
        chunk = names[start:start + batch]
        writer.write_on_batch_end(_TRAINER, None, ([tables[n] for n in chunk], tuple(chunk)), None, None, 0, 0)
    writer.on_predict_end(_TRAINER, None)
    return folder / f"{df_id}-0.parquet"


def _flat(path: Path, tables: dict, row_group_size=None) -> Path:
    """Residue embeddings as earlier versions wrote them: one row per chain, its table flattened into list<float32>."""
    rows = pa.array([table.numpy().ravel() for table in tables.values()], type=pa.list_(pa.float32()))
    pq.write_table(pa.table({"id": list(tables), "embedding": rows}), path, row_group_size=row_group_size)
    return path


def _changelog_nest():
    """The function nest of the CHANGELOG's entry on this change, defined by its code as written there."""
    text = (_ROOT / "CHANGELOG.md").read_text()
    entry = text[text.index("**BREAKING — residue embeddings in Parquet keep their shape.**"):]
    namespace = {}
    exec(textwrap.dedent(re.search(r"```python\n(.*?)\n *```", entry, re.S).group(1)), namespace)
    return namespace["nest"]


class ResidueParquetTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def folder(self, name: str) -> Path:
        folder = self.tmp / name
        folder.mkdir()
        return folder

    def assert_tables(self, dataset, tables: dict):
        """The items of ``dataset`` (ResidueEmbeddingFromParquet) are ``tables``, in order and bit for bit."""
        items = [dataset[i] for i in range(len(dataset))]
        self.assertEqual([name for _, name in items], list(tables))
        for (tensor, name), expected in zip(items, tables.values()):
            with self.subTest(item=name):
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertTrue(torch.equal(tensor, expected), (name, tuple(tensor.shape)))

    def refusal(self, read) -> str:
        with self.assertRaises(RefusalError) as refused:
            read()
        return str(refused.exception)

    def test_residue_embeddings_keep_their_shape(self):
        """One row per chain, one list of D values per residue: list<fixed_size_list<float32, D>>, D in the type, no
        metadata, for the ESM3 fixtures (D 1536, BOS and EOS rows included) and for D 64. The row groups after the
        first keep its schema, and the readers give back every table bit for bit: an assembly, its chains' tables
        concatenated; a list of two shards (of two df_ids, both written at rank 0), their tables in order."""
        esm3 = _esm3()
        for dim, tables in ((1536, esm3), (64, _tables(64))):
            with self.subTest(dim=dim):
                path = _write(self.folder(f"d{dim}"), tables, buffer_size=2)
                schema = pq.read_schema(path)
                self.assertEqual(schema.names, ["id", "embedding"])
                self.assertEqual(schema.field("embedding").type, pa.list_(pa.list_(pa.float32(), dim)))
                self.assertEqual(schema.field("embedding").type.value_type.list_size, dim)
                self.assertIsNone(schema.metadata)
                self.assertEqual(pq.ParquetFile(path).metadata.num_row_groups, (len(tables) + 1) // 2)
                self.assert_tables(ResidueEmbeddingFromParquet(path), tables)
        assembly = ResidueAssemblyEmbeddingFromParquet(
            src_stream=[("1acb", str(_RESOURCES / "pdb" / "1acb.cif"), "1", "1acb-1")],
            parquet_path=self.tmp / "d1536" / "inference-0.parquet", src_location=SrcLocation.stream)
        tensor, name = assembly[0]
        self.assertEqual(name, "1acb-1")
        self.assertTrue(torch.equal(tensor, torch.cat([esm3["1acb.A"], esm3["1acb.B"]])))
        shards = self.folder("shards")
        first, second = _tables(64, prefix="a"), _tables(64, (4, 2), prefix="b", seed=1)
        self.assert_tables(ResidueEmbeddingFromParquet([_write(shards, first, df_id="a"),
                                                        _write(shards, second, df_id="b")]), {**first, **second})

    def test_row_groups_of_about_256_mb(self):
        """Residue tables are written in row groups of about 256 MB, the default byte budget (as with a residue column),
        not of 16,384 chains (buffer_size): at 6 KiB per residue (D 1536) these would buffer tens of GB, and a flush
        takes several times its row group. A table counts its bytes, so a given budget cuts the row groups. Without
        residue_embeddings the writer has no budget, which keeps its bytes: tests/test_esm_codes_build_channel.py."""
        writer = ParquetBatchWriter(str(self.tmp), "x", residue_embeddings=True)
        self.assertEqual(writer.byte_budget, DEFAULT_BYTE_BUDGET)
        tables = _tables(64)  # 2,304, 256 and 1,280 bytes, and 4 bytes of id each: the first crosses 2,000 bytes
        path = _write(self.folder("budget"), tables, batch=1, byte_budget=2000)
        meta = pq.ParquetFile(path).metadata
        self.assertEqual([meta.row_group(i).num_rows for i in range(meta.num_row_groups)], [1, 2])
        self.assert_tables(ResidueEmbeddingFromParquet(path), tables)

    def test_the_writer_refuses_what_it_cannot_store(self):
        """With residue_embeddings: an embedding that is not a 2-D [L, D] table of D > 0 (one vector, a stack of
        tables, no values per residue) and a second D, in the batch of the first or in a later one, refused before any
        of the batch is buffered; residue embeddings and a residue output together. Without it, a table: this writer
        stores one vector per row."""
        stores = "this Parquet writer stores residue embeddings"
        cases = {"one vector": ([torch.zeros(64)], f"{stores}, one [L, D] table per row, but x.A has shape [64]"),
                 "a stack": ([torch.zeros(1, 5, 64)], f"{stores}, one [L, D] table per row, but x.A has shape [1, 5, 64]"),
                 "no values": ([torch.zeros(5, 0)], f"{stores}, one [L, D] table per row, but x.A has shape [5, 0]"),
                 "two dimensions": ([torch.zeros(5, 64), torch.zeros(3, 32)],
                                    f"{stores} of one dimension, but y.A has 32 values per residue and x.A has 64")}
        for name, (embeddings, message) in cases.items():
            with self.subTest(case=name):
                folder = self.folder(name.replace(" ", "-"))
                writer = ParquetBatchWriter(str(folder), "inference", residue_embeddings=True)
                prediction = (embeddings, ("x.A", "y.A")[:len(embeddings)])
                self.assertEqual(self.refusal(lambda: writer.write_on_batch_end(_TRAINER, None, prediction, None, None,
                                                                                0, 0)), message)
                writer.on_predict_end(_TRAINER, None)
                self.assertEqual(list(folder.iterdir()), [])
        later = self.folder("later")
        tables = _tables(64, (5,), prefix="x")
        writer = ParquetBatchWriter(str(later), "inference", residue_embeddings=True)
        writer.write_on_batch_end(_TRAINER, None, (list(tables.values()), tuple(tables)), None, None, 0, 0)
        self.assertEqual(self.refusal(lambda: writer.write_on_batch_end(
            _TRAINER, None, ([torch.zeros(2, 64), torch.zeros(3, 32)], ("y.A", "z.A")), None, None, 0, 0)),
            f"{stores} of one dimension, but z.A has 32 values per residue and x0.A has 64")
        writer.on_predict_end(_TRAINER, None)
        self.assert_tables(ResidueEmbeddingFromParquet(later / "inference-0.parquet"), tables)
        system = scoring_system.get(scoring_system.DEFAULT_ESM_CODES)
        with self.assertRaises(ValueError) as refused:
            ParquetBatchWriter(str(self.tmp), "x", residue_output="residue_codes", system=system,
                               residue_embeddings=True)
        self.assertEqual(str(refused.exception), "a residue_codes writer stores chain embeddings, not residue embeddings")
        plain = self.folder("plain")
        writer = ParquetBatchWriter(str(plain), "inference")
        prediction = ([torch.zeros(64), torch.zeros(5, 64)], ("v.A", "r.A"))
        self.assertEqual(self.refusal(lambda: writer.write_on_batch_end(_TRAINER, None, prediction, None, None, 0, 0)),
                         "this Parquet writer stores one embedding vector per row, but r.A has shape [5, 64]: residue "
                         "embeddings are written with residue_embeddings=True")
        writer.on_predict_end(_TRAINER, None)
        self.assertEqual(list(plain.iterdir()), [])

    def test_the_plain_writer_is_unchanged(self):
        """Without residue_embeddings, embedding vectors are written as before: list<float32>, no metadata, one vector
        of D values per row, [1, D] and [D, 1] tensors included (one axis longer than 1). Byte for byte against the
        writer before P5.3: tests/test_esm_codes_build_channel.py."""
        generator = torch.Generator().manual_seed(2)
        vectors = {"a.A": torch.randn(64, generator=generator), "b.A": torch.randn(1, 64, generator=generator),
                   "c.A": torch.randn(64, 1, generator=generator)}
        folder = self.folder("plain")
        writer = ParquetBatchWriter(str(folder), "inference")
        self.assertEqual(writer.schema, pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.float32()))]))
        writer.write_on_batch_end(_TRAINER, None, (list(vectors.values()), tuple(vectors)), None, None, 0, 0)
        writer.on_predict_end(_TRAINER, None)
        path = folder / "inference-0.parquet"
        schema = pq.read_schema(path)
        self.assertEqual(schema.field("embedding").type, pa.list_(pa.float32()))
        self.assertIsNone(schema.metadata)
        table = pq.read_table(path)
        self.assertEqual(table.column("id").to_pylist(), list(vectors))
        np.testing.assert_array_equal(np.stack(table.column("embedding").to_numpy(zero_copy_only=False)),
                                      np.stack([v.numpy().ravel() for v in vectors.values()]))

    def test_the_readers_refuse_a_flat_file(self):
        """A flat 'embedding' column, each table flattened into one row as earlier versions wrote residue embeddings,
        is refused by both readers, naming the file: they reshaped it with the embedding_dim they were given."""
        flat = _flat(self.tmp / "inference-0.parquet", _esm3())
        assemblies = [("1acb", str(_RESOURCES / "pdb" / "1acb.cif"), "1", "1acb-1")]
        readers = {"ResidueEmbeddingFromParquet": lambda: ResidueEmbeddingFromParquet(flat),
                   "ResidueAssemblyEmbeddingFromParquet": lambda: ResidueAssemblyEmbeddingFromParquet(
                       src_stream=assemblies, parquet_path=flat, src_location=SrcLocation.stream)}
        for name, read in readers.items():
            with self.subTest(reader=name):
                self.assertEqual(self.refusal(read), f"{flat} {_FLAT}")

    def test_the_readers_refuse_other_columns_and_nulls(self):
        """An 'embedding' column of another type is refused, naming its type and the type of residue embeddings; a
        column of lists of D values per residue (list<list<double>>, as pandas writes [L, D] lists) is read once cast
        as the message says. A null row or value is refused too, which Arrow's flatten would skip, shifting the tables
        after it. pyarrow 24 refuses to write a null residue in the middle of the column, and writes one at its start
        or end but cannot read it back."""
        folder = self.folder("other")
        doubles = pa.array([[[0.5, 0.25], [1.0, 2.0]]])
        types = {"text": (pa.array(["0.1,0.2"]), "string"),
                 "lists of any length": (pa.array([[[0.1, 0.2], [0.3]]], type=pa.list_(pa.list_(pa.float32()))),
                                         "list<element: list<element: float>>"),
                 "lists of D doubles": (doubles, "list<element: list<element: double>>"),
                 "integers": (pa.array([[[1, 2]]], type=pa.list_(pa.list_(pa.int64(), 2))),
                              "list<element: fixed_size_list<element: int64>[2]>")}
        for name, (column, kind) in types.items():
            with self.subTest(column=name):
                path = folder / f"{name.replace(' ', '-')}.parquet"
                pq.write_table(pa.table({"id": ["x.A"], "embedding": column}), path)
                self.assertEqual(self.refusal(lambda: ResidueEmbeddingFromParquet(path)),
                                 f"{path}: its 'embedding' column is of type {kind}, not residue embeddings, which are "
                                 f"of type list<fixed_size_list<float32, D>> (D in the type), as "
                                 f"ParquetBatchWriter(residue_embeddings=True) writes them. A column of lists of D "
                                 f"values per residue converts with .cast(pa.list_(pa.list_(pa.float32(), D))).")
        path = folder / "cast.parquet"
        pq.write_table(pa.table({"id": ["x.A"], "embedding": doubles.cast(pa.list_(pa.list_(pa.float32(), 2)))}), path)
        self.assert_tables(ResidueEmbeddingFromParquet(path), {"x.A": torch.tensor([[0.5, 0.25], [1.0, 2.0]])})
        kind = pa.list_(pa.list_(pa.float32(), 2))
        nulls = {"row": [[[0.1, 0.2]], None, [[0.5, 0.6]]],
                 "value": [[[0.1, 0.2]], [[None, 0.4]], [[0.5, 0.6]]]}
        for name, rows in nulls.items():
            with self.subTest(null=name):
                path = folder / f"null-{name}.parquet"
                pq.write_table(pa.table({"id": ["a.A", "b.A", "c.A"], "embedding": pa.array(rows, type=kind)}), path)
                self.assertEqual(self.refusal(lambda: ResidueEmbeddingFromParquet(path)),
                                 f"{path}: its 'embedding' column holds a null row, residue or value.")

    def test_a_file_that_cannot_be_read_is_refused(self):
        """A file that cannot be read as Parquet is refused, naming it, instead of stopping on pyarrow's message, which
        names no file: an empty file and a shard whose first page is corrupted, which fails when read, not when opened.
        So is a folder, which the readers used to read through pq.read_table: they read one file, or a list of files
        (ResidueEmbeddingFromParquet only), such as the shards of a multi-device run that
        sorted(Path(OUT).glob("*.parquet")) lists. A missing path is refused with the system's reason."""
        shards = self.folder("shards")
        tables = _tables(64)
        data = _write(shards, tables).read_bytes()
        empty, corrupted = self.tmp / "empty.parquet", self.tmp / "corrupted.parquet"
        empty.write_bytes(b"")
        corrupted.write_bytes(data[:4] + bytes(16) + data[20:])  # the first page header follows the magic bytes PAR1
        again = "Write it again: an interrupted run or a damaged copy leaves an unreadable file."
        causes = {empty: "Parquet file size is 0 bytes",
                  corrupted: "Couldn't deserialize thrift: TProtocolException: Invalid data"}
        for path, cause in causes.items():
            with self.subTest(file=path.name):
                self.assertEqual(self.refusal(lambda: ResidueEmbeddingFromParquet(path)),
                                 f"{path} could not be read as Parquet ({cause}). {again}")
        with self.subTest(file="folder"):
            self.assertEqual(self.refusal(lambda: ResidueEmbeddingFromParquet(shards)),
                             f"{shards} is a folder, not a Parquet file.")
            self.assert_tables(ResidueEmbeddingFromParquet(sorted(shards.glob("*.parquet"))), tables)
        with self.subTest(file="missing"):
            missing = self.tmp / "missing.parquet"
            self.assertEqual(self.refusal(lambda: ResidueEmbeddingFromParquet(missing)),
                             f"{missing} could not be opened (No such file or directory).")

    def test_build_and_query_refuse_residue_embeddings(self):
        """stream_embeddings, the reader of 'fm-search build embeddings' and 'fm-search query embedding', refuses
        residue embeddings in Parquet, naming the file, whatever their row lengths: also one chain, whose flattened
        row it could not tell from a vector. A build writes no database."""
        esm3 = _esm3()
        cases = {"five-chains": esm3, "one-chain": {"1acb.B": esm3["1acb.B"]}, "width-64": _tables(64)}
        for name, tables in cases.items():
            with self.subTest(tables=name):
                path = _write(self.folder(name), tables)
                for batch_size in (32768, 1):
                    self.assertEqual(self.refusal(lambda: list(stream_embeddings(path.parent, batch_size=batch_size))),
                                     f"{path} {_RESIDUES}")
        self.enterContext(mock.patch.object(search, "set_log_level"))
        db = self.folder("db") / "residues"
        with self.assertRaises(RefusalError):
            search.build_database_from_embeddings(embedding_folder=self.tmp / "five-chains", output_db=str(db),
                                                  file_extension=None, use_gpu_index=False)
        self.assertEqual(list(db.parent.iterdir()), [])

    def test_the_changelog_conversion(self):
        """The CHANGELOG's nest, defined as written there, turns a flat file (in row groups of two chains; D 1536 and
        64) into one that the readers read bit for bit and stream_embeddings refuses as residue embeddings. It
        refuses a D that does not divide every row, and writes nothing then."""
        nest = _changelog_nest()
        for dim, tables in ((1536, _esm3()), (64, _tables(64))):
            with self.subTest(dim=dim):
                flat = _flat(self.folder(f"d{dim}") / "inference-0.parquet", tables, row_group_size=2)
                nested = self.folder(f"nested{dim}") / "inference-0.parquet"
                nest(str(flat), str(nested), dim)
                self.assertEqual(pq.read_schema(nested).field("embedding").type, pa.list_(pa.list_(pa.float32(), dim)))
                self.assert_tables(ResidueEmbeddingFromParquet(nested), tables)
                self.assertEqual(self.refusal(lambda: list(stream_embeddings(nested))), f"{nested} {_RESIDUES}")
        flat, wrong = self.tmp / "d64" / "inference-0.parquet", self.tmp / "wrong.parquet"
        with self.assertRaises(ValueError) as refused:
            nest(str(flat), str(wrong), 7)
        self.assertEqual(str(refused.exception), f"{flat}: a row is not a multiple of 7 values")
        self.assertFalse(wrong.exists())


if __name__ == "__main__":
    unittest.main()
