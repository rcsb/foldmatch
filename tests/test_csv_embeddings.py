"""P9.B1, P9.B3: a .pt or .csv embedding file holds one vector.

P9.B1: CsvBatchWriter writes a chain or assembly vector as one column, one value per line. stream_embeddings averaged
any .csv of more than one row over its rows, as residue embeddings, so each vector became its mean: 'fm-search build
embeddings' built a database of dimension 1 from such files (every score +1 or -1), and 'fm-search query embedding'
with them failed on the dimension.

P9.B3: residue embeddings (one row per residue: a 2-D tensor, or a .csv of one line of D values per residue) were still
averaged over their rows, and the mean was indexed or queried as a chain embedding, which it is not. A .pt or .csv
file must now hold one vector ([D], [1, D], [D, 1]; a .csv of one column or one line): a table is refused, naming the
file and its shape, and so are a file that cannot be read, a file of no values and NaN or infinite values."""
import io
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import faiss
import numpy as np
import pandas as pd
import torch

from foldmatch.cli import search
from foldmatch.errors import RefusalError
from foldmatch.utils.data import stream_embeddings
from foldmatch.writer.batch_writer import CsvBatchWriter, TensorBatchWriter

_IDS = ["1abc-1", "2xyz-1", "3def-1"]
_RESIDUES = Path(__file__).parent / "resources" / "embeddings"


def _vectors() -> dict:
    generator = torch.Generator().manual_seed(0)
    return {dom_id: torch.randn(1536, generator=generator) for dom_id in _IDS}


def _residues() -> dict:
    """Residue embeddings: the ESM3 ones of chain 1acb.B (65 x 1536, its BOS and EOS rows included) and a 64-wide
    table, since no width is assumed."""
    return {"1acb.B": torch.load(_RESIDUES / "1acb.B.pt", map_location="cpu", weights_only=True),
            "r5": torch.randn(5, 64, generator=torch.Generator().manual_seed(1))}


def _read(folder: Path, extension: str = None) -> dict:
    vectors = {}
    for ids, batch in stream_embeddings(folder, file_extension=extension):
        vectors.update(zip(ids, batch))
    return vectors


class CsvEmbeddingsTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def write(self, writer, embeddings: dict, name: str) -> Path:
        """The files fm-embedding and inference write, through their writer."""
        folder = self.tmp / name
        folder.mkdir()
        files = writer(str(folder))
        for dom_id, embedding in embeddings.items():
            files._write_embedding(embedding, dom_id)
        return folder

    def refusal(self, path: Path) -> str:
        """The message of the RefusalError with which stream_embeddings refuses the file or folder ``path``."""
        with self.assertRaises(RefusalError) as refused:
            _read(path)
        return str(refused.exception)

    def test_a_vector_is_one_vector(self):
        vectors = _vectors()
        csv_folder = self.write(CsvBatchWriter, vectors, "csv")
        self.assertEqual(len((csv_folder / "1abc-1.csv").read_text().splitlines()), 1536)  # one value per line
        from_csv = _read(csv_folder, ".csv")
        from_pt = _read(self.write(TensorBatchWriter, vectors, "pt"), ".pt")
        self.assertEqual(sorted(from_csv), _IDS)
        for dom_id, vector in vectors.items():
            with self.subTest(item=dom_id):
                self.assertEqual(from_csv[dom_id].shape, (1536,))
                np.testing.assert_array_equal(from_pt[dom_id], vector.numpy())
                np.testing.assert_allclose(from_csv[dom_id], vector.numpy(), rtol=1e-6, atol=0)

    def test_residue_embeddings_are_refused(self):
        """Residue embeddings as fm-embedding and inference write them ([L, D], one row per residue) are refused, not
        averaged: the file and its shape are named, with the ways to chain embeddings for its format."""
        for writer, ext in ((CsvBatchWriter, "csv"), (TensorBatchWriter, "pt")):
            folder = self.write(writer, _residues(), ext)
            for dom_id, table in _residues().items():
                with self.subTest(file=f"{dom_id}.{ext}"):
                    message = self.refusal(folder / f"{dom_id}.{ext}")
                    rows, columns = table.shape
                    self.assertIn(f"{folder / dom_id}.{ext} holds a table of {rows} x {columns} values, not one "
                                  f"embedding vector.", message)
                    self.assertIn("compute chain embeddings with 'fm-embedding from-structures chain' or "
                                  "'fm-embedding from-sequences chain'", message)
                    self.assertIn(f"'inference chain-embedding --src-file LIST --res-embedding-location DIR "
                                  f"--res-embedding-format {ext} --output-folder OUT', LIST holding one line "
                                  f"NAME,,CHAIN per file.", message)
                    self.assertNotIn("header", message)  # the hints of pandas' layouts are for other tables
            with self.subTest(folder=ext):
                self.assertIn(f"{folder / '1acb.B'}.{ext} holds a table", self.refusal(folder))

    def test_one_row_or_one_column_is_a_vector(self):
        """A [1, D] or [D, 1] tensor and a one-line CSV hold one vector, read bit for bit ([D, 1] was read as the mean
        of its D values); a bfloat16 tensor, which numpy cannot take, is read as its float32 values."""
        vector = torch.randn(1536, generator=torch.Generator().manual_seed(2))
        for name, writer, ext, shaped, expected in (
                ("row", CsvBatchWriter, "csv", vector.reshape(1, -1), vector),
                ("row", TensorBatchWriter, "pt", vector.reshape(1, -1), vector),
                ("column", TensorBatchWriter, "pt", vector.reshape(-1, 1), vector),
                ("bfloat16", TensorBatchWriter, "pt", vector.bfloat16(), vector.bfloat16().float())):
            with self.subTest(file=f"{name}.{ext}", shape=tuple(shaped.shape)):
                folder = self.write(writer, {name: shaped}, f"{name}-{ext}")
                if ext == "csv":
                    self.assertEqual(len((folder / "row.csv").read_text().splitlines()), 1)
                read = _read(folder / f"{name}.{ext}")
                np.testing.assert_array_equal(read[name], expected.numpy())

    def test_nan_and_infinity_are_refused(self):
        """A NaN or infinite value is refused, in a .csv (with what makes one in a CSV) as in a .pt: such an embedding
        is never found, and as a query it finds nothing. A .pt table with a NaN is refused as a table: in a .pt the
        shape is checked first."""
        vectors = _vectors()
        vectors["2xyz-1"][100] = float("nan")
        vectors["3def-1"][7] = float("inf")
        for writer, ext in ((CsvBatchWriter, "csv"), (TensorBatchWriter, "pt")):
            folder = self.write(writer, vectors, ext)
            for dom_id in ("2xyz-1", "3def-1"):
                with self.subTest(file=f"{dom_id}.{ext}"):
                    message = self.refusal(folder / f"{dom_id}.{ext}")
                    self.assertIn(f"{folder / dom_id}.{ext} holds 1 NaN or infinite value(s) among its 1536: such an "
                                  f"embedding is never found, and as a query it finds nothing.", message)
                    self.assertEqual("In a CSV an empty field reads as NaN" in message, ext == "csv")
            with self.subTest(folder=ext):
                self.assertIn(f"{folder / '2xyz-1'}.{ext} holds 1 NaN", self.refusal(folder))
        table = _residues()["r5"]
        table[2, 3] = float("nan")
        folder = self.write(TensorBatchWriter, {"r5": table}, "nan-table")
        with self.subTest(file="r5.pt"):
            message = self.refusal(folder / "r5.pt")
            self.assertIn(f"{folder / 'r5.pt'} holds a table of 5 x 64 values, not one embedding vector.", message)
            self.assertNotIn("NaN", message)

    def test_empty_files_are_refused(self):
        """An empty tensor holds no values; an empty .csv (CsvBatchWriter's file for an empty tensor) and an empty .pt
        file (a run killed before it wrote one) cannot be read. Each is named."""
        folder = self.tmp / "empty"
        folder.mkdir()
        TensorBatchWriter(str(folder))._write_embedding(torch.zeros(0, 1536), "tensor")
        CsvBatchWriter(str(folder))._write_embedding(torch.zeros(0, 1536), "table")
        (folder / "file.pt").write_bytes(b"")
        expected = {"tensor.pt": "holds no values.",
                    "table.csv": "is not a CSV of numbers (No columns to parse from file). A .csv embedding holds one "
                                 "vector",
                    "file.pt": "could not be read as a tensor (EOFError). A .pt embedding holds one vector"}
        for name, text in expected.items():
            with self.subTest(file=name):
                self.assertIn(f"{folder / name} {text}", self.refusal(folder / name))

    def test_a_pt_that_is_not_a_tensor_is_named(self):
        """A .pt that torch.load cannot read (cut short, a pickled object, a text file) or that holds something other
        than a dense tensor is refused with its name, not with a traceback. The cause is not torch's advice for a pickle
        (load it with weights_only=False, which the command cannot do), nor a bare number."""
        folder = self.tmp / "not-tensors"
        folder.mkdir()
        torch.save(_vectors()["1abc-1"], folder / "whole.pt")
        data = (folder / "whole.pt").read_bytes()
        (folder / "cut.pt").write_bytes(data[:len(data) // 2])
        torch.save({"embedding": _vectors()["1abc-1"]}, folder / "dict.pt")
        torch.save({"embedding": Path("x.pt")}, folder / "pickled.pt")  # a class weights_only does not load
        (folder / "text.pt").write_text("hello world\n")
        torch.save(_vectors()["1abc-1"].to_sparse(), folder / "sparse.pt")
        expected = {"cut.pt": "could not be read as a tensor (PytorchStreamReader failed",
                    "pickled.pt": "could not be read as a tensor (it holds pickled objects other than tensors, such as "
                                  "a numpy array, or is not a torch file). A .pt embedding holds one vector",
                    "text.pt": "could not be read as a tensor (KeyError: 101). A .pt embedding holds one vector",
                    "dict.pt": "holds a dict, not a tensor. A .pt embedding holds one vector",
                    "sparse.pt": "holds a tensor that is not D dense values (can't convert Sparse layout tensor"}
        for name, text in expected.items():
            with self.subTest(file=name):
                message = self.refusal(folder / name)
                self.assertIn(f"{folder / name} {text}", message)
                self.assertLess(len(message), len(str(folder)) + 400)

    def test_a_csv_that_does_not_parse_is_named(self):
        """A .csv that is not a table of numbers (another separator, a text header, UTF-16, a line of more values, a
        number too large for a float) is refused with its name and the cause cut short: the error quoted the whole row
        (11 KB for a tab-separated vector), without the file."""
        values = [repr(float(x)) for x in _vectors()["1abc-1"]]
        folder = self.tmp / "unparsed"
        folder.mkdir()
        (folder / "tabs.csv").write_text("\t".join(values) + "\n")
        (folder / "named.csv").write_text("embedding\n" + "\n".join(values) + "\n")
        (folder / "utf16.csv").write_bytes("\n".join(values).encode("utf-16"))
        (folder / "ragged.csv").write_text("1,2,3\n4,5,6\n7,8,9,10\n")
        (folder / "huge.csv").write_text("1\n" + "9" * 400 + "\n")  # read as a Python int
        causes = {"tabs": f"could not convert string to float: '{values[0]}\\t{values[1]}",  # the cell's repr
                  "named": "could not convert string to float: 'embedding')",
                  "utf16": "'utf-8' codec can't decode byte 0xff",
                  "ragged": "Error tokenizing data. C error: Expected 3 fields in line 3, saw 4)",
                  "huge": "int too large to convert to float)"}
        for name, cause in causes.items():
            with self.subTest(file=name):
                message = self.refusal(folder / f"{name}.csv")
                self.assertIn(f"{folder / name}.csv is not a CSV of numbers ({cause}", message)
                self.assertIn("A .csv embedding holds one vector: numbers separated by commas, one per line or all on "
                              "one line, with no header line or index column.", message)
                self.assertLess(len(message), len(str(folder)) + 400)

    def test_a_csv_that_cannot_be_read_is_named(self):
        """A folder named NAME.csv in an embedding folder, which the glob of .csv files matches, is refused with its
        name, as a .pt that cannot be read is, not with a traceback."""
        folder = self.tmp / "folder"
        (folder / "x.A.csv").mkdir(parents=True)
        message = self.refusal(folder)
        self.assertTrue(message.startswith(f"{folder / 'x.A.csv'} could not be read ("), message)
        self.assertIn("Is a directory", message)

    def test_pandas_header_and_index_layouts_are_refused(self):
        """Files written with pandas' default header line (0,1,...,D-1) or index column are tables, refused with what
        makes them so: a vector on one line under a header (it was averaged with 0,1,...,1535), a vector in one
        column with an index (it was averaged into 2 values) and residue embeddings under a header (with the residue
        advice too: 'inference chain-embedding' reads the header as one more residue). With both, as to_csv writes
        by default, the empty field above the index reads as NaN: a vector written from a DataFrame or a Series, and
        residue embeddings, get the NaN message, which names that layout instead of a line cut short."""
        folder = self.tmp / "pandas"
        folder.mkdir()
        vector = _vectors()["1abc-1"].numpy()
        pd.DataFrame([vector]).to_csv(folder / "header.csv", index=False)
        pd.DataFrame(vector).to_csv(folder / "index.csv", header=False)
        pd.DataFrame(_residues()["1acb.B"].numpy()).to_csv(folder / "residues.csv", index=False)
        pd.DataFrame(vector).to_csv(folder / "default.csv")
        pd.Series(vector).to_csv(folder / "series.csv")
        pd.DataFrame(_residues()["1acb.B"].numpy()).to_csv(folder / "residues-default.csv")
        header = "Its first line is 0,1,...,1535: a column header such as pandas writes unless header=False; remove it."
        layout = "A header line, an index column or a decimal comma makes a vector a table."
        both = ("Its first line is a column header over an index column, as pandas' to_csv writes them unless "
                "header=False and index=False: the empty field above the index reads as NaN.")
        for name, shape, hints in (("header", "2 x 1536", (header, layout)), ("index", "1536 x 2", (layout,)),
                                   ("residues", "66 x 1536", (header,))):
            with self.subTest(file=name):
                message = self.refusal(folder / f"{name}.csv")
                self.assertIn(f"{folder / name}.csv holds a table of {shape} values, not one embedding vector.",
                              message)
                self.assertIn("--res-embedding-format csv --output-folder OUT", message)
                for hint in (header, layout):
                    self.assertEqual(hint in message, hint in hints, hint)
        for name, size in (("default", 1537 * 2), ("series", 1537 * 2), ("residues-default", 66 * 1537)):
            with self.subTest(file=name):
                path = folder / f"{name}.csv"
                self.assertEqual(self.refusal(path), f"{path} holds 1 NaN or infinite value(s) among its {size}: such "
                                 f"an embedding is never found, and as a query it finds nothing. {both}")

    def test_a_csv_line_cut_short_is_reported_as_such(self):
        """A residue CSV cut mid-line (a killed run, a full disk) pads its last line with NaN: it gets the NaN message
        and its likely causes, not the residue advice ('inference chain-embedding' made an all-NaN chain embedding of
        it). So does a vector with a comma at the end of each line."""
        folder = self.write(CsvBatchWriter, {"1acb.B": _residues()["1acb.B"]}, "cut")
        text = (folder / "1acb.B.csv").read_text()
        kept = text[:text.index(",", text.index("\n", len(text) // 2) + 100) + 1]  # cut after a field, mid-line
        (folder / "1acb.B.csv").write_text(kept)
        lines = kept.split("\n")
        missing = 1536 - len([v for v in lines[-1].split(",") if v])
        (folder / "comma.csv").write_text("".join(f"{x!r},\n" for x in map(float, _vectors()["1abc-1"])))
        hint = ("In a CSV an empty field reads as NaN: a line cut short by an interrupted write, or a comma at the end "
                "of each line.")
        for name, bad, size in (("1acb.B", missing, len(lines) * 1536), ("comma", 1536, 3072)):
            with self.subTest(file=name):
                message = self.refusal(folder / f"{name}.csv")
                self.assertIn(f"{folder / name}.csv holds {bad} NaN or infinite value(s) among its {size}: ", message)
                self.assertIn(hint, message)
                self.assertNotIn("inference chain-embedding", message)

    def test_the_command_refuses_residue_embeddings(self):
        """'fm-search build embeddings' on a folder of residue CSVs (the default output of 'fm-embedding
        from-structures residue'): exit status 1, one 'Error:' line that names the file, no database file."""
        folder = self.write(CsvBatchWriter, _residues(), "residues")
        db = self.tmp / "db"
        db.mkdir()
        argv = ["fm-search", "build", "embeddings", "--embedding-folder", str(folder), "--output-db",
                str(db / "residues")]
        stderr = self.enterContext(mock.patch("sys.stderr", new_callable=io.StringIO))
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as ctx:
                search.main()
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
        self.assertEqual(ctx.exception.code, 1)
        errors = [line for line in stderr.getvalue().splitlines() if line.startswith("Error: ")]
        self.assertEqual(len(errors), 1, stderr.getvalue())
        resolved = folder.resolve()  # --embedding-folder is resolved (/var -> /private/var on macOS)
        self.assertIn(f"Error: {resolved / '1acb.B.csv'} holds a table of 65 x 1536 values, not one embedding vector.",
                      errors[0])
        self.assertEqual(list(db.iterdir()), [])

    def test_build_and_query_from_csv(self):
        """'fm-search build embeddings' and 'fm-search query embedding' on CSV vectors: a database of dimension 1536,
        in which each vector finds itself. Logging is not under test: the commands leave it as it is."""
        self.enterContext(mock.patch.object(search, "set_log_level"))
        folder = self.write(CsvBatchWriter, _vectors(), "csv")
        db = self.tmp / "db" / "vectors"
        db.parent.mkdir()
        search.build_database_from_embeddings(embedding_folder=folder, output_db=str(db), file_extension=".csv",
                                              use_gpu_index=False)
        index = faiss.read_index(str(db.parent / "vectors.index"))
        self.assertEqual((index.d, index.ntotal), (1536, 3))
        hits = self.tmp / "hits.tsv"
        search.query_database_from_embedding(db_path=str(db), embedding_folder=folder, output_file=str(hits),
                                             file_extension=".csv", top_k=3, threshold=None, use_gpu_index=False)
        scores = {}
        for line in hits.read_text().splitlines():
            query, hit, score = line.split("\t")[:3]
            scores[query, hit] = float(score)
        self.assertEqual(len(scores), 9, hits.read_text())
        for (query, hit), score in scores.items():
            with self.subTest(query=query, hit=hit):
                if query == hit:
                    self.assertGreater(score, 0.9999)
                else:
                    self.assertLess(abs(score), 0.2)  # random vectors: not +1 or -1


if __name__ == "__main__":
    unittest.main()
