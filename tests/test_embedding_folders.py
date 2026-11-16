"""P9.X1: fm-search reads the embedding files of a folder, not its subfolders.

'fm-search build embeddings' and 'query embedding' list the entries of --embedding-folder named *.parquet, *.pt and
*.csv (utils.data.embedding_files). A folder so named (the dataset folder ``x.parquet`` that Spark writes, a codes
folder ``codes.pt`` inside an embedding folder) was listed too, and refused ("is a folder, not a Parquet file"), so the
whole embedding folder was refused. Such a folder is now skipped, with one WARNING per command that names the skipped
folders; a symbolic link to a file is a file, one to a folder a folder; a folder that then holds no embedding file
keeps the "No embedding files found" error."""
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from typer.testing import CliRunner

from foldmatch.cli import search
from foldmatch.utils import data
from foldmatch.utils.data import embedding_files, refuse_repeated_query_ids, stream_embeddings
from foldmatch.writer.batch_writer import CsvBatchWriter, ParquetBatchWriter, TensorBatchWriter

_LOGGER = "foldmatch.utils.data"
_ADVICE = ("The embedding files of a folder are read, not those of its subfolders; to read the files of such a folder "
           "(the parts of a Parquet dataset that Spark writes, say), give that folder itself.")


def _vectors(prefix: str, n: int = 4, dim: int = 16, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {f"{prefix}{i}.A": torch.randn(dim, generator=generator) for i in range(n)}


def _shard(folder: Path, embeddings: dict, name: str = "inference", rank: int = 0) -> Path:
    """``<name>-<rank>.parquet`` as fm-embedding writes chain embeddings."""
    writer = ParquetBatchWriter(str(folder), name)
    trainer = SimpleNamespace(global_rank=rank)
    writer.write_on_batch_end(trainer, None, (list(embeddings.values()), list(embeddings)), None, None, 0, 0)
    writer.on_predict_end(trainer, None)
    return folder / f"{name}-{rank}.parquet"


def _files(writer, folder: Path, embeddings: dict) -> Path:
    files = writer(str(folder))
    for dom_id, embedding in embeddings.items():
        files._write_embedding(embedding, dom_id)
    return folder


def _read(path, **kwargs):
    ids, batches = [], []
    for batch_ids, batch in stream_embeddings(path, **kwargs):
        ids.extend(batch_ids)
        batches.append(batch)
    return ids, np.concatenate(batches)


class EmbeddingFolderTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()

    def folder(self, name: str) -> Path:
        folder = self.tmp / name
        folder.mkdir()
        return folder

    def embeddings(self, name: str) -> Path:
        """A folder of chain embeddings: a Parquet shard (c0.A..c3.A) and a .pt file (p0.A)."""
        folder = self.folder(name)
        _shard(folder, _vectors("c"))
        _files(TensorBatchWriter, folder, _vectors("p", n=1, seed=1))
        return folder

    def with_folders(self, folder: Path) -> list:
        """Adds folders named like embedding files to ``folder``: a Spark dataset folder ``dataset.parquet`` whose
        part repeats the ids of the shard, a codes folder ``codes.pt``, an empty ``y.csv``, and ``link.parquet``, a
        symbolic link to a folder elsewhere; and ``file-link.csv``, a symbolic link to a .csv embedding elsewhere (a
        file, which is read). Returns the skipped folders' names, in the order of the listing (.pt, .csv, .parquet)."""
        _shard(self.folder_in(folder, "dataset.parquet"), _vectors("c", seed=2), name="part")
        _files(TensorBatchWriter, self.folder_in(folder, "codes.pt"), _vectors("p", n=1, seed=3))
        self.folder_in(folder, "y.csv")
        elsewhere = self.folder(f"{folder.name}-elsewhere")
        _shard(elsewhere, _vectors("c", seed=4))
        (folder / "link.parquet").symlink_to(elsewhere, target_is_directory=True)
        csv = _files(CsvBatchWriter, self.folder(f"{folder.name}-csv"), _vectors("v", n=1, seed=5)) / "v0.A.csv"
        (folder / "file-link.csv").symlink_to(csv)
        return ["codes.pt", "y.csv", "dataset.parquet", "link.parquet"]

    @staticmethod
    def folder_in(folder: Path, name: str) -> Path:
        (folder / name).mkdir()
        return folder / name

    def test_folders_named_like_embedding_files_are_skipped(self):
        """The files are read as without the folders, a link to a file among them; one WARNING names the folders."""
        plain = self.embeddings("plain")
        folder = self.embeddings("folders")
        skipped = self.with_folders(folder)
        with self.assertLogs(_LOGGER, "WARNING") as logs:
            parquet_files, point_files = embedding_files(folder)
        self.assertEqual(parquet_files, [folder / "inference-0.parquet"])
        self.assertEqual(point_files, [folder / "p0.A.pt", folder / "file-link.csv"])
        self.assertEqual(logs.output, [f"WARNING:{_LOGGER}:{folder}: 4 folder(s) named like embedding files skipped, "
                                       f"not read: {', '.join(skipped)}. {_ADVICE}"])
        with self.assertLogs(_LOGGER, "WARNING") as logs:
            ids, vectors = _read(folder)
        self.assertEqual(len(logs.output), 1)
        plain_ids, plain_vectors = _read(plain)
        self.assertEqual(ids, plain_ids + ["file-link"])
        np.testing.assert_array_equal(vectors[:len(plain_ids)], plain_vectors)
        with self.assertNoLogs(_LOGGER, "WARNING"):
            self.assertEqual(embedding_files(folder, warn_folders=False), (parquet_files, point_files))
            _read(folder, warn_folders=False)
            refuse_repeated_query_ids(folder)   # the part of dataset.parquet repeats the shard's ids: not read

    def test_a_folder_with_only_such_folders(self):
        """No embedding file: today's error, which names the skipped folders too; the WARNING as well."""
        folder = self.folder("only")
        self.folder_in(folder, "dataset.parquet")
        text = (f"No embedding files found with extensions ['.pt', '.csv', '.parquet'] in {folder} (1 folder(s) named "
                f"like embedding files skipped: dataset.parquet)")
        with self.assertLogs(_LOGGER, "WARNING") as logs, self.assertRaises(ValueError) as refused:
            _read(folder)
        self.assertEqual(str(refused.exception), text)
        self.assertEqual(len(logs.output), 1)
        with self.assertNoLogs(_LOGGER, "WARNING"), self.assertRaises(ValueError) as refused:
            embedding_files(folder, warn_folders=False)
        self.assertEqual(str(refused.exception), text)

    def test_many_folders_and_one_extension(self):
        """The first 5 folders named, the rest counted; --file-extension lists (and skips) that extension only."""
        folder = self.embeddings("many")
        for k in range(7):
            self.folder_in(folder, f"d{k}.parquet")
        self.folder_in(folder, "codes.pt")
        with self.assertLogs(_LOGGER, "WARNING") as logs:
            embedding_files(folder, ".parquet")
        self.assertIn("7 folder(s) named like embedding files skipped, not read: d0.parquet, d1.parquet, d2.parquet, "
                      "d3.parquet, d4.parquet and 2 more.", logs.output[0])
        with self.assertLogs(_LOGGER, "WARNING") as logs:
            self.assertEqual(embedding_files(folder, ".pt"), ([], [folder / "p0.A.pt"]))
        self.assertIn("1 folder(s) named like embedding files skipped, not read: codes.pt.", logs.output[0])

    def test_build_and_query_embeddings(self):
        """'fm-search build embeddings' and 'query embedding' on a folder with such folders: the index, metadata and
        hits of the folder without them, and one WARNING per command."""
        self.enterContext(mock.patch.object(search, "set_log_level"))
        plain = self.embeddings("plain")
        folder = self.embeddings("folders")
        self.with_folders(folder)
        (folder / "file-link.csv").unlink()     # the same files as the plain folder
        outputs = {}
        for name, source in (("plain", plain), ("folders", folder)):
            db = self.folder(f"db-{name}") / "db"
            hits = db.parent / "hits.tsv"
            warnings = []
            for args in (("build", "embeddings", "--embedding-folder", source, "--output-db", db),
                         ("query", "embedding", "--db-path", db, "--embedding-folder", source, "--output-file", hits,
                          "--threshold", "0")):
                with self.assertLogs(_LOGGER, "INFO") as logs:
                    logging.getLogger(_LOGGER).info("start")
                    result = CliRunner().invoke(search.app, [str(a) for a in args])
                self.assertEqual(result.exit_code, 0, f"{args}: {result.output} {result.exception!r}")
                warnings.append([line for line in logs.output if line.startswith("WARNING")])
            self.assertEqual([len(w) for w in warnings], [0, 0] if name == "plain" else [1, 1])
            outputs[name] = {suffix: Path(f"{db}{suffix}").read_bytes() for suffix in (".index", ".metadata")}
            outputs[name]["hits"] = hits.read_bytes()
        self.assertTrue(outputs["plain"]["hits"])
        self.assertEqual(outputs["folders"], outputs["plain"])

    def test_a_skipped_folder_is_not_opened(self):
        """A folder named *.parquet is never handed to the Parquet reader, which refuses a folder."""
        folder = self.embeddings("never-opened")
        self.folder_in(folder, "dataset.parquet")
        opened = []
        real = data._parquet_file
        with mock.patch.object(data, "_parquet_file", side_effect=lambda f: opened.append(f) or real(f)), \
                self.assertLogs(_LOGGER, "WARNING"):
            _read(folder)
        self.assertEqual(opened, [folder / "inference-0.parquet"])


if __name__ == "__main__":
    unittest.main()
