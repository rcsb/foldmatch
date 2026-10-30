"""P9.B1: a chain or assembly embedding saved as CSV is read as one vector.

CsvBatchWriter writes a chain or assembly vector as one column, one value per line. stream_embeddings averaged any
.csv of more than one row over its rows, as residue embeddings, so each vector became its mean: 'fm-search build
embeddings' built a database of dimension 1 from such files (every score +1 or -1), and 'fm-search query embedding'
with them failed on the dimension."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import faiss
import numpy as np
import torch

from foldmatch.cli import search
from foldmatch.utils.data import stream_embeddings
from foldmatch.writer.batch_writer import CsvBatchWriter, TensorBatchWriter

_IDS = ["1abc-1", "2xyz-1", "3def-1"]


def _vectors() -> dict:
    generator = torch.Generator().manual_seed(0)
    return {dom_id: torch.randn(1536, generator=generator) for dom_id in _IDS}


def _read(folder: Path, extension: str) -> dict:
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

    def test_residue_embeddings_are_averaged(self):
        """A residue table (one row per residue, as CsvBatchWriter writes [L, D]) is still averaged over its rows, as a
        2-D .pt is; one residue included."""
        generator = torch.Generator().manual_seed(1)
        residues = {"r7": torch.randn(7, 1536, generator=generator), "r1": torch.randn(1, 1536, generator=generator)}
        from_csv = _read(self.write(CsvBatchWriter, residues, "csv"), ".csv")
        from_pt = _read(self.write(TensorBatchWriter, residues, "pt"), ".pt")
        for dom_id, table in residues.items():
            with self.subTest(item=dom_id):
                expected = table.mean(dim=0).numpy()
                np.testing.assert_allclose(from_pt[dom_id], expected, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(from_csv[dom_id], expected, rtol=1e-5, atol=1e-6)

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
