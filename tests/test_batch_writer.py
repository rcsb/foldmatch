import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import torch

from foldmatch.writer.batch_writer import (
    CoreBatchWriter,
    CsvBatchWriter,
    JsonStorage,
    ParquetBatchWriter,
    TensorBatchWriter,
)

_EMBEDDINGS = [torch.arange(4, dtype=torch.float32), torch.arange(4, 8, dtype=torch.float32)]
_IDS = ("a", "b")
# A module with a residue head returns per-item payloads after (embeddings, ids);
# the plain writers must store exactly what a 2-tuple would have given them.
_PREDICTIONS = {
    2: (_EMBEDDINGS, _IDS),
    3: (_EMBEDDINGS, _IDS, [b"\x00" * 8, b"\x01" * 8]),
    4: (_EMBEDDINGS, _IDS, [b"\x00" * 8, b"\x01" * 8], [b"\x02" * 20, b"\x03" * 20]),
}
_TRAINER = SimpleNamespace(global_rank=0)


def _write(writer, prediction):
    writer.write_on_batch_end(_TRAINER, None, prediction, None, None, 0, 0)


class _RecordingWriter(CoreBatchWriter):
    def __init__(self):
        super().__init__(output_path=None, postfix="none")
        self.written = []

    def _write_embedding(self, embedding, dom_id):
        self.written.append((dom_id, embedding.tolist()))


class CoreBatchWriterTupleTest(unittest.TestCase):
    """CoreBatchWriter (and every writer built on it) accepts 2-, 3- and 4-tuples."""

    def test_core_writer_3_tuple(self):
        writer = _RecordingWriter()
        _write(writer, _PREDICTIONS[3])
        self.assertEqual(writer.written, [("a", [0.0, 1.0, 2.0, 3.0]), ("b", [4.0, 5.0, 6.0, 7.0])])

    def test_core_writer_tuple_sizes_write_the_same(self):
        expected = None
        for size, prediction in _PREDICTIONS.items():
            with self.subTest(size=size):
                writer = _RecordingWriter()
                _write(writer, prediction)
                if expected is None:
                    expected = writer.written
                self.assertEqual(writer.written, expected)

    def test_none_prediction_is_ignored(self):
        writer = _RecordingWriter()
        _write(writer, None)
        self.assertEqual(writer.written, [])

    def test_file_writers_accept_3_and_4_tuples(self):
        for size in (3, 4):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                _write(CsvBatchWriter(tmp), _PREDICTIONS[size])
                _write(TensorBatchWriter(tmp), _PREDICTIONS[size])
                for dom_id, emb in zip(_IDS, _EMBEDDINGS):
                    csv = pd.read_csv(os.path.join(tmp, f"{dom_id}.csv"), header=None)
                    self.assertEqual(csv[0].tolist(), emb.tolist())
                    self.assertTrue(torch.equal(torch.load(os.path.join(tmp, f"{dom_id}.pt")), emb))

    def test_json_storage_accepts_3_and_4_tuples(self):
        for size in (3, 4):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                writer = JsonStorage(tmp, "run")
                _write(writer, _PREDICTIONS[size])
                writer.on_predict_end(_TRAINER, None)
                df = pd.read_json(os.path.join(tmp, "run-0.json.gz"), orient="records")
                self.assertEqual(df["id"].tolist(), list(_IDS))


class ParquetBatchWriterTupleTest(unittest.TestCase):
    """The Parquet writer keeps its id + embedding schema whatever the tuple size."""

    def _table(self, prediction):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ParquetBatchWriter(tmp, "run")
            _write(writer, prediction)
            writer.on_predict_end(_TRAINER, None)
            return pq.read_table(os.path.join(tmp, "run-0.parquet"))

    def test_tuple_sizes_write_identical_tables(self):
        reference = self._table(_PREDICTIONS[2])
        self.assertEqual(reference.column_names, ["id", "embedding"])
        self.assertEqual(reference.column("id").to_pylist(), list(_IDS))
        for size in (3, 4):
            with self.subTest(size=size):
                self.assertTrue(self._table(_PREDICTIONS[size]).equals(reference))


if __name__ == "__main__":
    unittest.main()
