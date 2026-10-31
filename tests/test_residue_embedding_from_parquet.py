import os
import unittest
from pathlib import Path

import torch
import pyarrow as pa
import pyarrow.parquet as pq

from foldmatch.dataset.residue_embedding_from_parquet import ResidueEmbeddingFromParquet


class TestResidueEmbeddingFromParquet(unittest.TestCase):

    __tmp_path = os.path.join(Path(os.path.dirname(__file__)), Path("resources"), Path("tmp"))

    def setUp(self):
        os.makedirs(self.__tmp_path, exist_ok=True)
        for f in os.listdir(self.__tmp_path):
            fp = os.path.join(self.__tmp_path, f)
            if os.path.isfile(fp):
                os.unlink(fp)

    def _write_parquet(self, embeddings, ids, path):
        # One row per chain, one list of D values per residue.
        schema = pa.schema([
            ('id', pa.string()),
            ('embedding', pa.list_(pa.list_(pa.float32(), embeddings[0].shape[1])))
        ])
        table = pa.table(
            {
                'id': ids,
                'embedding': [emb.numpy().tolist() for emb in embeddings]
            },
            schema=schema
        )
        pq.write_table(table, path, compression='snappy')

    def test_round_trip(self):
        embedding_dim = 8
        originals = [
            torch.randn(10, embedding_dim),
            torch.randn(5, embedding_dim),
            torch.randn(20, embedding_dim),
        ]
        ids = ["chain_A", "chain_B", "chain_C"]

        parquet_path = Path(os.path.join(self.__tmp_path, "embeddings.parquet"))
        self._write_parquet(originals, ids, parquet_path)

        dataset = ResidueEmbeddingFromParquet(
            parquet_path=parquet_path
        )

        self.assertEqual(len(dataset), 3)
        for i in range(len(dataset)):
            tensor, item_id = dataset[i]
            self.assertEqual(item_id, ids[i])
            self.assertEqual(tensor.shape, originals[i].shape)
            self.assertTrue(torch.allclose(tensor, originals[i], atol=1e-6))

    def test_single_residue(self):
        embedding_dim = 16
        original = torch.randn(1, embedding_dim)

        parquet_path = os.path.join(self.__tmp_path, "single.parquet")
        self._write_parquet([original], ["single"], parquet_path)

        dataset = ResidueEmbeddingFromParquet(parquet_path)
        tensor, item_id = dataset[0]

        self.assertEqual(item_id, "single")
        self.assertEqual(tensor.shape, (1, embedding_dim))
        self.assertTrue(torch.allclose(tensor, original, atol=1e-6))
