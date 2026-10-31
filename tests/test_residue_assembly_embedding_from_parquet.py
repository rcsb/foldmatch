import os
import unittest
from pathlib import Path

import torch
import pyarrow as pa
import pyarrow.parquet as pq

from foldmatch.dataset.residue_assembly_embedding_from_parquet import ResidueAssemblyEmbeddingFromParquet
from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.types.api_types import SrcLocation, StructureFormat


class TestResidueAssemblyEmbeddingFromParquet(unittest.TestCase):

    __test_path = os.path.dirname(__file__)
    __tmp_path = os.path.join(Path(os.path.dirname(__file__)), Path("resources"), Path("tmp"))
    __embeddings_path = os.path.join(Path(os.path.dirname(__file__)), Path("resources"), Path("embeddings"))

    def setUp(self):
        os.makedirs(self.__tmp_path, exist_ok=True)
        for f in os.listdir(self.__tmp_path):
            fp = os.path.join(self.__tmp_path, f)
            if os.path.isfile(fp):
                os.unlink(fp)

    def _build_parquet_from_tensor_files(self):
        """Write the test .pt embeddings into a single Parquet file: one row per chain, one list of D values per
        residue."""
        ids = []
        embeddings = []
        for pt_file in sorted(os.listdir(self.__embeddings_path)):
            if pt_file.endswith('.pt'):
                tensor = torch.load(
                    os.path.join(self.__embeddings_path, pt_file),
                    map_location='cpu'
                )
                item_id = pt_file[:-3]  # strip .pt
                ids.append(item_id)
                embeddings.append(tensor.numpy().tolist())

        schema = pa.schema([
            ('id', pa.string()),
            ('embedding', pa.list_(pa.list_(pa.float32(), len(embeddings[0][0]))))
        ])
        table = pa.table({'id': ids, 'embedding': embeddings}, schema=schema)
        parquet_path = os.path.join(self.__tmp_path, "residue-embeddings.parquet")
        pq.write_table(table, parquet_path)
        return parquet_path

    def test_assembly_from_parquet_matches_tensor_files(self):
        parquet_path = self._build_parquet_from_tensor_files()

        src_stream = [
            ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1", "1acb.1"),
            ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "1", "2uzi.1")
        ]

        tensor_dataset = ResidueAssemblyEmbeddingFromTensorFile(
            src_stream=src_stream,
            res_embedding_location=self.__embeddings_path,
            src_location=SrcLocation.stream,
            structure_format=StructureFormat.mmcif
        )

        parquet_dataset = ResidueAssemblyEmbeddingFromParquet(
            src_stream=src_stream,
            parquet_path=parquet_path,
            src_location=SrcLocation.stream,
            structure_format=StructureFormat.mmcif
        )

        self.assertEqual(len(parquet_dataset), len(tensor_dataset))

        for i in range(len(parquet_dataset)):
            pq_tensor, pq_name = parquet_dataset[i]
            tf_tensor, tf_name = tensor_dataset[i]
            self.assertEqual(pq_name, tf_name)
            self.assertEqual(pq_tensor.shape, tf_tensor.shape)
            self.assertTrue(torch.allclose(pq_tensor, tf_tensor, atol=1e-6))
