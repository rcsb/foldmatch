import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch

from foldmatch.types.api_types import Accelerator, SrcProteinFrom, SrcLocation, \
    StructureFormat, SrcAssemblyFrom, SrcTensorFrom


class TestInference(unittest.TestCase):

    __test_path = os.path.dirname(__file__)

    def test_esm_inference_from_chain(self):
        from foldmatch.inference.esm_inference import predict
        esm_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "A", "1acb.A"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "A", "2uzi.A")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.chain,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(esm_embeddings[0][0][0].shape), (243, 1536))
        self.assertEqual(tuple(esm_embeddings[1][0][0].shape), (116, 1536))

    def test_esm_inference_from_structure(self):
        from foldmatch.inference.esm_inference import predict

        esm_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1acb"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(esm_embeddings), 5)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(esm_embeddings[0][0][0].shape), (243, 1536))
        self.assertEqual(tuple(esm_embeddings[1][0][0].shape), (65, 1536))
        self.assertEqual(tuple(esm_embeddings[2][0][0].shape), (116, 1536))
        self.assertEqual(tuple(esm_embeddings[3][0][0].shape), (106, 1536))
        self.assertEqual(tuple(esm_embeddings[4][0][0].shape), (168, 1536))

    def test_chain_inference_from_tensor_files(self):
        from foldmatch.inference.chain_inference import predict
        chain_embeddings = predict(
            src_stream=[
                (f"{self.__test_path}/resources/embeddings/1acb.A.pt", "1acb.A"),
                (f"{self.__test_path}/resources/embeddings/1acb.B.pt", "1acb.B"),
                (f"{self.__test_path}/resources/embeddings/2uzi.A.pt", "2uzi.A"),
                (f"{self.__test_path}/resources/embeddings/2uzi.B.pt", "2uzi.B"),
                (f"{self.__test_path}/resources/embeddings/2uzi.C.pt", "2uzi.C")
            ],
            src_location=SrcLocation.stream,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 5)
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[2][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[3][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[4][0][0].shape), (1536,))

    def test_chain_inference_from_parquet(self):
        from foldmatch.inference.chain_inference import predict
        from foldmatch.writer.batch_writer import ParquetBatchWriter
        tensor_files = [
            (f"{self.__test_path}/resources/embeddings/{name}.pt", name)
            for name in ("1acb.A", "1acb.B", "2uzi.A", "2uzi.B", "2uzi.C")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            # The same residue embeddings as the residue commands write them to Parquet: two chains per row group.
            writer = ParquetBatchWriter(tmp, "inference", residue_embeddings=True, buffer_size=2)
            trainer = SimpleNamespace(global_rank=0)
            for file, name in tensor_files:
                residues = torch.load(file, map_location="cpu", weights_only=True)
                writer.write_on_batch_end(trainer, None, ([residues], (name,)), None, None, 0, 0)
            writer.on_predict_end(trainer, None)
            self.assertEqual(pq.ParquetFile(f"{tmp}/inference-0.parquet").metadata.num_row_groups, 3)
            parquet_embeddings = predict(
                src_stream=f"{tmp}/inference-0.parquet",
                src_from=SrcTensorFrom.parquet,
                accelerator=Accelerator.cpu
            )
        chain_embeddings = predict(
            src_stream=tensor_files,
            src_location=SrcLocation.stream,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(parquet_embeddings), 5)
        self.assertEqual([ids for _, ids in parquet_embeddings], [ids for _, ids in chain_embeddings])
        for (parquet_embedding, ids), (chain_embedding, _) in zip(parquet_embeddings, chain_embeddings):
            self.assertEqual(tuple(parquet_embedding[0].shape), (1536,))
            self.assertTrue(torch.equal(parquet_embedding, chain_embedding), ids)

    def test_chain_inference_takes_no_embedding_dim(self):
        """embedding_dim, predict's 18th parameter, is gone: a call that passes it fails before anything runs, by name
        or by position (return_predictions, which took its place, is keyword-only, so D cannot land in it)."""
        from foldmatch.inference.chain_inference import predict
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'embedding_dim'"):
            predict(src_stream=[], src_from=SrcTensorFrom.parquet, embedding_dim=1536)
        with self.assertRaisesRegex(TypeError, "takes from 1 to 17 positional arguments but 18 were given"):
            predict(*([None] * 17), 1536)

    def test_chain_inference_from_structure(self):
        from foldmatch.inference.chain_inference import predict
        chain_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1acb"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi"),
            ],
            res_embedding_location=Path(f"{self.__test_path}/resources/embeddings"),
            src_location=SrcLocation.stream,
            src_from=SrcTensorFrom.structure,
            structure_format=StructureFormat.mmcif,
            min_res_n=0,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 5)
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[2][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[3][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[4][0][0].shape), (1536,))

    def test_structure_inference_from_chain(self):
        from foldmatch.inference.structure_inference import predict

        chain_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "A", "1acb.A"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "A", "2uzi.A")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.chain,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))

    def test_structure_inference_from_structure(self):
        from foldmatch.inference.structure_inference import predict

        chain_embeddings = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1acb"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi")
            ],
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=StructureFormat.mmcif,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 5)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[2][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[3][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[4][0][0].shape), (1536,))

    def test_assembly_inference_from_tensor_files(self):
        from foldmatch.inference.assembly_inferece import predict

        assembly_embedding = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1", "1acb.1"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "1", "2uzi.1")
            ],
            res_embedding_location=Path(f"{self.__test_path}/resources/embeddings"),
            src_location=SrcLocation.stream,
            src_from=SrcAssemblyFrom.assembly,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(assembly_embedding), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(assembly_embedding[0][0][0].shape), (1536,))
        self.assertEqual(tuple(assembly_embedding[1][0][0].shape), (1536,))

    def test_assembly_inference_from_structure(self):
        from foldmatch.inference.assembly_inferece import predict

        assembly_embedding = predict(
            src_stream=[
                ("1acb", f"{self.__test_path}/resources/pdb/1acb.cif", "1acb"),
                ("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi")
            ],
            res_embedding_location=Path(f"{self.__test_path}/resources/embeddings"),
            src_location=SrcLocation.stream,
            src_from=SrcAssemblyFrom.structure,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(assembly_embedding), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(assembly_embedding[0][0][0].shape), (1536,))
        self.assertEqual(tuple(assembly_embedding[1][0][0].shape), (1536,))