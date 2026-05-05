
import os
import shutil
import unittest

from foldmatch.cli.structure_embedding import scan_structure_folder
from foldmatch.types.api_types import OutFormat, StructureFormat, Accelerator


class TestScanStructureFolder(unittest.TestCase):
    __test_path = os.path.dirname(__file__)

    def test_scan_mmcif_folder(self):
        entries = scan_structure_folder(
            f"{self.__test_path}/resources/pdb",
            StructureFormat.mmcif
        )
        self.assertEqual(len(entries), 2)
        names = [e[0] for e in entries]
        self.assertIn("1acb", names)
        self.assertIn("2uzi", names)
        for name, path, item_name in entries:
            self.assertEqual(name, item_name)
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(path.endswith(".cif"))

    def test_scan_empty_folder_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(Exception):
                scan_structure_folder(tmpdir, StructureFormat.mmcif)


class TestCliEmbedding(unittest.TestCase):
    __test_path = os.path.dirname(__file__)

    def test_residue_embedding(self):
        _remove_files_in_directory(f"{self.__test_path}/resources/tmp")
        from foldmatch.cli.structure_embedding import residue_embedding
        residue_embedding(
            src_folder=f"{self.__test_path}/resources/pdb",
            output_path=f"{self.__test_path}/resources/tmp",
            output_format=OutFormat.separated,
            structure_format=StructureFormat.mmcif,
            min_res_n=0,
            batch_size=1,
            num_workers=0,
            num_nodes=1,
            accelerator=Accelerator.cpu,
            write_tensor=False
        )
        # 1acb has 2 chains (A, B), 2uzi has 3 chains (A, B, C) = 5 csv files
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/1acb.A.csv"))
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/1acb.B.csv"))
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/2uzi.A.csv"))
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/2uzi.B.csv"))
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/2uzi.C.csv"))

    def test_chain_embedding_with_precomputed_residues(self):
        _remove_files_in_directory(f"{self.__test_path}/resources/tmp")
        from foldmatch.cli.structure_embedding import chain_embedding
        chain_embedding(
            src_folder=f"{self.__test_path}/resources/pdb",
            output_path=f"{self.__test_path}/resources/tmp",
            res_embedding_folder=f"{self.__test_path}/resources/embeddings",
            output_format=OutFormat.grouped,
            output_name="chain-inference",
            structure_format=StructureFormat.mmcif,
            min_res_n=0,
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_aggregator=1,
            num_workers_aggregator=0,
            num_nodes_aggregator=1,
            accelerator=Accelerator.cpu,
            compute_residue_embedding=False
        )
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/chain-inference-0.json.gz"))

    def test_chain_embedding_end_to_end(self):
        _remove_files_in_directory(f"{self.__test_path}/resources/tmp")
        from foldmatch.cli.structure_embedding import chain_embedding
        chain_embedding(
            src_folder=f"{self.__test_path}/resources/pdb",
            output_path=f"{self.__test_path}/resources/tmp",
            res_embedding_folder=f"{self.__test_path}/resources/tmp",
            output_format=OutFormat.grouped,
            output_name="chain-inference",
            structure_format=StructureFormat.mmcif,
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_aggregator=1,
            num_workers_aggregator=0,
            num_nodes_aggregator=1,
            accelerator=Accelerator.cpu,
            compute_residue_embedding=True
        )
        # Residue tensors should have been created
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/1acb.A.pt"))
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/2uzi.A.pt"))
        # Chain embeddings should be grouped
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/chain-inference-0.json.gz"))

    def test_assembly_embedding_with_precomputed_residues(self):
        _remove_files_in_directory(f"{self.__test_path}/resources/tmp")
        from foldmatch.cli.structure_embedding import assembly_embedding
        assembly_embedding(
            src_folder=f"{self.__test_path}/resources/pdb",
            res_embedding_folder=f"{self.__test_path}/resources/embeddings",
            output_path=f"{self.__test_path}/resources/tmp",
            output_format=OutFormat.grouped,
            output_name="assembly-inference",
            structure_format=StructureFormat.mmcif,
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_aggregator=1,
            num_workers_aggregator=0,
            num_nodes_aggregator=1,
            accelerator=Accelerator.cpu,
            compute_residue_embedding=False
        )
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/assembly-inference-0.json.gz"))

    def test_assembly_embedding_end_to_end(self):
        _remove_files_in_directory(f"{self.__test_path}/resources/tmp")
        from foldmatch.cli.structure_embedding import assembly_embedding
        assembly_embedding(
            src_folder=f"{self.__test_path}/resources/pdb",
            res_embedding_folder=f"{self.__test_path}/resources/tmp",
            output_path=f"{self.__test_path}/resources/tmp",
            output_format=OutFormat.grouped,
            output_name="assembly-inference",
            structure_format=StructureFormat.mmcif,
            batch_size_res=1,
            num_workers_res=0,
            num_nodes_res=1,
            batch_size_aggregator=1,
            num_workers_aggregator=0,
            num_nodes_aggregator=1,
            accelerator=Accelerator.cpu,
            compute_residue_embedding=True
        )
        # Residue tensors should have been created
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/1acb.A.pt"))
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/2uzi.A.pt"))
        # Assembly embeddings should be grouped
        self.assertTrue(os.path.exists(f"{self.__test_path}/resources/tmp/assembly-inference-0.json.gz"))


def _remove_files_in_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
