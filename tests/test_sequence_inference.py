import os
import shutil
import unittest

from foldmatch.dataset.esm_prot_from_fasta import parse_fasta
from foldmatch.types.api_types import Accelerator, OutFormat, SrcEsmFrom, SrcLocation, SrcTensorFrom


class TestParseFasta(unittest.TestCase):

    __test_path = os.path.dirname(__file__)

    def test_parse_fasta(self):
        sequences = parse_fasta(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences[0][0], "1acb_E")
        self.assertEqual(len(sequences[0][1]), 245)
        self.assertEqual(sequences[1][0], "2uzi_A")
        self.assertEqual(len(sequences[1][1]), 58)

    def test_parse_fasta_sequence_content(self):
        sequences = parse_fasta(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        self.assertTrue(sequences[0][1].startswith("CGVPAIQPVLS"))
        self.assertTrue(sequences[0][1].endswith("TLAAN"))
        self.assertTrue(sequences[1][1].startswith("RPDFCLEPP"))


class TestEsmProtFromFasta(unittest.TestCase):

    __test_path = os.path.dirname(__file__)

    def test_dataset_length(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        dataset = EsmProtFromFasta(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        self.assertEqual(len(dataset), 2)

    def test_dataset_min_res_filter(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        dataset = EsmProtFromFasta(
            f"{self.__test_path}/resources/fasta/test_sequences.fasta",
            min_res_n=100
        )
        self.assertEqual(len(dataset), 1)

    def test_dataset_getitem(self):
        from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
        dataset = EsmProtFromFasta(f"{self.__test_path}/resources/fasta/test_sequences.fasta")
        esm_prot, name = dataset[0]
        self.assertEqual(name, "1acb_E")
        self.assertEqual(len(esm_prot), 245)


class TestSequenceInference(unittest.TestCase):

    __test_path = os.path.dirname(__file__)
    __tmp_path = os.path.join(os.path.dirname(__file__), "resources", "tmp")

    def test_sequence_residue_inference(self):
        from foldmatch.inference.sequence_inference import predict
        embeddings = predict(
            fasta_file=f"{self.__test_path}/resources/fasta/test_sequences.fasta",
            accelerator=Accelerator.cpu,
            out_format=OutFormat.pt
        )
        self.assertEqual(len(embeddings), 2)
        # [batch_index][item_embedding,item_name][item_index]
        self.assertEqual(tuple(embeddings[0][0][0].shape), (247, 1536))
        self.assertEqual(tuple(embeddings[1][0][0].shape), (60, 1536))

    def test_sequence_residue_inference_with_output(self):
        _remove_files_in_directory(self.__tmp_path)
        from foldmatch.inference.sequence_inference import predict
        predict(
            fasta_file=f"{self.__test_path}/resources/fasta/test_sequences.fasta",
            accelerator=Accelerator.cpu,
            out_folder=self.__tmp_path,
            out_format=OutFormat.pt
        )
        self.assertTrue(os.path.exists(f"{self.__tmp_path}/1acb_E.pt"))
        self.assertTrue(os.path.exists(f"{self.__tmp_path}/2uzi_A.pt"))

    def test_sequence_chain_inference(self):
        _remove_files_in_directory(self.__tmp_path)
        from foldmatch.inference.sequence_inference import predict as sequence_predict
        from foldmatch.inference.chain_inference import predict as chain_predict

        fasta_file = f"{self.__test_path}/resources/fasta/test_sequences.fasta"

        sequence_predict(
            fasta_file=fasta_file,
            accelerator=Accelerator.cpu,
            out_folder=self.__tmp_path,
            out_format=OutFormat.pt
        )

        # Build (tensor_path, sequence_name) stream from the FASTA we just
        # produced .pt files for, to feed the chain-level inference.
        sequences = parse_fasta(fasta_file)
        src_stream = tuple(
            (os.path.join(self.__tmp_path, f"{name}.pt"), name)
            for name, _ in sequences
        )
        chain_embeddings = chain_predict(
            src_stream=src_stream,
            src_location=SrcLocation.stream,
            src_from=SrcTensorFrom.file,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 2)
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))

    def test_sequence_chain_inference_from_parquet(self):
        _remove_files_in_directory(self.__tmp_path)
        from foldmatch.inference.sequence_inference import predict as sequence_predict
        from foldmatch.inference.chain_inference import predict as chain_predict

        fasta_file = f"{self.__test_path}/resources/fasta/test_sequences.fasta"
        parquet_path = f"{self.__tmp_path}/residue-embeddings-0.parquet"

        sequence_predict(
            fasta_file=fasta_file,
            accelerator=Accelerator.cpu,
            out_folder=self.__tmp_path,
            out_format=OutFormat.parquet,
            out_name="residue-embeddings"
        )
        self.assertTrue(os.path.exists(parquet_path))

        chain_embeddings = chain_predict(
            src_stream=parquet_path,
            src_from=SrcTensorFrom.parquet,
            accelerator=Accelerator.cpu
        )

        self.assertEqual(len(chain_embeddings), 2)
        self.assertEqual(tuple(chain_embeddings[0][0][0].shape), (1536,))
        self.assertEqual(tuple(chain_embeddings[1][0][0].shape), (1536,))

    def test_sequence_chain_complete_inference_min_res_filter(self):
        # 1acb_E has 245 residues and 2uzi_A has 58: min_res_n=100 must drop
        # 2uzi_A before it is embedded, not just from downstream stores.
        from foldmatch.inference.chain_complete_inference import predict
        embeddings = predict(
            src_stream=f"{self.__test_path}/resources/fasta/test_sequences.fasta",
            src_from=SrcEsmFrom.fasta,
            min_res_n=100,
            accelerator=Accelerator.cpu
        )
        # [batch_index][batch_embeddings,batch_names]
        names = [name for _, batch_names in embeddings for name in batch_names]
        self.assertEqual(names, ["1acb_E"])
        self.assertEqual(tuple(embeddings[0][0].shape), (1, 1536))


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
