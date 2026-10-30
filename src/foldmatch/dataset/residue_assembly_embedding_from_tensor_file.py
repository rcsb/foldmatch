import argparse
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset, DataLoader

from foldmatch.dataset.utils import get_structure_location
from foldmatch.types.api_types import StructureLocation, StructureFormat, SrcLocation, ResEmbeddingFormat, \
    FileOrStreamTuple
from foldmatch.utils.data import stringio_from_url, concatenate_tensors
from foldmatch.utils.structure_parser import get_protein_chains
from foldmatch.utils.structure_provider import StructureProvider


class ResidueAssemblyEmbeddingFromTensorFile(Dataset):

    STREAM_NAME_ATTR = 'stream_name'
    STREAM_ATTR = 'stream'
    ASSEMBLY_ATTR = 'assembly_id'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [STREAM_NAME_ATTR, STREAM_ATTR, ASSEMBLY_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream: FileOrStreamTuple,
            res_embedding_location: Path,
            src_location: SrcLocation = SrcLocation.file,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res_n: int = 0,
            max_res_n: int = sys.maxsize,
            res_embedding_format: ResEmbeddingFormat = ResEmbeddingFormat.pt,
            structure_provider: StructureProvider = None
    ):
        super().__init__()
        self.res_embedding_location = res_embedding_location
        self.src_location = src_location
        self.structure_format = structure_format
        self.min_res_n = min_res_n
        self.max_res_n = max_res_n
        self.res_embedding_format = res_embedding_format
        self.data = pd.DataFrame()
        self.__load_stream(src_stream)
        self.__structure_provider = structure_provider if structure_provider is not None else StructureProvider()

    def __load_stream(self, src_stream):
        self.data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=ResidueAssemblyEmbeddingFromTensorFile.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=ResidueAssemblyEmbeddingFromTensorFile.COLUMNS
        )
        self.data = self.data.sort_values(by=self.data.columns[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_name = self.data.iloc[idx][ResidueAssemblyEmbeddingFromTensorFile.STREAM_NAME_ATTR]
        src_structure = self.data.iloc[idx][ResidueAssemblyEmbeddingFromTensorFile.STREAM_ATTR]
        assembly_id = self.data.iloc[idx][ResidueAssemblyEmbeddingFromTensorFile.ASSEMBLY_ATTR]
        item_name = self.data.iloc[idx][ResidueAssemblyEmbeddingFromTensorFile.ITEM_NAME_ATTR]
        structure = self.__structure_provider.get_structure(
            src_name=src_name,
            src_structure=stringio_from_url(src_structure) if get_structure_location(src_structure) == StructureLocation.remote else src_structure,
            structure_format=self.structure_format,
            assembly_id=assembly_id
        )
        residue_embedding_files = [
            f"{self.res_embedding_location}/{src_name}.{ch}.{self.res_embedding_format.value}" for ch in get_protein_chains(structure, self.min_res_n)
        ]
        if len(residue_embedding_files) == 0:
            raise ValueError(f"No chains found for {src_name}-{assembly_id} in structure {src_structure}.")
        return concatenate_tensors(residue_embedding_files, self.max_res_n, res_embedding_format=self.res_embedding_format), item_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=argparse.FileType('r'), required=True)
    parser.add_argument('--res_embeddings_path', required=True)
    args = parser.parse_args()

    dataset = ResidueAssemblyEmbeddingFromTensorFile(
        src_stream=args.file_list,
        res_embedding_location=args.res_embeddings_path,
        src_location=SrcLocation.file,
        structure_format=StructureFormat.bciff
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda _: _
    )

    for _batch in dataloader:
        print(_batch)
