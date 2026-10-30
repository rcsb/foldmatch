import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from foldmatch.dataset.residue_embedding_from_tensor_file import ResidueEmbeddingFromTensorFile
from foldmatch.dataset.utils import get_structure_location, refuse_shared_names
from foldmatch.types.api_types import SrcLocation, StructureLocation, StructureFormat, ResEmbeddingFormat, \
    FileOrStreamTuple
from foldmatch.utils.data import stringio_from_url
from foldmatch.utils.structure_parser import get_protein_chains
from foldmatch.utils.structure_provider import StructureProvider


class ResidueEmbeddingFromStructure(ResidueEmbeddingFromTensorFile):

    STREAM_NAME_ATTR = 'stream_name'
    STREAM_ATTR = 'stream'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [STREAM_NAME_ATTR, STREAM_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream: FileOrStreamTuple,
            res_embedding_location: Path,
            src_location: SrcLocation = SrcLocation.file,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res_n: int = 0,
            res_embedding_format: ResEmbeddingFormat = ResEmbeddingFormat.pt,
            structure_provider: StructureProvider = None
    ):
        if not os.path.isdir(res_embedding_location):
            raise FileNotFoundError(f"Folder {res_embedding_location} does not exist")
        self.res_embedding_location = res_embedding_location
        self.src_location = src_location
        self.structure_format = structure_format
        self.min_res_n = min_res_n
        self.res_embedding_format = res_embedding_format
        self.__structure_provider = structure_provider if structure_provider is not None else StructureProvider()
        super().__init__(
            src_stream=self.__get_chains(src_stream),
            src_location=SrcLocation.stream,
            res_embedding_format=res_embedding_format
        )

    def __get_chains(self, src_stream):
        chains = []
        data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=ResidueEmbeddingFromStructure.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=ResidueEmbeddingFromStructure.COLUMNS
        )
        data = data.sort_values(by=data.columns[0])
        refuse_shared_names(data, ResidueEmbeddingFromStructure.STREAM_NAME_ATTR,
                            ResidueEmbeddingFromStructure.STREAM_ATTR)
        progress = tqdm(data.iterrows(), total=len(data), desc="Loading structures")
        for idx, row in progress:
            src_name = row[ResidueEmbeddingFromStructure.STREAM_NAME_ATTR]
            src_structure = row[ResidueEmbeddingFromStructure.STREAM_ATTR]
            item_name = row[ResidueEmbeddingFromStructure.ITEM_NAME_ATTR]
            progress.set_postfix_str(src_structure)
            structure = self.__structure_provider.get_structure(
                src_name=src_name,
                src_structure=stringio_from_url(src_structure) if get_structure_location(src_structure) == StructureLocation.remote else src_structure,
                structure_format=self.structure_format
            )
            for ch in get_protein_chains(structure, self.min_res_n):
                chains.append((os.path.join(self.res_embedding_location, f"{src_name}.{ch}.{self.res_embedding_format.value}"), f"{item_name}.{ch}"))
        return tuple(chains)
