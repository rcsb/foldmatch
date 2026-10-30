import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.dataset.utils import get_structure_location, refuse_shared_names
from foldmatch.types.api_types import SrcLocation, StructureLocation, StructureFormat, ResEmbeddingFormat, \
    FileOrStreamTuple
from foldmatch.utils.data import stringio_from_url
from foldmatch.utils.structure_parser import get_assemblies
from foldmatch.utils.structure_provider import StructureProvider


class ResidueAssemblyDatasetFromStructure(ResidueAssemblyEmbeddingFromTensorFile):

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
            max_res_n: int = sys.maxsize,
            res_embedding_format: ResEmbeddingFormat = ResEmbeddingFormat.pt,
            structure_provider: StructureProvider = None
    ):
        self.src_location = src_location
        self.structure_format = structure_format
        self.min_res_n = min_res_n
        self.max_res_n = max_res_n
        super().__init__(
            src_stream=self.__get_assemblies(src_stream),
            res_embedding_location=res_embedding_location,
            src_location=SrcLocation.stream,
            structure_format=structure_format,
            min_res_n=min_res_n,
            max_res_n=max_res_n,
            res_embedding_format=res_embedding_format,
            structure_provider=structure_provider
        )

    def __get_assemblies(self, src_stream):
        assemblies = []
        data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=ResidueAssemblyDatasetFromStructure.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=ResidueAssemblyDatasetFromStructure.COLUMNS
        )
        data = data.sort_values(by=data.columns[0])
        refuse_shared_names(data, ResidueAssemblyDatasetFromStructure.STREAM_NAME_ATTR,
                            ResidueAssemblyDatasetFromStructure.STREAM_ATTR)
        progress = tqdm(data.iterrows(), total=len(data), desc="Loading structures")
        for idx, row in progress:
            src_name = row[ResidueAssemblyDatasetFromStructure.STREAM_NAME_ATTR]
            src_structure = row[ResidueAssemblyDatasetFromStructure.STREAM_ATTR]
            progress.set_postfix_str(src_structure)
            structure = stringio_from_url(src_structure) if get_structure_location(src_structure) == StructureLocation.remote else src_structure
            item_name = row[ResidueAssemblyDatasetFromStructure.ITEM_NAME_ATTR]
            for assembly_id in get_assemblies(structure=structure, structure_format=self.structure_format):
                assemblies.append((src_name, src_structure, str(assembly_id), f"{item_name}-{assembly_id}"))

        return tuple(assemblies)
