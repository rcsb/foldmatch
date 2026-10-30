import logging
import sys
from pathlib import Path

import torch
import pandas as pd
import pyarrow.parquet as pq

from torch.utils.data import Dataset

from foldmatch.dataset.utils import get_structure_location
from foldmatch.types.api_types import StructureLocation, StructureFormat, SrcLocation, FileOrStreamTuple
from foldmatch.utils.data import stringio_from_url
from foldmatch.utils.structure_parser import get_protein_chains
from foldmatch.utils.structure_provider import StructureProvider


class ResidueAssemblyEmbeddingFromParquet(Dataset):

    STREAM_NAME_ATTR = 'stream_name'
    STREAM_ATTR = 'stream'
    ASSEMBLY_ATTR = 'assembly_id'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [STREAM_NAME_ATTR, STREAM_ATTR, ASSEMBLY_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream: FileOrStreamTuple,
            parquet_path: Path,
            embedding_dim: int,
            src_location: SrcLocation = SrcLocation.file,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res_n: int = 0,
            max_res_n: int = sys.maxsize,
            structure_provider: StructureProvider = None
    ):
        super().__init__()
        self.src_location = src_location
        self.structure_format = structure_format
        self.min_res_n = min_res_n
        self.max_res_n = max_res_n
        self.embedding_dim = embedding_dim
        self.data = pd.DataFrame()
        self.__load_stream(src_stream)
        self.__structure_provider = structure_provider if structure_provider is not None else StructureProvider()
        self.table = pq.read_table(parquet_path)
        self._id_index = {
            self.table.column('id')[i].as_py(): i
            for i in range(len(self.table))
        }

    def __load_stream(self, src_stream):
        self.data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=ResidueAssemblyEmbeddingFromParquet.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            dtype=str,
            names=ResidueAssemblyEmbeddingFromParquet.COLUMNS
        )
        self.data = self.data.sort_values(by=self.data.columns[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        logger = logging.getLogger(__name__)
        src_name = self.data.iloc[idx][ResidueAssemblyEmbeddingFromParquet.STREAM_NAME_ATTR]
        src_structure = self.data.iloc[idx][ResidueAssemblyEmbeddingFromParquet.STREAM_ATTR]
        assembly_id = self.data.iloc[idx][ResidueAssemblyEmbeddingFromParquet.ASSEMBLY_ATTR]
        item_name = self.data.iloc[idx][ResidueAssemblyEmbeddingFromParquet.ITEM_NAME_ATTR]
        structure = self.__structure_provider.get_structure(
            src_name=src_name,
            src_structure=stringio_from_url(src_structure) if get_structure_location(src_structure) == StructureLocation.remote else src_structure,
            structure_format=self.structure_format,
            assembly_id=assembly_id
        )
        chain_ids = get_protein_chains(structure, self.min_res_n)
        tensors = []
        total_residues = 0
        for ch in chain_ids:
            lookup_id = f"{src_name}.{ch}"
            if lookup_id not in self._id_index:
                logger.warning(f"Chain {lookup_id} not found in parquet file")
                continue
            row_idx = self._id_index[lookup_id]
            embedding_flat = self.table.column('embedding')[row_idx].as_py()
            tensor = torch.tensor(embedding_flat, dtype=torch.float32).reshape(-1, self.embedding_dim)
            total_residues += tensor.shape[0]
            tensors.append(tensor)
            if total_residues > self.max_res_n:
                break
        if not tensors:
            raise ValueError(f"No chains found for {src_name}-{assembly_id} in parquet file.")
        return torch.cat(tensors, dim=0), item_name
