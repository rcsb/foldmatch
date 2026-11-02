import logging

import pandas as pd
import torch
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from torch.utils.data import IterableDataset, get_worker_info

from foldmatch.dataset.utils import get_structure_location, read_structure_list, refuse_shared_names
from foldmatch.types.api_types import StructureLocation, StructureFormat, SrcLocation, FileOrStreamTuple
from foldmatch.utils.data import stringio_from_url
from foldmatch.utils.structure_parser import protein_chain_iter, rename_atom_attr, filter_residues
from foldmatch.utils.structure_provider import StructureProvider

logger = logging.getLogger(__name__)

class EsmProtFromStructure(IterableDataset):

    STREAM_NAME_ATTR = 'stream_name'
    STREAM_ATTR = 'stream'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [STREAM_NAME_ATTR, STREAM_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream: FileOrStreamTuple,
            src_location: SrcLocation = SrcLocation.file,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res_n: int = 0,
            structure_provider: StructureProvider = None
    ):
        super().__init__()
        self.min_res_n = min_res_n
        self.src_location = src_location
        self.structure_format = structure_format
        self.__structure_provider = structure_provider if structure_provider is not None else StructureProvider()
        self.data = pd.DataFrame()
        self.__load_stream(src_stream)

    def __load_stream(self, src_stream):
        # The structure list, whose columns are COLUMNS: a row of more than 3 fields is refused
        self.data = read_structure_list(src_stream, self.src_location, keep_default_na=False)
        refuse_shared_names(self.data, EsmProtFromStructure.STREAM_NAME_ATTR, EsmProtFromStructure.STREAM_ATTR)

    def __iter__(self):
        # Handle multiple workers by splitting data across workers
        worker_info = get_worker_info()

        # Get distributed info
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        if worker_info is None:
            # Single worker: just handle DDP split
            per_rank = int(len(self.data) / world_size)
            iter_start = rank * per_rank
            iter_end = iter_start + per_rank if rank < world_size - 1 else len(self.data)
            iter_data = self.data.iloc[iter_start:iter_end]
            logging.debug(f"Rank {rank} processing {iter_start}:{iter_end} structures")
        else:
            # Multiple workers: split by rank first, then by worker
            per_rank = int(len(self.data) / world_size)
            rank_start = rank * per_rank
            rank_end = rank_start + per_rank if rank < world_size - 1 else len(self.data)
            rank_data = self.data.iloc[rank_start:rank_end]

            per_worker = int(len(rank_data) / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker if worker_id < worker_info.num_workers - 1 else len(rank_data)
            iter_data = rank_data.iloc[iter_start:iter_end]
            logging.debug(f"Rank {rank} processing {rank_start}:{rank_end} worker {worker_id} processing {iter_start}:{iter_end} structures")

        # Iterate through structures and yield chains
        for idx, row in iter_data.iterrows():
            src_name = row[EsmProtFromStructure.STREAM_NAME_ATTR]
            src_structure = row[EsmProtFromStructure.STREAM_ATTR]
            item_name = row[EsmProtFromStructure.ITEM_NAME_ATTR]

            # Load structure once
            structure = self.__structure_provider.get_structure(
                src_name=src_name,
                src_structure=stringio_from_url(src_structure) if get_structure_location(src_structure) == StructureLocation.remote else src_structure,
                structure_format=self.structure_format
            )

            # Each protein chain of the structure once, its pieces joined
            for atom_ch in protein_chain_iter(structure, self.min_res_n):
                chain_id = atom_ch.chain_id[0]
                try:
                    atom_ch = filter_residues(atom_ch)
                    atom_ch = rename_atom_attr(atom_ch)
                    protein_chain = ProteinChain.from_atomarray(atom_ch)
                    protein_chain = ESMProtein.from_protein_chain(protein_chain)
                except Exception as e:
                    raise IOError(f"Error while creating ESMProtein from structure chain {src_name}.{chain_id}: {e}")

                if len(protein_chain) == 0:
                    raise IOError(f"No atoms were found in structure chain {src_name}.{chain_id}")

                yield protein_chain, f"{item_name}.{chain_id}"
