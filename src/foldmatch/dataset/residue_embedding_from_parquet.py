import logging
from pathlib import Path
from typing import List

import torch

from torch.utils.data import Dataset

from foldmatch.utils.data import residue_embeddings_from_parquet

logger = logging.getLogger(__name__)

class ResidueEmbeddingFromParquet(Dataset):

    def __init__(
            self,
            parquet_path: Path | List[Path]
    ):
        super().__init__()
        if isinstance(parquet_path, (str, Path)):
            paths = [parquet_path]
        else:
            paths = parquet_path
        logger.info(f"Loading {len(paths)} parquet file(s): {[str(p) for p in paths]}")
        # One row per chain, one list of D values per residue: the [L, D] tables, D from each file's column type.
        self.ids = []
        self.embeddings = []
        for path in paths:
            ids, embeddings = residue_embeddings_from_parquet(path)
            self.ids.extend(ids)
            self.embeddings.extend(embeddings)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), self.ids[idx]
