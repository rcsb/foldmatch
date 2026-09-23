import logging
import os
from pathlib import Path
from secrets import token_hex
from typing import Iterator, Optional

import numpy as np
import pyarrow.parquet as pq

from foldmatch.inference.assembly_complete_inference import predict as assembly_predict
from foldmatch.inference.chain_complete_inference import predict as chain_predict
from foldmatch.types.api_types import (
    StructureFormat,
    SrcLocation,
    OutFormat,
    Accelerator,
    Granularity, SrcEsmFrom, Devices, Strategy,
)

logger = logging.getLogger(__name__)

class EmbeddingComputer:
    """Compute chain or assembly embeddings via a residue->chain inference pipeline.

    Owns the per-run scratch directories and the multi-GPU bookkeeping: each
    rank's __init__ creates its own per-rank scratch dir, then after residue
    inference all ranks consolidate onto rank 0's directory so chain/assembly
    inference and tensor loading see a single shared location.
    """

    def __init__(
            self,
            embedding_folder: Path,
    ):
        self.embedding_folder = embedding_folder
        self.out_name = f"emb_{_consolidate_id()}"

    def compute_from_structures(
            self,
            structure_folder: Path,
            structure_format: StructureFormat = StructureFormat.mmcif,
            min_res: int = 10,
            granularity: Granularity = Granularity.chain,
            file_extension: Optional[str] = None,
            batch_size: int = 1,
            num_workers: int = 0,
            num_nodes: int = 1,
            accelerator: Accelerator = Accelerator.auto,
            devices: Devices = 'auto',
            strategy: Strategy = Strategy.auto,
    ):
        """Compute chain or assembly embeddings from a directory of structure files."""
        if file_extension is None:
            file_extension = '.cif' if structure_format == StructureFormat.mmcif else '.pdb'

        structure_folder = Path(structure_folder)
        if not structure_folder.exists():
            raise ValueError(f"Structure directory does not exist: {structure_folder}")

        logging.info(f"Listing structure files from: {structure_folder}")
        structure_files = tuple(structure_folder.glob(f"*{file_extension}"))
        if not structure_files:
            raise ValueError(
                f"No structure files found with extension {file_extension} in {structure_folder}"
            )

        if granularity == 'chain':
            chain_predict(
                src_stream=[
                    (str_file.stem, str_file, str_file.stem)
                    for str_file in structure_files
                ],
                src_location=SrcLocation.stream,
                structure_format=structure_format,
                min_res_n=min_res,
                out_folder=self.embedding_folder,
                out_name=self.out_name,
                accelerator=accelerator,
                batch_size=batch_size,
                num_workers=num_workers,
                num_nodes=num_nodes,
                devices=devices,
                strategy=strategy,
                out_format=OutFormat.parquet,
                return_predictions=False,
            )
        else:
            assembly_predict(
                src_stream=[
                    (str_file.stem, str_file, str_file.stem)
                    for str_file in structure_files
                ],
                src_location=SrcLocation.stream,
                structure_format=structure_format,
                min_res_n=min_res,
                out_folder=self.embedding_folder,
                out_name=self.out_name,
                accelerator=accelerator,
                batch_size=batch_size,
                num_workers=num_workers,
                num_nodes=num_nodes,
                devices=devices,
                strategy=strategy,
                out_format=OutFormat.parquet,
                return_predictions=False,
            )

    def compute_from_fasta(
            self,
            fasta_file: Path,
            min_res_n: int = 0,
            batch_size: int = 1,
            num_workers: int = 0,
            num_nodes: int = 1,
            devices: Devices = 'auto',
            strategy: Strategy = Strategy.auto,
            accelerator: Accelerator = Accelerator.auto,
            three_di: Optional[str] = None,
            three_di_output: str = "codes",
    ):
        """Compute chain embeddings from protein sequences in a FASTA file.

        ``three_di`` names a scoring system (e.g. ``'3di-v1-coded'``) whose predicted 3Di
        encoding is written alongside each embedding, from the same ESM3 pass. This is the
        sequence-only case: there are no coordinates to read a structural alphabet from,
        so predicting it is the only way a sequence database can carry one at all.
        """

        chain_predict(
            src_stream=fasta_file,
            src_from=SrcEsmFrom.fasta,
            three_di=three_di,
            three_di_output=three_di_output,
            min_res_n=min_res_n,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            accelerator=accelerator,
            devices=devices,
            out_format=OutFormat.parquet,
            out_folder=self.embedding_folder,
            out_name=self.out_name,
            strategy=strategy,
            return_predictions=False,
        )

    def get_three_di_batches(self, load_batch_size: int = 32768) -> Iterator[tuple[str, np.ndarray]]:
        """Stream ``(id, codes)`` from the shards' ``three_di`` column. Rank-0 only.

        A separate pass over the same Parquet files as :meth:`get_embedding_batches`,
        projecting a different column, so the index build is untouched by the existence
        of 3Di codes - and so a database can be built without them by simply not calling
        this.
        """
        parquet_files = sorted(Path(self.embedding_folder).glob(f"{self.out_name}-*.parquet"))
        for parquet_file in parquet_files:
            pf = pq.ParquetFile(parquet_file)
            if 'three_di' not in pf.schema_arrow.names:
                raise ValueError(
                    f"{parquet_file} has no 'three_di' column: the embeddings were "
                    f"computed without a 3Di head, so this database cannot carry codes")
            for record in pf.iter_batches(batch_size=load_batch_size,
                                          columns=['id', 'three_di']):
                ids = record.column('id').to_pylist()
                codes = record.column('three_di').to_pylist()
                for identifier, blob in zip(ids, codes):
                    yield identifier, np.frombuffer(blob, dtype=np.uint8)

    def get_three_di_profiles(self, load_batch_size: int = 32768) -> dict:
        """``{id: (L, 20) float32}`` from the shards' ``three_di`` column.

        The query counterpart of :meth:`get_three_di_batches`: the same column, written
        by a run configured with ``three_di_output='profile'``, read back as the
        distribution rather than as codes. Returned whole rather than streamed because a
        query set is small and the aligner wants them all at once; a *database* must use
        the code path instead, which is 1/40th the size.
        """
        profiles = {}
        parquet_files = sorted(Path(self.embedding_folder).glob(f"{self.out_name}-*.parquet"))
        for parquet_file in parquet_files:
            pf = pq.ParquetFile(parquet_file)
            if 'three_di' not in pf.schema_arrow.names:
                raise ValueError(
                    f"{parquet_file} has no 'three_di' column: these embeddings were "
                    f"computed without a 3Di head")
            for record in pf.iter_batches(batch_size=load_batch_size,
                                          columns=['id', 'three_di']):
                for identifier, blob in zip(record.column('id').to_pylist(),
                                            record.column('three_di').to_pylist()):
                    flat = np.frombuffer(blob, dtype=np.float16)
                    if flat.size % 20:
                        raise ValueError(
                            f"{identifier}: {flat.size} float16 values is not a whole "
                            f"number of 20-state rows — was this shard written with "
                            f"three_di_output='codes'?")
                    profiles[identifier] = flat.reshape(-1, 20).astype(np.float32)
        return profiles

    def get_embedding_batches(self, load_batch_size: int = 32768) -> Iterator[tuple[list[str], np.ndarray]]:
        """Stream (ids, [B, D] float32) batches from all per-rank Parquet shards. Rank-0 only."""

        parquet_files = sorted(Path(self.embedding_folder).glob(f"{self.out_name}-*.parquet"))
        logging.info(f"Streaming embeddings from {len(parquet_files)} Parquet shard(s) in {self.embedding_folder}/{self.out_name}-*.parquet")
        for parquet_file in parquet_files:
            logging.info(f"   Parquet file {parquet_file}")
            pf = pq.ParquetFile(parquet_file)
            for record in pf.iter_batches(batch_size=load_batch_size, columns=['id', 'embedding']):
                ids = record.column('id').to_pylist()
                flat = record.column('embedding').values.to_numpy(zero_copy_only=False)
                arr = np.ascontiguousarray(
                    flat.reshape(len(record), -1).astype(np.float32, copy=False)
                )
                yield ids, arr

def compute_from_structure_file(
        query_structure: Path,
        structure_format: StructureFormat = StructureFormat.mmcif,
        granularity: Granularity = Granularity.chain,
        chain_id: str = None,
        assembly_id: str = None,
        min_res: int = 10,
        max_res: int = None,
        accelerator: Accelerator = Accelerator.auto,
) -> Iterator[tuple[list[str], np.ndarray]]:
    """Compute per-chain (or per-assembly) embeddings for a single structure
    file and yield them in the same ``(ids, [B, D] float32)`` batch shape as
    :meth:`get_embedding_batches`. A single structure produces one batch
    containing all of its chains (or assemblies).
    """
    query_path = Path(query_structure)
    if not query_path.exists():
        raise ValueError(f"Query structure file does not exist: {query_structure}")

    if granularity != 'chain' and granularity != 'assembly':
        raise ValueError(f"Unknown granularity value {granularity}")

    structure_name = query_path.stem
    logging.info(f"Processing residue embeddings: {structure_name}")

    from foldmatch.foldmatch import FoldMatch
    foldmatch = FoldMatch(
        min_res=min_res,
        max_res=max_res
    )
    import torch
    if accelerator == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif accelerator == "cpu":
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cuda")
    foldmatch.load_models(device=torch_device)

    residue_embeddings = foldmatch.residue_embedding_by_chain(
        src_structure=query_structure,
        structure_format=structure_format,
        chain_id=chain_id
    ) if granularity == 'chain' else foldmatch.residue_embedding_by_assembly(
        src_structure=query_structure,
        structure_format=structure_format,
        assembly_id=assembly_id
    )

    if not residue_embeddings:
        if chain_id:
            raise ValueError(f"Chain {chain_id} not found or does not meet minimum residue requirements")
        elif assembly_id:
            raise ValueError(f"Assembly {assembly_id} not found or does not meet minimum residue requirements")
        else:
            raise ValueError("No valid chains found in query structure")

    granularity_str = granularity.value if hasattr(granularity, 'value') else granularity
    separator = '.' if granularity_str == 'chain' else '-'

    ids: list[str] = []
    embeddings: list[np.ndarray] = []
    for embedding_id, residue_embedding in residue_embeddings.items():
        logging.info(
            f"Aggregating {structure_name} {granularity_str} {embedding_id} "
            f"({residue_embedding.shape[0]} residues)..."
        )
        full_embedding = foldmatch.aggregator_embedding(residue_embedding)
        # aggregator returns a torch.Tensor; coerce to a 1-D float32 numpy vector
        if hasattr(full_embedding, 'detach'):
            full_embedding = full_embedding.detach().cpu().numpy()
        full_embedding = np.asarray(full_embedding, dtype=np.float32).reshape(-1)
        ids.append(f"{structure_name}{separator}{embedding_id}")
        embeddings.append(full_embedding)

    if not ids:
        return
    arr = np.ascontiguousarray(np.stack(embeddings).astype(np.float32, copy=False))
    yield ids, arr

def _consolidate_id():
    return os.environ.get('SLURM_JOB_ID', token_hex(16))