import logging
import re
from pathlib import Path
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
from foldmatch.utils.dist import RUN_ID_ENV, exported_run_id, run_id
from foldmatch.utils.sequence import scan_fasta

logger = logging.getLogger(__name__)

class EmbeddingComputer:
    """Compute chain or assembly embeddings via a residue->chain inference pipeline.

    Owns the per-run shard naming and the multi-GPU bookkeeping. Every rank
    writes its embeddings to ``{embedding_folder}/emb_{run_id}-{rank}.parquet``
    under one run id shared by all ranks (:mod:`foldmatch.utils.dist`), and
    rank 0 reads back exactly that run's shards. Before a pass every rank
    refuses to start if shards with this run id already exist, and after a
    FASTA pass rank 0 checks that the shards hold one embedding per input
    entry, so a lost rank or a shard of an earlier run with the same id fails
    loudly instead of being overwritten or merged.
    """

    def __init__(
            self,
            embedding_folder: Path,
    ):
        self.embedding_folder = embedding_folder
        # Raises on a rank > 0 that was given no id to share, before any model loads.
        self.run_id = run_id()
        self.out_name = f"emb_{self.run_id}"
        # Number of embeddings rank 0 must find in the shards; known for FASTA input.
        self.expected_count: Optional[int] = None

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
        self.refuse_existing_shards()

        logging.info(f"Listing structure files from: {structure_folder}")
        structure_files = tuple(structure_folder.glob(f"*{file_extension}"))
        if not structure_files:
            raise ValueError(
                f"No structure files found with extension {file_extension} in {structure_folder}"
            )

        # The other ranks Lightning launches inherit the exported run id.
        with exported_run_id(self.run_id):
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
    ):
        """Compute chain embeddings from protein sequences in a FASTA file."""

        # One embedding per entry that passes min_res_n, the same filter the
        # dataset applies; rank 0 checks the shards against it after the pass.
        self.expected_count = sum(
            1 for _offset, _name, length in scan_fasta(fasta_file)
            if min_res_n <= 0 or length >= min_res_n
        )
        self.refuse_existing_shards()
        # The other ranks Lightning launches inherit the exported run id.
        with exported_run_id(self.run_id):
            chain_predict(
                src_stream=fasta_file,
                src_from=SrcEsmFrom.fasta,
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

    def shard_files(self) -> list[Path]:
        """This run's per-rank Parquet shards, ``emb_{run_id}-{rank}.parquet``.

        The glob is narrowed by an exact match, so a run whose id merely starts
        with this one (``abc`` vs ``abc-1``) is never picked up.
        """
        shard_name = re.compile(rf"{re.escape(self.out_name)}-\d+\.parquet")
        return sorted(
            path for path in Path(self.embedding_folder).glob(f"{self.out_name}-*.parquet")
            if shard_name.fullmatch(path.name)
        )

    def refuse_existing_shards(self) -> None:
        """Raise ``FileExistsError`` if shards with this run id already exist. Read-only.

        Every rank runs it before the pass, so before ESM3 loads and before any
        rank writes (the ranks only write after they have all joined the process
        group). Shards with this id can only come from an earlier run with the
        same id, e.g. the same command run twice in one srun step, or a reused
        ``FOLDMATCH_RUN_ID``: this run would overwrite them, or rank 0 would
        mix them into its own.
        """
        existing = self.shard_files()
        if existing:
            raise FileExistsError(
                f"{len(existing)} embedding shard(s) of an earlier run with the same run id "
                f"({self.run_id}) are in {self.embedding_folder} ({existing[0].name}, ...). "
                f"This run would overwrite them or mix them into its own: remove them, use "
                f"another --tmp-embedding-folder, or set a different {RUN_ID_ENV}."
            )

    def count_embeddings(self) -> int:
        """Rows over this run's shards, from the Parquet footers (no data read)."""
        return sum(pq.ParquetFile(path).metadata.num_rows for path in self.shard_files())

    def check_embedding_count(self) -> None:
        """Raise unless the shards hold exactly :attr:`expected_count` embeddings. Rank-0 only.

        A shortfall means a rank's shard is missing; an excess means shards of
        an earlier run with the same id (e.g. the same SLURM step) are mixed in.
        A no-op when the expected count is unknown (structure input).
        """
        if self.expected_count is None:
            return
        found = self.count_embeddings()
        if found != self.expected_count:
            raise ValueError(
                f"Found {found} embeddings in {len(self.shard_files())} shard(s) "
                f"{self.embedding_folder}/{self.out_name}-*.parquet, but the input has "
                f"{self.expected_count} entries. Either a rank's shard is missing, or "
                f"shards of an earlier run with the same run id ({self.run_id}) are mixed "
                f"in: remove them or use another --tmp-embedding-folder."
            )

    def remove_shards(self) -> None:
        """Delete this run's shards. Rank-0 only, once every rank has finished writing."""
        for path in self.shard_files():
            path.unlink(missing_ok=True)

    def get_embedding_batches(self, load_batch_size: int = 32768) -> Iterator[tuple[list[str], np.ndarray]]:
        """Stream (ids, [B, D] float32) batches from all per-rank Parquet shards. Rank-0 only.

        Checks the embedding count first (:meth:`check_embedding_count`), so a
        mismatch fails before anything is built from the batches.
        """

        self.check_embedding_count()
        parquet_files = self.shard_files()
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
