import glob
import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, TypeVar

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
from foldmatch.search.query_encodings import QueryEncodingIndex
from foldmatch.search.residue_shards import ResidueCodeBatch, check_residue_output, iter_residue_codes
from foldmatch.utils.dist import RUN_ID_ENV, exported_run_id, is_run_id, run_id
from foldmatch.utils.fasta import read_id_at, read_record_at
from foldmatch.utils.sequence import FastaPreflight, normalize_sequence, preflight_fasta

logger = logging.getLogger(__name__)

_V = TypeVar("_V")

# A run's shards are {embedding folder}/emb_{run id}-{rank}.parquet. Run ids may contain '-' (utils.dist), so the
# rank is the last '-'-separated field.
_SHARD_PREFIX = "emb_"
_ANY_SHARD = re.compile(rf"{_SHARD_PREFIX}(?P<run_id>.+)-\d+\.parquet")


def run_shard_files(folder: Path, run_id: str) -> list[Path]:
    """The shards of run ``run_id`` in ``folder``, in the order rank 0 reads them (sorted by path).

    The glob is narrowed by an exact match, so a run whose id merely starts
    with this one (``abc`` vs ``abc-1``) is never picked up. The id is
    escaped in the glob: run ids hold no glob metacharacters
    (:func:`foldmatch.utils.dist.run_id`), but a name found in a folder may.
    """
    out_name = f"{_SHARD_PREFIX}{run_id}"
    shard_name = re.compile(rf"{re.escape(out_name)}-\d+\.parquet")
    return sorted(
        path for path in Path(folder).glob(f"{glob.escape(out_name)}-*.parquet")
        if shard_name.fullmatch(path.name)
    )


def shard_run_ids(folder: Path) -> list[str]:
    """The run ids of every shard in ``folder``, sorted. Read-only, file names only.

    Only well-formed run ids (:func:`foldmatch.utils.dist.is_run_id`) count:
    a file such as ``emb_copy[1]-0.parquet`` is not a shard of any run.
    """
    names = (path.name for path in Path(folder).glob(f"{_SHARD_PREFIX}*.parquet"))
    return sorted({match["run_id"] for match in map(_ANY_SHARD.fullmatch, names)
                   if match and is_run_id(match["run_id"])})


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
            *,
            run: Optional[str] = None,
    ):
        """``run`` binds the computer to the shards of that run id (``build sequences-index`` reads a finished run's
        shards): :func:`~foldmatch.utils.dist.run_id` is then never called, and a malformed id raises
        ``ValueError``. Without it, this run's id is drawn as every pass draws it."""
        self.embedding_folder = embedding_folder
        if run is None:
            # Raises on a rank > 0 that was given no id to share, before any model loads.
            self.run_id = run_id()
        elif is_run_id(run):
            self.run_id = run
        else:
            raise ValueError(
                f"{run!r} is not a run id: letters, digits, '.', '_' and '-' only, starting with a letter or digit."
            )
        self.out_name = f"{_SHARD_PREFIX}{self.run_id}"
        # Number of embeddings rank 0 must find in the shards; known for FASTA input.
        self.expected_count: Optional[int] = None
        # The pre-flight of the FASTA pass (its kept records, in input order); None for structure input.
        self.preflight: Optional[FastaPreflight] = None

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
            preflight: Optional[FastaPreflight] = None,
            residue_output: Optional[str] = None,
            scoring_system=None,
    ):
        """Compute chain embeddings from protein sequences in a FASTA file.

        ``preflight`` is the :func:`preflight_fasta` result for this file and
        ``min_res_n``, as the CLI computes it on every rank before the pass;
        its offsets feed the dataset, so the file is scanned once. Without it
        the pre-flight runs here (problems are only logged, or refused with a
        residue output).

        ``residue_output`` (``"residue_codes"`` for a build, ``"query_encoding"``
        for a query) and ``scoring_system`` go to the pass unchanged (see
        :func:`foldmatch.inference.chain_complete_inference.predict`); the
        shards then also hold the per-residue payloads, read back with
        :meth:`iter_residue_codes` or :meth:`build_query_encodings`. Without
        them the pass is called exactly as before.
        """

        residue_output = check_residue_output(residue_output)
        preflight = self.attach_preflight(fasta_file, min_res_n, preflight, strict=residue_output is not None)
        self.refuse_existing_shards()
        # Only a residue pass gets the residue keywords: a plain pass is called exactly as before.
        residue = {} if residue_output is None else {
            "residue_output": residue_output, "scoring_system": scoring_system,
        }
        # The other ranks Lightning launches inherit the exported run id.
        with exported_run_id(self.run_id):
            chain_predict(
                # EsmProtFromFasta takes the pre-flight for the path and reuses its offsets.
                src_stream=preflight,
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
                **residue,
            )
        # A rank's shard is complete only once that rank has left the pass: wait for every rank, so rank 0 may read
        # the shards (get_embedding_batches, iter_residue_codes, build_query_encodings) as soon as this returns.
        _wait_for_every_rank()

    def attach_preflight(self, fasta_file: Path, min_res_n: int, preflight: Optional[FastaPreflight] = None, *,
                         strict: bool = False) -> FastaPreflight:
        """Bind the FASTA pre-flight of this run's input to the computer and return it.

        ``preflight`` must be :func:`preflight_fasta`'s result for ``fasta_file`` and ``min_res_n`` (a mismatch
        raises ``ValueError``); without it the pre-flight runs here, its problems refused (``strict``) or logged.
        Sets :attr:`expected_count` (one embedding per kept entry, which rank 0 checks the shards against) and
        :attr:`preflight` (the input order), as a FASTA pass does before it starts.
        """
        if preflight is None:
            preflight = preflight_fasta(fasta_file, min_res_n)
            preflight.check(strict=strict)
        elif Path(preflight.path).resolve() != Path(fasta_file).resolve() or preflight.min_res_n != min_res_n:
            raise ValueError(
                f"Pre-flight of {preflight.path} (min_res_n={preflight.min_res_n}) does not "
                f"match {fasta_file} (min_res_n={min_res_n})"
            )
        # One embedding per kept entry; rank 0 checks the shards against it after the pass.
        self.expected_count = preflight.entries_kept
        self.preflight = preflight
        return preflight

    def shard_files(self) -> list[Path]:
        """This run's per-rank Parquet shards, ``emb_{run_id}-{rank}.parquet`` (:func:`run_shard_files`)."""
        return run_shard_files(self.embedding_folder, self.run_id)

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

    def get_embedding_batches(self, load_batch_size: int = 32768, *,
                              rows: bool = False) -> Iterator[tuple[list, np.ndarray]]:
        """Stream (ids, [B, D] float32) batches from all per-rank Parquet shards. Rank-0 only.

        Checks the embedding count first (:meth:`check_embedding_count`), so a
        mismatch fails before anything is built from the batches.

        With ``rows`` each batch is keyed by the rows' numbers in this read order
        (0, 1, ...) instead of their ids, so that a repeated id is still two rows:
        what :meth:`in_input_order` takes.
        """

        self.check_embedding_count()
        parquet_files = self.shard_files()
        logging.info(f"Streaming embeddings from {len(parquet_files)} Parquet shard(s) in {self.embedding_folder}/{self.out_name}-*.parquet")
        row = 0
        for parquet_file in parquet_files:
            logging.info(f"   Parquet file {parquet_file}")
            pf = pq.ParquetFile(parquet_file)
            for record in pf.iter_batches(batch_size=load_batch_size, columns=['id', 'embedding']):
                ids = record.column('id').to_pylist()
                flat = record.column('embedding').values.to_numpy(zero_copy_only=False)
                arr = np.ascontiguousarray(
                    flat.reshape(len(record), -1).astype(np.float32, copy=False)
                )
                if rows:
                    yield list(range(row, row + len(ids))), arr
                    row += len(ids)
                else:
                    yield ids, arr

    def input_ids(self) -> List[str]:
        """The ids of the kept records of the FASTA pass, in input order (duplicates included): the order of a
        1-rank run's shard. Read from the FASTA at the pre-flight's offsets, as the pass's dataset reads them."""
        preflight = self._fasta_preflight()
        with open(preflight.path, "r") as fh:
            return [read_id_at(fh, offset) for offset in preflight.offsets.tolist()]

    def input_sequences(self) -> Dict[str, str]:
        """``{id: sequence}`` of the kept records of the FASTA pass, in input order: each sequence normalised exactly
        as the pass's dataset gives it to ESM3, and a repeated id's the one of its last kept record, the record whose
        embedding a search keeps (:meth:`in_input_order`). Records the pass did not embed (below ``min_res_n``) are
        not in it, so Stage 2 aligns the sequence behind each query's Stage-1 hits."""
        preflight = self._fasta_preflight()
        out: Dict[str, str] = {}
        with open(preflight.path, "r") as fh:
            for offset in preflight.offsets.tolist():
                name, sequence = read_record_at(fh, offset)
                out[name] = normalize_sequence(sequence, allow_non_ascii=True)
        return out

    def _fasta_preflight(self) -> FastaPreflight:
        if self.preflight is None:
            raise ValueError("the input order is known only after a FASTA pass (compute_from_fasta)")
        return self.preflight

    def _row_positions(self, names: List[str]) -> np.ndarray:
        """The input position (index in ``names``, :meth:`input_ids`) of every shard row, in the order
        :meth:`get_embedding_batches` reads the rows.

        A rank's shard holds its entries in the order the distributed sampler dealt them, round robin: row ``i`` of
        rank ``r``'s shard (``emb_{run id}-{r}``) is entry ``r + i * world_size``, and a 1-rank shard is in input
        order. Checked against the ids: when every row's id is the input's at that position, the positions are
        exact, a repeated id included. Otherwise (shards not dealt that way: never by Lightning's predict sampler)
        an id is placed by its own position, which is exact for an id that occurs once; a repeated id's rows then
        take its positions in read order, with a warning. An id that is not in the input raises ``ValueError``
        (shards of another run mixed in)."""
        shards = self.shard_files()
        prefix = f"{self.out_name}-"
        ranks = [int(path.name[len(prefix):-len(".parquet")]) for path in shards]
        row_ids = [pq.ParquetFile(path).read(columns=["id"]).column("id").to_pylist() for path in shards]
        first: Dict[str, int] = {}
        for k, name in enumerate(names):
            first.setdefault(name, k)
        unknown = [qid for ids in row_ids for qid in ids if qid not in first]
        if unknown:
            raise ValueError(
                f"{len(unknown)} query id(s) from the embedding shards of run {self.run_id} are not in "
                f"{self.preflight.path} (e.g. {', '.join(map(repr, unknown[:3]))}): shards of another run are "
                f"mixed in. Remove them or use another --tmp-embedding-folder.")
        total, world = len(names), len(shards)
        if sorted(ranks) == list(range(world)):
            dealt = [np.arange(rank, total, world, dtype=np.int64) for rank in ranks]
            if all(len(pos) == len(ids) and all(names[p] == qid for p, qid in zip(pos.tolist(), ids))
                   for pos, ids in zip(dealt, row_ids)):
                return np.concatenate(dealt) if dealt else np.zeros(0, np.int64)
        repeated = len(first) < total
        if repeated:
            logger.warning(
                f"The embedding shards of run {self.run_id} are not in the order the distributed sampler deals the "
                f"entries; a query id that occurs more than once in {self.preflight.path} may be reported with the "
                f"embedding of another of its records.")
        later: Dict[str, List[int]] = {}
        for k, name in enumerate(names):
            later.setdefault(name, []).append(k)
        for positions in later.values():
            positions.reverse()                      # pop() gives an id's positions in input order
        out = []
        for ids in row_ids:
            for qid in ids:
                positions = later[qid]
                out.append(positions.pop() if positions else first[qid])
        return np.asarray(out, dtype=np.int64)

    def in_input_order(self, results: Mapping[int, _V]) -> Dict[str, _V]:
        """``results`` of a search keyed by shard row (as built from :meth:`get_embedding_batches` with ``rows``),
        keyed by query id in the input FASTA's order, with the values a 1-rank run gives. Rank-0 only, after a
        FASTA pass.

        Rank 0 reads the shards rank by rank, and the distributed sampler deals the entries round-robin, so in a
        multi-rank run the rows come in rank order (rank 0's first). Put back in input order
        (:meth:`_row_positions`), a multi-rank run writes the queries of a 1-rank run, whose shard is already in
        input order: each id at its first kept record, and a repeated id (duplicate ids, which only an embedding
        search and BLOSUM62 accept) with the value of its last kept record, as a 1-rank run's search (a dict, the
        later record replacing the earlier) gives it. Values are never changed. A row that is not in ``results``
        (a filtered subset) is left out.
        """
        names = self.input_ids()
        positions = self._row_positions(names)
        by_position: Dict[int, _V] = {}
        for row, value in results.items():
            if not 0 <= row < len(positions):
                raise ValueError(f"row {row!r} is not a row of the embedding shards of run {self.run_id} "
                                 f"({len(positions)} rows)")
            by_position[int(positions[row])] = value
        last: Dict[str, int] = {}
        for position in sorted(by_position):
            last[names[position]] = position          # dict order: each id at its first position
        return {name: by_position[position] for name, position in last.items()}

    def iter_residue_codes(self, system, load_batch_size: int = 32768) -> Iterator[ResidueCodeBatch]:
        """Stream this run's residue codes, one shard at a time in shard order. Rank-0 only.

        Yields :class:`~foldmatch.search.residue_shards.ResidueCodeBatch` ``(ids, codes, seq_sha1s)`` (one uint8
        ``[L, 8]`` array per entry; what ``ResidueCodeStore.create`` takes), ``load_batch_size`` entries at a time,
        reading only the ``id``, ``residue_codes`` and ``seq_sha1`` columns.
        Before anything is yielded: the embedding count (:meth:`check_embedding_count`) and every shard's schema
        (:func:`~foldmatch.search.residue_shards.check_shards`): shards without residue codes, mixed shards and
        shards of another encoding than ``system``'s are refused (``ResidueShardError``).
        """
        self.check_embedding_count()
        return iter_residue_codes(self.shard_files(), system, load_batch_size)

    def build_query_encodings(self, dst, system) -> QueryEncodingIndex:
        """This run's query encodings as a per-run fp16 memmap at ``dst`` (:meth:`QueryEncodingIndex.build`, one
        pass in shard order; the same shard checks, plus one encoding per input entry). Rank-0 only."""
        self.check_embedding_count()
        return QueryEncodingIndex.build(self.shard_files(), dst, system=system, expected_count=self.expected_count)

def _wait_for_every_rank() -> None:
    """``torch.distributed.barrier()`` when the pass ran in a process group (every rank calls the pass, so every rank
    reaches it); nothing otherwise."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


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
