import glob
import os
import logging
import sqlite3
import typer
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, List, Mapping, Optional

from foldmatch import __version__
from foldmatch.cli.args_utils import DOWNLOAD_MODELS_HELP, DOWNLOAD_MODELS_SYSTEM_HELP, arg_devices, \
    download_models_command, set_log_level
from foldmatch.errors import RefusalError
from foldmatch.search.db_path import parse_db_path
from foldmatch.search.output import (
    CLUSTER_OUTPUT_FIELDS,
    DEFAULT_FORMAT_OUTPUT,
    EMBEDDING_OUTPUT_FIELDS,
    ESM_CODES_OUTPUT_FIELDS,
    SUPPORTED_OUTPUT_FIELDS,
    write_embedding_results,
)
from foldmatch.types.api_types import (
    StructureFormat,
    Accelerator,
    Strategy,
    Granularity,
    IndexConfig,
    IndexType,
    LogLevel,
    SignificanceMode,
    FormatMode,
    ScoringChoice,
    DEFAULT_SCORING,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help=f"3D structure search using embeddings and FAISS. Version: {__version__}."
)

build_db_app = typer.Typer(
    add_completion=False,
    help="Build an embedding database."
)
app.add_typer(build_db_app, name="build")

query_db_app = typer.Typer(
    add_completion=False,
    help="Query an existing embedding database."
)
app.add_typer(query_db_app, name="query")

# Help text for the Stage-2 --format-output option, shared by the commands that
# run sequence-identity alignment.
# Every search command writes the same shape of file: tab-separated rows, no
# header. Embedding-only searches emit the EMBEDDING_OUTPUT_FIELDS columns;
# sequence-identity (Stage-2) searches emit the --format-output columns.
_OUTPUT_FILE_HELP = (
    "Path to write results, as tab-separated rows with no header. "
    f"An embedding-only search writes: {', '.join(EMBEDDING_OUTPUT_FIELDS)}."
)

_FORMAT_MODE_HELP = (
    "Layout of the output file: 'headless_tsv' (tab-separated, no header) "
    "or 'tsv' (same, with a header row naming each column)."
)

_FORMAT_OUTPUT_HELP = (
    "Comma-separated columns for the Stage-2 sequence-identity output, written "
    "as tab-separated rows with no header. "
    f"Available fields: {', '.join(SUPPORTED_OUTPUT_FIELDS)}. "
    f"Only 'query sequences --scoring esm-codes' fills {', '.join(ESM_CODES_OUTPUT_FIELDS)}; "
    "they are empty under BLOSUM62 scoring (and in 'query db')."
)

# Stage 2 keeps a hit only if min(qcov, tcov) >= --min-coverage.
_MIN_COVERAGE_HELP = (
    "When --seq-identity is set, keep only hits whose query coverage and "
    "subject coverage both reach this fraction (0-1); a hit is dropped when "
    "either side is below it."
)

# query sequences: one Stage-2 scoring per run (plan P6.1, D7). The filter defaults depend on it (D8), and the
# BLOSUM62-only options are refused under esm-codes (D20: exit status 2).
_SCORING_HELP = (
    "Stage-2 scoring, one per run: 'blosum62' (BLOSUM62 Smith-Waterman with BLAST-style "
    "statistics) or 'esm-codes' (alignment of the ESM3 residue codes, with E-values calibrated "
    "over the database's residues; needs a database built with 'build sequences "
    "--residue-codes', and uses the scoring system its residue-code manifest names). Under "
    "esm-codes the identity and coverage filters default to 0, --gap-open, --gap-extend, "
    "--comp-bias-corr, --significance-mode and --no-seq-identity are refused, and query "
    "FASTA files with duplicate ids, '<' or '>' are refused."
)

_QUERY_MIN_SEQ_IDENTITY_HELP = (
    "When --seq-identity is set, drop hits whose alignment identity is below this "
    "fraction (0-1). Default: 0.3 under --scoring blosum62, 0 under esm-codes."
)

_QUERY_MIN_COVERAGE_HELP = (
    f"{_MIN_COVERAGE_HELP} Default: 0.8 under --scoring blosum62, 0 under esm-codes."
)

_QUERY_MAX_EVALUE_HELP = (
    "When --seq-identity is set, drop hits whose database E-value is above this "
    "(default 1e-3); pass 'inf' to disable E-value filtering. Under --scoring blosum62 "
    "it forces the significance pass (and, in --significance-mode default, its 11/1 gap "
    "penalties); under esm-codes the E-value is the calibrated one, over the database's "
    "residue count."
)

_BLOSUM62_ONLY = " BLOSUM62 scoring only: refused with --scoring esm-codes."

_CLEANUP_EMBEDDINGS_HELP = (
    "Delete this build's intermediate embedding shards "
    "(emb_<run id>-<rank>.parquet) from --tmp-embedding-folder once the "
    "database is built. By default they are kept."
)

# Plan section 8, P5.6 (verbatim).
_RESIDUE_CODES_HELP = (
    "Also compute 8-byte per-residue ESM3 codes in the same pass, enabling "
    "`query sequences --scoring esm-codes`. Adds 8 B per residue plus up to "
    "about 100 B per entry for the index, and a one-time ~26 MB download. "
    "Sequence (FASTA) databases only."
)

_FROM_SHARDS_HELP = (
    "Folder holding the embedding shards that this database's 'build sequences "
    "--residue-codes' run kept (its --tmp-embedding-folder; a build keeps them "
    "unless --cleanup-embeddings)."
)


@build_db_app.command(
    name="structures",
    help="Build an embedding database from a directory of structure files."
)
def build_database_from_structures(
        structure_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing structure files.'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        tmp_embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder for intermediate embeddings.'
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif, binarycif, or pdb).'
        )] = StructureFormat.mmcif,
        granularity: Annotated[Granularity, typer.Option(
            help='Calculate embeddings for "chain" or "assembly" level.'
        )] = Granularity.chain,
        file_extension: Annotated[Optional[str], typer.Option(
            help='File extension to filter (e.g., .cif, .bcif, or .pdb). If not specified, uses default for format.'
        )] = None,
        min_res: Annotated[int, typer.Option(
            help='Minimum residue length for chains.'
        )] = 10,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = ['auto'],
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        index_type: Annotated[IndexType, typer.Option(
            help='FAISS index variant: auto (legacy heuristic), flat, hnsw, or ivf_pq (on-disk).'
        )] = IndexType.auto,
        ivf_nlist: Annotated[int, typer.Option(
            help='IVF cell count for ivf_pq index.'
        )] = 16384,
        ivf_nprobe: Annotated[Optional[int], typer.Option(
            help='Cells probed per query for ivf_pq (defaults to nlist // 64).'
        )] = None,
        load_batch_size: Annotated[int, typer.Option(
            help='Number of vectors per batch streamed into the FAISS index. Larger '
                 'batches (e.g. 65536+) better saturate many-core nodes during '
                 'IVF-PQ build; smaller batches use less memory.'
        )] = 32768,
        cleanup_embeddings: Annotated[bool, typer.Option(
            help=_CLEANUP_EMBEDDINGS_HELP
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    """Build an embedding database from structure files."""

    set_log_level(log_level)

    # Parse output_db into directory and prefix
    # Files will be saved as: {output_db}.index and {output_db}.metadata
    output_db_path = Path(output_db)
    db_dir = output_db_path.parent
    index_name = output_db_path.name

    # Ensure we have a valid directory and prefix
    if not index_name:
        index_name = "embeddings"
    if db_dir == Path('.'):
        db_dir = Path.cwd()  # Use current directory explicitly
    output_db = str(db_dir / index_name)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS index: enabled")

    # Read-only, on every rank, before ESM3.
    _refuse_existing_outputs(output_db, _INDEX_SUFFIXES)

    from foldmatch.search.embedding_computer import EmbeddingComputer
    embedding_computer = EmbeddingComputer(
        embedding_folder=tmp_embedding_folder,
    )
    embedding_computer.compute_from_structures(
        structure_folder=structure_folder,
        structure_format=structure_format,
        min_res=min_res,
        granularity=granularity,
        file_extension=file_extension,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        strategy=strategy,
    )

    if not _is_rank_zero():
        return

    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=output_db
    )
    embedding_db.create_db(
        embedding_batches=embedding_computer.get_embedding_batches(load_batch_size=load_batch_size),
        use_gpu_index=use_gpu_index,
        index_type=index_type,
        index_config=IndexConfig(nlist=ivf_nlist, nprobe=ivf_nprobe),
    )
    if cleanup_embeddings:
        embedding_computer.remove_shards()


@build_db_app.command(
    name="embeddings",
    help="Build an embedding database from a directory of pre-computed embedding files (.pt, .csv, or .parquet)."
)
def build_database_from_embeddings(
        embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing pre-computed embedding files (.pt, .csv, or .parquet).'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        file_extension: Annotated[Optional[str], typer.Option(
            help='Restrict loading to a single extension (.pt, .csv, or .parquet). If unset, all three are collected.'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        index_type: Annotated[IndexType, typer.Option(
            help='FAISS index variant: auto (legacy heuristic), flat, hnsw, or ivf_pq (on-disk).'
        )] = IndexType.auto,
        ivf_nlist: Annotated[int, typer.Option(
            help='IVF cell count for ivf_pq index.'
        )] = 16384,
        ivf_nprobe: Annotated[Optional[int], typer.Option(
            help='Cells probed per query for ivf_pq (defaults to nlist // 64).'
        )] = None,
        load_batch_size: Annotated[int, typer.Option(
            help='Number of vectors per batch streamed into the FAISS index. Larger '
                 'batches (e.g. 65536+) better saturate many-core nodes during '
                 'IVF-PQ build; smaller batches use less memory.'
        )] = 32768,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Build an embedding database from pre-computed embedding files."""

    set_log_level(log_level)

    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=output_db
    )
    from foldmatch.utils.data import stream_embeddings
    embedding_db.create_db(
        embedding_batches=stream_embeddings(
            path=embedding_folder,
            file_extension=file_extension,
            batch_size=load_batch_size
        ),
        use_gpu_index=use_gpu_index,
        index_type=index_type,
        index_config=IndexConfig(nlist=ivf_nlist, nprobe=ivf_nprobe),
    )


@build_db_app.command(
    name="sequences",
    help="Build an embedding database from protein sequences in a FASTA file."
)
def build_database_from_fasta(
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences.'
        )],
        output_db: Annotated[str, typer.Option(
            help='Path to save the FAISS database.'
        )],
        tmp_embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Directory for intermediate embeddings.'
        )],
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS index (requires faiss-gpu).'
        )] = False,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto".'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together for residue embedding inference.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        index_type: Annotated[IndexType, typer.Option(
            help='FAISS index variant: auto (legacy heuristic), flat, hnsw, or ivf_pq (on-disk).'
        )] = IndexType.auto,
        ivf_nlist: Annotated[int, typer.Option(
            help='IVF cell count for ivf_pq index.'
        )] = 16384,
        ivf_nprobe: Annotated[Optional[int], typer.Option(
            help='Cells probed per query for ivf_pq (defaults to nlist // 64).'
        )] = None,
        load_batch_size: Annotated[int, typer.Option(
            help='Number of vectors per batch streamed into the FAISS index. Larger '
                 'batches (e.g. 65536+) better saturate many-core nodes during '
                 'IVF-PQ build; smaller batches use less memory.'
        )] = 32768,
        cleanup_embeddings: Annotated[bool, typer.Option(
            help=_CLEANUP_EMBEDDINGS_HELP
        )] = False,
        residue_codes: Annotated[bool, typer.Option(
            "--residue-codes/--no-residue-codes",
            help=_RESIDUE_CODES_HELP
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Build an embedding database from protein sequences in a FASTA file."""

    set_log_level(log_level)
    if residue_codes:
        preflight, embedding_computer, prepared = _residue_codes_preflight(
            fasta_file, min_res_n, output_db, tmp_embedding_folder, num_nodes
        )
        # The pass also writes the residue codes and seq_sha1 into each shard.
        from foldmatch.search.residue_shards import RESIDUE_CODES
        residue = {"residue_output": RESIDUE_CODES, "scoring_system": prepared}
    else:
        preflight = _fasta_preflight(fasta_file, min_res_n, output_db=output_db)

        from foldmatch.search.embedding_computer import EmbeddingComputer
        embedding_computer = EmbeddingComputer(
            embedding_folder=tmp_embedding_folder,
        )
        prepared = None
        # A plain build calls the pass exactly as before.
        residue = {}
    # A --residue-codes build reports every refusal as a RefusalError (D20:
    # exit 1); a plain build raises exactly what it raised before.
    refusals = _refusals if residue_codes else nullcontext
    with refusals():
        embedding_computer.compute_from_fasta(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            accelerator=accelerator,
            devices=arg_devices(devices),
            strategy=strategy,
            preflight=preflight,
            **residue,
        )

    if not _is_rank_zero():
        return

    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=output_db
    )
    with refusals():
        embedding_db.create_db(
            embedding_batches=embedding_computer.get_embedding_batches(load_batch_size=load_batch_size),
            use_gpu_index=use_gpu_index,
            index_type=index_type,
            index_config=IndexConfig(nlist=ivf_nlist, nprobe=ivf_nprobe),
        )

    # Persist a sidecar id->sequence store so this database can be the
    # subject (or query) of a Stage-2 pairwise-alignment search. Built
    # directly from the FASTA — the DB ids ARE the FASTA headers — using the
    # same min_res_n filter so store and index stay in lock-step. Rank 0 only,
    # after the pass, so ranks never race on the SQLite file.
    from foldmatch.search.sequence_store import SequenceStore
    store = SequenceStore(embedding_db.db_folder, embedding_db.index_name)
    with refusals():
        store.create(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
        )
        _check_store_matches_preflight(store, preflight, output_db)
    if prepared is not None:
        # Last, once the database it describes is complete; its manifest is
        # written last of all. A refusal leaves the database (BLOSUM62) and the
        # shards (for --from-shards) in place.
        _build_residue_code_store(
            embedding_computer, embedding_db, store, preflight, prepared.system,
            min_res_n=min_res_n, load_batch_size=load_batch_size,
        )
    if cleanup_embeddings:
        embedding_computer.remove_shards()


@build_db_app.command(
    name="residue-codes",
    help="Build the residue-code store of a sequence database from the embedding "
         "shards its 'build sequences --residue-codes' run kept, without running "
         "ESM3 again, and validate it against the database. DB_FOLDER and "
         "INDEX_NAME name the database (its --output-db was DB_FOLDER/INDEX_NAME)."
)
def build_residue_codes(
        db_folder: Annotated[Path, typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder of the database.'
        )],
        index_name: Annotated[str, typer.Argument(
            help='Name of the database in DB_FOLDER: its files are '
                 'INDEX_NAME.index, INDEX_NAME.sequences, ...'
        )],
        from_shards: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_FROM_SHARDS_HELP
        )],
        run_id: Annotated[Optional[str], typer.Option(
            help='Run id of the build whose shards to use '
                 '(emb_<run id>-<rank>.parquet). Needed only when the folder '
                 'holds the residue-code shards of more than one run.'
        )] = None,
        force: Annotated[bool, typer.Option(
            "--force/--no-force",
            help='Replace the residue-code store if the database already has '
                 'one. The new codes are written aside and replace the old '
                 'store only once they have passed validation, so a refused '
                 'rebuild leaves it as it was (the rebuild needs room for both '
                 'meanwhile).'
        )] = False,
        load_batch_size: Annotated[int, typer.Option(
            min=1,
            help='Number of entries read from the shards per batch.'
        )] = 32768,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Build a sequence database's residue-code store from kept build shards (no ESM3)."""

    set_log_level(log_level)
    # Flag refusals (D20: exit 2), before anything is read.
    if not index_name or index_name in (".", "..") or Path(index_name).name != index_name:
        raise typer.BadParameter(
            f"{index_name!r} is not a database name: INDEX_NAME is the name of the "
            f"database's files in DB_FOLDER, without a folder.", param_hint="INDEX_NAME"
        )
    from foldmatch.utils.dist import is_run_id
    if run_id is not None and not is_run_id(run_id):
        raise typer.BadParameter(
            f"{run_id!r} is not a run id: letters, digits, '.', '_' and '-' only, "
            f"starting with a letter or digit.", param_hint="--run-id"
        )
    with _refusals():
        manifest = _residue_codes_from_shards(
            db_folder, index_name, from_shards, run_id=run_id, force=force,
            load_batch_size=load_batch_size,
        )
    logging.info(
        f"Residue-code store of {db_folder / index_name} built from the shards of run "
        f"{manifest['build']['run_id']}: {manifest['entries']} entries, "
        f"{manifest['total_residues']} residues."
    )


@query_db_app.command(
    name="structure",
    help="Search the database using a structure file."
)
def query_database_from_structure(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        query_structure: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='Path to query structure file.'
        )],
        output_file: Annotated[str, typer.Option(
            help=_OUTPUT_FILE_HELP
        )],
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format (mmcif, binarycif or pdb)'
        )] = StructureFormat.mmcif,
        granularity: Annotated[Granularity, typer.Option(
            help='Query database for "chain" or "assembly" embeddings.'
        )] = Granularity.chain,
        chain_id: Annotated[Optional[str], typer.Option(
            help='When "granularity=chain", specific chain to search (if not specified, searches all chains).'
        )] = None,
        assembly_id: Annotated[Optional[str], typer.Option(
            help='When "granularity=assembly", specific assembly to search (if not specified, searches by asymmetric unit).'
        )] = None,
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per chain.'
        )] = 100,
        threshold: Annotated[Optional[float], typer.Option(
            help='Similarity score threshold to filter results (only return matches with score >= threshold).'
        )] = 0.8,
        min_res: Annotated[int, typer.Option(
            help='Minimum residue length for chains.'
        )] = 10,
        max_res: Annotated[Optional[int], typer.Option(
            help='Maximum residue length for structures (None for no limit).'
        )] = None,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu).'
        )] = False,
        nprobe: Annotated[Optional[int], typer.Option(
            help='Override the search-time nprobe (cells probed per query) for '
                 'ivf_pq indexes. Higher = better recall, slower search. '
                 'Defaults to the value baked in at build time. '
                 'Ignored for flat/hnsw indexes.'
        )] = None,
        format_mode: Annotated[FormatMode, typer.Option(
            help=_FORMAT_MODE_HELP
        )] = FormatMode.headless_tsv,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    """Search database for similar structures."""

    set_log_level(log_level)

    from foldmatch.search.embedding_computer import compute_from_structure_file
    embedding_batches = compute_from_structure_file(
        query_structure=query_structure,
        structure_format=structure_format,
        granularity=granularity,
        chain_id=chain_id,
        assembly_id=assembly_id,
        min_res=min_res,
        max_res=max_res,
        accelerator=accelerator
    )

    logging.info("Loading database...")
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=db_path,
        use_gpu_for_search=use_gpu_index
    )
    embedding_db.load(nprobe=nprobe)
    results = embedding_db.search_by_embeddings(
        embedding_batches=embedding_batches,
        top_k=top_k
    )

    # Filter by threshold if specified
    results = _filter_results_by_threshold(results, threshold)

    write_embedding_results(results, output_file, format_mode)


@query_db_app.command(
    name="embedding",
    help="Search the database using a pre-computed embedding file (.pt, .csv, or .parquet)."
)
def query_database_from_embedding(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Directory containing pre-computed embedding files (.pt, .csv, or .parquet).'
        )],
        output_file: Annotated[str, typer.Option(
            help=_OUTPUT_FILE_HELP
        )],
        file_extension: Annotated[Optional[str], typer.Option(
            help='Restrict loading to a single extension (.pt, .csv, or .parquet). If unset, all three are collected.'
        )] = None,
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return.'
        )] = 100,
        threshold: Annotated[Optional[float], typer.Option(
            help='Similarity score threshold to filter results (only return matches with score >= threshold).'
        )] = 0.8,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu).'
        )] = False,
        nprobe: Annotated[Optional[int], typer.Option(
            help='Override the search-time nprobe (cells probed per query) for '
                 'ivf_pq indexes. Higher = better recall, slower search. '
                 'Defaults to the value baked in at build time. '
                 'Ignored for flat/hnsw indexes.'
        )] = None,
        format_mode: Annotated[FormatMode, typer.Option(
            help=_FORMAT_MODE_HELP
        )] = FormatMode.headless_tsv,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Search database using a pre-computed embedding file."""

    set_log_level(log_level)

    logging.info("Loading database...")
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=db_path,
        use_gpu_for_search=use_gpu_index
    )
    embedding_db.load(nprobe=nprobe)
    from foldmatch.utils.data import stream_embeddings
    results = embedding_db.search_by_embeddings(
        embedding_batches=stream_embeddings(
            path=embedding_folder,
            file_extension=file_extension
        ),
        top_k=top_k
    )

    if not results:
        raise ValueError(f"No embeddings found in {embedding_folder}")

    logging.info(f"Found {len(results)} results from {embedding_folder}")
    results = _filter_results_by_threshold(results, threshold)

    write_embedding_results(results, output_file, format_mode)


@query_db_app.command(
    name="sequences",
    help="Search the database using protein sequences from a FASTA file. Each sequence is used as a separate query."
)
def query_database_from_fasta(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences. Each sequence is used as a separate query.'
        )],
        tmp_embedding_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Directory for intermediate embeddings.'
        )],
        output_file: Annotated[str, typer.Option(
            help=_OUTPUT_FILE_HELP
        )],
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per query.'
        )] = 100,
        threshold: Annotated[Optional[float], typer.Option(
            help='Similarity score threshold to filter results (only return matches with score >= threshold).'
        )] = 0.8,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu).'
        )] = False,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto".'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together for residue embedding inference.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        nprobe: Annotated[Optional[int], typer.Option(
            help='Override the search-time nprobe (cells probed per query) for '
                 'ivf_pq indexes. Higher = better recall, slower search. '
                 'Defaults to the value baked in at build time. '
                 'Ignored for flat/hnsw indexes.'
        )] = None,
        scoring: Annotated[ScoringChoice, typer.Option(
            help=_SCORING_HELP
        )] = DEFAULT_SCORING,
        seq_identity: Annotated[Optional[bool], typer.Option(
            help='Stage 2: after the embedding prefilter, pairwise-align each '
                 'query against its candidate subjects and report exact sequence '
                 'identity. Default: AUTO — on when the subject database has a '
                 'sequence store (built from FASTA), else embedding-only. Use '
                 '--no-seq-identity to force off, --seq-identity to force on '
                 '(errors if the store is missing). --scoring esm-codes always '
                 'runs Stage 2.'
        )] = None,
        min_seq_identity: Annotated[Optional[float], typer.Option(
            help=_QUERY_MIN_SEQ_IDENTITY_HELP
        )] = None,
        min_coverage: Annotated[Optional[float], typer.Option(
            help=_QUERY_MIN_COVERAGE_HELP
        )] = None,
        max_evalue: Annotated[Optional[float], typer.Option(
            help=_QUERY_MAX_EVALUE_HELP
        )] = None,
        gap_open: Annotated[Optional[int], typer.Option(
            help='Gap-open penalty (positive) for Stage-2 alignment. Default: 11.' + _BLOSUM62_ONLY
        )] = None,
        gap_extend: Annotated[Optional[int], typer.Option(
            help='Gap-extend penalty (positive) for Stage-2 alignment. Default: 1.' + _BLOSUM62_ONLY
        )] = None,
        comp_bias_corr: Annotated[Optional[bool], typer.Option(
            help='Correct Stage-2 alignment scores for locally biased amino-acid '
                 'composition. Changes the alignment itself (coordinates, '
                 'identity, coverage) as well as the raw score, bits and E-value. '
                 'Default: off.' + _BLOSUM62_ONLY
        )] = None,
        comp_bias_corr_scale: Annotated[float, typer.Option(
            min=0.0, max=1.0,
            help='Scale applied to the compositional bias correction when '
                 '--comp-bias-corr is set (0-1).'
        )] = 1.0,
        significance_mode: Annotated[Optional[SignificanceMode], typer.Option(
            help="Scale for the BitScore/Pvalue/Evalue columns. 'default' uses "
                 "calibrated statistics, but requires the default gap penalties "
                 "11/1; 'sampled' works at any gap penalties but its absolute "
                 "values are unreliable and comparable only within this tool. "
                 "Default: 'default'." + _BLOSUM62_ONLY
        )] = None,
        significance_sample_size: Annotated[int, typer.Option(
            help="For --significance-mode sampled: number of random alignments "
                 "sampled to estimate lambda/K for the BitScore/Pvalue/Evalue "
                 "columns. Larger is slower; ignored when mode is 'default'."
        )] = 500,
        align_workers: Annotated[Optional[int], typer.Option(
            help='Process-pool size for Stage-2 alignment. Defaults to all CPUs '
                 'on the node (widening any scheduler --cpu-bind pinning); set '
                 'to 0 or 1 to run serially.'
        )] = None,
        format_output: Annotated[str, typer.Option(
            help=_FORMAT_OUTPUT_HELP
        )] = DEFAULT_FORMAT_OUTPUT,
        format_mode: Annotated[FormatMode, typer.Option(
            help=_FORMAT_MODE_HELP
        )] = FormatMode.headless_tsv,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Search database using protein sequences from a FASTA file."""

    set_log_level(log_level)
    _check_comp_bias_corr_scale(comp_bias_corr_scale)
    # The per-scoring defaults; under esm-codes the BLOSUM62-only options are refused (exit 2), before anything is read.
    stage2 = resolve_query_stage2_options(
        scoring, seq_identity=seq_identity, min_seq_identity=min_seq_identity, min_coverage=min_coverage,
        max_evalue=max_evalue, gap_open=gap_open, gap_extend=gap_extend, comp_bias_corr=comp_bias_corr,
        significance_mode=significance_mode,
    )
    if stage2.scoring is ScoringChoice.esm_codes:
        _query_sequences_esm_codes(
            db_path=db_path, fasta_file=fasta_file, tmp_embedding_folder=tmp_embedding_folder,
            output_file=output_file, min_res_n=min_res_n, top_k=top_k, threshold=threshold,
            use_gpu_index=use_gpu_index, accelerator=accelerator, num_nodes=num_nodes, devices=devices,
            strategy=strategy, batch_size=batch_size, num_workers=num_workers, nprobe=nprobe, stage2=stage2,
            align_workers=align_workers, format_output=format_output, format_mode=format_mode,
        )
        return
    min_seq_identity, min_coverage, max_evalue = stage2.min_seq_identity, stage2.min_coverage, stage2.max_evalue
    gap_open, gap_extend, comp_bias_corr = stage2.gap_open, stage2.gap_extend, stage2.comp_bias_corr
    significance_mode = stage2.significance_mode
    preflight = _fasta_preflight(fasta_file, min_res_n)
    # Read-only, on every rank, before ESM3: whether Stage 2 runs, and whether
    # the subject's sequence store can serve it.
    from foldmatch.search.sequence_store import SequenceStore
    subject_store = SequenceStore(*parse_db_path(db_path)[:2])
    run_stage2 = _resolve_stage2(
        seq_identity,
        subject_store.exists(),
        f"Subject database has no sequence store ({subject_store.path}); "
        f"--seq-identity requires a database built from FASTA (fm-search build sequences).",
    )
    if run_stage2:
        _check_stage2_stores(subject_store, max_evalue)

    from foldmatch.search.embedding_computer import EmbeddingComputer
    embedding_computer = EmbeddingComputer(
        embedding_folder=tmp_embedding_folder,
    )
    embedding_computer.compute_from_fasta(
        fasta_file=fasta_file,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        strategy=strategy,
        preflight=preflight,
    )

    if not _is_rank_zero():
        return

    logging.info("Loading database...")
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=db_path,
        use_gpu_for_search=use_gpu_index,
    )
    embedding_db.load(nprobe=nprobe)
    results = embedding_db.search_by_embeddings(
        embedding_batches=embedding_computer.get_embedding_batches(),
        top_k=top_k
    )
    # The query embeddings are only needed for this search.
    embedding_computer.remove_shards()
    logging.info(f"Searched {len(results)} sequence(s)")
    results = _filter_results_by_threshold(results, threshold)

    if run_stage2:
        from foldmatch.utils.fasta import iter_fasta
        from foldmatch.utils.sequence import normalize_sequence
        # Align the sequences ESM3 embedded, normalised like the stored subjects.
        query_sequences = {
            name: normalize_sequence(sequence, allow_non_ascii=True)
            for name, sequence in iter_fasta(fasta_file)
        }
        _stage2_align_and_report(
            embedding_db=embedding_db,
            prefilter_results=results,
            query_sequences=query_sequences,
            min_seq_identity=min_seq_identity,
            min_coverage=min_coverage,
            max_evalue=max_evalue,
            gap_open=gap_open,
            gap_extend=gap_extend,
            comp_bias_corr=comp_bias_corr,
            comp_bias_corr_scale=comp_bias_corr_scale,
            num_workers=align_workers,
            output_file=output_file,
            significance_mode=significance_mode,
            significance_sample_size=significance_sample_size,
            format_output=format_output,
            format_mode=format_mode,
        )
    else:
        write_embedding_results(results, output_file, format_mode)


@query_db_app.command(
    name="db",
    help="Compare entries from a query database with a subject database."
)
def query_database_from_database(
        query_db_path: Annotated[str, typer.Option(
            help='Path to the query FAISS database.'
        )],
        subject_db_path: Annotated[str, typer.Option(
            help='Path to the subject FAISS database to search against.'
        )],
        output_file: Annotated[str, typer.Option(
            help=_OUTPUT_FILE_HELP
        )],
        top_k: Annotated[int, typer.Option(
            help='Number of top results to return per query chain.'
        )] = 100,
        threshold: Annotated[Optional[float], typer.Option(
            help='Similarity score threshold to filter results (only return matches with score >= threshold).'
        )] = 0.8,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS search (requires faiss-gpu).'
        )] = False,
        nprobe: Annotated[Optional[int], typer.Option(
            help='Override the search-time nprobe (cells probed per query) for '
                 'ivf_pq indexes. Higher = better recall, slower search. '
                 'Defaults to the value baked in at build time. '
                 'Ignored for flat/hnsw indexes.'
        )] = None,
        seq_identity: Annotated[Optional[bool], typer.Option(
            help='Stage 2: after the embedding prefilter, pairwise-align each '
                 'query against its candidate subjects and report exact sequence '
                 'identity. Default: AUTO — on when BOTH databases have a sequence '
                 'store (built from FASTA), else embedding-only. Use '
                 '--no-seq-identity to force off, --seq-identity to force on '
                 '(errors if a store is missing).'
        )] = None,
        min_seq_identity: Annotated[float, typer.Option(
            help='When --seq-identity is set, drop hits whose alignment identity '
                 'is below this fraction (0-1).'
        )] = 0.3,
        min_coverage: Annotated[float, typer.Option(
            help=_MIN_COVERAGE_HELP
        )] = 0.8,
        max_evalue: Annotated[float, typer.Option(
            help='When --seq-identity is set, drop hits whose database E-value is '
                 'above this. Forces the significance pass (and, in '
                 "--significance-mode default, its 11/1 gap penalties); pass "
                 "'inf' to disable E-value filtering."
        )] = 1e-3,
        gap_open: Annotated[int, typer.Option(
            help='Gap-open penalty (positive) for Stage-2 alignment.'
        )] = 11,
        gap_extend: Annotated[int, typer.Option(
            help='Gap-extend penalty (positive) for Stage-2 alignment.'
        )] = 1,
        comp_bias_corr: Annotated[bool, typer.Option(
            help='Correct Stage-2 alignment scores for locally biased amino-acid '
                 'composition. Changes the alignment itself (coordinates, '
                 'identity, coverage) as well as the raw score, bits and E-value.'
        )] = False,
        comp_bias_corr_scale: Annotated[float, typer.Option(
            min=0.0, max=1.0,
            help='Scale applied to the compositional bias correction when '
                 '--comp-bias-corr is set (0-1).'
        )] = 1.0,
        significance_mode: Annotated[SignificanceMode, typer.Option(
            help="Scale for the BitScore/Pvalue/Evalue columns. 'default' uses "
                 "calibrated statistics, but requires the default gap penalties "
                 "11/1; 'sampled' works at any gap penalties but its absolute "
                 "values are unreliable and comparable only within this tool."
        )] = SignificanceMode.default,
        significance_sample_size: Annotated[int, typer.Option(
            help="For --significance-mode sampled: number of random alignments "
                 "sampled to estimate lambda/K for the BitScore/Pvalue/Evalue "
                 "columns. Larger is slower; ignored when mode is 'default'."
        )] = 500,
        align_workers: Annotated[Optional[int], typer.Option(
            help='Process-pool size for Stage-2 alignment. Defaults to all CPUs '
                 'on the node (widening any scheduler --cpu-bind pinning); set '
                 'to 0 or 1 to run serially.'
        )] = None,
        format_output: Annotated[str, typer.Option(
            help=_FORMAT_OUTPUT_HELP
        )] = DEFAULT_FORMAT_OUTPUT,
        format_mode: Annotated[FormatMode, typer.Option(
            help=_FORMAT_MODE_HELP
        )] = FormatMode.headless_tsv,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    """Search subject database using all entries from query database."""

    set_log_level(log_level)
    _check_comp_bias_corr_scale(comp_bias_corr_scale)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS search: enabled")

    # Stage 2 needs sequences for BOTH sides: queries from the query DB's store,
    # subjects from the subject DB's store. Decided, and the stores checked,
    # before the search.
    from foldmatch.search.sequence_store import SequenceStore
    query_store = SequenceStore(*parse_db_path(query_db_path)[:2])
    subject_store = SequenceStore(*parse_db_path(subject_db_path)[:2])
    missing = []
    if not query_store.exists():
        missing.append(f"query database ({query_store.path})")
    if not subject_store.exists():
        missing.append(f"subject database ({subject_store.path})")
    run_stage2 = _resolve_stage2(
        seq_identity,
        not missing,
        f"Stage-2 sequence identity requires a sequence store for: {', '.join(missing)}. "
        f"Build databases with 'fm-search build sequences'.",
    )
    if run_stage2:
        _check_stage2_stores(subject_store, max_evalue, query_store=query_store)

    # Load subject database
    logging.info("\nLoading subject database...")
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=subject_db_path,
        use_gpu_for_search=use_gpu_index
    )
    embedding_db.load(nprobe=nprobe)

    # Display database statistics
    subject_stats = embedding_db.get_db_statistics()
    logging.info(f"Subject database contains {subject_stats['total_embeddings']} chains")

    # Perform database-to-database search
    logging.info("Performing search...")
    results = embedding_db.search_by_database(
        query_db_path=query_db_path,
        top_k=top_k
    )

    # Filter by threshold if specified
    results = _filter_results_by_threshold(results, threshold)

    if run_stage2:
        query_sequences = query_store.fetch(results.keys())
        _stage2_align_and_report(
            embedding_db=embedding_db,
            prefilter_results=results,
            query_sequences=query_sequences,
            min_seq_identity=min_seq_identity,
            min_coverage=min_coverage,
            max_evalue=max_evalue,
            gap_open=gap_open,
            gap_extend=gap_extend,
            comp_bias_corr=comp_bias_corr,
            comp_bias_corr_scale=comp_bias_corr_scale,
            num_workers=align_workers,
            output_file=output_file,
            significance_mode=significance_mode,
            significance_sample_size=significance_sample_size,
            format_output=format_output,
            format_mode=format_mode,
        )
    else:
        write_embedding_results(results, output_file, format_mode)


@app.command(
    name="stats",
    help="Display database statistics."
)
def show_statistics(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database (directory + prefix, e.g., ./db/my_db).'
        )],
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    """Display database statistics."""

    set_log_level(log_level)

    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=db_path
    )
    embedding_db.load()
    stats = embedding_db.get_db_statistics()

    logging.info("DATABASE STATISTICS")
    logging.info("="*80)
    logging.info(f"Database path:    {stats['db_path']}")
    logging.info(f"Index name:       {stats['index_name']}")
    logging.info(f"Index type:       {stats['index_type']}")
    logging.info(f"Dimension:        {stats['dimension']}")
    logging.info(f"Total embeddings: {stats['total_embeddings']}")
    logging.info(f"On GPU:           {stats['on_gpu']}")
    logging.info(f"GPU available:    {stats['gpu_available']}")
    logging.info("="*80 + "\n")


@app.command(
    name="cluster",
    help="Cluster database embeddings using Leiden algorithm on similarity graph."
)
def cluster_database(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        threshold: Annotated[float, typer.Option(
            help='Similarity threshold for edge creation (0-1, where 1.0 = identical).'
        )] = 0.8,
        resolution: Annotated[float, typer.Option(
            help='Leiden resolution parameter (higher = more clusters).'
        )] = 1.0,
        output_file: Annotated[str, typer.Option(
            help='Path to save cluster assignments, as tab-separated rows with '
                 f'no header: {", ".join(CLUSTER_OUTPUT_FIELDS)}.'
        )] = "clusters.tsv",
        max_neighbors: Annotated[int, typer.Option(
            help='Maximum number of neighbors to consider per chain'
        )] = 1000,
        min_cluster_size: Annotated[Optional[int], typer.Option(
            help='Minimum cluster size to include in output (filters smaller clusters).'
        )] = None,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS operations (requires faiss-gpu).'
        )] = False,
        seed: Annotated[Optional[int], typer.Option(
            help='Random seed for reproducibility.'
        )] = None,
        format_mode: Annotated[FormatMode, typer.Option(
            help=_FORMAT_MODE_HELP
        )] = FormatMode.headless_tsv,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    """Cluster database embeddings using Leiden algorithm."""

    set_log_level(log_level)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS operations: enabled")

    # Initialize clusterer
    logging.info("Initializing clusterer...")
    from foldmatch.search.clustering import EmbeddingClusterer
    clusterer = EmbeddingClusterer(
        db_path=db_path,
        use_gpu=use_gpu_index
    )

    # Build similarity graph
    clusterer.build_similarity_graph(
        threshold=threshold,
        max_neighbors=max_neighbors
    )

    # Perform clustering
    clusterer.cluster_leiden(
        resolution=resolution,
        seed=seed
    )

    # Display statistics
    clusterer.print_statistics()

    # Export results
    clusterer.export_clusters(
        output_file=output_file,
        min_cluster_size=min_cluster_size,
        format_mode=format_mode,
    )


@app.command(
    name="similarity-graph",
    help="Build a similarity graph from database embeddings and export it in GraphML format."
)
def similarity_graph(
        db_path: Annotated[str, typer.Option(
            help='Path to the FAISS database.'
        )],
        threshold: Annotated[float, typer.Option(
            help='Similarity threshold for edge creation (0-1, where 1.0 = identical).'
        )] = 0.8,
        output_file: Annotated[str, typer.Option(
            help='Path to save the similarity graph (GraphML format).'
        )] = "similarity_graph.graphml",
        max_neighbors: Annotated[int, typer.Option(
            help='Maximum number of neighbors to consider per chain.'
        )] = 1000,
        use_gpu_index: Annotated[bool, typer.Option(
            help='Use GPU for FAISS operations (requires faiss-gpu).'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging verbosity level.'
        )] = LogLevel.info
):
    """Build and export a similarity graph from database embeddings."""

    set_log_level(log_level)

    if use_gpu_index:
        logging.info("GPU acceleration for FAISS operations: enabled")

    logging.info("Initializing clusterer...")
    from foldmatch.search.clustering import EmbeddingClusterer
    clusterer = EmbeddingClusterer(
        db_path=db_path,
        use_gpu=use_gpu_index
    )

    graph = clusterer.build_similarity_graph(
        threshold=threshold,
        max_neighbors=max_neighbors
    )

    output_path = Path(output_file)
    graph.write_graphml(str(output_path))
    logging.info(f"Similarity graph saved to {output_path}")


@app.command(
    name="download-models",
    help=DOWNLOAD_MODELS_HELP
)
def download_models(
        system: Annotated[Optional[List[str]], typer.Option(
            "--system",
            help=DOWNLOAD_MODELS_SYSTEM_HELP
        )] = None
):
    """The same command as ``fm-embedding download-models`` and ``inference download-models``."""
    download_models_command(system)


def version_callback(value: bool):
    if value:
        typer.echo(f"{__version__}")
        raise typer.Exit()


@app.callback()
def _app_callback(
        version: bool = typer.Option(
            None,
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit.",
        )
):
    pass

def _filter_results_by_threshold(results, threshold: float | None):
    """Filter search results by similarity threshold."""
    if threshold is None:
        return results

    logging.debug(f"Filtering results with similarity score threshold >= {threshold}")
    filtered_results = {}
    total_before = sum(len(ids) for ids, _ in results.values())
    for query_chain, (chain_ids, scores) in results.items():
        filtered_pairs = [(cid, score) for cid, score in zip(chain_ids, scores) if score >= threshold]
        if filtered_pairs:
            filtered_chain_ids, filtered_scores = zip(*filtered_pairs)
            filtered_results[query_chain] = (list(filtered_chain_ids), list(filtered_scores))
        else:
            filtered_results[query_chain] = ([], [])
    total_after = sum(len(ids) for ids, _ in filtered_results.values())
    logging.debug(f"Filtered from {total_before} to {total_after} results")
    return filtered_results

def _check_comp_bias_corr_scale(scale: float):
    """Reject a scale Click's range check lets through (NaN) before Stage 1 runs."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"--comp-bias-corr-scale must be between 0 and 1, got {scale}.")


# Files a database build writes next to ``--output-db``. Any of them already
# present refuses the build before ESM3 (create_db and SequenceStore.create
# would only refuse it after the whole pass).
_INDEX_SUFFIXES = (".index", ".metadata", ".invlists")
_SEQUENCE_DB_SUFFIXES = _INDEX_SUFFIXES + (".sequences",)


def _refuse_existing_outputs(output_db: str, suffixes, sidecar_globs=()) -> None:
    """Raise ``FileExistsError`` if any output of this build already exists. Read-only."""
    # parse_db_path does not import FAISS: the build still loads it only after
    # the ESM3 pass, as before.
    db_folder, index_name, db_path = parse_db_path(output_db)
    existing = [Path(f"{db_path}{suffix}") for suffix in suffixes]
    existing = [path for path in existing if path.exists()]
    for pattern in sidecar_globs:
        existing.extend(sorted(db_folder.glob(f"{glob.escape(index_name)}{pattern}")))
    if existing:
        raise FileExistsError(
            f"Database already exists at {db_path} "
            f"(found: {', '.join(str(path) for path in existing)}). "
            f"Choose a different --output-db, or delete the existing files first."
        )


def _fasta_preflight(fasta_file: Path, min_res_n: int, strict: bool = False, output_db: Optional[str] = None):
    """Read-only checks every rank runs before ESM3 is loaded.

    Every rank reaches the same verdict from the same files, so a refusal
    stops all ranks together instead of one rank failing after the pass.
    For a build (``output_db``), existing outputs are refused. The FASTA scan
    (:func:`preflight_fasta`) counts the entries the pass must produce;
    ``strict`` refuses duplicate ids, '<'/'>' and non-ASCII sequences, which
    otherwise are logged as warnings and processed as before.
    """
    if output_db is not None:
        _refuse_existing_outputs(output_db, _SEQUENCE_DB_SUFFIXES, sidecar_globs=(".rescodes.*",))
    from foldmatch.utils.sequence import preflight_fasta
    preflight = preflight_fasta(fasta_file, min_res_n)
    preflight.check(strict=strict)
    if output_db is not None and preflight.entries_kept == 0:
        raise ValueError(
            f"{fasta_file} has no sequence with at least {min_res_n} residues "
            f"({preflight.entries} entries in all): there is nothing to build."
        )
    return preflight


def _check_store_matches_preflight(store, preflight, output_db: str) -> None:
    """The sequence store must hold exactly what the pre-flight counted.

    A mismatch means the FASTA changed during the build. The database is
    removed rather than left disagreeing with itself: the store and the index
    files this build wrote (the pre-flight refused any that existed before), so
    the same build can be run again.
    """
    meta = store.meta()
    expected = {
        "entry_count": preflight.unique_entries,
        "total_residues": preflight.unique_residues,
        "duplicates_ignored": preflight.duplicates,
    }
    if meta != expected:
        removed = _remove_outputs(output_db, _SEQUENCE_DB_SUFFIXES)
        raise ValueError(
            f"Sequence store built from {preflight.path} ({meta}) does not match its "
            f"pre-flight ({expected}); was the FASTA modified during the build? "
            f"The database files it wrote were removed ({', '.join(path.name for path in removed)})."
        )


def _remove_outputs(output_db: str, suffixes) -> List[Path]:
    """Delete this build's ``{output_db}{suffix}`` files; returns the ones removed."""
    _, _, db_path = parse_db_path(output_db)
    removed = []
    for suffix in suffixes:
        path = Path(f"{db_path}{suffix}")
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


@contextmanager
def _refusals(note: Optional[str] = None):
    """Report a refused check of the block as a :class:`RefusalError` (D20: exit status 1).

    The check's own exception (a ``ValueError``, an ``OSError`` such as a
    missing or existing file or a full disk, or an SQLite file that is not a
    database) becomes its ``__cause__``; ``note`` is appended to the message.
    """
    try:
        yield
    except RefusalError as exc:
        if note is None:
            raise
        raise RefusalError(f"{exc} {note}") from exc
    except (ValueError, OSError, sqlite3.DatabaseError) as exc:
        raise RefusalError(str(exc) if note is None else f"{exc} {note}") from exc


def _residue_codes_preflight(fasta_file: Path, min_res_n: int, output_db: str,
                             tmp_embedding_folder: Path, num_nodes: int):
    """Every rank's read-only, deterministic checks of a --residue-codes build, before ESM3.

    In the order of plan section 3 (BUILD): the strict FASTA pre-flight
    (duplicate ids, '<' / '>' and non-ASCII sequences refused) and the
    outputs-exist check, ``.rescodes.*`` included; this run's id and its
    shards; then the default scoring system's bundle, which only the process
    that starts the run may download (every other rank finds it in the cache:
    :func:`~foldmatch.inference.chain_complete_inference.prepare_scoring_system`).
    The same files give every rank the same verdict, so a refusal stops them
    all together. Returns ``(preflight, embedding_computer, prepared_system)``.
    """
    with _refusals():
        preflight = _fasta_preflight(fasta_file, min_res_n, strict=True, output_db=output_db)
        from foldmatch.search.embedding_computer import EmbeddingComputer
        embedding_computer = EmbeddingComputer(embedding_folder=tmp_embedding_folder)
        embedding_computer.refuse_existing_shards()
        from foldmatch.inference.chain_complete_inference import prepare_scoring_system
        from foldmatch.search.residue_shards import RESIDUE_CODES
        prepared = prepare_scoring_system(residue_output=RESIDUE_CODES, num_nodes=num_nodes)
    return preflight, embedding_computer, prepared


def _code_store_build(run_id: str, shard_infos, *, from_shards: bool):
    """``(pass record, manifest build block)`` of a code store built from a run's shards.

    The build block describes the ESM3 pass that computed the codes, from the
    shards' pass record (:func:`~foldmatch.search.residue_shards.pass_record`):
    the device type and its ESM3 precision, the ranks (``devices``), the
    encoder precision and the software versions of the pass. ``from_shards``
    tells whether the store was built by the pass's own command or later by
    ``build residue-codes --from-shards``.
    """
    from foldmatch.search.residue_shards import shards_pass_record
    record = shards_pass_record(shard_infos)
    build = {
        "run_id": run_id,
        "from_shards": from_shards,
        "accelerator": record["accelerator"],
        "devices": record["world_size"],
        "esm_precision": record["esm_precision"],
        "encoder_precision": record["encoder_precision"],
        "foldmatch_version": record["foldmatch_version"],
        "torch": record["torch"],
        "esm": record["esm"],
        "platform": record["platform"],
    }
    return record, build


def _build_residue_code_store(embedding_computer, embedding_db, sequence_store, preflight, system, *,
                              min_res_n: int, load_batch_size: int):
    """Rank 0, after the FAISS index and the sequence store: the residue-code store from this run's shards.

    ``create`` streams the shards' codes into the pending ``.rescodes.bin``
    and ``.rescodes.idx``; ``validate`` checks them against the sequence
    store, the pre-flight and the FAISS count, then moves them into place and
    writes the manifest last. The manifest's ``build`` block comes from the
    shards' pass record. A refusal removes only the code store and keeps the
    shards.
    """
    from foldmatch.search.residue_code_store import ResidueCodeStore
    from foldmatch.search.residue_shards import RESIDUE_CODES, check_shards
    code_store = ResidueCodeStore(embedding_db.db_folder, embedding_db.index_name)
    kept = (f"This build's shards are kept in {embedding_computer.embedding_folder} "
            f"(run id {embedding_computer.run_id}).")
    with _refusals(note=kept):
        infos = check_shards(embedding_computer.shard_files(), RESIDUE_CODES, system)
        _, build = _code_store_build(embedding_computer.run_id, infos, from_shards=False)
        code_store.create(
            embedding_computer.iter_residue_codes(system, load_batch_size=load_batch_size),
            system, min_res_n=min_res_n, build=build,
        )
        manifest = code_store.validate(sequence_store, len(embedding_db.db.chain_ids), preflight=preflight)
    logging.info(
        f"Residue-code store written to {code_store.manifest_path} ({manifest['entries']} entries, "
        f"{manifest['total_residues']} residues, system {manifest['system_name']})"
    )
    return manifest


def _residue_codes_from_shards(db_folder: Path, index_name: str, shards_folder: Path, *, run_id: Optional[str],
                               force: bool, load_batch_size: int):
    """``build residue-codes --from-shards``: the residue-code store of a database from a run's kept shards.

    Refused before anything is written: a database without a sequence store or
    FAISS files, or whose files cannot be read; a complete code store unless
    ``force``; no shards, or the shards of several runs without ``run_id``;
    shards without residue codes, mixed shards or another encoding than the
    default system's (:func:`~foldmatch.search.residue_shards.check_shards`);
    shards that do not record their pass's ``--min-res-n``; an entry count
    that differs between the shards, the sequence store and the FAISS index.
    Then ``create`` and ``validate`` without a pre-flight; an old store (with
    ``force``) is replaced only once the new codes have passed validation. The
    manifest's ``min_res_n`` and ``build`` block come from the shards' pass
    record, so they describe the ESM3 pass that computed the codes. Returns
    the manifest.
    """
    import pyarrow.parquet as pq
    from foldmatch.search import scoring_system
    from foldmatch.search.faiss_database import FaissEmbeddingDatabase
    from foldmatch.search.residue_code_store import ResidueCodeStore
    from foldmatch.search.residue_shards import RESIDUE_CODES, check_shards, iter_residue_codes
    from foldmatch.search.sequence_store import SequenceStore

    database = db_folder / index_name
    sequence_store = SequenceStore(db_folder, index_name)
    if not sequence_store.exists():
        raise RefusalError(
            f"{database} has no sequence store ({sequence_store.path}): residue codes are "
            f"only built for sequence databases ('fm-search build sequences')."
        )
    try:
        entry_count = sequence_store.meta()["entry_count"]
    except sqlite3.DatabaseError as exc:
        raise RefusalError(f"{sequence_store.path} is not a readable sequence store ({exc}).") from exc
    faiss_entries = FaissEmbeddingDatabase(db_folder, index_name).saved_entry_count()
    code_store = ResidueCodeStore(db_folder, index_name)
    if code_store.exists() and not force:
        raise RefusalError(
            f"{database} already has a residue-code store ({code_store.manifest_path}). "
            f"Pass --force to replace it."
        )
    system = scoring_system.get(scoring_system.DEFAULT_ESM_CODES)
    run_id, shards = _residue_code_shards(shards_folder, run_id)
    # Every shard's footer is checked here, before any data and before anything is written.
    infos = check_shards(shards, RESIDUE_CODES, system)
    rows = sum(pq.read_metadata(path).num_rows for path in shards)
    if not rows == entry_count == faiss_entries:
        raise RefusalError(
            f"The {len(shards)} shard(s) of run {run_id} in {shards_folder} hold {rows} entries, "
            f"but {database} has {entry_count} in its sequence store and {faiss_entries} in its "
            f"FAISS index: they are not this database's build shards, or a rank's shard is "
            f"missing. Nothing was changed."
        )
    record, build = _code_store_build(run_id, infos, from_shards=True)
    if not isinstance(record["min_res_n"], int):
        raise RefusalError(
            f"The shards of run {run_id} in {shards_folder} do not record the --min-res-n of the "
            f"build that wrote them (written by an earlier FoldMatch). Rebuild the database with "
            f"'fm-search build sequences --residue-codes'. Nothing was changed."
        )
    logging.info(f"Building the residue-code store of {database} from {len(shards)} shard(s) of run {run_id}")
    code_store.create(iter_residue_codes(shards, system, load_batch_size), system, force=force,
                      min_res_n=record["min_res_n"], build=build, esm_revision=record["esm_revision"])
    return code_store.validate(sequence_store, faiss_entries, preflight=None)


def _residue_code_shards(folder: Path, run_id: Optional[str]):
    """``(run id, shard paths)`` of the residue-code shards in ``folder`` (paths in the order a build reads them).

    With ``run_id``, that run's shards. Without, the only run whose shards
    hold residue codes; none, or several, are refused.
    """
    from foldmatch.search.embedding_computer import run_shard_files, shard_run_ids
    from foldmatch.search.residue_shards import RESIDUE_CODES, read_shard_info
    if run_id is not None:
        shards = run_shard_files(folder, run_id)
        if not shards:
            runs = shard_run_ids(folder)
            raise RefusalError(
                f"No embedding shards of run {run_id} (emb_{run_id}-<rank>.parquet) in {folder}; "
                f"it holds shards of the run(s): {', '.join(runs) if runs else 'none'}."
            )
        return run_id, shards
    runs = {rid: run_shard_files(folder, rid) for rid in shard_run_ids(folder)}
    if not runs:
        raise RefusalError(
            f"No embedding shards (emb_<run id>-<rank>.parquet) in {folder}. A build keeps "
            f"them in its --tmp-embedding-folder unless --cleanup-embeddings."
        )
    with_codes = {}
    for rid, paths in runs.items():
        try:
            info = read_shard_info(paths[0])
        except ValueError as exc:
            raise RefusalError(
                f"{exc}. Pass --run-id to choose the run that built this database, or remove "
                f"the file if it is not a shard."
            ) from exc
        if info.residue_output == RESIDUE_CODES:
            with_codes[rid] = paths
    if not with_codes:
        raise RefusalError(
            f"None of the embedding shards in {folder} hold residue codes (run(s) "
            f"{', '.join(runs)}): they come from a build without --residue-codes. Rebuild the "
            f"database with 'fm-search build sequences --residue-codes'."
        )
    if len(with_codes) > 1:
        listed = ", ".join(f"{rid} ({len(paths)} shard(s))" for rid, paths in with_codes.items())
        raise RefusalError(
            f"{folder} holds the residue-code shards of {len(with_codes)} runs: {listed}. "
            f"Pass --run-id to choose the run that built this database."
        )
    return next(iter(with_codes.items()))


def _check_stage2_stores(subject_store, max_evalue: Optional[float], query_store=None) -> None:
    """Read-only checks of the sequence stores Stage 2 will read, before ESM3 or the search.

    Reading each store's totals refuses an incomplete store, and an E-value
    filter needs a non-empty subject database (:func:`check_evalue_search_space`).
    Both would otherwise only fail after the embedding pass.
    """
    from foldmatch.search.alignment import check_evalue_search_space
    if query_store is not None:
        query_store.meta()
    check_evalue_search_space(max_evalue, subject_store.total_residues())


def _resolve_stage2(seq_identity: Optional[bool], stores_available: bool, missing_msg: str) -> bool:
    """Decide whether to run Stage-2 sequence-identity alignment.

    ``seq_identity`` is tri-state: ``None`` = auto (run iff the required sequence
    store(s) exist), ``True`` = force on (error if stores are missing),
    ``False`` = force off. Returns whether to run Stage 2.
    """
    if seq_identity is False:
        return False
    if seq_identity is True:
        if not stores_available:
            raise ValueError(missing_msg)
        return True
    # auto
    if stores_available:
        logging.info(
            "Sequence store detected; running Stage-2 sequence-identity alignment "
            "(disable with --no-seq-identity)."
        )
        return True
    logging.info(
        "No sequence store for this database; reporting embedding similarity only. "
        "Build with 'fm-search build sequences' to enable sequence identity."
    )
    return False


# =====================================================================================================================
# query sequences --scoring (plan P6.1-P6.3)
# =====================================================================================================================
#: Defaults of the Stage-2 options of ``query sequences`` per scoring (P6.1). BLOSUM62's are the values the options
#: had before --scoring; esm-codes filters on the E-value only (D8).
BLOSUM62_QUERY_DEFAULTS = MappingProxyType(dict(
    min_seq_identity=0.3, min_coverage=0.8, max_evalue=1e-3, gap_open=11, gap_extend=1, comp_bias_corr=False,
    significance_mode=SignificanceMode.default,
))
ESM_CODES_QUERY_DEFAULTS = MappingProxyType(dict(min_seq_identity=0.0, min_coverage=0.0, max_evalue=1e-3))
#: The BLOSUM62-only options, refused under esm-codes whatever their value (D20: exit status 2).
_BLOSUM62_ONLY_OPTIONS = (("gap_open", "--gap-open"), ("gap_extend", "--gap-extend"),
                          ("comp_bias_corr", "--comp-bias-corr/--no-comp-bias-corr"),
                          ("significance_mode", "--significance-mode"))


@dataclass(frozen=True)
class QueryStage2Options:
    """The Stage-2 options of one ``query sequences`` run, resolved for its scoring (:func:`resolve_query_stage2_options`).
    The BLOSUM62-only fields are None under esm-codes."""
    scoring: ScoringChoice
    min_seq_identity: float
    min_coverage: float
    max_evalue: Optional[float]
    gap_open: Optional[int] = None
    gap_extend: Optional[int] = None
    comp_bias_corr: Optional[bool] = None
    significance_mode: Optional[SignificanceMode] = None


def resolve_query_stage2_options(scoring=DEFAULT_SCORING, *, seq_identity: Optional[bool] = None,
                                 min_seq_identity: Optional[float] = None, min_coverage: Optional[float] = None,
                                 max_evalue: Optional[float] = None, gap_open: Optional[int] = None,
                                 gap_extend: Optional[int] = None, comp_bias_corr: Optional[bool] = None,
                                 significance_mode: Optional[SignificanceMode] = None) -> QueryStage2Options:
    """``query sequences``' Stage-2 options for ``scoring`` (P6.1): an option left out (None) takes the scoring's
    default (:data:`BLOSUM62_QUERY_DEFAULTS`, :data:`ESM_CODES_QUERY_DEFAULTS`), an explicit value is kept.

    Under esm-codes, every BLOSUM62-only option given (``--gap-open``, ``--gap-extend``,
    ``--comp-bias-corr/--no-comp-bias-corr``, ``--significance-mode``) and ``--no-seq-identity`` are refused, and
    so are filter values out of range: ``typer.BadParameter`` (exit status 2, D20), before anything is read.
    Under blosum62 nothing is checked here, as before.
    """
    try:
        scoring = ScoringChoice(scoring)
    except ValueError:
        raise typer.BadParameter(f"{scoring!r} is not a scoring; expected one of "
                                 f"{', '.join(c.value for c in ScoringChoice)}.", param_hint="--scoring") from None
    given = dict(min_seq_identity=min_seq_identity, min_coverage=min_coverage, max_evalue=max_evalue)
    if scoring is ScoringChoice.blosum62:
        blosum = dict(gap_open=gap_open, gap_extend=gap_extend, comp_bias_corr=comp_bias_corr,
                      significance_mode=significance_mode)
        values = {name: BLOSUM62_QUERY_DEFAULTS[name] if value is None else value
                  for name, value in {**given, **blosum}.items()}
        if values["significance_mode"] is not None:
            values["significance_mode"] = SignificanceMode(values["significance_mode"])
        return QueryStage2Options(scoring=scoring, **values)
    refused = [flag for name, flag in _BLOSUM62_ONLY_OPTIONS
               if {"gap_open": gap_open, "gap_extend": gap_extend, "comp_bias_corr": comp_bias_corr,
                   "significance_mode": significance_mode}[name] is not None]
    problems = []
    if refused:
        problems.append(
            f"{', '.join(refused)} {'is a BLOSUM62 option' if len(refused) == 1 else 'are BLOSUM62 options'}: "
            f"--scoring esm-codes takes its gap costs and E-value statistics from the scoring system the "
            f"database's residue codes were built for")
    if seq_identity is False:
        problems.append("--no-seq-identity asks for an embedding-only search, which has no Stage-2 scoring: "
                        "drop --scoring esm-codes for it")
    if problems:
        raise typer.BadParameter("; ".join(problems) + ". Remove them, or use --scoring blosum62.",
                                 param_hint="--scoring")
    values = {name: ESM_CODES_QUERY_DEFAULTS[name] if value is None else value for name, value in given.items()}
    for name, flag in (("min_seq_identity", "--min-seq-identity"), ("min_coverage", "--min-coverage")):
        if not 0.0 <= values[name] <= 1.0:  # NaN fails too
            raise typer.BadParameter(f"must be a fraction between 0 and 1, got {values[name]}.", param_hint=flag)
    if not values["max_evalue"] >= 0.0:
        raise typer.BadParameter(f"must be >= 0 ('inf' for no E-value filter), got {values['max_evalue']}.",
                                 param_hint="--max-evalue")
    return QueryStage2Options(scoring=scoring, **values)


def _esm_codes_output_fields(format_output: str) -> List[str]:
    """The ``--format-output`` fields of an esm-codes query, refused (exit status 2) before anything is read."""
    from foldmatch.search import output
    try:
        return output.parse_format_output(format_output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--format-output") from None


@dataclass(frozen=True)
class EsmCodesQueryDatabase:
    """What :func:`resolve_esm_codes_database` found: the database's stores, its code-store manifest, its scoring
    system (the registry entry the manifest names), the kernel and the prepared (sha256-verified) bundle."""
    db_path: str
    sequence_store: Any
    code_store: Any
    manifest: Mapping[str, Any]
    system: Any
    kernel: Any
    prepared: Any

    @property
    def total_residues(self) -> int:
        """N of the E-values: the database's residue count (manifest ``total_residues``)."""
        return int(self.manifest["total_residues"])


def resolve_esm_codes_database(db_path: str, max_evalue: Optional[float], *, num_nodes: int = 1
                               ) -> EsmCodesQueryDatabase:
    """The read-only resolution of ``query sequences --scoring esm-codes`` (plan P6.2), run on every rank before
    ESM3, so a refusal stops every rank alike. Each step refuses with its own message, as a
    :class:`~foldmatch.errors.RefusalError` (exit status 1, D20):

    1. the database has a sequence store and a complete residue-code store: a database built without
       ``--residue-codes`` and a partial build of the codes are told apart;
    2. the registry knows the system the manifest names (else the database comes from a newer FoldMatch);
    3. ``check_compatible`` of the code store against the system and the sequence store's meta;
    4. the native kernel loads (else the message gives the rebuild command);
    5. ``prepare`` of the system's bundle, pinned by its registry entry: downloaded only by the process that starts
       the run, and found in the cache by every other rank
       (:func:`~foldmatch.inference.chain_complete_inference.prepare_scoring_system`);
    6. the option combinations that depend on the database: a finite ``max_evalue`` needs residues to count.

    The live FAISS count is checked later, on rank 0, once the index is loaded.
    """
    from foldmatch.search import scoring_system as ss
    from foldmatch.search import swkern
    from foldmatch.search.residue_code_store import (
        IncompatibleCodeStoreError, ResidueCodeStore, ResidueCodeStoreError,
    )
    code_store = ResidueCodeStore.from_db_path(db_path)
    database = code_store.base
    blosum = "--scoring blosum62 still works with this database."
    sequence_store = code_store.sequence_store()
    if not sequence_store.exists():
        raise RefusalError(
            f"--scoring esm-codes: {database} has no sequence store ({sequence_store.path}). It needs a sequence "
            f"database with residue codes, built with 'fm-search build sequences --residue-codes'.")
    # 1. The code store and its manifest.
    from_shards = (f"'fm-search build residue-codes {code_store.db_folder} {code_store.index_name} "
                   f"--from-shards DIR'")
    try:
        manifest = code_store.manifest()
    except FileNotFoundError as exc:
        raise RefusalError(
            f"--scoring esm-codes: {database} was built without --residue-codes: it has no residue codes "
            f"({code_store.manifest_path} not found). Rebuild it with 'fm-search build sequences --residue-codes', "
            f"or add the codes from the shards its build kept with {from_shards}. {blosum}") from exc
    except IncompatibleCodeStoreError as exc:
        raise RefusalError(f"--scoring esm-codes: {exc}. {blosum}") from exc
    except ResidueCodeStoreError as exc:
        raise RefusalError(
            f"--scoring esm-codes: the residue codes of {database} are a partial build: "
            f"{code_store.base}.rescodes.* has no manifest ({code_store.manifest_path}), so the build that wrote "
            f"them stopped before they were validated. Rebuild them with {from_shards}, or rebuild the database "
            f"with 'fm-search build sequences --residue-codes'. {blosum}") from exc
    # 2. The registry.
    name = manifest.get("system_name")
    try:
        system = ss.get(name)
    except ss.UnknownScoringSystemError as exc:
        raise RefusalError(
            f"--scoring esm-codes: the residue codes of {database} were built for the scoring system {name!r}, "
            f"which this FoldMatch ({__version__}) does not know (it knows: {', '.join(ss.known_names())}): the "
            f"database was built by a newer FoldMatch; upgrade FoldMatch to query it with esm-codes. {blosum}"
        ) from exc
    # 3. The codes against the system and the sequence store.
    with _refusals():
        manifest = code_store.check_compatible(system)
    # 4. The kernel.
    try:
        kernel = swkern.load()
    except swkern.KernelError as exc:
        raise RefusalError(
            f"--scoring esm-codes needs the native Smith-Waterman kernel, which cannot be used here: {exc}. "
            f"{blosum}") from exc
    # 5. The bundle: one fetch, then cached-only on every other rank.
    from foldmatch.inference.chain_complete_inference import prepare_scoring_system
    from foldmatch.search.residue_shards import QUERY_ENCODING
    with _refusals():
        prepared = prepare_scoring_system(system, residue_output=QUERY_ENCODING, num_nodes=num_nodes)
    # 6. The option combinations the database decides.
    from foldmatch.search.alignment import check_evalue_search_space
    with _refusals():
        check_evalue_search_space(max_evalue, manifest["total_residues"])
    return EsmCodesQueryDatabase(db_path=db_path, sequence_store=sequence_store, code_store=code_store,
                                 manifest=manifest, system=system, kernel=kernel, prepared=prepared)


def _query_sequences_esm_codes(*, db_path: str, fasta_file: Path, tmp_embedding_folder: Path, output_file: str,
                               min_res_n: int, top_k: int, threshold: Optional[float], use_gpu_index: bool,
                               accelerator, num_nodes: int, devices, strategy, batch_size: int, num_workers: int,
                               nprobe: Optional[int], stage2: QueryStage2Options, align_workers: Optional[int],
                               format_output: str, format_mode: FormatMode) -> None:
    """``query sequences --scoring esm-codes`` (plan section 3, QUERY; P6.1-P6.2): every rank's read-only checks
    before ESM3. The remaining option checks (exit status 2), the strict query pre-flight (duplicate ids, '<' / '>'
    and non-ASCII sequences refused) and :func:`resolve_esm_codes_database` (exit status 1). The query pass with the
    query encodings and the esm-codes Stage 2 come with P6.3.
    """
    if align_workers is not None and align_workers < 0:
        raise typer.BadParameter(f"must be >= 0, got {align_workers}.", param_hint="--align-workers")
    _esm_codes_output_fields(format_output)
    with _refusals():
        _fasta_preflight(fasta_file, min_res_n, strict=True)
    resolve_esm_codes_database(db_path, stage2.max_evalue, num_nodes=num_nodes)
    raise RefusalError("--scoring esm-codes: the database and the query pass every check, but the query pass with "
                       "query encodings is not implemented yet (plan P6.3). --scoring blosum62 works.")


def _stage2_align_and_report(
        embedding_db,
        prefilter_results,
        query_sequences,
        min_seq_identity: float,
        min_coverage: float,
        max_evalue: Optional[float],
        gap_open: int,
        gap_extend: int,
        comp_bias_corr: bool,
        comp_bias_corr_scale: float,
        num_workers: Optional[int],
        output_file: str,
        significance_mode: SignificanceMode = SignificanceMode.default,
        significance_sample_size: int = 500,
        format_output: str = DEFAULT_FORMAT_OUTPUT,
        format_mode: FormatMode = FormatMode.headless_tsv,
):
    """Run Stage-2 pairwise alignment on prefilter candidates and report.

    The subject sequences are recovered from the subject database's sidecar
    sequence store; raises a clear error if that store is missing (e.g. a
    database built from structures rather than FASTA). Results are written as
    MMseqs2-style tab-separated rows over the ``--format-output`` columns.
    """
    from foldmatch.search.sequence_store import SequenceStore
    from foldmatch.search import alignment, output

    # Validate the requested columns before doing any alignment work, so a typo
    # fails fast with the full list of supported fields.
    output_fields = output.parse_format_output(format_output)

    subject_store = SequenceStore(embedding_db.db_folder, embedding_db.index_name)
    if not subject_store.exists():
        raise ValueError(
            f"Subject database has no sequence store ({subject_store.path}); "
            f"Stage-2 sequence identity requires a database built from FASTA "
            f"sequences (fm-search build sequences)."
        )

    logging.info("Computing pairwise sequence alignments (Stage 2)...")
    aligned = alignment.align_candidates(
        query_sequences=query_sequences,
        prefilter_results=prefilter_results,
        fetch_subject_sequences=subject_store.fetch,
        min_seq_identity=min_seq_identity,
        min_coverage=min_coverage,
        max_evalue=max_evalue,
        gap_open=gap_open,
        gap_extend=gap_extend,
        comp_bias_corr=comp_bias_corr,
        comp_bias_corr_scale=comp_bias_corr_scale,
        num_workers=num_workers,
        subject_db_size=subject_store.total_residues(),
        # Only pay for the significance pass when a significance column
        # (evalue/bits) is actually requested — align_candidates turns it back
        # on by itself when max_evalue asks for an E-value to filter on.
        compute_significance=output.needs_significance(output_fields),
        significance_mode=significance_mode,
        significance_sample_size=significance_sample_size,
        output_fields=output_fields,
    )

    output.write_aligned_results(aligned, output_fields, output_file, format_mode)


def _is_rank_zero():
    """Check if the current process is running in distributed mode."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        return True
    return False


def main():
    """Entry point that renders expected errors as clean messages (no traceback)."""
    try:
        app()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
