import os
import logging
import typer
from pathlib import Path
from typing import Annotated, Optional, List

from foldmatch import __version__
from foldmatch.cli.args_utils import arg_devices, set_log_level
from foldmatch.search.output import (
    CLUSTER_OUTPUT_FIELDS,
    DEFAULT_FORMAT_OUTPUT,
    EMBEDDING_OUTPUT_FIELDS,
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
    f"Available fields: {', '.join(SUPPORTED_OUTPUT_FIELDS)}."
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
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Build an embedding database from protein sequences in a FASTA file."""

    set_log_level(log_level)

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

    # Persist a sidecar id->sequence store so this database can be the
    # subject (or query) of a Stage-2 pairwise-alignment search. Built
    # directly from the FASTA — the DB ids ARE the FASTA headers — using the
    # same min_res_n filter so store and index stay in lock-step.
    from foldmatch.search.sequence_store import SequenceStore
    SequenceStore(embedding_db.db_folder, embedding_db.index_name).create(
        fasta_file=fasta_file,
        min_res_n=min_res_n,
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
        seq_identity: Annotated[Optional[bool], typer.Option(
            help='Stage 2: after the embedding prefilter, pairwise-align each '
                 'query against its candidate subjects and report exact sequence '
                 'identity. Default: AUTO — on when the subject database has a '
                 'sequence store (built from FASTA), else embedding-only. Use '
                 '--no-seq-identity to force off, --seq-identity to force on '
                 '(errors if the store is missing).'
        )] = None,
        min_seq_identity: Annotated[float, typer.Option(
            help='When --seq-identity is set, drop hits whose alignment identity '
                 'is below this fraction (0-1).'
        )] = 0.3,
        min_coverage: Annotated[float, typer.Option(
            help='When --seq-identity is set, drop hits whose query AND subject '
                 'coverage are below this fraction (0-1).'
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
            help='Logging level.'
        )] = LogLevel.info
):
    """Search database using protein sequences from a FASTA file."""

    set_log_level(log_level)
    _check_comp_bias_corr_scale(comp_bias_corr_scale)

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
    logging.info(f"Searched {len(results)} sequence(s)")
    results = _filter_results_by_threshold(results, threshold)

    from foldmatch.search.sequence_store import SequenceStore
    subject_store = SequenceStore(embedding_db.db_folder, embedding_db.index_name)
    run_stage2 = _resolve_stage2(
        seq_identity,
        subject_store.exists(),
        f"Subject database has no sequence store ({subject_store.path}); "
        f"--seq-identity requires a database built from FASTA (fm-search build sequences).",
    )

    if run_stage2:
        from foldmatch.utils.fasta import iter_fasta
        query_sequences = dict(iter_fasta(fasta_file))
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
            help='When --seq-identity is set, drop hits whose query AND subject '
                 'coverage are below this fraction (0-1).'
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

    # Stage 2 needs sequences for BOTH sides: queries from the query DB's store,
    # subjects from the subject DB's store.
    from foldmatch.search.embedding_database import _parse_db_path
    from foldmatch.search.sequence_store import SequenceStore
    q_folder, q_name, _ = _parse_db_path(query_db_path)
    query_store = SequenceStore(q_folder, q_name)
    subject_store = SequenceStore(embedding_db.db_folder, embedding_db.index_name)
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
