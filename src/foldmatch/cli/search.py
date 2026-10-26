import glob
import os
import logging
import shlex
import sqlite3
import sys
import time
import typer
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, List, Mapping, Optional, Tuple

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
    "unless --cleanup-embeddings), or those of a 'build sequences-embed "
    "--residue-codes' run (its --shards-folder)."
)

# Plan P8b (CLI): the split build's step 1, verbatim but for --kill-on-invalid-dep=yes (without it, the job of step 2
# stays pending for ever once step 1 fails).
_EMBED_HELP = (
    "Step 1 of a split 'build sequences' for multi-node GPU runs: the pre-flight and "
    "the ESM3 pass (chain embeddings, and residue codes with --residue-codes) on every "
    "rank, writing the embedding shards and, last, a run record into --shards-folder. "
    "No database file is written. Then build the database with 'build sequences-index' "
    "as its own one-process job on a CPU node (e.g. sbatch --dependency=afterok:<this "
    "job> --kill-on-invalid-dep=yes), so that these nodes are released."
)

_EMBED_OUTPUT_DB_HELP = (
    "Path of the database 'build sequences-index' will build (as --output-db of 'build "
    "sequences'). Refused if any of its files exist; recorded in the run record."
)

_SHARDS_FOLDER_HELP = (
    "Folder for this step's output: the embedding shards (emb_<run id>-<rank>.parquet) "
    "and, written last, the run record (emb_<run id>.run.json) that 'build "
    "sequences-index --from-shards' reads. It must be on storage every node can write "
    "and that outlives the job (e.g. $SCRATCH), with room for about 6 KiB per entry "
    "plus 8 B per residue with --residue-codes. Use one folder per build."
)

# Plan P8b (CLI), verbatim: the split build's step 2.
_INDEX_HELP = (
    "Step 2 of a split 'build sequences': build the database from the shards of one "
    "'build sequences-embed' run, after checking against its run record that they are "
    "complete and were computed from this FASTA: the FAISS index, the sequence store and, "
    "when the run computed residue codes, the residue-code store, as 'build sequences' "
    "builds them after its ESM3 pass. No ESM3, GPU or download. One process: run it as its "
    "own job on a CPU node, not inside the GPU job; FAISS uses the node's physical cores "
    "unless OMP_NUM_THREADS is set."
)

_INDEX_FASTA_HELP = (
    "The FASTA file the run embedded. Checked against the run record (size, sha256, "
    "pre-flight counts) and against the id of every shard row; it may have moved since."
)

_INDEX_OUTPUT_DB_HELP = (
    "Path to save the FAISS database (as for 'build sequences'). Usually the run's "
    "--output-db, which the record holds; another path is allowed."
)

_INDEX_FROM_SHARDS_HELP = (
    "Folder holding the shards and the run record of a 'build sequences-embed' run (its "
    "--shards-folder)."
)

_INDEX_RUN_ID_HELP = (
    "Run id of the run to build from (its record is emb_<run id>.run.json; 'build "
    "sequences-embed' logs it). Needed when the folder holds more than one run, unfinished "
    "ones included."
)

_INDEX_RESIDUE_CODES_HELP = (
    "Build the residue-code store from the run's codes (enabling `query sequences --scoring "
    "esm-codes`). Default: exactly when the run computed them ('build sequences-embed "
    "--residue-codes'). --residue-codes refuses a run without codes; --no-residue-codes "
    "leaves the store out (add it later with 'build residue-codes --from-shards')."
)

_INDEX_CLEANUP_EMBEDDINGS_HELP = (
    "Delete the run's record, then its shards, from --from-shards once the database is "
    "built. Use it only on the last build from this run. Refused together with "
    "--no-residue-codes on a run with residue codes. By default they are kept."
)

# Plan P8b (CLI), verbatim: the split build's recovery.
_SEAL_HELP = (
    "Recovery: write the run record of embedding shards that have none, so that 'build "
    "sequences-index' can build from them: a 'build sequences-embed' run whose rank 0 stopped "
    "after the pass (wall time, out of memory, a full disk), or the shards a 'build sequences' "
    "run kept when its index phase failed. Checks the shards against the FASTA (entry count, "
    "each rank's rows, the shard schema and the id of every row) without ESM3 or a GPU, in one "
    "process. The record says it was sealed: it binds the shards by these checks, not by the "
    "pass."
)

_SEAL_FASTA_HELP = "The FASTA file the run embedded."

_SEAL_FROM_SHARDS_HELP = "Folder holding the run's shards."

_SEAL_RUN_ID_HELP = "Run id of the shards to seal (emb_<run id>-<rank>.parquet)."

_SEAL_MIN_RES_N_HELP = (
    "The --min-res-n the run used. Required for a run without residue codes, whose shards do "
    "not record it; for a run with codes it is read from the shards, and a different value is "
    "refused."
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
            help='Number of vectors per batch streamed into the FAISS index; larger '
                 'batches use more memory (FAISS adds IVF-PQ vectors in chunks of '
                 '32768 whatever this value).'
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
            help='Number of vectors per batch streamed into the FAISS index; larger '
                 'batches use more memory (FAISS adds IVF-PQ vectors in chunks of '
                 '32768 whatever this value).'
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
            help='Number of vectors per batch streamed into the FAISS index; larger '
                 'batches use more memory (FAISS adds IVF-PQ vectors in chunks of '
                 '32768 whatever this value).'
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
    # Read only, for the index phase's log lines: Lightning's DDP launcher sets OMP_NUM_THREADS during the pass when
    # it is unset (to the allowed CPUs // devices per node), and FAISS starts with that value.
    omp_env_at_start = os.environ.get("OMP_NUM_THREADS")
    run = _sequence_build_preflight(fasta_file, min_res_n, output_db, tmp_embedding_folder, num_nodes, residue_codes)
    # A --residue-codes build reports every refusal as a RefusalError (D20:
    # exit 1); a plain build raises exactly what it raised before.
    with run.refusals():
        _run_sequence_pass(run, fasta_file=fasta_file, min_res_n=min_res_n, batch_size=batch_size,
                           num_workers=num_workers, num_nodes=num_nodes, accelerator=accelerator, devices=devices,
                           strategy=strategy)

    if not _is_rank_zero():
        return
    if num_nodes > 1:
        logging.info(
            f"The index phase runs in this one process: the other {num_nodes - 1} node(s) stay allocated until it "
            f"ends. 'build sequences-embed' and then 'build sequences-index' (as its own job on a CPU node) "
            f"release them when the ESM3 pass ends."
        )

    _build_sequence_database(
        run.embedding_computer, run.preflight, fasta_file=fasta_file, output_db=output_db, min_res_n=min_res_n,
        system=None if run.prepared is None else run.prepared.system,
        index=_IndexOptions(use_gpu_index, index_type, ivf_nlist, ivf_nprobe, load_batch_size),
        refusals=run.refusals, omp_env_at_start=omp_env_at_start,
    )
    if cleanup_embeddings:
        run.embedding_computer.remove_shards()


@build_db_app.command(
    name="sequences-embed",
    help=_EMBED_HELP
)
def build_sequences_embed(
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences.'
        )],
        output_db: Annotated[str, typer.Option(
            help=_EMBED_OUTPUT_DB_HELP
        )],
        shards_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_SHARDS_FOLDER_HELP
        )],
        min_res_n: Annotated[int, typer.Option(
            min=0,
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
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
        residue_codes: Annotated[bool, typer.Option(
            "--residue-codes/--no-residue-codes",
            help=_RESIDUE_CODES_HELP
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Step 1 of a split sequence build: the pre-flight and the ESM3 pass on every rank, then rank 0 checks the shards
    and writes the run's record last. No database file."""

    # Every rank, read-only, before ESM3. Every refusal is a RefusalError, in both modes (D32).
    set_log_level(log_level)
    from foldmatch.search import split_run
    before, before_error = _fasta_stat_or_error(fasta_file, "before the pre-flight")
    start = time.monotonic()
    with _refusals():
        run = _sequence_build_preflight(fasta_file, min_res_n, output_db, shards_folder, num_nodes, residue_codes)
    preflight_seconds = time.monotonic() - start
    with _refusals():
        split_run.refuse_existing_record(shards_folder, run.embedding_computer.run_id)

    # Every rank: the pass of the one-shot build, up to its barrier.
    with _refusals():
        _run_sequence_pass(run, fasta_file=fasta_file, min_res_n=min_res_n, batch_size=batch_size,
                           num_workers=num_workers, num_nodes=num_nodes, accelerator=accelerator, devices=devices,
                           strategy=strategy)
    after, after_error = _fasta_stat_or_error(fasta_file, "after the pass")
    facts = _gather_rank_facts({
        "before": before,
        "after": after,
        "error": "; ".join(error for error in (before_error, after_error) if error) or None,
        "size": None if before is None else before.bytes,
        "preflight": split_run.preflight_counts(run.preflight),
        "seconds": preflight_seconds,
    })
    if not _is_rank_zero():
        return

    options = {"accelerator": accelerator.value, "devices": list(devices), "num_nodes": num_nodes,
               "strategy": strategy.value, "batch_size": batch_size, "num_workers": num_workers}
    record, path = _record_embed_run(run, facts, fasta_file=fasta_file, output_db=output_db,
                                     shards_folder=shards_folder, min_res_n=min_res_n, options=options)
    codes = f", residue codes of {record['system']['name']}" if record["system"] is not None else ""
    logging.info(
        f"Run {record['run_id']} complete: {record['rows']} entries in {len(record['shards'])} shard(s) of "
        f"{record['world_size']} rank(s){codes}; record {path}"
    )
    logging.info("Build the database with: " + shlex.join([
        "fm-search", "build", "sequences-index", "--fasta-file", str(fasta_file), "--output-db", record["output_db"],
        "--from-shards", str(shards_folder), "--run-id", record["run_id"],
    ]))
    logging.info(
        f"Add --index-type to choose its FAISS index: auto picks hnsw at {_AUTO_HNSW_ENTRIES:,} entries or more; flat "
        f"and hnsw hold every embedding in memory, about 12 KiB per entry at peak; ivf_pq is bounded."
    )


@build_db_app.command(
    name="sequences-index",
    help=_INDEX_HELP
)
def build_sequences_index(
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help=_INDEX_FASTA_HELP
        )],
        output_db: Annotated[str, typer.Option(
            help=_INDEX_OUTPUT_DB_HELP
        )],
        from_shards: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_INDEX_FROM_SHARDS_HELP
        )],
        run_id: Annotated[Optional[str], typer.Option(
            help=_INDEX_RUN_ID_HELP
        )] = None,
        residue_codes: Annotated[Optional[bool], typer.Option(
            "--residue-codes/--no-residue-codes",
            help=_INDEX_RESIDUE_CODES_HELP
        )] = None,
        cleanup_embeddings: Annotated[bool, typer.Option(
            help=_INDEX_CLEANUP_EMBEDDINGS_HELP
        )] = False,
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
            help='Number of vectors per batch streamed into the FAISS index; larger '
                 'batches use more memory (FAISS adds IVF-PQ vectors in chunks of '
                 '32768 whatever this value).'
        )] = 32768,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Step 2 of a split sequence build: one process checks a run's shards against its record and the FASTA, then
    builds the database from them as 'build sequences' does after its pass. No ESM3, GPU or download."""

    # Every refusal is a RefusalError, in both modes (D32). Up to the index phase nothing is written but the build lock
    # (and the database's folder when it is missing), which a refusal removes.
    set_log_level(log_level)
    start = time.monotonic()
    from foldmatch.utils.dist import is_run_id
    if run_id is not None and not is_run_id(run_id):
        raise typer.BadParameter(
            f"{run_id!r} is not a run id: letters, digits, '.', '_' and '-' only, "
            f"starting with a letter or digit.", param_hint="--run-id"
        )
    _refuse_multi_node_job("build sequences-index")
    # Before torch, FAISS or pyarrow start a thread pool: the threads they start inherit this mask (D29).
    from foldmatch.search.alignment import _expand_cpu_affinity
    _expand_cpu_affinity(purpose="the index phase")

    from foldmatch.search import split_run
    with _refusals():
        run_id, record_path = split_run.find_run(from_shards, run_id)
        record = split_run.read_run_record(record_path)
        codes, system = _split_code_store(record, record_path, residue_codes=residue_codes,
                                          cleanup_embeddings=cleanup_embeddings)
    db_path = parse_db_path(output_db)[2]
    with _database_folder(output_db), split_run.build_lock(db_path, record, shards_folder=from_shards):
        with _refusals():
            paths, preflight, fasta_stat = _split_index_checks(record, fasta_file=fasta_file, output_db=output_db,
                                                               from_shards=from_shards, codes=codes, system=system)
        _log_split_build(record, db_path=db_path, from_shards=from_shards, shards=len(paths), codes=codes)
        from foldmatch.search.embedding_computer import EmbeddingComputer
        computer = EmbeddingComputer(from_shards, run=run_id)
        with _refusals():
            computer.attach_preflight(fasta_file, record["min_res_n"], preflight)
        with _whole_node_threads():
            _build_sequence_database(
                computer, preflight, fasta_file=fasta_file, output_db=output_db, min_res_n=record["min_res_n"],
                system=system if codes else None,
                index=_IndexOptions(use_gpu_index, index_type, ivf_nlist, ivf_nprobe, load_batch_size),
                refusals=_refusals, split=record,
                after_sequence_store=_fasta_unchanged_guard(fasta_file, fasta_stat, output_db),
            )
        if cleanup_embeddings:
            # The record first: no record outlives its shards.
            with _refusals():
                record_path.unlink()
                computer.remove_shards()
    logging.info(f"Database {db_path} built from run {run_id} in {time.monotonic() - start:.1f} s")


@build_db_app.command(
    name="sequences-seal",
    help=_SEAL_HELP
)
def build_sequences_seal(
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help=_SEAL_FASTA_HELP
        )],
        from_shards: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_SEAL_FROM_SHARDS_HELP
        )],
        run_id: Annotated[str, typer.Option(
            help=_SEAL_RUN_ID_HELP
        )],
        min_res_n: Annotated[Optional[int], typer.Option(
            min=0,
            help=_SEAL_MIN_RES_N_HELP
        )] = None,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    """Recovery of a split sequence build: one process checks the shards of a run that has no run record against the
    FASTA, then writes their record, sealed, so that 'build sequences-index' can build from them. No ESM3 or GPU."""

    # Every refusal is a RefusalError, in both modes (D32). Nothing is written but the record, last.
    set_log_level(log_level)
    from foldmatch.utils.dist import is_run_id
    if not is_run_id(run_id):
        raise typer.BadParameter(
            f"{run_id!r} is not a run id: letters, digits, '.', '_' and '-' only, "
            f"starting with a letter or digit.", param_hint="--run-id"
        )
    _refuse_multi_node_job("build sequences-seal")

    from foldmatch.search import split_run
    with _refusals():
        split_run.refuse_existing_record(from_shards, run_id, command=split_run.SEAL_COMMAND)
        record = _seal_run(fasta_file, from_shards, run_id, min_res_n)
        path = split_run.write_run_record(from_shards, record)
    codes = f", residue codes of {record['system']['name']}" if record["system"] is not None else ""
    logging.info(
        f"Run {run_id} sealed: {record['rows']} entries in {len(record['shards'])} shard(s) of "
        f"{record['world_size']} rank(s){codes}; record {path}"
    )
    logging.warning(
        f"The record of run {run_id} is sealed: it binds the shards to {fasta_file} by their entry count, each rank's "
        f"rows and the id of every row"
        + (" and its seq_sha1, the sha1 of the entry's sequence" if codes else "")
        + ", not by the pass that wrote them, which 'build sequences-seal' did not see."
    )
    logging.info("Build the database DB with: " + shlex.join([
        "fm-search", "build", "sequences-index", "--fasta-file", str(fasta_file), "--output-db", "DB",
        "--from-shards", str(from_shards), "--run-id", run_id,
    ]))


@build_db_app.command(
    name="residue-codes",
    help="Build the residue-code store of a sequence database from the embedding "
         "shards its 'build sequences --residue-codes' run kept, or those of the "
         "'build sequences-embed --residue-codes' run it was built from, without "
         "running ESM3 again, and validate it against the database. DB_FOLDER and "
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
        comp_bias_corr_scale: Annotated[Optional[float], typer.Option(
            min=0.0, max=1.0,
            help='Scale applied to the compositional bias correction when '
                 '--comp-bias-corr is set (0-1). Default: 1.0.' + _BLOSUM62_ONLY
        )] = None,
        significance_mode: Annotated[Optional[SignificanceMode], typer.Option(
            help="Scale for the BitScore/Pvalue/Evalue columns. 'default' uses "
                 "calibrated statistics, but requires the default gap penalties "
                 "11/1; 'sampled' works at any gap penalties but its absolute "
                 "values are unreliable and comparable only within this tool. "
                 "Default: 'default'." + _BLOSUM62_ONLY
        )] = None,
        significance_sample_size: Annotated[Optional[int], typer.Option(
            help="For --significance-mode sampled: number of random alignments "
                 "sampled to estimate lambda/K for the BitScore/Pvalue/Evalue "
                 "columns. Larger is slower; ignored when mode is 'default'. "
                 "Default: 500." + _BLOSUM62_ONLY
        )] = None,
        align_workers: Annotated[Optional[int], typer.Option(
            help='Process-pool size for Stage-2 alignment; set to 0 or 1 to '
                 'run serially. Default under --scoring blosum62: all CPUs on '
                 'the node (widening any scheduler --cpu-bind pinning); under '
                 'esm-codes: up to one worker per physical core (SMT siblings '
                 'count once; the pinning widened the same way), only as many '
                 'as the work needs. An esm-codes worker needs about 5 bytes '
                 'per residue pair of the pair it aligns (up to 1.3 GiB for the '
                 'longest pairs aligned, 2^28 residue pairs).'
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
    # The per-scoring defaults; under esm-codes the BLOSUM62-only options are refused (exit 2), before anything is read.
    stage2 = resolve_query_stage2_options(
        scoring, seq_identity=seq_identity, min_seq_identity=min_seq_identity, min_coverage=min_coverage,
        max_evalue=max_evalue, gap_open=gap_open, gap_extend=gap_extend, comp_bias_corr=comp_bias_corr,
        comp_bias_corr_scale=comp_bias_corr_scale, significance_mode=significance_mode,
        significance_sample_size=significance_sample_size,
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
    comp_bias_corr_scale, significance_mode = stage2.comp_bias_corr_scale, stage2.significance_mode
    significance_sample_size = stage2.significance_sample_size
    _check_comp_bias_corr_scale(comp_bias_corr_scale)
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
        # The Stage-2 columns, refused (exit status 2) before ESM3, as under esm-codes.
        _output_fields(format_output)
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
        embedding_batches=embedding_computer.get_embedding_batches(rows=True),
        top_k=top_k
    )
    # The shards are read rank by rank: the queries are written in input order, and a repeated id keeps its last
    # record's hits, as by a 1-rank run.
    results = embedding_computer.in_input_order(results)
    # The query embeddings are only needed for this search.
    embedding_computer.remove_shards()
    logging.info(f"Searched {len(results)} sequence(s)")
    results = _filter_results_by_threshold(results, threshold)

    if run_stage2:
        # Align the sequences ESM3 embedded (each query's last kept record, the one behind its Stage-1 hits),
        # normalised like the stored subjects.
        query_sequences = embedding_computer.input_sequences()
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
    if output_db is not None:
        _refuse_nothing_to_build(fasta_file, min_res_n, preflight)
    return preflight


def _refuse_nothing_to_build(fasta_file: Path, min_res_n: int, preflight) -> None:
    """Raise ``ValueError`` when the pre-flight kept no entry of the FASTA: there is nothing to build."""
    if preflight.entries_kept == 0:
        raise ValueError(
            f"{fasta_file} has no sequence with at least {min_res_n} residues "
            f"({preflight.entries} entries in all): there is nothing to build."
        )


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


@dataclass(frozen=True)
class _IndexOptions:
    """The FAISS options of a sequence build's index phase, as ``build sequences`` takes them."""
    use_gpu_index: bool
    index_type: IndexType
    ivf_nlist: int
    ivf_nprobe: Optional[int]
    load_batch_size: int

    def index_config(self) -> IndexConfig:
        """A fresh ``IndexConfig`` for each build: an ivf_pq build sets its ``nprobe``."""
        return IndexConfig(nlist=self.ivf_nlist, nprobe=self.ivf_nprobe)


@dataclass(frozen=True)
class _SequencePass:
    """What every rank of a sequence build has after its pre-flight: the FASTA pre-flight, the embedding computer
    (this run's id and shards), the prepared scoring system (``--residue-codes``; else ``None``) and the mode."""
    preflight: Any
    embedding_computer: Any
    prepared: Any
    residue_codes: bool

    @property
    def residue(self) -> dict:
        """The residue keywords of the pass: the codes and seq_sha1 go into each shard under ``--residue-codes``;
        a plain build calls the pass exactly as before (none)."""
        if not self.residue_codes:
            return {}
        from foldmatch.search.residue_shards import RESIDUE_CODES
        return {"residue_output": RESIDUE_CODES, "scoring_system": self.prepared}

    def refusals(self):
        """The refusal wrapper of this mode: :func:`_refusals` under ``--residue-codes`` (D20), else none."""
        return _refusals() if self.residue_codes else nullcontext()


def _sequence_build_preflight(fasta_file: Path, min_res_n: int, output_db: str, shards_folder: Path, num_nodes: int,
                              residue_codes: bool) -> _SequencePass:
    """Every rank's read-only checks of a sequence build, before ESM3; returns the :class:`_SequencePass`.

    A plain build runs the FASTA pre-flight with the outputs-exist check and draws this run's id, raising what it
    always raised; a ``--residue-codes`` build runs :func:`_residue_codes_preflight` (every refusal a
    :class:`RefusalError`).
    """
    if residue_codes:
        preflight, embedding_computer, prepared = _residue_codes_preflight(
            fasta_file, min_res_n, output_db, shards_folder, num_nodes
        )
    else:
        preflight = _fasta_preflight(fasta_file, min_res_n, output_db=output_db)

        from foldmatch.search.embedding_computer import EmbeddingComputer
        embedding_computer = EmbeddingComputer(
            embedding_folder=shards_folder,
        )
        prepared = None
    return _SequencePass(preflight, embedding_computer, prepared, residue_codes)


def _run_sequence_pass(run: _SequencePass, *, fasta_file: Path, min_res_n: int, batch_size: int, num_workers: int,
                       num_nodes: int, accelerator, devices, strategy) -> None:
    """Every rank: the ESM3 pass of a sequence build into this run's shards, ending at the pass barrier
    (``compute_from_fasta``). The caller wraps it in ``run.refusals()``."""
    run.embedding_computer.compute_from_fasta(
        fasta_file=fasta_file,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        strategy=strategy,
        preflight=run.preflight,
        **run.residue,
    )


#: ``omp_env_at_start`` of an index phase whose command has no pass (``build sequences-index``): nothing in the process
#: changes ``OMP_NUM_THREADS`` before it.
_OMP_ENV_NOT_RECORDED = object()

#: The faiss-cpu version pyproject.toml pins, named by the index phase's line about a single-threaded FAISS BLAS: its
#: Linux wheels bundle an OpenMP OpenBLAS (the 1.13.2 wheels' OpenBLAS was single-threaded).
_FAISS_CPU_PIN = "1.15.1"


def _in_faiss_folder(path: str, package: str) -> bool:
    """Whether the library at ``path`` (a real path) is one FAISS ships: inside the ``faiss`` package folder
    ``package``, or in a wheel's library folder beside it (``faiss_cpu.libs``)."""
    folder = os.path.dirname(path)
    return path.startswith(package + os.sep) or (
        os.path.dirname(folder) == os.path.dirname(package)
        and os.path.basename(folder).startswith("faiss") and folder.endswith(".libs"))


def _faiss_openmp_blas(runtimes) -> Optional[str]:
    """The file name of FAISS's own BLAS among threadpoolctl's ``runtimes`` when it is an OpenMP build (the faiss-cpu
    1.15.1 wheel's ``libopenblaso``), else None. Such a BLAS runs the OpenMP threads FAISS runs, whatever
    ``OPENBLAS_NUM_THREADS`` says."""
    import faiss
    package = os.path.dirname(os.path.realpath(faiss.__file__))
    for runtime in runtimes:
        path = os.path.realpath(str(runtime.get("filepath")))
        if runtime.get("user_api") == "blas" and runtime.get("threading_layer") == "openmp" \
                and _in_faiss_folder(path, package):
            return os.path.basename(path)
    return None


def _build_sequence_database(embedding_computer, preflight, *, fasta_file: Path, output_db: str, min_res_n: int,
                             system, index: _IndexOptions, refusals, split: Optional[Mapping] = None,
                             after_sequence_store=None, omp_env_at_start=_OMP_ENV_NOT_RECORDED) -> Optional[dict]:
    """The index phase of a sequence build, in one process: the FAISS index from the run's shards, the sequence
    store from the FASTA, and with ``system`` the residue-code store; returns the code store's manifest, else
    ``None``. ``omp_env_at_start`` is ``OMP_NUM_THREADS`` as the command found it, before its pass
    (:func:`_log_index_phase`).

    ``refusals`` is the mode's refusal wrapper (``_SequencePass.refusals``). ``after_sequence_store``, if given, is
    called once the sequence store has passed :func:`_check_store_matches_preflight`, before the code store.
    ``split`` is the run record of a split build (``build sequences-index``); ``None`` for ``build sequences``. With
    it, a refusal of the FAISS index says that step 2 can be rerun with other index options (I21), and the code
    store's manifest says the build was split (D26).
    """
    from foldmatch.search.embedding_database import EmbeddingDatabase
    embedding_db = EmbeddingDatabase(
        db_path=output_db
    )
    _log_index_phase(omp_env_at_start)
    with refusals() if split is None else _refusals(
            note=f"The shards and the record of run {split['run_id']} are kept: 'build sequences-index' can be "
                 f"rerun with other index options, without a new ESM3 pass."):
        start = time.monotonic()
        embedding_db.create_db(
            embedding_batches=embedding_computer.get_embedding_batches(load_batch_size=index.load_batch_size),
            use_gpu_index=index.use_gpu_index,
            index_type=index.index_type,
            index_config=index.index_config(),
        )
        logging.info(f"FAISS index built in {time.monotonic() - start:.1f} s")

    # Persist a sidecar id->sequence store so this database can be the
    # subject (or query) of a Stage-2 pairwise-alignment search. Built
    # directly from the FASTA — the DB ids ARE the FASTA headers — using the
    # same min_res_n filter so store and index stay in lock-step. Rank 0 only,
    # after the pass, so ranks never race on the SQLite file.
    from foldmatch.search.sequence_store import SequenceStore
    store = SequenceStore(embedding_db.db_folder, embedding_db.index_name)
    with refusals():
        start = time.monotonic()
        store.create(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
        )
        logging.info(f"Sequence store built in {time.monotonic() - start:.1f} s")
        _check_store_matches_preflight(store, preflight, output_db)
        if after_sequence_store is not None:
            after_sequence_store()
    if system is None:
        return None
    # Last, once the database it describes is complete; its manifest is
    # written last of all. A refusal leaves the database (BLOSUM62) and the
    # shards (for --from-shards) in place.
    start = time.monotonic()
    manifest = _build_residue_code_store(
        embedding_computer, embedding_db, store, preflight, system,
        min_res_n=min_res_n, load_batch_size=index.load_batch_size, split=split is not None,
    )
    logging.info(f"Residue-code store built in {time.monotonic() - start:.1f} s")
    return manifest


def _log_index_phase(omp_env_at_start=_OMP_ENV_NOT_RECORDED) -> None:
    """Log the threads the index phase gets (FAISS's OpenMP, the CPUs this process may use, each thread-pool runtime)
    and warn, except on macOS (where ``foldmatch`` pins ``OMP_NUM_THREADS=1``), when FAISS has fewer threads than
    the usable physical cores, naming the cause. Diagnostics only: nothing is changed.

    ``omp_env_at_start`` is ``OMP_NUM_THREADS`` when the command started (``None``: unset). When it was unset then and
    is set now, the pass set it: Lightning's DDP launcher does so for a multi-device run (to the allowed CPUs // the
    devices per node, also for ranks srun starts), and FAISS, first loaded here, starts with that value.

    One more line (a WARNING, INFO on macOS) when FAISS's BLAS runs on one thread by build: a BLAS runtime at one
    thread whose threading layer is ``disabled``, or one in FAISS's own library folder (``faiss_cpu.libs`` beside the
    package, or inside it) whose threading layer threadpoolctl does not report as threaded. The faiss-cpu 1.13.2
    wheel's OpenBLAS is such a library (its ivf_pq training and adds ran on one core); the OpenMP OpenBLAS of the
    1.15.1 wheel FoldMatch pins (:data:`_FAISS_CPU_PIN`) is not.
    """
    import faiss
    from foldmatch.search.code_alignment import _physical_cpus
    omp_threads = faiss.omp_get_max_threads()
    cpus = os.cpu_count() or 1
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:  # no affinity masks (macOS)
        allowed = cpus
    physical = _physical_cpus()
    omp_env = os.environ.get("OMP_NUM_THREADS")
    set_by_pass = omp_env_at_start is None and omp_env is not None
    logging.info(
        f"Index phase: FAISS {faiss.__version__}, {omp_threads} OpenMP thread(s); {allowed} of {cpus} CPUs allowed "
        f"({physical} physical cores); OMP_NUM_THREADS={omp_env if omp_env is not None else 'unset'}"
        + (" (set during the ESM3 pass by Lightning: the allowed CPUs // devices per node; unset at the start)"
           if set_by_pass else "")
    )
    try:
        import threadpoolctl
        runtimes = threadpoolctl.threadpool_info()
    except Exception as exc:  # diagnostics only: never fail a build over them
        logging.debug(f"threadpoolctl could not list the thread pools: {exc}")
        runtimes = []
    for runtime in runtimes:
        logging.info(
            f"Thread pool: {runtime.get('internal_api')}, {runtime.get('num_threads')} thread(s), "
            f"{os.path.basename(str(runtime.get('filepath')))}"
        )
    # The k-means of an ivf_pq training and the IVF assignment of its adds are BLAS products: on a BLAS built without
    # threads they run on one core whatever FAISS's OpenMP threads. A threaded BLAS that a variable limits to one
    # thread (the faiss-cpu 1.15.1 wheel's OpenMP OpenBLAS at OMP_NUM_THREADS=1, for example) is not flagged.
    package = os.path.dirname(os.path.realpath(faiss.__file__))
    for runtime in runtimes:
        path = os.path.realpath(str(runtime.get("filepath")))
        own = _in_faiss_folder(path, package)
        layer = runtime.get("threading_layer")
        if runtime.get("user_api") == "blas" and runtime.get("num_threads") == 1 and (
                layer == "disabled" or (own and layer not in ("pthreads", "openmp"))):
            logging.log(
                logging.INFO if sys.platform == "darwin" else logging.WARNING,
                f"FAISS's BLAS, {os.path.basename(path)}, is single-threaded, so the ivf_pq training and adds run on "
                f"one core whatever FAISS's thread count: install the faiss-cpu {_FAISS_CPU_PIN} FoldMatch pins (pip "
                f"install faiss-cpu=={_FAISS_CPU_PIN}), whose wheel bundles an OpenMP OpenBLAS, or a FAISS built on "
                f"MKL (README, 'FAISS threads')."
            )
            break
    if sys.platform != "darwin" and omp_threads < physical:
        # With OMP_NUM_THREADS unset, OpenMP starts one thread per CPU of the affinity mask, never fewer than its
        # physical cores: fewer threads come from the variable, or from a limit set inside this process or by another
        # OpenMP variable (not from the mask).
        if set_by_pass:
            cause = (f"OMP_NUM_THREADS={omp_env}, which Lightning's DDP launcher set during the ESM3 pass (it was "
                     f"unset at the start; Lightning sets it to the allowed CPUs // devices per node)")
            remedy = (" To choose FAISS's threads, export OMP_NUM_THREADS before the run (it also sets each rank's "
                      "torch threads in the pass), or build in two steps ('build sequences-embed', then 'build "
                      "sequences-index' as its own job).")
        elif omp_env is not None:
            cause, remedy = f"OMP_NUM_THREADS={omp_env}", ""
        else:
            cause, remedy = ("an OpenMP thread limit set inside this process or by another OpenMP variable "
                             "(OMP_NUM_THREADS is unset)"), ""
        logging.warning(
            f"FAISS has {omp_threads} OpenMP thread(s) for {physical} usable physical cores, because of {cause}: "
            f"the index phase may be slower than this node allows.{remedy}"
        )


# FaissEmbeddingDatabase.N_EMBEDDINGS_DB_THR, where --index-type auto switches from flat to hnsw (the module imports
# FAISS, which step 1 never needs).
_AUTO_HNSW_ENTRIES = 10000


def _fasta_stat_or_error(fasta_file: Path, when: str):
    """``(FastaStat, None)``, or ``(None, error text)`` when the FASTA cannot be stat'ed: step 1 keeps a rank's stat
    error in its facts for rank 0 to refuse (E11), so that every rank still reaches the gather."""
    from foldmatch.search.split_run import fasta_stat
    try:
        return fasta_stat(fasta_file), None
    except OSError as exc:
        return None, f"stat {when}: {exc}"


def _gather_rank_facts(facts: dict) -> List[dict]:
    """Every rank's ``facts``, in rank order: ``torch.distributed.all_gather_object`` when the pass ran in a process
    group (it outlives the pass, whose barrier every rank has just left; under NCCL the objects travel on the device
    Lightning set for the rank), else ``[facts]``. ``build sequences-embed`` calls it on every rank right after the
    pass, with nothing before it that can raise, so no rank waits in it for a rank that refused."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        return [facts]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, facts)
    return gathered


def _record_embed_run(run: _SequencePass, facts: List[dict], *, fasta_file: Path, output_db: str, shards_folder: Path,
                      min_res_n: int, options: dict) -> Tuple[dict, Path]:
    """Rank 0 of ``build sequences-embed``, after the pass: the post-pass checks, then the run record, written last;
    returns ``(record, path)``.

    The world is the ranks that gathered their ``facts`` (:func:`_gather_rank_facts`). Refused, each leaving no record
    and the shards in place: the embedding count and the deal of the kept entries over those ranks (E9), the shard
    footers (E10), the FASTA, hashed here, against every rank's stat bracket, size and pre-flight counts (E11), and
    the record's own staging, check and commit (E12). From E11 on, the refusal also names the ``build sequences-seal``
    command that records the shards once the cause is fixed.
    """
    from foldmatch.search import split_run
    from foldmatch.search.residue_shards import RESIDUE_CODES, pass_record
    computer = run.embedding_computer
    run_id = computer.run_id
    world_size = len(facts)
    seconds = [rank_facts["seconds"] for rank_facts in facts]
    logging.info(f"Pre-flight on {world_size} rank(s): min {min(seconds):.1f} s, max {max(seconds):.1f} s")
    kept = f"No run record was written; the shards are kept in {shards_folder} as run {run_id}."
    seal = shlex.join(["fm-search", "build", "sequences-seal", "--fasta-file", str(fasta_file), "--from-shards",
                       str(shards_folder), "--run-id", run_id, "--min-res-n", str(min_res_n)])
    with _refusals(note=kept):
        computer.check_embedding_count()
        paths = computer.shard_files()
        shards = split_run.shard_entries(paths)
        _check_shards_are_the_deal(shards, run.preflight.entries_kept, world_size, shards_folder, run_id)
        if run.residue_codes:
            run_pass = split_run.check_codes_footers(paths, system=run.prepared.system, min_res_n=min_res_n,
                                                     world_size=world_size)
        else:
            from foldmatch.utils.esm.loaders import ESM_REVISION
            split_run.check_plain_footers(paths)
            run_pass = pass_record(min_res_n=min_res_n, esm_revision=ESM_REVISION, world_size=world_size)
    with _refusals(note=f"{kept} Once the cause is fixed, 'build sequences-seal' can record them: {seal}"):
        sha256, hash_seconds = split_run.fasta_sha256(fasta_file)
        final = split_run.fasta_stat(fasta_file)
        logging.info(f"FASTA sha256 of {fasta_file} ({final.bytes} bytes) in {hash_seconds:.1f} s")
        _check_rank_fasta_facts(fasta_file, facts, final)
        system = None if run.prepared is None else run.prepared.system
        record = {
            "kind": split_run.RUN_RECORD_KIND,
            "record_version": split_run.RUN_RECORD_VERSION,
            "run_id": run_id,
            "command": split_run.EMBED_COMMAND,
            "sealed": False,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "foldmatch_version": __version__,
            "fasta": {"path": str(Path(fasta_file).resolve()), "bytes": final.bytes, "mtime_ns": final.mtime_ns,
                      "sha256": sha256},
            "min_res_n": min_res_n,
            "strict": run.residue_codes,
            "preflight": split_run.preflight_counts(run.preflight),
            # The user's string is what step 2 gives create_db; the record holds the path it names.
            "output_db": os.path.abspath(parse_db_path(output_db)[2]),
            "residue_output": RESIDUE_CODES if run.residue_codes else None,
            "system": None if system is None else {
                "name": system.name, "digest": system.digest(), "encoding_digest": system.encoding_digest(),
            },
            "world_size": world_size,
            "pass": run_pass,
            "options": options,
            "shards": shards,
            "rows": sum(shard["rows"] for shard in shards),
        }
        path = split_run.write_run_record(shards_folder, record)
    return record, path


def _check_shards_are_the_deal(shards: List[dict], entries: int, world_size: int, folder: Path, run_id: str) -> None:
    """Refuse (E9) unless the shards' ranks and rows are the pass's deal of ``entries`` over ``world_size`` ranks
    (:func:`~foldmatch.search.split_run.dealt_rows`): a rank of the world that wrote no shard, a shard of a rank the
    deal gives nothing, or a rank with other rows."""
    from foldmatch.search.split_run import dealt_rows
    deal = dealt_rows(entries, world_size)
    found = {shard["rank"]: shard["rows"] for shard in shards}
    if found == deal:
        return
    missing = sorted(set(deal) - set(found))
    extra = sorted(set(found) - set(deal))
    other = [rank for rank in sorted(set(deal) & set(found)) if found[rank] != deal[rank]]

    def listed(ranks) -> str:
        return ", ".join(map(str, ranks[:10])) + (", ..." if len(ranks) > 10 else "")

    problems = []
    if missing:
        problems.append(f"no shard of rank(s) {listed(missing)}")
    if extra:
        problems.append(f"a shard of rank(s) {listed(extra)}, which the deal gives no entry")
    problems.extend(f"rank {rank} has {found[rank]} row(s), not {deal[rank]}" for rank in other[:10])
    raise RefusalError(
        f"The {len(shards)} shard(s) of run {run_id} in {folder} are not the deal of its {entries} entries over the "
        f"{world_size} rank(s) of the pass ({'; '.join(problems)}): the shards cannot be indexed."
    )


def _check_rank_fasta_facts(fasta_file: Path, facts: List[dict], final) -> None:
    """Refuse (E11) unless every rank read the same, unchanged FASTA: no rank's stat failed; rank 0's stat before its
    pre-flight equals ``final`` (its stat after hashing), and every other rank's equals its stat after the pass; every
    rank's size and pre-flight counts are rank 0's. A rank may read a node-local copy (another inode): each copy is
    bracketed on its own rank."""
    from foldmatch.search.split_run import PREFLIGHT_KEYS

    errors = [f"rank {rank}: {rank_facts['error']}" for rank, rank_facts in enumerate(facts) if rank_facts["error"]]
    if errors:
        raise RefusalError(
            f"The FASTA {fasta_file} could not be stat'ed on every rank ({'; '.join(errors[:10])}), so it cannot be "
            f"shown unchanged during the pass."
        )
    changed = [(0, facts[0]["before"], final)] if facts[0]["before"] != final else []
    changed.extend((rank, rank_facts["before"], rank_facts["after"]) for rank, rank_facts in enumerate(facts)
                   if rank > 0 and rank_facts["before"] != rank_facts["after"])
    if changed:
        raise RefusalError(
            f"The FASTA was modified during the pass ("
            + "; ".join(f"rank {rank}: before {_stat_text(a)}, after {_stat_text(b)}" for rank, a, b in changed[:10])
            + f"): the shards may not be the embeddings of {fasta_file} as it is now."
        )
    first = facts[0]
    differ = []
    for rank, rank_facts in enumerate(facts[1:], 1):
        if rank_facts["size"] != first["size"]:
            differ.append(f"rank {rank}: {rank_facts['size']} bytes, rank 0: {first['size']}")
        counts = [key for key in PREFLIGHT_KEYS if rank_facts["preflight"].get(key) != first["preflight"].get(key)]
        if counts:
            differ.append(f"rank {rank}: " + ", ".join(f"{key} {rank_facts['preflight'].get(key)}" for key in counts)
                          + ", rank 0: " + ", ".join(f"{key} {first['preflight'].get(key)}" for key in counts))
    if differ:
        raise RefusalError(
            f"The ranks did not read the same FASTA ({'; '.join(differ[:10])}): {fasta_file} was modified during the "
            f"pass, or the ranks' copies of it differ."
        )


def _stat_text(stat) -> str:
    """A :class:`~foldmatch.search.split_run.FastaStat` as a refusal shows it."""
    return f"{stat.bytes} bytes, mtime_ns {stat.mtime_ns}, inode {stat.ino}"


def _refuse_multi_node_job(command: str) -> None:
    """The launch checks of a command that runs as one process (``build sequences-index``, ``build sequences-seal``),
    from the launch environment alone.

    Refused: a process that is one of several ranks (I2, :func:`~foldmatch.utils.dist.multi_rank_launch`, which names
    the variable), and the batch step of a SLURM job of more than one node (I3, D34): the job's other nodes would stay
    allocated while one process works. A one-task srun step of such a job, and a one-node job with GPUs, get a
    WARNING. The advice never makes the command's job depend on this job, which it refuses (its ``afterok`` would
    never be met) or in which it keeps running: it is submitted once this job has ended, or next time from the GPU
    job's script before its work, with ``$SLURM_JOB_ID`` left for that script's shell.
    """
    from foldmatch.utils.dist import _whole_number, multi_rank_launch, srun_step_id
    launch = multi_rank_launch()
    if launch is not None:
        raise RefusalError(
            f"'{command}' runs as one process ({launch}): launch it once, directly in the batch script or with "
            f"srun -n 1."
        )
    job = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job:
        return
    nodes = _whole_number("SLURM_JOB_NUM_NODES")
    if nodes is not None and nodes > 1:
        chained = ("submit it from the GPU job's script before the job's work starts: sbatch -C cpu -N 1 "
                   "--dependency=afterok:$SLURM_JOB_ID --kill-on-invalid-dep=yes ...")
        if srun_step_id() is None:
            raise RefusalError(
                f"SLURM job {job} holds {nodes} nodes, and '{command}' runs in one process on one of them: the others "
                f"would stay allocated until it ends. Run it as its own one-node CPU job instead: once this job has "
                f"ended, sbatch -C cpu -N 1 ... (no dependency is needed then); next time, {chained}"
            )
        logging.warning(
            f"'{command}' runs in a one-task srun step of SLURM job {job}, which holds {nodes} nodes: the others stay "
            f"allocated until it ends. Next time, run it as its own one-node CPU job: {chained}"
        )
        return
    gpus = [f"{name}={os.environ[name].strip()}" for name in ("SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")
            if os.environ.get(name, "").strip()]
    if gpus:
        logging.warning(
            f"SLURM job {job} has GPUs ({', '.join(gpus)}), which '{command}' does not use: run it as its own job on a "
            f"CPU node (sbatch -C cpu -N 1 ...); a project with GPU hours only may run it on one GPU node, where "
            f"ivf_pq fits (flat and hnsw may not)."
        )


def _split_code_store(record: dict, record_path: Path, *, residue_codes: Optional[bool],
                      cleanup_embeddings: bool) -> Tuple[bool, Any]:
    """``(codes, system)``: whether ``build sequences-index`` builds the residue-code store, and the registry entry of
    the run's system when it does (else ``None``).

    ``--residue-codes`` unset follows the run (D24). Refused: ``--residue-codes`` on a run without codes (I8);
    ``--no-residue-codes --cleanup-embeddings`` on a run with codes (I9), which would delete the only input of a later
    ``build residue-codes --from-shards``; a run system this FoldMatch does not register, or registers with other
    digests (I10). Only the registry entry is needed: no bundle.
    """
    from foldmatch.search import scoring_system
    from foldmatch.search.residue_shards import RESIDUE_CODES
    run_id = record["run_id"]
    run_codes = record["residue_output"] == RESIDUE_CODES
    codes = run_codes if residue_codes is None else residue_codes
    if codes and not run_codes:
        raise RefusalError(
            f"Run {run_id} computed no residue codes (its record {record_path} has no residue output), so there is no "
            f"residue-code store to build from its shards: run step 1 with --residue-codes ('build sequences-embed "
            f"--residue-codes'), or build without --residue-codes."
        )
    if run_codes and not codes and cleanup_embeddings:
        raise RefusalError(
            f"--no-residue-codes with --cleanup-embeddings would delete the only copy of run {run_id}'s residue codes "
            f"(its shards in {record_path.parent}), which 'build residue-codes --from-shards' needs to add the "
            f"residue-code store later. Drop --cleanup-embeddings, or build the store too."
        )
    if not codes:
        return False, None
    recorded = record["system"]
    plain = "--no-residue-codes builds the database without its residue codes."
    with _refusals(note=f"Run {run_id}'s residue codes are of {recorded['name']}: {plain}"):
        system = scoring_system.get(recorded["name"])
    if (system.digest(), system.encoding_digest()) != (recorded["digest"], recorded["encoding_digest"]):
        raise RefusalError(
            f"Run {run_id}'s residue codes are of {recorded['name']} with digest {recorded['digest']} and "
            f"encoding_digest {recorded['encoding_digest']}, but this FoldMatch's {system.name} has digest "
            f"{system.digest()} and encoding_digest {system.encoding_digest()}: they are another FoldMatch's codes of "
            f"that name. {plain}"
        )
    return True, system


@contextmanager
def _database_folder(output_db: str):
    """The folder of ``--output-db``, created with its parents when it is missing (as the builders of ``build
    sequences`` create it), so that the build lock can be taken there; the folders created here are removed again,
    if still empty, when the block raises."""
    folder = Path(parse_db_path(output_db)[0])
    created = []
    for parent in (folder, *folder.parents):
        if parent.exists():
            break
        created.append(parent)
    with _refusals():
        folder.mkdir(parents=True, exist_ok=True)
    try:
        yield folder
    except BaseException:
        for path in created:  # the deepest first
            try:
                path.rmdir()
            except OSError:
                break
        raise


def _split_index_checks(record: dict, *, fasta_file: Path, output_db: str, from_shards: Path, codes: bool,
                        system) -> Tuple[List[Path], Any, Any]:
    """The checks ``build sequences-index`` makes inside its build lock, before any database file is written; returns
    ``(shard paths, pre-flight, FASTA stat)``.

    In this order: no output exists (I12, with the ``build residue-codes`` hint when only the code store is missing);
    the run's shards are the recorded ones, of the recorded size and rows (I13, I14); their footers (I15): the codes
    of ``system``, or under ``--no-residue-codes`` the codes of the record's system (self-consistency only, so that a
    system unknown here still gives the plain database), or plain shards; the FASTA's size (I16, not hashed when it
    differs), then its sha256 (I17), against the record; its pre-flight, with the record's ``min_res_n`` and
    strictness (I18), and the counts the record holds (I19); the id of every shard row at its dealt position (I20).
    The stat taken before the FASTA is read brackets this build's FASTA reads with :func:`_fasta_unchanged_guard`.
    """
    from foldmatch.search import split_run
    from foldmatch.search.residue_shards import RESIDUE_CODES
    run_id = record["run_id"]
    command = split_run.code_store_command(parse_db_path(output_db)[2], record, from_shards)
    with _refusals(note=None if command is None else
                   f"The database is complete but for its residue-code store: '{command}' adds it."):
        _refuse_existing_outputs(output_db, _SEQUENCE_DB_SUFFIXES, sidecar_globs=(".rescodes.*",))
    paths = split_run.check_run_shards(from_shards, record)
    footing = {"min_res_n": record["min_res_n"], "world_size": record["world_size"]}
    if codes:
        split_run.check_codes_footers(paths, system=system, **footing)
    elif record["residue_output"] == RESIDUE_CODES:
        split_run.check_codes_footers(paths, expect=record["system"], **footing)
    else:
        split_run.check_plain_footers(paths)

    fasta = record["fasta"]
    stat = split_run.fasta_stat(fasta_file)
    if stat.bytes != fasta["bytes"]:
        raise RefusalError(
            f"{fasta_file} has {stat.bytes} bytes, but the FASTA run {run_id} embedded ({fasta['path']}) had "
            f"{fasta['bytes']}: it is not the FASTA of this run."
        )
    sha256, seconds = split_run.fasta_sha256(fasta_file)
    logging.info(f"FASTA sha256 of {fasta_file} ({stat.bytes} bytes) in {seconds:.1f} s")
    if sha256 != fasta["sha256"]:
        raise RefusalError(
            f"{fasta_file} (sha256 {sha256}) is not the FASTA run {run_id} embedded ({fasta['path']}, sha256 "
            f"{fasta['sha256']}), though it has its size."
        )
    moved = [what for what, differs in (("path", str(Path(fasta_file).resolve()) != fasta["path"]),
                                        ("modification time", stat.mtime_ns != fasta["mtime_ns"])) if differs]
    if moved:
        logging.info(
            f"{fasta_file} has the size and sha256 of the FASTA run {run_id} embedded, with another "
            f"{' and '.join(moved)} (recorded: {fasta['path']}, mtime_ns {fasta['mtime_ns']})"
        )
    preflight = _fasta_preflight(fasta_file, record["min_res_n"], strict=record["strict"], output_db=output_db)
    counts = split_run.preflight_counts(preflight)
    differ = [key for key in split_run.PREFLIGHT_KEYS if counts[key] != record["preflight"][key]]
    if differ:
        raise RefusalError(
            f"The pre-flight of {fasta_file} counts " + ", ".join(f"{key} {counts[key]}" for key in differ)
            + f", but run {run_id}'s record holds " + ", ".join(f"{key} {record['preflight'][key]}" for key in differ)
            + f" for the same bytes: this FoldMatch's pre-flight counts differently from the one that ran step 1 "
              f"(FoldMatch {record['foldmatch_version']}). Run step 1 again with this FoldMatch."
        )
    seconds = split_run.check_dealt_ids(paths, record, preflight)
    logging.info(f"Ids of the {record['rows']} shard rows checked against {fasta_file} in {seconds:.1f} s")
    return paths, preflight, stat


def _log_split_build(record: dict, *, db_path: str, from_shards: Path, shards: int, codes: bool) -> None:
    """The start of ``build sequences-index``'s index phase: a WARNING when the run was embedded by another FoldMatch
    (D30) or its record was sealed, an INFO line when the database goes elsewhere than the recorded ``--output-db``
    (D28), then the run it is built from.

    The FoldMatch of the pass is the record's ``foldmatch_version`` for a record step 1 wrote. A sealed record holds
    the seal's version there, and for a plain run also in its pass record, which the seal filled from its own process;
    only residue-code shards record their pass's version (``pass.foldmatch_version``). For a sealed plain run that
    version is unknown, and the sealed WARNING says so."""
    run_id = record["run_id"]
    sealed = record["sealed"]
    if not sealed:
        pass_version = record["foldmatch_version"]
    elif record["residue_output"] is not None and isinstance(record["pass"].get("foldmatch_version"), str):
        pass_version = record["pass"]["foldmatch_version"]
    else:
        pass_version = None
    if pass_version is not None and pass_version != __version__:
        logging.warning(
            f"Run {run_id} was embedded by FoldMatch {pass_version}, and this is FoldMatch {__version__}: its record, "
            f"shard schemas and digests, pre-flight counts and ids were checked, and the database is built by this "
            f"FoldMatch."
        )
    if sealed:
        logging.warning(
            f"The record of run {run_id} was written by 'build sequences-seal' (FoldMatch "
            f"{record['foldmatch_version']}): it binds the shards to the FASTA by their entry count, rows and ids "
            f"(and, with residue codes, their seq_sha1), not by the pass that wrote them"
            + ("." if pass_version is not None else
               ", whose FoldMatch version is not recorded (shards without residue codes do not record it).")
        )
    if record["output_db"] is not None and record["output_db"] != os.path.abspath(db_path):
        logging.info(f"Building {db_path}, not the --output-db the run recorded ({record['output_db']})")
    if codes:
        built = f"residue codes of {record['system']['name']}"
    elif record["system"] is not None:
        built = f"its residue codes of {record['system']['name']} left out (--no-residue-codes)"
    else:
        built = "no residue codes"
    if sealed:
        when = (f"sealed {record['created_utc']} by FoldMatch {record['foldmatch_version']}; embedded by "
                + (f"FoldMatch {pass_version}" if pass_version is not None else "a FoldMatch it does not record"))
    else:
        when = f"embedded {record['created_utc']} by FoldMatch {record['foldmatch_version']}"
    logging.info(
        f"Building {db_path} from run {run_id} in {from_shards}: {record['rows']} entries in {shards} shard(s), "
        f"{when} on {record['world_size']} rank(s) ({record['pass']['accelerator'] or 'device not recorded'}), {built}"
        + ("; sealed record" if sealed else "")
    )


def _fasta_unchanged_guard(fasta_file: Path, stat, output_db: str):
    """The ``after_sequence_store`` hook of ``build sequences-index`` (I23): once the sequence store is built, the
    FASTA must still have ``stat``, taken before it was hashed. Otherwise the database files this build wrote are
    removed and the build is refused."""
    def guard() -> None:
        from foldmatch.search.split_run import fasta_stat
        try:
            now = fasta_stat(fasta_file)
            shown = _stat_text(now)
        except OSError as exc:
            now, shown = None, f"cannot be stat'ed ({exc})"
        if now == stat:
            return
        removed = _remove_outputs(output_db, _SEQUENCE_DB_SUFFIXES)
        raise RefusalError(
            f"The FASTA changed during this build ({fasta_file}: {_stat_text(stat)} when it was checked, now "
            f"{shown}), so the database may not hold the FASTA the run embedded. The database files it wrote were "
            f"removed ({', '.join(path.name for path in removed)})."
        )
    return guard


# The variables that set a BLAS runtime's threads; any of them set leaves BLAS to it (D29).
_BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


@contextmanager
def _whole_node_threads():
    """FAISS's OpenMP and the BLAS thread pools at the usable physical cores for the block (D29), restored on exit.

    FAISS gets ``faiss.omp_set_num_threads(n)`` unless ``OMP_NUM_THREADS`` is set; BLAS a ``threadpoolctl`` limit of
    ``n`` unless one of :data:`_BLAS_THREAD_VARS` is set, which reaches only the BLAS libraries threadpoolctl finds
    (the INFO line names them). Nothing is raised on macOS, where ``foldmatch`` pins ``OMP_NUM_THREADS=1`` against the
    second libomp. ``n`` is :func:`~foldmatch.search.code_alignment._physical_cpus` of the affinity mask, so call it
    after the mask was widened. Gains for ivf_pq (the k-means of its training and the encoding of its adds), not for
    flat, whose adds are copies after an L2 normalisation of 1000 rows per call (below the size FAISS parallelises);
    HNSW adds 100 vectors per call and scales poorly. With the faiss-cpu 1.15.1 wheel the OPQ step of an ivf_pq
    training does not gain: on a 128-core node it ran fastest at 4-16 threads and 2.6x slower at 128 (P9), and a
    30M-entry training took 76 min there at 128 threads against 40 min on a 64-core node at 64. D29 still gives the
    whole block ``n``.

    The limit also reaches the OpenMP OpenBLAS of the faiss-cpu 1.15.1 wheel (``libopenblaso``), which keeps no
    thread count of its own (OpenBLAS 0.3.15): ``openblas_set_num_threads`` sets the calling thread's OpenMP count,
    the one FAISS reads (FAISS and its OpenBLAS share an OpenMP runtime), and every BLAS call runs that count, lowered
    to the build's MAX_THREADS (128 in the x86_64 wheel) when it is larger. The limit gives it the ``n`` FAISS was just
    given, so FAISS keeps ``n``, at most MAX_THREADS (above it the INFO line still says ``n``, and the index phase warns
    of fewer threads than cores). On exit threadpoolctl sets every pool it lists back to its count at entry, in its
    listing order, which leaves that OpenMP count at OpenBLAS's (it lags FAISS's until a BLAS call); FAISS's previous
    count is restored after it. ``OPENBLAS_NUM_THREADS`` does not set that OpenBLAS's threads (FAISS's OpenMP count
    does), as the INFO line says when such a variable leaves BLAS alone.
    """
    import faiss
    from foldmatch.search.code_alignment import _physical_cpus
    n = _physical_cpus()
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:  # no affinity masks (macOS)
        allowed = os.cpu_count() or 1
    if sys.platform == "darwin":
        logging.info("FAISS/BLAS threads left as they are on macOS (foldmatch pins OMP_NUM_THREADS)")
        yield
        return
    raised, kept = [], []
    with ExitStack() as stack:
        if "OMP_NUM_THREADS" in os.environ:
            kept.append(f"FAISS threads left to OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
        else:
            previous = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(n)
            stack.callback(faiss.omp_set_num_threads, previous)
            raised.append("FAISS")
        blas_set = [f"{name}={os.environ[name]}" for name in _BLAS_THREAD_VARS if name in os.environ]
        limited = None
        if blas_set:
            follows = None
            if raised:  # FAISS was given n: an OpenMP BLAS of FAISS's runs FAISS's count, whatever the variable
                try:  # diagnostics only
                    import threadpoolctl
                    follows = _faiss_openmp_blas(threadpoolctl.threadpool_info())
                except Exception as exc:
                    logging.debug(f"threadpoolctl could not list the thread pools: {exc}")
            kept.append(f"BLAS threads left to {', '.join(blas_set)}" + (
                f", except FAISS's OpenMP OpenBLAS, {follows}, which runs FAISS's OpenMP threads" if follows else ""))
        else:
            import threadpoolctl
            stack.enter_context(threadpoolctl.threadpool_limits(limits=n, user_api="blas"))
            raised.append("BLAS")
            try:  # diagnostics only: the libraries the limit reached
                limited = sorted({os.path.basename(str(runtime.get("filepath")))
                                  for runtime in threadpoolctl.threadpool_info() if runtime.get("user_api") == "blas"})
            except Exception as exc:
                logging.debug(f"threadpoolctl could not list the thread pools: {exc}")
        set_to = [f"{'/'.join(raised)} threads set to {n} (physical cores of the {allowed} allowed CPUs)"]
        if limited is not None:
            set_to.append(f"BLAS libraries limited: {', '.join(limited)}" if limited else
                          "threadpoolctl found no BLAS library to limit")
        logging.info("; ".join((set_to if raised else []) + kept))
        yield


def _seal_run(fasta_file: Path, from_shards: Path, run_id: str, min_res_n: Optional[int]) -> dict:
    """The run record ``build sequences-seal`` writes for the shards of ``run_id``, which have none, once the checks
    that bind them to the FASTA have passed. Writes nothing.

    In this order: the run has shards (S4); their footers say what the pass was (S5, S6: :func:`_seal_footers`);
    then, between two stats of the FASTA, its sha256, its pre-flight with the run's ``min_res_n`` and the strictness
    of its mode (S7), the shards' rows and ranks, which must be the deal of the kept entries over the pass's ranks
    (S8), and the id of every shard row at its dealt position, with residue codes also its seq_sha1 against the
    entry's sequence (S9); the FASTA must not change meanwhile (S10). The record is sealed: it names no database and no
    pass options, and a plain run's pass record has no ESM3 revision.
    """
    from foldmatch.search import split_run
    from foldmatch.search.residue_shards import RESIDUE_CODES, run_shard_files
    paths = run_shard_files(from_shards, run_id)
    if not paths:
        runs = split_run._runs(from_shards)
        raise RefusalError(
            f"No embedding shards of run {run_id} (emb_{run_id}-<rank>.parquet) in {from_shards}, so there is nothing "
            f"to seal. It holds "
            + ("; ".join(f"run {rid} ({state.describe()})" for rid, state in runs.items()) if runs else "no run")
            + "."
        )
    system, min_res_n, world_size, run_pass = _seal_footers(paths, folder=from_shards, run_id=run_id,
                                                            min_res_n=min_res_n)
    codes = system is not None

    stat = split_run.fasta_stat(fasta_file)
    sha256, seconds = split_run.fasta_sha256(fasta_file)
    logging.info(f"FASTA sha256 of {fasta_file} ({stat.bytes} bytes) in {seconds:.1f} s")
    preflight = _fasta_preflight(fasta_file, min_res_n, strict=codes)
    _refuse_nothing_to_build(fasta_file, min_res_n, preflight)
    shards = split_run.shard_entries(paths)
    rows = sum(shard["rows"] for shard in shards)
    with _refusals(note="The shards are incomplete: run 'build sequences-embed' again"
                        + ("." if codes else ", or give the --min-res-n the run used.")):
        if rows != preflight.entries_kept:
            raise RefusalError(
                f"The {len(shards)} shard(s) of run {run_id} in {from_shards} hold {rows} row(s), but {fasta_file} has "
                f"{preflight.entries_kept} entries with at least {min_res_n} residues."
            )
        _check_shards_are_the_deal(shards, preflight.entries_kept, world_size, from_shards, run_id)
    seconds = split_run.check_dealt_ids(paths, {"run_id": run_id, "world_size": world_size}, preflight,
                                        sequences=codes)
    logging.info(f"Ids{' and seq_sha1' if codes else ''} of the {rows} shard rows checked against {fasta_file} in "
                 f"{seconds:.1f} s")
    try:
        now = split_run.fasta_stat(fasta_file)
        shown = _stat_text(now)
    except OSError as exc:
        now, shown = None, f"cannot be stat'ed ({exc})"
    if now != stat:
        raise RefusalError(
            f"The FASTA changed during the seal ({fasta_file}: {_stat_text(stat)} before it was hashed, now {shown}), "
            f"so the shards were not checked against one FASTA. No record was written."
        )
    return {
        "kind": split_run.RUN_RECORD_KIND,
        "record_version": split_run.RUN_RECORD_VERSION,
        "run_id": run_id,
        "command": split_run.SEAL_COMMAND,
        "sealed": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "foldmatch_version": __version__,
        "fasta": {"path": str(Path(fasta_file).resolve()), "bytes": stat.bytes, "mtime_ns": stat.mtime_ns,
                  "sha256": sha256},
        "min_res_n": min_res_n,
        "strict": codes,
        "preflight": split_run.preflight_counts(preflight),
        "output_db": None,
        "residue_output": RESIDUE_CODES if codes else None,
        "system": system,
        "world_size": world_size,
        "pass": run_pass,
        "options": {},
        "shards": shards,
        "rows": rows,
    }


def _seal_footers(paths: List[Path], *, folder: Path, run_id: str,
                  min_res_n: Optional[int]) -> Tuple[Optional[dict], int, int, dict]:
    """What the footers of a run's shards say about its pass, for ``build sequences-seal``: ``(system block or None,
    min_res_n, world size, pass record)``.

    Refused (S5): shards of more than one kind (plain and residue codes, or the codes of several systems); plain
    shards that are not legacy ones (:func:`~foldmatch.search.split_run.check_plain_footers`); residue shards that
    are not residue codes, or whose pass record (``foldmatch.pass``) is missing or lacks its ``min_res_n`` and world
    size. Refused (S6): a plain run without ``min_res_n``, which its shards do not record; for codes, shards that
    disagree on their pass's ``min_res_n`` or world size, a ``min_res_n`` other than theirs, or a world smaller than
    the shard count. A plain run's world is its shard count: a pass on W ranks writes min(W, N) shards of its N
    entries, and their deal over that many ranks is the same.
    """
    from foldmatch.search import split_run
    from foldmatch.search.residue_shards import ResidueShardError, pass_record, read_shard_info
    try:
        infos = [read_shard_info(path) for path in paths]
    except ResidueShardError as exc:
        raise RefusalError(str(exc)) from exc
    groups = {}
    for info in infos:
        groups.setdefault(info.key, []).append(info)
    if len(groups) > 1:
        raise RefusalError(
            f"The {len(infos)} shards of run {run_id} in {folder} are mixed: "
            + "; ".join(f"{len(group)} shard(s) with {group[0].describe()} (e.g. {group[0].path.name})"
                        for group in groups.values())
            + ". The shards of one run come from one pass, so they cannot be sealed as one run."
        )
    if infos[0].residue_output is None:
        split_run.check_plain_footers(paths)
        if min_res_n is None:
            raise RefusalError(
                f"The {len(infos)} shard(s) of run {run_id} in {folder} hold no residue codes, so they do not record "
                f"the --min-res-n of their pass: give the --min-res-n the run used."
            )
        return None, min_res_n, len(infos), pass_record(min_res_n=min_res_n, world_size=len(infos))

    unrecorded = [info.path.name for info in infos if info.pass_record is None]
    if unrecorded:
        raise RefusalError(
            f"{len(unrecorded)} of the {len(infos)} shard(s) of run {run_id} in {folder} record no pass "
            f"(foldmatch.pass): {', '.join(unrecorded[:10])}{', ...' if len(unrecorded) > 10 else ''}. Their "
            f"--min-res-n and world size are unknown, so they cannot be sealed: run 'build sequences-embed' again."
        )
    first = infos[0]
    for key in ("min_res_n", "world_size"):
        other = [info for info in infos if info.pass_record.get(key) != first.pass_record.get(key)]
        if other:
            raise RefusalError(
                f"The shards of run {run_id} in {folder} disagree on their pass's {key}: "
                f"{first.pass_record.get(key)!r} in {first.path.name}, {other[0].pass_record.get(key)!r} in "
                f"{other[0].path.name}{f' and {len(other) - 1} more' if len(other) > 1 else ''}. The shards of one run "
                f"come from one pass, so they cannot be sealed as one run."
            )
    shards_min_res_n, world_size = first.pass_record.get("min_res_n"), first.pass_record.get("world_size")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= floor
               for value, floor in ((shards_min_res_n, 0), (world_size, 1))):
        raise RefusalError(
            f"The shards of run {run_id} in {folder} record a pass without a valid --min-res-n and world size "
            f"(min_res_n {shards_min_res_n!r}, world_size {world_size!r}), so they cannot be sealed: run 'build "
            f"sequences-embed' again."
        )
    if min_res_n is not None and min_res_n != shards_min_res_n:
        raise RefusalError(
            f"--min-res-n is {min_res_n}, but the shards of run {run_id} in {folder} record a pass with --min-res-n "
            f"{shards_min_res_n}: leave --min-res-n out for a run with residue codes, or give the run's."
        )
    if world_size < len(infos):
        raise RefusalError(
            f"The shards of run {run_id} in {folder} record a pass on {world_size} rank(s), but there are "
            f"{len(infos)} of them: they are not the shards of one pass."
        )
    system = {"name": first.system, "digest": first.system_digest, "encoding_digest": first.encoding_digest}
    run_pass = split_run.check_codes_footers(paths, expect=system, min_res_n=shards_min_res_n, world_size=world_size)
    if None in system.values():
        raise RefusalError(
            f"The shards of run {run_id} in {folder} hold residue codes without their system's name and digests "
            f"({', '.join(f'{key} {value}' for key, value in system.items())}), so they cannot be sealed."
        )
    return system, shards_min_res_n, world_size, run_pass


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


def _code_store_build(run_id: str, shard_infos, *, from_shards: bool, split: bool = False):
    """``(pass record, manifest build block)`` of a code store built from a run's shards.

    The build block describes the ESM3 pass that computed the codes, from the
    shards' pass record (:func:`~foldmatch.search.residue_shards.pass_record`):
    the device type and its ESM3 precision, the ranks (``devices``), the
    encoder precision and the software versions of the pass. ``from_shards``
    tells whether the store was built by the pass's own command or later by
    ``build residue-codes --from-shards``; ``split`` adds ``"split": true``
    for the index phase of a split build (``build sequences-index``, D26),
    and the key is absent otherwise.
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
    if split:
        build["split"] = True
    return record, build


def _build_residue_code_store(embedding_computer, embedding_db, sequence_store, preflight, system, *,
                              min_res_n: int, load_batch_size: int, split: bool = False):
    """Rank 0, after the FAISS index and the sequence store: the residue-code store from this run's shards.

    ``create`` streams the shards' codes into the pending ``.rescodes.bin``
    and ``.rescodes.idx``; ``validate`` checks them against the sequence
    store, the pre-flight and the FAISS count, then moves them into place and
    writes the manifest last. The manifest's ``build`` block comes from the
    shards' pass record. A refusal removes only the code store and keeps the
    shards. ``split`` (``build sequences-index``): the pass ran in another
    process, so the manifest records the ESM3 revision of the shards' pass
    record, as ``build residue-codes --from-shards`` does; ``build
    sequences`` records this installation's pin, the same value under one
    installation.
    """
    from foldmatch.search.residue_code_store import ResidueCodeStore
    from foldmatch.search.residue_shards import RESIDUE_CODES, check_shards
    code_store = ResidueCodeStore(embedding_db.db_folder, embedding_db.index_name)
    kept = (f"This build's shards are kept in {embedding_computer.embedding_folder} "
            f"(run id {embedding_computer.run_id}).")
    with _refusals(note=kept):
        infos = check_shards(embedding_computer.shard_files(), RESIDUE_CODES, system)
        run_pass, build = _code_store_build(embedding_computer.run_id, infos, from_shards=False, split=split)
        revision = {"esm_revision": run_pass["esm_revision"]} if split else {}
        code_store.create(
            embedding_computer.iter_residue_codes(system, load_batch_size=load_batch_size),
            system, min_res_n=min_res_n, build=build, **revision,
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
    comp_bias_corr_scale=1.0, significance_mode=SignificanceMode.default, significance_sample_size=500,
))
ESM_CODES_QUERY_DEFAULTS = MappingProxyType(dict(min_seq_identity=0.0, min_coverage=0.0, max_evalue=1e-3))
#: The BLOSUM62-only options, refused under esm-codes whatever their value (D20: exit status 2).
_BLOSUM62_ONLY_OPTIONS = (("gap_open", "--gap-open"), ("gap_extend", "--gap-extend"),
                          ("comp_bias_corr", "--comp-bias-corr/--no-comp-bias-corr"),
                          ("comp_bias_corr_scale", "--comp-bias-corr-scale"),
                          ("significance_mode", "--significance-mode"),
                          ("significance_sample_size", "--significance-sample-size"))


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
    comp_bias_corr_scale: Optional[float] = None
    significance_mode: Optional[SignificanceMode] = None
    significance_sample_size: Optional[int] = None


def resolve_query_stage2_options(scoring=DEFAULT_SCORING, *, seq_identity: Optional[bool] = None,
                                 min_seq_identity: Optional[float] = None, min_coverage: Optional[float] = None,
                                 max_evalue: Optional[float] = None, gap_open: Optional[int] = None,
                                 gap_extend: Optional[int] = None, comp_bias_corr: Optional[bool] = None,
                                 comp_bias_corr_scale: Optional[float] = None,
                                 significance_mode: Optional[SignificanceMode] = None,
                                 significance_sample_size: Optional[int] = None) -> QueryStage2Options:
    """``query sequences``' Stage-2 options for ``scoring`` (P6.1): an option left out (None) takes the scoring's
    default (:data:`BLOSUM62_QUERY_DEFAULTS`, :data:`ESM_CODES_QUERY_DEFAULTS`), an explicit value is kept.

    Under esm-codes, every BLOSUM62-only option given (``--gap-open``, ``--gap-extend``,
    ``--comp-bias-corr/--no-comp-bias-corr``, ``--comp-bias-corr-scale``, ``--significance-mode``,
    ``--significance-sample-size``) and ``--no-seq-identity`` are refused, and so are filter values out of range:
    ``typer.BadParameter`` (exit status 2, D20), before anything is read. Under blosum62 nothing is checked here, as
    before.
    """
    try:
        scoring = ScoringChoice(scoring)
    except ValueError:
        raise typer.BadParameter(f"{scoring!r} is not a scoring; expected one of "
                                 f"{', '.join(c.value for c in ScoringChoice)}.", param_hint="--scoring") from None
    given = dict(min_seq_identity=min_seq_identity, min_coverage=min_coverage, max_evalue=max_evalue)
    blosum = dict(gap_open=gap_open, gap_extend=gap_extend, comp_bias_corr=comp_bias_corr,
                  comp_bias_corr_scale=comp_bias_corr_scale, significance_mode=significance_mode,
                  significance_sample_size=significance_sample_size)
    if scoring is ScoringChoice.blosum62:
        values = {name: BLOSUM62_QUERY_DEFAULTS[name] if value is None else value
                  for name, value in {**given, **blosum}.items()}
        if values["significance_mode"] is not None:
            values["significance_mode"] = SignificanceMode(values["significance_mode"])
        return QueryStage2Options(scoring=scoring, **values)
    refused = [flag for name, flag in _BLOSUM62_ONLY_OPTIONS if blosum[name] is not None]
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


def _output_fields(format_output: str) -> List[str]:
    """The ``--format-output`` fields of a ``query sequences`` Stage 2, parsed before ESM3: an empty list or an
    unknown field is refused with ``typer.BadParameter`` (exit status 2)."""
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


def query_encodings_path(embedding_computer) -> Path:
    """The per-run query-encoding memmap of an esm-codes query, next to the run's shards (deleted after the run)."""
    return Path(embedding_computer.embedding_folder) / f"query_encodings_{embedding_computer.run_id}.f16"


def _release_pass_models() -> None:
    """Rank 0, after the query pass: free the ESM3 model and the aggregator before Stage 2 (P6.3).

    Nothing holds them once the pass has returned but the reference cycles of its Lightning objects, so a collection
    frees them; the GPU allocators' caches are emptied too."""
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


@contextmanager
def _own_shard_removed_if_the_pass_fails(embedding_computer):
    """Around the query pass of an esm-codes query: if it fails, this process deletes the shard it wrote (its query
    embeddings and encodings, 128 B per residue), then the error goes on. Only a shard this run wrote: one that
    existed before the pass (an earlier run's, refused by the pass) is left alone. Every rank cleans its own shard;
    rank 0's ``finally`` removes the run's files once the pass has finished."""
    before = {path.name for path in embedding_computer.shard_files()}
    try:
        yield
    except BaseException:
        import torch.distributed as dist
        from foldmatch.utils.dist import launch_rank
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else launch_rank()
        own = Path(embedding_computer.embedding_folder) / f"{embedding_computer.out_name}-{rank}.parquet"
        if own.name not in before:
            try:
                own.unlink(missing_ok=True)
            except OSError as exc:
                logging.warning(f"Could not delete the query shard {own} of the failed pass: {exc}")
        raise


def _query_sequences_esm_codes(*, db_path: str, fasta_file: Path, tmp_embedding_folder: Path, output_file: str,
                               min_res_n: int, top_k: int, threshold: Optional[float], use_gpu_index: bool,
                               accelerator, num_nodes: int, devices, strategy, batch_size: int, num_workers: int,
                               nprobe: Optional[int], stage2: QueryStage2Options, align_workers: Optional[int],
                               format_output: str, format_mode: FormatMode) -> None:
    """``query sequences --scoring esm-codes`` (plan section 3, QUERY; P6.1-P6.3).

    Every rank, read-only, before ESM3: the remaining option checks (exit status 2), the strict query pre-flight
    (duplicate ids, '<' / '>' and non-ASCII sequences refused) and :func:`resolve_esm_codes_database` (exit status 1).
    The ESM3 pass writes each query's chain embedding, S5 encoding and seq_sha1 into the run's shards. Then rank 0:
    frees the models, loads the FAISS index and checks its count against the code store, builds the run's
    query-encoding memmap in shard order, searches (Stage 1), and aligns with the codes
    (:func:`~foldmatch.search.code_alignment.align_candidates_esm_codes`, N = the manifest's ``total_residues``).
    The run's shards and the memmap are deleted at the end, whatever happens (G7); a process whose pass fails deletes
    the shard it wrote.
    """
    if align_workers is not None and align_workers < 0:
        raise typer.BadParameter(f"must be >= 0, got {align_workers}.", param_hint="--align-workers")
    output_fields = _output_fields(format_output)
    with _refusals():
        preflight = _fasta_preflight(fasta_file, min_res_n, strict=True)
    database = resolve_esm_codes_database(db_path, stage2.max_evalue, num_nodes=num_nodes)

    from foldmatch.search.embedding_computer import EmbeddingComputer
    from foldmatch.search.residue_shards import QUERY_ENCODING
    with _refusals():
        embedding_computer = EmbeddingComputer(embedding_folder=tmp_embedding_folder)
        encodings_path = query_encodings_path(embedding_computer)
        if encodings_path.exists():
            raise FileExistsError(
                f"{encodings_path}, the query encodings of an earlier run with the same run id "
                f"({embedding_computer.run_id}), is in the way: remove it, or use another --tmp-embedding-folder.")
        with _own_shard_removed_if_the_pass_fails(embedding_computer):
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
                residue_output=QUERY_ENCODING,
                scoring_system=database.prepared,
            )

    if not _is_rank_zero():
        return

    query_index = None
    try:
        _release_pass_models()
        logging.info("Loading database...")
        from foldmatch.search.embedding_database import EmbeddingDatabase
        embedding_db = EmbeddingDatabase(db_path=db_path, use_gpu_for_search=use_gpu_index)
        embedding_db.load(nprobe=nprobe)
        with _refusals():
            database.code_store.check_faiss_count(len(embedding_db.db.chain_ids))
        query_index = embedding_computer.build_query_encodings(encodings_path, database.system)
        logging.info(f"Query encodings of {len(query_index)} sequence(s) ({query_index.rows} residues) in "
                     f"{encodings_path}")
        results = embedding_db.search_by_embeddings(
            embedding_batches=embedding_computer.get_embedding_batches(rows=True),
            top_k=top_k
        )
        # The shards are read rank by rank: the queries are aligned and written in input order, as by a 1-rank run.
        results = embedding_computer.in_input_order(results)
        # The query embeddings are only needed for this search.
        embedding_computer.remove_shards()
        logging.info(f"Searched {len(results)} sequence(s)")
        results = _filter_results_by_threshold(results, threshold)

        from foldmatch.search import output
        from foldmatch.search.code_alignment import CodeAlignmentPool, align_candidates_esm_codes
        # The sequences ESM3 encoded (the strict pre-flight refused non-ASCII ones), normalised like the stored
        # subjects; Stage 2 checks each against its encoding's length and seq_sha1.
        query_sequences = {name: sequence for name, sequence in embedding_computer.input_sequences().items()
                           if name in results}
        logging.info("Computing esm-codes alignments (Stage 2)...")
        with CodeAlignmentPool(align_workers) as pool:
            aligned = align_candidates_esm_codes(
                query_index,
                results,
                database.code_store,
                database.sequence_store.fetch,
                database.prepared,
                database.total_residues,
                pool,
                query_sequences=query_sequences,
                min_seq_identity=stage2.min_seq_identity,
                min_coverage=stage2.min_coverage,
                max_evalue=stage2.max_evalue,
                output_needs=output_fields,
            )
        output.write_aligned_results(aligned, output_fields, output_file, format_mode)
    finally:
        # QueryEncodingIndex.build removes its own file when it fails.
        if query_index is not None:
            query_index.remove()
        embedding_computer.remove_shards()


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
