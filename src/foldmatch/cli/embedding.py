"""Unified ``fm-embedding`` CLI for protein embedding inference.

Two subcommand groups reflect the input modality:

- ``from-structures``: residue / chain / assembly embeddings from a folder of
  mmCIF, binaryCIF, or PDB files (via ESM3-from-structure).
- ``from-sequences``: residue / chain embeddings from protein sequences in a
  FASTA file (via ESM3-from-sequence; no 3D structure required).

The split mirrors the two input modalities exposed by the inference layer.
Assembly-level embeddings are only available under ``from-structures`` since
the notion of an assembly is intrinsic to 3D structures.
"""
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import typer

from typing import Annotated, Optional, List, Tuple

from foldmatch import __version__
from foldmatch.cli.args_utils import DOWNLOAD_MODELS_HELP, DOWNLOAD_MODELS_SYSTEM_HELP, arg_devices, \
    download_models_command, set_log_level
from foldmatch.types.api_types import (
    Accelerator,
    LogLevel,
    OutFormat,
    SrcEsmFrom,
    SrcLocation,
    SrcProteinFrom,
    Strategy,
    StructureFormat,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_OUTPUT_FOLDER_HELP = (
    "Output folder to store predictions. A run that would replace a file of residue codes there (as 'fm-embedding "
    "from-sequences ... --output-codes-folder' writes them) is refused."
)
_OUTPUT_CODES_FOLDER_HELP = (
    "Also compute each sequence's residue codes in the same ESM3 pass and write them to this folder, which must exist "
    "and must not be --output-folder (a subfolder of it will do): the 8-byte per-residue ESM3 codes that 'fm-search "
    "build sequences --residue-codes' stores, 8 integers (0-255) per residue. They are written in --output-format, as "
    "the embeddings are: <id>.csv or <id>.pt per sequence, or <output-codes-name>-<rank>.parquet or .json.gz. The "
    "embedding files are those of a run without it, and --output-folder holds them only ('fm-search build embeddings' "
    "and 'query embedding' read it as before). Row i of a sequence's codes is its residue i, which is row i + 1 of the "
    "residue embeddings that 'from-sequences residue' writes (ESM3's BOS and EOS rows included). Refuses FASTA "
    "records whose sequence holds '<', '>' or non-ASCII characters. A sequence whose codes cannot be computed gets an "
    "empty table, and the run exits with status 1 once every file is written. A run that would replace an embedding "
    "file there is refused. The esm-codes bundle (~26 MB) is "
    "downloaded once; on several nodes, or under srun or torchrun, run 'fm-embedding download-models' first."
)
_OUTPUT_CODES_NAME_HELP = (
    "File name for storing residue codes, with --output-codes-folder. Used when output-format is parquet or json: "
    "<name>-<rank>.parquet or <name>-<rank>.json.gz (default <output-name>-codes)."
)

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help=f"FoldMatch embedding inference. Version: {__version__}.",
)

from_structures_app = typer.Typer(
    add_completion=False,
    help="Compute embeddings from a folder of 3D structure files.",
)
app.add_typer(from_structures_app, name="from-structures")

from_sequences_app = typer.Typer(
    add_completion=False,
    help="Compute embeddings from protein sequences in a FASTA file.",
)
app.add_typer(from_sequences_app, name="from-sequences")


# =====================================================================
# from-structures
# =====================================================================

@from_structures_app.command(
    name="residue",
    help="Calculate residue-level embeddings from a folder of structure files using ESM3."
)
def from_structures_residue(
        src_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing structure files. All chains in each structure will be processed.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_OUTPUT_FOLDER_HELP
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json. Parquet holds one row per chain, with one list of D values per residue.'
        )] = OutFormat.csv,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        structure_file_extension: Annotated[Optional[str], typer.Option(
            help='Override the default file extension used to filter structure files in src-folder. Pass an empty string to disable extension filtering and process every file in the folder. When unset, the defaults for the chosen structure-format are used.'
        )] = None,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only chains with at least <min_res_n> residues.'
        )] = 0,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices.'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    from foldmatch.inference.esm_inference import predict
    set_log_level(log_level)

    from foldmatch.utils.data import scan_structure_folder
    src_stream = scan_structure_folder(src_folder, structure_format, structure_file_extension)
    predict(
        src_stream=src_stream,
        src_location=SrcLocation.stream,
        src_from=SrcProteinFrom.structure,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_format=output_format,
        out_name=output_name,
        out_folder=output_folder,
        strategy=strategy,
        return_predictions=False,
    )


@from_structures_app.command(
    name="chain",
    help="Calculate chain-level embeddings from a folder of structure files."
)
def from_structures_chain(
        src_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing structure files. All chains in each structure will be processed.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_OUTPUT_FOLDER_HELP
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json.'
        )] = OutFormat.csv,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        structure_file_extension: Annotated[Optional[str], typer.Option(
            help='Override the default file extension used to filter structure files in src-folder.'
        )] = None,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only chains with at least <min_res_n> residues.'
        )] = 0,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
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
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    set_log_level(log_level)

    from foldmatch.utils.data import scan_structure_folder
    src_stream = scan_structure_folder(src_folder, structure_format, structure_file_extension)
    from foldmatch.inference.chain_complete_inference import predict
    predict(
        src_stream=src_stream,
        src_location=SrcLocation.stream,
        src_from=SrcEsmFrom.structure,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_format=output_format,
        out_folder=output_folder,
        out_name=output_name,
        strategy=strategy,
        return_predictions=False,
    )


@from_structures_app.command(
    name="assembly",
    help="Calculate assembly-level embeddings from a folder of structure files."
)
def from_structures_assembly(
        src_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing structure files. All assemblies in each structure will be processed.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_OUTPUT_FOLDER_HELP
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json.'
        )] = OutFormat.csv,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        structure_file_extension: Annotated[Optional[str], typer.Option(
            help='Override the default file extension used to filter structure files in src-folder.'
        )] = None,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only assembly chains with at least <min_res_n> residues.'
        )] = 0,
        max_res_n: Annotated[int, typer.Option(
            help='Stop adding assembly chains when number of residues is greater than <max_res_n> residues.'
        )] = sys.maxsize,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
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
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    from foldmatch.inference.assembly_complete_inference import predict
    set_log_level(log_level)

    from foldmatch.utils.data import scan_structure_folder
    src_stream = scan_structure_folder(src_folder, structure_format, structure_file_extension)
    predict(
        src_stream=src_stream,
        src_location=SrcLocation.stream,
        structure_format=structure_format,
        min_res_n=min_res_n,
        max_res_n=max_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_folder=output_folder,
        out_format=output_format,
        out_name=output_name,
        strategy=strategy,
        return_predictions=False,
    )


# =====================================================================
# from-sequences
# =====================================================================

@from_sequences_app.command(
    name="residue",
    help="Calculate residue-level embeddings from protein sequences in a FASTA file using ESM3."
)
def from_sequences_residue(
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_OUTPUT_FOLDER_HELP
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json. Parquet holds one row per sequence, with one list of D values per residue.'
        )] = OutFormat.csv,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
        output_codes_folder: Annotated[Optional[Path], typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_OUTPUT_CODES_FOLDER_HELP
        )] = None,
        output_codes_name: Annotated[Optional[str], typer.Option(
            help=_OUTPUT_CODES_NAME_HELP
        )] = None,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto".'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    codes = _codes_options(output_codes_folder, output_codes_name, output_folder, output_name)
    from foldmatch.inference.sequence_inference import predict
    set_log_level(log_level)

    with _refusals(codes is not None):
        predict(
            # Its pre-flight, which the dataset reads its records' offsets from
            fasta_file=_refuse_fasta_conflicts(fasta_file, output_folder, output_format, min_res_n, codes=codes),
            min_res_n=min_res_n,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            accelerator=accelerator,
            devices=arg_devices(devices),
            out_format=output_format,
            out_name=output_name,
            out_folder=output_folder,
            strategy=strategy,
            return_predictions=False,
            **_codes_keywords(codes),
        )


@from_sequences_app.command(
    name="chain",
    help="Calculate chain-level embeddings from protein sequences in a FASTA file."
)
def from_sequences_chain(
        fasta_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_OUTPUT_FOLDER_HELP
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json.'
        )] = OutFormat.csv,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
        output_codes_folder: Annotated[Optional[Path], typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help=_OUTPUT_CODES_FOLDER_HELP
        )] = None,
        output_codes_name: Annotated[Optional[str], typer.Option(
            help=_OUTPUT_CODES_NAME_HELP
        )] = None,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
        )] = 0,
        batch_size: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
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
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = LogLevel.info
):
    codes = _codes_options(output_codes_folder, output_codes_name, output_folder, output_name)
    set_log_level(log_level)

    from foldmatch.inference.chain_complete_inference import predict
    with _refusals(codes is not None):
        predict(
            # Its pre-flight, which the dataset reads its records' offsets from
            src_stream=_refuse_fasta_conflicts(fasta_file, output_folder, output_format, min_res_n, codes=codes),
            src_location=SrcLocation.file,
            src_from=SrcEsmFrom.fasta,
            min_res_n=min_res_n,
            batch_size=batch_size,
            num_workers=num_workers,
            num_nodes=num_nodes,
            accelerator=accelerator,
            devices=arg_devices(devices),
            out_format=output_format,
            out_folder=output_folder,
            out_name=output_name,
            strategy=strategy,
            return_predictions=False,
            **_codes_keywords(codes),
        )


def _codes_options(output_codes_folder: Optional[Path], output_codes_name: Optional[str], output_folder: Path,
                   output_name: str) -> Optional[Tuple[Path, str]]:
    """The codes folder and the codes name (--output-codes-name, else default_codes_name) of a run with residue codes,
    None without --output-codes-folder. Before anything is imported, usage errors (typer.BadParameter, exit status 2,
    D20): --output-codes-name without --output-codes-folder, and --output-codes-folder that is --output-folder,
    compared as files (same_folder_problem; the library raises the same message as a ValueError)."""
    if output_codes_folder is None:
        if output_codes_name is not None:
            raise typer.BadParameter("names the residue-code files of --output-codes-folder, which is not given.",
                                     param_hint="--output-codes-name")
        return None
    from foldmatch.utils.output_files import default_codes_name, same_folder_problem
    problem = same_folder_problem(output_folder, output_codes_folder, _CODES_OPTIONS)
    if problem is not None:
        raise typer.BadParameter(problem, param_hint="--output-codes-folder")
    return output_codes_folder, default_codes_name(output_name) if output_codes_name is None else output_codes_name


#: The folders of a run with residue codes, as its refusals name them.
_CODES_OPTIONS = ("--output-folder", "--output-codes-folder")


def _codes_keywords(codes: Optional[Tuple[Path, str]]) -> dict:
    """The residue-code keywords of predict: none without --output-codes-folder, so a plain run calls it as before."""
    return {} if codes is None else {"out_codes_folder": codes[0], "out_codes_name": codes[1]}


def _refusals(codes: bool):
    """A run with residue codes (--output-codes-folder) reports every refusal of its checks as a RefusalError (D20:
    exit status 1), as 'fm-search build sequences --residue-codes' does; a plain run raises exactly what it raised
    before."""
    return _as_refusals() if codes else nullcontext()


@contextmanager
def _as_refusals():
    from foldmatch.errors import RefusalError
    try:
        yield
    except RefusalError:
        raise
    except (ValueError, OSError) as exc:
        raise RefusalError(str(exc)) from exc


def _refuse_fasta_conflicts(fasta_file: Path, output_folder: Path, output_format: OutFormat, min_res_n: int,
                            codes: Optional[Tuple[Path, str]] = None):
    """The FASTA checks of from-sequences, before ESM3 (P9.N), on every rank alike: a record id repeated among the
    records kept (at least min_res_n residues), in every format, found by the residue pass's pre-flight
    (preflight_fasta, its duplicates); with csv and pt, ids that cannot name a file <id>.<format> of the output folder,
    or that the folder takes for one file (abc and ABC on APFS), the FASTA file among the output files, and residue
    codes in the files that the run would replace (refuse_other_kind). With ``codes`` (--output-codes-folder, the
    codes name) and csv or pt, the same of the code files <id>.<format> of the codes folder, as that folder compares
    names (embeddings in the files that the run would replace there), and neither folder inside the other named as
    one of the other's files (check_code_outputs). The files read are remembered for the library's check of the same
    folders (file_kinds.remembering). The pre-flight, whose offsets the dataset reads."""
    from foldmatch.errors import RefusalError
    from foldmatch.utils.file_kinds import remembering
    from foldmatch.utils.output_files import ITEM_FORMATS, check_code_outputs, check_outputs
    from foldmatch.utils.sequence import preflight_fasta, scan_fasta
    preflight = preflight_fasta(fasta_file, min_res_n)
    if preflight.duplicates:
        raise RefusalError(
            f"{fasta_file}: {preflight.problems()[0]}. fm-embedding writes each record's embedding under its id, the "
            f"first word of its header (the file <id>.csv or <id>.pt, or the id of its row in Parquet and JSON), so a "
            f"repeated id keeps one of them: give each record its own id."
        )
    if output_format.value in ITEM_FORMATS:
        ids, labels = [], []
        for k, (_, name, length) in enumerate(scan_fasta(fasta_file), start=1):
            if min_res_n <= 0 or length >= min_res_n:
                ids.append(name)
                labels.append(f"record {k}")
        inputs = [("the FASTA file", fasta_file)]
        with remembering():
            outputs = check_outputs(output_folder, output_format, None, names=ids, labels=labels,
                                    source=str(fasta_file), inputs=inputs, what="FASTA id", given="those records")
            if codes is not None:
                check_code_outputs(codes[0], output_format, codes[1], out_folder=output_folder, names=ids,
                                   labels=labels, source=str(fasta_file), inputs=inputs, what="FASTA id",
                                   given="those records", outputs=outputs, options=_CODES_OPTIONS)
    return preflight


# =====================================================================
# top-level
# =====================================================================

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


def main():
    """Entry point that renders expected errors as clean messages (no traceback)."""
    try:
        app()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
