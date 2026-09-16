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
from pathlib import Path

import typer

from typing import Annotated, Optional, List

from foldmatch import __version__
from foldmatch.cli.args_utils import arg_devices, set_log_level
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
            help='Output folder to store predictions.'
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
            help='Output folder to store predictions.'
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
            help='Output folder to store predictions.'
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
            help='Output folder to store predictions.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json.'
        )] = OutFormat.csv,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
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
    from foldmatch.inference.sequence_inference import predict
    set_log_level(log_level)

    predict(
        fasta_file=fasta_file,
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
            help='Output folder to store predictions.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json.'
        )] = OutFormat.csv,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
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
    set_log_level(log_level)

    from foldmatch.inference.chain_complete_inference import predict
    predict(
        src_stream=fasta_file,
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
    )


# =====================================================================
# top-level
# =====================================================================

@app.command(
    name="download-models",
    help="Download ESM3 and aggregator models from Hugging Face."
)
def download_models():
    from foldmatch.utils.model import get_residue_model, get_aggregator_model
    get_residue_model()
    get_aggregator_model()


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
