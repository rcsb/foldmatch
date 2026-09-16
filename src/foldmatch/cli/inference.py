import os
import sys
from pathlib import Path

import typer

from typing import Annotated, List

from foldmatch import __version__
from foldmatch.cli.args_utils import arg_devices, set_log_level
from foldmatch.types.api_types import StructureFormat, Accelerator, SrcLocation, SrcProteinFrom, \
    SrcAssemblyFrom, SrcTensorFrom, OutFormat, Strategy, LogLevel, ResEmbeddingFormat
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help=f"RCSB Embedding Model CLI. Version: {__version__}."
)


@app.command(
    name="residue-embedding",
    help="Calculate residue level embeddings of protein structures using ESM3. Predictions are stored as torch tensor files."
)
def residue_embedding(
        src_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 columns: Structure Name | Structure File Path or URL (switch structure-location) | Chain Id (asym_i for cif files) | Output Embedding Name.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as torch tensor files.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json.'
        )] = OutFormat.pt,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        min_res_n: Annotated[int, typer.Option(
            help='When using all chains in a structure, consider only chains with at least <min_res_n> residues.'
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
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    from foldmatch.inference.esm_inference import predict
    set_log_level(log_level)

    predict(
        src_stream=src_file,
        src_location=SrcLocation.file,
        src_from=SrcProteinFrom.chain,
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


@app.command(
    name="structure-embedding",
    help="Calculate single-chain protein embeddings from structural files. Predictions are stored in a single pandas DataFrame file."
)
def structure_embedding(
        src_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 columns: Structure Name | Structure File Path or URL (switch structure-location) | Chain Id (asym_i for cif files) | Output Embedding Name.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as a single DataFrame file (see out-df-name).'
        )],
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings as a single JSON file.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        min_res_n: Annotated[int, typer.Option(
            help='When using all chains in a structure, consider only chains with at least <min_res_n> residues.'
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
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    from foldmatch.inference.structure_inference import predict
    set_log_level(log_level)

    predict(
        src_stream=src_file,
        src_location=SrcLocation.file,
        src_from=SrcProteinFrom.chain,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_folder=output_folder,
        out_name=output_name,
        strategy=strategy,
        return_predictions=False,
    )


@app.command(
    name="chain-embedding",
    help="Calculate single-chain protein embeddings from residue level embeddings stored as torch tensor files. Predictions are stored as csv files."
)
def chain_embedding(
        src_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 columns: Structure Name | Structure File Path or URL (switch structure-location) | Chain Id (asym_i for cif files) | Output Embedding Name.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as csv files.'
        )],
        res_embedding_location: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Path where residue level embeddings are located.'
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
        res_embedding_format: Annotated[ResEmbeddingFormat, typer.Option(
            help='Format of the residue level embedding files. Options: pt (torch tensor files) or csv.'
        )] = ResEmbeddingFormat.pt,
        min_res_n: Annotated[int, typer.Option(
            help='When using all chains in a structure, consider only chains with at least <min_res_n> residues.'
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
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    from foldmatch.inference.chain_inference import predict
    from foldmatch.utils.data import adapt_csv_to_embedding_chain_stream
    set_log_level(log_level)

    predict(
        src_stream=adapt_csv_to_embedding_chain_stream(src_file, res_embedding_location, res_embedding_format=res_embedding_format),
        res_embedding_location=res_embedding_location,
        src_location=SrcLocation.stream,
        src_from=SrcTensorFrom.file,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_folder=output_folder,
        out_format=output_format,
        out_name=output_name,
        strategy=strategy,
        res_embedding_format=res_embedding_format,
        return_predictions=False,
    )

@app.command(
    name="assembly-embedding",
    help="Calculate assembly embeddings from residue level embeddings stored as torch tensor files. Predictions are stored as csv files."
)
def assembly_embedding(
        src_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 columns: Structure Name | Structure File Path or URL (switch structure-location) | Assembly Id | Output embedding name.'
        )],
        res_embedding_location: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Path where residue level embeddings for single chains are located.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as csv files.'
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
        res_embedding_format: Annotated[ResEmbeddingFormat, typer.Option(
            help='Format of the residue level embedding files. Options: pt (torch tensor files) or csv.'
        )] = ResEmbeddingFormat.pt,
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
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):
    from foldmatch.inference.assembly_inferece import predict
    set_log_level(log_level)

    predict(
        src_stream=src_file,
        res_embedding_location=res_embedding_location,
        src_location=SrcLocation.file,
        src_from=SrcAssemblyFrom.assembly,
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
        res_embedding_format=res_embedding_format,
        return_predictions=False,
    )

@app.command(
    name="complete-embedding",
    help="Calculate chain and assembly embeddings from structural files. Predictions are stored as csv files."
)
def complete_embedding(
        src_chain_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 columns: Structure Name | Structure File Path or URL (switch structure-location) | Chain Id (asym_i for cif files) | Output Embedding Name.'
        )],
        src_assembly_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file 4 columns: Structure Name | Structure File Path or URL (switch structure-location) | Assembly Id | Output embedding name.'
        )],
        output_res_path: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store residue embeddings. Residue embeddings are stored in separated files.'
        )],
        output_chain_path: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store chain embeddings. Embeddings are stored as a single JSON file (see output_chain_name).'
        )],
        output_assembly_path: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store assembly embeddings. Embeddings are stored as a single JSON file (see output_assembly_name).'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json.'
        )] = OutFormat.csv,
        output_chain_name: Annotated[str, typer.Option(
            help='File name for storing chain embeddings. Used when output-format is parquet or json.'
        )] = 'chain-inference',
        output_assembly_name: Annotated[str, typer.Option(
            help='File name for storing assembly embeddings. Used when output-format is parquet or json.'
        )] = 'chain-inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        min_res_n: Annotated[int, typer.Option(
            help='When using all chains in a structure, consider only chains with at least <min_res_n> residues.'
        )] = 0,
        max_res_n: Annotated[int, typer.Option(
            help='Stop adding assembly chains when number of residues is greater than <max_res_n> residues.'
        )] = sys.maxsize,
        batch_size_res: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_res: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        batch_size_chain: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_chain: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        batch_size_assembly: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_assembly: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes: Annotated[int, typer.Option(
            help='Number of nodes to use for inference.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = Accelerator.auto,
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = ['auto'],
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = Strategy.auto,
        log_level: Annotated[LogLevel, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = LogLevel.info
):

    residue_embedding(
        src_file=src_chain_file,
        output_folder=output_res_path,
        output_format=OutFormat.pt,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size_res,
        num_workers=num_workers_res,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        log_level=log_level
    )
    chain_embedding(
        src_file=src_chain_file,
        output_folder=output_chain_path,
        output_format=output_format,
        output_name=output_chain_name,
        res_embedding_location=output_res_path,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size_chain,
        num_workers=num_workers_chain,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        log_level=log_level
    )
    assembly_embedding(
        src_file=src_assembly_file,
        output_folder=output_assembly_path,
        output_format=output_format,
        output_name=output_assembly_name,
        res_embedding_location=output_res_path,
        structure_format=structure_format,
        min_res_n=min_res_n,
        max_res_n=max_res_n,
        batch_size=batch_size_assembly,
        num_workers=num_workers_assembly,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        log_level=log_level
    )

@app.command(
    name="download-models",
    help="Download models from huggingface and store them in the default location."
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
