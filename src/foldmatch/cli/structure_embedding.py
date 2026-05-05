import os
import sys
import logging
import typer

from typing import Annotated, List, Optional

from foldmatch import __version__
from foldmatch.cli.args_utils import arg_devices, set_log_level
from foldmatch.types.api_types import StructureFormat, Accelerator, SrcLocation, SrcProteinFrom, \
    SrcAssemblyFrom, SrcTensorFrom, OutFormat, Strategy, LogLevel, ResEmbeddingFormat

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STRUCTURE_FORMAT_EXTENSIONS = {
    StructureFormat.pdb: ('.pdb', '.pdb.gz'),
    StructureFormat.mmcif: ('.cif', '.cif.gz'),
    StructureFormat.bciff: ('.bcif', '.bcif.gz'),
}

app = typer.Typer(
    add_completion=False,
    help=f"RCSB Embedding Model CLI. Compute embeddings from a folder of structure files. Version: {__version__}."
)


def scan_structure_folder(folder_path, structure_format, file_extension=None):
    """Scan a folder for structure files and return stream tuples (name, path, name)."""
    if file_extension is not None:
        extensions = (file_extension,)
    else:
        extensions = STRUCTURE_FORMAT_EXTENSIONS.get(structure_format)
        if extensions is None:
            raise typer.BadParameter(f"Unknown structure format: {structure_format}")

    entries = []
    for filename in sorted(os.listdir(folder_path)):
        if any(filename.endswith(ext) for ext in extensions):
            file_path = os.path.join(folder_path, filename)
            name = filename
            for ext in extensions:
                if ext and name.endswith(ext):
                    name = name[:-len(ext)]
                    break
            entries.append((name, file_path, name))

    if not entries:
        raise typer.BadParameter(
            f"No structure files with extensions {extensions} found in {folder_path}"
        )

    return tuple(entries)


@app.command(
    name="residue",
    help="Calculate residue level embeddings of protein structures from a folder of structure files. Predictions are stored as torch tensor files or csv files."
)
def residue_embedding(
        src_folder: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing structure files. All chains in each structure will be processed.'
        )],
        output_path: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as csv or tensor files.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: separated (predictions are stored in single files) or grouped (predictions are stored in a single JSON file).'
        )] = OutFormat.separated,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings as a single JSON file. Used when output-format=grouped.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        structure_file_extension: Annotated[Optional[str], typer.Option(
            help='Override the default file extension used to filter structure files in src-folder. Pass an empty string to disable extension filtering and process every file in the folder. When unset, the defaults for the chosen structure-format are used.'
        )] = None,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only chains with more than <min_res_n> residues.'
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
        )] = 'auto',
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = tuple(['auto']),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = 'auto',
        write_tensor: Annotated[bool, typer.Option(
            help='If output-format=separated, write residue embeddings as torch tensor (.pt) files instead of csv files.'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = 'info'
):
    from foldmatch.inference.esm_inference import predict
    set_log_level(log_level)

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
        out_path=output_path,
        strategy=strategy,
        write_tensor=write_tensor
    )


@app.command(
    name="chain",
    help="Calculate single-chain protein embeddings from a folder of structure files using precomputed residue level embeddings. Predictions are stored as csv files."
)
def chain_embedding(
        src_folder: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing structure files. All chains in each structure will be processed.'
        )],
        output_path: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as csv or tensor files.'
        )],
        res_embedding_folder: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Path where residue level embeddings are located.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: separated (predictions are stored in single files) or grouped (predictions are stored in a single JSON file).'
        )] = OutFormat.separated,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings as a single JSON file. Used when output-format=grouped.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        structure_file_extension: Annotated[Optional[str], typer.Option(
            help='Override the default file extension used to filter structure files in src-folder. Pass an empty string to disable extension filtering and process every file in the folder. When unset, the defaults for the chosen structure-format are used.'
        )] = None,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only chains with more than <min_res_n> residues.'
        )] = 0,
        batch_size_res: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_res: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes_res: Annotated[int, typer.Option(
            help='Number of nodes to use for inference of residue embeddings.'
        )] = 1,
        batch_size_aggregator: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_aggregator: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes_aggregator: Annotated[int, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = 'auto',
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = tuple(['auto']),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = 'auto',
        compute_residue_embedding: Annotated[bool, typer.Option(
            help='Compute residue level embeddings as a first step. When enabled, residue embeddings are stored in res-embedding-location before computing chain embeddings.'
        )] = True,
        res_embedding_format: Annotated[ResEmbeddingFormat, typer.Option(
            help='Format of the precomputed residue embedding files read from res-embedding-location when compute-residue-embedding=False. Options: pt (torch tensor files) or csv.'
        )] = ResEmbeddingFormat.pt,
        write_tensor: Annotated[bool, typer.Option(
            help='If output-format=separated, write chain embeddings as torch tensor (.pt) files instead of csv files.'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = 'info'
):
    from foldmatch.inference.chain_inference import predict
    set_log_level(log_level)

    src_stream = scan_structure_folder(src_folder, structure_format, structure_file_extension)

    if compute_residue_embedding:
        from foldmatch.inference.esm_inference import predict as esm_predict
        esm_predict(
            src_stream=src_stream,
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=structure_format,
            min_res_n=min_res_n,
            batch_size=batch_size_res,
            num_workers=num_workers_res,
            num_nodes=num_nodes_res,
            accelerator=accelerator,
            devices=arg_devices(devices),
            out_format=OutFormat.separated,
            out_path=res_embedding_folder,
            strategy=strategy,
            write_tensor=True
        )
        res_embedding_format = ResEmbeddingFormat.pt

    predict(
        src_stream=src_stream,
        res_embedding_location=res_embedding_folder,
        src_location=SrcLocation.stream,
        src_from=SrcTensorFrom.structure,
        structure_format=structure_format,
        min_res_n=min_res_n,
        batch_size=batch_size_aggregator,
        num_workers=num_workers_aggregator,
        num_nodes=num_nodes_aggregator,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path,
        out_format=output_format,
        out_name=output_name,
        strategy=strategy,
        write_tensor=write_tensor,
        res_embedding_format=res_embedding_format
    )


@app.command(
    name="assembly",
    help="Calculate assembly embeddings from a folder of structure files using precomputed residue level embeddings. Predictions are stored as csv files."
)
def assembly_embedding(
        src_folder: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Folder containing structure files. All assemblies in each structure will be processed.'
        )],
        res_embedding_folder: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Path where residue level embeddings for single chains are located.'
        )],
        output_path: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as csv or tensor files.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: separated (predictions are stored in single files) or grouped (predictions are stored in a single JSON file).'
        )] = OutFormat.separated,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings as a single JSON file. Used when output-format=grouped.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        structure_file_extension: Annotated[Optional[str], typer.Option(
            help='Override the default file extension used to filter structure files in src-folder. Pass an empty string to disable extension filtering and process every file in the folder. When unset, the defaults for the chosen structure-format are used.'
        )] = None,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only assembly chains with more than <min_res_n> residues.'
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
        num_nodes_res: Annotated[int, typer.Option(
            help='Number of nodes to use for inference of residue embeddings.'
        )] = 1,
        batch_size_aggregator: Annotated[int, typer.Option(
            help='Number of samples processed together in one iteration.'
        )] = 1,
        num_workers_aggregator: Annotated[int, typer.Option(
            help='Number of subprocesses to use for data loading.'
        )] = 0,
        num_nodes_aggregator: Annotated[int, typer.Option(
            help='Number of nodes to use for inference of embeddings.'
        )] = 1,
        accelerator: Annotated[Accelerator, typer.Option(
            help='Device used for inference.'
        )] = 'auto',
        devices: Annotated[List[str], typer.Option(
            help='The devices to use. Can be set to a positive number or "auto". Repeat this argument to indicate multiple indices of devices. "auto" for automatic selection based on the chosen accelerator.'
        )] = tuple(['auto']),
        strategy: Annotated[Strategy, typer.Option(
            help='Lightning strategy to control distribution of inference.'
        )] = 'auto',
        compute_residue_embedding: Annotated[bool, typer.Option(
            help='Compute residue level embeddings as a first step. When enabled, residue embeddings are stored in res-embedding-location before computing assembly embeddings.'
        )] = True,
        res_embedding_format: Annotated[ResEmbeddingFormat, typer.Option(
            help='Format of the precomputed residue embedding files read from res-embedding-location when compute-residue-embedding=False. Options: pt (torch tensor files) or csv.'
        )] = ResEmbeddingFormat.pt,
        write_tensor: Annotated[bool, typer.Option(
            help='If output-format=separated, write assembly embeddings as torch tensor (.pt) files instead of csv files.'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = 'info'
):
    from foldmatch.inference.assembly_inferece import predict
    set_log_level(log_level)

    src_stream = scan_structure_folder(src_folder, structure_format, structure_file_extension)

    if compute_residue_embedding:
        from foldmatch.inference.esm_inference import predict as esm_predict
        esm_predict(
            src_stream=src_stream,
            src_location=SrcLocation.stream,
            src_from=SrcProteinFrom.structure,
            structure_format=structure_format,
            min_res_n=min_res_n,
            batch_size=batch_size_res,
            num_workers=num_workers_res,
            num_nodes=num_nodes_res,
            accelerator=accelerator,
            devices=arg_devices(devices),
            out_format=OutFormat.separated,
            out_path=res_embedding_folder,
            strategy=strategy,
            write_tensor=True
        )
        res_embedding_format = ResEmbeddingFormat.pt

    predict(
        src_stream=src_stream,
        res_embedding_location=res_embedding_folder,
        src_location=SrcLocation.stream,
        src_from=SrcAssemblyFrom.structure,
        structure_format=structure_format,
        min_res_n=min_res_n,
        max_res_n=max_res_n,
        batch_size=batch_size_aggregator,
        num_workers=num_workers_aggregator,
        num_nodes=num_nodes_aggregator,
        accelerator=accelerator,
        devices=arg_devices(devices),
        out_path=output_path,
        out_format=output_format,
        out_name=output_name,
        strategy=strategy,
        write_tensor=write_tensor,
        res_embedding_format=res_embedding_format
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
def main(
        version: bool = typer.Option(
            None,
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show the version and exit.",
        )
):
    pass

if __name__ == "__main__":
    app()
