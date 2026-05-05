import os
import typer

from typing import Annotated, List

from foldmatch import __version__
from foldmatch.cli.args_utils import arg_devices, set_log_level
from foldmatch.types.api_types import Accelerator, OutFormat, Strategy, LogLevel, ResEmbeddingFormat

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = typer.Typer(
    add_completion=False,
    help=f"RCSB Embedding Model CLI. Compute embeddings from protein sequences in FASTA files. Version: {__version__}."
)


def scan_fasta_sequences(fasta_file, res_embedding_location):
    """Build stream tuples (tensor_file_path, sequence_name) from a FASTA file and a residue embedding directory."""
    from foldmatch.dataset.esm_prot_from_fasta import parse_fasta
    sequences = parse_fasta(fasta_file)
    entries = []
    for name, _ in sequences:
        tensor_path = os.path.join(res_embedding_location, f"{name}.pt")
        entries.append((tensor_path, name))
    return tuple(entries)


@app.command(
    name="residue",
    help="Calculate residue level ESM embeddings from protein sequences in a FASTA file. Predictions are stored as torch tensor files or csv files."
)
def residue_embedding(
        fasta_file: Annotated[str, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences.'
        )],
        output_path: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as torch tensor files.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: separated (predictions are stored in single files) or grouped (predictions are stored in a single JSON file).'
        )] = OutFormat.separated,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings as a single JSON file. Used when output-format=grouped.'
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
        out_path=output_path,
        strategy=strategy,
        write_tensor=write_tensor
    )


@app.command(
    name="chain",
    help="Calculate chain level protein embeddings from sequences in a FASTA file using the aggregator model. Residue embeddings are computed first, then aggregated into chain embeddings."
)
def chain_embedding(
        fasta_file: Annotated[str, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='FASTA file containing protein sequences.'
        )],
        output_path: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions. Embeddings are stored as csv files.'
        )],
        res_embedding_folder: Annotated[str, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Path where residue level embeddings are stored.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: separated (predictions are stored in single files) or grouped (predictions are stored in a single JSON file).'
        )] = OutFormat.separated,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings as a single JSON file. Used when output-format=grouped.'
        )] = 'inference',
        min_res_n: Annotated[int, typer.Option(
            help='Consider only sequences with at least <min_res_n> residues.'
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
        )] = 'auto',
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
            help='If output-format=separated, write residue embeddings as torch tensor (.pt) files instead of csv files.'
        )] = False,
        log_level: Annotated[LogLevel, typer.Option(
            help='Logging level.'
        )] = 'info'
):
    from foldmatch.inference.sequence_inference import predict as sequence_predict
    from foldmatch.inference.chain_inference import predict as chain_predict
    from foldmatch.types.api_types import SrcLocation, SrcTensorFrom
    set_log_level(log_level)

    dev = arg_devices(devices)

    if compute_residue_embedding:
        sequence_predict(
            fasta_file=fasta_file,
            min_res_n=min_res_n,
            batch_size=batch_size_res,
            num_workers=num_workers_res,
            num_nodes=num_nodes_res,
            accelerator=accelerator,
            devices=dev,
            out_format=OutFormat.separated,
            out_path=res_embedding_folder,
            strategy=strategy,
            write_tensor=True
        )
        res_embedding_format = ResEmbeddingFormat.pt

    src_stream = scan_fasta_sequences(fasta_file, res_embedding_folder)
    chain_predict(
        src_stream=src_stream,
        src_location=SrcLocation.stream,
        src_from=SrcTensorFrom.file,
        batch_size=batch_size_aggregator,
        num_workers=num_workers_aggregator,
        num_nodes=num_nodes_aggregator,
        accelerator=accelerator,
        devices=dev,
        out_path=output_path,
        out_format=output_format,
        out_name=output_name,
        strategy=strategy,
        res_embedding_format=res_embedding_format,
        write_tensor=write_tensor
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
