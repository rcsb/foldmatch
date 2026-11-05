import io
import os
import sys
from pathlib import Path

import typer

from typing import Annotated, List, Optional

from foldmatch import __version__
from foldmatch.cli.args_utils import DOWNLOAD_MODELS_HELP, DOWNLOAD_MODELS_SYSTEM_HELP, arg_devices, \
    download_models_command, set_log_level
from foldmatch.types.api_types import StructureFormat, Accelerator, SrcLocation, SrcProteinFrom, \
    SrcAssemblyFrom, SrcTensorFrom, OutFormat, Strategy, LogLevel, ResEmbeddingFormat
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#: The help of an assembly list option.
_ASSEMBLY_LIST_HELP = ('CSV file of 4 columns, no header: Structure Name | Structure File Path or URL | Assembly Id | '
                       'Embedding Name, <Structure Name>-<Assembly Id> when the 4th column is missing or empty. Each '
                       'row needs its own Embedding Name.')

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help=f"RCSB Embedding Model CLI. Version: {__version__}."
)


@app.command(
    name="residue-embedding",
    help="Calculate residue level embeddings of protein structures using ESM3. Predictions are stored in the format of output-format, torch tensor files by default."
)
def residue_embedding(
        src_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file of 4 columns, no header: Structure Name | Structure File Path or URL | Chain Id (label_asym_id in mmCIF and BinaryCIF files, the author chain ID in PDB files) | Embedding Name. The residue embedding of each row is stored under its Embedding Name, <Structure Name>.<Chain Id> when the 4th column is missing or empty.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions, in the format of output-format: one file per chain for csv and pt, one file per process for parquet and json.'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json. Parquet holds one row per chain, with one list of D values per residue.'
        )] = OutFormat.pt,
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings. Used when output-format is parquet or json.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
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
            help='Logging level.'
        )] = LogLevel.info
):
    from foldmatch.inference.esm_inference import predict
    set_log_level(log_level)

    predict(
        src_stream=src_file,
        src_location=SrcLocation.file,
        src_from=SrcProteinFrom.chain,
        structure_format=structure_format,
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
    help="Calculate single-chain protein embeddings from structural files. Predictions are stored as Parquet, one file per process: <output-name>-<rank>.parquet."
)
def structure_embedding(
        src_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file of 4 columns, no header: Structure Name | Structure File Path or URL | Chain Id (label_asym_id in mmCIF and BinaryCIF files, the author chain ID in PDB files) | Embedding Name. The chain embedding of each row is stored under its Embedding Name, <Structure Name>.<Chain Id> when the 4th column is missing or empty.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions: the Parquet file <output-name>-<rank>.parquet of each process.'
        )],
        output_name: Annotated[str, typer.Option(
            help='File name for storing embeddings: <output-name>-<rank>.parquet.'
        )] = 'inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
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
            help='Logging level.'
        )] = LogLevel.info
):
    from foldmatch.inference.structure_inference import predict
    set_log_level(log_level)

    predict(
        src_stream=src_file,
        src_location=SrcLocation.file,
        src_from=SrcProteinFrom.chain,
        structure_format=structure_format,
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
    help="Calculate single-chain protein embeddings from residue level embeddings stored as torch tensor or csv files (res-embedding-format). Predictions are stored in the format of output-format, csv files by default."
)
def chain_embedding(
        src_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file of 4 columns, no header, as for residue-embedding: Structure Name | Structure File Path or URL (not read) | Chain Id | Embedding Name, <Structure Name>.<Chain Id> when the 4th column is missing or empty. The residue embedding <Embedding Name>.<res-embedding-format> is read from res-embedding-location, and the chain embedding is stored under the Embedding Name.'
        )],
        output_folder: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store predictions, in the format of output-format: one file per chain for csv and pt, one file per process for parquet and json.'
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
        res_embedding_format: Annotated[ResEmbeddingFormat, typer.Option(
            help='Format of the residue level embedding files. Options: pt (torch tensor files) or csv.'
        )] = ResEmbeddingFormat.pt,
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
            help='Logging level.'
        )] = LogLevel.info
):
    from foldmatch.dataset.utils import CHAIN_LIST_DEFAULT_NAME, read_chain_list, refuse_repeated_names, row_labels
    from foldmatch.inference.chain_inference import predict
    from foldmatch.utils.output_files import check_outputs
    set_log_level(log_level)

    # Before any model: one name per row, one file per name, and no output over its residue embedding or the list
    chains = read_chain_list(src_file)
    refuse_repeated_names(chains, str(src_file), CHAIN_LIST_DEFAULT_NAME)
    _refuse_chains_over_residues(output_folder, output_format, res_embedding_location, res_embedding_format)
    check_outputs(output_folder, output_format, output_name, names=list(chains.item_name), labels=row_labels(chains),
                  source=str(src_file), inputs=[("the chain list", src_file)], stem_what="--output-name")
    # The residue embedding of each row is the file of its embedding name, as residue-embedding writes it
    predict(
        src_stream=[(os.path.join(res_embedding_location, f"{name}.{res_embedding_format.value}"), name)
                    for name in chains.item_name],
        res_embedding_location=res_embedding_location,
        src_location=SrcLocation.stream,
        src_from=SrcTensorFrom.file,
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
    help="Calculate assembly embeddings from residue level embeddings stored as torch tensor or csv files (res-embedding-format). Predictions are stored in the format of output-format, csv files by default."
)
def assembly_embedding(
        src_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help=_ASSEMBLY_LIST_HELP
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
            help='Output path to store predictions, in the format of output-format: one file per assembly for csv and pt, one file per process for parquet and json.'
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
            help='Logging level.'
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
    help="Calculate chain and assembly embeddings from structural files. Residue embeddings are stored as torch tensor files, chain and assembly embeddings in the format of output-format, csv files by default."
)
def complete_embedding(
        src_chain_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help='CSV file of 4 columns, no header, as for residue-embedding: Structure Name | Structure File Path or URL | Chain Id (label_asym_id in mmCIF and BinaryCIF files, the author chain ID in PDB files) | Embedding Name, which must be <Structure Name>.<Chain Id> or left out: the assembly step reads the residue embeddings under that name.'
        )],
        src_assembly_file: Annotated[Path, typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help=_ASSEMBLY_LIST_HELP
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
            help='Output path to store chain embeddings, in the format of output-format: one file per chain for csv and pt, one file per process for parquet and json (see output-chain-name).'
        )],
        output_assembly_path: Annotated[Path, typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help='Output path to store assembly embeddings, in the format of output-format: one file per assembly for csv and pt, one file per process for parquet and json (see output-assembly-name).'
        )],
        output_format: Annotated[OutFormat, typer.Option(
            help='Format of the output. Options: csv, pt, parquet, json.'
        )] = OutFormat.csv,
        output_chain_name: Annotated[str, typer.Option(
            help='File name for storing chain embeddings. Used when output-format is parquet or json.'
        )] = 'chain-inference',
        output_assembly_name: Annotated[str, typer.Option(
            help='File name for storing assembly embeddings. Used when output-format is parquet or json.'
        )] = 'assembly-inference',
        structure_format: Annotated[StructureFormat, typer.Option(
            help='Structure file format.'
        )] = StructureFormat.mmcif,
        min_res_n: Annotated[int, typer.Option(
            help='Consider only assembly chains with at least <min_res_n> residues.'
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
            help='Logging level.'
        )] = LogLevel.info
):
    from foldmatch.dataset.utils import read_assembly_list, refuse_shared_names
    # Before any model is built: the assembly step reads the residue embeddings as <structure name>.<chain ID>
    _refuse_other_embedding_names(src_chain_file)
    # The assembly list, read as the assembly step reads it: a row of more than 4 fields, or a structure name given to
    # two files, is refused here, before the residue step, rather than after the chain step. A list that can be read
    # once (a pipe, a named pipe or <(...) on macOS; /dev/stdin or /dev/fd/N, whose open shares the descriptor's offset
    # on macOS, even on a regular file; any path under /dev) is read once, here, and its text given to the assembly
    # step, so that it is checked as a list in a regular file is
    src_assembly_list = _read_once(src_assembly_file)
    assemblies = read_assembly_list(src_assembly_list)
    refuse_shared_names(assemblies, "stream_name", "stream")
    # One name per row, one file per name, no step's file over another's or over an input, before any model
    _refuse_conflicting_steps(src_chain_file, _read_once(src_assembly_list), assemblies, output_format=output_format,
                              folders=(output_res_path, output_chain_path, output_assembly_path),
                              names=(output_chain_name, output_assembly_name))
    residue_embedding(
        src_file=src_chain_file,
        output_folder=output_res_path,
        output_format=OutFormat.pt,
        structure_format=structure_format,
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
        batch_size=batch_size_chain,
        num_workers=num_workers_chain,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        log_level=log_level
    )
    assembly_embedding(
        src_file=_read_once(src_assembly_list),
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


class _ListText(io.StringIO):
    """The text of a list that can be read once, read into memory, under the name of the path it was read from (the
    name that refusals give it, and by which it is an input of the run when it is a file)."""

    def __init__(self, text: str, name: str):
        super().__init__(text)
        self.name = name


def _read_once(src_file):
    """A list path as it can be read more than once: a regular file not under /dev as it is; any other path (a pipe,
    /dev/stdin, /dev/fd/N, a path under /dev) read once, into a _ListText; a _ListText, a new one of its text."""
    if isinstance(src_file, _ListText):
        return _ListText(src_file.getvalue(), src_file.name)
    if os.path.isfile(src_file) and not os.fspath(src_file).startswith("/dev/"):
        return src_file
    # As pandas reads a list file: UTF-8, line ends kept
    with open(src_file, encoding="utf-8", newline="") as f:
        return _ListText(f.read(), os.fspath(src_file))


def _refuse_chains_over_residues(output_folder: Path, output_format: OutFormat, res_embedding_location: Path,
                                 res_embedding_format: ResEmbeddingFormat):
    """chain-embedding writes <embedding name>.<output-format> and reads <embedding name>.<res-embedding-format>: in
    one folder and one format, each chain embedding would replace the residue embedding it is computed from, which is
    refused (RefusalError), naming both options (P9.N D2)."""
    from foldmatch.errors import RefusalError
    from foldmatch.utils.output_files import same_folder
    if output_format.value == res_embedding_format.value and same_folder(output_folder, res_embedding_location):
        other = "csv" if output_format.value == "pt" else "pt"
        raise RefusalError(
            f"--output-folder and --res-embedding-location are one folder ({output_folder}), and --output-format is "
            f"--res-embedding-format ({output_format.value}): each chain embedding <embedding name>."
            f"{output_format.value} would replace the residue embedding it is computed from. Give another "
            f"--output-folder, or --output-format {other}, parquet or json."
        )


def _refuse_conflicting_steps(src_chain_file: Path, src_assembly_file, assemblies, *, output_format: OutFormat,
                              folders: tuple, names: tuple):
    """complete-embedding's checks of its three steps' files, before its residue step (P9.N): each step's as the step
    itself checks them (one name per row, one file per name, an assembly not named <structure name>.<...> beside the
    residue embeddings in their format), no output of any step over an input of the run (both lists and the structure
    files of both), and no file written by two steps in one folder (residue <res>/<chain name>.pt, chain
    <chain>/<chain name>.<format> or <chain>/<chain name>-*, assembly <assembly>/<assembly name>.<format> or
    <assembly>/<assembly name>-*). ``src_assembly_file`` is the assembly list's path, or its text (_ListText) when it
    can be read once, and ``assemblies`` its frame."""
    from foldmatch.dataset.utils import ASSEMBLY_LIST_DEFAULT_NAME, CHAIN_LIST_DEFAULT_NAME, list_source, \
        read_chain_list, refuse_repeated_names, row_labels
    from foldmatch.errors import RefusalError
    from foldmatch.inference.utils import list_inputs
    from foldmatch.utils.output_files import check_outputs, file_id, probe_folding, refuse_assemblies_over_residues, \
        same_folder, shown
    res_path, chain_path, assembly_path = folders
    chain_name, assembly_name = names
    foldings = {}

    def folding(folder):
        key = file_id(folder)
        if key not in foldings:
            foldings[key] = probe_folding(folder)
        return foldings[key]

    def of_list(frame, src, list_name):
        # The inputs of one list, its structure files named with the list, as both lists' are checked together
        return [(described if described == f"the {list_name}" else f"{described} of the {list_name}", path)
                for described, path in list_inputs(frame, src, SrcLocation.file, list_name=list_name)]

    chains = read_chain_list(src_chain_file)
    refuse_repeated_names(chains, str(src_chain_file), CHAIN_LIST_DEFAULT_NAME)
    chain_names, chain_rows = list(chains.item_name), row_labels(chains)
    assembly_source = list_source(src_assembly_file, SrcLocation.file, "assembly list")
    refuse_repeated_names(assemblies, assembly_source, ASSEMBLY_LIST_DEFAULT_NAME)
    assembly_names, assembly_rows = list(assemblies.item_name), row_labels(assemblies)
    # Every input of the run, which no step's output may replace: a list read by one step can be written over by
    # another's output before it is read (the assembly list by the residue or chain step), or after (the chain list)
    inputs = of_list(chains, src_chain_file, "chain list") + of_list(assemblies, src_assembly_file, "assembly list")
    # Chains that differ only as the folder does not tell apart (A and a of one structure): complete-embedding names
    # them <structure name>.<chain ID>, and its residue step writes pt files, so only another folder keeps them apart
    named = "complete-embedding names each chain <structure name>.<chain ID>"
    steps = [
        ("--output-res-path", "residue", res_path, check_outputs(
            res_path, OutFormat.pt, None, names=chain_names, labels=chain_rows, source=str(src_chain_file),
            inputs=inputs, folding=folding(res_path),
            folded_advice=f"give --output-res-path a folder of a file system that tells {{apart}} apart ({named}, "
                          f"and its residue step writes pt files)")),
        ("--output-chain-path", "chain", chain_path, check_outputs(
            chain_path, output_format, chain_name, names=chain_names, labels=chain_rows, source=str(src_chain_file),
            inputs=inputs, folding=folding(chain_path), stem_what="--output-chain-name",
            folded_advice=f"give --output-chain-path a folder of a file system that tells {{apart}} apart, or use "
                          f"--output-format parquet or json ({named})")),
        ("--output-assembly-path", "assembly", assembly_path, check_outputs(
            assembly_path, output_format, assembly_name, names=assembly_names, labels=assembly_rows,
            source=assembly_source, folding=folding(assembly_path), stem_what="--output-assembly-name",
            inputs=inputs)),
    ]
    refuse_assemblies_over_residues(assembly_names, assemblies.stream_name, out_folder=assembly_path,
                                    out_format=output_format, res_folder=res_path, res_format=OutFormat.pt,
                                    source=assembly_source, labels=assembly_rows)
    for k, (option, step, folder, outputs) in enumerate(steps):
        for other_option, other_step, other_folder, other_outputs in steps[k + 1:]:
            if not same_folder(folder, other_folder):
                continue
            shared = outputs.shared_with(other_outputs)
            if shared:
                hint = ("an --output-format other than pt (residue embeddings are pt files)" if step == "residue"
                        else "other names (--output-chain-name and --output-assembly-name for parquet and json, "
                             "the lists' embedding names for csv and pt)")
                raise RefusalError(
                    f"{option} and {other_option} are one folder ({folder}), in which complete-embedding's {step} and "
                    f"{other_step} embeddings would share {len(shared)} file(s): {shown(shared)}. Each step would "
                    f"replace the other's files: give the steps different folders, or {hint}."
                )


def _refuse_other_embedding_names(src_chain_file: Path):
    """complete-embedding's assembly step reads each chain's residue embedding as <structure name>.<chain ID>, as
    assembly-embedding does: a chain list row with another embedding name is refused (RefusalError), before anything is
    computed (such a list failed, or ran on the residue embeddings found under <structure name>.<chain ID>: the row's
    own, for whitespace at a line's start, an earlier run's, or another row's, as when two rows swap names)."""
    from foldmatch.dataset.utils import read_chain_list
    from foldmatch.errors import RefusalError
    chains = read_chain_list(src_chain_file)
    other = chains[chains.item_name != chains.stream_name + "." + chains.chain_id]
    if len(other) > 0:
        # Each field quoted, so that whitespace shows
        listed = "; ".join(f"{row.stream_name!r} chain {row.chain_id!r} named {row.item_name!r}"
                           for row in other.head(5).itertuples())
        more = f"; and {len(other) - 5} more rows" if len(other) > 5 else ""
        raise RefusalError(
            f"{src_chain_file}: {len(other)} row(s) with an embedding name other than <structure name>.<chain ID>: "
            f"{listed}{more}. The assembly step of complete-embedding reads each chain's residue embedding as "
            f"<structure name>.<chain ID>, as assembly-embedding does: name those rows so, or leave their 4th "
            f"column out."
        )


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
