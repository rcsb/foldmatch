import logging
import sys
from pathlib import Path

from foldmatch.dataset.resdiue_assembly_embedding_from_structure import ResidueAssemblyDatasetFromStructure
from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.types.api_types import FileOrStreamTuple, SrcLocation, Accelerator, Devices, Strategy, \
    StructureFormat, SrcAssemblyFrom, OutFormat, ResEmbeddingFormat
from foldmatch.inference.chain_inference import predict as chain_predict
from foldmatch.dataset.utils import list_source, row_labels
from foldmatch.inference.utils import list_inputs
from foldmatch.utils.output_files import check_outputs, refuse_assemblies_over_residues, \
    refuse_outputs_over_inputs, residue_files


def predict(
        src_stream: FileOrStreamTuple,
        res_embedding_location: Path,
        src_location: SrcLocation = SrcLocation.file,
        src_from: SrcAssemblyFrom = SrcAssemblyFrom.assembly,
        structure_format: StructureFormat = StructureFormat.mmcif,
        min_res_n: int = 0,
        max_res_n: int = sys.maxsize,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        strategy: Strategy = Strategy.auto,
        out_format: OutFormat = OutFormat.csv,
        out_name: str = 'inference',
        out_folder: Path = None,
        res_embedding_format: ResEmbeddingFormat = ResEmbeddingFormat.pt,
        return_predictions: bool = True,
):
    logger = logging.getLogger(__name__)

    inference_set = ResidueAssemblyEmbeddingFromTensorFile(
        src_stream=src_stream,
        res_embedding_location=res_embedding_location,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n,
        max_res_n=max_res_n,
        res_embedding_format=res_embedding_format
    ) if src_from == SrcAssemblyFrom.assembly else ResidueAssemblyDatasetFromStructure(
        src_stream=src_stream,
        res_embedding_location=res_embedding_location,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n,
        max_res_n=max_res_n,
        res_embedding_format=res_embedding_format
    )
    # Before the model: each assembly names a file of its own in out_folder, and none an input, its residue embeddings
    # included (P9.N). From a structure list, the reader's rows are its own, one per assembly
    data, listed = inference_set.data.sort_index(), src_from == SrcAssemblyFrom.assembly  # in input order
    list_name = "assembly list" if listed else "structure list"
    source = list_source(src_stream, src_location, list_name)
    labels = row_labels(data) if listed else None
    outputs = check_outputs(out_folder, out_format, out_name, names=list(data.item_name), labels=labels,
                            source=source, inputs=list_inputs(data, src_stream, src_location, rows=listed,
                                                              list_name=list_name))
    refuse_assemblies_over_residues(list(data.item_name), data.stream_name, out_folder=out_folder,
                                    out_format=out_format, res_folder=res_embedding_location,
                                    res_format=res_embedding_format, source=source, labels=labels)
    if outputs is not None and outputs.existing():
        # A residue embedding that is an output's file through a link from another folder: only a file that exists
        # can be (none in a new output folder, where the residue folder is not listed)
        refuse_outputs_over_inputs(outputs, residue_files(res_embedding_location, data.stream_name,
                                                          res_embedding_format))
    logger.info(f"assembly-inference set contains {len(inference_set)} samples")

    logger.info(f"Delegating assembly-inference to chain-inference method")
    return chain_predict(
        src_stream=src_stream,
        src_location=src_location,
        batch_size=batch_size,
        num_workers=num_workers,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        out_format=out_format,
        out_name=out_name,
        out_folder=out_folder,
        inference_set=inference_set,
        res_embedding_format=res_embedding_format,
        return_predictions=return_predictions,
    )