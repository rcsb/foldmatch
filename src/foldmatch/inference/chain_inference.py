import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.residue_embedding_from_parquet import ResidueEmbeddingFromParquet
from foldmatch.dataset.residue_embedding_from_structure import ResidueEmbeddingFromStructure
from foldmatch.dataset.residue_embedding_from_tensor_file import ResidueEmbeddingFromTensorFile
from foldmatch.dataset.utils import list_source, row_labels
from foldmatch.errors import RefusalError
from foldmatch.inference.utils import list_inputs, refuse_min_res_n
from foldmatch.modules.chain_module import ChainModule
from foldmatch.types.api_types import Accelerator, Devices, Strategy, FileOrStreamTuple, SrcLocation, \
    SrcTensorFrom, StructureFormat, OutFormat, ResEmbeddingFormat
from foldmatch.utils.data import collate_embeddings
from foldmatch.utils.model import get_aggregator_model
from foldmatch.utils.output_files import check_outputs, format_value, shown
from foldmatch.writer.batch_writer import CsvBatchWriter, TensorBatchWriter, ParquetBatchWriter, JsonStorage


def predict(
        src_stream: FileOrStreamTuple,
        res_embedding_location: Path = None,
        src_location: SrcLocation = SrcLocation.file,
        src_from: SrcTensorFrom = SrcTensorFrom.file,
        structure_format: StructureFormat = StructureFormat.mmcif,
        min_res_n: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        strategy: Strategy = Strategy.auto,
        out_format: OutFormat = OutFormat.csv,
        out_name: str = 'inference',
        out_folder: Path = None,
        inference_set=None,
        res_embedding_format: ResEmbeddingFormat = ResEmbeddingFormat.pt,
        *,
        return_predictions: bool = True,
):
    logger = logging.getLogger(__name__)

    if inference_set is not None or src_from != SrcTensorFrom.structure:
        refuse_min_res_n(min_res_n, "the chains of an inference_set or of residue embedding files or Parquet "
                                    "(src_from file or parquet)")
    if inference_set is None:
        if src_from == SrcTensorFrom.parquet:
            inference_set = ResidueEmbeddingFromParquet(
                parquet_path=src_stream
            )
            _check_parquet_outputs(inference_set, src_stream, out_folder, out_format, out_name)
        elif src_from == SrcTensorFrom.file:
            inference_set = ResidueEmbeddingFromTensorFile(
                src_stream=src_stream,
                src_location=src_location,
                res_embedding_format=res_embedding_format
            )
            listed = inference_set.data.sort_index()  # in input order
            check_outputs(out_folder, out_format, out_name, names=list(listed.item_name), labels=row_labels(listed),
                          source=list_source(src_stream, src_location, "residue file list"),
                          inputs=list_inputs(listed, src_stream, src_location,
                                             list_name="residue file list", column="file",
                                             what="residue embedding file"))
        else:
            inference_set = ResidueEmbeddingFromStructure(
                src_stream=src_stream,
                res_embedding_location=res_embedding_location,
                src_location=src_location,
                structure_format=structure_format,
                min_res_n=min_res_n,
                res_embedding_format=res_embedding_format
            )
            # The reader's own list of residue files, <structure name>.<chain ID>, named <embedding name>.<chain ID>
            check_outputs(out_folder, out_format, out_name, names=list(inference_set.data.item_name),
                          source=list_source(src_stream, src_location, "structure list"),
                          inputs=list_inputs(inference_set.data, src_stream, src_location, list_name="structure list",
                                             column="file", what="residue embedding file", rows=False))
    logger.info(f"chain-inference set contains {len(inference_set)} samples")

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_embeddings
    )

    logger.info(f"Loading rcsb-aggregator module")
    aggregator_model = get_aggregator_model(
        device=torch.device("cpu")
    )
    module = ChainModule(
        model=aggregator_model
    )
    logger.info(f"rcsb-aggregator module ready")

    if out_folder is not None:
        if out_format == OutFormat.parquet:
            inference_writer = ParquetBatchWriter(out_folder, out_name)
        elif out_format == OutFormat.json:
            inference_writer = JsonStorage(out_folder, out_name)
        elif out_format == OutFormat.pt:
            inference_writer = TensorBatchWriter(out_folder)
        else:
            inference_writer = CsvBatchWriter(out_folder)
    else:
        inference_writer = None
    trainer = Trainer(
        callbacks=[inference_writer] if inference_writer is not None else None,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False
    )

    logger.info(f"chain-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader,
        return_predictions=return_predictions,
    )

    return prediction


def _check_parquet_outputs(inference_set: ResidueEmbeddingFromParquet, src_stream, out_folder, out_format, out_name):
    """The checks of a run on residue embeddings in Parquet (P9.N), before the model: an id held by more than one row
    of the input, in any output format (each row's output is written under its id); then those of check_outputs, the
    input files among the inputs."""
    paths = [src_stream] if isinstance(src_stream, (str, os.PathLike)) else list(src_stream)
    seen, repeated = set(), []
    for chain_id in inference_set.ids:
        if chain_id in seen and chain_id not in repeated:
            repeated.append(chain_id)
        seen.add(chain_id)
    if repeated:
        raise RefusalError(
            f"The residue embeddings in Parquet ({', '.join(str(p) for p in paths)}): {len(repeated)} id(s) held by "
            f"more than one row: {shown([repr(i) for i in repeated])}. Each row's chain embedding is written under its "
            f"id, so a repeated id keeps one of them: give each chain its own id."
        )
    if out_folder is not None and format_value(out_format) in ("csv", "pt"):
        names = list(inference_set.ids)
    else:
        names = None
    check_outputs(out_folder, out_format, out_name, names=names, source="The residue embeddings in Parquet",
                  inputs=[("the residue embedding Parquet file", p) for p in paths])
