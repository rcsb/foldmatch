import logging
from datetime import timedelta
from pathlib import Path

import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_assembly_from_structure import EsmAssemblyFromStructure
from foldmatch.dataset.utils import list_source, row_labels
from foldmatch.inference.utils import is_distributed, list_inputs
from foldmatch.modules.assembly_complete_module import AssemblyCompleteModule
from foldmatch.types.api_types import StructureFormat, Accelerator, Devices, Strategy, \
    FileOrStreamTuple, SrcLocation, OutFormat
from foldmatch.utils.data import identity_collate
from foldmatch.utils.model import get_residue_model, get_aggregator_model
from foldmatch.utils.output_files import check_outputs
from foldmatch.writer.batch_writer import ParquetBatchWriter, JsonStorage, TensorBatchWriter, CsvBatchWriter


def predict(
        src_stream: FileOrStreamTuple,
        src_location: SrcLocation = SrcLocation.file,
        structure_format: StructureFormat = StructureFormat.mmcif,
        min_res_n: int = 0,
        max_res_n: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        num_nodes: int = 1,
        accelerator: Accelerator = Accelerator.auto,
        devices: Devices = 'auto',
        strategy: Strategy = Strategy.auto,
        out_format: OutFormat = OutFormat.csv,
        out_name: str = 'inference',
        out_folder: Path = None,
        return_predictions: bool = True,
):
    logger = logging.getLogger(__name__)

    inference_set = EsmAssemblyFromStructure(
        src_stream=src_stream,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n
    )

    # Before the models: each structure's outputs, <embedding name>-<assembly ID>, name files of their own in
    # out_folder, and none an input (P9.N)
    data = inference_set.data.sort_index()  # in input order
    check_outputs(out_folder, out_format, out_name, prefixes=list(data.item_name), separator="-",
                  labels=row_labels(data), source=list_source(src_stream, src_location, "structure list"),
                  inputs=list_inputs(data, src_stream, src_location, list_name="structure list"))
    logger.info("assembly-inference set running as iterator")

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=identity_collate
    )

    logger.info("Loading rcsb-esm + rcsb-aggregator module")
    res_model = get_residue_model(
        device=torch.device("cpu")
    )
    aggregator_model = get_aggregator_model(
        device=torch.device("cpu")
    )
    module = AssemblyCompleteModule(
        res_model=res_model,
        aggregator_model=aggregator_model,
        max_res_n=max_res_n
    )
    logger.info("rcsb-esm + rcsb-aggregator module ready")

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
    if strategy == Strategy.auto and is_distributed(devices, num_nodes):
        strategy = DDPStrategy(timeout=timedelta(days=1))
    trainer = Trainer(
        callbacks=[inference_writer] if inference_writer is not None else None,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False
    )

    logger.info("assembly-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader,
        return_predictions=return_predictions,
    )

    return prediction
