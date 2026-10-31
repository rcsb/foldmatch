import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.modules.esm_module import EsmModule
from foldmatch.types.api_types import StructureFormat, Accelerator, Devices, Strategy, \
    SrcProteinFrom, FileOrStreamTuple, SrcLocation, OutFormat
from foldmatch.utils.data import identity_collate
from foldmatch.utils.model import get_residue_model
from foldmatch.writer.batch_writer import TensorBatchWriter, CsvBatchWriter, ParquetBatchWriter, JsonStorage


def predict(
        src_stream: FileOrStreamTuple,
        src_location: SrcLocation = SrcLocation.file,
        src_from: SrcProteinFrom = SrcProteinFrom.chain,
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
        return_predictions: bool = True,
):
    logger = logging.getLogger(__name__)

    inference_set = EsmProtFromChain(
        src_stream=src_stream,
        src_location=src_location,
        structure_format=structure_format
    ) if src_from == SrcProteinFrom.chain else EsmProtFromStructure(
        src_stream=src_stream,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n
    )
    if hasattr(inference_set, '__len__'):
        logger.info(f"esm-inference set contains {len(inference_set)} samples")
    else:
        logger.info(f"esm-inference set running as iterator")

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=identity_collate
    )

    logger.info(f"Loading rcsb-esm module")
    esm_model = get_residue_model(
        device=torch.device("cpu")
    )
    module = EsmModule(
        model=esm_model
    )
    logger.info(f"rcsb-esm module ready")

    if out_folder is not None:
        if out_format == OutFormat.parquet:
            inference_writer = ParquetBatchWriter(out_folder, out_name, residue_embeddings=True)
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

    logger.info(f"esm-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader,
        return_predictions=return_predictions,
    )

    return prediction
