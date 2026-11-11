import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.dataset.utils import list_source
from foldmatch.inference.utils import list_inputs, refuse_min_res_n
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.types.api_types import StructureFormat, Accelerator, Devices, Strategy, \
    SrcProteinFrom, FileOrStreamTuple, SrcLocation
from foldmatch.utils.data import identity_collate
from foldmatch.utils.model import get_residue_model, get_aggregator_model
from foldmatch.utils.output_files import check_outputs
from foldmatch.writer.batch_writer import ParquetBatchWriter


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
        out_name: str = 'inference',
        out_folder: Path = None,
        return_predictions: bool = True,
):
    logger = logging.getLogger(__name__)

    if src_from == SrcProteinFrom.chain:
        refuse_min_res_n(min_res_n, "the chains of a chain list (src_from chain)")
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
    # Before the models: the output name names files of the output folder, and none of them is an input (P9.N)
    list_name = "chain list" if src_from == SrcProteinFrom.chain else "structure list"
    if out_name is not None:
        check_outputs(out_folder, "parquet", out_name, source=list_source(src_stream, src_location, list_name),
                      inputs=list_inputs(inference_set.data, src_stream, src_location, list_name=list_name))
    logger.info(f"structure-inference set running as iterator")

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=identity_collate
    )

    logger.info(f"Loading rcsb-esm + rcsb-aggregator module")
    res_model = get_residue_model(
        device=torch.device("cpu")
    )
    aggregator_model = get_aggregator_model(
        device=torch.device("cpu")
    )
    module = ChainCompleteModule(
        res_model=res_model,
        aggregator_model=aggregator_model
    )
    logger.info(f"rcsb-esm + rcsb-aggregator module ready")

    inference_writer = ParquetBatchWriter(out_folder, out_name) if out_folder is not None and out_name is not None else None
    trainer = Trainer(
        callbacks=[inference_writer] if inference_writer is not None else None,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False
    )

    logger.info(f"structure-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader,
        return_predictions=return_predictions,
    )

    return prediction
