import logging
from datetime import timedelta
from pathlib import Path

import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.inference.utils import is_distributed
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.types.api_types import StructureFormat, Accelerator, Devices, Strategy, \
    SrcEsmFrom, FileOrStreamTuple, SrcLocation, OutFormat
from foldmatch.utils.data import identity_collate
from foldmatch.utils.model import get_residue_model, get_aggregator_model
from foldmatch.writer.batch_writer import ParquetBatchWriter, JsonStorage, TensorBatchWriter, CsvBatchWriter


def predict(
        src_stream: FileOrStreamTuple,
        src_location: SrcLocation = SrcLocation.file,
        src_from: SrcEsmFrom = SrcEsmFrom.structure,
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
        three_di: str = None,
        three_di_output: str = "codes",
):
    """``three_di`` names a scoring system (e.g. ``'3di-v1-coded'``) whose target encoding
    is written alongside each embedding, from the same ESM3 pass. ``None`` - the default -
    is the behaviour this function has always had, byte for byte.
    """
    logger = logging.getLogger(__name__)

    # Validate the 3Di request before touching the filesystem or downloading ESM3: a bad
    # flag combination should not cost a multi-gigabyte download to discover.
    three_di_system = None
    if three_di is not None:
        from foldmatch.search import scoring_system
        if src_from == SrcEsmFrom.structure:
            # A structure database has coordinates, so its 3Di can be read from them
            # exactly rather than predicted - and measurably should be: on held-out CATH
            # pairs, predicted targets cost about 0.02-0.03 of detection at a fixed
            # E-value against exact ones, in both bands. Predicting here would spend a
            # forward pass to make the database worse.
            raise ValueError(
                "3Di prediction is for sequence-only databases; this build has "
                "structures, so encode their exact 3Di from coordinates instead")
        if out_format != OutFormat.parquet:
            raise ValueError(
                f"3Di codes are only written to Parquet shards, not {out_format.name}")
        three_di_system = scoring_system.get(three_di)
        if three_di_system.target_repr != "coded":
            raise ValueError(
                f"scoring system '{three_di}' does not encode predicted targets "
                f"(target_repr={three_di_system.target_repr!r})")

    inference_set = EsmProtFromStructure(
        src_stream=src_stream,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n
    ) if src_from == SrcEsmFrom.structure else EsmProtFromFasta(
        fasta_file=src_stream,
        min_res_n=min_res_n
    )

    logger.info(f"{src_from.name}-inference set running as iterator")

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
    three_di_head = three_di_scoring = None
    if three_di_system is not None:
        from foldmatch.search import scoring_system
        from foldmatch.utils.model import get_three_di_head
        system = three_di_system
        three_di_scoring = scoring_system.prepare(system)
        three_di_head, bundle = get_three_di_head(device=torch.device("cpu"))
        # The kernel's confidence buckets only mean anything at the temperature the head
        # was calibrated at; refuse rather than write codes that index the wrong rows.
        fitted = bundle["provenance"].get("profile_temperature")
        if fitted is not None and abs(fitted - system.profile_temperature) > 1e-9:
            raise ValueError(
                f"head was calibrated at T={fitted} but scoring system '{three_di}' "
                f"scores at T={system.profile_temperature}")
        logger.info(f"3Di head loaded ({bundle['provenance']['name']}), "
                    f"encoding under '{system.name}'")

    module = ChainCompleteModule(
        res_model=res_model,
        aggregator_model=aggregator_model,
        three_di_head=three_di_head,
        three_di_scoring=three_di_scoring,
        three_di_output=three_di_output
    )
    logger.info(f"rcsb-esm + rcsb-aggregator module ready")

    if out_folder is not None:
        if out_format == OutFormat.parquet:
            inference_writer = ParquetBatchWriter(
                out_folder, out_name, with_three_di=three_di is not None)
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

    logger.info(f"{src_from.name}-inference starts")
    prediction = trainer.predict(
        module,
        inference_dataloader,
        return_predictions=return_predictions,
    )

    return prediction
