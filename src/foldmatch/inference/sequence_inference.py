import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
from foldmatch.modules.esm_module import EsmModule
from foldmatch.search.residue_shards import RESIDUE_CODES
from foldmatch.types.api_types import Accelerator, Devices, Strategy, OutFormat
from foldmatch.utils.data import identity_collate
from foldmatch.utils.model import get_residue_model
from foldmatch.utils.output_files import check_outputs
from foldmatch.utils.sequence import FastaPreflight
from foldmatch.writer.batch_writer import TensorBatchWriter, CsvBatchWriter, ParquetBatchWriter, JsonStorage


def predict(
        fasta_file: Path,
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
        out_codes_folder: Optional[Path] = None,
        out_codes_name: Optional[str] = None,
        scoring_system=None,
):
    """Residue embeddings of FASTA sequences (ESM3): one ``[L + 2, 1536]`` table per sequence, BOS and EOS included.
    Each prediction (``return_predictions``) is ``(embeddings, ids)``.

    ``out_codes_folder`` (fm-embedding's ``--output-codes-folder``) also computes each sequence's residue codes in
    the same ESM3 pass, the esm-codes target codes ``uint8 [L, 8]`` that ``fm-search build sequences
    --residue-codes`` stores, and writes them to that folder, which is not ``out_folder``, in ``out_format``:
    ``<id>.csv`` or ``<id>.pt``, else ``<out_codes_name>-<rank>.parquet`` or ``.json.gz`` (``out_codes_name`` by
    default ``<out_name>-codes``; :func:`~foldmatch.inference.chain_complete_inference.residue_codes_pass`). 0-based
    row ``i`` of a sequence's codes is character ``i`` of the normalised sequence (residue ``i + 1``, counting from 1)
    and 0-based row ``i + 1`` of its embeddings. Each prediction is then ``(embeddings,
    ids, codes, seq_sha1s)``: ``codes`` one CPU ``uint8`` tensor ``[L, 8]`` per id, the object that its ``.pt`` code
    file holds (``[0, 8]`` for a sequence without codes), and ``seq_sha1s`` the 20-byte SHA-1 of each normalised
    sequence. The embeddings and their files are those of a run without it. An ``out_codes_folder`` that is
    ``out_folder`` (compared as files) is a ValueError; a code file that the call would replace there and that holds
    embeddings is a RefusalError, before ESM3, as is a residue-code file in ``out_folder`` with or without the codes
    (:func:`~foldmatch.utils.output_files.refuse_other_kind`). A sequence whose codes cannot be computed gets an
    empty ``[0, 8]`` table, and the call raises :class:`~foldmatch.errors.MissingResidueCodesError` at the end, once
    every file is written (with several ranks, rank 0 raises it for the run, once every rank has ended its pass), its
    ``predictions`` those this call would have returned. Not under a strategy that spawns or forks its ranks
    (``ddp_spawn``, ``ddp_fork``, or ``ddp_notebook``, Lightning's choice for several devices in a notebook): this
    process then does not learn the ranks' counts, so it logs a WARNING and raises nothing; each rank logs its
    sequences without codes as ERRORs.

    ``scoring_system`` is the scoring system whose encoders compute the codes, used only with ``out_codes_folder``
    (otherwise a ValueError): its name, its :class:`~foldmatch.search.scoring_system.EsmCodesSystem` or a
    :class:`~foldmatch.search.scoring_system.PreparedSystem`
    (:func:`~foldmatch.inference.chain_complete_inference.prepare_scoring_system`). ``None``, the default, selects
    ``DEFAULT_ESM_CODES``, ``esm-codes-v1``, as for ``chain_complete_inference.predict`` and the builds (fm-embedding
    has no option for it: it uses the default).
    """
    logger = logging.getLogger(__name__)

    # Before the model: the output name names files of the output folder, none of them is the FASTA file (P9.N; the
    # ids, and so the files of csv and pt output, are fm-embedding's to check), and none that the run would replace
    # holds residue codes (the ids tell those of csv and pt output)
    from foldmatch.inference.chain_complete_inference import fasta_item_ids, residue_codes_pass
    fasta = fasta_file.path if isinstance(fasta_file, FastaPreflight) else fasta_file
    inputs = [("the FASTA file", fasta)] if isinstance(fasta, (str, os.PathLike)) else []
    ids = fasta_item_ids(fasta, min_res_n, out_format, out_folder, out_codes_folder)
    outputs = check_outputs(out_folder, out_format, out_name, source=str(fasta), inputs=inputs, replaced_names=ids)
    codes = None
    if out_codes_folder is not None:
        codes = residue_codes_pass(fasta_file, out_codes_folder, min_res_n=min_res_n, out_folder=out_folder,
                                   out_format=out_format, out_name=out_name, out_codes_name=out_codes_name,
                                   scoring_system=scoring_system, num_nodes=num_nodes, inputs=inputs,
                                   outputs=outputs, replaced_names=ids)
        fasta_file = codes.preflight
    elif out_codes_name is not None or scoring_system is not None:
        raise ValueError("out_codes_name and scoring_system are only used with out_codes_folder, which is not given")
    inference_set = EsmProtFromFasta(
        fasta_file=fasta_file,
        min_res_n=min_res_n
    )
    logger.info(f"sequence-inference set contains {len(inference_set)} samples")

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
    ) if codes is None else EsmModule(
        model=esm_model,
        residue_head=codes.head,
        residue_output=RESIDUE_CODES,
        code_files=True,
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
    callbacks = [inference_writer] if inference_writer is not None else []
    if codes is not None:
        callbacks.append(codes.writer(out_format))
    trainer = Trainer(
        callbacks=callbacks or None,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False
    )

    logger.info(f"sequence-inference starts")
    if codes is not None:
        return codes.run(trainer, module, inference_dataloader, out_format, return_predictions)
    prediction = trainer.predict(
        module,
        inference_dataloader,
        return_predictions=return_predictions,
    )

    return prediction
