import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.inference.utils import is_distributed
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.search import scoring_system as _ss
from foldmatch.search.residue_shards import QUERY_ENCODING, check_residue_output
from foldmatch.types.api_types import StructureFormat, Accelerator, Devices, Strategy, \
    SrcEsmFrom, FileOrStreamTuple, SrcLocation, OutFormat
from foldmatch.utils.data import identity_collate
from foldmatch.utils.dist import launch_rank, srun_step_id
from foldmatch.utils.model import get_residue_model, get_aggregator_model
from foldmatch.utils.sequence import FastaPreflight, preflight_fasta
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
        residue_output: Optional[str] = None,
        scoring_system=None,
):
    """Chain embeddings of structures or FASTA sequences (ESM3 + the residue aggregator).

    ``residue_output`` (plan P5.2) also computes a per-residue payload in the same ESM3 pass:
    ``"residue_codes"`` (builds) or ``"query_encoding"`` (queries), with the encoders of ``scoring_system``. That is
    the system's name, its :class:`~foldmatch.search.scoring_system.EsmCodesSystem` (exactly the registry entry) or
    a :class:`~foldmatch.search.scoring_system.PreparedSystem` from :func:`prepare_scoring_system`; a build defaults
    to ``DEFAULT_ESM_CODES``, a query must name the database's system. Each prediction is then
    ``(chain_embeddings, ids, payloads, seq_sha1s)`` and a Parquet output gets the residue columns
    (:class:`ParquetBatchWriter`). Refused before ESM3 loads, on every rank alike: structure input (the encoders
    were trained on sequence-only ESM3), an output other than Parquet, a system other than the registry entry of its
    name or a bundle that does not match it, and FASTA input the method cannot take (duplicate ids, ``<``/``>``,
    non-ASCII; :meth:`FastaPreflight.problems`). Any accelerator works, the Apple GPU (MPS) included (D18).

    Without ``residue_output`` nothing changes: same module, same writer, same outputs byte for byte.
    """
    logger = logging.getLogger(__name__)

    residue_output = check_residue_output(residue_output)
    residue_head = None
    prepared = None
    if residue_output is not None:
        prepared = check_residue_pass(residue_output, scoring_system, src_from=src_from, out_folder=out_folder,
                                      out_format=out_format, num_nodes=num_nodes)
        src_stream = strict_fasta_preflight(src_stream, min_res_n)
        logger.info(f"{residue_output} with {prepared.system.name}: loading the residue-code head")
        residue_head = prepared.load_head(device=torch.device("cpu"))
    elif scoring_system is not None:
        raise ValueError("scoring_system is only used with a residue_output")

    if out_folder is not None:
        if out_format == OutFormat.parquet:
            inference_writer = ParquetBatchWriter(out_folder, out_name) if residue_output is None else \
                ParquetBatchWriter(out_folder, out_name, residue_output=residue_output, system=prepared.system)
        elif out_format == OutFormat.json:
            inference_writer = JsonStorage(out_folder, out_name)
        elif out_format == OutFormat.pt:
            inference_writer = TensorBatchWriter(out_folder)
        else:
            inference_writer = CsvBatchWriter(out_folder)
    else:
        inference_writer = None

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
    module = ChainCompleteModule(
        res_model=res_model,
        aggregator_model=aggregator_model
    ) if residue_output is None else ChainCompleteModule(
        res_model=res_model,
        aggregator_model=aggregator_model,
        residue_head=residue_head,
        residue_output=residue_output,
    )
    logger.info(f"rcsb-esm + rcsb-aggregator module ready")

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


# =====================================================================================================================
# residue pass pre-flight: read-only and deterministic, so every rank refuses together, before ESM3
# =====================================================================================================================
def _env_int(name: str) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value.isdigit() else 0


def bundle_fetch_allowed(num_nodes: int = 1) -> bool:
    """Whether this process may download the esm-codes bundle, or must find it cached.

    Only the process that starts a run fetches it: rank 0 of a single-node run whose other ranks are started later
    by Lightning (they inherit the cache rank 0 filled, and check it cached-only). When the ranks are started
    independently (the tasks of a multi-task ``srun`` step, a ``torchrun`` world, or the nodes of a multi-node run),
    no rank fetches, so a bundle missing from the cache is refused by every rank alike instead of by all but one;
    run ``fm-embedding download-models`` first there.
    """
    if launch_rank() > 0 or num_nodes > 1:
        return False
    if srun_step_id() is not None and (_env_int("SLURM_STEP_NUM_TASKS") or _env_int("SLURM_NTASKS")) > 1:
        return False
    return _env_int("WORLD_SIZE") <= 1


def prepare_scoring_system(scoring_system=None, *, residue_output: str = "residue_codes",
                           local_files_only: Optional[bool] = None, num_nodes: int = 1) -> "_ss.PreparedSystem":
    """The checked bundle of the scoring system of a residue pass (:func:`scoring_system.prepare`).

    ``scoring_system`` is a name, an ``EsmCodesSystem`` (refused unless it is exactly the registry entry of its
    name) or an already ``PreparedSystem`` (re-checked against the registry, not fetched again). ``None`` is the
    default system for a build and refused for a query, which must use the database's system. The bundle is
    fetched only where :func:`bundle_fetch_allowed` says so (``local_files_only=None``); everywhere else it must be
    cached already, and the check reads and writes nothing else.
    """
    residue_output = check_residue_output(residue_output)
    if isinstance(scoring_system, _ss.PreparedSystem):
        _ss.registered(scoring_system.system)
        return scoring_system
    if scoring_system is None:
        if residue_output == QUERY_ENCODING:
            raise ValueError("a query-encoding pass needs the scoring system of the database it searches (the one "
                             "its residue-code manifest names); there is no default")
        system = _ss.get(_ss.DEFAULT_ESM_CODES)
    elif isinstance(scoring_system, str):
        system = _ss.get(scoring_system)
    else:
        system = _ss.registered(scoring_system)
    if local_files_only is None:
        local_files_only = not bundle_fetch_allowed(num_nodes)
    return _ss.prepare(system, local_files_only=local_files_only)


def check_residue_pass(residue_output, scoring_system=None, *, src_from=SrcEsmFrom.fasta, out_folder=None,
                       out_format=OutFormat.parquet, num_nodes: int = 1) -> "_ss.PreparedSystem":
    """The refusals of a residue pass that do not need the input (see :func:`predict`), then the prepared system."""
    residue_output = check_residue_output(residue_output)
    if residue_output is None:
        raise ValueError("check_residue_pass needs a residue output")
    if src_from != SrcEsmFrom.fasta:
        raise ValueError(f"{residue_output} needs sequence (FASTA) input: the residue-code encoders were trained on "
                         "sequence-only ESM3 embeddings, and structure input gives ESM3 coordinates")
    if out_folder is not None and out_format != OutFormat.parquet:
        raise ValueError(f"{residue_output} is written to Parquet shards only, not {OutFormat(out_format).value}")
    return prepare_scoring_system(scoring_system, residue_output=residue_output, num_nodes=num_nodes)


def strict_fasta_preflight(src_stream, min_res_n: int) -> FastaPreflight:
    """The FASTA pre-flight of a residue pass: ``src_stream`` (a path, or its :class:`FastaPreflight`) refused if it
    has duplicate ids, ``<``/``>`` or non-ASCII sequences, whose residues would not line up with stored codes."""
    if isinstance(src_stream, FastaPreflight):
        if src_stream.min_res_n != min_res_n:
            raise ValueError(f"Pre-flight of {src_stream.path} was made with min_res_n={src_stream.min_res_n}, "
                             f"not {min_res_n}")
        problems = src_stream.problems()
        if problems:
            raise ValueError(f"{src_stream.path}: " + "; ".join(problems) + ".")
        return src_stream
    if not isinstance(src_stream, (str, os.PathLike)):
        raise ValueError(f"a residue pass reads a FASTA file, not {type(src_stream).__name__}")
    preflight = preflight_fasta(Path(src_stream), min_res_n)
    preflight.check(strict=True)
    return preflight
