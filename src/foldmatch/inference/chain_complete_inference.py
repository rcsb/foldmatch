import itertools
import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from lightning import Trainer

from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.utils import list_source, row_labels
from foldmatch.errors import MissingResidueCodesError, RefusalError
from foldmatch.inference.utils import is_distributed, list_inputs
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.search import scoring_system as _ss
from foldmatch.search.residue_shards import QUERY_ENCODING, RESIDUE_CODES, check_residue_output
from foldmatch.types.api_types import StructureFormat, Accelerator, Devices, Strategy, \
    SrcEsmFrom, FileOrStreamTuple, SrcLocation, OutFormat
from foldmatch.utils.data import identity_collate
from foldmatch.utils.dist import launch_rank, srun_step_id
from foldmatch.utils.model import get_residue_model, get_aggregator_model
from foldmatch.utils.output_files import ITEM_FORMATS, SHOWN, check_code_outputs, check_outputs, default_codes_name, \
    format_value
from foldmatch.utils.sequence import FastaPreflight, preflight_fasta, scan_fasta
from foldmatch.writer.batch_writer import ParquetBatchWriter, JsonStorage, TensorBatchWriter, CsvBatchWriter, \
    residue_code_tables, residue_code_writer


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
        out_codes_folder: Optional[Path] = None,
        out_codes_name: Optional[str] = None,
):
    """Chain embeddings of structures or FASTA sequences (ESM3 + the residue aggregator).

    ``residue_output`` (plan P5.2) also computes a per-residue payload in the same ESM3 pass:
    ``"residue_codes"`` (builds) or ``"query_encoding"`` (queries), with the encoders of ``scoring_system``. That is
    the system's name, its :class:`~foldmatch.search.scoring_system.EsmCodesSystem` (exactly the registry entry) or
    a :class:`~foldmatch.search.scoring_system.PreparedSystem` from :func:`prepare_scoring_system`; a build defaults
    to ``DEFAULT_ESM_CODES``, a query must name the database's system. Each prediction is then
    ``(chain_embeddings, ids, payloads, seq_sha1s)``, each payload the raw ``bytes`` that the shards store, and a
    Parquet output gets the residue columns (:class:`ParquetBatchWriter`). Refused before ESM3 loads, on every rank
    alike: structure input (the encoders were trained on sequence-only ESM3), an output other than Parquet, a system
    other than the registry entry of its name or a bundle that does not match it, and FASTA input the method cannot
    take (duplicate ids, ``<``/``>``, non-ASCII; :meth:`FastaPreflight.problems`). Any accelerator works, the Apple
    GPU (MPS) included (D18).

    ``out_codes_folder`` (fm-embedding's ``--output-codes-folder``) also computes each sequence's residue codes in
    the same ESM3 pass, the esm-codes target codes ``uint8 [L, 8]`` of ``scoring_system`` (``None``, the default,
    selects ``DEFAULT_ESM_CODES``, ``esm-codes-v1``: what ``fm-search build sequences --residue-codes`` stores), and
    writes them to that folder, which is not ``out_folder``, in ``out_format``: ``<id>.csv`` or ``<id>.pt``, else
    ``<out_codes_name>-<rank>.parquet`` or ``.json.gz`` (``out_codes_name`` by default ``<out_name>-codes``;
    :func:`residue_codes_pass`). The chain embeddings and their files are those of a run without it (the Parquet file
    the legacy ``id | embedding``), and ``out_folder`` holds embeddings only. Each prediction (``return_predictions``)
    is then ``(chain_embeddings, ids, codes, seq_sha1s)``: ``codes`` one CPU ``uint8`` tensor ``[L, 8]`` per id, the
    object that its ``.pt`` code file holds (``[0, 8]`` for a sequence without codes), and ``seq_sha1s`` the 20-byte
    SHA-1 of each normalised sequence. Not with ``residue_output``, the shard pass of a build or a query; refused
    before ESM3 loads as :func:`residue_codes_pass` says (an ``out_codes_folder`` that is ``out_folder`` is a
    ValueError; a code file that the call would replace and that holds embeddings is a RefusalError). A sequence
    whose codes cannot be computed gets an empty ``[0, 8]`` table, and the call raises
    :class:`~foldmatch.errors.MissingResidueCodesError` at the end, once every file is written
    (:meth:`ResidueCodesPass.check_after_pass`; with several ranks, rank 0 raises it for the run, once every rank has
    ended its pass), its ``predictions`` those this call would have returned. Not under a strategy that spawns or
    forks its ranks (``ddp_spawn``, ``ddp_fork``, or ``ddp_notebook``, Lightning's choice for several devices in a
    notebook): this process then does not learn the ranks' counts, so it logs a WARNING and raises nothing; each rank
    logs its sequences without codes as ERRORs.

    Without ``residue_output`` or ``out_codes_folder`` nothing changes: same module, same writer, same outputs byte
    for byte, same predictions ``(chain_embeddings, ids)``. Whatever the options, a file that the call would replace
    in ``out_folder`` and that holds fm-embedding's residue codes is refused before ESM3 (RefusalError;
    :func:`~foldmatch.utils.output_files.refuse_other_kind`).
    """
    logger = logging.getLogger(__name__)

    residue_output = check_residue_output(residue_output)
    if out_codes_folder is not None and residue_output is not None:
        raise ValueError(f"out_codes_folder writes the residue codes to files of their own, and residue_output "
                         f"({residue_output}) to the shards of a build or a query: pass one or the other")
    if out_codes_folder is not None and src_from != SrcEsmFrom.fasta:
        raise ValueError(_STRUCTURE_INPUT.format(what="residue codes"))
    residue_head = None
    prepared = None
    codes = None
    inputs = []
    outputs = None
    if src_from == SrcEsmFrom.fasta:
        # Before any model: the output name names files of the output folder, none of them is the FASTA file (P9.N;
        # the ids, and so the files of csv and pt output, are fm-embedding's to check), and none that the run would
        # replace holds residue codes (the ids tell those of csv and pt output)
        fasta = src_stream.path if isinstance(src_stream, FastaPreflight) else src_stream
        inputs = [("the FASTA file", fasta)] if isinstance(fasta, (str, os.PathLike)) else []
        ids = fasta_item_ids(fasta, min_res_n, out_format, out_folder, out_codes_folder)
        outputs = check_outputs(out_folder, out_format, out_name, source=str(fasta), inputs=inputs,
                                replaced_names=ids)
    if out_codes_folder is not None:
        codes = residue_codes_pass(src_stream, out_codes_folder, min_res_n=min_res_n, out_folder=out_folder,
                                   out_format=out_format, out_name=out_name, out_codes_name=out_codes_name,
                                   scoring_system=scoring_system, num_nodes=num_nodes, inputs=inputs,
                                   outputs=outputs, replaced_names=ids)
        src_stream = codes.preflight
    elif out_codes_name is not None:
        raise ValueError("out_codes_name names the residue-code files of out_codes_folder, which is not given")
    elif residue_output is not None:
        prepared = check_residue_pass(residue_output, scoring_system, src_from=src_from, out_folder=out_folder,
                                      out_format=out_format, num_nodes=num_nodes)
        src_stream = strict_fasta_preflight(src_stream, min_res_n)
        logger.info(f"{residue_output} with {prepared.system.name}: loading the residue-code head")
        residue_head = prepared.load_head(device=torch.device("cpu"))
    elif scoring_system is not None:
        raise ValueError("scoring_system is only used with residue_output or out_codes_folder, and neither is given")

    if out_folder is not None:
        if out_format == OutFormat.parquet:
            inference_writer = ParquetBatchWriter(out_folder, out_name) if residue_output is None else \
                residue_shard_writer(out_folder, out_name, residue_output, prepared.system, min_res_n=min_res_n)
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

    inference_set = EsmProtFromStructure(
        src_stream=src_stream,
        src_location=src_location,
        structure_format=structure_format,
        min_res_n=min_res_n
    ) if src_from == SrcEsmFrom.structure else EsmProtFromFasta(
        fasta_file=src_stream,
        min_res_n=min_res_n
    )

    if src_from == SrcEsmFrom.structure:
        # Before the models: each structure's outputs, <embedding name>.<chain ID>, name files of their own in
        # out_folder, and none an input (P9.N)
        data = inference_set.data.sort_index()  # in input order
        check_outputs(out_folder, out_format, out_name, prefixes=list(data.item_name), separator=".",
                      labels=row_labels(data), source=list_source(src_stream, src_location, "structure list"),
                      inputs=list_inputs(data, src_stream, src_location, list_name="structure list"))
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
    if codes is not None:
        module = ChainCompleteModule(
            res_model=res_model,
            aggregator_model=aggregator_model,
            residue_head=codes.head,
            residue_output=RESIDUE_CODES,
            code_files=True,
        )
    else:
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
        callbacks=callbacks or None,
        num_nodes=num_nodes,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False
    )

    logger.info(f"{src_from.name}-inference starts")
    if codes is not None:
        return codes.run(trainer, module, inference_dataloader, out_format, return_predictions)
    prediction = trainer.predict(
        module,
        inference_dataloader,
        return_predictions=return_predictions,
    )

    return prediction


def residue_shard_writer(out_folder, out_name: str, residue_output: str, system, *,
                         min_res_n: int) -> ParquetBatchWriter:
    """The Parquet shard writer of a residue pass (:class:`ParquetBatchWriter` with the residue columns). Its pass
    record (schema metadata ``foldmatch.pass``) holds ``min_res_n`` and the ESM3 revision the pass loads, the pinned
    one (``load_head`` has checked it is the system's ESM3); the writer adds the device and the ranks."""
    from foldmatch.utils.esm.loaders import ESM_REVISION
    return ParquetBatchWriter(out_folder, out_name, residue_output=residue_output, system=system,
                              min_res_n=min_res_n, esm_revision=ESM_REVISION)


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
        return _ss.resolve(scoring_system)
    if scoring_system is None:
        if residue_output == QUERY_ENCODING:
            raise ValueError("a query-encoding pass needs the scoring system of the database it searches (the one "
                             "its residue-code manifest names); there is no default")
        scoring_system = _ss.DEFAULT_ESM_CODES
    system = _ss.resolve_system(scoring_system)
    if local_files_only is None:
        local_files_only = not bundle_fetch_allowed(num_nodes)
    return _ss.resolve(system, local_files_only=local_files_only)


def check_residue_pass(residue_output, scoring_system=None, *, src_from=SrcEsmFrom.fasta, out_folder=None,
                       out_format=OutFormat.parquet, num_nodes: int = 1) -> "_ss.PreparedSystem":
    """The refusals of a residue pass that do not need the input (see :func:`predict`), then the prepared system."""
    residue_output = check_residue_output(residue_output)
    if residue_output is None:
        raise ValueError("check_residue_pass needs a residue output")
    if src_from != SrcEsmFrom.fasta:
        raise ValueError(_STRUCTURE_INPUT.format(what=residue_output))
    if out_folder is not None and out_format != OutFormat.parquet:
        raise ValueError(f"{residue_output} is written to Parquet shards only, not {OutFormat(out_format).value}")
    return prepare_scoring_system(scoring_system, residue_output=residue_output, num_nodes=num_nodes)


_STRUCTURE_INPUT = ("{what} needs sequence (FASTA) input: the residue-code encoders were trained on sequence-only "
                    "ESM3 embeddings, and structure input gives ESM3 coordinates")


@dataclass(frozen=True)
class ResidueCodesPass:
    """What an fm-embedding pass with residue codes (``--output-codes-folder``) has after its pre-flight
    (:func:`residue_codes_pass`): the strict FASTA pre-flight (the dataset reads its offsets), the prepared scoring
    system, its residue-code head, the codes folder, the codes name and ``min_res_n``."""
    preflight: FastaPreflight
    prepared: Any
    head: Any
    out_codes_folder: Any
    out_codes_name: str
    min_res_n: int

    def writer(self, out_format):
        """The writer of the codes in ``out_format`` to the codes folder
        (:func:`foldmatch.writer.batch_writer.residue_code_writer`): csv and pt ``<id>.<format>``; parquet
        ``<codes name>-<rank>.parquet`` with the system and the pass record (``min_res_n``, the pinned ESM3 revision
        that ``load_head`` checked; the writer adds the device and the ranks); json ``<codes name>-<rank>.json.gz``."""
        from foldmatch.utils.esm.loaders import ESM_REVISION
        return residue_code_writer(out_format, self.out_codes_folder, self.out_codes_name, self.prepared.system,
                                   min_res_n=self.min_res_n, esm_revision=ESM_REVISION)

    def run(self, trainer, module, dataloader, out_format, return_predictions):
        """``trainer.predict`` of the pass, its predictions with the codes as tensors (:func:`code_predictions`; the
        writers have stored the payloads during the pass), then :meth:`check_after_pass`; the predictions, None without
        ``return_predictions``. A rank whose pass raises tells the other ranks so before it re-raises, and counts as
        having read their reports (:func:`_post_report`), so that no rank waits for it at the end."""
        pass_id = next(_PASSES)
        try:
            predictions = trainer.predict(module, dataloader, return_predictions=return_predictions)
            predictions = code_predictions(predictions, self.prepared.system.code_bytes)
        except BaseException as exc:
            _post_report(trainer, pass_id, {"error": type(exc).__name__})
            raise
        self.check_after_pass(trainer, module, out_format, predictions, pass_id=pass_id)
        return predictions

    def check_after_pass(self, trainer, module, out_format, predictions=None, *, pass_id: Optional[int] = None):
        """After the pass, once this rank's writers have flushed (``trainer.predict`` has returned): the ranks tell
        each other how many of their sequences got no codes and their first ids (:func:`_rank_reports`, through the
        process group's store, not a collective of the backend; a rank waits there until every rank has ended its
        pass, so that no rank fails while another still writes, which its launcher would then kill). Then rank 0
        raises if any sequence of the run got no codes (:class:`~foldmatch.errors.MissingResidueCodesError`, naming
        how many, on which ranks, the first ids and where their empty tables are) or, with none, if another rank's pass
        ended with an error (a :class:`~foldmatch.errors.RefusalError`; that rank raises its own error). Only rank 0
        raises, so the run's exit status is non-zero whatever the launcher (Lightning's, srun, torchrun) and no
        launcher kills a rank that has not ended; the other ranks log their count as an ERROR and return. One rank:
        no store, the same error."""
        count = int(getattr(module, "residue_failures", 0))
        ids = list(getattr(module, "residue_failed_ids", ()))
        rank, world_size = int(getattr(trainer, "global_rank", 0)), int(getattr(trainer, "world_size", 1))
        reports = _rank_reports(trainer, {"count": count, "ids": ids[:_KEPT_IDS]},
                                next(_PASSES) if pass_id is None else pass_id)
        if reports is None:   # several ranks without a process group (ddp_spawn's parent): this rank's own
            logging.getLogger(__name__).warning(
                f"This process started its {world_size} ranks with a spawn or fork strategy and does not learn their "
                f"counts of sequences without residue codes: it raises no MissingResidueCodesError for them. Each "
                f"rank has logged its own as ERROR lines; their codes are empty [0, 8] tables.")
            reports, rank, world_size = [{"count": count, "ids": ids[:_KEPT_IDS]}], 0, 1
        counts = [int(report.get("count", 0)) for report in reports]
        total = sum(counts)
        errors = [r for r, report in enumerate(reports) if "error" in report]
        if not total and not errors:
            return
        if rank != 0:
            if count:
                logging.getLogger(__name__).error(
                    f"{count} sequence(s) on rank {rank} of {world_size} got no residue codes; rank 0 ends the run "
                    f"with an error that names them")
            return
        failed = ""
        if errors:
            failed = (f"rank(s) {_ranks(errors)} of {world_size} ended the pass with an error, which they report "
                      f"themselves; their files may be incomplete.")
        if not total:
            raise RefusalError(f"R{failed[1:]} The files of the other ranks are written.")
        run_ids = [i for report in reports for i in report.get("ids", ())][:_KEPT_IDS]
        shown = run_ids[:SHOWN]
        listed = ", ".join(repr(i) for i in shown) + (f" and {total - len(shown)} more" if total > len(shown) else "")
        on = [r for r, n in enumerate(counts) if n]
        if world_size <= 1:
            where_ranks = ""
        elif len(on) == 1:
            where_ranks = f" on rank {on[0]} of {world_size}"
        else:
            where_ranks = (f" on {len(on)} ranks of {world_size} (" +
                           ", ".join(f"rank {r}: {counts[r]}" for r in on) + ")")
        raise MissingResidueCodesError(
            f"{total} sequence(s){where_ranks} got no residue codes: {listed}. ESM3 gave rows that do not match the "
            f"sequence's length, or the encoding failed (each one's reason is logged above"
            f"{', by its rank' if world_size > 1 else ''}). Their codes are written as empty [0, 8] tables "
            f"({self._empty_tables(out_format, on)}); their embeddings and every other file of the run are written. "
            f"Check those sequences, or leave them out of the FASTA file." + (f" Also, {failed}" if failed else ""),
            count=total, ids=run_ids, rank=rank, world_size=world_size, rank_counts=counts, predictions=predictions)

    def _empty_tables(self, out_format, ranks) -> str:
        """Where the empty tables of the sequences without codes of ``ranks`` are, in ``out_format``."""
        fmt = format_value(out_format)
        folder = os.fspath(self.out_codes_folder)
        if fmt in ("parquet", "json"):
            suffix = ".parquet" if fmt == "parquet" else ".json.gz"
            files = [os.path.join(folder, f"{self.out_codes_name}-{rank}{suffix}") for rank in ranks]
            files = files[0] if len(files) == 1 else ", ".join(files[:-1]) + " and " + files[-1]
            return f"{'rows' if fmt == 'parquet' else 'records'} of {files} without residues"
        return f"{os.path.join(folder, f'<id>.{fmt}')} of each, " + ("an empty file" if fmt == "csv" else
                                                                     "a uint8 tensor [0, 8]")


def code_predictions(predictions, code_bytes: int = 8):
    """What ``predict`` returns from a pass with residue codes in files of their own: each prediction ``(embeddings,
    ids, payloads, seq_sha1s)`` of the pass as ``(embeddings, ids, codes, seq_sha1s)``, ``codes`` one CPU ``uint8``
    tensor ``[L, code_bytes]`` per id, the object that its ``.pt`` code file holds
    (:func:`~foldmatch.writer.batch_writer.residue_code_tables`, as the code writers read the payloads; ``[0,
    code_bytes]`` for a sequence without codes). None (no ``return_predictions``) stays None. Only this pass's
    predictions are converted: the shard pass of a build or a query (``residue_output``) returns its payloads as the
    ``bytes`` that the shards store."""
    if predictions is None:
        return None
    converted = []
    for prediction in predictions:
        if prediction is None:
            converted.append(None)
            continue
        _, tables, _ = residue_code_tables(prediction, code_bytes)
        converted.append((prediction[0], prediction[1], tuple(torch.from_numpy(t) for t in tables), prediction[3]))
    return converted


#: The ids of the sequences without codes that each rank tells rank 0 (MissingResidueCodesError.ids).
_KEPT_IDS = 10
#: How long a rank that has ended its pass waits for the other ranks (as long as the chain pass's DDP timeout).
RANK_WAIT = timedelta(days=1)
#: How long rank 0 waits for the other ranks to read the reports (they have them as soon as rank 0 has).
READ_WAIT = timedelta(minutes=10)
#: The passes of this process, numbered alike on every rank (every rank runs the same passes): their store keys.
_PASSES = itertools.count()


def _ranks(ranks) -> str:
    return ", ".join(str(r) for r in ranks)


def _default_store():
    """The store of the default process group (rendezvous; ``torch.distributed``), or None without one."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.distributed_c10d._get_default_store()
    except Exception:  # noqa: BLE001  (no store: no exchange)
        pass
    return None


def _report_key(pass_id: int, rank) -> str:
    return f"foldmatch/residue-codes/{pass_id}/{rank}"


def _post_report(trainer, pass_id: int, report: dict):
    """Tell the other ranks the report of a pass ``pass_id`` that raised on this rank ({"error"}), without waiting
    (best effort: none without a process group of several ranks). This rank never reads the others' reports, so it
    also counts as having read them (:func:`_count_read`): rank 0 does not wait :data:`READ_WAIT` for it."""
    world_size = int(getattr(trainer, "world_size", 1))
    store = _default_store() if world_size > 1 else None
    if store is None:
        return
    try:
        store.set(_report_key(pass_id, int(getattr(trainer, "global_rank", 0))), json.dumps(report))
        _count_read(store, pass_id, world_size)
    except Exception:  # noqa: BLE001  (the error being raised is what matters)
        pass


def _count_read(store, pass_id: int, world_size: int):
    """Count one more rank done with the reports of pass ``pass_id`` (read, or never to read them: its pass raised);
    the last one tells rank 0 (key ``all-read``)."""
    if store.add(_report_key(pass_id, "read"), 1) == world_size:
        store.set(_report_key(pass_id, "all-read"), "1")


def _duration(wait: timedelta) -> str:
    """``wait`` for a message: "1 day", "10 minutes", "200 ms"."""
    seconds = wait.total_seconds()
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)):
        if seconds >= size:
            n = seconds / size
            return f"{n:g} {unit}{'' if n == 1 else 's'}"
    return f"{seconds * 1000:g} ms"


def _rank_reports(trainer, report: dict, pass_id: int) -> Optional[list]:
    """Every rank's report of pass ``pass_id``, by rank: ``report`` is this rank's ({"count", "ids"}; a rank whose
    pass raised posts {"error"}). One rank: ``[report]``. Several: each rank sets its report in the store of the
    default process group (the rendezvous that every launcher sets up, here still up after the pass) and waits until
    every rank has set its own, at most :data:`RANK_WAIT`. Store operations only, no collective of the backend (no
    NCCL watchdog, no teardown to wait for); a rank that dies inside the pass gets the others killed by its launcher
    (Lightning's observer, torchrun's agent, srun with KillOnBadExit), or tells them (:meth:`ResidueCodesPass.run`,
    :func:`_post_report`; it then counts as having read the reports); rank 0's death closes the store where rank 0's
    process holds it (Lightning's launcher, srun), and the others' wait fails at once, with a RefusalError that says
    so. Rank 0 then waits until every rank has read the reports (or posted an error), at most :data:`READ_WAIT`, so
    that it does not end before them. None for several ranks without a process group."""
    world_size, rank = int(getattr(trainer, "world_size", 1)), int(getattr(trainer, "global_rank", 0))
    if world_size <= 1:
        return [report]
    store = _default_store()
    if store is None:
        return None
    keys = [_report_key(pass_id, r) for r in range(world_size)]
    try:
        store.set(keys[rank], json.dumps(report))
        store.wait(keys, RANK_WAIT)
        reports = [json.loads(store.get(key)) for key in keys]
        _count_read(store, pass_id, world_size)
    except Exception as exc:  # noqa: BLE001  (a store timeout, or rank 0's store gone)
        missing, reachable = [], True
        for r, key in enumerate(keys):
            try:
                if not store.check([key]):
                    missing.append(r)
            except Exception:  # noqa: BLE001
                reachable = False
                break
        if not reachable:
            why = "the process group's store could not be reached" + (
                " (rank 0's process holds it under Lightning's launcher and srun: rank 0 has most likely ended, with "
                "an error of its own)" if rank else "")
        elif missing:
            why = (f"rank(s) {_ranks(missing)} did not report the end of its pass (a rank waits for that at most "
                   f"{_duration(RANK_WAIT)})")
        else:
            why = "the reports could not be read"
        raise RefusalError(
            f"Rank {rank} of {world_size} ended its pass, with its files written, but could not learn whether the "
            f"other ranks' sequences got residue codes: {why}; {type(exc).__name__}: {exc}."
        ) from exc
    if rank == 0:
        try:
            store.wait([_report_key(pass_id, "all-read")], READ_WAIT)
        except Exception as exc:  # noqa: BLE001  (the reports are read: only the other ranks' reading is unsure)
            logging.getLogger(__name__).warning(f"rank 0 of {world_size}: not every rank read the residue-code "
                                                f"reports within {_duration(READ_WAIT)} ({type(exc).__name__}: "
                                                f"{exc})")
    return reports


def residue_codes_pass(src_stream, out_codes_folder, *, min_res_n: int = 0, out_folder=None,
                       out_format=OutFormat.csv, out_name: str = "inference", out_codes_name: Optional[str] = None,
                       scoring_system=None,
                       num_nodes: int = 1, src_from=SrcEsmFrom.fasta,
                       inputs: Iterable[Tuple[str, object]] = (), outputs=None,
                       replaced_names: Optional[Iterable[str]] = None) -> ResidueCodesPass:
    """The pre-flight of a pass with residue codes (``from-sequences residue`` and ``chain`` with
    ``--output-codes-folder``), before ESM3 loads, read-only and deterministic, so every rank refuses alike, in this
    order: structure input (the encoders were trained on sequence-only ESM3); the codes folder and the codes name
    (``out_codes_name``, by default :func:`default_codes_name` of ``out_name``;
    :func:`~foldmatch.utils.output_files.check_code_outputs`: the codes folder is not ``out_folder``, compared as
    files (a ValueError), and is an existing folder, the codes name names files of it, no input is a code file, no
    code file that the pass would replace holds embeddings (``replaced_names``, the ids of csv and pt output, tell
    those files: :func:`fasta_item_ids`), neither folder inside the other is named as one of the other's files; the
    ids of csv and pt output are fm-embedding's to check, as for the embeddings); the strict FASTA pre-flight of a
    residue pass (duplicate ids, ``<``/``>``, non-ASCII); the scoring system's bundle, fetched only where
    :func:`bundle_fetch_allowed` says so, else it must be cached (``download-models``); then its residue-code head, on
    the CPU (Lightning moves it with the module). ``inputs`` are the run's inputs (described, path), ``outputs`` the
    embedding files (:func:`check_outputs` of ``out_folder``)."""
    if src_from != SrcEsmFrom.fasta:
        raise ValueError(_STRUCTURE_INPUT.format(what="residue codes"))
    if out_codes_name is None:
        out_codes_name = default_codes_name(out_name)
    check_code_outputs(out_codes_folder, out_format, out_codes_name, out_folder=out_folder, inputs=inputs,
                       outputs=outputs, replaced_names=replaced_names)
    preflight = strict_fasta_preflight(src_stream, min_res_n)
    prepared = prepare_scoring_system(scoring_system, residue_output=RESIDUE_CODES, num_nodes=num_nodes)
    logging.getLogger(__name__).info(f"residue codes of {prepared.system.name}: loading the residue-code head")
    head = prepared.load_head(device=torch.device("cpu"))
    return ResidueCodesPass(preflight, prepared, head, out_codes_folder, out_codes_name, min_res_n)


def fasta_item_ids(fasta, min_res_n: int, out_format, *folders) -> Optional[List[str]]:
    """The ids of the records of a FASTA file that a pass keeps (at least ``min_res_n`` residues), which name its csv
    and pt files ``<id>.<format>`` in each of ``folders`` (its output folder, its codes folder), so that its checks
    know which files it would replace (one scan of the file); None for parquet and json, without a folder, or for
    input that is not a path."""
    if format_value(out_format) not in ITEM_FORMATS or all(folder is None for folder in folders) or \
            not isinstance(fasta, (str, os.PathLike)):
        return None
    return [name for _, name, length in scan_fasta(fasta) if min_res_n <= 0 or length >= min_res_n]


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
