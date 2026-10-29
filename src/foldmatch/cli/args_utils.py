import logging
import os
import sys
import warnings
from pathlib import Path

from foldmatch.types.api_types import LogLevel


def arg_devices(devices):
    if len(devices) == 1:
        return devices[0] if devices[0] == "auto" else int(devices[0])
    return [int(x) for x in devices]


class RankFilter(logging.Filter):
    """Sets ``record.rank``: the rank of torch.distributed's process group when one is initialised, else the
    launcher's ``RANK``, ``GLOBAL_RANK``, ``SLURM_PROCID`` or ``LOCAL_RANK``, else 0. torch.distributed is asked only
    when it is already imported: without it there is no process group, and importing it here would import torch at a
    command's first log record."""

    def filter(self, record):
        dist = sys.modules.get("torch.distributed")
        try:
            if dist is not None and dist.is_available() and dist.is_initialized():
                record.rank = str(dist.get_rank())
                return True
        except Exception:
            pass
        record.rank = (
            os.environ.get("RANK")
            or os.environ.get("GLOBAL_RANK")
            or os.environ.get("SLURM_PROCID")
            or os.environ.get("LOCAL_RANK")
            or "0"
        )
        return True


#: ``(level, has handlers)`` of the root logger when ``set_log_level(..., defer_lightning=True)`` first ran with
#: Lightning not imported: the state in which ``set_log_level`` imports Lightning when it is called again.
_ROOT_BEFORE_LIGHTNING = {}


def _import_lightning():
    """Import Lightning, as ``set_log_level`` does before it sets up logging.

    Lightning sets up its loggers when it is imported, from the state of the root logger: when it has no handler,
    'lightning.pytorch' and 'lightning.fabric' get handlers of their own and stop propagating, and other libraries it
    imports log as the root logger then allows (matplotlib's DEBUG records, say). After a ``set_log_level(...,
    defer_lightning=True)`` the import runs with the root logger's handlers and level as that call found them, so that
    it sets up the same loggers as before the call; when Lightning was imported meanwhile with the handler that call
    added, those two loggers get the handlers it would have given them."""
    before = _ROOT_BEFORE_LIGHTNING.pop("root", None)
    if "lightning" in sys.modules:
        if before is not None and not before[1]:
            for name in ("lightning.fabric", "lightning.pytorch"):
                logger = logging.getLogger(name)
                if not logger.handlers and logger.propagate:
                    logger.addHandler(logging.StreamHandler())
                    logger.propagate = False
        return
    if before is None:
        import lightning  # noqa: F401
        return
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        if not before[1]:
            for handler in handlers:
                root.removeHandler(handler)
        root.setLevel(before[0])
        import lightning  # noqa: F401
    finally:
        for handler in [h for h in root.handlers if h not in handlers]:  # as basicConfig(force=True) after it
            root.removeHandler(handler)
            handler.close()
        for handler in handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(level)


def set_log_level(level: LogLevel, *, defer_lightning: bool = False):
    """Set up logging for a command at ``level`` (``--log-level``): ``info``, ``warn`` or ``debug``, the values of
    :class:`LogLevel`. Any other value is refused with ``ValueError``.

    Lightning is imported first (:func:`_import_lightning`), so that the levels set here for its loggers, and at
    ``info`` the filter that ignores every warning, come after those its import sets. Its import also brings torch
    and (through torchmetrics and transformers) scikit-learn, which sets ``KMP_DUPLICATE_LIB_OK``: the commands rely
    on that state, as on torch being loaded before FAISS (on macOS a process that runs both, each with its own
    libomp, aborts or crashes otherwise).

    ``defer_lightning=True`` sets up the same logging without importing Lightning or anything that imports torch, for
    a command whose refusals should come within seconds of its launch (``build sequences-index`` and ``build
    sequences-seal``; the import takes minutes from a cold file system on a compute node). Such a command calls
    ``set_log_level(level)`` again before anything that imports torch, Lightning or FAISS (the seal imports none):
    that call imports Lightning as this one would have, then sets up logging again, so that the process ends in the
    state ``set_log_level(level)`` leaves.
    Meanwhile the Hugging Face progress bars are left as they are, unless huggingface_hub is already imported (no
    download comes before the second call)."""
    try:
        level = LogLevel(level)
    except ValueError:
        raise ValueError(f"unknown log level {level!r}: one of {', '.join(m.value for m in LogLevel)}") from None
    if not defer_lightning:
        _import_lightning()
    elif "lightning" not in sys.modules:
        root = logging.getLogger()
        _ROOT_BEFORE_LIGHTNING.setdefault("root", (root.level, root.hasHandlers()))
    handler = logging.StreamHandler()
    handler.addFilter(RankFilter())
    if level is LogLevel.info:
        warnings.filterwarnings("ignore")
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        logging.getLogger("lightning.fabric.utilities.distributed").setLevel(logging.ERROR)
        handler.setFormatter(logging.Formatter('[proc-%(rank)s] %(message)s'))
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
        if not defer_lightning or "huggingface_hub" in sys.modules:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
    elif level is LogLevel.warn:
        handler.setFormatter(logging.Formatter(
            '> %(asctime)s - [proc-%(rank)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        ))
        logging.basicConfig(level=logging.WARN, handlers=[handler], force=True)
    else:
        handler.setFormatter(logging.Formatter(
            '> %(asctime)s - [proc-%(rank)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        ))
        logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)


DOWNLOAD_MODELS_HELP = (
    "Download ESM3, the aggregator model and the esm-codes scoring bundles from Hugging Face into its cache and "
    "verify their sha256, so that later runs work offline (HF_HUB_OFFLINE=1). Download only: no model is loaded and "
    "no GPU is used.")

DOWNLOAD_MODELS_SYSTEM_HELP = (
    "Fetch only this esm-codes scoring system's bundle (repeatable). Default: the bundles of every registered "
    "system. ESM3 and the aggregator are always fetched.")


def download_models_command(systems=None):
    """The ``download-models`` command of ``fm-search``, ``fm-embedding`` and ``inference``: ESM3 (the pinned
    snapshot), the aggregator and the esm-codes bundles of every registered system (or of ``systems``), into the
    Hugging Face cache, sha256-verified, so later runs work with ``HF_HUB_OFFLINE=1``. Download only: no model is
    built, torch is not imported and no GPU is touched (a GPU login node is safe). Unknown system names are refused
    before anything is downloaded."""
    import typer
    from foldmatch.search import scoring_system
    from foldmatch.utils import model
    from foldmatch.utils.esm import loaders
    known = scoring_system.known_names()
    names = list(dict.fromkeys(systems)) if systems else list(known)
    for name in names:
        if name not in known:
            raise typer.BadParameter(f"unknown esm-codes scoring system {name!r}; this FoldMatch knows: "
                                     f"{', '.join(known) or 'none'}", param_hint="--system")
    snapshot = loaders.download_esm3_snapshot()
    typer.echo(f"ESM3: {loaders.ESM_REPO_ID}@{loaders.ESM_REVISION}, weights sha256 verified, in {snapshot}")
    aggregator = model.download_aggregator_weights()
    typer.echo(f"aggregator: {model.REPO_ID}@{model.REVISION}, sha256 verified, in {aggregator.parent}")
    for name, files in model.download_esm_codes_bundles(names).items():
        s = scoring_system.get(name)
        folder = Path(files[scoring_system.BUNDLE_JSON]).parent
        typer.echo(f"esm-codes bundle {name}: {s.bundle_repo}@{s.bundle_revision}, {len(files)} files, sha256 "
                   f"verified, in {folder}")
