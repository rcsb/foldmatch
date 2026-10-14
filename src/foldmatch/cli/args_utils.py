import logging
import os
import warnings
from pathlib import Path

from foldmatch.types.api_types import LogLevel


def arg_devices(devices):
    if len(devices) == 1:
        return devices[0] if devices[0] == "auto" else int(devices[0])
    return [int(x) for x in devices]


class RankFilter(logging.Filter):
    def filter(self, record):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
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


def set_log_level(level: LogLevel):
    handler = logging.StreamHandler()
    handler.addFilter(RankFilter())

    import lightning
    if level == 'info':
        warnings.filterwarnings("ignore")
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        logging.getLogger("lightning.fabric.utilities.distributed").setLevel(logging.ERROR)
        handler.setFormatter(logging.Formatter('[proc-%(rank)s] %(message)s'))
        logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    elif level == 'warning':
        handler.setFormatter(logging.Formatter(
            '> %(asctime)s - [proc-%(rank)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        ))
        logging.basicConfig(level=logging.WARN, handlers=[handler], force=True)
    elif level == 'debug':
        handler.setFormatter(logging.Formatter(
            '> %(asctime)s - [proc-%(rank)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        ))
        logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)


DOWNLOAD_MODELS_SYSTEM_HELP = (
    "Fetch only this esm-codes scoring system's bundle (repeatable). Default: the bundles of every registered "
    "system. ESM3 and the aggregator are always fetched.")


def download_models_command(systems=None):
    """The ``download-models`` commands: ESM3, the aggregator and the esm-codes bundles of every registered system
    (or of ``systems``), into the Hugging Face cache, sha256-verified, so later runs work with ``HF_HUB_OFFLINE=1``.
    Unknown system names are refused before anything is downloaded."""
    import typer
    from foldmatch.search import scoring_system
    from foldmatch.utils import model
    known = scoring_system.known_names()
    names = list(dict.fromkeys(systems)) if systems else list(known)
    for name in names:
        if name not in known:
            raise typer.BadParameter(f"unknown esm-codes scoring system {name!r}; this FoldMatch knows: "
                                     f"{', '.join(known) or 'none'}", param_hint="--system")
    model.get_residue_model()
    model.get_aggregator_model()
    for name, files in model.download_esm_codes_bundles(names).items():
        s = scoring_system.get(name)
        folder = Path(files[scoring_system.BUNDLE_JSON]).parent
        typer.echo(f"esm-codes bundle {name}: {s.bundle_repo}@{s.bundle_revision}, {len(files)} files, sha256 "
                   f"verified, in {folder}")
