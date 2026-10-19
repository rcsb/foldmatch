"""Model downloads from Hugging Face: the aggregator, ESM3 and the esm-codes scoring bundles.

torch and ESM3 are imported only by the functions that build models, so the esm-codes bundle loader
(:func:`get_esm_codes_bundle`) and the download-only functions behind ``download-models``
(:func:`download_aggregator_weights`, :func:`foldmatch.utils.esm.loaders.download_esm3_snapshot`,
:func:`download_esm_codes_bundles`) work in processes without torch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional

from huggingface_hub import constants as hf_constants, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError,
)

from foldmatch.search import scoring_system
from foldmatch.search.scoring_system import BundleMismatchError, BundleUnavailableError, EsmCodesSystem

REPO_ID = "rcsb/rcsb-embedding-model"
FILE_NAME = "rcsb-embedding-model.pt"
REVISION = "410606e40b1bb7968ce318c41009355c3ac32503"
# sha256 of FILE_NAME at REVISION (its Hugging Face LFS blob name), checked by download-models.
AGGREGATOR_SHA256 = "0dc20e4fb5f5767bd1c30c9e67b011186c2c68d8eef596ffad121388c567b687"


class AggregatorWeightsMismatchError(ValueError):
    """The aggregator weights file is not the pinned one (``AGGREGATOR_SHA256``). A ``ValueError``, so the
    command-line tools report it as a one-line ``Error: ...``."""


def get_aggregator_model(device=None):
    import torch
    from foldmatch.model.residue_embedding_aggregator import ResidueEmbeddingAggregator
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILE_NAME,
        revision=REVISION
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.load(model_path, weights_only=True, map_location=device)
    aggregator_model = ResidueEmbeddingAggregator()
    aggregator_model.load_state_dict(weights)
    return aggregator_model


def get_residue_model(device=None):
    import torch
    from foldmatch.utils.esm.loaders import esm_open
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return esm_open(device)


def download_aggregator_weights(*, cache_dir=None) -> Path:
    """``download-models``: fetch the pinned aggregator weights (``REPO_ID``/``FILE_NAME`` at ``REVISION``, the file
    :func:`get_aggregator_model` loads) into the Hugging Face cache (``cache_dir``, else ``HF_HUB_CACHE`` /
    ``HF_HOME``) and verify their sha256 (``AGGREGATOR_SHA256``, a full hash). Returns the file's path.

    Download only: no torch, no model, no device. A cached file is not downloaded again (no network at a commit
    revision). A cached file whose sha256 differs is downloaded once more (``force_download``); it is refused
    (:class:`AggregatorWeightsMismatchError`) if that fails or the fresh file still differs.
    """
    where = f"{REPO_ID}@{REVISION}"
    try:
        path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, revision=REVISION, cache_dir=cache_dir))
    except LocalEntryNotFoundError as e:   # offline or Hub unreachable, and not cached
        raise FileNotFoundError(f"aggregator weights {FILE_NAME} ({where}) are not in the Hugging Face cache and "
                                f"could not be downloaded ({e})") from e
    got = _file_sha256(path)
    if got == AGGREGATOR_SHA256:
        return path
    corrupt = (f"aggregator weights {path} ({where}) have sha256 {got}, expected {AGGREGATOR_SHA256}: the cached "
               "copy is corrupt or was replaced.")
    try:
        path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, revision=REVISION, cache_dir=cache_dir,
                                    force_download=True))
    except (OSError, ValueError) as e:  # offline, Hub unreachable (requests errors are OSErrors), a full disk
        raise AggregatorWeightsMismatchError(f"{corrupt} Downloading it again failed ({type(e).__name__}: {e}).") from e
    got = _file_sha256(path)
    if got != AGGREGATOR_SHA256:
        raise AggregatorWeightsMismatchError(
            f"aggregator weights {FILE_NAME} ({where}) have sha256 {got} after a fresh download, expected "
            f"{AGGREGATOR_SHA256}: the file at the pinned revision is not the pinned one")
    return path


def release_mps_cache(device):
    """On the Apple GPU (MPS), wait for the batch's kernels and return the allocator's cached
    memory to the system. Without it an ESM3 pass grew to 83 GB and slowed to 41 residues/s."""
    import torch
    if torch.device(device).type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


# =====================================================================================================================
# esm-codes bundles
# =====================================================================================================================
def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_file(system: EsmCodesSystem, filename: str, *, cache_dir, local_files_only: bool,
                force_download: bool = False) -> str:
    """One bundle file through ``hf_hub_download`` at the system's commit; every failure becomes a bundle error."""
    where = f"{system.bundle_repo}@{system.bundle_revision}"
    cache = cache_dir or hf_constants.HF_HUB_CACHE
    try:
        return hf_hub_download(repo_id=system.bundle_repo, filename=filename, revision=system.bundle_revision,
                               cache_dir=cache_dir, local_files_only=local_files_only, force_download=force_download)
    except LocalEntryNotFoundError as e:
        if local_files_only:
            mode = "cached-only use was requested (local_files_only=True)"
        elif hf_constants.HF_HUB_OFFLINE:
            mode = "offline mode is on (HF_HUB_OFFLINE=1)"
        else:
            mode = "the Hugging Face Hub is unreachable"
        raise BundleUnavailableError(
            f"esm-codes bundle {where}: {filename} is not in the Hugging Face cache ({cache}) and {mode}. Run "
            "`fm-embedding download-models` once where the Hub is reachable, with the same HF_HOME / HF_HUB_CACHE."
        ) from e
    except (EntryNotFoundError, RevisionNotFoundError, RepositoryNotFoundError) as e:
        raise BundleMismatchError(f"esm-codes bundle {where} has no {filename} ({type(e).__name__}); the "
                                  f"registry entry {system.name} does not match the Hub") from e
    except PermissionError as e:
        raise BundleUnavailableError(
            f"esm-codes bundle {where}: {filename} is not in the Hugging Face cache ({cache}) and the cache is not "
            f"writable ({e}). Run `fm-embedding download-models` with a writable HF_HOME / HF_HUB_CACHE; a read-only "
            "cache works once it holds the bundle.") from e
    except OSError as e:          # HTTP errors (requests' exceptions are OSErrors), a full disk, ...
        raise BundleUnavailableError(f"esm-codes bundle {where}: {filename} could not be fetched into {cache} "
                                     f"({type(e).__name__}: {e})") from e
    except ValueError as e:
        if not force_download:
            raise
        # hf_hub_download refuses a forced download offline and wraps a failed one in a ValueError
        raise BundleUnavailableError(f"esm-codes bundle {where}: {filename} could not be downloaded again into "
                                     f"{cache} ({e})") from e


def get_esm_codes_bundle(system: EsmCodesSystem, *, cache_dir=None, local_files_only: bool = False,
                         redownload_corrupt: bool = False) -> Dict[str, Path]:
    """The files of ``system``'s bundle, ``{file name: local path}``, every one sha256-verified.

    ``system`` must be a registry entry, exactly as registered (:func:`scoring_system.registered`): the repo, the
    revision and the file sha256s come only from the registry (``system.bundle_repo``, ``system.bundle_revision``,
    ``system.bundle_files``), and a system from elsewhere (a manifest) that differs from its entry is refused before
    any Hub call. Each file goes through ``hf_hub_download`` at that commit: a cached file is returned without any
    network access or write (so ``HF_HUB_OFFLINE=1`` and read-only caches work), a missing one is downloaded into
    the cache (``cache_dir``, else ``HF_HUB_CACHE`` / ``HF_HOME``) unless ``local_files_only``. Every call hashes
    every file (about 26 MB for esm-codes-v1, tens of ms) and writes no marker files, since the cache may be shared
    or read-only.

    A cached file whose sha256 differs is refused (:class:`BundleMismatchError`, naming the cache blob to delete);
    with ``redownload_corrupt`` (``download-models``) it is downloaded once more with ``force_download`` and checked
    again, and refused only if the Hub's file itself differs from the pin.

    Refused: anything but an :class:`EsmCodesSystem`, an unpinned revision (None, a branch, a short hash), a system
    that is not its registry entry (:class:`UnknownScoringSystemError`), :class:`BundleUnavailableError` when a file is
    neither cached nor downloadable (offline, cached-only, Hub unreachable, a read-only cache without the bundle), and
    :class:`BundleMismatchError` when a file is missing at the pinned revision or its sha256 differs.
    """
    if not isinstance(system, EsmCodesSystem):
        raise TypeError(f"get_esm_codes_bundle needs a registry EsmCodesSystem, got {type(system).__name__}")
    if not isinstance(system.bundle_revision, str) or not scoring_system.is_commit(system.bundle_revision):
        raise ValueError(f"{system.name}: bundle revision {system.bundle_revision!r} is not a 40-hex commit; "
                         "an unpinned bundle download is refused")
    scoring_system.registered(system)
    where = f"{system.bundle_repo}@{system.bundle_revision}"
    files: Dict[str, Path] = {}
    for filename, sha256 in system.bundle_files:
        path = _fetch_file(system, filename, cache_dir=cache_dir, local_files_only=local_files_only)
        got = _file_sha256(path)
        if got != sha256 and redownload_corrupt and not local_files_only:
            path = _fetch_file(system, filename, cache_dir=cache_dir, local_files_only=False, force_download=True)
            got = _file_sha256(path)
            if got != sha256:
                raise BundleMismatchError(
                    f"esm-codes bundle file {filename} ({where}) has sha256 {got} after a fresh download, expected "
                    f"{sha256} for {system.name}: the file at the pinned revision differs from the registry entry")
        if got != sha256:
            blob = Path(path).resolve()
            linked = f" (the cache blob {path} links to)" if blob != Path(path).absolute() else ""
            raise BundleMismatchError(
                f"esm-codes bundle file {path} ({where}) has sha256 {got}, expected {sha256} for {system.name}: the "
                "cached copy is corrupt or was replaced. Run `fm-embedding download-models`, which downloads such a "
                f"file again, or delete {blob}{linked} and fetch again.")
        files[filename] = Path(path)
    return files


def download_esm_codes_bundles(names: Optional[Iterable[str]] = None, *, cache_dir=None) -> Dict[str, Dict[str, Path]]:
    """Fetch and verify the bundles of the named systems (default: every registered system), ``{name: files}``.
    Unknown names are refused (listing the known ones) before anything is downloaded; a corrupt cached file is
    downloaded again (``redownload_corrupt``)."""
    names = list(dict.fromkeys(names)) if names else list(scoring_system.known_names())
    systems = [scoring_system.get(n) for n in names]
    return {s.name: get_esm_codes_bundle(s, cache_dir=cache_dir, redownload_corrupt=True) for s in systems}
