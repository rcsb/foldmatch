"""ESM3: the pinned snapshot, its weights check, the model builder and the per-residue embedding.

torch and esm are imported only by the functions that build or run the model, so the pins, the weights check and
:func:`download_esm3_snapshot` (``download-models``) work in processes without torch, and importing this module
never touches a GPU.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
from huggingface_hub.errors import LocalEntryNotFoundError

ESM_REPO_ID = "rcsb/rcsb-esm"
# Pinned: the aggregator and everything calibrated downstream read these weights as frozen
# features, so a new upload to the repo must not change them silently.
ESM_REVISION = "636acb420daf674df37ab0220f56061a5de2f169"
ESM_WEIGHTS_FILE = "data/weights/esm3_sm_open_v1.pth"
# The same blob is served by rcsb/rcsb-esm@636acb42 and the EvolutionaryScale/esm3-sm-open-v1 snapshots.
ESM_WEIGHTS_SHA256 = "5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9"

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
# Files fully hashed by this process, keyed by identity, so one process never hashes the
# same file twice. Nothing is written to disk: the Hugging Face cache may be shared or read-only.
_hashed = {}


class EsmWeightsMismatchError(ValueError):
    """The ESM3 weights file is not the pinned release (``ESM_WEIGHTS_SHA256``).

    A ``ValueError``, so the command-line tools report it as a one-line
    ``Error: ...`` with its remedy instead of a traceback.
    """


class DataRoot:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance.path = _esm_snapshot()
            cls._instance = instance
        return cls._instance

    def get(self):
        return self.path


def data_root():
    return DataRoot().get()


def _esm_snapshot():
    # A file that was not in the cache before this call is hashed once after its download;
    # a cached one is checked by its blob name only (see verify_esm_weights).
    cached = isinstance(
        try_to_load_from_cache(ESM_REPO_ID, ESM_WEIGHTS_FILE, revision=ESM_REVISION), str
    )
    path = Path(snapshot_download(repo_id=ESM_REPO_ID, revision=ESM_REVISION))
    verify_esm_weights(path / ESM_WEIGHTS_FILE, full_hash=not cached)
    return path


def verify_esm_weights(weights_path, full_hash=False):
    """Check that ``weights_path`` holds the pinned ESM3 weights (``ESM_WEIGHTS_SHA256``).

    In the Hugging Face cache layout the file is a symlink to ``blobs/<sha256>``, so the
    blob name already is the content hash and the check costs nothing. When the layout
    hides it (``local_dir`` downloads, copied snapshots, filesystems without symlinks), or
    when ``full_hash`` is set, the file is hashed, once per process.
    Returns ``"blob-name"`` or ``"sha256"``, the way the file was verified.
    """
    weights_path = Path(weights_path)
    resolved = Path(os.path.realpath(weights_path))
    if not full_hash and resolved.parent.name == "blobs" and _SHA256_HEX.fullmatch(resolved.name):
        _check_esm_sha256(weights_path, resolved.name)
        return "blob-name"
    _check_esm_sha256(weights_path, _sha256_once(resolved))
    return "sha256"


def _sha256_once(resolved):
    """The sha256 of the file ``resolved``, hashed at most once per process for one file identity."""
    stat = resolved.stat()
    key = (str(resolved), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _hashed:
        _hashed[key] = _file_sha256(resolved)
    return _hashed[key]


def _check_esm_sha256(weights_path, sha256):
    if sha256 != ESM_WEIGHTS_SHA256:
        raise EsmWeightsMismatchError(
            f"ESM3 weights {weights_path} have sha256 {sha256}, expected {ESM_WEIGHTS_SHA256} "
            f"({ESM_REPO_ID}@{ESM_REVISION}). The file is corrupt or not the pinned release: "
            f"remove it and run download-models again."
        )


def _file_sha256(path, chunk_size=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_esm3_snapshot(*, cache_dir=None) -> Path:
    """``download-models``: fetch the pinned ESM3 snapshot (``ESM_REPO_ID`` at ``ESM_REVISION``, the snapshot
    :func:`data_root` serves) into the Hugging Face cache (``cache_dir``, else ``HF_HUB_CACHE`` / ``HF_HOME``) and
    verify the weights against ``ESM_WEIGHTS_SHA256`` by a full sha256 (once per process). Returns the snapshot folder.

    Download only: no torch, no model, no device. A cached snapshot is not downloaded again (with ``HF_HUB_OFFLINE=1``
    nothing touches the network). A cached weights file whose content differs from the pin (a truncated or corrupted
    blob, whose name alone would pass) is downloaded once more (``force_download``); it is refused
    (:class:`EsmWeightsMismatchError`) if that fails or the fresh file still differs.
    """
    try:
        snapshot = Path(snapshot_download(repo_id=ESM_REPO_ID, revision=ESM_REVISION, cache_dir=cache_dir))
    except LocalEntryNotFoundError as e:   # offline or Hub unreachable, and not cached
        raise FileNotFoundError(f"ESM3 snapshot {ESM_REPO_ID}@{ESM_REVISION} is not in the Hugging Face cache and "
                                f"could not be downloaded ({e})") from e
    weights = snapshot / ESM_WEIGHTS_FILE
    if not weights.is_file():
        raise FileNotFoundError(
            f"ESM3 weights {ESM_WEIGHTS_FILE} are missing from the cached snapshot {snapshot} "
            f"({ESM_REPO_ID}@{ESM_REVISION}); run download-models with the Hugging Face Hub reachable "
            "(HF_HUB_OFFLINE unset)")
    try:
        verify_esm_weights(weights, full_hash=True)
        return snapshot
    except EsmWeightsMismatchError as mismatch:
        corrupt = mismatch
    try:
        hf_hub_download(repo_id=ESM_REPO_ID, filename=ESM_WEIGHTS_FILE, revision=ESM_REVISION, cache_dir=cache_dir,
                        force_download=True)
    except (OSError, ValueError) as e:  # offline, Hub unreachable (requests errors are OSErrors), a full disk
        raise EsmWeightsMismatchError(f"{corrupt} Downloading it again failed ({type(e).__name__}: {e}).") from e
    got = _sha256_once(Path(os.path.realpath(weights)))
    if got != ESM_WEIGHTS_SHA256:
        raise EsmWeightsMismatchError(
            f"ESM3 weights {ESM_WEIGHTS_FILE} ({ESM_REPO_ID}@{ESM_REVISION}) have sha256 {got} after a fresh "
            f"download, expected {ESM_WEIGHTS_SHA256}: the file at the pinned revision is not the pinned release")
    return snapshot


def structure_encoder(device: torch.device | str = "cpu"):
    import torch
    from esm.models.vqvae import StructureTokenEncoder
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root() / "data/weights/esm3_structure_encoder_v0.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    return model



def get_model_tokenizers():
    from esm.tokenization import TokenizerCollection, EsmSequenceTokenizer, StructureTokenizer, \
        SecondaryStructureTokenizer, SASADiscretizingTokenizer, InterProQuantizedTokenizer, ResidueAnnotationsTokenizer

    class CustomAnnotationsTokenizer(ResidueAnnotationsTokenizer):
        def __init__(self, csv_path: str | None = None, max_annotations: int = 16):
            from esm.utils.constants import esm3 as C
            super().__init__("none", max_annotations)
            if csv_path is None:
                csv_path = str(data_root() / C.RESID_CSV)
            self.csv_path = csv_path

    return TokenizerCollection(
        sequence=EsmSequenceTokenizer(),
        structure=StructureTokenizer(),
        secondary_structure=SecondaryStructureTokenizer(kind="ss8"),
        sasa=SASADiscretizingTokenizer(),
        function=InterProQuantizedTokenizer(),
        residue_annotations=CustomAnnotationsTokenizer(),
    )


def esm_open(device: torch.device | str = "cpu"):
    import torch
    from esm.models.esm3 import ESM3
    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=structure_encoder,
            structure_decoder_fn=lambda x: x,
            function_decoder_fn=lambda x: x,
            tokenizers=get_model_tokenizers(),
        ).eval()
    state_dict = torch.load(
        data_root() / ESM_WEIGHTS_FILE, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    return model


# One re-entrant lock per ESM3 model. geometric_attention_off flips a flag on the model itself,
# so a forward on the same model in another thread must not run while the flag is off.
_MODEL_LOCKS = weakref.WeakKeyDictionary()
_MODEL_LOCKS_GUARD = threading.Lock()
_SHARED_LOCK = threading.RLock()  # for objects that cannot be weakly referenced


def _model_lock(model):
    with _MODEL_LOCKS_GUARD:
        try:
            lock = _MODEL_LOCKS.get(model)
            if lock is None:
                lock = _MODEL_LOCKS[model] = threading.RLock()
        except TypeError:
            lock = _SHARED_LOCK
        return lock


@contextmanager
def geometric_attention_off(model, off=True):
    """Switch off ESM3's geometric attention (block 0) for the duration of the block, then restore it.

    Without coordinates every frame is masked, so geometric attention adds exactly zero to the
    residue stream, but it still allocates an O(L^2) tensor per head: ESM3 ran out of memory at
    2,750 aa on a 40 GB A100, and fits 16,000 aa in about 10 GiB with it off. The embeddings are
    bit-identical (checked on GPU for 300-2,000 aa and on CPU for 300-1,000 aa).

    The flag belongs to the model, which other threads may share, so the block holds the model's
    lock (also with ``off=False``): a forward on the same model in another thread that goes
    through :func:`per_residue_embedding` waits, instead of running without geometric attention.
    """
    with _model_lock(model):
        blocks = [block for block in model.transformer.blocks if block.use_geom_attn] if off else []
        for block in blocks:
            block.use_geom_attn = False
        try:
            yield
        finally:
            for block in blocks:
                block.use_geom_attn = True


def per_residue_embedding(model, esm_prot):
    """ESM3 per-residue embedding of one protein, (L + 2, 1536) with BOS and EOS.

    Sequence-only proteins (no coordinates) run without geometric attention; proteins with
    coordinates take ESM3's path unchanged. Every call holds the model's lock, so calls on one
    model from several threads run one at a time (calls on different models do not wait).
    """
    from esm.sdk.api import SamplingConfig
    with geometric_attention_off(model, off=esm_prot.coordinates is None):
        return model.forward_and_sample(
            model.encode(esm_prot), SamplingConfig(return_per_residue_embeddings=True)
        ).per_residue_embedding
