import hashlib
import os
import re
from pathlib import Path
import torch

from esm.models.esm3 import ESM3
from esm.models.vqvae import StructureTokenEncoder
from esm.tokenization import TokenizerCollection, EsmSequenceTokenizer, StructureTokenizer, SecondaryStructureTokenizer, \
    SASADiscretizingTokenizer, InterProQuantizedTokenizer, ResidueAnnotationsTokenizer

from huggingface_hub import  snapshot_download, try_to_load_from_cache

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
    stat = resolved.stat()
    key = (str(resolved), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _hashed:
        _hashed[key] = _file_sha256(resolved)
    _check_esm_sha256(weights_path, _hashed[key])
    return "sha256"


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


def structure_encoder(device: torch.device | str = "cpu"):
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
