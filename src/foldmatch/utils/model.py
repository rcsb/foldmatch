import torch

from huggingface_hub import hf_hub_download
from foldmatch.model.residue_embedding_aggregator import ResidueEmbeddingAggregator
from foldmatch.model.three_di_head import build_head
from foldmatch.utils.esm.loaders import esm_open

REPO_ID = "rcsb/rcsb-embedding-model"
FILE_NAME = "rcsb-embedding-model.pt"
REVISION = "410606e40b1bb7968ce318c41009355c3ac32503"

# The sequence->3Di head. Unlike the aggregator, whose architecture is the no-argument
# default of its class, this bundle carries its own architecture: `nhead` cannot be
# recovered from the weights, so a hardcoded guess here would load cleanly and predict
# nonsense. See foldmatch/model/three_di_head.py.
THREE_DI_REPO_ID = "rcsb/rcsb-three-di-head"
THREE_DI_FILE_NAME = "three-di-head-v1.pt"
THREE_DI_REVISION = "6b88ce659351894bdb8954ad7a854d046db12527"
THREE_DI_FORMAT = "foldmatch-three-di-head/1"
#: sha256 of the bundle exported from epoch=18-nll=1.1824.ckpt. The revision already pins
#: the content; this is here so the number that identifies the served model is greppable
#: next to the constants calibrated against it (see search/scoring_system.py).
THREE_DI_SHA256 = "8b6608445f72359437332195c1b7e615752f86f441cb7a455a1b4170dd4d05e2"


def get_aggregator_model(device=None):
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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return esm_open(device)


def load_three_di_head(path, device=None):
    """Build the 3Di head from a bundle on disk, using the architecture it records.

    Returns ``(head, bundle)`` with the head in eval mode; the bundle's ``provenance``
    carries the profile temperature the model was calibrated at, which the caller must
    match against the scoring system it intends to use.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(path, weights_only=True, map_location=device)
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise ValueError(
            f"{path} is not a 3Di head bundle: expected a dict with 'state_dict', "
            f"'architecture' and 'provenance'. A bare state dict cannot be loaded, "
            f"because the architecture it needs (notably nhead) is not recoverable "
            f"from the weights.")
    if bundle.get("format") != THREE_DI_FORMAT:
        raise ValueError(
            f"{path}: unsupported bundle format {bundle.get('format')!r}, "
            f"expected {THREE_DI_FORMAT!r}")
    head = build_head(bundle["architecture"])
    # strict=True: a silently partial load is how a wrong architecture would survive.
    head.load_state_dict(bundle["state_dict"], strict=True)
    return head.to(device).eval(), bundle


def get_three_di_head(device=None, revision=THREE_DI_REVISION):
    """Download and build the released sequence->3Di head."""
    if revision is None:
        raise ValueError(
            "THREE_DI_REVISION is unset: pin it to the Hugging Face commit the head was "
            "uploaded at before serving it, so a later upload cannot silently change the "
            "model behind the calibrated lambda/K, temperature and target kernel.")
    model_path = hf_hub_download(
        repo_id=THREE_DI_REPO_ID,
        filename=THREE_DI_FILE_NAME,
        revision=revision,
    )
    return load_three_di_head(model_path, device=device)

