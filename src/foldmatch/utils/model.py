import torch

from huggingface_hub import hf_hub_download
from foldmatch.model.residue_embedding_aggregator import ResidueEmbeddingAggregator
from foldmatch.utils.esm.loaders import esm_open

REPO_ID = "rcsb/rcsb-embedding-model"
FILE_NAME = "rcsb-embedding-model.pt"
REVISION = "410606e40b1bb7968ce318c41009355c3ac32503"


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


def release_mps_cache(device):
    """On the Apple GPU (MPS), wait for the batch's kernels and return the allocator's cached
    memory to the system. Without it an ESM3 pass grew to 83 GB and slowed to 41 residues/s."""
    if torch.device(device).type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
