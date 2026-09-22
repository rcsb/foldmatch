from pathlib import Path
import torch

from esm.models.esm3 import ESM3
from esm.models.vqvae import StructureTokenEncoder
from esm.tokenization import TokenizerCollection, EsmSequenceTokenizer, StructureTokenizer, SecondaryStructureTokenizer, \
    SASADiscretizingTokenizer, InterProQuantizedTokenizer, ResidueAnnotationsTokenizer

from huggingface_hub import  snapshot_download

ESM_REPO_ID = "rcsb/rcsb-esm"
# Pinned deliberately. Every model downstream reads these weights as FROZEN features: the
# stage-1 aggregator was trained on them, and the sequence->3Di head's accuracy, its
# fitted profile temperature, its target kernel and the Karlin-Altschul lambda/K in
# search/scoring_system.py were all calibrated against this exact backbone. Left
# unpinned, a new upload to the repo would silently change every one of those numbers
# without changing a line of code here.
ESM_REVISION = "636acb420daf674df37ab0220f56061a5de2f169"


class DataRoot:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.path = Path(
                snapshot_download(repo_id=ESM_REPO_ID, revision=ESM_REVISION)
            )
        return cls._instance

    def get(self):
        return self.path


def data_root():
    return DataRoot().get()


def structure_encoder(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root() / "data/weights/esm3_structure_encoder_v0.pth", map_location=device
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
        data_root() / "data/weights/esm3_sm_open_v1.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model