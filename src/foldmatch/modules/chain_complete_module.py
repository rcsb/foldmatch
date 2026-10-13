import logging

from lightning import LightningModule

from foldmatch.utils.data import collate_seq_embeddings
from foldmatch.utils.esm.loaders import per_residue_embedding
from foldmatch.utils.model import release_mps_cache

logger = logging.getLogger(__name__)

class ChainCompleteModule(LightningModule):

    def __init__(
            self,
            res_model,
            aggregator_model
    ):
        super().__init__()
        self.esm3 = res_model
        self.aggregator =  aggregator_model

    def on_predict_start(self):
        logger.info(f"ESM + Aggregator device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")

    def predict_step(self, prot_batch, batch_idx):
        prot_embeddings = []
        prot_names = []
        for esm_prot, name in prot_batch:
            embeddings = per_residue_embedding(self.esm3, esm_prot)
            prot_embeddings.append(embeddings)
            prot_names.append(name)
        res_batch_embedding, res_batch_mask = collate_seq_embeddings(prot_embeddings)
        chain_embeddings = self.aggregator(res_batch_embedding, res_batch_mask)
        release_mps_cache(self.device)
        return chain_embeddings, tuple(prot_names)