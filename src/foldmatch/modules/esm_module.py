import logging

from lightning import LightningModule

from foldmatch.utils.esm.loaders import per_residue_embedding

logger = logging.getLogger(__name__)

class EsmModule(LightningModule):

    def __init__(
            self,
            model
    ):
        super().__init__()
        self.esm3 = model

    def on_predict_start(self):
        logger.info(f"ESM device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")

    def predict_step(self, prot_batch, batch_idx):
        return tuple([self.__compute_embeddings(esm_prot) for esm_prot, name in prot_batch]), tuple([name for esm_prot, name in prot_batch])

    def __compute_embeddings(self, esm_prot):
        return per_residue_embedding(self.esm3, esm_prot)