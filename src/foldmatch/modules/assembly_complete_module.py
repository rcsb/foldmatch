import logging

from lightning import LightningModule
from torch import cat

from foldmatch.utils.data import collate_seq_embeddings
from foldmatch.utils.esm.loaders import per_residue_embedding

logger = logging.getLogger(__name__)

class AssemblyCompleteModule(LightningModule):

    def __init__(
            self,
            res_model,
            aggregator_model,
            max_res_n=0
    ):
        super().__init__()
        self.esm3 = res_model
        self.aggregator =  aggregator_model
        self.max_res_n = max_res_n

    def on_predict_start(self):
        logger.info(f"ESM + Aggregator device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")

    def predict_step(self, prot_batch, batch_idx):
        assembly_embeddings = []
        assembly_names = []
        for esm_chains, name in prot_batch:
            prot_embeddings = []
            assembly_n_res = 0
            for esm_prot in esm_chains:
                embeddings = per_residue_embedding(self.esm3, esm_prot)
                prot_embeddings.append(embeddings)
                assembly_n_res += embeddings.shape[0]
                if 0 < self.max_res_n < assembly_n_res:
                    break
            assembly_embeddings.append(cat(prot_embeddings, dim=0))
            assembly_names.append(name)
        res_batch_embedding, res_batch_mask = collate_seq_embeddings(assembly_embeddings)

        return self.aggregator(res_batch_embedding, res_batch_mask), tuple(assembly_names)