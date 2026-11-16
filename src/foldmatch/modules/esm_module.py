import logging

from lightning import LightningModule

from foldmatch.modules.residue_payloads import ResiduePayloads
from foldmatch.utils.esm.loaders import per_residue_embedding
from foldmatch.utils.model import release_mps_cache

logger = logging.getLogger(__name__)

class EsmModule(ResiduePayloads, LightningModule):
    """ESM3 per-residue embeddings, ``(L + 2, 1536)`` per protein with BOS and EOS: each step returns
    ``(embeddings, names)``.

    With ``residue_output`` ``"residue_codes"`` and the scoring system's ``residue_head`` (fm-embedding's
    ``--output-codes-folder``), each protein's residue codes are also computed from the same ESM3 rows
    (:class:`ResiduePayloads`, the embeddings returned unchanged), and a step returns ``(embeddings, names, payloads,
    seq_sha1s)``.
    """

    def __init__(
            self,
            model,
            residue_head=None,
            residue_output=None,
            code_files=False,
    ):
        super().__init__()
        self.esm3 = model
        self._init_residue_payloads(residue_head, residue_output, code_files)

    def on_predict_start(self):
        logger.info(f"ESM device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")

    def predict_step(self, prot_batch, batch_idx):
        embeddings = tuple([self.__compute_embeddings(esm_prot) for esm_prot, name in prot_batch])
        if self.residue_output is None:
            release_mps_cache(self.device)
            return embeddings, tuple([name for esm_prot, name in prot_batch])
        payloads, digests = self._residue_payloads(prot_batch, embeddings)
        release_mps_cache(self.device)
        return embeddings, tuple([name for esm_prot, name in prot_batch]), payloads, digests

    def __compute_embeddings(self, esm_prot):
        return per_residue_embedding(self.esm3, esm_prot)
