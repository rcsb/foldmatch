import logging

from lightning import LightningModule

from foldmatch.modules.residue_payloads import ResiduePayloads
from foldmatch.utils.data import collate_seq_embeddings
from foldmatch.utils.esm.loaders import per_residue_embedding
from foldmatch.utils.model import release_mps_cache

logger = logging.getLogger(__name__)


class ChainCompleteModule(ResiduePayloads, LightningModule):
    """ESM3 per-residue embeddings -> chain embeddings, optionally with a per-residue payload (plan P5.1).

    The aggregator always gets ESM3's per-residue embeddings unchanged, BOS and EOS included. With
    ``residue_output`` (``"residue_codes"`` or ``"query_encoding"``) and the ``residue_head`` of the scoring system
    (:class:`foldmatch.model.residue_code_encoder.ResidueCodeHead`), the rows ``1..L`` of each protein (S1) are also
    encoded, one protein at a time: S4 target codes (``L * 8`` bytes) or S5 query encodings (``float16[L, 64]``
    little-endian, ``L * 128`` bytes). The step then returns ``(chain_embeddings, names, payloads, seq_sha1s)``
    instead of ``(chain_embeddings, names)``, ``seq_sha1`` being the SHA-1 of the normalised sequence (S15).

    ``predict_step`` never raises for a residue payload: under DDP a rank that raised would leave the others waiting
    for the timeout. A protein whose ESM3 rows are not ``L + 2`` (the tokenizer did not keep one token per residue),
    whose sequence is not ASCII, or whose encoding failed gets an empty payload instead, which the code-store
    validation refuses on rank 0 after the pass (:class:`ResiduePayloads`; with ``code_files``, fm-embedding's
    ``--output-codes-folder``, it is written as an empty ``[0, 8]`` table).
    """

    def __init__(
            self,
            res_model,
            aggregator_model,
            residue_head=None,
            residue_output=None,
            code_files=False,
    ):
        super().__init__()
        self.esm3 = res_model
        self.aggregator =  aggregator_model
        self._init_residue_payloads(residue_head, residue_output, code_files)

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
        if self.residue_output is None:
            release_mps_cache(self.device)
            return chain_embeddings, tuple(prot_names)
        payloads, digests = self._residue_payloads(prot_batch, prot_embeddings)
        release_mps_cache(self.device)
        return chain_embeddings, tuple(prot_names), payloads, digests
