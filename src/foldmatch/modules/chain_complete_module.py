import logging

import numpy as np
import torch
from esm.sdk.api import SamplingConfig
from lightning import LightningModule

from foldmatch.utils.data import collate_seq_embeddings

logger = logging.getLogger(__name__)

class ChainCompleteModule(LightningModule):

    def __init__(
            self,
            res_model,
            aggregator_model,
            three_di_head=None,
            three_di_scoring=None,
            three_di_output="codes"
    ):
        """``three_di_head`` predicts a 3Di state distribution per residue from the same
        ESM3 pass, and ``three_di_scoring`` (a ``PreparedScoring``) interprets it. Both
        None - the default - leaves every existing path untouched.

        ``three_di_output`` picks what is emitted, because the two sides of a search need
        different things from the same forward pass:

        * ``codes``   - one byte per residue (state x confidence bucket). What a database
          entry stores: it is what the aligner can consume as a target symbol, and it is
          1/40th the size of the distribution it came from.
        * ``profile`` - the raw (L, 20) distribution, float16. What a QUERY needs: the
          position-specific matrix is built from the full distribution, and quantising to
          a byte and back would lose exactly the gradation that makes a soft query worth
          +0.06 F1 over a hard one.
        """
        super().__init__()
        self.esm3 = res_model
        self.aggregator =  aggregator_model
        self.three_di_head = three_di_head
        self.three_di_scoring = three_di_scoring
        if three_di_output not in ("codes", "profile"):
            raise ValueError("three_di_output must be 'codes' or 'profile'")
        self.three_di_output = three_di_output
        if (three_di_head is None) != (three_di_scoring is None):
            raise ValueError("a 3Di head needs its scoring system, and vice versa")

    def on_predict_start(self):
        logger.info(f"ESM + Aggregator device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")
        if self.three_di_head is not None:
            logger.info(
                f"3Di head active: encoding targets as "
                f"{self.three_di_scoring.system.n_codes} codes under scoring system "
                f"'{self.three_di_scoring.system.name}'")

    def predict_step(self, prot_batch, batch_idx):
        prot_embeddings = []
        prot_names = []
        three_di = [] if self.three_di_head is not None else None
        for esm_prot, name in prot_batch:
            embeddings = self.esm3.forward_and_sample(
                self.esm3.encode(esm_prot), SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding
            if three_di is not None:
                # The residue tensor carries BOS and EOS; the head was trained on the
                # residues alone, so it reads embeddings[1:-1]. The slice is deliberately
                # local to this branch - the aggregator must keep receiving the tensor
                # exactly as it always has, or every stage-1 embedding changes.
                three_di.append(self._encode_three_di(embeddings[1:-1]))
            prot_embeddings.append(embeddings)
            prot_names.append(name)
        res_batch_embedding, res_batch_mask = collate_seq_embeddings(prot_embeddings)
        aggregated = self.aggregator(res_batch_embedding, res_batch_mask)
        if three_di is None:
            return aggregated, tuple(prot_names)
        return aggregated, tuple(prot_names), tuple(three_di)

    def _encode_three_di(self, residue_embedding):
        """The head's output for one protein, in whichever form was asked for."""
        with torch.no_grad():
            logits = self.three_di_head(residue_embedding.unsqueeze(0).float())[0]
            profile = torch.softmax(logits, dim=-1).to('cpu', torch.float64).numpy()
        if self.three_di_output == "profile":
            # float16 is deliberate: the matrix this feeds is rounded to int32 at scale
            # 100, so the extra precision cannot reach a score, and the profile is 20x
            # wider than the codes it replaces.
            return profile.astype(np.float16)
        return self.three_di_scoring.encode_target(profile)