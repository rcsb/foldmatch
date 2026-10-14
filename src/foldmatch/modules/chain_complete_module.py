import hashlib
import logging

import numpy as np
from lightning import LightningModule

from foldmatch.search.residue_shards import QUERY_ENCODING_DTYPE, RESIDUE_CODES, check_residue_output
from foldmatch.utils.data import collate_seq_embeddings
from foldmatch.utils.esm.loaders import per_residue_embedding
from foldmatch.utils.model import release_mps_cache
from foldmatch.utils.sequence import normalize_sequence, seq_sha1

logger = logging.getLogger(__name__)

# Rows whose residue payload failed are logged one by one up to this many per rank, then only counted.
_LOGGED_FAILURES = 10


class ChainCompleteModule(LightningModule):
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
    validation refuses on rank 0 after the pass.
    """

    def __init__(
            self,
            res_model,
            aggregator_model,
            residue_head=None,
            residue_output=None,
    ):
        super().__init__()
        self.esm3 = res_model
        self.aggregator =  aggregator_model
        residue_output = check_residue_output(residue_output)
        if (residue_head is None) != (residue_output is None):
            raise ValueError("residue_head and residue_output go together: pass both (a residue pass) or neither")
        self.residue_head = residue_head
        self.residue_output = residue_output
        self.residue_failures = 0

    def on_predict_start(self):
        logger.info(f"ESM + Aggregator device: {self.device} | {self.trainer.global_rank + 1}/{self.trainer.world_size}")

    def on_predict_end(self):
        if self.residue_failures:
            logger.error(
                f"{self.residue_failures} protein(s) on this rank got an empty {self.residue_output} payload; "
                f"the residue store validation will refuse them"
            )

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
        payloads = []
        digests = []
        for (esm_prot, name), embeddings in zip(prot_batch, prot_embeddings):
            payload, digest = self._residue_payload(esm_prot.sequence, name, embeddings)
            payloads.append(payload)
            digests.append(digest)
        release_mps_cache(self.device)
        return chain_embeddings, tuple(prot_names), tuple(payloads), tuple(digests)

    def _residue_payload(self, sequence, name, embeddings):
        """``(payload bytes, 20-byte seq_sha1)`` of one protein; an empty payload instead of any exception."""
        sequence = sequence or ""
        if not sequence.isascii():
            # S15 refuses non-ASCII input before the pass; the row still gets a 20-byte seq_sha1 (of the UTF-8).
            digest = hashlib.sha1(normalize_sequence(sequence, allow_non_ascii=True).encode("utf-8", "surrogatepass"))
            return self._failed(name, "the sequence is not ASCII"), digest.digest()
        digest = seq_sha1(sequence)
        if embeddings.dim() != 2 or embeddings.shape[0] != len(sequence) + 2:
            return self._failed(name, f"ESM3 gave {tuple(embeddings.shape)} rows for {len(sequence)} residues "
                                      f"(expected {len(sequence) + 2} with BOS and EOS)"), digest
        try:
            rows = embeddings[1:-1]  # S1: residues 1..L, BOS and EOS removed (a view; the aggregator input is untouched)
            if self.residue_output == RESIDUE_CODES:
                payload = self.residue_head.encode_target(rows).cpu().numpy()
            else:
                payload = np.ascontiguousarray(self.residue_head.encode_query(rows).cpu().numpy(),
                                               dtype=QUERY_ENCODING_DTYPE)
            return payload.tobytes(), digest
        except Exception as e:  # noqa: BLE001  (never raise inside the distributed pass; validated after it)
            return self._failed(name, f"{type(e).__name__}: {e}"), digest

    def _failed(self, name, reason):
        self.residue_failures += 1
        if self.residue_failures <= _LOGGED_FAILURES:
            logger.error(f"{name}: no {self.residue_output} payload ({reason}); the row is written empty")
        elif self.residue_failures == _LOGGED_FAILURES + 1:
            logger.error(f"more proteins without a {self.residue_output} payload; only counting them from now on")
        return b""
