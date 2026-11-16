"""The per-residue payload of an ESM3 pass (plan P5.1), shared by the modules that compute it from ESM3's rows."""
import hashlib
import logging

import numpy as np

from foldmatch.search.residue_shards import QUERY_ENCODING_DTYPE, RESIDUE_CODES, check_residue_output
from foldmatch.utils.sequence import normalize_sequence, seq_sha1

logger = logging.getLogger(__name__)

# Rows whose residue payload failed are logged one by one up to this many per rank, then only counted.
_LOGGED_FAILURES = 10


class ResiduePayloads:
    """A LightningModule mixin: with ``residue_output`` (``"residue_codes"`` or ``"query_encoding"``) and the
    ``residue_head`` of the scoring system (:class:`foldmatch.model.residue_code_encoder.ResidueCodeHead`, a submodule,
    so Lightning moves it with the module), the rows ``1..L`` of each protein's ESM3 embeddings (S1) are encoded, one
    protein at a time: S4 target codes (``L * 8`` bytes) or S5 query encodings (``float16[L, 64]`` little-endian,
    ``L * 128`` bytes), with ``seq_sha1``, the SHA-1 of the normalised sequence (S15).

    Never raises for a payload: under DDP a rank that raised would leave the others waiting for the timeout. A protein
    whose ESM3 rows are not ``L + 2`` (the tokenizer did not keep one token per residue), whose sequence is not ASCII,
    or whose encoding failed gets an empty payload instead, logged as an ERROR (the first ``_LOGGED_FAILURES`` by name)
    and counted at the end of the pass (``residue_failures``; the first ids in ``residue_failed_ids``). ``code_files``:
    the payloads are residue codes that fm-embedding writes to files of their own (``--output-codes-folder``), where an
    empty payload is an empty ``[0, 8]`` table, and which raises after the pass when there are any
    (``ResidueCodesPass.check_after_pass``); otherwise they go to a build's or a query's shards, whose validation
    refuses an empty one after the pass.
    """

    def _init_residue_payloads(self, residue_head=None, residue_output=None, code_files: bool = False):
        residue_output = check_residue_output(residue_output)
        if (residue_head is None) != (residue_output is None):
            raise ValueError("residue_head and residue_output go together: pass both (a residue pass) or neither")
        if code_files and residue_output != RESIDUE_CODES:
            raise ValueError(f"code files hold {RESIDUE_CODES}, not {residue_output or 'no residue output'}")
        self.residue_head = residue_head
        self.residue_output = residue_output
        self.code_files = bool(code_files)
        self.residue_failures = 0
        self.residue_failed_ids = []

    def on_predict_end(self):
        if self.residue_failures:
            if self.code_files:
                logger.error(f"{self.residue_failures} protein(s) on this rank got no residue codes: their codes are "
                             f"written as empty [0, 8] tables")
            else:
                logger.error(
                    f"{self.residue_failures} protein(s) on this rank got an empty {self.residue_output} payload; "
                    f"the residue store validation will refuse them"
                )

    def _residue_payloads(self, prot_batch, prot_embeddings):
        """``(payloads, seq_sha1s)`` of a batch, as tuples, from its proteins' ESM3 embeddings."""
        payloads = []
        digests = []
        for (esm_prot, name), embeddings in zip(prot_batch, prot_embeddings):
            payload, digest = self._residue_payload(esm_prot.sequence, name, embeddings)
            payloads.append(payload)
            digests.append(digest)
        return tuple(payloads), tuple(digests)

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
            rows = embeddings[1:-1]  # S1: residues 1..L, BOS and EOS removed (a view; the embeddings are untouched)
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
        if len(self.residue_failed_ids) < _LOGGED_FAILURES:
            self.residue_failed_ids.append(name)
        if self.residue_failures <= _LOGGED_FAILURES:
            if self.code_files:
                logger.error(f"{name}: no residue codes ({reason}); its codes are written as an empty [0, 8] table")
            else:
                logger.error(f"{name}: no {self.residue_output} payload ({reason}); the row is written empty")
        elif self.residue_failures == _LOGGED_FAILURES + 1:
            if self.code_files:
                logger.error("more proteins without residue codes; only counting them from now on")
            else:
                logger.error(f"more proteins without a {self.residue_output} payload; only counting them from now on")
        return b""
