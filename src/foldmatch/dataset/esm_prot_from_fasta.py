import logging
import os

import numpy as np
from esm.sdk.api import ESMProtein
from torch.utils.data import Dataset

# Re-exported for backwards compatibility: callers historically imported
# parse_fasta from this module. The canonical implementations now live in the
# dependency-light foldmatch.utils.fasta.
from foldmatch.utils.fasta import iter_fasta, iter_fasta_offsets, parse_fasta, read_record_at
from foldmatch.utils.sequence import FastaPreflight, normalize_sequence, scan_fasta

logger = logging.getLogger(__name__)

__all__ = ["EsmProtFromFasta", "parse_fasta", "iter_fasta"]


class EsmProtFromFasta(Dataset):
    """Random-access FASTA dataset that never materializes the sequences.

    On init it scans the FASTA once and keeps only a compact ``int64`` array of
    per-record byte offsets (those passing ``min_res_n``). ``__getitem__`` seeks
    to the offset and reads that single record on demand, so peak memory is
    ~8 bytes per record regardless of sequence length — letting the inference
    pipeline handle FASTA files with hundreds of millions of sequences.

    Sequences are normalised (:func:`normalize_sequence`: ASCII whitespace
    removed, ASCII letters uppercased) before ESM3 sees them, and
    ``min_res_n`` applies to the normalised length, as in the sequence store.

    ``fasta_file`` may also be the :class:`FastaPreflight` of the file, whose
    offsets come from the same scan; the pre-flight pass then doubles as the
    index build and the file is not read twice.
    """

    def __init__(
        self,
        fasta_file,
        min_res_n=0
    ):
        super().__init__()
        if isinstance(fasta_file, FastaPreflight):
            if fasta_file.min_res_n != min_res_n:
                raise ValueError(
                    f"Pre-flight of {fasta_file.path} was made with min_res_n="
                    f"{fasta_file.min_res_n}, not {min_res_n}"
                )
            self.fasta_file = str(fasta_file.path)
            self._offsets = fasta_file.offsets
        else:
            self.fasta_file = str(fasta_file)
            # np.fromiter streams the generator straight into the array without an
            # intermediate Python list, keeping the index build itself low-memory.
            self._offsets = np.fromiter(
                (
                    offset
                    for offset, _name, length in scan_fasta(self.fasta_file)
                    if min_res_n <= 0 or length >= min_res_n
                ),
                dtype=np.int64,
            )
        # Per-process file handle, opened lazily. Never pickled (see __getstate__)
        # so each DataLoader worker opens its own handle and seeks independently.
        self._fh = None
        self._fh_pid = None

    def __len__(self):
        return len(self._offsets)

    def _handle(self):
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(self.fasta_file, 'r')
            self._fh_pid = pid
        return self._fh

    def __getitem__(self, idx):
        offset = int(self._offsets[idx])
        name, sequence = read_record_at(self._handle(), offset)
        # Non-ASCII characters are kept one for one, as before (ESM3 reads them as <unk>).
        return ESMProtein(sequence=normalize_sequence(sequence, allow_non_ascii=True)), name

    def __getstate__(self):
        # Drop the open file handle so the dataset pickles cleanly to workers.
        state = self.__dict__.copy()
        state['_fh'] = None
        state['_fh_pid'] = None
        return state
