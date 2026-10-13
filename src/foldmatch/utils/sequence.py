"""Protein sequence normalisation shared by every FASTA input path.

One function, :func:`normalize_sequence`, decides what string a FASTA record
stands for: the sequence ESM3 tokenises, the one the sequence store keeps, the
one Stage-2 aligns and the one :func:`seq_sha1` hashes. Keeping them identical
is what lets an embedding, a stored sequence and a residue index line up.

Dependency-free (no torch/numpy/esm) so lightweight modules can use it.
"""
import hashlib
import string
from pathlib import Path
from typing import Iterator, Tuple

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# ASCII whitespace only (string.whitespace). Python's str.split() also treats
# \x1c-\x1f as whitespace, so it is used only as a fast test for "none at all".
_DELETE_WHITESPACE = str.maketrans("", "", string.whitespace)
_UPPER_ASCII_DELETE_WHITESPACE = str.maketrans(
    string.ascii_lowercase, string.ascii_uppercase, string.whitespace
)


def normalize_sequence(sequence: str, allow_non_ascii: bool = False) -> str:
    """Remove all ASCII whitespace and uppercase ASCII letters.

    Every other character is kept as it is, one for one, so the ESM3
    tokenizer still sees one token per residue: ``_`` is ``<mask>``,
    ``* J 0-9`` are ``<unk>`` and ``- . |`` are tokens of their own. Lowercase
    letters, which ESM3 would read as ``<unk>``, become residues.

    Non-ASCII input raises ``ValueError``. With ``allow_non_ascii`` it is kept
    unchanged instead (only ASCII letters are uppercased): the plain build and
    query paths warn about such input rather than refuse it.
    """
    if sequence.isascii():
        # On an ASCII string str.upper() changes ASCII letters only.
        sequence = sequence.upper()
    elif allow_non_ascii:
        return sequence.translate(_UPPER_ASCII_DELETE_WHITESPACE)
    else:
        raise ValueError("Sequence contains non-ASCII characters")
    if len(sequence.split()) != 1:
        sequence = sequence.translate(_DELETE_WHITESPACE)
    return sequence


def seq_sha1(sequence: str) -> bytes:
    """SHA-1 (20 bytes) of the ASCII bytes of ``normalize_sequence(sequence)``."""
    return hashlib.sha1(normalize_sequence(sequence).encode("ascii")).digest()


def _normalized_length(line: str) -> int:
    """``len(normalize_sequence(line, allow_non_ascii=True))`` without building it."""
    if len(line.split()) == 1:
        return len(line)
    return len(line.translate(_DELETE_WHITESPACE))


def scan_fasta(fasta_file: Path) -> Iterator[Tuple[int, str, int]]:
    """Single pass yielding ``(header_byte_offset, id, length)`` per record.

    The record rules are those of :func:`foldmatch.utils.fasta.iter_fasta_offsets`
    (id = first token of the header, blank lines and empty records skipped),
    but ``length`` is the length of the *normalised* sequence, so a
    ``min_res_n`` filter on it agrees with the sequence store and with what
    ESM3 is given.

    Uses explicit ``tell``/``readline`` rather than ``for line in f`` because
    the latter's read-ahead buffering makes ``tell()`` offsets unreliable.
    """
    with open(fasta_file, 'r') as f:
        header_offset = None
        name = None
        length = 0
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('>'):
                if name is not None and length > 0:
                    yield header_offset, name, length
                header_offset = pos
                name = stripped[1:].split()[0]
                length = 0
            else:
                length += _normalized_length(stripped)
        if name is not None and length > 0:
            yield header_offset, name, length
