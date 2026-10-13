"""Protein sequence normalisation and the FASTA pre-flight.

One function, :func:`normalize_sequence`, decides what string a FASTA record
stands for: the sequence ESM3 tokenises, the one the sequence store keeps, the
one Stage-2 aligns and the one :func:`seq_sha1` hashes. Keeping them identical
is what lets an embedding, a stored sequence and a residue index line up.

:func:`preflight_fasta` is the read-only pass every rank makes over a FASTA
file before ESM3 is loaded. It is the scan the dataset needs anyway (the
record offsets), extended with the counts and problems that decide whether
the run may start, so a refusal stops every rank at the same point.

No torch or esm imports, so lightweight modules can use it.
"""
import hashlib
import heapq
import logging
import string
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from foldmatch.utils.fasta import read_record_at

logger = logging.getLogger(__name__)

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_STANDARD_BYTES = STANDARD_AMINO_ACIDS.encode("ascii")

# ASCII whitespace only (string.whitespace). Python's str.split() also treats
# \x1c-\x1f as whitespace, so it is used only as a fast test for "none at all".
_DELETE_WHITESPACE = str.maketrans("", "", string.whitespace)
_UPPER_ASCII_DELETE_WHITESPACE = str.maketrans(
    string.ascii_lowercase, string.ascii_uppercase, string.whitespace
)

# Ids quoted per problem in a pre-flight message.
_EXAMPLE_IDS = 5


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


def _scan_records(fasta_file: Path) -> Iterator[Tuple[int, str, int, int, bool, bool]]:
    """Single pass yielding, per record, ``(header_byte_offset, id, length,
    nonstandard, non_ascii, angle_bracket)``.

    The record rules are those of :func:`foldmatch.utils.fasta.iter_fasta_offsets`
    (id = first token of the header, blank lines and empty records skipped),
    but every figure is about the *normalised* sequence: ``length`` residues,
    of which ``nonstandard`` are outside the 20 standard amino acids, and
    whether it holds a non-ASCII character or a ``<``/``>``.

    Uses explicit ``tell``/``readline`` rather than ``for line in f`` because
    the latter's read-ahead buffering makes ``tell()`` offsets unreliable.
    """
    with open(fasta_file, 'r') as f:
        header_offset = None
        name = None
        length = nonstandard = 0
        non_ascii = angle_bracket = False
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
                    yield header_offset, name, length, nonstandard, non_ascii, angle_bracket
                header_offset = pos
                name = stripped[1:].split()[0]
                length = nonstandard = 0
                non_ascii = angle_bracket = False
                continue
            if stripped.isascii():
                residues = stripped.upper()
                if len(residues.split()) != 1:
                    residues = residues.translate(_DELETE_WHITESPACE)
                nonstandard += len(residues.encode("ascii").translate(None, _STANDARD_BYTES))
            else:
                residues = stripped.translate(_UPPER_ASCII_DELETE_WHITESPACE)
                nonstandard += sum(c not in STANDARD_AMINO_ACIDS for c in residues)
                non_ascii = True
            length += len(residues)
            if '<' in residues or '>' in residues:
                angle_bracket = True
        if name is not None and length > 0:
            yield header_offset, name, length, nonstandard, non_ascii, angle_bracket


def scan_fasta(fasta_file: Path) -> Iterator[Tuple[int, str, int]]:
    """Single pass yielding ``(header_byte_offset, id, length)`` per record,
    ``length`` being the length of the normalised sequence (see :func:`_scan_records`)."""
    for offset, name, length, *_ in _scan_records(fasta_file):
        yield offset, name, length


def _id_hash(name: str) -> int:
    # Deterministic across processes (unlike hash()), so every rank finds the same duplicates.
    return int.from_bytes(
        hashlib.blake2b(name.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little"
    )


@dataclass(frozen=True)
class FastaPreflight:
    """What :func:`preflight_fasta` found in one FASTA file.

    Counts over all records are ``entries``; every other figure is over the
    *kept* records, those with at least ``min_res_n`` residues, which are the
    ones that get embedded and stored.
    """
    path: Path
    min_res_n: int
    entries: int
    entries_kept: int  # one embedding each
    residues: int
    duplicates: int  # kept records whose id repeats an earlier kept record
    unique_residues: int  # residues over the first record of each id (what the store keeps)
    nonstandard_entries: int
    nonstandard_residues: int
    non_ascii_entries: int
    angle_bracket_entries: int  # records containing '<' or '>'
    max_length: int
    duplicate_ids: Tuple[str, ...]  # a few examples of each problem, in file order
    non_ascii_ids: Tuple[str, ...]
    angle_bracket_ids: Tuple[str, ...]
    offsets: np.ndarray = field(repr=False, compare=False)  # header offsets of the kept records

    @property
    def unique_entries(self) -> int:
        """Kept records with distinct ids: the rows of the sequence store."""
        return self.entries_kept - self.duplicates

    def problems(self) -> List[str]:
        """One message per kind of input a strict run refuses."""
        found = []
        if self.duplicates:
            found.append(
                f"{self.duplicates} record(s) repeat an earlier id "
                f"(e.g. {', '.join(self.duplicate_ids)})"
            )
        if self.angle_bracket_entries:
            found.append(
                f"{self.angle_bracket_entries} sequence(s) contain '<' or '>' "
                f"(e.g. {', '.join(self.angle_bracket_ids)}); ESM3 reads a bracketed "
                f"name such as '<mask>' as a single token, so residues and positions "
                f"no longer line up"
            )
        if self.non_ascii_entries:
            found.append(
                f"{self.non_ascii_entries} sequence(s) contain non-ASCII characters "
                f"(e.g. {', '.join(self.non_ascii_ids)})"
            )
        return found

    def check(self, strict: bool = False) -> None:
        """Refuse (strict) or warn about (plain) every :meth:`problems` entry.

        Strict raises one ``ValueError`` naming all of them. A plain run logs
        them and goes ahead exactly as before: duplicates keep their first
        sequence in the store, and the other characters reach ESM3 unchanged.
        """
        found = self.problems()
        if strict and found:
            raise ValueError(f"{self.path}: " + "; ".join(found) + ".")
        for problem in found:
            logger.warning(f"{self.path}: {problem}.")
        logger.info(
            f"Pre-flight {self.path}: {self.entries_kept} of {self.entries} entries with "
            f">= {self.min_res_n} residues, {self.residues} residues, longest {self.max_length}"
        )


def preflight_fasta(fasta_file: Path, min_res_n: int = 0) -> FastaPreflight:
    """Read-only pass over a FASTA file: record offsets plus counts and problems.

    Deterministic, so every rank that runs it reaches the same verdict.
    Duplicate ids are found from sorted 64-bit id hashes, 8 bytes per record,
    with an exact comparison of the ids whose hashes collide, so memory stays
    bounded however large the file.
    """
    offsets = array('q')
    hashes = array('Q')
    entries = entries_kept = residues = max_length = 0
    nonstandard_entries = nonstandard_residues = non_ascii_entries = angle_bracket_entries = 0
    non_ascii_ids: List[str] = []
    angle_bracket_ids: List[str] = []
    for offset, name, length, nonstandard, non_ascii, angle_bracket in _scan_records(fasta_file):
        entries += 1
        if min_res_n > 0 and length < min_res_n:
            continue
        entries_kept += 1
        residues += length
        max_length = max(max_length, length)
        offsets.append(offset)
        hashes.append(_id_hash(name))
        if nonstandard:
            nonstandard_entries += 1
            nonstandard_residues += nonstandard
        if non_ascii:
            non_ascii_entries += 1
            if len(non_ascii_ids) < _EXAMPLE_IDS:
                non_ascii_ids.append(name)
        if angle_bracket:
            angle_bracket_entries += 1
            if len(angle_bracket_ids) < _EXAMPLE_IDS:
                angle_bracket_ids.append(name)

    offsets = np.frombuffer(offsets, dtype=np.int64) if entries_kept else np.empty(0, dtype=np.int64)
    duplicates, duplicate_residues, duplicate_ids = _find_duplicates(fasta_file, offsets, hashes)
    return FastaPreflight(
        path=Path(fasta_file),
        min_res_n=min_res_n,
        entries=entries,
        entries_kept=entries_kept,
        residues=residues,
        duplicates=duplicates,
        unique_residues=residues - duplicate_residues,
        nonstandard_entries=nonstandard_entries,
        nonstandard_residues=nonstandard_residues,
        non_ascii_entries=non_ascii_entries,
        angle_bracket_entries=angle_bracket_entries,
        max_length=max_length,
        duplicate_ids=duplicate_ids,
        non_ascii_ids=tuple(non_ascii_ids),
        angle_bracket_ids=tuple(angle_bracket_ids),
        offsets=offsets,
    )


def _find_duplicates(fasta_file: Path, offsets: np.ndarray, hashes: array) -> Tuple[int, int, Tuple[str, ...]]:
    """``(duplicates, their residues, example ids)`` among the kept records.

    A record is a duplicate when its id equals that of an earlier kept record
    (the store keeps the first). Only records whose 64-bit id hash occurs more
    than once are re-read, to compare the actual ids and measure the repeats.
    """
    if len(hashes) < 2:
        return 0, 0, ()
    hashes = np.frombuffer(hashes, dtype=np.uint64)
    ordered = np.sort(hashes)
    repeated = ordered[1:][ordered[1:] == ordered[:-1]]
    del ordered
    if repeated.size == 0:
        return 0, 0, ()
    # File-order positions of the records whose hash is shared, regrouped by
    # hash; the stable sort keeps file order within each group.
    candidates = np.flatnonzero(np.isin(hashes, repeated))
    candidates = candidates[np.argsort(hashes[candidates], kind='stable')]

    duplicates = duplicate_residues = 0
    examples: List[Tuple[int, str]] = []  # max-heap (negated positions) of the first few
    first_seen: set = set()
    previous_hash = None
    with open(fasta_file, 'r') as fh:
        for position in candidates.tolist():
            if hashes[position] != previous_hash:
                previous_hash = hashes[position]
                first_seen.clear()
            fh.seek(int(offsets[position]))
            name = fh.readline().strip()[1:].split()[0]
            if name not in first_seen:
                first_seen.add(name)
                continue
            _, sequence = read_record_at(fh, int(offsets[position]))
            duplicates += 1
            duplicate_residues += len(normalize_sequence(sequence, allow_non_ascii=True))
            heapq.heappush(examples, (-position, name))
            if len(examples) > _EXAMPLE_IDS:
                heapq.heappop(examples)
    return duplicates, duplicate_residues, tuple(dict.fromkeys(name for _, name in sorted(examples, reverse=True)))
