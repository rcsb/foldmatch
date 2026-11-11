"""Lightweight, dependency-free FASTA parsing helpers.

A single home for FASTA reading so the streaming generator and the
list-materializing parser cannot drift apart. Intentionally free of heavy
imports (no torch/pandas/esm) so it can be used from lightweight modules such
as the sequence store as well as from the inference dataset.
"""
from pathlib import Path
from typing import IO, Iterator, List, Tuple


def iter_fasta(fasta_file: Path) -> Iterator[Tuple[str, str]]:
    """Stream ``(id, sequence)`` pairs from a FASTA file one record at a time.

    Never holds more than a single record in memory, so it scales to
    arbitrarily large FASTA files. The id is the first whitespace-delimited
    token of the header. Records with an empty sequence are skipped.
    """
    name = None
    chunks: list = []
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None and chunks:
                    yield name, ''.join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None and chunks:
        yield name, ''.join(chunks)


def parse_fasta(fasta_file: Path) -> List[Tuple[str, str]]:
    """Materialize all ``(id, sequence)`` pairs from a FASTA file into a list.

    Convenience wrapper over :func:`iter_fasta`; loads the whole file into
    memory, so prefer :func:`iter_fasta` for large corpora.
    """
    return list(iter_fasta(fasta_file))


def iter_fasta_offsets(fasta_file: Path) -> Iterator[Tuple[int, str, int]]:
    """Single-pass scan yielding ``(header_byte_offset, id, sequence_length)``.

    Used to build a compact random-access index over a FASTA without holding
    the sequences in memory: only byte offsets (and transiently a length, for
    filtering) are needed. Records with an empty sequence are skipped.

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
                length += len(stripped)
        if name is not None and length > 0:
            yield header_offset, name, length


def read_id_at(fh: IO, offset: int) -> str:
    """The id of the record whose header (``>``) line starts at ``offset`` of an open handle: the first token of
    the header, as :func:`read_record_at` reads it. Leaves ``fh`` just after the header line."""
    fh.seek(offset)
    header = fh.readline().strip()
    return header[1:].split()[0]


def read_record_at(fh: IO, offset: int) -> Tuple[str, str]:
    """Read a single ``(id, sequence)`` record from an open handle at ``offset``.

    ``offset`` must be the byte position of a header (``>``) line, as produced
    by :func:`iter_fasta_offsets`. Reads forward until the next header or EOF.
    The caller owns ``fh`` (it is re-``seek``-ed on each call, so the trailing
    header that terminates the record need not be consumed).
    """
    name = read_id_at(fh, offset)
    chunks: list = []
    while True:
        line = fh.readline()
        if not line:
            break
        stripped = line.strip()
        if stripped.startswith('>'):
            break
        if stripped:
            chunks.append(stripped)
    return name, ''.join(chunks)
