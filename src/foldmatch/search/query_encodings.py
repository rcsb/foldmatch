"""Per-run query encodings for Stage 2 (plan P5.4): one fp16 memmap plus an id -> (offset, length, seq_sha1) map.

The query pass writes each query's S5 encoding (``float16[L, 64]``) into its Parquet shards
(:mod:`foldmatch.search.residue_shards`). :meth:`QueryEncodingIndex.build` reads them once, in shard order, and
appends every encoding to one raw little-endian fp16 file, ``[total rows, 64]``, that Stage-2 workers map read-only
(:func:`open_memmap`) instead of receiving arrays through pickles. The map gives each query's first row, its row
count and the SHA-1 of its normalised sequence (S15).

No torch: workers import this module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Optional

import numpy as np

from foldmatch.search.residue_shards import (
    QUERY_ENCODING, QUERY_ENCODING_DTYPE, ResidueShardError, check_shards, iter_payload_batches, payload_row_bytes,
    residue_width,
)

_EXAMPLES = 10


class QueryEncodingEntry(NamedTuple):
    offset: int        # first row in the memmap
    length: int        # rows (= residues)
    seq_sha1: bytes    # 20 bytes


def open_memmap(path, rows: int, width: int) -> np.ndarray:
    """The encodings file as a read-only ``float16 [rows, width]`` array (an empty array for 0 rows, which
    ``np.memmap`` cannot map)."""
    if rows == 0:
        return np.empty((0, width), dtype=QUERY_ENCODING_DTYPE)
    return np.memmap(path, dtype=QUERY_ENCODING_DTYPE, mode="r", shape=(rows, width))


class QueryEncodingIndex:
    """The query encodings of one run: ``path`` (raw fp16), ``rows``, ``width`` and ``entries``."""

    def __init__(self, path, rows: int, width: int, entries: Dict[str, QueryEncodingEntry], system_name: str = ""):
        self.path = Path(path)
        self.rows = int(rows)
        self.width = int(width)
        self.entries = entries
        self.system_name = system_name
        self._memmap = None

    @classmethod
    def build(cls, shards: Iterable, dst, *, system, expected_count: Optional[int] = None,
              load_batch_size: int = 8192) -> "QueryEncodingIndex":
        """One sequential pass over ``shards``, in the given order, writing ``dst`` (which must not exist).

        Refused (:class:`ResidueShardError`), with ``dst`` removed again: shards that are not query encodings of
        ``system`` or are mixed (:func:`check_shards`, footers only, before ``dst`` is created); a duplicate id; a
        query with an empty payload (its ESM3 rows did not line up with its sequence) or one whose size is not a
        whole number of rows; and, with ``expected_count``, a different number of queries.
        """
        shards = [Path(p) for p in shards]
        check_shards(shards, QUERY_ENCODING, system)
        row_bytes = payload_row_bytes(QUERY_ENCODING, system)
        width = residue_width(QUERY_ENCODING, system)
        dst = Path(dst)
        out = open(dst, "xb")  # never another run's file: FileExistsError before anything is written
        try:
            with out:
                entries, rows = _append_encodings(out, shards, row_bytes, load_batch_size, expected_count)
            if os.path.getsize(dst) != rows * row_bytes:
                raise ResidueShardError(f"{dst}: wrote {os.path.getsize(dst)} bytes, expected {rows * row_bytes}")
        except BaseException:
            dst.unlink(missing_ok=True)
            raise
        return cls(dst, rows, width, entries, system.name)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name) -> bool:
        return name in self.entries

    def memmap(self) -> np.ndarray:
        """The encodings, read-only ``float16 [rows, width]``."""
        if self._memmap is None:
            self._memmap = open_memmap(self.path, self.rows, self.width)
        return self._memmap

    def get(self, name: str) -> np.ndarray:
        """Query ``name``'s encoding, a ``float16 [length, width]`` view of the memmap (``KeyError`` if unknown)."""
        entry = self.entries[name]
        return self.memmap()[entry.offset:entry.offset + entry.length]

    def memmap_args(self) -> tuple:
        """``(path, rows, width)`` for :func:`open_memmap` in a worker (picklable)."""
        return str(self.path), self.rows, self.width

    def remove(self) -> None:
        """Delete the encodings file (after the run)."""
        self._memmap = None
        self.path.unlink(missing_ok=True)


def _append_encodings(out, shards, row_bytes: int, load_batch_size: int, expected_count: Optional[int]):
    entries: Dict[str, QueryEncodingEntry] = {}
    rows = 0
    empty, ragged, duplicates = [], [], []
    for shard in shards:
        for batch in iter_payload_batches(shard, QUERY_ENCODING, load_batch_size):
            sizes = np.diff(batch.offsets)
            for k, name in enumerate(batch.ids):
                size = int(sizes[k])
                if size == 0:
                    empty.append(name)
                elif size % row_bytes:
                    ragged.append(name)
                if name in entries:
                    duplicates.append(name)
                    continue
                entries[name] = QueryEncodingEntry(rows + int(batch.offsets[k]) // row_bytes, size // row_bytes,
                                                   batch.seq_sha1s[k])
            out.write(memoryview(batch.data))  # a refused batch is written too; build() removes the file
            rows += batch.data.size // row_bytes
    problems = []
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate query id(s) (e.g. {duplicates[:_EXAMPLES]})")
    if empty:
        problems.append(f"{len(empty)} query(ies) without an encoding: their ESM3 rows did not match their sequence "
                        f"length (e.g. {empty[:_EXAMPLES]})")
    if ragged:
        problems.append(f"encoding(s) that are not a whole number of {row_bytes}-byte rows (e.g. {ragged[:_EXAMPLES]})")
    if expected_count is not None and not problems and len(entries) != expected_count:
        problems.append(f"{len(entries)} query encodings in {len(shards)} shard(s), but the input has "
                        f"{expected_count} queries")
    if problems:
        raise ResidueShardError("query encodings refused: " + "; ".join(problems))
    return entries, rows
