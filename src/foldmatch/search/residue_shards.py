"""Parquet embedding shards with a residue column: the schema contract between the ESM3 pass and its readers.

A pass with a residue output (plan P5) writes, next to each chain embedding, one per-residue payload and the SHA-1
of the sequence it was computed from:

* builds, ``residue_codes``: the S4 target codes, ``uint8[L, 8]`` as ``L * 8`` bytes;
* queries, ``query_encoding``: the S5 query encodings, ``float16[L, 64]`` little-endian as ``L * 128`` bytes.

Columns ``id: string | embedding: list<float32> | <residue_output>: large_binary | seq_sha1: fixed_size_binary(20)``.
``large_binary`` has 64-bit offsets, so a row group may hold more than 2 GiB of payloads. ``seq_sha1`` is
``sha1(normalize_sequence(sequence))`` (S15). A row whose payload could not be computed (the ESM3 rows did not match
the sequence length) has an empty payload; the readers pass it on as length 0 and the code-store validation refuses
it after the pass (the ESM3 pass itself never raises, P5.1).

Schema metadata, fixed when the writer is constructed:

* ``foldmatch.residue_output``: ``residue_codes`` or ``query_encoding``;
* ``foldmatch.system``: the scoring system's name;
* ``foldmatch.encoding_digest``: :meth:`EsmCodesSystem.encoding_digest`, the fields the target codes depend on;
* ``foldmatch.system_digest``: :meth:`EsmCodesSystem.digest`. The query encodings also depend on the query encoder,
  which only the full digest covers.

The readers (:func:`check_shards`, :func:`iter_residue_codes`, ``foldmatch.search.query_encodings``) read every
shard's footer before any data and refuse shards without the column, mixed shards (different kinds, systems or
digests) and stale shards (another system's encoding): residue codes need the system's ``encoding_digest`` (a system
that differs only in its statistics may reuse them), query encodings its ``digest`` too.

A plain pass (no residue output) keeps the legacy schema ``id | embedding`` with no metadata, byte for byte.

pyarrow and numpy only: no torch, so Stage-2 code can import it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, NamedTuple, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RESIDUE_CODES = "residue_codes"
QUERY_ENCODING = "query_encoding"
RESIDUE_OUTPUTS = (RESIDUE_CODES, QUERY_ENCODING)

ID = "id"
EMBEDDING = "embedding"
SEQ_SHA1 = "seq_sha1"
SEQ_SHA1_BYTES = 20

META_RESIDUE_OUTPUT = b"foldmatch.residue_output"
META_SYSTEM = b"foldmatch.system"
META_ENCODING_DIGEST = b"foldmatch.encoding_digest"
META_SYSTEM_DIGEST = b"foldmatch.system_digest"
META_KEYS = (META_RESIDUE_OUTPUT, META_SYSTEM, META_ENCODING_DIGEST, META_SYSTEM_DIGEST)

#: The legacy shard schema (``id | embedding``), unchanged for plain passes.
BASE_FIELDS = ((ID, pa.string()), (EMBEDDING, pa.list_(pa.float32())))
PAYLOAD_TYPE = pa.large_binary()
SEQ_SHA1_TYPE = pa.binary(SEQ_SHA1_BYTES)

#: Payload byte order: little-endian, the native order of every supported platform.
QUERY_ENCODING_DTYPE = np.dtype("<f2")


class ResidueShardError(ValueError):
    """Embedding shards that cannot feed this residue channel: no residue column, mixed or stale shards, or a
    malformed payload."""


def check_residue_output(value) -> Optional[str]:
    """``None``, ``"residue_codes"`` or ``"query_encoding"``; anything else raises ``ValueError``."""
    if value is None:
        return None
    value = getattr(value, "value", value)
    if value not in RESIDUE_OUTPUTS:
        raise ValueError(f"unknown residue output {value!r}; expected None or one of {list(RESIDUE_OUTPUTS)}")
    return value


def residue_width(residue_output: str, system) -> int:
    """Payload values per residue: the code bytes (8) or the query encoding's dimensions (64; both encoders map
    to ``code_bits`` outputs)."""
    return system.code_bytes if check_residue_output(residue_output) == RESIDUE_CODES else system.code_bits


def payload_row_bytes(residue_output: str, system) -> int:
    """Payload bytes per residue: 8 for residue codes, 128 for query encodings (64 x float16)."""
    if check_residue_output(residue_output) == RESIDUE_CODES:
        return system.code_bytes
    if np.dtype(system.query_dtype) != np.dtype(np.float16):
        raise ResidueShardError(f"{system.name}: query_dtype {system.query_dtype!r} is not implemented (float16)")
    return system.code_bits * QUERY_ENCODING_DTYPE.itemsize
