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
  which only the full digest covers;
* ``foldmatch.pass``: how the pass computed the payloads (:func:`pass_record`, JSON): its ``min_res_n``, the device
  type and precision ESM3 ran on, the number of ranks, the ESM3 revision and the software versions. The writer
  completes it at the start of the pass, once the device is known, before any row is written. The code store's
  manifest takes its ``build`` block from it, also when the store is rebuilt from the shards later.

The readers (:func:`check_shards`, :func:`iter_residue_codes`, ``foldmatch.search.query_encodings``) read every
shard's footer before any data and refuse shards without the column, mixed shards (different kinds, systems or
digests) and stale shards (another system's encoding): residue codes need the system's ``encoding_digest`` (a system
that differs only in its statistics may reuse them), query encodings its ``digest`` too.

A plain pass (no residue output) keeps the legacy schema ``id | embedding`` with no metadata, byte for byte.

pyarrow and numpy only: no torch, so Stage-2 code can import it.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence

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
META_PASS = b"foldmatch.pass"

#: The fields of a ``foldmatch.pass`` record (:func:`pass_record`).
PASS_KEYS = ("min_res_n", "accelerator", "world_size", "esm_precision", "encoder_precision", "esm_revision",
             "foldmatch_version", "torch", "esm", "platform")
#: The residue-code encoders always run in fp32, without TF32 or autocast (method spec S3).
ENCODER_PRECISION = "fp32-no-tf32"

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


def shard_metadata(residue_output: str, system) -> dict:
    return {
        META_RESIDUE_OUTPUT: check_residue_output(residue_output).encode("ascii"),
        META_SYSTEM: system.name.encode("ascii"),
        META_ENCODING_DIGEST: system.encoding_digest().encode("ascii"),
        META_SYSTEM_DIGEST: system.digest().encode("ascii"),
    }


def esm_precision(device_type: Optional[str]) -> Optional[str]:
    """How ESM3 computes on ``device_type``: ``"bf16-autocast"`` on CUDA (esm 3.2.3 runs ``ESM3.logits`` under bf16
    autocast there, and only there), ``"fp32"`` elsewhere (CPU, Apple MPS); ``None`` for an unknown device."""
    if device_type is None:
        return None
    return "bf16-autocast" if device_type == "cuda" else "fp32"


def _dist_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def pass_record(*, min_res_n: Optional[int] = None, esm_revision: Optional[str] = None,
                accelerator: Optional[str] = None, world_size: Optional[int] = None) -> Dict[str, Any]:
    """The ``foldmatch.pass`` record of a residue pass (``PASS_KEYS``): ``min_res_n``, ``accelerator`` (the device
    type the module ran on: ``cpu``, ``cuda``, ``mps``), ``world_size`` (the pass's ranks), the ESM3 and encoder
    precision, the ESM3 revision, and the versions of FoldMatch, torch and esm and the platform of this process.
    ``None`` where unknown."""
    from foldmatch import __version__
    return {
        "min_res_n": min_res_n,
        "accelerator": accelerator,
        "world_size": world_size,
        "esm_precision": esm_precision(accelerator),
        "encoder_precision": ENCODER_PRECISION,
        "esm_revision": esm_revision,
        "foldmatch_version": __version__,
        "torch": _dist_version("torch"),
        "esm": _dist_version("esm"),
        "platform": f"{platform.system()}-{platform.machine()}",
    }


def pass_metadata(record: Dict[str, Any]) -> dict:
    """The schema metadata entry that holds a :func:`pass_record` (compact, sorted JSON)."""
    return {META_PASS: json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")}


def shard_schema(residue_output: Optional[str] = None, system=None, record: Optional[Dict[str, Any]] = None
                 ) -> pa.Schema:
    """The shard schema: the legacy ``id | embedding`` (no metadata) without a residue output, else with the
    residue column, ``seq_sha1`` and the metadata of ``system`` (required then) and of the pass ``record``
    (:func:`pass_record`; one with every field unknown by default)."""
    residue_output = check_residue_output(residue_output)
    if residue_output is None:
        if record is not None:
            raise ValueError("a pass record is only kept in a shard with a residue output")
        return pa.schema(list(BASE_FIELDS))
    if system is None:
        raise ValueError(f"a {residue_output} shard schema needs the scoring system")
    metadata = shard_metadata(residue_output, system)
    metadata.update(pass_metadata(pass_record() if record is None else record))
    return pa.schema(list(BASE_FIELDS) + [(residue_output, PAYLOAD_TYPE), (SEQ_SHA1, SEQ_SHA1_TYPE)],
                     metadata=metadata)


def _concat_payloads(payloads: Sequence[bytes], total: int):
    """One contiguous buffer holding ``payloads`` back to back (``total`` bytes)."""
    return b"".join(payloads)


def large_binary_array(payloads: Sequence[bytes]) -> pa.LargeBinaryArray:
    """A ``large_binary`` array of ``payloads`` with explicit 64-bit offsets (no 2 GiB limit per chunk)."""
    n = len(payloads)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.fromiter((len(p) for p in payloads), dtype=np.int64, count=n), out=offsets[1:])
    data = _concat_payloads(payloads, int(offsets[-1]))
    return pa.LargeBinaryArray.from_buffers(PAYLOAD_TYPE, n, [None, pa.py_buffer(offsets), pa.py_buffer(data)])
