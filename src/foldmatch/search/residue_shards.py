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
import logging
import platform
from dataclasses import dataclass, field
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

logger = logging.getLogger(__name__)

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


# =====================================================================================================================
# readers
# =====================================================================================================================
@dataclass(frozen=True)
class ShardInfo:
    """What a shard's footer says (read with ``pq.read_schema``: no data). ``pass_record`` is its parsed
    ``foldmatch.pass`` metadata (:func:`pass_record`), ``None`` if it has none; it describes the pass, so it is not
    part of :attr:`key` (the ranks of one run may be told apart by it in principle; :func:`shards_pass_record`)."""
    path: Path
    residue_output: Optional[str]
    system: Optional[str]
    encoding_digest: Optional[str]
    system_digest: Optional[str]
    column_types: tuple
    pass_record: Optional[Dict[str, Any]] = field(default=None, compare=False)

    @property
    def key(self) -> tuple:
        return self.residue_output, self.system, self.encoding_digest, self.system_digest, self.column_types

    def describe(self) -> str:
        if self.residue_output is None:
            return "no residue column (a plain pass)"
        return (f"{self.residue_output} of {self.system} (encoding_digest {self.encoding_digest}, system_digest "
                f"{self.system_digest})")


def _type_name(t: pa.DataType) -> str:
    # Parquet names a list's item field "element" where Arrow says "item": compare the value types only.
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return f"{'large_' if pa.types.is_large_list(t) else ''}list<{_type_name(t.value_type)}>"
    return str(t)


def _column_types(schema: pa.Schema) -> tuple:
    return tuple((f.name, _type_name(f.type)) for f in schema)


def read_shard_info(path) -> ShardInfo:
    """The footer of the shard at ``path``; a file that is not a readable Parquet shard (empty, truncated, missing)
    is a :class:`ResidueShardError` naming it."""
    try:
        schema = pq.read_schema(path)
    except (pa.ArrowException, OSError) as e:
        raise ResidueShardError(f"{path}: not a readable Parquet embedding shard ({e})") from e
    meta = schema.metadata or {}

    def text(key):
        value = meta.get(key)
        return value.decode("utf-8", "replace") if value is not None else None

    record = None
    if META_PASS in meta:
        try:
            record = json.loads(meta[META_PASS].decode("utf-8"))
        except ValueError:
            record = None
        if not isinstance(record, dict):
            raise ResidueShardError(f"{path}: its {META_PASS.decode()} metadata is not a JSON object")
    return ShardInfo(path=Path(path), residue_output=text(META_RESIDUE_OUTPUT), system=text(META_SYSTEM),
                     encoding_digest=text(META_ENCODING_DIGEST), system_digest=text(META_SYSTEM_DIGEST),
                     column_types=_column_types(schema), pass_record=record)


def shards_pass_record(infos: Sequence[ShardInfo]) -> Dict[str, Any]:
    """The pass record of one run's shards (``PASS_KEYS``, every field ``None`` when they hold none): the first
    shard's, which is rank 0's in path order. The ranks of a run share their settings, so a field on which the
    shards disagree, or a shard without a record, is only logged."""
    out: Dict[str, Any] = dict.fromkeys(PASS_KEYS)
    records = [info.pass_record for info in infos if info.pass_record is not None]
    if not records:
        return out
    first = records[0]
    out.update({key: first.get(key) for key in PASS_KEYS})
    differing = sorted({key for record in records[1:] for key in PASS_KEYS if record.get(key) != first.get(key)})
    if differing or len(records) != len(infos):
        logger.warning(f"{len(infos)} shard(s) ({infos[0].path.name}, ...): {len(records)} record their pass"
                       + (f", and they differ in {', '.join(differing)}" if differing else "")
                       + f"; using {infos[0].path.name}'s")
    return out


def check_shards(paths: Iterable, residue_output: str, system) -> List[ShardInfo]:
    """Refuse (:class:`ResidueShardError`) unless every shard in ``paths`` holds ``residue_output`` computed with
    ``system``'s encoding. Reads the footers only, all of them before any data.

    Mixed shards (different kinds, systems, digests or column types between them) are refused naming each group;
    then the common metadata must be ``system``'s: the ``encoding_digest`` for residue codes, and also the
    ``system_digest`` for query encodings. The columns must be exactly :func:`shard_schema`'s.
    """
    residue_output = check_residue_output(residue_output)
    if residue_output is None:
        raise ValueError("check_shards needs a residue output")
    infos = [read_shard_info(p) for p in paths]
    if not infos:
        return infos
    groups: dict = {}
    for info in infos:
        groups.setdefault(info.key, []).append(info)
    if len(groups) > 1:
        shown = "; ".join(f"{len(g)} shard(s) with {g[0].describe()} (e.g. {g[0].path.name})"
                          for g in groups.values())
        raise ResidueShardError(f"mixed embedding shards: {shown}. Shards of one run must come from one pass; remove "
                                "the shards of other runs or use another folder.")
    info = infos[0]
    where = f"{len(infos)} shard(s) ({info.path.name}, ...)"
    if info.residue_output != residue_output:
        raise ResidueShardError(f"{where} hold {info.describe()}, not {residue_output}: they were written by a pass "
                                f"without --residue-codes or for another purpose")
    want_types = _column_types(shard_schema(residue_output, system))
    if info.column_types != want_types:
        raise ResidueShardError(f"{where} have columns {list(info.column_types)}, expected {list(want_types)}")
    stale = []
    if info.encoding_digest != system.encoding_digest():
        stale.append(f"encoding_digest {info.encoding_digest}, {system.name} has {system.encoding_digest()}")
    if residue_output == QUERY_ENCODING and info.system_digest != system.digest():
        stale.append(f"system_digest {info.system_digest}, {system.name} has {system.digest()}")
    if stale:
        raise ResidueShardError(f"stale embedding shards: {where} hold {info.describe()}, which is not the encoding "
                                f"of {system.name} ({'; '.join(stale)}). Rebuild them with this FoldMatch.")
    return infos


class PayloadBatch(NamedTuple):
    """One record batch of a residue shard: ``offsets`` (int64 ``[n + 1]``, in bytes, starting at 0) delimit each
    row's payload in ``data`` (uint8)."""
    ids: List[str]
    offsets: np.ndarray
    data: np.ndarray
    seq_sha1s: List[bytes]


def _payload_buffers(column: pa.Array, where: str):
    if column.null_count:
        raise ResidueShardError(f"{where}: {column.null_count} null payload(s)")
    if not isinstance(column, pa.LargeBinaryArray):
        raise ResidueShardError(f"{where}: payload column is {column.type}, expected large_binary")
    _, offsets_buf, data_buf = column.buffers()
    n = len(column)
    offsets = np.frombuffer(offsets_buf, dtype=np.int64, count=column.offset + n + 1)[column.offset:]
    start, stop = int(offsets[0]), int(offsets[-1])
    data = (np.frombuffer(data_buf, dtype=np.uint8, count=stop)[start:stop] if stop
            else np.empty(0, dtype=np.uint8))
    return offsets - start, data


def iter_payload_batches(path, residue_output: str, batch_size: int = 32768) -> Iterator[PayloadBatch]:
    """Stream one shard's ``id``, payload and ``seq_sha1`` columns (column projection: the embeddings are not
    read). Opens the shard when iteration starts and closes it when the shard is done."""
    residue_output = check_residue_output(residue_output)
    where = f"{Path(path).name}"
    try:
        with pq.ParquetFile(path) as pf:
            for record in pf.iter_batches(batch_size=batch_size, columns=[ID, residue_output, SEQ_SHA1]):
                offsets, data = _payload_buffers(record.column(residue_output), where)
                sha1 = record.column(SEQ_SHA1)
                if sha1.null_count:
                    raise ResidueShardError(f"{where}: {sha1.null_count} null seq_sha1 value(s)")
                yield PayloadBatch(record.column(ID).to_pylist(), offsets, data, sha1.to_pylist())
    except (pa.ArrowException, OSError) as e:
        # A shard damaged after its footer was checked (truncated data pages, a vanished file).
        raise ResidueShardError(f"{path}: the shard could not be read ({e})") from e


class ResidueCodeBatch(NamedTuple):
    """Residue codes of consecutive entries, in shard order, as the code store's ``create()`` takes them: a
    3-tuple ``(ids, codes, seq_sha1s)`` with one uint8 ``[L, 8]`` array per entry in ``codes`` (``[0, 8]`` for an
    empty payload). The arrays are read-only, C-contiguous views of the batch's Arrow buffer, consecutive in it, so
    ``codes[k].tobytes()`` is entry ``k``'s byte layout in the store."""
    ids: List[str]
    codes: List[np.ndarray]
    seq_sha1s: List[bytes]

    @property
    def lengths(self) -> np.ndarray:
        """Residues per entry (int64)."""
        return np.fromiter((len(c) for c in self.codes), dtype=np.int64, count=len(self.codes))


def iter_residue_codes(paths: Sequence, system, load_batch_size: int = 32768) -> Iterator[ResidueCodeBatch]:
    """:func:`check_shards` on every shard (before anything is yielded), then the residue codes one shard at a
    time, in the given order, ``load_batch_size`` entries per batch."""
    paths = [Path(p) for p in paths]
    check_shards(paths, RESIDUE_CODES, system)
    return _residue_code_batches(paths, system.code_bytes, load_batch_size)


def _residue_code_batches(paths, code_bytes: int, load_batch_size: int) -> Iterator[ResidueCodeBatch]:
    for path in paths:
        for batch in iter_payload_batches(path, RESIDUE_CODES, load_batch_size):
            sizes = np.diff(batch.offsets)
            bad = np.flatnonzero(sizes % code_bytes)
            if bad.size:
                raise ResidueShardError(f"{path.name}: {bad.size} residue-code payload(s) are not a multiple of "
                                        f"{code_bytes} bytes (e.g. {batch.ids[int(bad[0])]!r}, "
                                        f"{int(sizes[bad[0]])} bytes)")
            codes = batch.data.reshape(-1, code_bytes)
            bounds = batch.offsets[1:-1] // code_bytes
            yield ResidueCodeBatch(batch.ids, np.split(codes, bounds) if len(batch.ids) else [], batch.seq_sha1s)
