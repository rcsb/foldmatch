"""What an existing output file of fm-embedding holds: embeddings, or the residue codes of ``fm-embedding
from-sequences residue|chain --output-codes-folder`` (:func:`file_kind`).

A run refuses to replace a file of the other kind (``foldmatch.utils.output_files.refuse_other_kind``): embeddings in
its codes folder, residue codes in its output folder. Identification is positive only: a file is of a kind when its
first bytes say so, as the writers write it, and anything else (an empty file, a file cut short, one that another rank
of the run is writing, a corrupt or foreign file) is of no kind, None, and is replaced as before. A file that the run
itself writes is always of the right kind, so a run never refuses one of its own files, whatever another rank has
written of it when it is looked at.

Each file is read as little as can tell it: a Parquet file's footer (its schema and metadata); a ``.pt`` file's
tensor, loaded memory-mapped (its type and shape; the values are not read); a ``.csv`` file's first line (its first 4
KiB); a ``.json.gz`` file's first record's keys (its first 16 KiB, decompressed as far as they go). The suffix is
matched in any case, as a case-insensitive folder takes ``X.PT`` for the file ``x.pt`` that a run would write. A
command that checks a folder before the library checks it again remembers what it read (:func:`remembering`), and the
library's check takes each file's kind from there once (while the file is unchanged: its device, inode, size and
modification time) and forgets it: the memory is held from one check to the next only, never for the whole run.

pyarrow, torch, gzip and json are imported only when a file of their format is read.
"""
import contextlib
import json
import os
import re
import zlib
from typing import Optional

#: A file of FoldMatch embeddings: chain, residue, structure or assembly embeddings, in any format (Parquet shards of
#: a build included, whose ``embedding`` column holds chain embeddings).
EMBEDDINGS = "embeddings"
#: A file of fm-embedding's residue codes (``--output-codes-folder``): ``uint8 [L, 8]`` tables.
RESIDUE_CODES = "residue codes"

#: The bytes of one residue's code in a residue-code file.
CODE_BYTES = 8
#: How much of a .csv file is read: a code line takes at most 32 bytes, and an embedding line's first values tell it.
CSV_HEAD = 4096
#: How much of a .json.gz file is read, compressed: the first record's id and the key after it.
JSON_HEAD = 16384

_INTEGER = re.compile(r"[0-9]+")
_FLOAT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[+-]?(?:inf|nan)")
_SIGNED_INTEGER = re.compile(r"[+-]?[0-9]+")

#: (device, inode, size, mtime_ns) -> kind, of the files read within remembering() and not yet read again.
_KNOWN = {}
_remembering = 0


@contextlib.contextmanager
def remembering():
    """Remember the kind of each file read within it (:func:`file_kind`), for the library's check of the same files
    after it: there each is taken from memory once, and forgotten. A command checks its folders within it before it
    calls the library, which checks them again (fm-embedding from-sequences, csv and pt; the legacy chain-embedding and
    complete-embedding); a check that refuses ends the run, and what was remembered is forgotten."""
    global _remembering
    _remembering += 1
    try:
        yield
    except BaseException:
        _KNOWN.clear()
        raise
    finally:
        _remembering -= 1


def file_kind(path) -> Optional[str]:
    """:data:`EMBEDDINGS`, :data:`RESIDUE_CODES`, or None (not positively either, or not readable), by the file's
    format, its name's suffix (``.parquet``, ``.pt``, ``.csv``, ``.json.gz``): :func:`parquet_kind`, :func:`pt_kind`,
    :func:`csv_kind`, :func:`json_kind`; the suffix in any case (``X.PT``). Within :func:`remembering`, the kind is
    remembered; outside it, a remembered kind is taken once and forgotten."""
    name = os.fspath(path)
    lowered = name.lower()
    read = _KINDS.get(next((suffix for suffix in _KINDS if lowered.endswith(suffix)), None))
    if read is None:
        return None
    try:
        stat = os.stat(name)
    except (OSError, ValueError):
        return None
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key in _KNOWN:
        return _KNOWN[key] if _remembering else _KNOWN.pop(key)
    try:
        kind = read(name) if stat.st_size else None
    except Exception:   # noqa: BLE001 - a file that cannot be read is of no kind
        kind = None
    if _remembering:
        _KNOWN[key] = kind
    return kind


def parquet_kind(path) -> Optional[str]:
    """A Parquet file by its footer: residue codes when its metadata ``foldmatch.residue_output`` is
    ``residue_code_table`` (``search.residue_shards.code_table_schema``); embeddings when it has the columns ``id``
    and ``embedding``, ``embedding`` a list of floats (chain, structure, assembly embeddings, and a build's shards) or
    a list of lists of floats (residue embeddings). A file being written has no footer yet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from foldmatch.search.residue_shards import EMBEDDING, ID, META_RESIDUE_OUTPUT, RESIDUE_CODE_TABLE
    schema = pq.read_schema(path)
    if (schema.metadata or {}).get(META_RESIDUE_OUTPUT) == RESIDUE_CODE_TABLE.encode("ascii"):
        return RESIDUE_CODES
    if ID not in schema.names or EMBEDDING not in schema.names:
        return None
    lists = (pa.types.is_list, pa.types.is_large_list, pa.types.is_fixed_size_list)
    kind = schema.field(EMBEDDING).type
    for _ in range(2):
        if not any(is_list(kind) for is_list in lists):
            return None
        kind = kind.value_type
        if pa.types.is_floating(kind):
            return EMBEDDINGS
    return None


def pt_kind(path) -> Optional[str]:
    """A ``.pt`` file by its tensor, loaded memory-mapped on the CPU (``weights_only``; its values are not read):
    residue codes when it is ``uint8 [L, 8]``, embeddings when it is a floating-point tensor. A file being written is
    not yet a whole zip archive, which cannot be loaded."""
    import torch
    tensor = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.dtype == torch.uint8 and tensor.dim() == 2 and tensor.shape[1] == CODE_BYTES:
        return RESIDUE_CODES
    return EMBEDDINGS if tensor.is_floating_point() else None


def csv_kind(path) -> Optional[str]:
    """A ``.csv`` file by its first line (in its first :data:`CSV_HEAD` bytes; no header, as the writers write it):
    residue codes when it is a whole line of 8 integers 0-255; embeddings when its values are numbers, at least one
    of them written as a float ('0.5', '1e-05'), as every float value is (a chain embedding is one value per line, a
    residue embedding one residue per line). A line not ended in the bytes read (cut short, or long) is judged on its
    values but the last, which may be cut; 8 integers are never codes then. An empty file (an empty code table, say)
    is of no kind."""
    with open(path, "rb") as f:
        head = f.read(CSV_HEAD)
    line, ended, _ = head.partition(b"\n")
    try:
        text = line.decode("ascii")
    except UnicodeDecodeError:
        return None
    fields = (text[:-1] if ended and text.endswith("\r") else text).split(",")
    if ended and len(fields) == CODE_BYTES and all(_INTEGER.fullmatch(f) and int(f) <= 255 for f in fields):
        return RESIDUE_CODES
    if not ended:
        fields = fields[:-1]
    if fields and all(_FLOAT.fullmatch(f) for f in fields) and \
            not all(_SIGNED_INTEGER.fullmatch(f) for f in fields):
        return EMBEDDINGS
    return None


def json_kind(path) -> Optional[str]:
    """A ``.json.gz`` file by its first record (orient records, as the writers write it: ``[{"id": ..., "<key>":
    [...]}, ...]``), decompressed from its first :data:`JSON_HEAD` bytes as far as they go: residue codes when the key
    after ``id`` is ``residue_codes``, embeddings when it is ``embedding``, each a list. A file cut short is judged on
    what it holds."""
    with open(path, "rb") as f:
        head = f.read(JSON_HEAD)
    text = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(head, 4 * JSON_HEAD).decode("utf-8", "replace")
    decoder = json.JSONDecoder()
    at = _skip(text, 0)
    for expected in "[{":
        if not text.startswith(expected, at):
            return None
        at = _skip(text, at + 1)
    key, at = decoder.raw_decode(text, at)
    if key != "id":
        return None
    at = _skip(text, _expect(text, _skip(text, at), ":"))
    _, at = decoder.raw_decode(text, at)
    at = _skip(text, _expect(text, _skip(text, at), ","))
    key, at = decoder.raw_decode(text, at)
    at = _skip(text, _expect(text, _skip(text, at), ":"))
    if not text.startswith("[", at):
        return None
    return {"embedding": EMBEDDINGS, "residue_codes": RESIDUE_CODES}.get(key) if isinstance(key, str) else None


def _skip(text: str, at: int) -> int:
    while at < len(text) and text[at] in " \t\r\n":
        at += 1
    return at


def _expect(text: str, at: int, character: str) -> int:
    if not text.startswith(character, at):
        raise ValueError(f"{character!r} expected at {at}")
    return at + 1


_KINDS = {".parquet": parquet_kind, ".pt": pt_kind, ".csv": csv_kind, ".json.gz": json_kind}


def describe(path, kind: str) -> str:
    """A file of a kind, as a refusal names it: 'an embedding' (a .csv or .pt file), 'embeddings' (a Parquet or JSON
    file of many), 'residue codes'."""
    if kind == EMBEDDINGS and os.fspath(path).lower().endswith((".csv", ".pt")):
        return "an embedding"
    return kind
