"""The run record of a split sequence build (plan P8b): what ``build sequences-embed`` leaves for ``build
sequences-index``.

Step 1, ``build sequences-embed``, runs the pre-flight and the ESM3 pass of ``build sequences`` on every rank and
keeps the shards (``emb_<run id>-<rank>.parquet``); rank 0 then checks them and writes, last, this run's record
``emb_<run id>.run.json`` beside them. Step 2, ``build sequences-index``, one process on a CPU node, picks one run by
its record (:func:`find_run`), checks the shards against it (:func:`check_run_shards` and the footers) and the FASTA
(size, sha256 and pre-flight counts, then the id of every shard row at its dealt position, :func:`check_dealt_ids`),
and runs the index phase of ``build sequences`` on them. ``build sequences-seal`` writes the record of complete
shards that have none (a step 1 whose rank 0 stopped after the pass, or the shards a ``build sequences`` kept).

The record:

* is staged as ``emb_<run id>.run.json.tmp`` (created exclusively), verified as step 2 reads it, and committed with a
  hard link that fails if a record exists (:func:`write_run_record`). It is the last file its command writes and is
  never rewritten;
* names its shards by file name, so a copied folder still works. It is no shard (shard names end in
  ``-<rank>.parquet``) and no embedding file of ``build embeddings`` (``.pt``, ``.csv``, ``.parquet``);
* binds the shards to the FASTA by the pass's deal: Lightning's predict sampler gives rank ``r`` of ``W`` the entries
  ``r, r + W, r + 2W, ...`` (``indices[rank::num_replicas]``), and a rank given no entry writes no shard
  (:func:`dealt_rows`).

pyarrow, numpy and the standard library only: no torch. Every refusal is a :class:`~foldmatch.errors.RefusalError`.
"""
from __future__ import annotations

import errno
import glob
import hashlib
import json
import logging
import os
import re
import socket
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from foldmatch.errors import RefusalError
from foldmatch.search.db_path import parse_db_path
from foldmatch.search.residue_code_store import _fsync_dir
from foldmatch.search.residue_shards import (
    _SHARD_PREFIX, BASE_FIELDS, ID, PASS_KEYS, PAYLOAD_TYPE, RESIDUE_CODES, SEQ_SHA1, SEQ_SHA1_TYPE,
    ResidueShardError, _column_types, check_shards, read_shard_info, run_shard_files, shard_run_ids, shard_schema, shards_pass_record,
)
from foldmatch.search.sequence_store import SequenceStore
from foldmatch.utils.dist import RUN_ID_ENV, is_run_id
from foldmatch.utils.fasta import read_id_at, read_record_at
from foldmatch.utils.sequence import seq_sha1

logger = logging.getLogger(__name__)

RUN_RECORD_KIND = "foldmatch-sequence-run"
RUN_RECORD_VERSION = 1
RUN_RECORD_SUFFIX = ".run.json"
#: The commands that write a record (its ``command``); only the seal's is ``sealed``.
EMBED_COMMAND = "build sequences-embed"
SEAL_COMMAND = "build sequences-seal"
#: The pre-flight counts a record holds (:class:`~foldmatch.utils.sequence.FastaPreflight`), which step 2's own
#: pre-flight must reproduce.
PREFLIGHT_KEYS = ("entries", "entries_kept", "residues", "duplicates", "unique_entries", "unique_residues",
                  "nonstandard_entries", "nonstandard_residues", "non_ascii_entries", "angle_bracket_entries",
                  "max_length")
#: The keys of a record's ``fasta`` and ``system`` blocks.
FASTA_KEYS = ("path", "bytes", "mtime_ns", "sha256")
SYSTEM_KEYS = ("name", "digest", "encoding_digest")
BUILD_LOCK_SUFFIX = ".build.lock"

# A record is named as its run's shards are, emb_<run id>, with its own suffix.
_RECORD_PREFIX = _SHARD_PREFIX
_TMP_SUFFIX = ".tmp"
_RECORD_NAME = re.compile(rf"{_RECORD_PREFIX}(?P<run_id>.+){re.escape(RUN_RECORD_SUFFIX)}")
_TMP_RECORD_NAME = re.compile(rf"{_RECORD_PREFIX}(?P<run_id>.+){re.escape(RUN_RECORD_SUFFIX + _TMP_SUFFIX)}")
_SHARD_NAME = re.compile(rf"{_RECORD_PREFIX}(?P<run_id>.+)-(?P<rank>\d+)\.parquet")
_CREATED_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
#: ``hashlib.file_digest`` takes no chunk size: the FASTA is hashed in reads of this many bytes.
HASH_CHUNK = 16 << 20
#: Where a hard link cannot be made, the record is committed with ``os.replace`` after an existence check.
_LINK_UNSUPPORTED = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP})
#: Ids shown in a refusal.
_EXAMPLES = 10
#: Offsets turned into Python ints at a time by :func:`check_dealt_ids`.
_OFFSET_CHUNK = 1 << 16
#: A database's files (``cli.search._SEQUENCE_DB_SUFFIXES``) and its residue-code store's (``{index name}.rescodes.*``).
_DATABASE_SUFFIXES = (".index", ".metadata", ".invlists", ".sequences")
_CODE_STORE_GLOB = ".rescodes.*"

#: The type(s) of every key of a record; ``None`` stands for JSON null.
_RECORD_TYPES = {
    "kind": (str,), "record_version": (int,), "run_id": (str,), "command": (str,), "sealed": (bool,),
    "created_utc": (str,), "foldmatch_version": (str,), "fasta": (dict,), "min_res_n": (int,), "strict": (bool,),
    "preflight": (dict,), "output_db": (str, None), "residue_output": (str, None), "system": (dict, None),
    "world_size": (int,), "pass": (dict,), "options": (dict,), "shards": (list,), "rows": (int,),
}
_FASTA_TYPES = dict(zip(FASTA_KEYS, ((str,), (int,), (int,), (str,))))
_PASS_TYPES = {"min_res_n": (int, None), "world_size": (int, None)}
_SHARD_TYPES = {"name": (str,), "rank": (int,), "rows": (int,), "bytes": (int,)}


def _is(value, kind) -> bool:
    """JSON typing: ``None`` is null, an ``int`` is never a ``bool``."""
    if kind is None:
        return value is None
    if kind is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, kind)


def _type_names(kinds) -> str:
    return " or ".join("null" if kind is None else {dict: "object", list: "array", str: "string", int: "integer",
                                                     bool: "boolean"}[kind] for kind in kinds)


# =====================================================================================================================
# paths and discovery
# =====================================================================================================================
def run_record_path(folder, run_id: str) -> Path:
    """``{folder}/emb_{run_id}.run.json``."""
    if not is_run_id(run_id):
        raise RefusalError(f"{run_id!r} is not a run id: letters, digits, '.', '_' and '-' only, starting with a "
                           f"letter or digit.")
    return Path(folder) / f"{_RECORD_PREFIX}{run_id}{RUN_RECORD_SUFFIX}"


def run_record_tmp_path(folder, run_id: str) -> Path:
    """``{folder}/emb_{run_id}.run.json.tmp``, where the record is staged."""
    path = run_record_path(folder, run_id)
    return path.with_name(path.name + _TMP_SUFFIX)


def _ids_matching(folder, pattern: re.Pattern) -> List[str]:
    names = (path.name for path in Path(folder).glob(f"{_RECORD_PREFIX}*"))
    return sorted({match["run_id"] for match in map(pattern.fullmatch, names)
                   if match and is_run_id(match["run_id"])})


def record_run_ids(folder) -> List[str]:
    """The run ids of the run records in ``folder``, sorted. File names only; ``emb_copy[1].run.json`` names no
    run (only well-formed ids count, :func:`~foldmatch.utils.dist.is_run_id`)."""
    return _ids_matching(folder, _RECORD_NAME)


def tmp_record_run_ids(folder) -> List[str]:
    """The run ids of the staged (``.tmp``) run records in ``folder``, sorted: records whose command stopped before
    committing them."""
    return _ids_matching(folder, _TMP_RECORD_NAME)


def _record_name_id(path: Path) -> Optional[str]:
    """The run id in a record's file name (staged or not), ``None`` if it is not a record's name."""
    match = _RECORD_NAME.fullmatch(path.name) or _TMP_RECORD_NAME.fullmatch(path.name)
    return match["run_id"] if match else None


def _shard_rank(path) -> int:
    match = _SHARD_NAME.fullmatch(Path(path).name)
    if match is None:
        raise RefusalError(f"{path} is not an embedding shard (emb_<run id>-<rank>.parquet).")
    return int(match["rank"])


# =====================================================================================================================
# FASTA facts
# =====================================================================================================================
class FastaStat(NamedTuple):
    """What a stat of the FASTA says about its content: size, modification time and inode. ``st_dev`` is left out:
    the nodes that mount one Lustre file system may report different devices for the same file."""
    bytes: int
    mtime_ns: int
    ino: int


def fasta_stat(path) -> FastaStat:
    st = os.stat(path)
    return FastaStat(st.st_size, st.st_mtime_ns, st.st_ino)


def fasta_sha256(path) -> Tuple[str, float]:
    """``(sha256 hex digest, seconds)`` of the file at ``path``, read in :data:`HASH_CHUNK` pieces."""
    start = time.monotonic()
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), time.monotonic() - start


def preflight_counts(preflight) -> Dict[str, int]:
    """The :data:`PREFLIGHT_KEYS` counts of a :class:`~foldmatch.utils.sequence.FastaPreflight`."""
    return {key: int(getattr(preflight, key)) for key in PREFLIGHT_KEYS}


# =====================================================================================================================
# shard facts
# =====================================================================================================================
def dealt_rows(n: int, world_size: int) -> Dict[int, int]:
    """``{rank: rows}`` of the pass's deal of ``n`` entries over ``world_size`` ranks: rank ``r`` gets entries ``r, r
    + W, ...``, so the ranks that write a shard are exactly ``0 ... min(W, n) - 1``."""
    return {rank: len(range(rank, n, world_size)) for rank in range(min(world_size, n))}


def shard_entries(paths: Sequence) -> List[Dict[str, Any]]:
    """A record's ``shards`` entries, in the order of ``paths``: ``name``, ``rank`` (from the name), ``rows`` (from
    the Parquet footer) and ``bytes``."""
    out = []
    for path in map(Path, paths):
        rank = _shard_rank(path)
        try:
            rows = pq.read_metadata(path).num_rows
            size = path.stat().st_size
        except (pa.ArrowException, OSError) as exc:
            raise RefusalError(f"{path}: not a readable Parquet embedding shard ({exc}).") from exc
        out.append({"name": path.name, "rank": rank, "rows": int(rows), "bytes": int(size)})
    return out


def _read_schema(path) -> pa.Schema:
    try:
        return pq.read_schema(path)
    except (pa.ArrowException, OSError) as exc:
        raise RefusalError(f"{path}: not a readable Parquet embedding shard ({exc}).") from exc


def _listed(paths, limit: int = _EXAMPLES) -> str:
    names = [Path(p).name for p in paths]
    return ", ".join(names[:limit]) + (", ..." if len(names) > limit else "")


def check_plain_footers(paths: Sequence) -> None:
    """Refuse unless every shard is a plain one: the legacy schema ``id | embedding`` (``shard_schema(None)``) with no
    ``foldmatch.*`` metadata, as a pass without ``--residue-codes`` writes it."""
    want = _column_types(shard_schema(None))
    bad = []
    for path in paths:
        schema = _read_schema(path)
        foldmatch_keys = [key for key in (schema.metadata or {}) if key.startswith(b"foldmatch.")]
        if _column_types(schema) != want or foldmatch_keys:
            bad.append((Path(path), schema, foldmatch_keys))
    if bad:
        path, schema, keys = bad[0]
        shown = (f"{path.name} has columns {[name for name, _ in _column_types(schema)]}"
                 + (f" and the metadata {', '.join(k.decode('utf-8', 'replace') for k in keys)}" if keys else ""))
        raise RefusalError(
            f"{len(bad)} of the {len(paths)} shard(s) are not plain embedding shards (columns id | embedding, no "
            f"residue codes): {_listed(p for p, _, _ in bad)}. E.g. {shown}. A run without residue codes has only "
            f"plain shards: these were written by another pass, or changed since."
        )


_CODES_COLUMNS = _column_types(pa.schema(list(BASE_FIELDS) + [(RESIDUE_CODES, PAYLOAD_TYPE),
                                                               (SEQ_SHA1, SEQ_SHA1_TYPE)]))


def _pass_value(record, key):
    return None if not isinstance(record, dict) else record.get(key)


def check_codes_footers(paths: Sequence, *, system=None, expect: Optional[Dict[str, Any]] = None, min_res_n: int,
                        world_size: int) -> Dict[str, Any]:
    """Refuse unless every shard holds the residue codes of one pass with ``min_res_n`` on ``world_size`` ranks;
    returns the shards' pass record (:func:`~foldmatch.search.residue_shards.shards_pass_record`).

    With ``system`` (an ``EsmCodesSystem``) the shards must hold its encoding
    (:func:`~foldmatch.search.residue_shards.check_shards`: mixed and stale shards are refused). With ``expect``, a
    record's ``system`` block, they must form one group of residue-code shards whose system name, digest and
    encoding digest are the block's: a check of the run's self-consistency that needs no registry entry. Exactly one
    of the two is given. Then every shard must record its pass (``foldmatch.pass``) with this ``min_res_n`` and
    ``world_size``.
    """
    paths = [Path(p) for p in paths]
    if (system is None) == (expect is None):
        raise ValueError("check_codes_footers takes exactly one of system and expect")
    if not paths:
        raise RefusalError("There are no shards to check.")
    try:
        infos = check_shards(paths, RESIDUE_CODES, system) if system is not None else list(map(read_shard_info, paths))
    except ResidueShardError as exc:
        raise RefusalError(str(exc)) from exc
    if expect is not None:
        groups: Dict[tuple, list] = {}
        for info in infos:
            groups.setdefault(info.key, []).append(info)
        if len(groups) > 1:
            shown = "; ".join(f"{len(g)} shard(s) with {g[0].describe()} (e.g. {g[0].path.name})"
                              for g in groups.values())
            raise RefusalError(f"Mixed embedding shards: {shown}. The shards of one run come from one pass.")
        info = infos[0]
        found = {"name": info.system, "digest": info.system_digest, "encoding_digest": info.encoding_digest}
        want = {key: (expect or {}).get(key) for key in SYSTEM_KEYS}
        where = f"{len(infos)} shard(s) ({_listed(paths, 3)})"
        if info.residue_output != RESIDUE_CODES or info.column_types != _CODES_COLUMNS:
            raise RefusalError(f"{where} hold {info.describe()} with columns {[n for n, _ in info.column_types]}, "
                               f"not the residue codes of {want['name']}.")
        if found != want:
            raise RefusalError(f"{where} hold the residue codes of {found['name']} (digest {found['digest']}, "
                               f"encoding_digest {found['encoding_digest']}), not those of {want['name']} (digest "
                               f"{want['digest']}, encoding_digest {want['encoding_digest']}).")
    bad = [info for info in infos
           if not (_is(_pass_value(info.pass_record, "min_res_n"), int)
                   and _pass_value(info.pass_record, "min_res_n") == min_res_n
                   and _is(_pass_value(info.pass_record, "world_size"), int)
                   and _pass_value(info.pass_record, "world_size") == world_size)]
    if bad:
        shown = ", ".join(
            f"{info.path.name} (" + ("no pass record" if info.pass_record is None else
                                     f"min_res_n {_pass_value(info.pass_record, 'min_res_n')}, world_size "
                                     f"{_pass_value(info.pass_record, 'world_size')}") + ")"
            for info in bad[:_EXAMPLES])
        raise RefusalError(
            f"{len(bad)} of the {len(infos)} shard(s) do not record a pass with --min-res-n {min_res_n} on "
            f"{world_size} rank(s): {shown}{', ...' if len(bad) > _EXAMPLES else ''}."
        )
    return shards_pass_record(infos)


# =====================================================================================================================
# record I/O
# =====================================================================================================================
def refuse_existing_record(folder, run_id: str, *, command: str = EMBED_COMMAND) -> None:
    """Refuse if ``folder`` holds a run record of ``run_id``, or its staged ``.tmp``. Read-only.

    The advice is ``command``'s: step 1 (:data:`EMBED_COMMAND`) found an earlier run with its id; the seal
    (:data:`SEAL_COMMAND`) found the run it would seal recorded already (use the record), or a record whose writer
    stopped (delete the ``.tmp``, then seal again)."""
    path = run_record_path(folder, run_id)
    sealing = command == SEAL_COMMAND
    if path.exists():
        if sealing:
            raise RefusalError(
                f"{path} exists: run {run_id} in {folder} has its run record already, so there is nothing to seal. "
                f"Use it: 'build sequences-index --from-shards {folder} --run-id {run_id}' builds from it."
            )
        raise RefusalError(
            f"{path} exists: a run record of an earlier run with the same run id ({run_id}) is in {folder}. Remove "
            f"it and its shards, use another --shards-folder, or set a different {RUN_ID_ENV}."
        )
    tmp = run_record_tmp_path(folder, run_id)
    if tmp.exists() and sealing:
        raise RefusalError(
            f"{tmp} exists: the run record of run {run_id} was being written when its command stopped. Delete the "
            f".tmp, then run 'build sequences-seal' again."
        )
    if tmp.exists():
        raise RefusalError(
            f"{tmp} exists: an earlier run with the same run id ({run_id}) stopped while it wrote its run record. "
            f"Delete the .tmp (and that run's shards, unless you seal them), use another --shards-folder, or set a "
            f"different {RUN_ID_ENV}."
        )


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def write_run_record(folder, record: Dict[str, Any]) -> Path:
    """Write ``record`` as ``{folder}/emb_{run id}.run.json`` and return its path.

    The record is staged in ``emb_{run id}.run.json.tmp``, created exclusively (an existing ``.tmp`` is refused and
    left alone), as sorted, indented JSON, and fsynced; the staged file is verified as step 2 reads it
    (:func:`read_run_record` and :func:`check_run_shards`), then committed with ``os.link``, which fails if a record
    exists; the ``.tmp`` is unlinked and the folder fsynced. Where hard links are not supported, an existing record
    is refused and the ``.tmp`` is renamed into place (``os.replace``). Any failure removes the ``.tmp`` and raises a
    :class:`RefusalError` naming the file.
    """
    folder = Path(folder)
    run_id = record.get("run_id") if isinstance(record, dict) else None
    final = run_record_path(folder, run_id)
    tmp = run_record_tmp_path(folder, run_id)
    try:
        text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise RefusalError(f"The run record {final} cannot be written as JSON ({exc}).") from exc
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        raise RefusalError(f"The run record cannot be staged in {tmp} ({exc}).") from exc
    try:
        try:
            _write_all(fd, text.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            check_run_shards(folder, read_run_record(tmp))
        except RefusalError as exc:
            raise RefusalError(f"The staged run record {tmp} failed its check: {exc}") from exc
        try:
            os.link(tmp, final)
        except OSError as exc:
            if exc.errno not in _LINK_UNSUPPORTED:
                raise
            if final.exists():
                raise FileExistsError(errno.EEXIST, "a run record exists", str(final)) from exc
            logger.info(f"{folder} does not support hard links ({exc.strerror}): the run record is renamed into "
                        f"place instead")
            os.replace(tmp, final)
    except BaseException as exc:
        tmp.unlink(missing_ok=True)
        if isinstance(exc, RefusalError):
            raise
        if isinstance(exc, OSError):
            raise RefusalError(f"The run record {final} cannot be written ({exc}); {tmp} was removed.") from exc
        raise
    try:
        tmp.unlink(missing_ok=True)
    except OSError as exc:  # the record is committed: a leftover .tmp is only reported
        logger.warning(f"Could not remove {tmp} after committing {final} ({exc}); delete it.")
    _fsync_dir(folder)
    return final


def _bad(path, problem: str) -> RefusalError:
    return RefusalError(f"{path} is not a valid run record: {problem}.")


def _check_fields(path, block: Dict[str, Any], types: Dict[str, tuple], where: str) -> None:
    for key, kinds in types.items():
        if key not in block:
            raise _bad(path, f"the key {where}{key} is missing")
        if not any(_is(block[key], kind) for kind in kinds):
            raise _bad(path, f"{where}{key} is {block[key]!r}, expected {_type_names(kinds)}")


def read_run_record(path) -> Dict[str, Any]:
    """The run record at ``path``, checked; refused (naming the file and the key) if it cannot be read, is not a
    JSON object, has another ``kind`` or ``record_version`` (a larger one: written by a newer FoldMatch), lacks a key
    or has one of a wrong type, a negative count, or fields that disagree: the run id and the file name, the strict
    mode and the residue output, the command and ``sealed``, the pass's ``min_res_n`` and ``world_size`` and the
    record's, the shard rows and ``rows`` and the pre-flight's kept entries, and the shards and the deal of those
    entries over ``world_size`` ranks (:func:`dealt_rows`). Unknown keys are ignored."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusalError(f"The run record {path} cannot be read ({exc}).") from exc
    try:
        record = json.loads(text)
    except ValueError as exc:
        raise _bad(path, f"it is not JSON ({exc})") from exc
    if not isinstance(record, dict):
        raise _bad(path, "it is not a JSON object")
    if record.get("kind") != RUN_RECORD_KIND:
        raise _bad(path, f"kind is {record.get('kind')!r}, not {RUN_RECORD_KIND!r}")
    version = record.get("record_version")
    if _is(version, int) and version > RUN_RECORD_VERSION:
        raise RefusalError(f"The run record {path} was written by a newer FoldMatch (record_version {version}; this "
                           f"FoldMatch reads version {RUN_RECORD_VERSION}): upgrade FoldMatch.")
    if not _is(version, int) or version != RUN_RECORD_VERSION:
        raise _bad(path, f"record_version is {version!r}, not {RUN_RECORD_VERSION}")
    _check_fields(path, record, _RECORD_TYPES, "")
    _check_fields(path, record["fasta"], _FASTA_TYPES, "fasta.")
    _check_fields(path, record["preflight"], dict.fromkeys(PREFLIGHT_KEYS, (int,)), "preflight.")
    _check_fields(path, record["pass"], {key: _PASS_TYPES.get(key, (str, None)) for key in PASS_KEYS}, "pass.")
    if record["system"] is not None:
        _check_fields(path, record["system"], dict.fromkeys(SYSTEM_KEYS, (str,)), "system.")
    for k, shard in enumerate(record["shards"]):
        if not isinstance(shard, dict):
            raise _bad(path, f"shards[{k}] is {shard!r}, expected object")
        _check_fields(path, shard, _SHARD_TYPES, f"shards[{k}].")

    run_id = record["run_id"]
    if not is_run_id(run_id):
        raise _bad(path, f"run_id {run_id!r} is not a run id")
    name_id = _record_name_id(path)
    if name_id != run_id:
        raise _bad(path, f"run_id is {run_id!r}, but the file is named for "
                         + (f"run {name_id!r}" if name_id else "no run (emb_<run id>.run.json)"))
    if record["command"] not in (EMBED_COMMAND, SEAL_COMMAND):
        raise _bad(path, f"command is {record['command']!r}, not {EMBED_COMMAND!r} or {SEAL_COMMAND!r}")
    if record["sealed"] != (record["command"] == SEAL_COMMAND):
        raise _bad(path, f"sealed is {record['sealed']} for the command {record['command']!r}")
    try:
        datetime.strptime(record["created_utc"], _CREATED_FORMAT)
    except ValueError as exc:
        raise _bad(path, f"created_utc {record['created_utc']!r} is not {_CREATED_FORMAT}") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", record["fasta"]["sha256"]):
        raise _bad(path, f"fasta.sha256 {record['fasta']['sha256']!r} is not a sha256 hex digest")
    counts = {"min_res_n": record["min_res_n"], "rows": record["rows"], "fasta.bytes": record["fasta"]["bytes"],
              **{f"preflight.{key}": value for key, value in record["preflight"].items() if key in PREFLIGHT_KEYS},
              **{f"shards[{k}].{key}": shard[key] for k, shard in enumerate(record["shards"])
                 for key in ("rank", "rows", "bytes")}}
    negative = [key for key, value in counts.items() if value < 0]
    if negative:
        raise _bad(path, f"{negative[0]} is negative ({counts[negative[0]]})")
    if record["world_size"] < 1:
        raise _bad(path, f"world_size is {record['world_size']}, not at least 1")
    if record["residue_output"] not in (None, RESIDUE_CODES):
        raise _bad(path, f"residue_output is {record['residue_output']!r}, not {RESIDUE_CODES!r} or null")
    codes = record["residue_output"] is not None
    if record["strict"] != codes:
        raise _bad(path, f"strict is {record['strict']}, but residue_output is {record['residue_output']!r}")
    if (record["system"] is not None) != codes:
        raise _bad(path, f"system is {'set' if record['system'] is not None else 'null'}, but residue_output is "
                         f"{record['residue_output']!r}")
    for key in ("min_res_n", "world_size"):
        if record["pass"][key] != record[key]:
            raise _bad(path, f"pass.{key} is {record['pass'][key]!r}, but {key} is {record[key]}")
    preflight = record["preflight"]
    if preflight["unique_entries"] != preflight["entries_kept"] - preflight["duplicates"]:
        raise _bad(path, f"preflight.unique_entries is {preflight['unique_entries']}, not entries_kept - "
                         f"duplicates ({preflight['entries_kept'] - preflight['duplicates']})")
    if preflight["entries_kept"] > preflight["entries"]:
        raise _bad(path, f"preflight.entries_kept ({preflight['entries_kept']}) exceeds preflight.entries "
                         f"({preflight['entries']})")
    shards = record["shards"]
    if sum(shard["rows"] for shard in shards) != record["rows"]:
        raise _bad(path, f"rows is {record['rows']}, but the shards hold {sum(s['rows'] for s in shards)}")
    if record["rows"] != preflight["entries_kept"]:
        raise _bad(path, f"rows is {record['rows']}, but preflight.entries_kept is {preflight['entries_kept']}")
    names = [shard["name"] for shard in shards]
    for k, shard in enumerate(shards):
        want = f"{_RECORD_PREFIX}{run_id}-{shard['rank']}.parquet"
        if shard["name"] != want:
            raise _bad(path, f"shards[{k}] is named {shard['name']!r}, not {want!r} (run {run_id}, rank "
                             f"{shard['rank']})")
    if len(set(names)) != len(names) or names != sorted(names):
        raise _bad(path, "the shards are not listed once each, in file-name order")
    deal = dealt_rows(record["rows"], record["world_size"])
    found = {shard["rank"]: shard["rows"] for shard in shards}
    if found != deal:
        raise _bad(path, f"the shards ({_deal_text(found)}) are not the deal of {record['rows']} entries over "
                         f"{record['world_size']} rank(s) ({_deal_text(deal)})")
    return record


def _deal_text(rows: Dict[int, int]) -> str:
    if not rows:
        return "no shard"
    return ", ".join(f"rank {rank}: {n} row(s)" for rank, n in sorted(rows.items())[:_EXAMPLES]) + (
        ", ..." if len(rows) > _EXAMPLES else "")


# =====================================================================================================================
# selection and checks
# =====================================================================================================================
class _RunState(NamedTuple):
    shards: int
    record: bool
    tmp: bool

    def describe(self) -> str:
        if self.record:
            return f"run record, {self.shards} shard(s)"
        if self.tmp:
            return f"{self.shards} shard(s), an unfinished run record (.tmp)"
        return f"{self.shards} shard(s), no run record"


def _runs(folder: Path) -> Dict[str, _RunState]:
    """Every run in ``folder``: a well-formed run id with shards, a record or a staged record."""
    records, tmps = set(record_run_ids(folder)), set(tmp_record_run_ids(folder))
    ids = set(shard_run_ids(folder)) | records | tmps
    return {rid: _RunState(len(run_shard_files(folder, rid)), rid in records, rid in tmps) for rid in sorted(ids)}


def _seal_command(folder: Path, run_id: str) -> str:
    return (f"'fm-search build sequences-seal --fasta-file FASTA --from-shards {folder} --run-id {run_id}' (with "
            f"--min-res-n for a run without residue codes)")


def find_run(folder, run_id: Optional[str] = None) -> Tuple[str, Path]:
    """``(run id, record path)`` of the run step 2 builds from (plan D33).

    With ``run_id``, that run, which must have a record. Without, the folder must hold exactly one run of any kind
    (shards, a record or a staged ``.tmp``), and it must have a record: a run without one did not finish (or its
    shards were kept by ``build sequences``), and building silently from an older run beside it is refused.
    """
    folder = Path(folder)
    runs = _runs(folder)
    if run_id is not None:
        path = run_record_path(folder, run_id)
        if path.is_file():
            return run_id, path
        state = runs.get(run_id)
        with_records = [rid for rid, s in runs.items() if s.record]
        if state is None:
            what = "neither a run record nor embedding shards"
        else:
            what = state.describe()
        advice = (f" If its shards are complete, {_seal_command(folder, run_id)} records them."
                  if state is not None and state.shards else "")
        if state is not None and state.tmp:
            advice = f" Delete {run_record_tmp_path(folder, run_id)} first.{advice}"
        raise RefusalError(
            f"Run {run_id} has no run record ({path.name}) in {folder}: {what}.{advice} Runs with a record here: "
            f"{', '.join(with_records) if with_records else 'none'}."
        )
    if not runs:
        raise RefusalError(
            f"No run in {folder}: no run record (emb_<run id>.run.json) and no embedding shards. Pass the "
            f"--shards-folder of a 'build sequences-embed' run."
        )
    if len(runs) > 1:
        listed = "; ".join(f"{rid} ({state.describe()})" for rid, state in runs.items())
        raise RefusalError(
            f"{folder} holds {len(runs)} runs: {listed}. Pass --run-id to choose one ('build sequences-embed' logs "
            f"its run id; its record is emb_<run id>.run.json)."
        )
    (rid, state), = runs.items()
    if state.record:
        return rid, run_record_path(folder, rid)
    first = f"delete {run_record_tmp_path(folder, rid)} first, then " if state.tmp else ""
    raise RefusalError(
        f"Run {rid} in {folder} has no run record ({state.describe()}): it did not finish, or its shards were kept "
        f"by 'build sequences'. If they are complete, {first}run {_seal_command(folder, rid)}; otherwise run 'build "
        f"sequences-embed' again."
    )


def check_run_shards(folder, record: Dict[str, Any]) -> List[Path]:
    """The shards of the record's run in ``folder``, in the order step 2 reads them (:func:`run_shard_files`), after
    checking that they are exactly the recorded ones, with the recorded size and rows (from the footers)."""
    folder = Path(folder)
    run_id = record["run_id"]
    paths = run_shard_files(folder, run_id)
    recorded = {shard["name"]: shard for shard in record["shards"]}
    found = {path.name for path in paths}
    missing, extra = sorted(set(recorded) - found), sorted(found - set(recorded))
    if missing or extra:
        raise RefusalError(
            f"The shards of run {run_id} in {folder} are not the {len(recorded)} its record lists: "
            + "; ".join(part for part in (f"missing {_listed(missing)}" if missing else "",
                                          f"not in the record {_listed(extra)}" if extra else "") if part)
            + ". They changed since the run (a purge, a partial copy, or shards of another run with the same id)."
        )
    changed = []
    for path in paths:
        shard = recorded[path.name]
        try:
            size = path.stat().st_size
            rows = pq.read_metadata(path).num_rows
        except (pa.ArrowException, OSError) as exc:
            raise RefusalError(f"{path}: the footer of a shard of run {run_id} cannot be read ({exc}); it changed "
                               f"since the run.") from exc
        if size != shard["bytes"] or rows != shard["rows"]:
            changed.append(f"{path.name} ({size} bytes, {rows} rows; recorded {shard['bytes']} bytes, "
                           f"{shard['rows']} rows)")
    if changed:
        raise RefusalError(f"{len(changed)} shard(s) of run {run_id} in {folder} changed since the run: "
                           f"{', '.join(changed[:_EXAMPLES])}{', ...' if len(changed) > _EXAMPLES else ''}.")
    return paths


def _sha1_rows(column: pa.ChunkedArray, path: Path) -> np.ndarray:
    """A shard's ``seq_sha1`` column as a ``uint8`` array of ``[rows, 20]`` (20 B per row, no Python objects)."""
    array = column.combine_chunks()
    if array.type != SEQ_SHA1_TYPE or array.null_count:
        raise RefusalError(f"{path}: its seq_sha1 column is not {SEQ_SHA1_TYPE} without nulls ({array.type}, "
                           f"{array.null_count} null(s)).")
    width = SEQ_SHA1_TYPE.byte_width
    if not len(array):
        return np.zeros((0, width), np.uint8)
    data = np.frombuffer(array.buffers()[1], dtype=np.uint8)
    return data[array.offset * width:(array.offset + len(array)) * width].reshape(-1, width)


def check_dealt_ids(paths: Sequence, record: Dict[str, Any], preflight, *, sequences: bool = False) -> float:
    """Refuse unless every shard row holds the id of the FASTA entry the pass's deal gave it: kept entry ``k`` (in
    the order of ``preflight.offsets``) is row ``k // W`` of rank ``k % W``'s shard, ``W`` the record's
    ``world_size``. One sequential pass over the kept records, each shard's id column loaded once (about 100 B per
    entry). Returns the seconds it took.

    With ``sequences`` (residue-code shards), every row's ``seq_sha1`` must also be the
    :func:`~foldmatch.utils.sequence.seq_sha1` of its entry's sequence, read in the same pass (20 B more per row):
    shards computed from another FASTA with the same ids, order and lengths are refused too. Their pre-flight was
    strict, so every sequence is ASCII."""
    start = time.monotonic()
    world = record["world_size"]
    total = preflight.entries_kept
    by_rank: Dict[int, list] = {}
    sha1s: Dict[int, Any] = {}
    names: Dict[int, str] = {}
    for path in map(Path, paths):
        rank = _shard_rank(path)
        try:
            table = pq.read_table(path, columns=[ID, SEQ_SHA1] if sequences else [ID])
            by_rank[rank] = table.column(ID).to_pylist()
            if sequences:
                sha1s[rank] = _sha1_rows(table.column(SEQ_SHA1), path)
        except (pa.ArrowException, OSError, KeyError) as exc:
            what = "ids and seq_sha1" if sequences else "ids"
            raise RefusalError(f"{path}: the {what} of a shard of run {record['run_id']} cannot be read ({exc}).") \
                from exc
        names[rank] = path.name
    found = {rank: len(ids) for rank, ids in by_rank.items()}
    deal = dealt_rows(total, world)
    if found != deal:
        raise RefusalError(
            f"The shards of run {record['run_id']} ({_deal_text(found)}) are not the deal of the {total} entries of "
            f"{preflight.path} over {world} rank(s) ({_deal_text(deal)})."
        )
    mismatches, shown = 0, []
    sha1_mismatches, sha1_shown = 0, []
    offsets = preflight.offsets
    with open(preflight.path, "r") as fh:
        for begin in range(0, total, _OFFSET_CHUNK):
            for k, offset in enumerate(offsets[begin:begin + _OFFSET_CHUNK].tolist(), begin):
                rank, row = k % world, k // world
                got = by_rank[rank][row]
                if sequences:
                    want, sequence = read_record_at(fh, offset)
                else:
                    want = read_id_at(fh, offset)
                if got != want:
                    mismatches += 1
                    if len(shown) < _EXAMPLES:
                        shown.append(f"{names[rank]} row {row} holds {got!r}, entry {k} is {want!r}")
                elif sequences and sha1s[rank][row].tobytes() != seq_sha1(sequence):
                    sha1_mismatches += 1
                    if len(sha1_shown) < _EXAMPLES:
                        sha1_shown.append(f"{names[rank]} row {row} ({got!r}, entry {k})")
    if mismatches:
        raise RefusalError(
            f"{mismatches} shard row(s) of run {record['run_id']} do not hold the id of the entry of "
            f"{preflight.path} the pass dealt them: {'; '.join(shown)}{'; ...' if mismatches > len(shown) else ''}. "
            f"These shards were not computed from this FASTA in this order."
        )
    if sha1_mismatches:
        raise RefusalError(
            f"{sha1_mismatches} shard row(s) of run {record['run_id']} hold a seq_sha1 that is not the sha1 of the "
            f"sequence of the entry of {preflight.path} the pass dealt them, though they hold its id: "
            f"{'; '.join(sha1_shown)}{'; ...' if sha1_mismatches > len(sha1_shown) else ''}. These shards were not "
            f"computed from this FASTA's sequences."
        )
    return time.monotonic() - start


# =====================================================================================================================
# the build lock
# =====================================================================================================================
def _database_files(db_path: str) -> List[Path]:
    db_folder, index_name, _ = parse_db_path(db_path)
    present = [Path(f"{db_path}{suffix}") for suffix in _DATABASE_SUFFIXES]
    present = [path for path in present if path.exists()]
    present.extend(sorted(Path(db_folder).glob(f"{glob.escape(index_name)}{_CODE_STORE_GLOB}")))
    return present


def _only_code_store_pending(db_path: str) -> bool:
    """The database's ``.index``, ``.metadata`` and complete ``.sequences`` exist, and no residue-code manifest."""
    db_folder, index_name, _ = parse_db_path(db_path)
    if not all(Path(f"{db_path}{suffix}").exists() for suffix in (".index", ".metadata")):
        return False
    if Path(f"{db_path}.rescodes.json").exists():
        return False
    store = SequenceStore(Path(db_folder), index_name)
    if not store.exists():
        return False
    try:
        store.meta()
    except (ValueError, OSError, sqlite3.DatabaseError):
        return False
    return True


@contextmanager
def build_lock(db_path, record: Dict[str, Any], *, shards_folder=None) -> Iterator[Path]:
    """Hold ``{db_path}.build.lock`` (``db_path``: :func:`~foldmatch.search.db_path.parse_db_path`'s resolved path)
    for a step-2 build of the record's run.

    The lock is created exclusively and holds the run id, the host, the pid, ``SLURM_JOB_ID`` and the start time. It
    is removed when the block ends, also on an exception (which then propagates), so it stays only when the process
    is killed (wall time, out of memory). An existing lock is refused with its content, the database files present
    and the recovery; when only the residue-code store is missing from an otherwise complete database, the refusal
    also names the ``build residue-codes`` command that adds it (``shards_folder`` is the run's folder).
    """
    db_path = str(db_path)
    lock = Path(f"{db_path}{BUILD_LOCK_SUFFIX}")
    content = {
        "run_id": record["run_id"],
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID") or None,
        "started_utc": datetime.now(timezone.utc).strftime(_CREATED_FORMAT),
    }
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise RefusalError(_lock_refusal(lock, db_path, record, shards_folder)) from None
    except OSError as exc:
        raise RefusalError(f"The build lock {lock} cannot be created ({exc}).") from exc
    try:
        try:
            _write_all(fd, (json.dumps(content, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            os.fsync(fd)
            own = os.fstat(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        lock.unlink(missing_ok=True)
        raise RefusalError(f"The build lock {lock} cannot be written ({exc}).") from exc
    try:
        yield lock
    finally:
        try:
            current = os.stat(lock)
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (own.st_dev, own.st_ino):
            lock.unlink()


def code_store_command(db_path, record: Dict[str, Any], shards_folder=None) -> Optional[str]:
    """The ``fm-search build residue-codes`` command that adds the residue-code store to the database at ``db_path``
    from the record's run (``shards_folder`` holds its shards), when the run computed residue codes and the database
    is complete but for that store (a step-2 build stopped after its sequence store); else ``None``."""
    db_path = str(db_path)
    if record.get("residue_output") != RESIDUE_CODES or not _only_code_store_pending(db_path):
        return None
    db_folder, index_name, _ = parse_db_path(db_path)
    return (f"fm-search build residue-codes {db_folder} {index_name} --from-shards "
            f"{shards_folder if shards_folder is not None else 'SHARDS'} --run-id {record['run_id']}")


def _lock_refusal(lock: Path, db_path: str, record: Dict[str, Any], shards_folder) -> str:
    try:
        held = " ".join(lock.read_text(encoding="utf-8").split())
    except (OSError, UnicodeDecodeError) as exc:
        held = f"unreadable: {exc}"
    files = _database_files(db_path)
    message = (
        f"{lock} exists ({held}): a 'build sequences-index' of {db_path} is running, or was killed. Database files "
        f"present: {', '.join(path.name for path in files) if files else 'none'}. If no such job runs, delete the "
        f"lock and these files, then rerun."
    )
    command = code_store_command(db_path, record, shards_folder)
    if command is not None:
        message += (
            f" The database is complete but for its residue-code store: '{command}' adds it (delete the lock first)."
        )
    return message
