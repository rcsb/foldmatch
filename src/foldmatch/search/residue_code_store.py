"""Residue-code store: the per-residue ESM3 codes of a sequence database (plan section 8 P5.5, Appendix B; D10).

Three files sit next to a FAISS database ``P`` = ``{db_folder}/{index_name}`` (the path split of
:func:`foldmatch.search.db_path.parse_db_path`, as for ``P.sequences``):

* ``P.rescodes.bin``: raw ``uint8[N_res, code_bytes]`` (8 bytes per residue for esm-codes-v1, method spec S4), the
  entries' codes back to back in the order :meth:`ResidueCodeStore.create` received them. Read it with
  :meth:`ResidueCodeStore.open_memmap`.
* ``P.rescodes.idx``: SQLite, ``entries(id TEXT PRIMARY KEY, offset INTEGER, length INTEGER, seq_sha1 BLOB)``
  (``WITHOUT ROWID``). ``offset`` and ``length`` count residues, i.e. rows of the ``.bin``, so the codes of entry
  ``id`` are ``codes[offset:offset + length]``. ``seq_sha1`` is :func:`foldmatch.utils.sequence.seq_sha1` of the
  sequence the codes were computed from (20 bytes). A ``meta`` table holds the build's totals and digests; create()
  writes it last, so an idx without it is the remains of an interrupted build.
* ``P.rescodes.json``: the manifest (Appendix B). It is the commit point. :meth:`ResidueCodeStore.validate` writes it
  last, only after every check passed, through a temporary file, fsync and ``os.replace``. Without it the store is
  partial (:meth:`ResidueCodeStore.is_partial`) and nothing reads it.

Lifecycle (plan section 3, BUILD; rank 0, after the ESM3 pass and the FAISS build)::

    store.create(batches, system)                       # P.rescodes.{bin,idx}.tmp, fsynced: the pending build
    store.validate(sequence_store, faiss_ids_count, preflight=...)   # checks (a)-(e), then the swap, manifest last

:meth:`ResidueCodeStore.create` writes the new codes aside, as ``P.rescodes.bin.tmp`` and ``P.rescodes.idx.tmp``, so a
complete store it replaces (``force``) stays whole and usable until the new codes have passed validation. Only then
does :meth:`ResidueCodeStore.validate` swap them in: the new manifest is written to ``P.rescodes.json.tmp`` and
fsynced, the old manifest is removed, the two files are renamed into place, and the new manifest is renamed last. A
failed validation, or a failed or interrupted create(), discards the pending files only: the store it was to replace
(if any), the sequence store and FAISS are untouched, so the database still serves BLOSUM62, and its codes too when it
had them. A query calls :meth:`ResidueCodeStore.check_compatible` on every rank before ESM3 (read-only and cheap), then
:meth:`ResidueCodeStore.check_faiss_count` on rank 0 once the index is loaded.

Digests (recorded in the manifest):

* ``bin_sha256``: sha256 of the ``.bin`` bytes, computed while they are written.
* ``entries_digest``: sha256 over ``ENTRIES_DIGEST_TAG`` followed, for every entry in ascending id order (the order of
  the ids' UTF-8 bytes, which is SQLite's ``BINARY`` order), by the uint32 little-endian byte length of the UTF-8 id,
  the id, and its 20-byte ``seq_sha1`` (:func:`entries_digest`). It does not depend on the order of the entries.

No torch: Stage-2 workers open the memmap. Only the ``_INSTALLED`` defaults that name this installation's ESM3 pin
(``create(esm_revision=...)``, ``check_compatible(esm_weights_sha256=...)``) import ``foldmatch.utils.esm.loaders``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import platform
import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from foldmatch.search.db_path import parse_db_path
from foldmatch.search.scoring_system import ENCODING_FIELDS, EsmCodesSystem
from foldmatch.search.scoring_system import get as get_system
from foldmatch.search.sequence_store import _INSERT_BATCH, _SQLITE_VAR_CHUNK, SequenceStore
from foldmatch.utils.sequence import STANDARD_AMINO_ACIDS

logger = logging.getLogger(__name__)

RESCODES_BIN_SUFFIX = ".rescodes.bin"
RESCODES_IDX_SUFFIX = ".rescodes.idx"
RESCODES_MANIFEST_SUFFIX = ".rescodes.json"
RESCODES_SUFFIXES = (RESCODES_BIN_SUFFIX, RESCODES_IDX_SUFFIX, RESCODES_MANIFEST_SUFFIX)
_TMP_SUFFIX = ".tmp"
_SQLITE_SIDECARS = ("-journal", "-wal", "-shm")

MANIFEST_VERSION = 1
MANIFEST_KIND = "foldmatch-residue-codes"
IDX_FORMAT = "foldmatch-residue-codes-idx/1"
ENTRIES_DIGEST_TAG = b"foldmatch-residue-codes-entries/1\n"
SEQ_SHA1_BYTES = 20

#: Column names of a pyarrow batch given to :meth:`ResidueCodeStore.create` (the build shards' columns, P5.3).
ID_COLUMN = "id"
CODES_COLUMN = "residue_codes"
SEQ_SHA1_COLUMN = "seq_sha1"

# Ids quoted per failed check (validate) or per refused lookup.
_EXAMPLE_IDS = 10

_ESM_KEYS = ("repo", "revision", "weights_sha256")
_BUNDLE_KEYS = ("repo", "revision", "files")
_CROSS_CHECK_KEYS = ("preflight_entries", "preflight_residues", "sequence_store_entries", "sequence_store_residues",
                     "faiss_entries", "seq_sha1_mismatches", "length_mismatches")
_BUILD_KEYS = ("foldmatch_version", "created_utc", "run_id", "from_shards", "accelerator", "devices", "esm_precision",
               "encoder_precision", "torch", "esm", "platform")
#: Every key of a version-1 manifest (Appendix B). check_compatible() refuses a manifest that lacks any of them,
#: at the top level or inside the blocks of ``_NESTED_KEYS``.
MANIFEST_KEYS = ("manifest_version", "kind", "system_name", "system_digest", "encoding_digest", "system",
                 "code_bytes", "bitorder", "esm", "bundle", "entries", "total_residues", "min_res_n", "bin_sha256",
                 "entries_digest", "nonstandard_entries", "input_normalisation", "cross_checks", "build")
_NESTED_KEYS = {
    "system": tuple(f.name for f in dataclasses.fields(EsmCodesSystem)),
    "esm": _ESM_KEYS,
    "bundle": _BUNDLE_KEYS,
    "cross_checks": _CROSS_CHECK_KEYS,
    "build": _BUILD_KEYS,
}
# What create() records in the idx meta table, for validate() to put in the manifest. Written in one transaction
# after the last entry, so either all are present or the idx is incomplete.
_PENDING_KEYS = ("store_format", "system_json", "esm_revision", "min_res_n", "build_json", "entries",
                 "total_residues", "bin_sha256", "entries_digest")

_STANDARD_BYTES = STANDARD_AMINO_ACIDS.encode("ascii")
_REBUILD_HINT = ("Rebuild the codes from the kept build shards with "
                 "'fm-search build residue-codes DB_FOLDER INDEX_NAME --from-shards DIR', "
                 "or rebuild the database with 'fm-search build sequences --residue-codes'")
# For entries the ESM3 pass could not encode (empty payloads): the same shards would fail again.
_FRESH_BUILD_HINT = ("The empty payloads were written by the ESM3 pass itself, so rebuilding from the same shards "
                     "would fail the same way: rebuild the database with 'fm-search build sequences "
                     "--residue-codes' (and check those entries' sequences)")


class ResidueCodeStoreError(ValueError):
    """The residue-code store refused: malformed input to create(), a duplicate or unknown id, a partial store.

    A ``ValueError``, so the command-line tools report it as a one-line error."""


class CodeStoreValidationError(ResidueCodeStoreError):
    """:meth:`ResidueCodeStore.validate` found faults (and removed the code store).

    ``failures`` maps each failed check, ``"a"`` to ``"e"``, to ``(count, examples)``: for (a)-(d) the number of
    entries and the first ids (at most 10, ascending); for (e) the number of disagreeing totals and their
    descriptions."""

    def __init__(self, message: str, failures: Mapping[str, Tuple[int, Tuple[str, ...]]]):
        super().__init__(message)
        self.failures = dict(failures)


class IncompatibleCodeStoreError(ResidueCodeStoreError):
    """The residue-code store cannot be used with this system, sequence store or installation (check_compatible)."""


class _Installed:
    """Type of :data:`_INSTALLED`."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "_INSTALLED"


#: Default of every check_compatible() argument: compare with what this installation uses. An omitted argument is
#: therefore never a skipped check, and ``None`` is refused (TypeError) instead of meaning "don't check".
_INSTALLED = _Installed()


class CodeEntry(NamedTuple):
    """One entry of the index: its codes are ``codes[offset:offset + length]`` of :meth:`open_memmap`."""
    offset: int
    length: int
    seq_sha1: bytes


def entries_digest(pairs: Iterable[Tuple[str, bytes]]) -> str:
    """``entries_digest`` (module docstring) of ``(id, seq_sha1)`` pairs given in any order."""
    return _entries_digest_sorted(sorted(pairs, key=lambda pair: pair[0].encode("utf-8")))


def _entries_digest_sorted(rows: Iterable[Tuple[str, bytes]]) -> str:
    h = hashlib.sha256(ENTRIES_DIGEST_TAG)
    for entry_id, sha1 in rows:
        data = entry_id.encode("utf-8")
        h.update(struct.pack("<I", len(data)))
        h.update(data)
        h.update(sha1)
    return h.hexdigest()


class ResidueCodeStore:
    """The ``P.rescodes.{bin,idx,json}`` store of a database (module docstring).

    ``exists()`` means complete: the manifest is there. Readers (:meth:`lookup`, :meth:`open_memmap`,
    :meth:`check_compatible`) refuse a store without it."""

    def __init__(self, db_folder: Path, index_name: str):
        self.db_folder = Path(db_folder)
        self.index_name = index_name
        base = self.db_folder / index_name
        self.base = base
        self.bin_path = Path(f"{base}{RESCODES_BIN_SUFFIX}")
        self.idx_path = Path(f"{base}{RESCODES_IDX_SUFFIX}")
        self.manifest_path = Path(f"{base}{RESCODES_MANIFEST_SUFFIX}")
        # The pending build: create() writes these, validate() checks them and renames them into place.
        self.tmp_bin_path = Path(f"{self.bin_path}{_TMP_SUFFIX}")
        self.tmp_idx_path = Path(f"{self.idx_path}{_TMP_SUFFIX}")
        self.tmp_manifest_path = Path(f"{self.manifest_path}{_TMP_SUFFIX}")

    @classmethod
    def from_db_path(cls, db_path: str) -> "ResidueCodeStore":
        """The store of the database named by a ``--db-path`` / ``--output-db`` string."""
        db_folder, index_name, _ = parse_db_path(db_path)
        return cls(db_folder, index_name)

    def sequence_store(self) -> SequenceStore:
        """The database's sequence store (same folder and index name)."""
        return SequenceStore(self.db_folder, self.index_name)

    # -----------------------------------------------------------------------------------------------------------------
    # state
    # -----------------------------------------------------------------------------------------------------------------
    def _pending_paths(self) -> Tuple[Path, ...]:
        """The files of a pending build (create() not yet validated, or validate() interrupted)."""
        return (self.tmp_manifest_path, self.tmp_bin_path, self.tmp_idx_path,
                *(Path(f"{self.tmp_idx_path}{s}") for s in _SQLITE_SIDECARS))

    def _other_paths(self) -> Tuple[Path, ...]:
        """Every file of the store except the manifest."""
        return (*self._pending_paths(), self.bin_path, self.idx_path,
                *(Path(f"{self.idx_path}{s}") for s in _SQLITE_SIDECARS))

    def exists(self) -> bool:
        """A complete store: its manifest, written last, is there."""
        return self.manifest_path.exists()

    def is_partial(self) -> bool:
        """Store files without a manifest: a build that stopped (or has not reached validate()) before its end."""
        return not self.exists() and any(path.exists() for path in self._other_paths())

    def has_pending_build(self) -> bool:
        """Files of a build that create() wrote and validate() has not swapped in (next to a complete store when it
        is a replacement). Readers never see them."""
        return any(path.exists() for path in self._pending_paths())

    def remove(self) -> List[Path]:
        """Delete every file of the store, the manifest first (so no manifest ever names missing data), and any
        pending build. Returns the files removed."""
        return self._unlink((self.manifest_path, *self._other_paths()))

    def _discard_pending(self) -> List[Path]:
        """Delete the pending build only; a complete store stays as it is."""
        return self._unlink(self._pending_paths())

    def _unlink(self, paths) -> List[Path]:
        removed = []
        for path in paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            removed.append(path)
        if removed:
            _fsync_dir(self.db_folder)
        return removed

    # -----------------------------------------------------------------------------------------------------------------
    # build
    # -----------------------------------------------------------------------------------------------------------------
    def create(self, batches, system: EsmCodesSystem, force: bool = False, *, min_res_n: int = 0,
               build: Optional[Mapping[str, Any]] = None, esm_revision=_INSTALLED) -> Dict[str, Any]:
        """Write the pending ``.bin`` and ``.idx`` from ``batches`` (streamed); :meth:`validate` then checks them,
        swaps them in and writes the manifest.

        ``batches`` is an iterable of batches, each either ``(ids, codes, seq_sha1s)`` (equal-length sequences:
        ``str`` ids; codes as ``bytes`` or ``uint8 [L, code_bytes]`` arrays, an empty payload allowed; 20-byte
        ``seq_sha1``\\ s) or a pyarrow ``RecordBatch``/``Table`` with the columns ``id``, ``residue_codes`` and
        ``seq_sha1``. Entries are stored in the order given. An empty payload (the build channel's mark for an entry
        it could not encode, P5.1) is stored as such; validate() check (c) refuses it.

        The codes are written aside (``P.rescodes.{bin,idx}.tmp``). A complete store (manifest present) is refused
        unless ``force``; with ``force`` it stays whole and readable until :meth:`validate` has accepted the new codes
        and swaps them in, so a failed rebuild leaves it as it was. A partial store is cleared, and so are the pending
        files of an earlier create() that was never validated. The ``.bin`` sha256 is computed while writing; index
        rows go in in transactions of 50,000; a duplicate id raises (pre-flight refuses duplicates, so it is a bug);
        both files are fsynced; then ``entries_digest`` is computed and the totals recorded. Any failure removes what
        was written (the pending files only). ``build`` adds to (or overrides) the manifest's ``build`` block
        (``run_id``, ``from_shards``, ``accelerator``, ``devices``, ``esm_precision``, ``encoder_precision``, ...);
        ``esm_revision`` is the ESM3 repo revision recorded in the manifest (by default this installation's pin, when
        it is the system's ESM3). ``system`` is used as given (the build passes the registry entry).

        Returns ``{"entries", "total_residues", "bin_sha256", "entries_digest"}``.
        """
        if not isinstance(system, EsmCodesSystem):
            raise TypeError(f"create needs an EsmCodesSystem, got {type(system).__name__}")
        if isinstance(min_res_n, bool) or not isinstance(min_res_n, int) or min_res_n < 0:
            raise TypeError(f"min_res_n must be an int >= 0, got {min_res_n!r}")
        build_json = json.dumps(_build_info(build), sort_keys=True, allow_nan=False)
        esm_revision = _installed_esm_revision(system) if esm_revision is _INSTALLED else esm_revision
        if esm_revision is not None and not isinstance(esm_revision, str):
            raise TypeError(f"esm_revision must be a str or None, got {esm_revision!r}")

        if self.exists():
            if not force:
                raise FileExistsError(
                    f"Residue-code store already exists at {self.manifest_path}. Pass force (--force) to replace "
                    f"it, or choose a different database path.")
            logger.info(f"Building a replacement for the residue-code store {self.base}.rescodes.*: the current "
                        f"store stays in place until the new codes have passed validation")
        elif self.is_partial():
            logger.warning(f"Clearing a partial residue-code store at {self.base} (no manifest: an earlier build "
                           f"stopped before its end)")
            self.remove()  # the manifest-less remains, pending files included
        if self._discard_pending():
            logger.warning(f"Cleared the unvalidated residue codes an earlier build left at {self.tmp_bin_path}")
        self.db_folder.mkdir(parents=True, exist_ok=True)

        code_bytes = system.code_bytes
        bin_hash = hashlib.sha256()
        entries = total = 0
        conn = None
        try:
            with open(self.tmp_bin_path, "xb") as out:
                conn = sqlite3.connect(str(self.tmp_idx_path))
                # A derived artefact (rebuildable from the shards) that nothing reads before its manifest exists:
                # durability comes from the fsync before the manifest, not from the journal.
                conn.execute("PRAGMA journal_mode = MEMORY")
                conn.execute("PRAGMA synchronous = OFF")
                conn.execute("CREATE TABLE entries (id TEXT PRIMARY KEY NOT NULL, offset INTEGER NOT NULL, "
                             "length INTEGER NOT NULL, seq_sha1 BLOB NOT NULL) WITHOUT ROWID")
                conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value)")
                conn.commit()
                rows: list = []
                for batch in batches:
                    ids, payloads, sha1s = _columns(batch)
                    chunks = []
                    for entry_id, payload, sha1 in zip(ids, payloads, sha1s):
                        if not isinstance(entry_id, str):
                            raise ResidueCodeStoreError(f"entry ids must be str, got {entry_id!r}")
                        data = _payload_bytes(payload, code_bytes, entry_id)
                        length = len(data) // code_bytes
                        rows.append((entry_id, total, length, _seq_sha1_bytes(sha1, entry_id)))
                        chunks.append(data)
                        total += length
                        if len(rows) >= _INSERT_BATCH:
                            _insert(conn, rows)
                            entries += len(rows)
                            rows = []
                    data = b"".join(chunks)
                    out.write(data)
                    bin_hash.update(data)
                if rows:
                    _insert(conn, rows)
                    entries += len(rows)
                out.flush()
                os.fsync(out.fileno())

            digest = _entries_digest_sorted(conn.execute("SELECT id, seq_sha1 FROM entries ORDER BY id"))
            pending = {
                "store_format": IDX_FORMAT, "system_json": system.to_json(), "esm_revision": esm_revision,
                "min_res_n": min_res_n, "build_json": build_json, "entries": entries, "total_residues": total,
                "bin_sha256": bin_hash.hexdigest(), "entries_digest": digest,
            }
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [(k, pending[k]) for k in _PENDING_KEYS])
            conn.commit()
            conn.close()
            conn = None
            _fsync_file(self.tmp_idx_path)
            _fsync_dir(self.db_folder)
        except BaseException:
            if conn is not None:
                conn.close()
            self._discard_pending()
            raise
        logger.info(f"Residue codes written to {self.tmp_bin_path} ({entries} entries, {total} residues); "
                    f"validation pending")
        return {"entries": entries, "total_residues": total, "bin_sha256": pending["bin_sha256"],
                "entries_digest": digest}

    def _read_pending(self) -> Dict[str, Any]:
        """The idx meta create() recorded; ResidueCodeStoreError if create() did not finish."""
        if not self.tmp_idx_path.exists():
            raise ResidueCodeStoreError(
                f"No pending residue-code build at {self.tmp_idx_path}: create() has not run (validate() checks the "
                f"codes create() wrote, then swaps them in)")
        with _sqlite_errors(self.tmp_idx_path):
            conn = _connect_ro(self.tmp_idx_path)
            try:
                has_meta = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone() is not None
                pending = dict(conn.execute("SELECT key, value FROM meta").fetchall()) if has_meta else {}
            finally:
                conn.close()
        missing = [k for k in _PENDING_KEYS if k not in pending]
        if missing or pending.get("store_format") != IDX_FORMAT:
            raise ResidueCodeStoreError(
                f"Residue-code index {self.tmp_idx_path} is incomplete (its build stopped before it recorded "
                f"{missing or 'its format'}). {_REBUILD_HINT}.")
        return pending

    def validate(self, sequence_store: SequenceStore, faiss_ids_count: int, *, preflight) -> Dict[str, Any]:
        """Check the pending build (:meth:`create`) against the database, then swap it in and write the manifest
        (the commit point). Returns the manifest.

        The checks, run via ``ATTACH`` of the sequence store (both read-only):

        * (a) no sequence-store entry without codes;
        * (b) no codes entry without a sequence-store entry;
        * (c) no length mismatch, empty payloads included;
        * (d) ``seq_sha1 == sha1(stored sequence)`` (both sides went through ``normalize_sequence``, so no
          ``upper()``);
        * (e) the entry count and ``N_res`` equal the sequence store's meta, the pre-flight's counts and the FAISS
          count, and the store's own records (its index rows, its ``.bin`` size).

        ``preflight`` is the build's :class:`foldmatch.utils.sequence.FastaPreflight` (its ``unique_entries`` and
        ``unique_residues``), or ``None`` when there was none (a rebuild from shards); it has no default, so it is
        never left out by accident. Any failure discards the pending build and raises
        (:class:`CodeStoreValidationError` naming each failed check with its first 10 ids); a complete store it was to
        replace, the sequence store and FAISS are untouched, so the database still serves BLOSUM62.

        The swap, once every check passed: the manifest goes to ``P.rescodes.json.tmp`` and is fsynced (the only
        step that needs space, taken while an old store is still whole), the old manifest is removed, the pending
        ``.bin`` and ``.idx`` are renamed into place, and the new manifest is renamed last, each step made durable
        with a directory fsync. A failure during the renames leaves no manifest and removes the store's files.
        """
        if not (hasattr(sequence_store, "meta") and hasattr(sequence_store, "path")):
            raise TypeError(f"validate needs a SequenceStore, got {type(sequence_store).__name__}")
        if isinstance(faiss_ids_count, bool) or not isinstance(faiss_ids_count, (int, np.integer)) \
                or faiss_ids_count < 0:
            raise TypeError(f"faiss_ids_count must be an int >= 0, got {faiss_ids_count!r}")
        if preflight is not None and not (hasattr(preflight, "unique_entries")
                                          and hasattr(preflight, "unique_residues")):
            raise TypeError(f"preflight must be a FastaPreflight or None, got {type(preflight).__name__}")
        replacing = self.exists()
        try:
            pending = self._read_pending()
            system = EsmCodesSystem.from_json(pending["system_json"])
            seq_meta = _read_sequence_meta(sequence_store, ResidueCodeStoreError)
            failures, scan = self._scan(sequence_store)
            problems = self._totals_problems(pending, system, scan, seq_meta, preflight, int(faiss_ids_count))
            if problems:
                failures["e"] = (len(problems), tuple(problems))
            if failures:
                raise CodeStoreValidationError(
                    self._failure_message(failures, scan["empty_payloads"], replacing), failures)
            manifest = _manifest(system, pending, scan, seq_meta, preflight, int(faiss_ids_count))
            # Written and fsynced (not yet renamed) while a store being replaced is still whole: the only step of
            # the swap that needs space.
            _write_synced(self.tmp_manifest_path,
                          json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        except BaseException:
            self._discard_pending()
            raise
        self._swap_in()
        logger.info(f"Residue-code store {self.manifest_path} validated ({manifest['entries']} entries, "
                    f"{manifest['total_residues']} residues)" + ("; it replaces the previous store" if replacing
                                                                 else ""))
        return manifest

    def _swap_in(self) -> None:
        """Replace the store by the validated pending build: old manifest out, ``.bin`` and ``.idx`` renamed into
        place, the new manifest renamed last (each step made durable). On failure no manifest is left, and the store's
        files are removed (it is partial)."""
        try:
            if self.manifest_path.exists():
                os.unlink(self.manifest_path)  # from here on the old store is partial, never mixed with the new one
                _fsync_dir(self.db_folder)
            for sidecar in _SQLITE_SIDECARS:
                Path(f"{self.idx_path}{sidecar}").unlink(missing_ok=True)
            os.replace(self.tmp_bin_path, self.bin_path)
            os.replace(self.tmp_idx_path, self.idx_path)
            _fsync_dir(self.db_folder)
            os.replace(self.tmp_manifest_path, self.manifest_path)
            _fsync_dir(self.db_folder)
        except BaseException:
            self.remove()
            raise

    def _scan(self, sequence_store: SequenceStore) -> Tuple[Dict[str, Tuple[int, Tuple[str, ...]]], Dict[str, int]]:
        """Checks (a)-(d) and the totals, in SQL over the pending idx with the sequence store attached."""
        seq_path = Path(sequence_store.path)
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence store not found at {seq_path}")
        with _sqlite_errors(f"{self.tmp_idx_path} with the sequence store {seq_path}"):
            return self._scan_sql(seq_path)

    def _scan_sql(self, seq_path: Path):
        failures: Dict[str, Tuple[int, Tuple[str, ...]]] = {}
        conn = _connect_ro(self.tmp_idx_path)
        try:
            conn.execute("ATTACH DATABASE ? AS seq", (_ro_uri(seq_path),))
            conn.create_function("fm_seq_sha1", 1, _sql_seq_sha1, deterministic=True)
            conn.create_function("fm_nonstandard", 1, _sql_nonstandard, deterministic=True)
            rows, residues = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(length), 0) FROM main.entries").fetchone()

            def count_and_ids(where: str, id_expr: str) -> Tuple[int, Tuple[str, ...]]:
                n = conn.execute(f"SELECT COUNT(*) {where}").fetchone()[0]
                ids = conn.execute(f"SELECT {id_expr} {where} ORDER BY {id_expr} LIMIT {_EXAMPLE_IDS}").fetchall() \
                    if n else []
                return n, tuple(r[0] for r in ids)

            missing = count_and_ids(
                "FROM seq.sequences s WHERE NOT EXISTS (SELECT 1 FROM main.entries c WHERE c.id = s.id)", "s.id")
            extra = count_and_ids(
                "FROM main.entries c WHERE NOT EXISTS (SELECT 1 FROM seq.sequences s WHERE s.id = c.id)", "c.id")
            join = "FROM main.entries c JOIN seq.sequences s ON s.id = c.id"
            _, length_mm, empty, sha1_mm, nonstandard = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(c.length != s.length), 0), "
                f"COALESCE(SUM(c.length = 0 AND s.length > 0), 0), "
                f"COALESCE(SUM(c.seq_sha1 IS NOT fm_seq_sha1(s.sequence)), 0), "
                f"COALESCE(SUM(fm_nonstandard(s.sequence)), 0) {join}").fetchone()
            if missing[0]:
                failures["a"] = missing
            if extra[0]:
                failures["b"] = extra
            if length_mm:
                failures["c"] = count_and_ids(f"{join} WHERE c.length != s.length", "c.id")
            if sha1_mm:
                failures["d"] = count_and_ids(f"{join} WHERE c.seq_sha1 IS NOT fm_seq_sha1(s.sequence)", "c.id")
        finally:
            conn.close()
        bin_size = self.tmp_bin_path.stat().st_size if self.tmp_bin_path.exists() else None
        return failures, {"rows": rows, "residues": residues, "empty_payloads": empty, "length_mismatches": length_mm,
                          "seq_sha1_mismatches": sha1_mm, "nonstandard_entries": nonstandard, "bin_size": bin_size}

    def _totals_problems(self, pending, system, scan, seq_meta, preflight, faiss_ids_count) -> List[str]:
        """Check (e): every count of entries and of residues agrees."""
        entries, residues = int(pending["entries"]), int(pending["total_residues"])
        problems = []

        def compare(what: str, got_entries, got_residues=None):
            if got_entries != entries:
                problems.append(f"{what} has {got_entries} entries, the code store {entries}")
            if got_residues is not None and got_residues != residues:
                problems.append(f"{what} has {got_residues} residues, the code store {residues}")

        compare("the code index", scan["rows"], scan["residues"])
        want_size = residues * system.code_bytes
        if scan["bin_size"] != want_size:
            problems.append(f"{self.tmp_bin_path.name} is {scan['bin_size']} bytes, expected {want_size} "
                            f"({residues} residues x {system.code_bytes})")
        compare("the sequence store meta", seq_meta["entry_count"], seq_meta["total_residues"])
        if preflight is not None:
            compare("the pre-flight", preflight.unique_entries, preflight.unique_residues)
        compare("the FAISS index", faiss_ids_count)
        return problems

    def _failure_message(self, failures, empty_payloads: int, replacing: bool) -> str:
        what = {
            "a": "sequence-store entr(ies) have no residue codes",
            "b": "residue-code entr(ies) are not in the sequence store",
            "c": "entr(ies) have codes of another length than their sequence",
            "d": "entr(ies) have a seq_sha1 that is not the sha1 of their stored sequence",
        }
        lines = []
        for check in ("a", "b", "c", "d"):
            if check in failures:
                n, ids = failures[check]
                empty = (f", {empty_payloads} of them with an empty payload (the build could not encode them)"
                         if check == "c" and empty_payloads else "")
                lines.append(f"({check}) {n} {what[check]}{empty} (first: {', '.join(ids)})")
        if "e" in failures:
            lines.append(f"(e) the totals disagree: {'; '.join(failures['e'][1])}")
        hint = _FRESH_BUILD_HINT if "c" in failures and empty_payloads else _REBUILD_HINT
        if replacing:
            return (f"The new residue codes for {self.base} failed validation and were discarded: " + "; ".join(lines)
                    + f". The existing residue-code store {self.manifest_path} and the database are unchanged. "
                      f"{hint}.")
        return (f"Residue-code store {self.base}.rescodes.* failed validation and was removed: " + "; ".join(lines)
                + f". The database itself is unchanged and still serves --scoring blosum62. {hint}.")

    # -----------------------------------------------------------------------------------------------------------------
    # read
    # -----------------------------------------------------------------------------------------------------------------
    def manifest(self) -> Dict[str, Any]:
        """The parsed manifest (read-only)."""
        if not self.manifest_path.exists():
            if self.is_partial():
                raise ResidueCodeStoreError(
                    f"Residue-code store {self.base}.rescodes.* is incomplete (no manifest): its build stopped "
                    f"before it was validated. {_REBUILD_HINT}.")
            raise FileNotFoundError(
                f"No residue-code store for {self.base} ({self.manifest_path} not found): the database was built "
                f"without --residue-codes. {_REBUILD_HINT}.")
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise IncompatibleCodeStoreError(f"{self.manifest_path} is not readable JSON: {e}") from e
        if not isinstance(data, dict):
            raise IncompatibleCodeStoreError(f"{self.manifest_path} is not a JSON object")
        return data

    def check_compatible(self, system=_INSTALLED, *, sequence_store=_INSTALLED,
                         esm_weights_sha256=_INSTALLED) -> Dict[str, Any]:
        """Refuse a store this query cannot use; return its manifest. Read-only and cheap (no hashing, no scan), so
        it runs on every rank before ESM3.

        Refused (:class:`IncompatibleCodeStoreError`, all problems named): an unknown ``manifest_version`` or
        ``kind``; any missing key (``MANIFEST_KEYS`` and the nested blocks); an ``encoding_digest`` other than the
        system's; ESM3 weights, target encoder or transform content hashes (sha256, not commit strings) other than the
        system's, or ESM3 weights other than the ones this FoldMatch loads; any other encoding field, the code width
        or bit order other than the system's; ``entries`` or ``total_residues`` other than the sequence store meta; a
        missing index or a ``.bin`` of the wrong size.

        Every argument defaults to :data:`_INSTALLED`, what this installation uses: ``system`` the registry entry the
        manifest names (an unknown name raises
        :class:`~foldmatch.search.scoring_system.UnknownScoringSystemError`), ``sequence_store`` the database's own
        sequence store (or pass a ``SequenceStore`` or its ``meta()`` mapping), ``esm_weights_sha256`` the ESM3
        weights this FoldMatch pins. So leaving an argument out never skips its check, and ``None`` is a TypeError.
        A statistics-only system (another name, the same encoding) is compatible: the codes depend only on the
        encoding fields.
        """
        for name, value in (("system", system), ("sequence_store", sequence_store),
                            ("esm_weights_sha256", esm_weights_sha256)):
            if value is None:
                raise TypeError(f"check_compatible: {name}=None would skip a check; leave it out to compare with "
                                f"this installation")
        manifest = self.manifest()
        where = str(self.manifest_path)
        version = manifest.get("manifest_version")
        if isinstance(version, bool) or version != MANIFEST_VERSION:
            raise IncompatibleCodeStoreError(
                f"{where}: unknown manifest_version {version!r}; this FoldMatch reads version {MANIFEST_VERSION} (a "
                f"store written by a newer FoldMatch needs a FoldMatch upgrade)")
        if manifest.get("kind") != MANIFEST_KIND:
            raise IncompatibleCodeStoreError(f"{where}: kind {manifest.get('kind')!r}, expected {MANIFEST_KIND!r}")
        missing = _missing_keys(manifest)
        if missing:
            raise IncompatibleCodeStoreError(f"{where}: the manifest lacks {', '.join(missing)}. {_REBUILD_HINT}.")
        if system is _INSTALLED:
            system = get_system(manifest["system_name"])
        elif not isinstance(system, EsmCodesSystem):
            raise TypeError(f"check_compatible: system must be an EsmCodesSystem, got {type(system).__name__}")
        if esm_weights_sha256 is _INSTALLED:
            from foldmatch.utils.esm.loaders import ESM_WEIGHTS_SHA256 as esm_weights_sha256
        if sequence_store is _INSTALLED:
            sequence_store = self.sequence_store()
        seq_meta = _sequence_meta(sequence_store)

        built = manifest["system"]
        want = system.to_dict()
        problems = []
        if manifest["encoding_digest"] != system.encoding_digest():
            problems.append(f"encoding_digest {manifest['encoding_digest']!r}, {system.name} has "
                            f"{system.encoding_digest()!r}")
        if built["name"] != manifest["system_name"]:
            problems.append(f"system_name {manifest['system_name']!r} but the recorded system is {built['name']!r}")
        for label, field in (("ESM3 weights", "esm_weights_sha256"), ("target encoder", "target_encoder_sha256"),
                             ("transform", "transform_sha256")):
            if built[field] != want[field]:
                problems.append(f"built with {label} sha256 {built[field]}, {system.name} has {want[field]}")
        if manifest["esm"]["weights_sha256"] != want["esm_weights_sha256"]:
            problems.append(f"built with ESM3 weights {manifest['esm']['weights_sha256']}, {system.name} has "
                            f"{want['esm_weights_sha256']}")
        if manifest["esm"]["weights_sha256"] != esm_weights_sha256:
            problems.append(f"built with ESM3 weights {manifest['esm']['weights_sha256']}, this FoldMatch loads "
                            f"{esm_weights_sha256}")
        for field in ENCODING_FIELDS:
            if field not in ("esm_weights_sha256", "target_encoder_sha256", "transform_sha256") \
                    and built[field] != want[field]:
                problems.append(f"built with {field} {built[field]!r}, {system.name} has {want[field]!r}")
        if manifest["code_bytes"] != system.code_bytes or manifest["bitorder"] != system.bitorder:
            problems.append(f"codes of {manifest['code_bytes']!r} bytes, bit order {manifest['bitorder']!r}; "
                            f"{system.name} has {system.code_bytes} bytes, {system.bitorder!r}")
        if manifest["entries"] != seq_meta["entry_count"]:
            problems.append(f"{manifest['entries']!r} entries, the sequence store {seq_meta['entry_count']}")
        if manifest["total_residues"] != seq_meta["total_residues"]:
            problems.append(f"{manifest['total_residues']!r} residues, the sequence store "
                            f"{seq_meta['total_residues']}")
        if not self.idx_path.exists():
            problems.append(f"{self.idx_path} is missing")
        if not self.bin_path.exists():
            problems.append(f"{self.bin_path} is missing")
        elif isinstance(manifest["total_residues"], int) and isinstance(manifest["code_bytes"], int):
            want_size = manifest["total_residues"] * manifest["code_bytes"]
            size = self.bin_path.stat().st_size
            if size != want_size:
                problems.append(f"{self.bin_path.name} is {size} bytes, the manifest needs {want_size}")
        if problems:
            raise IncompatibleCodeStoreError(
                f"Residue-code store {self.base}.rescodes.* cannot be used with {system.name}: " + "; ".join(problems)
                + f". --scoring blosum62 still works. {_REBUILD_HINT}.")
        return manifest

    def check_faiss_count(self, faiss_ids_count: int) -> None:
        """Refuse a store whose entry count is not the live FAISS index's (rank 0, once the index is loaded)."""
        entries = self.manifest().get("entries")
        if entries != faiss_ids_count:
            raise IncompatibleCodeStoreError(
                f"Residue-code store {self.manifest_path} has {entries!r} entries, the FAISS index "
                f"{faiss_ids_count}. --scoring blosum62 still works. {_REBUILD_HINT}.")

    def _require_complete(self) -> None:
        if not self.exists():
            self.manifest()  # raises the partial / not-found error

    def lookup(self, ids: Iterable[str]) -> Dict[str, CodeEntry]:
        """``{id: CodeEntry(offset, length, seq_sha1)}`` for every id (repeats collapse); an id the store lacks
        raises :class:`ResidueCodeStoreError` naming the first 10. Queried in ``IN`` chunks of 900."""
        self._require_complete()
        unique = list(dict.fromkeys(ids))
        out: Dict[str, CodeEntry] = {}
        if unique:
            for entry_id, offset, length, sha1 in self._select(unique, "offset, length, seq_sha1"):
                out[entry_id] = CodeEntry(offset, length, sha1)
        self._refuse_missing(unique, out)
        return out

    def lookup_rows(self, ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """``(offsets, lengths)``, int64 arrays aligned with ``ids`` (repeats allowed): the code rows of each id, as
        :meth:`lookup` gives them, without building an entry per id (Stage 2 looks up a whole batch of candidates
        at once). Missing ids raise as in :meth:`lookup`."""
        self._require_complete()
        ids = ids if isinstance(ids, list) else list(ids)
        unique = list(dict.fromkeys(ids))
        rows: Dict[str, Tuple[int, int]] = {}
        if unique:
            for entry_id, offset, length in self._select(unique, "offset, length"):
                rows[entry_id] = (offset, length)
        self._refuse_missing(unique, rows)
        if not ids:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        table = np.array([rows[i] for i in unique], np.int64).reshape(-1, 2)
        position = dict(zip(unique, range(len(unique))))
        picks = np.fromiter((position[i] for i in ids), np.int64, len(ids))
        found = table[picks]
        return np.ascontiguousarray(found[:, 0]), np.ascontiguousarray(found[:, 1])

    def _select(self, unique: List[str], columns: str) -> Iterator[tuple]:
        """``(id, *columns)`` rows of the ids, queried in ``IN`` chunks of 900 on one read-only connection."""
        with _sqlite_errors(self.idx_path):
            conn = _connect_ro(self.idx_path)
            try:
                for start in range(0, len(unique), _SQLITE_VAR_CHUNK):
                    chunk = unique[start:start + _SQLITE_VAR_CHUNK]
                    placeholders = ",".join("?" * len(chunk))
                    yield from conn.execute(f"SELECT id, {columns} FROM entries WHERE id IN ({placeholders})", chunk)
            finally:
                conn.close()

    def _refuse_missing(self, unique: List[str], found: Mapping[str, object]) -> None:
        missing = [i for i in unique if i not in found]
        if missing:
            raise ResidueCodeStoreError(
                f"{len(missing)} id(s) have no residue codes in {self.idx_path} (first: "
                f"{', '.join(map(str, missing[:_EXAMPLE_IDS]))}): the code store and the database disagree. "
                f"{_REBUILD_HINT}.")

    def open_memmap(self) -> np.ndarray:
        """The codes, ``uint8 [N_res, code_bytes]``, memory-mapped read-only. An empty store gives an empty array
        (``np.memmap`` cannot map 0 bytes)."""
        manifest = self.manifest()
        code_bytes, total = manifest.get("code_bytes"), manifest.get("total_residues")
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (code_bytes, total)) \
                or code_bytes == 0:
            raise IncompatibleCodeStoreError(f"{self.manifest_path}: unusable code_bytes {code_bytes!r} / "
                                             f"total_residues {total!r}")
        size = self.bin_path.stat().st_size
        if size != total * code_bytes:
            raise IncompatibleCodeStoreError(f"{self.bin_path} is {size} bytes, the manifest needs "
                                             f"{total * code_bytes}. {_REBUILD_HINT}.")
        if total == 0:
            return np.empty((0, code_bytes), dtype=np.uint8)
        return np.memmap(self.bin_path, dtype=np.uint8, mode="r", shape=(total, code_bytes))


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _columns(batch) -> Tuple[list, list, list]:
    if hasattr(batch, "schema") and hasattr(batch, "column"):
        names = set(batch.schema.names)
        lacking = [c for c in (ID_COLUMN, CODES_COLUMN, SEQ_SHA1_COLUMN) if c not in names]
        if lacking:
            raise ResidueCodeStoreError(f"a residue-code batch lacks the column(s) {lacking} (it has {sorted(names)})")
        columns = [batch.column(c).to_pylist() for c in (ID_COLUMN, CODES_COLUMN, SEQ_SHA1_COLUMN)]
    else:
        try:
            columns = list(batch)
        except TypeError:
            columns = []
        if len(columns) != 3:
            raise ResidueCodeStoreError("a residue-code batch must be (ids, codes, seq_sha1s) or a pyarrow batch with "
                                        f"the columns {ID_COLUMN}, {CODES_COLUMN}, {SEQ_SHA1_COLUMN}")
    lengths = {len(c) for c in columns}
    if len(lengths) != 1:
        raise ResidueCodeStoreError(f"a residue-code batch has columns of different lengths {sorted(lengths)}")
    return columns[0], columns[1], columns[2]


def _payload_bytes(value, code_bytes: int, entry_id: str) -> bytes:
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8 or value.ndim not in (1, 2) or (value.ndim == 2 and value.shape[1] != code_bytes):
            raise ResidueCodeStoreError(f"{entry_id!r}: codes must be uint8 [L, {code_bytes}], got {value.dtype} "
                                        f"{value.shape}")
        data = np.ascontiguousarray(value).tobytes()
    elif isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, memoryview) and value.itemsize == 1:
        data = value.tobytes()
    else:
        raise ResidueCodeStoreError(f"{entry_id!r}: codes must be bytes or a uint8 array, got "
                                    f"{type(value).__name__}")
    if len(data) % code_bytes:
        raise ResidueCodeStoreError(f"{entry_id!r}: {len(data)} code bytes is not a whole number of "
                                    f"{code_bytes}-byte residues")
    return data


def _seq_sha1_bytes(value, entry_id: str) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if len(data) == SEQ_SHA1_BYTES:
            return data
    raise ResidueCodeStoreError(f"{entry_id!r}: seq_sha1 must be {SEQ_SHA1_BYTES} bytes, got {value!r:.80}")


def _insert(conn: sqlite3.Connection, rows: list) -> None:
    """One transaction of index rows; a duplicate id (in ``rows`` or already stored) raises naming it."""
    try:
        conn.executemany("INSERT INTO entries (id, offset, length, seq_sha1) VALUES (?, ?, ?, ?)", rows)
        conn.commit()
    except sqlite3.IntegrityError as e:
        conn.rollback()
        ids = [r[0] for r in rows]
        seen, duplicates = set(), []
        for entry_id in ids:
            if entry_id in seen:
                duplicates.append(entry_id)
            seen.add(entry_id)
        unique = list(dict.fromkeys(ids))
        for start in range(0, len(unique), _SQLITE_VAR_CHUNK):
            chunk = unique[start:start + _SQLITE_VAR_CHUNK]
            duplicates.extend(r[0] for r in conn.execute(
                f"SELECT id FROM entries WHERE id IN ({','.join('?' * len(chunk))})", chunk))
        duplicates = sorted(set(duplicates))
        raise ResidueCodeStoreError(
            f"duplicate entry id(s) in the residue codes: {', '.join(duplicates[:_EXAMPLE_IDS]) or e}. The "
            f"pre-flight refuses duplicate ids before ESM3, so this is a bug.") from e


def _sql_seq_sha1(sequence) -> Optional[bytes]:
    """SQL ``fm_seq_sha1``: sha1 of the ASCII bytes of an already normalised sequence (NULL if not ASCII, which then
    mismatches every stored seq_sha1)."""
    if not isinstance(sequence, str) or not sequence.isascii():
        return None
    return hashlib.sha1(sequence.encode("ascii")).digest()


def _sql_nonstandard(sequence) -> int:
    """SQL ``fm_nonstandard``: 1 if the sequence has a residue outside the 20 standard amino acids."""
    if not isinstance(sequence, str):
        return 1
    if not sequence.isascii():
        return 1
    return int(bool(sequence.encode("ascii").translate(None, _STANDARD_BYTES)))


def _ro_uri(path) -> str:
    return Path(path).resolve().as_uri() + "?mode=ro"


def _connect_ro(path) -> sqlite3.Connection:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found")
    return sqlite3.connect(_ro_uri(path), uri=True)


def _fsync_file(path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path) -> None:
    """Make renames and unlinks in ``path`` durable; a no-op where directories cannot be opened or fsynced."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_synced(path: Path, text: str) -> None:
    """Write ``path`` and fsync it (the caller renames it into place)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def _sqlite_errors(what):
    """A file that is not the SQLite database it should be: a :class:`ResidueCodeStoreError` naming it (SQLite's own
    message, e.g. 'file is not a database', names no file)."""
    try:
        yield
    except sqlite3.DatabaseError as e:
        if isinstance(e, sqlite3.IntegrityError):
            raise
        raise ResidueCodeStoreError(f"{what}: not a readable residue-code index or sequence store ({e})") from e


def _read_sequence_meta(sequence_store, error: type) -> Mapping[str, Any]:
    """``sequence_store.meta()``, a corrupt file reported as ``error`` naming it."""
    try:
        return sequence_store.meta()
    except sqlite3.DatabaseError as e:
        raise error(f"{sequence_store.path} is not a readable sequence store ({e})") from e


def _dist_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _build_info(build: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    from foldmatch import __version__
    info: Dict[str, Any] = {
        "foldmatch_version": __version__,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": None, "from_shards": None, "accelerator": None, "devices": None, "esm_precision": None,
        "encoder_precision": None,
        "torch": _dist_version("torch"), "esm": _dist_version("esm"),
        "platform": f"{platform.system()}-{platform.machine()}",
    }
    if build is not None:
        if not isinstance(build, Mapping) or not all(isinstance(k, str) for k in build):
            raise TypeError(f"build must be a mapping with str keys, got {build!r}")
        info.update(build)
    return info


def _installed_esm_revision(system: EsmCodesSystem) -> Optional[str]:
    """This installation's ESM3 revision pin, when it serves the system's ESM3 weights."""
    from foldmatch.utils.esm import loaders
    if (loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_SHA256) == (system.esm_repo, system.esm_weights_sha256):
        return loaders.ESM_REVISION
    return None


def _sequence_meta(sequence_store) -> Mapping[str, Any]:
    if hasattr(sequence_store, "meta"):
        meta = _read_sequence_meta(sequence_store, IncompatibleCodeStoreError)
    elif isinstance(sequence_store, Mapping):
        meta = sequence_store
    else:
        raise TypeError(f"sequence_store must be a SequenceStore or its meta() mapping, got "
                        f"{type(sequence_store).__name__}")
    for key in ("entry_count", "total_residues"):
        if key not in meta:
            raise TypeError(f"sequence_store meta lacks {key!r}")
    return meta


def _missing_keys(manifest: Mapping[str, Any]) -> List[str]:
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    for block, keys in _NESTED_KEYS.items():
        if block not in manifest:
            continue
        value = manifest[block]
        if not isinstance(value, Mapping):
            missing.append(f"{block} (not an object)")
            continue
        missing.extend(f"{block}.{k}" for k in keys if k not in value)
    return missing


def _manifest(system: EsmCodesSystem, pending, scan, seq_meta, preflight, faiss_ids_count: int) -> Dict[str, Any]:
    """The Appendix B manifest of a store that passed validation."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "kind": MANIFEST_KIND,
        "system_name": system.name,
        "system_digest": system.digest(),
        "encoding_digest": system.encoding_digest(),
        "system": system.to_dict(),
        "code_bytes": system.code_bytes,
        "bitorder": system.bitorder,
        "esm": {"repo": system.esm_repo, "revision": pending["esm_revision"],
                "weights_sha256": system.esm_weights_sha256},
        "bundle": {"repo": system.bundle_repo, "revision": system.bundle_revision,
                   "files": dict(system.bundle_files)},
        "entries": int(pending["entries"]),
        "total_residues": int(pending["total_residues"]),
        "min_res_n": int(pending["min_res_n"]),
        "bin_sha256": pending["bin_sha256"],
        "entries_digest": pending["entries_digest"],
        "nonstandard_entries": int(scan["nonstandard_entries"]),
        "input_normalisation": system.input_normalisation,
        "cross_checks": {
            "preflight_entries": None if preflight is None else int(preflight.unique_entries),
            "preflight_residues": None if preflight is None else int(preflight.unique_residues),
            "sequence_store_entries": int(seq_meta["entry_count"]),
            "sequence_store_residues": int(seq_meta["total_residues"]),
            "faiss_entries": faiss_ids_count,
            "seq_sha1_mismatches": int(scan["seq_sha1_mismatches"]),
            "length_mismatches": int(scan["length_mismatches"]),
        },
        "build": json.loads(pending["build_json"]),
    }
