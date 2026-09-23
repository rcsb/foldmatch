"""Predicted 3Di codes for a sequence-only database, and the manifest that makes them safe.

A sequence database has no coordinates, so it cannot read a structural alphabet from
anything - the only way it can carry one is to predict it. Each residue is stored as ONE
BYTE: the predicted 3Di state times a confidence bucket (240 codes), which the search
interprets through the fitted kernel named in the manifest.

Why a sidecar rather than a column on ``sequences``: that table's premise is that it is
rebuildable from the source FASTA (:class:`SequenceStore` says so, and its build is a
second independent pass over the same file). These codes are not - reproducing them needs
the ESM3 revision, the head revision, a temperature and a kernel. Keeping them in their
own versioned file keeps that promise true and makes the model-derived data removable.

The manifest is the point of this module as much as the bytes are. A code is meaningless
without the kernel that interprets it, and the kernel is meaningless at a different
temperature, and the whole thing is meaningless if the backbone that produced the
embeddings has moved. None of those mismatches would raise on their own: they would
produce plausible alignments and confident E-values for a scoring system nobody
calibrated. So the manifest records every input, the search checks it, and a database
built under one scoring system cannot be searched under another.
"""
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Sidecar holding ``id -> codes`` (``{index_name}.threedi``), beside ``.sequences``.
THREE_DI_STORE_SUFFIX = ".threedi"
#: Plain JSON, deliberately: ``.metadata`` is a pickle that cannot be read without an
#: importable foldmatch, which makes a stale database impossible to diagnose from outside.
THREE_DI_MANIFEST_SUFFIX = ".threedi.json"

#: Bumped when the on-disk layout changes in a way older readers cannot handle.
MANIFEST_VERSION = 1

#: Sentinel: "use whatever revision this install loads", so a caller cannot skip the
#: revision check by simply not passing it. Pass None to opt out deliberately.
_INSTALLED = object()

_SQLITE_VAR_CHUNK = 900
_INSERT_BATCH = 50_000


class ThreeDiStoreError(RuntimeError):
    """A 3Di sidecar that cannot be trusted for the search being asked for."""


class ThreeDiStore:
    """SQLite-backed ``id -> per-residue codes`` store beside a FAISS database.

    Reads are random-access by id, like :class:`SequenceStore`: Stage 2 only ever needs
    the handful of candidates the embedding prefilter returned, so the corpus is never
    loaded into RAM.
    """

    def __init__(self, db_folder: Path, index_name: str):
        self.db_folder = Path(db_folder)
        self.index_name = index_name
        self.path = self.db_folder / f"{index_name}{THREE_DI_STORE_SUFFIX}"
        self.manifest_path = self.db_folder / f"{index_name}{THREE_DI_MANIFEST_SUFFIX}"

    # -- existence -------------------------------------------------------------------

    def exists(self) -> bool:
        """True only when the build finished.

        The manifest is written last, so a sidecar without one is an interrupted build
        rather than a usable store, and is reported as absent instead of silently
        serving a partial corpus.
        """
        return self.path.exists() and self.manifest_path.exists()

    def is_partial(self) -> bool:
        return self.path.exists() and not self.manifest_path.exists()

    # -- build -----------------------------------------------------------------------

    def create(self, entries: Iterable[Tuple[str, np.ndarray]], force: bool = False) -> int:
        """Build the store from ``(id, codes)`` pairs. Returns the number of rows.

        The manifest is NOT written here - :meth:`write_manifest` is a separate call the
        builder makes only once every id is accounted for, so an interrupted build leaves
        a store that :meth:`exists` reports as absent.
        """
        if self.path.exists():
            if not force:
                raise FileExistsError(
                    f"3Di store already exists at {self.path}. Choose a different output, "
                    f"or delete it first.")
            self.path.unlink()
        if self.manifest_path.exists():
            # Remove it up front: from here until write_manifest the store is incomplete,
            # and a manifest left from a previous build would declare it finished.
            self.manifest_path.unlink()

        self.db_folder.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        try:
            conn.execute("PRAGMA journal_mode = MEMORY")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute(
                "CREATE TABLE three_di ("
                "id TEXT PRIMARY KEY, codes BLOB NOT NULL, length INTEGER NOT NULL)")
            insert = "INSERT OR IGNORE INTO three_di (id, codes, length) VALUES (?, ?, ?)"
            batch, seen = [], 0
            for identifier, codes in entries:
                codes = np.ascontiguousarray(codes, dtype=np.uint8)
                batch.append((identifier, codes.tobytes(), int(codes.size)))
                seen += 1
                if len(batch) >= _INSERT_BATCH:
                    conn.executemany(insert, batch)
                    conn.commit()
                    batch.clear()
            if batch:
                conn.executemany(insert, batch)
                conn.commit()
            stored = conn.execute("SELECT COUNT(*) FROM three_di").fetchone()[0]
        finally:
            conn.close()
        if seen - stored:
            logger.warning(f"{seen - stored} duplicate id(s) in the 3Di store; kept the first")
        logger.info(f"3Di store written to {self.path} ({stored} entries)")
        return stored

    def write_manifest(self, system, model_revision: str, esm_revision: str,
                       entry_count: int, extra: Optional[dict] = None) -> dict:
        """Record what produced these codes, and mark the build complete.

        Called last, after every id is accounted for. ``system`` is the
        :class:`~foldmatch.search.scoring_system.ScoringSystem` the codes were encoded
        under; its whole JSON is stored, not only its digest, so a database stays
        diagnosable after the shipped registry has been refitted and no longer contains
        the constants this database was built with.
        """
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "scoring_system": json.loads(system.to_json()),
            "scoring_system_digest": system.digest(),
            # Repeated out of the scoring system so a mismatch can be named precisely
            # rather than reported as "the digest differs".
            "n_codes": system.n_codes,
            "kernel_sha256": system.kernel_sha256,
            "profile_temperature": system.profile_temperature,
            # The bytes are a function of these too, and neither is in the scoring system.
            "three_di_model_revision": model_revision,
            "esm3_revision": esm_revision,
            "entry_count": entry_count,
        }
        if extra:
            manifest.update(extra)
        self.db_folder.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash must not leave a half-written manifest declaring an
        # incomplete store complete.
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        temporary.replace(self.manifest_path)
        logger.info(f"3Di manifest written to {self.manifest_path}")
        return manifest

    def remove(self) -> None:
        """Delete the sidecar and its manifest, for a forced rebuild."""
        for path in (self.manifest_path, self.path):
            if path.exists():
                path.unlink()

    # -- read ------------------------------------------------------------------------

    def read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"3Di manifest not found at {self.manifest_path}")
        return json.loads(self.manifest_path.read_text())

    def check_compatible(self, system, model_revision: str = _INSTALLED,
                         esm_revision: str = _INSTALLED,
                         expected_entries: Optional[int] = None) -> dict:
        """Refuse to search this store under a scoring system it was not built with.

        Every one of these mismatches produces plausible numbers rather than an error if
        left unchecked: the codes still index a kernel, the alignment still scores, and
        the E-value still comes out of a calibrated-looking constant. Only the answer is
        wrong, so each is a refusal rather than a warning.

        The model and ESM3 revisions are compared when supplied, because the stored bytes
        are a function of both - a different backbone means different embeddings, hence
        different states and different confidences, under the same scoring system.
        """
        manifest = self.read_manifest()
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            raise ThreeDiStoreError(
                f"{self.manifest_path}: manifest version "
                f"{manifest.get('manifest_version')!r}, this FoldMatch writes "
                f"{MANIFEST_VERSION}")
        built = manifest.get("scoring_system", {})
        for field in ("n_codes", "kernel_sha256", "profile_temperature"):
            want, have = getattr(system, field), manifest.get(field, built.get(field))
            if want != have:
                raise ThreeDiStoreError(
                    f"{self.path} was built with {field}={have!r} but the search asks for "
                    f"{field}={want!r}; the stored codes would be interpreted against a "
                    f"different {'kernel' if 'kernel' in field else 'encoding'}.")
        # Default to the revisions this install actually uses, so a caller cannot skip
        # the check by forgetting to pass them; None explicitly opts out.
        if model_revision is _INSTALLED or esm_revision is _INSTALLED:
            from foldmatch.utils.esm.loaders import ESM_REVISION
            from foldmatch.utils.model import THREE_DI_REVISION
            if model_revision is _INSTALLED:
                model_revision = THREE_DI_REVISION
            if esm_revision is _INSTALLED:
                esm_revision = ESM_REVISION
        for label, key, want in (("3Di model", "three_di_model_revision", model_revision),
                                 ("ESM3 backbone", "esm3_revision", esm_revision)):
            have = manifest.get(key)
            if want is not None and have is not None and want != have:
                raise ThreeDiStoreError(
                    f"{self.path} was built with {label} revision {have}, but this "
                    f"install uses {want}; the embeddings behind these codes differ.")

        if expected_entries is not None:
            # A store short of the index drops those subjects per candidate with only a
            # log line (the aligner cannot tell "no codes" from "not a hit"), so a
            # shortfall is a refusal here rather than quietly missing results.
            stored = self.count()
            if stored < expected_entries:
                raise ThreeDiStoreError(
                    f"{self.path} holds {stored} entries but the index has "
                    f"{expected_entries}; {expected_entries - stored} subject(s) would "
                    f"silently never match. Rebuild the sidecar.")
            if stored > expected_entries:
                logger.warning(
                    f"{self.path} holds {stored} entries against the index's "
                    f"{expected_entries}; the extra rows are unreachable but harmless.")
        if manifest.get("scoring_system_digest") != system.digest():
            # Reached only when the difference is in a field checked above's siblings -
            # weights, gaps, scale, lambda/K. Those change the score and its statistics
            # without changing the bytes, so name the whole system rather than a field.
            logger.warning(
                f"{self.path} was built under scoring system "
                f"{built.get('name')!r} (digest {manifest.get('scoring_system_digest')}) "
                f"but is being searched under {system.name!r} (digest {system.digest()}); "
                f"the codes are compatible but the scores and E-values are not the ones "
                f"this database was calibrated with.")
        return manifest

    def fetch(self, ids: Iterable[str]) -> Dict[str, np.ndarray]:
        """``{id: (L,) uint8}`` for the ids present. Missing ids are simply absent."""
        if not self.path.exists():
            raise FileNotFoundError(f"3Di store not found at {self.path}")
        unique = list(dict.fromkeys(ids))
        if not unique:
            return {}
        conn = sqlite3.connect(str(self.path))
        try:
            out: Dict[str, np.ndarray] = {}
            for start in range(0, len(unique), _SQLITE_VAR_CHUNK):
                chunk = unique[start:start + _SQLITE_VAR_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT id, codes FROM three_di WHERE id IN ({placeholders})", chunk)
                for identifier, blob in rows.fetchall():
                    out[identifier] = np.frombuffer(blob, dtype=np.uint8)
            return out
        finally:
            conn.close()

    def missing(self, ids: Iterable[str]) -> set:
        """Which of ``ids`` have no codes, without loading any codes.

        The build has to check every indexed entry, and ``fetch`` would materialise the
        whole corpus's byte arrays to do it - fine for a test fixture, fatal for a
        UniProt-scale build. This selects the id column alone.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"3Di store not found at {self.path}")
        unique = list(dict.fromkeys(ids))
        present = set()
        conn = sqlite3.connect(str(self.path))
        try:
            for start in range(0, len(unique), _SQLITE_VAR_CHUNK):
                chunk = unique[start:start + _SQLITE_VAR_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                present.update(
                    row[0] for row in conn.execute(
                        f"SELECT id FROM three_di WHERE id IN ({placeholders})", chunk))
        finally:
            conn.close()
        return set(unique) - present

    def count(self) -> int:
        if not self.path.exists():
            raise FileNotFoundError(f"3Di store not found at {self.path}")
        conn = sqlite3.connect(str(self.path))
        try:
            return conn.execute("SELECT COUNT(*) FROM three_di").fetchone()[0]
        finally:
            conn.close()
