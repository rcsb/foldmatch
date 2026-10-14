"""Versioned esm-codes scoring systems (plan section 8 P4; method spec S1-S16, ``docs/esm-codes-method.md``).

An :class:`EsmCodesSystem` names every number and convention that shapes esm-codes output: the ESM3 weights and
embedding tap, the encoders and transform (by content digest), the code layout, the Smith-Waterman setting, the panel,
the E-value table and its rules, the calibrated domain, and the Hugging Face bundle (repo, revision and file sha256s)
that holds the weights, the panel and the table. It is pure metadata: no torch, no numpy and no I/O.

Principles (plan section 1):

* **Versioned.** Registry entries are immutable. Any change, even a statistics-only one, is a new system name (D17);
  golden digests in ``tests/test_scoring_system.py`` fail CI on any edit of a frozen entry.
* **One source of truth.** The bundle identity (repo, revision, file sha256s) exists only in the registry entry;
  nothing else in FoldMatch names a bundle revision.
* **Refuse, don't degrade.** ``__post_init__`` refuses inconsistent values; ``from_json`` refuses unknown or missing
  fields; :func:`get` refuses unknown names and lists the known ones.

Digests: :meth:`EsmCodesSystem.digest` covers every field; :meth:`EsmCodesSystem.encoding_digest` covers only the
fields that determine the stored residue codes (``ENCODING_FIELDS``), so a statistics-only system can share a
database's codes but never its digest. Both are the first 16 hex digits of a sha256 over a tag line and the canonical
JSON (sorted keys, no whitespace, shortest round-trip floats), so they are identical on every platform.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

SYSTEM_DIGEST_TAG = "foldmatch-esm-codes-system/1"
ENCODING_DIGEST_TAG = "foldmatch-esm-codes-encoding/1"
DIGEST_HEX_DIGITS = 16

#: The system every ``--residue-codes`` build uses (v1 has no user choice). Queries always use the system named in the
#: database manifest, never this default.
DEFAULT_ESM_CODES = "esm-codes-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_HEX = re.compile(r"[0-9a-f]{40}")
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_FILE = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*")
TABLE_SIZE = 6


class UnknownScoringSystemError(ValueError):
    """No registry entry has this name (for a database: it was built by a newer FoldMatch)."""


# =====================================================================================================================
# value checks
# =====================================================================================================================
def _str(owner: str, name: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner}.{name} must be a non-empty string, got {value!r}")
    return value


def _hex(owner: str, name: str, value, pattern: re.Pattern, what: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{owner}.{name} must be {what}, got {value!r}")
    return value


def _int(owner: str, name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{owner}.{name} must be an integer, got {value!r}")
    return int(value)


def _float(owner: str, name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{owner}.{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{owner}.{name} must be finite, got {value!r}")
    return value


def _floats(owner: str, name: str, value, size: int) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(f"{owner}.{name} must be a sequence of {size} numbers, got {value!r}")
    out = tuple(_float(owner, f"{name}[{i}]", v) for i, v in enumerate(value))
    if len(out) != size:
        raise ValueError(f"{owner}.{name} must hold {size} numbers, got {len(out)}")
    return out


def _nested(cls, owner: str, name: str, value):
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_dict(value, where=f"{owner}.{name}")
    raise ValueError(f"{owner}.{name} must be a {cls.__name__} or a mapping, got {value!r}")


def _set(obj, name: str, value):
    object.__setattr__(obj, name, value)


def _fields_from(cls, data, where: str) -> dict:
    """The mapping's fields for ``cls``, refusing unknown and missing ones."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: expected a JSON object, got {type(data).__name__}")
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(data) - set(names))
    missing = [n for n in names if n not in data]
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown}; {cls.__name__} has {names}")
    if missing:
        raise ValueError(f"{where}: missing field(s) {missing}")
    return {n: data[n] for n in names}


def _plain(value):
    """JSON form of a field value: nested dataclasses become objects, tuples become lists."""
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def canonical_json(obj) -> str:
    """Sorted keys, no whitespace, ASCII, no NaN / Inf; Python's float repr is the shortest round-trip form, so the
    text (and every digest over it) is the same on every platform."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(tag: str, obj) -> str:
    return hashlib.sha256(tag.encode("ascii") + b"\n" + canonical_json(obj).encode("ascii")).hexdigest()[
        :DIGEST_HEX_DIGITS]


# =====================================================================================================================
# nested parts
# =====================================================================================================================
@dataclass(frozen=True)
class Normalisation:
    """S7 rowcol z: ``z_ij = 0.5 ((s_ij - mu_i) / (sigma_i + eps) + (s_ij - nu_j) / (tau_j + eps))`` with ``std``
    standard deviations; a row or column whose std is below ``degenerate_std_lt`` contributes exactly 0."""
    name: str
    std: str
    eps: float
    degenerate_std_lt: float

    def __post_init__(self):
        o = "Normalisation"
        _set(self, "name", _str(o, "name", self.name))
        _set(self, "std", _str(o, "std", self.std))
        for n in ("eps", "degenerate_std_lt"):
            v = _float(o, n, getattr(self, n))
            if v <= 0:
                raise ValueError(f"{o}.{n} must be > 0, got {v!r}")
            _set(self, n, v)

    @classmethod
    def from_dict(cls, data, where: str = "Normalisation") -> "Normalisation":
        return cls(**_fields_from(cls, data, where))


@dataclass(frozen=True)
class CalibratedDomain:
    """S13 / D14 / D15: a hit is inside the calibrated domain (``ecal = 1``) only when the query length is in
    ``[query_min_length, query_max_length]``, the target length is at most ``target_max_length``, both sequences have
    at most ``max_nonstandard_fraction`` residues outside the 20 standard amino acids, and the target has at least
    ``target_min_distinct_codes`` distinct residue codes."""
    query_min_length: int
    query_max_length: int
    target_max_length: int
    max_nonstandard_fraction: float
    target_min_distinct_codes: int

    def __post_init__(self):
        o = "CalibratedDomain"
        for n in ("query_min_length", "query_max_length", "target_max_length", "target_min_distinct_codes"):
            v = _int(o, n, getattr(self, n))
            if v < 1:
                raise ValueError(f"{o}.{n} must be >= 1, got {v!r}")
            _set(self, n, v)
        if self.query_min_length > self.query_max_length:
            raise ValueError(f"{o}: query_min_length {self.query_min_length} > query_max_length "
                             f"{self.query_max_length}")
        f = _float(o, "max_nonstandard_fraction", self.max_nonstandard_fraction)
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"{o}.max_nonstandard_fraction must be in [0, 1], got {f!r}")
        _set(self, "max_nonstandard_fraction", f)

    @classmethod
    def from_dict(cls, data, where: str = "CalibratedDomain") -> "CalibratedDomain":
        return cls(**_fields_from(cls, data, where))


# =====================================================================================================================
# the system
# =====================================================================================================================
IDENTITY_FIELDS = ("name", "calibration_version")
#: The fields that determine the stored residue codes' bytes (S1-S4); :meth:`EsmCodesSystem.encoding_digest`. The
#: encoder architecture and precision are among them: with the same weights, GELU-tanh instead of exact GELU, or a
#: different precision, changes the codes (P2 NotRecoverableFromWeightsTest), and the content digests cover only the
#: weight tensors.
ENCODING_FIELDS = (
    "esm_repo", "esm_weights_sha256", "embedding_tap", "input_normalisation",
    "input_cast", "transform_sha256", "transform_output_cast",
    "target_encoder_sha256", "encoder_architecture", "encoder_dtype", "code_bits", "code_bytes", "bitorder",
    "sign_rule",
)
SCORING_FIELDS = (
    "query_encoder_sha256", "query_dtype", "query_prep", "query_prep_eps", "similarity", "determinism",
    "normalisation", "offset", "gap_open", "gap_extend", "dp", "score_form", "score_diff",
)
STATISTICS_FIELDS = (
    "panel_sha256", "panel_size", "panel_residues", "panel_quantile", "quantile_method",
    "bin_centres", "lam", "lnK", "low_m_rule", "high_m_rule", "calibrated_domain", "degenerate_query_rule",
)
BUNDLE_FIELDS = ("bundle_repo", "bundle_revision", "bundle_files")

_STR_FIELDS = ("calibration_version", "esm_repo", "embedding_tap", "input_normalisation", "input_cast",
               "transform_output_cast", "encoder_architecture", "encoder_dtype", "bitorder", "sign_rule",
               "query_dtype", "query_prep", "similarity", "determinism", "dp", "score_form", "score_diff",
               "quantile_method", "low_m_rule", "high_m_rule", "degenerate_query_rule")
_SHA_FIELDS = ("esm_weights_sha256", "transform_sha256", "target_encoder_sha256", "query_encoder_sha256",
               "panel_sha256")


@dataclass(frozen=True)
class EsmCodesSystem:
    """One esm-codes scoring system (method spec S1-S16). Every field is required; see the module docstring.

    Construction normalises (sequences become tuples, integral numbers of float fields become floats, ``bundle_files``
    becomes ``((name, sha256), ...)`` sorted by name, nested mappings become :class:`Normalisation` /
    :class:`CalibratedDomain`) and refuses inconsistent values with :class:`ValueError`.
    """
    # identity
    name: str
    calibration_version: str
    # encoding (S1-S4): what the stored residue codes depend on
    esm_repo: str
    esm_weights_sha256: str
    embedding_tap: str
    input_normalisation: str
    input_cast: str
    transform_sha256: str
    transform_output_cast: str
    target_encoder_sha256: str
    encoder_architecture: str        # S3: the encoder module (bundle.json encoding.architecture.name)
    encoder_dtype: str               # S3: "float32" = IEEE fp32, no TF32, no autocast (encoding.encoder.dtype)
    code_bits: int
    code_bytes: int
    bitorder: str
    sign_rule: str
    # scoring (S5-S9)
    query_encoder_sha256: str
    query_dtype: str
    query_prep: str
    query_prep_eps: float
    similarity: str
    determinism: str
    normalisation: Normalisation
    offset: float
    gap_open: float
    gap_extend: float
    dp: str
    score_form: str
    score_diff: str
    # panel and statistics (S10-S13b)
    panel_sha256: str
    panel_size: int
    panel_residues: int
    panel_quantile: float
    quantile_method: str
    bin_centres: Tuple[float, ...]
    lam: Tuple[float, ...]
    lnK: Tuple[float, ...]
    low_m_rule: str
    high_m_rule: str
    calibrated_domain: CalibratedDomain
    degenerate_query_rule: str
    # bundle: the only place a bundle repo, revision or file sha256 is named
    bundle_repo: str
    bundle_revision: str
    bundle_files: Tuple[Tuple[str, str], ...]

    def __post_init__(self):
        o = "EsmCodesSystem"
        name = self.name
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"{o}.name must match {_NAME.pattern}, got {name!r}")
        for n in _STR_FIELDS:
            _set(self, n, _str(o, n, getattr(self, n)))
        for n in _SHA_FIELDS:
            _set(self, n, _hex(o, n, getattr(self, n), _SHA256, "a lowercase 64-hex sha256"))
        for n in ("code_bits", "code_bytes", "panel_size", "panel_residues"):
            v = _int(o, n, getattr(self, n))
            if v < 1:
                raise ValueError(f"{o}.{n} must be >= 1, got {v!r}")
            _set(self, n, v)
        if self.code_bits != 8 * self.code_bytes:
            raise ValueError(f"{o}: code_bits {self.code_bits} != 8 * code_bytes {self.code_bytes}")
        if self.panel_residues < self.panel_size:
            raise ValueError(f"{o}: panel_residues {self.panel_residues} < panel_size {self.panel_size}")
        for n in ("query_prep_eps", "offset", "gap_open", "gap_extend", "panel_quantile"):
            _set(self, n, _float(o, n, getattr(self, n)))
        if self.query_prep_eps <= 0:
            raise ValueError(f"{o}.query_prep_eps must be > 0, got {self.query_prep_eps!r}")
        if not self.gap_open >= self.gap_extend >= 0.0:
            raise ValueError(f"{o}: needs gap_open >= gap_extend >= 0, got gap_open={self.gap_open!r}, "
                             f"gap_extend={self.gap_extend!r}")
        if not 0.0 < self.panel_quantile < 1.0:
            raise ValueError(f"{o}.panel_quantile must be in (0, 1), got {self.panel_quantile!r}")
        for n in ("bin_centres", "lam", "lnK"):
            _set(self, n, _floats(o, n, getattr(self, n), TABLE_SIZE))
        c = self.bin_centres
        if c[0] <= 0 or any(b <= a for a, b in zip(c, c[1:])):
            raise ValueError(f"{o}.bin_centres must be positive and strictly increasing, got {c!r}")
        if any(v <= 0 for v in self.lam):
            raise ValueError(f"{o}.lam must be > 0, got {self.lam!r}")
        _set(self, "normalisation", _nested(Normalisation, o, "normalisation", self.normalisation))
        _set(self, "calibrated_domain", _nested(CalibratedDomain, o, "calibrated_domain", self.calibrated_domain))
        if not isinstance(self.bundle_repo, str) or not _REPO.fullmatch(self.bundle_repo):
            raise ValueError(f"{o}.bundle_repo must be a Hugging Face repo id 'owner/name', got {self.bundle_repo!r}")
        _set(self, "bundle_revision", _hex(o, "bundle_revision", self.bundle_revision, _COMMIT_HEX,
                                          "a lowercase 40-hex commit (never a branch, a tag or None)"))
        _set(self, "bundle_files", _bundle_files(o, self.bundle_files))

    # ---------------------------------------------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict: nested parts as objects, tuples as lists, ``bundle_files`` as ``[[name, sha256], ...]``."""
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def to_json(self) -> str:
        """Canonical JSON text (:func:`canonical_json`); ``from_json(s.to_json()) == s``."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data, where: str = "EsmCodesSystem") -> "EsmCodesSystem":
        return cls(**_fields_from(cls, data, where))

    @classmethod
    def from_json(cls, text) -> "EsmCodesSystem":
        """From JSON text or an already-parsed mapping; unknown and missing fields (at any level) are refused."""
        data = json.loads(text) if isinstance(text, (str, bytes, bytearray)) else text
        return cls.from_dict(data)

    def digest(self) -> str:
        """16 hex digits over every field."""
        return _digest(SYSTEM_DIGEST_TAG, self.to_dict())

    def encoding_digest(self) -> str:
        """16 hex digits over ``ENCODING_FIELDS`` only (what the stored residue codes depend on)."""
        d = self.to_dict()
        return _digest(ENCODING_DIGEST_TAG, {k: d[k] for k in ENCODING_FIELDS})

    def bundle_sha256(self, filename: str) -> str:
        """The pinned sha256 of one bundle file."""
        for name, sha in self.bundle_files:
            if name == filename:
                return sha
        raise KeyError(f"{self.name}: {filename!r} is not a bundle file ({[n for n, _ in self.bundle_files]})")


BUNDLE_JSON = "bundle.json"


def _bundle_files(o: str, value) -> Tuple[Tuple[str, str], ...]:
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError(f"{o}.bundle_files must be (name, sha256) pairs or a mapping, got {value!r}")
    else:
        items = []
        for pair in value:
            if isinstance(pair, (str, bytes)) or not hasattr(pair, "__iter__") or len(tuple(pair)) != 2:
                raise ValueError(f"{o}.bundle_files entries must be (name, sha256) pairs, got {pair!r}")
            items.append(tuple(pair))
    out = []
    for name, sha in items:
        if not isinstance(name, str) or not _FILE.fullmatch(name) or ".." in name.split("/"):
            raise ValueError(f"{o}.bundle_files: {name!r} is not a relative file name in the repo")
        out.append((name, _hex(o, f"bundle_files[{name!r}]", sha, _SHA256, "a lowercase 64-hex sha256")))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{o}.bundle_files names a file twice: {sorted(names)}")
    if BUNDLE_JSON not in names:
        raise ValueError(f"{o}.bundle_files must include {BUNDLE_JSON}, got {sorted(names)}")
    return tuple(sorted(out))


_PARTITION = IDENTITY_FIELDS + ENCODING_FIELDS + SCORING_FIELDS + STATISTICS_FIELDS + BUNDLE_FIELDS
if sorted(_PARTITION) != sorted(f.name for f in dataclasses.fields(EsmCodesSystem)) or \
        len(set(_PARTITION)) != len(_PARTITION):
    raise AssertionError("the field groups must partition EsmCodesSystem's fields")


# =====================================================================================================================
# registry
# =====================================================================================================================
def _registry(*systems: EsmCodesSystem) -> Mapping[str, EsmCodesSystem]:
    out: Dict[str, EsmCodesSystem] = {}
    for s in systems:
        if s.name in out:
            raise AssertionError(f"registry: {s.name} registered twice")
        out[s.name] = s
    return MappingProxyType(out)


#: Every system this FoldMatch can build and query, by name. Entries are immutable once frozen (golden digests in
#: tests/test_scoring_system.py); a change of any value is a new system name, never an edit (D17).
REGISTRY: Mapping[str, EsmCodesSystem] = _registry()


def known_names() -> Tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def get(name: str) -> EsmCodesSystem:
    """The registered system ``name``; :class:`UnknownScoringSystemError` (listing the known names) otherwise."""
    system = REGISTRY.get(name) if isinstance(name, str) else None
    if system is None:
        known = ", ".join(known_names()) or "none"
        raise UnknownScoringSystemError(
            f"unknown esm-codes scoring system {name!r}; this FoldMatch knows: {known}. A database built by a newer "
            "FoldMatch needs a FoldMatch upgrade.")
    return system
