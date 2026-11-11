"""Versioned esm-codes scoring systems (plan section 8 P4; method spec S1-S16, ``docs/esm-codes-method.md``).

An :class:`EsmCodesSystem` names every number and convention that shapes esm-codes output: the ESM3 weights and
embedding tap, the encoders and transform (by content digest), the code layout, the Smith-Waterman setting, the panel,
the E-value table and its rules, the calibrated domain, and the Hugging Face bundle (repo, revision and file sha256s)
that holds the weights, the panel and the table. It is pure metadata: no torch, no numpy and no I/O.

:func:`prepare` turns a system into a :class:`PreparedSystem`: it fetches the bundle from the registry pin
(:func:`foldmatch.utils.model.get_esm_codes_bundle`, sha256-verified) and refuses it unless ``bundle.json`` and the
scoring npz agree with the system exactly. :meth:`PreparedSystem.for_worker` gives Stage-2 workers a picklable
:class:`WorkerSpec` (paths and scalars). Importing this module loads neither torch nor numpy.

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
from pathlib import Path
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


class BundleUnavailableError(ValueError):
    """A bundle file is neither in the Hugging Face cache nor downloadable (offline mode, Hub unreachable)."""


class BundleMismatchError(ValueError):
    """The bundle does not match its registry entry: a file sha256, a ``bundle.json`` / ``meta_json`` field, the
    panel or the table differs, or a file is missing at the pinned revision. Never worked around."""


def is_commit(revision) -> bool:
    """A full 40-hex lowercase commit hash (what every bundle revision must be)."""
    return isinstance(revision, str) and _COMMIT_HEX.fullmatch(revision) is not None


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
REGISTRY: Mapping[str, EsmCodesSystem] = _registry(
    # esm-codes-v1: model M8_s0 (L2m_B8_s0), setting D, S = S_f - S_r, UniRef50 1,000-member panel, M3t999 table.
    # FROZEN 2026-10-14 (after P0.6, P0.7 and the upload of the bundle to rcsb/foldmatch-esm-codes). Every value
    # below equals the bundle's bundle.json / meta_json (prepare() checks it); the content digests are
    # foldmatch-tensor-digest/1; the table is panel/fit2/results/ship_final.json (md5 c51735f9...).
    EsmCodesSystem(
        name="esm-codes-v1",
        calibration_version="foldmatch-evalue-M8_s0-D_fmr-pnlU50s1000-M3t999-v1",
        esm_repo="rcsb/rcsb-esm",
        esm_weights_sha256="5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9",
        embedding_tap="per_residue_embedding",
        input_normalisation="ascii_upper_nows_v1",
        input_cast="float16",
        transform_sha256="e8989b1cefc839784555ff973dabb9a2f7ef4bdf81ac78ad47767714227c3336",
        transform_output_cast="float16",
        target_encoder_sha256="58da263e005a79fd1689b0f5605adbb85b51532bca99e4ef7330916a50df6686",
        encoder_architecture="esm-codes-mlp-v1",
        encoder_dtype="float32",
        code_bits=64,
        code_bytes=8,
        bitorder="big",
        sign_rule="bit_k = (h_k >= 0)",
        query_encoder_sha256="d8bed195068be23a60586632fa471a08f3a767fe8cf34b06aa6e733d90cd3bf8",
        query_dtype="float16",
        query_prep="fp64_seq_k_asc",
        query_prep_eps=1e-12,
        similarity="pm1_fp32_blas_gemm_per_pair",
        determinism="tolerance_s16_v1",
        normalisation=Normalisation(name="rowcol_z", std="population", eps=1e-08, degenerate_std_lt=1e-06),
        offset=1.0,
        gap_open=5.0,
        gap_extend=0.5,
        dp="fmsw/1",
        score_form="fwd_minus_rev_view",
        score_diff="float64",
        panel_sha256="7aea2b00a24ee41c5a9f9adb91a5723ee02dff12dca23998767ab87da4b9dcf7",
        panel_size=1000,
        panel_residues=355626,
        panel_quantile=0.9,
        quantile_method="linear",
        bin_centres=(32.58437573116228, 50.34161310783617, 80.8700900525453, 149.85910754690784,
                     276.49344608131264, 594.9175724688861),
        lam=(0.18775162858061797, 0.1227859873586807, 0.10488523441851651, 0.08222582327223689,
             0.04784353519669357, 0.041619283794126286),
        lnK=(-13.42382627313042, -13.969792295008805, -13.73906582316063, -13.454607332784638,
             -14.484498724905178, -15.878915750325046),
        low_m_rule="clamp",
        high_m_rule="clamp",
        calibrated_domain=CalibratedDomain(query_min_length=21, query_max_length=966, target_max_length=3000,
                                           max_nonstandard_fraction=0.01, target_min_distinct_codes=2),
        degenerate_query_rule="no_hits",
        bundle_repo="rcsb/foldmatch-esm-codes",
        bundle_revision="3a14f46b72f5971f47203a57ba418e87054142bd",
        bundle_files=(
            ("LICENSE", "49f7a86fb623fa33dcb951fd53f840795989eadb9f03dda5c0587a2f451258d1"),
            ("README.md", "70bd8e84c48b2a511ec3b37cededa7eaa1f46e7b3da6ea86f5881126bc02416c"),
            ("bundle.json", "dcd29bddbff0bb7dc93209bbe292d422d7bff21dd48c80f6e1f6b74f23789fe8"),
            ("esm_codes_v1_encoders.pt", "1696ab91089ece8bce787d38b9d4d68bbc8ca4abff06f836f6e9363a23a1c02b"),
            ("esm_codes_v1_scoring.npz", "a5bebd626ec874626fb9372a63d10c32a001ed09c358717448bc9e1fd0ddedd6"),
        ),
    ),
)


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


def registered(system: EsmCodesSystem) -> EsmCodesSystem:
    """``system``, refused unless it is exactly this FoldMatch's registry entry of its name.

    A system that comes from elsewhere (parsed from a database manifest, say) and differs from the entry of its name
    in any value, the bundle repo, revision and file sha256s included, is not that system: nothing is fetched or run
    for it. :class:`UnknownScoringSystemError` names the differing fields (or lists the known names)."""
    if not isinstance(system, EsmCodesSystem):
        raise TypeError(f"expected an EsmCodesSystem, got {type(system).__name__}")
    entry = get(system.name)
    if entry != system:
        diff = [f.name for f in dataclasses.fields(EsmCodesSystem) if getattr(entry, f.name) != getattr(system, f.name)]
        raise UnknownScoringSystemError(
            f"this {system.name} differs from this FoldMatch's registry entry {system.name} in {diff}: only the "
            "registered system is used, exactly as registered (any change is a new system name)")
    return system


# =====================================================================================================================
# what this FoldMatch implements
# =====================================================================================================================
class UnsupportedSystemError(ValueError):
    """The system names a method this FoldMatch does not implement (a newer DP, rule or encoding)."""


BUNDLE_FORMAT = "foldmatch-esm-codes-bundle/1"
ENCODERS_FORMAT = "foldmatch-esm-codes-encoders/1"      # = foldmatch.model.residue_code_encoder.ENCODERS_FORMAT
SCORING_FORMAT = "foldmatch-esm-codes-scoring/1"

#: Method values hard-coded by this FoldMatch's encoder module, kernel (fmsw/1) and statistics. A system naming
#: anything else is refused by :func:`check_implemented`, never run with this code's arithmetic.
IMPLEMENTED = MappingProxyType({
    "embedding_tap": "per_residue_embedding",
    "input_normalisation": "ascii_upper_nows_v1",
    "input_cast": "float16",
    "transform_output_cast": "float16",
    "code_bits": 64,
    "code_bytes": 8,
    "bitorder": "big",
    "sign_rule": "bit_k = (h_k >= 0)",
    "encoder_architecture": "esm-codes-mlp-v1",     # = a residue_code_encoder.ARCHITECTURES key
    "encoder_dtype": "float32",                     # = residue_code_encoder.IMPLEMENTED_CONVENTIONS encoder.dtype
    "query_dtype": "float16",
    "query_prep": "fp64_seq_k_asc",
    "query_prep_eps": 1e-12,
    "similarity": "pm1_fp32_blas_gemm_per_pair",
    "determinism": "tolerance_s16_v1",
    "normalisation": Normalisation(name="rowcol_z", std="population", eps=1e-08, degenerate_std_lt=1e-06),
    "score_form": "fwd_minus_rev_view",
    "score_diff": "float64",
    "quantile_method": "linear",
    "degenerate_query_rule": "no_hits",
})


def check_implemented(system: EsmCodesSystem) -> None:
    """Refuse (:class:`UnsupportedSystemError`) a system this FoldMatch cannot run exactly: its DP must be the
    kernel's (``swkern.DP_ID``, ``fmsw/1``) with setting D (``swkern.SETTING_D``), its m rules ones that
    :mod:`code_statistics` implements, and every value in ``IMPLEMENTED`` must match."""
    from foldmatch.search import code_statistics, swkern       # numpy + ctypes only; loads no library
    bad = []
    if system.dp != swkern.DP_ID:
        bad.append(f"dp={system.dp!r} (this kernel is {swkern.DP_ID!r})")
    setting = (system.offset, system.gap_open, system.gap_extend)
    kernel_setting = (swkern.SETTING_D.offset, swkern.SETTING_D.gap_open, swkern.SETTING_D.gap_extend)
    if setting != kernel_setting:
        bad.append(f"(offset, gap_open, gap_extend)={setting} (fmsw/1 setting D is {kernel_setting})")
    for rule in ("low_m_rule", "high_m_rule"):
        if getattr(system, rule) not in code_statistics.IMPLEMENTED_M_RULES:
            bad.append(f"{rule}={getattr(system, rule)!r} (implemented: {code_statistics.IMPLEMENTED_M_RULES})")
    if (system.code_bits, system.code_bytes) != (swkern.DIM, swkern.CODE_BYTES):
        bad.append(f"code layout {system.code_bits} bits / {system.code_bytes} bytes (the kernel reads "
                   f"{swkern.DIM} / {swkern.CODE_BYTES})")
    for name, value in IMPLEMENTED.items():
        got = getattr(system, name)
        if got != value or type(got) is not type(value):
            bad.append(f"{name}={got!r} (implemented: {value!r})")
    if bad:
        raise UnsupportedSystemError(f"{system.name}: this FoldMatch does not implement " + "; ".join(bad))


# =====================================================================================================================
# prepare(): the verified bundle of a system
# =====================================================================================================================
def _both(field_path, doc_path):
    return field_path, doc_path, doc_path


#: (system field path, bundle.json path, scoring npz meta_json path or None): every system value that the bundle
#: also states. prepare() requires each pair to be equal exactly (numbers as float64, strings as text).
DOCUMENT_FIELDS = (
    _both(("name",), ("system",)),
    _both(("calibration_version",), ("calibration_version",)),
    _both(("esm_repo",), ("esm3", "repo")),
    _both(("esm_weights_sha256",), ("esm3", "weights_sha256")),
    _both(("embedding_tap",), ("esm3", "embedding_tap")),
    _both(("input_normalisation",), ("esm3", "input_normalisation")),
    (("input_cast",), ("encoding", "input_cast"), None),
    (("transform_sha256",), ("content_digests", "transform_sha256"), None),
    (("transform_output_cast",), ("encoding", "transform_output_cast"), None),
    (("target_encoder_sha256",), ("content_digests", "target_encoder_sha256"), None),
    (("encoder_architecture",), ("encoding", "architecture", "name"), None),
    (("encoder_dtype",), ("encoding", "encoder", "dtype"), None),
    (("code_bits",), ("encoding", "code_bits"), ("code", "bits")),
    (("code_bytes",), ("encoding", "code_bytes"), ("code", "bytes")),
    (("bitorder",), ("encoding", "bitorder"), ("code", "bitorder")),
    (("sign_rule",), ("encoding", "sign_rule"), ("code", "sign_rule")),
    (("query_encoder_sha256",), ("content_digests", "query_encoder_sha256"), None),
    (("query_dtype",), ("encoding", "query_dtype"), None),
    _both(("query_prep",), ("scoring", "query_prep")),
    _both(("query_prep_eps",), ("scoring", "query_prep_eps")),
    _both(("similarity",), ("scoring", "similarity")),
    _both(("determinism",), ("scoring", "determinism")),
    *(_both(("normalisation", k), ("scoring", "normalisation", k)) for k in ("name", "std", "eps",
                                                                           "degenerate_std_lt")),
    *(_both((k,), ("scoring", k)) for k in ("offset", "gap_open", "gap_extend", "dp", "score_form", "score_diff")),
    _both(("panel_sha256",), ("panel", "panel_sha256")),
    (("panel_sha256",), ("content_digests", "panel_sha256"), None),
    _both(("panel_size",), ("panel", "size")),
    _both(("panel_residues",), ("panel", "residues")),
    _both(("panel_quantile",), ("panel", "quantile")),
    _both(("quantile_method",), ("panel", "quantile_method")),
    *(_both((k,), ("statistics", k)) for k in ("bin_centres", "lam", "lnK", "low_m_rule", "high_m_rule")),
    *(_both(("calibrated_domain", k), ("calibrated_domain", k)) for k in (
        "query_min_length", "query_max_length", "target_max_length", "max_nonstandard_fraction",
        "target_min_distinct_codes")),
    _both(("degenerate_query_rule",), ("degenerate_query_rule",)),
    (("bundle_repo",), ("hf_repo",), None),
)
_MISSING = object()


def _lookup(obj, path):
    for key in path:
        if dataclasses.is_dataclass(obj):
            obj = getattr(obj, key)
        elif isinstance(obj, Mapping) and key in obj:
            obj = obj[key]
        else:
            return _MISSING
    return obj


def _same(want, got) -> bool:
    """Exact equality: strings as text, integers as integers, floats as float64 (an integral JSON number equals the
    same float), sequences element by element. bool never equals a number."""
    if isinstance(want, bool) or isinstance(got, bool):
        return type(want) is type(got) and want == got
    if isinstance(want, str):
        return isinstance(got, str) and got == want
    if isinstance(want, int):
        return isinstance(got, int) and got == want
    if isinstance(want, float):
        return isinstance(got, (int, float)) and float(got) == want
    if isinstance(want, tuple):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(map(_same, want, got))
    return want == got


def _compare(system: EsmCodesSystem, doc, column: int, what: str, problems: list):
    for entry in DOCUMENT_FIELDS:
        doc_path = entry[column]
        if doc_path is None:
            continue
        want, got = _lookup(system, entry[0]), _lookup(doc, doc_path)
        if got is _MISSING:
            problems.append(f"{what} has no {'.'.join(doc_path)}")
        elif not _same(want, got):
            problems.append(f"{what} {'.'.join(doc_path)} = {got!r}, the system has {'.'.join(entry[0])} = "
                            f"{want!r}")


def _table_digest(system: EsmCodesSystem) -> str:
    import numpy as np
    from foldmatch.utils.tensor_digest import tensor_digest
    return tensor_digest({k: np.array(getattr(system, k), np.float64) for k in ("bin_centres", "lam", "lnK")})


def _panel_digest(codes, offsets) -> str:
    from foldmatch.utils.tensor_digest import tensor_digest
    return tensor_digest({"codes": codes, "offsets": offsets})


def _bundle_roles(system: EsmCodesSystem, bundle: Mapping, problems: list) -> Dict[str, str]:
    """bundle.json ``files``: each listed file pinned with the same sha256 in the system; the encoders and scoring
    files identified by their format tags."""
    files = bundle.get("files")
    roles: Dict[str, str] = {}
    if not isinstance(files, Mapping) or not files:
        problems.append("bundle.json has no files")
        return roles
    pinned = dict(system.bundle_files)
    for name, info in files.items():
        sha = info.get("sha256") if isinstance(info, Mapping) else None
        if name not in pinned:
            problems.append(f"bundle.json lists {name}, which the system does not pin")
        elif sha != pinned[name]:
            problems.append(f"bundle.json gives {name} sha256 {sha}, the system pins {pinned[name]}")
        fmt = info.get("format") if isinstance(info, Mapping) else None
        if fmt in (ENCODERS_FORMAT, SCORING_FORMAT):
            if fmt in roles:
                problems.append(f"bundle.json lists two {fmt} files")
            roles[fmt] = name
    for fmt in (ENCODERS_FORMAT, SCORING_FORMAT):
        if fmt not in roles:
            problems.append(f"bundle.json lists no {fmt} file")
    return roles


def _load_scoring_npz(path) -> Dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _check_scoring(system: EsmCodesSystem, arrays: Mapping[str, Any], problems: list):
    import numpy as np
    meta_raw = arrays.get("meta_json")
    try:
        meta = json.loads(str(meta_raw)) if meta_raw is not None else None
    except ValueError:
        meta = None
    if not isinstance(meta, Mapping):
        problems.append("the scoring npz has no readable meta_json")
    else:
        if meta.get("format") != SCORING_FORMAT:
            problems.append(f"the scoring npz format is {meta.get('format')!r}, expected {SCORING_FORMAT!r}")
        _compare(system, meta, 2, "meta_json", problems)
    for k in ("bin_centres", "lam", "lnK"):
        a = arrays.get(k)
        want = np.array(getattr(system, k), np.float64)
        if a is None or a.dtype != np.float64 or a.shape != want.shape:
            problems.append(f"the scoring npz {k} is not float64 {list(want.shape)}")
        elif not np.array_equal(a.view(np.uint64), want.view(np.uint64)):
            first = int(np.flatnonzero(a.view(np.uint64) != want.view(np.uint64))[0])
            problems.append(f"the scoring npz {k} = {a.tolist()} differs from the system's {want.tolist()} at index "
                            f"{first} (exact float64 comparison)")
    codes, offsets, lengths = (arrays.get(k) for k in ("panel_codes", "panel_offsets", "panel_lengths"))
    shape_ok = (codes is not None and codes.dtype == np.uint8 and codes.shape == (system.panel_residues,
                                                                                  system.code_bytes)
                and offsets is not None and offsets.dtype == np.int64 and offsets.shape == (system.panel_size + 1,))
    if not shape_ok:
        problems.append(f"the scoring npz panel is not uint8 [{system.panel_residues}, {system.code_bytes}] codes "
                        f"with int64 [{system.panel_size + 1}] offsets")
        return
    if offsets[0] != 0 or offsets[-1] != system.panel_residues or np.any(np.diff(offsets) < 1):
        problems.append("the scoring npz panel offsets are not 0 .. panel_residues with every member non-empty")
    if lengths is None or lengths.shape != (system.panel_size,) or not np.array_equal(lengths, np.diff(offsets)):
        problems.append("the scoring npz panel_lengths do not equal diff(panel_offsets)")
    got = _panel_digest(codes, offsets)
    if got != system.panel_sha256:
        problems.append(f"the panel content digest is {got}, the system has panel_sha256 {system.panel_sha256}")


def _raise_problems(system: EsmCodesSystem, problems: list):
    if problems:
        shown = problems[:10] + ([f"... and {len(problems) - 10} more"] if len(problems) > 10 else [])
        raise BundleMismatchError(f"{system.name}: the bundle {system.bundle_repo}@{system.bundle_revision} does "
                                  "not match the registry entry: " + "; ".join(shown))


@dataclass(frozen=True)
class WorkerSpec:
    """What a Stage-2 worker needs, as picklable paths and scalars (no arrays, no torch): the scoring npz (sha256
    re-checked by :meth:`load_panel`), the DP setting, the panel quantile and the E-value table with its rules (so a
    WorkerSpec is also a ``table`` for :mod:`code_statistics`)."""
    system_name: str
    system_digest: str
    scoring_npz: str
    scoring_sha256: str
    dp: str
    offset: float
    gap_open: float
    gap_extend: float
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

    def dp_params(self):
        from foldmatch.search import swkern
        return swkern.DPParams(offset=self.offset, gap_open=self.gap_open, gap_extend=self.gap_extend)

    def load_panel(self):
        """``(codes uint8 [panel_residues, 8], offsets int64 [panel_size + 1])`` from the scoring npz, after checking
        its sha256 again (the file may have changed since prepare())."""
        from foldmatch.utils.model import _file_sha256
        got = _file_sha256(self.scoring_npz)
        if got != self.scoring_sha256:
            raise BundleMismatchError(f"{self.scoring_npz} has sha256 {got}, expected {self.scoring_sha256} "
                                      f"({self.system_name})")
        a = _load_scoring_npz(self.scoring_npz)
        codes, offsets = a["panel_codes"], a["panel_offsets"]
        if codes.shape[0] != self.panel_residues or len(offsets) != self.panel_size + 1:
            raise BundleMismatchError(f"{self.scoring_npz}: panel shape differs from {self.system_name}")
        return codes, offsets


@dataclass(frozen=True, eq=False)
class PreparedSystem:
    """A system whose bundle has been fetched and checked against it (:func:`prepare`)."""
    system: EsmCodesSystem
    files: Mapping[str, Any]                 # file name -> local path (sha256-verified)
    bundle: Mapping[str, Any]                # parsed bundle.json
    encoders_file: str
    scoring_file: str

    @property
    def encoders_path(self):
        return self.files[self.encoders_file]

    @property
    def scoring_path(self):
        return self.files[self.scoring_file]

    def for_worker(self) -> WorkerSpec:
        s = self.system
        return WorkerSpec(
            system_name=s.name, system_digest=s.digest(), scoring_npz=str(self.scoring_path),
            scoring_sha256=s.bundle_sha256(self.scoring_file), dp=s.dp, offset=s.offset, gap_open=s.gap_open,
            gap_extend=s.gap_extend, panel_size=s.panel_size, panel_residues=s.panel_residues,
            panel_quantile=s.panel_quantile, quantile_method=s.quantile_method, bin_centres=s.bin_centres,
            lam=s.lam, lnK=s.lnK, low_m_rule=s.low_m_rule, high_m_rule=s.high_m_rule,
            calibrated_domain=s.calibrated_domain, degenerate_query_rule=s.degenerate_query_rule)

    def load_head(self, device="cpu"):
        """The residue-code head (torch) from the encoders file, refused unless its content digests, architecture,
        encoder dtype and system name equal the system's and the system's ESM3 weights are the ones this FoldMatch
        loads."""
        from foldmatch.model.residue_code_encoder import load_esm_codes_encoders
        from foldmatch.utils.esm import loaders
        s = self.system
        if (s.esm_repo, s.esm_weights_sha256) != (loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_SHA256):
            raise UnsupportedSystemError(
                f"{s.name} needs ESM3 weights {s.esm_weights_sha256} from {s.esm_repo}; this FoldMatch loads "
                f"{loaders.ESM_WEIGHTS_SHA256} from {loaders.ESM_REPO_ID}")
        head = load_esm_codes_encoders(self.encoders_path, sha256=s.bundle_sha256(self.encoders_file), device=device)
        got = head.content_digests()
        want = {k: getattr(s, k) for k in ("transform_sha256", "query_encoder_sha256", "target_encoder_sha256")}
        problems = [f"{k} = {got.get(k)}, the system has {v}" for k, v in want.items() if got.get(k) != v]
        arch = head.meta.get("architecture") if isinstance(head.meta.get("architecture"), Mapping) else {}
        if arch.get("name") != s.encoder_architecture:
            problems.append(f"the encoders file architecture is {arch.get('name')!r}, the system has "
                            f"encoder_architecture {s.encoder_architecture!r}")
        conventions = head.meta.get("conventions") if isinstance(head.meta.get("conventions"), Mapping) else {}
        encoder = conventions.get("encoder") if isinstance(conventions.get("encoder"), Mapping) else {}
        if encoder.get("dtype") != s.encoder_dtype:
            problems.append(f"the encoders file encoder dtype is {encoder.get('dtype')!r}, the system has "
                            f"encoder_dtype {s.encoder_dtype!r}")
        if head.meta.get("system") != s.name:
            problems.append(f"the encoders file is for system {head.meta.get('system')!r}")
        _raise_problems(s, problems)
        return head


def prepare(system: EsmCodesSystem, *, cache_dir=None, local_files_only: bool = False) -> PreparedSystem:
    """Fetch the bundle of ``system`` (a registry entry, see :func:`registered`) from its registry pin
    (:func:`foldmatch.utils.model.get_esm_codes_bundle`, sha256 verified) and refuse it unless ``bundle.json`` and
    the scoring npz agree with the system exactly.

    Checked, deterministic and, once the bundle is cached, read-only (so it can run on every rank before ESM3):
    :func:`check_implemented`; that ``system`` is the registry entry of its name; ``bundle.json`` format, every
    ``DOCUMENT_FIELDS`` value (numbers as float64), that every file it lists is pinned with the same sha256, and its
    table digest; the scoring npz ``meta_json`` likewise, its table arrays bit for bit, and the panel's shape, offsets
    and content digest. Any difference raises :class:`BundleMismatchError` naming the fields. torch is not needed
    (only :meth:`PreparedSystem.load_head` imports it).

    ``local_files_only=True`` never downloads: a bundle file that is not cached raises
    :class:`BundleUnavailableError`. Use it for the every-rank pre-flight after one process (``download-models``, or
    rank 0) has fetched the bundle, so that no rank writes to the cache and a network failure cannot split the ranks.
    """
    if not isinstance(system, EsmCodesSystem):
        raise TypeError(f"prepare needs an EsmCodesSystem, got {type(system).__name__}")
    check_implemented(system)
    from foldmatch.utils.model import get_esm_codes_bundle
    from foldmatch.utils.tensor_digest import DIGEST_METHOD
    files = get_esm_codes_bundle(system, cache_dir=cache_dir, local_files_only=local_files_only)
    problems: list = []
    try:
        bundle = json.loads(Path(files[BUNDLE_JSON]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BundleMismatchError(f"{system.name}: {files[BUNDLE_JSON]} is not readable JSON: {e}") from e
    if not isinstance(bundle, Mapping) or bundle.get("bundle_format") != BUNDLE_FORMAT:
        got = bundle.get("bundle_format") if isinstance(bundle, Mapping) else None
        raise BundleMismatchError(f"{system.name}: bundle.json format {got!r}, this FoldMatch reads "
                                  f"{BUNDLE_FORMAT!r}")
    _compare(system, bundle, 1, "bundle.json", problems)
    digests = bundle.get("content_digests") if isinstance(bundle.get("content_digests"), Mapping) else {}
    if digests.get("method") != DIGEST_METHOD:
        problems.append(f"bundle.json content_digests.method = {digests.get('method')!r}, expected {DIGEST_METHOD!r}")
    table = _table_digest(system)
    if digests.get("table_sha256") != table:
        problems.append(f"bundle.json content_digests.table_sha256 = {digests.get('table_sha256')!r}, the system's "
                        f"table digests to {table}")
    roles = _bundle_roles(system, bundle, problems)
    if SCORING_FORMAT in roles and roles[SCORING_FORMAT] in files:
        _check_scoring(system, _load_scoring_npz(files[roles[SCORING_FORMAT]]), problems)
    _raise_problems(system, problems)
    return PreparedSystem(system=system, files=MappingProxyType(dict(files)), bundle=bundle,
                          encoders_file=roles[ENCODERS_FORMAT], scoring_file=roles[SCORING_FORMAT])
