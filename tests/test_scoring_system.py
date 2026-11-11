"""P4: the versioned esm-codes scoring system, its bundle loader and download-models (plan section 8 P4).

The structural tests run on a synthetic system (``_synthetic``) whose values only need to be well formed. The bundle
tests write a synthetic bundle into a Hugging Face cache layout (``_SyntheticBundle``) and fetch it through the real
``hf_hub_download`` (a cached file at a commit revision needs no network), or mock the hub where the calls themselves
are under test.
"""
import copy
import dataclasses
import hashlib
import io
import json
import math
import os
import pickle
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

import numpy as np
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from typer.testing import CliRunner

import foldmatch
from foldmatch.search import code_statistics as cs
from foldmatch.search import scoring_system as ss
from foldmatch.search import swkern
from foldmatch.search.scoring_system import (
    BUNDLE_FIELDS, ENCODING_FIELDS, IDENTITY_FIELDS, SCORING_FIELDS, STATISTICS_FIELDS, BundleMismatchError,
    BundleUnavailableError, CalibratedDomain, EsmCodesSystem, Normalisation, UnknownScoringSystemError,
    UnsupportedSystemError,
)
from foldmatch.types.api_types import DEFAULT_SCORING, ScoringChoice
from foldmatch.utils import model
from foldmatch.utils.tensor_digest import tensor_digest


def _synthetic_dict(**overrides):
    d = {
        "name": "esm-codes-test",
        "calibration_version": "test-calibration-v1",
        "esm_repo": "owner/esm",
        "esm_weights_sha256": "1" * 64,
        "embedding_tap": "per_residue_embedding",
        "input_normalisation": "ascii_upper_nows_v1",
        "input_cast": "float16",
        "transform_sha256": "2" * 64,
        "transform_output_cast": "float16",
        "target_encoder_sha256": "3" * 64,
        "encoder_architecture": "esm-codes-mlp-v1",
        "encoder_dtype": "float32",
        "code_bits": 64,
        "code_bytes": 8,
        "bitorder": "big",
        "sign_rule": "bit_k = (h_k >= 0)",
        "query_encoder_sha256": "4" * 64,
        "query_dtype": "float16",
        "query_prep": "fp64_seq_k_asc",
        "query_prep_eps": 1e-12,
        "similarity": "pm1_fp32_blas_gemm_per_pair",
        "determinism": "tolerance_s16_v1",
        "normalisation": {"name": "rowcol_z", "std": "population", "eps": 1e-08, "degenerate_std_lt": 1e-06},
        "offset": 1.0,
        "gap_open": 5.0,
        "gap_extend": 0.5,
        "dp": "fmsw/1",
        "score_form": "fwd_minus_rev_view",
        "score_diff": "float64",
        "panel_sha256": "5" * 64,
        "panel_size": 3,
        "panel_residues": 40,
        "panel_quantile": 0.9,
        "quantile_method": "linear",
        "bin_centres": [30.0, 50.0, 80.0, 150.0, 280.0, 600.0],
        "lam": [0.19, 0.12, 0.1, 0.08, 0.05, 0.04],
        "lnK": [-13.4, -14.0, -13.7, -13.5, -14.5, -15.9],
        "low_m_rule": "clamp",
        "high_m_rule": "clamp",
        "calibrated_domain": {"query_min_length": 21, "query_max_length": 966, "target_max_length": 3000,
                              "max_nonstandard_fraction": 0.01, "target_min_distinct_codes": 2},
        "degenerate_query_rule": "no_hits",
        "bundle_repo": "owner/bundle",
        "bundle_revision": "a" * 40,
        "bundle_files": [["bundle.json", "6" * 64], ["encoders.pt", "7" * 64], ["scoring.npz", "8" * 64]],
    }
    d.update(overrides)
    return d


def _synthetic(**overrides):
    return EsmCodesSystem.from_dict(_synthetic_dict(**overrides))


def _flip_hex(h):
    return h[:-1] + ("0" if h[-1] != "0" else "1")


def _up(x):
    return math.nextafter(x, math.inf)


def _mutations(d):
    """(label, mutated dict) for every leaf of the system's JSON form; each mutation keeps the system valid."""
    out = []
    for key, value in d.items():
        if key in ("normalisation", "calibrated_domain"):
            for sub, v in value.items():
                m = copy.deepcopy(d)
                m[key][sub] = (v + "x") if isinstance(v, str) else (v + 1 if isinstance(v, int) else _up(v))
                out.append((f"{key}.{sub}", m))
            continue
        m = copy.deepcopy(d)
        if key in ("code_bits", "code_bytes"):
            m["code_bits"], m["code_bytes"] = 128, 16
        elif key == "bundle_files":
            m[key][1][1] = _flip_hex(m[key][1][1])
        elif key in ("bin_centres", "lam", "lnK"):
            m[key][-1] = _up(m[key][-1])
        elif key.endswith("sha256") or key == "bundle_revision":
            m[key] = _flip_hex(value)
        elif isinstance(value, str):
            m[key] = value + "x"
        elif isinstance(value, int):
            m[key] = value + 1
        else:
            m[key] = _up(value)
        out.append((key, m))
    return out


class RoundTripTest(unittest.TestCase):

    def test_json_round_trip(self):
        s = _synthetic()
        text = s.to_json()
        self.assertEqual(EsmCodesSystem.from_json(text), s)
        self.assertEqual(EsmCodesSystem.from_json(json.loads(text)), s)
        self.assertEqual(EsmCodesSystem.from_json(text).digest(), s.digest())
        self.assertEqual(EsmCodesSystem.from_json(text).encoding_digest(), s.encoding_digest())
        self.assertEqual(text, json.dumps(json.loads(text), sort_keys=True, separators=(",", ":")))
        self.assertNotIn(" ", text.replace("bit_k = (h_k >= 0)", ""))

    def test_floats_survive_the_text_exactly(self):
        lam = [0.18775162858061797, 0.1227859873586807, 0.10488523441851651, 0.08222582327223689,
               0.04784353519669357, 0.041619283794126286]
        s = _synthetic(lam=lam)
        self.assertEqual(EsmCodesSystem.from_json(s.to_json()).lam, tuple(lam))

    def test_digests_are_16_hex(self):
        s = _synthetic()
        for d in (s.digest(), s.encoding_digest()):
            self.assertRegex(d, r"^[0-9a-f]{16}$")
        self.assertNotEqual(s.digest(), s.encoding_digest())

    def test_digest_is_hashable_and_frozen(self):
        s = _synthetic()
        self.assertEqual(hash(s), hash(_synthetic()))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.lam = (1.0,) * 6
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.normalisation.eps = 1.0


class DigestSensitivityTest(unittest.TestCase):
    """Every field changes digest(); exactly the ENCODING_FIELDS change encoding_digest() (so statistics, panel,
    scoring and bundle changes never flip it)."""

    def test_every_leaf(self):
        base = _synthetic()
        d = base.to_dict()
        seen = set()
        for label, m in _mutations(d):
            with self.subTest(field=label):
                s = EsmCodesSystem.from_dict(m)
                self.assertNotEqual(s, base)
                self.assertNotEqual(s.digest(), base.digest())
                top = label.split(".")[0]
                seen.add(top)
                if top in ENCODING_FIELDS:
                    self.assertNotEqual(s.encoding_digest(), base.encoding_digest())
                else:
                    self.assertEqual(s.encoding_digest(), base.encoding_digest())
        self.assertEqual(seen, {f.name for f in dataclasses.fields(EsmCodesSystem)})

    def test_architecture_and_precision_flip_the_encoding_digest(self):
        base = _synthetic()
        for change in (dict(encoder_architecture="esm-codes-mlp-v2"), dict(encoder_dtype="bfloat16")):
            with self.subTest(**change):
                self.assertNotEqual(dataclasses.replace(base, **change).encoding_digest(), base.encoding_digest())

    def test_statistics_fields_do_not_flip_the_encoding_digest(self):
        base = _synthetic()
        s = dataclasses.replace(base, lam=tuple(v * 1.01 for v in base.lam), lnK=tuple(v - 1 for v in base.lnK),
                                panel_quantile=0.95, high_m_rule="extrapolate", calibration_version="other",
                                bundle_revision="b" * 40, name="esm-codes-test2", offset=0.5)
        self.assertEqual(s.encoding_digest(), base.encoding_digest())
        self.assertNotEqual(s.digest(), base.digest())


class NormalisationTest(unittest.TestCase):

    def test_lists_become_tuples(self):
        s = _synthetic()
        for name in ("bin_centres", "lam", "lnK", "bundle_files"):
            self.assertIsInstance(getattr(s, name), tuple, name)
        self.assertTrue(all(isinstance(p, tuple) for p in s.bundle_files))
        self.assertIsInstance(s.normalisation, Normalisation)
        self.assertIsInstance(s.calibrated_domain, CalibratedDomain)
        r = dataclasses.replace(s, lam=list(s.lam))
        self.assertIsInstance(r.lam, tuple)
        self.assertEqual(r, s)

    def test_integral_floats_and_file_order_do_not_change_the_digest(self):
        s = _synthetic()
        r = _synthetic(offset=1, gap_open=5, bin_centres=[30, 50, 80, 150, 280, 600],
                       bundle_files={"scoring.npz": "8" * 64, "bundle.json": "6" * 64, "encoders.pt": "7" * 64})
        self.assertEqual(r, s)
        self.assertEqual(r.digest(), s.digest())
        self.assertIsInstance(r.offset, float)
        self.assertEqual([n for n, _ in r.bundle_files], ["bundle.json", "encoders.pt", "scoring.npz"])

    def test_bundle_sha256(self):
        s = _synthetic()
        self.assertEqual(s.bundle_sha256("encoders.pt"), "7" * 64)
        with self.assertRaises(KeyError):
            s.bundle_sha256("README.md")


class UnknownFieldsTest(unittest.TestCase):

    def test_unknown_top_level_field(self):
        with self.assertRaisesRegex(ValueError, r"unknown field\(s\) \['colour'\]"):
            EsmCodesSystem.from_json(json.dumps(_synthetic_dict(colour="blue")))

    def test_unknown_nested_fields(self):
        for key in ("normalisation", "calibrated_domain"):
            d = _synthetic_dict()
            d[key] = dict(d[key], extra=1)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, rf"{key}: unknown field\(s\) \['extra'\]"):
                EsmCodesSystem.from_dict(d)

    def test_missing_fields(self):
        d = _synthetic_dict()
        del d["high_m_rule"]
        with self.assertRaisesRegex(ValueError, r"missing field\(s\) \['high_m_rule'\]"):
            EsmCodesSystem.from_dict(d)
        d = _synthetic_dict()
        del d["calibrated_domain"]["target_min_distinct_codes"]
        with self.assertRaisesRegex(ValueError, "missing field"):
            EsmCodesSystem.from_dict(d)

    def test_not_an_object(self):
        with self.assertRaises(ValueError):
            EsmCodesSystem.from_json("[1, 2]")


class InconsistentValuesTest(unittest.TestCase):

    CASES = {
        "five centres": dict(bin_centres=[30, 50, 80, 150, 280]),
        "seven lambdas": dict(lam=[0.1] * 7),
        "centres not increasing": dict(bin_centres=[30, 50, 50, 150, 280, 600]),
        "centres decreasing": dict(bin_centres=[600, 280, 150, 80, 50, 30]),
        "centre not positive": dict(bin_centres=[0, 50, 80, 150, 280, 600]),
        "lambda zero": dict(lam=[0.19, 0.12, 0.0, 0.08, 0.05, 0.04]),
        "lambda negative": dict(lam=[0.19, 0.12, -0.1, 0.08, 0.05, 0.04]),
        "lnK NaN": dict(lnK=[-13.4, float("nan"), -13.7, -13.5, -14.5, -15.9]),
        "lambda infinite": dict(lam=[0.19, float("inf"), 0.1, 0.08, 0.05, 0.04]),
        "gap_extend above gap_open": dict(gap_open=0.5, gap_extend=5.0),
        "negative gap_extend": dict(gap_extend=-0.5),
        "quantile 0": dict(panel_quantile=0.0),
        "quantile 1": dict(panel_quantile=1.0),
        "quantile 1.2": dict(panel_quantile=1.2),
        "revision None": dict(bundle_revision=None),
        "revision main": dict(bundle_revision="main"),
        "revision short": dict(bundle_revision="a" * 12),
        "revision upper": dict(bundle_revision="A" * 40),
        "sha short": dict(transform_sha256="2" * 63),
        "sha upper": dict(panel_sha256="F" * 64),
        "sha None": dict(esm_weights_sha256=None),
        "code bits vs bytes": dict(code_bits=63),
        "code bytes bool": dict(code_bytes=True),
        "panel size float": dict(panel_size=3.0),
        "panel residues below size": dict(panel_residues=2),
        "offset string": dict(offset="1.0"),
        "offset bool": dict(offset=True),
        "empty dp": dict(dp=""),
        "bad name": dict(name="ESM codes"),
        "bad repo": dict(bundle_repo="no-slash"),
        "no bundle.json": dict(bundle_files=[["encoders.pt", "7" * 64]]),
        "file twice": dict(bundle_files=[["bundle.json", "6" * 64], ["bundle.json", "7" * 64]]),
        "file outside the repo": dict(bundle_files=[["bundle.json", "6" * 64], ["../x", "7" * 64]]),
        "absolute file": dict(bundle_files=[["bundle.json", "6" * 64], ["/x", "7" * 64]]),
        "bad file sha": dict(bundle_files=[["bundle.json", "6" * 60]]),
        "query lengths reversed": dict(calibrated_domain={"query_min_length": 967, "query_max_length": 966,
                                                          "target_max_length": 3000,
                                                          "max_nonstandard_fraction": 0.01,
                                                          "target_min_distinct_codes": 2}),
        "nonstandard fraction above 1": dict(calibrated_domain={"query_min_length": 21, "query_max_length": 966,
                                                                "target_max_length": 3000,
                                                                "max_nonstandard_fraction": 1.5,
                                                                "target_min_distinct_codes": 2}),
        "eps zero": dict(normalisation={"name": "rowcol_z", "std": "population", "eps": 0.0,
                                        "degenerate_std_lt": 1e-06}),
        "query_prep_eps negative": dict(query_prep_eps=-1e-12),
    }

    def test_each_case_raises(self):
        for label, overrides in self.CASES.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                _synthetic(**overrides)

    def test_replace_validates_too(self):
        with self.assertRaises(ValueError):
            dataclasses.replace(_synthetic(), gap_extend=6.0)


class RegistryTest(unittest.TestCase):

    def test_get_unknown_lists_the_known_names(self):
        a, b = _synthetic(name="esm-codes-a"), _synthetic(name="esm-codes-b")
        with mock.patch.object(ss, "REGISTRY", ss._registry(b, a)):
            self.assertIs(ss.get("esm-codes-a"), a)
            self.assertEqual(ss.known_names(), ("esm-codes-a", "esm-codes-b"))
            with self.assertRaisesRegex(UnknownScoringSystemError, "esm-codes-a, esm-codes-b") as ctx:
                ss.get("esm-codes-v9")
            self.assertIn("'esm-codes-v9'", str(ctx.exception))
            self.assertIn("newer FoldMatch", str(ctx.exception))
            with self.assertRaises(UnknownScoringSystemError):
                ss.get(None)
        self.assertTrue(issubclass(UnknownScoringSystemError, ValueError))

    def test_registry_is_read_only(self):
        self.assertIsInstance(ss.REGISTRY, MappingProxyType)
        with self.assertRaises(TypeError):
            ss.REGISTRY["x"] = _synthetic()
        with self.assertRaises(AssertionError):
            ss._registry(_synthetic(), _synthetic())

    def test_default_name(self):
        self.assertEqual(ss.DEFAULT_ESM_CODES, "esm-codes-v1")


class FieldGroupsTest(unittest.TestCase):

    def test_groups_partition_the_fields(self):
        groups = IDENTITY_FIELDS + ENCODING_FIELDS + SCORING_FIELDS + STATISTICS_FIELDS + BUNDLE_FIELDS
        self.assertEqual(sorted(groups), sorted(f.name for f in dataclasses.fields(EsmCodesSystem)))
        self.assertEqual(len(set(groups)), len(groups))

    def test_encoding_fields_are_the_plans(self):
        """The plan's encoding fields, plus the encoder architecture and precision (P4 review): the content digests
        cover only the weight tensors, and the same weights give other codes with GELU-tanh or another precision."""
        self.assertEqual(set(ENCODING_FIELDS), {
            "esm_repo", "esm_weights_sha256", "embedding_tap", "input_normalisation", "input_cast",
            "transform_sha256", "transform_output_cast", "target_encoder_sha256", "code_bits", "code_bytes",
            "bitorder", "sign_rule"} | {"encoder_architecture", "encoder_dtype"})
        self.assertEqual(BUNDLE_FIELDS, ("bundle_repo", "bundle_revision", "bundle_files"))

    def test_every_field_is_required(self):
        for f in dataclasses.fields(EsmCodesSystem):
            self.assertIs(f.default, dataclasses.MISSING, f.name)
            self.assertIs(f.default_factory, dataclasses.MISSING, f.name)


class ScoringChoiceTest(unittest.TestCase):

    def test_values_and_default(self):
        self.assertEqual([c.value for c in ScoringChoice], ["blosum62", "esm-codes"])
        self.assertIs(DEFAULT_SCORING, ScoringChoice.blosum62)
        self.assertIs(ScoringChoice("esm-codes"), ScoringChoice.esm_codes)



# =====================================================================================================================
# a synthetic bundle in a Hugging Face cache layout
# =====================================================================================================================
def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _npz_bytes(arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


class _SyntheticBundle:
    """A bundle laid out like rcsb/foldmatch-esm-codes (bundle.json, encoders, scoring npz) for ``self.system``,
    written as ``<cache>/models--owner--bundle/{blobs/<sha256>, snapshots/<rev>/<file> -> blob}``.

    ``bundle_json``, ``meta`` and ``arrays`` may be edited by the caller before :meth:`write`, which pins the
    resulting files' sha256s in the system (so a content check, not the sha256 check, must catch an edit)."""

    ENCODERS = "esm_codes_test_encoders.pt"
    SCORING = "esm_codes_test_scoring.npz"

    def __init__(self, **overrides):
        d = _synthetic_dict(**overrides)
        rng = np.random.default_rng(3)
        lengths = np.array([10, 13, 17], np.int64)
        self.codes = rng.integers(0, 256, (int(lengths.sum()), 8), dtype=np.uint8)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        d["panel_size"], d["panel_residues"] = len(lengths), int(lengths.sum())
        d["panel_sha256"] = tensor_digest({"codes": self.codes, "offsets": self.offsets})
        self.d = d
        table = {k: np.array(d[k], np.float64) for k in ("bin_centres", "lam", "lnK")}
        scoring = {k: d[k] for k in ("dp", "offset", "gap_open", "gap_extend", "score_form", "score_diff",
                                     "query_prep", "query_prep_eps", "similarity", "determinism")}
        scoring["normalisation"] = dict(d["normalisation"], formula="z = ...")
        esm3 = {"model": "esm3_sm_open_v1", "repo": d["esm_repo"], "weights_sha256": d["esm_weights_sha256"],
                "embedding_tap": d["embedding_tap"], "input_normalisation": d["input_normalisation"]}
        statistics = {k: d[k] for k in ("bin_centres", "lam", "lnK", "low_m_rule", "high_m_rule")}
        statistics["fit_db_residues"] = 8557236.0
        panel = {"size": d["panel_size"], "residues": d["panel_residues"], "quantile": d["panel_quantile"],
                 "quantile_method": d["quantile_method"], "panel_sha256": d["panel_sha256"]}
        self.meta = {
            "format": ss.SCORING_FORMAT, "system": d["name"], "calibration_version": d["calibration_version"],
            "code": {"bits": d["code_bits"], "bytes": d["code_bytes"], "bitorder": d["bitorder"],
                     "sign_rule": d["sign_rule"]},
            "scoring": copy.deepcopy(scoring), "statistics": copy.deepcopy(statistics), "panel": dict(panel),
            "calibrated_domain": dict(d["calibrated_domain"]), "degenerate_query_rule": d["degenerate_query_rule"],
            "esm3": dict(esm3),
        }
        self.arrays = {"panel_codes": self.codes, "panel_offsets": self.offsets, "panel_lengths": lengths,
                       "panel_ids": np.array(["UniRef50_A", "UniRef50_B", "UniRef50_C"]), **table}
        self.encoders = b"synthetic encoders (prepare() does not read them)"
        self.bundle_json = {
            "bundle_format": ss.BUNDLE_FORMAT, "system": d["name"], "calibration_version": d["calibration_version"],
            "hf_repo": d["bundle_repo"],
            "files": {self.ENCODERS: {"sha256": None, "format": ss.ENCODERS_FORMAT},
                      self.SCORING: {"sha256": None, "format": ss.SCORING_FORMAT}},
            "content_digests": {"method": "foldmatch-tensor-digest/1", "transform_sha256": d["transform_sha256"],
                                "query_encoder_sha256": d["query_encoder_sha256"],
                                "target_encoder_sha256": d["target_encoder_sha256"],
                                "panel_sha256": d["panel_sha256"], "table_sha256": tensor_digest(table)},
            "esm3": esm3,
            "encoding": {**{k: d[k] for k in ("input_cast", "transform_output_cast", "code_bits", "code_bytes",
                                              "bitorder", "sign_rule", "query_dtype")},
                         "architecture": {"name": d["encoder_architecture"], "activation": "gelu_exact"},
                         "encoder": {"dtype": d["encoder_dtype"], "tf32": False, "autocast": False}},
            "scoring": scoring, "statistics": statistics, "panel": panel,
            "calibrated_domain": dict(d["calibrated_domain"]), "degenerate_query_rule": d["degenerate_query_rule"],
            "licence": {"spdx": "MIT"},
        }

    def write(self, cache: Path) -> EsmCodesSystem:
        npz = _npz_bytes(dict(self.arrays, meta_json=np.array(json.dumps(self.meta))))
        files = {self.ENCODERS: self.encoders, self.SCORING: npz}
        bj = copy.deepcopy(self.bundle_json)
        for name in (self.ENCODERS, self.SCORING):
            if name in bj["files"] and bj["files"][name]["sha256"] is None:
                bj["files"][name]["sha256"] = _sha(files[name])
        files[ss.BUNDLE_JSON] = json.dumps(bj, indent=1).encode()
        d = dict(self.d, bundle_files=[[n, _sha(b)] for n, b in files.items()])
        system = EsmCodesSystem.from_dict(d)
        repo = cache / ("models--" + system.bundle_repo.replace("/", "--"))
        for name, data in files.items():
            blob = repo / "blobs" / _sha(data)
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
            link = repo / "snapshots" / system.bundle_revision / name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(blob, link.parent))
        self.files = files
        return system


def _listing(root: Path):
    return sorted((str(p.relative_to(root)), p.is_symlink() or p.stat().st_mtime_ns) for p in root.rglob("*"))


def _subprocess_env(**extra):
    env = dict(os.environ)
    src = str(Path(foldmatch.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra)
    return env


def _registered(*systems):
    """The registry holding exactly ``systems`` (the bundle loader serves registry entries only)."""
    return mock.patch.object(ss, "REGISTRY", ss._registry(*systems))


class _CacheCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "hub"
        self.cache.mkdir()

    def register(self, *systems):
        """Make ``systems`` the registry for the rest of the test; returns the first."""
        patch = _registered(*systems)
        patch.start()
        self.addCleanup(patch.stop)
        return systems[0]

    def tearDown(self):
        for p in [self.cache, *self.cache.rglob("*")]:        # undo read-only modes so the cleanup can delete
            if p.is_dir() and not p.is_symlink():
                p.chmod(stat.S_IRWXU)
        self._tmp.cleanup()


# =====================================================================================================================
# get_esm_codes_bundle
# =====================================================================================================================
class _FakeHub:
    """Stands in for hf_hub_download: serves ``files[(repo, revision, filename)]`` from a temp dir (``forced[key]``
    instead on a ``force_download``), records the calls and their ``local_files_only`` / ``force_download``."""

    def __init__(self, root: Path, files, forced=None):
        self.root, self.files, self.forced, self.calls, self.flags = root, files, dict(forced or {}), [], []

    def __call__(self, *, repo_id, filename, revision, cache_dir=None, local_files_only=False, force_download=False):
        self.calls.append((repo_id, revision, filename, cache_dir))
        self.flags.append((filename, local_files_only, force_download))
        key = (repo_id, revision, filename)
        if key not in self.files:
            raise LocalEntryNotFoundError(f"{key} not cached")
        path = self.root / repo_id.replace("/", "--") / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.forced.get(key, self.files[key]) if force_download else self.files[key])
        return str(path)


def _hub_system(name, revision, contents):
    return _synthetic(name=name, bundle_revision=revision,
                      bundle_files=[[f, _sha(b)] for f, b in contents.items()])


class BundleLoaderTest(_CacheCase):

    CONTENTS = {"bundle.json": b"{}", "encoders.pt": b"weights", "scoring.npz": b"panel"}

    def _hub(self, *systems):
        files = {(s.bundle_repo, s.bundle_revision, f): self.CONTENTS[f] for s in systems for f, _ in s.bundle_files}
        return _FakeHub(self.cache, files)

    def test_returns_verified_paths(self):
        s = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        hub = self._hub(s)
        with mock.patch.object(model, "hf_hub_download", hub):
            files = model.get_esm_codes_bundle(s)
        self.assertEqual(sorted(files), sorted(self.CONTENTS))
        for name, path in files.items():
            self.assertIsInstance(path, Path)
            self.assertEqual(path.read_bytes(), self.CONTENTS[name])
        self.assertEqual({c[:2] for c in hub.calls}, {(s.bundle_repo, s.bundle_revision)})
        self.assertEqual({f[1:] for f in hub.flags}, {(False, False)})

    def test_every_call_hashes_every_file_and_writes_nothing(self):
        system = self.register(_SyntheticBundle().write(self.cache))
        before = _listing(self.cache)
        spy = mock.Mock(wraps=model._file_sha256)
        with mock.patch.object(model, "_file_sha256", spy):
            first = model.get_esm_codes_bundle(system, cache_dir=self.cache)
            second = model.get_esm_codes_bundle(system, cache_dir=self.cache)
        self.assertEqual(first, second)
        self.assertEqual(spy.call_count, 2 * len(system.bundle_files))
        self.assertEqual(_listing(self.cache), before)

    def test_sha256_mismatch_is_refused(self):
        s = self.register(_hub_system("esm-codes-a", "a" * 40, dict(self.CONTENTS, **{"scoring.npz": b"other panel"})))
        with mock.patch.object(model, "hf_hub_download", self._hub(s)):
            with self.assertRaisesRegex(BundleMismatchError, r"scoring\.npz.*expected " + _sha(b"other panel")):
                model.get_esm_codes_bundle(s)

    def test_corrupt_cache_file_is_refused(self):
        """The message names the blob to delete: deleting the snapshot link alone re-links the same corrupt blob."""
        bundle = _SyntheticBundle()
        system = self.register(bundle.write(self.cache))
        blob = self.cache / "models--owner--bundle" / "blobs" / system.bundle_sha256(bundle.SCORING)
        blob.write_bytes(b"truncated")
        with self.assertRaisesRegex(BundleMismatchError, "corrupt") as ctx:
            model.get_esm_codes_bundle(system, cache_dir=self.cache)
        self.assertIn(f"delete {blob.resolve()} (the cache blob ", str(ctx.exception))
        self.assertIn("download-models", str(ctx.exception))

    def test_download_models_redownloads_a_corrupt_file(self):
        s = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        key = (s.bundle_repo, s.bundle_revision, "scoring.npz")
        files = {(s.bundle_repo, s.bundle_revision, f): b for f, b in self.CONTENTS.items()}
        files[key] = b"truncated"
        hub = _FakeHub(self.cache, files, forced={key: self.CONTENTS["scoring.npz"]})
        with mock.patch.object(model, "hf_hub_download", hub):
            with self.assertRaisesRegex(BundleMismatchError, "corrupt"):          # an ordinary call never forces
                model.get_esm_codes_bundle(s)
            self.assertFalse(any(f[2] for f in hub.flags))
            hub.flags.clear()
            got = model.download_esm_codes_bundles(["esm-codes-a"])["esm-codes-a"]
        self.assertEqual(got["scoring.npz"].read_bytes(), self.CONTENTS["scoring.npz"])
        self.assertEqual([f for f in hub.flags if f[2]], [("scoring.npz", False, True)])   # forced once, that file

    def test_redownload_that_still_differs_or_fails(self):
        s = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        key = (s.bundle_repo, s.bundle_revision, "scoring.npz")
        files = {(s.bundle_repo, s.bundle_revision, f): b for f, b in self.CONTENTS.items()}
        files[key] = b"truncated"
        with mock.patch.object(model, "hf_hub_download", _FakeHub(self.cache, files, forced={key: b"other"})):
            with self.assertRaisesRegex(BundleMismatchError, "after a fresh download"):
                model.get_esm_codes_bundle(s, redownload_corrupt=True)
        hub = _FakeHub(self.cache, files)

        def failing(**kw):
            if kw.get("force_download"):
                raise ValueError("Force download failed due to the above error.")
            return hub(**kw)
        with mock.patch.object(model, "hf_hub_download", failing):
            with self.assertRaisesRegex(BundleUnavailableError, "could not be downloaded again"):
                model.get_esm_codes_bundle(s, redownload_corrupt=True)
            with self.assertRaisesRegex(BundleMismatchError, "corrupt"):         # cached-only never forces
                model.get_esm_codes_bundle(s, redownload_corrupt=True, local_files_only=True)

    def test_unpinned_revisions_are_refused_before_any_download(self):
        s = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        hub = self._hub(s)
        for bad in (None, "main", "a" * 12, "A" * 40):
            loose = copy.copy(s)
            object.__setattr__(loose, "bundle_revision", bad)
            with self.subTest(revision=bad), mock.patch.object(model, "hf_hub_download", hub):
                with self.assertRaisesRegex(ValueError, "unpinned"):
                    model.get_esm_codes_bundle(loose)
        self.assertEqual(hub.calls, [])

    def test_only_registry_systems(self):
        for bad in (None, {"bundle_revision": "a" * 40}, SimpleNamespace(**_synthetic().__dict__)):
            with self.subTest(system=type(bad).__name__), self.assertRaises(TypeError):
                model.get_esm_codes_bundle(bad)

    def test_only_the_registered_entry(self):
        """A system parsed from elsewhere (e.g. a database manifest) is fetched only if it equals the registry entry
        of its name; a foreign repo or revision under the registered name, or an unknown name, is refused before
        any Hub call (P4 review)."""
        entry = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        d = json.loads(entry.to_json())
        d.update(bundle_repo="someone/else", bundle_revision="d" * 40)
        cases = {"foreign repo and revision": (EsmCodesSystem.from_json(json.dumps(d)),
                                               r"in \['bundle_repo', 'bundle_revision'\]"),
                 "other name": (dataclasses.replace(entry, name="esm-codes-z"), "knows: esm-codes-a"),
                 "one ulp": (dataclasses.replace(entry, lam=[_up(entry.lam[0])] + list(entry.lam[1:])),
                             r"in \['lam'\]")}
        hub = self._hub(entry)
        for label, (system, pattern) in cases.items():
            for fetch in (model.get_esm_codes_bundle, ss.prepare):
                with self.subTest(case=label, call=fetch.__name__), \
                        mock.patch.object(model, "hf_hub_download", hub):
                    with self.assertRaisesRegex(UnknownScoringSystemError, pattern):
                        fetch(system)
        self.assertEqual(hub.calls, [])
        with mock.patch.object(model, "hf_hub_download", hub):
            model.get_esm_codes_bundle(EsmCodesSystem.from_json(entry.to_json()))       # an equal copy is the entry
        self.assertEqual({c[:2] for c in hub.calls}, {(entry.bundle_repo, entry.bundle_revision)})
        self.assertIs(ss.registered(entry), entry)

    def test_not_cached_and_offline(self):
        s = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        with mock.patch.object(model, "hf_hub_download", side_effect=LocalEntryNotFoundError("offline")):
            with self.assertRaisesRegex(BundleUnavailableError, "download-models") as ctx:
                model.get_esm_codes_bundle(s)
        self.assertIn(f"{s.bundle_repo}@{s.bundle_revision}", str(ctx.exception))

    def test_cached_only(self):
        """local_files_only (the every-rank pre-flight after one fetch): passed to every call, never downloads."""
        s = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        hub = self._hub(s)
        with mock.patch.object(model, "hf_hub_download", hub):
            model.get_esm_codes_bundle(s, local_files_only=True)
        self.assertEqual(hub.flags, [(f, True, False) for f, _ in s.bundle_files])
        bundle = _SyntheticBundle()
        system = self.register(bundle.write(self.cache))
        before = _listing(self.cache)
        ss.prepare(system, cache_dir=self.cache, local_files_only=True)                 # the real hf_hub_download
        missing = self.register(dataclasses.replace(system, bundle_revision="c" * 40))
        with self.assertRaisesRegex(BundleUnavailableError, r"cached-only use was requested \(local_files_only"):
            ss.prepare(missing, cache_dir=self.cache, local_files_only=True)
        self.assertEqual(_listing(self.cache), before)

    def test_read_only_cache_without_the_bundle(self):
        """A reachable Hub but a cache that cannot be written: a bundle refusal naming download-models, not a raw
        PermissionError (P4 review). Other OS / HTTP errors are refusals too."""
        s = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        from huggingface_hub.errors import HfHubHTTPError
        errors = {"read-only": (PermissionError(13, "Permission denied", "/ro/hub/models--owner--bundle"),
                                "the cache is not writable"),
                  "disk full": (OSError(28, "No space left on device"), "could not be fetched"),
                  "server error": (HfHubHTTPError("500 Server Error"), "HfHubHTTPError")}
        for label, (error, pattern) in errors.items():
            with self.subTest(case=label), mock.patch.object(model, "hf_hub_download", side_effect=error):
                with self.assertRaisesRegex(BundleUnavailableError, pattern):
                    model.get_esm_codes_bundle(s)

    def test_missing_at_the_pinned_revision(self):
        s = self.register(_hub_system("esm-codes-a", "a" * 40, self.CONTENTS))
        with mock.patch.object(model, "hf_hub_download", side_effect=EntryNotFoundError("404")):
            with self.assertRaisesRegex(BundleMismatchError, "does not match the Hub"):
                model.get_esm_codes_bundle(s)

    def test_the_bundle_comes_only_from_the_registry(self):
        """A second registered system with another revision downloads that revision, and only that one."""
        a = _hub_system("esm-codes-a", "a" * 40, self.CONTENTS)
        b = _hub_system("esm-codes-b", "b" * 40, self.CONTENTS)
        hub = self._hub(a, b)
        with _registered(a, b), mock.patch.object(model, "hf_hub_download", hub):
            model.get_esm_codes_bundle(ss.get("esm-codes-b"))
            self.assertEqual({c[1] for c in hub.calls}, {"b" * 40})
            hub.calls.clear()
            model.download_esm_codes_bundles(["esm-codes-b"], cache_dir="/some/cache")
            self.assertEqual({(c[1], c[3]) for c in hub.calls}, {("b" * 40, "/some/cache")})
            hub.calls.clear()
            got = model.download_esm_codes_bundles()
            self.assertEqual(list(got), ["esm-codes-a", "esm-codes-b"])
            self.assertEqual(sorted({c[1] for c in hub.calls}), ["a" * 40, "b" * 40])
            self.assertEqual(len(hub.calls), 2 * len(self.CONTENTS))
            hub.calls.clear()
            with self.assertRaisesRegex(UnknownScoringSystemError, "esm-codes-a, esm-codes-b"):
                model.download_esm_codes_bundles(["esm-codes-b", "esm-codes-x"])
            self.assertEqual(hub.calls, [])


# =====================================================================================================================
# prepare()
# =====================================================================================================================
class PrepareTest(_CacheCase):

    def test_prepare_and_worker_view(self):
        bundle = _SyntheticBundle()
        system = self.register(bundle.write(self.cache))
        prepared = ss.prepare(system, cache_dir=self.cache)
        self.assertIs(prepared.system, system)
        self.assertEqual((prepared.encoders_file, prepared.scoring_file), (bundle.ENCODERS, bundle.SCORING))
        self.assertEqual(prepared.bundle["system"], system.name)
        spec = prepared.for_worker()
        clone = pickle.loads(pickle.dumps(spec))
        self.assertEqual(clone, spec)
        for f in dataclasses.fields(spec):
            value = getattr(spec, f.name)
            leaves = value if isinstance(value, tuple) else (
                dataclasses.astuple(value) if dataclasses.is_dataclass(value) else (value,))
            self.assertTrue(all(isinstance(v, (str, int, float)) for v in leaves), f.name)
        self.assertEqual(spec.system_digest, system.digest())
        self.assertEqual(spec.dp_params(), swkern.SETTING_D)
        codes, offsets = spec.load_panel()
        np.testing.assert_array_equal(codes, bundle.codes)
        np.testing.assert_array_equal(offsets, bundle.offsets)
        self.assertEqual(cs.lam_lnk(100, spec), cs.lam_lnk(100, system))     # a WorkerSpec is a stats table

    def test_worker_rechecks_the_npz(self):
        bundle = _SyntheticBundle()
        system = self.register(bundle.write(self.cache))
        spec = ss.prepare(system, cache_dir=self.cache).for_worker()
        Path(spec.scoring_npz).resolve().write_bytes(b"replaced")
        with self.assertRaisesRegex(BundleMismatchError, "sha256"):
            spec.load_panel()

    def _refused(self, bundle, pattern, system=None):
        written = bundle.write(self.cache)
        candidate = self.register(system(written) if system else written)
        with self.assertRaisesRegex(BundleMismatchError, pattern) as ctx:
            ss.prepare(candidate, cache_dir=self.cache)
        return str(ctx.exception)

    def test_table_off_by_one_ulp_is_refused(self):
        for key, i in (("lam", 0), ("lnK", 5), ("bin_centres", 2)):
            with self.subTest(field=key):
                def ulp(s, key=key, i=i):
                    v = list(getattr(s, key))
                    v[i] = math.nextafter(v[i], math.inf)
                    return dataclasses.replace(s, **{key: v})
                self._refused(_SyntheticBundle(), rf"bundle\.json statistics\.{key} = .*the system has {key}", ulp)

    def test_document_values_are_compared_exactly(self):
        cases = {
            "bundle.json gap_open one ulp": (lambda b: b.bundle_json["scoring"].update(gap_open=_up(5.0)),
                                             r"bundle\.json scoring\.gap_open"),
            "meta_json lam one ulp": (lambda b: b.meta["statistics"]["lam"].__setitem__(1, _up(0.12)),
                                      r"meta_json statistics\.lam"),
            "npz lnK array one ulp": (lambda b: b.arrays.__setitem__("lnK", np.nextafter(b.arrays["lnK"], 0)),
                                      r"scoring npz lnK = \[-13\.39999.* differs from the system's \[-13\.4, "
                                      r"-14\.0, .*\] at index 0 \(exact float64"),
            "encoder architecture": (lambda b: b.bundle_json["encoding"]["architecture"].update(name="other-mlp"),
                                     r"bundle\.json encoding\.architecture\.name = 'other-mlp'.*encoder_architecture"),
            "encoder dtype": (lambda b: b.bundle_json["encoding"]["encoder"].update(dtype="bfloat16"),
                              r"bundle\.json encoding\.encoder\.dtype = 'bfloat16'"),
            "no architecture": (lambda b: b.bundle_json["encoding"].pop("architecture"),
                                r"bundle\.json has no encoding\.architecture\.name"),
            "a number as text": (lambda b: b.bundle_json["scoring"].update(offset="1.0"),
                                 r"scoring\.offset = '1\.0'"),
            "missing section": (lambda b: b.bundle_json.pop("calibrated_domain"),
                                r"bundle\.json has no calibrated_domain\.query_min_length"),
            "calibrated domain": (lambda b: b.meta["calibrated_domain"].update(target_max_length=1000),
                                  r"meta_json calibrated_domain\.target_max_length = 1000"),
            "high-m rule": (lambda b: b.bundle_json["statistics"].update(high_m_rule="extrapolate"),
                            r"statistics\.high_m_rule"),
            "esm weights": (lambda b: b.meta["esm3"].update(weights_sha256="0" * 64), r"meta_json esm3\.weights"),
            "hf repo": (lambda b: b.bundle_json.update(hf_repo="other/repo"), r"hf_repo"),
            "table digest": (lambda b: b.bundle_json["content_digests"].update(table_sha256="0" * 64),
                             r"table_sha256"),
            "digest method": (lambda b: b.bundle_json["content_digests"].update(method="other/1"),
                              r"content_digests\.method"),
            "meta format": (lambda b: b.meta.update(format="other/1"), r"scoring npz format"),
            "panel byte": (lambda b: b.arrays.__setitem__("panel_codes", b.codes ^ np.eye(40, 8, dtype=np.uint8)),
                           r"panel content digest"),
            "panel lengths": (lambda b: b.arrays.__setitem__("panel_lengths", b.arrays["panel_lengths"][::-1]),
                              r"panel_lengths"),
            "unpinned file": (lambda b: b.bundle_json["files"].update({"extra.bin": {"sha256": "0" * 64}}),
                              r"lists extra\.bin, which the system does not pin"),
            "file sha in bundle.json": (lambda b: b.bundle_json["files"][b.ENCODERS].update(sha256="0" * 64),
                                        r"gives esm_codes_test_encoders\.pt sha256 0+"),
            "no scoring file": (lambda b: b.bundle_json["files"][b.SCORING].update(format="other/1"),
                                r"lists no foldmatch-esm-codes-scoring/1 file"),
        }
        for label, (edit, pattern) in cases.items():
            with self.subTest(case=label):
                b = _SyntheticBundle()
                edit(b)
                self.assertNotIn("np.float64(", self._refused(b, pattern))

    def test_integral_json_numbers_equal_floats(self):
        b = _SyntheticBundle()
        b.bundle_json["scoring"].update(gap_open=5, offset=1)
        system = self.register(b.write(self.cache))
        ss.prepare(system, cache_dir=self.cache)

    def test_unknown_bundle_format(self):
        b = _SyntheticBundle()
        b.bundle_json["bundle_format"] = "foldmatch-esm-codes-bundle/2"
        system = self.register(b.write(self.cache))
        with self.assertRaisesRegex(BundleMismatchError, "bundle.json format"):
            ss.prepare(system, cache_dir=self.cache)

    def test_unsupported_systems_are_refused_before_any_download(self):
        cases = {"dp": dict(dp="fmsw/2"), "setting": dict(gap_open=6.0), "high-m rule": dict(high_m_rule="extra"),
                 "similarity": dict(similarity="pm1_fp32_fixed_order"), "eps": dict(
                     normalisation={"name": "rowcol_z", "std": "population", "eps": 1e-7, "degenerate_std_lt": 1e-6}),
                 "code layout": dict(code_bits=128, code_bytes=16), "degenerate rule": dict(degenerate_query_rule="x"),
                 "architecture": dict(encoder_architecture="esm-codes-mlp-v2"),
                 "encoder dtype": dict(encoder_dtype="bfloat16")}
        for label, overrides in cases.items():
            with self.subTest(case=label), mock.patch.object(model, "hf_hub_download") as hub:
                with self.assertRaises(UnsupportedSystemError):
                    ss.prepare(_synthetic(**overrides))
                hub.assert_not_called()
        ss.check_implemented(_synthetic())

    def test_dp_values_come_from_the_kernel(self):
        s = _synthetic()
        self.assertEqual((s.offset, s.gap_open, s.gap_extend), (swkern.SETTING_D.offset, swkern.SETTING_D.gap_open,
                                                                swkern.SETTING_D.gap_extend))
        with mock.patch.object(swkern, "DP_ID", "fmsw/9"), self.assertRaisesRegex(UnsupportedSystemError, "fmsw/9"):
            ss.check_implemented(s)
        with mock.patch.object(swkern, "SETTING_D", swkern.DPParams(1.0, 5.0, 1.0)), \
                self.assertRaisesRegex(UnsupportedSystemError, "setting D"):
            ss.check_implemented(s)

    def test_format_tags_match_the_encoder_module(self):
        from foldmatch.model import residue_code_encoder as rce
        self.assertEqual(ss.ENCODERS_FORMAT, rce.ENCODERS_FORMAT)
        self.assertEqual(rce.DIGEST_METHOD, "foldmatch-tensor-digest/1")
        self.assertIn(ss.IMPLEMENTED["encoder_architecture"], rce.ARCHITECTURES)
        self.assertEqual(ss.IMPLEMENTED["encoder_dtype"], rce.IMPLEMENTED_CONVENTIONS[("encoder", "dtype")])


class _OfflineBase(_CacheCase):
    """The bundle loader and prepare() in a subprocess with HF_HUB_OFFLINE=1, an unreachable endpoint (any network
    use would fail) and, optionally, a read-only cache; without torch (Stage 2 never imports it)."""

    CODE = textwrap.dedent("""
        import json, sys
        sys.modules["torch"] = None
        from pathlib import Path
        from foldmatch.search import code_statistics as cs, scoring_system as ss
        from foldmatch.utils import model
        system = ss.EsmCodesSystem.from_json(Path(sys.argv[1]).read_text())
        if sys.argv[2] == "register":                   # a synthetic system; the published one is registered
            ss.REGISTRY = ss._registry(system)
        files = model.get_esm_codes_bundle(system)
        spec = ss.prepare(system).for_worker()
        codes, offsets = spec.load_panel()
        print(json.dumps(dict(files=sorted(files), residues=int(offsets[-1]), lnE=cs.log_evalue(3.0, 100, 1e9, spec),
                              torch=sys.modules.get("torch") is not None)))
    """)

    REGISTER = True

    def _run(self, system, *, offline=True):
        spec = Path(self._tmp.name) / "system.json"
        spec.write_text(system.to_json())
        env = _subprocess_env(HF_HUB_CACHE=str(self.cache), HF_ENDPOINT="http://127.0.0.1:9",
                              HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_ETAG_TIMEOUT="2", HF_HUB_DOWNLOAD_TIMEOUT="2")
        env.pop("HF_HUB_OFFLINE", None)
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
        return subprocess.run([sys.executable, "-c", self.CODE, str(spec), "register" if self.REGISTER else "-"],
                              env=env, capture_output=True, text=True, timeout=300)

    def _check(self, system, **kw):
        before = _listing(self.cache)
        r = self._run(system, **kw)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(out["files"], sorted(n for n, _ in system.bundle_files))
        self.assertEqual(out["residues"], system.panel_residues)
        self.assertFalse(out["torch"])
        self.assertEqual(_listing(self.cache), before)
        return out

    def _require_read_only(self):
        """Skip when the permissions do not make the cache read-only (e.g. a CI container running as root)."""
        probe = self.cache / ".write-probe"
        try:
            probe.write_text("x")
        except OSError:
            return
        probe.unlink()
        self.skipTest("the cache stays writable after chmod (running as root?): read-only use cannot be tested here")

    def _read_only(self):
        for p in [self.cache, *self.cache.rglob("*")]:
            if p.is_dir() and not p.is_symlink():
                p.chmod(stat.S_IRUSR | stat.S_IXUSR)
            elif p.is_file() and not p.is_symlink():
                p.chmod(stat.S_IRUSR)


class OfflineReadOnlyTest(_OfflineBase):

    def test_offline(self):
        self._check(_SyntheticBundle().write(self.cache))

    def test_read_only_cache_offline_and_online(self):
        system = _SyntheticBundle().write(self.cache)
        self._read_only()
        self._require_read_only()
        self._check(system, offline=True)
        self._check(system, offline=False)      # a cached file at a commit revision: no network, no write

    def test_offline_and_not_cached(self):
        system = _SyntheticBundle().write(self.cache)
        missing = dataclasses.replace(system, bundle_revision="c" * 40)
        r = self._run(missing)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("BundleUnavailableError", r.stderr)
        self.assertIn("offline mode is on", r.stderr)


# =====================================================================================================================
# download-models
# =====================================================================================================================
class DownloadModelsTest(unittest.TestCase):

    def setUp(self):
        self.a, self.b = _synthetic(name="esm-codes-a"), _synthetic(name="esm-codes-b", bundle_revision="b" * 40)
        self.fetched = []

        def fake_bundle(system, cache_dir=None, **kw):
            self.fetched.append(system.name)
            return {n: Path("/cache/snap") / n for n, _ in system.bundle_files}
        from foldmatch.utils.esm import loaders
        patches = [mock.patch.object(ss, "REGISTRY", ss._registry(self.a, self.b)),
                   mock.patch.object(model, "get_esm_codes_bundle", side_effect=fake_bundle),
                   mock.patch.object(loaders, "download_esm3_snapshot", return_value=Path("/cache/esm3")),
                   mock.patch.object(model, "download_aggregator_weights", return_value=Path("/cache/agg/a.pt"))]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

    def _apps(self):
        from foldmatch.cli import embedding, inference, search
        return {"fm-search": search.app, "fm-embedding": embedding.app, "inference": inference.app}

    def test_every_registered_system_by_default(self):
        for label, app in self._apps().items():
            with self.subTest(cli=label):
                self.fetched.clear()
                r = CliRunner().invoke(app, ["download-models"])
                self.assertEqual(r.exit_code, 0, r.output)
                self.assertEqual(self.fetched, ["esm-codes-a", "esm-codes-b"])
                self.assertIn(f"esm-codes-b: owner/bundle@{'b' * 40}, 3 files, sha256 verified", r.output)

    def test_system_restricts(self):
        for label, app in self._apps().items():
            with self.subTest(cli=label):
                self.fetched.clear()
                r = CliRunner().invoke(app, ["download-models", "--system", "esm-codes-b"])
                self.assertEqual(r.exit_code, 0, r.output)
                self.assertEqual(self.fetched, ["esm-codes-b"])

    def test_unknown_system_is_refused_before_any_download(self):
        esm3, aggregator = self.mocks[2], self.mocks[3]
        for label, app in self._apps().items():
            with self.subTest(cli=label):
                r = CliRunner().invoke(app, ["download-models", "--system", "esm-codes-a", "--system", "nope"])
                self.assertEqual(r.exit_code, 2, r.output)
                self.assertIn("unknown esm-codes scoring system 'nope'", r.output)
                self.assertIn("esm-codes-a, esm-codes-b", r.output)
                self.assertNotIn("database", r.output)
        self.assertEqual(self.fetched, [])
        esm3.assert_not_called()
        aggregator.assert_not_called()

    def test_help_names_the_option(self):
        for label, app in self._apps().items():
            with self.subTest(cli=label):
                r = CliRunner().invoke(app, ["download-models", "--help"])
                self.assertEqual(r.exit_code, 0)
                self.assertIn("--system", r.output)
                self.assertIn("esm-codes", r.output)


# =====================================================================================================================
# the frozen registry (P4.4) and the published bundle
# =====================================================================================================================
# The golden digests of the registry freeze live in tests/test_scoring_system_golden.py (standard library only, so the
# CI kernel jobs on Linux and macOS run them too).
_FIXTURES = Path(__file__).parent / "resources" / "esm_codes"


class RegistryEntryTest(unittest.TestCase):

    def test_default_is_registered(self):
        self.assertEqual(ss.known_names(), ("esm-codes-v1",))
        self.assertIs(ss.get(ss.DEFAULT_ESM_CODES), ss.REGISTRY["esm-codes-v1"])

    def test_every_entry_is_implemented(self):
        for s in ss.REGISTRY.values():
            ss.check_implemented(s)
            self.assertEqual(s.dp, swkern.DP_ID)
            self.assertEqual(swkern.DPParams(s.offset, s.gap_open, s.gap_extend), swkern.SETTING_D)

    def test_esm3_pin_is_the_loaders(self):
        from foldmatch.utils.esm import loaders
        for s in ss.REGISTRY.values():
            self.assertEqual((s.esm_repo, s.esm_weights_sha256), (loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_SHA256))

    def test_v1_matches_the_frozen_fixtures(self):
        """The v1 entry against the P0.8 fixtures (made from the research artefacts, independently of the bundle):
        the E-value table of ship_final.json bit for bit, the setting and the identity strings."""
        s = ss.get("esm-codes-v1")
        f4 = json.loads((_FIXTURES / "evalue_grid_v1.json").read_text())
        self.assertEqual(s.bin_centres, tuple(f4["table"]["center_len"]))
        self.assertEqual(s.lam, tuple(f4["table"]["lam"]))
        self.assertEqual(s.lnK, tuple(f4["table"]["lnK"]))
        fixtures = json.loads((_FIXTURES / "FIXTURES.json").read_text())
        self.assertEqual((s.name, s.calibration_version), (fixtures["system"], fixtures["calibration_version"]))
        setting = fixtures["setting"]
        self.assertEqual((s.offset, s.gap_open, s.gap_extend), (setting["offset"], setting["gap_open"],
                                                               setting["gap_extend"]))
        self.assertEqual((s.normalisation.degenerate_std_lt, s.query_prep_eps), (setting["s7_eps"],
                                                                                 setting["qprep_eps"]))
        self.assertEqual(s.esm_weights_sha256, fixtures["esm3"]["weights_blob_sha256"])
        self.assertEqual(f4["calibration_version"], s.calibration_version)

    def test_v1_encoder_is_the_encoder_modules(self):
        from foldmatch.model import residue_code_encoder as rce
        s = ss.get("esm-codes-v1")
        self.assertEqual((s.encoder_architecture, s.encoder_dtype), ("esm-codes-mlp-v1", "float32"))
        self.assertEqual(rce.ARCHITECTURES[s.encoder_architecture]["activation"], "gelu_exact")

    def test_v1_calibrated_domain_and_rules(self):
        """S13 / S13b / D14 / D15 as decided 2026-10-12."""
        s = ss.get("esm-codes-v1")
        self.assertEqual(s.calibrated_domain, CalibratedDomain(21, 966, 3000, 0.01, 2))
        self.assertEqual((s.low_m_rule, s.high_m_rule, s.degenerate_query_rule), ("clamp", "clamp", "no_hits"))
        self.assertEqual((s.similarity, s.determinism), ("pm1_fp32_blas_gemm_per_pair", "tolerance_s16_v1"))
        self.assertEqual((s.panel_size, s.panel_residues, s.panel_quantile, s.quantile_method),
                         (1000, 355626, 0.9, "linear"))

    def test_bundle_json_states_every_value_but_the_pins(self):
        stated = {entry[0][0] for entry in ss.DOCUMENT_FIELDS}
        self.assertEqual(stated, {f.name for f in dataclasses.fields(EsmCodesSystem)} - {"bundle_revision",
                                                                                         "bundle_files"})


class SingleSourceOfTruthTest(unittest.TestCase):
    """A bundle revision is written down exactly once: in its registry entry (plan principle 2, R5)."""

    SKIP_DIRS = {".git", "build", "dist", "__pycache__", ".venv", "venv", ".idea", ".pytest_cache"}
    SKIP_SUFFIXES = {".so", ".dylib", ".pyc", ".o"}

    def test_bundle_revision_exists_only_in_the_registry_entry(self):
        root = Path(__file__).resolve().parents[1]
        if not (root / "src" / "foldmatch").is_dir():
            self.skipTest("not a source checkout")
        revisions = {s.bundle_revision.encode() for s in ss.REGISTRY.values()}
        hits = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS and not d.endswith(".egg-info")
                           and (not d.startswith(".") or d == ".github")]
            for f in filenames:
                path = Path(dirpath) / f
                if path.suffix in self.SKIP_SUFFIXES or path.is_symlink():
                    continue
                data = path.read_bytes()
                for rev in revisions:
                    if rev in data:
                        hits[(path.relative_to(root).as_posix(), rev.decode())] = data.count(rev)
        self.assertEqual(hits, {("src/foldmatch/search/scoring_system.py", r.decode()): 1 for r in revisions})


class RealBundleTest(unittest.TestCase):
    """prepare() and the head on the published bundle at the registry's revision (downloaded once, sha256-verified)."""

    @classmethod
    def setUpClass(cls):
        import esm_codes_bundle
        cls.files = esm_codes_bundle.registry_files()
        cls.system = ss.get(ss.DEFAULT_ESM_CODES)
        cls.prepared = ss.prepare(cls.system)

    def test_files_are_the_pinned_ones(self):
        self.assertEqual(sorted(self.files), [n for n, _ in self.system.bundle_files])
        for name, sha in self.system.bundle_files:
            self.assertEqual(model._file_sha256(self.files[name]), sha, name)
            self.assertEqual(Path(self.files[name]).parent.name, self.system.bundle_revision)

    def test_prepare_accepts_the_published_bundle(self):
        p = self.prepared
        self.assertEqual((p.encoders_file, p.scoring_file), ("esm_codes_v1_encoders.pt", "esm_codes_v1_scoring.npz"))
        self.assertEqual(p.bundle["licence"]["spdx"], "MIT")
        spec = p.for_worker()
        self.assertEqual(pickle.loads(pickle.dumps(spec)), spec)
        codes, offsets = spec.load_panel()
        self.assertEqual((codes.shape, codes.dtype, offsets.shape), ((355626, 8), np.uint8, (1001,)))
        self.assertEqual(tensor_digest({"codes": codes, "offsets": offsets}), self.system.panel_sha256)

    def test_table_off_by_one_ulp_is_refused(self):
        """A registry entry one ulp off the bundle (as if mistyped and registered) is refused by prepare(); an
        unregistered variant never reaches the bundle (test_unregistered_variants_are_refused)."""
        for key, i, direction in (("lam", 3, math.inf), ("lnK", 0, -math.inf), ("bin_centres", 5, math.inf)):
            v = list(getattr(self.system, key))
            v[i] = math.nextafter(v[i], direction)
            variant = dataclasses.replace(self.system, **{key: v})
            with self.subTest(field=key), _registered(variant), \
                    self.assertRaisesRegex(BundleMismatchError, rf"statistics\.{key}"):
                ss.prepare(variant)

    def test_other_differences_are_refused(self):
        cases = {"panel_quantile": 0.95, "calibration_version": "foldmatch-evalue-other",
                 "calibrated_domain": CalibratedDomain(21, 966, 1000, 0.01, 2), "panel_residues": 355627,
                 "target_encoder_sha256": "0" * 64, "name": "esm-codes-v2"}
        for key, value in cases.items():
            variant = dataclasses.replace(self.system, **{key: value})
            with self.subTest(field=key), _registered(variant), self.assertRaises(BundleMismatchError):
                ss.prepare(variant)

    def test_unregistered_variants_are_refused(self):
        d = json.loads(self.system.to_json())
        d["bundle_revision"] = "d" * 40
        variants = {"foreign revision": EsmCodesSystem.from_json(json.dumps(d)),
                    "other quantile": dataclasses.replace(self.system, panel_quantile=0.95)}
        for label, variant in variants.items():
            with self.subTest(case=label), mock.patch.object(model, "hf_hub_download") as hub:
                with self.assertRaises(UnknownScoringSystemError):
                    ss.prepare(variant)
                hub.assert_not_called()
        self.assertEqual(ss.registered(EsmCodesSystem.from_json(self.system.to_json())), self.system)

    def test_load_head(self):
        head = self.prepared.load_head()
        self.assertEqual(head.content_digests(), {k: getattr(self.system, k) for k in (
            "transform_sha256", "query_encoder_sha256", "target_encoder_sha256")})
        self.assertEqual(head.meta["system"], self.system.name)
        other = dataclasses.replace(self.prepared, system=dataclasses.replace(self.system,
                                                                              query_encoder_sha256="0" * 64))
        with self.assertRaisesRegex(BundleMismatchError, "query_encoder_sha256"):
            other.load_head()
        foreign = dataclasses.replace(self.prepared, system=dataclasses.replace(self.system,
                                                                                esm_weights_sha256="0" * 64))
        with self.assertRaisesRegex(UnsupportedSystemError, "ESM3 weights"):
            foreign.load_head()
        for change, pattern in ((dict(encoder_architecture="esm-codes-mlp-v2"), "architecture is 'esm-codes-mlp-v1'"),
                                (dict(encoder_dtype="bfloat16"), "encoder dtype is 'float32'")):
            other = dataclasses.replace(self.prepared, system=dataclasses.replace(self.system, **change))
            with self.subTest(**change), self.assertRaisesRegex(BundleMismatchError, pattern):
                other.load_head()


class RealBundleOfflineTest(_OfflineBase):
    """The published bundle, copied from the cache it was downloaded into, used offline and read-only, without
    torch and with an unreachable endpoint."""

    REGISTER = False                # the published system is the real registry entry

    def setUp(self):
        super().setUp()
        import shutil
        import esm_codes_bundle
        self.system = ss.get(ss.DEFAULT_ESM_CODES)
        repo = Path(esm_codes_bundle.registry_files()[ss.BUNDLE_JSON]).parents[2]
        shutil.copytree(repo, self.cache / repo.name, symlinks=True)

    def test_offline(self):
        out = self._check(self.system)
        self.assertEqual(out["lnE"], cs.log_evalue(3.0, 100, 1e9, self.system))

    def test_read_only_cache_offline_and_online(self):
        self._read_only()
        self._require_read_only()
        self._check(self.system, offline=True)
        self._check(self.system, offline=False)


class BundleHelperSkipRuleTest(unittest.TestCase):
    """tests/esm_codes_bundle.py skips the real-bundle tests only when offline mode is on and the bundle is not
    cached; an unreachable Hub with offline mode off, a sha256 mismatch or any other failure is a test error, so CI
    never turns green without the acceptance checks (P4 review)."""

    def setUp(self):
        import esm_codes_bundle
        self.helper = esm_codes_bundle
        esm_codes_bundle._fetch.cache_clear()
        self.addCleanup(esm_codes_bundle._fetch.cache_clear)

    def _classify(self, *, offline, error=None):
        from huggingface_hub import constants as hf_constants
        self.helper._fetch.cache_clear()
        files = {"bundle.json": Path("/cache/snap/bundle.json")}
        with mock.patch.object(model, "get_esm_codes_bundle", side_effect=error, return_value=files), \
                mock.patch.object(hf_constants, "HF_HUB_OFFLINE", offline):
            try:
                got = self.helper.registry_files()
            except unittest.SkipTest:
                return "skip"
            except RuntimeError as e:
                self.assertIsInstance(e.__cause__, type(error))
                return "error"
        self.assertEqual(got, files)
        return "ok"

    def test_skip_only_offline_and_not_cached(self):
        unavailable = BundleUnavailableError("bundle.json is not in the Hugging Face cache and offline mode is on")
        unreachable = BundleUnavailableError("bundle.json is not in the cache and the Hugging Face Hub is unreachable")
        cases = {
            "offline, not cached": (dict(offline=True, error=unavailable), "skip"),
            "Hub unreachable, offline mode off": (dict(offline=False, error=unreachable), "error"),
            "sha256 mismatch offline": (dict(offline=True, error=BundleMismatchError("sha256 differs")), "error"),
            "sha256 mismatch online": (dict(offline=False, error=BundleMismatchError("sha256 differs")), "error"),
            "not the registry entry": (dict(offline=True, error=UnknownScoringSystemError("x")), "error"),
            "a bug": (dict(offline=True, error=TypeError("x")), "error"),
            "fetched": (dict(offline=False), "ok"),
            "fetched offline": (dict(offline=True), "ok"),
        }
        for label, (kw, want) in cases.items():
            with self.subTest(case=label):
                self.assertEqual(self._classify(**kw), want)

    def test_one_fetch_per_session(self):
        from huggingface_hub import constants as hf_constants
        with mock.patch.object(model, "get_esm_codes_bundle", side_effect=BundleUnavailableError("down")) as fetch, \
                mock.patch.object(hf_constants, "HF_HUB_OFFLINE", False):
            for _ in range(3):
                with self.assertRaises(RuntimeError):
                    self.helper.registry_files()
        self.assertEqual(fetch.call_count, 1)


class NoTorchImportTest(unittest.TestCase):

    def test_bundle_loader_and_statistics_import_without_torch(self):
        code = ("import sys; sys.modules['torch'] = None\n"
                "import foldmatch.search.scoring_system, foldmatch.search.code_statistics, foldmatch.utils.model\n"
                "import foldmatch.utils.tensor_digest\n"
                "print('ok')\n")
        r = subprocess.run([sys.executable, "-c", code], env=_subprocess_env(), capture_output=True, text=True,
                           timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(r.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
