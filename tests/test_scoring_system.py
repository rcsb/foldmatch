"""P4: the versioned esm-codes scoring system (plan section 8 P4; ``foldmatch.search.scoring_system``).

The structural tests run on a synthetic system (``_synthetic``) whose values only need to be well formed.
"""
import copy
import dataclasses
import json
import math
import unittest
from types import MappingProxyType
from unittest import mock

from foldmatch.search import scoring_system as ss
from foldmatch.search.scoring_system import (
    BUNDLE_FIELDS, ENCODING_FIELDS, IDENTITY_FIELDS, SCORING_FIELDS, STATISTICS_FIELDS, CalibratedDomain,
    EsmCodesSystem, Normalisation, UnknownScoringSystemError,
)
from foldmatch.types.api_types import DEFAULT_SCORING, ScoringChoice


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


if __name__ == "__main__":
    unittest.main()
