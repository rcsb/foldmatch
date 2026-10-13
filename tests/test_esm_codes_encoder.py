"""P2.2: ESM3 residue-code encoders (plan §6 P2.2, spec S2-S5).

Structural tests use tiny synthetic bundles (a 16-dimensional embedding, registered as a test-only architecture) and
always run. Tests that need the real esm-codes-v1 weights read the bundle from the directory named by
``FOLDMATCH_ESM_CODES_BUNDLE`` (the files of the HF repo rcsb/foldmatch-esm-codes) and are skipped when it is not set;
P4 replaces the variable with the pinned, sha256-verified download.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from foldmatch.model import residue_code_encoder as rce
from foldmatch.model.residue_code_encoder import (
    ARCHITECTURES, ENCODERS_FORMAT, IMPLEMENTED_CONVENTIONS, EsmCodesBundleError, ResidueCodeHead,
    head_from_bundle, load_esm_codes_encoders, packbits_rows, strict_fp32, tensor_digest,
)

_TEST_PATH = Path(__file__).parent
_FIXTURES = _TEST_PATH / "resources" / "esm_codes"
_EMBEDDINGS = _TEST_PATH / "resources" / "embeddings"
_SRC = Path(rce.__file__).resolve().parents[2]
BUNDLE_ENV = "FOLDMATCH_ESM_CODES_BUNDLE"
ENCODERS_FILE = "esm_codes_v1_encoders.pt"
SCORING_FILE = "esm_codes_v1_scoring.npz"

_TINY = {
    "name": "tiny-test",
    "d_embed": 16,
    "dims_dropped": [3, 11],
    "d_in": 14,
    "hidden": 8,
    "d_out": 64,
    "activation": "gelu_exact",
    "skip": "linear",
    "bias": True,
    "dropout": 0.0,
    "encoders": ["f_q", "f_t"],
}


def _conventions():
    out = {}
    for (section, key), value in IMPLEMENTED_CONVENTIONS.items():
        out.setdefault(section, {})[key] = value
    return out


def _tiny_bundle(seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    def state():
        return {"lin.weight": r(64, 14), "lin.bias": r(64), "fc1.weight": r(8, 14), "fc1.bias": r(8),
                "fc2.weight": r(64, 8), "fc2.bias": r(64)}

    dims = torch.tensor([i for i in range(16) if i not in (3, 11)], dtype=torch.int64)
    return {"format": ENCODERS_FORMAT, "system": "test", "architecture": dict(_TINY),
            "transform": {"dims": dims, "offset": r(14), "M": r(14, 14) / 4},
            "f_q": state(), "f_t": state(), "conventions": _conventions(), "provenance": {"note": "synthetic"}}


def _save(obj, path):
    torch.save(obj, path)
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row_ulps(a, b):
    """max |a - b| in fp16 ulps of each row's largest |value| (per-element ulps are misleading near 0)."""
    a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
    rowmax = np.maximum(np.abs(a32), np.abs(b32)).max(axis=1, keepdims=True)
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(rowmax, 2.0 ** -14))) - 10)
    return float((np.abs(a32 - b32) / ulp).max())


class _TinyArchitecture(unittest.TestCase):
    """Registers the synthetic architecture for the duration of each test."""

    def setUp(self):
        patcher = mock.patch.dict(ARCHITECTURES, {"tiny-test": _TINY})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def assertRefused(self, obj_or_path, pattern, **kwargs):
        path = obj_or_path if isinstance(obj_or_path, Path) else _save(obj_or_path, self.tmp / "b.pt")
        with self.assertRaisesRegex(EsmCodesBundleError, pattern):
            load_esm_codes_encoders(path, **kwargs)


class LoaderTest(_TinyArchitecture):
    """load_esm_codes_encoders: weights_only loading, strict rebuild, and every refusal."""

    def test_valid_bundle_loads_with_weights_only(self):
        path = _save(_tiny_bundle(), self.tmp / "ok.pt")
        raw = torch.load(path, map_location="cpu", weights_only=True)      # the format is weights_only-safe
        self.assertEqual(raw["format"], ENCODERS_FORMAT)
        with mock.patch.object(rce.torch, "load", wraps=torch.load) as spy:
            head = load_esm_codes_encoders(path, sha256=_sha256(path))
        self.assertTrue(all(c.kwargs.get("weights_only") is True for c in spy.call_args_list))
        self.assertIsInstance(head, ResidueCodeHead)
        self.assertFalse(head.training)
        self.assertTrue(all(not p.requires_grad and p.dtype == torch.float32 for p in head.parameters()))
        self.assertEqual(head.meta["architecture"]["name"], "tiny-test")
        for e in ("f_q", "f_t"):
            for k, v in raw[e].items():
                self.assertTrue(torch.equal(getattr(head, e).state_dict()[k], v))

    def test_loading_does_not_draw_from_the_global_rng(self):
        path = _save(_tiny_bundle(), self.tmp / "ok.pt")
        state = torch.get_rng_state()
        load_esm_codes_encoders(path)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_tampered_file_refused(self):
        path = _save(_tiny_bundle(), self.tmp / "ok.pt")
        good = _sha256(path)
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0x01
        bad = self.tmp / "tampered.pt"
        bad.write_bytes(bytes(data))
        self.assertRefused(bad, "sha256", sha256=good)
        truncated = self.tmp / "truncated.pt"
        truncated.write_bytes(path.read_bytes()[: len(data) // 3])
        self.assertRefused(truncated, "weights_only")

    def test_tampered_tensors_refused(self):
        b = _tiny_bundle()
        b["f_t"]["fc1.weight"] = torch.randn(9, 14)
        self.assertRefused(b, "does not match architecture")
        b = _tiny_bundle()
        b["f_q"]["extra.weight"] = torch.zeros(1)
        self.assertRefused(b, "does not match architecture")
        b = _tiny_bundle()
        del b["f_q"]["fc2.bias"]
        self.assertRefused(b, "does not match architecture")
        b = _tiny_bundle()
        b["f_t"]["lin.weight"] = b["f_t"]["lin.weight"].double()
        self.assertRefused(b, "float32")
        b = _tiny_bundle()
        b["transform"]["M"][2, 3] = float("nan")
        self.assertRefused(b, "non-finite")
        b = _tiny_bundle()
        b["transform"]["offset"] = b["transform"]["offset"].half()
        self.assertRefused(b, "transform.offset")
        b = _tiny_bundle()
        b["transform"]["dims"] = torch.arange(14, dtype=torch.int64)
        self.assertRefused(b, "transform.dims")
        b = _tiny_bundle()
        del b["f_t"]
        self.assertRefused(b, "missing")

    def test_bare_state_dict_refused(self):
        b = _tiny_bundle()
        bare = {f"f_q.{k}": v for k, v in b["f_q"].items()} | {f"f_t.{k}": v for k, v in b["f_t"].items()}
        self.assertRefused(bare, "bare state_dict")
        self.assertRefused({"architecture": _TINY}, "without a format tag")
        self.assertRefused([1, 2], "not an esm-codes encoder bundle")

    def test_unknown_format_refused(self):
        b = _tiny_bundle()
        b["format"] = "foldmatch-esm-codes-encoders/2"
        self.assertRefused(b, "unknown format")
        b["format"] = "foldmatch-three-di-head/1"
        self.assertRefused(b, "unknown format")

    def test_wrong_architecture_refused(self):
        b = _tiny_bundle()
        b["architecture"]["name"] = "esm-codes-mlp-v9"
        self.assertRefused(b, "unknown encoder architecture")
        b = _tiny_bundle()
        b["architecture"]["hidden"] = 16
        self.assertRefused(b, r"differs from the known one in \['hidden'\]")
        b = _tiny_bundle()
        b["architecture"]["activation"] = "gelu_tanh"
        self.assertRefused(b, "activation")
        b = _tiny_bundle()
        b["architecture"]["dims_dropped"] = (3, 11)       # a tuple is not what a bundle stores
        self.assertRefused(b, "dims_dropped")
        # the real architecture entry with the tiny tensors: every shape is wrong
        b = _tiny_bundle()
        b["architecture"] = dict(ARCHITECTURES["esm-codes-mlp-v1"])
        self.assertRefused(b, "transform")

    def test_other_conventions_refused(self):
        for section, key, value in (("code", "bitorder", "little"), ("transform", "folded_into_encoders", True),
                                    ("transform", "output_cast", "float32"), ("query", "dtype", "float32"),
                                    ("code", "bits", 64.0)):
            b = _tiny_bundle()
            b["conventions"][section][key] = value
            self.assertRefused(b, f"convention {section}.{key}")

    def test_not_weights_only_refused(self):
        b = _tiny_bundle()
        b["provenance"]["array"] = np.zeros(3)          # numpy objects need full unpickling
        self.assertRefused(b, "weights_only")
        b = _tiny_bundle()
        b["provenance"]["exc"] = ValueError("x")
        self.assertRefused(b, "weights_only")


class PackbitsTest(unittest.TestCase):
    """On-device packing equals numpy.packbits(bits, axis=1) (MSB first)."""

    def _devices(self):
        devs = ["cpu"]
        if torch.backends.mps.is_available():
            devs.append("mps")
        if torch.cuda.is_available():
            devs.append("cuda")
        return devs

    def test_equals_numpy_packbits(self):
        rng = np.random.default_rng(7)
        for dev in self._devices():
            for rows, width in ((0, 64), (1, 64), (37, 64), (5, 8), (300, 16)):
                bits = rng.integers(0, 2, size=(rows, width)).astype(bool)
                got = packbits_rows(torch.from_numpy(bits).to(dev))
                self.assertEqual(got.dtype, torch.uint8)
                self.assertEqual(got.device.type, dev)
                np.testing.assert_array_equal(got.cpu().numpy(), np.packbits(bits, axis=1), err_msg=dev)
        every_byte = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(bool)
        np.testing.assert_array_equal(packbits_rows(torch.from_numpy(every_byte)).numpy()[:, 0], np.arange(256))

    def test_bad_shapes_refused(self):
        with self.assertRaises(ValueError):
            packbits_rows(torch.zeros(3, 12, dtype=torch.bool))
        with self.assertRaises(ValueError):
            packbits_rows(torch.zeros(64, dtype=torch.bool))


class EncodeTest(_TinyArchitecture):
    """encode_target / encode_query on synthetic weights: shapes, casts, guards, autocast."""

    def setUp(self):
        super().setUp()
        self.bundle = _tiny_bundle(3)
        self.head = head_from_bundle(self.bundle)
        self.x = torch.randn(23, 16, generator=torch.Generator().manual_seed(5)) * 40

    def _reference(self, x):
        """S2-S5 written out with plain tensor ops (F.linear) from the bundle dict."""
        b = self.bundle
        x16 = x.to(torch.float16)
        xs = x16[:, b["transform"]["dims"]].to(torch.float32) - b["transform"]["offset"]
        xw = (xs @ b["transform"]["M"]).to(torch.float16).to(torch.float32)

        def f(s):
            h1 = F.gelu(F.linear(xw, s["fc1.weight"], s["fc1.bias"]), approximate="none")
            return F.linear(xw, s["lin.weight"], s["lin.bias"]) + F.linear(h1, s["fc2.weight"], s["fc2.bias"])
        with torch.no_grad():
            ht, hq = f(b["f_t"]), f(b["f_q"])
        return np.packbits((ht >= 0).numpy(), axis=1), hq.to(torch.float16).numpy()

    def test_outputs_match_the_written_out_spec(self):
        codes = self.head.encode_target(self.x)
        q = self.head.encode_query(self.x)
        self.assertEqual((codes.dtype, tuple(codes.shape)), (torch.uint8, (23, 8)))
        self.assertEqual((q.dtype, tuple(q.shape)), (torch.float16, (23, 64)))
        ref_codes, ref_q = self._reference(self.x)
        np.testing.assert_array_equal(codes.numpy(), ref_codes)
        np.testing.assert_array_equal(q.numpy().view(np.uint16), ref_q.view(np.uint16))

    def test_transform_output_is_fp16_exact(self):
        xw = self.head.transform(self.x)
        self.assertEqual(xw.dtype, torch.float32)
        self.assertTrue(torch.equal(xw, xw.to(torch.float16).to(torch.float32)))

    def test_input_is_cast_to_fp16_first(self):
        x16 = self.x.to(torch.float16)
        self.assertTrue(torch.equal(self.head.encode_target(self.x), self.head.encode_target(x16)))
        self.assertTrue(torch.equal(self.head.encode_query(self.x), self.head.encode_query(x16.double())))

    def test_empty_and_single_residue(self):
        self.assertEqual(tuple(self.head.encode_target(torch.zeros(0, 16)).shape), (0, 8))
        self.assertEqual(tuple(self.head.encode_query(torch.zeros(0, 16)).shape), (0, 64))
        self.assertTrue(torch.equal(self.head.encode_target(self.x[4:5]), self.head.encode_target(self.x)[4:5]))

    def test_wrong_width_refused(self):
        with self.assertRaises(ValueError):
            self.head.encode_target(torch.zeros(4, 15))
        with self.assertRaises(ValueError):
            self.head.encode_query(torch.zeros(16))

    def test_autocast_is_disabled(self):
        q = self.head.encode_query(self.x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            q_ac = self.head.encode_query(self.x)
            c_ac = self.head.encode_target(self.x)
            self.assertTrue(torch.is_autocast_enabled("cpu"))           # restored after each call
        self.assertTrue(torch.equal(q, q_ac))
        self.assertTrue(torch.equal(self.head.encode_target(self.x), c_ac))

    def test_reduced_precision_weights_refused(self):
        self.head.half()
        with self.assertRaisesRegex(RuntimeError, "float32"):
            self.head.encode_target(self.x)

    def test_sign_rule_zero_is_a_one_bit(self):
        """S4: bit_k = (h_k >= 0), so an output of exactly +0.0 or -0.0 is a 1 bit (and the smallest negative
        subnormal a 0 bit). Real weights never produce an exact zero on the fixtures, so the rule is pinned here."""
        # (a) through the real code path: f_t outputs 0, 9 and 63 have all-zero weights and biases -> h is exactly 0
        zeroed = (0, 9, 63)
        st = self.bundle["f_t"]
        for key in ("lin.weight", "lin.bias", "fc2.weight", "fc2.bias"):
            st[key][list(zeroed)] = 0.0
        head = head_from_bundle(self.bundle)
        with torch.no_grad():
            h = head.f_t(head.transform(self.x))
        self.assertTrue(bool((h[:, list(zeroed)] == 0).all()))
        self.assertTrue(bool((h[:, [k for k in range(64) if k not in zeroed]] != 0).all()))
        codes = head.encode_target(self.x).numpy()
        np.testing.assert_array_equal(codes, np.packbits((h >= 0).numpy(), axis=1))
        self.assertTrue(np.unpackbits(codes, axis=1)[:, list(zeroed)].all(), "h = 0 must give bit 1")

        # (b) signed zeros and subnormals, through encode_target with f_t replaced by fixed outputs
        tiny = float(np.nextafter(np.float32(0), np.float32(1)))
        row = torch.tensor([0.0, -0.0, tiny, -tiny, 1.0, -1.0, 3.5e38, -3.5e38] * 8, dtype=torch.float32)
        fixed = torch.stack([row, -row, row.roll(1), torch.zeros(64), -torch.zeros(64)])

        class _Fixed(torch.nn.Module):
            def forward(self, xw):
                return fixed[: xw.shape[0]]
        head.f_t = _Fixed()
        codes = head.encode_target(self.x[:5]).numpy()
        bits = np.unpackbits(codes, axis=1)
        np.testing.assert_array_equal(codes, np.packbits((fixed >= 0).numpy(), axis=1))
        np.testing.assert_array_equal(bits[0, :8], [1, 1, 1, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(bits[1, :8], [1, 1, 0, 1, 0, 1, 0, 1])
        self.assertTrue(bits[3:].all(), "+0.0 and -0.0 rows must be all 1 bits")

    def test_content_digests(self):
        d = self.head.content_digests()
        self.assertEqual(d["transform_sha256"], tensor_digest(self.bundle["transform"]))
        self.assertEqual(d["query_encoder_sha256"], tensor_digest(self.bundle["f_q"]))
        self.assertEqual(d["target_encoder_sha256"], tensor_digest(self.bundle["f_t"]))
        # numpy and torch containers give the same digest; any change of a value, dtype or name changes it
        t = self.bundle["transform"]
        self.assertEqual(tensor_digest({k: v.numpy() for k, v in t.items()}), d["transform_sha256"])
        bumped = dict(t, M=t["M"].clone())
        bumped["M"][0, 0] = torch.nextafter(bumped["M"][0, 0], torch.tensor(1e9))
        self.assertNotEqual(tensor_digest(bumped), d["transform_sha256"])
        self.assertNotEqual(tensor_digest(dict(t, M=t["M"].double())), d["transform_sha256"])
        self.assertNotEqual(tensor_digest({"m": t["M"], "dims": t["dims"], "offset": t["offset"]}),
                            d["transform_sha256"])


_CONTEXT_SCRIPT = textwrap.dedent("""
    import json, sys, warnings
    warnings.simplefilter("ignore")
    import torch
    from foldmatch.model import residue_code_encoder as rce

    def snapshot():
        out = {}
        for path in ("", "cudnn", "mkldnn", "cuda.matmul", "cudnn.conv", "cudnn.rnn", "mkldnn.matmul", "mkldnn.conv",
                     "mkldnn.rnn"):
            try:
                out["new:" + path] = rce._backend(path).fp32_precision
            except (AttributeError, RuntimeError) as e:
                out["new:" + path] = "unavailable" if isinstance(e, AttributeError) else "refused"
        for name, get in (("matmul_precision", torch.get_float32_matmul_precision),
                          ("cublas_tf32", torch._C._get_cublas_allow_tf32),
                          ("cudnn_tf32", torch._C._get_cudnn_allow_tf32),
                          ("autocast_cpu", lambda: torch.is_autocast_enabled("cpu"))):
            try:
                out[name] = get()
            except RuntimeError:
                out[name] = "refused"
        return out

    scenario = sys.argv[1]
    if scenario == "legacy_path":
        rce._fp32_precision_handles = lambda paths: []      # torch < 2.9 code path
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif scenario == "legacy_high":
        torch.set_float32_matmul_precision("high")
    elif scenario == "legacy_medium":
        torch.set_float32_matmul_precision("medium")
    elif scenario == "new_tf32":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    elif scenario == "new_generic_tf32":
        torch.backends.fp32_precision = "tf32"
    elif scenario == "mkldnn_bf16":
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
    before = snapshot()
    ac = torch.autocast("cpu", dtype=torch.bfloat16) if scenario == "autocast" else None
    if ac is not None:
        ac.__enter__()
        before = snapshot()
    with rce.strict_fp32("cpu"):
        inside = snapshot()
    after = snapshot()
    try:
        with rce.strict_fp32():
            raise KeyError("boom")
    except KeyError:
        pass
    after_error = snapshot()
    if ac is not None:
        ac.__exit__(None, None, None)
    print(json.dumps(dict(before=before, inside=inside, after=after, after_error=after_error)))
""")


class Fp32ContextTest(unittest.TestCase):
    """strict_fp32: inside, full fp32 matmuls and no autocast; on exit (also after an exception) every flag is back,
    whichever TF32 API the caller used. Each scenario runs in its own interpreter: the flags are process-global."""

    SCENARIOS = ("default", "legacy_high", "legacy_medium", "new_tf32", "new_generic_tf32", "mkldnn_bf16", "autocast",
                 "legacy_path")

    def _run(self, scenario):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(_SRC), os.environ.get("PYTHONPATH", "")]))
        r = subprocess.run([sys.executable, "-c", _CONTEXT_SCRIPT, scenario], capture_output=True, text=True,
                           env=env, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_flags_inside_and_restored(self):
        for scenario in self.SCENARIOS:
            with self.subTest(scenario=scenario):
                s = self._run(scenario)
                if scenario == "legacy_path":
                    # torch < 2.9 has only the legacy flags; on this torch the simulation leaves the per-backend
                    # settings unmanaged, so only the legacy flags are compared
                    s = {k: {f: v for f, v in snap.items() if not f.startswith("new:")} for k, snap in s.items()}
                self.assertEqual(s["after"], s["before"])
                self.assertEqual(s["after_error"], s["before"])
                inside = s["inside"]
                self.assertIs(inside["autocast_cpu"], False)
                self.assertIs(inside["cublas_tf32"], False)
                self.assertEqual(inside["matmul_precision"], "highest")
                if scenario == "legacy_path":
                    self.assertIs(inside["cudnn_tf32"], False)
                else:
                    for path in ("cuda.matmul", "cudnn.conv", "cudnn.rnn", "mkldnn.matmul", "mkldnn.conv",
                                 "mkldnn.rnn"):
                        if inside["new:" + path] != "unavailable":
                            self.assertEqual(inside["new:" + path], "ieee", path)


# =====================================================================================================================
# Tests with the real esm-codes-v1 bundle
# =====================================================================================================================
def _bundle_dir():
    d = os.environ.get(BUNDLE_ENV)
    return Path(d) if d else None


_SKIP_NO_BUNDLE = unittest.skipIf(
    _bundle_dir() is None,
    f"{BUNDLE_ENV} is not set: point it at a directory holding the esm-codes-v1 bundle files ({ENCODERS_FILE}, "
    f"{SCORING_FILE}, bundle.json); P4 adds the pinned HF download")


class _RealBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = _bundle_dir()
        cls.dir = d
        for f in (ENCODERS_FILE, SCORING_FILE, "bundle.json"):
            if not (d / f).is_file():
                raise AssertionError(f"{BUNDLE_ENV}={d} has no {f}")
        cls.bundle_json = json.loads((d / "bundle.json").read_text())
        cls.head = load_esm_codes_encoders(d / ENCODERS_FILE,
                                           sha256=cls.bundle_json["files"][ENCODERS_FILE]["sha256"])


@_SKIP_NO_BUNDLE
class BundleFilesTest(_RealBundle):
    """The published files themselves: weights_only, sha256s, content digests, torch-free scoring npz."""

    def test_encoders_file_is_weights_only(self):
        raw = torch.load(self.dir / ENCODERS_FILE, map_location="cpu", weights_only=True)
        self.assertEqual(raw["format"], ENCODERS_FORMAT)
        self.assertEqual(raw["system"], self.bundle_json["system"])
        self.assertEqual(raw["architecture"], ARCHITECTURES["esm-codes-mlp-v1"])
        self.assertEqual(self.head.meta["architecture"]["name"], "esm-codes-mlp-v1")
        self.assertEqual(sum(p.numel() for p in self.head.parameters()), 3_471_360)

    def test_sha256s_match_bundle_json(self):
        for f in (ENCODERS_FILE, SCORING_FILE):
            self.assertEqual(_sha256(self.dir / f), self.bundle_json["files"][f]["sha256"], f)
        with self.assertRaisesRegex(EsmCodesBundleError, "sha256"):
            load_esm_codes_encoders(self.dir / ENCODERS_FILE, sha256="0" * 64)

    def test_content_digests_match_bundle_json(self):
        want = self.bundle_json["content_digests"]
        self.assertEqual(want["method"], rce.DIGEST_METHOD)
        for k, v in self.head.content_digests().items():
            self.assertEqual(v, want[k], k)

    def test_scoring_npz_loads_without_torch(self):
        code = textwrap.dedent(f"""
            import json, sys
            sys.modules["torch"] = None
            import numpy as np
            with np.load({str(self.dir / SCORING_FILE)!r}, allow_pickle=False) as z:
                a = {{k: z[k] for k in z.files}}
            meta = json.loads(str(a["meta_json"]))
            print(json.dumps(dict(files=sorted(a), system=meta["system"], codes=[list(a["panel_codes"].shape),
                  str(a["panel_codes"].dtype)], members=len(a["panel_offsets"]) - 1,
                  residues=int(a["panel_offsets"][-1]), lam=a["lam"].tolist(), dtype_lam=str(a["lam"].dtype),
                  torch_loaded=sys.modules.get("torch") is not None)))
        """)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        out = json.loads(r.stdout)
        self.assertEqual(out["files"], ["bin_centres", "lam", "lnK", "meta_json", "panel_codes", "panel_ids",
                                        "panel_lengths", "panel_offsets"])
        self.assertEqual(out["system"], "esm-codes-v1")
        self.assertEqual(out["codes"], [[355626, 8], "uint8"])
        self.assertEqual((out["members"], out["residues"]), (1000, 355626))
        self.assertEqual((len(out["lam"]), out["dtype_lam"]), (6, "float64"))
        self.assertEqual(out["lam"], self.bundle_json["statistics"]["lam"])
        self.assertFalse(out["torch_loaded"])


def _f2_items():
    z = np.load(_FIXTURES / "encoder_rows_v1.npz", allow_pickle=False)
    o = z["offsets"]
    return [torch.from_numpy(z["rows"][o[k]:o[k + 1]].copy()) for k in range(len(o) - 1)], z["codes"], z["q"]


def _f2c_items():
    z = np.load(_FIXTURES / "fasta_rows_codes_v1.npz", allow_pickle=False)
    ro = z["rows_offsets"]
    rows = [torch.from_numpy(z["rows"][ro[k] + 1:ro[k + 1] - 1].copy()) for k in range(len(ro) - 1)]  # S1: no BOS/EOS
    return rows, z["codes"], z["q"]


def _f2b_items():
    z = np.load(_FIXTURES / "ood_rows_codes_v1.npz", allow_pickle=False)
    rows = [torch.load(_EMBEDDINGS / f"{name}.pt", map_location="cpu", weights_only=True)
            for name in z["ids"].astype(str)]
    return rows, z["codes"], z["q"]


def _encode(head, items, device=None):
    if device is not None:
        head = head.to(device)
    codes = np.concatenate([head.encode_target(x).cpu().numpy() for x in items])
    q = np.concatenate([head.encode_query(x).cpu().numpy() for x in items])
    return codes, q


def _bits_equal(a, b):
    return int((np.unpackbits(a, axis=1) == np.unpackbits(b, axis=1)).sum()), int(a.size * 8)


@_SKIP_NO_BUNDLE
class EncoderParityTest(_RealBundle):
    """Acceptance (plan §6): F2 bits >= 0.99999 and q within 1 fp16 ulp; F2b and F2c codes identical to the reference
    encoder on the same rows.

    "Within 1 fp16 ulp" is measured against each row's largest |q| (``_row_ulps``), on every platform: the fixtures
    were made on an Apple silicon CPU, and other CPUs round the fp32 GEMMs differently. On Linux x86-64 (OpenBLAS,
    torch 2.9 and 2.11) F2 has 19,631 of 20,416 q values bit-equal, with per-element distances up to 927 ulps on
    near-zero values but at most 1 ulp of the row maximum; the codes are identical."""

    def test_f2(self):
        items, codes_ref, q_ref = _f2_items()
        codes, q = _encode(self.head, items)
        equal, n = _bits_equal(codes, codes_ref)
        self.assertEqual(n, 20_416)
        self.assertGreaterEqual(equal / n, 0.99999, f"F2 code bits {equal}/{n}")
        self.assertLessEqual(_row_ulps(q, q_ref), 1.0, "F2 q beyond 1 fp16 ulp of the row maximum")

    def test_f2c(self):
        items, codes_ref, q_ref = _f2c_items()
        codes, q = _encode(self.head, items)
        np.testing.assert_array_equal(codes, codes_ref)
        self.assertLessEqual(_row_ulps(q, q_ref), 1.0)
        # the same rows as fp32 (what ESM3 hands over) give the same encodings: S2 starts with fp16(x)
        codes32, q32 = _encode(self.head, [x.float() for x in items])
        np.testing.assert_array_equal(codes32, codes)
        np.testing.assert_array_equal(q32.view(np.uint16), q.view(np.uint16))

    def test_f2b(self):
        items, codes_ref, q_ref = _f2b_items()
        self.assertEqual([x.dtype for x in items], [torch.float32, torch.float32])
        codes, q = _encode(self.head, items)
        np.testing.assert_array_equal(codes, codes_ref)
        self.assertLessEqual(_row_ulps(q, q_ref), 1.0)

    @unittest.skipUnless(torch.backends.mps.is_available() or torch.cuda.is_available(), "no GPU (MPS or CUDA)")
    def test_gpu_matches_cpu(self):
        """D18: the encoders on the GPU (on-device packing, strict fp32, autocast off) agree with the CPU at least as
        well as the pre-registered Mac-vs-Linux CPU gate (A4-xcpu, 0.999)."""
        device = "cuda" if torch.cuda.is_available() else "mps"
        items, codes_ref, q_ref = _f2c_items()
        head = load_esm_codes_encoders(self.dir / ENCODERS_FILE, device=device)
        codes, q = _encode(head, items)
        equal, n = _bits_equal(codes, codes_ref)
        self.assertGreaterEqual(equal / n, 0.999, f"{device} code bits {equal}/{n}")
        self.assertLessEqual(_row_ulps(q, q_ref), 4.0)


@_SKIP_NO_BUNDLE
class NotRecoverableFromWeightsTest(_RealBundle):
    """The weights alone do not define the codes: GELU-tanh, dropping an fp16 cast or folding the whitener into the
    first layer each changes the reference codes and the query encodings.

    Each variant is compared with the module's own output on the same machine (the arithmetic differs only by the
    variant), and its codes with the fixture codes (which the module reproduces on every platform). Comparing variant
    q with the fixture q instead would mix in the platform's own last-bit differences (785 of 20,416 F2 values on
    Linux x86-64)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.items, cls.codes_ref, cls.q_ref = _f2_items()
        cls.codes, cls.q = _encode(cls.head, cls.items)

    def _variant(self, xw_fn, approximate="none", items=None):
        h = self.head

        def f(m, xw):
            return m.lin(xw) + m.fc2(F.gelu(m.fc1(xw), approximate=approximate))
        codes, q = [], []
        with strict_fp32("cpu"), torch.no_grad():
            for x in (items if items is not None else self.items):
                xw = xw_fn(x)
                codes.append(packbits_rows(f(h.f_t, xw) >= 0).numpy())
                q.append(f(h.f_q, xw).to(torch.float16).numpy())
        return np.concatenate(codes), np.concatenate(q)

    def _assert_changes(self, codes, q, codes_ref, q_module):
        self.assertFalse(np.array_equal(codes, codes_ref), "codes unchanged")
        changed = int((q.view(np.uint16) != q_module.view(np.uint16)).sum())
        self.assertGreater(changed, q.size // 100, f"only {changed} of {q.size} q values changed")

    def test_module_reproduces_the_reference(self):
        np.testing.assert_array_equal(self.codes, self.codes_ref)

    def test_variant_harness_reproduces_the_module(self):
        """Control: the harness with the module's own arithmetic gives the module's codes and q bit for bit, so every
        difference below comes from the variant."""
        codes, q = self._variant(self.head.transform)
        np.testing.assert_array_equal(codes, self.codes)
        np.testing.assert_array_equal(q.view(np.uint16), self.q.view(np.uint16))

    def test_gelu_tanh(self):
        codes, q = self._variant(self.head.transform, approximate="tanh")
        self._assert_changes(codes, q, self.codes_ref, self.q)

    def test_without_the_xw_fp16_cast(self):
        t = self.head.transform

        def xw(x):
            return (x.to(torch.float16).index_select(1, t.dims).float() - t.offset) @ t.M
        codes, q = self._variant(xw)
        self._assert_changes(codes, q, self.codes_ref, self.q)

    def test_whitener_folded_into_the_first_layer(self):
        t, h = self.head.transform, self.head

        def run(m, xs):
            return (xs @ (t.M @ m.lin.weight.T) + m.lin.bias) + m.fc2(F.gelu(xs @ (t.M @ m.fc1.weight.T) + m.fc1.bias))
        codes, q = [], []
        with strict_fp32("cpu"), torch.no_grad():
            for x in self.items:
                xs = x.to(torch.float16).index_select(1, t.dims).float() - t.offset
                codes.append(packbits_rows(run(h.f_t, xs) >= 0).numpy())
                q.append(run(h.f_q, xs).to(torch.float16).numpy())
        self._assert_changes(np.concatenate(codes), np.concatenate(q), self.codes_ref, self.q)

    def test_without_the_input_fp16_cast(self):
        items, codes_ref, _q_ref = _f2b_items()     # fp32 rows: the input cast only matters for them
        codes_module, q_module = _encode(self.head, items)
        np.testing.assert_array_equal(codes_module, codes_ref)
        t = self.head.transform

        def xw(x):
            return ((x.float().index_select(1, t.dims) - t.offset) @ t.M).half().float()
        codes, q = self._variant(xw, items=items)
        self._assert_changes(codes, q, codes_ref, q_module)


if __name__ == "__main__":
    unittest.main()
