import ast
import hashlib
import os
import pickle
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import foldmatch
from foldmatch.utils.data import load_residue_embedding
from foldmatch.utils.esm import loaders

_PACKAGE = Path(foldmatch.__file__).parent


def _listing(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


def _package_modules():
    for path in sorted(_PACKAGE.rglob("*.py")):
        yield path.relative_to(_PACKAGE.parent), ast.parse(path.read_text(), filename=str(path))


def _call_name(call):
    func = call.func
    if isinstance(func, ast.Attribute):
        prefix = func.value.id + "." if isinstance(func.value, ast.Name) else ""
        return prefix + func.attr
    return func.id if isinstance(func, ast.Name) else None


class EsmWeightsPinTest(unittest.TestCase):
    """G1: the ESM3 snapshot is pinned and its weights are verified without writing anything."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        loaders._hashed.clear()

    def tearDown(self):
        loaders._hashed.clear()
        self._tmp.cleanup()

    def _cache_layout(self, blob_name, content=b"weights"):
        """A Hugging Face cache entry: snapshots/<rev>/<file> -> ../../blobs/<blob_name>."""
        blob = self.root / "blobs" / blob_name
        blob.parent.mkdir(parents=True)
        blob.write_bytes(content)
        link = self.root / "snapshots" / loaders.ESM_REVISION / loaders.ESM_WEIGHTS_FILE
        link.parent.mkdir(parents=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
        return link

    def _copied_layout(self, content):
        path = self.root / "local" / loaders.ESM_WEIGHTS_FILE
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        return path

    def test_pins(self):
        self.assertEqual(loaders.ESM_REPO_ID, "rcsb/rcsb-esm")
        self.assertEqual(loaders.ESM_REVISION, "636acb420daf674df37ab0220f56061a5de2f169")
        self.assertEqual(
            loaders.ESM_WEIGHTS_SHA256, "5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9"
        )
        self.assertEqual(loaders.ESM_WEIGHTS_FILE, "data/weights/esm3_sm_open_v1.pth")

    def test_blob_name_verifies_without_hashing(self):
        link = self._cache_layout(loaders.ESM_WEIGHTS_SHA256)
        before = _listing(self.root)
        with mock.patch.object(loaders, "_file_sha256", side_effect=AssertionError("hashed")):
            self.assertEqual(loaders.verify_esm_weights(link), "blob-name")
        self.assertEqual(_listing(self.root), before)

    def test_wrong_blob_name_is_refused(self):
        link = self._cache_layout("0" * 64)
        with self.assertRaisesRegex(loaders.EsmWeightsMismatchError, "expected " + loaders.ESM_WEIGHTS_SHA256):
            loaders.verify_esm_weights(link)

    def test_hidden_layout_is_hashed_once_per_process(self):
        content = b"copied weights"
        path = self._copied_layout(content)
        before = _listing(self.root)
        sha = hashlib.sha256(content).hexdigest()
        spy = mock.Mock(wraps=loaders._file_sha256)
        with mock.patch.object(loaders, "ESM_WEIGHTS_SHA256", sha), mock.patch.object(loaders, "_file_sha256", spy):
            self.assertEqual(loaders.verify_esm_weights(path), "sha256")
            self.assertEqual(loaders.verify_esm_weights(path), "sha256")
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(_listing(self.root), before)

    def test_hidden_layout_with_wrong_content_is_refused(self):
        path = self._copied_layout(b"not the weights")
        with self.assertRaisesRegex(loaders.EsmWeightsMismatchError, "sha256 " + hashlib.sha256(b"not the weights").hexdigest()):
            loaders.verify_esm_weights(path)

    def test_full_hash_reads_the_content_behind_a_blob_name(self):
        link = self._cache_layout(loaders.ESM_WEIGHTS_SHA256, content=b"truncated download")
        with self.assertRaisesRegex(loaders.EsmWeightsMismatchError, "is corrupt"):
            loaders.verify_esm_weights(link, full_hash=True)

    def _data_root(self, cached):
        """Resolve DataRoot against a fake cache; returns (path, snapshot_download mock, verify mock)."""
        snapshot = self.root / "snapshots" / loaders.ESM_REVISION
        snapshot.mkdir(parents=True)
        cached_value = str(snapshot / loaders.ESM_WEIGHTS_FILE) if cached else None
        saved = loaders.DataRoot._instance
        loaders.DataRoot._instance = None
        try:
            with mock.patch.object(loaders, "try_to_load_from_cache", return_value=cached_value) as lookup, \
                    mock.patch.object(loaders, "snapshot_download", return_value=str(snapshot)) as download, \
                    mock.patch.object(loaders, "verify_esm_weights") as verify:
                path = loaders.data_root()
                self.assertIs(loaders.DataRoot(), loaders.DataRoot())
                self.assertEqual(verify.call_count, 1)
        finally:
            loaders.DataRoot._instance = saved
        lookup.assert_called_once_with(loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_FILE, revision=loaders.ESM_REVISION)
        download.assert_called_once_with(repo_id=loaders.ESM_REPO_ID, revision=loaders.ESM_REVISION)
        self.assertEqual(path, snapshot)
        return verify

    def test_data_root_checks_a_cached_snapshot_by_blob_name(self):
        verify = self._data_root(cached=True)
        verify.assert_called_once_with(
            self.root / "snapshots" / loaders.ESM_REVISION / loaders.ESM_WEIGHTS_FILE, full_hash=False
        )

    def test_data_root_hashes_a_fresh_download_once(self):
        verify = self._data_root(cached=False)
        verify.assert_called_once_with(
            self.root / "snapshots" / loaders.ESM_REVISION / loaders.ESM_WEIGHTS_FILE, full_hash=True
        )

    def test_failed_verification_leaves_no_data_root(self):
        saved = loaders.DataRoot._instance
        loaders.DataRoot._instance = None
        try:
            with mock.patch.object(loaders, "try_to_load_from_cache", return_value=None), \
                    mock.patch.object(loaders, "snapshot_download", return_value=str(self.root)), \
                    mock.patch.object(loaders, "verify_esm_weights", side_effect=loaders.EsmWeightsMismatchError("mismatch")):
                with self.assertRaisesRegex(loaders.EsmWeightsMismatchError, "mismatch"):
                    loaders.data_root()
            self.assertIsNone(loaders.DataRoot._instance)
        finally:
            loaders.DataRoot._instance = saved

    def test_mismatch_is_reported_as_a_clean_cli_error(self):
        """The CLIs render ValueError as one 'Error: ...' line (exit 1), not a traceback."""
        self.assertTrue(issubclass(loaders.EsmWeightsMismatchError, ValueError))
        link = self._cache_layout("0" * 64)
        import importlib
        import io
        from contextlib import redirect_stderr
        for name in ("foldmatch.cli.search", "foldmatch.cli.embedding", "foldmatch.cli.inference"):
            cli = importlib.import_module(name)
            with self.subTest(cli=name), \
                    mock.patch.object(cli, "app", side_effect=lambda: loaders.verify_esm_weights(link)):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
                    cli.main()
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("Error: ESM3 weights", stderr.getvalue())
                self.assertIn("run download-models again", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_local_cache_holds_the_pinned_blob(self):
        cached = loaders.try_to_load_from_cache(
            loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_FILE, revision=loaders.ESM_REVISION
        )
        if not isinstance(cached, str):
            self.skipTest(f"{loaders.ESM_REPO_ID}@{loaders.ESM_REVISION} is not in the local Hugging Face cache")
        if not re.fullmatch(r"[0-9a-f]{64}", Path(os.path.realpath(cached)).name):
            self.skipTest("the local Hugging Face cache does not use blob symlinks")
        with mock.patch.object(loaders, "_file_sha256", side_effect=AssertionError("hashed")):
            self.assertEqual(loaders.verify_esm_weights(cached), "blob-name")


class _NotAWeight:
    pass


class TorchLoadWeightsOnlyTest(unittest.TestCase):
    """G5: every torch.load in the package refuses pickled code (weights_only=True)."""

    def test_every_torch_load_passes_weights_only(self):
        calls = []
        for module, tree in _package_modules():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _call_name(node) == "torch.load":
                    weights_only = [k.value for k in node.keywords if k.arg == "weights_only"]
                    calls.append((f"{module}:{node.lineno}", weights_only))
        self.assertGreaterEqual(len(calls), 4)
        for where, weights_only in calls:
            self.assertEqual(len(weights_only), 1, where)
            self.assertIsInstance(weights_only[0], ast.Constant, where)
            self.assertIs(weights_only[0].value, True, where)

    def test_residue_embedding_loader_refuses_pickled_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.pt"
            torch.save(torch.arange(6.0).reshape(2, 3), path)
            self.assertTrue(torch.equal(load_residue_embedding(path), torch.arange(6.0).reshape(2, 3)))
            torch.save({"embedding": _NotAWeight()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                load_residue_embedding(path)


if __name__ == "__main__":
    unittest.main()
