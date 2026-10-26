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
from foldmatch.utils import model
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


class DependencyPinsTest(unittest.TestCase):
    """The runtime dependencies in pyproject.toml are exact pins; threadpoolctl is one of them (P7.0d: the esm-codes
    Stage 2 checks and limits its BLAS threads with it, S6/S16)."""

    def setUp(self):
        path = _PACKAGE.parent.parent / "pyproject.toml"
        if not path.is_file():
            self.skipTest(f"no pyproject.toml next to the package ({path}): not a source tree")
        import tomllib
        self.dependencies = tomllib.loads(path.read_text())["project"]["dependencies"]

    def test_every_dependency_is_an_exact_pin(self):
        self.assertGreaterEqual(len(self.dependencies), 13)
        for requirement in self.dependencies:
            with self.subTest(requirement=requirement):
                self.assertRegex(requirement, r"^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][0-9A-Za-z.+!]*$")

    def test_threadpoolctl_is_declared(self):
        self.assertIn("threadpoolctl==3.6.0", self.dependencies)

    def test_faiss_cpu_is_the_threaded_wheel(self):
        """faiss-cpu 1.15.1 (P9): its Linux wheels bundle an OpenMP OpenBLAS, where 1.13.2's was single-threaded."""
        self.assertIn("faiss-cpu==1.15.1", self.dependencies)
        self.assertFalse([r for r in self.dependencies if r.startswith("faiss") and r != "faiss-cpu==1.15.1"])

    def test_the_index_phase_names_the_pinned_faiss_cpu(self):
        """The index phase's line about a single-threaded FAISS BLAS tells to install the pinned faiss-cpu."""
        from foldmatch.cli import search
        self.assertIn(f"faiss-cpu=={search._FAISS_CPU_PIN}", self.dependencies)


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


# Hugging Face calls that take a revision; esm's own from_pretrained fetches EvolutionaryScale
# repositories at whatever revision main points to, so the package must not call it.
_HF_FETCHES = {"snapshot_download", "hf_hub_download", "try_to_load_from_cache", "hf_hub_url"}
_UNPINNED_FETCHES = {"from_pretrained"}
_COMMIT = re.compile(r"[0-9a-f]{40}")
_UNRESOLVED = "<not a literal>"
# ``revision=<system>.bundle_revision``: the esm-codes bundle loader takes its revision from a scoring-system registry
# entry, the only place a bundle revision exists (plan P4). Such a call is pinned when every registry entry's
# bundle_revision is a 40-hex commit (EsmCodesSystem refuses anything else; checked again here).
_REGISTRY_PIN = "<scoring-system registry bundle_revision>"


def _registry_is_pinned():
    from foldmatch.search import scoring_system
    return all(isinstance(s.bundle_revision, str) and _COMMIT.fullmatch(s.bundle_revision) is not None
               for s in scoring_system.REGISTRY.values())


def _is_pinned(revision):
    if revision == _REGISTRY_PIN:
        return _registry_is_pinned()
    return isinstance(revision, str) and _COMMIT.fullmatch(revision) is not None


def _resolve(node, scope, seen=()):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute) and node.attr == "bundle_revision":
        return _REGISTRY_PIN
    if isinstance(node, ast.Name) and node.id in scope and node.id not in seen:
        return _resolve(scope[node.id], scope, seen + (node.id,))
    return _UNRESOLVED


def _is_main_guard(node):
    test = node.test if isinstance(node, ast.If) else None
    return (
        isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
        and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)
    )


def _revisions_in(tree, module):
    """Every Hugging Face revision a module declares or passes, as (where, value).

    A *REVISION* constant or a ``revision=`` argument resolves through literals, module
    constants and parameter defaults; anything else, or a missing ``revision=``, is not a pin.
    """
    constants = {
        target.id: node.value
        for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name) and node.value is not None
    }
    found = [
        (f"{module}:{name}", _resolve(value, constants))
        for name, value in constants.items() if "REVISION" in name.upper()
    ]

    def visit(node, scope):
        if _is_main_guard(node):
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            positional = args.posonlyargs + args.args
            defaults = dict(zip([a.arg for a in positional[len(positional) - len(args.defaults):]], args.defaults))
            defaults.update({a.arg: d for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None})
            scope = {**scope, **{a.arg: defaults.get(a.arg, ast.Name(id="<parameter>")) for a in
                                 positional + args.kwonlyargs}}
        if isinstance(node, ast.Call):
            name = (_call_name(node) or "").split(".")[-1]
            where = f"{module}:{node.lineno} {name}()"
            if name in _HF_FETCHES:
                revision = [k.value for k in node.keywords if k.arg == "revision"]
                found.append((where, _resolve(revision[0], scope) if revision else None))
            elif name in _UNPINNED_FETCHES:
                found.append((where, None))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, constants)
    return found


class ReleasePinTest(unittest.TestCase):
    """G11: every Hugging Face revision in the package is a 40-hex commit; None (unpinned) is refused."""

    @classmethod
    def setUpClass(cls):
        cls.revisions = [found for module, tree in _package_modules() for found in _revisions_in(tree, module)]

    def test_every_revision_is_a_commit(self):
        unpinned = [(where, value) for where, value in self.revisions if not _is_pinned(value)]
        self.assertEqual(unpinned, [])

    def test_known_pins_are_found(self):
        found = dict(self.revisions)
        self.assertEqual(found["foldmatch/utils/esm/loaders.py:ESM_REVISION"], loaders.ESM_REVISION)
        self.assertEqual(found["foldmatch/utils/model.py:REVISION"], model.REVISION)
        calls = [where.split()[-1] for where, _ in self.revisions if where.endswith("()")]
        for fetch in ("snapshot_download()", "hf_hub_download()", "try_to_load_from_cache()"):
            self.assertIn(fetch, calls)
        bundle_calls = [where for where, value in self.revisions if value == _REGISTRY_PIN]
        self.assertEqual(len(bundle_calls), 1, bundle_calls)
        self.assertTrue(bundle_calls[0].startswith("foldmatch/utils/model.py:"), bundle_calls)
        self.assertTrue(bundle_calls[0].endswith("hf_hub_download()"), bundle_calls)

    def test_runtime_pins(self):
        for revision in (loaders.ESM_REVISION, model.REVISION):
            self.assertTrue(_is_pinned(revision), revision)

    def test_unpinned_revisions_are_refused(self):
        sha = "636acb420daf674df37ab0220f56061a5de2f169"
        cases = {
            f'REVISION = "{sha}"': sha,
            "REVISION = None": None,
            'REVISION = "main"': "main",
            f'REVISION = "{sha[:12]}"': sha[:12],
            f'REVISION = "{sha.upper()}"': sha.upper(),
            'snapshot_download(repo_id="r")': None,
            'hf_hub_download(repo_id="r", filename="f", revision=None)': None,
            'hf_hub_download(repo_id="r", filename="f", revision=REV)': _UNRESOLVED,
            f'R = "{sha}"\nsnapshot_download(repo_id="r", revision=R)': sha,
            'def get(revision=None):\n    return hf_hub_download(repo_id="r", filename="f", revision=revision)': None,
            f'def get(revision="{sha}"):\n    return hf_hub_download(repo_id="r", filename="f", revision=revision)': sha,
            'def get(revision):\n    return hf_hub_download(repo_id="r", filename="f", revision=revision)': _UNRESOLVED,
            'ESM3.from_pretrained("esm3_sm_open_v1")': None,
            'if __name__ == "__main__":\n    ESM3.from_pretrained("esm3_sm_open_v1")': "<none>",
        }
        for source, expected in cases.items():
            values = [value for _where, value in _revisions_in(ast.parse(source), "m.py")]
            if expected == "<none>":
                self.assertEqual(values, [], source)
                continue
            self.assertEqual(values[-1], expected, source)
            self.assertEqual(_is_pinned(values[-1]), expected == sha, source)

    def test_registry_revisions(self):
        """revision=<system>.bundle_revision resolves to the registry, pinned exactly when every entry is a commit;
        any other attribute is not a pin."""
        source = 'def get(system):\n    return hf_hub_download(repo_id=system.bundle_repo, filename="f", ' \
                 'revision=system.bundle_revision)'
        self.assertEqual([v for _w, v in _revisions_in(ast.parse(source), "m.py")], [_REGISTRY_PIN])
        source = 'def get(system):\n    return hf_hub_download(repo_id="r", filename="f", revision=system.revision)'
        self.assertEqual([v for _w, v in _revisions_in(ast.parse(source), "m.py")], [_UNRESOLVED])
        self.assertTrue(_is_pinned(_REGISTRY_PIN))
        from foldmatch.search import scoring_system
        loose = mock.Mock(bundle_revision="main")
        with mock.patch.object(scoring_system, "REGISTRY", {"x": loose}):
            self.assertFalse(_is_pinned(_REGISTRY_PIN))


if __name__ == "__main__":
    unittest.main()
