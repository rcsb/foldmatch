"""P6.0: ``download-models`` downloads and verifies only (plan section 8, P6.0).

One shared command (``fm-search``, ``fm-embedding`` and ``inference download-models``) fetches the pinned ESM3
snapshot, the aggregator weights and the esm-codes bundles into the Hugging Face cache and checks their sha256s. It
builds no model, loads no weights, never imports torch and makes no CUDA call, so it is safe on a shared GPU login
node.

ESM3 (2.8 GB) and the aggregator (690 MB) are synthetic in most tests: a fake Hub (``_FakeHub``) serves small files in
their place in the Hugging Face cache layout, and the pins are patched to their sha256s. The esm-codes bundle is the
published one (``tests/esm_codes_bundle.py``: skipped only when offline and not cached). ``RealCacheTest`` runs the
command on the real cached files, offline and without torch; ``RealBundleDownloadTest`` downloads the published bundle
into an empty cache (it needs the Hub, so it is skipped with ``HF_HUB_OFFLINE=1``).
"""
import functools
import hashlib
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr
from pathlib import Path
from unittest import mock

import huggingface_hub
import typer
from huggingface_hub import constants as hf_constants
from typer.testing import CliRunner

import foldmatch
from foldmatch.cli import args_utils
from foldmatch.search import scoring_system as ss
from foldmatch.utils import model
from foldmatch.utils.esm import loaders

CLIS = {"fm-search": "search", "fm-embedding": "embedding", "inference": "inference"}

ESM3_FILES = {
    loaders.ESM_WEIGHTS_FILE: b"synthetic ESM3 weights",
    "data/weights/esm3_structure_encoder_v0.pth": b"synthetic structure encoder",
    "README.md": b"synthetic readme",
}
AGGREGATOR = b"synthetic aggregator weights"
_REAL_SNAPSHOT_DOWNLOAD = huggingface_hub.snapshot_download
_REAL_HF_HUB_DOWNLOAD = huggingface_hub.hf_hub_download


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SYNTHETIC_PINS = {"esm3": _sha(ESM3_FILES[loaders.ESM_WEIGHTS_FILE]), "aggregator": _sha(AGGREGATOR)}


def _cli(label):
    return importlib.import_module("foldmatch.cli." + CLIS[label])


def _listing(root):
    """Every path under ``root`` (symlinked directories followed) with its mtime, or "link" for a symlink."""
    root = Path(root)
    out = []
    for folder, dirs, files in os.walk(root, followlinks=True):
        for name in dirs + files:
            p = Path(folder) / name
            out.append((str(p.relative_to(root)), "link" if p.is_symlink() else p.stat().st_mtime_ns))
    return sorted(out)


@functools.lru_cache(maxsize=None)
def _published_bundle_bytes():
    """The published esm-codes-v1 bundle, ``{file name: bytes}`` (``unittest.SkipTest`` only offline and not
    cached)."""
    import esm_codes_bundle
    return {name: Path(path).read_bytes() for name, path in esm_codes_bundle.registry_files().items()}


def _repo_dir(cache: Path, repo_id: str) -> Path:
    return cache / ("models--" + repo_id.replace("/", "--"))


class _FakeHub:
    """The Hugging Face Hub behind ``snapshot_download`` and ``hf_hub_download``, patched where ``loaders`` and
    ``model`` call them. Online it writes a requested file that is not cached (or every file, on a
    ``force_download``) into ``cache`` in the Hub cache layout (``blobs/<sha256>``, ``snapshots/<rev>/<file>`` ->
    blob) and records the download. Offline (``offline = True``) it hands the call to the real huggingface_hub
    function with ``local_files_only=True``, which serves the cache or raises, as with ``HF_HUB_OFFLINE=1``.
    ``served[(repo, file)]`` is what the Hub serves when it differs from the pinned content (a bad upload)."""

    def __init__(self, cache: Path, bundle):
        system = ss.get(ss.DEFAULT_ESM_CODES)
        self.cache = cache
        self.repos = {loaders.ESM_REPO_ID: (loaders.ESM_REVISION, dict(ESM3_FILES)),
                      model.REPO_ID: (model.REVISION, {model.FILE_NAME: AGGREGATOR}),
                      system.bundle_repo: (system.bundle_revision, dict(bundle))}
        self.served = {}
        self.offline = False
        self.calls, self.downloads = [], []

    def all_files(self):
        return sorted((repo, name) for repo, (_rev, files) in self.repos.items() for name in files)

    def pointer(self, repo_id, filename):
        return _repo_dir(self.cache, repo_id) / "snapshots" / self.repos[repo_id][0] / filename

    def write(self, repo_id, filename, data, *, blob_name=None):
        """Put ``data`` in the cache as ``repo_id``'s ``filename`` (``blob_name`` defaults to its sha256; a test plants
        a corrupt file by giving the pinned name to other bytes, as a truncated download does)."""
        blob = _repo_dir(self.cache, repo_id) / "blobs" / (blob_name or _sha(data))
        blob.parent.mkdir(parents=True, exist_ok=True)
        tmp = blob.with_name(blob.name + ".incomplete")
        tmp.write_bytes(data)
        os.replace(tmp, blob)
        link = self.pointer(repo_id, filename)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(blob, link.parent))
        return link

    def _download(self, repo_id, filename):
        self.downloads.append((repo_id, filename))
        return self.write(repo_id, filename, self.served.get((repo_id, filename), self.repos[repo_id][1][filename]))

    def hf_hub_download(self, *, repo_id, filename, revision, cache_dir=None, local_files_only=False,
                        force_download=False, **kwargs):
        self.calls.append(("hf_hub_download", repo_id, filename, revision, force_download))
        assert cache_dir is None and revision == self.repos[repo_id][0], (cache_dir, revision)
        if self.offline or local_files_only:
            return _REAL_HF_HUB_DOWNLOAD(repo_id=repo_id, filename=filename, revision=revision,
                                         cache_dir=str(self.cache), local_files_only=True,
                                         force_download=force_download)
        link = self.pointer(repo_id, filename)
        if link.exists() and not force_download:
            return str(link)
        return str(self._download(repo_id, filename))

    def snapshot_download(self, *, repo_id, revision, cache_dir=None, **kwargs):
        self.calls.append(("snapshot_download", repo_id, revision))
        assert cache_dir is None and revision == self.repos[repo_id][0], (cache_dir, revision)
        if self.offline:
            return _REAL_SNAPSHOT_DOWNLOAD(repo_id=repo_id, revision=revision, cache_dir=str(self.cache),
                                           local_files_only=True)
        for filename in self.repos[repo_id][1]:
            if not self.pointer(repo_id, filename).exists():
                self._download(repo_id, filename)
        return str(_repo_dir(self.cache, repo_id) / "snapshots" / revision)

    def patches(self):
        return [mock.patch.object(loaders, "snapshot_download", self.snapshot_download),
                mock.patch.object(loaders, "hf_hub_download", self.hf_hub_download),
                mock.patch.object(model, "hf_hub_download", self.hf_hub_download)]


def _synthetic_pins():
    return [mock.patch.object(loaders, "ESM_WEIGHTS_SHA256", SYNTHETIC_PINS["esm3"]),
            mock.patch.object(model, "AGGREGATOR_SHA256", SYNTHETIC_PINS["aggregator"])]


@contextmanager
def _no_models_no_weights_no_cuda():
    """Every way download-models could build a model (any ``torch.nn.Module``, ESM3, the aggregator, the model
    getters), load weights (``torch.load``), move anything to a device or call CUDA (every function in
    ``torch.cuda``) raises and is recorded in the yielded list."""
    import torch
    from esm.models.esm3 import ESM3
    from foldmatch.model.residue_embedding_aggregator import ResidueEmbeddingAggregator
    calls = []

    def refuse(name):
        def refused(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"download-models called {name}")
        return refused

    targets = [(torch.nn.Module, "__init__"), (ESM3, "__init__"), (ResidueEmbeddingAggregator, "__init__"),
               (torch.nn.Module, "to"), (torch.nn.Module, "cuda"), (torch.Tensor, "to"), (torch.Tensor, "cuda"),
               (torch, "load"), (model, "get_residue_model"), (model, "get_aggregator_model"),
               (loaders, "esm_open"), (loaders, "structure_encoder"), (loaders, "data_root")]
    cuda = sorted(name for name, obj in vars(torch.cuda).items()
                  if not name.startswith("__") and (inspect.isfunction(obj) or inspect.isbuiltin(obj)))
    assert {"is_available", "device_count", "init", "_lazy_init", "set_device", "mem_get_info"} <= set(cuda), cuda
    targets += [(torch.cuda, name) for name in cuda]
    with ExitStack() as stack:
        for owner, name in targets:
            label = f"{getattr(owner, '__name__', type(owner).__name__)}.{name}"
            stack.enter_context(mock.patch.object(owner, name, refuse(label)))
        yield calls


class _HubCase(unittest.TestCase):
    """A temporary cache served by a fake Hub, with the synthetic ESM3 and aggregator pins."""

    @classmethod
    def setUpClass(cls):
        cls.bundle = _published_bundle_bytes()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        loaders._hashed.clear()
        self.addCleanup(loaders._hashed.clear)

    def hub(self, name="hub") -> _FakeHub:
        cache = Path(self._tmp.name) / name
        cache.mkdir()
        hub = _FakeHub(cache, self.bundle)
        stack = ExitStack()
        for patch in hub.patches() + _synthetic_pins():
            stack.enter_context(patch)
        self.addCleanup(stack.close)
        return hub

    def invoke(self, label, *args):
        return CliRunner().invoke(_cli(label).app, ["download-models", *args])

    def assertOk(self, r):
        self.assertEqual(r.exit_code, 0, f"{r.output}\n{r.exception!r}")


# =====================================================================================================================
# download only: the three CLIs, no model, no weights loaded, no CUDA; a second run is a cache hit
# =====================================================================================================================
class DownloadOnlyTest(_HubCase):

    def test_downloads_and_verifies_without_models_or_cuda_then_hits_the_cache(self):
        system = ss.get(ss.DEFAULT_ESM_CODES)
        for label in CLIS:
            with self.subTest(cli=label):
                loaders._hashed.clear()
                hub = self.hub(label)
                with _no_models_no_weights_no_cuda() as refused:
                    first = self.invoke(label)
                    self.assertOk(first)
                    self.assertEqual(sorted(hub.downloads), hub.all_files())     # the first run downloads all
                    after_first = _listing(hub.cache)
                    hub.downloads.clear()
                    second = self.invoke(label)                                  # online again: a cache hit
                    self.assertOk(second)
                    hub.offline = True                                           # and offline (real hf, cache only)
                    third = self.invoke(label)
                    self.assertOk(third)
                self.assertEqual(refused, [])
                self.assertEqual(hub.downloads, [])
                self.assertEqual(_listing(hub.cache), after_first)              # no new or rewritten file
                lines = first.output.strip().splitlines()
                self.assertEqual(len(lines), 3, first.output)
                self.assertTrue(lines[0].startswith(
                    f"ESM3: {loaders.ESM_REPO_ID}@{loaders.ESM_REVISION}, weights sha256 verified, in "), lines[0])
                self.assertTrue(lines[1].startswith(
                    f"aggregator: {model.REPO_ID}@{model.REVISION}, sha256 verified, in "), lines[1])
                self.assertTrue(lines[2].startswith(
                    f"esm-codes bundle {system.name}: {system.bundle_repo}@{system.bundle_revision}, "
                    f"{len(system.bundle_files)} files, sha256 verified, in "), lines[2])
                self.assertEqual(second.output, first.output)
                self.assertEqual(third.output, first.output)

    def test_every_file_is_hashed(self):
        """The full sha256 of every file is checked (not only the blob names): ESM3 once per process."""
        hub = self.hub()
        spy_esm = mock.Mock(wraps=loaders._file_sha256)
        spy_model = mock.Mock(wraps=model._file_sha256)
        with mock.patch.object(loaders, "_file_sha256", spy_esm), mock.patch.object(model, "_file_sha256", spy_model):
            self.assertOk(self.invoke("fm-search"))
            self.assertOk(self.invoke("fm-search"))
        hashed = {Path(os.path.realpath(c.args[0])).name for c in spy_model.call_args_list}
        self.assertEqual(spy_esm.call_count, 1)
        self.assertIn(SYNTHETIC_PINS["aggregator"], hashed)
        self.assertTrue({sha for _n, sha in ss.get(ss.DEFAULT_ESM_CODES).bundle_files} <= hashed)

    def test_unknown_system_is_refused_before_any_download(self):
        for label in CLIS:
            with self.subTest(cli=label):
                hub = self.hub(label)
                with _no_models_no_weights_no_cuda() as refused:
                    r = self.invoke(label, "--system", ss.DEFAULT_ESM_CODES, "--system", "nope")
                self.assertEqual(r.exit_code, 2, r.output)
                text = " ".join(r.output.replace("\u2502", " ").split())     # rich wraps the message in a box
                self.assertIn("unknown esm-codes scoring system 'nope'; this FoldMatch knows: "
                              f"{', '.join(ss.known_names())}", text)
                self.assertEqual((hub.calls, refused, _listing(hub.cache)), ([], [], []))

    def test_system_option_fetches_that_bundle(self):
        hub = self.hub()
        r = self.invoke("fm-search", "--system", ss.DEFAULT_ESM_CODES)
        self.assertOk(r)
        system = ss.get(ss.DEFAULT_ESM_CODES)
        self.assertEqual(sorted(f for repo, f in hub.downloads if repo == system.bundle_repo),
                         sorted(n for n, _ in system.bundle_files))


# =====================================================================================================================
# corrupt, missing and mismatching cached files
# =====================================================================================================================
class CacheProblemsTest(_HubCase):
    """A cached file whose content differs from the pin is downloaded once more; a Hub file that differs, a failed
    re-download and a file that is neither cached nor downloadable are refused with one clear error."""

    def plant(self, hub, *, esm3=None, aggregator=None):
        """Cache every file, with ``esm3`` / ``aggregator`` as the weights' content under the pinned blob name."""
        for repo, name in hub.all_files():
            hub.write(repo, name, hub.repos[repo][1][name])
        if esm3 is not None:
            hub.write(loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_FILE, esm3, blob_name=SYNTHETIC_PINS["esm3"])
        if aggregator is not None:
            hub.write(model.REPO_ID, model.FILE_NAME, aggregator, blob_name=SYNTHETIC_PINS["aggregator"])

    def test_corrupt_cached_files_are_downloaded_again(self):
        hub = self.hub()
        self.plant(hub, esm3=b"truncated", aggregator=b"truncated")
        snapshot = loaders.download_esm3_snapshot()
        aggregator = model.download_aggregator_weights()
        self.assertEqual(hub.downloads, [(loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_FILE),
                                         (model.REPO_ID, model.FILE_NAME)])
        self.assertEqual((snapshot / loaders.ESM_WEIGHTS_FILE).read_bytes(), ESM3_FILES[loaders.ESM_WEIGHTS_FILE])
        self.assertEqual(aggregator.read_bytes(), AGGREGATOR)
        forced = [c for c in hub.calls if c[0] == "hf_hub_download" and c[-1]]
        self.assertEqual(len(forced), 2)

    def test_hub_file_that_differs_from_the_pin_is_refused(self):
        hub = self.hub()
        self.plant(hub, esm3=b"truncated", aggregator=b"truncated")
        hub.served = {(loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_FILE): b"another upload",
                      (model.REPO_ID, model.FILE_NAME): b"another upload"}
        with self.assertRaisesRegex(loaders.EsmWeightsMismatchError, "after a fresh download"):
            loaders.download_esm3_snapshot()
        with self.assertRaisesRegex(model.AggregatorWeightsMismatchError, "after a fresh download"):
            model.download_aggregator_weights()

    def test_offline_corrupt_file_is_refused_with_the_reason(self):
        hub = self.hub()
        self.plant(hub, esm3=b"truncated", aggregator=b"truncated")
        hub.offline = True
        with self.assertRaisesRegex(loaders.EsmWeightsMismatchError, r"is corrupt.*Downloading it again failed"):
            loaders.download_esm3_snapshot()
        with self.assertRaisesRegex(model.AggregatorWeightsMismatchError,
                                    r"cached copy is corrupt.*Downloading it again failed"):
            model.download_aggregator_weights()
        self.assertEqual(hub.downloads, [])

    def test_offline_and_not_cached(self):
        hub = self.hub()
        hub.offline = True
        with self.assertRaisesRegex(FileNotFoundError, rf"ESM3 snapshot {loaders.ESM_REPO_ID}@{loaders.ESM_REVISION} "
                                                       "is not in the Hugging Face cache"):
            loaders.download_esm3_snapshot()
        with self.assertRaisesRegex(FileNotFoundError, rf"aggregator weights {model.FILE_NAME} \({model.REPO_ID}@"
                                                       rf"{model.REVISION}\) are not in the Hugging Face cache"):
            model.download_aggregator_weights()
        self.assertEqual(_listing(hub.cache), [])

    def test_weights_missing_from_a_cached_snapshot(self):
        hub = self.hub()
        hub.write(loaders.ESM_REPO_ID, "README.md", ESM3_FILES["README.md"])
        hub.offline = True
        with self.assertRaisesRegex(FileNotFoundError, "missing from the cached snapshot"):
            loaders.download_esm3_snapshot()

    def test_the_clis_report_a_refusal_as_one_line(self):
        hub = self.hub()
        self.plant(hub, esm3=b"truncated")
        hub.offline = True
        for label in CLIS:
            with self.subTest(cli=label), mock.patch.object(sys, "argv", [label, "download-models"]):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
                    _cli(label).main()
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("Error: ESM3 weights", stderr.getvalue())
                self.assertIn("Downloading it again failed", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())


# =====================================================================================================================
# real huggingface_hub in a subprocess: HF_HUB_OFFLINE=1, an unreachable endpoint, torch cannot be imported
# =====================================================================================================================
_SCRIPT = textwrap.dedent("""
    import importlib, json, sys
    from pathlib import Path
    sys.modules["torch"] = None                       # any torch import fails
    cli, pins, stub = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3] == "stub"
    from foldmatch.utils import model
    from foldmatch.utils.esm import loaders
    if pins:                                          # the test's synthetic ESM3 and aggregator files
        loaders.ESM_WEIGHTS_SHA256, model.AGGREGATOR_SHA256 = pins["esm3"], pins["aggregator"]
    if stub:                                          # the bundle download test does not fetch 3.5 GB of weights
        loaders.download_esm3_snapshot = lambda **kw: Path("/stub/esm3")
        model.download_aggregator_weights = lambda **kw: Path("/stub/aggregator/weights.pt")
    sys.argv = [cli, "download-models", *sys.argv[4:]]
    code = "returned"
    try:
        importlib.import_module("foldmatch.cli." + cli).main()
    except SystemExit as e:
        code = e.code
    print(json.dumps({"exit": code, "esm": "esm" in sys.modules, "lightning": "lightning" in sys.modules}))
""")


def _env(cache, *, offline=True):
    env = dict(os.environ)
    src = str(Path(foldmatch.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env.update(HF_HUB_CACHE=str(cache), HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_ETAG_TIMEOUT="10",
               HF_HUB_DOWNLOAD_TIMEOUT="30")
    env.pop("HF_HUB_OFFLINE", None)
    if offline:
        env.update(HF_HUB_OFFLINE="1", HF_ENDPOINT="http://127.0.0.1:9")       # any network use fails
    return env


def _run(label, cache, *args, pins=None, stub=False, offline=True):
    """``label download-models args`` in a fresh interpreter without torch; returns (exit code, stdout lines
    before the result, the result dict, stderr)."""
    r = subprocess.run([sys.executable, "-c", _SCRIPT, CLIS[label], json.dumps(pins or {}),
                        "stub" if stub else "-", *args],
                       env=_env(cache, offline=offline), capture_output=True, text=True, timeout=600)
    lines = r.stdout.strip().splitlines()
    if not lines:
        raise AssertionError(f"no output (exit {r.returncode}): {r.stderr[-3000:]}")
    result = json.loads(lines[-1])
    return result, lines[:-1], r.stderr


class OfflineSubprocessTest(_HubCase):

    def test_offline_after_the_download_without_torch(self):
        hub = self.hub()
        self.assertOk(self.invoke("fm-search"))                 # the download (fake Hub)
        before = _listing(hub.cache)
        for label in CLIS:
            with self.subTest(cli=label):
                result, lines, stderr = _run(label, hub.cache, pins=SYNTHETIC_PINS)
                self.assertEqual(result["exit"], 0, stderr[-3000:])
                self.assertFalse(result["esm"] or result["lightning"], result)
                self.assertEqual([line.split(",")[0] for line in lines],
                                 [f"ESM3: {loaders.ESM_REPO_ID}@{loaders.ESM_REVISION}",
                                  f"aggregator: {model.REPO_ID}@{model.REVISION}",
                                  f"esm-codes bundle {ss.DEFAULT_ESM_CODES}: "
                                  f"{ss.get(ss.DEFAULT_ESM_CODES).bundle_repo}@"
                                  f"{ss.get(ss.DEFAULT_ESM_CODES).bundle_revision}"])
                self.assertEqual(_listing(hub.cache), before)

    def test_offline_before_any_download_is_a_one_line_error(self):
        cache = Path(self._tmp.name) / "empty"
        cache.mkdir()
        result, lines, stderr = _run("fm-search", cache, pins=SYNTHETIC_PINS)
        self.assertEqual(result["exit"], 1, stderr[-3000:])
        self.assertIn(f"Error: ESM3 snapshot {loaders.ESM_REPO_ID}@{loaders.ESM_REVISION} is not in the Hugging Face "
                      "cache", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(_listing(cache), [])


# =====================================================================================================================
# the same command on the three CLIs
# =====================================================================================================================
class SameCommandTest(unittest.TestCase):

    def test_same_help_and_option(self):
        helps = {}
        for label in CLIS:
            command = typer.main.get_command(_cli(label).app).commands["download-models"]
            with self.subTest(cli=label):
                self.assertEqual(command.help, args_utils.DOWNLOAD_MODELS_HELP)
                [param] = command.params
                self.assertEqual((param.name, param.opts, param.multiple, param.default, param.help),
                                 ("system", ["--system"], True, None, args_utils.DOWNLOAD_MODELS_SYSTEM_HELP))
                r = CliRunner().invoke(_cli(label).app, ["download-models", "--help"])
                self.assertEqual(r.exit_code, 0, r.output)
                helps[label] = r.output
        self.assertEqual(len(set(helps.values())), 1, helps)
        self.assertIn("Download only", helps["fm-search"])
        self.assertIn("--system", helps["fm-search"])
        top = CliRunner().invoke(_cli("fm-search").app, ["--help"])
        self.assertIn("download-models", top.output)

    def test_every_cli_calls_the_shared_function(self):
        for label in CLIS:
            self.assertIs(_cli(label).download_models_command, args_utils.download_models_command)
            with self.subTest(cli=label), mock.patch.object(_cli(label), "download_models_command") as shared:
                r = CliRunner().invoke(_cli(label).app, ["download-models", "--system", "a", "--system", "b"])
                self.assertEqual(r.exit_code, 0, r.output)
                shared.assert_called_once_with(["a", "b"])

    def test_download_path_imports_without_torch(self):
        code = ("import sys; sys.modules['torch'] = None\n"
                "import foldmatch.utils.esm.loaders, foldmatch.utils.model, foldmatch.cli.args_utils\n"
                "import foldmatch.cli.search, foldmatch.cli.embedding, foldmatch.cli.inference\n"
                "print(sorted(m for m in ('esm', 'biotite', 'lightning') if m in sys.modules))\n")
        r = subprocess.run([sys.executable, "-c", code], env=_env(Path(tempfile.gettempdir())), capture_output=True,
                           text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(r.stdout.strip(), "[]")


# =====================================================================================================================
# the real files
# =====================================================================================================================
class RealCacheTest(unittest.TestCase):
    """The command on the real pinned files in this machine's Hugging Face cache, offline, without torch, with an
    unreachable endpoint: the pins (ESM3, aggregator, bundle) match the files, and nothing is written."""

    def test_real_files_offline(self):
        system = ss.get(ss.DEFAULT_ESM_CODES)
        _published_bundle_bytes()                                # skipped only offline and not cached
        missing = [f"{repo}/{name}" for repo, name, rev in (
            (loaders.ESM_REPO_ID, loaders.ESM_WEIGHTS_FILE, loaders.ESM_REVISION),
            (model.REPO_ID, model.FILE_NAME, model.REVISION))
            if not isinstance(huggingface_hub.try_to_load_from_cache(repo, name, revision=rev), str)]
        if missing:
            self.skipTest(f"not in the Hugging Face cache {hf_constants.HF_HUB_CACHE}: {missing} (run download-models)")
        cache = Path(hf_constants.HF_HUB_CACHE)
        repos = [_repo_dir(cache, r) for r in (loaders.ESM_REPO_ID, model.REPO_ID, system.bundle_repo)]
        before = [_listing(r) for r in repos]
        result, lines, stderr = _run("fm-search", cache)
        self.assertEqual(result["exit"], 0, stderr[-3000:])
        self.assertFalse(result["esm"] or result["lightning"], result)
        self.assertEqual(len(lines), 3, lines)
        self.assertIn(f"{loaders.ESM_REPO_ID}@{loaders.ESM_REVISION}, weights sha256 verified", lines[0])
        self.assertIn(f"{model.REPO_ID}@{model.REVISION}, sha256 verified", lines[1])
        self.assertIn(f"{system.bundle_repo}@{system.bundle_revision}, 5 files, sha256 verified", lines[2])
        self.assertEqual([_listing(r) for r in repos], before)


class RealBundleDownloadTest(unittest.TestCase):
    """``fm-search download-models`` downloads the published bundle into an empty cache (ESM3 and the aggregator
    stubbed: 3.5 GB), then serves it offline with no new file. Needs the Hub: skipped with HF_HUB_OFFLINE=1."""

    def test_fresh_download_then_offline(self):
        if hf_constants.HF_HUB_OFFLINE:
            self.skipTest("needs the Hugging Face Hub (HF_HUB_OFFLINE=1)")
        system = ss.get(ss.DEFAULT_ESM_CODES)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "hub"
            cache.mkdir()
            result, lines, stderr = _run("fm-search", cache, "--system", system.name, stub=True, offline=False)
            self.assertEqual(result["exit"], 0, stderr[-3000:])
            snapshot = _repo_dir(cache, system.bundle_repo) / "snapshots" / system.bundle_revision
            self.assertEqual(lines[-1], f"esm-codes bundle {system.name}: {system.bundle_repo}@"
                                        f"{system.bundle_revision}, 5 files, sha256 verified, in {snapshot}")
            for name, sha in system.bundle_files:
                self.assertEqual(model._file_sha256(snapshot / name), sha, name)
            before = _listing(cache)
            result, lines2, stderr = _run("fm-search", cache, "--system", system.name, stub=True, offline=True)
            self.assertEqual(result["exit"], 0, stderr[-3000:])
            self.assertEqual(lines2, lines)
            self.assertEqual(_listing(cache), before)


if __name__ == "__main__":
    unittest.main()
