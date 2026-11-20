"""NetworKit ParallelLeiden as the optional Leiden backend (P9.CL D5): ``cluster_leiden(backend="networkit")``,
``cluster_greedy_leiden(backend="networkit")`` and ``fm-search cluster --leiden-backend networkit --leiden-threads``.

Without NetworKit, a fake ``networkit`` module in ``sys.modules`` checks the calls: the thread count set before
GraphFromCsr, NetworKit's and FAISS's thread counts saved and restored (also on an error in run()), the seed, the
Partition held in a variable when getQuality gets it, the renumbering, the version check from the package metadata
(no import), and every refusal; the CSR builder is checked against a sorted reference on graphs with rescued edges.

With NetworKit 11.3 importable (its folder on PYTHONPATH), the real library: one thread and one seed give the same file
in two fresh processes at the default FAISS thread count; the stored modularity is igraph's of the same partition; the
partition agrees with igraph's within igraph's seed band on two families; and, with torch and FAISS loaded and the
default OMP_NUM_THREADS of the platform (FoldMatch pins 1 on macOS), FAISS searches the same before and after
ParallelLeiden."""
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

# As the commands run (set_log_level): Lightning first, which brings torch and, through scikit-learn,
# KMP_DUPLICATE_LIB_OK; on macOS, FAISS's OpenMP regions abort a process that loaded torch's libomp without it.
import lightning  # noqa: F401
import faiss
import numpy as np
import typer
from typer.testing import CliRunner

import foldmatch
from foldmatch.cli import search
from foldmatch.errors import RefusalError
from foldmatch.search import clustering
from foldmatch.search.clustering import EmbeddingClusterer, SimilarityEdges
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.types.api_types import ClusterMethod, EdgeRule, IndexConfig, IndexType, LeidenBackend


def _real_networkit() -> bool:
    """NetworKit 11.3 importable, without importing it."""
    if importlib.util.find_spec("networkit") is None:
        return False
    try:
        return importlib.metadata.version("networkit") == "11.3"
    except importlib.metadata.PackageNotFoundError:
        return False


_REAL = _real_networkit()
# The Leiden package FoldMatch no longer uses, as a backend name that is refused; spelled in two parts, as
# tests/test_clustering.py does, so that its name is nowhere in the repository
_REMOVED_PACKAGE = "leiden" + "alg"


def _clustered_vectors(seed: int, n: int, d: int, n_centers: int, noise: float) -> np.ndarray:
    """Unit vectors around ``n_centers`` random centres (tests/test_clustering.py's fixture)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_centers, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    members = np.sort(rng.integers(0, n_centers, n))
    levels = noise * rng.uniform(0.5, 1.5, (n, 1))
    vectors = centers[members] + levels * rng.standard_normal((n, d)) / math.sqrt(d)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.astype(np.float32)


def _planted_lost_chains(n_groups: int = 3, n_core: int = 8, n_satellites: int = 3) -> np.ndarray:
    """tests/test_clustering.py's planted lost chains: groups of core chains, then satellites whose own hits at 0.8
    are all lower chains (rescued at k = 8), then one chain that scores 0 with all."""
    d = 64
    vectors = []
    for g in range(n_groups):
        for t in range(n_core):
            v = np.zeros(d)
            v[16 * g] = 1.0
            v[16 * g + 1 + t] = 0.05 + 0.01 * t
            vectors.append(v)
    for g in range(n_groups):
        for u in range(n_satellites):
            v = np.zeros(d)
            v[16 * g] = 1.0
            v[16 * g + 1 + n_core + u] = 0.6
            vectors.append(v)
    lone = np.zeros(d)
    lone[d - 1] = 1.0
    vectors.append(lone)
    vectors = np.array(vectors)
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


def _two_families(seed: int, d: int = 32) -> np.ndarray:
    """Two orthogonal families of 6 sub-centres each, each sub-centre with 4 to 9 near-duplicates, shuffled
    (test_clustering_greedy_leiden's fixture)."""
    rng = np.random.default_rng(seed)
    vectors = []
    for f in range(2):
        axis = np.zeros(d)
        axis[f] = 1.0
        for _ in range(6):
            sub = axis + 0.42 * rng.standard_normal(d) * (np.arange(d) >= 2) / math.sqrt(d - 2)
            sub /= np.linalg.norm(sub)
            for _ in range(int(rng.integers(4, 10))):
                v = sub + 0.08 * rng.standard_normal(d) / math.sqrt(d)
                vectors.append(v / np.linalg.norm(v))
    return np.asarray(vectors, dtype=np.float32)[rng.permutation(len(vectors))]


def _build(folder: Path, name: str, vectors: np.ndarray, index_type: IndexType = IndexType.flat,
           config: IndexConfig = None) -> str:
    db = FaissEmbeddingDatabase(folder, name)
    db.create_database(iter([([f"chain{i}" for i in range(len(vectors))], vectors)]), index_type=index_type,
                       index_config=config)
    return str(folder / name)


def _invoke(*args):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        return CliRunner().invoke(search.app, [str(a) for a in args])
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def _flat(text: str) -> str:
    """Text on one line, typer's error panel borders and colour codes dropped."""
    import re
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


def _reference_csr(edges: SimilarityEdges):
    """The CSR by a stable sort of the whole edge stream on the lower end."""
    pairs = np.concatenate([np.asarray(b, dtype=np.int64) for b in edges.pair_blocks]) if edges.n_edges else \
        np.empty((0, 2), dtype=np.int64)
    weights = np.concatenate([np.asarray(b) for b in edges.weight_blocks]) if edges.n_edges else np.empty(0)
    order = np.argsort(pairs[:, 0], kind="stable")
    indptr = np.zeros(edges.n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pairs[:, 0], minlength=edges.n), out=indptr[1:])
    return indptr, pairs[order, 1], weights[order].astype(np.float64)


def _reference_numbering(raw) -> np.ndarray:
    """Cluster ids by decreasing size, ties by the smallest chain index, from any ids."""
    raw = np.asarray(raw)
    groups = {}
    for i, c in enumerate(raw.tolist()):
        groups.setdefault(c, []).append(i)
    ordered = sorted(groups.values(), key=lambda members: (-len(members), members[0]))
    out = np.empty(len(raw), dtype=np.int64)
    for new, members in enumerate(ordered):
        out[members] = new
    return out


class _FakeNetworkit:
    """A fake ``networkit`` package: ``engineering`` (thread count and seed), ``graph.GraphFromCsr`` and
    ``community.ParallelLeiden`` / ``community.Modularity``, recording every call in ``self.calls``. The partition is
    the connected components of the graph, with ids ``7 + 3 * component`` (not 0..C-1); ``run_error`` makes run()
    raise."""

    def __init__(self, max_threads: int = 6, run_error: Exception = None):
        self.calls = []
        self.threads = max_threads
        self.run_error = run_error
        self.partitions = []
        self.quality_args = []
        fake = self

        class Graph:
            def __init__(self, csr):
                self.csr = csr

        class Partition:
            def __init__(self, vector):
                self.vector = vector

            def getVector(self):
                fake.calls.append(("getVector",))
                return list(self.vector)

        class ParallelLeiden:
            def __init__(self, graph, iterations=3, randomize=True, gamma=1.0):
                fake.calls.append(("ParallelLeiden", iterations, randomize, gamma, fake.threads))
                self.graph = graph
                self.vector = None

            def run(self):
                fake.calls.append(("run", fake.threads))
                if fake.run_error is not None:
                    raise fake.run_error
                csr = self.graph.csr
                n = csr.shape[0]
                parent = list(range(n))

                def root(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i

                for i in range(n):
                    for j in csr.indices[csr.indptr[i]:csr.indptr[i + 1]].tolist():
                        parent[root(i)] = root(j)
                roots = [root(i) for i in range(n)]
                self.vector = [7 + 3 * r for r in roots]

            def getPartition(self):
                fake.calls.append(("getPartition",))
                partition = Partition(self.vector)
                fake.partitions.append(partition)
                return partition

        class Modularity:
            def getQuality(self, partition, graph):
                # Whether the caller holds the partition in a variable of its own (the segfault of networkit 11.3
                # comes with a temporary)
                caller = sys._getframe(1).f_locals
                held = any(value is partition for value in caller.values())
                fake.calls.append(("getQuality",))
                fake.quality_args.append((partition, held))
                return 0.25

        def GraphFromCsr(csr, directed=True, weighted=False):
            fake.calls.append(("GraphFromCsr", directed, weighted, fake.threads))
            fake.csr = csr
            return Graph(csr)

        def setNumberOfThreads(n):
            fake.calls.append(("setNumberOfThreads", n))
            fake.threads = n

        def getMaxNumberOfThreads():
            return fake.threads

        def setSeed(seed, use_thread_id):
            fake.calls.append(("setSeed", seed, use_thread_id, fake.threads))

        self.module = types.ModuleType("networkit")
        self.module.engineering = types.SimpleNamespace(setNumberOfThreads=setNumberOfThreads,
                                                        getMaxNumberOfThreads=getMaxNumberOfThreads, setSeed=setSeed)
        self.module.graph = types.SimpleNamespace(GraphFromCsr=GraphFromCsr)
        self.module.community = types.SimpleNamespace(ParallelLeiden=ParallelLeiden, Modularity=Modularity)

    def names(self):
        return [c[0] for c in self.calls]


class _FakeFaissThreads:
    """faiss.omp_get_max_threads / omp_set_num_threads stand-ins (FAISS's real count is left alone)."""

    def __init__(self, threads: int = 3):
        self.threads = threads
        self.sets = []

    def get(self):
        return self.threads

    def set(self, n):
        self.sets.append(n)
        self.threads = n


class _FakeCase(unittest.TestCase):
    """A flat database with planted lost chains (rescued edges) and the fake NetworKit, version 11.3."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.vectors = _planted_lost_chains()
        cls.n = len(cls.vectors)
        cls.db_path = _build(cls.folder, "planted", cls.vectors)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fake(self, platform: str = "linux", **kwargs):
        """Run the test body with the fake module, its metadata version 11.3, and fake FAISS thread calls, the library
        on ``platform`` (Linux by default: on macOS more than one NetworKit thread is refused)."""
        nk = _FakeNetworkit(**kwargs)
        threads = _FakeFaissThreads()
        stack = [mock.patch.dict(sys.modules, {"networkit": nk.module}),
                 mock.patch.object(clustering.importlib.metadata, "version", side_effect=self.version),
                 mock.patch.object(clustering.faiss, "omp_get_max_threads", threads.get),
                 mock.patch.object(clustering.faiss, "omp_set_num_threads", threads.set),
                 mock.patch.object(clustering, "sys", types.SimpleNamespace(platform=platform))]
        for patch in stack:
            patch.start()
            self.addCleanup(patch.stop)
        return nk, threads

    @staticmethod
    def version(name, _real=importlib.metadata.version):
        return "11.3" if name == "networkit" else _real(name)

    def edges(self, rule=EdgeRule.rescue, threshold=0.8, k=8):
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.build_similarity_edges(threshold=threshold, max_neighbors=k, edge_rule=rule)
        return clusterer


class FakeBackendTest(_FakeCase):

    def test_fixture(self):
        clusterer = self.edges()
        self.assertGreater(clusterer.edges.n_rescued_edges, 0)

    def test_calls_in_order(self):
        """Threads set before GraphFromCsr; the seed after; ParallelLeiden's arguments; the partition held when
        getQuality gets it; both thread counts restored."""
        nk, threads = self.fake()
        clusterer = self.edges()
        clusterer.cluster_leiden(resolution=2.5, n_iterations=4, seed=9, backend="networkit", threads=2)
        self.assertEqual(nk.calls, [
            ("setNumberOfThreads", 2), ("GraphFromCsr", False, True, 2), ("setSeed", 9, True, 2),
            ("ParallelLeiden", 4, True, 2.5, 2), ("run", 2), ("getPartition",), ("getQuality",), ("getVector",),
            ("setNumberOfThreads", 6)])
        self.assertEqual(threads.sets, [3])
        self.assertEqual(len(nk.partitions), 1)
        self.assertEqual(nk.quality_args, [(nk.partitions[0], True)])
        self.assertEqual(clusterer.modularity, 0.25)
        self.assertEqual(clusterer.method, "leiden")
        self.assertIsNone(clusterer.graph)
        # The edges are kept for another run
        self.assertIsNotNone(clusterer.edges)

    def test_default_threads_are_networkits(self):
        nk, threads = self.fake(max_threads=5)
        clusterer = self.edges()
        with self.assertLogs(level="INFO") as logs:
            clusterer.cluster_leiden(seed=None, backend=LeidenBackend.networkit)
        self.assertEqual([c for c in nk.calls if c[0] == "setNumberOfThreads"],
                         [("setNumberOfThreads", 5), ("setNumberOfThreads", 5)])
        self.assertEqual(nk.calls[1], ("GraphFromCsr", False, True, 5))
        self.assertIn("INFO:root:Running NetworKit ParallelLeiden on 5 threads (resolution=1.0, at most 10 "
                      "iterations)...", logs.output)
        self.assertEqual(threads.sets, [3])

    def test_macos_one_thread(self):
        """On macOS: 1 thread by default (not NetworKit's maximum), and more refused before NetworKit is used."""
        nk, threads = self.fake(platform="darwin", max_threads=5)
        clusterer = self.edges()
        with self.assertLogs(level="INFO") as logs:
            clusterer.cluster_leiden(seed=4, backend="networkit")
        self.assertEqual([c for c in nk.calls if c[0] == "setNumberOfThreads"],
                         [("setNumberOfThreads", 1), ("setNumberOfThreads", 5)])
        self.assertEqual(nk.calls[1], ("GraphFromCsr", False, True, 1))
        self.assertIn("INFO:root:Running NetworKit ParallelLeiden on 1 threads (resolution=1.0, at most 10 "
                      "iterations)...", logs.output)
        self.assertFalse(any(line.startswith("WARNING") for line in logs.output))
        nk.calls.clear()
        for count in (2, 5):
            with self.subTest(threads=count):
                with self.assertRaises(ValueError) as caught:
                    clusterer.cluster_leiden(seed=4, backend="networkit", threads=count)
                self.assertEqual(str(caught.exception), clustering._NETWORKIT_DARWIN_REFUSAL)
                with self.assertRaises(ValueError):
                    clusterer.cluster_greedy_leiden(self.folder / "darwin-work", backend="networkit", threads=count)
        self.assertEqual((nk.calls, threads.sets), ([], [3]))
        self.assertFalse((self.folder / "darwin-work").exists())
        self.assertEqual(clustering._NETWORKIT_DARWIN_REFUSAL, (
            "--leiden-threads above 1 is refused on macOS: there NetworKit's own OpenMP runtime, a third one next to "
            "torch's and FAISS's, crashes the process on more than one thread. Use --leiden-threads 1."))
        clusterer.cluster_leiden(seed=4, backend="networkit", threads=1)
        self.assertIn(("ParallelLeiden", 10, True, 1.0, 1), nk.calls)

    def test_restored_on_an_error_in_run(self):
        nk, threads = self.fake(max_threads=6, run_error=RuntimeError("injected fault"))
        clusterer = self.edges()
        with self.assertRaisesRegex(RuntimeError, "injected fault"):
            clusterer.cluster_leiden(seed=1, backend="networkit", threads=1)
        self.assertEqual(nk.names()[-2:], ["run", "setNumberOfThreads"])
        self.assertEqual(nk.calls[-1], ("setNumberOfThreads", 6))
        self.assertEqual(threads.sets, [3])
        self.assertIsNone(clusterer.clusters)

    def test_seed(self):
        nk, _ = self.fake()
        clusterer = self.edges()
        clusterer.cluster_leiden(seed=2 ** 64 - 1, backend="networkit", threads=1)
        self.assertIn(("setSeed", 2 ** 64 - 1, True, 1), nk.calls)
        # Without a seed, one drawn from os.urandom (8 bytes, little-endian)
        nk.calls.clear()
        with mock.patch.object(clustering.os, "urandom", return_value=bytes(range(1, 9))) as urandom:
            clusterer.cluster_leiden(backend="networkit", threads=1)
        urandom.assert_called_once_with(8)
        self.assertIn(("setSeed", int.from_bytes(bytes(range(1, 9)), "little"), True, 1), nk.calls)

    def test_seed_warning_above_one_thread(self):
        warning = ("WARNING:root:--seed with --leiden-backend networkit on 3 threads: ParallelLeiden does not give "
                   "the same clusters for the same seed on more than one thread; use --leiden-threads 1 for that.")
        for seed, threads, warned in ((4, 3, True), (4, 1, False), (None, 3, False), (0, None, True)):
            with self.subTest(seed=seed, threads=threads):
                self.fake(max_threads=3)
                clusterer = self.edges()
                with self.assertLogs(level="INFO") as logs:
                    clusterer.cluster_leiden(seed=seed, backend="networkit", threads=threads)
                self.assertEqual(warning in logs.output, warned)
                self.assertEqual(sum(line.startswith("WARNING") for line in logs.output), int(warned))

    def test_renumbering(self):
        """The fake's component ids (7 + 3 c) become 0..C-1 by decreasing size, ties by the smallest chain index."""
        nk, _ = self.fake()
        clusterer = self.edges()
        clusters = clusterer.cluster_leiden(seed=1, backend="networkit", threads=1)
        raw = nk.partitions[0].vector
        self.assertGreater(min(raw), 0)
        np.testing.assert_array_equal(clusters, _reference_numbering(raw))
        self.assertEqual(clusters.dtype, np.int32)
        sizes = np.bincount(clusters)
        self.assertTrue(np.all(np.diff(sizes) <= 0))
        self.assertGreater(len(sizes), 1)

    def test_the_csr(self):
        """GraphFromCsr gets the reference CSR of the rescue graph (rescued pairs (j, i) out of row order), int64
        indices and float64 data, in a plain object; at chunks of 1, 7 and the default."""
        clusterer = self.edges()
        reference = _reference_csr(clusterer.edges)
        lower = np.concatenate([np.asarray(b[:, 0]) for b in clusterer.edges.pair_blocks])
        self.assertTrue(np.any(np.diff(lower) < 0), "rescued pairs out of row order")
        for chunk in (1, 7, None):
            with self.subTest(chunk=chunk):
                nk, _ = self.fake()
                with mock.patch.object(clustering, "_GRAPH_EDGE_CHUNK", chunk or clustering._GRAPH_EDGE_CHUNK):
                    clusterer.cluster_leiden(seed=1, backend="networkit", threads=1)
                csr = nk.csr
                self.assertIsInstance(csr, types.SimpleNamespace)
                self.assertEqual(csr.shape, (self.n, self.n))
                self.assertEqual((csr.indptr.dtype, csr.indices.dtype, csr.data.dtype),
                                 (np.dtype(np.int64), np.dtype(np.int64), np.dtype(np.float64)))
                for got, want in zip((csr.indptr, csr.indices, csr.data), reference):
                    np.testing.assert_array_equal(got, want)
                # Upper-triangular: every pair once, lower end first
                rows = np.repeat(np.arange(self.n), np.diff(csr.indptr))
                self.assertTrue(np.all(rows < csr.indices))

    def test_csr_on_random_blocks(self):
        """Random pairs (lower end first, rows in any order) in several blocks, memory-mapped too."""
        rng = np.random.default_rng(3)
        n = 50
        blocks, weights = [], []
        for size in (0, 13, 1, 40, 7):
            i = rng.integers(0, n - 1, size)
            j = i + 1 + rng.integers(0, n - 1 - i)
            blocks.append(np.stack([i, j], axis=1).astype(np.int32))
            weights.append(rng.random(size).astype(np.float32))
        edges = SimilarityEdges(n, [b for b in blocks if len(b)], [w for w in weights if len(w)])
        for chunk in (1, 5, 1 << 20):
            with self.subTest(chunk=chunk), mock.patch.object(clustering, "_GRAPH_EDGE_CHUNK", chunk):
                csr = clustering._networkit_csr(edges)
                for got, want in zip((csr.indptr, csr.indices, csr.data), _reference_csr(edges)):
                    np.testing.assert_array_equal(got, want)

    def test_no_edge(self):
        """A graph with no edge is not passed to NetworKit: every chain its own cluster, NaN modularity (igraph's)."""
        nk, threads = self.fake()
        clusterer = self.edges(threshold=2.0)
        self.assertEqual(clusterer.edges.n_edges, 0)
        clusters = clusterer.cluster_leiden(seed=1, backend="networkit")
        np.testing.assert_array_equal(clusters, np.arange(self.n))
        self.assertTrue(math.isnan(clusterer.modularity))
        self.assertEqual((nk.calls, threads.sets), ([], []))

    def test_statistics_read_the_stored_modularity(self):
        self.fake()
        clusterer = self.edges()
        clusterer.cluster_leiden(seed=1, backend="networkit", threads=1)
        stats = clusterer.get_cluster_statistics()
        self.assertEqual(stats["modularity"], 0.25)
        with self.assertLogs(level="INFO") as logs:
            clusterer.print_statistics()
        self.assertIn("INFO:root:Modularity:          0.2500", logs.output)

    def test_library_refusals(self):
        self.fake()
        clusterer = EmbeddingClusterer(self.db_path)
        with self.assertRaisesRegex(ValueError, r"^Edges not built\. Call build_similarity_edges\(\) first\.$"):
            clusterer.cluster_leiden(backend="networkit")
        # A graph alone is not enough: NetworKit reads the edges
        clusterer.build_similarity_graph(threshold=0.8, max_neighbors=8)
        with self.assertRaisesRegex(ValueError, r"^Edges not built\."):
            clusterer.cluster_leiden(backend="networkit")
        clusterer = self.edges()
        cases = [(dict(backend="igraph", threads=2), r"^threads \(--leiden-threads\) applies to the networkit backend "
                                                     r"only; igraph runs on one thread\.$"),
                 (dict(threads=2), r"applies to the networkit backend only"),
                 (dict(backend="networkit", threads=0), r"^--leiden-threads must be at least 1\.$"),
                 (dict(backend="networkit", seed=2 ** 64), r"^--seed must be below 2\*\*64 with --leiden-backend "
                                                           r"networkit"),
                 (dict(backend=_REMOVED_PACKAGE), rf"'{_REMOVED_PACKAGE}' is not a valid LeidenBackend")]
        for kwargs, message in cases:
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                clusterer.cluster_leiden(**kwargs)
        # Random directions score below 0 with some others
        rng = np.random.default_rng(2)
        vectors = rng.standard_normal((12, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        clusterer = EmbeddingClusterer(_build(self.folder, "random", vectors))
        clusterer.build_similarity_edges(threshold=-1.0, max_neighbors=12)
        self.assertLess(clusterer.edges.min_weight, 0)
        with self.assertRaisesRegex(ValueError, r"^Leiden needs non-negative edge weights: the edges' smallest "
                                                r"weight is"):
            clusterer.cluster_leiden(backend="networkit")


class VersionCheckTest(unittest.TestCase):
    """NetworKit 11.3 from the package metadata, with no import; refused otherwise (RefusalError), also on macOS
    (the U9 (a) test passed there, so darwin is not refused)."""

    def check(self, version):
        def metadata(name):
            self.assertEqual(name, "networkit")
            if version is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return version

        before = set(sys.modules)
        with mock.patch.object(clustering.importlib.metadata, "version", side_effect=metadata), \
                mock.patch.dict(sys.modules):
            sys.modules.pop("networkit", None)
            try:
                clustering._check_networkit()
            finally:
                self.assertNotIn("networkit", set(sys.modules) - before)

    def test_versions(self):
        install = ("pip install 'foldmatch[networkit]' (NetworKit 11.3 has wheels for Linux x86_64, macOS 26 arm64 and "
                   "Windows, none for Linux aarch64 or macOS before 26).")
        for version, message in ((None, "networkit is not installed"), ("11.2", "networkit 11.2 is installed"),
                                 ("11.3.1", "networkit 11.3.1 is installed")):
            with self.subTest(version=version):
                with self.assertRaises(RefusalError) as caught:
                    self.check(version)
                self.assertEqual(str(caught.exception), f"--leiden-backend networkit needs NetworKit 11.3, and "
                                                        f"{message}: {install}")
        self.check("11.3")
        for platform in ("darwin", "linux", "win32"):
            with self.subTest(platform=platform), mock.patch.object(sys, "platform", platform):
                self.check("11.3")

    def test_the_extra_pins_the_checked_version(self):
        import tomllib
        path = Path(foldmatch.__file__).parent.parent.parent / "pyproject.toml"
        if not path.is_file():
            self.skipTest(f"no pyproject.toml next to the package ({path})")
        extras = tomllib.loads(path.read_text())["project"]["optional-dependencies"]
        self.assertEqual(extras["networkit"], [f"networkit=={clustering._NETWORKIT_VERSION}"])
        self.assertEqual(clustering._NETWORKIT_VERSION, "11.3")


class FakeGreedyLeidenTest(_FakeCase):
    """greedy-leiden with the networkit backend reads the memory-mapped edge files; the leiden record of run.json names
    the backend and the threads; a rerun with another backend searches nothing."""

    def test_greedy_leiden(self):
        nk, threads = self.fake()
        work = self.folder / "gl-work"
        clusterer = EmbeddingClusterer(self.db_path)
        with mock.patch.object(clustering, "_networkit_csr", wraps=clustering._networkit_csr) as csr:
            clusters = clusterer.cluster_greedy_leiden(work, greedy_threshold=0.99, threshold=0.8, max_neighbors=8,
                                                       seed=3, backend="networkit", threads=1, resolution=5.0)
        edges = csr.call_args.args[0]
        self.assertIsInstance(edges.pair_blocks[0], np.memmap)
        self.assertIn(("ParallelLeiden", 10, True, 5.0, 1), nk.calls)
        self.assertEqual(threads.sets, [3])
        self.assertEqual(clusterer.modularity, 0.25)
        np.testing.assert_array_equal(clusters, clusters[clusterer.chain_representatives])
        record = json.loads((work / "run.json").read_text())
        self.assertEqual(record["runs"][-1]["leiden"], {"resolution": 5.0, "n_iterations": 10, "seed": 3,
                                                        "backend": "networkit", "threads": 1})
        # igraph on the same folder: every step reused (no graph search, no copy), the leiden record names igraph
        again = EmbeddingClusterer(self.db_path)
        with mock.patch.object(EmbeddingClusterer, "_search_edge_blocks", side_effect=AssertionError("searched")), \
                mock.patch.object(FaissEmbeddingDatabase, "subset", side_effect=AssertionError("subset")):
            again.cluster_greedy_leiden(work, greedy_threshold=0.99, threshold=0.8, max_neighbors=8, seed=3,
                                        resolution=5.0)
        record = json.loads((work / "run.json").read_text())
        self.assertTrue(all(step.get("skipped") for step in record["runs"][-1]["steps"].values()))
        self.assertEqual(record["runs"][-1]["leiden"]["backend"], "igraph")
        self.assertIsNone(record["runs"][-1]["leiden"]["threads"])

    def test_checked_before_greedy(self):
        def missing(name):
            raise importlib.metadata.PackageNotFoundError(name)

        clusterer = EmbeddingClusterer(self.db_path)
        with mock.patch.object(clustering.importlib.metadata, "version", side_effect=missing), \
                mock.patch.object(EmbeddingClusterer, "_greedy_pass", side_effect=AssertionError("greedy ran")):
            with self.assertRaises(RefusalError):
                clusterer.cluster_greedy_leiden(self.folder / "missing", backend="networkit")
            with self.assertRaisesRegex(ValueError, "applies to the networkit backend only"):
                clusterer.cluster_greedy_leiden(self.folder / "missing", threads=4)
        self.assertFalse((self.folder / "missing").exists())


class CliTest(_FakeCase):
    """fm-search cluster --leiden-backend and --leiden-threads: the option table, the refusals before the load, the
    missing NetworKit (exit status 1), and the fake run through the CLI."""

    def invoke_unloaded(self, *args):
        with mock.patch.object(clustering, "EmbeddingClusterer", side_effect=AssertionError("the database was loaded")):
            return _invoke(*args)

    def refused(self, *args) -> str:
        result = self.invoke_unloaded(*args)
        self.assertEqual(result.exit_code, 2, f"{result.output}\n{result.exception!r}")
        return _flat(result.output)

    def test_option_table(self):
        base = ("cluster", "--db-path", "no/such/db", "--method", "greedy")
        leiden = "(--method leiden or greedy-leiden)"
        output = self.refused(*base, "--leiden-backend", "networkit", "--leiden-threads", "4")
        self.assertIn(f"Invalid value for --method: --method greedy does not take --leiden-backend {leiden}, "
                      f"--leiden-threads {leiden}. Remove them, or use a method that takes them.", output)
        output = self.refused(*base, "--leiden-backend", "igraph")
        self.assertIn(f"--method greedy does not take --leiden-backend {leiden}. Remove it, or use a method that "
                      f"takes it.", output)

    def test_refusals(self):
        base = ("cluster", "--db-path", "no/such/db")
        threads_igraph = ("Invalid value for --leiden-threads: --leiden-threads applies to --leiden-backend networkit "
                          "only; igraph runs on one thread.")
        cases = [(("--leiden-threads", "2"), threads_igraph),
                 (("--leiden-backend", "igraph", "--leiden-threads", "2"), threads_igraph),
                 (("--method", "greedy-leiden", "--work-folder", "w", "--leiden-threads", "1"), threads_igraph),
                 (("--leiden-backend", "networkit", "--leiden-threads", "0"),
                  "Invalid value for --leiden-threads: --leiden-threads must be at least 1."),
                 (("--leiden-backend", "networkit", "--seed", str(2 ** 64)),
                  "Invalid value for --seed: --seed must be below 2**64 with --leiden-backend networkit (NetworKit's "
                  "seed is an unsigned 64-bit integer)."),
                 (("--leiden-backend", _REMOVED_PACKAGE), "--leiden-backend")]
        for extra, message in cases:
            with self.subTest(extra=extra):
                self.assertIn(message, self.refused(*base, *extra))

    def test_macos_refuses_more_than_one_thread(self):
        """--leiden-backend networkit --leiden-threads above 1 on macOS: exit status 2 before the load (and before the
        NetworKit check); 1 thread, or another platform, is not refused there."""
        message = f"Invalid value for --leiden-threads: {clustering._NETWORKIT_DARWIN_REFUSAL}"
        base = ("cluster", "--db-path", "no/such/db", "--leiden-backend", "networkit")
        with mock.patch.object(search.sys, "platform", "darwin"), \
                mock.patch.object(clustering, "_check_networkit", side_effect=AssertionError("checked")):
            for extra in (("--leiden-threads", "2"), ("--leiden-threads", "64"),
                          ("--method", "greedy-leiden", "--work-folder", self.folder / "w", "--leiden-threads", "2")):
                with self.subTest(extra=extra):
                    self.assertIn(message, self.refused(*base, *extra))
        self.fake()
        for platform, threads in (("darwin", "1"), ("linux", "2")):
            with self.subTest(platform=platform), mock.patch.object(search.sys, "platform", platform):
                result = self.invoke_unloaded(*base, "--leiden-threads", threads)
                self.assertEqual(str(result.exception), "the database was loaded")

    def test_missing_networkit_exit_1_before_the_load(self):
        for version in (None, "11.2"):
            def metadata(name, version=version):
                if version is None:
                    raise importlib.metadata.PackageNotFoundError(name)
                return version

            for extra in ((), ("--method", "greedy-leiden", "--work-folder", self.folder / "w")):
                with self.subTest(version=version, extra=extra), \
                        mock.patch.object(clustering.importlib.metadata, "version", side_effect=metadata):
                    result = self.invoke_unloaded("cluster", "--db-path", "no/such/db", "--leiden-backend",
                                                  "networkit", *extra)
                    self.assertEqual(result.exit_code, 1, result.output)
                    self.assertIsInstance(result.exception, RefusalError)
                    self.assertIn("--leiden-backend networkit needs NetworKit 11.3", str(result.exception))

    def test_igraph_never_checks_networkit(self):
        with mock.patch.object(clustering, "_check_networkit", side_effect=AssertionError("checked")):
            result = self.invoke_unloaded("cluster", "--db-path", "no/such/db", "--leiden-backend", "igraph")
        self.assertEqual(str(result.exception), "the database was loaded")

    def test_fake_run(self):
        """cluster --leiden-backend networkit --leiden-threads 1: the edges without an igraph graph, the file of the
        fake's partition, the statistics' modularity line; --leiden-threads 2 with --seed: the WARNING."""
        nk, _ = self.fake()
        out = self.folder / "nk.tsv"
        with mock.patch.object(EmbeddingClusterer, "build_similarity_graph",
                               side_effect=AssertionError("igraph graph built")):
            result = _invoke("cluster", "--db-path", self.db_path, "--threshold", "0.8", "--max-neighbors", "8",
                             "--seed", "5", "--leiden-backend", "networkit", "--leiden-threads", "1",
                             "--output-file", out)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        clusters = _reference_numbering(nk.partitions[0].vector)
        sizes = np.bincount(clusters)
        self.assertEqual(out.read_text().splitlines(),
                         [f"chain{i}\t{c}\t{sizes[c]}" for i, c in enumerate(clusters.tolist())])
        self.assertIn(("setSeed", 5, True, 1), nk.calls)

    def test_help(self):
        command = typer.main.get_command(search.app).commands["cluster"]
        helps = {p.name: p.help for p in command.params}
        self.assertEqual(helps["leiden_backend"], (
            "Leiden implementation. igraph (default): python-igraph's community_leiden, one thread, the same --seed "
            "gives the same clusters. networkit: NetworKit's ParallelLeiden on --leiden-threads threads, 5-25x faster "
            "on large graphs (measured at 64 threads on Linux); needs pip install 'foldmatch\\[networkit]' (NetworKit "
            "11.3: wheels for Linux x86_64 and macOS 26 arm64, where it runs on 1 thread only; none for Linux "
            "aarch64). With more than one thread the same --seed does not give the same clusters. Refused with "
            "--method greedy."))
        self.assertEqual(helps["leiden_threads"], (
            "With --leiden-backend networkit, the threads of ParallelLeiden and of its graph build (at least 1). "
            "Default: every OpenMP thread (NetworKit's maximum, OMP_NUM_THREADS when set); 1 on macOS, where more are "
            "refused (NetworKit's OpenMP runtime crashes there next to torch's and FAISS's). 1 gives the same "
            "clusters for the same --seed. Refused with --leiden-backend igraph and with --method greedy."))
        self.assertIn("(with --leiden-backend networkit only on one --leiden-threads; on more, a WARNING)",
                      helps["seed"])
        self.assertIn(f"networkit=={clustering._NETWORKIT_VERSION}".replace("networkit==", "NetworKit "),
                      helps["leiden_backend"])

    def test_rendered_help_names_the_extra(self):
        # typer renders help as Rich markup: an unescaped [networkit] was dropped, leaving pip install 'foldmatch'.
        result = CliRunner().invoke(search.app, ["cluster", "--help"], env={"COLUMNS": "200"})
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("needs pip install 'foldmatch[networkit]' (NetworKit 11.3: wheels for Linux x86_64",
                      _flat(result.output))


def _cli_in_a_fresh_process(*args) -> subprocess.CompletedProcess:
    """``fm-search`` in a new Python process of this interpreter, the environment as it is (PYTHONPATH included),
    asserting it runs this test's foldmatch."""
    child = ("import sys, foldmatch\n"
             "assert foldmatch.__file__ == sys.argv[1], foldmatch.__file__\n"
             "sys.argv = ['fm-search'] + sys.argv[2:]\n"
             "from foldmatch.cli.search import main\n"
             "main()\n")
    return subprocess.run([sys.executable, "-c", child, foldmatch.__file__, *[str(a) for a in args]],
                          capture_output=True, text=True, timeout=600)


@unittest.skipUnless(_REAL, "NetworKit 11.3 is not importable")
class RealNetworkitTest(unittest.TestCase):
    """The real NetworKit 11.3: one thread and one seed give the same file in two fresh processes (leiden on a flat
    database, greedy-leiden on an ivf_pq one), at the default FAISS thread count; the stored modularity is igraph's
    modularity of the same partition; the partition agrees with igraph's within igraph's seed band on two
    families; both thread counts are left as they were."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.flat = _build(cls.folder, "flat", _clustered_vectors(seed=5, n=600, d=64, n_centers=30, noise=1.0))
        cls.pq = _build(cls.folder, "pq", _clustered_vectors(seed=7, n=1000, d=64, n_centers=40, noise=0.8),
                        IndexType.ivf_pq, IndexConfig(nlist=16, m=8, nbits=4, nprobe=4, opq_train_size=1000))
        cls.families = _build(cls.folder, "fam", _two_families(seed=3))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_two_fresh_processes(self):
        cases = [("flat", ("--db-path", self.flat, "--threshold", "0.5", "--max-neighbors", "10")),
                 ("ivf_pq", ("--db-path", self.pq, "--method", "greedy-leiden", "--greedy-threshold", "0.9",
                             "--max-neighbors", "20", "--resolution", "5"))]
        for name, args in cases:
            with self.subTest(db=name):
                files = []
                for attempt in range(2):
                    out = self.folder / f"{name}.{attempt}.tsv"
                    extra = ("--work-folder", self.folder / f"{name}.{attempt}.work") if name == "ivf_pq" else ()
                    run = _cli_in_a_fresh_process("cluster", *args, *extra, "--leiden-backend", "networkit",
                                                  "--leiden-threads", "1", "--seed", "7", "--output-file", out)
                    self.assertEqual(run.returncode, 0, run.stderr[-3000:])
                    self.assertIn("Running NetworKit ParallelLeiden on 1 threads", run.stderr)
                    self.assertNotIn("WARNING", run.stderr)
                    files.append(out.read_bytes())
                self.assertEqual(files[0], files[1])
                self.assertGreater(len(set(line.split(b"\t")[1] for line in files[0].splitlines())), 5)

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    def test_more_than_one_thread_refused_on_macos(self):
        """In fresh processes with torch and FAISS loaded (where 2 NetworKit threads crashed the process with a
        segmentation fault): the CLI refuses --leiden-threads 2 (exit status 2, before the load), the library refuses
        threads=2 before NetworKit runs, and its default runs on 1 thread."""
        out = self.folder / "darwin2.tsv"
        run = _cli_in_a_fresh_process("cluster", "--db-path", self.flat, "--threshold", "0.5", "--max-neighbors", "10",
                                      "--leiden-backend", "networkit", "--leiden-threads", "2", "--output-file", out)
        self.assertEqual(run.returncode, 2, run.stderr[-3000:])
        self.assertIn(f"Invalid value for --leiden-threads: {clustering._NETWORKIT_DARWIN_REFUSAL}", _flat(run.stderr))
        self.assertNotIn("Initializing clusterer", run.stderr)
        self.assertFalse(out.exists())
        child = ("import sys, foldmatch\n"
                 "assert foldmatch.__file__ == sys.argv[1], foldmatch.__file__\n"
                 "import logging\n"
                 "import lightning\n"
                 "from foldmatch.search.clustering import EmbeddingClusterer\n"
                 "logging.basicConfig(level=logging.INFO)\n"
                 "clusterer = EmbeddingClusterer(sys.argv[2])\n"
                 "clusterer.build_similarity_edges(threshold=0.5, max_neighbors=10)\n"
                 "try:\n"
                 "    clusterer.cluster_leiden(seed=3, backend='networkit', threads=2)\n"
                 "except ValueError as exc:\n"
                 "    print('refused', exc)\n"
                 "clusters = clusterer.cluster_leiden(seed=3, backend='networkit')\n"
                 "print('clusters', int(clusters.max()) + 1)\n")
        run = subprocess.run([sys.executable, "-c", child, foldmatch.__file__, str(self.flat)], capture_output=True,
                             text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr[-3000:])
        self.assertIn(f"refused {clustering._NETWORKIT_DARWIN_REFUSAL}", run.stdout)
        self.assertIn("Running NetworKit ParallelLeiden on 1 threads", run.stderr)
        self.assertGreater(int(run.stdout.split("clusters ")[1]), 5)

    def test_modularity_and_threads(self):
        import networkit
        nk_before, faiss_before = networkit.engineering.getMaxNumberOfThreads(), faiss.omp_get_max_threads()
        clusterer = EmbeddingClusterer(self.flat)
        clusterer.build_similarity_edges(threshold=0.5, max_neighbors=10)
        clusters = clusterer.cluster_leiden(seed=3, backend="networkit", threads=1)
        self.assertEqual((networkit.engineering.getMaxNumberOfThreads(), faiss.omp_get_max_threads()),
                         (nk_before, faiss_before))
        graph = EmbeddingClusterer(self.flat).build_similarity_graph(threshold=0.5, max_neighbors=10)
        self.assertAlmostEqual(clusterer.modularity, graph.modularity(clusters.tolist(), weights="weight"), places=9)
        self.assertEqual(clusterer.get_cluster_statistics()["modularity"], clusterer.modularity)
        sizes = np.bincount(clusters)
        self.assertTrue(np.all(np.diff(sizes) <= 0))
        # The same seed on one thread, in the same process: the same partition
        np.testing.assert_array_equal(clusterer.cluster_leiden(seed=3, backend="networkit", threads=1), clusters)

    def test_agrees_with_igraph_within_its_seed_band(self):
        from sklearn.metrics import adjusted_mutual_info_score as ami
        direct = EmbeddingClusterer(self.families)
        direct.build_similarity_graph(threshold=0.8, max_neighbors=50)
        igraph_runs = [direct.cluster_leiden(seed=s).copy() for s in range(1, 5)]
        band = min(ami(a, b) for i, a in enumerate(igraph_runs) for b in igraph_runs[i + 1:])
        nk = EmbeddingClusterer(self.families)
        nk.build_similarity_edges(threshold=0.8, max_neighbors=50)
        for seed in (1, 2):
            with self.subTest(seed=seed):
                clusters = nk.cluster_leiden(seed=seed, backend="networkit", threads=1)
                self.assertGreater(len(np.unique(clusters)), 1)
                for reference in igraph_runs:
                    self.assertGreaterEqual(ami(clusters, reference), band)


_OPENMP_PROBE = """
import os, sys, shutil, tempfile
from pathlib import Path
import foldmatch
assert foldmatch.__file__ == sys.argv[1], foldmatch.__file__
import torch  # noqa: F401
import lightning  # noqa: F401
import faiss
import numpy as np
from foldmatch.search.clustering import EmbeddingClusterer
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.types.api_types import IndexType

folder = Path(sys.argv[2])
rng = np.random.default_rng(5)
centers = rng.standard_normal((30, 64))
v = centers[np.sort(rng.integers(0, 30, 3000))] + 0.7 * rng.standard_normal((3000, 64)) / 8
v = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
db = FaissEmbeddingDatabase(folder, "flat")
db.create_database(iter([([f"c{i}" for i in range(len(v))], v)]), index_type=IndexType.flat)
clusterer = EmbeddingClusterer(str(folder / "flat"))
before = clusterer.db.index.search(v[:500].copy(), 50)
clusterer.build_similarity_edges(threshold=0.6, max_neighbors=50)
import networkit
threads = networkit.engineering.getMaxNumberOfThreads()
clusters = clusterer.cluster_leiden(seed=42, backend="networkit")
after = clusterer.db.index.search(v[:500].copy(), 50)
print("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
print("networkit threads", threads, networkit.engineering.getMaxNumberOfThreads())
print("clusters", int(clusters.max()) + 1)
print("same search", bool(np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])))
"""


@unittest.skipUnless(_REAL, "NetworKit 11.3 is not importable")
class OpenMpProbeTest(unittest.TestCase):
    """U9 (a): with torch and FAISS imported and the platform's default OMP_NUM_THREADS (FoldMatch pins 1 on macOS
    when it is unset), a FAISS search, ParallelLeiden through cluster_leiden, and the same search again give the same
    results in one fresh process; on macOS NetworKit runs on 1 thread."""

    def test_faiss_then_networkit_then_faiss(self):
        env = dict(os.environ)
        env.pop("OMP_NUM_THREADS", None)
        with tempfile.TemporaryDirectory() as folder:
            run = subprocess.run([sys.executable, "-c", _OPENMP_PROBE, foldmatch.__file__, folder],
                                 capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(run.returncode, 0, run.stderr[-3000:])
        lines = dict(line.split(" ", 1) for line in run.stdout.splitlines() if " " in line)
        self.assertEqual(lines["same"], "search True")
        self.assertGreater(int(lines["clusters"]), 1)
        if sys.platform == "darwin":
            self.assertEqual(lines["OMP_NUM_THREADS"], "1")
            self.assertEqual(lines["networkit"], "threads 1 1")


if __name__ == "__main__":
    unittest.main()
