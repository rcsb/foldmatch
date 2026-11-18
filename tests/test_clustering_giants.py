"""Giant clusters in the statistics and the greedy-leiden resolution default (P9.CL D4): the largest cluster's share,
the clusters above 1.745% of the chains (in integers, at the exact boundary) and the largest sizes, at INFO under every
method; the giant WARNING under greedy-leiden only, from _GIANT_WARN_MIN_CHAINS chains on; --resolution 1.0 under
leiden and 20 under greedy-leiden when not given."""
import inspect
import logging
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# As the commands run (set_log_level): Lightning first, which brings torch and, through scikit-learn,
# KMP_DUPLICATE_LIB_OK; on macOS, FAISS's OpenMP regions abort a process that loaded torch's libomp without it.
import lightning  # noqa: F401
import igraph as ig
import numpy as np
import typer
from typer.testing import CliRunner

from foldmatch.cli import search
from foldmatch.search import clustering
from foldmatch.search.clustering import EmbeddingClusterer
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.types.api_types import IndexType


def _build(folder: Path, name: str, vectors: np.ndarray) -> str:
    db = FaissEmbeddingDatabase(folder, name)
    db.create_database(iter([([f"chain{i}" for i in range(len(vectors))], vectors)]), index_type=IndexType.flat)
    return str(folder / name)


def _two_families(seed: int, d: int = 32):
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


def _invoke(*args):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        return CliRunner().invoke(search.app, [str(a) for a in args])
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


class GiantRuleTest(unittest.TestCase):
    """The statistics of a partition set on a clusterer: the giant rule at its exact boundary, in integers."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(1)
        vectors = rng.standard_normal((8, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        cls.db_path = _build(Path(cls.tmp.name), "tiny", vectors)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stats(self, sizes, n=None):
        """The statistics of clusters of these sizes (cluster ids by decreasing size), padded with singletons to n
        chains."""
        sizes = list(sizes)
        if n is not None:
            sizes += [1] * (n - sum(sizes))
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.clusters = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
        return clusterer, clusterer.get_cluster_statistics()

    def test_constants(self):
        self.assertEqual((clustering._GIANT_PER_100K, clustering._GIANT_WARN_MIN_CHAINS,
                          clustering._LARGEST_CLUSTERS), (1_745, 1_000_000, 10))

    def test_boundary(self):
        """n = 100,000: 1,745 chains (exactly 1.745%) is not a giant, 1,746 is; the same at 1,000,000 chains with
        17,450 and 17,451 (CA's 17,450 chains at 1M)."""
        for n, size, giants in ((100_000, 1_745, 0), (100_000, 1_746, 1), (1_000_000, 17_450, 0),
                                (1_000_000, 17_451, 1)):
            with self.subTest(n=n, size=size):
                _, stats = self.stats([size], n)
                self.assertEqual(stats["n_chains"], n)
                self.assertEqual(stats["n_giant_clusters"], giants)
                self.assertEqual(stats["largest_cluster_share"], size / n)
                self.assertEqual(stats["largest_cluster_sizes"], [size] + [1] * 9)

    def test_several_giants_and_few_clusters(self):
        _, stats = self.stats([40, 30, 20, 7, 3])
        self.assertEqual(stats["n_giant_clusters"], 5)
        self.assertEqual(stats["largest_cluster_sizes"], [40, 30, 20, 7, 3])
        self.assertEqual(stats["largest_cluster_share"], 0.4)
        # 2,000 chains: 1.745% is 34.9 chains, so 35 is a giant and 34 is not; 12 clusters, the 10 largest listed
        _, stats = self.stats([400, 35, 34, 34] + [2] * 8, 2_000)
        self.assertEqual(stats["n_giant_clusters"], 2)
        self.assertEqual(stats["largest_cluster_sizes"], [400, 35, 34, 34, 2, 2, 2, 2, 2, 2])

    def test_sizes_listed_largest_first_whatever_the_numbering(self):
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.clusters = np.array([3, 0, 3, 1, 3, 1, 2, 3], dtype=np.int32)
        stats = clusterer.get_cluster_statistics()
        self.assertEqual(stats["largest_cluster_sizes"], [4, 2, 1, 1])
        self.assertEqual(stats["largest_cluster_share"], 0.5)

    def test_info_lines(self):
        clusterer, stats = self.stats([1_746, 900], 100_000)
        with self.assertLogs(level="INFO") as logs:
            clusterer.print_statistics()
        self.assertIn("INFO:root:Largest cluster:     1746 chains (1.746% of the chains)", logs.output)
        self.assertIn("INFO:root:Giant clusters (more than 1.745% of the chains): 1", logs.output)
        self.assertIn("INFO:root:Largest 10 clusters: 1746 (1.746%), 900 (0.900%), " + ", ".join(["1 (0.001%)"] * 8),
                      logs.output)
        self.assertFalse(any(line.startswith("WARNING") for line in logs.output))


class GiantWarningTest(unittest.TestCase):
    """The giant lines under leiden, greedy and greedy-leiden; the WARNING under greedy-leiden only, at
    _GIANT_WARN_MIN_CHAINS chains or more (patched down to the fixture's size)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        vectors = _two_families(seed=3)
        cls.n = len(vectors)
        cls.db_path = _build(cls.folder, "fam", vectors)
        cls.chained = EmbeddingClusterer(cls.db_path)
        # At resolution 1 each family is one cluster: two giants
        cls.chained.cluster_greedy_leiden(cls.folder / "work", seed=2, resolution=1.0)
        cls.direct = EmbeddingClusterer(cls.db_path)
        cls.direct.build_similarity_graph(threshold=0.8, max_neighbors=50)
        cls.direct.cluster_leiden(seed=2)
        cls.greedy = EmbeddingClusterer(cls.db_path)
        cls.greedy.cluster_greedy(threshold=0.9, max_neighbors=1000)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def logs(self, clusterer, min_chains):
        with mock.patch.object(clustering, "_GIANT_WARN_MIN_CHAINS", min_chains), \
                self.assertLogs(level="INFO") as logs:
            clusterer.print_statistics()
        return logs.output

    def test_fixture(self):
        for clusterer in (self.chained, self.direct, self.greedy):
            with self.subTest(method=clusterer.method):
                self.assertGreater(clusterer.get_cluster_statistics()["n_giant_clusters"], 1)

    def test_warning_under_greedy_leiden_only(self):
        n_giants = self.chained.get_cluster_statistics()["n_giant_clusters"]
        warning = (f"WARNING:root:{n_giants} clusters each hold more than 1.745% of the chains; a higher --resolution "
                   f"splits large clusters (measured on TrEMBL samples of 1M-10M chains).")
        for min_chains, warned in ((self.n, True), (self.n - 1, True), (self.n + 1, False)):
            with self.subTest(min_chains=min_chains):
                output = self.logs(self.chained, min_chains)
                self.assertEqual(warning in output, warned)
                self.assertIn(f"INFO:root:Giant clusters (more than 1.745% of the chains): {n_giants}", output)
        for clusterer in (self.direct, self.greedy):
            with self.subTest(method=clusterer.method):
                n_giants = clusterer.get_cluster_statistics()["n_giant_clusters"]
                output = self.logs(clusterer, 1)
                self.assertFalse(any(line.startswith("WARNING:root:") and "1.745%" in line for line in output))
                self.assertIn(f"INFO:root:Giant clusters (more than 1.745% of the chains): {n_giants}", output)

    def test_one_giant_in_the_singular(self):
        stats = dict(self.chained.get_cluster_statistics(), n_giant_clusters=1)
        with mock.patch.object(self.chained, "get_cluster_statistics", return_value=stats):
            output = self.logs(self.chained, 1)
        self.assertIn("WARNING:root:1 cluster holds more than 1.745% of the chains; a higher --resolution splits "
                      "large clusters (measured on TrEMBL samples of 1M-10M chains).", output)

    def test_no_giant_no_warning(self):
        stats = dict(self.chained.get_cluster_statistics(), n_giant_clusters=0)
        with mock.patch.object(self.chained, "get_cluster_statistics", return_value=stats):
            output = self.logs(self.chained, 1)
        self.assertFalse(any(line.startswith("WARNING:root:") and "1.745%" in line for line in output))


class ResolutionDefaultTest(unittest.TestCase):
    """--resolution: 1.0 under leiden and 20 under greedy-leiden when not given, through to python-igraph's call; the
    library defaults; the help text from the same constant."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.db_path = _build(cls.folder, "fam", _two_families(seed=3))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_library_defaults(self):
        self.assertEqual(clustering._GREEDY_LEIDEN_RESOLUTION, 20.0)
        defaults = {name: inspect.signature(getattr(EmbeddingClusterer, name)).parameters["resolution"].default
                    for name in ("cluster_leiden", "cluster_greedy_leiden")}
        self.assertEqual(defaults, {"cluster_leiden": 1.0, "cluster_greedy_leiden": 20.0})

    def test_cli_reaches_igraph(self):
        real = ig.Graph.community_leiden
        calls = []

        def community_leiden(graph, *args, **kwargs):
            calls.append(kwargs["resolution"])
            return real(graph, *args, **kwargs)

        out = self.folder / "out.tsv"
        base = ("cluster", "--db-path", self.db_path, "--output-file", out, "--seed", "1")
        cases = [((), 1.0), (("--method", "leiden"), 1.0), (("--resolution", "3"), 3.0),
                 (("--method", "greedy-leiden", "--work-folder", self.folder / "w1"), 20.0),
                 (("--method", "greedy-leiden", "--work-folder", self.folder / "w2", "--resolution", "2"), 2.0)]
        for extra, resolution in cases:
            with self.subTest(extra=extra):
                calls.clear()
                with mock.patch.object(ig.Graph, "community_leiden", community_leiden):
                    result = _invoke(*base, *extra)
                self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
                self.assertEqual(calls, [resolution])

    def test_help(self):
        command = typer.main.get_command(search.app).commands["cluster"]
        text = next(p.help for p in command.params if p.name == "resolution")
        self.assertEqual(text, "Leiden resolution parameter (higher = more, smaller clusters). Default: 1.0 with "
                               "--method leiden; 20 with --method greedy-leiden (measured on TrEMBL samples of 1M-10M "
                               "chains: largest cluster about 2% of the chains or less; Pfam family agreement higher "
                               "than at 10, best at 50, the top of the measured grid; a lower value such as 10 keeps "
                               "more superfamilies whole). Refused with --method greedy.")
        self.assertIn(f"; {clustering._GREEDY_LEIDEN_RESOLUTION:g} with --method greedy-leiden", text)


if __name__ == "__main__":
    unittest.main()
