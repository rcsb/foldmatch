"""fm-search cluster --method greedy-leiden (P9.CL D2, D3, D7): greedy representatives, the representatives' database
copied from the parent, their graph with its edges in the work folder, Leiden, and every chain in its representative's
cluster. The identity control (a greedy threshold above every score gives --method leiden's file) on flat and ivf_pq;
two families; the statistics; the work folder (run.json, greedy checkpoints and resumes after a fault, the graph
restart, Leiden-only reruns, every refusal); the hnsw refusal; the one option table of cluster."""
import json
import logging
import math
import os
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

# As the commands run (set_log_level): Lightning first, which brings torch and, through scikit-learn,
# KMP_DUPLICATE_LIB_OK; on macOS, FAISS's OpenMP regions abort a process that loaded torch's libomp without it.
import lightning  # noqa: F401
import faiss
import igraph as ig
import numpy as np
from typer.testing import CliRunner

from foldmatch.cli import search
from foldmatch.search import clustering
from foldmatch.search.clustering import EmbeddingClusterer, SimilarityEdges
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.types.api_types import EdgeRule, IndexConfig, IndexType

_PQ_CONFIG = dict(nlist=16, m=8, nbits=4, nprobe=4, opq_train_size=1000)


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


def _build(folder: Path, name: str, vectors: np.ndarray, index_type: IndexType, config: IndexConfig = None) -> str:
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


def _rows(path) -> list:
    return [line.split("\t") for line in Path(path).read_text().splitlines()]


class _Fixtures:
    """A flat database (600 x 64) and an ivf_pq one (1000 x 64, nlist 16, PQ8x4, stored nprobe 4), each with some
    near-duplicates so that greedy at 0.9 takes members."""

    @classmethod
    def build_fixtures(cls, folder: Path):
        flat = _clustered_vectors(seed=5, n=600, d=64, n_centers=30, noise=1.0)
        pq = _clustered_vectors(seed=7, n=1000, d=64, n_centers=40, noise=0.8)
        rng = np.random.default_rng(11)
        for vectors in (flat, pq):
            # Near-duplicates: every fifth chain from 100 on is a copy of an earlier chain, slightly moved
            for i in range(100, len(vectors), 5):
                v = vectors[i - 97] + 0.05 * rng.standard_normal(vectors.shape[1]) / math.sqrt(vectors.shape[1])
                vectors[i] = v / np.linalg.norm(v)
        return {"flat": _build(folder, "flat", flat, IndexType.flat),
                "ivf_pq": _build(folder, "pq", pq, IndexType.ivf_pq, IndexConfig(**_PQ_CONFIG))}


class IdentityControlTest(_Fixtures, unittest.TestCase):
    """--greedy-threshold above every score: every chain is its own representative, the representatives' database holds
    every row (the same vectors, or the same codes), so the searches are --method leiden's: the TSV's first three
    columns are --method leiden's file byte for byte, and the fourth is the first."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.dbs = cls.build_fixtures(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_identity(self):
        cases = [("--threshold", "0.5", "--max-neighbors", "10", "--seed", "3"),
                 ("--threshold", "0.5", "--max-neighbors", "10", "--seed", "3", "--edge-rule", "lower"),
                 ("--threshold", "0.6", "--max-neighbors", "30", "--seed", "8", "--resolution", "2.5",
                  "--leiden-iterations", "4"),
                 ("--threshold", "0.5", "--max-neighbors", "1000", "--seed", "1")]
        for name, db_path in self.dbs.items():
            for i, options in enumerate(cases):
                with self.subTest(db=name, options=options):
                    leiden, chained = self.folder / f"{name}{i}.leiden.tsv", self.folder / f"{name}{i}.gl.tsv"
                    result = _invoke("cluster", "--db-path", db_path, *options, "--output-file", leiden)
                    self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
                    result = _invoke("cluster", "--db-path", db_path, *options, "--method", "greedy-leiden",
                                     "--greedy-threshold", "2.0", "--work-folder", self.folder / f"{name}{i}.work",
                                     "--output-file", chained)
                    self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
                    rows = _rows(chained)
                    self.assertEqual("".join("\t".join(r[:3]) + "\n" for r in rows).encode(), leiden.read_bytes())
                    self.assertEqual([r[3] for r in rows], [r[0] for r in rows])
                    self.assertGreater(len(set(r[1] for r in rows)), 5)
                    self.assertLess(len(set(r[1] for r in rows)), len(rows))

    def test_identity_in_the_library(self):
        """The same through the library, at the default edge rule: the clusters equal cluster_leiden's array, the
        representatives' database holds every chain, and the modularity is that of the direct graph."""
        for name, db_path in self.dbs.items():
            with self.subTest(db=name):
                direct = EmbeddingClusterer(db_path)
                direct.build_similarity_graph(threshold=0.5, max_neighbors=10)
                expected = direct.cluster_leiden(seed=5)
                work = self.folder / f"{name}.lib.work"
                chained = EmbeddingClusterer(db_path)
                clusters = chained.cluster_greedy_leiden(work, greedy_threshold=2.0, threshold=0.5, max_neighbors=10,
                                                         seed=5)
                self.assertEqual(clusters.dtype, expected.dtype)
                np.testing.assert_array_equal(clusters, expected)
                np.testing.assert_array_equal(chained.chain_representatives, np.arange(len(expected)))
                self.assertEqual(chained.n_representatives, len(expected))
                self.assertEqual(chained.modularity, direct.graph.modularity(expected.tolist(), weights="weight"))
                reps = FaissEmbeddingDatabase(work, "representatives")
                reps.load_database()
                self.assertEqual(reps.chain_ids, chained.chain_ids)


def _two_families(seed: int, d: int = 32):
    """Two orthogonal families of 6 sub-centres each (sub-centres of a family score about 0.85 with each other), each
    sub-centre with 4 to 9 near-duplicates (scores above 0.97), in a shuffled order. Returns the vectors and each
    chain's family."""
    rng = np.random.default_rng(seed)
    vectors, family = [], []
    for f in range(2):
        axis = np.zeros(d)
        axis[f] = 1.0
        for _ in range(6):
            sub = axis + 0.42 * rng.standard_normal(d) * (np.arange(d) >= 2) / math.sqrt(d - 2)
            sub /= np.linalg.norm(sub)
            for _ in range(int(rng.integers(4, 10))):
                v = sub + 0.08 * rng.standard_normal(d) / math.sqrt(d)
                vectors.append(v / np.linalg.norm(v))
                family.append(f)
    order = rng.permutation(len(vectors))
    return np.asarray(vectors, dtype=np.float32)[order], np.asarray(family)[order]


class TwoFamiliesTest(unittest.TestCase):
    """Two well-separated families with near-duplicate members (flat): every member is in its representative's final
    cluster, the fourth column is the representative cluster_greedy gives each chain, no cluster spans both families,
    the numbering follows the size rule, and the statistics add up (modularity on the representatives' graph)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.vectors, cls.family = _two_families(seed=3)
        cls.n = len(cls.vectors)
        cls.db_path = _build(cls.folder, "fam", cls.vectors, IndexType.flat)
        cls.clusterer = EmbeddingClusterer(cls.db_path)
        cls.clusterer.cluster_greedy_leiden(cls.folder / "work", seed=2)
        greedy = EmbeddingClusterer(cls.db_path)
        greedy.cluster_greedy(threshold=0.9, max_neighbors=1000)
        cls.greedy_reps = greedy.representatives[greedy.clusters]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixture(self):
        scores = self.vectors @ self.vectors.T
        across = scores[self.family[:, None] != self.family[None, :]]
        self.assertLess(across.max(), 0.5)
        # Greedy takes members, and the representatives' graph has edges
        self.assertLess(self.clusterer.n_representatives, self.n / 3)
        self.assertGreater(self.clusterer.graph_stats["n_edges"], 10)

    def test_members_follow_their_representative(self):
        clusters, reps = self.clusterer.clusters, self.clusterer.chain_representatives
        np.testing.assert_array_equal(clusters, clusters[reps])
        np.testing.assert_array_equal(reps, self.greedy_reps)
        self.assertEqual(int(np.count_nonzero(reps == np.arange(self.n))), self.clusterer.n_representatives)
        for c in np.unique(clusters):
            self.assertEqual(len(set(self.family[clusters == c].tolist())), 1)

    def test_size_rule(self):
        clusters = self.clusterer.clusters
        sizes = np.bincount(clusters)
        self.assertTrue(np.all(np.diff(sizes) <= 0), sizes.tolist())
        first = [int(np.flatnonzero(clusters == c)[0]) for c in range(len(sizes))]
        for c in range(len(sizes) - 1):
            if sizes[c] == sizes[c + 1]:
                self.assertLess(first[c], first[c + 1])

    def test_export(self):
        out = self.folder / "fam.tsv"
        self.clusterer.export_clusters(str(out))
        rows = _rows(out)
        ids = self.clusterer.chain_ids
        sizes = np.bincount(self.clusterer.clusters)
        self.assertEqual(rows, [[ids[i], str(c), str(sizes[c]), ids[self.greedy_reps[i]]]
                                for i, c in enumerate(self.clusterer.clusters.tolist())])
        # With chunks of 7 chains, and with --min-cluster-size above every size: an empty file
        with mock.patch.object(clustering, "_EXPORT_CHUNK", 7):
            self.clusterer.export_clusters(str(self.folder / "fam7.tsv"))
        self.assertEqual((self.folder / "fam7.tsv").read_bytes(), out.read_bytes())
        self.clusterer.export_clusters(str(self.folder / "none.tsv"), min_cluster_size=self.n + 1)
        self.assertEqual((self.folder / "none.tsv").read_bytes(), b"")

    def test_statistics(self):
        stats = self.clusterer.get_cluster_statistics()
        n_reps = self.clusterer.n_representatives
        edges = SimilarityEdges.from_folder(self.folder / "work")
        graph = ig.Graph(n=edges.n, edges=[tuple(p) for p in np.asarray(edges.pair_blocks[0]).tolist()])
        graph.es["weight"] = np.asarray(edges.weight_blocks[0]).tolist()
        rep_rows = np.flatnonzero(self.clusterer.chain_representatives == np.arange(self.n))
        membership = self.clusterer.clusters[rep_rows].tolist()
        self.assertAlmostEqual(stats["modularity"], graph.modularity(membership, weights="weight"), places=12)
        self.assertEqual(stats["method"], "greedy-leiden")
        self.assertEqual((stats["n_representatives"], stats["graph_nodes"]), (n_reps, n_reps))
        self.assertEqual(stats["graph_edges"], edges.n_edges)
        self.assertEqual(stats["n_chains"], self.n)
        self.assertEqual(stats["n_clusters"], len(np.unique(self.clusterer.clusters)))
        self.assertLessEqual(stats["n_searched"], self.n)
        self.assertGreaterEqual(stats["n_searched"], n_reps)
        self.assertEqual((stats["n_saturated"], stats["n_at_cap"]), (0, 0))
        with self.assertLogs(level="INFO") as logs:
            self.clusterer.print_statistics()
        self.assertIn(f"INFO:root:Modularity (representatives' graph): {stats['modularity']:.4f}", logs.output)
        self.assertIn(f"INFO:root:Representatives:     {n_reps} of {self.n} ({n_reps / self.n:.2%})", logs.output)
        self.assertIn("INFO:root:Representatives with all --greedy-max-neighbors neighbours at or above "
                      f"--greedy-threshold: 0 of {n_reps} (0.00%); their greedy clusters may be cut, and a larger "
                      "--greedy-max-neighbors can change them.", logs.output)
        self.assertIn(f"INFO:root:Representatives' graph: {n_reps} nodes, {edges.n_edges} edges "
                      f"({stats['n_rescued_chains']} chains rescued with {stats['n_rescued_edges']} edges)",
                      logs.output)
        self.assertFalse(any(line.startswith("INFO:root:Modularity: ") for line in logs.output))
        # D2's order: the greedy step, then the representatives' graph, then the final partition over chains
        starts = ["INFO:root:Total embeddings:", "INFO:root:Method:              greedy-leiden",
                  "INFO:root:Representatives:", "INFO:root:Chains searched:", "INFO:root:Representatives with all",
                  "INFO:root:Greedy clusters of at least", "INFO:root:Representatives' graph:",
                  "INFO:root:Number of clusters:", "INFO:root:Cluster size (med):",
                  "INFO:root:Modularity (representatives' graph):"]
        lines = [next(i for i, line in enumerate(logs.output) if line.startswith(start)) for start in starts]
        self.assertEqual(lines, sorted(lines))

    def test_saturated_share_uses_the_representatives(self):
        """The saturated share is over n_representatives, not over the final clusters (fewer here)."""
        stats = self.clusterer.get_cluster_statistics()
        self.assertLess(stats["n_clusters"], stats["n_representatives"])
        n_reps = stats["n_representatives"]
        for n_saturated, warned in ((n_reps // 2, False), (n_reps // 2 + 1, True)):
            with self.subTest(n_saturated=n_saturated):
                patched = dict(stats, n_saturated=n_saturated)
                with mock.patch.object(self.clusterer, "get_cluster_statistics", return_value=patched), \
                        self.assertLogs(level="INFO") as logs:
                    self.clusterer.print_statistics()
                self.assertIn(f"INFO:root:Representatives with all --greedy-max-neighbors neighbours at or above "
                              f"--greedy-threshold: {n_saturated} of {n_reps} ({n_saturated / n_reps:.2%}); their "
                              f"greedy clusters may be cut, and a larger --greedy-max-neighbors can change them.",
                              logs.output)
                warning = ("WARNING:root:Most greedy clusters are bounded by --greedy-max-neighbors, not "
                           "--greedy-threshold: a representative or singleton may score at or above "
                           "--greedy-threshold with an earlier representative.")
                self.assertEqual(warning in logs.output, warned)

    def test_indexes_released(self):
        self.assertIsNone(self.clusterer.db.index)
        self.assertIsNone(self.clusterer.graph)
        self.assertIsNone(self.clusterer.representatives)
        self.assertEqual(len(self.clusterer.chain_ids), self.n)


class _FailAfter:
    """Wraps a function: the call after the first ``n`` raises."""

    def __init__(self, real, n: int):
        self.real, self.n, self.calls = real, n, 0

    def __call__(self, *args, **kwargs):
        if self.calls >= self.n:
            raise RuntimeError("injected fault")
        self.calls += 1
        return self.real(*args, **kwargs)


class _Searcher:
    """An index stand-in: reconstruct_n and search pass through, and the search after the first ``fail_after`` raises
    (None: never)."""

    def __init__(self, index, fail_after=None):
        self.index, self.fail_after, self.searches = index, fail_after, 0

    def reconstruct_n(self, start, n):
        return self.index.reconstruct_n(start, n)

    def search(self, x, k):
        if self.fail_after is not None and self.searches >= self.fail_after:
            raise RuntimeError("injected fault")
        self.searches += 1
        return self.index.search(x, k)

    def __getattr__(self, name):
        return getattr(self.index, name)


class ResumeTest(_Fixtures, unittest.TestCase):
    """D3: greedy checkpoints every _CHECKPOINT_SECONDS and resumes after a fault, the graph restarts from block 0
    without graph.done, the representatives' database is rebuilt without its marker, and Leiden-only reruns search
    nothing. Blocks of 50 rows (patched row cap), checkpoints after every block (_CHECKPOINT_SECONDS 0)."""

    rows = 50

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.dbs = cls.build_fixtures(cls.folder)
        cls.reference = {}
        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", cls.rows):
            for name, db_path in cls.dbs.items():
                for method in ("greedy", "greedy-leiden"):
                    cls.reference[name, method] = cls.cluster_run(db_path, method, cls.folder / f"ref-{name}-{method}")
                plain = EmbeddingClusterer(db_path)
                plain.cluster_greedy(threshold=0.9, max_neighbors=1000)
                cls.reference[name, "plain"] = plain.clusters, plain.representatives

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def cluster_run(db_path, method, work, out=None, clusterer=None, resolution=1.0):
        clusterer = clusterer or EmbeddingClusterer(db_path)
        if method == "greedy":
            clusterer.cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
        else:
            clusterer.cluster_greedy_leiden(work, threshold=0.6, max_neighbors=10, seed=4, resolution=resolution)
        out = Path(out or f"{work}.tsv")
        clusterer.export_clusters(str(out))
        return out.read_bytes()

    def test_reference(self):
        for name in self.dbs:
            with self.subTest(db=name):
                greedy = self.reference[name, "greedy"].decode().splitlines()
                clusters, reps = self.reference[name, "plain"]
                self.assertEqual(len(greedy), len(clusters))
                self.assertEqual(len(set(r.split("\t")[3] for r in greedy)), len(reps))
                self.assertLess(len(reps), len(clusters))

    def test_greedy_resumed_after_a_fault(self):
        """A fault after the state of block b is written; the same call again gives the uninterrupted file."""
        for name, db_path in self.dbs.items():
            n_blocks = math.ceil(len(EmbeddingClusterer(db_path).chain_ids) / self.rows)
            for method in ("greedy", "greedy-leiden"):
                for b in (1, 2, 5, n_blocks - 2):
                    with self.subTest(db=name, method=method, b=b):
                        work = self.folder / f"fault-{name}-{method}-{b}"
                        failing = _FailAfter(clustering._WorkFolder.save_greedy_state, b)
                        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows), \
                                mock.patch.object(clustering, "_CHECKPOINT_SECONDS", 0), \
                                mock.patch.object(clustering._WorkFolder, "save_greedy_state",
                                                  lambda self, *a, **k: failing(self, *a, **k)), \
                                self.assertRaisesRegex(RuntimeError, "injected fault"):
                            self.cluster_run(db_path, method, work)
                        state = np.load(work / "greedy.state.npz")
                        self.assertEqual(int(state["next_start"]), b * self.rows)
                        self.assertFalse(bool(state["done"]))
                        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows), \
                                self.assertLogs(level="INFO") as logs:
                            self.assertEqual(self.cluster_run(db_path, method, work), self.reference[name, method])
                        self.assertTrue(any(f"Greedy pass resumed from the work folder at chain {b * self.rows} of"
                                            in line for line in logs.output), logs.output)
                        record = json.loads((work / "run.json").read_text())
                        self.assertEqual(len(record["runs"]), 2)
                        self.assertEqual(record["runs"][1]["steps"]["greedy"]["resumed_at"], b * self.rows)

    def test_greedy_resumed_after_a_fault_inside_a_block(self):
        """--method greedy: a search that fails in the middle of the pass; the blocks after the last state are
        searched again, so the resumed file is the uninterrupted one."""
        for name, db_path in self.dbs.items():
            for fail_after in (3, 7):
                with self.subTest(db=name, fail_after=fail_after):
                    work = self.folder / f"inside-{name}-{fail_after}"
                    clusterer = EmbeddingClusterer(db_path)
                    clusterer.db.index = _Searcher(clusterer.db.index, fail_after)
                    with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows), \
                            mock.patch.object(clustering, "_CHECKPOINT_SECONDS", 0), \
                            self.assertRaisesRegex(RuntimeError, "injected fault"):
                        clusterer.cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
                    self.assertGreater(int(np.load(work / "greedy.state.npz")["next_start"]), 0)
                    with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows):
                        self.assertEqual(self.cluster_run(db_path, "greedy", work), self.reference[name, "greedy"])

    def test_no_checkpoint_before_the_interval(self):
        """With the default interval, a pass shorter than it writes its state only at the end: a fault leaves none,
        and the rerun starts from chain 0."""
        name, db_path = "flat", self.dbs["flat"]
        work = self.folder / "interval"
        clusterer = EmbeddingClusterer(db_path)
        clusterer.db.index = _Searcher(clusterer.db.index, 5)
        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows), \
                self.assertRaisesRegex(RuntimeError, "injected fault"):
            clusterer.cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
        self.assertFalse((work / "greedy.state.npz").exists())
        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows):
            self.assertEqual(self.cluster_run(db_path, "greedy", work), self.reference[name, "greedy"])

    def test_checkpoint_interval(self):
        """A state is written once _CHECKPOINT_SECONDS have passed since the last one (a clock of 400 s a block: every
        third block), with the rate in searched chains a second, then at the end with done set."""
        db_path = self.dbs["ivf_pq"]
        work = self.folder / "clock"
        ticks = iter(range(0, 10 ** 6, 400))
        saved = []
        real = clustering._WorkFolder.save_greedy_state

        def save(folder, labels, next_start, *args, done):
            saved.append((next_start, done))
            return real(folder, labels, next_start, *args, done=done)

        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows), \
                mock.patch.object(clustering, "_clock", lambda: next(ticks)), \
                mock.patch.object(clustering._WorkFolder, "save_greedy_state", save), \
                self.assertLogs(level="INFO") as logs:
            EmbeddingClusterer(db_path).cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
        self.assertEqual(saved, [(150, False), (300, False), (450, False), (600, False), (750, False),
                                 (900, False), (1000, True)])
        lines = [line for line in logs.output if "Greedy checkpoint at chain" in line]
        self.assertEqual(len(lines), 6)
        self.assertRegex(lines[0], r"^INFO:root:Greedy checkpoint at chain 150 of 1000: \d+ representatives, \d+ "
                                   r"chains searched, \d+\.\d searched chains a second since the last checkpoint$")
        state = np.load(work / "greedy.state.npz")
        self.assertTrue(bool(state["done"]))
        self.assertEqual(float(state["seconds"]), 400.0 * 20)

    def test_finished_greedy_is_reused(self):
        """A rerun of a finished pass searches nothing (the index is a stand-in that refuses to search)."""
        for name, db_path in self.dbs.items():
            with self.subTest(db=name):
                work = self.folder / f"reuse-{name}"
                with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows):
                    self.cluster_run(db_path, "greedy", work)
                    clusterer = EmbeddingClusterer(db_path)
                    clusterer.db.index = _Searcher(clusterer.db.index, 0)
                    self.assertEqual(self.cluster_run(db_path, "greedy", work, clusterer=clusterer),
                                     self.reference[name, "greedy"])
                clusters, reps = self.reference[name, "plain"]
                np.testing.assert_array_equal(clusterer.clusters, clusters)
                np.testing.assert_array_equal(clusterer.representatives, reps)
                record = json.loads((work / "run.json").read_text())
                self.assertEqual(record["runs"][1]["steps"]["greedy"]["skipped"], True)

    def test_greedy_folder_continued_by_greedy_leiden(self):
        """--method greedy's work folder, then greedy-leiden at the same greedy settings: the pass is reused."""
        db_path = self.dbs["ivf_pq"]
        work = self.folder / "continued"
        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows):
            self.cluster_run(db_path, "greedy", work)
            self.assertEqual(self.cluster_run(db_path, "greedy-leiden", work),
                             self.reference["ivf_pq", "greedy-leiden"])
        record = json.loads((work / "run.json").read_text())
        self.assertEqual([r["method"] for r in record["runs"]], ["greedy", "greedy-leiden"])
        self.assertTrue(record["runs"][1]["steps"]["greedy"]["skipped"])
        self.assertEqual(record["graph"], {"threshold": 0.6, "max_neighbors": 10, "edge_rule": "rescue"})

    def test_graph_restarts_from_block_0(self):
        """A fault after b blocks of the graph's edges were written: no graph.done or graph.json; the rerun reuses the
        greedy pass and the representatives' database, writes the edge files again from block 0 (the same bytes) and
        gives the uninterrupted file."""
        for name, db_path in self.dbs.items():
            ref_work = self.folder / f"ref-{name}-greedy-leiden"
            for b in (1, 3):
                with self.subTest(db=name, b=b):
                    work = self.folder / f"graph-{name}-{b}"
                    real = EmbeddingClusterer._search_edge_blocks

                    def failing(clusterer, threshold, k, edge_rule, write):
                        return real(clusterer, threshold, k, edge_rule, _FailAfter(write, b))

                    with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows), \
                            mock.patch.object(EmbeddingClusterer, "_search_edge_blocks", failing), \
                            self.assertRaisesRegex(RuntimeError, "injected fault"):
                        self.cluster_run(db_path, "greedy-leiden", work)
                    self.assertTrue((work / "representatives.done").exists())
                    self.assertFalse((work / "graph.done").exists())
                    self.assertFalse((work / "graph.json").exists())
                    self.assertGreater((work / "graph.weights.f32").stat().st_size, 0)
                    with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows):
                        self.assertEqual(self.cluster_run(db_path, "greedy-leiden", work),
                                         self.reference[name, "greedy-leiden"])
                    for file in ("graph.pairs.i32", "graph.weights.f32", "graph.json"):
                        self.assertEqual((work / file).read_bytes(), (ref_work / file).read_bytes(), file)
                    steps = json.loads((work / "run.json").read_text())["runs"][1]["steps"]
                    self.assertTrue(steps["greedy"]["skipped"])
                    self.assertTrue(steps["representatives"]["skipped"])
                    self.assertNotIn("skipped", steps["graph"])

    def test_representatives_rebuilt_without_their_marker(self):
        for name, db_path in self.dbs.items():
            with self.subTest(db=name):
                work = self.folder / f"reps-{name}"
                with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows):
                    self.cluster_run(db_path, "greedy-leiden", work)
                    index_bytes = (work / "representatives.index").read_bytes()
                    (work / "representatives.done").unlink()
                    with mock.patch.object(FaissEmbeddingDatabase, "subset", autospec=True,
                                           side_effect=FaissEmbeddingDatabase.subset) as subset:
                        self.assertEqual(self.cluster_run(db_path, "greedy-leiden", work),
                                         self.reference[name, "greedy-leiden"])
                self.assertEqual(subset.call_count, 1)
                self.assertTrue(subset.call_args.kwargs["overwrite"])
                self.assertEqual((work / "representatives.index").read_bytes(), index_bytes)
                self.assertTrue((work / "representatives.done").exists())
                self.assertTrue((work / "graph.done").exists())

    def test_greedy_rerun_invalidates_the_later_steps(self):
        """Without greedy's state (removed), the pass runs again, and the representatives' database and the graph are
        made again although their markers were there."""
        db_path = self.dbs["ivf_pq"]
        work = self.folder / "greedy-again"
        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows):
            self.cluster_run(db_path, "greedy-leiden", work)
            (work / "greedy.state.npz").unlink()
            with mock.patch.object(FaissEmbeddingDatabase, "subset", autospec=True,
                                   side_effect=FaissEmbeddingDatabase.subset) as subset, \
                    mock.patch.object(EmbeddingClusterer, "_search_edges", autospec=True,
                                      side_effect=EmbeddingClusterer._search_edges) as search_edges:
                self.assertEqual(self.cluster_run(db_path, "greedy-leiden", work),
                                 self.reference["ivf_pq", "greedy-leiden"])
        self.assertEqual((subset.call_count, search_edges.call_count), (1, 1))
        steps = json.loads((work / "run.json").read_text())["runs"][1]["steps"]
        self.assertFalse(any(step.get("skipped") for step in steps.values()), steps)

    def test_leiden_only_rerun_searches_nothing(self):
        """A rerun at another resolution reuses greedy, the representatives and the graph: no FAISS search (the
        parent's index is a stand-in that refuses to search or read), no representatives' database loaded or made;
        its file equals a run from scratch at that resolution."""
        for name, db_path in self.dbs.items():
            with self.subTest(db=name):
                work = self.folder / f"leiden-only-{name}"
                with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", self.rows):
                    first = self.cluster_run(db_path, "greedy-leiden", work)
                    fresh = self.cluster_run(db_path, "greedy-leiden", self.folder / f"fresh-{name}",
                                             resolution=30.0)
                    clusterer = EmbeddingClusterer(db_path)
                    searcher = _Searcher(clusterer.db.index, 0)
                    searcher.reconstruct_n = mock.Mock(side_effect=AssertionError("reconstruct_n"))
                    clusterer.db.index = searcher
                    with mock.patch.object(FaissEmbeddingDatabase, "load_database",
                                           side_effect=AssertionError("load_database")), \
                            mock.patch.object(FaissEmbeddingDatabase, "subset", side_effect=AssertionError("subset")):
                        again = self.cluster_run(db_path, "greedy-leiden", work,
                                                 out=self.folder / f"again-{name}.tsv", clusterer=clusterer,
                                                 resolution=30.0)
                self.assertEqual(again, fresh)
                self.assertNotEqual(again, first)
                record = json.loads((work / "run.json").read_text())
                self.assertEqual([r["leiden"]["resolution"] for r in record["runs"]], [1.0, 30.0])
                self.assertTrue(all(step.get("skipped") for step in record["runs"][1]["steps"].values()))


class EdgeFilesTest(unittest.TestCase):
    """build_similarity_edges(work_folder=...): the edge files hold the in-memory edges in order, by bits; graph.json
    the counts; from_folder memory-maps them, refuses files that do not match, and handles a graph with no edge."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        vectors = _clustered_vectors(seed=7, n=1000, d=64, n_centers=40, noise=0.8)
        cls.db_path = _build(cls.folder, "pq", vectors, IndexType.ivf_pq, IndexConfig(**_PQ_CONFIG))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_edges_as_in_memory(self):
        for rule in (EdgeRule.rescue, EdgeRule.lower):
            for k in (5, 10, 1000):
                with self.subTest(rule=rule.value, k=k):
                    memory = EmbeddingClusterer(self.db_path).build_similarity_edges(0.5, k, rule)
                    work = self.folder / f"edges-{rule.value}-{k}"
                    clusterer = EmbeddingClusterer(self.db_path)
                    files = clusterer.build_similarity_edges(0.5, k, rule, work_folder=work)
                    self.assertIs(clusterer.edges, files)
                    self.assertEqual(clusterer.min_weight, memory.min_weight)
                    self.assertIsInstance(files.pair_blocks[0], np.memmap)
                    pairs = np.concatenate(memory.pair_blocks)
                    weights = np.concatenate(memory.weight_blocks)
                    self.assertEqual(pairs.dtype, np.int32)
                    self.assertEqual(np.asarray(files.pair_blocks[0]).tobytes(), pairs.tobytes())
                    self.assertEqual(np.asarray(files.weight_blocks[0]).tobytes(), weights.tobytes())
                    self.assertEqual((work / "graph.pairs.i32").read_bytes(), pairs.tobytes())
                    self.assertEqual((work / "graph.weights.f32").read_bytes(), weights.tobytes())
                    self.assertEqual((files.n, files.n_edges, files.n_rescued_chains, files.n_rescued_edges,
                                      files.min_weight),
                                     (memory.n, memory.n_edges, memory.n_rescued_chains, memory.n_rescued_edges,
                                      memory.min_weight))
                    record = json.loads((work / "graph.json").read_text())
                    self.assertEqual(record, {"n": 1000, "n_edges": memory.n_edges,
                                              "n_rescued_chains": memory.n_rescued_chains,
                                              "n_rescued_edges": memory.n_rescued_edges,
                                              "min_weight": memory.min_weight, "pairs_dtype": "int32",
                                              "threshold": 0.5, "k": min(k, 1000), "edge_rule": rule.value})
                    self.assertTrue((work / "graph.done").exists())
                    if rule is EdgeRule.rescue and k == 5:
                        self.assertGreater(memory.n_rescued_edges, 0)
                    # The graph and Leiden from the files equal those from memory
                    direct = EmbeddingClusterer(self.db_path)
                    direct.build_similarity_graph(0.5, k, rule)
                    np.testing.assert_array_equal(clusterer.cluster_leiden(seed=2), direct.cluster_leiden(seed=2))

    def test_no_edge(self):
        work = self.folder / "empty"
        edges = EmbeddingClusterer(self.db_path).build_similarity_edges(2.0, 10, work_folder=work)
        self.assertEqual((edges.n, edges.n_edges, edges.min_weight, edges.pair_blocks), (1000, 0, float("inf"), []))
        self.assertEqual((work / "graph.pairs.i32").stat().st_size, 0)
        self.assertIsNone(json.loads((work / "graph.json").read_text())["min_weight"])

    def test_refusals(self):
        work = self.folder / "broken"
        EmbeddingClusterer(self.db_path).build_similarity_edges(0.5, 10, work_folder=work)
        with open(work / "graph.weights.f32", "ab") as f:
            f.write(b"\0\0\0\0")
        with self.assertRaisesRegex(ValueError, r"do not hold the \d+ edges of its graph.json"):
            SimilarityEdges.from_folder(work)
        (work / "graph.done").unlink()
        with self.assertRaisesRegex(ValueError, "graph.done is missing"):
            SimilarityEdges.from_folder(work)

    def test_restart_replaces_the_files(self):
        """A second build into the same folder replaces the files (here at another threshold)."""
        work = self.folder / "twice"
        EmbeddingClusterer(self.db_path).build_similarity_edges(0.3, 10, work_folder=work)
        second = EmbeddingClusterer(self.db_path).build_similarity_edges(0.6, 10, work_folder=work)
        memory = EmbeddingClusterer(self.db_path).build_similarity_edges(0.6, 10)
        self.assertEqual((work / "graph.weights.f32").read_bytes(), np.concatenate(memory.weight_blocks).tobytes())
        self.assertEqual(second.n_edges, memory.n_edges)


class WorkFolderRefusalTest(_Fixtures, unittest.TestCase):
    """run.json: every recorded setting and the parent's fingerprint are compared on a rerun; a difference is refused
    (exit status 2) naming each field with both values; --nprobe equal to the stored value is no difference; a moved
    parent with the same files, another FAISS thread count, and a work folder of --method greedy are accepted."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.dbs = cls.build_fixtures(cls.folder)
        cls.work = cls.folder / "work"
        cls.base = ("cluster", "--db-path", cls.dbs["ivf_pq"], "--method", "greedy-leiden", "--work-folder", cls.work,
                    "--max-neighbors", "10", "--threshold", "0.6", "--seed", "1")
        result = _invoke(*cls.base, "--output-file", cls.folder / "first.tsv")
        assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
        cls.first = (cls.folder / "first.tsv").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def refused(self, *args, db=None):
        args = list(self.base) + [str(a) for a in args] + ["--output-file", str(self.folder / "refused.tsv")]
        if db is not None:
            args[2] = db
        result = _invoke(*args)
        self.assertEqual(result.exit_code, 2, f"{result.output}\n{result.exception!r}")
        self.assertFalse((self.folder / "refused.tsv").exists())
        return _flat(result.output)

    def message(self, *differences) -> str:
        return (f"Invalid value for --work-folder: The work folder {self.work} holds the steps of a run with other "
                f"settings or another database: {'; '.join(differences)}. Use a new --work-folder for this run, or "
                f"rerun with the recorded settings.")

    def test_each_setting(self):
        cases = [
            (("--threshold", "0.65"), "graph threshold 0.6 (this run: 0.65)"),
            (("--max-neighbors", "12"), "graph max_neighbors 10 (this run: 12)"),
            (("--edge-rule", "lower"), "graph edge_rule 'rescue' (this run: 'lower')"),
            (("--greedy-threshold", "0.95"), "greedy threshold 0.9 (this run: 0.95)"),
            (("--greedy-max-neighbors", "500"), "greedy max_neighbors 1000 (this run: 500)"),
            (("--nprobe", "8"), "nprobe 4 (this run: 8)"),
        ]
        for extra, difference in cases:
            with self.subTest(extra=extra):
                self.assertIn(self.message(difference), self.refused(*extra))
        # Several in one message, in the record's order
        self.assertIn(self.message("nprobe 4 (this run: 8)", "greedy threshold 0.9 (this run: 0.95)",
                                   "graph edge_rule 'rescue' (this run: 'lower')"),
                      self.refused("--edge-rule", "lower", "--nprobe", "8", "--greedy-threshold", "0.95"))
        # --threshold given as the recorded value is no difference; the resolved defaults are compared
        ok = _invoke(*self.base, "--nprobe", "4", "--threshold", "0.60", "--greedy-threshold", "0.9",
                     "--greedy-max-neighbors", "1000", "--edge-rule", "rescue", "--output-file",
                     self.folder / "same.tsv")
        self.assertEqual(ok.exit_code, 0, f"{ok.output}\n{ok.exception!r}")
        self.assertEqual((self.folder / "same.tsv").read_bytes(), self.first)

    def test_nprobe_above_the_cell_count(self):
        """run.json records and compares the cells probed: on nlist 16, --nprobe 21, 16 and 30 are one search; the
        stored nprobe 4 is another."""
        work = self.folder / "cells"
        args = list(self.base)
        args[args.index("--work-folder") + 1] = str(work)
        files = []
        for nprobe in ("21", "16", "30"):
            with self.subTest(nprobe=nprobe):
                out = self.folder / f"cells-{nprobe}.tsv"
                result = _invoke(*args, "--nprobe", nprobe, "--output-file", out)
                self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
                self.assertEqual(json.loads((work / "run.json").read_text())["nprobe"], 16)
                files.append(out.read_bytes())
        self.assertEqual(files[1:], files[:1] * 2)
        result = _invoke(*args, "--output-file", self.folder / "cells-stored.tsv")
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn(f"The work folder {work} holds the steps of a run with other settings or another database: "
                      f"nprobe 16 (this run: 4).", _flat(result.output))

    def test_another_parent(self):
        """The fingerprint: another database (here the flat one, so also another chain count), and a copy of the
        parent with one changed metadata file (the same chain count)."""
        output = self.refused(db=self.dbs["flat"])
        self.assertIn("parent chain count 1000 (this run: 600)", output)
        self.assertIn("parent .index sha256", output)
        self.assertIn("nprobe 4 (this run: None)", output)
        copy = self.folder / "copy"
        copy.mkdir(exist_ok=True)
        for suffix in (".index", ".invlists", ".metadata"):
            shutil.copy(f"{self.dbs['ivf_pq']}{suffix}", copy / f"pq{suffix}")
        changed = FaissEmbeddingDatabase(copy, "pq")
        changed.load_database()
        changed.chain_ids = [f"other{i}" for i in range(len(changed.chain_ids))]
        changed._save()
        output = self.refused(db=copy / "pq")
        record = json.loads((self.work / "run.json").read_text())
        old = record["parent"]["metadata_sha256"]
        new = changed._fingerprint()["metadata_sha256"]
        self.assertIn(self.message(f"parent .metadata sha256 {old!r} (this run: {new!r})"), output)

    def test_moved_parent_is_accepted(self):
        moved = self.folder / "moved"
        moved.mkdir(exist_ok=True)
        for suffix in (".index", ".invlists", ".metadata"):
            shutil.copy(f"{self.dbs['ivf_pq']}{suffix}", moved / f"pq{suffix}")
        args = list(self.base) + ["--output-file", str(self.folder / "moved.tsv")]
        args[2] = str(moved / "pq")
        result = _invoke(*args)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        self.assertEqual((self.folder / "moved.tsv").read_bytes(), self.first)
        self.assertIn(f"The parent database is now {(moved / 'pq').resolve()}", result.output)

    def test_other_thread_count_is_a_warning(self):
        # The first run is on this process's FAISS thread count (1 on macOS, OMP_NUM_THREADS elsewhere).
        threads = faiss.omp_get_max_threads()
        other = 7 if threads != 7 else 8
        work = self.folder / "threads"
        clusterer = EmbeddingClusterer(self.dbs["flat"])
        clusterer.cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
        with mock.patch.object(clustering.faiss, "omp_get_max_threads", return_value=other), \
                self.assertLogs(level="INFO") as logs:
            EmbeddingClusterer(self.dbs["flat"]).cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
        self.assertIn(f"WARNING:root:FAISS runs on {other} threads; this work folder's earlier steps ran on {threads}: "
                      f"the graph can differ by a few edges from a run at one thread count.", logs.output)
        record = json.loads((work / "run.json").read_text())
        self.assertEqual([r["faiss_threads"] for r in record["runs"]], [threads, other])

    def test_other_version_is_a_warning(self):
        work = self.folder / "versions"
        EmbeddingClusterer(self.dbs["flat"]).cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
        with mock.patch.object(clustering, "__version__", "9.9.9"), self.assertLogs(level="INFO") as logs:
            EmbeddingClusterer(self.dbs["flat"]).cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
        self.assertTrue(any(line.startswith("WARNING:root:This work folder's earlier steps ran on other FoldMatch or "
                                            "FAISS versions") for line in logs.output), logs.output)

    def test_greedy_settings_under_greedy(self):
        """--method greedy records --threshold and --max-neighbors as the greedy step's."""
        work = self.folder / "greedy-only"
        base = ("cluster", "--db-path", self.dbs["flat"], "--method", "greedy", "--work-folder", work,
                "--output-file", self.folder / "g.tsv")
        self.assertEqual(_invoke(*base).exit_code, 0)
        record = json.loads((work / "run.json").read_text())
        self.assertEqual((record["greedy"], record["graph"], record["nprobe"]),
                         ({"threshold": 0.8, "max_neighbors": 1000}, None, None))
        result = _invoke(*base, "--threshold", "0.85")
        self.assertEqual(result.exit_code, 2)
        self.assertIn("greedy threshold 0.8 (this run: 0.85)", _flat(result.output))
        result = _invoke(*base, "--max-neighbors", "20")
        self.assertIn("greedy max_neighbors 1000 (this run: 20)", _flat(result.output))

    def test_foreign_folders(self):
        """Work files without run.json (an older layout), a run.json of another format or not JSON: refused."""
        cases = []
        older = self.folder / "older"
        older.mkdir()
        (older / "greedy.state.npz").write_bytes(b"")
        cases.append((older, f"The work folder {older} holds work files (greedy.state.npz) but no run.json, as an "
                             f"older FoldMatch version leaves it: use a new --work-folder."))
        other = self.folder / "other-format"
        other.mkdir()
        (other / "run.json").write_text(json.dumps({"format": 0}))
        cases.append((other, f"The work folder {other} was made by another FoldMatch layout (run.json format 0, this "
                             f"version writes 1): use a new --work-folder."))
        garbage = self.folder / "garbage"
        garbage.mkdir()
        (garbage / "run.json").write_text("{")
        cases.append((garbage, f"The work folder {garbage} has a run.json that is not readable"))
        for folder, message in cases:
            with self.subTest(folder=folder.name):
                args = list(self.base)
                args[6] = folder
                result = _invoke(*args, "--output-file", self.folder / "foreign.tsv")
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn(f"Invalid value for --work-folder: {message}", _flat(result.output))
        # Other files are left alone: a folder holding only an earlier output file is used
        mine = self.folder / "mine"
        mine.mkdir()
        (mine / "clusters.tsv").write_text("x\n")
        args = list(self.base)
        args[6] = mine
        result = _invoke(*args, "--output-file", mine / "clusters.tsv")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual((mine / "clusters.tsv").read_bytes(), self.first)

    def test_output_file_over_a_work_file(self):
        for name in ("run.json", "graph.json", "graph.weights.f32", "representatives.index"):
            with self.subTest(name=name):
                before = (self.work / name).read_bytes()
                result = _invoke(*self.base, "--output-file", self.work / name)
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn(f"--output-file {self.work / name} is the work file {self.work / name}: the results "
                              f"would replace an input of this command.", str(result.exception))
                self.assertEqual((self.work / name).read_bytes(), before)

    def test_corrupt_state(self):
        work = self.folder / "corrupt"
        EmbeddingClusterer(self.dbs["flat"]).cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)
        with np.load(work / "greedy.state.npz") as data:
            state = {key: data[key] for key in data.files}
        state["labels"] = state["labels"][:-1]
        with open(work / "greedy.state.npz", "wb") as f:
            np.savez(f, **state)
        with self.assertRaisesRegex(ValueError, r"greedy.state.npz does not fit this database \(600 chains"):
            EmbeddingClusterer(self.dbs["flat"]).cluster_greedy(threshold=0.9, max_neighbors=1000, work_folder=work)


class MethodRefusalTest(unittest.TestCase):
    """cluster's option table (D7) and the other refusals before the load (exit status 2); the hnsw refusal and the
    negative-weight refusal after it."""

    def invoke(self, *args):
        with mock.patch.object(clustering, "EmbeddingClusterer", side_effect=AssertionError("the database was loaded")):
            return _invoke(*args)

    def refused(self, *args) -> str:
        result = self.invoke(*args)
        self.assertEqual(result.exit_code, 2, f"{result.output}\n{result.exception!r}")
        return _flat(result.output)

    def test_option_table(self):
        base = ("cluster", "--db-path", "no/such/db")
        gl = "(--method greedy-leiden)"
        cases = [
            (("--method", "leiden", "--greedy-threshold", "0.9"),
             f"--method leiden does not take --greedy-threshold {gl}. Remove it, or use a method that takes it."),
            (("--greedy-max-neighbors", "5"),
             f"--method leiden does not take --greedy-max-neighbors {gl}. Remove it, or use a method that takes it."),
            (("--work-folder", "w"), "--method leiden does not take --work-folder (--method greedy or greedy-leiden). "
                                     "Remove it, or use a method that takes it."),
            (("--method", "greedy", "--greedy-threshold", "0.9", "--greedy-max-neighbors", "5"),
             "--method greedy does not take --greedy-threshold (--method greedy-leiden; --method greedy takes its "
             "threshold from --threshold), --greedy-max-neighbors (--method greedy-leiden; --method greedy takes its "
             "neighbours from --max-neighbors). Remove them, or use a method that takes them."),
            (("--method", "greedy-leiden", "--work-folder", "w", "--use-gpu-index"),
             "--method greedy-leiden does not take --use-gpu-index (--method leiden or greedy; the representatives' "
             "database is built from the CPU index; GPU search is not supported for this method yet). Remove it, or "
             "use a method that takes it."),
            (("--method", "greedy", "--resolution", "2", "--edge-rule", "lower", "--greedy-threshold", "0.9",
              "--work-folder", "w", "--use-gpu-index"),
             "--method greedy does not take --resolution (--method leiden or greedy-leiden), --edge-rule (--method "
             "leiden or greedy-leiden), --greedy-threshold (--method greedy-leiden; --method greedy takes its "
             "threshold from --threshold). Remove them, or use a method that takes them."),
            (("--work-folder", "w", "--greedy-threshold", "0.9", "--seed", "-1", "--leiden-iterations", "0"),
             f"--method leiden does not take --greedy-threshold {gl}, --work-folder (--method greedy or "
             f"greedy-leiden). Remove them, or use a method that takes them."),
        ]
        for extra, message in cases:
            with self.subTest(extra=extra):
                output = self.refused(*base, *extra)
                self.assertIn(f"Invalid value for --method: {message}", output)
                self.assertEqual(output.count("does not take"), 1)

    def test_taken_options_are_not_refused(self):
        base = ("cluster", "--db-path", "no/such/db")
        cases = [("--method", "greedy-leiden", "--work-folder", "w", "--greedy-threshold", "0.8",
                  "--greedy-max-neighbors", "5", "--resolution", "3", "--leiden-iterations", "2", "--seed", "1",
                  "--edge-rule", "lower", "--nprobe", "3", "--max-neighbors", "7"),
                 ("--method", "greedy", "--work-folder", "w", "--use-gpu-index", "--nprobe", "2"),
                 ("--method", "leiden", "--use-gpu-index", "--edge-rule", "rescue")]
        for extra in cases:
            with self.subTest(extra=extra):
                result = self.invoke(*base, *extra)
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertEqual(str(result.exception), "the database was loaded")

    def test_work_folder_required(self):
        output = self.refused("cluster", "--db-path", "no/such/db", "--method", "greedy-leiden")
        self.assertIn("Invalid value for --work-folder: --method greedy-leiden needs --work-folder: a folder for its "
                      "intermediate results (greedy labels, the representatives' database, the representatives' "
                      "graph), from which a run that stops is resumed.", output)

    def test_ranges(self):
        base = ("cluster", "--db-path", "no/such/db", "--method", "greedy-leiden", "--work-folder", "w")
        for extra, message in ((("--greedy-max-neighbors", "0"), "Invalid value for --greedy-max-neighbors: "
                                                                 "--greedy-max-neighbors must be at least 1."),
                               (("--max-neighbors", "0"), "Invalid value for --max-neighbors: --max-neighbors must be "
                                                          "at least 1."),
                               (("--leiden-iterations", "0"), "Invalid value for --leiden-iterations: "
                                                              "--leiden-iterations must be at least 1."),
                               (("--seed", "-2"), "Invalid value for --seed: --seed must be at least 0."),
                               (("--nprobe", "0"), "Invalid value for --nprobe: --nprobe must be at least 1.")):
            with self.subTest(extra=extra):
                self.assertIn(message, self.refused(*base, *extra))
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "file"
            file.write_text("")
            output = self.refused("cluster", "--db-path", "no/such/db", "--method", "greedy-leiden", "--work-folder",
                                  file)
            self.assertIn(f"Invalid value for --work-folder: {file} is a file, not a folder.", output)

    def test_help_defaults(self):
        """The defaults the help states are the code's constants."""
        import typer
        command = typer.main.get_command(search.app).commands["cluster"]
        helps = {p.name: p.help for p in command.params}
        self.assertIn(f"Default: {clustering._GREEDY_LEIDEN_GREEDY_THRESHOLD}.", helps["greedy_threshold"])
        self.assertIn(f"Default: {clustering._GREEDY_LEIDEN_GREEDY_MAX_NEIGHBORS}.", helps["greedy_max_neighbors"])
        self.assertIn(f"Default: 1000; {clustering._GREEDY_LEIDEN_MAX_NEIGHBORS} with --method greedy-leiden, where "
                      f"it sets the representatives' graph.", helps["max_neighbors"])
        self.assertEqual(helps["greedy_threshold"], "With --method greedy-leiden, a representative takes the "
                                                    "still-unclustered chains scoring at least this. Default: 0.9. "
                                                    "Refused with the other methods (use --threshold).")
        self.assertIn("Needs --work-folder and an ivf_pq or flat database.", helps["method"])
        self.assertTrue(helps["work_folder"].endswith("It may hold tens of GiB at 200M chains: put it on scratch."))
        for name in ("resolution", "leiden_iterations"):
            self.assertTrue(helps[name].endswith("Refused with --method greedy."), helps[name])

    def test_method_values(self):
        output = self.refused("cluster", "--db-path", "no/such/db", "--method", "cascade")
        self.assertIn("--method", output)
        self.assertIn("greedy-leiden", output)


class AfterTheLoadRefusalTest(unittest.TestCase):
    """hnsw parents (an auto database that is hnsw included) are refused after the load, exit status 2, before the work
    folder is touched; an auto database that is flat is taken; a negative kept weight on the representatives' graph is
    refused with --method leiden's message."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        vectors = _clustered_vectors(seed=3, n=300, d=16, n_centers=20, noise=1.0)
        cls.hnsw = _build(cls.folder, "hnsw", vectors, IndexType.hnsw)
        with mock.patch.object(FaissEmbeddingDatabase, "N_EMBEDDINGS_DB_THR", 100):
            cls.auto_hnsw = _build(cls.folder, "auto_hnsw", vectors, IndexType.auto)
        cls.auto_flat = _build(cls.folder, "auto_flat", vectors, IndexType.auto)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_hnsw_refused(self):
        message = ("Invalid value for --method: --method greedy-leiden needs an ivf_pq or flat database; this one is "
                   "hnsw. Use --method leiden, or build the database with --index-type ivf_pq or flat.")
        for db_path in (self.hnsw, self.auto_hnsw):
            with self.subTest(db=db_path):
                work = self.folder / f"work-{Path(db_path).name}"
                result = _invoke("cluster", "--db-path", db_path, "--method", "greedy-leiden", "--work-folder", work,
                                 "--output-file", self.folder / "h.tsv")
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn(message, _flat(result.output))
                self.assertFalse(work.exists())
        clusterer = EmbeddingClusterer(self.hnsw)
        with self.assertRaisesRegex(ValueError, "this one is hnsw"):
            clusterer.cluster_greedy_leiden(self.folder / "lib-work")
        self.assertIsNotNone(clusterer.db.index)

    def test_auto_flat_taken(self):
        work = self.folder / "auto-flat-work"
        result = _invoke("cluster", "--db-path", self.auto_flat, "--method", "greedy-leiden", "--work-folder", work,
                         "--threshold", "0.3", "--max-neighbors", "10", "--seed", "1", "--output-file",
                         self.folder / "af.tsv")
        self.assertEqual(result.exit_code, 0, result.output)
        reps = FaissEmbeddingDatabase(work, "representatives")
        reps.load_database()
        self.assertEqual(reps.index_type, IndexType.flat)

    def test_negative_weights(self):
        vectors = np.array([[1, 0, 0], [-0.6, 0.8, 0], [0, -0.6, 0.8]], dtype=np.float32)
        db_path = _build(self.folder, "neg", vectors, IndexType.flat)
        clusterer = EmbeddingClusterer(db_path)
        clusterer.build_similarity_graph(threshold=-1.0, max_neighbors=3)
        min_weight = clusterer.min_weight
        self.assertLess(min_weight, 0)
        out = self.folder / "neg.tsv"
        result = _invoke("cluster", "--db-path", db_path, "--method", "greedy-leiden", "--work-folder",
                         self.folder / "neg-work", "--threshold", "-1", "--output-file", out)
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn(f"Invalid value for --threshold: with --method greedy-leiden, Leiden's modularity needs "
                      f"non-negative edge weights, and this --threshold kept a score of {min_weight:.4g} on the "
                      f"representatives' graph. Use a --threshold of 0 or more with a new --work-folder (this one "
                      f"records this --threshold, so greedy runs again), or --method greedy.",
                      _flat(result.output))
        self.assertFalse(out.exists())
        # As the message says: the same folder at another --threshold is refused (run.json records -1.0)
        result = _invoke("cluster", "--db-path", db_path, "--method", "greedy-leiden", "--work-folder",
                         self.folder / "neg-work", "--threshold", "0.5", "--output-file", out)
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("graph threshold -1.0 (this run: 0.5)", _flat(result.output))


class NprobeBothSearchesTest(unittest.TestCase):
    """--nprobe under greedy-leiden reaches both searches: the parent's index and the representatives' database (its
    metadata, so a load keeps it), and run.json records the effective value."""

    def test_nprobe(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            vectors = _clustered_vectors(seed=7, n=1000, d=64, n_centers=40, noise=0.8)
            db_path = _build(folder, "pq", vectors, IndexType.ivf_pq, IndexConfig(**_PQ_CONFIG))
            made = []
            real_init = EmbeddingClusterer.__init__

            def init(clusterer, *args, **kwargs):
                real_init(clusterer, *args, **kwargs)
                made.append((kwargs.get("nprobe"), clusterer.nprobe,
                             faiss.extract_index_ivf(clusterer.db.index).nprobe))

            for nprobe in (None, 6):
                with self.subTest(nprobe=nprobe), mock.patch.object(EmbeddingClusterer, "__init__", init):
                    made.clear()
                    work = folder / f"work-{nprobe}"
                    extra = () if nprobe is None else ("--nprobe", nprobe)
                    result = _invoke("cluster", "--db-path", db_path, "--method", "greedy-leiden", "--work-folder",
                                     work, *extra, "--output-file", folder / "out.tsv")
                    self.assertEqual(result.exit_code, 0, result.output)
                    expected = 4 if nprobe is None else 6
                    self.assertEqual(made, [(nprobe, expected, expected), (expected, expected, expected)])
                    self.assertEqual(json.loads((work / "run.json").read_text())["nprobe"], expected)
                    reps = FaissEmbeddingDatabase(work, "representatives")
                    reps.load_database()
                    self.assertEqual(faiss.extract_index_ivf(reps.index).nprobe, expected)
                    self.assertEqual(reps.index_config.nprobe, expected)


if __name__ == "__main__":
    unittest.main()
