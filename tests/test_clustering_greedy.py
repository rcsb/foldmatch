"""EmbeddingClusterer.cluster_greedy and fm-search cluster --method greedy: greedy representatives in database order,
computed in streamed blocks that search only the chains no representative has taken at the block's start. On hnsw, and
on flat below FAISS's BLAS switch, the partition equals the sequential rule on one query per chain bit for bit, at every
block size; on ivf_pq it is checked against the result rows the method used. Also the cluster ids (by decreasing size,
ties by representative index), the statistics, the 4-column output, the refusals, and the Leiden output unchanged."""
import logging
import math
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# As the commands run (set_log_level): Lightning first, which brings torch and, through scikit-learn,
# KMP_DUPLICATE_LIB_OK; on macOS, FAISS's OpenMP regions abort a process that loaded torch's libomp without it.
import lightning  # noqa: F401
import faiss
import leidenalg
import numpy as np
from typer.testing import CliRunner

from foldmatch.cli import search
from foldmatch.cli.search import cluster_database
from foldmatch.search import clustering
from foldmatch.search.clustering import EmbeddingClusterer
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.search.output import CLUSTER_OUTPUT_FIELDS, GREEDY_CLUSTER_OUTPUT_FIELDS, format_spec, \
    write_cluster_results
from foldmatch.types.api_types import ClusterMethod, FormatMode, IndexConfig, IndexType


def _clustered_vectors(seed: int, n: int, d: int, n_centers: int, noise: float) -> np.ndarray:
    """Unit vectors around ``n_centers`` random centres, at a noise level of 0.5 to 1.5 times ``noise`` each; the
    members of a centre are contiguous (tests/test_clustering.py's fixture)."""
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


def _load(db_path: str):
    path = Path(db_path)
    db = FaissEmbeddingDatabase(path.parent, path.name)
    db.load_database()
    return db


def _budget(rows: int, k: int) -> int:
    """The result budget that gives blocks of ``rows`` chains at ``k`` neighbours."""
    return rows * k * 12


def _block_rows(n: int, k: int) -> int:
    """The rows of a block at the module's budget and row cap (build_similarity_graph's formula)."""
    return max(1, min(n, clustering._GRAPH_MAX_ROWS, clustering._GRAPH_RESULT_BYTES // (k * 12)))


def _one_query_results(index, n: int, k: int) -> list:
    """Each chain's result row from a search of its stored vector alone (flat and hnsw), as Python lists."""
    results = []
    for i in range(n):
        query = np.ascontiguousarray(index.reconstruct(i), dtype=np.float32).reshape(1, -1)
        scores, ids = index.search(query, k)
        results.append((scores[0].tolist(), ids[0].tolist()))
    return results


def _sequential(results: list, n: int, threshold: float, k: int, rows: int, max_neighbors: int = None):
    """The sequential rule on full result rows, in plain Python: chains in database order; a chain in a cluster is
    passed over; any other becomes a representative and takes every id of its row with id >= 0, a score at least the
    threshold (compared in float32, as NumPy compares float32 scores with a Python float) and no cluster yet. Blocks
    of ``rows`` chains only count the searched chains: those in no cluster at their block's start. Cluster ids by
    decreasing size, ties by representative index. Clusters at the cap: of at least ``max_neighbors`` (default k)
    chains."""
    threshold = float(np.float32(threshold))
    label = [-1] * n
    searched = saturated = 0
    for start in range(0, n, rows):
        end = min(start + rows, n)
        searched += sum(1 for i in range(start, end) if label[i] < 0)
        for i in range(start, end):
            if label[i] >= 0:
                continue
            label[i] = i
            scores, ids = results[i]
            for j, score in zip(ids, scores):
                if j >= 0 and score >= threshold and label[j] < 0:
                    label[j] = i
            if k < n and ids[-1] >= 0 and scores[-1] >= threshold:
                saturated += 1
    members = {}
    for i, rep in enumerate(label):
        members.setdefault(rep, []).append(i)
    reps = sorted(members, key=lambda r: (-len(members[r]), r))
    cluster_of = {rep: c for c, rep in enumerate(reps)}
    sizes = [len(members[r]) for r in reps]
    return SimpleNamespace(clusters=[cluster_of[label[i]] for i in range(n)], representatives=reps, sizes=sizes,
                           label=label, n_searched=searched, n_saturated=saturated,
                           n_at_cap=sum(1 for s in sizes if s >= (max_neighbors or k)), saturated_reps=[
                               r for r in reps if k < n and results[r][1][-1] >= 0
                               and results[r][0][-1] >= threshold])


class _GreedyCase(unittest.TestCase):

    def greedy(self, db_path: str, threshold: float, max_neighbors: int, rows: int = None):
        """cluster_greedy on a fresh clusterer, with blocks of ``rows`` chains (the default budget when None)."""
        clusterer = EmbeddingClusterer(db_path)
        budget = clustering._GRAPH_RESULT_BYTES if rows is None else _budget(rows, min(max_neighbors,
                                                                                         len(clusterer.chain_ids)))
        with mock.patch.object(clustering, "_GRAPH_RESULT_BYTES", budget):
            clusters = clusterer.cluster_greedy(threshold=threshold, max_neighbors=max_neighbors)
        self.assertIs(clusters, clusterer.clusters)
        self.assertEqual(clusterer.method, "greedy")
        return clusterer

    def assert_same_as(self, clusterer: EmbeddingClusterer, ref):
        self.assertEqual(clusterer.clusters.tolist(), ref.clusters)
        self.assertEqual(clusterer.representatives.tolist(), ref.representatives)
        self.assertEqual(np.bincount(clusterer.clusters).tolist(), ref.sizes)
        stats = clusterer.get_cluster_statistics()
        self.assertEqual((stats["n_saturated"], stats["n_at_cap"], stats["n_searched"]),
                         (ref.n_saturated, ref.n_at_cap, ref.n_searched))
        self.assertEqual((clusterer.n_saturated, clusterer.n_searched), (ref.n_saturated, ref.n_searched))

    def assert_cluster_ids(self, clusterer: EmbeddingClusterer):
        """Ids by non-increasing size, equal sizes in ascending representative index, and representatives[i] the
        representative of cluster i (a chain of cluster i)."""
        sizes = np.bincount(clusterer.clusters)
        reps = clusterer.representatives
        self.assertEqual(len(reps), len(sizes))
        self.assertTrue(np.all(np.diff(sizes) <= 0))
        ties = np.diff(sizes) == 0
        self.assertTrue(np.all(np.diff(reps)[ties] > 0))
        self.assertTrue(np.array_equal(clusterer.clusters[reps], np.arange(len(reps))))
        # The representative is the cluster's lowest chain index (every member comes after it in database order).
        lowest = np.full(len(reps), len(clusterer.clusters))
        np.minimum.at(lowest, clusterer.clusters, np.arange(len(clusterer.clusters)))
        self.assertTrue(np.array_equal(lowest, reps))


class HnswGreedyTest(_GreedyCase):
    """hnsw: FAISS results do not depend on the batch, so cluster_greedy equals the sequential rule bit for bit at
    every block size."""

    n, d, threshold, max_neighbors = 2000, 1536, 0.6, 40
    # A --max-neighbors small enough that it, not the threshold, bounds many clusters.
    small_k = 5

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        vectors = _clustered_vectors(seed=3, n=cls.n, d=cls.d, n_centers=150, noise=1.0)
        cls.db_path = _build(Path(cls.tmp.name), "hnsw", vectors, IndexType.hnsw)
        index = _load(cls.db_path).index
        cls.results = _one_query_results(index, cls.n, cls.max_neighbors)
        cls.results_small_k = _one_query_results(index, cls.n, cls.small_k)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def reference(self, rows=None, small=False):
        k = self.small_k if small else self.max_neighbors
        return _sequential(self.results_small_k if small else self.results, self.n, self.threshold, k,
                           rows or _block_rows(self.n, k))

    def test_fixture(self):
        ref = self.reference()
        sizes = np.array(ref.sizes)
        self.assertGreater(len(sizes), 100)
        self.assertGreater(np.count_nonzero(sizes == 1), 10)
        self.assertGreater(np.count_nonzero(sizes > 5), 50)
        self.assertNotEqual(self.n % 7, 0)
        # At small_k many representatives keep their k-th neighbour, some with clusters below k.
        small = self.reference(small=True)
        self.assertGreater(small.n_saturated, 50)
        self.assertGreater(small.n_at_cap, 10)
        self.assertTrue(any(small.sizes[small.representatives.index(r)] < self.small_k
                            for r in small.saturated_reps))

    def test_same_as_the_sequential_rule(self):
        for small in (False, True):
            with self.subTest(small_k=small):
                k = self.small_k if small else self.max_neighbors
                clusterer = self.greedy(self.db_path, self.threshold, k)
                self.assert_same_as(clusterer, self.reference(small=small))
                self.assertEqual(clusterer.n_searched, self.n)  # one block holds every chain
                self.assert_cluster_ids(clusterer)

    def test_block_sizes(self):
        """The same partition with blocks of 1, 7 (a short last block) and more than n chains; the searched chains
        are those in no cluster at their block's start: as many as clusters with blocks of 1, n with one block."""
        for small in (False, True):
            k = self.small_k if small else self.max_neighbors
            for rows in (1, 7, self.n + 5):
                with self.subTest(small_k=small, rows=rows):
                    clusterer = self.greedy(self.db_path, self.threshold, k, rows)
                    ref = self.reference(min(rows, self.n), small=small)
                    self.assert_same_as(clusterer, ref)
                    if rows == 1:
                        self.assertEqual(clusterer.n_searched, len(ref.sizes))
                    elif rows > self.n:
                        self.assertEqual(clusterer.n_searched, self.n)
                    else:
                        self.assertLess(len(ref.sizes), clusterer.n_searched)
                        self.assertLess(clusterer.n_searched, self.n)

    def test_blocks_read_and_searched(self):
        """Each block is read once with reconstruct_n, only when it has a chain in no cluster, and one FAISS call
        searches those chains' stored vectors, in ascending order."""
        clusterer = EmbeddingClusterer(self.db_path)
        stored = clusterer.db.index.reconstruct_n(0, self.n)
        recorder = _Recorder(clusterer.db.index)
        clusterer.db.index = recorder
        with mock.patch.object(clustering, "_GRAPH_RESULT_BYTES", _budget(7, self.max_neighbors)):
            clusterer.cluster_greedy(threshold=self.threshold, max_neighbors=self.max_neighbors)
        ref = self.reference(7)
        expected_reads, expected_rows = [], []
        for start in range(0, self.n, 7):
            end = min(start + 7, self.n)
            # A chain is in a cluster from its representative's turn on
            todo = [i for i in range(start, end) if ref.label[i] >= start]
            if todo:
                expected_reads.append((start, end - start))
                expected_rows.append(todo)
        self.assertEqual(sum(map(len, expected_rows)), ref.n_searched)
        self.assertEqual(recorder.reads, expected_reads)
        self.assertLess(len(expected_reads), math.ceil(self.n / 7))
        self.assertEqual(len(recorder.searches), len(expected_rows))
        for (queries, k, _, _), rows in zip(recorder.searches, expected_rows):
            self.assertEqual(k, self.max_neighbors)
            self.assertTrue(queries.flags.c_contiguous)
            self.assertEqual(queries.dtype, np.float32)
            self.assertTrue(np.array_equal(queries, stored[rows]))

    def test_row_cap_sets_the_blocks(self):
        """With the budget left large (more than n rows at k 40), _GRAPH_MAX_ROWS of 7 makes the blocks 7 chains:
        each read once, only when it has a chain in no cluster, and the searched chains and their count as the
        reference at 7 rows."""
        self.assertGreater(clustering._GRAPH_RESULT_BYTES // (self.max_neighbors * 12), self.n)
        clusterer = EmbeddingClusterer(self.db_path)
        stored = clusterer.db.index.reconstruct_n(0, self.n)
        recorder = _Recorder(clusterer.db.index)
        clusterer.db.index = recorder
        with mock.patch.object(clustering, "_GRAPH_MAX_ROWS", 7):
            clusterer.cluster_greedy(threshold=self.threshold, max_neighbors=self.max_neighbors)
        ref = self.reference(7)
        expected_reads, expected_rows = [], []
        for start in range(0, self.n, 7):
            end = min(start + 7, self.n)
            todo = [i for i in range(start, end) if ref.label[i] >= start]
            if todo:
                expected_reads.append((start, end - start))
                expected_rows.append(todo)
        self.assertEqual(recorder.reads, expected_reads)
        self.assertEqual([len(q) for q, _, _, _ in recorder.searches], [len(r) for r in expected_rows])
        for (queries, _, _, _), rows in zip(recorder.searches, expected_rows):
            self.assertTrue(np.array_equal(queries, stored[rows]))
        self.assertLess(ref.n_searched, self.n)
        self.assert_same_as(clusterer, ref)

    def test_int64_labels_when_the_largest_id_is_above_the_int32_limit(self):
        ref = self.reference()
        for max_id, dtype in ((self.n - 1, np.int32), (self.n - 2, np.int64)):
            with self.subTest(max_id=max_id), mock.patch.object(clustering, "_GRAPH_INT32_MAX_ID", max_id):
                clusterer = self.greedy(self.db_path, self.threshold, self.max_neighbors)
                self.assertEqual(clusterer.clusters.dtype, dtype)
                self.assert_same_as(clusterer, ref)


class FlatGreedyTest(_GreedyCase):
    """flat below FAISS's BLAS switch: one query per call and one block of every chain give the same scores, so the
    same partition bit for bit."""

    n, d, threshold, max_neighbors = 1500, 64, 0.5, 20

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        vectors = _clustered_vectors(seed=5, n=cls.n, d=cls.d, n_centers=60, noise=1.0)
        cls.db_path = _build(Path(cls.tmp.name), "flat", vectors, IndexType.flat)
        index = _load(cls.db_path).index
        cls.results = _one_query_results(index, cls.n, cls.max_neighbors)
        cls.results_all = _one_query_results(index, cls.n, cls.n)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixture(self):
        # Every FAISS call of the tests stays below the BLAS switch (nq * d < distance_compute_blas_threshold).
        self.assertLess(self.n * self.d, faiss.cvar.distance_compute_blas_threshold)
        ref = _sequential(self.results, self.n, self.threshold, self.max_neighbors, self.n)
        self.assertGreater(len(ref.sizes), 50)
        self.assertGreater(ref.n_saturated, 10)
        self.assertGreater(ref.n_at_cap, 0)

    def test_same_as_the_sequential_rule(self):
        for k, results in ((self.max_neighbors, self.results), (self.n, self.results_all)):
            for rows in (None, 1, 7, self.n + 5):
                with self.subTest(k=k, rows=rows):
                    clusterer = self.greedy(self.db_path, self.threshold, k, rows)
                    ref = _sequential(results, self.n, self.threshold, k, min(rows or self.n, self.n))
                    self.assert_same_as(clusterer, ref)
                    self.assert_cluster_ids(clusterer)

    def test_max_neighbors_above_n(self):
        """max_neighbors above n is clamped to n: every search asks for n neighbours, and blocks are sized at k = n."""
        for rows in (None, 7):
            with self.subTest(rows=rows):
                clusterer = EmbeddingClusterer(self.db_path)
                recorder = _Recorder(clusterer.db.index)
                clusterer.db.index = recorder
                budget = clustering._GRAPH_RESULT_BYTES if rows is None else _budget(rows, self.n)
                with mock.patch.object(clustering, "_GRAPH_RESULT_BYTES", budget):
                    clusterer.cluster_greedy(threshold=self.threshold, max_neighbors=self.n + 10)
                ref = _sequential(self.results_all, self.n, self.threshold, self.n, rows or self.n, self.n + 10)
                self.assert_same_as(clusterer, ref)
                self.assertEqual(clusterer.n_saturated, 0)
                self.assertEqual(clusterer.k, self.n)
                self.assertTrue(recorder.searches)
                self.assertEqual({k for _, k, _, _ in recorder.searches}, {self.n})
                self.assertEqual({count for _, count in recorder.reads[:-1]}, {rows} if rows else set())


def _blobs(seed: int, sizes, d: int, noise: float, scattered: int = 0) -> tuple:
    """Tight blobs of the given sizes around orthogonal centres (the first axes), plus ``scattered`` random unit
    vectors in the other axes, in a random order: the vectors and each chain's blob (-1 for scattered chains)."""
    rng = np.random.default_rng(seed)
    blob = np.repeat(np.arange(len(sizes)), sizes)
    vectors = np.zeros((len(blob), d))
    vectors[np.arange(len(blob)), blob] = 1.0
    vectors += noise * rng.standard_normal(vectors.shape) / math.sqrt(d)
    if scattered:
        other = np.zeros((scattered, d))
        other[:, len(sizes):] = rng.standard_normal((scattered, d - len(sizes)))
        vectors = np.concatenate([vectors, other])
        blob = np.concatenate([blob, np.full(scattered, -1)])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    order = rng.permutation(len(blob))
    return vectors[order].astype(np.float32), blob[order]


class PlantedClustersTest(_GreedyCase):
    """Well-separated blobs smaller than k, a threshold between the intra- and inter-blob scores: each blob is one
    cluster, represented by its lowest chain index, and no representative keeps its k-th neighbour."""

    d, threshold, max_neighbors = 32, 0.5, 40
    sizes = (30, 1, 7, 12, 25, 3, 18, 9, 30, 2, 14, 1)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.vectors, cls.blob = _blobs(seed=13, sizes=cls.sizes, d=cls.d, noise=0.1)
        cls.n = len(cls.blob)
        cls.db_path = _build(Path(cls.tmp.name), "planted", cls.vectors, IndexType.flat)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixture(self):
        scores = self.vectors @ self.vectors.T
        same = self.blob[:, None] == self.blob[None, :]
        self.assertGreater(scores[same].min(), self.threshold + 0.2)
        self.assertLess(scores[~same].max(), self.threshold - 0.2)
        self.assertLess(max(self.sizes), self.max_neighbors)

    def test_each_blob_is_one_cluster(self):
        for rows in (None, 1, 7):
            with self.subTest(rows=rows):
                clusterer = self.greedy(self.db_path, self.threshold, self.max_neighbors, rows)
                for b, size in enumerate(self.sizes):
                    chains = np.flatnonzero(self.blob == b)
                    self.assertEqual(len(set(clusterer.clusters[chains].tolist())), 1)
                    cluster = clusterer.clusters[chains[0]]
                    self.assertEqual(clusterer.representatives[cluster], chains.min())
                    self.assertEqual(np.count_nonzero(clusterer.clusters == cluster), size)
                self.assertEqual(len(clusterer.representatives), len(self.sizes))
                self.assertEqual(clusterer.n_saturated, 0)
                self.assert_cluster_ids(clusterer)
                stats = clusterer.get_cluster_statistics()
                self.assertEqual((stats["n_clusters"], stats["n_singletons"], stats["max_cluster_size"],
                                  stats["n_at_cap"]), (len(self.sizes), 2, 30, 0))


class _Recorder:
    """The database index, recording the rows read (and a copy of what reconstruct_n returned) and each search's
    queries and results."""

    def __init__(self, index):
        self._index = index
        self.reads = []
        self.read_vectors = []
        self.searches = []

    def reconstruct_n(self, start, n):
        self.reads.append((start, n))
        vectors = self._index.reconstruct_n(start, n)
        self.read_vectors.append(vectors.copy())
        return vectors

    def search(self, x, k):
        scores, ids = self._index.search(x, k)
        self.searches.append((x.copy(), k, scores.copy(), ids.copy()))
        return scores, ids

    def __getattr__(self, name):
        return getattr(self._index, name)


class SizeCapTest(_GreedyCase):
    """One blob larger than k: no cluster above k + 1, clusters at the cap, saturated representatives (some with
    clusters below k), none at k = n; the WARNING when most representatives are saturated."""

    d, threshold, k = 32, 0.5, 50

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.vectors, cls.blob = _blobs(seed=17, sizes=(300,), d=cls.d, noise=0.6, scattered=100)
        cls.n = len(cls.blob)
        cls.db_path = _build(Path(cls.tmp.name), "cap", cls.vectors, IndexType.flat)
        index = _load(cls.db_path).index
        cls.results = _one_query_results(index, cls.n, cls.k)
        cls.results_all = _one_query_results(index, cls.n, cls.n)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixture(self):
        self.assertLess(self.n * self.d, faiss.cvar.distance_compute_blas_threshold)
        ref = _sequential(self.results, self.n, self.threshold, self.k, self.n)
        self.assertGreater(ref.n_at_cap, 0)
        self.assertGreater(ref.n_saturated, ref.n_at_cap)
        self.assertLess(ref.n_saturated, len(ref.sizes) / 2)

    def test_cap_and_saturation(self):
        for rows in (None, 1, 7):
            with self.subTest(rows=rows):
                clusterer = self.greedy(self.db_path, self.threshold, self.k, rows)
                ref = _sequential(self.results, self.n, self.threshold, self.k, min(rows or self.n, self.n))
                self.assert_same_as(clusterer, ref)
                sizes = np.bincount(clusterer.clusters)
                self.assertLessEqual(sizes.max(), self.k + 1)
                # A saturated representative whose cluster is below k: its row's earlier ids were taken already.
                below = [r for r in ref.saturated_reps if ref.sizes[ref.representatives.index(r)] < self.k]
                self.assertGreater(len(below), 0)

    def test_nothing_saturated_at_k_equal_n(self):
        """At k = n every chain is in the row: nothing is cut, even when the row's last cell is kept."""
        for threshold in (self.threshold, -1.0):
            with self.subTest(threshold=threshold):
                clusterer = self.greedy(self.db_path, threshold, self.n)
                ref = _sequential(self.results_all, self.n, threshold, self.n, self.n)
                self.assert_same_as(clusterer, ref)
                self.assertEqual(clusterer.n_saturated, 0)
        self.assertEqual(len(clusterer.representatives), 1)

    def test_warning_when_most_representatives_are_saturated(self):
        warning = ("Most clusters are bounded by --max-neighbors, not --threshold: a representative or singleton may "
                   "score at or above --threshold with an earlier representative.")
        clusterer = self.greedy(self.db_path, self.threshold, self.k)
        with self.assertLogs(level="INFO") as logs:
            clusterer.print_statistics()
        self.assertNotIn(f"WARNING:root:{warning}", logs.output)
        n_clusters = len(clusterer.representatives)
        self.assertIn(f"INFO:root:Representatives with all --max-neighbors neighbours at or above --threshold: "
                      f"{clusterer.n_saturated} of {n_clusters} ({clusterer.n_saturated / n_clusters:.2%}); their "
                      f"clusters may be cut, and a larger --max-neighbors can change them.",
                      logs.output)
        # Every cell kept: every representative saturated.
        clusterer = self.greedy(self.db_path, -1.0, self.k)
        self.assertEqual(clusterer.n_saturated, len(clusterer.representatives))
        with self.assertLogs(level="INFO") as logs:
            clusterer.print_statistics()
        self.assertEqual(logs.output.count(f"WARNING:root:{warning}"), 1)
        # The bound: above 50%, not at it.
        stats = clusterer.get_cluster_statistics()
        for n_saturated, warned in ((2, False), (3, True)):
            with self.subTest(n_saturated=n_saturated):
                four = dict(stats, n_clusters=4, n_saturated=n_saturated)
                with mock.patch.object(clusterer, "get_cluster_statistics", return_value=four), \
                        self.assertLogs(level="INFO") as logs:
                    clusterer.print_statistics()
                self.assertEqual(f"WARNING:root:{warning}" in logs.output, warned)


class IvfPqGreedyTest(_GreedyCase):
    """ivf_pq (no exact reference across batch sizes): invariants against the result rows the method used, recorded by
    a spy on the index; the queries are the normalised decoded embeddings."""

    n, d, threshold = 1000, 64, 0.5
    config = dict(nlist=16, m=8, nbits=4, nprobe=16, opq_train_size=1000)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        vectors = _clustered_vectors(seed=7, n=cls.n, d=cls.d, n_centers=40, noise=0.8)
        # Planted exact duplicates: chain 500 + t is chain 10 * t.
        cls.duplicates = [(10 * t, 500 + t) for t in range(20)]
        for a, b in cls.duplicates:
            vectors[b] = vectors[a]
        cls.db_path = _build(Path(cls.tmp.name), "pq", vectors, IndexType.ivf_pq, IndexConfig(**cls.config))
        decoded = _load(cls.db_path).index.reconstruct_n(0, cls.n)
        cls.unnormalised = decoded

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def spied(self, threshold, max_neighbors, rows=None, nprobe=None):
        clusterer = EmbeddingClusterer(self.db_path)
        if nprobe is not None:
            clusterer.db.set_nprobe(nprobe)
        recorder = _Recorder(clusterer.db.index)
        clusterer.db.index = recorder
        budget = clustering._GRAPH_RESULT_BYTES if rows is None else _budget(rows, min(max_neighbors, self.n))
        with mock.patch.object(clustering, "_GRAPH_RESULT_BYTES", budget):
            clusterer.cluster_greedy(threshold=threshold, max_neighbors=max_neighbors)
        return clusterer, recorder

    def searched_rows(self, recorder):
        """Each searched chain's recorded (scores, ids), found by its query: within the block read before the search,
        the queries are the normalised decoded rows of the chains searched, in ascending order."""
        self.assertEqual(len(recorder.reads), len(recorder.searches))
        rows = {}
        for (start, count), read, (queries, k, scores, ids) in zip(recorder.reads, recorder.read_vectors,
                                                                    recorder.searches):
            block = read.copy()
            faiss.normalize_L2(block)
            at = 0
            for q, query in enumerate(queries):
                while at < count and not np.array_equal(block[at], query):
                    at += 1
                self.assertLess(at, count, "a query is not a normalised decoded row of its block, in order")
                self.assertNotIn(start + at, rows)
                rows[start + at] = (scores[q], ids[q])
                at += 1
        return rows

    def assert_invariants(self, clusterer, recorder, threshold):
        threshold32 = np.float32(threshold)
        rows = self.searched_rows(recorder)
        self.assertEqual(len(rows), clusterer.n_searched)
        clusters, reps = clusterer.clusters, clusterer.representatives
        self.assertEqual(len(clusters), self.n)
        self.assertTrue(np.all((clusters >= 0) & (clusters < len(reps))))
        rep_of = reps[clusters]
        # Every label a representative, each representative searched and its own representative
        self.assertTrue(np.array_equal(rep_of[reps], reps))
        self.assertTrue(set(reps.tolist()) <= set(rows))
        for chain in range(self.n):
            rep = int(rep_of[chain])
            scores, ids = rows[rep]
            if chain != rep:
                # A member is in its representative's row, at a score at least the threshold
                at = np.flatnonzero(ids == chain)
                self.assertEqual(len(at), 1, f"chain {chain} not in its representative {rep}'s row")
                self.assertGreaterEqual(scores[at[0]], threshold32)
        for rep in reps.tolist():
            scores, ids = rows[rep]
            kept = ids[(ids >= 0) & (scores >= threshold32)]
            # Every kept id is taken by the time the representative's turn is over: by it or an earlier one
            self.assertTrue(np.all(rep_of[kept] <= rep))
        self.assert_cluster_ids(clusterer)
        return rows

    def test_invariants(self):
        for max_neighbors, rows in ((self.n, None), (self.n, 7), (self.n, 1), (50, None), (50, 7)):
            with self.subTest(max_neighbors=max_neighbors, rows=rows):
                clusterer, recorder = self.spied(self.threshold, max_neighbors, rows)
                self.assert_invariants(clusterer, recorder, self.threshold)
                self.assertGreater(len(clusterer.representatives), 10)
                self.assertGreater(np.bincount(clusterer.clusters).max(), 10)
                if max_neighbors == self.n:
                    for a, b in self.duplicates:
                        self.assertEqual(clusterer.clusters[a], clusterer.clusters[b])
                    self.assertEqual(clusterer.n_saturated, 0)

    def test_queries_are_normalised(self):
        """The queries are unit vectors (searched_rows finds each among its block's normalised decoded rows), and
        normalising matters on this fixture: the decoded embeddings are not unit vectors."""
        norms = np.linalg.norm(self.unnormalised, axis=1)
        self.assertGreater(np.abs(norms - 1).max(), 1e-3)
        for rows in (None, 7):
            with self.subTest(rows=rows):
                clusterer, recorder = self.spied(self.threshold, 50, rows)
                self.assertEqual(len(self.searched_rows(recorder)), clusterer.n_searched)
                searched = np.concatenate([q for q, _, _, _ in recorder.searches])
                np.testing.assert_allclose(np.linalg.norm(searched, axis=1), 1, rtol=0, atol=1e-5)

    def test_fill_ids_are_never_labels(self):
        """With nprobe < nlist and max_neighbors >= n, FAISS fills rows with -1 (scored -FLT_MAX): no chain is labelled
        from them, even at a threshold of -inf, which every fill's score passes."""
        for threshold in (-10.0, float("-inf")):
            with self.subTest(threshold=threshold):
                clusterer, recorder = self.spied(threshold, self.n + 5, nprobe=1)
                fills = [scores[ids == -1] for _, _, scores, ids in recorder.searches]
                self.assertGreater(sum(map(len, fills)), 0)
                if threshold == float("-inf"):
                    self.assertTrue(all(np.all(f >= threshold) for f in fills))
                self.assert_invariants(clusterer, recorder, threshold)

    def test_no_direct_map_and_parallel_mode_3(self):
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.cluster_greedy(threshold=self.threshold, max_neighbors=50)
        ivf = faiss.extract_index_ivf(clusterer.db.index)
        self.assertEqual(ivf.direct_map.type, faiss.DirectMap.NoMap)
        self.assertEqual(ivf.parallel_mode, 3)


class StatisticsTest(unittest.TestCase):
    """The modularity is for Leiden partitions only; greedy adds its own keys and lines."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        vectors, _ = _blobs(seed=19, sizes=(20, 5, 9, 1, 14), d=16, noise=0.3)
        cls.n = len(vectors)
        cls.db_path = _build(Path(cls.tmp.name), "stats", vectors, IndexType.flat)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_greedy_after_a_graph_then_leiden(self):
        clusterer = EmbeddingClusterer(self.db_path)
        self.assertIsNone(clusterer.method)
        self.assertIsNone(clusterer.representatives)
        clusterer.build_similarity_graph(threshold=0.5, max_neighbors=10)
        clusterer.cluster_greedy(threshold=0.5, max_neighbors=10)
        self.assertIsNotNone(clusterer.graph)
        stats = clusterer.get_cluster_statistics()
        self.assertIsNone(stats["modularity"])
        self.assertEqual(stats["method"], "greedy")
        with self.assertLogs(level="INFO") as logs:
            clusterer.print_statistics()
        self.assertFalse(any("Modularity" in line for line in logs.output))
        n_clusters = stats["n_clusters"]
        self.assertIn(f"INFO:root:Chains searched:     {stats['n_searched']} of {self.n} "
                      f"({stats['n_searched'] / self.n:.2%})", logs.output)
        self.assertIn(f"INFO:root:Clusters of at least --max-neighbors chains: {stats['n_at_cap']}", logs.output)
        self.assertIn("INFO:root:Method:              greedy", logs.output)
        sizes = np.bincount(clusterer.clusters)
        self.assertEqual(stats["n_at_cap"], np.count_nonzero(sizes >= 10))
        self.assertGreater(stats["n_at_cap"], 0)
        self.assertLessEqual(stats["max_cluster_size"], 11)

        out = Path(self.tmp.name) / "greedy.tsv"
        clusterer.export_clusters(str(out))
        self.assertEqual({len(line.split("\t")) for line in out.read_text().splitlines()}, {4})

        clusterer.cluster_leiden(seed=1)
        self.assertEqual(clusterer.method, "leiden")
        self.assertIsNone(clusterer.representatives)
        stats = clusterer.get_cluster_statistics()
        self.assertEqual(stats["modularity"], clusterer.graph.modularity(clusterer.clusters, weights="weight"))
        self.assertNotIn("method", stats)
        self.assertNotIn("n_searched", stats)
        with self.assertLogs(level="INFO") as logs:
            clusterer.print_statistics()
        self.assertTrue(any(line.startswith("INFO:root:Modularity:") for line in logs.output))
        self.assertFalse(any("searched" in line for line in logs.output))
        clusterer.export_clusters(str(out))
        self.assertEqual({len(line.split("\t")) for line in out.read_text().splitlines()}, {3})

    def test_a_partition_set_without_either_method_keeps_the_modularity(self):
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.build_similarity_graph(threshold=0.5, max_neighbors=10)
        clusterer.clusters = [0] * self.n
        stats = clusterer.get_cluster_statistics()
        self.assertEqual(stats["modularity"], clusterer.graph.modularity(clusterer.clusters, weights="weight"))


class EdgeCaseTest(_GreedyCase):

    n, d = 60, 16

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        vectors, _ = _blobs(seed=23, sizes=(10, 10, 10, 10, 10, 4), d=cls.d, noise=0.5, scattered=6)
        cls.db_path = _build(folder, "flat", vectors, IndexType.flat)
        cls.one_path = _build(folder, "one", vectors[:1], IndexType.flat)
        # Scores a threshold can equal exactly: 0.6 and 0.8 from chain 0, and 0 with the last chain.
        cls.exact_vectors = np.array([[1, 0, 0], [0.6, 0.8, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype=np.float32)
        cls.exact_path = _build(folder, "exact", cls.exact_vectors, IndexType.flat)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_chain(self):
        clusterer = self.greedy(self.one_path, 0.0, 10)
        self.assertEqual(clusterer.clusters.tolist(), [0])
        self.assertEqual(clusterer.representatives.tolist(), [0])
        self.assertEqual((clusterer.n_searched, clusterer.n_saturated), (1, 0))
        stats = clusterer.get_cluster_statistics()
        self.assertEqual((stats["n_clusters"], stats["n_singletons"], stats["n_at_cap"]), (1, 1, 0))

    def test_threshold_above_every_score(self):
        clusterer = self.greedy(self.db_path, 2.0, 10)
        self.assertEqual(clusterer.clusters.tolist(), list(range(self.n)))
        self.assertEqual(clusterer.representatives.tolist(), list(range(self.n)))
        self.assertEqual((clusterer.n_searched, clusterer.n_saturated), (self.n, 0))
        stats = clusterer.get_cluster_statistics()
        self.assertEqual((stats["n_clusters"], stats["n_singletons"], stats["max_cluster_size"]),
                         (self.n, self.n, 1))
        self.assertIsNone(stats["modularity"])
        clusterer.print_statistics()

    def test_max_neighbors_above_n(self):
        ref = _sequential(_one_query_results(_load(self.db_path).index, self.n, self.n), self.n, 0.5, self.n, self.n,
                          self.n + 10)
        self.assertGreater(len(ref.sizes), 6)
        clusterer = self.greedy(self.db_path, 0.5, self.n + 10)
        self.assert_same_as(clusterer, ref)

    def test_score_equal_to_threshold_is_taken(self):
        """A float32 score equal to the threshold takes the chain: 0 (the axis chain), 0.6, the score itself, and a
        float64 just above it that rounds to it in float32, as the graph's edges."""
        stored = _load(self.exact_path).index.reconstruct_n(0, 4)
        score = float(stored[0] @ stored[1])
        self.assertEqual(np.float32(np.nextafter(score, 1.0)), np.float32(score))
        for threshold in (0.0, 0.6, score, float(np.nextafter(score, 1.0))):
            with self.subTest(threshold=threshold):
                clusterer = self.greedy(self.exact_path, threshold, 10)
                self.assertEqual(clusterer.clusters[1], clusterer.clusters[0])
                self.assertEqual(clusterer.representatives[clusterer.clusters[1]], 0)
                if threshold == 0.0:
                    self.assertEqual(clusterer.clusters.tolist(), [0, 0, 0, 0])

    def test_max_neighbors_below_1_is_refused_before_any_read(self):
        clusterer = EmbeddingClusterer(self.db_path)
        recorder = _Recorder(clusterer.db.index)
        clusterer.db.index = recorder
        for max_neighbors in (0, -1):
            with self.subTest(max_neighbors=max_neighbors):
                with self.assertRaisesRegex(ValueError, r"^--max-neighbors must be at least 1\.$"):
                    clusterer.cluster_greedy(threshold=0.5, max_neighbors=max_neighbors)
        self.assertEqual((recorder.reads, recorder.searches), ([], []))
        self.assertIsNone(clusterer.clusters)


def _c40d4e1_export(chain_ids, membership, output_file: str, min_cluster_size=None,
                    format_mode=FormatMode.headless_tsv):
    """c40d4e1's export_clusters and write_cluster_results, verbatim but for the file handling: a list of every
    (chain_id, cluster_id, cluster_size), then each row joined from its three str()."""
    cluster_sizes = np.bincount(membership).tolist()
    assignments = [
        (chain_id, cluster_id, cluster_sizes[cluster_id])
        for chain_id, cluster_id in zip(chain_ids, membership)
        if min_cluster_size is None or cluster_sizes[cluster_id] >= min_cluster_size
    ]
    spec = format_spec(format_mode)
    lines = [spec.delimiter.join(CLUSTER_OUTPUT_FIELDS)] if spec.header else []
    lines += [spec.delimiter.join((str(chain_id), str(cluster_id), str(cluster_size)))
              for chain_id, cluster_id, cluster_size in assignments]
    Path(output_file).write_text("".join(line + "\n" for line in lines))


class ClusterCommandTest(unittest.TestCase):
    """cluster_database, called directly as test_cli_search's test_23: the greedy rows, and the Leiden file as
    c40d4e1 wrote it, at several export chunks, with and without --min-cluster-size."""

    n, d, threshold, max_neighbors, seed = 600, 128, 0.6, 20, 42

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        vectors = _clustered_vectors(seed=29, n=cls.n, d=cls.d, n_centers=60, noise=1.0)
        cls.db_path = _build(Path(cls.tmp.name), "hnsw", vectors, IndexType.hnsw)
        clusterer = EmbeddingClusterer(cls.db_path)
        cls.chain_ids = clusterer.chain_ids
        graph = clusterer.build_similarity_graph(threshold=cls.threshold, max_neighbors=cls.max_neighbors)
        # c40d4e1's cluster_leiden call, made here and not through the code under test: the reference membership
        cls.membership = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=1.0,
            n_iterations=-1, seed=cls.seed).membership
        greedy = EmbeddingClusterer(cls.db_path)
        greedy.cluster_greedy(threshold=cls.threshold, max_neighbors=cls.max_neighbors)
        cls.greedy_clusters = greedy.clusters
        cls.greedy_reps = greedy.representatives

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def chunks(self):
        # 1, 3, an exact divisor of n, and more than n (None: the default, also more than n)
        self.assertEqual(self.n % 120, 0)
        self.assertNotEqual(self.n % 7, 0)
        return (None, 1, 3, 7, 120, self.n + 5)

    def test_fixture(self):
        sizes = np.bincount(self.greedy_clusters)
        self.assertGreater(np.count_nonzero(sizes == 1), 5)
        self.assertGreater(np.count_nonzero(sizes >= 2), 20)

    def test_greedy_rows(self):
        sizes = np.bincount(self.greedy_clusters)
        for chunk in self.chunks():
            for min_size in (None, 2):
                with self.subTest(chunk=chunk, min_cluster_size=min_size):
                    out = Path(self.tmp.name) / "greedy.tsv"
                    with mock.patch.object(clustering, "_EXPORT_CHUNK", chunk or clustering._EXPORT_CHUNK):
                        cluster_database(db_path=self.db_path, method=ClusterMethod.greedy, threshold=self.threshold,
                                         output_file=str(out), max_neighbors=self.max_neighbors,
                                         min_cluster_size=min_size, format_mode=FormatMode.tsv)
                    lines = out.read_text().splitlines()
                    self.assertEqual(lines[0].split("\t"), list(GREEDY_CLUSTER_OUTPUT_FIELDS))
                    expected = [
                        f"{self.chain_ids[i]}\t{c}\t{sizes[c]}\t{self.chain_ids[self.greedy_reps[c]]}"
                        for i, c in enumerate(self.greedy_clusters.tolist()) if min_size is None or sizes[c] >= min_size]
                    self.assertEqual(lines[1:], expected)
                    rows = [line.split("\t") for line in lines[1:]]
                    counts = Counter(cluster for _, cluster, _, _ in rows)
                    for chain, cluster, size, rep in rows:
                        self.assertEqual(int(size), counts[cluster])
                        if rep == chain:
                            self.assertEqual(self.greedy_reps[int(cluster)], self.chain_ids.index(chain))
                    self.assertEqual(sum(1 for chain, _, _, rep in rows if chain == rep), len(counts))
                    if min_size:
                        self.assertTrue(all(int(size) >= 2 for _, _, size, _ in rows))
                        self.assertLess(len(rows), self.n)
                    # Headless: the same rows without the header
                    with mock.patch.object(clustering, "_EXPORT_CHUNK", chunk or clustering._EXPORT_CHUNK):
                        cluster_database(db_path=self.db_path, method="greedy", threshold=self.threshold,
                                         output_file=str(out), max_neighbors=self.max_neighbors,
                                         min_cluster_size=min_size)
                    self.assertEqual(out.read_text().splitlines(), lines[1:])

    def test_leiden_membership_as_c40d4e1(self):
        """cluster_leiden returns c40d4e1's membership, cluster ids by non-increasing size (0 the largest)."""
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.build_similarity_graph(threshold=self.threshold, max_neighbors=self.max_neighbors)
        self.assertEqual(list(clusterer.cluster_leiden(seed=self.seed)), self.membership)
        sizes = np.bincount(self.membership)
        self.assertGreater(len(set(sizes.tolist())), 2)
        self.assertTrue(np.all(np.diff(sizes) <= 0), sizes.tolist())

    def test_leiden_file_as_c40d4e1(self):
        for chunk in self.chunks():
            for min_size in (None, 2):
                for format_mode in (FormatMode.headless_tsv, FormatMode.tsv):
                    with self.subTest(chunk=chunk, min_cluster_size=min_size, format_mode=format_mode.value):
                        ref = Path(self.tmp.name) / "ref.tsv"
                        _c40d4e1_export(self.chain_ids, self.membership, str(ref), min_size, format_mode)
                        for method in (None, ClusterMethod.leiden):
                            out = Path(self.tmp.name) / "leiden.tsv"
                            kwargs = {} if method is None else {"method": method}
                            with mock.patch.object(clustering, "_EXPORT_CHUNK", chunk or clustering._EXPORT_CHUNK):
                                cluster_database(db_path=self.db_path, threshold=self.threshold, output_file=str(out),
                                                 max_neighbors=self.max_neighbors, min_cluster_size=min_size,
                                                 seed=self.seed, format_mode=format_mode, **kwargs)
                            self.assertEqual(out.read_bytes(), ref.read_bytes())
        self.assertGreater(len(ref.read_text().splitlines()), 100)

    def test_methods_route(self):
        """greedy calls cluster_greedy and builds no graph; leiden (the default) the graph, then Leiden at resolution
        1.0 when --resolution is not given."""
        out = str(Path(self.tmp.name) / "routed.tsv")
        cases = [
            (dict(), [mock.call.build_similarity_graph(threshold=0.7, max_neighbors=9),
                      mock.call.cluster_leiden(resolution=1.0, seed=None)]),
            (dict(method=ClusterMethod.leiden, resolution=2.5, seed=4),
             [mock.call.build_similarity_graph(threshold=0.7, max_neighbors=9),
              mock.call.cluster_leiden(resolution=2.5, seed=4)]),
            (dict(method=ClusterMethod.greedy), [mock.call.cluster_greedy(threshold=0.7, max_neighbors=9)]),
        ]
        for kwargs, calls in cases:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                with mock.patch.object(clustering, "EmbeddingClusterer") as made:
                    cluster_database(db_path=self.db_path, threshold=0.7, max_neighbors=9, output_file=out, **kwargs)
                clusterer = made.return_value
                self.assertEqual([c for c in clusterer.mock_calls if c[0] in (
                    "build_similarity_graph", "cluster_leiden", "cluster_greedy")], calls)


def _flat(text: str) -> str:
    """Text on one line, typer's error panel borders and colour codes dropped."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


class ClusterRowLengthTest(unittest.TestCase):
    """write_cluster_results refuses rows whose length does not match the fields."""

    def test_rows_must_match_the_fields(self):
        cases = [(GREEDY_CLUSTER_OUTPUT_FIELDS, [("a", 0, 1)], r"^cluster row \('a', 0, 1\) has 3 values for the 4 "
                  r"fields chain_id, cluster_id, cluster_size, representative$"),
                 (CLUSTER_OUTPUT_FIELDS, [("a", 0, 1, "a")], r"^cluster row \('a', 0, 1, 'a'\) has 4 values for the "
                  r"3 fields chain_id, cluster_id, cluster_size$")]
        with tempfile.TemporaryDirectory() as d:
            for fields, rows, message in cases:
                with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, message):
                    write_cluster_results(rows, str(Path(d) / "c.tsv"), FormatMode.tsv, fields)


class RefusalTest(unittest.TestCase):
    """Through the CLI: exit status 2 before the database is loaded."""

    def invoke(self, *args):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            with mock.patch.object(clustering, "EmbeddingClusterer",
                                   side_effect=AssertionError("the database was loaded")):
                return CliRunner().invoke(search.app, [str(a) for a in args])
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)

    def refused(self, *args) -> str:
        result = self.invoke(*args)
        self.assertEqual(result.exit_code, 2, f"{result.output}\n{result.exception!r}")
        return _flat(result.output)

    def test_leiden_options_under_greedy_in_one_message(self):
        base = ("cluster", "--db-path", "no/such/db", "--method", "greedy")
        cases = (
            (("--resolution", "1.0"), "--resolution is a Leiden option: --method greedy has no resolution and is "
                                      "deterministic. Remove it, or use --method leiden."),
            (("--seed", "7"), "--seed is a Leiden option: --method greedy has no resolution and is deterministic. "
                              "Remove it, or use --method leiden."),
            (("--seed", "7", "--resolution", "2"), "--resolution, --seed are Leiden options: --method greedy has no "
                                                   "resolution and is deterministic. Remove them, or use --method "
                                                   "leiden."),
        )
        for extra, message in cases:
            with self.subTest(extra=extra):
                output = self.refused(*base, *extra)
                self.assertIn(f"Invalid value for --method: {message}", output)
                self.assertEqual(output.count("Leiden option"), 1)

    def test_max_neighbors_below_1(self):
        for command in (("cluster",), ("cluster", "--method", "leiden"), ("cluster", "--method", "greedy"),
                        ("similarity-graph",)):
            for value in ("0", "-3"):
                with self.subTest(command=command, value=value):
                    output = self.refused(*command, "--db-path", "no/such/db", "--max-neighbors", value)
                    self.assertIn("Invalid value for --max-neighbors: --max-neighbors must be at least 1.", output)

    def test_leiden_options_with_leiden_are_not_refused(self):
        for method in ((), ("--method", "leiden")):
            with self.subTest(method=method):
                result = self.invoke("cluster", "--db-path", "no/such/db", *method, "--resolution", "2",
                                     "--seed", "3")
                self.assertEqual(result.exit_code, 1)
                self.assertEqual(str(result.exception), "the database was loaded")

    def test_unknown_method(self):
        output = self.refused("cluster", "--db-path", "no/such/db", "--method", "components")
        self.assertIn("--method", output)


if __name__ == "__main__":
    unittest.main()
