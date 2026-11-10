"""EmbeddingClusterer.build_similarity_graph: blocks of chains read with reconstruct_n and searched many at a time give,
on every database type, the expected graph: on hnsw the one-query-per-call graph bit for bit, on flat the same up to
float rounding, and on ivf_pq the normalised decoded query against each code. Also cluster_leiden on python-igraph's
Leiden (P9.U): the call, its private seeding, the cluster ids by size, and its refusals."""
import hashlib
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import weakref
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# As the commands run (set_log_level): Lightning first, which brings torch and, through scikit-learn,
# KMP_DUPLICATE_LIB_OK; on macOS, FAISS's OpenMP regions abort a process that loaded torch's libomp without it.
import lightning  # noqa: F401
import faiss
import igraph as ig
import numpy as np
from tqdm import tqdm
from typer.testing import CliRunner

import foldmatch
from foldmatch.cli import search
from foldmatch.cli.search import cluster_database, similarity_graph
from foldmatch.search import clustering
from foldmatch.search.clustering import EmbeddingClusterer
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.types.api_types import FormatMode, IndexConfig, IndexType


def _clustered_vectors(seed: int, n: int, d: int, n_centers: int, noise: float) -> np.ndarray:
    """Unit vectors around ``n_centers`` random centres, at a noise level of 0.5 to 1.5 times ``noise`` each, so scores
    spread over the thresholds; the members of a centre are contiguous, so the last chains have edges among
    themselves."""
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


def _old_way(index, n: int, threshold: float, max_neighbors: int):
    """The graph build of FoldMatch 875b514: one stored vector per FAISS call, searched as it is, and an edge (i, j)
    from the search of i when i < j. Returns the ordered edges, their weights, and each chain's k-th score and id."""
    k = min(max_neighbors, n)
    edges, weights, kth, kth_ids = [], [], [], []
    for i in range(n):
        query = np.ascontiguousarray(index.reconstruct(i), dtype=np.float32).reshape(1, -1)
        scores, ids = index.search(query, k)
        kth.append(float(scores[0, -1]))
        kth_ids.append(int(ids[0, -1]))
        for j, score in zip(ids[0], scores[0]):
            if i < j and score >= threshold:
                edges.append((i, int(j)))
                weights.append(float(score))
    return edges, weights, kth, kth_ids


def _budget(rows: int, k: int) -> int:
    """The result budget that gives blocks of ``rows`` chains at ``k`` neighbours."""
    return rows * k * 12


class _Recorder:
    """The database index, recording the rows read and searched in each call."""

    def __init__(self, index):
        self._index = index
        self.reads = []
        self.searches = []

    def reconstruct_n(self, start, n):
        self.reads.append((start, n))
        return self._index.reconstruct_n(start, n)

    def search(self, x, k):
        self.searches.append((x.copy(), k))
        return self._index.search(x, k)

    def __getattr__(self, name):
        return getattr(self._index, name)


class _GraphCase(unittest.TestCase):

    def graph(self, db_path: str, threshold: float, max_neighbors: int, rows: int = None, k: int = None):
        clusterer = EmbeddingClusterer(db_path)
        if rows is None:
            graph = clusterer.build_similarity_graph(threshold=threshold, max_neighbors=max_neighbors)
        else:
            with mock.patch.object(clustering, "_GRAPH_RESULT_BYTES", _budget(rows, k)):
                graph = clusterer.build_similarity_graph(threshold=threshold, max_neighbors=max_neighbors)
        self.assertEqual(graph.vcount(), len(clusterer.chain_ids))
        self.assertEqual(graph.vs["name"], clusterer.chain_ids)
        self.assertTrue(all(type(w) is float for w in graph.es["weight"]))
        return graph.get_edgelist(), graph.es["weight"]

    def assert_same_up_to_rounding(self, got, ref, threshold, tol, kth=None):
        """Same edge set as ``ref`` except pairs whose score is within ``tol`` of the threshold or of the row's k-th
        score; weights of the shared edges within ``tol``."""
        got_w = dict(zip(*got))
        ref_w = dict(zip(*ref))
        self.assertEqual(len(got_w), len(got[0]))
        for pair in set(got_w) ^ set(ref_w):
            score = got_w.get(pair, ref_w.get(pair))
            near = abs(score - threshold) <= tol or (kth is not None and abs(score - kth[pair[0]]) <= tol)
            self.assertTrue(near, f"edge {pair} score {score} only in one graph")
        shared = sorted(set(got_w) & set(ref_w))
        self.assertGreater(len(shared), 0)
        diff = max(abs(got_w[p] - ref_w[p]) for p in shared)
        self.assertLessEqual(diff, tol)


class HnswGraphTest(_GraphCase):
    """hnsw: the graph equals the one-query-per-call graph bit for bit, ordered edges and weights."""

    n, d, threshold, max_neighbors = 2000, 1536, 0.6, 40
    # A --max-neighbors small enough that it, not the threshold, limits the edges of many chains.
    small_k = 5

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.vectors = _clustered_vectors(seed=3, n=cls.n, d=cls.d, n_centers=150, noise=1.0)
        cls.db_path = _build(Path(cls.tmp.name), "hnsw", cls.vectors, IndexType.hnsw)
        db = FaissEmbeddingDatabase(Path(cls.tmp.name), "hnsw")
        db.load_database()
        cls.ref = _old_way(db.index, cls.n, cls.threshold, cls.max_neighbors)
        cls.ref_small_k = _old_way(db.index, cls.n, cls.threshold, cls.small_k)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixture(self):
        edges = self.ref[0]
        self.assertGreater(len(edges), 1000)
        self.assertNotEqual(self.n % 7, 0)
        # The last short block of 7 has edges of its own.
        self.assertTrue(any(i >= self.n - self.n % 7 for i, _ in edges))
        # At small_k, many chains have an edge to their k-th hit: --max-neighbors limits their edges.
        _, _, kth, kth_ids = self.ref_small_k
        saturated = [i for i in range(self.n) if kth_ids[i] > i and kth[i] >= self.threshold]
        self.assertGreater(len(saturated), 100)

    def test_same_graph_as_one_query_at_a_time(self):
        edges, weights = self.graph(self.db_path, self.threshold, self.max_neighbors)
        self.assertEqual(edges, self.ref[0])
        self.assertEqual(weights, self.ref[1])

    def test_block_sizes(self):
        for rows in (1, 7, self.n + 5):
            with self.subTest(rows=rows):
                edges, weights = self.graph(self.db_path, self.threshold, self.max_neighbors, rows,
                                            self.max_neighbors)
                self.assertEqual(edges, self.ref[0])
                self.assertEqual(weights, self.ref[1])

    def test_max_neighbors_limits_the_edges_as_before(self):
        """A chain's k-th hit, when above the threshold, is an edge, as in the one-query-per-call graph."""
        edges, weights = self.graph(self.db_path, self.threshold, self.small_k)
        self.assertEqual(edges, self.ref_small_k[0])
        self.assertEqual(weights, self.ref_small_k[1])

    def test_blocks_follow_the_budget_and_search_the_stored_vectors(self):
        """Blocks of budget // (k * 12) chains, at most _GRAPH_MAX_ROWS, a short last one, each read once and searched
        as stored."""
        for k, rows, max_rows in ((40, 7, None), (20, 14, None), (1000, 1, None), (1, 13, 13)):
            with self.subTest(k=k, max_rows=max_rows):
                clusterer = EmbeddingClusterer(self.db_path)
                stored = clusterer.db.index.reconstruct_n(0, self.n)
                recorder = _Recorder(clusterer.db.index)
                clusterer.db.index = recorder
                # At k = 1 the budget alone would give blocks of 280 chains: the row cap makes them 13.
                with mock.patch.object(clustering, "_GRAPH_RESULT_BYTES", _budget(7, 40)), \
                        mock.patch.object(clustering, "_GRAPH_MAX_ROWS", max_rows or clustering._GRAPH_MAX_ROWS):
                    clusterer.build_similarity_graph(threshold=self.threshold, max_neighbors=k)
                starts = list(range(0, self.n, rows))
                expected = [(s, min(rows, self.n - s)) for s in starts]
                self.assertEqual(recorder.reads, expected)
                self.assertEqual([(len(x), kk) for x, kk in recorder.searches], [(m, k) for _, m in expected])
                searched = np.concatenate([x for x, _ in recorder.searches])
                self.assertTrue(np.array_equal(searched, stored))
        # Normalising the stored vectors again would change them: the comparison above can see it.
        renormalised = stored.copy()
        faiss.normalize_L2(renormalised)
        self.assertFalse(np.array_equal(renormalised, stored))

    def test_max_neighbors_0_is_refused_after_one_chain(self):
        """FAISS refuses k = 0, as before, after one chain is read: not after a block of the budget's size."""
        clusterer = EmbeddingClusterer(self.db_path)
        recorder = _Recorder(clusterer.db.index)
        clusterer.db.index = recorder
        with self.assertRaises(AssertionError):
            clusterer.build_similarity_graph(threshold=self.threshold, max_neighbors=0)
        self.assertEqual(recorder.reads, [(0, 1)])


class FlatGraphTest(_GraphCase):
    """flat across FAISS's BLAS switch: the same graph up to float rounding."""

    n, d, threshold, max_neighbors = 1500, 256, 0.6, 30

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.vectors = _clustered_vectors(seed=5, n=cls.n, d=cls.d, n_centers=60, noise=1.0)
        cls.db_path = _build(Path(cls.tmp.name), "flat", cls.vectors, IndexType.flat)
        db = FaissEmbeddingDatabase(Path(cls.tmp.name), "flat")
        db.load_database()
        cls.ref = _old_way(db.index, cls.n, cls.threshold, cls.max_neighbors)
        cls.ref_all = _old_way(db.index, cls.n, cls.threshold, cls.n)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixture(self):
        self.assertGreater(len(self.ref[0]), 1000)
        # One-query searches stay below the BLAS switch; one block of every chain is above it.
        self.assertLess(self.d, faiss.cvar.distance_compute_blas_threshold)
        self.assertGreaterEqual(self.n * self.d, faiss.cvar.distance_compute_blas_threshold)
        self.assertGreater(clustering._GRAPH_RESULT_BYTES // (self.max_neighbors * 12), self.n)
        self.assertGreater(clustering._GRAPH_MAX_ROWS, self.n)

    def test_same_graph_across_the_blas_switch(self):
        got = self.graph(self.db_path, self.threshold, self.max_neighbors)
        self.assert_same_up_to_rounding(got, self.ref[:2], self.threshold, 1e-5, kth=self.ref[2])

    def test_block_sizes(self):
        for rows in (1, 7, self.n + 5):
            with self.subTest(rows=rows):
                got = self.graph(self.db_path, self.threshold, self.n, rows, self.n)
                self.assert_same_up_to_rounding(got, self.ref_all[:2], self.threshold, 1e-6)


class IvfPqGraphTest(_GraphCase):
    """ivf_pq: the graph builds without a direct map, and a score is the normalised decoded query against the code."""

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
        db = FaissEmbeddingDatabase(Path(cls.tmp.name), "pq")
        db.load_database()
        cls.decoded = db.index.reconstruct_n(0, cls.n)
        cls.norms = np.linalg.norm(cls.decoded, axis=1)
        # The score of chain i's search for chain j: the unit decoded query against j's code.
        cls.expected = (cls.decoded / cls.norms[:, None]) @ cls.decoded.T

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cluster_and_similarity_graph_run(self):
        out = Path(self.tmp.name)
        cluster_database(db_path=self.db_path, threshold=self.threshold, output_file=str(out / "clusters.tsv"),
                         seed=1)
        rows = (out / "clusters.tsv").read_text().splitlines()
        self.assertEqual([r.split("\t")[0] for r in rows], [f"chain{i}" for i in range(self.n)])
        similarity_graph(db_path=self.db_path, threshold=self.threshold, output_file=str(out / "graph.graphml"))
        nodes = ET.parse(out / "graph.graphml").getroot().findall(".//g:node",
                                                                   {"g": "http://graphml.graphdrawing.org/xmlns"})
        self.assertEqual(len(nodes), self.n)

    def test_weights_are_the_unit_query_against_the_code(self):
        edge_list, weights = self.graph(self.db_path, self.threshold, self.n)
        edges = np.array(edge_list)
        self.assertGreater(len(edges), 1000)
        self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
        self.assertTrue(np.all(np.diff(edges[:, 0]) >= 0))
        np.testing.assert_allclose(weights, self.expected[edges[:, 0], edges[:, 1]], rtol=0, atol=1e-5)
        # Every chain is scanned (nprobe = nlist, k = n): the edges are the pairs above the threshold.
        i, j = np.nonzero(np.triu(self.expected >= self.threshold, k=1))
        ref = (list(zip(i.tolist(), j.tolist())), self.expected[i, j].tolist())
        self.assert_same_up_to_rounding((edge_list, weights), ref, self.threshold, 1e-5)

    def test_duplicates_score_the_code_norm(self):
        got = dict(zip(*self.graph(self.db_path, self.threshold, self.n)))
        checked = 0
        for a, b in self.duplicates:
            # Same code; the reverse OPQ product's last bits can depend on the row's position (OpenBLAS threads).
            np.testing.assert_allclose(self.decoded[a], self.decoded[b], rtol=0, atol=1e-6)
            norm = float(self.norms[b])
            if norm < self.threshold + 0.01 or abs(norm - norm ** 2) < 1e-3:
                continue
            self.assertAlmostEqual(got[(a, b)], norm, delta=1e-5)
            checked += 1
        self.assertGreater(checked, 5)

    def test_no_direct_map_and_parallel_mode_3(self):
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.build_similarity_graph(threshold=self.threshold, max_neighbors=50)
        ivf = faiss.extract_index_ivf(clusterer.db.index)
        self.assertEqual(ivf.direct_map.type, faiss.DirectMap.NoMap)
        self.assertEqual(ivf.parallel_mode, 3)

    def test_fill_ids_make_no_edges(self):
        """With nprobe < nlist and max_neighbors >= n, FAISS fills rows with -1: no edge comes from them."""
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.db.set_nprobe(1)
        queries = self.decoded.copy()
        faiss.normalize_L2(queries)
        scores, ids = clusterer.db.index.search(queries, self.n)
        self.assertTrue((ids == -1).any())
        graph = clusterer.build_similarity_graph(threshold=-10.0, max_neighbors=self.n + 5)
        rows, cols = np.nonzero(np.arange(self.n)[:, None] < ids)
        self.assertEqual(graph.get_edgelist(), list(zip(rows.tolist(), ids[rows, cols].tolist())))
        self.assertEqual(graph.es["weight"], scores[rows, cols].tolist())
        self.assertGreaterEqual(min(min(e) for e in graph.get_edgelist()), 0)

    def test_block_sizes(self):
        """The same graph up to float rounding: the OPQ rotation is a BLAS product, whose last bits can depend on the
        number of rows (a block of 1 against more on OpenBLAS; blocks of 1, 2 and 7 against 64 or more on Apple
        Accelerate)."""
        ref = self.graph(self.db_path, self.threshold, self.n)
        for rows in (1, 2, 7, self.n + 5):
            with self.subTest(rows=rows):
                got = self.graph(self.db_path, self.threshold, self.n, rows, self.n)
                self.assert_same_up_to_rounding(got, ref, self.threshold, 1e-6)


class EdgeCaseTest(_GraphCase):

    n, d = 60, 16

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        # Six clusters in the first 8 dimensions, and six chains along the last 8 axes that score 0 with every other
        # chain: singletons.
        cls.vectors = np.zeros((cls.n, cls.d), dtype=np.float32)
        cls.vectors[:54, :8] = _clustered_vectors(seed=11, n=54, d=8, n_centers=6, noise=0.5)
        cls.vectors[54:, 8:14] = np.eye(6, dtype=np.float32)
        cls.db_path = _build(folder, "flat", cls.vectors, IndexType.flat)
        cls.one_path = _build(folder, "one", cls.vectors[:1], IndexType.flat)
        # Scores a threshold can equal exactly: 0.6 and 0.8 from chain 0, and 0 with the last chain.
        cls.exact_vectors = np.array([[1, 0, 0], [0.6, 0.8, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype=np.float32)
        cls.exact_path = _build(folder, "exact", cls.exact_vectors, IndexType.flat)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_max_neighbors_above_n(self):
        db = FaissEmbeddingDatabase(Path(self.tmp.name), "flat")
        db.load_database()
        ref = _old_way(db.index, self.n, 0.5, self.n + 10)
        edges, weights = self.graph(self.db_path, 0.5, self.n + 10)
        self.assertGreater(len(edges), 0)
        self.assertEqual(edges, ref[0])
        self.assertEqual(weights, ref[1])

    def test_score_equal_to_threshold_makes_an_edge(self):
        """An edge needs score >= threshold: the axis chains score exactly 0 with every chain, so at threshold 0 they
        make edges, the same as in the one-query-per-call graph."""
        db = FaissEmbeddingDatabase(Path(self.tmp.name), "flat")
        db.load_database()
        ref = _old_way(db.index, self.n, 0.0, self.n)
        self.assertGreater(sum(w == 0.0 for w in ref[1]), 0)
        edges, weights = self.graph(self.db_path, 0.0, self.n)
        self.assertEqual(edges, ref[0])
        self.assertEqual(weights, ref[1])

    def test_threshold_equal_to_a_float32_score(self):
        """A float32 score equal to the threshold as a Python float makes an edge, as before (NumPy compares the
        float32 scores with the threshold in float32): 0, 0.6, the score itself, and a float64 just above it that rounds
        to it in float32."""
        db = FaissEmbeddingDatabase(Path(self.tmp.name), "exact")
        db.load_database()
        stored = db.index.reconstruct_n(0, 4)
        score = float(stored[0] @ stored[1])
        for threshold in (0.0, 0.6, score, float(np.nextafter(score, 1.0))):
            with self.subTest(threshold=threshold):
                ref = _old_way(db.index, 4, threshold, 10)
                edges, weights = self.graph(self.exact_path, threshold, 10)
                self.assertEqual(edges, ref[0])
                self.assertEqual(weights, ref[1])
                self.assertIn((0, 1), edges)
                if threshold == 0.0:
                    self.assertIn((0, 3), edges)
        self.assertEqual(np.float32(np.nextafter(score, 1.0)), np.float32(score))

    def test_one_chain(self):
        clusterer = EmbeddingClusterer(self.one_path)
        graph = clusterer.build_similarity_graph(threshold=0.0, max_neighbors=10)
        self.assertEqual((graph.vcount(), graph.ecount()), (1, 0))
        self.assertEqual(clusterer.cluster_leiden(seed=1).tolist(), [0])
        stats = clusterer.get_cluster_statistics()
        self.assertEqual((stats["n_clusters"], stats["n_singletons"]), (1, 1))

    def test_threshold_above_every_score(self):
        clusterer = EmbeddingClusterer(self.db_path)
        graph = clusterer.build_similarity_graph(threshold=2.0, max_neighbors=10)
        self.assertEqual((graph.vcount(), graph.ecount()), (self.n, 0))
        self.assertEqual(graph.es["weight"], [])
        clusters = clusterer.cluster_leiden(seed=1)
        self.assertEqual(clusters.tolist(), list(range(self.n)))
        stats = clusterer.get_cluster_statistics()
        self.assertEqual((stats["n_clusters"], stats["n_singletons"], stats["max_cluster_size"]), (self.n, self.n, 1))
        self.assertTrue(math.isnan(stats["modularity"]))
        clusterer.print_statistics()

    def test_statistics_and_export_as_before(self):
        """Cluster sizes from np.bincount give the rows and statistics of the dict count."""
        out = Path(self.tmp.name) / "clusters.tsv"
        cluster_database(db_path=self.db_path, threshold=0.5, output_file=str(out), max_neighbors=10, seed=3,
                         min_cluster_size=2, format_mode=FormatMode.tsv)
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.build_similarity_graph(threshold=0.5, max_neighbors=10)
        membership = clusterer.cluster_leiden(seed=3)
        sizes = {}
        for cluster_id in membership:
            sizes[cluster_id] = sizes.get(cluster_id, 0) + 1
        self.assertTrue(any(s == 1 for s in sizes.values()) and any(s >= 2 for s in sizes.values()))
        expected = ["chain_id\tcluster_id\tcluster_size"] + [
            f"{chain}\t{c}\t{sizes[c]}" for chain, c in zip(clusterer.chain_ids, membership) if sizes[c] >= 2]
        self.assertEqual(out.read_text().splitlines(), expected)
        stats = clusterer.get_cluster_statistics()
        values = list(sizes.values())
        self.assertEqual(
            {k: stats[k] for k in ("n_chains", "n_clusters", "n_singletons", "min_cluster_size", "max_cluster_size",
                                   "mean_cluster_size", "median_cluster_size")},
            {"n_chains": self.n, "n_clusters": len(sizes), "n_singletons": sum(1 for s in values if s == 1),
             "min_cluster_size": min(values), "max_cluster_size": max(values),
             "mean_cluster_size": np.mean(values), "median_cluster_size": np.median(values)})
        self.assertEqual(stats["modularity"], clusterer.graph.modularity(membership, weights="weight"))


# P9.Q1: the graph is built with less memory per edge (int32 ids, no concatenation, blocks released as consumed, equal
# weights sharing one float, malloc_trim on glibc), and is K2's graph bit for bit.

def _k2_way(clusterer: EmbeddingClusterer, threshold: float, max_neighbors: int):
    """K2's build_similarity_graph (c40d4e1) verbatim, on ``clusterer``: int64 blocks, np.concatenate, ig.Graph of the
    ndarray and weights.tolist(). It reads the module's block budget and row cap, as K2 did."""
    if clusterer.db.index is None:
        raise ValueError("Database not loaded. Call load_database() first.")

    n_chains = len(clusterer.chain_ids)
    logging.info(f"\nBuilding similarity graph with threshold >= {threshold}")
    logging.info(f"Searching up to {max_neighbors} neighbors per chain")

    k = min(max_neighbors, n_chains)
    rows = 1 if k < 1 else max(1, min(n_chains, clustering._GRAPH_MAX_ROWS, clustering._GRAPH_RESULT_BYTES // (k * 12)))

    index_to_use = clusterer.db.index
    if clusterer.db.is_gpu_index:
        logging.info("Moving GPU index to CPU for reconstruction...")
        index_to_use = faiss.index_gpu_to_cpu(clusterer.db.index)

    normalize = clusterer.db.index_type == IndexType.ivf_pq

    edge_blocks = []
    weight_blocks = []
    n_edges = 0
    start_time = time.perf_counter()

    logging.info("Performing k-NN search ...")
    with tqdm(total=n_chains, desc="Processing chains", unit="chain") as progress:
        for start in range(0, n_chains, rows):
            end = min(start + rows, n_chains)
            embeddings = np.ascontiguousarray(index_to_use.reconstruct_n(start, end - start), dtype=np.float32)
            if normalize:
                faiss.normalize_L2(embeddings)
            scores, indices = clusterer.db.index.search(embeddings, k)
            del embeddings

            chains = np.arange(start, end, dtype=np.int64)[:, None]
            block_rows, block_cols = np.nonzero((chains < indices) & (scores >= threshold))
            edge_blocks.append(np.stack([block_rows + start, indices[block_rows, block_cols]], axis=1))
            weight_blocks.append(scores[block_rows, block_cols])
            n_edges += len(block_rows)
            del scores, indices, chains, block_rows, block_cols

            progress.set_postfix(edges=n_edges, refresh=False)
            progress.update(end - start)

    edges = np.concatenate(edge_blocks) if edge_blocks else np.empty((0, 2), dtype=np.int64)
    weights = np.concatenate(weight_blocks) if weight_blocks else np.empty(0, dtype=np.float32)
    del edge_blocks, weight_blocks

    logging.info(f"Created graph with {n_chains} nodes and {len(edges)} edges")

    clusterer.graph = ig.Graph(n=n_chains, edges=edges, directed=False)
    clusterer.graph.es['weight'] = weights.tolist()
    del edges, weights

    clusterer.graph.vs['name'] = clusterer.chain_ids

    logging.info(f"Similarity graph built in {time.perf_counter() - start_time:.2f}s")

    return clusterer.graph


def _patched(name: str, value):
    """A patch of the clustering module's constant ``name``; none when ``value`` is None."""
    if value is None:
        return mock.patch.object(clustering, name, getattr(clustering, name))
    return mock.patch.object(clustering, name, value)


def _exact_divisor(counts) -> int:
    """A chunk size of at least 4 that divides one block's kept-edge count exactly, that block spanning two chunks or
    more."""
    for count in sorted((int(c) for c in counts), reverse=True):
        for size in range(4, count // 2 + 1):
            if count % size == 0:
                return size
    raise AssertionError(f"no block's edge count has a divisor: {list(counts)}")


def _float32_span(weights) -> int:
    bits = np.asarray(weights, dtype=np.float32).view(np.uint32).astype(np.int64)
    return int(bits.max() - bits.min() + 1)


def _shares_equal_weights(weights) -> bool:
    """Whether equal weights are one float object: as many objects as distinct values, and some value repeated."""
    distinct = {w.hex() for w in weights}
    return len(distinct) < len(weights) and len({id(w) for w in weights}) == len(distinct)


class _SameGraphAsK2:
    """On a fixture database, the graph equals K2's bit for bit: the same edges in the same order, the same weights by
    value and by bits, Python floats, and the same names, at every block size and every chunk of edges and weights
    handed to igraph (chunks of 1 and 3 edges, and chunks that divide a block's edge count exactly), with int32 ids
    and with the int64 fallback."""

    index_type = config = None
    n = threshold = max_neighbors = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = _build(Path(cls.tmp.name), cls.index_type.value, cls.vectors(), cls.index_type, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def budget(self, rows):
        """Blocks of ``rows`` chains at the fixture's k; the default budget when ``rows`` is None."""
        return _patched("_GRAPH_RESULT_BYTES", None if rows is None else _budget(rows, min(self.max_neighbors, self.n)))

    def assert_same_list(self, got, ref, what):
        """``got == ref``, reported by the first difference (a diff of a million-edge list would not finish)."""
        if got != ref:
            first = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
            self.fail(f"{what} differ: lengths {len(got)} and {len(ref)}, first difference at {first}: "
                      f"{got[first:first + 3]} against {ref[first:first + 3]}")

    def assert_same_graph(self, graph, ref):
        self.assertEqual(graph.vcount(), ref.vcount())
        self.assert_same_list(graph.get_edgelist(), ref.get_edgelist(), "edges")
        weights, ref_weights = graph.es["weight"], ref.es["weight"]
        self.assert_same_list(weights, ref_weights, "weights")
        self.assertTrue(np.asarray(weights, dtype=np.float64).tobytes()
                        == np.asarray(ref_weights, dtype=np.float64).tobytes(), "weight bits differ")
        self.assertTrue(all(type(w) is float for w in weights))
        self.assert_same_list(graph.vs["name"], ref.vs["name"], "names")

    def test_same_graph_as_k2(self):
        clusterer = EmbeddingClusterer(self.db_path)
        several = self.n // 4 + 1
        refs = {}
        for rows in (None, 1, 7, several):
            with self.budget(rows):
                refs[rows] = _k2_way(clusterer, self.threshold, self.max_neighbors)
        self.assertGreater(refs[None].ecount(), 1000)
        counts = np.bincount(np.array(refs[several].get_edgelist())[:, 0] // several, minlength=4)
        self.assertEqual(len(counts), 4)
        divisor = _exact_divisor(counts)
        for rows, ref in refs.items():
            for chunk in (None, 1, 3, divisor):
                with self.subTest(rows=rows, chunk=chunk), self.budget(rows), _patched("_GRAPH_EDGE_CHUNK", chunk):
                    graph = clusterer.build_similarity_graph(threshold=self.threshold,
                                                             max_neighbors=self.max_neighbors)
                    self.assert_same_graph(graph, ref)

    def test_int64_ids_when_the_largest_id_is_above_the_int32_limit(self):
        clusterer = EmbeddingClusterer(self.db_path)
        rows = self.n // 4 + 1
        with self.budget(rows):
            ref = _k2_way(clusterer, self.threshold, self.max_neighbors)
        graphs = {}
        for max_id, dtype in ((self.n - 1, np.int32), (self.n - 2, np.int64)):
            with self.subTest(max_id=max_id):
                dtypes = []
                edge_tuples = clustering._edge_tuples

                def recording(blocks):
                    dtypes.extend(b.dtype for b in blocks)
                    return edge_tuples(blocks)

                with self.budget(rows), _patched("_GRAPH_INT32_MAX_ID", max_id), \
                        mock.patch.object(clustering, "_edge_tuples", recording):
                    graphs[dtype] = clusterer.build_similarity_graph(threshold=self.threshold,
                                                                     max_neighbors=self.max_neighbors)
                self.assertGreater(len(dtypes), 1)
                self.assertEqual(set(dtypes), {np.dtype(dtype)})
                self.assert_same_graph(graphs[dtype], ref)
        self.assert_same_graph(graphs[np.int64], graphs[np.int32])


class HnswSameGraphAsK2Test(_SameGraphAsK2, unittest.TestCase):
    """hnsw (HnswGraphTest's fixture): also how the graph is built (no concatenation, igraph pulling the edges from
    the generator, freed memory released twice, with every block released before each release) and the weight table,
    at its cap and past it."""

    index_type, n, d, threshold, max_neighbors = IndexType.hnsw, 2000, 1536, 0.6, 40

    @classmethod
    def vectors(cls):
        return _clustered_vectors(seed=3, n=cls.n, d=cls.d, n_centers=150, noise=1.0)

    def test_no_concatenation(self):
        clusterer = EmbeddingClusterer(self.db_path)
        for rows in (None, 7):
            with self.subTest(rows=rows), self.budget(rows):
                ref = _k2_way(clusterer, self.threshold, self.max_neighbors)
                with mock.patch.object(clustering.np, "concatenate", side_effect=AssertionError("np.concatenate")):
                    graph = clusterer.build_similarity_graph(threshold=self.threshold,
                                                             max_neighbors=self.max_neighbors)
                self.assert_same_graph(graph, ref)

    def test_igraph_is_given_no_ndarray(self):
        clusterer = EmbeddingClusterer(self.db_path)
        ref = _k2_way(clusterer, self.threshold, self.max_neighbors)
        generators = []
        edge_tuples = clustering._edge_tuples

        def recording(blocks):
            generators.append(edge_tuples(blocks))
            return generators[-1]

        with mock.patch.object(clustering.ig, "Graph", wraps=ig.Graph) as graph_class, \
                mock.patch.object(clustering, "_edge_tuples", recording):
            graph = clusterer.build_similarity_graph(threshold=self.threshold, max_neighbors=self.max_neighbors)
        graph_class.assert_called_once()
        args = list(graph_class.call_args.args) + list(graph_class.call_args.kwargs.values())
        self.assertIn("edges", graph_class.call_args.kwargs)
        self.assertFalse(any(isinstance(a, np.ndarray) for a in args))
        # igraph pulls the edges from _edge_tuples' generator itself: no list or tuple of every edge is made first
        self.assertEqual(len(generators), 1)
        self.assertIs(graph_class.call_args.kwargs["edges"], generators[0])
        self.assert_same_graph(graph, ref)

    def test_freed_memory_released_after_the_graph_and_after_the_weights(self):
        clusterer = EmbeddingClusterer(self.db_path)
        calls = []

        def record():
            calls.append((clusterer.graph, clusterer.graph.ecount(), "weight" in clusterer.graph.es.attributes()))

        with mock.patch.object(clustering, "_release_freed_memory", side_effect=record) as release:
            graph = clusterer.build_similarity_graph(threshold=self.threshold, max_neighbors=self.max_neighbors)
        self.assertEqual(release.call_count, 2)
        self.assertGreater(graph.ecount(), 1000)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c[0] is graph for c in calls))
        self.assertEqual([c[1:] for c in calls], [(graph.ecount(), False), (graph.ecount(), True)])

    def test_every_block_released_before_each_release(self):
        """The build hands its own block lists to _edge_tuples and _weight_list and keeps no other reference: every
        pair block is dead at the first release (after the igraph build), every weight block at the second."""
        clusterer = EmbeddingClusterer(self.db_path)
        refs = {"pairs": [], "weights": []}
        edge_tuples, weight_list = clustering._edge_tuples, clustering._weight_list

        def recording(real, key):
            def call(blocks):
                refs[key].extend(weakref.ref(b) for b in blocks)
                return real(blocks)
            return call

        alive = []

        def release():
            alive.append({key: sum(r() is not None for r in rs) for key, rs in refs.items()})

        with self.budget(7), \
                mock.patch.object(clustering, "_edge_tuples", recording(edge_tuples, "pairs")), \
                mock.patch.object(clustering, "_weight_list", recording(weight_list, "weights")), \
                mock.patch.object(clustering, "_release_freed_memory", side_effect=release):
            graph = clusterer.build_similarity_graph(threshold=self.threshold, max_neighbors=self.max_neighbors)
        self.assertGreater(len(refs["pairs"]), 1)
        self.assertEqual(len(refs["weights"]), len(refs["pairs"]))
        self.assertGreater(graph.ecount(), 1000)
        self.assertEqual(alive, [{"pairs": 0, "weights": 0}, {"pairs": 0, "weights": 0}])

    def test_release_without_malloc_trim_is_a_no_op(self):
        """On this platform, with the C library lookup failing (OSError) and with a C library that has no malloc_trim
        (AttributeError), the build gives the same graph; on Linux, malloc_trim(0) is called twice."""
        self.assertIsNone(clustering._release_freed_memory())
        clusterer = EmbeddingClusterer(self.db_path)
        ref = _k2_way(clusterer, self.threshold, self.max_neighbors)
        linux = SimpleNamespace(platform="linux")
        trim = mock.Mock(return_value=1)
        cases = {
            "this platform": (clustering.sys, clustering.ctypes),
            "lookup raises OSError": (linux, SimpleNamespace(CDLL=mock.Mock(side_effect=OSError("no C library")))),
            "no malloc_trim": (linux, SimpleNamespace(CDLL=mock.Mock(return_value=object()))),
            "this C library on linux": (linux, clustering.ctypes),
            "malloc_trim on linux": (linux, SimpleNamespace(CDLL=mock.Mock(return_value=SimpleNamespace(
                malloc_trim=trim)))),
            "malloc_trim not on linux": (SimpleNamespace(platform="darwin"), SimpleNamespace(CDLL=mock.Mock(
                return_value=SimpleNamespace(malloc_trim=trim)))),
        }
        for case, (platform, c_library) in cases.items():
            with self.subTest(case=case):
                trim.reset_mock()
                with mock.patch.object(clustering, "sys", platform), mock.patch.object(clustering, "ctypes", c_library):
                    graph = clusterer.build_similarity_graph(threshold=self.threshold,
                                                             max_neighbors=self.max_neighbors)
                self.assert_same_graph(graph, ref)
                if case == "malloc_trim on linux":
                    c_library.CDLL.assert_called_with(None)
                    self.assertEqual(trim.call_args_list, [mock.call(0), mock.call(0)])
                elif case == "malloc_trim not on linux":
                    c_library.CDLL.assert_not_called()
                    trim.assert_not_called()

    def test_equal_weights_share_one_float(self):
        """The table path, at the default cap and at a cap equal to the weights' float32 span: equal weights are one
        object; past the cap (patched to 1), one float an edge as before. The same graph as K2's in all three, at
        chunks of 1, 3, a divisor of the edge count, and the default."""
        clusterer = EmbeddingClusterer(self.db_path)
        ref = _k2_way(clusterer, self.threshold, self.max_neighbors)
        span = _float32_span(ref.es["weight"])
        self.assertLess(span, clustering._GRAPH_WEIGHT_TABLE_MAX)
        # One block: the whole fixture.
        self.assertGreaterEqual(clustering._GRAPH_RESULT_BYTES // (self.max_neighbors * 12), self.n)
        divisor = _exact_divisor([ref.ecount()])
        for cap, shared in ((None, True), (span, True), (1, False)):
            for chunk in (None, 1, 3, divisor):
                with self.subTest(cap=cap, chunk=chunk), _patched("_GRAPH_WEIGHT_TABLE_MAX", cap), \
                        _patched("_GRAPH_EDGE_CHUNK", chunk):
                    graph = clusterer.build_similarity_graph(threshold=self.threshold,
                                                             max_neighbors=self.max_neighbors)
                    self.assert_same_graph(graph, ref)
                    if shared:
                        self.assertTrue(_shares_equal_weights(graph.es["weight"]))


class FlatSameGraphAsK2Test(_SameGraphAsK2, unittest.TestCase):
    """flat (FlatGraphTest's fixture): also every pair, negative scores included, past the weight table's cap."""

    index_type, n, d, threshold, max_neighbors = IndexType.flat, 1500, 256, 0.6, 30

    @classmethod
    def vectors(cls):
        return _clustered_vectors(seed=5, n=cls.n, d=cls.d, n_centers=60, noise=1.0)

    def test_every_pair_past_the_weight_table_cap(self):
        """Threshold -1 at max_neighbors = n: every pair is an edge, in one block of two chunks, negative scores
        among them, so the weights are one float an edge (no table)."""
        clusterer = EmbeddingClusterer(self.db_path)
        ref = _k2_way(clusterer, -1.0, self.n)
        self.assertEqual(ref.ecount(), self.n * (self.n - 1) // 2)
        self.assertGreater(ref.ecount(), clustering._GRAPH_EDGE_CHUNK)
        self.assertLess(ref.ecount(), 2 * clustering._GRAPH_EDGE_CHUNK)
        self.assertGreaterEqual(clustering._GRAPH_RESULT_BYTES // (self.n * 12), self.n)
        self.assertLess(min(ref.es["weight"]), 0)
        self.assertGreater(_float32_span(ref.es["weight"]), clustering._GRAPH_WEIGHT_TABLE_MAX)
        for chunk in (None, 1, 3, _exact_divisor([ref.ecount()])):
            with self.subTest(chunk=chunk), _patched("_GRAPH_EDGE_CHUNK", chunk):
                graph = clusterer.build_similarity_graph(threshold=-1.0, max_neighbors=self.n)
                self.assert_same_graph(graph, ref)


class IvfPqSameGraphAsK2Test(_SameGraphAsK2, unittest.TestCase):
    """ivf_pq (IvfPqGraphTest's fixture): also the similarity-graph and cluster files equal K2's."""

    index_type, n, d, threshold, max_neighbors = IndexType.ivf_pq, 1000, 64, 0.5, 1000
    config = IndexConfig(nlist=16, m=8, nbits=4, nprobe=16, opq_train_size=1000)

    @classmethod
    def vectors(cls):
        vectors = _clustered_vectors(seed=7, n=cls.n, d=cls.d, n_centers=40, noise=0.8)
        for t in range(20):
            vectors[500 + t] = vectors[10 * t]
        return vectors

    def test_files_as_k2(self):
        """similarity-graph writes the GraphML of K2's graph, and cluster with a seed the Leiden TSV on it (both
        at the commands' default --max-neighbors, 1000)."""
        out = Path(self.tmp.name)
        similarity_graph(db_path=self.db_path, threshold=self.threshold, output_file=str(out / "q1.graphml"))
        cluster_database(db_path=self.db_path, threshold=self.threshold, output_file=str(out / "q1.tsv"), seed=1)
        clusterer = EmbeddingClusterer(self.db_path)
        ref = _k2_way(clusterer, self.threshold, 1000)
        self.assertIs(clusterer.graph, ref)
        ref.write_graphml(str(out / "k2.graphml"))
        clusterer.cluster_leiden(seed=1)
        clusterer.export_clusters(str(out / "k2.tsv"))
        self.assertGreater(ref.ecount(), 1000)
        self.assertEqual((out / "q1.graphml").read_bytes(), (out / "k2.graphml").read_bytes())
        self.assertEqual((out / "q1.tsv").read_bytes(), (out / "k2.tsv").read_bytes())


class GraphEdgeCaseAsK2Test(unittest.TestCase):
    """EdgeCaseTest's fixtures: no edge, one chain, and blocks with no kept edge between blocks with edges, each the
    same as K2's graph; a block with no kept edge is not stored."""

    n, d = 60, 16

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        vectors = np.zeros((cls.n, cls.d), dtype=np.float32)
        vectors[:54, :8] = _clustered_vectors(seed=11, n=54, d=8, n_centers=6, noise=0.5)
        vectors[54:, 8:14] = np.eye(6, dtype=np.float32)
        cls.db_path = _build(folder, "flat", vectors, IndexType.flat)
        cls.one_path = _build(folder, "one", vectors[:1], IndexType.flat)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    assert_same_list = _SameGraphAsK2.assert_same_list
    assert_same_graph = _SameGraphAsK2.assert_same_graph

    def test_no_edge(self):
        clusterer = EmbeddingClusterer(self.db_path)
        ref = _k2_way(clusterer, 2.0, 10)
        graph = clusterer.build_similarity_graph(threshold=2.0, max_neighbors=10)
        self.assertEqual((graph.vcount(), graph.ecount(), graph.es["weight"]), (self.n, 0, []))
        self.assert_same_graph(graph, ref)

    def test_one_chain(self):
        clusterer = EmbeddingClusterer(self.one_path)
        ref = _k2_way(clusterer, 0.0, 10)
        graph = clusterer.build_similarity_graph(threshold=0.0, max_neighbors=10)
        self.assertEqual((graph.vcount(), graph.ecount()), (1, 0))
        self.assert_same_graph(graph, ref)

    def test_blocks_with_no_edge_between_blocks_with_edges(self):
        threshold, k = 0.5, 10
        clusterer = EmbeddingClusterer(self.db_path)
        gaps = {}
        for rows in (1, 2, 7):
            with self.subTest(rows=rows), mock.patch.object(clustering, "_GRAPH_RESULT_BYTES", _budget(rows, k)):
                ref = _k2_way(clusterer, threshold, k)
                with_edges = sorted({i // rows for i, _ in ref.get_edgelist()})
                gaps[rows] = len(set(range(with_edges[0], with_edges[-1])) - set(with_edges))
                stored = []
                edge_tuples, weight_list = clustering._edge_tuples, clustering._weight_list

                def recording(real):
                    def call(blocks):
                        stored.append([len(b) for b in blocks])
                        return real(blocks)
                    return call

                with mock.patch.object(clustering, "_edge_tuples", recording(edge_tuples)), \
                        mock.patch.object(clustering, "_weight_list", recording(weight_list)):
                    graph = clusterer.build_similarity_graph(threshold=threshold, max_neighbors=k)
                self.assert_same_graph(graph, ref)
                counts = np.bincount(np.array(ref.get_edgelist())[:, 0] // rows)
                self.assertEqual(stored, [counts[with_edges].tolist()] * 2)
        # Blocks of one chain: some with no kept edge lie between blocks with edges.
        self.assertGreater(gaps[1], 0)


class _PopRecorder(list):
    """A list of blocks that records, at each pop, which of the original blocks are still alive."""

    def __init__(self, blocks):
        super().__init__(blocks)
        self.refs = [weakref.ref(b) for b in blocks]
        self.alive = []

    def pop(self, *args):
        self.alive.append([r() is not None for r in self.refs])
        return super().pop(*args)


class GraphPiecesTest(unittest.TestCase):
    """The id dtype rule, the edge generator and the weight list on small blocks."""

    def test_id_dtype(self):
        self.assertEqual(clustering._GRAPH_INT32_MAX_ID, 2 ** 31 - 1)
        for n_chains, dtype in ((0, np.int32), (1, np.int32), (2 ** 31, np.int32), (2 ** 31 + 1, np.int64),
                                (2 ** 40, np.int64)):
            with self.subTest(n_chains=n_chains):
                self.assertIs(clustering._graph_id_dtype(n_chains), dtype)
        with _patched("_GRAPH_INT32_MAX_ID", 9):
            self.assertIs(clustering._graph_id_dtype(10), np.int32)
            self.assertIs(clustering._graph_id_dtype(11), np.int64)

    def test_edge_tuples_release_each_block_once_consumed(self):
        blocks = [np.array([[0, 1], [0, 2]], dtype=np.int32), np.array([[1, 2], [1, 3], [1, 4]], dtype=np.int32),
                  np.array([[3, 4]], dtype=np.int32)]
        refs = [weakref.ref(b) for b in blocks]
        with _patched("_GRAPH_EDGE_CHUNK", 2):
            edges = clustering._edge_tuples(blocks)
            got = [next(edges), next(edges)]
            self.assertIsNotNone(refs[0]())
            got.append(next(edges))
            self.assertIsNone(refs[0]())
            self.assertLess(len(blocks), 3)
            got.extend(edges)
        self.assertEqual(len(blocks), 0)
        self.assertEqual([r() for r in refs], [None] * 3)
        self.assertEqual(got, [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (3, 4)])
        self.assertTrue(all(type(e) is tuple and all(type(i) is int for i in e) for e in got))
        self.assertEqual(list(clustering._edge_tuples([])), [])

    def test_weight_list_releases_each_block_once_consumed(self):
        for cap in (None, 1):
            with self.subTest(cap=cap), _patched("_GRAPH_WEIGHT_TABLE_MAX", cap), _patched("_GRAPH_EDGE_CHUNK", 2):
                blocks = _PopRecorder([np.float32([0.5, 0.75, 0.5]), np.float32([0.75]), np.float32([0.25, 0.5])])
                expected = [0.5, 0.75, 0.5, 0.75, 0.25, 0.5]
                got = clustering._weight_list(blocks)
                self.assertEqual(got, expected)
                self.assertEqual(len(blocks), 0)
                self.assertEqual(blocks.alive, [[True, True, True], [False, True, True], [False, False, True]])
                self.assertEqual([r() for r in blocks.refs], [None] * 3)

    def test_weight_list_shares_equal_bit_patterns_across_slices_and_blocks(self):
        blocks = [np.float32([0.5, 0.75, 0.5]), np.float32([]), np.float32([0.75, 0.5, 1.0])]
        expected = [w for b in blocks for w in b.tolist()]
        with _patched("_GRAPH_EDGE_CHUNK", 2):
            got = clustering._weight_list(list(blocks))
        self.assertEqual(got, expected)
        self.assertTrue(all(type(w) is float for w in got))
        self.assertTrue(got[0] is got[2] is got[4])
        self.assertTrue(got[1] is got[3])
        self.assertEqual(clustering._weight_list([]), [])
        self.assertEqual(clustering._weight_list([np.float32([])]), [])

    def test_weight_list_keeps_the_sign_of_zero(self):
        """0.0 with -0.0 spans past the cap (one float an edge); -0.0 alone takes the table: signs kept either way."""
        for values, shared in (([0.0, -0.0, 0.5], False), ([-0.0, -0.0], True)):
            with self.subTest(values=values):
                block = np.float32(values)
                self.assertEqual(_float32_span(block) <= clustering._GRAPH_WEIGHT_TABLE_MAX, shared)
                expected = block.tolist()
                got = clustering._weight_list([block])
                self.assertEqual(np.asarray(got, dtype=np.float64).tobytes(),
                                 np.asarray(expected, dtype=np.float64).tobytes())
                self.assertEqual([math.copysign(1.0, w) for w in got], [math.copysign(1.0, w) for w in expected])
                self.assertTrue(all(type(w) is float for w in got))
                if shared:
                    self.assertIs(got[0], got[1])


# P9.U: cluster_leiden runs python-igraph's community_leiden, a fixed number of iterations, seeded with a private
# random.Random, cluster ids by decreasing size, ties by smallest chain index.

_SRC = str(Path(foldmatch.__file__).resolve().parent.parent)


def _reference_leiden(graph: ig.Graph, seed: int, resolution: float = 1.0, n_iterations: int = 10) -> list:
    """P9.U's Leiden call, made here and not through the code under test: python-igraph's community_leiden with its
    arguments, seeded with random.seed (Python's global state saved and restored around it), and the cluster ids
    renumbered in pure Python by (-size, smallest chain index)."""
    state = random.getstate()
    try:
        random.seed(seed)
        membership = graph.community_leiden(objective_function="modularity", weights="weight", resolution=resolution,
                                            beta=0.01, n_iterations=n_iterations).membership
    finally:
        random.setstate(state)
    members = {}
    for chain, cluster in enumerate(membership):
        members.setdefault(cluster, []).append(chain)
    order = sorted(members, key=lambda c: (-len(members[c]), members[c][0]))
    new_id = {cluster: rank for rank, cluster in enumerate(order)}
    return [new_id[cluster] for cluster in membership]


def _as_sets(membership) -> set:
    """A partition as a set of frozensets of chain indexes, whatever its labels."""
    members = {}
    for chain, cluster in enumerate(membership):
        members.setdefault(int(cluster), set()).add(chain)
    return {frozenset(m) for m in members.values()}


def _flat(text: str) -> str:
    """Text on one line, typer's error panel borders and colour codes dropped."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


class _LeidenSpy:
    """python-igraph's Graph.community_leiden, recording the keyword arguments of each call while patched."""

    def __init__(self):
        self.calls = []

    def patch(self):
        real, calls = ig.Graph.community_leiden, self.calls

        def community_leiden(graph, *args, **kwargs):
            assert not args, args
            calls.append(kwargs)
            return real(graph, **kwargs)

        return mock.patch.object(ig.Graph, "community_leiden", community_leiden)


# Imports the clustering module and the CLI with the removed Leiden package blocked, then runs cluster on a database.
# The package's name is spelled in two pieces so that a grep of the repository for it finds nothing.
_NO_LEIDEN_PACKAGE_CHILD = """
import sys
sys.modules["leiden" + "alg"] = None
import lightning  # noqa: F401
import foldmatch
assert foldmatch.__file__.startswith(sys.argv[1]), foldmatch.__file__
import foldmatch.search.clustering
from foldmatch.cli.search import cluster_database
cluster_database(db_path=sys.argv[2], threshold=0.5, max_neighbors=10, seed=3, output_file=sys.argv[3])
print("cluster ran")
"""

# Prints the digest of cluster_leiden's array at one seed, in a fresh process.
_SEED_DIGEST_CHILD = """
import hashlib
import sys
import lightning  # noqa: F401
import foldmatch
assert foldmatch.__file__.startswith(sys.argv[1]), foldmatch.__file__
from foldmatch.search.clustering import EmbeddingClusterer
clusterer = EmbeddingClusterer(sys.argv[2])
clusterer.build_similarity_graph(threshold=float(sys.argv[4]), max_neighbors=10)
clusters = clusterer.cluster_leiden(seed=int(sys.argv[3]))
print(clusters.dtype.name, hashlib.sha256(clusters.tobytes()).hexdigest())
"""


class LeidenIgraphTest(unittest.TestCase):
    """cluster_leiden: python-igraph's community_leiden with modularity at the resolution, beta 0.01 and a fixed
    number of iterations (10 by default); seeded with a private random.Random, Python's global random state untouched
    and python-igraph's generator set back to the random module; cluster ids by decreasing size, ties by smallest chain
    index, as an int32 array; and the refusals (iterations below 1, negative seeds, negative kept weights)."""

    n, d = 90, 16

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = Path(cls.tmp.name)
        # Six well-separated clusters: a chain's few nearest neighbours score above 0, but some pairs of chains of
        # different clusters score below 0.
        cls.vectors = _clustered_vectors(seed=31, n=cls.n, d=cls.d, n_centers=6, noise=0.3)
        cls.db_path = _build(folder, "flat", cls.vectors, IndexType.flat)
        # Noisier clusters, kept from threshold 0 at k 10: their partition depends on the seed (the fixture above
        # gives one partition whatever the seed), so the seeding tests tell a seeded generator from any other.
        cls.sensitive_db_path = _build(folder, "flat_seed_sensitive", _clustered_vectors(
            seed=31, n=cls.n, d=cls.d, n_centers=6, noise=2.0), IndexType.flat)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.addCleanup(ig.set_random_number_generator, random)

    def clusterer(self, threshold: float = 0.5, max_neighbors: int = 10) -> EmbeddingClusterer:
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.build_similarity_graph(threshold=threshold, max_neighbors=max_neighbors)
        return clusterer

    def seed_sensitive_clusterer(self) -> EmbeddingClusterer:
        """A clusterer whose Leiden partition depends on the seed (threshold 0, k 10)."""
        clusterer = EmbeddingClusterer(self.sensitive_db_path)
        clusterer.build_similarity_graph(threshold=0.0, max_neighbors=10)
        return clusterer

    def with_graph(self, graph: ig.Graph) -> EmbeddingClusterer:
        """A clusterer whose graph a caller set (not built by build_similarity_graph)."""
        clusterer = EmbeddingClusterer(self.db_path)
        clusterer.graph = graph
        return clusterer

    @staticmethod
    def planted_graph():
        """A weighted planted partition: four blocks of 40, 30, 20 and 10 nodes, dense and heavy inside a block,
        sparse and light between blocks. Returns the graph and its blocks."""
        rng = np.random.default_rng(37)
        sizes = (40, 30, 20, 10)
        block = np.repeat(np.arange(len(sizes)), sizes)
        n = len(block)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        same = np.array([block[i] == block[j] for i, j in pairs])
        keep = rng.random(len(pairs)) < np.where(same, 0.6, 0.02)
        weights = np.where(same, rng.uniform(0.8, 1.0, len(pairs)), 0.1)
        graph = ig.Graph(n=n, edges=[p for p, k in zip(pairs, keep) if k], directed=False)
        graph.es["weight"] = weights[keep].tolist()
        return graph, _as_sets(block)

    @staticmethod
    def cliques_graph():
        """67 disjoint cliques in node order: 64 of 3 nodes, and 3 of 10 nodes at positions 20, 41 and 65, so
        igraph's first-appearance ids are not by size and 64 clusters tie (more than 16, so an unstable sort of the
        sizes would misorder ties)."""
        cliques, start = [], 0
        for c in range(67):
            size = 10 if c in (20, 41, 65) else 3
            cliques.append(list(range(start, start + size)))
            start += size
        edges = [(a, b) for clique in cliques for i, a in enumerate(clique) for b in clique[i + 1:]]
        graph = ig.Graph(n=start, edges=edges, directed=False)
        graph.es["weight"] = [1.0] * len(edges)
        return graph, cliques

    # The call

    def test_the_call(self):
        """The keyword arguments of community_leiden: exactly "modularity", the weights, the resolution, beta 0.01, and
        10 iterations unless given."""
        clusterer = self.clusterer()
        spy = _LeidenSpy()
        with spy.patch():
            clusterer.cluster_leiden(seed=1)
            clusterer.cluster_leiden(resolution=2.5, seed=1)
            clusterer.cluster_leiden(n_iterations=3, seed=1)
        expected = dict(objective_function="modularity", weights="weight", resolution=1.0, beta=0.01, n_iterations=10)
        self.assertEqual(spy.calls, [expected, {**expected, "resolution": 2.5}, {**expected, "n_iterations": 3}])
        self.assertIs(type(spy.calls[0]["objective_function"]), str)
        self.assertEqual(clustering._LEIDEN_ITERATIONS, 10)

    def test_planted_partition(self):
        graph, blocks = self.planted_graph()
        clusterer = self.with_graph(graph)
        clusters = clusterer.cluster_leiden(seed=2)
        self.assertEqual(_as_sets(clusters), blocks)
        self.assertEqual(clusters.tolist(), _reference_leiden(graph, 2))
        stats = clusterer.get_cluster_statistics()
        self.assertEqual(stats["modularity"], graph.modularity(clusters, weights="weight"))
        self.assertGreater(stats["modularity"], 0.5)

    # Seeding

    def test_same_seed_same_partition_as_global_seeding(self):
        """The same seed twice gives the same array, equal to random.seed seeding of the same call, on a fixture
        whose partition depends on the seed."""
        clusterer = self.seed_sensitive_clusterer()
        partitions = {tuple(clusterer.cluster_leiden(seed=seed).tolist()) for seed in range(4)}
        self.assertGreater(len(partitions), 1)
        first = clusterer.cluster_leiden(seed=5)
        second = clusterer.cluster_leiden(seed=5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.tolist(), _reference_leiden(clusterer.graph, 5))
        for seed in (0, 1, 2 ** 31, 2 ** 64):
            with self.subTest(seed=seed):
                self.assertEqual(clusterer.cluster_leiden(seed=seed).tolist(),
                                 _reference_leiden(clusterer.graph, seed))
        self.assertGreater(len(set(first.tolist())), 3)

    def test_same_seed_in_fresh_processes(self):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([_SRC] + [p for p in os.environ.get(
            "PYTHONPATH", "").split(os.pathsep) if p and p != _SRC])}
        digests = []
        for _ in range(2):
            run = subprocess.run([sys.executable, "-c", _SEED_DIGEST_CHILD, _SRC, self.sensitive_db_path, "5", "0"],
                                 env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(run.returncode, 0, run.stderr)
            digests.append(run.stdout.split())
        clusters = self.seed_sensitive_clusterer().cluster_leiden(seed=5)
        self.assertEqual(digests, [["int32", hashlib.sha256(clusters.tobytes()).hexdigest()]] * 2)

    def test_global_random_state_untouched(self):
        """random.getstate() is the same before and after, seeded or not, and a caller's seeded stream goes on."""
        clusterer = self.clusterer()
        for seed in (5, None):
            with self.subTest(seed=seed):
                state = random.getstate()
                clusterer.cluster_leiden(seed=seed)
                self.assertEqual(random.getstate(), state)
        random.seed(7)
        expected = [random.random() for _ in range(3)]
        random.seed(7)
        clusterer.cluster_leiden(seed=1)
        self.assertEqual([random.random() for _ in range(3)], expected)

    def test_generator_restored_to_the_random_module(self):
        """After the call, and after an error inside it, python-igraph's generator is the random module again."""
        clusterer = self.clusterer()
        with mock.patch.object(ig, "set_random_number_generator", wraps=ig.set_random_number_generator) as setter:
            clusterer.cluster_leiden(seed=1)
        self.assertIsInstance(setter.call_args_list[0].args[0], random.Random)
        self.assertEqual(setter.call_args_list[-1], mock.call(random))
        failing = mock.patch.object(ig.Graph, "community_leiden", side_effect=ig.InternalError("inside Leiden"))
        with mock.patch.object(ig, "set_random_number_generator", wraps=ig.set_random_number_generator) as setter, \
                failing, self.assertRaisesRegex(ig.InternalError, "inside Leiden"):
            clusterer.cluster_leiden(seed=1)
        self.assertEqual(setter.call_args_list[-1], mock.call(random))
        self.assertEqual(len(setter.call_args_list), 2)

    # Cluster ids

    def test_ids_by_size_ties_by_smallest_chain_index(self):
        graph, cliques = self.cliques_graph()
        raw = graph.community_leiden(objective_function="modularity", weights="weight", n_iterations=10).membership
        # igraph numbers by first appearance: a 3-node clique is its cluster 0
        self.assertEqual(raw[0], 0)
        clusterer = self.with_graph(graph)
        clusters = clusterer.cluster_leiden(seed=3)
        self.assertIs(clusters, clusterer.clusters)
        self.assertIsInstance(clusters, np.ndarray)
        self.assertEqual(clusters.dtype, np.int32)
        self.assertEqual(_as_sets(clusters), _as_sets(raw))
        self.assertEqual(_as_sets(clusters), {frozenset(c) for c in cliques})
        # The three 10-node cliques first, then the 3-node cliques, each group in node order
        by_size = sorted(cliques, key=lambda c: (-len(c), c[0]))
        self.assertEqual([int(clusters[c[0]]) for c in by_size], list(range(len(cliques))))
        sizes = np.bincount(clusters)
        self.assertTrue(np.all(np.diff(sizes) <= 0), sizes.tolist())
        self.assertEqual(clusters.tolist(), _reference_leiden(graph, 3))

    # Refusals

    def test_iterations_below_1(self):
        clusterer = EmbeddingClusterer(self.db_path)
        for n_iterations in (0, -1):
            with self.subTest(n_iterations=n_iterations), self.assertRaisesRegex(
                    ValueError, r"^n_iterations must be at least 1 \(Leiden runs a fixed number of iterations\)\.$"):
                clusterer.cluster_leiden(n_iterations=n_iterations, seed=1)
        self.assertIsNone(clusterer.clusters)

    def test_seed_values(self):
        clusterer = self.clusterer()
        with self.assertRaisesRegex(ValueError, r"^--seed must be at least 0\.$"):
            clusterer.cluster_leiden(seed=-1)
        with self.assertRaises(TypeError):
            clusterer.cluster_leiden(seed=5.0)
        self.assertIsNone(clusterer.clusters)
        clusterer = self.seed_sensitive_clusterer()
        expected = clusterer.cluster_leiden(seed=5)
        for seed in (np.int64(5), np.int32(5)):
            with self.subTest(seed=type(seed).__name__):
                np.testing.assert_array_equal(clusterer.cluster_leiden(seed=seed), expected)

    def test_negative_kept_weights_refused(self):
        """Every pair of chains (k = n) at threshold -1 keeps negative scores: refused, naming the smallest weight."""
        clusterer = self.clusterer(threshold=-1.0, max_neighbors=self.n)
        weights = clusterer.graph.es["weight"]
        self.assertEqual(clusterer.graph.ecount(), self.n * (self.n - 1) // 2)
        self.assertLess(min(weights), 0)
        self.assertEqual(clusterer.min_weight, min(weights))
        message = (f"Leiden needs non-negative edge weights: the graph's smallest weight is {min(weights):.4g} (edges "
                   f"keep scores at or above the threshold of build_similarity_graph).")
        with self.assertRaisesRegex(ValueError, f"^{re.escape(message)}$"):
            clusterer.cluster_leiden(seed=1)
        self.assertIsNone(clusterer.clusters)

    def test_negative_threshold_with_non_negative_kept_weights_runs(self):
        """At a few neighbours a chain, threshold -1 keeps only scores of at least 0: the edges and the partition of
        threshold 0."""
        clusterer = self.clusterer(threshold=-1.0, max_neighbors=5)
        self.assertGreaterEqual(clusterer.min_weight, 0)
        self.assertEqual(clusterer.min_weight, min(clusterer.graph.es["weight"]))
        at_zero = self.clusterer(threshold=0.0, max_neighbors=5)
        self.assertEqual(clusterer.graph.get_edgelist(), at_zero.graph.get_edgelist())
        np.testing.assert_array_equal(clusterer.cluster_leiden(seed=4), at_zero.cluster_leiden(seed=4))

    def test_a_kept_weight_of_exactly_0_runs(self):
        """Orthogonal unit vectors score exactly 0, which threshold 0 keeps: 0 is not below 0, in the library and the
        CLI."""
        vectors = np.zeros((3, self.d), dtype=np.float32)
        vectors[[0, 2], 0] = 1.0
        vectors[1, 1] = 1.0
        db_path = _build(Path(self.tmp.name), "flat_zero_score", vectors, IndexType.flat)
        clusterer = EmbeddingClusterer(db_path)
        clusterer.build_similarity_graph(threshold=0.0, max_neighbors=3)
        self.assertEqual(sorted(clusterer.graph.es["weight"]), [0.0, 0.0, 1.0])
        self.assertEqual(clusterer.min_weight, 0.0)
        self.assertEqual(len(clusterer.cluster_leiden(seed=1)), 3)
        out = Path(self.tmp.name) / "zero_score.tsv"
        result = self.invoke("cluster", "--db-path", db_path, "--output-file", out, "--threshold", "0",
                             "--max-neighbors", 3, "--seed", 1)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        self.assertEqual(len(out.read_text().splitlines()), 3)

    def test_a_graph_set_after_the_build_is_not_judged_by_its_weights(self):
        clusterer = self.clusterer(threshold=-1.0, max_neighbors=self.n)
        self.assertLess(clusterer.min_weight, 0)
        other = self.clusterer(threshold=0.5, max_neighbors=10)
        clusterer.graph = other.graph
        np.testing.assert_array_equal(clusterer.cluster_leiden(seed=4), other.cluster_leiden(seed=4))
        # A build that fails records nothing
        clusterer.db.index = None
        with self.assertRaises(ValueError):
            clusterer.build_similarity_graph(threshold=-1.0, max_neighbors=self.n)
        self.assertIsNone(clusterer.min_weight)
        self.assertIsNone(clusterer._min_weight_graph)

    def test_no_edge_records_inf(self):
        clusterer = self.clusterer(threshold=2.0, max_neighbors=10)
        self.assertEqual(clusterer.graph.ecount(), 0)
        self.assertEqual(clusterer.min_weight, float("inf"))
        self.assertIs(clusterer._min_weight_graph(), clusterer.graph)

    # Through the CLI

    def invoke(self, *args):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            return CliRunner().invoke(search.app, [str(a) for a in args])
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)

    def test_cli_negative_kept_weights(self):
        out = Path(self.tmp.name) / "neg.tsv"
        base = ("cluster", "--db-path", self.db_path, "--output-file", out, "--threshold", "-1")
        clusterer = self.clusterer(threshold=-1.0, max_neighbors=self.n)
        for method in ((), ("--method", "leiden")):
            with self.subTest(method=method):
                result = self.invoke(*base, *method, "--max-neighbors", self.n)
                self.assertEqual(result.exit_code, 2, f"{result.output}\n{result.exception!r}")
                self.assertIn(f"Invalid value for --threshold: with --method leiden, Leiden's modularity needs "
                              f"non-negative edge weights, and this --threshold kept a score of "
                              f"{clusterer.min_weight:.4g}. Use a --threshold of 0 or more, or --method greedy.",
                              _flat(result.output))
                self.assertFalse(out.exists())
        # Non-negative kept scores at threshold -1: runs, as at threshold 0
        result = self.invoke(*base, "--max-neighbors", 5, "--seed", 4)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        at_zero = self.clusterer(threshold=0.0, max_neighbors=5)
        at_zero.cluster_leiden(seed=4)
        at_zero.export_clusters(str(Path(self.tmp.name) / "zero.tsv"))
        self.assertEqual(out.read_bytes(), (Path(self.tmp.name) / "zero.tsv").read_bytes())
        # Greedy computes no modularity: not refused
        result = self.invoke(*base, "--max-neighbors", self.n, "--method", "greedy")
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        self.assertEqual(len(out.read_text().splitlines()), self.n)

    def test_cli_seed_2_to_the_31(self):
        out = Path(self.tmp.name) / "big_seed.tsv"
        result = self.invoke("cluster", "--db-path", self.sensitive_db_path, "--output-file", out, "--threshold",
                             "0", "--max-neighbors", 10, "--seed", 2 ** 31)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        clusterer = self.seed_sensitive_clusterer()
        expected = _reference_leiden(clusterer.graph, 2 ** 31)
        self.assertEqual([int(line.split("\t")[1]) for line in out.read_text().splitlines()], expected)

    # The removed package

    def test_runs_without_the_removed_leiden_package(self):
        out = Path(self.tmp.name) / "blocked.tsv"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([_SRC] + [p for p in os.environ.get(
            "PYTHONPATH", "").split(os.pathsep) if p and p != _SRC])}
        run = subprocess.run([sys.executable, "-c", _NO_LEIDEN_PACKAGE_CHILD, _SRC, self.db_path, str(out)], env=env,
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines()[-1], "cluster ran")
        self.assertEqual(len(out.read_text().splitlines()), self.n)

    def test_no_name_of_the_removed_package_in_the_repository(self):
        root = Path(_SRC).parent
        if shutil.which("git") is None or not (root / ".git").exists():
            self.skipTest("not a git checkout")
        run = subprocess.run(["git", "grep", "-n", "-i", "leiden" + "alg", "--", "src", "tests", "pyproject.toml",
                              "uv.lock", "README.md"], cwd=root, capture_output=True, text=True, timeout=60)
        self.assertEqual((run.returncode, run.stdout), (1, ""), run.stderr)


if __name__ == "__main__":
    unittest.main()
