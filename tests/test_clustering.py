"""EmbeddingClusterer.build_similarity_graph: blocks of chains read with reconstruct_n and searched many at a time give,
on every database type, the expected graph: on hnsw the one-query-per-call graph bit for bit, on flat the same up to
float rounding, and on ivf_pq the normalised decoded query against each code."""
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

# As the commands run (set_log_level): Lightning first, which brings torch and, through scikit-learn,
# KMP_DUPLICATE_LIB_OK; on macOS, FAISS's OpenMP regions abort a process that loaded torch's libomp without it.
import lightning  # noqa: F401
import faiss
import numpy as np

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
        self.assertEqual(clusterer.cluster_leiden(seed=1), [0])
        stats = clusterer.get_cluster_statistics()
        self.assertEqual((stats["n_clusters"], stats["n_singletons"]), (1, 1))

    def test_threshold_above_every_score(self):
        clusterer = EmbeddingClusterer(self.db_path)
        graph = clusterer.build_similarity_graph(threshold=2.0, max_neighbors=10)
        self.assertEqual((graph.vcount(), graph.ecount()), (self.n, 0))
        self.assertEqual(graph.es["weight"], [])
        clusters = clusterer.cluster_leiden(seed=1)
        self.assertEqual(sorted(clusters), list(range(self.n)))
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


if __name__ == "__main__":
    unittest.main()
