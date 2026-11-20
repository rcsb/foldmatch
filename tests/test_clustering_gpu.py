"""GPU search for fm-search cluster and similarity-graph (P9.CL D10): ``FaissEmbeddingDatabase.move_to_gpu(clustering=
True)`` (one replica a visible GPU, not sharded; float16 lookup tables on ivf_pq, not on flat; the CPU index kept for
reconstruct_n and the code copy), the clusterer's nprobe set before the move, --use-gpu-index under every --method
(greedy-leiden copies the representatives' database from the CPU index, then moves it), and the refusals: FAISS without
GPU support (macOS's Metal backend included), k or nprobe above the GPU limit, hnsw.

There is no CUDA GPU here, so a fake GPU layer stands in for FAISS's: ``index_cpu_to_all_gpus`` returns an index that
searches the CPU index it was cloned from at the nprobe it had then (a GPU clone keeps its own), counts its searches and
refuses reconstruct_n. These tests check the routing and the refusals; the GPU's own results (float16 tables, the 2048
limit, the clone of an OPQ index with on-disk lists) are for a GPU node (V2.7, V3.7)."""
import json
import logging
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# As the commands run (set_log_level): Lightning first, which brings torch and, through scikit-learn,
# KMP_DUPLICATE_LIB_OK; on macOS, FAISS's OpenMP regions abort a process that loaded torch's libomp without it.
import lightning  # noqa: F401
import faiss
import numpy as np
from typer.testing import CliRunner

from foldmatch.cli import search
from foldmatch.errors import RefusalError
from foldmatch.search import clustering, faiss_database
from foldmatch.search.clustering import EmbeddingClusterer
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.types.api_types import IndexConfig, IndexType

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


def _load(db_path: str) -> FaissEmbeddingDatabase:
    path = Path(db_path)
    db = FaissEmbeddingDatabase(path.parent, path.name)
    db.load_database()
    return db


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


def _edges(edges):
    """The pairs and weights of a SimilarityEdges, in order."""
    pairs = list(edges.pair_blocks)
    weights = list(edges.weight_blocks)
    return (np.concatenate(pairs) if pairs else np.empty((0, 2)), np.concatenate(weights) if weights else np.empty(0))


class _FakeOptions:
    """GpuMultipleClonerOptions without a GPU: every option unset (None), so that a test sees which ones are set."""

    def __init__(self):
        self.shard = None
        self.useFloat16 = None


class _FakeGpuIndex:
    """A GPU index stand-in: it searches the CPU index it was cloned from at the nprobe that index had at the clone
    (a GPU clone keeps its own), records each search, and has no reconstruct_n (the stored vectors are read from the
    CPU index)."""

    def __init__(self, cpu_index, options, ngpu):
        self.cpu_index = cpu_index
        self.ntotal, self.d = cpu_index.ntotal, cpu_index.d
        self.options = {"shard": options.shard, "useFloat16": options.useFloat16}
        self.ngpu = ngpu
        self.ivf = None
        if not isinstance(faiss.downcast_index(cpu_index), faiss.IndexFlat):
            self.ivf = faiss.extract_index_ivf(cpu_index)
        self.nprobe = None if self.ivf is None else int(self.ivf.nprobe)
        self.searches = []

    def search(self, x, k):
        self.searches.append((len(x), k))
        if self.ivf is None:
            return self.cpu_index.search(x, k)
        saved = self.ivf.nprobe
        self.ivf.nprobe = self.nprobe
        try:
            return self.cpu_index.search(x, k)
        finally:
            self.ivf.nprobe = saved

    def reconstruct_n(self, *args):
        raise AssertionError("the stored vectors were read from the GPU index")


class FakeGpuLayer:
    """FAISS's GPU layer as FoldMatch sees it, on a machine without one: GPU support, ``n_gpus`` visible GPUs, and
    ``index_cpu_to_all_gpus`` recording each clone (``self.clones``)."""

    def __init__(self, n_gpus: int = 3):
        self.n_gpus = n_gpus
        self.clones = []

    def clone(self, index, co=None, ngpu=-1):
        gpu = _FakeGpuIndex(index, co, ngpu)
        self.clones.append(gpu)
        return gpu

    def __enter__(self):
        self.patches = [mock.patch.object(faiss_database, "_gpu_support_problem", return_value=None),
                        mock.patch.object(faiss, "get_num_gpus", return_value=self.n_gpus),
                        mock.patch.object(faiss, "GpuMultipleClonerOptions", _FakeOptions, create=True),
                        mock.patch.object(faiss, "index_cpu_to_all_gpus", self.clone, create=True)]
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self.patches):
            patch.stop()
        return False


class _Fixtures:
    """A flat database (400 x 64), an ivf_pq one (1000 x 64, nlist 16, PQ8x4, stored nprobe 4) with near-duplicates,
    and an hnsw one (300 x 16)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        flat = _clustered_vectors(seed=5, n=400, d=64, n_centers=25, noise=1.0)
        pq = _clustered_vectors(seed=7, n=1000, d=64, n_centers=40, noise=0.8)
        rng = np.random.default_rng(11)
        for vectors in (flat, pq):
            for i in range(100, len(vectors), 5):
                v = vectors[i - 97] + 0.05 * rng.standard_normal(vectors.shape[1]) / math.sqrt(vectors.shape[1])
                vectors[i] = v / np.linalg.norm(v)
        cls.flat = _build(cls.folder, "flat", flat, IndexType.flat)
        cls.pq = _build(cls.folder, "pq", pq, IndexType.ivf_pq, IndexConfig(**_PQ_CONFIG))
        cls.hnsw = _build(cls.folder, "hnsw", _clustered_vectors(seed=3, n=300, d=16, n_centers=20, noise=1.0),
                          IndexType.hnsw)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class GpuSupportTest(unittest.TestCase):
    """FAISS's GPU support: its CUDA layer and a visible GPU; macOS's Metal backend is not counted (F27)."""

    def problem(self, platform, **fake):
        with mock.patch.object(faiss_database, "sys", SimpleNamespace(platform=platform)), \
                mock.patch.object(faiss_database, "faiss", SimpleNamespace(__version__="1.15.1", **fake)):
            return faiss_database._gpu_support_problem(), faiss_database._has_gpu_support()

    def test_problems(self):
        gpu = dict(StandardGpuResources=object, get_num_gpus=lambda: 4)
        self.assertEqual(self.problem("linux", **gpu), (None, True))
        self.assertEqual(self.problem("darwin", **gpu),
                         ("the macOS FAISS wheel, whose Metal backend FoldMatch does not use", False))
        self.assertEqual(self.problem("linux", get_num_gpus=lambda: 4),
                         ("a FAISS 1.15.1 without GPU support (faiss-cpu)", False))
        self.assertEqual(self.problem("linux", StandardGpuResources=object, get_num_gpus=lambda: 0),
                         ("no visible GPU (CUDA_VISIBLE_DEVICES, or a node without GPUs)", False))

        def broken():
            raise RuntimeError("no driver")

        self.assertEqual(self.problem("linux", StandardGpuResources=object, get_num_gpus=broken),
                         ("no GPU count from FAISS (RuntimeError: no driver)", False))

    def test_refusal(self):
        with mock.patch.object(faiss_database, "_gpu_support_problem", return_value="no visible GPU"):
            with self.assertRaises(RefusalError) as caught:
                clustering._check_gpu_search()
        self.assertEqual(str(caught.exception),
                         f"GPU search (--use-gpu-index) needs FAISS's CUDA GPU support and a visible GPU, and this is "
                         f"no visible GPU: install faiss-gpu {faiss.__version__} (conda), or search on the CPU without "
                         f"--use-gpu-index.")
        with mock.patch.object(faiss_database, "_gpu_support_problem", return_value=None):
            clustering._check_gpu_search()

    @unittest.skipUnless(sys.platform == "darwin", "the macOS FAISS wheel")
    def test_this_mac(self):
        """The macOS faiss-cpu wheel has a StandardGpuResources (Metal) and counts a GPU, but no GPU support counts."""
        self.assertFalse(faiss_database._has_gpu_support())
        self.assertEqual(faiss_database._gpu_support_problem(),
                         "the macOS FAISS wheel, whose Metal backend FoldMatch does not use")


class MoveToGpuTest(_Fixtures, unittest.TestCase):
    """move_to_gpu(clustering=True): replicas on every GPU, float16 tables on ivf_pq only, the CPU index kept."""

    def test_ivf_pq_and_flat(self):
        for db_path, float16 in ((self.pq, True), (self.flat, False)):
            with self.subTest(db=Path(db_path).name):
                db = _load(db_path)
                cpu = db.index
                with FakeGpuLayer(n_gpus=3) as layer:
                    db.move_to_gpu(clustering=True)
                (gpu,) = layer.clones
                self.assertIs(gpu.cpu_index, cpu)
                self.assertEqual((gpu.options, gpu.ngpu), ({"shard": False, "useFloat16": float16}, 3))
                self.assertIs(db.index, gpu)
                self.assertIs(db.cpu_index, cpu)
                self.assertIs(db.host_index(), cpu)
                self.assertEqual((db.is_gpu_index, db.n_gpus), (True, 3))
                db.move_to_cpu()
                self.assertIs(db.index, cpu)
                self.assertEqual((db.cpu_index, db.is_gpu_index, db.n_gpus), (None, False, 0))

    def test_release(self):
        db = _load(self.pq)
        with FakeGpuLayer():
            db.move_to_gpu(clustering=True)
        db.release_index()
        self.assertEqual((db.index, db.cpu_index, db.is_gpu_index, db.n_gpus), (None, None, False, 0))
        self.assertEqual(len(db.chain_ids), 1000)

    def test_refusals(self):
        db = _load(self.pq)
        cpu = db.index
        with mock.patch.object(faiss_database, "_gpu_support_problem", return_value="no visible GPU"), \
                mock.patch.object(faiss, "index_cpu_to_all_gpus", create=True) as clone:
            with self.assertRaisesRegex(RefusalError, r"^GPU search \(--use-gpu-index\) needs FAISS's CUDA GPU "
                                                      r"support and a visible GPU, and this is no visible GPU: "):
                db.move_to_gpu(clustering=True)
        clone.assert_not_called()
        self.assertEqual((db.index, db.is_gpu_index, db.cpu_index), (cpu, False, None))
        hnsw = _load(self.hnsw)
        with FakeGpuLayer() as layer:
            with self.assertRaisesRegex(ValueError, r"^GPU search takes flat and ivf_pq databases; .* is hnsw "
                                                    r"\(built with --index-type hnsw\), which FAISS searches on the "
                                                    r"CPU only\.$"):
                hnsw.move_to_gpu(clustering=True)
        self.assertEqual((layer.clones, hnsw.is_gpu_index), ([], False))

    def test_the_default_mode_is_unchanged(self):
        """Without clustering: one GPU, default options, the CPU index dropped (the query commands' path)."""
        db = _load(self.flat)
        cloned = []

        class Resources:
            def setTempMemory(self, size):
                cloned.append(("temp memory", size))

        def to_gpu(resources, device, index):
            cloned.append(("clone", device, index))
            return "gpu index"

        with mock.patch.object(faiss_database, "_gpu_support_problem", return_value=None), \
                mock.patch.object(faiss, "StandardGpuResources", Resources, create=True), \
                mock.patch.object(faiss, "index_cpu_to_gpu", to_gpu, create=True):
            cpu = db.index
            db.move_to_gpu()
        self.assertEqual(cloned, [("temp memory", 1 << 30), ("clone", 0, cpu)])
        self.assertEqual((db.index, db.cpu_index, db.host_index(), db.n_gpus), ("gpu index", None, None, 0))

    def test_without_gpu_support_the_default_mode_stays_on_the_cpu(self):
        """load_database(use_gpu=True) without GPU support keeps the CPU index, as before; on macOS it raised an
        AttributeError in setTempMemory (F27)."""
        with mock.patch.object(faiss_database, "_gpu_support_problem", return_value="no visible GPU"):
            path = Path(self.flat)
            db = FaissEmbeddingDatabase(path.parent, path.name)
            db.load_database(use_gpu=True)
        self.assertFalse(db.is_gpu_index)
        self.assertEqual(len(db.search_batch(np.ones((1, 64), dtype=np.float32), 3)[0][0]), 3)

    @unittest.skipUnless(sys.platform == "darwin", "the macOS FAISS wheel")
    def test_without_gpu_support_on_this_mac(self):
        path = Path(self.pq)
        db = FaissEmbeddingDatabase(path.parent, path.name)
        db.load_database(use_gpu=True)
        self.assertFalse(db.is_gpu_index)

    def test_subset_reads_the_cpu_index(self):
        """The code copy of a database on the GPU is read from the CPU index it kept: the same subset as on the CPU."""
        for db_path in (self.pq, self.flat):
            with self.subTest(db=Path(db_path).name):
                name = Path(db_path).name
                db = _load(db_path)
                rows = np.arange(0, len(db.chain_ids), 3)
                on_cpu = _load(db_path).subset(rows, self.folder / f"{name}-subsets", "cpu")
                with FakeGpuLayer() as layer:
                    db.move_to_gpu(clustering=True)
                    on_gpu = db.subset(rows, self.folder / f"{name}-subsets", "gpu")
                self.assertEqual(layer.clones[0].searches, [])
                self.assertFalse(on_gpu.is_gpu_index)
                self.assertEqual(on_gpu.chain_ids, on_cpu.chain_ids)
                n = len(on_cpu.chain_ids)
                self.assertEqual(on_gpu.index.reconstruct_n(0, n).tobytes(), on_cpu.index.reconstruct_n(0, n).tobytes())
                self.assertEqual(on_gpu.index_config, on_cpu.index_config)
        dropped = _load(self.flat)
        dropped.is_gpu_index = True
        with self.assertRaisesRegex(ValueError, r"^subset reads the CPU index, which this move to the GPU dropped"):
            dropped.subset(np.arange(5), self.folder / "dropped", "db")


class ClustererGpuTest(_Fixtures, unittest.TestCase):
    """EmbeddingClusterer(use_gpu=True): nprobe set before the move; searches on the GPU index, stored vectors from the
    CPU index, so the fake GPU gives the CPU's results; greedy-leiden copies the representatives' database from the CPU
    index and moves it too; the refusals."""

    def test_nprobe_set_before_the_move(self):
        for nprobe, expected in ((3, 3), (None, 4), (40, 40)):
            with self.subTest(nprobe=nprobe):
                with FakeGpuLayer() as layer:
                    clusterer = EmbeddingClusterer(self.pq, use_gpu=True, nprobe=nprobe)
                self.assertEqual((layer.clones[0].nprobe, clusterer.nprobe), (expected, expected))
                self.assertEqual(faiss.extract_index_ivf(clusterer.db.cpu_index).nprobe, expected)

    def test_graph_and_greedy_as_on_the_cpu(self):
        for db_path in (self.flat, self.pq):
            with self.subTest(db=Path(db_path).name):
                cpu = EmbeddingClusterer(db_path)
                expected = _edges(cpu.build_similarity_edges(threshold=0.5, max_neighbors=10))
                expected_labels = cpu.cluster_greedy(threshold=0.9, max_neighbors=20).copy()
                with FakeGpuLayer() as layer:
                    gpu = EmbeddingClusterer(db_path, use_gpu=True)
                    found = _edges(gpu.build_similarity_edges(threshold=0.5, max_neighbors=10))
                    n_graph_searches = len(layer.clones[0].searches)
                    labels = gpu.cluster_greedy(threshold=0.9, max_neighbors=20)
                self.assertEqual(found[0].tobytes(), expected[0].tobytes())
                self.assertEqual(found[1].tobytes(), expected[1].tobytes())
                self.assertEqual(labels.tobytes(), expected_labels.tobytes())
                # Every search went to the GPU index: one call a block
                self.assertGreater(n_graph_searches, 0)
                self.assertGreater(len(layer.clones[0].searches), n_graph_searches)
                self.assertEqual({k for _, k in layer.clones[0].searches}, {10, 20})

    def test_greedy_leiden(self):
        """Both searches on the GPU: the parent's, then the representatives' database, copied from the parent's CPU
        index and then moved (a second clone, of the representatives' CPU index); the result is the CPU's; both
        indexes are released before Leiden; run.json records the GPUs, and a CPU rerun of the folder warns."""
        for db_path in (self.pq, self.flat):
            name = Path(db_path).name
            with self.subTest(db=name):
                options = dict(greedy_threshold=0.9, threshold=0.5, max_neighbors=10, resolution=1.0, seed=3)
                cpu = EmbeddingClusterer(db_path)
                expected = cpu.cluster_greedy_leiden(self.folder / f"{name}-cpu-work", **options).copy()
                work = self.folder / f"{name}-gpu-work"
                with FakeGpuLayer(n_gpus=2) as layer:
                    gpu = EmbeddingClusterer(db_path, use_gpu=True)
                    clusters = gpu.cluster_greedy_leiden(work, **options)
                self.assertEqual(clusters.tobytes(), expected.tobytes())
                self.assertEqual(gpu.chain_representatives.tobytes(), cpu.chain_representatives.tobytes())
                parent, reps = layer.clones
                self.assertEqual(reps.ntotal, gpu.n_representatives)
                self.assertLess(reps.ntotal, len(gpu.chain_ids))
                float16 = name == "pq"
                for clone in (parent, reps):
                    self.assertEqual((clone.options, clone.ngpu), ({"shard": False, "useFloat16": float16}, 2))
                    self.assertTrue(clone.searches)
                if float16:
                    self.assertEqual(reps.nprobe, parent.nprobe)
                self.assertEqual({k for _, k in parent.searches}, {min(1000, len(gpu.chain_ids))})
                self.assertEqual({k for _, k in reps.searches}, {10})
                self.assertEqual((gpu.db.index, gpu.db.cpu_index, gpu.db.is_gpu_index), (None, None, False))
                record = json.loads((work / "run.json").read_text())
                self.assertEqual(record["runs"][-1]["gpus"], 2)
                self.assertEqual({step["gpus"] for step in record["runs"][-1]["steps"].values()}, {2})
                with self.assertLogs(level=logging.WARNING) as logs:
                    again = EmbeddingClusterer(db_path).cluster_greedy_leiden(work, **options)
                self.assertEqual(again.tobytes(), expected.tobytes())
                self.assertIn("This run searches on the CPU; this work folder's earlier steps searched on the GPU: "
                              "the GPU's float16 lookup tables move ivf_pq scores slightly, so reused or resumed steps "
                              "may differ from a run on one of them.", "\n".join(logs.output))

    def test_k_limit(self):
        with FakeGpuLayer() as layer, mock.patch.object(faiss_database, "GPU_MAX_K", 5):
            clusterer = EmbeddingClusterer(self.pq, use_gpu=True)
            with self.assertRaises(clustering._GpuSearchRefused) as caught:
                clusterer.build_similarity_edges(threshold=0.5, max_neighbors=6)
            self.assertEqual((str(caught.exception), caught.exception.param),
                             ("--max-neighbors 6 is above 5, the most neighbours a FAISS GPU index returns: use at "
                              "most 5 with --use-gpu-index, or search on the CPU.", "--max-neighbors"))
            with self.assertRaisesRegex(clustering._GpuSearchRefused, r"^--max-neighbors 6 is above 5"):
                clusterer.cluster_greedy(threshold=0.9, max_neighbors=6, work_folder=self.folder / "k-greedy")
            for kwargs, flag in ((dict(greedy_max_neighbors=6, max_neighbors=5), "--greedy-max-neighbors"),
                                 (dict(greedy_max_neighbors=5, max_neighbors=6), "--max-neighbors")):
                with self.subTest(**kwargs):
                    with self.assertRaises(clustering._GpuSearchRefused) as caught:
                        clusterer.cluster_greedy_leiden(self.folder / "k-gl", **kwargs)
                    self.assertTrue(str(caught.exception).startswith(f"{flag} 6 is above 5"), caught.exception)
                    self.assertEqual(caught.exception.param, flag)
            self.assertEqual(layer.clones[0].searches, [])
            self.assertFalse((self.folder / "k-greedy").exists())
            self.assertFalse((self.folder / "k-gl").exists())
            clusterer.build_similarity_edges(threshold=0.5, max_neighbors=5)
            self.assertEqual({k for _, k in layer.clones[0].searches}, {5})
        # k is the clamped one: every chain of a database of no more chains than the limit
        with FakeGpuLayer(), mock.patch.object(faiss_database, "GPU_MAX_K", 400):
            EmbeddingClusterer(self.flat, use_gpu=True).build_similarity_edges(threshold=0.5, max_neighbors=10 ** 6)

    def test_nprobe_limit_and_hnsw(self):
        with FakeGpuLayer() as layer, mock.patch.object(faiss_database, "GPU_MAX_K", 5):
            for nprobe, message in ((6, "--nprobe 6 is above 5"), (None, None)):
                with self.subTest(nprobe=nprobe):
                    if message is None:
                        EmbeddingClusterer(self.pq, use_gpu=True, nprobe=nprobe)
                        continue
                    with self.assertRaises(clustering._GpuSearchRefused) as caught:
                        EmbeddingClusterer(self.pq, use_gpu=True, nprobe=nprobe)
                    self.assertEqual((str(caught.exception), caught.exception.param),
                                     ("--nprobe 6 is above 5, the most cells a FAISS GPU index probes: give --nprobe "
                                      "5 or less with --use-gpu-index, or search on the CPU.", "--nprobe"))
            self.assertEqual(len(layer.clones), 1)
        with FakeGpuLayer() as layer, mock.patch.object(faiss_database, "GPU_MAX_K", 3):
            with self.assertRaisesRegex(clustering._GpuSearchRefused,
                                        r"^--nprobe 4 \(the value stored at build\) is above 3, "):
                EmbeddingClusterer(self.pq, use_gpu=True)
            with self.assertRaises(clustering._GpuSearchRefused) as caught:
                EmbeddingClusterer(self.hnsw, use_gpu=True)
            self.assertEqual((str(caught.exception), caught.exception.param),
                             ("--use-gpu-index takes flat and ivf_pq databases; this one is hnsw, which FAISS searches "
                              "on the CPU only. Search it without --use-gpu-index.", "--use-gpu-index"))
            self.assertEqual(layer.clones, [])

    def test_no_gpu_support_refused_before_the_load(self):
        with mock.patch.object(faiss_database, "_gpu_support_problem", return_value="no visible GPU"), \
                mock.patch.object(FaissEmbeddingDatabase, "load_database",
                                  side_effect=AssertionError("the database was loaded")):
            with self.assertRaisesRegex(RefusalError, r"^GPU search \(--use-gpu-index\) needs FAISS's CUDA GPU "):
                EmbeddingClusterer(self.pq, use_gpu=True)


class CliGpuTest(_Fixtures, unittest.TestCase):
    """cluster and similarity-graph with --use-gpu-index: the fake GPU gives the CPU's files under every method; the
    limits are usage errors before the load, before the GPU support check; without GPU support exit status 1 before
    the load; the stored nprobe and hnsw after it."""

    def run_ok(self, *args):
        result = _invoke(*args)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")

    def test_files_as_on_the_cpu(self):
        cases = [("flat", ("cluster", "--threshold", "0.5", "--max-neighbors", "10", "--seed", "2")),
                 ("pq", ("cluster", "--threshold", "0.5", "--max-neighbors", "10", "--seed", "2", "--nprobe", "8")),
                 ("pq", ("cluster", "--method", "greedy", "--threshold", "0.9", "--max-neighbors", "50")),
                 ("flat", ("cluster", "--method", "greedy-leiden", "--threshold", "0.5", "--resolution", "1",
                           "--seed", "4")),
                 ("pq", ("cluster", "--method", "greedy-leiden", "--threshold", "0.5", "--resolution", "1",
                         "--seed", "4")),
                 ("pq", ("similarity-graph", "--threshold", "0.5", "--max-neighbors", "10"))]
        for i, (name, options) in enumerate(cases):
            with self.subTest(db=name, options=options):
                db_path = getattr(self, name)
                files = {}
                for device in ("cpu", "gpu"):
                    out = self.folder / f"cli{i}.{device}.out"
                    extra = ("--work-folder", self.folder / f"cli{i}.{device}.work") if "greedy-leiden" in options \
                        else ()
                    with FakeGpuLayer() as layer:
                        self.run_ok(*options, "--db-path", db_path, *extra, "--output-file", out,
                                    *(("--use-gpu-index",) if device == "gpu" else ()))
                    files[device] = out.read_bytes()
                    self.assertEqual(len(layer.clones), {"cpu": 0, "gpu": 2 if "greedy-leiden" in options else 1}[
                        device])
                    if device == "gpu" and "--nprobe" in options:
                        self.assertEqual(layer.clones[0].nprobe, 8)
                self.assertEqual(files["gpu"], files["cpu"])
                self.assertGreater(len(files["cpu"]), 100)

    def test_limits_before_the_load(self):
        cases = [(("cluster", "--max-neighbors", "2049"), "--max-neighbors",
                  "--max-neighbors 2049 is above 2048, the most neighbours a FAISS GPU index returns: use at most "
                  "2048 with --use-gpu-index, or search on the CPU."),
                 (("cluster", "--method", "greedy", "--max-neighbors", "5000"), "--max-neighbors",
                  "--max-neighbors 5000 is above 2048"),
                 (("cluster", "--method", "greedy-leiden", "--work-folder", self.folder / "lim",
                   "--greedy-max-neighbors", "2049"), "--greedy-max-neighbors",
                  "--greedy-max-neighbors 2049 is above 2048, the most neighbours a FAISS GPU index returns: use at "
                  "most 2048 with --use-gpu-index, or search on the CPU."),
                 (("cluster", "--nprobe", "2049"), "--nprobe",
                  "--nprobe 2049 is above 2048, the most cells a FAISS GPU index probes: give --nprobe 2048 or less "
                  "with --use-gpu-index, or search on the CPU."),
                 (("similarity-graph", "--max-neighbors", "2049"), "--max-neighbors",
                  "--max-neighbors 2049 is above 2048"),
                 (("similarity-graph", "--nprobe", "3000"), "--nprobe", "--nprobe 3000 is above 2048")]
        for args, flag, message in cases:
            with self.subTest(args=args):
                with mock.patch.object(clustering, "EmbeddingClusterer",
                                       side_effect=AssertionError("the database was loaded")), \
                        mock.patch.object(clustering, "_check_gpu_search",
                                          side_effect=AssertionError("GPU support checked")):
                    result = _invoke(*args, "--db-path", self.pq, "--use-gpu-index", "--output-file",
                                     self.folder / "limits.out")
                self.assertEqual(result.exit_code, 2, f"{result.output}\n{result.exception!r}")
                self.assertIn(f"Invalid value for {flag}: {message}", _flat(result.output))
        self.assertFalse((self.folder / "lim").exists())

    def test_without_gpu_support(self):
        """No GPU support (this machine's own state where it has none): exit status 1 before the load, at the limit
        2048 too, under every method and in similarity-graph."""
        problem = faiss_database._gpu_support_problem() or "no visible GPU"
        for args in (("cluster", "--max-neighbors", "2048", "--nprobe", "2048"), ("cluster", "--method", "greedy"),
                     ("cluster", "--method", "greedy-leiden", "--work-folder", self.folder / "nogpu"),
                     ("similarity-graph",)):
            with self.subTest(args=args):
                with mock.patch.object(faiss_database, "_gpu_support_problem", return_value=problem), \
                        mock.patch.object(clustering, "EmbeddingClusterer",
                                          side_effect=AssertionError("the database was loaded")):
                    result = _invoke(*args, "--db-path", self.pq, "--use-gpu-index", "--output-file",
                                     self.folder / "nogpu.out")
                self.assertEqual(result.exit_code, 1, f"{result.output}\n{result.exception!r}")
                self.assertIsInstance(result.exception, RefusalError)
                self.assertEqual(str(result.exception),
                                 f"GPU search (--use-gpu-index) needs FAISS's CUDA GPU support and a visible GPU, and "
                                 f"this is {problem}: install faiss-gpu {faiss.__version__} (conda), or search on the "
                                 f"CPU without --use-gpu-index.")
        self.assertFalse((self.folder / "nogpu").exists())
        self.assertFalse((self.folder / "nogpu.out").exists())

    @unittest.skipUnless(sys.platform == "darwin", "the macOS FAISS wheel")
    def test_this_mac(self):
        result = _invoke("cluster", "--db-path", self.flat, "--use-gpu-index", "--output-file", self.folder / "m.tsv")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("this is the macOS FAISS wheel, whose Metal backend FoldMatch does not use", str(result.exception))

    def test_after_the_load(self):
        with FakeGpuLayer() as layer, mock.patch.object(faiss_database, "GPU_MAX_K", 3):
            result = _invoke("cluster", "--db-path", self.pq, "--use-gpu-index", "--max-neighbors", "3",
                             "--output-file", self.folder / "after.tsv")
            self.assertEqual(result.exit_code, 2, result.output)
            self.assertIn("Invalid value for --nprobe: --nprobe 4 (the value stored at build) is above 3, the most "
                          "cells a FAISS GPU index probes: give --nprobe 3 or less with --use-gpu-index, or search on "
                          "the CPU.", _flat(result.output))
            for command in ("cluster", "similarity-graph"):
                result = _invoke(command, "--db-path", self.hnsw, "--use-gpu-index", "--max-neighbors", "3",
                                 "--output-file", self.folder / "after.out")
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("Invalid value for --use-gpu-index: --use-gpu-index takes flat and ivf_pq databases; "
                              "this one is hnsw, which FAISS searches on the CPU only. Search it without "
                              "--use-gpu-index.", _flat(result.output))
            self.assertEqual(layer.clones, [])
        self.assertFalse((self.folder / "after.tsv").exists())

    def test_help_states_the_limit(self):
        self.assertIn(f"--nprobe at most {faiss_database.GPU_MAX_K}.", search._CLUSTER_GPU_HELP)
        for command in ("cluster", "similarity-graph"):
            result = _invoke(command, "--help")
            self.assertEqual(result.exit_code, 0)
            self.assertIn("one replica of the index on each GPU", _flat(result.output))


if __name__ == "__main__":
    unittest.main()
