"""FaissEmbeddingDatabase search of an ivf_pq database: FAISS 1.14.2 to 1.15.1 lose coarse centroids in a batch; the
subset of a database copied from its stored codes or vectors (FaissEmbeddingDatabase.subset)."""
import hashlib
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Lightning first, as set_log_level: torch and FAISS each bring a libomp on macOS; scikit-learn sets KMP_DUPLICATE_LIB_OK.
import lightning  # noqa: F401
import faiss
import numpy as np

from foldmatch.search import faiss_database
from foldmatch.search.faiss_database import FaissEmbeddingDatabase
from foldmatch.types.api_types import IndexConfig, IndexType


def _write_one_vector_per_cell_database(folder: Path, name: str, nlist: int) -> np.ndarray:
    """An IVF database with one vector per cell, saved as FoldMatch saves an ivf_pq one, as a query's top hit is the id of
    its nearest cell only if the coarse search looks at that cell. Training a real OPQ + PQ on 39 * nlist vectors is not
    needed here, and the defect is in the coarse stage. Returns the vectors."""
    d = 8
    vectors = np.random.default_rng(0).standard_normal((nlist, d)).astype(np.float32)
    faiss.normalize_L2(vectors)
    quantizer = faiss.IndexFlatIP(d)
    quantizer.add(vectors)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.is_trained = True
    index.add(vectors)
    faiss.write_index(index, str(folder / f"{name}.index"))
    with open(folder / f"{name}.metadata", "wb") as f:
        pickle.dump({"embedding_ids": [f"chain{i}" for i in range(nlist)], "dimension": d,
                     "index_type": IndexType.ivf_pq, "index_config": IndexConfig(nlist=nlist, nprobe=4)}, f)
    return vectors


@unittest.skipIf(sys.platform == "darwin", "foldmatch pins OMP_NUM_THREADS=1 on macOS: FAISS deadlocks with more threads")
class IvfBatchSearchTest(unittest.TestCase):
    """Every query of a batch finds its own cell, whatever the batch size and the thread count."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.threads = faiss.omp_get_max_threads()
        self.addCleanup(faiss.omp_set_num_threads, self.threads)
        self.nlist = 16384
        self.vectors = _write_one_vector_per_cell_database(Path(self.tmp.name), "db", self.nlist)
        self.db = FaissEmbeddingDatabase(Path(self.tmp.name), "db")
        self.db.load_database()

    def test_batches_find_the_nearest_cell_at_every_thread_count(self):
        # The cells at the end of the table are the ones the defect never looks at.
        for threads in (1, 2, 3, 4, 8, 16, 17, 32):
            faiss.omp_set_num_threads(threads)
            for nq in (1, 2, 3, threads - 1, threads, threads + 1, threads * threads - 1, threads * threads):
                if nq < 1:
                    continue
                with self.subTest(threads=threads, nq=nq):
                    cells = [self.nlist - 1 - j for j in range(nq)]
                    results = self.db.search_batch(self.vectors[cells], top_k=1)
                    self.assertEqual([ids[0] for ids, _ in results], [f"chain{c}" for c in cells])

    def test_batch_equals_one_query_at_a_time(self):
        faiss.omp_set_num_threads(8)
        cells = list(range(self.nlist - 40, self.nlist))
        batch = self.db.search_batch(self.vectors[cells], top_k=4)
        single = [self.db.search_batch(self.vectors[c], top_k=4)[0] for c in cells]
        self.assertEqual([ids for ids, _ in batch], [ids for ids, _ in single])


class IvfParallelModeTest(unittest.TestCase):
    """parallel_mode is not stored in the index file: the database sets it wherever an ivf_pq index enters."""

    def test_load_and_build_set_parallel_mode_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            d = 16
            vectors = np.random.default_rng(1).standard_normal((1024, d)).astype(np.float32)
            built = FaissEmbeddingDatabase(folder, "built")
            built.create_database(iter([([f"c{i}" for i in range(len(vectors))], vectors)]), index_type=IndexType.ivf_pq,
                                  index_config=IndexConfig(nlist=8, m=4, nbits=4, nprobe=2, opq_train_size=1024))
            self.assertEqual(faiss.extract_index_ivf(built.index).parallel_mode, 3)
            loaded = FaissEmbeddingDatabase(folder, "built")
            loaded.load_database()
            self.assertEqual(faiss.extract_index_ivf(loaded.index).parallel_mode, 3)


def _build(folder: Path, name: str, vectors: np.ndarray, index_type: IndexType, config: IndexConfig = None):
    """A database of ``vectors``, chain ids c0, c1, ..., built and saved as the build commands do, then loaded."""
    built = FaissEmbeddingDatabase(folder, name)
    built.create_database(iter([([f"c{i}" for i in range(len(vectors))], vectors)]), index_type=index_type,
                          index_config=config)
    db = FaissEmbeddingDatabase(folder, name)
    db.load_database()
    return db


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lists(index):
    """Each IVF list of ``index`` as (ids, codes) arrays, copied."""
    ivf = faiss.extract_index_ivf(index)
    out = []
    for list_no in range(ivf.nlist):
        size = ivf.invlists.list_size(list_no)
        if size == 0:
            out.append((np.zeros(0, dtype=np.int64), np.zeros((0, ivf.code_size), dtype=np.uint8)))
            continue
        ids = faiss.rev_swig_ptr(ivf.invlists.get_ids(list_no), size).copy()
        codes = faiss.rev_swig_ptr(ivf.invlists.get_codes(list_no), size * ivf.code_size).copy()
        out.append((ids, codes.reshape(size, ivf.code_size)))
    return out


def _queries(db: FaissEmbeddingDatabase, rows) -> np.ndarray:
    """The stored (on ivf_pq, decoded) vectors of ``rows``, normalized, as the clustering searches query them."""
    queries = np.ascontiguousarray(db.index.reconstruct_n(0, len(db.chain_ids))[rows])
    faiss.normalize_L2(queries)
    return queries


class _SubsetChecks(unittest.TestCase):

    def assert_parent_top_k_restricted(self, parent: FaissEmbeddingDatabase, subset: FaissEmbeddingDatabase,
                                       rows: np.ndarray, queries: np.ndarray, k: int):
        """The subset's top k equals the parent's whole ranking restricted to ``rows``: per-rank scores equal by bits;
        the ids, as row ranks, equal as sets within each group of equal scores (a group cut by rank k: the subset's
        ids are in the parent's group)."""
        n = len(parent.chain_ids)
        rank = -np.ones(n, dtype=np.int64)
        rank[rows] = np.arange(len(rows))
        pd, pi = parent.index.search(queries, n)
        sd, si = subset.index.search(queries, k)
        for q in range(len(queries)):
            kept = (pi[q] >= 0) & (rank[np.maximum(pi[q], 0)] >= 0)
            self.assertEqual(int(kept.sum()), len(rows), "the parent's whole ranking holds every subset row")
            p_ids = rank[pi[q][kept]]
            p_bits = pd[q][kept].view(np.uint32)
            s_bits = sd[q].view(np.uint32)
            self.assertEqual(s_bits.tolist(), p_bits[:k].tolist(), f"query {q}: per-rank scores")
            self.assertTrue(bool(np.all(si[q] >= 0)))
            for bits in np.unique(s_bits):
                group = set(p_ids[p_bits == bits].tolist())
                got = set(si[q][s_bits == bits].tolist())
                if np.count_nonzero(p_bits[:k] == bits) == len(group):
                    self.assertEqual(got, group, f"query {q}: ids of score bits {bits}")
                else:
                    self.assertLessEqual(got, group, f"query {q}: ids of score bits {bits} (cut at rank {k})")


class IvfPqSubsetTest(_SubsetChecks):
    """The ivf_pq subset is a code copy: the parent's PQ codes, the parent's lists, the parent's search restricted."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.n, cls.d, cls.nlist, cls.stored_nprobe = 1024, 16, 8, 2
        vectors = np.random.default_rng(1).standard_normal((cls.n, cls.d)).astype(np.float32)
        cls.config = IndexConfig(nlist=cls.nlist, m=4, nbits=4, nprobe=cls.stored_nprobe, opq_train_size=1024)
        _build(cls.folder / "parent", "db", vectors, IndexType.ivf_pq, cls.config)
        cls.rows = np.sort(np.random.default_rng(2).choice(cls.n, 300, replace=False))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def parent(self, nprobe=None):
        db = FaissEmbeddingDatabase(self.folder / "parent", "db")
        db.load_database()
        if nprobe is not None:
            db.set_nprobe(nprobe)
        return db

    def out(self, name):
        folder = Path(tempfile.mkdtemp(dir=self.tmp.name))
        return folder, name

    def decoded(self, parent, rows):
        """The parent's decode of these rows in one batch of their size, as a subset of them decodes them.

        Not rows of the parent's ``reconstruct_n(0, n)``: the OPQ reverse transform's rounding can depend on the batch
        size (FAISS 1.15.1 on Linux), so the bytes are compared at the same size."""
        xt = np.ascontiguousarray(faiss.extract_index_ivf(parent.index).reconstruct_n(0, self.n)[np.asarray(rows)])
        x = np.empty_like(xt)
        parent.index.reverse_chain(len(xt), faiss.swig_ptr(xt), faiss.swig_ptr(x))
        return x

    def test_loads_as_a_database(self):
        parent = self.parent()
        folder, name = self.out("reps")
        subset = parent.subset(self.rows, folder, name, parent_record={"greedy_threshold": 0.9, "k": 1000})
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["reps.index", "reps.invlists", "reps.metadata"])
        for db in (subset, FaissEmbeddingDatabase(folder, name)):
            if db is not subset:
                db.load_database()
            with self.subTest(reloaded=db is not subset):
                self.assertEqual(db.index_type, IndexType.ivf_pq)
                self.assertEqual(db.dimension, self.d)
                self.assertEqual(db.chain_ids, [f"c{r}" for r in self.rows])
                self.assertEqual(db.index.ntotal, len(self.rows))
                ivf = faiss.extract_index_ivf(db.index)
                self.assertEqual(ivf.ntotal, len(self.rows))
                self.assertEqual(ivf.parallel_mode, 3)
                self.assertEqual(ivf.nprobe, self.stored_nprobe)
                self.assertFalse(db.is_gpu_index)
                self.assertEqual((db.index_config.nlist, db.index_config.m, db.index_config.nbits,
                                  db.index_config.nprobe), (self.nlist, 4, 4, self.stored_nprobe))
        with open(folder / "reps.metadata", "rb") as f:
            metadata = pickle.load(f)
        base = (self.folder / "parent").resolve() / "db"
        self.assertEqual(metadata["parent"], {
            "path": str(base), "index_name": "db", "n_chains": self.n,
            "index_sha256": _sha256(Path(f"{base}.index")), "metadata_sha256": _sha256(Path(f"{base}.metadata")),
            "invlists_bytes": Path(f"{base}.invlists").stat().st_size,
            "settings": {"greedy_threshold": 0.9, "k": 1000}})
        # The parent is unchanged: still loaded, its stored config untouched.
        self.assertEqual(parent.index.ntotal, self.n)
        self.assertEqual(parent.index_config.nprobe, self.stored_nprobe)

    def test_codes_are_the_parents_bytes(self):
        parent = self.parent()
        folder, name = self.out("reps")
        subset = parent.subset(self.rows, folder, name)
        rank = -np.ones(self.n, dtype=np.int64)
        rank[self.rows] = np.arange(len(self.rows))
        for list_no, ((p_ids, p_codes), (s_ids, s_codes)) in enumerate(zip(_lists(parent.index),
                                                                            _lists(subset.index))):
            kept = rank[p_ids] >= 0
            with self.subTest(list_no=list_no):
                self.assertEqual(s_ids.tolist(), rank[p_ids[kept]].tolist())
                self.assertEqual(s_codes.tobytes(), p_codes[kept].tobytes())
        # The IVF level (codes decoded, before the OPQ reverse transform) row by row; then the whole chain.
        p_ivf, s_ivf = faiss.extract_index_ivf(parent.index), faiss.extract_index_ivf(subset.index)
        self.assertEqual(s_ivf.reconstruct_n(0, len(self.rows)).tobytes(),
                         p_ivf.reconstruct_n(0, self.n)[self.rows].tobytes())
        self.assertEqual(self.decoded(parent, np.arange(self.n)).tobytes(),
                         parent.index.reconstruct_n(0, self.n).tobytes())
        self.assertEqual(subset.index.reconstruct_n(0, len(self.rows)).tobytes(),
                         self.decoded(parent, self.rows).tobytes())

    def test_the_effective_nprobe_is_stored(self):
        for nprobe in (None, 5, self.nlist + 3):
            with self.subTest(nprobe=nprobe):
                parent = self.parent(nprobe)
                folder, name = self.out("reps")
                parent.subset(self.rows, folder, name)
                expected = self.stored_nprobe if nprobe is None else nprobe
                with open(folder / "reps.metadata", "rb") as f:
                    self.assertEqual(pickle.load(f)["index_config"].nprobe, expected)
                reloaded = FaissEmbeddingDatabase(folder, name)
                reloaded.load_database()
                self.assertEqual(faiss.extract_index_ivf(reloaded.index).nprobe, expected)
                self.assertEqual(parent.index_config.nprobe, self.stored_nprobe)

    def test_a_clusterers_nprobe_carries_over(self):
        # The clusterer is the one source of nprobe (D6): a clusterer on the subset searches at the parent clusterer's.
        from foldmatch.search.clustering import EmbeddingClusterer
        parent = EmbeddingClusterer(db_path=str(self.folder / "parent" / "db"), nprobe=6)
        folder, name = self.out("reps")
        subset = EmbeddingClusterer(database=parent.db.subset(self.rows, folder, name))
        self.assertEqual((parent.nprobe, subset.nprobe), (6, 6))
        reloaded = EmbeddingClusterer(db_path=str(folder / name))
        self.assertEqual(reloaded.nprobe, 6)

    def test_search_is_the_parents_restricted_at_every_cell(self):
        parent = self.parent(self.nlist)
        folder, name = self.out("reps")
        subset = parent.subset(self.rows, folder, name)
        self.assertEqual(faiss.extract_index_ivf(subset.index).nprobe, self.nlist)
        # Queries from the subset and from outside it, as the clustering would make them from the parent.
        outside = np.setdiff1d(np.arange(self.n), self.rows)[:40]
        queries = _queries(parent, np.concatenate([self.rows[:60], outside]))
        for k in (1, 10, 50):
            with self.subTest(k=k):
                self.assert_parent_top_k_restricted(parent, subset, self.rows, queries, k)

    def test_every_row_searches_as_the_parent_by_bits(self):
        parent = self.parent()
        folder, name = self.out("all")
        subset = parent.subset(np.arange(self.n), folder, name)
        self.assertEqual(subset.chain_ids, parent.chain_ids)
        queries = _queries(parent, np.arange(0, self.n, 3))
        for k in (10, 100):
            with self.subTest(k=k):
                pd, pi = parent.index.search(queries, k)
                sd, si = subset.index.search(queries, k)
                self.assertEqual(sd.tobytes(), pd.tobytes())
                self.assertEqual(si.tolist(), pi.tolist())

    def test_a_subset_of_one_row(self):
        parent = self.parent()
        folder, name = self.out("one")
        subset = parent.subset([self.n - 1], folder, name)
        self.assertEqual(subset.chain_ids, [f"c{self.n - 1}"])
        self.assertEqual(subset.index.ntotal, 1)
        self.assertEqual(subset.index.reconstruct_n(0, 1).tobytes(),
                         parent.index.reconstruct_n(self.n - 1, 1).tobytes())

    def test_refusals(self):
        parent = self.parent()
        folder, name = self.out("reps")
        bad_rows = {
            "not ascending": ([5, 3], "strictly ascending"),
            "a repeat": ([3, 3, 7], "strictly ascending"),
            "below 0": ([-1, 3], r"\[0, 1024\)"),
            "at n": ([3, self.n], r"\[0, 1024\)"),
            "empty": ([], "at least one row"),
            "floats": (np.array([1.0, 2.0]), "integers"),
            "booleans": (np.ones(self.n, dtype=bool), "integers"),
            "two dimensions": (np.array([[1, 2]]), "one-dimensional"),
        }
        for label, (rows, message) in bad_rows.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, message):
                    parent.subset(rows, folder, name)
                self.assertFalse(folder.joinpath("reps.metadata").exists())
        before = {p.name: _sha256(p) for p in (self.folder / "parent").iterdir()}
        with self.assertRaisesRegex(ValueError, "its own parent's files"):
            parent.subset(self.rows, self.folder / "parent" / ".." / "parent", "db")
        self.assertEqual({p.name: _sha256(p) for p in (self.folder / "parent").iterdir()}, before)
        with self.assertRaisesRegex(ValueError, "Database not loaded"):
            FaissEmbeddingDatabase(self.folder / "parent", "db").subset(self.rows, folder, name)
        on_gpu = self.parent()
        on_gpu.is_gpu_index = True
        with self.assertRaisesRegex(ValueError, "reads the CPU index"):
            on_gpu.subset(self.rows, folder, name)

    def test_an_existing_database_is_refused_unless_overwrite(self):
        parent = self.parent()
        folder, name = self.out("reps")
        parent.subset(self.rows, folder, name)
        before = {p.name: _sha256(p) for p in folder.iterdir()}
        with self.assertRaisesRegex(FileExistsError, "Database already exists"):
            parent.subset(self.rows[:10], folder, name)
        self.assertEqual({p.name: _sha256(p) for p in folder.iterdir()}, before)
        # A lone file of an interrupted copy is refused too.
        lone_folder, _ = self.out("lone")
        (lone_folder / "reps.invlists").write_bytes(b"partial")
        with self.assertRaisesRegex(FileExistsError, "reps.invlists"):
            parent.subset(self.rows, lone_folder, name)
        for target in (folder, lone_folder):
            with self.subTest(target=target.name):
                subset = parent.subset(self.rows[:10], target, name, overwrite=True)
                self.assertEqual(subset.chain_ids, [f"c{r}" for r in self.rows[:10]])
                reloaded = FaissEmbeddingDatabase(target, name)
                reloaded.load_database()
                self.assertEqual(reloaded.index.ntotal, 10)
                self.assertEqual(reloaded.index.reconstruct_n(0, 10).tobytes(),
                                 self.decoded(parent, self.rows[:10]).tobytes())


class FlatSubsetTest(_SubsetChecks):
    """The flat subset is an exact IndexFlatIP of the parent's stored vectors; hnsw parents are refused."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.tmp.name)
        cls.n, cls.d = 500, 16
        cls.vectors = np.random.default_rng(3).standard_normal((cls.n, cls.d)).astype(np.float32)
        cls.rows = np.sort(np.random.default_rng(4).choice(cls.n, 120, replace=False))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_flat_subset(self, parent: FaissEmbeddingDatabase):
        folder = Path(tempfile.mkdtemp(dir=self.tmp.name))
        subset = parent.subset(self.rows, folder, "reps", parent_record={"k": 7})
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["reps.index", "reps.metadata"])
        self.assertEqual(subset.index_type, IndexType.flat)
        self.assertIs(type(faiss.downcast_index(subset.index)), faiss.IndexFlatIP)
        self.assertEqual(subset.chain_ids, [f"c{r}" for r in self.rows])
        self.assertEqual(subset.index.ntotal, len(self.rows))
        self.assertEqual(subset.index.reconstruct_n(0, len(self.rows)).tobytes(),
                         parent.index.reconstruct_n(0, self.n)[self.rows].tobytes())
        with open(folder / "reps.metadata", "rb") as f:
            metadata = pickle.load(f)
        self.assertEqual(metadata["parent"]["n_chains"], self.n)
        self.assertEqual(metadata["parent"]["settings"], {"k": 7})
        self.assertIsNone(metadata["parent"]["invlists_bytes"])
        # Exact search over the subset: fewer than 20 queries a call keep FAISS off its BLAS path, whose scores
        # depend on the call's rows.
        queries = _queries(parent, np.concatenate([self.rows[:8], np.setdiff1d(np.arange(self.n), self.rows)[:8]]))
        for k in (1, 10, len(self.rows)):
            with self.subTest(k=k):
                self.assert_parent_top_k_restricted(parent, subset, self.rows, queries, k)
        return subset

    def test_flat_parent(self):
        parent = _build(self.folder / "flat", "db", self.vectors, IndexType.flat)
        self.check_flat_subset(parent)

    def test_auto_parent_that_is_flat(self):
        parent = _build(self.folder / "auto_flat", "db", self.vectors, IndexType.auto)
        self.assertEqual(parent.index_type, IndexType.auto)
        self.check_flat_subset(parent)

    def test_blocks_of_the_copy(self):
        parent = _build(self.folder / "flat_blocks", "db", self.vectors, IndexType.flat)
        whole = self.check_flat_subset(parent)
        for block in (1, 7, self.n - 1, self.n):
            with self.subTest(block=block), mock.patch.object(faiss_database, "_SUBSET_FLAT_BLOCK", block):
                folder = Path(tempfile.mkdtemp(dir=self.tmp.name))
                subset = parent.subset(self.rows, folder, "reps")
                self.assertEqual(subset.index.reconstruct_n(0, len(self.rows)).tobytes(),
                                 whole.index.reconstruct_n(0, len(self.rows)).tobytes())

    def test_hnsw_parents_are_refused(self):
        hnsw = _build(self.folder / "hnsw", "db", self.vectors, IndexType.hnsw)
        with mock.patch.object(FaissEmbeddingDatabase, "N_EMBEDDINGS_DB_THR", 100):
            auto = _build(self.folder / "auto_hnsw", "db", self.vectors, IndexType.auto)
        self.assertIsInstance(faiss.downcast_index(auto.index), faiss.IndexHNSW)
        for label, parent in (("hnsw", hnsw), ("auto", auto)):
            with self.subTest(label):
                folder = Path(tempfile.mkdtemp(dir=self.tmp.name))
                with self.assertRaisesRegex(ValueError, f"hnsw database is not supported .*--index-type {label}\\)"):
                    parent.subset(self.rows, folder, "reps")
                self.assertEqual(list(folder.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
