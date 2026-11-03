"""FaissEmbeddingDatabase search of an ivf_pq database: FAISS 1.14.2 to 1.15.1 lose coarse centroids in a batch."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

# Lightning first, as set_log_level: torch and FAISS each bring a libomp on macOS; scikit-learn sets KMP_DUPLICATE_LIB_OK.
import lightning  # noqa: F401
import faiss
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
