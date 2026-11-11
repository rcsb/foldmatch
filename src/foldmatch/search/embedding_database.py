import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from tqdm import tqdm
import numpy as np
import time

from foldmatch.search.db_path import parse_db_path as _parse_db_path
from foldmatch.search.faiss_database import FaissEmbeddingDatabase

logger = logging.getLogger(__name__)

class EmbeddingDatabase:
    """Search for similar protein structures using embeddings."""

    def __init__(
            self,
            db_path: str,
            use_gpu_for_search: bool = False
    ):
        """
        Initialize the embedding database wrapper.

        This does NOT load any existing files on disk — call :meth:`load`
        explicitly when you need to read. Build paths (``create_db``) never
        load, so they can construct freely against a path that may or may not
        already hold a database.

        Args:
            db_path: Path to FAISS database (will be parsed into folder + name).
            use_gpu_for_search: Default GPU preference used when ``load`` is
                invoked without an explicit ``use_gpu`` argument.
        """
        self.db_folder, self.index_name, self.db_path = _parse_db_path(db_path)
        self.db = FaissEmbeddingDatabase(self.db_folder, self.index_name)
        self._use_gpu_for_search = use_gpu_for_search

    def load(self, use_gpu: Optional[bool] = None, nprobe: Optional[int] = None):
        """Load the on-disk FAISS index and metadata into memory.

        Read-side callers (search, stats, export) must call this before using
        any read method. Raises if the database files don't exist.

        Args:
            use_gpu: Override the constructor-time GPU preference for this load.
            nprobe: Optional override of the search-time IVF nprobe. When set,
                supersedes the value baked into the index at build time. No-op
                for non-IVF indexes (flat / hnsw).
        """
        if use_gpu is None:
            use_gpu = self._use_gpu_for_search
        self.db.load_database(use_gpu=use_gpu)
        if nprobe is not None:
            self.db.set_nprobe(nprobe)

    def set_nprobe(self, nprobe: int) -> None:
        """Override the search-time ``nprobe`` for IVF-family indexes.

        No-op for non-IVF indexes. Must be called after :meth:`load`. Most
        callers can pass ``nprobe`` directly to :meth:`load` instead.
        """
        self.db.set_nprobe(nprobe)

    def create_db(
            self,
            embedding_batches,
            use_gpu_index,
            index_type,
            index_config,
            force: bool = False,
    ):
        """Build a fresh FAISS database from the given batches.

        Raises ``FileExistsError`` if any database file already exists at the
        target path, unless ``force=True`` is given — in which case the stale
        files are deleted first.
        """
        # Safety guard against accidental overwrites (and against the mmap /
        # write conflict that occurs when a previous index is still on disk).
        candidates = [
            Path(f"{self.db_path}.index"),
            Path(f"{self.db_path}.metadata"),
            Path(f"{self.db_path}.invlists"),
        ]
        existing = [p for p in candidates if p.exists()]
        if existing:
            if not force:
                paths = ", ".join(str(p) for p in existing)
                raise FileExistsError(
                    f"Database already exists at {self.db_path} "
                    f"(found: {paths}). "
                    f"Choose a different --output-db, or delete the existing files first."
                )
            for p in existing:
                p.unlink()

        start_time = time.time()
        self.db.create_database(
            embedding_batches=embedding_batches,
            index_type=index_type,
            index_config=index_config,
            use_gpu=use_gpu_index,
        )
        database_time = time.time() - start_time
        logging.info(f"Creating database completed in {database_time:.2f} seconds")

        logging.info("Database build complete!")
        logging.info(f"Database location: {self.db_path}")
        logging.info(f"Total embeddings: {len(self.db.chain_ids)}")
        logging.info(f"You can now search this database using:")
        logging.info(f"   fm-search query structure --db-path {self.db_path} --query-structure <path_to_structure>")

    def search_by_embeddings(
            self,
            embedding_batches: Iterable[Tuple[List[str], np.ndarray]],
            top_k: int = 10,
    ):
        results: Dict[str, Tuple[List[str], List[float]]] = {}
        for ids_batch, emb_batch in embedding_batches:
            batch_results = self.db.search_batch(emb_batch, top_k=top_k)
            for qid, res in zip(ids_batch, batch_results):
                results[qid] = res
        return results

    def search_by_database(
            self,
            query_db_path: str,
            top_k: int = 10,
            batch_size: int = 4096,
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Search the subject database using every chain embedding from another database.

        Args:
            query_db_path: Path to the query FAISS database directory
            query_index_name: Name of the query FAISS index
            top_k: Number of top results to return per query chain
            batch_size: Number of query vectors processed per FAISS call.

        Returns:
            Dictionary mapping query chain ID to (matching_chain_ids, similarity_scores)
        """
        logging.info("Loading query database...")
        query_db_folder, query_index_name, _ = _parse_db_path(query_db_path)
        query_db = FaissEmbeddingDatabase(query_db_folder, query_index_name)
        query_db.load_database()

        n = len(query_db.chain_ids)
        logging.info(f"Query database contains {n} embeddings")

        results = {}
        with tqdm(total=n, desc="Querying database") as pbar:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_ids = query_db.chain_ids[start:end]
                batch_vecs = query_db.index.reconstruct_n(start, end - start)
                batch_results = self.db.search_batch(batch_vecs, top_k=top_k)
                for qid, res in zip(batch_ids, batch_results):
                    results[qid] = res
                pbar.update(end - start)

        logging.info(f"Completed {len(results)} queries")
        return results

    def get_db_statistics(self) -> Dict:
        """Get database statistics."""
        return self.db.get_statistics()
