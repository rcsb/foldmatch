import logging
import faiss
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from foldmatch.types.api_types import IndexConfig, IndexType

logger = logging.getLogger(__name__)

def _has_gpu_support():
    """Check if FAISS has GPU support available."""
    try:
        return hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
    except:
        return False


# FAISS's documented minimum: k-means training degenerates below this
# many points per centroid, producing an index whose search can segfault.
# See https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
_MIN_TRAINING_POINTS_PER_CELL = 39


def _normalize_L2_batched(embedding_array: np.ndarray, batch_size: int = 1000):
    """Normalize embeddings in-place by batches to avoid segfaults on very large arrays."""
    n = embedding_array.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        faiss.normalize_L2(embedding_array[start:end])


def _index_add_batched(index, embedding_array: np.ndarray, batch_size: int = 100):
    """Add embeddings to a FAISS index by batches to avoid segfaults on very large arrays."""
    n = embedding_array.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        index.add(np.ascontiguousarray(embedding_array[start:end]))


class FaissEmbeddingDatabase:
    """FAISS-based database for structure chain embeddings."""

    N_EMBEDDINGS_DB_THR = 10000

    def __init__(self, db_folder: Path, index_name: str = "structure_embeddings"):
        """
        Initialize FAISS database.

        Args:
            db_folder: Path to store the FAISS database
            index_name: Name of the index (used for file naming)
        """
        self.db_folder = db_folder
        self.index_name = index_name
        self.index = None
        self.chain_ids = None
        self.dimension = None
        self.gpu_resources = None
        self.is_gpu_index = False
        self.index_type = None
        self.index_config: Optional[IndexConfig] = None

    def create_database(
            self,
            embedding_batches: Iterable[Tuple[List[str], np.ndarray]],
            index_type: IndexType = IndexType.auto,
            index_config: Optional[IndexConfig] = None,
            use_gpu: bool = False,
    ):
        """
        Create a new FAISS database from a stream of embedding batches.

        Args:
            embedding_batches: Iterable yielding ``(ids, [B, D] float32 ndarray)`` tuples.
            index_type: FAISS index variant. See :class:`IndexType`.
            index_config: Tuning knobs (see :class:`IndexConfig`). Defaults if None.
            use_gpu: Move the in-memory index to GPU after build (ignored for ivf_pq).
        """
        config = index_config or IndexConfig()
        self.index_config = config
        self.index_type = index_type
        if index_type == IndexType.ivf_pq:
            if use_gpu:
                logger.warning("use_gpu is not supported for ivf_pq builds; running on CPU.")
            self._create_ivf_pq_ondisk(embedding_batches, config)
        else:
            self._create_in_memory(embedding_batches, index_type, config, use_gpu)
        self._save()

    def _create_in_memory(
            self,
            batches: Iterable[Tuple[List[str], np.ndarray]],
            index_type: IndexType,
            config: IndexConfig,
            use_gpu: bool,
    ):
        """Build an in-memory FlatIP/HNSW index by materializing all batches."""
        chain_ids: list[str] = []
        arrs: list[np.ndarray] = []
        for ids_batch, emb_batch in batches:
            chain_ids.extend(ids_batch)
            arrs.append(emb_batch)
        if not arrs:
            raise ValueError("No embeddings provided to create_database")
        embedding_array = np.ascontiguousarray(np.concatenate(arrs, axis=0), dtype=np.float32)
        del arrs

        self.dimension = embedding_array.shape[1]
        n_embeddings = embedding_array.shape[0]

        _normalize_L2_batched(embedding_array)

        if index_type == IndexType.auto:
            if n_embeddings < FaissEmbeddingDatabase.N_EMBEDDINGS_DB_THR:
                index_type = IndexType.flat
            else:
                index_type = IndexType.hnsw

        logger.info(f"Building {index_type.name} index with {n_embeddings} embeddings")
        if index_type == IndexType.flat:
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == IndexType.hnsw:
            self.index = faiss.IndexHNSWFlat(
                self.dimension, config.hnsw_m, faiss.METRIC_INNER_PRODUCT
            )

        if use_gpu:
            if _has_gpu_support():
                self.gpu_resources = faiss.StandardGpuResources()
                self.gpu_resources.setTempMemory(1024 * 1024 * 1024)
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                self.is_gpu_index = True

        _index_add_batched(self.index, embedding_array)
        self.chain_ids = chain_ids

    def _create_ivf_pq_ondisk(
            self,
            batches: Iterable[Tuple[List[str], np.ndarray]],
            config: IndexConfig,
    ):
        """Build OPQ + IVF-PQ with OnDiskInvertedLists. Bounded RAM ≈ training sample size."""

        logger.info(f"Building {IndexType.ivf_pq.name} index")
        self.db_folder.mkdir(parents=True, exist_ok=True)
        invlists_path = str(self.db_folder / f"{self.index_name}.invlists")

        # Phase A: accumulate the training sample. Keep raw (un-normalized) batches so
        # we can add them to the index after training without double-normalization.
        train_buffer: list[tuple[list[str], np.ndarray]] = []
        n_buffered = 0
        batches_iter = iter(batches)
        for ids_batch, emb_batch in batches_iter:
            train_buffer.append((ids_batch, emb_batch))
            n_buffered += emb_batch.shape[0]
            if n_buffered >= config.opq_train_size:
                break
        if not train_buffer:
            raise ValueError("No embeddings provided to create_database")

        self.dimension = train_buffer[0][1].shape[1]
        if self.dimension % config.m != 0:
            raise ValueError(
                f"PQ subquantizer count m={config.m} must divide embedding dim D={self.dimension}"
            )

        # Guard against degenerate IVF k-means: too few training points per
        # centroid produces an index whose search path can segfault inside
        # FAISS's C++. FAISS's documented minimum is 39 points per cell; below
        # this, k-means cannot converge to meaningful centroids.
        training_size = min(n_buffered, config.opq_train_size)
        points_per_cell = training_size / config.nlist
        if points_per_cell < _MIN_TRAINING_POINTS_PER_CELL:
            max_nlist = max(1, training_size // _MIN_TRAINING_POINTS_PER_CELL)
            # Round down to nearest power of two for FAISS-conventional nlist values.
            recommended_nlist = 1 << (max_nlist.bit_length() - 1)
            raise ValueError(
                f"IVF training would be degenerate: nlist={config.nlist} with "
                f"{training_size} training points yields {points_per_cell:.1f} points/cell "
                f"(FAISS minimum is {_MIN_TRAINING_POINTS_PER_CELL}; below this k-means "
                f"produces centroids that can cause segfaults during search). "
                f"Pass --ivf-nlist {recommended_nlist} (or smaller) for this corpus, "
                f"or use --index-type flat for small datasets."
            )

        # Phase B: build the empty index, train on a normalized copy of the sample.
        factory_string = f"OPQ{config.m},IVF{config.nlist},PQ{config.m}x{config.nbits}"
        logger.info(
            f"Training {factory_string} on {n_buffered} vectors (cap {config.opq_train_size})"
        )
        self.index = faiss.index_factory(
            self.dimension, factory_string, faiss.METRIC_INNER_PRODUCT
        )

        train_arr = np.ascontiguousarray(
            np.concatenate([a for _, a in train_buffer], axis=0)[:config.opq_train_size],
            dtype=np.float32,
        )
        faiss.normalize_L2(train_arr)
        self.index.train(train_arr)
        del train_arr
        logger.info("Training complete")

        # Phase C: swap inverted lists onto disk so subsequent add() writes go to mmap.
        ivf_index = faiss.extract_index_ivf(self.index)
        invlists = faiss.OnDiskInvertedLists(
            ivf_index.nlist, ivf_index.code_size, invlists_path
        )
        ivf_index.replace_invlists(invlists, True)
        # SWIG ownership transfer: the IVF index now owns the C++ invlists
        # object and will free it on destruction. Without disown(), Python's
        # wrapper would *also* try to free it at interpreter shutdown — a
        # double-free that manifests as a segfault during process teardown.
        invlists.this.disown()

        # Phase D: add the training-sample batches, then the rest of the stream.
        self.chain_ids = []
        total_added = 0
        for ids_batch, emb_batch in train_buffer:
            total_added += self._normalize_and_add(ids_batch, emb_batch)
        del train_buffer
        for ids_batch, emb_batch in batches_iter:
            total_added += self._normalize_and_add(ids_batch, emb_batch)

        if config.nprobe is None:
            config.nprobe = max(1, config.nlist // 64)
        ivf_index.nprobe = config.nprobe

        # Phase E: persist the populated index. This is the canonical FAISS
        # save order — write_index serializes both the trained header AND the
        # OnDisk invlists metadata (per-cell offsets + sizes). Without this,
        # a subsequent load would create a fresh empty OnDiskInvertedLists
        # (constructor doesn't scan the file for existing metadata), and
        # every query would find an empty cell. The earlier disown() makes
        # this safe — without it, write_index can segfault from a SWIG
        # double-free of the invlists object.
        header_path = self.db_folder / f"{self.index_name}.index"
        faiss.write_index(self.index, str(header_path))

        logger.info(f"IVF-PQ build complete: {total_added} vectors, nprobe={config.nprobe}")

    def _normalize_and_add(self, ids_batch: list[str], emb_batch: np.ndarray) -> int:
        """Normalize a copy of the batch and add to the index. Returns the count added."""
        arr = np.ascontiguousarray(emb_batch, dtype=np.float32).copy()
        faiss.normalize_L2(arr)
        self.index.add(arr)
        self.chain_ids.extend(ids_batch)
        return arr.shape[0]

    def load_database(self, use_gpu: bool = False):
        """
        Load an existing FAISS database.

        Args:
            use_gpu: Whether to load the index to GPU (if available)
        """
        index_file = self.db_folder / f"{self.index_name}.index"
        metadata_file = self.db_folder / f"{self.index_name}.metadata"

        if not index_file.exists() or not metadata_file.exists():
            raise ValueError(f"Database files not found in: {self.db_folder}")

        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        self.chain_ids = metadata['embedding_ids']
        self.dimension = metadata['dimension']
        self.index_type = metadata.get('index_type', IndexType.auto)
        self.index_config = metadata.get('index_config')

        if self.index_type == IndexType.ivf_pq:
            # The populated index was saved by _create_ivf_pq_ondisk after the
            # OnDisk swap, so the header includes both the trained state AND the
            # per-cell invlists metadata (offsets + sizes). IO_FLAG_ONDISK_SAME_DIR
            # tells FAISS to look for the .invlists file in the same directory as
            # the .index file — gives portability if the DB folder is moved.
            self.index = faiss.read_index(
                str(index_file), faiss.IO_FLAG_ONDISK_SAME_DIR
            )
            if self.index_config is not None and self.index_config.nprobe is not None:
                ivf_index = faiss.extract_index_ivf(self.index)
                ivf_index.nprobe = self.index_config.nprobe
        else:
            self.index = faiss.read_index(str(index_file))

        logger.info(
            f"Loaded database with {len(self.chain_ids)} embeddings (index_type={self.index_type.name})"
        )

        if use_gpu:
            self.move_to_gpu()

    def saved_entry_count(self) -> int:
        """Number of embeddings in the database on disk, from its metadata file; the index itself is not read.

        Raises ``ValueError`` if the database files do not exist, as :meth:`load_database` does, or if the
        metadata file is not readable FAISS metadata (truncated, or another file), naming it.
        """
        index_file = self.db_folder / f"{self.index_name}.index"
        metadata_file = self.db_folder / f"{self.index_name}.metadata"
        if not index_file.exists() or not metadata_file.exists():
            raise ValueError(f"Database files not found in: {self.db_folder}")
        try:
            with open(metadata_file, 'rb') as f:
                return len(pickle.load(f)['embedding_ids'])
        except Exception as e:  # unpickling a damaged file can raise almost anything
            raise ValueError(
                f"{metadata_file} is not readable FAISS database metadata ({type(e).__name__}: {e})"
            ) from e

    def set_nprobe(self, nprobe: int) -> None:
        """Override the search-time ``nprobe`` for IVF-family indexes.

        ``nprobe`` controls how many IVF cells are scanned per query. Higher
        values improve recall at the cost of search latency. Calling this on a
        non-IVF index (``flat`` / ``hnsw``) is a silent no-op so callers can be
        index-type-agnostic.
        """
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")
        if self.index_type != IndexType.ivf_pq:
            return
        if nprobe < 1:
            raise ValueError(f"nprobe must be >= 1, got {nprobe}")
        ivf_index = faiss.extract_index_ivf(self.index)
        ivf_index.nprobe = nprobe
        logger.info(f"Set search-time nprobe = {nprobe}")

    def move_to_gpu(self, gpu_id: int = 0):
        """
        Move the index to GPU.

        Args:
            gpu_id: GPU device ID to use (default: 0)
        """
        if self.is_gpu_index:
            logger.info("Index is already on GPU")
            return

        if not _has_gpu_support():
            logger.info("WARNING: GPU not available. Keeping index on CPU.")
            logger.info("To enable GPU support, install: pip install faiss-gpu")
            return

        if self.index is None:
            raise ValueError("No index loaded. Call load_database() first.")

        logger.info(f"Moving index to GPU {gpu_id}...")
        self.gpu_resources = faiss.StandardGpuResources()
        self.gpu_resources.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory
        self.index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, self.index)
        self.is_gpu_index = True
        logger.info("Index successfully moved to GPU")

    def move_to_cpu(self):
        """Move the index from GPU to CPU."""
        if not self.is_gpu_index:
            logger.info("Index is already on CPU")
            return

        logger.info("Moving index to CPU...")
        self.index = faiss.index_gpu_to_cpu(self.index)
        self.is_gpu_index = False
        self.gpu_resources = None
        logger.info("Index successfully moved to CPU")

    def search_batch(
            self,
            query_embeddings,
            top_k: int = 10,
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Batched cosine-similarity search for one or many query vectors.

        Args:
            query_embeddings: torch.Tensor or np.ndarray. 1-D is treated as a single
                query; 2-D as a ``[B, D]`` batch.
            top_k: Number of top results per query.

        Returns:
            One ``(chain_ids, scores)`` tuple per query, in input order. Scores are
            cosine similarities in ``[-1, 1]`` (1.0 means identical, like TM-score).
        """
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()

        arr = np.ascontiguousarray(query_embeddings, dtype=np.float32).copy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"query_embeddings must be 1-D or 2-D, got {arr.ndim}-D"
            )

        faiss.normalize_L2(arr)
        scores, indices = self.index.search(arr, top_k)

        results: List[Tuple[List[str], List[float]]] = []
        for i in range(arr.shape[0]):
            ids = [self.chain_ids[idx] for idx in indices[i] if idx >= 0]
            sc = [scores[i][j] for j, idx in enumerate(indices[i]) if idx >= 0]
            results.append((ids, sc))
        return results

    def search(
            self,
            query_embedding,
            top_k: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """
        Cosine-similarity search for a single query vector.

        For batched queries use :meth:`search_batch`.
        """
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()
        arr = np.asarray(query_embedding)
        if arr.ndim != 1:
            raise ValueError(
                f"search() takes a single 1-D query (got {arr.ndim}-D). "
                f"Use search_batch for [B, D] inputs."
            )
        return self.search_batch(arr, top_k=top_k)[0]

    def search_by_chain_ids(
            self,
            query_chain_ids: List[str],
            top_k: int = 10
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Search database using existing chain IDs as queries.

        Missing IDs are logged and skipped. Returns one entry per resolved ID.
        """
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        id_to_idx = {cid: i for i, cid in enumerate(self.chain_ids)}
        resolved_ids: List[str] = []
        resolved_idxs: List[int] = []
        for cid in query_chain_ids:
            idx = id_to_idx.get(cid)
            if idx is None:
                logger.info(f"Chain {cid} not found in database")
                continue
            resolved_ids.append(cid)
            resolved_idxs.append(idx)

        if not resolved_ids:
            return {}

        embeddings = np.stack([self.index.reconstruct(i) for i in resolved_idxs])
        batch_results = self.search_batch(embeddings, top_k=top_k)
        return dict(zip(resolved_ids, batch_results))

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        return {
            "total_embeddings": len(self.chain_ids),
            "dimension": self.dimension,
            "index_name": self.index_name,
            "db_path": str(self.db_folder),
            "index_type": type(self.index).__name__,
            "on_gpu": self.is_gpu_index,
            "gpu_available": _has_gpu_support()
        }

    def _save(self):
        """Save FAISS index and metadata to disk."""
        self.db_folder.mkdir(parents=True, exist_ok=True)

        index_file = self.db_folder / f"{self.index_name}.index"
        metadata_file = self.db_folder / f"{self.index_name}.metadata"

        if self.index_type == IndexType.ivf_pq:
            # Header was already persisted by _create_ivf_pq_ondisk before the
            # OnDiskInvertedLists swap; calling faiss.write_index on a swapped
            # index segfaults at scale. Only the metadata pickle remains here.
            pass
        elif self.is_gpu_index:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_file))
        else:
            faiss.write_index(self.index, str(index_file))

        metadata = {
            'embedding_ids': self.chain_ids,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'index_config': self.index_config,
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

