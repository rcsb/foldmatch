import dataclasses
import hashlib
import logging
import faiss
import torch
import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from foldmatch.errors import RefusalError
from foldmatch.types.api_types import IndexConfig, IndexType

logger = logging.getLogger(__name__)

# The most neighbours (k) a FAISS GPU index returns and the most cells (nprobe) it probes: GPU_MAX_SELECTION_K of
# FAISS's CUDA build, as read from FAISS, not yet run against it here (the GPU benchmark probed up to 2048 cells only,
# work/p9ivf/bench/IVF_BENCH.md). The clustering GPU path refuses more.
GPU_MAX_K = 2048


def _gpu_support_problem() -> Optional[str]:
    """Why FAISS cannot search on a GPU here, or None when it can: FAISS's CUDA GPU layer and at least one visible GPU.
    The macOS faiss-cpu wheel's Metal backend (a ``StandardGpuResources`` without ``setTempMemory``, so FoldMatch's
    GPU paths fail in it) is not counted."""
    if sys.platform == "darwin":
        return "the macOS FAISS wheel, whose Metal backend FoldMatch does not use"
    if not hasattr(faiss, 'StandardGpuResources'):
        return f"a FAISS {faiss.__version__} without GPU support (faiss-cpu)"
    try:
        n_gpus = faiss.get_num_gpus()
    except Exception as exc:
        return f"no GPU count from FAISS ({type(exc).__name__}: {exc})"
    if n_gpus < 1:
        return "no visible GPU (CUDA_VISIBLE_DEVICES, or a node without GPUs)"
    return None


def _require_gpu_support():
    """Refuse (:class:`~foldmatch.errors.RefusalError`) unless FAISS can search on a GPU here
    (:func:`_gpu_support_problem`)."""
    problem = _gpu_support_problem()
    if problem is not None:
        raise RefusalError(f"GPU search (--use-gpu-index) needs FAISS's CUDA GPU support and a visible GPU, and this "
                           f"is {problem}: install faiss-gpu {faiss.__version__} (conda), or search on the CPU without "
                           f"--use-gpu-index.")


def _has_gpu_support():
    """Check if FAISS has GPU support available (FAISS's CUDA GPU layer and a visible GPU; not macOS's Metal
    backend)."""
    return _gpu_support_problem() is None


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


def _quantize_outside_parallel_regions(index) -> None:
    """Make ``index.search`` of an IVF index run its coarse quantizer outside OpenMP parallel regions (FAISS parallel_mode 3).

    FAISS 1.14.2 to 1.15.1 lose half or more of the coarse centroids otherwise, with no error. In parallel_mode 0 (the
    default) ``IndexIVF::search`` calls the quantizer inside a parallel loop over the queries. With T threads, nlist >=
    max(10000, 1024 * T), and fewer than T queries in a thread's slice, the flat quantizer splits its centroids into T
    segments, one per thread, but the nested region has only one thread: only the first nlist // T centroids are
    searched (facebookresearch/faiss#5824). parallel_mode 3 calls the quantizer once for the whole batch, outside the
    region, and then splits the same per-query scan over the threads. It is not stored in the index file.

    The defect is likely solved in the next FAISS release: the upstream fix (facebookresearch/faiss#5824) was merged on
    2026-10-31, but it is not in 1.15.1, the latest release and FoldMatch's pin. Re-examine this workaround when the
    faiss-cpu pin moves to a release that has it.
    """
    faiss.extract_index_ivf(index).parallel_mode = 3


# The suffixes of a database's files (the .invlists file is the on-disk lists of an ivf_pq database).
_DATABASE_SUFFIXES = (".index", ".invlists", ".metadata")
# Stored vectors read with one reconstruct_n call when a flat database is subset (16,384 x d x 4 bytes, 96 MiB at
# d = 1536).
_SUBSET_FLAT_BLOCK = 16_384


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subset_rows(rows, n: int) -> np.ndarray:
    """The rows of :meth:`FaissEmbeddingDatabase.subset` as an int64 array, checked: one dimension, integers, at least
    one, strictly ascending (so no repeats), every row in [0, n)."""
    arr = np.asarray(rows)
    if arr.ndim != 1:
        raise ValueError(f"rows must be one-dimensional, got {arr.ndim} dimensions.")
    if arr.size == 0:
        raise ValueError("rows must name at least one row.")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"rows must be integers, got {arr.dtype}.")
    arr = arr.astype(np.int64, copy=False)
    if arr.size > 1 and not bool(np.all(arr[1:] > arr[:-1])):
        raise ValueError("rows must be strictly ascending (database order, no repeats).")
    if arr[0] < 0 or arr[-1] >= n:
        raise ValueError(f"rows must lie in [0, {n}) for a database of {n} chains; got {arr[0]} to {arr[-1]}.")
    return arr


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
        # The CPU index kept by a move to the GPU in the clustering mode (move_to_gpu(clustering=True)), for
        # reconstruct_n and subset; None otherwise. And the number of GPUs the index is on (0 on the CPU).
        self.cpu_index = None
        self.n_gpus = 0
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
        _quantize_outside_parallel_regions(self.index)

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

    def subset(
            self,
            rows,
            db_folder: Path,
            index_name: str,
            parent_record: Optional[Dict] = None,
            overwrite: bool = False,
    ) -> "FaissEmbeddingDatabase":
        """Write a database of some of this database's chains, from its stored data, and return it loaded on the CPU.

        Nothing is trained or re-encoded. ``ivf_pq``: the trained OPQ, coarse quantizer and PQ are read from this
        database's ``.index`` file without its lists, and each kept chain's PQ code is copied, list by list, into new
        on-disk lists, in this database's list order; so a search of the subset is a search of this database restricted
        to the subset's chains. The subset searches at this database's current nprobe (as set by :meth:`set_nprobe`,
        else the stored one), which its metadata stores, so a load of the subset keeps it. ``flat``: an exact
        ``IndexFlatIP`` of the kept chains' stored vectors, held in memory (rows x dimension x 4 bytes), so for small
        sets. ``hnsw`` (an ``auto`` database that is hnsw included) is refused: an approximate graph is not subset. On
        the GPU, the CPU index kept by ``move_to_gpu(clustering=True)`` is read.

        Args:
            rows: Row numbers of the chains to keep, integers, strictly ascending (database order) in [0, n). Row r
                of this database becomes row ``rank(r)`` of the subset.
            db_folder: Folder of the subset's files (created if needed).
            index_name: Base name of the subset's ``.index``, ``.invlists`` (ivf_pq) and ``.metadata`` files.
            parent_record: The caller's settings (for example, how the rows were chosen), stored in the subset's
                metadata as ``metadata['parent']['settings']`` next to this database's path, chain count and the
                sha256 of its ``.index`` and ``.metadata`` files.
            overwrite: Replace a database already at ``db_folder``/``index_name``; otherwise it is refused.

        Raises:
            ValueError: the database is not loaded, or is on the GPU without its CPU index (:meth:`move_to_gpu`
                without ``clustering``); ``rows`` are not as above; the database is hnsw,
                or a flat index other than ``IndexFlatIP``; the subset would be this database's own files.
            FileExistsError: a file of the subset exists and ``overwrite`` is False.
        """
        if self.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")
        cpu_index = self.host_index()
        if cpu_index is None:
            raise ValueError("subset reads the CPU index, which this move to the GPU dropped: load the database "
                             "without use_gpu, or move it with move_to_gpu(clustering=True), which keeps it.")
        n = len(self.chain_ids)
        rows = _subset_rows(rows, n)
        index = faiss.downcast_index(cpu_index)
        if self.index_type != IndexType.ivf_pq:
            if isinstance(index, faiss.IndexHNSW):
                raise ValueError(
                    f"A subset of an hnsw database is not supported ({self.db_folder / self.index_name}, built with "
                    f"--index-type {self.index_type.name}): copy from an ivf_pq or flat database.")
            if type(index) is not faiss.IndexFlatIP:
                raise ValueError(f"A subset of a {type(index).__name__} index is not supported: copy from an ivf_pq "
                                 "or flat database.")
        db_folder = Path(db_folder)
        if (db_folder.resolve(), index_name) == (Path(self.db_folder).resolve(), self.index_name):
            raise ValueError(f"The subset would replace its own parent's files ({db_folder / index_name}): choose "
                             "another folder or index name.")
        paths = [db_folder / f"{index_name}{suffix}" for suffix in _DATABASE_SUFFIXES]
        existing = [p for p in paths if p.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"Database already exists at {db_folder / index_name} (found: "
                                  f"{', '.join(str(p) for p in existing)}). Pass overwrite=True to replace it.")
        db_folder.mkdir(parents=True, exist_ok=True)
        for path in existing:
            path.unlink()

        parent = self._fingerprint()
        parent["settings"] = dict(parent_record or {})
        config = self.index_config
        if self.index_type == IndexType.ivf_pq:
            nprobe = int(faiss.extract_index_ivf(cpu_index).nprobe)
            self._subset_ivf_pq(rows, db_folder, index_name, nprobe)
            if config is None:  # a database whose metadata has no config: its layout, read from the index
                ivf = faiss.downcast_index(faiss.extract_index_ivf(cpu_index))
                pq = getattr(ivf, 'pq', None)
                config = (IndexConfig(nlist=ivf.nlist) if pq is None
                          else IndexConfig(nlist=ivf.nlist, m=pq.M, nbits=pq.nbits))
            # load_database sets nprobe from the metadata, so the subset's config carries the value it searched at.
            config = dataclasses.replace(config, nprobe=nprobe)
            index_type = IndexType.ivf_pq
        else:
            self._subset_flat(rows, db_folder, index_name)
            config = dataclasses.replace(config) if config is not None else None
            index_type = IndexType.flat
        metadata = {
            'embedding_ids': [self.chain_ids[r] for r in rows.tolist()],
            'dimension': self.dimension,
            'index_type': index_type,
            'index_config': config,
            'parent': parent,
        }
        # The metadata file is written last: load_database refuses a folder without it.
        with open(db_folder / f"{index_name}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
        subset = FaissEmbeddingDatabase(db_folder, index_name)
        subset.load_database()
        return subset

    def _fingerprint(self) -> Dict:
        """This database's identity: its path, index name, chain count, the sha256 of its ``.index`` and ``.metadata``
        files and the size of its ``.invlists`` file (None if it has none)."""
        base = Path(self.db_folder).resolve() / self.index_name
        invlists = Path(f"{base}.invlists")
        return {
            'path': str(base),
            'index_name': self.index_name,
            'n_chains': len(self.chain_ids),
            'index_sha256': _file_sha256(Path(f"{base}.index")),
            'metadata_sha256': _file_sha256(Path(f"{base}.metadata")),
            'invlists_bytes': invlists.stat().st_size if invlists.exists() else None,
        }

    def _subset_ivf_pq(self, rows: np.ndarray, db_folder: Path, index_name: str, nprobe: int) -> None:
        """Write the ``.index`` and ``.invlists`` files of the ivf_pq subset of :meth:`subset` (see there)."""
        n = len(self.chain_ids)
        # The ids of an ivf_pq database are row numbers (added in order, from 0), so a kept id's new id is its rank
        # among the kept rows.
        keep = np.zeros(n, dtype=bool)
        keep[rows] = True
        rank = np.cumsum(keep, dtype=np.int32 if n <= np.iinfo(np.int32).max else np.int64)
        rank -= 1
        subset = faiss.read_index(str(self.db_folder / f"{self.index_name}.index"), faiss.IO_FLAG_SKIP_IVF_DATA)
        ivf = faiss.extract_index_ivf(subset)
        invlists = faiss.OnDiskInvertedLists(ivf.nlist, ivf.code_size, str(db_folder / f"{index_name}.invlists"))
        ivf.replace_invlists(invlists, True)
        # As in _create_ivf_pq_ondisk: the IVF index owns the lists now; without disown() Python would free them too.
        invlists.this.disown()
        parent_lists = faiss.extract_index_ivf(self.host_index()).invlists
        code_size = ivf.code_size
        total = 0
        for list_no in range(ivf.nlist):
            size = parent_lists.list_size(list_no)
            if size == 0:
                continue
            ids_ptr = parent_lists.get_ids(list_no)
            codes_ptr = parent_lists.get_codes(list_no)
            try:
                ids = faiss.rev_swig_ptr(ids_ptr, size)
                kept = keep[ids]
                n_kept = int(np.count_nonzero(kept))
                if n_kept:
                    new_ids = np.ascontiguousarray(rank[ids[kept]], dtype=np.int64)
                    new_codes = np.ascontiguousarray(
                        faiss.rev_swig_ptr(codes_ptr, size * code_size).reshape(size, code_size)[kept])
                    ivf.invlists.add_entries(list_no, n_kept, faiss.swig_ptr(new_ids), faiss.swig_ptr(new_codes))
                    total += n_kept
            finally:
                parent_lists.release_codes(list_no, codes_ptr)
                parent_lists.release_ids(list_no, ids_ptr)
        if total != len(rows):
            raise RuntimeError(f"The code copy found {total} of the {len(rows)} kept chains in the lists of "
                               f"{self.db_folder / self.index_name}: its lists do not hold one entry per chain.")
        # The header read keeps the parent's count on the IVF and on the OPQ pre-transform.
        ivf.ntotal = total
        subset.ntotal = total
        ivf.nprobe = nprobe
        _quantize_outside_parallel_regions(subset)
        faiss.write_index(subset, str(db_folder / f"{index_name}.index"))
        logger.info(f"Copied the PQ codes of {total} of {n} chains into {db_folder / index_name} "
                    f"(nlist {ivf.nlist}, nprobe {nprobe})")
        del subset

    def _subset_flat(self, rows: np.ndarray, db_folder: Path, index_name: str) -> None:
        """Write the ``.index`` file of the flat subset of :meth:`subset`: the kept rows' stored vectors (unit vectors
        already) in an exact ``IndexFlatIP``."""
        n = len(self.chain_ids)
        logger.info(f"Copying the stored vectors of {len(rows)} of {n} chains into an exact flat index at "
                    f"{db_folder / index_name}; it holds {len(rows) * self.dimension * 4:,} bytes in memory, so a flat "
                    "subset is for small sets")
        subset = faiss.IndexFlatIP(self.dimension)
        for start in range(0, n, _SUBSET_FLAT_BLOCK):
            stop = min(start + _SUBSET_FLAT_BLOCK, n)
            lo, hi = np.searchsorted(rows, [start, stop])
            if lo == hi:
                continue
            block = self.host_index().reconstruct_n(start, stop - start)
            subset.add(np.ascontiguousarray(block[rows[lo:hi] - start]))
        faiss.write_index(subset, str(db_folder / f"{index_name}.index"))

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
            _quantize_outside_parallel_regions(self.index)
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

    def move_to_gpu(self, gpu_id: int = 0, clustering: bool = False):
        """
        Move the index to GPU.

        Args:
            gpu_id: GPU device ID to use (default: 0; not used in the clustering mode, which takes every GPU)
            clustering: The clustering mode of ``fm-search cluster`` and ``similarity-graph`` (P9.CL D10): one replica
                of the index on every visible GPU (``index_cpu_to_all_gpus``, not sharded: each search call's queries
                are split between the GPUs), the PQ lookup tables of an ivf_pq index in float16 (an ivf_pq index of 64
                sub-quantizers of 8 bits is refused on an A100 with float32 tables), and the CPU index kept as
                ``self.cpu_index`` for ``reconstruct_n`` and :meth:`subset`. Set nprobe before the move: a GPU index
                keeps the one it was cloned with. Refused (:class:`~foldmatch.errors.RefusalError`) without GPU support
                instead of staying on the CPU; an hnsw index is refused (ValueError).
        """
        if self.is_gpu_index:
            logger.info("Index is already on GPU")
            return

        if clustering:
            self._clone_to_all_gpus()
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

    def _clone_to_all_gpus(self):
        """The clustering mode of :meth:`move_to_gpu` (see there)."""
        if self.index is None:
            raise ValueError("No index loaded. Call load_database() first.")
        _require_gpu_support()
        if isinstance(faiss.downcast_index(self.index), faiss.IndexHNSW):
            raise ValueError(f"GPU search takes flat and ivf_pq databases; {self.db_folder / self.index_name} is hnsw "
                             f"(built with --index-type {IndexType(self.index_type).value}), which FAISS searches on "
                             f"the CPU only.")
        n_gpus = faiss.get_num_gpus()
        options = faiss.GpuMultipleClonerOptions()
        options.shard = False
        # ivf_pq: float16 lookup tables (IVFPQ only; the coarse quantizer stays float32). On a flat index the same
        # option would store the vectors in float16, so it is off there.
        options.useFloat16 = self.index_type == IndexType.ivf_pq
        logger.info(f"Copying the index to {n_gpus} GPU(s), one replica each"
                    + (" (float16 lookup tables)" if options.useFloat16 else "")
                    + "; the CPU index is kept for reading the stored vectors")
        gpu_index = faiss.index_cpu_to_all_gpus(self.index, co=options, ngpu=n_gpus)
        self.cpu_index = self.index
        self.index = gpu_index
        self.is_gpu_index = True
        self.n_gpus = n_gpus
        logger.info("Index successfully moved to GPU")

    def host_index(self):
        """The CPU index: the index itself on the CPU, the one kept by ``move_to_gpu(clustering=True)`` on the GPU,
        None on the GPU without it (or when nothing is loaded)."""
        return self.cpu_index if self.is_gpu_index else self.index

    def release_index(self):
        """Drop the index (on the GPU, its replicas and the kept CPU index too); the chain ids stay."""
        self.index = None
        self.cpu_index = None
        self.is_gpu_index = False
        self.n_gpus = 0
        self.gpu_resources = None

    def move_to_cpu(self):
        """Move the index from GPU to CPU (the CPU index kept by the clustering mode, when there is one)."""
        if not self.is_gpu_index:
            logger.info("Index is already on CPU")
            return

        logger.info("Moving index to CPU...")
        self.index = self.cpu_index if self.cpu_index is not None else faiss.index_gpu_to_cpu(self.index)
        self.cpu_index = None
        self.n_gpus = 0
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

