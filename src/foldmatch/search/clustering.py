import ctypes
import logging
import sys
import time
import numpy as np
import faiss
import igraph as ig
import leidenalg
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from foldmatch.search import FaissEmbeddingDatabase
from foldmatch.search.embedding_database import _parse_db_path
from foldmatch.search.output import write_cluster_results
from foldmatch.types.api_types import FormatMode, IndexType

logger = logging.getLogger(__name__)

# Byte budget of the search results of one block of chains (k int64 ids and k float32 scores a chain): 16,384 chains
# a FAISS call at k = 1000. The block's embeddings are read with one reconstruct_n call of the same rows.
_GRAPH_RESULT_BYTES = 16_384 * 1000 * 12
# Row cap of a block, which bounds its embeddings at small k (16,384 x d x 4 bytes, 96 MiB at d = 1536).
_GRAPH_MAX_ROWS = 16_384
# Kept node ids are int32 (8 bytes an edge instead of 16 while the search runs) when the largest id fits, else int64.
_GRAPH_INT32_MAX_ID = np.iinfo(np.int32).max
# Edges (and weights) turned into Python objects at a time when handed to igraph.
_GRAPH_EDGE_CHUNK = 1 << 20
# An igraph attribute is a list of Python objects, one float an edge for the weights: equal weights share one float
# when the kept weights' float32 bit patterns span at most this many values, in a table of 128 MiB of pointers and 16
# MiB of flags at most (scores from 0.8 to 1.03 span about 3.6M); a wider span (scores of both signs, 0.0 with -0.0
# included, or, with scores up to about 1.03, scores below about 0.26) gives one float an edge, as ndarray.tolist().
_GRAPH_WEIGHT_TABLE_MAX = 1 << 24


def _graph_id_dtype(n_chains: int):
    """The dtype of the kept node ids of a graph of ``n_chains`` chains: int32 when the largest id fits, else int64."""
    return np.int32 if n_chains - 1 <= _GRAPH_INT32_MAX_ID else np.int64


def _edge_tuples(blocks: list):
    """Yield the edges of the (m, 2) blocks as tuples of two ints, block by block, in chunks of _GRAPH_EDGE_CHUNK.
    The list is emptied as it goes, so each block is released once consumed."""
    blocks.reverse()
    while blocks:
        block = blocks.pop()
        for s in range(0, len(block), _GRAPH_EDGE_CHUNK):
            yield from map(tuple, block[s:s + _GRAPH_EDGE_CHUNK].tolist())
        del block


def _weight_list(blocks: list) -> list:
    """The float32 weight blocks as one list of Python floats, in order, emptying the list as it goes. Each weight is
    the float of its own float32 value, as ndarray.tolist() gives; equal bit patterns share one float object when the
    patterns span at most _GRAPH_WEIGHT_TABLE_MAX values (0.0 and -0.0 stay apart)."""
    weights = []
    nonempty = [b for b in blocks if len(b)]
    if not nonempty:
        blocks.clear()
        return weights
    lo = min(int(b.view(np.uint32).min()) for b in nonempty)
    span = max(int(b.view(np.uint32).max()) for b in nonempty) - lo + 1
    del nonempty
    table = seen = None
    if span <= _GRAPH_WEIGHT_TABLE_MAX:
        table = np.empty(span, dtype=object)
        seen = np.zeros(span, dtype=bool)
    blocks.reverse()
    while blocks:
        block = blocks.pop()
        for s in range(0, len(block), _GRAPH_EDGE_CHUNK):
            if table is None:
                weights.extend(block[s:s + _GRAPH_EDGE_CHUNK].tolist())
                continue
            idx = block[s:s + _GRAPH_EDGE_CHUNK].view(np.uint32).astype(np.int64) - lo
            new = np.unique(idx[~seen[idx]])
            if len(new):
                table[new] = (new + lo).astype(np.uint32).view(np.float32).tolist()
                seen[new] = True
            weights.extend(table[idx].tolist())
        del block
    return weights


def _release_freed_memory():
    """Hand freed heap memory back to the OS with glibc's malloc_trim(0); a no-op elsewhere or without the symbol."""
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (OSError, AttributeError):
        pass


class EmbeddingClusterer:
    """Cluster protein structure embeddings using Leiden algorithm on similarity graphs."""

    def __init__(self, db_path: str, use_gpu: bool = False):
        """
        Initialize clustering — stashes db location; actual load happens in
        :meth:`load_database`.

        Args:
            db_path: Path to FAISS database directory
            index_name: Name of the FAISS index
        """
        db_folder, index_name, db_path = _parse_db_path(db_path)
        self.db = FaissEmbeddingDatabase(db_folder, index_name)
        self.db.load_database(use_gpu=use_gpu)
        self.chain_ids = self.db.chain_ids

        self.graph = None
        self.clusters = None

    def build_similarity_graph(
            self,
            threshold: float = 0.8,
            max_neighbors: int = 1000
    ) -> ig.Graph:
        """
        Build a similarity graph using FAISS k-NN search with threshold filtering.

        Args:
            threshold: Similarity threshold for edge creation (cosine; identical chains score 1, or close to 1 on
                ivf_pq, where scores are PQ estimates)
            max_neighbors: Maximum number of neighbors to consider per node

        Returns:
            igraph Graph object with edges weighted by similarity
        """
        if self.db.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")

        n_chains = len(self.chain_ids)
        logging.info(f"\nBuilding similarity graph with threshold >= {threshold}")
        logging.info(f"Searching up to {max_neighbors} neighbors per chain")

        # Clamp max_neighbors to actual database size
        k = min(max_neighbors, n_chains)
        # Chains per block, from the result budget and the row cap (k = 0 is left to FAISS to refuse, as before, after
        # one chain is read)
        rows = 1 if k < 1 else max(1, min(n_chains, _GRAPH_MAX_ROWS, _GRAPH_RESULT_BYTES // (k * 12)))

        # Get CPU index for reconstruction if using GPU
        index_to_use = self.db.index
        if self.db.is_gpu_index:
            logging.info("Moving GPU index to CPU for reconstruction...")
            index_to_use = faiss.index_gpu_to_cpu(self.db.index)

        # ivf_pq decodes its PQ codes, which are not unit vectors: the query is normalised, as in query db, so a score
        # is the inner product of the chain's unit decoded embedding with each stored code. flat and hnsw store the
        # normalised embeddings, searched as they are.
        normalize = self.db.index_type == IndexType.ivf_pq

        id_dtype = _graph_id_dtype(n_chains)
        pair_blocks = []
        weight_blocks = []
        n_edges = 0
        start_time = time.perf_counter()

        # Blocks of chains in ascending order, read with reconstruct_n (no direct map needed on ivf_pq) and searched
        # with one FAISS call each. An edge (i, j) is kept from the search of the lower chain i only, which also drops
        # self-hits and the -1 ids FAISS fills in; edges come in chain order, then in FAISS rank order. A block's kept
        # pairs are stored with int32 ids when the database's largest id fits, and a block with no kept edge is not
        # stored.
        logging.info("Performing k-NN search ...")
        with tqdm(total=n_chains, desc="Processing chains", unit="chain") as progress:
            for start in range(0, n_chains, rows):
                end = min(start + rows, n_chains)
                embeddings = np.ascontiguousarray(index_to_use.reconstruct_n(start, end - start), dtype=np.float32)
                if normalize:
                    faiss.normalize_L2(embeddings)
                scores, indices = self.db.index.search(embeddings, k)
                del embeddings

                chains = np.arange(start, end, dtype=np.int64)[:, None]
                block_rows, block_cols = np.nonzero((chains < indices) & (scores >= threshold))
                if len(block_rows):
                    pair_blocks.append(np.stack([block_rows + start, indices[block_rows, block_cols]],
                                                axis=1).astype(id_dtype, copy=False))
                    weight_blocks.append(scores[block_rows, block_cols])
                    n_edges += len(block_rows)
                del scores, indices, chains, block_rows, block_cols

                progress.set_postfix(edges=n_edges, refresh=False)
                progress.update(end - start)

        logging.info(f"Created graph with {n_chains} nodes and {n_edges} edges")

        # Create igraph from the blocks, with no concatenation, through a generator of int tuples: python-igraph 1.0.0
        # converts an edge ndarray through Python objects as well, at about 170 bytes an edge at its peak against 50 to
        # 57. Each block is released once consumed, and each weight block once its floats are in the list. An error
        # raised while igraph pulls the edges (MemoryError, KeyboardInterrupt) surfaces from python-igraph 1.0.0 as a
        # SystemError, with the original as its __cause__.
        self.graph = ig.Graph(n=n_chains, edges=_edge_tuples(pair_blocks), directed=False)
        # glibc keeps freed blocks in the heap: give them back before the weights and before Leiden.
        _release_freed_memory()
        self.graph.es['weight'] = _weight_list(weight_blocks)
        _release_freed_memory()

        # Add chain IDs as vertex attribute
        self.graph.vs['name'] = self.chain_ids

        logging.info(f"Similarity graph built in {time.perf_counter() - start_time:.2f}s")

        return self.graph

    def cluster_leiden(
            self,
            resolution: float = 1.0,
            n_iterations: int = -1,
            seed: Optional[int] = None
    ) -> List[int]:
        """
        Apply Leiden clustering algorithm to the similarity graph.

        Args:
            resolution: Resolution parameter (higher = more clusters, default: 1.0)
            n_iterations: Number of iterations (-1 for until convergence)
            seed: Random seed for reproducibility

        Returns:
            List of cluster IDs (one per chain)
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_similarity_graph() first.")

        logging.info(f"Running Leiden clustering (resolution={resolution})...")
        start_time = time.perf_counter()

        # Use RBConfigurationVertexPartition with weights
        # Higher edge weights (higher similarity) encourage nodes to be in same cluster
        partition = leidenalg.find_partition(
            self.graph,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=seed
        )

        self.clusters = partition.membership
        logging.info(f"Found {len(set(self.clusters))} clusters")
        logging.info(f"Leiden clustering completed in {time.perf_counter() - start_time:.2f}s")

        return self.clusters

    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about the clustering results.

        Returns:
            Dictionary with clustering statistics
        """
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call cluster_leiden() first.")

        # Leiden memberships are 0..C-1
        sizes = np.bincount(self.clusters).tolist()
        n_clusters = len(sizes)
        n_singletons = sum(1 for s in sizes if s == 1)

        # Calculate modularity if graph exists
        modularity = None
        if self.graph is not None:
            modularity = self.graph.modularity(self.clusters, weights='weight')

        stats = {
            'n_chains': len(self.clusters),
            'n_clusters': n_clusters,
            'n_singletons': n_singletons,
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes),
            'mean_cluster_size': np.mean(sizes),
            'median_cluster_size': np.median(sizes),
            'modularity': modularity
        }

        return stats

    def export_clusters(
            self,
            output_file: str,
            min_cluster_size: Optional[int] = None,
            format_mode: FormatMode = FormatMode.headless_tsv,
    ):
        """
        Export cluster assignments as tab-separated rows with no header.

        Columns are :data:`~foldmatch.search.output.CLUSTER_OUTPUT_FIELDS`
        (``chain_id, cluster_id, cluster_size``), matching the file convention
        used by the search commands.

        Args:
            output_file: Path to output file
            min_cluster_size: Only include clusters with at least this many members
            format_mode: File layout (header or headless); see :class:`FormatMode`
        """
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call cluster_leiden() first.")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate cluster sizes (Leiden memberships are 0..C-1)
        cluster_sizes = np.bincount(self.clusters).tolist()

        assignments = [
            (chain_id, cluster_id, cluster_sizes[cluster_id])
            for chain_id, cluster_id in zip(self.chain_ids, self.clusters)
            if min_cluster_size is None or cluster_sizes[cluster_id] >= min_cluster_size
        ]
        exported = write_cluster_results(assignments, str(output_path), format_mode)
        logging.info(f"Cluster assignments exported to {output_file}")

        if min_cluster_size:
            total_count = len(self.clusters)
            logging.info(f"Exported {exported}/{total_count} chains (min_cluster_size={min_cluster_size})")

    def print_statistics(self):
        """Print clustering statistics to console."""
        stats = self.get_cluster_statistics()
        logging.info("CLUSTERING STATISTICS")
        logging.info(f"Total embeddings:    {stats['n_chains']}")
        logging.info(f"Number of clusters:  {stats['n_clusters']}")
        logging.info(f"Singleton clusters:  {stats['n_singletons']}")
        logging.info(f"Cluster size (min):  {stats['min_cluster_size']}")
        logging.info(f"Cluster size (max):  {stats['max_cluster_size']}")
        logging.info(f"Cluster size (mean): {stats['mean_cluster_size']:.2f}")
        logging.info(f"Cluster size (med):  {stats['median_cluster_size']:.1f}")
        if stats['modularity'] is not None:
            logging.info(f"Modularity:          {stats['modularity']:.4f}")
