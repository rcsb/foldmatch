import ctypes
import logging
import operator
import random
import sys
import time
import weakref
import numpy as np
import faiss
import igraph as ig
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm

from foldmatch.search import FaissEmbeddingDatabase
from foldmatch.search.embedding_database import _parse_db_path
from foldmatch.search.output import CLUSTER_OUTPUT_FIELDS, GREEDY_CLUSTER_OUTPUT_FIELDS, write_cluster_results
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
# Chains whose output rows are made at a time by export_clusters.
_EXPORT_CHUNK = 1 << 20
# Leiden iterations of cluster_leiden: a fixed count, since run-to-convergence is open-ended (221 iterations on a
# 1M-chain graph).
_LEIDEN_ITERATIONS = 10


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
    """Cluster protein structure embeddings: Leiden on the similarity graph (:meth:`cluster_leiden`), or greedy
    representatives in one streamed pass over the database (:meth:`cluster_greedy`)."""

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
        # The smallest kept edge weight of the last build_similarity_graph, and a weak reference to the graph it was
        # recorded for (cluster_leiden refuses a negative one for that graph only)
        self.min_weight = None
        self._min_weight_graph = None
        # The method of the last clustering ("leiden" or "greedy"), and under greedy each cluster's representative
        self.method = None
        self.representatives = None

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
        self.min_weight = None
        self._min_weight_graph = None
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
        # The smallest kept weight (+inf with no edge), before the weight blocks are consumed: Leiden refuses negative
        # weights
        min_weight = min((float(b.min()) for b in weight_blocks), default=float("inf"))
        self.graph.es['weight'] = _weight_list(weight_blocks)
        _release_freed_memory()

        # Add chain IDs as vertex attribute
        self.graph.vs['name'] = self.chain_ids
        self.min_weight = min_weight
        self._min_weight_graph = weakref.ref(self.graph)

        logging.info(f"Similarity graph built in {time.perf_counter() - start_time:.2f}s")

        return self.graph

    def cluster_leiden(
            self,
            resolution: float = 1.0,
            n_iterations: int = _LEIDEN_ITERATIONS,
            seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply Leiden clustering algorithm to the similarity graph.

        The call is python-igraph's ``Graph.community_leiden(objective_function="modularity", weights="weight",
        resolution=resolution, beta=0.01, n_iterations=n_iterations)``: modularity at ``resolution`` with the edge
        weights. It runs a fixed number of iterations (``n_iterations``, the CLI's --leiden-iterations), 10 by default,
        since run-to-convergence is open-ended on sparse graphs (221 iterations on a 1M-chain graph).

        Seeding: python-igraph draws from a ``random.Random(seed)`` of its own during the call
        (``ig.set_random_number_generator``), so Python's global ``random`` state is neither used nor changed; the same
        seed gives the same partition, and ``seed=None`` (seeded from ``os.urandom``) a different one each call.
        Caveats: afterwards python-igraph's generator is set back to the ``random`` module, its default (python-igraph
        has no getter), so a caller's own igraph generator is replaced; and the generator is process-global, so igraph
        calls of other threads during the Leiden step draw from it.

        Cluster ids are numbered by decreasing cluster size, ties by the smallest chain index in the cluster (0 is the
        largest cluster), as under greedy.

        Args:
            resolution: Resolution parameter (higher = more clusters, default: 1.0)
            n_iterations: Number of Leiden iterations, at least 1 (default: 10)
            seed: Random seed for reproducibility, an integer of at least 0 (``operator.index``: NumPy integers are
                accepted, floats refused with TypeError)

        Returns:
            Array of cluster IDs (one per chain), int32 below 2**31 chains

        Raises:
            ValueError: ``n_iterations`` below 1, ``seed`` below 0, no graph, or a graph built by
                build_similarity_graph with a negative kept weight (modularity needs non-negative edge weights)
        """
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1 (Leiden runs a fixed number of iterations).")
        if seed is not None:
            seed = operator.index(seed)
            if seed < 0:
                raise ValueError("--seed must be at least 0.")
        if self.graph is None:
            raise ValueError("Graph not built. Call build_similarity_graph() first.")
        if self._min_weight_graph is not None and self._min_weight_graph() is self.graph and self.min_weight < 0:
            raise ValueError(f"Leiden needs non-negative edge weights: the graph's smallest weight is "
                             f"{self.min_weight:.4g} (edges keep scores at or above the threshold of "
                             f"build_similarity_graph).")

        logging.info(f"Running Leiden clustering (resolution={resolution})...")
        start_time = time.perf_counter()

        # Higher edge weights (higher similarity) encourage nodes to be in same cluster. The objective must be exactly
        # "modularity": python-igraph normalises the resolution only for that spelling. A private generator: Python's
        # global random state is left alone.
        rng = random.Random(seed)
        ig.set_random_number_generator(rng)
        try:
            partition = self.graph.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution=resolution,
                beta=0.01,
                n_iterations=n_iterations
            )
        finally:
            # python-igraph 1.0.0's default; it has no getter
            ig.set_random_number_generator(random)

        # Cluster ids by decreasing size, ties by smallest chain index (igraph numbers by first appearance)
        n = self.graph.vcount()
        dtype = _graph_id_dtype(n)
        membership = np.fromiter(partition.membership, dtype=dtype, count=n)
        del partition
        sizes = np.bincount(membership)
        # Every cluster has a member, so minimum.at overwrites the fill
        first = np.full(len(sizes), n - 1, dtype=dtype)
        np.minimum.at(first, membership, np.arange(n, dtype=dtype))
        order = np.lexsort((first, -sizes))
        new_id = np.empty(len(sizes), dtype=dtype)
        new_id[order] = np.arange(len(sizes), dtype=dtype)
        self.clusters = new_id[membership]
        del membership, first, order, new_id

        logging.info(f"Found {len(sizes)} clusters")
        logging.info(f"Leiden clustering completed in {time.perf_counter() - start_time:.2f}s")
        self.method = "leiden"
        self.representatives = None

        return self.clusters

    def cluster_greedy(
            self,
            threshold: float = 0.8,
            max_neighbors: int = 1000
    ) -> np.ndarray:
        """
        Greedy representatives in database order, in one streamed pass over the database (no graph).

        The partition: chains are taken in database order; a chain already in a cluster is passed over; any other
        chain becomes a representative, and its cluster takes every chain of its k-NN result row (k =
        min(max_neighbors, number of chains), searched as build_similarity_graph searches it) that scores at least
        ``threshold`` and is in no cluster yet. A member joins on its representative's score for it. A cluster has at
        most k + 1 chains, and a representative whose k-th neighbour scores at least ``threshold`` may lose neighbours
        above the threshold beyond rank k (its cluster may be cut): such a chain can become a representative, or a
        singleton, that scores at least ``threshold`` with an earlier representative.

        How it is computed: build_similarity_graph's blocks, in ascending order; a block searches, in one FAISS call,
        only its chains in no cluster at the block's start (a block with none is not read), then takes them in order,
        passing over those an earlier chain of the block has taken. Each block's results are dropped before the next,
        so memory grows with the number of chains, not with the edges. Where a chain's search result does not depend
        on the other chains of its FAISS call (hnsw, and flat while a block's rows x dimension stay below FAISS's BLAS
        switch, faiss.cvar.distance_compute_blas_threshold), this is the partition above bit for bit, whatever the
        block size; otherwise a score can move by float rounding with the call's rows (up to about 2e-6 on flat at
        1536 dimensions, a few 1e-7 on ivf_pq), so a pair at ``threshold`` or tied at the k-th rank can change. There
        is no randomness.

        Cluster ids are numbered by decreasing cluster size, ties by representative index (0 is the largest cluster);
        ``self.representatives[i]`` is the chain index of cluster i's representative.

        Args:
            threshold: A representative takes the chains in no cluster yet scoring at least this (cosine; identical
                chains score 1, or close to 1 on ivf_pq, where scores are PQ estimates)
            max_neighbors: Neighbours searched per representative, so also the cluster-size cap (this + 1)

        Returns:
            Array of cluster IDs (one per chain)
        """
        if self.db.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")
        if max_neighbors < 1:
            raise ValueError("--max-neighbors must be at least 1.")

        n_chains = len(self.chain_ids)
        # Clamp max_neighbors to actual database size
        k = min(max_neighbors, n_chains)
        logging.info(f"\nGreedy clustering with threshold >= {threshold}")
        logging.info(f"Searching up to {k} neighbors per representative, only chains in no cluster yet")
        # Chains per block, as build_similarity_graph's
        rows = 1 if k < 1 else max(1, min(n_chains, _GRAPH_MAX_ROWS, _GRAPH_RESULT_BYTES // (k * 12)))

        # Get CPU index for reconstruction if using GPU
        index_to_use = self.db.index
        if self.db.is_gpu_index:
            logging.info("Moving GPU index to CPU for reconstruction...")
            index_to_use = faiss.index_gpu_to_cpu(self.db.index)

        # ivf_pq queries are normalised, as in build_similarity_graph
        normalize = self.db.index_type == IndexType.ivf_pq

        # Each chain's representative (-1: in no cluster yet), int32 when the largest id fits
        labels = np.full(n_chains, -1, dtype=_graph_id_dtype(n_chains))
        n_searched = 0
        n_saturated = 0
        n_representatives = 0
        start_time = time.perf_counter()

        with tqdm(total=n_chains, desc="Clustering chains", unit="chain") as progress:
            for start in range(0, n_chains, rows):
                end = min(start + rows, n_chains)
                # The block's chains in no cluster at its start: the only ones searched
                todo = start + np.flatnonzero(labels[start:end] < 0)
                if len(todo):
                    embeddings = np.ascontiguousarray(index_to_use.reconstruct_n(start, end - start), dtype=np.float32)
                    if normalize:
                        faiss.normalize_L2(embeddings)
                    queries = np.ascontiguousarray(embeddings[todo - start], dtype=np.float32)
                    del embeddings
                    scores, ids = self.db.index.search(queries, k)
                    del queries
                    n_searched += len(todo)

                    # Kept cells: at least the threshold, and not the -1 ids FAISS fills in
                    ok = (scores >= threshold) & (ids >= 0)
                    for r, row in enumerate(todo.tolist()):
                        if labels[row] >= 0:
                            continue
                        labels[row] = row
                        n_representatives += 1
                        neighbors = ids[r][ok[r]]
                        labels[neighbors[labels[neighbors] < 0]] = row
                        # The k-th neighbour is kept: the cluster may be cut (never when every chain is searched)
                        if k < n_chains and ok[r, -1]:
                            n_saturated += 1
                    del scores, ids, ok

                progress.set_postfix(clusters=n_representatives, searched=n_searched, refresh=False)
                progress.update(end - start)

        # Cluster ids by decreasing size, ties by representative index
        is_rep = labels == np.arange(n_chains, dtype=labels.dtype)
        rank = (np.cumsum(is_rep, dtype=labels.dtype) - 1)[labels]
        sizes = np.bincount(rank)
        order = np.argsort(-sizes, kind="stable")
        new_id = np.empty_like(order, dtype=labels.dtype)
        new_id[order] = np.arange(len(order), dtype=labels.dtype)
        self.clusters = new_id[rank]
        self.representatives = np.flatnonzero(is_rep)[order]
        del labels, is_rep, rank, order, new_id

        self.method = "greedy"
        self.k = k
        self.max_neighbors = max_neighbors
        self.n_searched = n_searched
        self.n_saturated = n_saturated

        n_singletons = int(np.count_nonzero(sizes == 1))
        share = n_searched / n_chains if n_chains else 0.0
        logging.info(f"Searched {n_searched} of {n_chains} chains ({share:.2%})")
        logging.info(f"Found {n_representatives} clusters, {n_singletons} singletons")
        logging.info(f"Greedy clustering completed in {time.perf_counter() - start_time:.2f}s")

        return self.clusters

    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about the clustering results.

        Returns:
            Dictionary with clustering statistics
        """
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call cluster_leiden() or cluster_greedy() first.")

        # Memberships are 0..C-1 (Leiden and greedy)
        sizes = np.bincount(self.clusters).tolist()
        n_clusters = len(sizes)
        n_singletons = sum(1 for s in sizes if s == 1)

        # Calculate modularity if graph exists and the partition is not greedy's, which is not computed on a graph
        modularity = None
        if self.graph is not None and self.method != "greedy":
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
        if self.method == "greedy":
            stats.update({
                'method': self.method,
                'n_searched': self.n_searched,
                'n_saturated': self.n_saturated,
                'n_at_cap': sum(1 for s in sizes if s >= self.max_neighbors),
            })

        return stats

    def export_clusters(
            self,
            output_file: str,
            min_cluster_size: Optional[int] = None,
            format_mode: FormatMode = FormatMode.headless_tsv,
    ):
        """
        Export cluster assignments as tab-separated rows with no header, in chain order.

        Columns are :data:`~foldmatch.search.output.CLUSTER_OUTPUT_FIELDS`
        (``chain_id, cluster_id, cluster_size``), matching the file convention
        used by the search commands; after :meth:`cluster_greedy`,
        :data:`~foldmatch.search.output.GREEDY_CLUSTER_OUTPUT_FIELDS`, which add
        the representative's chain id.

        Args:
            output_file: Path to output file
            min_cluster_size: Only include clusters with at least this many members
            format_mode: File layout (header or headless); see :class:`FormatMode`
        """
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call cluster_leiden() or cluster_greedy() first.")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate cluster sizes (memberships are 0..C-1)
        cluster_sizes = np.bincount(self.clusters)
        greedy = self.method == "greedy"

        def _assignments():
            # Rows made _EXPORT_CHUNK chains at a time, not as one list of every chain
            for s in range(0, len(self.clusters), _EXPORT_CHUNK):
                cluster_ids = np.asarray(self.clusters[s:s + _EXPORT_CHUNK])
                sizes = cluster_sizes[cluster_ids]
                columns = [self.chain_ids[s:s + _EXPORT_CHUNK], cluster_ids.tolist(), sizes.tolist()]
                if greedy:
                    columns.append([self.chain_ids[r] for r in self.representatives[cluster_ids].tolist()])
                chunk = zip(*columns)
                if min_cluster_size is not None:
                    chunk = (row for row, keep in zip(chunk, (sizes >= min_cluster_size).tolist()) if keep)
                yield from chunk

        exported = write_cluster_results(_assignments(), str(output_path), format_mode,
                                         GREEDY_CLUSTER_OUTPUT_FIELDS if greedy else CLUSTER_OUTPUT_FIELDS)
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
        if stats.get('method') == "greedy":
            n_chains, n_clusters = stats['n_chains'], stats['n_clusters']
            saturated_share = stats['n_saturated'] / n_clusters
            logging.info(f"Method:              {stats['method']}")
            logging.info(f"Chains searched:     {stats['n_searched']} of {n_chains} "
                         f"({stats['n_searched'] / n_chains:.2%})")
            logging.info(f"Representatives with all --max-neighbors neighbours at or above --threshold: "
                         f"{stats['n_saturated']} of {n_clusters} ({saturated_share:.2%}); their clusters may be cut, "
                         f"and a larger --max-neighbors can change them.")
            logging.info(f"Clusters of at least --max-neighbors chains: {stats['n_at_cap']}")
            if saturated_share > 0.5:
                logging.warning("Most clusters are bounded by --max-neighbors, not --threshold: a representative or "
                                "singleton may score at or above --threshold with an earlier representative.")
