import ctypes
import json
import logging
import operator
import os
import random
import sys
import time
import weakref
from datetime import datetime, timezone
import numpy as np
import faiss
import igraph as ig
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm

from foldmatch import __version__
from foldmatch.search import FaissEmbeddingDatabase
from foldmatch.search.embedding_database import _parse_db_path
from foldmatch.search.output import CLUSTER_OUTPUT_FIELDS, GREEDY_CLUSTER_OUTPUT_FIELDS, write_cluster_results
from foldmatch.types.api_types import EdgeRule, FormatMode, IndexType

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

# greedy-leiden (P9.CL D2): the defaults of the greedy step and of the representatives' graph. The greedy threshold is
# the user's (2026-11-14); k = 100 is the k of every measured cascade (TrEMBL 1M-10M, work/p9t10m/a/CASCADE_SIZE.md).
_GREEDY_LEIDEN_GREEDY_THRESHOLD = 0.9
_GREEDY_LEIDEN_GREEDY_MAX_NEIGHBORS = 1000
_GREEDY_LEIDEN_MAX_NEIGHBORS = 100
# The Leiden resolution of greedy-leiden (P9.CL D4): on this method over TrEMBL samples of 1M and 10M chains (greedy
# 0.9, k = 100, P9.CL V3.1) gamma 20 left no cluster above 1.745% of the chains at 10M (at most one at 1M, against 4-7
# at gamma 1) and agreed better with Pfam families than 10, at some cost on domain superfamilies.
_GREEDY_LEIDEN_RESOLUTION = 20.0
# Giant clusters in the statistics (P9.CL D4): more than 1.745% of the chains, compared in integers (size x 100,000 >
# 1,745 x chains), the rule of the cascade tables the resolution default rests on (work/p9t10m/a/CASCADE_SIZE.md,
# work/p9res/a/RES_SWEEP.md). Under greedy-leiden they are a WARNING from _GIANT_WARN_MIN_CHAINS chains on, the
# smallest set that default was measured on (on small or redundant sets such a cluster is ordinary).
_GIANT_PER_100K = 1_745
_GIANT_WARN_MIN_CHAINS = 1_000_000
# The largest cluster sizes the statistics give
_LARGEST_CLUSTERS = 10
# Leiden memory a representatives' graph edge, logged after greedy for the edge bound k x R (measured whole-run peaks:
# igraph 64.2 B an edge on Swiss-Prot, 63.8 on TrEMBL; NetworKit's graph 32-34 B an edge plus about 70 B a node).
_IGRAPH_BYTES_PER_EDGE = 64
_NETWORKIT_BYTES_PER_EDGE = 34
_NETWORKIT_BYTES_PER_NODE = 70

# The work folder of greedy-leiden and of a resumable greedy (P9.CL D3). Greedy's state is written every
# _CHECKPOINT_SECONDS of greedy (and at its end); the clock is a module attribute so that tests can drive it.
_CHECKPOINT_SECONDS = 900
_clock = time.monotonic
# The layout of a work folder, recorded in its run.json: another value is refused.
_WORK_FOLDER_FORMAT = 1
_RUN_FILE = "run.json"
_GREEDY_STATE_FILE = "greedy.state.npz"
_REPRESENTATIVES_NAME = "representatives"
_REPRESENTATIVES_DONE = "representatives.done"
_GRAPH_RECORD_FILE = "graph.json"
_GRAPH_WEIGHTS_FILE = "graph.weights.f32"
_GRAPH_DONE = "graph.done"
# The edge pairs file, by id dtype (int32 when the largest id fits, as in memory; 8 or 16 bytes an edge)
_GRAPH_PAIRS_FILES = {np.dtype(np.int32): "graph.pairs.i32", np.dtype(np.int64): "graph.pairs.i64"}
# Every file a work folder may hold (temporary names included), for the --output-file refusal
WORK_FOLDER_FILES = (_RUN_FILE, f"{_RUN_FILE}.tmp", _GREEDY_STATE_FILE, f"{_GREEDY_STATE_FILE}.tmp",
                     *(f"{_REPRESENTATIVES_NAME}{suffix}" for suffix in (".index", ".invlists", ".metadata")),
                     _REPRESENTATIVES_DONE, *_GRAPH_PAIRS_FILES.values(), _GRAPH_WEIGHTS_FILE, _GRAPH_RECORD_FILE,
                     f"{_GRAPH_RECORD_FILE}.tmp", _GRAPH_DONE)


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


def _fsync_folder(folder: Path):
    """fsync a folder, so that a rename in it is on disk (a no-op where folders cannot be opened, as on Windows)."""
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomically(path: Path, write):
    """Write ``path`` through ``write(file)`` to ``<path>.tmp``, fsync it, then rename it over ``path``: a reader (or a
    run stopped while writing, a full disk included) sees the old file or the new one, never part of one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_folder(path.parent)


def _write_json(path: Path, record: dict):
    _write_atomically(path, lambda f: f.write((json.dumps(record, indent=1) + "\n").encode()))


def _touch_marker(path: Path):
    """A step's done marker, written once the step's files are on disk."""
    _write_atomically(path, lambda f: None)


class _WorkFolderRefused(ValueError):
    """A work folder that this run cannot use: other recorded settings or another parent database, or files of another
    layout (the CLI reports it as a usage error of --work-folder)."""


class _MethodNotApplicable(ValueError):
    """A database that greedy-leiden does not take (the CLI reports it as a usage error, after the load)."""


class _NegativeEdgeWeights(ValueError):
    """A representatives' graph with a negative kept weight, which Leiden's modularity cannot take."""

    def __init__(self, message: str, min_weight: float):
        super().__init__(message)
        self.min_weight = min_weight


class _WorkFolder:
    """The work folder of :meth:`EmbeddingClusterer.cluster_greedy_leiden` and of a resumable
    :meth:`EmbeddingClusterer.cluster_greedy` (P9.CL D3): ``run.json`` (the record), ``greedy.state.npz`` (greedy's
    labels and counters), the representatives' database and its ``representatives.done`` marker, and the
    representatives' graph (``graph.pairs.i32``, ``graph.weights.f32``, ``graph.json``, ``graph.done``)."""

    # The recorded settings compared on a rerun, by section: (key, description)
    _COMPARED = (("parent", (("n_chains", "parent chain count"), ("index_sha256", "parent .index sha256"),
                             ("metadata_sha256", "parent .metadata sha256"),
                             ("invlists_bytes", "parent .invlists size"))),
                 (None, (("nprobe", "nprobe"),)),
                 ("greedy", (("threshold", "greedy threshold"), ("max_neighbors", "greedy max_neighbors"))),
                 ("graph", (("threshold", "graph threshold"), ("max_neighbors", "graph max_neighbors"),
                            ("edge_rule", "graph edge_rule"))))

    def __init__(self, path):
        self.path = Path(path)
        self.record = None

    def file(self, name: str) -> Path:
        return self.path / name

    def has(self, name: str) -> bool:
        return self.file(name).exists()

    def remove(self, *names: str):
        for name in names:
            self.file(name).unlink(missing_ok=True)

    def open(self, record: dict, method: str):
        """Check ``record`` (this run's parent, nprobe and step settings) against ``run.json``, or create it, and
        append this run. Refused (:class:`_WorkFolderRefused`): a folder with work files but no ``run.json``, a
        ``run.json`` of another layout, or any compared setting that differs, each named with both values."""
        if self.path.exists() and not self.path.is_dir():
            raise _WorkFolderRefused(f"The work folder {self.path} is a file, not a folder.")
        self.path.mkdir(parents=True, exist_ok=True)
        run_file = self.file(_RUN_FILE)
        threads = faiss.omp_get_max_threads()
        if run_file.exists():
            try:
                stored = json.loads(run_file.read_text())
            except ValueError as exc:
                raise _WorkFolderRefused(f"The work folder {self.path} has a {_RUN_FILE} that is not readable "
                                         f"({exc}): use a new --work-folder.") from None
            if not isinstance(stored, dict) or stored.get("format") != _WORK_FOLDER_FORMAT:
                found = stored.get("format") if isinstance(stored, dict) else None
                raise _WorkFolderRefused(f"The work folder {self.path} was made by another FoldMatch layout "
                                         f"({_RUN_FILE} format {found!r}, this version writes {_WORK_FOLDER_FORMAT}): "
                                         f"use a new --work-folder.")
            differences = []
            for section, keys in self._COMPARED:
                old = stored if section is None else stored.get(section)
                new = record if section is None else record.get(section)
                if old is None or new is None:
                    continue
                for key, described in keys:
                    if old.get(key) != new.get(key):
                        differences.append(f"{described} {old.get(key)!r} (this run: {new.get(key)!r})")
            if differences:
                raise _WorkFolderRefused(
                    f"The work folder {self.path} holds the steps of a run with other settings or another "
                    f"database: {'; '.join(differences)}. Use a new --work-folder for this run, or rerun with the "
                    f"recorded settings.")
            for section in ("greedy", "graph"):
                if stored.get(section) is None and record.get(section) is not None:
                    stored[section] = record[section]
            if stored["parent"].get("path") != record["parent"].get("path"):
                logging.info(f"The parent database is now {record['parent']['path']} (recorded: "
                             f"{stored['parent'].get('path')}); its files are the same.")
            versions = {(run.get("foldmatch_version"), run.get("faiss_version")) for run in stored.get("runs", [])}
            if versions - {(__version__, faiss.__version__)}:
                logging.warning(f"This work folder's earlier steps ran on other FoldMatch or FAISS versions "
                                f"({', '.join(f'FoldMatch {a}, FAISS {b}' for a, b in sorted(versions, key=str))}; "
                                f"this run: FoldMatch {__version__}, FAISS {faiss.__version__}): reused or resumed "
                                f"steps may differ from a run on one version.")
            earlier = {step["faiss_threads"] for run in stored.get("runs", [])
                       for step in run.get("steps", {}).values() if not step.get("skipped")}
            if earlier - {threads}:
                logging.warning(f"FAISS runs on {threads} threads; this work folder's earlier steps ran on "
                                f"{', '.join(str(t) for t in sorted(earlier))}: the graph can differ by a few edges "
                                f"from a run at one thread count.")
        else:
            found = [name for name in WORK_FOLDER_FILES if self.has(name)]
            if found:
                raise _WorkFolderRefused(f"The work folder {self.path} holds work files ({', '.join(found)}) but no "
                                         f"{_RUN_FILE}, as an older FoldMatch version leaves it: use a new "
                                         f"--work-folder.")
            stored = dict(record, runs=[])
        stored["runs"].append({
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": method,
            "foldmatch_version": __version__,
            "faiss_version": faiss.__version__,
            "faiss_threads": threads,
            "steps": {},
        })
        self.record = stored
        _write_json(run_file, stored)
        logging.info(f"Work folder: {self.path}")

    def step(self, name: str, **fields):
        """Record a step of this run in ``run.json``, with FAISS's thread count."""
        self.record["runs"][-1]["steps"][name] = dict(fields, faiss_threads=faiss.omp_get_max_threads())
        _write_json(self.file(_RUN_FILE), self.record)

    def save_greedy_state(self, labels: np.ndarray, next_start: int, n_searched: int, n_saturated: int,
                          n_representatives: int, seconds: float, done: bool):
        """Greedy's state, written to a temporary name, fsynced, then renamed over the last one."""
        _write_atomically(self.file(_GREEDY_STATE_FILE), lambda f: np.savez(
            f, labels=labels, next_start=np.int64(next_start), n_searched=np.int64(n_searched),
            n_saturated=np.int64(n_saturated), n_representatives=np.int64(n_representatives),
            seconds=np.float64(seconds), done=np.bool_(done)))

    def load_greedy_state(self, n_chains: int, dtype, rows: int):
        """The saved greedy state as a dict, or None without one. A state that does not fit this database (chain count,
        label type, a next block start that is not a block boundary, or labels that are not chain indexes) is
        refused (ValueError)."""
        path = self.file(_GREEDY_STATE_FILE)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            state = {key: data[key] for key in data.files}
        labels = state.get("labels")
        next_start = int(state.get("next_start", -1))
        if (labels is None or labels.shape != (n_chains,) or labels.dtype != np.dtype(dtype)
                or not (next_start == n_chains or (0 <= next_start < n_chains and next_start % rows == 0))
                or bool(np.any(labels[:next_start] < 0)) or bool(np.any(labels >= n_chains))):
            raise ValueError(f"The greedy state {path} does not fit this database ({n_chains} chains, blocks of {rows} "
                             f"rows): use a new --work-folder.")
        return {"labels": labels.copy(), "next_start": next_start, "n_searched": int(state["n_searched"]),
                "n_saturated": int(state["n_saturated"]), "n_representatives": int(state["n_representatives"]),
                "seconds": float(state["seconds"]), "done": bool(state["done"])}


class SimilarityEdges:
    """The kept edges of a similarity graph of ``n`` chains (:meth:`EmbeddingClusterer.build_similarity_edges`), in
    the blocks of the search: ``pair_blocks``, (m, 2) arrays of chain indexes, the lower end first (int32 when the
    largest index fits, else int64), and ``weight_blocks``, one float32 score a pair, in the same order. A block with
    no kept edge is not stored.

    ``n_edges`` counts the pairs and ``min_weight`` is the smallest weight (+inf with no edge). Under the rescue rule
    (:class:`~foldmatch.types.api_types.EdgeRule`), ``n_rescued_chains`` chains got ``n_rescued_edges`` of the edges
    from their own search (0 and 0 under the lower rule).

    Edges written to a work folder (``build_similarity_edges(work_folder=...)``) are read back by :meth:`from_folder`
    as one block of each file, memory-mapped read-only (``np.memmap``), so they are paged in as they are read."""

    def __init__(self, n: int, pair_blocks: list, weight_blocks: list, n_rescued_chains: int = 0,
                 n_rescued_edges: int = 0, min_weight: Optional[float] = None):
        self.n = n
        self.pair_blocks = pair_blocks
        self.weight_blocks = weight_blocks
        self.n_edges = sum(len(b) for b in pair_blocks)
        if min_weight is None:
            min_weight = min((float(b.min()) for b in weight_blocks if len(b)), default=float("inf"))
        self.min_weight = min_weight
        self.n_rescued_chains = n_rescued_chains
        self.n_rescued_edges = n_rescued_edges

    @classmethod
    def from_folder(cls, folder) -> "SimilarityEdges":
        """The edges a ``build_similarity_edges(work_folder=folder)`` wrote: ``graph.json`` (counts, smallest weight,
        id type) and the pairs and weights files, memory-mapped. Refused (ValueError) without ``graph.done``, or when
        a file's size does not match ``graph.json``."""
        folder = Path(folder)
        if not (folder / _GRAPH_DONE).exists():
            raise ValueError(f"No complete graph in {folder}: {_GRAPH_DONE} is missing.")
        record = json.loads((folder / _GRAPH_RECORD_FILE).read_text())
        n_edges = int(record["n_edges"])
        dtype = np.dtype(record["pairs_dtype"])
        pairs_path, weights_path = folder / _GRAPH_PAIRS_FILES[dtype], folder / _GRAPH_WEIGHTS_FILE
        sizes = (pairs_path.stat().st_size, weights_path.stat().st_size)
        if sizes != (n_edges * 2 * dtype.itemsize, n_edges * 4):
            raise ValueError(f"The graph files of {folder} ({sizes[0]} and {sizes[1]} bytes) do not hold the "
                             f"{n_edges} edges of its {_GRAPH_RECORD_FILE}: use a new --work-folder.")
        pair_blocks, weight_blocks = [], []
        if n_edges:
            pair_blocks.append(np.memmap(pairs_path, dtype=dtype, mode="r", shape=(n_edges, 2)))
            weight_blocks.append(np.memmap(weights_path, dtype=np.float32, mode="r", shape=(n_edges,)))
        min_weight = record["min_weight"]
        return cls(int(record["n"]), pair_blocks, weight_blocks, int(record["n_rescued_chains"]),
                   int(record["n_rescued_edges"]), float("inf") if min_weight is None else float(min_weight))

    def chunks(self, size: int = None):
        """Yield the edges in order as ``(pairs, weights)`` views of at most ``size`` edges (default
        _GRAPH_EDGE_CHUNK), block by block: a chunk never spans two blocks. The blocks are kept."""
        size = _GRAPH_EDGE_CHUNK if size is None else size
        for pairs, weights in zip(self.pair_blocks, self.weight_blocks):
            for s in range(0, len(pairs), size):
                yield pairs[s:s + size], weights[s:s + size]


def _igraph_from_edges(edges: SimilarityEdges, names, hold=None) -> ig.Graph:
    """The igraph graph of ``edges`` (consuming its blocks), weighted, its vertices named ``names``; ``hold(graph)``
    is called with the graph as soon as it has its edges (before the weights)."""
    # Create igraph from the blocks, with no concatenation, through a generator of int tuples: python-igraph 1.0.0
    # converts an edge ndarray through Python objects as well, at about 170 bytes an edge at its peak against 50 to
    # 57. Each block is released once consumed, and each weight block once its floats are in the list. An error
    # raised while igraph pulls the edges (MemoryError, KeyboardInterrupt) surfaces from python-igraph 1.0.0 as a
    # SystemError, with the original as its __cause__.
    graph = ig.Graph(n=edges.n, edges=_edge_tuples(edges.pair_blocks), directed=False)
    if hold is not None:
        hold(graph)
    # glibc keeps freed blocks in the heap: give them back before the weights and before Leiden.
    _release_freed_memory()
    # The smallest kept weight (+inf with no edge) was taken before the weight blocks are consumed: Leiden refuses
    # negative weights
    graph.es['weight'] = _weight_list(edges.weight_blocks)
    _release_freed_memory()

    # Add chain IDs as vertex attribute
    graph.vs['name'] = names
    return graph


def _leiden_clusters(graph: ig.Graph, resolution: float, n_iterations: int, seed: Optional[int]) -> np.ndarray:
    """python-igraph's Leiden on ``graph`` (see :meth:`EmbeddingClusterer.cluster_leiden`), cluster ids renumbered by
    decreasing size, ties by the smallest chain index."""
    # Higher edge weights (higher similarity) encourage nodes to be in same cluster. The objective must be exactly
    # "modularity": python-igraph normalises the resolution only for that spelling. A private generator: Python's
    # global random state is left alone.
    rng = random.Random(seed)
    ig.set_random_number_generator(rng)
    try:
        partition = graph.community_leiden(
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
    n = graph.vcount()
    membership = np.fromiter(partition.membership, dtype=_graph_id_dtype(n), count=n)
    del partition
    return _by_size(membership)


def _by_size(membership: np.ndarray) -> np.ndarray:
    """Cluster ids 0..C-1 (every one used) renumbered by decreasing cluster size, ties by the smallest chain index in
    the cluster (0 is the largest cluster), in ``membership``'s dtype."""
    n = len(membership)
    dtype = membership.dtype
    sizes = np.bincount(membership)
    # Every cluster has a member, so minimum.at overwrites the fill
    first = np.full(len(sizes), n - 1, dtype=dtype)
    np.minimum.at(first, membership, np.arange(n, dtype=dtype))
    order = np.lexsort((first, -sizes))
    new_id = np.empty(len(sizes), dtype=dtype)
    new_id[order] = np.arange(len(sizes), dtype=dtype)
    return new_id[membership]


class _NprobeNotApplicable(ValueError):
    """nprobe given for a database that is not ivf_pq (the CLI reports it as a usage error, after the load)."""


class EmbeddingClusterer:
    """Cluster protein structure embeddings: Leiden on the similarity graph (:meth:`cluster_leiden`), greedy
    representatives in one streamed pass over the database (:meth:`cluster_greedy`), or greedy representatives then
    Leiden on the representatives' graph (:meth:`cluster_greedy_leiden`)."""

    def __init__(self, db_path: Optional[str] = None, use_gpu: bool = False, nprobe: Optional[int] = None,
                 database: Optional[FaissEmbeddingDatabase] = None):
        """
        Load the database to cluster: from ``db_path``, or a ``database`` already loaded. The index is loaded on the
        CPU, its nprobe set, and only then moved to the GPU when ``use_gpu`` (the IVF parameters of a GPU index are
        not reachable through extract_index_ivf).

        Args:
            db_path: Path to the FAISS database (``<folder>/<index name>``)
            use_gpu: Move the index to the GPU (if available) once nprobe is set
            nprobe: IVF cells probed per query, at least 1, ivf_pq databases only (default: the value the database
                stored at build). The one source of nprobe of this clusterer's searches; ``self.nprobe`` is the
                value the index searches at (None on flat and hnsw). FAISS probes at most every cell;
                ``self.effective_nprobe`` is the number of cells probed, min(nprobe, nlist).
            database: A loaded FaissEmbeddingDatabase on the CPU, instead of ``db_path``

        Raises:
            ValueError: both or neither of ``db_path`` and ``database``; ``database`` not loaded, or on the GPU;
                ``nprobe`` below 1 (before the load); ``nprobe`` given for a flat or hnsw database (after the load)
        """
        if (db_path is None) == (database is None):
            raise ValueError("Give exactly one of db_path and database.")
        if nprobe is not None:
            nprobe = operator.index(nprobe)
            if nprobe < 1:
                raise ValueError("--nprobe must be at least 1.")
        if database is None:
            db_folder, index_name, db_path = _parse_db_path(db_path)
            database = FaissEmbeddingDatabase(db_folder, index_name)
            database.load_database(use_gpu=False)
        elif database.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")
        elif database.is_gpu_index:
            raise ValueError("The database must be on the CPU: use_gpu=True moves it to the GPU once nprobe is set.")
        self.db = database

        self.nprobe = None
        # The cells a search probes (FAISS probes at most nlist), which a work folder's run.json records and compares
        self.effective_nprobe = None
        if self.db.index_type == IndexType.ivf_pq:
            if nprobe is not None:
                self.db.set_nprobe(nprobe)
            ivf_index = faiss.extract_index_ivf(self.db.index)
            self.nprobe = int(ivf_index.nprobe)
            probed = min(self.nprobe, ivf_index.nlist)
            self.effective_nprobe = probed
            logging.info(f"IVF cells probed per query: {probed} of {ivf_index.nlist}"
                         + (f" (nprobe {self.nprobe} is above the cell count)" if self.nprobe > probed else ""))
        elif nprobe is not None:
            index_type = IndexType(self.db.index_type)
            kind = "built with --index-type auto (flat or hnsw)" if index_type is IndexType.auto else index_type.value
            raise _NprobeNotApplicable(f"--nprobe applies to ivf_pq databases only; this database is {kind}.")
        if use_gpu:
            self.db.move_to_gpu()

        self.chain_ids = self.db.chain_ids

        self.graph = None
        # The edges of the last build_similarity_edges (a SimilarityEdges), until a graph is built from them
        self.edges = None
        self.clusters = None
        # The smallest kept edge weight of the last build_similarity_edges or build_similarity_graph, and a weak
        # reference to the graph it was recorded for (cluster_leiden refuses a negative one for that graph only)
        self.min_weight = None
        self._min_weight_graph = None
        # The method of the last clustering ("leiden", "greedy" or "greedy-leiden"); under greedy each cluster's
        # representative, under greedy-leiden each chain's (greedy) representative, both as chain indexes
        self.method = None
        self.representatives = None
        self.chain_representatives = None
        # The modularity stored by cluster_greedy_leiden (on the representatives' graph), and that graph's counts
        self.modularity = None
        self.graph_stats = None

    def build_similarity_edges(
            self,
            threshold: float = 0.8,
            max_neighbors: int = 1000,
            edge_rule: EdgeRule = EdgeRule.rescue,
            work_folder=None
    ) -> SimilarityEdges:
        """
        Find the edges of the similarity graph with FAISS k-NN search and threshold filtering, with no igraph graph.

        Sets ``self.edges`` and ``self.min_weight``; a graph built before is dropped (``self.graph`` is None).
        :meth:`cluster_leiden` builds the igraph graph from ``self.edges`` when it needs one.

        With ``work_folder``, the edges are appended to files there block by block as they are found, not kept in
        memory: ``graph.pairs.i32`` (``graph.pairs.i64`` above 2**31 chains; two ids an edge) and
        ``graph.weights.f32`` (one float32 an edge), then ``graph.json`` (the counts and the smallest weight) and the
        ``graph.done`` marker once the search ends. Files of an earlier build there are replaced (a build that stopped
        leaves no ``graph.done``). The edges returned are those files, memory-mapped
        (:meth:`SimilarityEdges.from_folder`).

        Args:
            threshold: Similarity threshold for edge creation (cosine; identical chains score 1, or close to 1 on
                ivf_pq, where scores are PQ estimates)
            max_neighbors: Maximum number of neighbors to consider per node
            edge_rule: Which hits become edges (:class:`~foldmatch.types.api_types.EdgeRule`). ``lower``: an edge
                (i, j), i < j, is kept from the search of the lower chain i only. ``rescue`` (default): the same
                edges, plus, for each chain i that has none of them, an edge (j, i) to each of its own hits j at or
                above the threshold, weighted by i's score for j; every such j is lower than i.
            work_folder: A folder for the edge files (above); None keeps the edges in memory

        Returns:
            The edges, a :class:`SimilarityEdges`
        """
        edge_rule = EdgeRule(edge_rule)
        self.min_weight = None
        self._min_weight_graph = None
        self.edges = None
        if self.db.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")
        self.graph = None

        logging.info(f"\nBuilding similarity graph with threshold >= {threshold}")
        logging.info(f"Searching up to {max_neighbors} neighbors per chain")

        # Clamp max_neighbors to actual database size
        k = min(max_neighbors, len(self.chain_ids))
        self.edges = self._search_edges(threshold, k, edge_rule, work_folder)
        self.min_weight = self.edges.min_weight
        return self.edges

    def _search_edges(self, threshold: float, k: int, edge_rule: EdgeRule, work_folder=None) -> SimilarityEdges:
        """The block loop of :meth:`build_similarity_edges`, at ``k`` neighbours a chain (already clamped); with
        ``work_folder``, each block's edges are appended to its edge files instead of kept."""
        if work_folder is None:
            return self._search_edge_blocks(threshold, k, edge_rule, None)
        folder = Path(work_folder)
        folder.mkdir(parents=True, exist_ok=True)
        # A restart replaces the files: no done marker or record is left from a build that stopped
        for name in (_GRAPH_DONE, _GRAPH_RECORD_FILE):
            (folder / name).unlink(missing_ok=True)
        id_dtype = np.dtype(_graph_id_dtype(len(self.chain_ids)))
        with open(folder / _GRAPH_PAIRS_FILES[id_dtype], "wb") as pairs_file, \
                open(folder / _GRAPH_WEIGHTS_FILE, "wb") as weights_file:
            def write(pairs, weights):
                pairs_file.write(np.ascontiguousarray(pairs, dtype=id_dtype).tobytes())
                weights_file.write(np.ascontiguousarray(weights, dtype=np.float32).tobytes())

            edges = self._search_edge_blocks(threshold, k, edge_rule, write)
            for f in (pairs_file, weights_file):
                f.flush()
                os.fsync(f.fileno())
        _write_json(folder / _GRAPH_RECORD_FILE, {
            "n": edges.n,
            "n_edges": edges.n_edges,
            "n_rescued_chains": edges.n_rescued_chains,
            "n_rescued_edges": edges.n_rescued_edges,
            "min_weight": None if edges.n_edges == 0 else edges.min_weight,
            "pairs_dtype": id_dtype.name,
            "threshold": threshold,
            "k": k,
            "edge_rule": EdgeRule(edge_rule).value,
        })
        _touch_marker(folder / _GRAPH_DONE)
        return SimilarityEdges.from_folder(folder)

    def _search_edge_blocks(self, threshold: float, k: int, edge_rule: EdgeRule, write) -> SimilarityEdges:
        """The block loop: each block's kept pairs and weights are kept, or handed to ``write(pairs, weights)`` when
        it is given (the edges returned then hold no block, only the counts and the smallest weight)."""
        n_chains = len(self.chain_ids)
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
        rescue = edge_rule is EdgeRule.rescue
        # Under rescue, whether each chain has an edge yet (1 byte a chain)
        has_edge = np.zeros(n_chains, dtype=bool) if rescue else None
        pair_blocks = []
        weight_blocks = []
        n_edges = 0
        n_rescued_chains = 0
        n_rescued_edges = 0
        min_weight = float("inf")

        # Blocks of chains in ascending order, read with reconstruct_n (no direct map needed on ivf_pq) and searched
        # with one FAISS call each. An edge (i, j) is kept from the search of the lower chain i, which also drops
        # self-hits and the -1 ids FAISS fills in; these edges come in chain order, then in FAISS rank order. A block's
        # kept pairs are stored with int32 ids when the database's largest id fits, and a block with no kept edge is
        # not stored.
        #
        # Rescue: once a block's own edges are marked on both ends, a chain of the block with no edge has none for
        # good, since every lower block is done and a higher chain's search gives no edge to a lower chain. Its own
        # hits j at or above the threshold (all lower: a higher one would be an edge of its own) become edges (j, i),
        # weighted by its score for j, stored after the block's other edges, in chain order, then in rank order. None
        # repeats an edge: had j's search kept i, i would have an edge.
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
                pairs = weights = None
                if len(block_rows):
                    pairs = np.stack([block_rows + start, indices[block_rows, block_cols]],
                                     axis=1).astype(id_dtype, copy=False)
                    weights = scores[block_rows, block_cols]
                if rescue:
                    if pairs is not None:
                        has_edge[pairs[:, 0]] = True
                        has_edge[pairs[:, 1]] = True
                    lone = np.flatnonzero(~has_edge[start:end])
                    lone_ids, lone_scores = indices[lone], scores[lone]
                    lone_rows, lone_cols = np.nonzero((lone_ids >= 0) & (lone_ids < chains[lone]) &
                                                      (lone_scores >= threshold))
                    if len(lone_rows):
                        m = 0 if pairs is None else len(pairs)
                        both = np.empty((m + len(lone_rows), 2), dtype=id_dtype)
                        both_weights = np.empty(m + len(lone_rows), dtype=scores.dtype)
                        if m:
                            both[:m] = pairs
                            both_weights[:m] = weights
                        both[m:, 0] = lone_ids[lone_rows, lone_cols]
                        both[m:, 1] = lone[lone_rows] + start
                        both_weights[m:] = lone_scores[lone_rows, lone_cols]
                        has_edge[both[m:, 0]] = True
                        has_edge[both[m:, 1]] = True
                        n_rescued_chains += len(np.unique(lone_rows))
                        n_rescued_edges += len(lone_rows)
                        pairs, weights = both, both_weights
                        del both, both_weights
                    del lone, lone_ids, lone_scores, lone_rows, lone_cols
                if pairs is not None:
                    if write is None:
                        pair_blocks.append(pairs)
                        weight_blocks.append(weights)
                    else:
                        write(pairs, weights)
                        min_weight = min(min_weight, float(weights.min()))
                    n_edges += len(pairs)
                del scores, indices, chains, block_rows, block_cols, pairs, weights

                progress.set_postfix(edges=n_edges, refresh=False)
                progress.update(end - start)

        logging.info(f"Created graph with {n_chains} nodes and {n_edges} edges")
        if rescue:
            logging.info(f"Rescued {n_rescued_chains} chains with no edge from a lower chain's search "
                         f"({n_rescued_edges} edges)")
        if write is not None:
            edges = SimilarityEdges(n_chains, [], [], n_rescued_chains, n_rescued_edges, min_weight)
            edges.n_edges = n_edges
            return edges
        return SimilarityEdges(n_chains, pair_blocks, weight_blocks, n_rescued_chains, n_rescued_edges)

    def build_similarity_graph(
            self,
            threshold: float = 0.8,
            max_neighbors: int = 1000,
            edge_rule: EdgeRule = EdgeRule.rescue
    ) -> ig.Graph:
        """
        Build a similarity graph using FAISS k-NN search with threshold filtering: :meth:`build_similarity_edges`,
        then the igraph graph of its edges. The edge blocks are released as the graph takes them (``self.edges`` is
        None afterwards).

        Args:
            threshold: Similarity threshold for edge creation (cosine; identical chains score 1, or close to 1 on
                ivf_pq, where scores are PQ estimates)
            max_neighbors: Maximum number of neighbors to consider per node
            edge_rule: Which hits become edges, as in :meth:`build_similarity_edges` (default: rescue)

        Returns:
            igraph Graph object with edges weighted by similarity
        """
        start_time = time.perf_counter()
        self.build_similarity_edges(threshold=threshold, max_neighbors=max_neighbors, edge_rule=edge_rule)
        self._graph_from_edges()
        logging.info(f"Similarity graph built in {time.perf_counter() - start_time:.2f}s")
        return self.graph

    def _graph_from_edges(self):
        """The igraph graph of ``self.edges``, as ``self.graph``; the edge blocks are consumed and ``self.edges`` is
        set to None."""
        edges = self.edges

        def hold(graph):
            self.graph = graph

        _igraph_from_edges(edges, self.chain_ids, hold)
        self.min_weight = edges.min_weight
        self._min_weight_graph = weakref.ref(self.graph)
        self.edges = None

    def cluster_leiden(
            self,
            resolution: float = 1.0,
            n_iterations: int = _LEIDEN_ITERATIONS,
            seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply Leiden clustering algorithm to the similarity graph. After :meth:`build_similarity_edges`, the graph is
        built from ``self.edges`` first.

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
            ValueError: ``n_iterations`` below 1, ``seed`` below 0, no graph (nor edges), or a graph built by
                build_similarity_graph or from build_similarity_edges with a negative kept weight (modularity needs
                non-negative edge weights)
        """
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1 (Leiden runs a fixed number of iterations).")
        if seed is not None:
            seed = operator.index(seed)
            if seed < 0:
                raise ValueError("--seed must be at least 0.")
        if self.graph is None and self.edges is not None:
            self._graph_from_edges()
        if self.graph is None:
            raise ValueError("Graph not built. Call build_similarity_graph() first.")
        if self._min_weight_graph is not None and self._min_weight_graph() is self.graph and self.min_weight < 0:
            raise ValueError(f"Leiden needs non-negative edge weights: the graph's smallest weight is "
                             f"{self.min_weight:.4g} (edges keep scores at or above the threshold of "
                             f"build_similarity_graph).")

        logging.info(f"Running Leiden clustering (resolution={resolution})...")
        start_time = time.perf_counter()
        self.clusters = _leiden_clusters(self.graph, resolution, n_iterations, seed)
        sizes = np.bincount(self.clusters)

        logging.info(f"Found {len(sizes)} clusters")
        logging.info(f"Leiden clustering completed in {time.perf_counter() - start_time:.2f}s")
        self.method = "leiden"
        self.representatives = None

        return self.clusters

    def cluster_greedy(
            self,
            threshold: float = 0.8,
            max_neighbors: int = 1000,
            work_folder=None
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

        With ``work_folder``, the pass can be resumed: its state (each chain's representative so far, the next block
        and the counters) is written to ``greedy.state.npz`` there every ``_CHECKPOINT_SECONDS`` (900 s) of the pass
        and at its end, and a call with the same folder resumes from the last state written, or reuses a finished
        pass; the blocks are fixed by the chain count and k, so a resumed pass makes the same FAISS calls. The folder's
        ``run.json`` records the database (the sha256 of its files) and the settings; a folder recorded with another
        database, ``threshold``, ``max_neighbors`` or nprobe is refused. It is the work folder of
        :meth:`cluster_greedy_leiden`, which can reuse a pass made here at its ``greedy_threshold`` and
        ``greedy_max_neighbors``.

        Cluster ids are numbered by decreasing cluster size, ties by representative index (0 is the largest cluster);
        ``self.representatives[i]`` is the chain index of cluster i's representative.

        Args:
            threshold: A representative takes the chains in no cluster yet scoring at least this (cosine; identical
                chains score 1, or close to 1 on ivf_pq, where scores are PQ estimates)
            max_neighbors: Neighbours searched per representative, so also the cluster-size cap (this + 1)
            work_folder: A folder for the pass's state, so that it can be resumed (above); None keeps nothing

        Returns:
            Array of cluster IDs (one per chain)
        """
        if self.db.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")
        if max_neighbors < 1:
            raise ValueError("--max-neighbors must be at least 1.")
        work = None
        if work_folder is not None:
            work = _WorkFolder(work_folder)
            work.open(self._work_record(greedy=dict(threshold=float(threshold),
                                                    max_neighbors=operator.index(max_neighbors))), "greedy")

        start_time = time.perf_counter()
        labels, k, n_searched, n_saturated, n_representatives = self._greedy_pass(threshold, max_neighbors, work)
        n_chains = len(self.chain_ids)

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
        self.chain_representatives = None
        self.modularity = None
        self.k = k
        self.max_neighbors = max_neighbors
        self.n_searched = n_searched
        self.n_saturated = n_saturated
        self.n_representatives = n_representatives

        n_singletons = int(np.count_nonzero(sizes == 1))
        share = n_searched / n_chains if n_chains else 0.0
        logging.info(f"Searched {n_searched} of {n_chains} chains ({share:.2%})")
        logging.info(f"Found {n_representatives} clusters, {n_singletons} singletons")
        logging.info(f"Greedy clustering completed in {time.perf_counter() - start_time:.2f}s")

        return self.clusters

    def _work_record(self, greedy: dict, graph: Optional[dict] = None) -> dict:
        """The record of a work folder's ``run.json`` for this database: the parent's fingerprint, the effective
        nprobe (the cells probed: --nprobe above the cell count is the cell count) and the settings of the stored
        steps."""
        return {
            "format": _WORK_FOLDER_FORMAT,
            "foldmatch_version": __version__,
            "faiss_version": faiss.__version__,
            "parent": self.db._fingerprint(),
            "nprobe": self.effective_nprobe,
            "greedy": greedy,
            "graph": graph,
        }

    def _greedy_pass(self, threshold: float, max_neighbors: int, work: Optional[_WorkFolder], before_run=None):
        """Greedy's block loop (:meth:`cluster_greedy`), resumed from and checkpointed to ``work`` when given;
        ``before_run()`` is called before the loop when the pass is not reused whole from ``work``. Returns each
        chain's representative (a chain index), k, and the chains searched, saturated representatives and
        representatives."""
        n_chains = len(self.chain_ids)
        # Clamp max_neighbors to actual database size
        k = min(max_neighbors, n_chains)
        logging.info(f"\nGreedy clustering with threshold >= {threshold}")
        logging.info(f"Searching up to {k} neighbors per representative, only chains in no cluster yet")
        # Chains per block, as build_similarity_graph's
        rows = 1 if k < 1 else max(1, min(n_chains, _GRAPH_MAX_ROWS, _GRAPH_RESULT_BYTES // (k * 12)))

        # Each chain's representative (-1: in no cluster yet), int32 when the largest id fits
        labels = np.full(n_chains, -1, dtype=_graph_id_dtype(n_chains))
        first_start = 0
        n_searched = 0
        n_saturated = 0
        n_representatives = 0
        seconds = 0.0
        if work is not None:
            state = work.load_greedy_state(n_chains, labels.dtype, rows)
            if state is not None:
                labels, first_start = state["labels"], state["next_start"]
                n_searched, n_saturated = state["n_searched"], state["n_saturated"]
                n_representatives, seconds = state["n_representatives"], state["seconds"]
                if state["done"]:
                    logging.info(f"Greedy pass reused from the work folder ({n_representatives} representatives, "
                                 f"{n_searched} chains searched, in {seconds:.2f}s)")
                    work.step("greedy", skipped=True)
                    return labels, k, n_searched, n_saturated, n_representatives
                logging.info(f"Greedy pass resumed from the work folder at chain {first_start} of {n_chains} "
                             f"({n_representatives} representatives, {n_searched} chains searched so far)")
        if before_run is not None:
            before_run()

        # Get CPU index for reconstruction if using GPU
        index_to_use = self.db.index
        if self.db.is_gpu_index:
            logging.info("Moving GPU index to CPU for reconstruction...")
            index_to_use = faiss.index_gpu_to_cpu(self.db.index)

        # ivf_pq queries are normalised, as in build_similarity_graph
        normalize = self.db.index_type == IndexType.ivf_pq

        resumed_at = first_start
        run_start = last_save = _clock()
        searched_at_last_save = n_searched
        with tqdm(total=n_chains, initial=first_start, desc="Clustering chains", unit="chain") as progress:
            for start in range(first_start, n_chains, rows):
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

                if work is not None and end < n_chains:
                    now = _clock()
                    if now - last_save >= _CHECKPOINT_SECONDS:
                        work.save_greedy_state(labels, end, n_searched, n_saturated, n_representatives,
                                               seconds + now - run_start, done=False)
                        # Greedy is not stationary (early blocks search every chain, late ones few): the rate is in
                        # chains searched
                        rate = (n_searched - searched_at_last_save) / max(now - last_save, 1e-9)
                        logging.info(f"Greedy checkpoint at chain {end} of {n_chains}: {n_representatives} "
                                     f"representatives, {n_searched} chains searched, {rate:.1f} searched chains a "
                                     f"second since the last checkpoint")
                        last_save, searched_at_last_save = now, n_searched

        if work is not None:
            run_seconds = _clock() - run_start
            work.save_greedy_state(labels, n_chains, n_searched, n_saturated, n_representatives,
                                   seconds + run_seconds, done=True)
            work.step("greedy", resumed_at=resumed_at, seconds=round(run_seconds, 3))
        return labels, k, n_searched, n_saturated, n_representatives

    def cluster_greedy_leiden(
            self,
            work_folder,
            greedy_threshold: float = _GREEDY_LEIDEN_GREEDY_THRESHOLD,
            greedy_max_neighbors: int = _GREEDY_LEIDEN_GREEDY_MAX_NEIGHBORS,
            threshold: float = 0.8,
            max_neighbors: int = _GREEDY_LEIDEN_MAX_NEIGHBORS,
            edge_rule: EdgeRule = EdgeRule.rescue,
            resolution: float = _GREEDY_LEIDEN_RESOLUTION,
            n_iterations: int = _LEIDEN_ITERATIONS,
            seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Greedy representatives, then Leiden on the representatives' similarity graph; every chain takes its
        representative's cluster. For collections whose direct graph does not fit in memory.

        Steps, each stored in ``work_folder`` (resumed or reused by a call with the same folder):

        1. Greedy on this database at ``greedy_threshold`` and ``greedy_max_neighbors`` (:meth:`cluster_greedy`,
           resumable). The log then gives the representatives R, the bound of k x R edges of their graph and Leiden's
           memory at that bound.
        2. The representatives' database, ``<work_folder>/representatives``, copied from this one
           (:meth:`FaissEmbeddingDatabase.subset`: on ivf_pq the representatives' PQ codes, on flat their vectors),
           searched at this clusterer's nprobe, then a ``representatives.done`` marker.
        3. The representatives' graph at ``threshold``, ``max_neighbors`` and ``edge_rule``
           (:meth:`build_similarity_edges`), its edges in files of the work folder, then ``graph.done``.
        4. This database's and the representatives' FAISS indexes are released (``self.db.index`` is None
           afterwards; the chain ids are kept), then python-igraph's Leiden on the representatives' graph, as
           :meth:`cluster_leiden` (a graph with no edge is not passed to Leiden: every representative is its own
           cluster).
        5. Each chain takes its representative's cluster; cluster ids are numbered over the chains by decreasing size,
           ties by the smallest chain index (0 is the largest cluster).

        Steps 4 and 5 are never stored, so a call with another ``resolution``, ``n_iterations`` or ``seed`` reuses the
        searches. The folder's ``run.json`` records this database (the sha256 of its files), the nprobe and the
        settings of steps 1-3; a folder recorded with other ones is refused.

        Sets ``self.clusters`` (one per chain), ``self.chain_representatives`` (each chain's representative, a chain
        index), ``self.representatives`` None, ``self.method`` "greedy-leiden" and ``self.modularity`` (of the
        representatives' graph with their clusters; None with no edge).

        Args:
            work_folder: The folder of the steps' files (required)
            greedy_threshold: A representative takes the chains in no cluster yet scoring at least this (default: 0.9)
            greedy_max_neighbors: Neighbours a representative takes from, so also the greedy cluster-size cap
                (default: 1000)
            threshold: The edge threshold of the representatives' graph (default: 0.8)
            max_neighbors: The k of the representatives' graph (default: 100)
            edge_rule: The edge rule of the representatives' graph (default: rescue)
            resolution: Leiden's resolution (default: 20.0, chosen on TrEMBL samples of 1M and 10M chains)
            n_iterations: Leiden iterations, at least 1 (default: 10)
            seed: Leiden's random seed, at least 0

        Returns:
            Array of cluster IDs (one per chain)

        Raises:
            ValueError: an hnsw database (an ``auto`` one that is hnsw included), a database on the GPU, an argument
                out of range, a work folder recorded with other settings or of another layout, or a representatives'
                graph with a negative kept weight
        """
        if work_folder is None:
            raise ValueError("cluster_greedy_leiden needs a work_folder.")
        if greedy_max_neighbors < 1:
            raise ValueError("--greedy-max-neighbors must be at least 1.")
        if max_neighbors < 1:
            raise ValueError("--max-neighbors must be at least 1.")
        if n_iterations < 1:
            raise ValueError("n_iterations must be at least 1 (Leiden runs a fixed number of iterations).")
        if seed is not None:
            seed = operator.index(seed)
            if seed < 0:
                raise ValueError("--seed must be at least 0.")
        if self.db.index is None:
            raise ValueError("Database not loaded. Call load_database() first.")
        # Judged on the loaded index: an auto database stores "auto", whichever index it built
        index_type = IndexType(self.db.index_type)
        if index_type is IndexType.hnsw or (index_type is IndexType.auto and not self.db.is_gpu_index and isinstance(
                faiss.downcast_index(self.db.index), faiss.IndexHNSW)):
            raise _MethodNotApplicable("--method greedy-leiden needs an ivf_pq or flat database; this one is hnsw. Use "
                                       "--method leiden, or build the database with --index-type ivf_pq or flat.")
        if self.db.is_gpu_index:
            raise ValueError("cluster_greedy_leiden copies the representatives' database from the CPU index: load the "
                             "database without use_gpu.")
        edge_rule = EdgeRule(edge_rule)
        work = _WorkFolder(work_folder)
        work.open(self._work_record(
            greedy=dict(threshold=float(greedy_threshold), max_neighbors=operator.index(greedy_max_neighbors)),
            graph=dict(threshold=float(threshold), max_neighbors=operator.index(max_neighbors),
                       edge_rule=edge_rule.value)), "greedy-leiden")
        start_time = time.perf_counter()
        n_chains = len(self.chain_ids)

        # 1. Greedy (a pass that runs here, not reused, invalidates the steps after it)
        labels, greedy_k, n_searched, n_saturated, n_reps = self._greedy_pass(
            greedy_threshold, greedy_max_neighbors, work, before_run=lambda: work.remove(_REPRESENTATIVES_DONE,
                                                                                         _GRAPH_DONE))
        is_rep = labels == np.arange(n_chains, dtype=labels.dtype)
        rep_rows = np.flatnonzero(is_rep)
        # Each chain's representative as a rank among the representatives
        rep_rank = (np.cumsum(is_rep, dtype=labels.dtype) - 1)[labels]
        greedy_sizes = np.bincount(rep_rank)
        del is_rep
        k = min(max_neighbors, n_reps)
        bound = k * n_reps
        logging.info(f"Representatives: {n_reps} of {n_chains} chains ({n_reps / max(n_chains, 1):.2%}); their graph "
                     f"at --max-neighbors {max_neighbors} has at most {bound} edges")
        logging.info(f"Leiden memory at that bound: igraph about {_IGRAPH_BYTES_PER_EDGE} B an edge, "
                     f"{bound * _IGRAPH_BYTES_PER_EDGE / 2 ** 30:.2f} GiB; NetworKit ParallelLeiden about "
                     f"{_NETWORKIT_BYTES_PER_EDGE} B an edge and {_NETWORKIT_BYTES_PER_NODE} B a node, "
                     f"{(bound * _NETWORKIT_BYTES_PER_EDGE + n_reps * _NETWORKIT_BYTES_PER_NODE) / 2 ** 30:.2f} GiB")

        # 2. The representatives' database, and 3. their graph
        rep_ids = [self.chain_ids[r] for r in rep_rows.tolist()]
        if work.has(_GRAPH_DONE) and work.has(_REPRESENTATIVES_DONE):
            edges = SimilarityEdges.from_folder(work.path)
            if edges.n != n_reps:
                raise ValueError(f"The graph of the work folder {work.path} has {edges.n} nodes for {n_reps} "
                                 f"representatives: use a new --work-folder.")
            logging.info(f"Representatives' graph reused from the work folder ({edges.n_edges} edges)")
            work.step("representatives", skipped=True)
            work.step("graph", skipped=True)
        else:
            work.remove(_GRAPH_DONE)
            reps_db = None
            if work.has(_REPRESENTATIVES_DONE):
                reps_db = FaissEmbeddingDatabase(work.path, _REPRESENTATIVES_NAME)
                reps_db.load_database()
                if reps_db.chain_ids != rep_ids:
                    raise ValueError(f"The representatives' database of the work folder {work.path} does not hold "
                                     f"this run's representatives: use a new --work-folder.")
                logging.info("Representatives' database reused from the work folder")
                work.step("representatives", skipped=True)
            else:
                step_start = time.perf_counter()
                reps_db = self.db.subset(rep_rows, work.path, _REPRESENTATIVES_NAME, overwrite=True, parent_record={
                    "greedy_threshold": float(greedy_threshold),
                    "greedy_max_neighbors": operator.index(greedy_max_neighbors), "nprobe": self.nprobe})
                _touch_marker(work.file(_REPRESENTATIVES_DONE))
                work.step("representatives", seconds=round(time.perf_counter() - step_start, 3))
            step_start = time.perf_counter()
            reps = EmbeddingClusterer(database=reps_db, nprobe=self.nprobe)
            edges = reps.build_similarity_edges(threshold=threshold, max_neighbors=max_neighbors,
                                                edge_rule=edge_rule, work_folder=work.path)
            work.step("graph", seconds=round(time.perf_counter() - step_start, 3))
            # Only the edges are kept
            reps.db.index = None
            reps.edges = None
            del reps, reps_db

        # 4. Leiden, with both indexes released
        self.db.index = None
        self.graph = None
        self.edges = None
        _release_freed_memory()
        work.record["runs"][-1]["leiden"] = {"resolution": float(resolution), "n_iterations": int(n_iterations),
                                             "seed": seed}
        _write_json(work.file(_RUN_FILE), work.record)
        n_graph_edges = edges.n_edges
        self.graph_stats = {"n_nodes": edges.n, "n_edges": n_graph_edges,
                            "n_rescued_chains": edges.n_rescued_chains, "n_rescued_edges": edges.n_rescued_edges}
        self.modularity = None
        if n_graph_edges and edges.min_weight < 0:
            raise _NegativeEdgeWeights(f"Leiden needs non-negative edge weights: the representatives' graph's "
                                       f"smallest weight is {edges.min_weight:.4g} (edges keep scores at or above "
                                       f"the threshold).", edges.min_weight)
        if n_graph_edges == 0:
            logging.info("The representatives' graph has no edge: every representative is its own cluster")
            rep_clusters = np.arange(n_reps, dtype=_graph_id_dtype(n_reps))
        else:
            graph = _igraph_from_edges(edges, rep_ids)
            del edges
            logging.info(f"Running Leiden clustering on the representatives' graph (resolution={resolution})...")
            leiden_start = time.perf_counter()
            rep_clusters = _leiden_clusters(graph, resolution, n_iterations, seed)
            self.modularity = graph.modularity(rep_clusters.tolist(), weights="weight")
            logging.info(f"Leiden clustering completed in {time.perf_counter() - leiden_start:.2f}s")
            del graph
            _release_freed_memory()
        del rep_ids

        # 5. Every chain takes its representative's cluster, renumbered over the chains
        self.clusters = _by_size(rep_clusters[rep_rank].astype(_graph_id_dtype(n_chains), copy=False))
        self.chain_representatives = labels
        self.representatives = None
        self.method = "greedy-leiden"
        self.k = greedy_k
        self.max_neighbors = greedy_max_neighbors
        self.n_searched = n_searched
        self.n_saturated = n_saturated
        self.n_representatives = n_reps
        self.n_at_cap = int(np.count_nonzero(greedy_sizes >= greedy_max_neighbors))
        del rep_rank, rep_clusters, greedy_sizes
        logging.info(f"Found {int(self.clusters.max()) + 1 if n_chains else 0} clusters of {n_chains} chains")
        logging.info(f"Greedy-Leiden clustering completed in {time.perf_counter() - start_time:.2f}s")
        return self.clusters

    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about the clustering results.

        Returns:
            Dictionary with clustering statistics
        """
        if self.clusters is None:
            raise ValueError("Clustering not performed. Call cluster_leiden() or cluster_greedy() first.")

        # Memberships are 0..C-1 (Leiden, greedy and greedy-leiden)
        size_array = np.bincount(self.clusters)
        sizes = size_array.tolist()
        n_clusters = len(sizes)
        n_singletons = sum(1 for s in sizes if s == 1)
        n_chains = len(self.clusters)
        # Giants: more than 1.745% of the chains, in integers
        n_giants = int(np.count_nonzero(size_array.astype(np.int64) * 100_000 > _GIANT_PER_100K * n_chains))
        largest = np.sort(size_array)[::-1][:_LARGEST_CLUSTERS].tolist()

        # Calculate modularity if graph exists and the partition is not greedy's, which is not computed on a graph;
        # greedy-leiden's is the one stored on the representatives' graph
        modularity = None
        if self.method == "greedy-leiden":
            modularity = self.modularity
        elif self.graph is not None and self.method != "greedy":
            modularity = self.graph.modularity(self.clusters, weights='weight')

        stats = {
            'n_chains': len(self.clusters),
            'n_clusters': n_clusters,
            'n_singletons': n_singletons,
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes),
            'mean_cluster_size': np.mean(sizes),
            'median_cluster_size': np.median(sizes),
            'modularity': modularity,
            # The largest cluster's chains over all chains; the clusters with more than 1.745% of the chains; the
            # _LARGEST_CLUSTERS largest sizes, largest first
            'largest_cluster_share': largest[0] / n_chains,
            'n_giant_clusters': n_giants,
            'largest_cluster_sizes': largest,
        }
        if self.method == "greedy":
            stats.update({
                'method': self.method,
                'n_representatives': self.n_representatives,
                'n_searched': self.n_searched,
                'n_saturated': self.n_saturated,
                'n_at_cap': sum(1 for s in sizes if s >= self.max_neighbors),
            })
        elif self.method == "greedy-leiden":
            # The greedy step's counts (its clusters at the cap among the greedy clusters), then the representatives'
            # graph
            stats.update({
                'method': self.method,
                'n_representatives': self.n_representatives,
                'n_searched': self.n_searched,
                'n_saturated': self.n_saturated,
                'n_at_cap': self.n_at_cap,
                'graph_nodes': self.graph_stats['n_nodes'],
                'graph_edges': self.graph_stats['n_edges'],
                'n_rescued_chains': self.graph_stats['n_rescued_chains'],
                'n_rescued_edges': self.graph_stats['n_rescued_edges'],
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
        used by the search commands; after :meth:`cluster_greedy` and
        :meth:`cluster_greedy_leiden`,
        :data:`~foldmatch.search.output.GREEDY_CLUSTER_OUTPUT_FIELDS`, which add
        the representative's chain id (under greedy-leiden the chain's greedy
        representative; the cluster is the final one).

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
        greedy_leiden = self.method == "greedy-leiden"

        def _assignments():
            # Rows made _EXPORT_CHUNK chains at a time, not as one list of every chain
            for s in range(0, len(self.clusters), _EXPORT_CHUNK):
                cluster_ids = np.asarray(self.clusters[s:s + _EXPORT_CHUNK])
                sizes = cluster_sizes[cluster_ids]
                columns = [self.chain_ids[s:s + _EXPORT_CHUNK], cluster_ids.tolist(), sizes.tolist()]
                if greedy:
                    # Each cluster's representative, by cluster id
                    columns.append([self.chain_ids[r] for r in self.representatives[cluster_ids].tolist()])
                elif greedy_leiden:
                    # Each chain's own representative, by chain
                    columns.append([self.chain_ids[r] for r in
                                    np.asarray(self.chain_representatives[s:s + _EXPORT_CHUNK]).tolist()])
                chunk = zip(*columns)
                if min_cluster_size is not None:
                    chunk = (row for row, keep in zip(chunk, (sizes >= min_cluster_size).tolist()) if keep)
                yield from chunk

        exported = write_cluster_results(_assignments(), str(output_path), format_mode,
                                         GREEDY_CLUSTER_OUTPUT_FIELDS if greedy or greedy_leiden
                                         else CLUSTER_OUTPUT_FIELDS)
        logging.info(f"Cluster assignments exported to {output_file}")

        if min_cluster_size:
            total_count = len(self.clusters)
            logging.info(f"Exported {exported}/{total_count} chains (min_cluster_size={min_cluster_size})")

    def print_statistics(self):
        """Print clustering statistics to console. Under greedy-leiden: the greedy step, then the representatives'
        graph, then the final partition over chains (P9.CL D2)."""
        stats = self.get_cluster_statistics()
        method = stats.get('method')
        logging.info("CLUSTERING STATISTICS")
        logging.info(f"Total embeddings:    {stats['n_chains']}")
        if method == "greedy-leiden":
            self._print_greedy_statistics(stats)
            logging.info(f"Representatives' graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges "
                         f"({stats['n_rescued_chains']} chains rescued with {stats['n_rescued_edges']} edges)")
        logging.info(f"Number of clusters:  {stats['n_clusters']}")
        logging.info(f"Singleton clusters:  {stats['n_singletons']}")
        logging.info(f"Cluster size (min):  {stats['min_cluster_size']}")
        logging.info(f"Cluster size (max):  {stats['max_cluster_size']}")
        logging.info(f"Cluster size (mean): {stats['mean_cluster_size']:.2f}")
        logging.info(f"Cluster size (med):  {stats['median_cluster_size']:.1f}")
        # The largest clusters and the giants, under every method
        n_chains = stats['n_chains']
        giant_share = f"{_GIANT_PER_100K / 1000:g}%"
        logging.info(f"Largest cluster:     {stats['largest_cluster_sizes'][0]} chains "
                     f"({stats['largest_cluster_share']:.3%} of the chains)")
        logging.info(f"Giant clusters (more than {giant_share} of the chains): {stats['n_giant_clusters']}")
        logging.info(f"Largest {len(stats['largest_cluster_sizes'])} clusters: "
                     + ", ".join(f"{s} ({s / n_chains:.3%})" for s in stats['largest_cluster_sizes']))
        if stats['modularity'] is not None:
            if method == "greedy-leiden":
                logging.info(f"Modularity (representatives' graph): {stats['modularity']:.4f}")
            else:
                logging.info(f"Modularity:          {stats['modularity']:.4f}")
        if method == "greedy":
            self._print_greedy_statistics(stats)
        # Giants are a WARNING under greedy-leiden only, on sets as large as those its resolution default was measured
        # on: on direct graphs every measured default run has giants, and a higher resolution there costs domain labels
        n_giants = stats['n_giant_clusters']
        if method == "greedy-leiden" and n_chains >= _GIANT_WARN_MIN_CHAINS and n_giants:
            held = "clusters each hold" if n_giants > 1 else "cluster holds"
            logging.warning(f"{n_giants} {held} more than {giant_share} of the chains; a higher --resolution splits "
                            f"large clusters (measured on TrEMBL samples of 1M-10M chains).")

    @staticmethod
    def _print_greedy_statistics(stats: dict):
        """The greedy step's lines of :meth:`print_statistics`, with the greedy step's options under each method."""
        method = stats['method']
        if method == "greedy":
            k_flag, threshold_flag, clusters = "--max-neighbors", "--threshold", "clusters"
        else:
            k_flag, threshold_flag, clusters = "--greedy-max-neighbors", "--greedy-threshold", "greedy clusters"
        n_chains, n_reps = stats['n_chains'], stats['n_representatives']
        saturated_share = stats['n_saturated'] / n_reps
        logging.info(f"Method:              {method}")
        if method == "greedy-leiden":
            logging.info(f"Representatives:     {n_reps} of {n_chains} ({n_reps / n_chains:.2%})")
        logging.info(f"Chains searched:     {stats['n_searched']} of {n_chains} "
                     f"({stats['n_searched'] / n_chains:.2%})")
        logging.info(f"Representatives with all {k_flag} neighbours at or above {threshold_flag}: "
                     f"{stats['n_saturated']} of {n_reps} ({saturated_share:.2%}); their {clusters} may be cut, "
                     f"and a larger {k_flag} can change them.")
        logging.info(f"{clusters[0].upper()}{clusters[1:]} of at least {k_flag} chains: {stats['n_at_cap']}")
        if saturated_share > 0.5:
            logging.warning(f"Most {clusters} are bounded by {k_flag}, not {threshold_flag}: a representative or "
                            f"singleton may score at or above {threshold_flag} with an earlier representative.")
