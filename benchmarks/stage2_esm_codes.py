#!/usr/bin/env python
"""esm-codes Stage-2 benchmark and profiling gate (not a test): plan section 7 P3 acceptance (e) and section 8 P7.

Data (all real; md5-checked):
  chains     ``L2m_B8_s0.chains.npz`` (md5 5519866044ab0f348c74b9319125acd2): the 559 chains of the full-chain
             evaluation set, target codes (and chain query encodings for the research queries of ``kernel``);
  chain seqs ``chains_seq.tsv`` (md5 7996b1728759fe8424366843f95d8961): their sequences (id, sequence, ...), for
             BLOSUM62, the round-2 metrics and the sequence store; every length equals the codes';
  queries    ``queries.L2m_B8_s0.npz`` (md5 e821497f59dda06a3899c18dce37286c): the 1,465 real benchmark queries (ESM3
             query encodings q fp16 and sequences; 21-966 aa, median 71).
Candidates: the research throughput draw ``np.random.default_rng(20260925).integers(0, 559, 10000)``; candidate k
is ``<chain>#<k>`` (a database of 10,000 entries, 2,218,273 residues); one query gets the first K, query j of a
batch the slice ``[(617 j) mod (10000 - K), + K)`` (research bench_batch). Fixed-length queries follow the research
rule on the real queries: the shortest real query with length >= Lq, cut to its first Lq residues; Lq = 1000 (> 966)
is the longest real query + the first residues of the next longest. N (the E-value database size) is
``--db-residues`` (default 15,889,271, the residues of the research E-value benchmark's 50,000 targets), so the
E-value filter keeps the real survivor fraction of these pairs.

FoldMatch runs the production Stage 2 (``code_alignment.align_candidates_esm_codes``) on a real sequence store and
residue-code store built in ``--work``, with the registry bundle's panel and statistics (HF cache), default output
fields; 1 core = the serial in-process path (BLAS limited to one thread), pin it with ``--core`` / taskset.
BLOSUM62 is ``alignment.align_candidates`` with the CLI defaults on the same target sequences.

Modes (each writes ``--out`` JSON and prints a summary):
  kernel        P3 gate: kernel-level fwd+rev vs the research production path (unchanged from P3).
  components    1 core: per-stage split of the worker work, ns per DP cell: query prep, similarity (decode +
                per-pair GEMM), rowcol z, DP (score only, S_f + S_r), DP with traceback flags, traceback (the probe
                library ``fmsw_probe.c``: the shipped kernel source compiled with the shipped flags plus clocks;
                checked bit for bit against the shipped library), path metrics and CIGAR (parent).
  grid          1 core, Lq x K grid: Stage 2 at max_evalue 1e-3 and inf (the panel on, default fields), the panel
                alone, rounds 1+2 without the panel (= total - panel alone; also measured with panel_policy never),
                BLOSUM62, survivors; plus one profiled 1e-3 run per cell.
  panel         panel pass, 1 core, at the real 71-aa and 966-aa queries: FoldMatch (aligner, 1 candidate; and
                kernel-level over the panel codes) vs the research path (``--research-src``); and the panel wall
                time of one 300-aa query with 1 vs ``--workers`` workers (warm persistent pool; cold reported).
  perquery      Stage 2 per query at K = 100: serial and a warm pool of each ``--workers``; BLOSUM62 likewise.
  node          ``--nq`` real queries x K on a persistent pool of ``--workers``: 1e-3, inf and no panel; pairs/s;
                a profiled 1e-3 run; ``--serial-nq`` of them serially for the per-core rate (Stage B).
  sweep         block_floats and task_cells sweeps on ``--nq`` x K; every configuration's output hashed (S16: bit
                for bit the same).
  profile       the profiling gate: ``--nq`` x K with FOLDMATCH_STAGE2_PROFILE=1 for each ``--workers`` (0/1 =
                serial), plus profile-off runs (overhead, identical output) and the output writing time.
  rss           worker RSS at Lq = 1000, K = 1000, default block, ``--workers`` workers (profiled).
  replay        the real pipeline's Stage 2, as captured by ``benchmarks/stage2_capture.py`` around ``fm-search
                query sequences --scoring esm-codes`` (real Stage-1 candidates, so the real E-filter survivor
                fraction): cold / warm profiled calls and profile-off calls per ``--workers``, BLOSUM62 on the same
                candidates; the output signature is checked against the CLI's.
  research-batch the research Stage-2 driver (``codepath.align_candidates_x`` of the research throughput bench,
                with the research ``swfast`` / ``libswkern``) on the same ``--nq`` x K workload, or the research P64
                single-query workload (``--research-query chain:300 --nq 1 --k 10000``), at ``--workers``; modes
                s0 (score only), s10 (score + traceback of the top 10 %), tb (traceback all, vector identity).
                Run with ``--research-bench`` (codepath.py), ``--research-build`` (swfast.py + libswkern) and
                optionally ``--research-fm-src`` (the FoldMatch snapshot the research ran with; put first on
                sys.path). Never imports this tree's swkern.
  bandwidth     STREAM-like triad with ``--workers`` processes (DRAM-sized and L2-sized arrays), GB/s.

Examples (M1: through bin/heavy.sh, one at a time):
  python benchmarks/stage2_esm_codes.py grid --chains C --chain-seqs S --queries Q --work W --out grid.json
  python benchmarks/stage2_esm_codes.py profile --nq 64 --k 1000 --workers 1 8 ... --out profile.json
  python benchmarks/stage2_esm_codes.py node --nq 640 --k 1000 --workers 64 ... --out node.json

Gate of mode ``kernel`` (EPYC 7763, NERSC): foldmatch@2^17 >= 0.9 x research@2^20 at Lq=300, K=1,000 and >= 2,500
pairs/s; the P7 targets are in PLAN section 8 P7 and are checked from these JSONs.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __name__ in ("__main__", "__mp_main__"):     # as a script (and its spawn workers): BLAS on one thread
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[_v] = "1"
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CHAINS_MD5 = "5519866044ab0f348c74b9319125acd2"
CHAIN_SEQS_MD5 = "7996b1728759fe8424366843f95d8961"
QUERIES_MD5 = "e821497f59dda06a3899c18dce37286c"
RESEARCH_MD5 = {"swkern.c": "438d5a9811153f61e66cee845f1ed037", "swmulti.c": "b12fbc2c03638659440b12282a4d8670"}
WORKLOAD_SEED, POOL, DRAW = 20260925, 559, 10000
QUERY_SEED = 20261019
BENCH_N = 15_889_271
QUERIES = {100: [("1j2vA", 0, 100)], 300: [("4xrtA", 0, 300)], 600: [("3zukA", 0, 600)],
           1000: [("4j5tA", 0, None), ("3dwbA", 0, 212)]}
OFF, GO, GE = 1.0, 5.0, 0.5
ESM_REV = "636acb420daf674df37ab0220f56061a5de2f169"
PROFILE_ENV = "FOLDMATCH_STAGE2_PROFILE"          # foldmatch.search.stage2_profile.PROFILE_ENV
_vp = C.c_void_p


def _swkern():
    from foldmatch.search import swkern
    return swkern


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _check_md5(path: Path, want: str):
    got = md5(path)
    if got != want:
        raise SystemExit(f"{path}: md5 {got} differs from {want}")


def host_info(with_kernel: bool = True) -> dict:
    info = dict(host=socket.gethostname(), platform=platform.platform(), machine=platform.machine(),
                python=platform.python_version(), cpus=os.cpu_count(), numpy=np.__version__)
    if with_kernel:
        sk = _swkern()
        info["blas"] = sk.blas_info()
        info["kernel"] = sk.load().info()
    try:
        if sys.platform == "darwin":
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            aff = sorted(os.sched_getaffinity(0))
            info["affinity"] = dict(count=len(aff), first=aff[:8], last=aff[-4:])
    except Exception:
        pass
    for k in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(k):
            info[k.lower()] = os.environ[k]
    return info


def median_rounds(f, rounds, warmup=True):
    if warmup:
        f()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [round(t, 6) for t in ts]


def _timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


# =====================================================================================================================
# data (numpy only)
# =====================================================================================================================
class ChainSet:
    """The 559 chains: codes, chain query encodings, sequences."""

    def __init__(self, chains: Path, chain_seqs: Path = None):
        _check_md5(chains, CHAINS_MD5)
        z = np.load(chains)
        self.q, self.codes, self.offsets = z["q"], z["codes"], z["offsets"].astype(np.int64)
        self.ids = [str(x) for x in z["ids"]]
        self.index = {s: i for i, s in enumerate(self.ids)}
        if len(self.ids) != POOL:
            raise SystemExit(f"expected {POOL} chains, found {len(self.ids)}")
        self.lens = np.diff(self.offsets)
        self.seqs = None
        if chain_seqs is not None:
            _check_md5(chain_seqs, CHAIN_SEQS_MD5)
            seq = {}
            for line in open(chain_seqs):
                f = line.rstrip("\n").split("\t")
                seq.setdefault(f[0], f[1])
            self.seqs = [seq[c] for c in self.ids]
            bad = [c for c, s, m in zip(self.ids, self.seqs, self.lens) if len(s) != m]
            if bad:
                raise SystemExit(f"{len(bad)} chain sequences differ in length from their codes: {bad[:5]}")
        self.draw = np.random.default_rng(WORKLOAD_SEED).integers(0, POOL, DRAW)
        self.cand_ids = [f"{self.ids[c]}#{k}" for k, c in enumerate(self.draw.tolist())]

    def chain_codes(self, i: int) -> np.ndarray:
        return self.codes[self.offsets[i]:self.offsets[i + 1]]

    def research_query(self, Lq: int):
        """The research throughput query of length Lq (chain-set q16 and sequence)."""
        rows, seqs, names = [], [], []
        for cid, a, b in QUERIES[Lq]:
            i = self.index[cid]
            rows.append(self.q[self.offsets[i]:self.offsets[i + 1]][a:b])
            if self.seqs is not None:
                seqs.append(self.seqs[i][a:b])
            names.append(f"{cid}[{a}:{b if b is not None else ''}]")
        q16 = np.ascontiguousarray(np.concatenate(rows), np.float16)
        assert len(q16) == Lq, (Lq, len(q16))
        return (f"chain{Lq}", "".join(seqs) if seqs else None, q16, "+".join(names))

    def slice_ids(self, j: int, K: int):
        start = 0 if K >= DRAW else (j * 617) % (DRAW - K)
        return self.cand_ids[start:start + K]


class RealQueries:
    """The 1,465 real benchmark queries (q fp16, sequences)."""

    def __init__(self, path: Path):
        _check_md5(path, QUERIES_MD5)
        z = np.load(path)
        self.ids = [str(x) for x in z["ids"]]
        self.off = z["offsets"].astype(np.int64)
        self.lens = np.diff(self.off)
        self.q = z["q"]
        sc = z["seq_concat"]
        self.seqs = [sc[self.off[i]:self.off[i + 1]].tobytes().decode("ascii") for i in range(len(self.ids))]

    def get(self, i: int):
        return (self.ids[i], self.seqs[i], np.ascontiguousarray(self.q[self.off[i]:self.off[i + 1]]), self.ids[i])

    def exact(self, L: int):
        """The first real query of length exactly L."""
        idx = np.flatnonzero(self.lens == L)
        if not idx.size:
            raise SystemExit(f"no real query of length {L}")
        return self.get(int(idx[0]))

    def fixed(self, Lq: int):
        """Research rule on the real queries: the shortest with length >= Lq cut to Lq; longer than every query: the
        longest + the first residues of the next longest."""
        order = np.argsort(self.lens, kind="stable")
        ok = [int(i) for i in order if self.lens[i] >= Lq]
        if ok:
            i = ok[0]
            qid, seq, q16, _ = self.get(i)
            return (f"{qid}@1-{Lq}", seq[:Lq], np.ascontiguousarray(q16[:Lq]), f"{qid} (length {self.lens[i]}) 1-{Lq}")
        i1, i2 = int(order[-1]), int(order[-2])
        a, b = self.get(i1), self.get(i2)
        rest = Lq - len(a[1])
        return (f"{a[0]}+{b[0]}@{Lq}", a[1] + b[1][:rest], np.ascontiguousarray(np.concatenate([a[2], b[2][:rest]])),
                f"{a[0]} (all {len(a[1])}) + {b[0]} 1-{rest}")

    def sample(self, nq: int, seed: int = QUERY_SEED):
        perm = np.random.default_rng(seed).permutation(len(self.ids))
        if nq > len(perm):
            raise SystemExit(f"--nq {nq} > {len(perm)} real queries")
        return [self.get(int(i)) for i in perm[:nq]]


def make_queries(a, cs: ChainSet):
    """The query set of a mode from --query-set: 'real' (--nq sampled real queries), 'fixed:<Lq>' (one fixed-length
    real query), 'chain:<Lq>' (the research chain-set query)."""
    spec = a.query_set
    if spec == "real":
        return RealQueries(a.queries).sample(a.nq)
    kind, _, lq = spec.partition(":")
    if kind == "fixed":
        return [RealQueries(a.queries).fixed(int(lq))]
    if kind == "chain":
        return [cs.research_query(int(lq))]
    raise SystemExit(f"unknown --query-set {spec!r}")


def batch_prefilter(cs: ChainSet, queries, K: int):
    emb = [float(x) for x in np.linspace(0.99, 0.5, K)]
    return {q[0]: (cs.slice_ids(j, K), emb) for j, q in enumerate(queries)}


# =====================================================================================================================
# FoldMatch world: a real sequence store + residue-code store of the 10,000 candidates, and a query memmap
# =====================================================================================================================
class World:
    def __init__(self, work: Path, cs: ChainSet, queries, db_residues: int = BENCH_N, n_cand: int = DRAW):
        from foldmatch.search import scoring_system as ss
        from foldmatch.search.query_encodings import QueryEncodingEntry, QueryEncodingIndex
        from foldmatch.search.residue_code_store import ResidueCodeStore
        from foldmatch.search.sequence_store import SequenceStore
        from foldmatch.utils.sequence import seq_sha1
        if cs.seqs is None:
            raise SystemExit("--chain-seqs is required")
        self.system = ss.get(ss.DEFAULT_ESM_CODES)
        self.prepared = ss.prepare(self.system, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")))
        work.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix="world_", dir=work))
        ids = cs.cand_ids[:n_cand]
        idx = cs.draw[:n_cand].tolist()
        t0 = time.perf_counter()
        fasta = self.folder / "db.fasta"
        with open(fasta, "w") as f:
            for tid, i in zip(ids, idx):
                f.write(f">{tid}\n{cs.seqs[i]}\n")
        self.seq_store = SequenceStore(self.folder, "db")
        self.seq_store.create(fasta)
        self.store = ResidueCodeStore(self.folder, "db")
        self.store.create([(ids, [cs.chain_codes(i) for i in idx], [seq_sha1(cs.seqs[i]) for i in idx])],
                          self.system, esm_revision=ESM_REV)
        self.store.validate(self.seq_store, len(ids), preflight=None)
        self.build_s = time.perf_counter() - t0
        self.store_residues = int(self.store.manifest()["total_residues"])
        self.db_residues = int(db_residues)
        path = self.folder / "queries.f16"
        entries, rows = {}, 0
        with open(path, "wb") as f:
            for qid, seq, q16, _src in queries:
                f.write(np.ascontiguousarray(q16, "<f2").tobytes())
                entries[qid] = QueryEncodingEntry(rows, len(q16), seq_sha1(seq))
                rows += len(q16)
        self.query_index = QueryEncodingIndex(path, rows, 64, entries, self.system.name)
        self.query_sequences = {qid: seq for qid, seq, _q, _s in queries}
        self.queries = queries

    def close(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def align(self, prefilter, pool=None, *, max_evalue=1e-3, panel_policy="auto", fields=None, profile=False,
              **kw):
        """(results, stats, wall seconds) of one production esm-codes Stage-2 call."""
        from foldmatch.search import code_alignment as ca
        from foldmatch.search import output
        fields = list(output.DEFAULT_OUTPUT_FIELDS) if fields is None else fields
        stats = {}
        old = os.environ.get(PROFILE_ENV)
        os.environ[PROFILE_ENV] = "1" if profile else ""
        try:
            t0 = time.perf_counter()
            res = ca.align_candidates_esm_codes(
                self.query_index, prefilter, self.store, self.seq_store.fetch, self.prepared, self.db_residues, pool,
                query_sequences=self.query_sequences, max_evalue=max_evalue, panel_policy=panel_policy,
                output_needs=fields, stats=stats, **kw)
            wall = time.perf_counter() - t0
        finally:
            if old is None:
                os.environ.pop(PROFILE_ENV, None)
            else:
                os.environ[PROFILE_ENV] = old
        if profile and "profile" not in stats:
            raise SystemExit("this FoldMatch has no Stage-2 profiling mode (foldmatch.search.stage2_profile)")
        return res, stats, wall

    def blosum(self, prefilter, num_workers=1, fields=None):
        from foldmatch.search import alignment, output
        fields = list(output.DEFAULT_OUTPUT_FIELDS) if fields is None else fields
        t0 = time.perf_counter()
        res = alignment.align_candidates(self.query_sequences, prefilter, self.seq_store.fetch,
                                         num_workers=num_workers, subject_db_size=self.db_residues,
                                         output_fields=fields)
        return res, time.perf_counter() - t0


def signature(results, fields=None) -> str:
    """sha256 over every metrics field (floats as hex) and the rendered row of every hit, in order."""
    import dataclasses
    from foldmatch.search import output
    fields = list(output.SUPPORTED_OUTPUT_FIELDS) if fields is None else fields
    h = hashlib.sha256()
    for qid, hits in results.items():
        h.update(qid.encode() + b"\0")
        for hit in hits:
            vals = [float(v).hex() if isinstance(v, float) else repr(v)
                    for v in (getattr(hit.metrics, f.name) for f in dataclasses.fields(hit.metrics))]
            h.update(("\t".join([hit.subject_id, float(hit.emb_score).hex(), *vals]) + "\n").encode())
            h.update((output.format_row(qid, hit, fields) + "\n").encode())
    return h.hexdigest()


def _counts(stats):
    keys = ("queries", "candidate_pairs", "panel_pairs", "survivors", "hits", "round1_tasks", "round2_tasks",
            "workers", "batches", "skipped_pairs")
    return {k: stats.get(k) for k in keys}


def _profile_brief(p: dict) -> dict:
    """The profile summary without the per-task list."""
    return {k: v for k, v in p.items() if k != "tasks"}


# =====================================================================================================================
# research path (unmodified research kernels) for kernel / panel
# =====================================================================================================================
def research_flags() -> list:
    m = platform.machine().lower()
    if sys.platform == "darwin" and m in ("arm64", "aarch64"):
        arch = ["-mcpu=apple-m1"]
    elif m in ("x86_64", "amd64"):
        arch = ["-march=znver3"]
    else:
        arch = ["-mcpu=native"]
    return ["-O3", *arch, "-ffp-contract=off", "-shared", "-fPIC"]


def build_research(src: Path, work: Path, cc: str, cflags: list) -> Path:
    for name, want in RESEARCH_MD5.items():
        _check_md5(src / name, want)
    work.mkdir(parents=True, exist_ok=True)
    ext = "dylib" if sys.platform == "darwin" else "so"
    for name in ("swkern", "swmulti"):
        cmd = [*shlex.split(cc), *cflags, f"-I{src}", "-o", str(work / f"lib{name}.{ext}"), str(src / f"{name}.c"),
               "-lm"]
        subprocess.run(cmd, check=True)
    return work


def research_from_args(a):
    if a.research_lib_dir:
        return Research(a.research_lib_dir), dict(lib_dir=str(a.research_lib_dir))
    if not a.research_src:
        raise SystemExit("--research-src or --research-lib-dir is required")
    flags = shlex.split(a.research_cflags) if a.research_cflags else research_flags()
    R = Research(build_research(a.research_src, a.work / "research_lib", a.research_cc, flags))
    return R, dict(src=str(a.research_src), cc=a.research_cc, cflags=flags, sources_md5=RESEARCH_MD5,
                   cc_version=subprocess.run(shlex.split(a.research_cc) + ["--version"], capture_output=True,
                                             text=True).stdout.splitlines()[0])


class Research:
    _LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32) * 2 - 1

    def __init__(self, libdir: Path):
        ext = "dylib" if sys.platform == "darwin" else "so"
        self.k = C.CDLL(str(libdir / f"libswkern.{ext}"))
        self.m = C.CDLL(str(libdir / f"libswmulti.{ext}"))
        self.k.sw_ad_ws_floats.restype = C.c_size_t
        self.k.sw_ad_ws_floats.argtypes = [C.c_int, C.c_int]
        self.k.swkern_vector_lanes.restype = C.c_int
        self.m.swz_ad_multi_f32.restype = None
        self.m.swz_ad_multi_f32.argtypes = [_vp, _vp, C.c_int, C.c_int, _vp, C.c_int, C.c_int, _vp, _vp, _vp, _vp,
                                            _vp, _vp]
        self.R = (np.array([OFF, OFF], np.float32), np.array([GO, GO], np.float32), np.array([GE, GE], np.float32),
                  np.array([0, 1], np.int32))
        self.ws = np.empty(0, np.float32)
        self.libdir = libdir

    @staticmethod
    def query_prep(q16):
        q = np.asarray(q16, dtype=np.float32)
        return q / (8.0 * np.linalg.norm(q, axis=1, keepdims=True))

    @staticmethod
    def blocks(lengths, n, block_floats):
        """research codepath._blocks: a candidate opens a new block when the current one is non-empty and would
        exceed block_floats similarity floats."""
        out, k0, acc = [], 0, 0
        for k, m in enumerate(lengths):
            if k > k0 and acc + n * m > block_floats:
                out.append((k0, k))
                k0, acc = k, 0
            acc += n * int(m)
        out.append((k0, len(lengths)))
        return out

    def multi_fwd_rev(self, sim, simr, col_off, out2):
        n, ld = sim.shape
        K = len(col_off) - 1
        mmax = int(np.max(np.diff(col_off)))
        need = 2 * (n + mmax) + int(self.k.sw_ad_ws_floats(n, mmax))
        if self.ws.size < need:
            self.ws = np.empty(need, np.float32)
        o, g, e, r = self.R
        self.m.swz_ad_multi_f32(sim.ctypes.data, simr.ctypes.data, n, ld, col_off.ctypes.data, K, 2, o.ctypes.data,
                                g.ctypes.data, e.ctypes.data, r.ctypes.data, out2.ctypes.data, self.ws.ctypes.data)

    def runner(self, codes, offsets, q16, block_floats, reuse=False):
        lens = np.diff(offsets)
        K = len(lens)
        blk = self.blocks(lens.tolist(), len(q16), block_floats)
        cols = [np.ascontiguousarray(offsets[k0:k1 + 1] - offsets[k0]) for k0, k1 in blk]
        out = np.empty((2, K), np.float32)
        bufs = {}

        def run():
            qp = self.query_prep(q16)
            n = len(qp)
            for (k0, k1), col in zip(blk, cols):
                j0, j1 = int(offsets[k0]), int(offsets[k1])
                w = j1 - j0
                o2 = np.empty((2, k1 - k0), np.float32)
                if reuse:
                    for key, size in (("T", w * 64), ("sim", n * w), ("simr", n * w)):
                        if bufs.get(key) is None or bufs[key].size < size:
                            bufs[key] = np.empty(size, np.float32)
                    T = bufs["T"][:w * 64].reshape(w, 8, 8)
                    np.take(self._LUT, codes[j0:j1], axis=0, out=T, mode="clip")
                    sim = bufs["sim"][:n * w].reshape(n, w)
                    np.matmul(qp, T.reshape(w, 64).T, out=sim)
                    simr = bufs["simr"][:n * w].reshape(n, w)
                    np.copyto(simr, sim[::-1])
                else:
                    T = np.unpackbits(codes[j0:j1], axis=1)[:, :64].astype(np.float32) * 2 - 1
                    sim = qp @ T.T
                    simr = np.ascontiguousarray(sim[::-1])
                self.multi_fwd_rev(sim, simr, col, o2)
                out[:, k0:k1] = o2
            return out
        return run


def foldmatch_runner(codes, offsets, q16, block_floats):
    sk = _swkern()
    k = sk.load()
    ws = sk.default_workspace()

    def run():
        qp = sk.query_prep(q16, k)
        return sk.score_fwd_rev(qp, codes, offsets, sk.SETTING_D, block_floats=block_floats, kernel=k, workspace=ws)
    return run


def kernel_workload(cs: ChainSet, K: int):
    cand = cs.draw[:K]
    parts = [cs.chain_codes(int(c)) for c in cand]
    codes = np.ascontiguousarray(np.concatenate(parts), np.uint8)
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
    return codes, offsets


# =====================================================================================================================
# mode: kernel (P3)
# =====================================================================================================================
def kernel_mode(a) -> dict:
    sk = _swkern()
    cs = ChainSet(a.chains)
    codes, offsets = kernel_workload(cs, a.k)
    K, residues = len(offsets) - 1, int(offsets[-1])
    R, research_build = research_from_args(a)
    res = dict(mode="kernel", host=host_info(), research=research_build,
               research_vector_lanes=int(R.k.swkern_vector_lanes()), K=K, residues=residues,
               workload=dict(chains=str(a.chains), seed=WORKLOAD_SEED, queries={str(k): v for k, v in QUERIES.items()}),
               rounds=a.rounds, results={})
    for Lq in a.lq:
        q16 = cs.research_query(Lq)[2]
        r = {}
        fm = foldmatch_runner(codes, offsets, q16, 1 << 17)()
        rs = R.runner(codes, offsets, q16, 1 << 20)()
        bound = sk.s16_score_bound(rs)
        got = np.stack([fm.s_fwd, fm.s_rev])
        r["s16"] = dict(max_ratio=float(np.max(np.abs(got - rs) / bound)),
                        bit_equal_fraction=float(np.mean(got.view(np.int32) == rs.view(np.int32))))
        variants = [(f"foldmatch.bf2^{e}", foldmatch_runner(codes, offsets, q16, 1 << e)) for e in a.blocks]
        for e in a.research_blocks:
            variants.append((f"research.bf2^{e}", R.runner(codes, offsets, q16, 1 << e)))
            variants.append((f"research.reuse.bf2^{e}", R.runner(codes, offsets, q16, 1 << e, reuse=True)))
        for name, f in variants:
            m, ts = median_rounds(f, a.rounds)
            r[name] = dict(s_med=m, s_rounds=ts, pairs_per_s=K / m, ns_per_cell=1e9 * m / (Lq * residues))
            print(f"Lq={Lq:5d} K={K} {name:28s} {K / m:10.1f} pairs/s  {1e9 * m / (Lq * residues):6.3f} ns/cell",
                  flush=True)
        fm_def = r["foldmatch.bf2^17"]["pairs_per_s"] if "foldmatch.bf2^17" in r else None
        if fm_def and "research.bf2^20" in r:
            r["ratio.foldmatch_bf2^17_over_research_bf2^20"] = fm_def / r["research.bf2^20"]["pairs_per_s"]
        best_res = max(v["pairs_per_s"] for k, v in r.items() if k.startswith("research"))
        best_fm = max(v["pairs_per_s"] for k, v in r.items() if k.startswith("foldmatch"))
        r["ratio.best_foldmatch_over_best_research"] = best_fm / best_res
        print(f"Lq={Lq} " + ", ".join(f"{k} {v:.3f}" for k, v in r.items() if k.startswith("ratio"))
              + f"; S16 max ratio {r['s16']['max_ratio']:.3g}", flush=True)
        res["results"][str(Lq)] = r
    return res


# =====================================================================================================================
# mode: components (probe)
# =====================================================================================================================
def build_probe(work: Path, cc=None) -> Path:
    """The probe library for this machine's kernel ISA: benchmarks/fmsw_probe.c (#includes the shipped fmsw.c) with
    the shipped build's exact flags for that ISA."""
    from foldmatch._native import SOURCE_DIR, library_filename
    from foldmatch._native import build as nb
    sk = _swkern()
    isa = sk.load().isa
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"probe_{library_filename(isa)}"
    cmd = nb.compile_command(isa, out, cc=cc)
    # the shipped command ends with "-o OUT SRC/fmsw.c -lm": compile the probe instead, with the source dir on -I
    i = cmd.index("-o")
    cmd = cmd[:i] + [f"-I{SOURCE_DIR}", "-o", str(out), str(HERE / "fmsw_probe.c"), "-lm"]
    subprocess.run(cmd, check=True)
    return out


class Probe:
    def __init__(self, path: Path):
        lib = C.CDLL(str(path))
        lib.probe_score_fwd_rev.restype = C.c_int
        lib.probe_score_fwd_rev.argtypes = [_vp, C.c_int, _vp, _vp, C.c_int, C.c_float, C.c_float, C.c_float,
                                            _vp, _vp, _vp, C.c_size_t, _vp]
        lib.probe_align_tb.restype = C.c_int
        lib.probe_align_tb.argtypes = [_vp, C.c_int, _vp, _vp, C.c_int, C.c_float, C.c_float, C.c_float, _vp, _vp,
                                       _vp, C.c_size_t, _vp, _vp, C.c_size_t, _vp, C.c_size_t, _vp]
        self.lib = lib


def components_mode(a) -> dict:
    """1 core: the worker's stages for one query x K candidates, timed block by block exactly as swkern runs them
    (same blocks, same per-pair GEMMs), with the kernel call replaced by the probe."""
    sk = _swkern()
    from foldmatch.search import code_alignment as ca
    cs = ChainSet(a.chains, a.chain_seqs)
    rq = RealQueries(a.queries)
    probe = Probe(build_probe(a.work, a.cc))
    kern = sk.load()
    params = sk.SETTING_D
    codes, offsets = kernel_workload(cs, a.k)
    lens = np.diff(offsets)
    starts = offsets[:-1]
    t_seqs = [cs.seqs[int(c)] for c in cs.draw[:a.k]]
    res = dict(mode="components", host=host_info(), K=a.k, residues=int(offsets[-1]), rounds=a.rounds,
               block_floats=a.block_floats, results={})
    for Lq in a.lq:
        qid, qseq, q16, src = rq.fixed(Lq)
        cells = Lq * int(offsets[-1])
        ws = sk.Workspace()
        qp = sk.query_prep(q16, kern)
        skipped = np.zeros(len(lens), bool)
        blocks = sk.plan_blocks(Lq, lens.tolist(), skipped.tolist(), a.block_floats)
        ref1 = sk.score_fwd_rev(qp, codes, offsets, params, block_floats=a.block_floats, kernel=kern)
        ref2 = sk.align(qp, codes, offsets, params, block_floats=a.block_floats, kernel=kern)

        def one_round():
            t = dict(prep=0.0, sim=0.0, rowcol_r1=0.0, dp_r1=0.0, rowcol_r2=0.0, dp_tb_r2=0.0, traceback_r2=0.0,
                     metrics=0.0, cigar=0.0)
            t0 = time.perf_counter()
            qp_ = sk.query_prep(q16, kern)
            t["prep"] += time.perf_counter() - t0
            s_f = np.full(len(lens), np.nan, np.float32)
            s_r = np.full(len(lens), np.nan, np.float32)
            sc2 = np.full(len(lens), np.nan, np.float32)
            ends = np.zeros((len(lens), 2), np.int32)
            paths, poffs = [], []
            for k0, k1 in blocks:
                t0 = time.perf_counter()
                b = sk._Block(kern, ws, qp_, codes, starts, lens, k0, k1, False)
                t["sim"] += time.perf_counter() - t0
                mmax = int(b.m.max())
                kw = ws.kernel_floats(kern.ws_floats(Lq, mmax))
                tt = np.zeros(3, np.float64)
                st = probe.lib.probe_score_fwd_rev(b.sim.ctypes.data, Lq, b.pair_off.ctypes.data, b.m.ctypes.data,
                                                   b.K, params.offset, params.gap_open, params.gap_extend,
                                                   s_f[k0:].ctypes.data, s_r[k0:].ctypes.data, kw.ctypes.data,
                                                   kw.size, tt.ctypes.data)
                assert st == 0, st
                t["rowcol_r1"] += tt[0]
                t["dp_r1"] += tt[1]
                fl = ws.flags(kern.flag_bytes(Lq, mmax))
                cap = int(b.K * Lq + int(b.m.sum()))
                pbuf = np.empty(2 * max(cap, 1), np.int32)
                poff = np.empty(b.K + 1, np.int64)
                tt[:] = 0
                st = probe.lib.probe_align_tb(b.sim.ctypes.data, Lq, b.pair_off.ctypes.data, b.m.ctypes.data, b.K,
                                              params.offset, params.gap_open, params.gap_extend,
                                              sc2[k0:].ctypes.data, ends[k0:].ctypes.data, pbuf.ctypes.data, cap,
                                              poff.ctypes.data, kw.ctypes.data, kw.size, fl.ctypes.data, fl.size,
                                              tt.ctypes.data)
                assert st == 0, st
                t["rowcol_r2"] += tt[0]
                t["dp_tb_r2"] += tt[1]
                t["traceback_r2"] += tt[2]
                paths.append(pbuf[:2 * int(poff[-1])].reshape(-1, 2))
                poffs.append(np.diff(poff))
            path = np.concatenate(paths)
            path_off = np.zeros(len(lens) + 1, np.int64)
            np.cumsum(np.concatenate(poffs), out=path_off[1:])
            t0 = time.perf_counter()
            q_bytes = np.frombuffer(ca.sanitized_bytes(qseq), np.uint8)
            t_clean = [ca.sanitized_bytes(s) for s in t_seqs]
            t_starts = np.zeros(len(t_clean), np.int64)
            np.cumsum(lens[:-1], out=t_starts[1:])
            ca.path_columns(path, path_off, q_bytes, np.frombuffer(b"".join(t_clean), np.uint8), t_starts)
            t["metrics"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for k in range(len(lens)):
                sk.cigar_from_path(path[path_off[k]:path_off[k + 1]])
            t["cigar"] += time.perf_counter() - t0
            return t, s_f, s_r, sc2, ends, path, path_off

        t, s_f, s_r, sc2, ends, path, path_off = one_round()
        same = (np.array_equal(s_f.view(np.uint32), ref1.s_fwd.view(np.uint32))
                and np.array_equal(s_r.view(np.uint32), ref1.s_rev.view(np.uint32))
                and np.array_equal(sc2.view(np.uint32), ref2.scores.view(np.uint32))
                and np.array_equal(ends, ref2.ends) and np.array_equal(path, ref2.path)
                and np.array_equal(path_off, ref2.path_off))
        rounds = [one_round()[0] for _ in range(a.rounds)]
        med = {k: float(np.median([r[k] for r in rounds])) for k in rounds[0]}
        # production wrappers for comparison: the whole score_fwd_rev / align calls
        m1, _ = median_rounds(lambda: sk.score_fwd_rev(sk.query_prep(q16, kern), codes, offsets, params,
                                                       block_floats=a.block_floats, kernel=kern, workspace=ws),
                              a.rounds)
        m2, _ = median_rounds(lambda: sk.align(sk.query_prep(q16, kern), codes, offsets, params,
                                               block_floats=a.block_floats, kernel=kern, workspace=ws), a.rounds)
        ns = {k: 1e9 * v / cells for k, v in med.items()}
        r = dict(query=src, Lq=Lq, cells=cells, probe_equals_shipped=bool(same), seconds=med, ns_per_cell=ns,
                 score_fwd_rev_call_s=m1, align_call_s=m2, score_fwd_rev_ns_per_cell=1e9 * m1 / cells,
                 align_ns_per_cell=1e9 * m2 / cells, blocks=len(blocks))
        res["results"][str(Lq)] = r
        print(f"Lq={Lq} K={a.k}: probe == shipped {same}; ns/cell " + ", ".join(f"{k} {v:.3f}" for k, v in ns.items())
              + f"; score_fwd_rev call {1e9 * m1 / cells:.3f}, align call {1e9 * m2 / cells:.3f}", flush=True)
    return res


# =====================================================================================================================
# mode: grid (1 core)
# =====================================================================================================================
def grid_mode(a) -> dict:
    from foldmatch.search import code_alignment as ca
    cs = ChainSet(a.chains, a.chain_seqs)
    rq = RealQueries(a.queries)
    queries = [rq.fixed(Lq) for Lq in a.lq]
    world = World(a.work, cs, queries, a.db_residues)
    res = dict(mode="grid", host=host_info(), db_residues=world.db_residues, store_residues=world.store_residues,
               rounds=a.rounds, blosum_rounds=a.blosum_rounds, results=[])
    serial = ca.CodeAlignmentPool(1)       # one in-process worker state for the run, as the CLI's serial Stage 2
    try:
        for (qid, seq, q16, src), Lq in zip(queries, a.lq):
            for K in a.ks:
                ids = cs.cand_ids[:K]
                emb = [float(x) for x in np.linspace(0.99, 0.5, K)]
                pf = {qid: (ids, emb)}
                pf1 = {qid: (ids[:1], emb[:1])}
                cell = dict(Lq=Lq, K=K, query=src, residues=int(sum(cs.lens[cs.draw[:K]])))
                runs = {}
                rounds = a.rounds if K < a.big_k else a.big_k_rounds
                cell["rounds"] = rounds
                for name, kw, p in (("e1e-3", dict(max_evalue=1e-3), pf), ("inf", dict(max_evalue=None), pf),
                                    ("nopanel_inf", dict(max_evalue=None, panel_policy="never"), pf),
                                    ("panel_only", dict(max_evalue=1e-3), pf1)):
                    walls, stats = [], None
                    if K < a.big_k or p is pf1:
                        world.align(p, serial, **kw)                               # warm-up
                    for _ in range(rounds if p is not pf1 else a.rounds):
                        _r, stats, w = world.align(p, serial, **kw)
                        walls.append(w)
                    runs[name] = dict(wall=float(np.median(walls)), walls=walls, counts=_counts(stats))
                # profiled runs: rounds 1+2 without the panel = wall - the panel's worker seconds (1 core, in-process)
                world.align(pf1, serial, max_evalue=1e-3, profile=True)                # profiled worker state
                prof_runs = {}
                for name, me in (("e1e-3", 1e-3), ("inf", None)):
                    excl, walls, last = [], [], None
                    for _ in range(rounds):
                        _r, pst, pw = world.align(pf, serial, max_evalue=me, profile=True)
                        last = _profile_brief(pst["profile"])
                        excl.append(pw - last["workers"].get("panel", {}).get("seconds", 0.0))
                        walls.append(pw)
                    prof_runs[name] = dict(wall=float(np.median(walls)), walls=walls,
                                           excl_panel=float(np.median(excl)), excl_panel_rounds=excl,
                                           parent=last["parent"], workers=last["workers"])
                cell["profiled"] = prof_runs
                br = a.blosum_rounds if K < a.big_k else 1
                bw = []
                for _ in range(br):
                    bres, w = world.blosum(pf, 1)
                    bw.append(w)
                runs["blosum62"] = dict(wall=float(np.median(bw)), walls=bw,
                                        hits=sum(len(v) for v in bres.values()))
                panel = runs["panel_only"]["wall"]
                b = runs["blosum62"]["wall"]
                ex3, exi = prof_runs["e1e-3"]["excl_panel"], prof_runs["inf"]["excl_panel"]
                sub3, subi = runs["e1e-3"]["wall"] - panel, runs["inf"]["wall"] - panel
                cell.update(runs=runs, panel_alone=panel,
                            panel_worker_s=prof_runs["e1e-3"]["workers"].get("panel", {}).get("seconds"),
                            excl_panel_e1e3=ex3, excl_panel_inf=exi,
                            excl_panel_e1e3_by_subtraction=sub3, excl_panel_inf_by_subtraction=subi,
                            survivor_fraction_e1e3=runs["e1e-3"]["counts"]["survivors"] / K,
                            ratio_blosum_over_code_excl_panel_e1e3=b / ex3,
                            ratio_blosum_over_code_excl_panel_inf=b / exi,
                            ratio_blosum_over_code_nopanel_inf=b / runs["nopanel_inf"]["wall"],
                            ratio_blosum_over_code_total_e1e3=b / runs["e1e-3"]["wall"],
                            ratio_blosum_over_code_total_inf=b / runs["inf"]["wall"],
                            pairs_per_s=dict(code_total_e1e3=K / runs["e1e-3"]["wall"],
                                             code_excl_panel_e1e3=K / ex3, code_excl_panel_inf=K / exi,
                                             code_nopanel_inf=K / runs["nopanel_inf"]["wall"], blosum62=K / b))
                res["results"].append(cell)
                print(f"Lq={Lq:5d} K={K:6d}: e1e-3 {runs['e1e-3']['wall']:.3f}s (survivors "
                      f"{runs['e1e-3']['counts']['survivors']}), inf {runs['inf']['wall']:.3f}s, no-panel inf "
                      f"{runs['nopanel_inf']['wall']:.3f}s, panel alone {panel:.3f}s, BLOSUM62 {b:.3f}s -> "
                      f"BLOSUM62 / code (excl. panel): 1e-3 {cell['ratio_blosum_over_code_excl_panel_e1e3']:.2f}x, "
                      f"inf {cell['ratio_blosum_over_code_excl_panel_inf']:.2f}x (no-panel run "
                      f"{cell['ratio_blosum_over_code_nopanel_inf']:.2f}x); with the panel: 1e-3 "
                      f"{cell['ratio_blosum_over_code_total_e1e3']:.2f}x, inf "
                      f"{cell['ratio_blosum_over_code_total_inf']:.2f}x", flush=True)
                _dump_partial(a, res)
    finally:
        serial.close()
        world.close()
    return res


def _dump_partial(a, res):
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".partial.json").write_text(json.dumps(res, indent=1, default=str))


# =====================================================================================================================
# mode: panel
# =====================================================================================================================
def panel_mode(a) -> dict:
    sk = _swkern()
    from foldmatch.search import code_alignment as ca
    cs = ChainSet(a.chains, a.chain_seqs)
    rq = RealQueries(a.queries)
    queries = [rq.exact(L) for L in a.panel_lengths] + [rq.fixed(a.pool_query_length)]
    world = World(a.work, cs, queries, a.db_residues)
    R, research_build = research_from_args(a)
    spec = world.prepared.for_worker()
    pcodes, poff = spec.load_panel()
    pcodes = np.ascontiguousarray(pcodes)
    poff = np.asarray(poff, np.int64)
    res = dict(mode="panel", host=host_info(), research=research_build, panel_members=len(poff) - 1,
               panel_residues=int(poff[-1]), rounds=a.rounds, results={}, pool={})
    serial = ca.CodeAlignmentPool(1)
    try:
        for qid, seq, q16, src in queries[:len(a.panel_lengths)]:
            L = len(seq)
            pf1 = {qid: (cs.cand_ids[:1], [0.99])}
            m_al, ts_al = median_rounds(lambda: world.align(pf1, serial, max_evalue=1e-3), a.rounds)
            m_fm, ts_fm = median_rounds(foldmatch_runner(pcodes, poff, q16, 1 << 17), a.rounds)
            m_rs, ts_rs = median_rounds(R.runner(pcodes, poff, q16, 1 << 20), a.rounds)
            m_rs17, _ = median_rounds(R.runner(pcodes, poff, q16, 1 << 17), a.rounds)
            fm = foldmatch_runner(pcodes, poff, q16, 1 << 17)()
            rs = R.runner(pcodes, poff, q16, 1 << 20)()
            got = np.stack([fm.s_fwd, fm.s_rev])
            s16 = float(np.max(np.abs(got - rs) / sk.s16_score_bound(rs)))
            world.align(pf1, serial, max_evalue=1e-3, profile=True)
            _r, pst, _w = world.align(pf1, serial, max_evalue=1e-3, profile=True)
            pw = pst["profile"]["workers"].get("panel", {})
            r = dict(query=src, length=L, aligner_panel_plus_1_s=m_al, aligner_rounds=ts_al, kernel_level_s=m_fm,
                     kernel_rounds=ts_fm, research_bf2_20_s=m_rs, research_rounds=ts_rs, research_bf2_17_s=m_rs17,
                     ratio_aligner_over_research=m_al / m_rs, ratio_kernel_over_research=m_fm / m_rs,
                     s16_max_ratio=s16, profile_panel=pw)
            res["results"][str(L)] = r
            print(f"panel pass L={L}: aligner (panel + 1 candidate) {m_al:.4f}s, kernel-level {m_fm:.4f}s, research "
                  f"{m_rs:.4f}s (2^17 {m_rs17:.4f}s) -> aligner / research {m_al / m_rs:.3f}, kernel / research "
                  f"{m_fm / m_rs:.3f}; S16 {s16:.3g}", flush=True)
        # one 300-aa query, panel wall time: 1 worker (serial) vs warm pools
        qid, seq, q16, src = queries[-1]
        pf1 = {qid: (cs.cand_ids[:1], [0.99])}
        m1, ts1 = median_rounds(lambda: world.align(pf1, serial, max_evalue=1e-3), a.rounds)
        res["pool"]["query"] = src
        res["pool"]["serial"] = dict(wall=m1, walls=ts1)
        for W in a.workers:
            if W <= 1:
                continue
            with ca.CodeAlignmentPool(W) as pool:
                _r, st0, cold = world.align(pf1, pool, max_evalue=1e-3)
                mW, tsW = median_rounds(lambda: world.align(pf1, pool, max_evalue=1e-3), a.rounds, warmup=False)
                _r, pst, pwall = world.align(pf1, pool, max_evalue=1e-3, profile=True)
            prof = _profile_brief(pst["profile"])
            res["pool"][str(W)] = dict(cold_first_call=cold, warm=mW, warm_walls=tsW, speedup_warm=m1 / mW,
                                       speedup_cold=m1 / cold, tasks=st0.get("round1_tasks"),
                                       workers=st0.get("workers"), profile=prof)
            print(f"panel wall, one {len(seq)}-aa query: serial {m1:.4f}s; {W} workers warm {mW:.4f}s "
                  f"({m1 / mW:.2f}x), cold first call {cold:.4f}s ({m1 / cold:.2f}x); {st0.get('round1_tasks')} "
                  f"round-1 tasks", flush=True)
    finally:
        serial.close()
        world.close()
    return res


# =====================================================================================================================
# mode: perquery (K = 100)
# =====================================================================================================================
def perquery_mode(a) -> dict:
    from foldmatch.search import code_alignment as ca
    cs = ChainSet(a.chains, a.chain_seqs)
    rq = RealQueries(a.queries)
    queries = [rq.fixed(Lq) for Lq in a.lq]
    if a.nq:
        queries += rq.sample(a.nq)
    world = World(a.work, cs, queries, a.db_residues)
    K = a.k
    res = dict(mode="perquery", host=host_info(), K=K, rounds=a.rounds, results=[])
    try:
        pools = {}
        for W in a.workers:
            pools[W] = ca.CodeAlignmentPool(max(W, 1))
            world.align({queries[0][0]: (cs.cand_ids[:K], [0.9] * K)}, pools[W], max_evalue=1e-3)  # start
        for j, (qid, seq, q16, src) in enumerate(queries):
            pf = {qid: (cs.slice_ids(j, K), [float(x) for x in np.linspace(0.99, 0.5, K)])}
            row = dict(query=src, Lq=len(seq), code={}, blosum62={})
            for W in a.workers:
                pool = pools.get(W)
                m, ts = median_rounds(lambda: world.align(pf, pool, max_evalue=1e-3), a.rounds)
                _r, st, _w = world.align(pf, pool, max_evalue=1e-3)
                row["code"][str(W)] = dict(wall=m, walls=ts, counts=_counts(st))
                if W <= 1 or a.blosum_pool:
                    mb, tb = median_rounds(lambda: world.blosum(pf, max(W, 1)), a.blosum_rounds)
                    row["blosum62"][str(W)] = dict(wall=mb, walls=tb)
            res["results"].append(row)
            print(f"{src} (Lq {len(seq)}) K={K}: " + ", ".join(
                f"code W={w} {v['wall']:.3f}s" for w, v in row["code"].items()) + "; " + ", ".join(
                f"BLOSUM62 W={w} {v['wall']:.3f}s" for w, v in row["blosum62"].items()), flush=True)
        for p in pools.values():
            p.close()
    finally:
        world.close()
    return res


# =====================================================================================================================
# mode: node / sweep / profile / rss (multi-query)
# =====================================================================================================================
def _workload(a):
    cs = ChainSet(a.chains, a.chain_seqs)
    queries = make_queries(a, cs)
    world = World(a.work, cs, queries, a.db_residues)
    pf = batch_prefilter(cs, queries, a.k)
    cells = sum(len(q[1]) * int(sum(cs.lens[cs.index[t.split("#")[0]]] for t in pf[q[0]][0])) for q in queries)
    info = dict(nq=len(queries), K=a.k, query_set=a.query_set, query_lengths=dict(
        min=min(len(q[1]) for q in queries), median=float(np.median([len(q[1]) for q in queries])),
        mean=float(np.mean([len(q[1]) for q in queries])), max=max(len(q[1]) for q in queries)),
        candidate_pairs=len(queries) * a.k, candidate_cells=int(cells), db_residues=world.db_residues,
        store_residues=world.store_residues)
    return cs, queries, world, pf, info


def node_mode(a) -> dict:
    from foldmatch.search import code_alignment as ca
    cs, queries, world, pf, info = _workload(a)
    res = dict(mode="node", host=host_info(), workload=info, rounds=a.rounds, results={})
    configs = (("e1e-3", dict(max_evalue=1e-3)), ("inf", dict(max_evalue=None)),
               ("nopanel_inf", dict(max_evalue=None, panel_policy="never")))
    try:
        for W in a.workers:
            out = res["results"][str(W)] = {}
            with ca.CodeAlignmentPool(W) as pool:
                # the first call sizes the persistent pool (never more workers than its first batch's tasks):
                # start it with the whole workload
                _r, _s, cold = world.align(pf, pool, max_evalue=1e-3)
                out["pool_start_call_s"] = cold
                out["pool_workers"] = pool.workers
                for name, kw in configs:
                    walls, st, sig = [], None, None
                    for _ in range(a.rounds):
                        r_, st, w = world.align(pf, pool, **kw)
                        walls.append(w)
                        s_ = signature(r_)
                        if sig is not None and s_ != sig:
                            raise SystemExit(f"node {name}: output differs between repeats")
                        sig = s_
                    m = float(np.median(walls))
                    pairs = info["candidate_pairs"]
                    out[name] = dict(wall=m, walls=walls, counts=_counts(st), signature=sig,
                                     candidate_pairs_per_s=pairs / m,
                                     all_pairs_per_s=(pairs + (st.get("panel_pairs") or 0)) / m,
                                     gcells_per_s=info["candidate_cells"] / m / 1e9)
                    print(f"node W={W} {name}: {m:.3f}s, {pairs / m:,.0f} candidate pairs/s "
                          f"({out[name]['all_pairs_per_s']:,.0f} incl. panel), survivors "
                          f"{st.get('survivors')}", flush=True)
                if a.profile_run:
                    for name, kw in configs[:2]:
                        world.align(pf, pool, profile=True, **kw)        # the profiled pool (a new worker setup)
                        _r, pst, pwall = world.align(pf, pool, profile=True, **kw)
                        out[f"profile_{name}"] = dict(wall=pwall, **_profile_brief(pst["profile"]))
                        out[f"profile_{name}_tasks"] = pst["profile"]["tasks"] if a.keep_tasks else None
            _dump_partial(a, res)
        if a.serial_nq:
            sub = {q[0]: pf[q[0]] for q in queries[:a.serial_nq]}
            npairs = sum(len(v[0]) for v in sub.values())
            ser = res["results"]["serial_subset"] = dict(nq=len(sub), pairs=npairs)
            serial = ca.CodeAlignmentPool(1)
            for name, kw in configs:
                world.align({queries[0][0]: (pf[queries[0][0]][0][:50], pf[queries[0][0]][1][:50])}, serial, **kw)
                _r, st, w = world.align(sub, serial, **kw)
                ser[name] = dict(wall=w, pairs_per_s=npairs / w, counts=_counts(st))
                print(f"serial subset {len(sub)} queries {name}: {w:.3f}s, {npairs / w:,.0f} pairs/s", flush=True)
                for W in a.workers:
                    full = res["results"][str(W)][name]["candidate_pairs_per_s"]
                    res["results"][str(W)][name]["linear_fraction"] = full / (W * npairs / w)
            serial.close()
    finally:
        world.close()
    return res


def sweep_mode(a) -> dict:
    from foldmatch.search import code_alignment as ca
    cs, queries, world, pf, info = _workload(a)
    res = dict(mode="sweep", host=host_info(), workload=info, rounds=a.rounds, block_floats={}, task_cells={})
    sigs = set()
    try:
        for W in a.workers:
            pool = ca.CodeAlignmentPool(max(W, 1))
            try:
                world.align(pf, pool)          # sizes the persistent pool with the whole workload
                res.setdefault("pool_workers", {})[str(W)] = pool.workers
                for e in a.block_exps:
                    walls, sig = [], None
                    for _ in range(a.rounds):
                        r_, st, w = world.align(pf, pool, max_evalue=1e-3, block_floats=1 << e)
                        walls.append(w)
                        sig = signature(r_)
                    sigs.add(sig)
                    m = float(np.median(walls))
                    res["block_floats"][f"{W}:2^{e}"] = dict(workers=W, block_floats=1 << e, wall=m, walls=walls,
                                                             signature=sig, pairs_per_s=info["candidate_pairs"] / m)
                    print(f"sweep W={W} block_floats 2^{e}: {m:.3f}s {info['candidate_pairs'] / m:,.0f} pairs/s "
                          f"sig {sig[:12]}", flush=True)
                for e in a.task_exps:
                    walls, sig, st = [], None, None
                    for _ in range(a.rounds):
                        r_, st, w = world.align(pf, pool, max_evalue=1e-3, task_cells=1 << e)
                        walls.append(w)
                        sig = signature(r_)
                    sigs.add(sig)
                    m = float(np.median(walls))
                    res["task_cells"][f"{W}:2^{e}"] = dict(workers=W, task_cells=1 << e, wall=m, walls=walls,
                                                           signature=sig, pairs_per_s=info["candidate_pairs"] / m,
                                                           round1_tasks=st.get("round1_tasks"))
                    print(f"sweep W={W} task_cells 2^{e}: {m:.3f}s {info['candidate_pairs'] / m:,.0f} pairs/s "
                          f"({st.get('round1_tasks')} round-1 tasks) sig {sig[:12]}", flush=True)
            finally:
                pool.close()
            _dump_partial(a, res)
    finally:
        world.close()
    res["all_outputs_identical"] = len(sigs) == 1
    res["signatures"] = sorted(sigs)
    print(f"sweep: {len(sigs)} distinct output signature(s)", flush=True)
    return res


def profile_mode(a) -> dict:
    """For each worker count (0/1 = serial in-process): a profiled call on a fresh pool (cold: with the worker
    start-up), a profiled call on the warm pool, then ``--rounds`` profile-off and profile-on calls (each set after
    one untimed call, since switching the mode restarts the workers) for the overhead and the identical output, and
    the time ``output.write_aligned_results`` takes for these hits (default fields and every field)."""
    from foldmatch.search import code_alignment as ca
    from foldmatch.search import output
    cs, queries, world, pf, info = _workload(a)
    res = dict(mode="profile", host=host_info(), workload=info, rounds=a.rounds, max_evalue=a.max_evalue,
               results={})
    me = a.max_evalue if math.isfinite(a.max_evalue) else None
    try:
        for W in a.workers:
            out = res["results"][str(W)] = {}
            with ca.CodeAlignmentPool(max(W, 1)) as pool:
                for label in ("cold", "warm"):
                    r_on, pst, w = world.align(pf, pool, max_evalue=me, profile=True)
                    prof = pst["profile"]
                    out[label] = dict(wall=w, counts=_counts(pst), **_profile_brief(prof))
                    if a.keep_tasks:
                        out[f"{label}_tasks"] = prof["tasks"]
                sig_on = signature(r_on)
                walls = {}
                for mode, flag in (("off", False), ("on", True)):
                    world.align(pf, pool, max_evalue=me, profile=flag)
                    walls[mode] = []
                    for _ in range(a.rounds):
                        r_, _s, w = world.align(pf, pool, max_evalue=me, profile=flag)
                        walls[mode].append(w)
                    if mode == "off":
                        r_off, sig_off = r_, signature(r_)
                out["off_walls"], out["on_walls"] = walls["off"], walls["on"]
                out["off_median"], out["on_median"] = float(np.median(walls["off"])), float(np.median(walls["on"]))
                out["profile_overhead_fraction"] = out["on_median"] / out["off_median"] - 1.0
                out["identical_output_on_off"] = sig_on == sig_off
                out["signature"] = sig_off
                with tempfile.TemporaryDirectory(dir=a.work) as td:
                    for label, fields in (("default", list(output.DEFAULT_OUTPUT_FIELDS)),
                                          ("all", list(output.SUPPORTED_OUTPUT_FIELDS))):
                        r_w = r_off if label == "default" else world.align(pf, pool, max_evalue=me, fields=fields)[0]
                        t0 = time.perf_counter()
                        output.write_aligned_results(r_w, fields, str(Path(td) / f"{label}.tsv"))
                        out[f"output_{label}_s"] = time.perf_counter() - t0
                        out[f"output_{label}_bytes"] = (Path(td) / f"{label}.tsv").stat().st_size
                        out[f"output_{label}_rows"] = sum(len(v) for v in r_w.values())
            for label in ("cold", "warm"):
                p = out[label]
                print(f"profile W={W} {label}: wall {p['wall']:.3f}s; parent " + ", ".join(
                    f"{k} {v:.3f}" for k, v in p["parent"].items()) + "; workers " + "; ".join(
                    f"{k}: {v['tasks']} tasks {v['seconds']:.3f}s (sim {v['sim']:.3f} kernel {v['kernel']:.3f})"
                    for k, v in p["workers"].items()), flush=True)
            print(f"profile W={W}: off {out['off_median']:.3f}s on {out['on_median']:.3f}s (overhead "
                  f"{100 * out['profile_overhead_fraction']:.1f} %), identical output {out['identical_output_on_off']}"
                  f"; output default {out['output_default_s']:.3f}s ({out['output_default_rows']} rows), all fields "
                  f"{out['output_all_s']:.3f}s", flush=True)
            _dump_partial(a, res)
    finally:
        world.close()
    return res


def rss_mode(a) -> dict:
    from foldmatch.search import code_alignment as ca
    a.query_set = f"fixed:{a.rss_lq}"
    cs, queries, world, pf, info = _workload(a)
    res = dict(mode="rss", host=host_info(), workload=info, results={})
    try:
        for W in a.workers:
            with ca.CodeAlignmentPool(W) as pool:
                for name, kw in (("e1e-3", dict(max_evalue=1e-3)), ("inf", dict(max_evalue=None))):
                    _r, pst, w = world.align(pf, pool, profile=True, **kw)
                    mem = pst["profile"]["memory"]
                    first = {}
                    for t in pst["profile"]["tasks"]:
                        first.setdefault(t["pid"], t)
                    res["results"][f"{W}:{name}"] = dict(wall=w, memory=mem, counts=_counts(pst),
                                                         worker_rss_first_task_bytes=sorted(
                                                             v.get("rss_max_bytes", 0) for v in first.values()))
                    print(f"rss W={W} {name}: worker RSS max {mem['worker_rss_max_bytes'] / 2**20:.1f} MiB, parent "
                          f"{mem['parent_rss_max_bytes'] / 2**20:.1f} MiB", flush=True)
    finally:
        world.close()
    return res


# =====================================================================================================================
# mode: replay (the real pipeline's Stage-2 inputs, captured by stage2_capture.py)
# =====================================================================================================================
class Replay:
    """A captured Stage-2 call: the database's stores, the query memmap, the Stage-1 candidates, N and options."""

    def __init__(self, capture: Path, db_dir: Path = None, lq_range=None, max_candidates=None):
        from foldmatch.search import scoring_system as ss
        from foldmatch.search.query_encodings import QueryEncodingEntry, QueryEncodingIndex
        from foldmatch.search.residue_code_store import RESCODES_BIN_SUFFIX, ResidueCodeStore
        from foldmatch.search.sequence_store import SequenceStore
        cap = json.loads((capture / "capture.json").read_text())
        self.cap = cap
        bin_path = Path(cap["code_store_bin"])
        name = bin_path.name[:-len(RESCODES_BIN_SUFFIX)]
        folder = Path(db_dir) if db_dir else bin_path.parent
        self.store = ResidueCodeStore(folder, name)
        self.seq_store = SequenceStore(folder, name)
        manifest = self.store.manifest()
        self.system = ss.get(manifest["system_name"])
        self.prepared = ss.prepare(self.system, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")))
        entries = {q: QueryEncodingEntry(int(o), int(n), bytes.fromhex(h)) for q, (o, n, h) in cap["entries"].items()}
        self.query_index = QueryEncodingIndex(capture / "queries.f16", cap["rows"], cap["width"], entries,
                                              cap["system_name"])
        self.query_sequences = cap["query_sequences"]
        pf = {q: (list(v[0]), list(v[1])) for q, v in cap["prefilter_results"].items()}
        if lq_range:
            lo, hi = lq_range
            pf = {q: v for q, v in pf.items() if lo <= len(self.query_sequences[q]) <= hi}
        if max_candidates:
            pf = {q: (v[0][:max_candidates], v[1][:max_candidates]) for q, v in pf.items()}
        self.prefilter = pf
        self.db_residues = int(cap["db_residues"])
        self.options = dict(cap["options"])
        self.fields = self.options.pop("output_needs")
        self.options.pop("max_evalue", None)
        self.max_evalue = cap["options"].get("max_evalue")

    def align(self, pool, *, max_evalue="capture", profile=False, prefilter=None, **kw):
        from foldmatch.search import code_alignment as ca
        me = self.max_evalue if max_evalue == "capture" else max_evalue
        opts = dict(self.options)
        opts.update(kw)
        stats = {}
        old = os.environ.get(PROFILE_ENV)
        os.environ[PROFILE_ENV] = "1" if profile else ""
        try:
            t0 = time.perf_counter()
            res = ca.align_candidates_esm_codes(
                self.query_index, prefilter if prefilter is not None else self.prefilter, self.store,
                self.seq_store.fetch, self.prepared, self.db_residues, pool, query_sequences=self.query_sequences,
                max_evalue=me, output_needs=self.fields, stats=stats, **opts)
            wall = time.perf_counter() - t0
        finally:
            if old is None:
                os.environ.pop(PROFILE_ENV, None)
            else:
                os.environ[PROFILE_ENV] = old
        return res, stats, wall

    def blosum(self, num_workers=1, prefilter=None):
        from foldmatch.search import alignment
        t0 = time.perf_counter()
        res = alignment.align_candidates(self.query_sequences, prefilter if prefilter is not None else self.prefilter,
                                         self.seq_store.fetch, num_workers=num_workers,
                                         subject_db_size=self.db_residues, output_fields=None)
        return res, time.perf_counter() - t0


def replay_mode(a) -> dict:
    """The captured real-pipeline Stage 2 for each ``--workers``: a cold profiled call (fresh pool), a warm profiled
    call, ``--rounds`` profile-off calls (wall, output signature: equal to the CLI's when the capture's machine has
    the same BLAS and CPU kernel), the same at ``--max-evalue`` inf when ``--also-inf``; BLOSUM62 on the same
    candidates with ``--blosum-workers``."""
    from foldmatch.search import code_alignment as ca
    rp = Replay(a.capture, a.db_dir, a.lq_range, a.max_candidates)
    pf = rp.prefilter
    lens = [len(rp.query_sequences[q]) for q in pf]
    npairs = sum(len(v[0]) for v in pf.values())
    info = dict(capture=str(a.capture), queries=len(pf), queries_with_candidates=sum(1 for v in pf.values() if v[0]),
                candidate_pairs=npairs, query_lengths=dict(min=min(lens), median=float(np.median(lens)),
                                                           max=max(lens)),
                db_residues=rp.db_residues, max_evalue=rp.max_evalue, options=rp.options, fields=rp.fields,
                cli_signature=rp.cap["signature"], cli_stage2_s=rp.cap["stage2_s"], cli_pool=rp.cap["pool"],
                lq_range=a.lq_range, max_candidates=a.max_candidates)
    res = dict(mode="replay", host=host_info(), workload=info, rounds=a.rounds, results={}, blosum62={})
    settings = [("capture", "capture")] + ([("inf", None)] if a.also_inf else [])
    for W in a.workers:
        out = res["results"][str(W)] = {}
        with ca.CodeAlignmentPool(max(W, 1)) as pool:
            for label in (("cold",) if a.no_warm_profile else ("cold", "warm")):
                _r, pst, w = rp.align(pool, profile=True)
                out[label] = dict(wall=w, counts=_counts(pst), **_profile_brief(pst["profile"]))
                if a.keep_tasks:
                    out[f"{label}_tasks"] = pst["profile"]["tasks"]
            for name, me in settings:
                if not a.no_warmup:
                    rp.align(pool, max_evalue=me)
                walls, sig, st = [], None, None
                for _ in range(a.rounds):
                    r_, st, w = rp.align(pool, max_evalue=me)
                    walls.append(w)
                    sig = signature(r_)
                out[name] = dict(wall=float(np.median(walls)), walls=walls, signature=sig, counts=_counts(st),
                                 pairs_per_s=npairs / float(np.median(walls)))
                if name == "inf":
                    _r, pst, w = rp.align(pool, max_evalue=None, profile=True)
                    out["inf_profiled"] = dict(wall=w, counts=_counts(pst), **_profile_brief(pst["profile"]))
            out["identical_to_cli"] = (out["capture"]["signature"] == rp.cap["signature"]
                                       if a.lq_range is None and a.max_candidates is None else None)
        c = out.get("warm", out["cold"])
        print(f"replay W={W}: cold {out['cold']['wall']:.3f}s, warm {c['wall']:.3f}s, off {out['capture']['wall']:.3f}s "
              f"({out['capture']['counts']['survivors']} survivors of {npairs}); identical to the CLI "
              f"{out['identical_to_cli']}; parent " + ", ".join(f"{k} {v:.3f}" for k, v in c["parent"].items())
              + "; workers " + "; ".join(f"{k}: {v['tasks']} tasks {v['seconds']:.3f}s" for k, v in
                                         c["workers"].items()), flush=True)
        _dump_partial(a, res)
    for W in a.blosum_workers:
        walls = []
        for _ in range(a.blosum_rounds):
            _r, w = rp.blosum(W)
            walls.append(w)
        res["blosum62"][str(W)] = dict(wall=float(np.median(walls)), walls=walls, pairs_per_s=npairs / np.median(walls))
        print(f"replay BLOSUM62 W={W}: {np.median(walls):.3f}s", flush=True)
    return res


# =====================================================================================================================
# mode: research-batch (the research driver; never imports this tree's swkern)
# =====================================================================================================================
def research_batch_mode(a) -> dict:
    """The research throughput bench's FoldMatch-shaped driver (codepath.align_candidates_x, scorer 'code': per block
    decode (np.unpackbits) + ONE block GEMM + rowcol z folded into the research strip SW; research setting offset
    0.5, gaps 3.0 / 0.3, cost-neutral vs setting D) on the same workload as the FoldMatch modes."""
    for p in (a.research_bench, a.research_build):
        if p is None:
            raise SystemExit("--research-bench and --research-build are required")
    paths = [str(a.research_build), str(a.research_bench)]
    if a.research_fm_src:
        paths.insert(0, str(a.research_fm_src))
    for p in reversed(paths):
        sys.path.insert(0, p)
    import foldmatch.search.alignment as FA  # noqa: F401  (the snapshot when --research-fm-src)
    import codepath as CP
    import swfast
    cs = ChainSet(a.chains, a.chain_seqs)
    queries = make_queries(a, cs)
    pf = batch_prefilter(cs, queries, a.k)
    seq_of = {t: cs.seqs[int(cs.draw[k])] for k, t in enumerate(cs.cand_ids)}
    code_of = {t: np.ascontiguousarray(cs.chain_codes(int(cs.draw[k]))) for k, t in enumerate(cs.cand_ids)}
    qseq = {q[0]: q[1] for q in queries}
    qside = {q[0]: np.ascontiguousarray(swfast.query_prep(q[2].astype(np.float32))) for q in queries}
    fetch = (lambda ids: {i: seq_of[i] for i in ids})
    codes = (lambda ids: {i: code_of[i] for i in ids})
    import multiprocessing as mp
    prewarmed = False
    if mp.get_start_method() == "fork":
        CP.worker_init(dict(scorer="code", mode="tb", tb_fraction=0.0, lam=CP.CODE_LAM, k=CP.CODE_K, gaps=None,
                            db_residues=100_000_000, needs=None, block_cells=CP.BLOCK_CELLS), {})
        prewarmed = True
    import foldmatch
    res = dict(mode="research-batch", host=host_info(with_kernel=False), foldmatch_imported_from=foldmatch.__file__,
               research_bench=str(a.research_bench), research_build=str(a.research_build),
               swfast_vector_lanes=int(swfast.VECTOR_LANES), start_method=mp.get_start_method(),
               parent_prewarmed=prewarmed, nq=len(queries), K=a.k, query_set=a.query_set, rounds=a.rounds,
               results={})
    npairs = len(queries) * a.k
    modes = dict(s0=dict(mode="score", tb_fraction=0.0, identity="vector"),
                 s10=dict(mode="score", tb_fraction=0.1, identity="vector"),
                 tb=dict(mode="tb", identity="vector"))
    for W in a.workers:
        for name in a.research_modes:
            kw = modes[name]
            q0 = queries[0][0]

            def call(sub=None):
                p = pf if sub is None else sub
                return CP.align_candidates_x({k: qseq[k] for k in p}, p, fetch, codes, {k: qside[k] for k in p},
                                             scorer="code", min_seq_identity=0.0, num_workers=W, min_coverage=0.8,
                                             max_evalue=1e-3, subject_db_size=100_000_000, **kw)
            call({q0: (pf[q0][0][:100], pf[q0][1][:100])})
            walls, meta = [], None
            for _ in range(a.rounds):
                t0 = time.perf_counter()
                _r, meta = call()
                walls.append(time.perf_counter() - t0)
            m = float(np.median(walls))
            res["results"][f"{W}:{name}"] = dict(workers=W, mode=name, wall=m, walls=walls, pairs=npairs,
                                                 pairs_per_s=npairs / m,
                                                 meta={k: v for k, v in (meta or {}).items() if k != "wall"})
            print(f"research W={W} {name}: {m:.3f}s {npairs / m:,.0f} pairs/s", flush=True)
            _dump_partial(a, res)
    return res


# =====================================================================================================================
# mode: bandwidth
# =====================================================================================================================
def _triad_worker(args):
    n, seconds, cpu = args
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    a_ = np.ones(n, np.float32)
    b_ = np.ones(n, np.float32)
    c_ = np.ones(n, np.float32)
    s = np.float32(1.5)
    reps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        np.multiply(c_, s, out=a_)
        np.add(a_, b_, out=a_)
        reps += 1
    dt = time.perf_counter() - t0
    return reps * 4 * n * 4 / dt          # bytes: read c, write a, read a, read b, write a ~ 4 streams per rep


def bandwidth_mode(a) -> dict:
    import multiprocessing as mp
    res = dict(mode="bandwidth", host=host_info(with_kernel=False), results={})
    ctx = mp.get_context("spawn")
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count()))
    for W in a.workers:
        for label, n in (("dram_64MiB", 1 << 24), ("l2_256KiB", 1 << 16)):
            pin = [cpus[i % len(cpus)] for i in range(W)] if a.pin else [None] * W
            with ctx.Pool(W) as pool:
                bw = pool.map(_triad_worker, [(n, a.seconds, c) for c in pin])
            res["results"][f"{W}:{label}"] = dict(workers=W, floats=n, gbytes_per_s_total=sum(bw) / 1e9,
                                                  gbytes_per_s_per_worker=float(np.median(bw)) / 1e9)
            print(f"bandwidth W={W} {label}: {sum(bw) / 1e9:.1f} GB/s total, {np.median(bw) / 1e9:.2f} per worker",
                  flush=True)
    return res


# =====================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["kernel", "components", "grid", "panel", "perquery", "node", "sweep", "profile",
                                     "rss", "replay", "research-batch", "bandwidth"])
    ap.add_argument("--chains", type=Path)
    ap.add_argument("--chain-seqs", type=Path)
    ap.add_argument("--queries", type=Path, help="queries.L2m_B8_s0.npz (real benchmark queries)")
    ap.add_argument("--work", type=Path, default=Path("stage2_bench_work"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--db-residues", type=int, default=BENCH_N, help="N for the E-values (default: the research "
                                                                     "E-value benchmark's 15,889,271 residues)")
    ap.add_argument("--lq", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--ks", type=int, nargs="+", default=[100, 1000, 10000])
    ap.add_argument("--nq", type=int, default=None, help="queries (default 64; perquery: 0 extra real queries)")
    ap.add_argument("--query-set", default="real", help="real | fixed:<Lq> | chain:<Lq>")
    ap.add_argument("--workers", type=int, nargs="+", default=[1])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--blosum-rounds", type=int, default=3)
    ap.add_argument("--blosum-pool", action="store_true", help="perquery: also BLOSUM62 with each worker count")
    ap.add_argument("--max-evalue", type=float, default=1e-3)
    ap.add_argument("--warm", action="store_true", help="profile: start the pool before the profiled run")
    ap.add_argument("--keep-tasks", action="store_true", help="keep the per-task profile records in the JSON")
    ap.add_argument("--profile-run", action="store_true", help="node: one profiled 1e-3 run per worker count")
    ap.add_argument("--serial-nq", type=int, default=0, help="node: queries of the serial per-core reference")
    ap.add_argument("--block-floats", type=int, default=1 << 17)
    ap.add_argument("--block-exps", type=int, nargs="+", default=[14, 15, 16, 17, 18, 19, 20])
    ap.add_argument("--task-exps", type=int, nargs="+", default=[20, 21, 22, 23, 24, 25, 26])
    ap.add_argument("--panel-lengths", type=int, nargs="+", default=[71, 966])
    ap.add_argument("--pool-query-length", type=int, default=300)
    ap.add_argument("--rss-lq", type=int, default=1000)
    ap.add_argument("--cc", default=None)
    # replay
    ap.add_argument("--capture", type=Path, help="replay: a stage2_capture.py directory")
    ap.add_argument("--db-dir", type=Path, help="replay: the database folder (default: the capture's)")
    ap.add_argument("--lq-range", type=int, nargs=2, default=None, help="replay: only queries of these lengths")
    ap.add_argument("--max-candidates", type=int, default=None, help="replay: the first K candidates per query")
    ap.add_argument("--also-inf", action="store_true", help="replay: also at max_evalue inf")
    ap.add_argument("--blosum-workers", type=int, nargs="*", default=[], help="replay: BLOSUM62 worker counts")
    ap.add_argument("--no-warm-profile", action="store_true", help="replay: only the cold profiled call")
    ap.add_argument("--no-warmup", action="store_true", help="replay: no untimed call before the timed ones")
    ap.add_argument("--big-k", type=int, default=10000, help="grid: K from which --big-k-rounds apply")
    ap.add_argument("--big-k-rounds", type=int, default=1, help="grid: rounds (no warm-up) for K >= --big-k")
    # research
    ap.add_argument("--research-src", type=Path, help="directory with the research swkern.c and swmulti.c")
    ap.add_argument("--research-lib-dir", type=Path, help="prebuilt research libswkern / libswmulti")
    ap.add_argument("--research-cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--research-cflags", default=None)
    ap.add_argument("--research-bench", type=Path, help="research throughput bench dir (codepath.py)")
    ap.add_argument("--research-build", type=Path, help="research build dir (swfast.py + libswkern)")
    ap.add_argument("--research-fm-src", type=Path, help="the FoldMatch src snapshot of the research runs")
    ap.add_argument("--research-modes", nargs="+", default=["s0", "s10", "tb"])
    ap.add_argument("--blocks", type=int, nargs="+", default=[17, 20], help="kernel: FoldMatch block exponents")
    ap.add_argument("--research-blocks", type=int, nargs="+", default=[20, 17])
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--core", type=int, default=None, help="Linux: pin this process to one core")
    a = ap.parse_args()
    if a.lq is None:
        a.lq = [300] if a.mode == "kernel" else [100, 300, 1000]
    if a.nq is None:
        a.nq = 0 if a.mode == "perquery" else 64
    if a.core is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {a.core})
    t0 = time.time()
    fn = dict(kernel=kernel_mode, components=components_mode, grid=grid_mode, panel=panel_mode,
              perquery=perquery_mode, node=node_mode, sweep=sweep_mode, profile=profile_mode, rss=rss_mode,
              replay=replay_mode, bandwidth=bandwidth_mode)
    fn["research-batch"] = research_batch_mode
    res = fn[a.mode](a)
    res["argv"] = sys.argv
    res["seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    part = a.out.with_suffix(".partial.json")
    if part.exists():
        part.unlink()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
