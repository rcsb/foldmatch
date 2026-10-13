#!/usr/bin/env python
"""esm-codes Stage-2 benchmark (not a test). P3 kernel gate now; P7 extends it (grid, BLOSUM62 baseline, pool).

Mode ``kernel`` (PLAN section 7, P3 acceptance (e)): one process, BLAS / OpenMP pinned to one thread, fwd+rev
score-only for one query against K chain-set candidates, INCLUDING query_prep and the similarity, for

  foldmatch     swkern.query_prep (C) + swkern.score_fwd_rev: per-pair numpy fp32 GEMM into 64-byte-aligned
                per-pair buffers + fmsw_score_fwd_rev (S_r on the row-reversed view), block_floats 2^17 (default)
                and 2^20
  research      the research production path (embcomp/evalue/score/score.py): numpy fp32 query_prep; per block of
                <= block_floats similarity floats: decode (np.unpackbits) + ONE block GEMM + reversed copy +
                swz_ad_multi_f32 (R = 2) from the unmodified research swkern.c / swmulti.c, built with the research
                flags; blocks 2^20 (score.py) and 2^17; '.reuse' = the same with every buffer reused
                (LUT decode into a buffer, GEMM out=, copyto) = the research path without per-block allocations

Workload = the research throughput workload (embcomp/throughput/bench/results/summary.txt), rebuilt from the
chain-set encodings ``L2m_B8_s0.chains.npz`` (md5 5519866044ab0f348c74b9319125acd2): queries Lq=100 1j2vA 1-100,
Lq=300 4xrtA 1-300, Lq=1000 4j5tA (all) + 3dwbA 1-212; candidates = the first K of
``np.random.default_rng(20260925).integers(0, 559, 10000)`` (reproduces the summary's 21,082 / 229,955 /
2,218,273 residues at K = 100 / 1,000 / 10,000). Each variant: one warm-up round, then ``--rounds`` rounds; the
MEDIAN round is reported as pairs/s. The run also checks FoldMatch vs research S_f / S_r within S16.

Gate (EPYC 7763, NERSC): foldmatch@2^17 >= 0.9 x research@2^20 (block GEMM, score.py) at Lq=300, K=1,000, and
>= 2,500 pairs/s. M1 Max is reported (target 4,700 pairs/s), not gated.

  python benchmarks/stage2_esm_codes.py kernel --chains .../L2m_B8_s0.chains.npz --research-src DIR \\
      --work DIR --out result.json [--lq 300] [--k 1000] [--rounds 7] [--core 3]

``--research-src`` holds swkern.c (md5 438d5a98...) and swmulti.c (md5 b12fbc2c...); they are compiled into --work
with ``--research-cc`` and ``--research-cflags`` (default: the research flags, -mcpu=apple-m1 on Apple Silicon,
-march=znver3 on x86_64 Linux, -mcpu=native on aarch64 Linux; plus -ffp-contract=off). Use ``--research-lib-dir``
for prebuilt libswkern / libswmulti instead.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

from foldmatch.search import swkern  # noqa: E402

CHAINS_MD5 = "5519866044ab0f348c74b9319125acd2"
RESEARCH_MD5 = {"swkern.c": "438d5a9811153f61e66cee845f1ed037", "swmulti.c": "b12fbc2c03638659440b12282a4d8670"}
WORKLOAD_SEED, POOL = 20260925, 559
QUERIES = {100: [("1j2vA", 0, 100)], 300: [("4xrtA", 0, 300)], 600: [("3zukA", 0, 600)],
           1000: [("4j5tA", 0, None), ("3dwbA", 0, 212)]}
OFF, GO, GE = 1.0, 5.0, 0.5
_vp = C.c_void_p


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def host_info() -> dict:
    info = dict(host=socket.gethostname(), platform=platform.platform(), machine=platform.machine(),
                python=platform.python_version(), cpus=os.cpu_count(), blas=swkern.blas_info())
    try:
        if sys.platform == "darwin":
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            info["affinity"] = sorted(os.sched_getaffinity(0))[:16]
    except Exception:
        pass
    return info


# =====================================================================================================================
# workload
# =====================================================================================================================
class Workload:
    def __init__(self, chains: Path, K: int):
        if md5(chains) != CHAINS_MD5:
            raise SystemExit(f"{chains}: md5 differs from {CHAINS_MD5}")
        z = np.load(chains)
        self.Q, self.codes_all, off = z["q"], z["codes"], z["offsets"].astype(np.int64)
        self.ids = {s: i for i, s in enumerate(z["ids"].astype(str))}
        self.chain_off = off
        if len(off) - 1 != POOL:
            raise SystemExit(f"expected {POOL} chains, found {len(off) - 1}")
        cand = np.random.default_rng(WORKLOAD_SEED).integers(0, POOL, 10000)[:K]
        parts = [self.codes_all[off[c]:off[c + 1]] for c in cand]
        self.codes = np.ascontiguousarray(np.concatenate(parts), np.uint8)
        self.offsets = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
        self.K, self.residues = K, int(self.offsets[-1])
        self.lens = np.diff(self.offsets)

    def query(self, Lq: int) -> np.ndarray:
        rows = []
        for cid, a, b in QUERIES[Lq]:
            i = self.ids[cid]
            q = self.Q[self.chain_off[i]:self.chain_off[i + 1]]
            rows.append(q[a:b])
        q16 = np.ascontiguousarray(np.concatenate(rows), np.float16)
        assert len(q16) == Lq, (Lq, len(q16))
        return q16


# =====================================================================================================================
# research path (unmodified research kernels)
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
        if md5(src / name) != want:
            raise SystemExit(f"{src / name}: md5 differs from {want}")
    work.mkdir(parents=True, exist_ok=True)
    ext = "dylib" if sys.platform == "darwin" else "so"
    for name in ("swkern", "swmulti"):
        cmd = [*shlex.split(cc), *cflags, f"-I{src}", "-o", str(work / f"lib{name}.{ext}"), str(src / f"{name}.c"),
               "-lm"]
        subprocess.run(cmd, check=True)
    return work


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

    def runner(self, W: Workload, q16, block_floats, reuse=False):
        blk = self.blocks(W.lens.tolist(), len(q16), block_floats)
        cols = [np.ascontiguousarray(W.offsets[k0:k1 + 1] - W.offsets[k0]) for k0, k1 in blk]
        out = np.empty((2, W.K), np.float32)
        bufs = {}

        def run():
            qp = self.query_prep(q16)
            n = len(qp)
            for (k0, k1), col in zip(blk, cols):
                j0, j1 = int(W.offsets[k0]), int(W.offsets[k1])
                w = j1 - j0
                o2 = np.empty((2, k1 - k0), np.float32)
                if reuse:
                    for key, size in (("T", w * 64), ("sim", n * w), ("simr", n * w)):
                        if bufs.get(key) is None or bufs[key].size < size:
                            bufs[key] = np.empty(size, np.float32)
                    T = bufs["T"][:w * 64].reshape(w, 8, 8)
                    np.take(self._LUT, W.codes[j0:j1], axis=0, out=T, mode="clip")
                    sim = bufs["sim"][:n * w].reshape(n, w)
                    np.matmul(qp, T.reshape(w, 64).T, out=sim)
                    simr = bufs["simr"][:n * w].reshape(n, w)
                    np.copyto(simr, sim[::-1])
                else:
                    T = np.unpackbits(W.codes[j0:j1], axis=1)[:, :64].astype(np.float32) * 2 - 1
                    sim = qp @ T.T
                    simr = np.ascontiguousarray(sim[::-1])
                self.multi_fwd_rev(sim, simr, col, o2)
                out[:, k0:k1] = o2
            return out
        return run


def foldmatch_runner(W: Workload, q16, block_floats):
    k = swkern.load()
    ws = swkern.default_workspace()

    def run():
        qp = swkern.query_prep(q16, k)
        return swkern.score_fwd_rev(qp, W.codes, W.offsets, swkern.SETTING_D, block_floats=block_floats, kernel=k,
                                    workspace=ws)
    return run


def median_rounds(f, rounds):
    f()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [round(t, 6) for t in ts]


# =====================================================================================================================
def kernel_mode(a) -> dict:
    W = Workload(a.chains, a.k)
    if a.research_lib_dir:
        R = Research(a.research_lib_dir)
        research_build = dict(lib_dir=str(a.research_lib_dir))
    else:
        flags = shlex.split(a.research_cflags) if a.research_cflags else research_flags()
        R = Research(build_research(a.research_src, a.work, a.research_cc, flags))
        research_build = dict(src=str(a.research_src), cc=a.research_cc, cflags=flags, sources_md5=RESEARCH_MD5,
                              cc_version=subprocess.run(shlex.split(a.research_cc) + ["--version"],
                                                        capture_output=True, text=True).stdout.splitlines()[0])
    res = dict(mode="kernel", host=host_info(), kernel=swkern.load().info(), research=research_build,
               research_vector_lanes=int(R.k.swkern_vector_lanes()), K=W.K, residues=W.residues,
               workload=dict(chains=str(a.chains), seed=WORKLOAD_SEED, queries={str(k): v for k, v in QUERIES.items()}),
               rounds=a.rounds, results={})
    for Lq in a.lq:
        q16 = W.query(Lq)
        r = {}
        # correctness: FoldMatch vs research within S16 (and the bit-equal fraction)
        fm = foldmatch_runner(W, q16, 1 << 17)()
        rs = R.runner(W, q16, 1 << 20)()
        bound = swkern.s16_score_bound(rs)
        got = np.stack([fm.s_fwd, fm.s_rev])
        r["s16"] = dict(max_ratio=float(np.max(np.abs(got - rs) / bound)),
                        bit_equal_fraction=float(np.mean(got.view(np.int32) == rs.view(np.int32))))
        variants = [(f"foldmatch.bf2^{e}", foldmatch_runner(W, q16, 1 << e)) for e in a.blocks]
        for e in a.research_blocks:
            variants.append((f"research.bf2^{e}", R.runner(W, q16, 1 << e)))
            variants.append((f"research.reuse.bf2^{e}", R.runner(W, q16, 1 << e, reuse=True)))
        for name, f in variants:
            m, ts = median_rounds(f, a.rounds)
            r[name] = dict(s_med=m, s_rounds=ts, pairs_per_s=W.K / m, ns_per_cell=1e9 * m / (Lq * W.residues))
            print(f"Lq={Lq:5d} K={W.K} {name:28s} {W.K / m:10.1f} pairs/s  {1e9 * m / (Lq * W.residues):6.3f} ns/cell",
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["kernel"])
    ap.add_argument("--chains", type=Path, required=True)
    ap.add_argument("--research-src", type=Path, help="directory with the research swkern.c and swmulti.c")
    ap.add_argument("--research-lib-dir", type=Path, help="prebuilt research libswkern / libswmulti")
    ap.add_argument("--research-cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--research-cflags", default=None)
    ap.add_argument("--work", type=Path, default=Path("stage2_bench_work"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lq", type=int, nargs="+", default=[300], choices=sorted(QUERIES))
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--blocks", type=int, nargs="+", default=[17, 20], help="FoldMatch block_floats exponents")
    ap.add_argument("--research-blocks", type=int, nargs="+", default=[20, 17])
    ap.add_argument("--core", type=int, default=None, help="Linux: pin this process to one core")
    a = ap.parse_args()
    if not (a.research_src or a.research_lib_dir):
        ap.error("--research-src or --research-lib-dir is required")
    if a.core is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {a.core})
    t0 = time.time()
    res = kernel_mode(a)
    res["argv"] = sys.argv
    res["seconds"] = round(time.time() - t0, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
