# FoldMatch

**Version** 0.13.0


## Overview

FoldMatch is a Python toolkit to encode macromolecular 3D structures into fixed-length vector embeddings for efficient large-scale structure similarity search and clustering.

Reference: [Multi-scale structural similarity embedding search across entire proteomes](https://doi.org/10.1093/bioinformatics/btag058).

A web-based implementation using this tool for structure similarity search is available at [RCSB.org](https://www.rcsb.org/search/advanced) under **3D Similarity**.

If you are interested in training a new model with a new structure dataset, visit the [rcsb-embedding-search repository](https://github.com/bioinsilico/rcsb-embedding-search), which provides scripts and documentation for training.


## Features

- **Residue-level embeddings** computed using the ESM3 protein language model
- **Sequence-based embeddings** from FASTA files without requiring 3D structures
- **Structure-level embeddings** aggregated via a transformer-based aggregator network 
- **Fast and efficient** FAISS-based similarity search
- **Two-stage sequence search** — an embedding prefilter followed by exact pairwise Smith-Waterman alignment on FoldMatch's native SIMD kernel, reporting sequence identity, coverage, and calibrated E-values over selectable output columns
- **Structural clustering**: Leiden on the similarity graph (python-igraph, or NetworKit's multithreaded ParallelLeiden as an optional backend), greedy representatives in one streamed pass, or greedy representatives then Leiden on the representatives' graph (`--method greedy-leiden`), resumable from a work folder, for collections whose direct graph does not fit in memory; FAISS search on the CPU or on every visible GPU
- **Command-line interface** implemented with Typer for high-throughput inference workflows  
- **Python API** for interactive embedding computation and integration into analysis pipelines  
- **High-performance inference** leveraging PyTorch Lightning, with multi-node and multi-GPU support  

---

## Installation

### From PyPI

```bash
pip install foldmatch
```

### From Source (Development)

```bash
git clone https://github.com/rcsb/foldmatch.git
cd foldmatch
pip install -e .
```

**Requirements:**

- Python ≥ 3.12
- ESM 3.2.3
- Lightning 2.6.1
- Typer 0.24.1
- Biotite 1.7.1
- FAISS 1.15.1 (the `faiss-cpu` wheel)
- igraph 1.0.0
- PyTorch with CUDA support (recommended for GPU acceleration)
- A C compiler (GCC 12 or later, or Clang 14 or later) when FoldMatch is built from source (an install from the
  sdist, `pip install -e .`): it compiles the native Smith-Waterman kernel of the sequence search's Stage 2. The
  platform wheels (Linux x86_64 and aarch64, macOS arm64 and x86_64) include it. Without a compiler the install
  still succeeds, with a warning, as a pure-Python package in which Stage 2 refuses to run and prints the reinstall
  command; everything else works.

**Optional Dependencies:**

- `faiss-gpu` for GPU-accelerated similarity search (instead of `faiss-cpu`)
- `pip install 'foldmatch[networkit]'` (NetworKit 11.3) for `fm-search cluster --leiden-backend networkit`: wheels for
  Linux x86_64, macOS 26 arm64 and Windows, none for Linux aarch64 or macOS before 26
- a FAISS built on MKL, for faster `ivf_pq` builds of large databases on Linux x86_64 with Python 3.12 (see *FAISS
  threads* below)

### FAISS threads

FoldMatch installs the `faiss-cpu` 1.15.1 wheel. On Linux the wheel bundles an OpenMP build of OpenBLAS
(`faiss_cpu.libs/libopenblaso-*.so`) and its own `libgomp`. FAISS's BLAS work therefore runs on FAISS's OpenMP
threads: the k-means and OPQ steps of an `ivf_pq` training, the IVF assignment of its adds, and batched flat searches.
Earlier FoldMatch versions pinned `faiss-cpu` 1.13.2, whose OpenBLAS is single-threaded. With it, a 30-million-entry
`ivf_pq` index took about 7 hours on any node.

- **Choosing the threads.** Export `OMP_NUM_THREADS` to set FAISS's threads. It also sets the threads of FAISS's
  OpenBLAS. `OPENBLAS_NUM_THREADS` has no effect on FAISS's OpenBLAS; it changes only the OpenBLAS copies of numpy
  and scipy. When `OMP_NUM_THREADS` is unset, `fm-search build sequences-index` gives FAISS the physical cores it may
  use and logs the count.
- **Build times.** On NERSC Perlmutter, `fm-search build sequences-index --residue-codes` over 30.4 million sequences
  built the `ivf_pq` index in 1 h 45 min on a GPU node (one 64-core AMD EPYC 7763, 64 threads; 2 h 47 min for the
  whole command) and in 2 h 37 min on a CPU node (two 64-core EPYC 7763, 128 threads; 3 h 33 min). Most of the
  difference is the training, 40 min against 1 h 16 min: the OPQ step of the wheel's training does not gain from
  more threads. In a benchmark on the CPU node, two OPQ iterations took 66 to 70 s at 4 to 16 threads and 173 s at
  128, while 10 iterations of the IVF k-means gained up to about 32 threads.
- **Recommended on large nodes.** With the wheel, run `build sequences-index` with `OMP_NUM_THREADS=64` on a node
  with more than 64 physical cores, such as a 128-core CPU node. The setting applies to the whole command. This
  advice rests on the 64-thread GPU-node build above and on the OPQ benchmark (121 s at 64 threads against 173 s at
  128); a full build at 64 threads on a 128-core node was not timed. With it, `build sequences-index` warns that FAISS
  has fewer threads than physical cores; with the wheel that warning is expected. With the MKL build below, leave
  `OMP_NUM_THREADS` unset: it built a 30-million-entry index in 43 minutes at 128 threads.
- **128 BLAS threads at most.** The x86_64 wheel's OpenBLAS is built with `MAX_THREADS=128`, so FAISS's BLAS work
  runs on at most 128 threads. That OpenBLAS also lowers a higher OpenMP thread count to 128 when FAISS calls it. On
  a node with more than 128 physical cores, FAISS therefore runs 128 threads, and `build sequences-index` warns that
  FAISS has fewer threads than physical cores.
- **Reproducibility.** Two `hnsw` or `ivf_pq` builds are byte-identical only at `OMP_NUM_THREADS=1`. The wheel's
  k-means results also change with the thread count. `OPENBLAS_NUM_THREADS=1` alone does not make FAISS
  single-threaded. With the wheel, search results can change with FAISS's thread count too, in the last digits of
  the scores and so in the order of tied hits: between 32 and 256 threads, about a quarter of 1,000 batched flat
  searches and 3 to 5 `ivf_pq` searches were reordered. For bit-identical reruns, set `OMP_NUM_THREADS` to the same
  value (unset, FAISS gets each node's core count).
- **Existing databases.** FAISS 1.13.2 and 1.15.1 read each other's index files. Searching one database with either
  version gives the same hits up to float rounding (flat and `ivf_pq` scores within about 3e-7), which can reorder
  tied hits and swap hits tied at the cut-off; with `hnsw`, 5 to 7% of 1,000 searches changed at least one hit. A
  rebuilt `ivf_pq` index is a different, equally valid index: at 30 million entries, rebuilds shared 57 to 59% of
  their top-100 hits with each other (1.13.2 against the MKL build below, the wheel against the MKL build, and two
  wheel builds at 64 and 128 threads), with about the same self-hit rates. Searches whose results you compare should
  all use the same database.
- **Start-up log.** On Linux x86_64, `import faiss` logs at INFO `Could not load library with AVX2 support due to:
  ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")`, then `Successfully loaded faiss.`. This is
  expected: the 1.15.1 wheel has one library, which picks the CPU's instructions at run time (its compile options
  include `DD`).
- **macOS.** The macOS wheel uses Apple Accelerate for BLAS and bundles its own `libomp`, so FoldMatch keeps
  `OMP_NUM_THREADS=1` there by default (see *macOS notes*).

**Optional: FAISS with MKL.** For Linux x86_64 and Python 3.12 only, the official FAISS conda channel, `pytorch`,
publishes a `faiss-cpu` 1.15.1 build on MKL. It builds large `ivf_pq` indexes faster than the wheel, mostly because
its OPQ training is faster. On one NERSC Perlmutter GPU node (64 cores), `build sequences-index --residue-codes` over
30.4 million sequences took 2 h 02 min with it, of which 59 min was the index (the wheel: 2 h 47 min and 1 h 45 min).
In a FAISS-only script on one CPU node (2× AMD EPYC 7763), with 128 threads bound to cores (`OMP_PROC_BIND=spread
OMP_PLACES=cores`; FoldMatch does not bind threads), it built a 30-million-entry `ivf_pq` index in 43 minutes. The
commands below replace the wheel with the conda build. They were tested on NERSC, in a conda env (Python 3.12) in
which FoldMatch and its dependencies had been installed with `pip`:

```bash
python -m pip uninstall -y faiss-cpu
conda install -y --no-deps --override-channels -c pytorch -c conda-forge \
  "pytorch::faiss-cpu=1.15.1=py3.12_hf65b397_0_cpu" "pytorch::libfaiss=1.15.1=py3.12_hf65b397_0_cpu" \
  "conda-forge::mkl=2025.3.1=h0e700b2_12" "conda-forge::onemkl-license=2025.3.1=hf2ce2f3_12" \
  "conda-forge::tbb=2023.1.0=haa8c3ee_2" "conda-forge::libhwloc=2.15.0=default_hd7c4d0c_1001" \
  "conda-forge::libxml2=2.15.4=h7df9aa5_1" "conda-forge::libxml2-16=2.15.4=hf3af7cc_1" \
  "conda-forge::libiconv=1.18=h0cb94f2_3" "conda-forge::llvm-openmp=23.1.4=h7148c6a_0" \
  "conda-forge::_openmp_mutex=4.5=8_kmp_llvm" "conda-forge::libsvs-runtime=0.4.0=g7f7c639_0_0" \
  "conda-forge::python_abi=3.12=9_cp312" "conda-forge::libgcc=16.2.0=ha9f2e26_7" \
  "conda-forge::libgcc-ng=16.2.0=h69a702a_7" "conda-forge::libgomp=16.2.0=he0feb66_7" \
  "conda-forge::libstdcxx=16.2.0=h934c35e_7"
python -c "import faiss; print(faiss.__version__, faiss.get_compile_options())"   # 1.15.1 OPTIMIZE AVX2 ...
```

- **Why `--no-deps`.** It keeps conda from replacing pip's numpy and its BLAS, but conda then checks nothing. The
  listed packages are what `libfaiss` and MKL needed in that env, which had the rest; `libfaiss` needs `libgcc` 16 or
  later. In another env, a library still missing makes `import faiss` fail with an `ImportError` that names it.
- **Reinstalling FoldMatch.** The conda package registers as the distribution `faiss`, not `faiss-cpu`, so `pip check`
  reports `foldmatch ... requires faiss-cpu, which is not installed`. That is expected. Reinstall or upgrade FoldMatch
  in this env only with `pip install --no-deps ...`. Without `--no-deps`, pip installs the `faiss-cpu` wheel over the
  conda FAISS.
- **Updating the env.** Never run `conda update`, or a `conda install` without `--no-deps`, in this env: conda's view
  of it is inconsistent on purpose. Repeat the `--no-deps` form instead.
- **OpenMP runtime.** It depends on the import order. FoldMatch's commands load torch first, so torch's `libgomp`
  serves FAISS and MKL. A script that imports `faiss` first gets LLVM's `libomp`. Both work.
- **Thread logs.** threadpoolctl 3.6.0 does not list this MKL, so the index phase's `Thread pool:` lines do not show
  FAISS's BLAS.



## Usage

The package provides two main interfaces:
1. **Command-line Interface (CLI)** for batch processing and high-throughput workflows
2. **Python API** for interactive use and integration into custom pipelines

---

## Command-Line Interface (CLI)

The toolkit ships two CLIs, `fm-embedding` and `fm-search`. Each is invoked with `--help` for full option documentation; the canonical examples below are enough to get started.

### `fm-embedding` — compute embeddings

Two subcommand groups reflect input modality:

```bash
# Output folders must exist; give each kind of embedding its own (residue and chain files share names)
mkdir -p out/residue out/chain out/assembly out/seq-residue out/seq-chain

# Residue / chain / assembly embeddings from a folder of 3D structures
fm-embedding from-structures residue  --src-folder data/pdb --output-folder out/residue  --structure-format mmcif
fm-embedding from-structures chain    --src-folder data/pdb --output-folder out/chain    --structure-format mmcif
fm-embedding from-structures assembly --src-folder data/pdb --output-folder out/assembly --structure-format mmcif

# Residue / chain embeddings from protein sequences in a FASTA file (no 3D required)
fm-embedding from-sequences  residue  --fasta-file seqs.fasta --output-folder out/seq-residue
fm-embedding from-sequences  chain    --fasta-file seqs.fasta --output-folder out/seq-chain

# One-shot model download
fm-embedding download-models
```

Assembly-level embeddings are only available under `from-structures` — there is no assembly concept for a bare sequence.

Each FASTA record's embedding is stored under its id, the first word of its header, so `from-sequences` refuses a repeated id before ESM3 is loaded. With `--output-format csv` or `pt` (one file per record or chain), an id is a file name: one that holds `/` or cannot name a file, ids that the output folder takes for one file (`abc` and `ABC` on macOS's default file system), and a `--fasta-file` named like an output are refused too, and a run stops at a second write of one file (chains `A` and `a` of one structure on macOS). Parquet and JSON keep any id. A rerun into the same folder replaces its own files, as before.

`from-sequences residue` and `from-sequences chain` take `--output-codes-folder FOLDER`: the same ESM3 pass also computes each sequence's residue codes, the 8-byte per-residue ESM3 codes that `fm-search build sequences --residue-codes` stores (esm-codes-v1 target codes: 8 integers, 0-255, per residue), and writes them to that folder. ESM3 runs once; the embedding files are byte for byte those of a run without the option, and the output folder holds them only, so `fm-search build embeddings` and `query embedding` read it as before (for chain embeddings). The codes folder must exist and must not be `--output-folder` (compared as folders on disk: a symbolic link, `.`, or another case of the name on macOS are the same folder; a usage error, exit status 2); a subfolder of it, or a folder that holds it, will do (`fm-search` reads the files of a folder, not its subfolders), unless it is named as a file of the other folder's output. The codes follow `--output-format`, in the layout of the embeddings, one `[L, 8]` table per sequence of `L` residues, one row per residue: row `i` of the codes is residue `i`, which is row `i + 1` of the residue embeddings (`[L + 2, 1536]`, ESM3's BOS and EOS rows included):

| Format | Residue-code files in the codes folder | Layout |
|---|---|---|
| `csv` | `<id>.csv`, as the embedding is `<output-folder>/<id>.csv` | L lines of 8 integers, no header or index |
| `pt` | `<id>.pt` | a CPU `uint8` tensor `[L, 8]` |
| `parquet` | `<output-codes-name>-<rank>.parquet` | columns `id`, `residue_codes` (`list<fixed_size_list<uint8, 8>>`), `seq_sha1`; the schema metadata names the scoring system, its digests and the pass (`--min-res-n`, device, ranks, versions) |
| `json` | `<output-codes-name>-<rank>.json.gz` | records `{"id", "residue_codes"}`, a list of 8 integers per residue |

`--output-codes-name NAME` names the parquet and json files (default `<output-name>-codes`; the output name itself is allowed, as the folders differ); like `--output-name`, it names no csv or pt file. It is refused without `--output-codes-folder`. Before ESM3 is loaded, a run with `--output-codes-folder` refuses a codes name that cannot name those files, ids (csv, pt) that the codes folder takes for one file, a `--fasta-file` named like a code file, one folder inside the other under the name of a file that the other one gets (`out/<id>.csv` as the codes folder of a csv run, say), and FASTA records whose sequence holds `<`, `>` or non-ASCII characters (as the build does). Nor does any `fm-embedding` run replace a file of the other kind: before ESM3 (exit status 1), it refuses to replace an embedding file in its codes folder (an earlier run's output folder given as the codes folder, say) or a residue-code file in its output folder, with or without `--output-codes-folder`; files of other names are left alone, and a rerun into the same folders replaces its own files as before. The esm-codes bundle (~26 MB) is downloaded once, as for builds; on several nodes, or ranks started by `srun` or `torchrun`, run `fm-embedding download-models` first. A sequence whose codes cannot be computed (ESM3's rows do not match its length) is logged as an ERROR and gets an empty table (an empty file in csv); the run still writes every file, then exits with status 1, naming how many sequences got no codes, the first ids and where their empty tables are. With several devices or nodes, each rank that ends its pass waits for the others, and rank 0 then fails for the whole run (the other ranks log their count), so that no rank fails while another one still writes: the exit status is 1 under Lightning's own launcher, `srun` or `torchrun` alike. The code files are neither embeddings nor embedding shards: `fm-search build embeddings`, `query embedding` and `build residue-codes --from-shards` refuse them if they are moved among embedding files.

```bash
mkdir -p out/seq-chain out/seq-codes
fm-embedding from-sequences chain --fasta-file seqs.fasta --output-folder out/seq-chain \
    --output-format parquet --output-codes-folder out/seq-codes
# out/seq-chain/inference-0.parquet (chain embeddings), out/seq-codes/inference-codes-0.parquet (residue codes)
fm-search build embeddings --embedding-folder out/seq-chain --output-db db/chains
```

Run `fm-embedding [from-structures|from-sequences] [command] --help` for full options (batch size, accelerator, devices, output format, distributed settings, etc.).

### `fm-search` — build and query FAISS databases

```bash
# Build a similarity-search database from structures, FASTA, or pre-computed embeddings
fm-search build structures  --structure-folder data/pdb --output-db dbs/my_db --tmp-embedding-folder tmp
fm-search build sequences   --fasta-file seqs.fasta     --output-db dbs/my_db --tmp-embedding-folder tmp
fm-search build embeddings  --embedding-folder out/chain --output-db dbs/my_db

# Query the database (--output-file is required)
fm-search query structure   --db-path dbs/my_db --query-structure q.cif --output-file hits.tsv
fm-search query sequences   --db-path dbs/my_db --fasta-file q.fasta --tmp-embedding-folder tmp --output-file hits.tsv
fm-search query embedding   --db-path dbs/my_db --embedding-folder out/chain --output-file hits.tsv
fm-search query db          --query-db-path dbs/queries --subject-db-path dbs/my_db --output-file hits.tsv

# Inspect, cluster, export
fm-search stats             --db-path dbs/my_db
fm-search cluster           --db-path dbs/my_db --output-file clusters.tsv
fm-search cluster           --db-path dbs/my_db --output-file clusters.tsv --method greedy
fm-search cluster           --db-path dbs/my_db --output-file clusters.tsv --method greedy-leiden --work-folder work
fm-search similarity-graph  --db-path dbs/my_db --output-file graph.graphml
```

All `build` commands accept `--index-type [auto|flat|hnsw|ivf_pq]` and IVF-PQ tuning flags (`--ivf-nlist`, `--ivf-nprobe`). See `fm-search <subcommand> --help` for the full surface.

`cluster` and `similarity-graph` work on every database type (`--method greedy-leiden` and `--use-gpu-index` on `flat` and `ivf_pq` only). On `ivf_pq` a score is estimated from the PQ codes (the chain's decoded embedding, normalised, against each stored code, as in `query db`), so two identical chains score close to, but not exactly, 1. With `--method leiden` (the default) the k-NN search takes time in proportion to the number of chains and `--max-neighbors`, whatever `--threshold`; `--method greedy` searches only chains no representative has taken, so its time also depends on `--threshold` and on database order. The graph's memory grows with the number of edges, which `--threshold` and `--max-neighbors` bound (at most `--max-neighbors` edges from each chain's search, plus the rescued edges below), and the Leiden step's memory also grows with the number of chains. The Leiden step runs a fixed number of iterations of Leiden (modularity at `--resolution`), 10 unless `--leiden-iterations` says otherwise; `--seed` makes it repeatable. With the default `--leiden-backend igraph` (python-igraph) it runs on one CPU thread. `--leiden-backend networkit` runs NetworKit's ParallelLeiden on `--leiden-threads` threads (default: every OpenMP thread; 1 on macOS, where more are refused), 5-25x faster on large graphs at 64 threads on Linux (126 s against 2,111 s on a 10M-chain graph); there `--leiden-iterations` is a maximum, and the same `--seed` gives the same clusters only with `--leiden-threads 1` (on more threads `--seed` logs a WARNING; repeats at 64 threads agreed at AMI 0.984-0.996 on CATH and Swiss-Prot graphs, 0.795-0.805 on decoy-heavy 10M- and 30.4M-chain graphs). It needs `pip install 'foldmatch[networkit]'`; without it the run is refused before the database is loaded.

**Edges.** `--edge-rule` (`leiden` and `greedy-leiden`, and `similarity-graph`) says which k-NN hits become edges. `lower` keeps an edge (i, j), i < j, only from the search of the lower chain i, the graph of FoldMatch up to 0.12.0. A chain whose own hits at or above `--threshold` are all lower chains, and that no lower chain's search keeps, gets no edge that way; such chains sit mostly at the end of the database, and their number grows with the database (11,503 chains, 0.115%, at 10M TrEMBL chains and `--max-neighbors 100`, 6.8x the count at 1M). `rescue`, the default, adds for each such chain an edge to each of its own hits at or above `--threshold`, weighted by its own score, in the same pass. On Swiss-Prot (573,010 chains, `--threshold 0.8`) it removed 4,616 / 1,892 / 551 / 207 singletons at `--max-neighbors` 30 / 50 / 100 / 300 for +1.59% / +0.62% / +0.14% / +0.012% edges, and the partitions agreed with `lower`'s at AMI 0.987 / 0.995 / 0.994 (30 / 50 / 100).

**Defaults by method.** `--max-neighbors` is 1000 under `leiden` and `greedy`, 100 under `greedy-leiden` (the k of its representatives' graph). `--resolution` is 1.0 under `leiden` and 20 under `greedy-leiden`. Options a method does not take are refused in one message before the database is loaded, naming each one and the methods that take it (`--resolution`, `--leiden-iterations`, `--seed`, `--edge-rule`, `--leiden-backend` and `--leiden-threads` under `greedy`; `--greedy-threshold` and `--greedy-max-neighbors` outside `greedy-leiden`; `--work-folder` under `leiden`).

**Search depth.** `--nprobe` (`cluster` under every method, and `similarity-graph`) sets the IVF cells probed per query of an `ivf_pq` database; the default is the value stored at build (`--ivf-nprobe`, or `--ivf-nlist` // 64). Higher finds more neighbours and is slower. On `flat` and `hnsw` databases it is refused.

**GPU.** `--use-gpu-index` (needs FAISS with CUDA GPU support, `faiss-gpu`) puts one replica of the index on every visible GPU (each search call's queries are split between them), with the PQ lookup tables of an `ivf_pq` index in float16 (float32 tables of an `ivf_pq` index of 64 sub-quantizers of 8 bits are refused on an A100). The stored vectors are read from the CPU index, which stays in memory, and `greedy-leiden` copies its representatives' database from it before moving that database to the GPUs too. On 4 A100s FAISS searched 28,188-32,607 queries a second at nlist 16,384, nprobe 256, k = 1000 on a 30.4M-chain database, against 1,011-1,752 on the node's 64 CPU cores. The float16 tables move scores slightly: GPU graphs of a 10M- and a 30.4M-chain database had 0.02% fewer and 2.07% more edges than the CPU graphs (edge Jaccard 0.96 and 0.91), and on a 10M-chain TrEMBL sample `--method greedy-leiden` on the GPUs kept 26 more representatives than on the CPU (1,886,496 against 1,886,470), and its label AMIs (Pfam clan and family, Gene3D, SUPERFAMILY, UniRef50) were within 0.0007 of the CPU's at `--resolution` 10 and 20, with greedy 7.5x and the graph 9.9x faster than on the same node's CPU. `--max-neighbors`, `--greedy-max-neighbors` and `--nprobe` are refused above 2048 (FAISS's GPU limit), and without GPU support the run is refused (exit status 1) before the database is loaded; macOS has none.

`cluster --method greedy` clusters collections whose similarity graph does not fit in memory. When most chains have `--max-neighbors` neighbours above `--threshold`, the Leiden route keeps about `n x --max-neighbors / 2` edges for `n` chains: 447,547,205 edges for 1,000,000 chains in an `ivf_pq` database (`--max-neighbors 1000`, `--threshold 0.8`), about 15.1e9 for 30.4M chains (`ivf_pq`); with fewer neighbours above the threshold, fewer (219,292,519 for the same 1,000,000 chains on `hnsw`, where few chains reach `--max-neighbors`). Leiden's memory grows with the edges (on that 1M-chain `ivf_pq` database, `cluster` peaked at 28.16 GiB, about 66 bytes an edge above the memory before the load, on Linux x86_64 at 128 threads), greedy's with the chains: it picks representatives in one streamed pass over the database, dropping each block's search results once used. A cluster is a representative and the still-unclustered chains among its `--max-neighbors` nearest that score at least `--threshold` (MMseqs2/CD-HIT style): chains are taken in database order, and a chain in no cluster yet becomes a representative. When most representatives have `--max-neighbors` neighbours above the threshold (the statistics report the share; 98.96% of rows in the first 10 blocks of a 30.4M-chain `ivf_pq` database), clusters are bounded by `--max-neighbors`, and a representative or singleton can score above the threshold with an earlier representative. Greedy clusters are smaller and tighter than Leiden's few large communities at 0.8: higher family purity, lower completeness and family ARI (measured at 100k and 1M chains). Results depend on database order (the order chains were added), as CD-HIT's depend on length order. `--max-neighbors` caps cluster size at `--max-neighbors` + 1 chains (on `hnsw`, FAISS's `efSearch` of 16 returns fewer: 481.7 of 1000 on average on a 100k-chain database, so a larger value adds few). Greedy needs no graph and computes no modularity; it has no resolution or iteration count and is deterministic, so `--resolution`, `--leiden-iterations`, `--seed` and the other Leiden options are refused with it. `--work-folder` keeps greedy's state for a resume (below).

`cluster --method greedy-leiden` combines the two for collections whose direct graph does not fit in memory: greedy representatives at `--greedy-threshold` (default 0.9) with `--greedy-max-neighbors` (default 1000, also the greedy cluster-size cap), then Leiden on the representatives' similarity graph at `--threshold` (default 0.8), `--max-neighbors` (default 100), `--edge-rule` and `--resolution` (default 20); every chain takes its representative's cluster. The representatives' database is copied from the parent, with nothing trained or re-encoded: on `ivf_pq` their PQ codes in the parent's lists, under its trained OPQ, coarse quantizer and PQ, so their search is the parent's search restricted to them, at the same `--nprobe`; on `flat` their stored vectors (held in memory, so for small sets). An `hnsw` database (an `auto` one that is hnsw included) is refused: use `--method leiden` there. The resolution default 20 was chosen on TrEMBL samples of 1M and 10M chains (greedy 0.9, `--max-neighbors` 100): it left no cluster above 1.745% of the chains at 10M (at most one at 1M, against 4-7 at resolution 1) and agreed with Pfam families better than 10 (+0.008 to +0.014 AMI), at a cost of up to 0.007 on Gene3D and SUPERFAMILY; a lower value such as 10 keeps more superfamilies whole. With `--greedy-threshold` above every score every chain is its own representative and the result is `--method leiden`'s at the same settings.

`--work-folder` is required by `greedy-leiden` (optional with `greedy`, refused with `leiden`). It holds `run.json` (the parent database's fingerprint, the effective nprobe, that is the cells probed, and the settings of each stored step, as resolved, and each run's steps with FAISS's thread count and GPUs), greedy's state (written every 15 minutes of greedy and at its end), the representatives' database and the representatives' graph's edges, each step with a `done` marker. Running the same command again resumes greedy from its last state and reuses every finished step; Leiden, the propagation to the chains and the export always run, so a rerun at another `--resolution`, `--leiden-iterations`, `--seed`, `--leiden-backend` or `--leiden-threads` searches nothing. A run with another recorded setting or another parent database is refused, naming each difference; another FAISS thread count, a change between CPU and GPU search, or another FoldMatch or FAISS version is a WARNING (the graph can differ by a few edges).

**The layout confound.** The share of representatives depends on the IVF layout as well as on `--greedy-threshold`: the same 1M TrEMBL chains gave 265,039 representatives at nlist 16,384 and 359,379 at nlist 4,096. Runs on databases of different `--ivf-nlist` are not comparable, and a resolution tuned at one layout may not transfer to another (the resolution default above was measured at nlist 16,384).

**Giants.** The statistics log gives, under every method, the largest cluster's size and share of the chains, the number of giant clusters (more than 1.745% of the chains) and the 10 largest cluster sizes. Under `greedy-leiden`, with 1,000,000 chains or more, giants are also a WARNING that a higher `--resolution` splits large clusters. Under `leiden` there is no such warning: on direct graphs every measured run at the default resolution had giants, and a higher resolution there cost domain labels (at 10M TrEMBL chains, resolution 10 lowered Gene3D agreement by 0.026 and SUPERFAMILY by 0.032 against resolution 1), so a giant there is not by itself a reason to raise it. On small or redundant sets a 1.745% cluster is ordinary.

#### Clustering very large collections

A recipe for one to a few hundred million chains, with `greedy-leiden` on one GPU node (64 CPU cores, 256 GB, 4 GPUs). Its figures for 200M chains are **projections** from measurements at up to 10M (TrEMBL) and 30.4M chains, not measurements.

1. Put the FASTA in a random order (for example sorted by a salted hash of the id) before the build: an `ivf_pq` build trains on the first 1,000,000 vectors of its input.
2. Build with `--index-type ivf_pq --ivf-nlist 16384`, the layout the greedy-leiden defaults were measured on.
3. In each job, set the FAISS threads with `OMP_NUM_THREADS` (for example 64; the thread count changes a graph by a few edges, so use the same one for reruns that must match), copy the database's `.index`, `.invlists` and `.metadata` files to node-local storage, and put `--work-folder` on scratch (tens of GiB at 200M chains). `--use-gpu-index` below searches on the node's GPUs (see *GPU* above for how its results compared with the CPU's at 10M chains); leave it out to search on the CPU:
   ```bash
   export OMP_NUM_THREADS=64
   fm-search cluster --db-path /local/db/my_db --method greedy-leiden --work-folder $SCRATCH/clusters-work \
       --leiden-backend networkit --leiden-threads 64 --use-gpu-index --output-file clusters.tsv
   ```
4. After greedy the log gives the number of representatives R, the bound of `--max-neighbors` x R edges of their graph and Leiden's memory at that bound for each backend. Expect up to about 0.47 x k x R edges (0.46 x k a chain on a direct 1M-chain TrEMBL graph at k = 100; 39.3 edges a representative on a 10M-chain cascade), so plan Leiden's node from the bound and that ratio.
5. A job that stops (wall-time limit, failure) is resumed by the same command; chain the jobs with `sbatch --dependency=afterany:<job id>`.
6. Other resolutions: the same command with another `--resolution` and the same `--work-folder`; only Leiden, the propagation and the export run.

Projections for 200M chains (unmeasured, from runs at 10M and 30.4M chains): 24M-38M representatives; 1.0-1.6e9 edges in their graph at k = 100; 12-19 GB of edge files and 15-23 GB in the work folder in all; NetworKit's Leiden 84-117 GiB, 5-8 minutes a run at 64 threads (igraph 51-81 minutes); the two searches about 65-200 hours on the CPU at nprobe 256 (12-34 hours at nprobe 64), about 4-7 hours on the node's 4 GPUs at nprobe 256.

#### Output format

Every command writes **tab-separated rows** to `--output-file` — one record per line. `query` results are ordered best-first per query, so rank is simply the row order.

`--output-file` cannot be one of the command's inputs: a file of a database it reads (`<db path>.index`, `.metadata`, `.invlists`, `.sequences` or `.rescodes.*`), the query FASTA or structure file, or a query embedding file; such a run is refused before any work. A file written by an earlier command is replaced. `query embedding` refuses query embeddings that share an id (results are kept by query id).

`--format-mode` selects the file layout:

| Mode | Layout |
|---|---|
| `headless_tsv` *(default)* | Tab-separated, **no header row** |
| `tsv` | Tab-separated, with a header row naming each column |

```bash
# Add a header row naming the columns
fm-search query sequences --db-path dbs/my_db --fasta-file q.fasta \
    --tmp-embedding-folder tmp --output-file hits.tsv --format-mode tsv
```

The header always names the columns actually emitted, in the order emitted — for a
sequence-identity search that is your `--format-output` selection. `--format-mode`
controls only the file's shape; `--format-output` controls which columns it holds.

An **embedding-only** search — `query structure`, `query embedding`, or any query whose database has no sequence store — emits three columns:

```
query   target   embscore
```

A **sequence-identity** search (Stage 2, below) emits the columns chosen with `--format-output`, which default to those same three followed by the alignment statistics.

`cluster` uses the same convention with its own columns:

```
chain_id   cluster_id   cluster_size
```

and, with `--method greedy` and `--method greedy-leiden`, a fourth column, the chain's greedy representative's chain id (equal to `chain_id` on the representative's own row):

```
chain_id   cluster_id   cluster_size   representative
```

Under `greedy` the cluster is the representative's greedy cluster. Under `greedy-leiden` `cluster_id` and `cluster_size` are the final cluster (the Leiden cluster of the chain's representative, counted in chains), and `representative` is still the greedy representative, so the chains of one greedy cluster share a `cluster_id`. `cluster_id` 0 is the largest cluster, under every method (ties under greedy in representative order, under `leiden` and `greedy-leiden` by the smallest chain index). Rows are in database order; `--min-cluster-size` drops the rows of smaller clusters (under `greedy-leiden`, by the final size).

(`similarity-graph` is the one exception: GraphML is a graph format, not tabular.)

#### Two-stage sequence search (exact identity)

`build sequences` also writes a sidecar `{db}.sequences` store next to the FAISS index. This lets sequence-built databases report **exact** sequence identity, not just embedding similarity: when you run `query sequences` (or `query db`) against such a database, a second stage pairwise-aligns each embedding hit (local Smith-Waterman, BLOSUM62) and re-ranks the surviving hits by bitscore.

Stage 2 aligns on FoldMatch's native SIMD Smith-Waterman kernel, a compiled library (see *Requirements*). It reproduces biotite 1.7.1's `align_optimal` (local, affine gaps), ties included, which Stage 2 used before: the same scores, coordinates and alignments, so the same output. Without the kernel, `query sequences` and `query db` refuse to run Stage 2 before the queries are embedded or the database is searched, and print the command that rebuilds it; `--no-seq-identity` still runs an embedding-only search.

Stage-2 hits use the same **tab-separated, no-header** format as every other query output. Pick the columns with `--format-output`:

```bash
# Stage 2 turns on automatically when the database has a sequence store
fm-search query sequences --db-path dbs/my_db --fasta-file q.fasta \
    --tmp-embedding-folder tmp --output-file hits.tsv

# Choose the output columns
fm-search query sequences --db-path dbs/my_db --fasta-file q.fasta \
    --tmp-embedding-folder tmp --output-file hits.tsv \
    --format-output "query,target,evalue,qaln,taln"
```

Default columns: `query,target,embscore,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits`.

Supported `--format-output` fields:

| Field | Meaning | Field | Meaning |
|-------|---------|-------|---------|
| `query` | Query sequence identifier | `tlen` | Target sequence length |
| `target` | Target sequence identifier | `alnlen` | Alignment length (aligned columns) |
| `embscore` | Stage-1 embedding similarity score | `raw` | Raw alignment score |
| `evalue` | E-value | `bits` | Bit score (integer) |
| `gapopen` | Number of gap **open events** (not gap chars) | `cigar` | CIGAR string (`M` match, `D` gap in query, `I` gap in target) |
| `pident` | Percentage of identical matches | `qseq` / `tseq` | Query / target sequence |
| `fident` | Fraction of identical matches | `qaln` / `taln` | Aligned query / target sequence with gaps |
| `nident` | Number of identical matches | `mismatch` | Number of mismatches |
| `qstart` / `qend` | 1-indexed alignment bounds in query | `qcov` / `tcov` | Fraction of query / target covered |
| `qlen` | Query sequence length | | |
| `tstart` / `tend` | 1-indexed alignment bounds in target | | |

- **Auto by default**: Stage 2 runs when the database(s) carry a sequence store and falls back to embedding-only otherwise. Force it with `--seq-identity` (errors if no store is present) or disable with `--no-seq-identity`. `query db` requires **both** databases to have sequence stores.
- Hits below `--min-seq-identity` (default `0.3`) or `--min-coverage` (default `0.8`) are dropped, as are hits above `--max-evalue` (default `1e-3`; pass `inf` to keep every hit). `--max-evalue` turns the significance pass on even when neither `evalue` nor `bits` is in `--format-output`.
- Tuning: `--gap-open` (default `11`) and `--gap-extend` (default `1`), where a gap of length L costs gap-open + (L − 1) × gap-extend, and `--align-workers` (defaults to all CPUs on the node). `--gap-open` must be at least 1 and `--gap-extend` at least 0, both at most 1,048,576; the sign is ignored. With `--significance-mode default`, a run that needs significance (an `evalue` or `bits` column, or a finite `--max-evalue`, which the default `1e-3` is) needs the gap costs 11/1: for other gap costs use `--significance-mode sampled`, or `--max-evalue inf` with neither column in `--format-output`. Both rules are checked before the queries are embedded (`query sequences`) or the database is searched (`query db`).
- Compositional bias correction: off by default, so alignments are scored with plain BLOSUM62. `--comp-bias-corr` (scaled by `--comp-bias-corr-scale`, default `1`) applies mmseqs's correction for locally biased amino-acid composition, which mmseqs enables by default; use it when comparing with `mmseqs search`. With the same setting on both tools, raw scores, bit scores and E-values match `mmseqs search` (18-8cc5c) on the same pairs, except for alignments containing a non-standard residue (`X`, `B`, `Z`, `J`, `U`, `O`, `*`), which the two tools score differently. The correction can shorten alignments over biased regions, so coordinates, identity and coverage change too.
- `evalue`/`bits` use calibrated Karlin–Altschul statistics by default (`--significance-mode default`). `--significance-mode sampled` estimates λ/K by sampling for other gap penalties, but its absolute magnitudes are relative-only. The significance pass is skipped automatically when neither `evalue` nor `bits` is in `--format-output`.

---

## Python API

The `FoldMatch` class provides methods for computing embeddings programmatically.

### Basic Usage

```python
from foldmatch import FoldMatch

# Initialize model
model = FoldMatch(min_res=10, max_res=5000)

# Load models (optional - loads automatically on first use)
model.load_models()  # Auto-detects CUDA
# or specify device:
# import torch
# model.load_models(device=torch.device("cuda:0"))
```

A protein chain is embedded only if it has at least `min_res` residues (default 10), counted as polymer amino-acid residues: free amino acids are not counted. With `max_res` (default `None`, no limit), a structure method raises `ValueError`, before any ESM3 pass, when the protein chains it is about to embed have more than `max_res` residues together, counted the same way: the chains with `chain_id` (in an assembly, one per copy), else every protein chain of the structure or assembly. `sequence_embedding` raises `ValueError` when the sequence is shorter than `min_res` or longer than `max_res`.

### Methods

#### `load_models(device=None)`

Load both residue and aggregator models.

```python
import torch
model.load_models(device=torch.device("cuda"))
```

---

#### `load_residue_embedding(device=None)`

Load only the ESM3 residue embedding model.

```python
model.load_residue_embedding()
```

---

#### `load_aggregator_embedding(device=None)`

Load only the aggregator model.

```python
model.load_aggregator_embedding()
```

---

#### `residue_embedding(src_structure, structure_format='mmcif', chain_id=None, assembly_id=None)`

Compute per-residue embeddings for a structure.

**Parameters:**
- `src_structure`: File path, URL, or file-like object
- `structure_format`: `'mmcif'`, `'binarycif'`, or `'pdb'`
- `chain_id`: Chain ID (optional): the protein chains with this ID (in an assembly, one per copy); every protein chain if None
- `assembly_id`: Assembly ID for biological assembly (optional)

**Returns:** `torch.Tensor` of shape `[L + 2, 1536]` for one chain of `L` residues: one row per residue, between ESM3's BOS and EOS rows. Several chains give their tensors one after another, each with its BOS and EOS rows. Raises `ValueError` when no protein chain qualifies, or when `max_res` is exceeded.

```python
# Single chain
residue_emb = model.residue_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    chain_id="A"
)

# All chains concatenated
all_residues = model.residue_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif"
)

# Biological assembly
assembly_residues = model.residue_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    assembly_id="1"
)
```

---

#### `residue_embedding_by_chain(src_structure, structure_format='mmcif', chain_id=None)`

Compute per-residue embeddings separately for each chain.

**Returns:** `dict[str, torch.Tensor]` mapping each chain ID to its embeddings, of shape `[L + 2, 1536]`; `{}` when no chain qualifies

```python
chain_embeddings = model.residue_embedding_by_chain(
    src_structure="1abc.cif",
    structure_format="mmcif"
)
# Returns: {'A': tensor(...), 'B': tensor(...), ...}

# Get specific chain
chain_a = model.residue_embedding_by_chain(
    src_structure="1abc.cif",
    chain_id="A"
)
```

---

#### `residue_embedding_by_assembly(src_structure, structure_format='mmcif', assembly_id=None)`

Compute residue embeddings for an assembly.

**Returns:** `dict[str, torch.Tensor]` mapping the assembly ID (`'0'` for the asymmetric unit when `assembly_id` is None) to the tensors of its protein chains (one per copy) one after another, each with its BOS and EOS rows; `{}` when no chain qualifies

```python
assembly_emb = model.residue_embedding_by_assembly(
    src_structure="1abc.cif",
    structure_format="mmcif",
    assembly_id="1"
)
# Returns: {'1': tensor(...)}
```

---

#### `sequence_embedding(sequence)`

Compute residue embeddings from amino acid sequence (no structural information).

**Parameters:**
- `sequence`: Amino acid sequence string (plain or FASTA format)

**Returns:** `torch.Tensor` of shape `[L + 2, 1536]` for a sequence of `L` residues (ESM3's BOS and EOS rows included)

```python
# Plain sequence
seq_emb = model.sequence_embedding("ACDEFGHIKLMNPQRSTVWY")

# FASTA format
fasta = """>Protein1
ACDEFGHIKLMNPQRSTVWY
ACDEFGHIKLMNPQRSTVWY"""
seq_emb = model.sequence_embedding(fasta)
```

---

#### `aggregator_embedding(residue_embedding)`

Aggregate residue embeddings into a single structure-level vector.

**Parameters:**
- `residue_embedding`: `torch.Tensor` from any of the residue embedding methods (for `residue_embedding_by_chain` and `residue_embedding_by_assembly`, a value of their dict)

**Returns:** `torch.Tensor` of shape `[1536]`

```python
residue_emb = model.residue_embedding("1abc.cif", chain_id="A")
structure_emb = model.aggregator_embedding(residue_emb)
```

---

#### `structure_embedding(src_structure, structure_format='mmcif', chain_id=None, assembly_id=None)`

End-to-end: compute residue embeddings and aggregate in one call.

```python
# Complete structure embedding
structure_emb = model.structure_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    chain_id="A"
)
# Returns: tensor of shape [1536]
```

---

### Complete Example

```python
from foldmatch import FoldMatch
import torch

# Initialize
model = FoldMatch(min_res=10, max_res=5000)

# Option 1: Full structure embedding (one-shot)
embedding = model.structure_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    chain_id="A"
)

# Option 2: Step-by-step with residue embeddings
residue_emb = model.residue_embedding(
    src_structure="1abc.cif",
    structure_format="mmcif",
    chain_id="A"
)
structure_emb = model.aggregator_embedding(residue_emb)

# Option 3: Process multiple chains
chain_embeddings = model.residue_embedding_by_chain(
    src_structure="1abc.cif"
)
for chain_id, res_emb in chain_embeddings.items():
    chain_emb = model.aggregator_embedding(res_emb)
    print(f"Chain {chain_id}: {chain_emb.shape}")

# Sequence-based embedding
seq_emb = model.sequence_embedding("ACDEFGHIKLMNPQRSTVWY")
structure_from_seq = model.aggregator_embedding(seq_emb)
```

See the `examples/` and `tests/` directories for more use cases.

---

## Model Architecture

The embedding model is trained to predict structural similarity by approximating TM-scores using cosine distances between embeddings. It consists of two main components:

- **Protein Language Model (PLM)**: Computes residue-level embeddings from a given 3D structure.
- **Residue Embedding Aggregator**: A transformer-based neural network that aggregates these residue-level embeddings into a single vector.

![Embedding model architecture](assets/embedding-model-architecture.png)

### **Protein Language Model (PLM)**
Residue-wise embeddings of protein structures are computed using the [ESM3](https://www.evolutionaryscale.ai/) generative protein language model.

### **Residue Embedding Aggregator**
The aggregation component consists of six transformer encoder layers, each with a 3,072-neuron feedforward layer and ReLU activations. After processing through these layers, a summation pooling operation is applied, followed by 12 fully connected residual layers that refine the embeddings into a single 1,536-dimensional vector.

---

## Testing

After installation, run the test suite:

```bash
pytest
```

### macOS notes

**The problem.** PyPI wheels for `faiss-cpu` and `torch` (pulled in via `lightning`) each bundle their own copy of `libomp.dylib`. On macOS, both copies get loaded into the same Python process. Whenever FAISS enters an OpenMP-parallel section (batched search with more than one query vector, `IndexHNSWFlat` graph construction, IVF-PQ training) the second OpenMP runtime fails to `pthread_mutex_init` and the call deadlocks — the CLI appears to hang indefinitely — or the process crashes (a segmentation fault at `OMP_NUM_THREADS=4` on an M1 Max, with torch 2.11 and `faiss-cpu` 1.13.2 or 1.15.1). Linux installs are unaffected: there the copies are GNU `libgomp` builds, which can be loaded together, and FoldMatch's commands, which load torch first, run FAISS on torch's copy.

**Affected commands** on macOS without mitigation:

- `fm-search build` with `--index-type hnsw` or `auto` past ~10k vectors, and any `--index-type ivf_pq`.
- `fm-search query embedding` with a multi-row `.parquet` file.
- `fm-search query sequences` with more than one input sequence.
- `fm-search query db` (database-to-database).
- `fm-search cluster` and `fm-search similarity-graph`, which search many chains per FAISS call (safe at the default `OMP_NUM_THREADS=1` pin).

`fm-search cluster --leiden-backend networkit` works on macOS 26 arm64 (NetworKit 11.3 ships its own `libomp.dylib` there): with torch and FAISS loaded and the default pin, a FAISS search, ParallelLeiden and the same search again gave the same results in one process. It runs on 1 thread there, and `--leiden-threads` above 1 is refused before the database is loaded: with torch and FAISS loaded, NetworKit's copy of `libomp.dylib` (a third one) failed its first parallel region on 2 or 3 threads (OMP Error #179) and crashed the process with a segmentation fault, after the searches. `--use-gpu-index` is refused on macOS: the `faiss-cpu` wheel's Metal backend is not FAISS's CUDA GPU layer, and the commands that fall back to the CPU without a GPU do so there.

Single-query paths (`fm-search query structure`, small `--index-type flat` builds) are unaffected.

**Possible fixes.**

1. **Fix the install environment** — install both libraries against a unified OpenMP runtime. On conda-forge:
   ```bash
   conda install -c conda-forge faiss-cpu pytorch llvm-openmp
   ```
   Once a single libomp is loaded, FAISS's parallel paths just work and you keep the full multi-threaded performance.

2. **Force single-threaded FAISS via environment variable** — set `OMP_NUM_THREADS=1` before invoking Python:
   ```bash
   export OMP_NUM_THREADS=1
   fm-search query db ...
   ```
   Sidesteps the parallel section entirely. Toolkit works, but FAISS runs single-threaded so large builds and queries are slower.

**What this package does by default.** To prevent macOS users from hitting a silent hang out of the box, `foldmatch/__init__.py` calls `os.environ.setdefault("OMP_NUM_THREADS", "1")` on `darwin` only — before any `torch` or `faiss` import. This is option 2 above, applied automatically. Linux installs are not touched (the branch is skipped). A user on macOS who has fixed their environment per option 1 can opt back into parallelism by exporting `OMP_NUM_THREADS=N` before launching Python — `setdefault` respects an existing value.

---

## Citation

Segura, J., et al. (2026). *Multi-scale structural similarity embedding search across entire proteomes*. (https://doi.org/10.1093/bioinformatics/btag058)

---

## License

This project uses the EvolutionaryScale ESM-3 model and is distributed under the
[Cambrian Non-Commercial License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement).