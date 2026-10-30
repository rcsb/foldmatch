# FoldMatch

**Version** 0.10.0


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
- **Two-stage sequence search** — an embedding prefilter followed by exact pairwise Smith-Waterman alignment, reporting sequence identity, coverage, and calibrated E-values over selectable output columns
- **Structural clustering** using the Leiden algorithm for biological assembly identification
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
- Biotite 1.6.0
- FAISS 1.15.1 (the `faiss-cpu` wheel)
- igraph 1.0.0
- leidenalg 0.11.0
- PyTorch with CUDA support (recommended for GPU acceleration)

**Optional Dependencies:**

- `faiss-gpu` for GPU-accelerated similarity search (instead of `faiss-cpu`)
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

The toolkit ships three CLIs. Each is invoked with `--help` for full option documentation; the canonical examples below are enough to get started.

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
fm-search similarity-graph  --db-path dbs/my_db --output-file graph.graphml
```

All `build` commands accept `--index-type [auto|flat|hnsw|ivf_pq]` and IVF-PQ tuning flags (`--ivf-nlist`, `--ivf-nprobe`). See `fm-search <subcommand> --help` for the full surface.

#### Output format

Every command writes **tab-separated rows** to `--output-file` — one record per line. `query` results are ordered best-first per query, so rank is simply the row order.

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

(`similarity-graph` is the one exception: GraphML is a graph format, not tabular.)

#### Two-stage sequence search (exact identity)

`build sequences` also writes a sidecar `{db}.sequences` store next to the FAISS index. This lets sequence-built databases report **exact** sequence identity, not just embedding similarity: when you run `query sequences` (or `query db`) against such a database, a second stage pairwise-aligns each embedding hit (local Smith-Waterman, BLOSUM62) and re-ranks the surviving hits by identity.

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
- Tuning: `--gap-open`, `--gap-extend`, and `--align-workers` (defaults to all CPUs on the node).
- Compositional bias correction: off by default, so alignments are scored with plain BLOSUM62. `--comp-bias-corr` (scaled by `--comp-bias-corr-scale`, default `1`) applies mmseqs's correction for locally biased amino-acid composition, which mmseqs enables by default; use it when comparing with `mmseqs search`. With the same setting on both tools, raw scores, bit scores and E-values match `mmseqs search` (18-8cc5c) on the same pairs, except for alignments containing a non-standard residue (`X`, `B`, `Z`, `J`, `U`, `O`, `*`), which the two tools score differently. The correction can shorten alignments over biased regions, so coordinates, identity and coverage change too.
- `evalue`/`bits` use calibrated Karlin–Altschul statistics by default (`--significance-mode default`). `--significance-mode sampled` estimates λ/K by sampling for other gap penalties, but its absolute magnitudes are relative-only. The significance pass is skipped automatically when neither `evalue` nor `bits` is in `--format-output`.

### `inference` — low-level inference subcommands

Lower-level entry point exposing individual inference passes (`residue-embedding`, `structure-embedding`, `chain-embedding`, `assembly-embedding`, `complete-embedding`). Mostly useful for advanced workflows that compose inference stages explicitly. Run `inference --help` for the command list.

---

## Python API

The `RcsbStructureEmbedding` class provides methods for computing embeddings programmatically.

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
- `chain_id`: Specific chain ID (optional, uses all chains if None)
- `assembly_id`: Assembly ID for biological assembly (optional)

**Returns:** `torch.Tensor` of shape `[num_residues, embedding_dim]`

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

**Returns:** `dict[str, torch.Tensor]` mapping chain IDs to embeddings

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

**Returns:** `dict[str, torch.Tensor]` mapping assembly ID to concatenated embeddings

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

**Returns:** `torch.Tensor` of shape `[sequence_length, embedding_dim]`

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
- `residue_embedding`: `torch.Tensor` from residue embedding methods

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