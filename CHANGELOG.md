# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, breaking changes are released in a minor bump.

## [Unreleased]

### Added

- **`--leiden-iterations`** on `fm-search cluster`: the number of Leiden
  iterations, a fixed count (default 10, at least 1; below 1 it is refused
  with exit status 2 before the database is loaded). Fewer iterations are
  faster, with somewhat noisier partitions: on a 1M-chain graph
  (`--max-neighbors` 50), the Leiden step took 26.9-28.4 s at 2 iterations
  against 77.7-78.5 s at 10 (one thread, NERSC CPU nodes), and two seeds
  agreed at a mean AMI of 0.7548 against 0.7659 (10 pairs of 5 seeds).
  `--method leiden` only: refused under `--method greedy`, with
  `--resolution` and `--seed`, in one message.
- **`--cleanup-embeddings`** on `fm-search build structures` and `fm-search
  build sequences`: deletes the build's intermediate embedding shards from
  `--tmp-embedding-folder` once the database is built. Off by default, so the
  shards are kept as before.
- **ESM3 residue-code scoring: the building blocks.** The parts of the
  opt-in Stage-2 scoring `esm-codes-v1`, an alternative to BLOSUM62: the
  residue-code encoders
  (`foldmatch.model.residue_code_encoder`), which turn ESM3 per-residue
  embeddings into 8-byte residue codes and query encodings with the files of
  the Hugging Face repo `rcsb/foldmatch-esm-codes`; a native C
  Smith-Waterman kernel with its Python wrapper (`foldmatch.search.swkern`);
  and the versioned scoring system (`foldmatch.search.scoring_system`), a
  frozen registry entry `esm-codes-v1` that names every number and
  convention behind the scores, including the bundle's repo, pinned revision
  and file sha256s (about 26 MB), with its E-value statistics
  (`foldmatch.search.code_statistics`). A changed value will always be a new
  system name. The database build (`--residue-codes`) and the query
  (`fm-search query sequences --scoring esm-codes`), both below, use them;
  existing commands, BLOSUM62 scoring and its output are unchanged; and the
  Stage-2 aligner is `foldmatch.search.code_alignment`
  (`align_candidates_esm_codes`). `python -m foldmatch.search.swkern` runs the kernel's
  self-test (reference hash and the BLAS similarity path) and prints a JSON
  report.
- **`fm-search build sequences --residue-codes`** (opt-in, off by default).
  The ESM3 pass that computes the chain embeddings also computes an 8-byte
  ESM3 code per residue with the `esm-codes-v1` encoders, and the build
  stores them next to the database: `<output-db>.rescodes.bin` (8 bytes per
  residue), a SQLite index `<output-db>.rescodes.idx` (about 60 bytes per
  entry) and a manifest `<output-db>.rescodes.json`, written last, that
  records the scoring system, the ESM3 and bundle pins, the entry and
  residue counts, the sha256 of the codes, and how the pass computed them
  (device type and ESM3 precision, number of ranks, software versions;
  the embedding shards record the same). The first such build downloads
  the scoring bundle (about 26 MB; `download-models` fetches it in advance).
  Before ESM3 is loaded, and on every rank alike, the build refuses FASTA
  input the codes cannot be lined up with (duplicate ids, `<` or `>` in a
  sequence, non-ASCII characters), any existing output (`.rescodes.*`
  included) and a bundle that cannot be fetched or verified. After the pass
  the codes are checked against the sequence store, the pre-flight counts
  and the FAISS index; if a check fails, only the code store is removed, so
  the database still serves BLOSUM62 searches, and the embedding shards are
  kept for `build residue-codes`. FASTA (sequence) databases only. Such a
  database is searched with its codes by `query sequences --scoring
  esm-codes` (below), and with BLOSUM62 as before. Without the option the
  build and all its files are unchanged.
- **`fm-search build residue-codes DB_FOLDER INDEX_NAME --from-shards DIR`**
  rebuilds and validates the residue-code store of a database from the
  embedding shards its `build sequences --residue-codes` run kept in
  `--tmp-embedding-folder` (they are kept unless `--cleanup-embeddings`),
  without running ESM3 again; the codes are byte-identical to the original
  build's, and the manifest's `min_res_n` and build information come from
  the shards, so they describe the pass that computed the codes. `--run-id`
  picks the run when `DIR` holds the shards of several. `--force` replaces
  an existing store, but only once the new codes have passed validation: the
  new store is written next to the old one (room for both is needed
  meanwhile), and a refused or interrupted rebuild leaves the old one as it
  was. Shards without codes, of another encoding, or whose entry count
  differs from the database's are refused before anything is changed.
- **A split sequence build for multi-node GPU runs: `fm-search build
  sequences-embed`, `build sequences-index` and `build sequences-seal`.**
  After its ESM3 pass, `build sequences` builds the FAISS index, the sequence
  store and the residue-code store in one process, while the other nodes of
  the job stay allocated: in a build of 30 million sequences on 8 GPU nodes
  this phase took 7.4 of the 20.6 hours. The split build ends the GPU job with
  the pass and builds the database in a job of its own on one CPU node; `build
  sequences` itself is unchanged. Step 1, `build sequences-embed`, takes the
  input and pass options of `build sequences` (`--fasta-file`, `--min-res-n`,
  `--accelerator`, `--devices`, `--num-nodes`, `--strategy`, `--batch-size`,
  `--num-workers`, `--residue-codes`), runs its pre-flight and ESM3 pass on
  every rank and keeps the embedding shards in `--shards-folder` (the
  `--tmp-embedding-folder` of `build sequences`, renamed); rank 0 then checks
  them and writes, last, a run record `emb_<run id>.run.json` beside them,
  which holds the FASTA's size, sha256 and pre-flight counts, each shard's
  rows and size, the pass and `--output-db`, the database step 2 is to build
  (refused if any of its files exists). Step 1 writes no database file and
  logs the command of step 2. Step 2, `build sequences-index`, runs in one
  process, without ESM3, a GPU or a download: it builds from the run in
  `--from-shards` (`--run-id` picks one when the folder holds more than one
  run, unfinished ones included), with the index options of `build sequences`
  (`--index-type`, `--ivf-nlist`, `--ivf-nprobe`, `--load-batch-size`,
  `--use-gpu-index`), into `--output-db`, which may differ from the recorded
  one. Its `--residue-codes` defaults to what the run computed;
  `--no-residue-codes` leaves the code store out (`build residue-codes
  --from-shards --run-id` adds it later), and `--cleanup-embeddings` deletes
  the record, then the shards, once the database is built. Before it writes
  any database file, step 2 checks the record, that the shards are the
  recorded ones with their sizes and rows, their schema (with codes, those of
  the system the record names, with this FoldMatch's digests), the FASTA
  against the record (its size, then its sha256, then its pre-flight counts)
  and the id of every shard row against the FASTA entry the pass gave it. Step
  1 takes a stat of the FASTA on every rank before its pre-flight and after
  its pass, and step 2 around its own reads, so a FASTA that changes during
  either step is refused (step 2 then removes the files it wrote). A rank may
  read a node-local copy of the FASTA, which must be a byte copy of rank 0's:
  only rank 0's copy is hashed. A run embedded by another FoldMatch version is
  built with a warning. A FAISS refusal (an `ivf_pq` setting the data cannot
  take, say) costs no GPU time: rerun step 2 with other options. From the same
  shards, the two steps write the files of `build sequences` byte for byte,
  except `build.created_utc` and `build.run_id` in the residue-code manifest
  (and the copy in its index) and the added `build.split: true`; `hnsw` and
  `ivf_pq` indexes are byte-identical, as between any two builds, only at one
  thread, and an `ivf_pq` index holds the path of its `.invlists`, so only at
  the same `--output-db`. Every refusal of the three commands is an error
  with exit status 1 (2 for a usage error), with or without
  `--residue-codes`. The recipe: run `fm-search download-models` once on a
  login node first (no rank of a multi-node `--residue-codes` run downloads);
  run step 1 in the GPU job as `srun --kill-on-bad-exit=1 ... fm-search build
  sequences-embed ...`, with a generous wall time (step 1 cannot resume) and
  one shards folder per build on storage that every node can write and that
  outlives the job (such as `$SCRATCH`); and run step 2 as its own job on one
  CPU node, submitted with `sbatch -C cpu -N 1 --dependency=afterok:<job 1>
  --kill-on-invalid-dep=yes` and started without srun, with a generous wall
  time too. At 30.4 million sequences with `--residue-codes`, step 2 with the
  `faiss-cpu` 1.15.1 wheel FoldMatch now pins (see *Changed*) took 3 h 33 min
  on one CPU node (128 cores and FAISS threads): 2 h 37 min for the `ivf_pq`
  index, 16 min for the sequence store, 30 min for the residue-code store and
  4 min for the shard id check. On one GPU node (64 cores and threads) it took
  2 h 47 min, 1 h 45 min of it for the index, and with FAISS 1.15.1 built on
  MKL (the official `pytorch` conda channel's build; see the README) 2 h 02
  min, 59 min of it for the index: ask for 6 hours. The `faiss-cpu` 1.13.2
  wheel's single-threaded BLAS made the index take about 7 to 8 hours whatever
  the thread count; the index phase logs a line, a warning except on macOS,
  when FAISS's BLAS is single-threaded. A step 2 killed by its wall time
  cannot resume: it leaves `<output-db>.build.lock`
  and partial database files, which the next step 2 lists and which must be
  deleted before FAISS is rebuilt from scratch. A project with GPU hours only
  may run step 2 on one GPU node, where `ivf_pq` fits (`flat` and `hnsw` may
  not). To keep several builds in one folder, set `FOLDMATCH_RUN_ID` in job 1
  and pass the same value to step 2 as its `--run-id`. Step 2 and the seal
  refuse to run as one of several ranks, or in the batch step of a job of more
  than one node, and warn in a one-task srun step of such a job and in a
  one-node job with GPUs. They refuse before Lightning and torch are
  imported (which can take minutes on a compute node), and so do step 2's
  other refusals up to its index phase (the run record, the residue-code
  options, the build lock, existing database files, the shards and the FASTA
  against the record) and every refusal of the seal, which never imports
  them: within seconds of the launch once the file system has the
  environment's files cached (a first launch on a node can take a minute).
  The refusal in the batch step, like the srun step's warning, ends with the
  command line to run in the one-node CPU job,
  as it was invoked (`fm-search ...` through the console script), after a
  `cd` to the folder it ran in when the line depends on it (a relative path,
  or `python -m`); the refusal as one of several ranks ends with the command
  to launch once. The command is on a line of its own, indented, so that it
  can be copied whole. Step 2 sets FAISS's OpenMP threads to the
  physical cores it may use (and the threads of the BLAS libraries
  threadpoolctl finds, which it logs), unless `OMP_NUM_THREADS` (for BLAS also
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` or `VECLIB_MAXIMUM_THREADS`) is
  set; on macOS it leaves them as they are. On Linux, FAISS's own OpenBLAS,
  in the `faiss-cpu` wheel, is one of those libraries: an OpenMP build, it
  runs FAISS's OpenMP threads (at most 128 in the x86_64 wheel), so
  `OMP_NUM_THREADS` sets it and `OPENBLAS_NUM_THREADS` does not (the log line
  says so). With the wheel, the OPQ step of an `ivf_pq` training does not gain
  from more threads, which is most of why the CPU node above took longer than
  the GPU node. With the wheel, run step 2 with `OMP_NUM_THREADS=64` on a node
  of more than 64 physical cores (it sets the threads of the whole step 2; not
  timed at this scale on a 128-core node), and expect the warning that FAISS
  has fewer threads than cores; with the MKL build, leave it unset. It holds
  a lock, `<output-db>.build.lock`, while it builds, so a second step 2 into
  the same database is refused; a lock left by a killed job is reported with
  its content, the database files present and what to delete. `build
  sequences-seal --fasta-file F --from-shards S --run-id X` recovers shards
  that have no record: those of a step 1 whose rank 0 stopped after the pass
  (wall time, out of memory, a full disk; when step 1 itself refuses then, its
  error gives the command), and those a `build sequences` kept when its index
  phase failed. In one process and without ESM3, it checks them against the
  FASTA (the entry count, each rank's rows, the shard schema and the id of
  every row, and with residue codes each row's `seq_sha1` against the sha1 of
  its sequence) and writes their record, marked as sealed (step 2 then warns
  that the shards are bound by these checks, not by the pass, and takes the
  FoldMatch version of the pass from residue-code shards, the only ones that
  record it). A run without residue codes needs the `--min-res-n` it used,
  which its shards do not record. Shards written by FoldMatch 0.8.0 and
  earlier embedded the FASTA's sequences before they were normalised, so a
  database sealed from them matches a fresh build only when the FASTA's
  sequences are uppercase with no whitespace inside their lines; otherwise
  embed them again.
- **`fm-search query sequences --scoring esm-codes`** (opt-in; `--scoring
  blosum62`, the BLOSUM62 Stage 2 as before, stays the default). Stage 2
  aligns each query with its Stage-1 candidates through their ESM3 residue
  codes, on a database built with `build sequences --residue-codes`, using
  the scoring system its residue-code manifest names (`esm-codes-v1`). The
  ESM3 pass that embeds the queries also computes their residue encodings;
  the native kernel scores every candidate, the E-value of a hit is
  calibrated against a fixed panel of 1,000 UniRef50 sequences and computed
  over the database's residue count, and only the hits that pass
  `--max-evalue` are traced back for the alignment columns. Hits are ranked
  by score. One scoring per run. Under esm-codes, `--min-seq-identity` and
  `--min-coverage` default to 0 and `--max-evalue` to `1e-3`, as under
  BLOSUM62 (explicit values are honoured; the help gives both scorings'
  defaults), and the BLOSUM62 options `--gap-open`, `--gap-extend`,
  `--comp-bias-corr`, `--comp-bias-corr-scale`, `--significance-mode`,
  `--significance-sample-size`, as well as `--no-seq-identity`, are refused
  (exit status 2), whatever their value. Before ESM3 is loaded, and on every
  rank alike, the query is refused with a one-line message (exit status 1)
  when its FASTA has duplicate ids, `<` or `>` in a sequence, or non-ASCII
  characters; when the database was built without `--residue-codes`, or
  its codes are a partial build; when the codes were built for a scoring
  system this FoldMatch does not know (a database from a newer FoldMatch)
  or do not match the system or the database's sequence store; when the
  native kernel cannot be loaded (the message gives the command that builds
  it); and when the scoring bundle cannot be fetched or verified (only the
  process that starts a run downloads it). Once the index is loaded, codes
  whose entry count differs from the FAISS index are refused. A hit is
  inside the calibrated domain of the E-values (`ecal=1`) when the query
  has 21 to 966 residues, the target at most 3,000, both at most 1%
  non-standard residues and the target at least 2 distinct codes; other
  hits are still reported, with `ecal=0`, and one warning per run gives the
  counts. A query of one residue, or whose encoding rows are all identical,
  reports no hits. Stage 2 runs in a pool of worker processes that load
  neither torch nor ESM3, after ESM3 and the aggregator have been freed:
  `--align-workers N` uses N workers; by default the pool has up to one
  worker per physical core (SMT siblings count once) and starts only as many
  as the work needs, so a small query does not start a worker on every CPU
  (a later batch of queries with more work grows the pool). Its output does
  not depend on the number of workers. A worker needs about 5 bytes per
  residue pair of the pair it aligns (up to 1.3 GiB for the longest pairs it
  aligns, 2^28 residue pairs; longer ones are skipped and counted) and keeps
  at most 128 MiB of it between tasks; a worker that is killed (for example
  by the out-of-memory killer) stops the run with an error. The query
  encodings are written next to the query's embedding shards in
  `--tmp-embedding-folder` and deleted with them at the end of the run, also
  when it fails (a process whose ESM3 pass fails deletes the shard it wrote).
- **New `--format-output` fields `rawrev`, `q90`, `rawadj` and `ecal`**, filled
  by `query sequences --scoring esm-codes`: the score of the reversed query,
  the query's 0.9 quantile over the scoring panel, the adjusted score behind
  `evalue` and `bits` (`raw - rawrev - q90`), and whether the hit is inside
  the calibrated domain (`1`/`0`). Under esm-codes `raw` is the forward score
  with three decimals, `evalue` is computed in log space, so it never
  underflows, and `bits` is rounded half up (`floor(x + 0.5)`, also for a
  negative bit score). The four fields render empty under BLOSUM62 and in
  `query db`. BLOSUM62 columns, `bits` included, and the default columns are
  unchanged.
- **esm-codes Stage 2 speed** (measured on one AMD EPYC 7763 node, OpenBLAS,
  one BLAS thread per worker; the output does not depend on any of the
  settings below). On one core, scoring a 300-residue query against 1,000
  candidates and tracing back the hits that pass takes 0.25 s at
  `--max-evalue 1e-3` (none passed in this benchmark) and 0.49 s at
  `--max-evalue inf` (all 1,000 traced back); BLOSUM62's Stage 2 takes
  2.65 s on the same targets. The per-query panel pass that calibrates the
  E-value adds 0.11 s at 71 residues, 0.38 s at 300 and 1.17 s at 966. With
  100 candidates a query takes 0.15 / 0.40 / 1.29 s at 100 / 300 / 1,000
  residues on one core, 1.5-1.6x the BLOSUM62 time (codes are faster from
  about 160 candidates on), and 0.03 / 0.07 / 0.20 s on 8 workers. On 64
  workers, 640 queries x 1,000 candidates run at 128,000-154,000 candidate
  pairs per second at `1e-3` and 58,000-67,000 at `inf` (the lower ends
  with the workers pinned to 65 cores, the higher with the default
  affinity). A worker's memory stays under 70 MiB at 1,000 residues x 1,000
  candidates.
- **`fm-search download-models`**: the same command, with the same `--system`
  option and help, as `fm-embedding download-models` and `inference
  download-models`, since `fm-search` now uses the scoring bundle too.
- **`fm-search cluster --method greedy`**: greedy representatives in one
  streamed pass over the database, for collections whose similarity graph does
  not fit in memory. Chains are taken in database order; a chain in no cluster
  yet becomes a representative and takes the chains among its
  `--max-neighbors` nearest that score at least `--threshold` and are in no
  cluster yet. Each block's search results are dropped once used, so memory
  grows with the number of chains, not with the number of edges, and only
  chains no representative has taken are searched. Deterministic (no seed).
  Clusters have at most `--max-neighbors` + 1 chains and, when most
  representatives have `--max-neighbors` neighbours above `--threshold`, are
  bounded by `--max-neighbors` rather than by `--threshold` (the statistics
  report that share, with a warning when it is above half). Results depend on
  database order. The output gains a 4th column, `representative` (the
  representative's chain id), and cluster ids are numbered by decreasing size
  under both methods (`cluster_id` 0 is the largest cluster; Leiden's already
  were). `--method leiden`, the graph and Leiden as before, stays the default.
  Measured with this command: 30,430,281 chains, `ivf_pq`, `--max-neighbors`
  1000, `--threshold` 0.8, one NERSC m4468 CPU node (2 x AMD EPYC 7763, 128
  cores / 256 threads, `OMP_NUM_THREADS` 128): 53.4 min from the load to the
  last block, 54:11 min wall for the whole command (with the start-up, the
  statistics and the export of the 1.65 GB file), peak RSS 5.57 GiB.

### Changed

- **`fm-search cluster` and `fm-search similarity-graph` build the graph many
  chains per FAISS call.** They searched one chain per FAISS call and went
  through its neighbours in Python; they now read and search blocks of at most
  16,384 chains (whose results take about 200 MB at the default
  `--max-neighbors` 1000) and keep the edges with NumPy, which is much faster.
  The edge rule is unchanged: an edge comes from the search of its
  lower-numbered chain, in chain order, then in FAISS's rank order. On `hnsw`
  the graph is the same, and so are the files of `cluster` (for the same
  `--seed`) and `similarity-graph`. On `flat` it is the same up to float
  rounding (scores within about 2e-6 at 1536 dimensions), so a pair at
  `--threshold` or tied at a chain's `--max-neighbors`-th score can gain or lose
  its edge, and near-tied neighbours of a chain can change order, which can
  change the clusters of the same `--seed`; the scores now depend on FAISS's
  thread count, as batched `flat` searches already do. Both commands log the
  time of the graph build, and `cluster` that of the Leiden clustering.
- **`fm-search cluster` and `fm-search similarity-graph` need much less memory
  per edge to build the similarity graph.** The kept edges are no longer
  concatenated into one array, they are handed to igraph without a temporary
  NumPy array conversion, and equal edge weights are stored once. The graph,
  its weights, the clusters and the output files are unchanged.
- **BREAKING — `fm-search cluster --method greedy` refuses `--resolution` and
  `--seed`**, the Leiden options (greedy has no resolution and is
  deterministic), in one message. **`--max-neighbors` below 1 is refused** by
  `cluster` and `similarity-graph` before the database is loaded (it raised a
  bare FAISS `AssertionError` after the load). `--resolution`'s default, 1.0,
  is now stated in its help instead of Typer's `[default: 1.0]`.
- **BREAKING — `fm-search cluster`'s Leiden step runs on python-igraph's
  `Graph.community_leiden`** (modularity, the same objective at
  `--resolution`, a fixed 10 iterations by default, set by the new
  `--leiden-iterations`) instead of `leidenalg` run to convergence;
  `leidenalg` is no longer a dependency. `--resolution` works as before: at
  0.5 and 2.0 on a 1M-chain graph (`--max-neighbors` 50) the new partitions'
  largest cluster, cluster count and modularity overlap `leidenalg`'s across
  seeds, and the new partitions agree with `leidenalg`'s about as well as two
  seeds of either library agree. **Partitions and cluster labels change for
  the same `--seed`**, by about as much as a change of seed: on a 1M-chain
  graph (`hnsw`, `--threshold` 0.8, `--max-neighbors` 1000), the new
  partitions agree with `leidenalg`'s converged ones at AMI 0.857 (ARI 0.778;
  3 seeds against 2), while two `leidenalg` seeds agree at AMI 0.854 (ARI
  0.763) and two new seeds at 0.859 (ARI 0.790); modularity 0.6663-0.6687
  against 0.6653-0.6696. The Leiden step took 679-852 s instead of
  6,514-7,671 s on that graph (one thread, NERSC CPU and GPU nodes), and
  needed about 5.1 GiB of working memory instead of 11.8 GiB in a standalone
  run of the Leiden step on that graph (`cluster`'s own `leidenalg` Leiden
  phase measured about 8.7 GiB there). In `cluster` itself (seeds 42-44,
  `--threshold` 0.8, `--max-neighbors` 1000, Linux x86_64 at 128 threads),
  the Leiden phase needed 5.1 GiB above the built graph instead of 8.7 GiB
  on that `hnsw` database, and the run peaked at 20.2 GiB instead of 23.79
  GiB (VmHWM); on a 1M-chain `ivf_pq` database (447,547,205 edges), 10.3
  GiB instead of 17.1 GiB, a peak of 28.16 GiB instead of 34.99 GiB, and a
  Leiden step of 1,714-1,936 s instead of 17,945 s (new: a NERSC GPU node;
  old: a CPU node). Cluster ids stay by decreasing size (0 the largest),
  ties now by the smallest chain index, as under greedy. **`--seed` below 0
  is refused** (exit status 2, before the load): the new generator gives
  seeds n and -n the same partition. `--seed` 2**31 and above now work
  (`leidenalg` raised `OverflowError`). **`--threshold` below 0 with
  `--method leiden` is refused when a kept edge score is negative** (exit
  status 2, after the graph is built; previously `leidenalg` raised a bare
  `BaseException` then). A negative threshold whose kept scores are all at
  least 0 worked and still works. **Library**:
  `EmbeddingClusterer.cluster_leiden` returns a NumPy array (it returned a
  list); its `n_iterations` defaults to 10 and must be at least 1 (it
  defaulted to -1, run to convergence); it raises `ValueError` for a seed
  below 0 (`leidenalg` accepted negative seeds; the new generator gives seeds
  n and -n the same partition) and for a graph built with a negative kept
  weight. It seeds python-igraph's random generator privately
  (`random.Random(seed)`), leaving Python's global `random` state untouched
  as `leidenalg` did, and restores python-igraph's generator to the `random`
  module afterwards (a caller's own igraph generator is replaced). Seeds are
  integers of at least 0 (`operator.index`: NumPy integers accepted, floats
  refused).
- **Structure-based chain selection keeps chains with at least `min_res_n`
  residues** (was more than `min_res_n`). This covers `fm-search build
  structures`, `fm-embedding` and `inference`, and matches `FoldMatch(min_res=...)`
  at query time and the FASTA paths, which already used "at least". Databases and
  embeddings built with a non-zero minimum now also include chains of exactly
  that length.
- **The ESM3 weights are pinned.** FoldMatch now downloads `rcsb/rcsb-esm` at
  revision `636acb420daf674df37ab0220f56061a5de2f169` instead of whatever its
  `main` branch points to, and refuses any `esm3_sm_open_v1.pth` other than the
  one with sha256
  `5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9`. These are
  the weights FoldMatch already used, so embeddings do not change. In the usual
  Hugging Face cache layout the check reads the file's blob name, which is its
  sha256, and costs nothing; when the layout hides it (a `local_dir` download or
  a copied snapshot), the 2.8 GB file is hashed once per process. An offline
  installation (`HF_HUB_OFFLINE=1`) whose cache holds only an older
  `rcsb/rcsb-esm` snapshot must run `download-models` once online.
  Every `torch.load` now passes `weights_only=True` (already PyTorch's default
  from 2.6), so model and residue-embedding files cannot run pickled code.
- **FASTA sequences are normalised before ESM3.** ASCII whitespace inside
  sequence lines is removed and ASCII letters are uppercased before a sequence
  is embedded, stored in the sequence store or aligned in Stage 2, on every
  FASTA path (`fm-search build sequences`, `fm-search query sequences`,
  `fm-embedding`, `inference`) and in `FoldMatch.sequence_embedding` (Python
  API). ESM3 used to read lowercase letters, and spaces or tabs inside a line,
  as unknown residues, so the chain embeddings of such records were wrong
  (cosine 0.31-0.34 to the correct ones on the test fixture): rebuild databases
  and redo embeddings made from lowercase or soft-masked FASTA. `--min-res-n`,
  and `min_res`/`max_res` of `FoldMatch.sequence_embedding`, now count residues
  without whitespace. Uppercase FASTA without whitespace inside its sequence
  lines gives exactly the same results as before.
- **FASTA problems and existing outputs are reported before the ESM3 pass.**
  `fm-search build sequences` now refuses an existing `.index`, `.metadata`,
  `.invlists`, `.sequences` or `.rescodes.*` file before computing any
  embedding; before, the existing database was only refused after the whole
  pass, and a leftover `.sequences` file made the build fail at the end.
  `fm-search build structures` checks `.index`, `.metadata` and `.invlists` up
  front in the same way. `build sequences` and `query sequences` scan the FASTA
  once on every rank before ESM3 and log a warning for duplicate ids, sequences
  containing `<` or `>`, and non-ASCII sequences, then go ahead as before. If
  the sequence store written after the pass disagrees with that scan (the FASTA
  changed during the build), the build fails and removes the database files it
  wrote, so it can simply be run again. A FASTA with no sequence of at least
  `--min-res-n` residues (or an empty one) is refused before ESM3 too, instead
  of failing after the pass with "No embeddings provided". `fm-search query
  sequences` and `fm-search query db` decide whether Stage 2 runs before ESM3
  or the search, and check the sequence stores it will read: `--seq-identity`
  without a store, an incomplete store and an E-value filter over an empty
  database are reported there.
- **An unknown `--format-output` field is refused before the ESM3 pass**
  by `fm-search query sequences --scoring blosum62` too, as under esm-codes:
  whenever Stage 2 runs, an empty list or an unknown field is now a usage
  error (exit status 2) reported before the queries are embedded. Before, the
  queries were embedded first and the command then failed with exit status 1
  (and the Python function `query_database_from_fasta` raised `ValueError`;
  it now raises `typer.BadParameter`). An embedding-only search does not use
  the option and still ignores it. `fm-search query db` now checks it too,
  before its search (see *BREAKING — the BLOSUM62 Stage 2 needs the native
  kernel* below).
- **An interrupted sequence-store build is detected.** A `.sequences` file
  whose build stopped part way (it now records its totals last) is refused
  with an "incomplete" error instead of being read as the whole database:
  delete the database and rebuild it.
- **Embedding shards are named per run.** The intermediate files in
  `--tmp-embedding-folder` are now `emb_<run id>-<rank>.parquet`. The run id is
  `FOLDMATCH_RUN_ID` when set; in a task of an `srun` step it is built from the
  job id, step id, restart count and a digest of the command's arguments, so
  the tasks of a step agree and the commands it runs one after another do not;
  otherwise, including the batch step of an `sbatch` script, the launching
  process draws a random id and passes it to the ranks it starts. `fm-search
  query sequences` deletes its query shards after the search; build shards are
  kept unless `--cleanup-embeddings` is given. A rank above 0 started without a
  shared id refuses to start: set `FOLDMATCH_RUN_ID` to the same value for every
  rank whenever something other than `srun` or Lightning starts the ranks, that
  is `torchrun`, or a multi-node run outside SLURM started on each node with
  `NODE_RANK` (set it on every node: otherwise the other nodes refuse and node 0
  waits for them until the one-day process-group timeout). Every rank also
  refuses to start when shards with its run id are already in
  `--tmp-embedding-folder` (the same command run twice in one `srun` step, or a
  reused `FOLDMATCH_RUN_ID`). After a FASTA pass, rank 0 checks that the shards
  hold exactly one embedding per entry, and a missing rank is now an error
  instead of being dropped silently.
- **`fm-search build sequences` logs its index phase.** When rank 0 starts the
  FAISS index it logs the FAISS version and OpenMP threads, how many of the
  node's CPUs (and physical cores) the process may use, and `OMP_NUM_THREADS`,
  with one line per thread pool threadpoolctl finds; a line naming FAISS's
  BLAS library when it is single-threaded, as the `faiss-cpu` 1.13.2 wheel's
  was (a warning, except on macOS: the `ivf_pq` training and adds then run on
  one core; the line says to install the pinned `faiss-cpu` 1.15.1, whose
  OpenBLAS is multi-threaded, or a FAISS built on MKL); and, except on
  macOS, a warning when FAISS has fewer threads than usable physical cores,
  naming the cause: `OMP_NUM_THREADS`, which Lightning's DDP launcher sets
  during a multi-device pass when it is unset (to the CPUs the rank may use
  divided by the devices per node; the lines say so, and exporting
  `OMP_NUM_THREADS` before the run chooses FAISS's threads); then the time
  taken by the FAISS index, the sequence store and the residue-code store. A
  build on more than one node (`--num-nodes`) also logs that the other nodes
  stay allocated through this one-process phase, and that `build
  sequences-embed` and `build sequences-index` (see *Added*) release them. The
  outputs are unchanged.
- **The `--load-batch-size` help of the build commands is corrected.** It said
  that larger batches (65536 and more) use many-core nodes better during an
  IVF-PQ build, but FAISS adds IVF-PQ vectors in chunks of 32768 whatever the
  batch: a larger one only takes more memory.
- **`--max-evalue` against an empty database is an error.** With an E-value
  filter (including the default `--max-evalue 1e-3`), `fm-search query
  sequences`, `fm-search query db` and `align_candidates` now refuse a database
  whose residue count is 0 or negative, with a message that names the likely
  cause (an empty or unbuilt sequence store) and the two ways out: delete the
  database and rebuild it with `fm-search build sequences`, or pass
  `--max-evalue inf` (`max_evalue=None`). The CLI reports it before the ESM3
  pass or the search. Before, a query against an empty sequence store returned
  no hits with only "missing from sequence store" warnings, and
  `align_candidates` failed with an unrelated error from deep inside the
  alignment loop.
- **`--min-coverage` help text corrected** on `fm-search query sequences` and
  `fm-search query db`: a hit is kept only if the query coverage and the subject
  coverage both reach the value, so it is dropped when either side is below.
  The old text suggested the opposite. Filtering itself is unchanged.
- **The BLOSUM62 Stage 2 aligns on FoldMatch's native SIMD kernel: the same
  output, about 23 times faster.** `fm-search query sequences`
  (`--scoring blosum62`, the default), `fm-search query db` and
  `align_candidates` align each (query, candidate) pair with `fmswi_align`, an
  integer Smith-Waterman added to FoldMatch's compiled library (the one of the
  esm-codes kernel), instead of biotite's `align_optimal`. It uses a striped
  query profile in 16-bit SIMD lanes (SSE2, AVX2, NEON, or a portable build) and
  redoes a pair in 32-bit lanes when its score could overflow them; the scores
  are exact integers, so every build gives the same results. It reproduces
  biotite 1.7.1's `align_optimal` with `local=True` and affine gaps, ties
  included: the same score, end cell and path, hence the same value in every
  output column. On the full nbt.3988 benchmark, the Stage-2 tables were
  byte-identical: 17,733,246 rows at `--threshold 0.7` and 9,503,077 at `0.8`.
  Stage 2 of the 0.7 search took 486 s instead of 11,403 s (468 to 486 s over
  the runs with the slices described below), on the CPU of one Perlmutter GPU
  node (a 64-core AMD EPYC 7763) with 128 workers. The kernel alone aligned the
  benchmark's largest query in 72 s instead of 9,864 s, and cut the end of the
  run that waited on the longest queries from 8,459 s to 4 s. Pairs are now also
  handed to the workers in slices of about a quarter of each worker's share of
  the DP cells (a pair larger than that is a slice of its own), the largest
  first, which shortens that end of the run to under 1 s. The composition-bias
  correction (`--comp-bias-corr`) is part of the query profile and costs nothing
  more. The traceback needs one byte per DP cell (359 MB for the benchmark's
  largest pair, where biotite took 4.87 GB). A worker process that cannot load
  the kernel now fails Stage 2 instead of hanging it.
  `python -m foldmatch.search.swkern`, the kernel's self-test, also checks
  `fmswi_align` against a golden digest and reports it under `fmswi`. biotite
  remains a dependency: it supplies the BLOSUM62 matrix and the
  `--significance-mode sampled` statistics.
- **BREAKING — the BLOSUM62 Stage 2 needs the native kernel, and its gap costs
  are checked before Stage 1.** A search that runs Stage 2 under BLOSUM62 now
  needs FoldMatch's compiled kernel, as `--scoring esm-codes` already did. An
  install without it (the pure-Python package a build from source falls back to
  when it finds no C compiler; see *Platform wheels* below) aligned with
  biotite; `fm-search query sequences` and `fm-search query db` now refuse it
  before Stage 1, with exit status 1 and the command that rebuilds or reinstalls
  the kernel, and `align_candidates` raises `KernelError`. There is no fallback.
  Embedding-only searches (`--no-seq-identity`, or a database without a sequence
  store) do not need the kernel. `--gap-open` must now be at least 1 (0 was
  accepted) and `--gap-extend` at least 0, both at most 1,048,576; the sign is
  still ignored. Called from Python, a gap cost must be a whole number: a float
  such as `11.0` is still taken as 11, but `10.5`, which biotite truncated to
  10, is now refused. Under `--significance-mode default`, gap costs other than
  11/1 are refused, as before, whenever the run needs significance (an `evalue`
  or `bits` column in `--format-output`, or a finite `--max-evalue`, `1e-3` by
  default). `--significance-sample-size` must now be at least 2 (the Python
  functions check it when they compute the sampled significance): under
  `--significance-mode sampled`, 1 and 0 gave infinite or NaN bit scores and
  E-values, and a negative size failed after Stage 1. These refusals now come
  before Stage 1, before the queries are embedded (`query sequences`, on every
  rank) or the database is loaded (`query db`), as usage errors (exit status 2)
  that name the options and, for 11/1, the ways out
  (`--gap-open 11 --gap-extend 1`; `--significance-mode sampled`; or
  `--max-evalue inf` with neither column); the 11/1 refusal came after Stage 1,
  with exit status 1 (the Python functions `query_database_from_fasta` and
  `query_database_from_database` now raise `typer.BadParameter` for them).
  `align_candidates` checks the gap costs the same way before it reads a
  sequence or starts a worker (`ValueError`, or `TypeError` for one that is not
  a whole number). A gap cost of `-11`/`-1` now counts as 11/1 under
  `--significance-mode default` (it was refused there). Whenever Stage 2 runs,
  `query db` also refuses an unknown or empty `--format-output` field before its
  search, as a usage error (exit status 2); it failed with exit status 1 after
  the search.
- **Long FASTA sequences need far less ESM3 memory.** For sequence-only input
  (the FASTA paths of `fm-search`, `fm-embedding` and `inference`, and
  `FoldMatch.sequence_embedding` in the Python API), ESM3 now runs without its
  geometric attention, which has no coordinates to attend to there but
  allocated a tensor that grows with the square of the sequence length. ESM3
  ran out of memory at 2,750 residues on a 40 GB A100; 16,000 residues now fit
  in about 10 GiB. On a CPU node a 4,000-residue sequence needed 83 GiB and now
  needs 8.3 GiB, and a 1,500-residue sequence takes 10 s instead of 56 s on an
  M1 Max CPU. Embeddings are bit-identical. Structure inputs are unchanged. In
  the Python API, ESM3 calls on one model from several threads now run one at
  a time, so that no structure embedding runs while another thread has
  geometric attention switched off.
- **Platform wheels; installing from source needs a C compiler.** FoldMatch now
  contains a small compiled library (the Smith-Waterman kernel above), so the
  release publishes platform wheels for Linux x86_64 and aarch64 (built in
  manylinux_2_28 images) and macOS arm64 and x86_64 (macOS 11 or later), and
  the classifiers name Linux and macOS instead of "OS Independent". Anything
  that builds FoldMatch from source compiles the kernel and needs GCC 12 or
  later, or Clang 14 or later: an install from the sdist (another platform, or
  `pip install --no-binary foldmatch`), an editable install (`pip install -e
  .`, `uv sync`) and `hatch test`. Without a usable compiler the install still
  succeeds, with a warning, as a pure-Python package: everything works as
  before except the Stage 2 of a search (under either scoring, and in `query
  db`: the BLOSUM62 Stage 2 aligns on the kernel too, see above), which refuses
  to run and prints the reinstall command, `pip install --no-cache-dir --force-reinstall
  --no-binary foldmatch foldmatch` (pip caches the pure-Python wheel, hence
  `--no-cache-dir`). Set `FOLDMATCH_REQUIRE_NATIVE=1` to make a failed compile
  an error instead; CI, the release wheels and the Docker image do. In a
  source checkout, rebuild the kernel after editing its C source with `python
  -m foldmatch._native.build`; a library built from other source is refused.
- **`threadpoolctl` is a declared dependency** (`threadpoolctl==3.6.0`,
  exactly pinned like the others). The esm-codes Stage 2 uses it to limit to
  one thread the BLAS of its worker processes and of an in-process run, which
  keeps reruns bit-identical; it already came with `esm` (through
  scikit-learn), and the code still works without it. threadpoolctl cannot
  limit Apple Accelerate (numpy's BLAS on macOS): an in-process run there
  keeps Accelerate's own thread count and says so once in the log, while the
  worker processes are limited through `VECLIB_MAXIMUM_THREADS=1` either way.
- **`faiss-cpu` 1.15.1** (was 1.13.2). On Linux its wheels bundle an OpenMP
  build of OpenBLAS, where 1.13.2's OpenBLAS was single-threaded. The OPQ and
  k-means training of an `ivf_pq` index, the IVF assignment of its adds and
  batched flat searches therefore run on FAISS's OpenMP threads, at most 128
  for BLAS in the x86_64 wheel (see the README, *FAISS threads*): an `ivf_pq`
  index of 30 million sequences, about 7 hours with 1.13.2 on any node, took
  1 h 45 min on a 64-core node at 64 threads and 2 h 37 min on a 128-core node
  at 128, where the wheel's OPQ training runs slower than at fewer threads.
  `OMP_NUM_THREADS` sets those threads; `OPENBLAS_NUM_THREADS` does not. The
  one-thread BLAS limit of an in-process esm-codes Stage 2 now reaches that
  OpenBLAS too, and so FAISS's OpenMP threads (and torch's, which share them
  when torch is loaded first) while it runs; it then sets every OpenMP runtime
  back to its thread count, where threadpoolctl's restore alone left them at
  the OpenBLAS's own, older count. Index files written by either version are
  read by both, and the two search one database alike up to float rounding,
  which can reorder tied hits (with `hnsw`, 5 to 7% of searches changed a
  hit). With the wheel, rounding and so the order of tied hits also depend on
  FAISS's thread count: set `OMP_NUM_THREADS` to the same value for
  bit-identical reruns. Any rebuild of an `ivf_pq` index gives a different,
  equally valid index (at 30 million entries, an MKL build below shared 57% of
  its top-100 hits with the 1.13.2 index, wheel builds 58 to 59% with an MKL
  build, and two wheel builds at 64 and 128 threads 59% with each other), so
  compare only searches made on the same database.
  A FAISS built on MKL, from the official `pytorch` conda channel (Linux
  x86_64, Python 3.12), is faster still for large `ivf_pq` builds; the README
  gives the install and its caveats. On macOS the wheel still uses Accelerate
  and bundles its own OpenMP runtime, so FoldMatch still sets
  `OMP_NUM_THREADS=1` there by default. On Linux x86_64, `import faiss` now
  logs at INFO that it "Could not load library with AVX2 support": expected,
  the wheel has one library that picks the CPU's instructions at run time. Its
  Linux wheels need glibc 2.27 (FoldMatch's own wheels need 2.28), its macOS
  wheels macOS 14 on arm64, as before, and macOS 15 on x86_64 (was 14);
  `faiss-cpu` still publishes no sdist.
  `uv.lock` is regenerated, and now also records FoldMatch 0.10.0 and its
  `threadpoolctl` pin.
- **`download-models` also fetches the esm-codes scoring bundle.** `fm-embedding
  download-models` and `inference download-models` now download, besides ESM3
  and the aggregator, the bundle of every registered esm-codes scoring system
  (one, about 26 MB, from `rcsb/foldmatch-esm-codes` at the revision pinned in
  FoldMatch) and check the sha256 of every file. `--system NAME` (repeatable)
  limits this to the named systems; an unknown name is refused before any
  download. A cached bundle file whose sha256 differs (a truncated or
  corrupted download) is downloaded again. The files go into the Hugging Face
  cache like the models, so later runs work with `HF_HUB_OFFLINE=1` and from
  a read-only cache; nothing else is written there.
- **`download-models` downloads only.** It fetches and checks the pinned files
  (the ESM3 snapshot, whose weights it hashes in full, the aggregator weights,
  now pinned by sha256 too, and the esm-codes bundles) without building ESM3
  or the aggregator, importing torch or touching a GPU. Before, it loaded both
  models, on the GPU when one was visible, which ran out of memory on shared
  GPU login nodes; the `CUDA_VISIBLE_DEVICES=` workaround is no longer
  needed. A cached ESM3 or aggregator file whose content differs from its pin
  is downloaded again, and the command prints one line per model with the
  verified pin.
- **The `--format-output` field list** in `fm-search query sequences --help`
  and `fm-search query db --help` names the four new esm-codes fields
  (`rawrev`, `q90`, `rawadj`, `ecal`); in `query db` they are always empty.
- **Docker image.** The image is built with a regular (non-editable) install
  that must compile the kernel, runs the kernel self-test at build time and
  again in the compiler-free runtime stage, and no longer copies the source
  tree into `/app`.
- **A `.pt` or `.csv` embedding file holds one vector: residue embeddings are
  refused.** `fm-search build embeddings` and `fm-search query embedding`
  read one embedding vector per `.pt` or `.csv` file: a tensor of shape
  `[D]`, `[1, D]` or `[D, 1]`, or a CSV of one column (as `fm-embedding` and
  `inference` write chain and assembly embeddings) or of one line. A file of
  residue embeddings (one row per residue, as `fm-embedding from-structures
  residue`, `fm-embedding from-sequences residue` and `inference
  residue-embedding` write them), or any other table, now stops the command
  with an error that names the file and its shape. Such a file was averaged
  over its rows without a message, and the mean indexed or queried as the
  chain's embedding, which it is not: on the test chains its cosine to the
  chain embedding is -0.03 to 0.01, and the means of any two chains, related
  or not, score 0.86 to 0.97. Compute chain embeddings with `fm-embedding
  from-structures chain` or `fm-embedding from-sequences chain`, or from
  residue files `NAME.CHAIN.pt` or `NAME.CHAIN.csv` with `inference
  chain-embedding --src-file LIST --res-embedding-location DIR
  --res-embedding-format pt|csv --output-folder OUT`, `LIST` holding one line
  `NAME,,CHAIN` per file. **Rebuild databases built from residue
  embeddings**: the mean has the width of the residue embeddings, which for
  FoldMatch's ESM3 ones is the dimension of its chain embeddings (1536), so
  `fm-search stats` cannot tell such a database apart, but building it again
  from the same folder now stops with this error, which identifies it.
  Results of such databases, and of residue embeddings used as queries, are
  meaningless. A `[D, 1]` tensor is now read as its `D` values; it was read
  as their mean, one value. A NaN or infinite value is now refused too (such
  an embedding was never found, and as a query it found nothing), and so are
  a file that cannot be read or holds no values, a `.pt` that does not hold a
  dense tensor, and a CSV that is not numbers (its error named no file and
  could quote a whole line), each with the file's name.
- **BREAKING — residue embeddings in Parquet keep their shape.** With
  `--output-format parquet`, `fm-embedding from-structures residue`,
  `fm-embedding from-sequences residue` and `inference residue-embedding`
  write one row per chain whose `embedding` holds one list of `D` values per
  residue (ESM3's BOS and EOS rows included, as before): its type is
  `list<fixed_size_list<float32, D>>`, so the file stores `D`. They wrote each
  chain's residue table flattened into one row of `list<float32>`, which
  stores no shape. The Python readers `ResidueEmbeddingFromParquet` and
  `ResidueAssemblyEmbeddingFromParquet`, and `chain_inference.predict`, lose
  their `embedding_dim` argument, so a call that passes it now fails (for
  `chain_inference.predict`, whatever its `src_from`), and the arguments of
  `ResidueAssemblyEmbeddingFromParquet` after `parquet_path`, and
  `return_predictions` of `chain_inference.predict`, are now keyword-only, so
  that an old positional call fails too. The readers take `D` from the file,
  read only the new layout and refuse a flat file, or one that cannot be read,
  naming it. They read one file, or a list of files
  (`ResidueEmbeddingFromParquet` only), such as the shards of a multi-device
  run: `sorted(Path(OUT).glob("*.parquet"))`. A folder, which they read
  before, is refused. `fm-search build embeddings` and `fm-search query
  embedding` refuse the new files as residue embeddings, whatever their row
  lengths. The files are written in row groups of about 256 MB, so a residue
  run no longer holds up to 16,384 chains in memory. `ParquetBatchWriter`
  writes the new layout with `residue_embeddings=True` (not with a residue
  output); without it, it now refuses a tensor with more than one axis longer
  than 1, and the chain, assembly and sequence embeddings and the residue-code
  shards it writes are unchanged byte for byte. Recompute the residue Parquet
  files of earlier versions, or convert one with this function (pyarrow only),
  giving the number of values per residue (1536 for FoldMatch's ESM3), as in
  `nest("inference-0.parquet", "inference-0-nested.parquet", 1536)`:

  ```python
  import pyarrow as pa
  import pyarrow.parquet as pq

  def nest(src, dst, d):
      table = pq.read_table(src)
      chunks = []
      for flat in table.column("embedding").chunks:
          offsets = flat.offsets.to_numpy() - flat.offsets[0].as_py()
          if (offsets % d).any():
              raise ValueError(f"{src}: a row is not a multiple of {d} values")
          values = flat.flatten().cast(pa.float32())
          offsets = pa.array(offsets // d, pa.int32())  # in residues
          residues = pa.FixedSizeListArray.from_arrays(values, d)
          chunks.append(pa.ListArray.from_arrays(offsets, residues))
      kind = pa.list_(pa.list_(pa.float32(), d))
      nested = pa.chunked_array(chunks, type=kind)
      i = table.column_names.index("embedding")
      pq.write_table(table.set_column(i, "embedding", nested), dst)
  ```
- **BREAKING — `--min-res-n` only where it selects chains.** `inference
  residue-embedding`, `inference structure-embedding` and `inference
  chain-embedding` took `--min-res-n`, whose help said that it keeps the
  chains of a structure with at least that many residues, but each row of
  their chain list names one chain, embedded as listed, so it did nothing.
  They no longer take it: a command line that passes it stops with `No such
  option` (exit status 2); leave it out. `inference complete-embedding`
  keeps it, with the help of `inference assembly-embedding` ("Consider only
  assembly chains with at least <min_res_n> residues."), and passes it to
  its assembly step only, where it selects the chains of each assembly. In
  the library, `esm_inference.predict` and `structure_inference.predict`
  given chains (`src_from=SrcProteinFrom.chain`, their default), and
  `chain_inference.predict` given residue embedding files (`src_from` file,
  its default) or Parquet, or an `inference_set`, now refuse a `min_res_n`
  other than 0 (`RefusalError`) before anything is read: `min_res_n` selects
  the protein chains of structures read whole (`src_from` structure). Leave
  it at 0 in these calls, or list only the chains to embed. Their signatures
  are unchanged.
- **BREAKING — one chain list, one name per row.** The chain lists of
  `inference residue-embedding`, `structure-embedding`, `chain-embedding` and
  `complete-embedding` (`--src-file`, `--src-chain-file`), and the file or
  stream of the library's `EsmProtFromChain`, are read one way: CSV without a
  header (a quoted field read as CSV reads it), of structure name, structure
  file or URL, chain ID and embedding name. The embedding name now defaults
  to `<structure name>.<chain ID>` when the 4th column is missing or empty:
  for such a row, `residue-embedding` stopped with torch's `invalid file
  name: .../.pt` (with `--output-format csv`, it wrote a hidden `.csv` that
  the next such row replaced; with `parquet` or `json`, an empty id), and
  `structure-embedding` wrote an empty id. In the stream of
  `EsmProtFromChain`, a row of fewer than four elements, which pandas refused
  unless another row had four (its missing elements then read as `nan`), is
  now read like a short line of a file, and a `None` name, read as `nan`,
  gets the default too. `chain-embedding` read the residue embedding
  `<structure name>.<chain ID>` from `--res-embedding-location` and wrote the
  chain embedding under that name, whatever the 4th column said, so it could
  not find what `residue-embedding` wrote under another name, and it skipped
  lines of one or two fields without a word. It now reads `<embedding
  name>.<res-embedding-format>`, writes under the embedding name and does not
  read the structure file column; a line of one or two fields gets an empty
  chain ID; and a missing residue file is reported with its path and the
  row's embedding name. Its old list reader,
  `adapt_csv_to_embedding_chain_stream` in `foldmatch.utils.data`, is
  removed. Lists that name each row `<structure name>.<chain ID>`, and the
  `NAME,,CHAIN` lists of residue files above, give the same files as before,
  except that: (1) a space, a tab or other whitespace at a line's start or
  end is part of a field (`chain-embedding` stripped each line; `x,,A ` now
  names `x.A `; and `complete-embedding`, which gave ` x,x.cif,A,x.A` the
  files of `x,x.cif,A,x.A`, refuses it, as its structure name is ` x`: remove
  the whitespace); (2) a quoted field, or a byte-order mark at the start of
  the file, is read as CSV (`chain-embedding` kept the quotes or the mark in
  the names, and stopped on the missing file; the new reading works); (3) a
  row of more than 4 fields is refused (`RefusalError`, naming the file or
  the stream and the row) by the four commands, before any model is loaded,
  and by `EsmProtFromChain`: `chain-embedding` ignored the fields after the
  3rd, and `residue-embedding`, `structure-embedding` and `EsmProtFromChain`
  misread them, or stopped on a bare pandas error (but read right a list
  whose every row started with an index column, as pandas'
  `to_csv(header=False)` writes it); remove the extra fields (a comma after
  the embedding name makes a 5th; write such a list with
  `to_csv(header=False, index=False)`); (4) a list whose file name
  ends in `.gz`, `.bz2`, `.zip`, `.xz`, `.zst` or `.tar` (letter case
  ignored) is read as compressed, as `residue-embedding` and
  `structure-embedding` read it (`chain-embedding` read it as plain text):
  rename a plain-text list so named. Any other pandas `ParserError` (an
  unclosed quote, say) is refused as in (3), with pandas' message.
  `complete-embedding`, whose assembly step reads residue embeddings as
  `<structure name>.<chain ID>`, as `inference assembly-embedding` does, now
  refuses a chain list with other embedding names before any model is loaded,
  naming up to 5 rows (it failed, or ran on the residue embeddings found
  under `<structure name>.<chain ID>`: the row's own, as in (1), an earlier
  run's, or another row's, as when two rows swap names): name them
  `<structure name>.<chain ID>`, or leave their 4th column out. The help of
  `--src-file` and `--src-chain-file` describes the columns and the names.
- **BREAKING — an assembly list row of more than 4 fields is refused.** The
  assembly lists of `inference assembly-embedding` (`--src-file`) and
  `inference complete-embedding` (`--src-assembly-file`), and the file or
  stream of the library's `ResidueAssemblyEmbeddingFromTensorFile` and
  `ResidueAssemblyEmbeddingFromParquet`, hold 4 fields per row: structure
  name, structure file or URL, assembly ID and embedding name. A row of more
  is now refused (`RefusalError`, naming the file or the stream and the row),
  by both commands before any model is loaded (by `complete-embedding`, for
  an assembly list in a regular file), as a chain list row of more than 4
  fields is (above): remove the extra fields (a comma after the embedding
  name makes a 5th). Before, pandas read the leading fields of a first such
  row as the index, which shifted the columns of every row: a trailing comma
  or a 5th field ended in a misleading error, such as `Structure file source
  is neither a recognized URL nor file: '1'` (the assembly ID read as the
  file) or the refusal of a structure name given to two files. A list whose
  every row started with one or more extra fields, such as the index column
  that pandas' `to_csv` writes unless `index=False`, worked (pandas took them
  as the index, so the columns lined up) and is now refused: write it without
  them (`to_csv(header=False, index=False)`). A longer later row stopped with
  a bare pandas `ParserError` that named no file, and a stream row of 5
  elements with pandas' `ValueError`. Any other pandas `ParserError` (an
  unclosed quote, say) is refused the same way, with pandas' message. Lists
  of at most 4 fields per row are read as before. `complete-embedding` now
  reads its assembly list before its residue step, so such a list, or one
  that gives one structure name to two files, is refused before the ESM3
  step instead of after its chain step, which had written the residue and
  chain embeddings. A list given as a pipe (a named pipe, or `<(...)` on
  macOS), or on macOS as `/dev/stdin` or `/dev/fd/N` (even of a regular
  file), which can be read once only, or at any other path under `/dev` (a
  file in `/dev/shm`, say), is read once, into memory, and its text given to
  the assembly step. The help
  of `--src-file` and `--src-assembly-file` describes the columns.
- **BREAKING — a structure list or residue file list row of too many fields
  is refused.** A structure list holds 3 fields per row (structure name,
  structure file or URL, embedding name): it is the file or stream of the
  library's `EsmProtFromStructure`, `EsmAssemblyFromStructure`,
  `ResidueEmbeddingFromStructure` and
  `ResidueAssemblyDatasetFromStructure`. A residue file list holds 2 (residue
  embedding file, embedding name): it is the file or stream of
  `ResidueEmbeddingFromTensorFile`. These lists are given in a file through
  the library only, to `predict` of `esm_inference`, `structure_inference`,
  `chain_inference`, `assembly_inferece`, `chain_complete_inference` and
  `assembly_complete_inference`. The commands that use these readers
  (`fm-embedding from-structures`, `fm-search build structures`,
  `inference chain-embedding` and `complete-embedding`) give them streams
  they build themselves, of the right length, and are unaffected. A row of
  more fields is now refused (`RefusalError`, naming the file or the stream
  and the row), before any model is loaded, as a chain or assembly list row
  is (above): remove the extra fields (a comma after the embedding name makes
  a 4th in a structure list, a 3rd in a residue file list). Before, pandas
  read the leading fields of a first such row as the index, which shifted the
  columns of every row (after a trailing comma, the embedding name was read
  as the file). A list whose every row started with one or more extra fields,
  such as the index column that pandas' `to_csv` writes unless `index=False`,
  worked (pandas took them as the index, so the columns lined up) and is now
  refused: write it without them (`to_csv(header=False, index=False)`). A
  longer later row stopped with a bare pandas `ParserError` that named no
  file, and a longer stream row with pandas' `ValueError`. Any other pandas
  `ParserError` (an unclosed quote, say) is refused the same way, with
  pandas' message. A residue file list in a file is now read as strings, as
  its stream and every other list are: otherwise pandas 3 reads an index
  column `0, 1, 2` like no index at all, and the refusal would miss it. So
  the names of a list whose names all look like numbers, or all like booleans
  (any of them may be empty), which pandas read as such, keep their spelling:
  `007` and `1.50` give `007.csv` and `1.50.csv` (or `.pt`) where they gave
  `7.0.csv` and `1.5.csv`, `TRUE` gives `TRUE.csv` where it gave `True.csv`,
  and `1` next to an empty name gives `1.csv` where it gave `1.0.csv`; `1`
  and `2` give `1.csv` and `2.csv`, as before. A space around a number, which
  pandas dropped, now stays in the name (`a.pt, 1` gives ` 1.csv`). In JSON
  output their ids are strings (`"007"`, `"1"`, `"TRUE"`) where they were
  numbers or booleans (`7.0`, `1`, `true`), and Parquet output, which stopped
  on them with pyarrow's `ArrowTypeError` and wrote nothing, now holds them
  as written. An empty name or `NA` stays NaN, as before: `nan.csv`, `null`
  in JSON, and that `ArrowTypeError` in Parquet. Lists of at most that many
  fields per row are otherwise read as before.

- **BREAKING — `inference chain-embedding` takes no `--structure-format`.** It
  reads residue embeddings and no structure file, so the option did nothing. A
  command line that passes it now stops with `No such option` (exit status 2);
  leave it out. `inference complete-embedding` keeps the option, for its
  residue and assembly steps, which read structures. In Python,
  `foldmatch.cli.inference.chain_embedding` no longer takes `structure_format`
  (`TypeError`); `chain_inference.predict` keeps it, for `src_from` structure.
- **BREAKING — `inference complete-embedding` names its assembly file
  `assembly-inference` by default.** `--output-assembly-name` defaulted to
  `chain-inference`, the default of `--output-chain-name`, so that with
  `--output-format parquet` or `json` and one folder for
  `--output-chain-path` and `--output-assembly-path`, the assembly file
  replaced the chain file. Both are now kept. In any folder, the default
  assembly file is now `assembly-inference-<rank>.parquet` or `.json.gz`: read
  it under that name. Output in `csv` or `pt` does not use the name.
- **BREAKING — every row of a list has an embedding name of its own.** The
  chain lists, assembly lists, structure lists and residue file lists of
  `inference` and of the library (`read_chain_list`, `read_assembly_list`,
  `read_structure_list`, `read_residue_file_list` and the readers built on
  them) read `NA`, `nan`, `null` and pandas' other NA words as text, and a
  missing or empty field as `''`, as chain lists already did; and a missing or
  empty embedding name is a default, as in chain lists
  (`<structure name>.<chain ID>`): `<structure name>-<assembly ID>` in an
  assembly list, the structure name in a structure list, the residue file's
  name without its extension in a residue file list (`x` for `x.pt`). Before,
  such a name was NaN, which gave the file `nan.csv` (every unnamed row of an
  assembly list into that one file), the id `nan` in JSON, an
  `ArrowTypeError` in Parquet, or, in a structure list, hidden files `.A.csv`
  shared by every unnamed structure. A name given to more than one row is now
  refused (`RefusalError`, naming the names and their rows), in every output
  format and before any model is built: with `csv` or `pt` each row replaced
  the one before, and in Parquet and JSON the rows could not be told apart
  (the assembly readers keep one). `read_structure_list` no longer takes
  `keep_default_na`. A list that relied on NaN names, or repeated a name, must
  give each row its own name.
- **BREAKING — an `inference` run refuses outputs that would replace each
  other or its inputs, before any model is built** (`RefusalError`, exit
  status 1). With `--output-format csv` or `pt`, an embedding name is a file
  name: one that is empty, `.` or `..`, holds `/` or NUL, or makes a file
  name over 255 bytes is refused (`../x` wrote outside the output folder), and
  so are two names that the output folder takes for one file (`A` and `a`,
  or `straße` and `strasse`, on macOS's case-insensitive file system; a
  case-sensitive folder keeps both), the names of a structure list's rows
  (`<name>.<chain ID>`) included. A folder that cannot be written is probed
  on the names it holds.
  `--output-name`, `--output-chain-name` and `--output-assembly-name` follow
  the same rules in every format (`<name>-<rank>.parquet`, a 6-digit rank
  reserved). No output may be an input of its run: the list file, a structure
  file of the list, or a residue embedding the run reads.
  `chain-embedding` refuses an `--output-folder` that is
  `--res-embedding-location` when `--output-format` is
  `--res-embedding-format` (each chain embedding replaced the residue
  embedding it was computed from); `assembly-embedding` refuses, in that same
  case, an assembly name `<structure name>.<anything>`, which can be a residue
  embedding's name (`1acb-1` is fine); `chain_inference.predict` refuses a
  Parquet input in `out_folder` named like its Parquet output. Folders, and
  inputs and output files joined by a link, are compared as files: a
  symbolic link to the folder is the folder, and a residue embedding linked
  (symbolic or hard link) from another folder to an output's file, or an
  output's file linked to it, is that file.
  `complete-embedding` checks all three steps before its residue step, and
  refuses two steps that would write one file: the residue and chain
  embeddings in one folder with `--output-format pt` (the chain step replaced
  its own input and the assembly step failed), or one name for
  `--output-chain-name` and `--output-assembly-name` with Parquet or JSON in
  one folder; and an output of any step over either list or a structure file
  of either. An assembly list that can be read once (a pipe, `/dev/fd/N`) is
  read once, before the residue step, and checked as one in a regular file is
  (before, it was not checked against the other steps). Files of earlier runs
  are not checked: a rerun into the same folders replaces its own earlier
  files, as before.
- **BREAKING — `fm-embedding from-sequences` refuses FASTA ids that conflict
  in its files, before ESM3.** A record id (the first word of a header)
  repeated among the records kept (`--min-res-n`) is refused in every format,
  as the residue-code pass of `fm-search build sequences --residue-codes`
  refuses it: in `csv` and `pt` the last record replaced the others, in
  Parquet and JSON two rows had one id. With `csv` or `pt`, an id that cannot
  name a file (`../x` wrote outside the output folder, `Q9/1-100` stopped the
  run with a traceback), ids that the output folder takes for one file (`abc`
  and `ABC` on macOS), and a `--fasta-file` among the output files are
  refused too; Parquet and JSON keep any id. `--output-name` follows the file
  name rules in every `fm-embedding` command. `fm-search` keeps accepting a
  repeated query id under BLOSUM62, with its warning.
- **BREAKING — a csv or pt writer refuses to write one file twice in a run.**
  `CsvBatchWriter` and `TensorBatchWriter` (`fm-embedding`, `inference` and
  the library with `csv` or `pt` output) keep the files they have written in
  the run, compared as the output folder compares names, and stop the run
  with a `RefusalError` naming the file at a second write of one: the first
  file is kept, the second item is not written. It catches what is known only
  while structures are read whole, such as chains `A` and `a` of one structure
  on macOS's case-insensitive file system (one of them was lost), and two
  structures whose `<name>.<chain ID>` coincide. The check is per process, so
  two ranks that write one file are not caught; it also refuses a name that
  cannot be a file name there.
- **BREAKING — `fm-search` refuses an `--output-file` that is one of its
  inputs, before any work.** `query structure`, `query embedding`, `query
  sequences`, `query db`, `cluster` and `similarity-graph` refuse
  (`RefusalError`, exit status 1) an output file that is a file of a
  database they read (`<db path>.index`, `.metadata`, `.invlists`,
  `.sequences` or `.rescodes.*`), the query FASTA or structure file, or a
  query embedding file that `query embedding` reads. Compared as files: a
  symbolic link to such a file, or on macOS another case of its name, is
  refused too. An output file written by an earlier command is not an input,
  even one named `<db path>.hits.tsv` beside the database, and is replaced as
  before.
- **BREAKING — `fm-search query embedding` refuses query embeddings that
  share an id**, before the database is loaded, naming the ids and their
  files: two rows of one id in a Parquet file, one id in two Parquet files,
  or a Parquet id that is also a `.pt` or `.csv` file's name. The results are
  kept by query id, so all but one of them disappeared from the output. Give
  each embedding its own id, or query the files apart (`--file-extension`).
- **`click` pinned at 8.3.1** (Typer's dependency, as in `uv.lock`): click
  8.5 words its usage errors differently. The tests no longer depend on
  that wording.

### Fixed

- **`ivf_pq` searches of several queries at once could silently return wrong
  hits.** With `faiss-cpu` 1.14.2 to 1.15.1, a FAISS call with 2 or more queries
  on an `ivf_pq` database of `--ivf-nlist` 10000 or more searched only the first
  `nlist // T` coarse cells for the queries in a slice of fewer than T queries,
  where T is the number of FAISS threads and `nlist >= 1024 * T`
  (facebookresearch/faiss#5824). With the default `--ivf-nlist` 16384 that means
  2 to 16 threads and fewer than T\*T queries per FAISS call: `query sequences`
  makes one call per query shard (one shard per rank), and `query db` one per
  4096 queries, so its last call is usually short. No error was raised.
  FoldMatch now sets FAISS's `parallel_mode` 3 on these databases, which gives
  the correct result at every thread count, and no rebuild is needed. Results in
  that range should be run again. A single-query call, an `nlist` under 10000 or
  under 1024 times the thread count (so 17 or more threads at the default
  16384), and `flat` or `hnsw` databases were not affected. The defect is likely
  solved in the next FAISS release: the upstream fix
  (facebookresearch/faiss#5824) was merged on 2026-10-31, but it is not in
  1.15.1, the latest release and FoldMatch's pin. When the `faiss-cpu` pin moves
  to a release that has it, this workaround is to be re-examined.
- **`fm-search cluster` and `fm-search similarity-graph` run on `ivf_pq`
  databases.** They failed on every `ivf_pq` database, at the first chain, with
  FAISS's `direct map not initialized`. The chains' embeddings are now read with
  `reconstruct_n`, which needs no direct map. On `ivf_pq` a score is the inner
  product of the chain's normalised decoded embedding with each stored PQ code,
  as in `query db`: an estimate of the cosine, so two identical chains score the
  norm of their code, close to but not exactly 1 (0.84 to 1.03 on a 30M-chain
  database, 98% of chains between 0.94 and 1.01).
- **Assembly embeddings of entries with more than one assembly.** Structures
  were cached under the entry name alone, so assemblies of an entry read one
  after another all got the first one's chains. `fm-search build structures
  --granularity assembly` and `fm-embedding from-structures assembly` gave
  every assembly of an entry the chains of its first; `inference
  assembly-embedding` and `inference complete-embedding` did the same for
  assemblies of one entry read in a row by one process (input rows are sorted
  by entry). `fm-search query structure --granularity assembly` read the
  requested assembly, so such queries were compared with wrong database
  vectors. The cache now keys on the assembly too, and each dataset has its
  own. An assembly with no protein chain long enough (`--min-res`,
  `--min-res-n`) is now skipped with a warning by the first two commands
  (before, it got the first assembly's chains, and a first assembly without
  one stopped the build); the `inference` commands refuse it when their input
  lists it (`No chains found for ...`), as they did for a first assembly.
  Given a URL, the assembly inference of the library
  (`foldmatch.inference.assembly_complete_inference`) read the download to its
  end while listing its assemblies, then failed on every entry; it now reads
  it from its start. **Rebuild assembly databases, and recompute assembly
  embeddings, of entries with more than one assembly.**
- **Assembly embeddings of assemblies with copies of a chain.** The copies of
  a chain that an assembly's operators make keep the chain's ID, and
  `fm-search build structures --granularity assembly` and `fm-embedding
  from-structures assembly`, with the assembly inference of the library
  (`foldmatch.inference.assembly_complete_inference`), read each protein
  chain of an assembly by its ID, from the first stretch of atoms with that
  ID: in mmCIF, BinaryCIF and PDB files alike, every copy of a chain was
  embedded with the first copy's coordinates. Each copy is now read from its
  own atoms, as `fm-search query structure --granularity assembly` already
  read it. ESM reads each chain as PDB text, which cannot hold a coordinate
  of -1000 Å or less, or of 10000 Å or more: a copy with one, embedded before
  with its first copy's coordinates, now stops the first two commands and the
  library (`Error while creating ESMProtein from structure chain ...`), as it
  already stopped the query (none of the 1,341 assemblies with copies
  checked, in 1,083 entries, has one). In mmCIF and BinaryCIF files laid out
  as the PDB archive's are (each ligand with an asym ID of its own, each
  chain's records in one block, with residue numbers that never decrease),
  the number and order of the chains are unchanged, and the first copy of
  each chain is read as before, bit for bit. In PDB files, which name chains
  by their author IDs, and in any other mmCIF or BinaryCIF file (one whose
  asym IDs are author chains, ligands included, as biotite writes from a PDB
  structure, say), one ID can cover several pieces of a chain: how such a
  chain is now read is in the next entry. `inference assembly-embedding` and
  `inference complete-embedding`, which repeat a chain's stored residue
  embeddings for each of its copies, and the chain-level commands are
  unaffected. ESM3 embeds each chain on its own, and a copy differs from the
  chain it copies only by a rigid motion and by the rounding of its
  coordinates to 0.001 Å (after the motion, each atom within 0.0015 Å of its
  first copy's, on 3,720 copies in 346 assemblies of 241 entries), so its
  embeddings stay close to its first copy's, which it was given before: on 10
  assemblies with copies, from C2 dimers to a 60-copy capsid, 30 of the
  copies' 17,875 structure tokens changed, and the assembly embeddings kept a
  cosine of at least 0.99987 to the old ones (at least 0.99999999 for 8 of
  them), no value moving by more than 0.31% of the vector's norm. Rebuilding
  assembly databases, or recomputing assembly embeddings, for the copies is
  therefore optional (for chains that one ID splits, see the next entry).
- **Chain and assembly embeddings of chains that one chain ID splits.** In
  PDB files, which name chains by their author IDs and numbers, and in mmCIF
  and BinaryCIF files not laid out as the archive's, one chain ID can cover
  several pieces of a chain: where its residue numbers decrease (a tag
  numbered 998-999 before residue 1, an insert numbered from 1001) and where
  records of other chains lie between its records (its free amino-acid
  ligands, HETATM groups listed after every chain, say). `fm-search build
  structures --granularity chain`, `fm-embedding from-structures residue` and
  `fm-embedding from-structures chain`, with the library's
  `chain_complete_inference`, `esm_inference` and `structure_inference` given
  structures (`foldmatch.inference`), listed such a chain once for each piece
  long enough and embedded, each time, the first piece of the atoms with its
  ID, in which blocks that only other chains separate run together, with the
  chain's free amino-acid ligands listed later: a database could hold the
  chain's id more than once, and only one piece of the chain, or all of it
  with its free amino acids appended. `inference residue-embedding`,
  `inference structure-embedding` and `inference complete-embedding`, with
  `esm_inference` and `structure_inference` given chains, embedded that first
  piece too; `fm-search query structure --granularity chain`
  (`FoldMatch.residue_embedding_by_chain`), with or without `--chain-id`,
  embedded every piece of at least `--min-res` amino acids and kept the last
  one, the others embedded for nothing; `fm-search query structure
  --granularity assembly` (`residue_embedding_by_assembly`), and since the
  fix above the assembly builds, embedded the pieces of each copy apart;
  `FoldMatch.residue_embedding` and `structure_embedding` embedded the pieces
  apart; and the datasets of stored residue embeddings (`inference
  assembly-embedding` and `inference complete-embedding`, and the library's
  `chain_inference` given structures, `assembly_inferece` and
  `ResidueAssemblyEmbeddingFromParquet`) took the chain's stored file once
  for each piece. Every path now reads a protein chain the same way: the
  pieces with its ID (in an assembly, of one copy), their amino-acid atoms
  joined in file order into one protein, embedded in one pass, leaving out
  the pieces whose amino acids are all HETATM records when it has amino acids
  in ATOM records (a chain made only of HETATM amino acids, a peptide of
  modified residues, say, keeps them all; HETATM groups within a piece of the
  chain, right after its records with rising numbers, stay in it, as before).
  It is a chain if its polymer amino-acid residues, counted piece by piece as
  before, reach `--min-res-n` (`--min-res` for `fm-search`) together, so a
  chain whose pieces are each shorter is now kept. Each chain is listed once
  (in an assembly, once per copy): the database and the query embed the same
  residues, each chain once (with `--chain-id`, that chain once), and the
  datasets of stored residue embeddings take each chain's file once, as the
  build writes it. A chain ID with no protein chain (water, or a lone free
  amino acid with an asym ID of its own) is refused by the `inference`
  commands above with an `OSError` that names the structure (the input's
  first column), the chain and the file (`No protein chain found for ...`); a
  lone amino acid was embedded before as a protein of one residue. In an
  assembly of a PDB file, each copy of a chain is a chain of its own: biotite
  numbers the copies made by each REMARK 350 list from 0, so FoldMatch now
  numbers them itself as biotite builds them, and the two copies of a chain
  in two lists no longer run together where its numbering rises from its last
  residue to its first (a tag numbered 998-999 before residue 1, say). In
  mmCIF and BinaryCIF files laid out as the archive's, every protein chain is
  one piece: the chains listed and their residues are unchanged, bit for bit,
  for every build; the query's threshold now counts polymer amino-acid
  residues, as the build does, instead of every amino acid, which changes
  only chains near `--min-res` and, at `--min-res` 0 or 1, free amino acids
  with an asym ID of their own, no longer embedded as chains.
  `FoldMatch.residue_embedding` and `structure_embedding` now read each chain
  as the other paths do: HETATM residues within a chain (selenomethionine,
  say), which ESM's PDB reader left out, are read, as X; a chain ID of two
  characters is accepted; each chain is embedded in one pass, with BOS and
  EOS rows at its ends only; and when no chain qualifies they raise `No
  protein chain ... found in the structure`, and
  `residue_embedding_by_assembly` returns an empty dict, as
  `residue_embedding_by_chain` does (so `fm-search query structure
  --granularity assembly` refuses the assembly with its own message), instead
  of torch's error. In any file, archive mmCIF included, the results of
  `residue_embedding` and `structure_embedding` for chains with HETATM
  residues therefore change, and now agree with the database: recompute them.
  `get_structure_from_src` and `StructureProvider.get_structure`
  (`foldmatch.utils`) no longer take a `chain_id`: a selection by chain ID
  joins the pieces that other chains separate; their `assembly_id`, which
  took its place, is now keyword-only, so that an old positional call fails
  too. In the database and the query, a chain that one ID does not split is
  embedded as before (on 420 archive entries, every build and the query at
  `--min-res` 10 gave the same chains and residues, bit for bit). On PDB
  files written from 206 of them, every path now reads each of the 546 chains
  that one ID splits once, as its archive mmCIF chain, bit for bit (before,
  the database had 13 of them right and listed 23 more than once at
  `--min-res` 10, 49 at 0, and the query kept the last piece of 26 at 10, and
  a wrong piece of every one at 0), and in 180 of their 206 assemblies each
  copy as in the mmCIF assembly, residue for residue (164 before; in the 26
  others, a chain keeps the free amino acids that the assembly leaves right
  after it, as before, while its stored file, which `inference
  assembly-embedding` and `inference complete-embedding` give each copy, now
  lacks them). With ESM3, the database vector of such a chain now equals the
  mmCIF chain's, bit for bit, and scores 1.0000 against the query vector of
  the same chain (13 chains of 8 entries; before, 0.19 to 0.83 for 9 chains
  cut where their numbering drops, and 0.999 for 2 with free amino acids
  appended). **Rebuild chain and assembly databases, and recompute chain,
  assembly and residue embeddings, made from PDB files, or from mmCIF and
  BinaryCIF files not laid out as the archive's, with chains that one ID
  splits**; a database built from them may hold a chain's id more than once.
- **`max_res` counts the residues of the chains a call embeds, as `min_res`
  counts them.** `FoldMatch.residue_embedding_by_chain` and
  `residue_embedding_by_assembly` (`fm-search query structure --max-res`,
  `--granularity chain` and `assembly`) counted every amino-acid residue of
  the structure or assembly, with its free amino acids, its chains under
  `min_res` and the pieces that a chain leaves out, even with `--chain-id`,
  so a chain within the limit was refused in a structure above it;
  `residue_embedding` and `structure_embedding` counted the same, or the
  atoms with the chain ID, their free amino acids included. Each now counts
  the protein chains it is about to embed, as `min_res` counts them (polymer
  amino-acid residues, piece by piece, summed over the pieces kept), all of
  them together: those with `chain_id` (in an assembly, one per copy), else
  every protein chain of the structure or assembly. Over `max_res`, the call
  raises `ValueError` before any ESM3 pass, with the count, the limit and
  what was counted. With no chain to embed, it gives what it gives without
  `max_res` (`residue_embedding` and `structure_embedding` their
  `ValueError`, the other two `{}`, and `fm-search query structure` its error
  that no chain qualifies), where the old count could refuse it as too long.
  Structures and chains refused before may now be embedded. The `--max-res`
  help says what is counted.
- **Chain and assembly embeddings saved as CSV.** `fm-embedding` and
  `inference` write a chain or assembly embedding to CSV (their default
  `--output-format`) as one column, one value per line. `fm-search build
  embeddings` and `fm-search query embedding` read a CSV of several rows as
  residue embeddings and averaged its rows, so each vector became one number:
  a database built from such files had dimension 1 (every score was +1 or -1),
  and a query with them failed. A one-column CSV is now read as one vector
  (and residue embeddings are refused: see *Changed*). **Rebuild
  databases built from CSV chain or assembly embeddings**: `fm-search stats`
  shows them as `Dimension: 1`, and the `fm-search query embedding`,
  `query db`, `cluster` and `similarity-graph` results made with them are
  meaningless (every score is 1 or -1).
- **Parquet embedding files, embedding dimensions and repeated ids.**
  `fm-search build embeddings` and `fm-search query embedding` read one
  embedding vector per row of a Parquet file's `embedding` column, a list of
  floating-point values, and every row of a file must hold as many values as
  its first. A file of residue embeddings, each chain's residue table
  flattened into one row as FoldMatch wrote them, was cut into equal slices
  that mixed chains (without a message when the values divided evenly) or
  stopped on a numpy error that named no file. It is now refused at any
  `--load-batch-size`, naming the file: compute chain embeddings with
  `fm-embedding from-structures chain` or `fm-embedding from-sequences chain`.
  A row with no embedding, whose batch was sliced the same way, a row with a
  NaN or infinite value (never found, and as a query it found nothing) and an
  `embedding` column of another type (a list of integers too, read before as
  floats) are refused, naming the row or the type. An unreadable Parquet file
  (empty, cut short by an interrupted run, or corrupted) and a file without
  the `id` or `embedding` column are now refused, naming the file; they
  stopped with pyarrow's message, which named no file, or with a traceback.
  The embeddings read must have one dimension: files of two dimensions
  stopped a build with a numpy error that named no file (or a FAISS error
  with `--index-type ivf_pq`), and query embeddings of another dimension than
  the database's stopped `fm-search query embedding` with an empty FAISS
  `AssertionError`. Both are now refused as they are read, before FAISS,
  naming the files (a Parquet file with its row) and the dimensions. A `.pt`
  and a `.csv` file of one name in one folder (left, for example, by a
  residue run and a chain run that wrote to one folder) were read as two
  embeddings of one id; they are now refused, and `--file-extension .pt` or
  `--file-extension .csv` reads one kind. A flat row stores no shape, so residue
  embeddings whose rows all have one length (a single chain, or chains of one
  length) still build a database, whose dimension is that length, but are
  refused as queries. **Rebuild databases built from Parquet residue
  embeddings** (`fm-search stats` shows a dimension other than that of the
  chain embeddings, such as 236544) **or from a folder that held a `.pt` and
  a `.csv` of one id.**
- **A structure name that names two files or URLs is refused.** The commands
  that read structures read a structure once for all the input rows that name
  it (one per chain or assembly), from a file or a download alike, so rows
  that gave one name to two files got the first file's structure for both.
  Such an input now stops with an error that names the files when the command
  reads its input list. A structure folder that holds one entry in two
  formats or compressions (`1acb.cif` and `1acb.cif.gz` both take the name
  `1acb`): keep one copy, or choose one with `--structure-file-extension`
  (`fm-embedding`) or `--file-extension` (`fm-search build structures`). An
  `inference` input that gives one name to several files, such as models of
  one target: give each its own name. One file under two spellings (a
  relative path, a link, a case variant on a case-insensitive volume) is one
  file. **Recompute embeddings, and rebuild databases, made from inputs that
  gave one name to more than one file or URL.**
- **`--log-level warn` sets up logging.** It set up none in every command of
  `fm-search`, `fm-embedding` and `inference` (the option's value is `warn`,
  the set-up looked for `warning`), so a warning printed in Python's default
  format (`WARNING:root:...`, `WARNING:foldmatch.<module>:...` or the bare
  message). It now prints in the format of `--log-level debug`
  (`> time - [proc-N] - WARNING - file:line - message`), at WARNING level. As
  at `debug`, Lightning's own INFO lines (`GPU available: ...`) still show;
  only `info` hides them.
- **`fm-search build structures --granularity assembly` now honours `--min-res`
  and `--batch-size`.** Both were dropped for assembly builds, so every chain
  went into the assembly embedding, while `fm-search query structure
  --granularity assembly` skips chains shorter than `--min-res`; database and
  query embeddings were computed over different chains. Assembly databases
  built with the default `--min-res 10` now leave out chains shorter than 10
  residues, so rebuild existing ones to match queries.
- **Multi-GPU runs without SLURM lost the embeddings of every rank but rank 0.**
  Each process drew its own shard-name prefix, so rank 0 built the database, or
  ran the query, from its own shard only and silently dropped the others. All
  ranks now share one run id (see *Changed*). Rebuild databases that were built
  outside SLURM on more than one device (`--devices` greater than 1, or the
  default `--devices auto` on a machine with several GPUs).
- **Apple Silicon GPU (MPS).** On Apple Silicon, where MPS is the default
  accelerator, embedding failed in the chain aggregator with
  `NotImplementedError: aten::_nested_tensor_from_mask_left_aligned`. The
  `PYTORCH_ENABLE_MPS_FALLBACK=1` workaround made it run, but gave wrong chain
  embeddings for chains padded within a batch, that is with `--batch-size`
  greater than 1. **Rebuild databases and redo embeddings computed on MPS with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` and `--batch-size` greater than 1.** FASTA
  input now runs on MPS without the fallback flag and agrees with CPU to about
  1e-6 (relative), and the MPS memory cache is released after every batch.
  Structure input on MPS still needs `PYTORCH_ENABLE_MPS_FALLBACK=1`, because
  ESM3's structure encoder uses an operator MPS lacks; its chain and assembly
  embeddings are now correct at any batch size as well. Chain and assembly
  embeddings on CPU and CUDA are byte-identical to before (CUDA checked on an
  A100). Code that calls `ResidueEmbeddingAggregator` directly with a padding
  mask outside the batched inference fast path (autograd enabled, training, or
  an unbatched input with a mask) now also leaves the padded rows out of the
  sum, which changes, and corrects, its output.
- **`EmbeddingComputer.compute_from_fasta` (Python API) in a multi-process
  run** now returns on every rank only once every rank has finished the pass.
  Rank 0 could read the shards right after it returned and find another
  rank's shard not yet written ("Found 1 embeddings in 1 shard(s) ... a
  rank's shard is missing"). `fm-search` already waited and is unaffected.
- **A FASTA header without an id** (a line holding only `>`) is now refused
  before the ESM3 pass with a one-line error naming the file and position;
  it used to stop the build with an `IndexError` traceback.
- **A multi-rank `fm-search query sequences` writes the queries in input
  order.** With more than one rank (`--devices 2`, several GPUs, or the tasks
  of a multi-task `srun` step), rank 0 read the query embeddings rank by rank,
  so the output file listed rank 0's queries first, then rank 1's, and so on,
  under `--scoring blosum62` (with or without Stage 2) and `--scoring
  esm-codes` alike. The queries are now written in the order of the FASTA
  file, so a multi-rank run writes the file a 1-rank run writes from the same
  embeddings; each query's hits were already the same. A 1-rank run's output
  is unchanged. A repeated query id (accepted, with a warning, only under
  BLOSUM62) is reported once, at its first record, with the hits of its last
  record, as on one rank; on several ranks it could take its embedding scores
  and candidates from one record and its alignments from the other. Stage 2
  now also aligns the record that Stage 1 searched when the last record of a
  repeated id is shorter than `--min-res-n`: before, that short record's
  sequence was aligned, even on one rank (the one case in which a 1-rank
  run's output changes).

- **The help of `inference` and `fm-search`.** Only help text changes.
  `--log-level` read "Number of nodes to use for inference of embeddings." in
  every `inference` embedding command and in `fm-search stats`, `cluster`,
  `build structures`, `query structure` and `query db`. `inference
  structure-embedding` named a pandas DataFrame file, an option `out-df-name`
  that does not exist and a JSON file; it writes
  `<output-name>-<rank>.parquet`. `inference residue-embedding`,
  `chain-embedding`, `assembly-embedding` and `complete-embedding` named one
  output format, whatever `--output-format` said (`complete-embedding` a
  single JSON file, where its default is one CSV file per chain or assembly);
  they now name the default and say what each format writes. `chain-embedding`
  and `assembly-embedding` also name the CSV residue embeddings they read
  (`--res-embedding-format`).
- **`inference` runs that destroyed their own output or input, with exit
  status 0.** Two rows of one embedding name kept the last row's embedding
  only; unnamed assemblies all went to `nan.csv`; an embedding name `../x`
  replaced a file outside the output folder; `chain-embedding` into its
  residue folder, in the residue format, replaced every residue embedding by a
  chain vector; `complete-embedding` with `--output-format pt` and one folder
  for residues and chains did the same, then failed in its assembly step; and
  the recipe of P9.H2, `--output-assembly-name chain-inference` with one
  folder for chains and assemblies, brought back the loss it fixed. All are
  now refused before any model is built (see *BREAKING — an `inference` run
  refuses outputs that would replace each other or its inputs* and *BREAKING
  — every row of a list has an embedding name of its own*).
- **`fm-embedding` runs that lost their own output or input, with exit status
  0.** `from-sequences` kept the last of the records of one id, wrote a record
  `../x` outside the output folder, replaced its own `--fasta-file` when it was
  named like an output, and on macOS kept one of two ids that differ in case;
  `from-structures` kept one of two chains `A` and `a` of a structure on
  macOS. Each is now refused (see the two *BREAKING* entries on
  `fm-embedding from-sequences` and on the csv and pt writers).
- **`fm-search` commands that destroyed their own input, with exit status 0.**
  `--output-file <db path>.index` (or `.metadata`, `.sequences`, ...)
  replaced a file of the database the command had just searched, so every
  later command on it failed until a rebuild; an output file named like the
  query FASTA, structure or embedding file replaced it; and `query embedding`
  dropped every query but one of a repeated id. Each is now refused (see the
  two *BREAKING* entries on `fm-search`).
- **A step-2 build could delete another build's lock on Linux.** At its
  end, `fm-search build sequences-index` removed `<output-db>.build.lock`
  when the file there had the inode of its own lock; Linux can give that
  inode to a lock created right after the first was deleted, so a lock
  deleted by hand during a build, then taken by a second build, was removed
  by the first. The lock now holds a random `lock_id`, and a build removes
  only the lock with its own.
- **`complete-embedding --src-assembly-file /dev/fd/N` on Linux** named, in
  its refusals, the file the descriptor is open on instead of the path given,
  and `/dev/stdin` from a pipe was refused as a file that does not exist. A
  path under `/dev` is now taken as given (other paths are resolved as
  before).

## [0.10.0] - 2026-09-15

### Added

- **Compositional bias correction** (`--comp-bias-corr`, `--comp-bias-corr-scale`)
  on `fm-search query sequences` and `fm-search query db`, and as
  `comp_bias_corr` / `comp_bias_corr_scale` on `align_candidates`. A port of
  mmseqs's correction: each query position's score is offset according to its
  local (40-residue) composition before Smith-Waterman, which lowers scores of
  locally biased regions. With the same settings, raw scores, bit scores and
  E-values match `mmseqs search` 18-8cc5c on identical pairs, except for
  alignments containing a non-standard residue (`X`, `B`, `Z`, `J`, `U`, `O`,
  `*`), which FoldMatch still scores with biotite's BLOSUM62. Off by default,
  unlike mmseqs: on a 48-query SCOP benchmark it cost more true positives than
  the false positives it removed. When on, E-values rise (median about 1.8x on
  that benchmark) and alignments over biased regions get shorter, lowering
  coverage. Pass `--comp-bias-corr` to compare with a default mmseqs run.

## [0.9.0] - 2026-08-27

### Added

- **`--max-evalue`** on `fm-search query sequences` and `fm-search query db`:
  drops Stage-2 hits whose database E-value is above the threshold. Defaults to
  `1e-3`, so a default search now reports only statistically significant
  alignments; pass `--max-evalue inf` to keep every hit. Because the E-value is
  a significance quantity, the option turns the significance pass on even when
  neither `evalue` nor `bits` is in `--format-output` (and so, under
  `--significance-mode default`, requires the 11/1 gap penalties).

### Changed

- **`--min-coverage` now defaults to `0.8`** (was `0.0`) on `fm-search query
  sequences` and `fm-search query db`, matching the coverage cut-off other
  alignment tools apply by default. Pass `--min-coverage 0` for the previous
  behaviour.
- **`align_candidates` thresholds now default to the CLI's values**
  (`min_seq_identity=0.3`, `min_coverage=0.8`, `max_evalue=1e-3`), so the Python
  entry point filters like the equivalent `fm-search query` run instead of
  returning every alignment. Callers that want the old permissive behaviour pass
  `min_coverage=0.0, max_evalue=None`; `max_evalue=None` is also required when
  no `subject_db_size` is available, since an E-value threshold needs a search
  space.

## [0.8.0] - 2026-07-21

This release reworks how `fm-search` reports results and makes its alignment
statistics calibrated. **It contains breaking changes to every `fm-search query`
command and to `cluster` / `similarity-graph`** — see *Migration* below.

### Added

- **Selectable output columns.** Sequence-identity (Stage-2) searches accept
  `--format-output`, a comma-separated list of columns, e.g.
  `--format-output "query,target,evalue,qaln,taln"`. Twenty-five columns are
  supported: `query`, `target`, `embscore`, `evalue`, `gapopen`, `pident`,
  `fident`, `nident`, `qstart`, `qend`, `qlen`, `tstart`, `tend`, `tlen`,
  `alnlen`, `raw`, `bits`, `cigar`, `qseq`, `tseq`, `qaln`, `taln`, `mismatch`,
  `qcov`, `tcov`. An unknown column fails immediately with the supported list.
- **`embscore` column**, carrying the Stage-1 embedding similarity score through
  to the Stage-2 output so both stages' files line up.
- **`--format-mode`**, selecting the layout of the output file: `headless_tsv`
  (tab-separated with no header row — the default, and the previous behaviour)
  or `tsv` (the same, with a header row naming each column). Available on every
  command that writes a result file: the four `query` commands and `cluster`.
  The header names the columns actually emitted, in the order emitted, so for a
  sequence-identity search it reflects your `--format-output` selection.
  `--format-mode` controls the file's shape; `--format-output` controls which
  columns it contains.
- **Calibrated alignment significance.** E-values and bit scores are now computed
  from published Karlin-Altschul constants with a finite-size-corrected search
  space, making them exact and reproducible rather than sampled. Selected with
  `--significance-mode`: `default` (calibrated; requires the default gap
  penalties 11/1) or `sampled` (any gap penalties, magnitudes relative-only).
  `--significance-sample-size` tunes the `sampled` estimator.
- **Bit score** (`bits`) and raw alignment score (`raw`) as reportable columns.
- **Faster Stage-2 alignment.** Work is parallelized over (query, subject)
  *candidate pairs* rather than whole queries, so a single query with many
  candidates now uses the whole node. `--align-workers` sets the pool size and
  defaults to all CPUs, widening any scheduler CPU pinning.
- The significance pass is skipped automatically when neither `evalue` nor
  `bits` is requested, which also lifts the 11/1 gap-penalty restriction for
  those runs.

### Changed

- **BREAKING — `--output-csv` is now `--output-file`, and it is required** on
  `query structure`, `query embedding`, `query sequences`, and `query db`.
- **BREAKING — all result files are now tab-separated with no header row**
  (previously comma-separated with a header). Pass `--format-mode tsv` to get a
  header row back.
- **BREAKING — results are always written to a file.** Printing results to
  stdout when no output path was given has been removed.
- **BREAKING — the Stage-2 column set changed.** Output is now the
  `--format-output` columns, defaulting to
  `query,target,embscore,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits`.
  The previous `Query,Rank,Match,EmbScore,SeqIdentity_aln,...` columns are gone.
- **BREAKING — an embedding-only search emits three columns**,
  `query`, `target`, `embscore`. The former `Rank` column is gone; hits are
  ordered best-first, so rank is the row order.
- **BREAKING — `cluster --output` is now `--output-file`**, and its output is
  tab-separated with no header: `chain_id`, `cluster_id`, `cluster_size`.
  The default filename changed from `clusters.csv` to `clusters.tsv`.
- **BREAKING — `similarity-graph --output` is now `--output-file`.** The file is
  still GraphML.
- Relicensed from the EvolutionaryScale Cambrian Non-Commercial License
  Agreement to the **MIT License**.

### Removed

- **BREAKING — cluster JSON output.** `cluster` previously emitted JSON when the
  output filename ended in `.json`; output is always TSV now.
- **BREAKING — `EmbeddingDatabase.print_results()` and
  `EmbeddingDatabase.export_results()`** (Python API). Use
  `foldmatch.search.output.write_embedding_results(results, output_file)`.

### Internal

- New `foldmatch.search.output` module owning the tabular output contract for
  every command — column sets, per-field renderers, and the shared writer.
  `alignment.py` and `clustering.py` are now computation-only.

### Migration

| Before | After |
|---|---|
| `fm-search query structure --db-path db --query-structure q.cif` | add `--output-file hits.tsv` (now required) |
| `--output-csv results.csv` | `--output-file results.tsv` |
| `fm-search cluster --output clusters.csv` | `fm-search cluster --output-file clusters.tsv` |
| `fm-search cluster --output clusters.json` | no JSON equivalent; output is TSV |
| `fm-search similarity-graph --output g.graphml` | `--output-file g.graphml` |
| parsing results with `csv.reader(f)` after skipping a header | split on `\t`; there is no header row by default (`--format-mode tsv` restores one) |
| `db.export_results(results, path)` | `output.write_embedding_results(results, path)` |

Scripts that read the Stage-2 columns positionally must be updated: pass an
explicit `--format-output` to pin the column order you expect.

## 0.7.3 and earlier

Not tracked in this file; see the git history and release tags.

[0.10.0]: https://github.com/rcsb/foldmatch/releases/tag/v0.10.0
[0.9.0]: https://github.com/rcsb/foldmatch/releases/tag/v0.9.0
[0.8.0]: https://github.com/rcsb/foldmatch/releases/tag/v0.8.0
