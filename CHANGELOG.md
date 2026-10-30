# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, breaking changes are released in a minor bump.

## [Unreleased]

### Added

- **`--max-evalue`** on `fm-search query sequences` and `fm-search query db`:
  drops Stage-2 hits whose database E-value is above the threshold. Defaults to
  `1e-3`, so a default search now reports only statistically significant
  alignments; pass `--max-evalue inf` to keep every hit. Because the E-value is
  a significance quantity, the option turns the significance pass on even when
  neither `evalue` nor `bits` is in `--format-output` (and so, under
  `--significance-mode default`, requires the 11/1 gap penalties).
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
  the option and still ignores it; `fm-search query db` is unchanged.
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
- **Faster BLOSUM62 Stage 2.** Sequence identity is computed vectorised from the
  alignment trace, which makes `align_candidates` about 1.1-1.3x faster in a
  serial run. The output is byte-identical.
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
  before except the residue-code scoring (`--scoring esm-codes`), which refuses
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

### Fixed

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
- **Chain and assembly embeddings saved as CSV.** `fm-embedding` and
  `inference` write a chain or assembly embedding to CSV (their default
  `--output-format`) as one column, one value per line. `fm-search build
  embeddings` and `fm-search query embedding` read a CSV of several rows as
  residue embeddings and averaged its rows, so each vector became one number:
  a database built from such files had dimension 1 (every score was +1 or -1),
  and a query with them failed. A one-column CSV is now read as one vector;
  residue embeddings (one row per residue) are still averaged. **Rebuild
  databases built from CSV chain or assembly embeddings**: `fm-search stats`
  shows them as `Dimension: 1`, and the `fm-search query embedding`,
  `query db`, `cluster` and `similarity-graph` results made with them are
  meaningless (every score is 1 or -1).
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

[0.8.0]: https://github.com/rcsb/foldmatch/releases/tag/v0.8.0
