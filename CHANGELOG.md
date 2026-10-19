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
  `--comp-bias-corr`, `--significance-mode`, as well as `--no-seq-identity`,
  are refused (exit status 2). Before ESM3 is loaded, and on every rank
  alike, the query is refused with a one-line message (exit status 1) when
  its FASTA has duplicate ids, `<` or `>` in a sequence, or non-ASCII
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
  reports no hits. Stage 2 runs in a pool of worker processes
  (`--align-workers`, all CPUs by default) that load neither torch nor
  ESM3, after ESM3 and the aggregator have been freed; its output does not
  depend on the number of workers; a worker that is killed (for example by
  the out-of-memory killer) stops the run with an error. The query
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
