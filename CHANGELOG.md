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

[0.8.0]: https://github.com/rcsb/foldmatch/releases/tag/v0.8.0
