"""Tabular output contract for search results.

This module owns everything about *how results are presented*: which columns
exist, how each one renders, and how rows reach disk. The modules that compute
results (:mod:`alignment`, :mod:`clustering`, the embedding search) stay pure
computation and hand their values here.

Every result file has the same shape — delimited rows, one record per line, no
header — so downstream parsing never has to branch on which command produced a
file, and a column name means the same thing everywhere.

Three column sets are defined here:

* :data:`EMBEDDING_OUTPUT_FIELDS` — an embedding-only search.
* :data:`OUTPUT_FIELD_DESCRIPTIONS` — the selectable Stage-2 alignment columns
  (``--format-output``), of which :data:`DEFAULT_OUTPUT_FIELDS` is the default.
* :data:`CLUSTER_OUTPUT_FIELDS` — ``fm-search cluster``.

--------------------------------------------------------------------------- #
mmseqs-style tabular output

Stage-2 hits are reported as a subset of the same columns MMseqs2 emits from
``mmseqs convertalis``, selected with ``--format-output`` and written as
tab-separated rows with no header (MMseqs2's ``--format-mode 0`` default). The
default column set below matches MMseqs2's own default.

Fidelity notes:
  * ``fident`` is nident/alnlen (identical columns over the whole alignment
    length, gaps included) — the same fraction MMseqs2 reports; ``pident`` is
    that fraction as a percentage (100*fident), per the field's name.
  * ``bits`` is emitted as an integer. BLOSUM62 keeps MMseqs2's
    ``int(bitScore + 0.5)``, so its output is unchanged; it truncates towards
    zero, so a negative bit score (``--significance-mode sampled`` with low
    gap costs gives some) prints one higher: -7.6 prints -7. esm-codes bit
    scores are rounded half up, ``floor(bitScore + 0.5)`` (method S11): -7.6
    prints -8.
  * ``evalue`` is ``%.3E``; under esm-codes it is rendered from ln E, so it
    never underflows (``5.076E-435``).
  * ``gapopen`` counts maximal gap RUNS (query- and target-side independently),
    not gap characters.
  * CIGAR ops follow MMseqs2: ``M`` aligned pair, ``D`` gap in query (deletion),
    ``I`` gap in target (insertion).

esm-codes columns (``query sequences --scoring esm-codes``, method S9-S13):
``raw`` is ``S_f`` as ``%.3f`` (an integer under BLOSUM62, as before); the
optional ``rawrev`` (``S_r``), ``q90`` (the query's panel quantile) and
``rawadj`` (``S' = S_f - S_r - Q90``) are ``%.3f``, and ``ecal`` is ``1`` for
a hit inside the calibrated domain (S13), else ``0``. These four render empty
under BLOSUM62 (and ``query db``); ``evalue``, ``bits``, ``q90`` and
``rawadj`` render empty when the panel pass was skipped (D11).
--------------------------------------------------------------------------- #
"""

import logging
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

from foldmatch.search.code_statistics import format_evalue_from_log, render_bits
from foldmatch.types.api_types import FormatMode

if TYPE_CHECKING:  # avoids a runtime import cycle; alignment imports this module
    from .alignment import Hit

logger = logging.getLogger(__name__)

# Default column separator. The active FormatMode is the authority (see
# _FORMATS); this is kept for callers that render a row outside a writer.
DELIMITER = "\t"


@dataclass(frozen=True)
class _FormatSpec:
    """The file-level shape a :class:`FormatMode` selects."""
    delimiter: str
    header: bool


# One entry per FormatMode. A new layout (say a comma-separated or commented-
# header variant) is added by defining the member in FormatMode and its spec
# here — no writer needs to change, since all three go through _emit().
_FORMATS: Dict[FormatMode, _FormatSpec] = {
    FormatMode.headless_tsv: _FormatSpec(delimiter="\t", header=False),
    FormatMode.tsv:          _FormatSpec(delimiter="\t", header=True),
}


def format_spec(format_mode: FormatMode) -> _FormatSpec:
    """Resolve a :class:`FormatMode` (or its plain string value) to its spec."""
    try:
        return _FORMATS[FormatMode(format_mode)]
    except ValueError:
        raise ValueError(
            f"Unknown format mode {format_mode!r}; expected one of: "
            f"{', '.join(m.value for m in FormatMode)}."
        ) from None


def write_rows(rows: Iterable[str], output_file: str) -> int:
    """Write pre-rendered rows verbatim to ``output_file``; return the count."""
    n = 0
    with open(output_file, "w", newline="") as fh:
        for row in rows:
            fh.write(row)
            fh.write("\n")
            n += 1
    return n


def _emit(field_names: Iterable[str], rows: Iterable[str], output_file: str,
          format_mode: FormatMode) -> int:
    """Write ``rows``, prefixed by a header when the format asks for one.

    Returns the number of *data* rows, so callers can log a count that doesn't
    silently include the header.
    """
    spec = format_spec(format_mode)

    def _all():
        if spec.header:
            yield spec.delimiter.join(field_names)
        yield from rows

    written = write_rows(_all(), output_file)
    return written - 1 if spec.header else written


# --------------------------------------------------------------------------- #
# Embedding-only search output
# --------------------------------------------------------------------------- #

# Columns emitted by an embedding-only search (i.e. no Stage-2 alignment: the
# structure/embedding queries, and the sequence queries when Stage 2 is off).
# These reuse the Stage-2 field names so the columns line up across both files.
EMBEDDING_OUTPUT_FIELDS: Tuple[str, ...] = ("query", "target", "embscore")


def format_embedding_score(score: float) -> str:
    """Render an embedding similarity score.

    Single source of truth, so an embedding-only file and Stage-2's
    ``embscore`` column format the same number identically.
    """
    return f"{score:.6f}"


def write_embedding_results(
        results: Dict[str, Tuple[List[str], List[float]]],
        output_file: str,
        format_mode: FormatMode = FormatMode.headless_tsv,
) -> None:
    """Write embedding-search hits as ``query, target, embscore`` rows.

    ``results`` is the ``{query_id: ([target_ids], [scores])}`` mapping the
    embedding search returns. Hits are emitted in the order the search produced
    them (descending similarity), so rank is simply the row order — there is no
    separate rank column.
    """
    delimiter = format_spec(format_mode).delimiter

    def _rows():
        for query_id, (target_ids, scores) in results.items():
            for target_id, score in zip(target_ids, scores):
                yield delimiter.join((query_id, target_id, format_embedding_score(score)))

    n = _emit(EMBEDDING_OUTPUT_FIELDS, _rows(), output_file, format_mode)
    logger.info(f"Wrote {n} result row(s) to {output_file}")


# --------------------------------------------------------------------------- #
# Clustering output
# --------------------------------------------------------------------------- #

# Columns emitted by ``fm-search cluster``. Clustering describes the database
# rather than a query/target pair, so it has its own column set — but the same
# tab-separated, no-header file convention.
CLUSTER_OUTPUT_FIELDS: Tuple[str, ...] = ("chain_id", "cluster_id", "cluster_size")


def write_cluster_results(
        assignments: Iterable[Tuple[str, int, int]],
        output_file: str,
        format_mode: FormatMode = FormatMode.headless_tsv,
) -> int:
    """Write cluster assignments in :data:`CLUSTER_OUTPUT_FIELDS` order.

    ``assignments`` is an iterable of ``(chain_id, cluster_id, cluster_size)``;
    the caller owns which rows to include (e.g. a ``min_cluster_size`` filter).
    Emitting the columns here keeps their order next to the declared field list,
    which the CLI help text is generated from. Returns the number of data rows.
    """
    delimiter = format_spec(format_mode).delimiter
    return _emit(
        CLUSTER_OUTPUT_FIELDS,
        (delimiter.join((str(chain_id), str(cluster_id), str(cluster_size)))
         for chain_id, cluster_id, cluster_size in assignments),
        output_file,
        format_mode,
    )


# --------------------------------------------------------------------------- #
# Stage-2 alignment output (selectable columns)
# --------------------------------------------------------------------------- #

# Ordered registry of every supported field -> one-line description (used for
# validation and the CLI help text). Order here is the canonical field order.
OUTPUT_FIELD_DESCRIPTIONS: Dict[str, str] = {
    "query": "Query sequence identifier",
    "target": "Target sequence identifier",
    "embscore": "Embedding similarity score from the Stage-1 prefilter",
    "evalue": "E-value",
    "gapopen": "Number of gap open events (not the number of gap characters)",
    "pident": "Percentage of identical matches",
    "fident": "Fraction of identical matches",
    "nident": "Number of identical matches",
    "qstart": "1-indexed alignment start position in query sequence",
    "qend": "1-indexed alignment end position in query sequence",
    "qlen": "Query sequence length",
    "tstart": "1-indexed alignment start position in target sequence",
    "tend": "1-indexed alignment end position in target sequence",
    "tlen": "Target sequence length",
    "alnlen": "Alignment length (number of aligned columns)",
    "raw": "Raw alignment score (blosum62: the integer Smith-Waterman score; esm-codes: S_f)",
    "bits": "Bit score",
    "cigar": "Alignment CIGAR string (M match, D gap in query, I gap in target)",
    "qseq": "Query sequence",
    "tseq": "Target sequence",
    "qaln": "Aligned query sequence with gaps",
    "taln": "Aligned target sequence with gaps",
    "mismatch": "Number of mismatches",
    "qcov": "Fraction of query sequence covered by alignment",
    "tcov": "Fraction of target sequence covered by alignment",
    "rawrev": "esm-codes: score of the row-reversed query, S_r (empty under blosum62)",
    "q90": "esm-codes: the query's 0.9 quantile of S over the score panel, Q90 (empty under blosum62)",
    "rawadj": "esm-codes: adjusted score S_f - S_r - Q90, the score behind evalue and bits (empty under blosum62)",
    "ecal": "esm-codes: 1 if the hit is inside the calibrated domain of the E-values, else 0 (empty under blosum62)",
}

SUPPORTED_OUTPUT_FIELDS: Tuple[str, ...] = tuple(OUTPUT_FIELD_DESCRIPTIONS)

# Fields only esm-codes Stage 2 computes; they render empty under BLOSUM62.
ESM_CODES_OUTPUT_FIELDS: Tuple[str, ...] = ("rawrev", "q90", "rawadj", "ecal")

# MMseqs2's own default column set, plus the Stage-1 embedding score right
# after the identifiers (this tool's prefilter has no MMseqs2 equivalent, and
# keeping it in the default output makes the two stages' files line up).
DEFAULT_OUTPUT_FIELDS: Tuple[str, ...] = (
    "query", "target", "embscore", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
)
DEFAULT_FORMAT_OUTPUT: str = ",".join(DEFAULT_OUTPUT_FIELDS)

# Fields whose value comes from Karlin-Altschul significance; requesting none of
# these lets the caller skip the significance pass entirely (and, in 'default'
# mode, its 11/1 gap-penalty requirement).
SIGNIFICANCE_FIELDS: frozenset = frozenset({"evalue", "bits"})


def parse_format_output(spec: Optional[str]) -> List[str]:
    """Parse a comma-separated ``--format-output`` spec into validated fields.

    ``None`` yields the default column set. Order and repeats are preserved
    (each field is emitted where the user placed it). Raises ``ValueError`` with
    the full supported list on an empty spec or any unknown field.
    """
    if spec is None:
        return list(DEFAULT_OUTPUT_FIELDS)
    fields = [tok.strip() for tok in spec.split(",") if tok.strip()]
    if not fields:
        raise ValueError(
            "Empty --format-output; give a comma-separated list of fields, e.g. "
            f"'{DEFAULT_FORMAT_OUTPUT}'."
        )
    unknown = [f for f in fields if f not in OUTPUT_FIELD_DESCRIPTIONS]
    if unknown:
        raise ValueError(
            f"Unknown --format-output field(s): {', '.join(unknown)}. "
            f"Supported fields: {', '.join(SUPPORTED_OUTPUT_FIELDS)}."
        )
    return fields


def needs_significance(fields: List[str]) -> bool:
    """Whether any requested field requires the significance (lambda/K) pass."""
    return any(f in SIGNIFICANCE_FIELDS for f in fields)


def _format_3f(value: Optional[float]) -> str:
    """``%.3f``, empty for None; a value that rounds to zero prints ``0.000``, never ``-0.000``."""
    if value is None:
        return ""
    text = f"{value:.3f}"
    return "0.000" if text == "-0.000" else text


def _render_raw(m) -> str:
    """The integer BLOSUM62 score as before; the float esm-codes S_f as ``%.3f``."""
    return str(m.score) if isinstance(m.score, numbers.Integral) else _format_3f(m.score)


def _render_bits(m) -> str:
    """BLOSUM62 (an integer score): MMseqs2's ``int(bits + 0.5)``, unchanged; esm-codes: ``floor(bits + 0.5)``."""
    if m.bit_score is None:
        return ""
    if isinstance(m.score, numbers.Integral):
        return str(int(m.bit_score + 0.5))
    return str(render_bits(m.bit_score))


def _render_evalue(m) -> str:
    """From ln E when the aligner gives it (esm-codes: no underflow), else the E-value as ``%.3E``."""
    if m.log_evalue is not None:
        return format_evalue_from_log(m.log_evalue)
    return f"{m.evalue:.3E}" if m.evalue is not None else ""


# Per-field renderers: ``(query_id, hit) -> str``. Significance-derived fields
# render to "" when significance was not computed; heavy strings render to ""
# when they were not requested (and so were never materialized); the esm-codes
# fields render to "" under BLOSUM62.
_FIELD_RENDERERS: Dict[str, Callable[[str, "Hit"], str]] = {
    "query":    lambda qid, h: qid,
    "target":   lambda qid, h: h.subject_id,
    "embscore": lambda qid, h: format_embedding_score(h.emb_score),
    "evalue":   lambda qid, h: _render_evalue(h.metrics),
    "gapopen":  lambda qid, h: str(h.metrics.gap_open),
    "pident":   lambda qid, h: f"{100.0 * h.metrics.identity_aln:.3f}",
    "fident":   lambda qid, h: f"{h.metrics.identity_aln:.3f}",
    "nident":   lambda qid, h: str(h.metrics.n_ident),
    "qstart":   lambda qid, h: str(h.metrics.q_start),
    "qend":     lambda qid, h: str(h.metrics.q_end),
    "qlen":     lambda qid, h: str(h.metrics.query_len),
    "tstart":   lambda qid, h: str(h.metrics.t_start),
    "tend":     lambda qid, h: str(h.metrics.t_end),
    "tlen":     lambda qid, h: str(h.metrics.subject_len),
    "alnlen":   lambda qid, h: str(h.metrics.aln_len),
    "raw":      lambda qid, h: _render_raw(h.metrics),
    "bits":     lambda qid, h: _render_bits(h.metrics),
    "cigar":    lambda qid, h: h.metrics.cigar or "",
    "qseq":     lambda qid, h: h.metrics.q_seq or "",
    "tseq":     lambda qid, h: h.metrics.t_seq or "",
    "qaln":     lambda qid, h: h.metrics.q_aln or "",
    "taln":     lambda qid, h: h.metrics.t_aln or "",
    "mismatch": lambda qid, h: str(h.metrics.mismatch),
    "qcov":     lambda qid, h: f"{h.metrics.query_coverage:.3f}",
    "tcov":     lambda qid, h: f"{h.metrics.subject_coverage:.3f}",
    "rawrev":   lambda qid, h: _format_3f(h.metrics.score_rev),
    "q90":      lambda qid, h: _format_3f(h.metrics.q90),
    "rawadj":   lambda qid, h: _format_3f(h.metrics.score_adj),
    "ecal":     lambda qid, h: "" if h.metrics.ecal is None else ("1" if h.metrics.ecal else "0"),
}


def format_row(query_id: str, hit: "Hit", output_fields: List[str],
               delimiter: str = DELIMITER) -> str:
    """Render one hit as a delimited row over ``output_fields`` (in order)."""
    return delimiter.join(_FIELD_RENDERERS[field](query_id, hit) for field in output_fields)


def write_aligned_results(
        results: Dict[str, List["Hit"]],
        output_fields: List[str],
        output_file: str,
        format_mode: FormatMode = FormatMode.headless_tsv,
) -> None:
    """Write Stage-2 hits as MMseqs2-style delimited rows.

    One row per (query, surviving hit) over the requested ``output_fields`` in
    the order given. Under the default headless format this matches ``mmseqs
    convertalis --format-mode 0``, and the embedding-only output written by
    :func:`write_embedding_results`.
    """
    delimiter = format_spec(format_mode).delimiter

    def _rows():
        for query_id, hits in results.items():
            for hit in hits:
                yield format_row(query_id, hit, output_fields, delimiter)

    n = _emit(output_fields, _rows(), output_file, format_mode)
    logger.info(f"Wrote {n} alignment row(s) to {output_file}")
