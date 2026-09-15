import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from foldmatch.types.api_types import SignificanceMode

from . import karlin_altschul
from .output import DEFAULT_OUTPUT_FIELDS

logger = logging.getLogger(__name__)

# Significance (p-/E-values, bit scores) comes in two flavors, selected by
# ``significance_mode``:
#
#   'default' — calibrated. Uses published ALP Karlin-Altschul constants and a
#       finite-size-corrected search space, so the values are exact and
#       reproducible (a table lookup, no sampling pass). Implementation and full
#       provenance in karlin_altschul.py. Only BLOSUM62 at gaps 11/1 is supported;
#       other gap penalties raise rather than fall back silently.
#       NOT comparable to BLAST: the nominal 11/1 gap cost here equals BLAST's
#       10/1, and these lambda/K differ from NCBI's published values.
#
#   'sampled' — the original behavior: lambda/K estimated by sampling random
#       alignments via biotite (see _estimate_lambda_k). The estimate does NOT
#       converge with sample size — measured at BLOSUM62 11/1, seed 0:
#           100x100   lambda 0.382  K 0.445
#           250x250   lambda 0.256  K 0.027
#           500x500   lambda 0.279  K 0.062   (this module's default)
#           1000x1000 lambda 0.222  K 0.008
#       i.e. K spans ~57x and lambda 0.22-0.38 purely as a function of the
#       sampling knobs. Since E scales linearly with K and exponentially in
#       lambda*S, the absolute values are unreliable; only the RANKING is
#       trustworthy (it is monotonic in the raw score for any lambda > 0).
#       Use this mode only for gap penalties 'default' cannot serve.

# Robinson & Robinson (1991) background amino-acid frequencies, used to generate
# the random sequences sampled when estimating lambda/K.
_ROBINSON_FREQUENCIES = {
    'A': 0.0780, 'R': 0.0512, 'N': 0.0448, 'D': 0.0536, 'C': 0.0193,
    'Q': 0.0426, 'E': 0.0629, 'G': 0.0737, 'H': 0.0220, 'I': 0.0514,
    'L': 0.0901, 'K': 0.0574, 'M': 0.0224, 'F': 0.0385, 'P': 0.0520,
    'S': 0.0712, 'T': 0.0584, 'W': 0.0133, 'Y': 0.0323, 'V': 0.0644,
}

# Which output columns require an expensive per-hit string, so the aligner can
# skip materializing (and pickling back) anything the active format never asks
# for. The columns themselves are declared and rendered in :mod:`output`; this
# is purely the compute-side gate.
@dataclass(frozen=True)
class _FieldNeeds:
    """Which heavy per-hit strings the requested columns actually require."""
    cigar: bool
    qaln: bool
    taln: bool
    qseq: bool
    tseq: bool


def _needs_from_fields(fields: List[str]) -> _FieldNeeds:
    fs = set(fields)
    return _FieldNeeds(
        cigar="cigar" in fs,
        qaln="qaln" in fs,
        taln="taln" in fs,
        qseq="qseq" in fs,
        tseq="tseq" in fs,
    )


def _available_cpus() -> int:
    """Number of CPUs usable by this process.

    Prefers ``sched_getaffinity`` so it honors SLURM/cgroup CPU pinning (a job
    allocated 8 of 64 cores sees 8); falls back to ``cpu_count`` on platforms
    without it (e.g. macOS).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _resolve_num_workers(requested: Optional[int], n_tasks: int) -> int:
    """Resolve the worker count: ``None`` -> all CPUs; never more than tasks."""
    avail = _available_cpus() if requested is None else requested
    return max(1, min(avail, n_tasks))


def _expand_cpu_affinity() -> int:
    """Widen this process's CPU affinity mask to all CPUs, returning the count.

    A process launched under a binding scheduler (e.g. NERSC ``srun`` pins each
    of N tasks to cpu_count/N cores) inherits a narrow mask; Stage-2 runs single-
    process on rank 0 and would otherwise see only that slice. Widening lets it
    (and its forked pool workers) use the whole node, so this must run *before*
    the pool is created.

    This is safe even on shared/allocated machines: the kernel intersects the
    requested set with the process's cpuset cgroup, so ``sched_setaffinity``
    cannot escape a real allocation — it just un-does soft ``--cpu-bind`` masks.
    No-op where ``sched_setaffinity`` is unavailable (non-Linux).
    """
    n = os.cpu_count() or 1
    if not hasattr(os, "sched_setaffinity"):
        return n
    before = len(os.sched_getaffinity(0))
    try:
        os.sched_setaffinity(0, range(n))
    except OSError as exc:
        logger.warning(f"Could not widen CPU affinity to all {n} CPUs: {exc}")
        return before
    after = len(os.sched_getaffinity(0))
    if after != before:
        logger.info(f"Widened CPU affinity for alignment: {before} -> {after} CPUs")
    return after

# The biotite ProteinSequence alphabet (20 canonical + ambiguity codes B/Z/X and
# the stop symbol *). Anything outside this set — selenocysteine U, pyrrolysine
# O, gaps, digits, whitespace — is mapped to X so alignment never crashes on a
# stray residue.
_BIOTITE_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYBZX*")

# Compositional bias correction — a port of MMseqs2's ``--comp-bias-corr``
# (SubstitutionMatrix::calcLocalAaBiasCorrection plus the rounding in
# SmithWaterman::ssw_init, soedinglab/MMseqs2 18-8cc5c). See
# :func:`local_composition_bias`. The offsets are computed in MMseqs2's own
# scoring alphabet and background, so they match MMseqs2 for any query letters.
# The substitution scores they are added to still come from biotite's BLOSUM62,
# which scores non-standard residues (X, B, Z, J, U, O, *) differently from
# MMseqs2, so alignments containing those can differ by a few score units.
_MMSEQS_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
# Letters MMseqs2 folds onto its alphabet (setupLetterMapping); any other letter
# outside the alphabet becomes X.
_MMSEQS_LETTER_MAP = {"B": "D", "Z": "E", "J": "L", "U": "X", "O": "X"}
# data/blosum62.out "# Background (precomputed optional)" line, in alphabet order.
_MMSEQS_BLOSUM62_BACKGROUND = (
    0.07422, 0.02469, 0.05363, 0.05431, 0.04742, 0.07415, 0.02621, 0.06792,
    0.05815, 0.09891, 0.02499, 0.04465, 0.03854, 0.03426, 0.05161, 0.05723,
    0.05089, 0.07292, 0.01303, 0.03228, 0.00001,
)
# Window over which the local composition is measured: [i-20, i+20).
_COMP_BIAS_WINDOW = 40

# Per-process globals populated by the multiprocessing initializer so the
# (immutable, picklable-but-large) substitution matrix is built once per worker.
_MATRIX = None
_GAP: Tuple[int, int] = (-11, -1)
# Karlin-Altschul E-value estimator (biotite) and the subject-DB residue total,
# also per-process. _ESTIMATOR is None unless significance_mode='sampled'.
_ESTIMATOR = None
_DB_RESIDUES: Optional[int] = None
# Calibrated ALP parameter set; None unless significance_mode='default'.
_ALP = None
# The active lambda/K (calibrated or sampled), kept so the bit score can be
# computed directly. None when significance is not being computed at all —
# this is the single gate for "is significance on?".
_LAMBDA: Optional[float] = None
_K: Optional[float] = None
# Which heavy per-hit strings the active output format requires; set per-process
# by the pool initializer so workers only build (and pickle back) what's needed.
_OUTPUT_NEEDS: "_FieldNeeds" = _FieldNeeds(False, False, False, False, False)
# Compositional bias correction scale, or None when the correction is off; set
# per-process by the pool initializer.
_COMP_BIAS_SCALE: Optional[float] = None
# MMseqs2's (integer BLOSUM62, background) pair for the correction, built lazily.
_MMSEQS_SCORING: Optional[Tuple[np.ndarray, np.ndarray]] = None


@dataclass
class AlignmentMetrics:
    """Outcome of one pairwise local alignment.

    Carries every quantity needed to render any ``--format-output`` column. The
    scalar fields are always computed (cheap); the heavy strings (``cigar``,
    ``q_aln``, ``t_aln``, ``q_seq``, ``t_seq``) are ``None`` unless the active
    output format requested them (see :class:`_FieldNeeds`).
    """
    identity_aln: float        # identical positions / alignment length == fident
    identity_shorter: float    # identical positions / length of the shorter sequence
    query_coverage: float      # aligned query residues / full query length == qcov
    subject_coverage: float    # aligned subject residues / full subject length == tcov
    aln_len: int               # alignment length (columns, gaps included) == alnlen
    score: int                 # raw Smith-Waterman score (comp-bias corrected when on) == raw
    # Karlin-Altschul significance. Scale depends on significance_mode:
    # calibrated ('default') or relative-only ('sampled').
    # Never BLAST-comparable — see module header.
    bit_score: Optional[float] = None  # normalized score (lambda*S - lnK)/ln2 == bits
    pvalue: Optional[float] = None   # pairwise p-value (n = subject length)
    evalue: Optional[float] = None   # E-value over the subject DB (n = total residues)
    # Full sequence lengths (qlen/tlen). Default 0 for the degenerate empty
    # alignment; overwritten with the real lengths otherwise.
    query_len: int = 0
    subject_len: int = 0
    # Column tallies over the alignment.
    n_ident: int = 0           # identical aligned columns == nident
    mismatch: int = 0          # aligned non-gap columns that differ == mismatch
    gap_open: int = 0          # number of maximal gap runs (both sides) == gapopen
    # 1-indexed alignment bounds within each sequence (0 for an empty alignment).
    q_start: int = 0
    q_end: int = 0
    t_start: int = 0
    t_end: int = 0
    # Heavy strings — only populated when the output format requests them.
    cigar: Optional[str] = None
    q_aln: Optional[str] = None
    t_aln: Optional[str] = None
    q_seq: Optional[str] = None
    t_seq: Optional[str] = None


@dataclass
class Hit:
    """A Stage-1 candidate annotated with its Stage-2 alignment metrics."""
    subject_id: str
    emb_score: float
    metrics: AlignmentMetrics


def sanitize_sequence(sequence: str) -> str:
    """Uppercase and coerce a sequence into the biotite protein alphabet."""
    return "".join(c if c in _BIOTITE_ALPHABET else "X" for c in sequence.strip().upper())


def _to_protein(sequence: str):
    """Build a biotite ProteinSequence from a sequence string.

    Returns ``(protein, clean_str, length)``; ``(None, "", 0)`` for an empty
    sequence. The sanitized ``clean_str`` is what is actually aligned, so the
    ``qseq``/``tseq``/``qaln``/``taln`` columns and the position indices are all
    derived from it (keeping ``qaln`` a gapped substring of ``qseq``).
    """
    import biotite.sequence as seq
    clean = sanitize_sequence(sequence)
    if not clean:
        return None, "", 0
    return seq.ProteinSequence(clean), clean, len(clean)


def _mmseqs_blosum62() -> Tuple[np.ndarray, np.ndarray]:
    """MMseqs2's integer BLOSUM62 and background, both in ``_MMSEQS_ALPHABET`` order.

    MMseqs2 regenerates its integer matrix from data/blosum62.out; on the 20
    amino acids that equals biotite's BLOSUM62, while its X row and column are a
    flat -1. readProbMatrix scales the 20 amino-acid background entries by
    ``1 - pBack[X]``.
    """
    global _MMSEQS_SCORING
    if _MMSEQS_SCORING is None:
        import biotite.sequence.align as align
        std = align.SubstitutionMatrix.std_protein_matrix()
        idx = [std.get_alphabet1().encode(c) for c in _MMSEQS_ALPHABET[:20]]
        matrix = np.full((21, 21), -1, dtype=np.int64)
        matrix[:20, :20] = std.score_matrix()[np.ix_(idx, idx)]
        background = np.array(_MMSEQS_BLOSUM62_BACKGROUND, dtype=np.float64)
        background[:20] *= 1.0 - background[20]
        _MMSEQS_SCORING = (matrix, background)
    return _MMSEQS_SCORING


def local_composition_bias(sequence: str, scale: float = 1.0) -> np.ndarray:
    """MMseqs2's per-position compositional bias correction for a query sequence.

    Returns one integer offset per residue (of the stripped, uppercased
    sequence, i.e. aligned with :func:`sanitize_sequence`). MMseqs2 adds offset
    ``i`` to the score of query position ``i`` against *every* target residue,
    lowering the score where the query is locally enriched in residues that
    score well against each other. The target is never corrected.

    For window ``W_i = [max(0, i-20), min(N, i+20))``::

        cb[i] = scale * (sum_a pBack[a] * S(q_i, a)
                         - sum_{j in W_i, j != i} S(q_i, q_j) / |W_i|)

    rounded half away from zero (``|W_i|`` counts position ``i`` although the
    sum skips it). The arithmetic follows the C++ types — a float32 accumulator,
    summed in alphabet order — so values near a .5 boundary round as MMseqs2
    rounds them.
    """
    text = sequence.strip().upper()
    n = len(text)
    if n == 0 or scale == 0:
        return np.zeros(n, dtype=np.int64)
    matrix, background = _mmseqs_blosum62()
    lookup = {c: i for i, c in enumerate(_MMSEQS_ALPHABET)}
    codes = np.fromiter(
        (lookup.get(_MMSEQS_LETTER_MAP.get(c, c), 20) for c in text), dtype=np.int64, count=n,
    )
    # Integer window sums from per-letter prefix counts: S(q_i, .) . counts(W_i).
    onehot = np.zeros((n + 1, 21), dtype=np.int64)
    onehot[np.arange(1, n + 1), codes] = 1
    prefix = np.cumsum(onehot, axis=0)
    pos = np.arange(n)
    lo = np.maximum(0, pos - _COMP_BIAS_WINDOW // 2)
    hi = np.minimum(n, pos + _COMP_BIAS_WINDOW // 2)
    rows = matrix[codes]
    window_sum = (rows * (prefix[hi] - prefix[lo])).sum(axis=1) - rows[pos, codes]
    # float deltaS_i = (float) sum; deltaS_i /= -1.0 * (float) windowLength;
    delta = (window_sum.astype(np.float32).astype(np.float64)
             / (-1.0 * (hi - lo).astype(np.float32).astype(np.float64))).astype(np.float32)
    # deltaS_i += pBack[a] * (float) subMat[a];  (double sum stored back into a float)
    for a in range(len(_MMSEQS_ALPHABET)):
        delta = (delta.astype(np.float64)
                 + background[a] * rows[:, a].astype(np.float32).astype(np.float64)).astype(np.float32)
    corrected = (np.float32(scale) * delta).astype(np.float64)
    return np.trunc(np.where(corrected < 0.0, corrected - 0.5, corrected + 0.5)).astype(np.int64)


def _positional_query(query_protein, bias: np.ndarray):
    """Score a query position-specifically: BLOSUM62 row of each residue plus its bias.

    biotite has no per-position scoring hook, so the query becomes a sequence
    over its own positions with an ``L x |protein alphabet|`` substitution
    matrix. With zero bias this reproduces the plain BLOSUM62 alignment exactly
    (score and trace).
    """
    import biotite.sequence as seq
    import biotite.sequence.align as align
    n = len(query_protein)
    alphabet = seq.Alphabet(list(range(n)))
    positional = seq.GeneralSequence(alphabet, list(range(n)))
    scores = _MATRIX.score_matrix()[query_protein.code] + bias[:, None]
    matrix = align.SubstitutionMatrix(alphabet, _MATRIX.get_alphabet2(), scores.astype(np.int32))
    return positional, matrix


def _count_runs(mask: np.ndarray) -> int:
    """Number of maximal runs of ``True`` in a 1-D boolean mask (gap-open count)."""
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.count_nonzero(mask[1:] & ~mask[:-1]))


def _run_length_encode(ops: str) -> str:
    """Collapse a per-column op string (e.g. 'MMMDDMM') to CIGAR ('3M2D2M')."""
    if not ops:
        return ""
    out: List[str] = []
    prev = ops[0]
    count = 1
    for ch in ops[1:]:
        if ch == prev:
            count += 1
        else:
            out.append(f"{count}{prev}")
            prev = ch
            count = 1
    out.append(f"{count}{prev}")
    return "".join(out)


def _derive_alignment_extras(trace, query_str: str, subject_str: str, needs: "_FieldNeeds"):
    """Compute the mmseqs column values that need the alignment trace.

    Returns ``(n_ident, mismatch, gap_open, q_start, q_end, t_start, t_end,
    cigar, q_aln, t_aln)``. The trace must be non-empty; column ``(-1, -1)`` never
    occurs (biotite emits no double-gap columns), so the three masks partition
    the columns exactly.
    """
    q_idx = trace[:, 0]
    s_idx = trace[:, 1]
    q_gap = q_idx == -1
    s_gap = s_idx == -1
    both = ~q_gap & ~s_gap

    q_arr = np.frombuffer(query_str.encode("ascii"), dtype=np.uint8)
    s_arr = np.frombuffer(subject_str.encode("ascii"), dtype=np.uint8)

    # 1-indexed bounds over each side's non-gap columns (min/max, so a terminal
    # gap column — should one ever occur — cannot skew the reported extent).
    q_res = q_idx[~q_gap]
    s_res = s_idx[~s_gap]
    q_start = int(q_res.min()) + 1
    q_end = int(q_res.max()) + 1
    t_start = int(s_res.min()) + 1
    t_end = int(s_res.max()) + 1

    # Identity/mismatch only over columns aligned on both sides.
    matches = q_arr[q_idx[both]] == s_arr[s_idx[both]]
    n_ident = int(np.count_nonzero(matches))
    aligned_pairs = int(both.sum())
    mismatch = aligned_pairs - n_ident

    # A gap-open event is one maximal run of gaps; count query- and target-side
    # runs independently (no column is a gap on both sides).
    gap_open = _count_runs(q_gap) + _count_runs(s_gap)

    cigar = q_aln = t_aln = None
    if needs.cigar:
        ops = np.empty(trace.shape[0], dtype="U1")
        ops[both] = "M"
        ops[q_gap] = "D"   # deletion: gap in query
        ops[s_gap] = "I"   # insertion: gap in target
        cigar = _run_length_encode("".join(ops.tolist()))
    if needs.qaln:
        q_aln = "".join(query_str[i] if i != -1 else "-" for i in q_idx)
    if needs.taln:
        t_aln = "".join(subject_str[i] if i != -1 else "-" for i in s_idx)

    return n_ident, mismatch, gap_open, q_start, q_end, t_start, t_end, cigar, q_aln, t_aln


def _robinson_frequency_vector(alphabet) -> np.ndarray:
    """Robinson background frequencies as a vector aligned to ``alphabet`` order."""
    freq = np.array(
        [_ROBINSON_FREQUENCIES.get(sym, 0.0) for sym in alphabet.get_symbols()],
        dtype=float,
    )
    return freq / freq.sum()


def _estimate_lambda_k(
        gap_open: int,
        gap_extend: int,
        sample_size: int,
        sample_length: int,
        seed: int,
) -> Tuple[float, float]:
    """Estimate Karlin-Altschul ``(lambda, K)`` for BLOSUM62 at the given gaps
    by sampling random alignments via biotite.

    The estimate is biased relative to NCBI's published constants, so the
    derived p-/E-values are an approximate, relative-only signal (see module
    docstring). Seeded for reproducibility.
    """
    import biotite.sequence as seq
    import biotite.sequence.align as align
    alphabet = seq.ProteinSequence.alphabet
    np.random.seed(seed)
    estimator = align.EValueEstimator.from_samples(
        alphabet,
        align.SubstitutionMatrix.std_protein_matrix(),
        (-abs(gap_open), -abs(gap_extend)),
        _robinson_frequency_vector(alphabet),
        sample_length=sample_length,
        sample_size=sample_size,
    )
    logger.info(
        f"Estimated lambda={estimator.lam:.4f}, K={estimator.k:.5f} by sampling "
        f"{sample_size} alignments (approximate significance)."
    )
    return float(estimator.lam), float(estimator.k)


def _evalue_from_log(log_e: float) -> float:
    """E-value from biotite's log10(E), guarding against overflow to inf."""
    with np.errstate(over='ignore'):
        return float(10.0 ** log_e)


def _pvalue_from_evalue(evalue: float) -> float:
    """p = 1 - exp(-E), numerically stable for small and large E."""
    return float(-np.expm1(-evalue))


def _bit_score(score: int, lam: float, k: float) -> float:
    """Normalized (bit) score S' = (lambda*S - ln K) / ln 2.

    Uses the same lambda/K as the p-/E-values, so it inherits the active mode's
    scale: calibrated under 'default', relative-only under 'sampled'.
    Never BLAST-comparable (see module header). Because lambda/K are fixed for a
    whole run, the bit score is a strictly increasing function of the raw score,
    so ranking by bit score matches ranking by raw Smith-Waterman score in either
    mode.
    """
    return (lam * score - math.log(k)) / math.log(2.0)


def _worker_init(gap_open: int, gap_extend: int, lam: Optional[float], k: Optional[float],
                 db_residues: Optional[int], alp_params=None, needs: Optional["_FieldNeeds"] = None,
                 comp_bias_scale: Optional[float] = None):
    global _MATRIX, _GAP, _ESTIMATOR, _DB_RESIDUES, _LAMBDA, _K, _ALP, _OUTPUT_NEEDS, _COMP_BIAS_SCALE
    import biotite.sequence.align as align
    _OUTPUT_NEEDS = needs if needs is not None else _FieldNeeds(False, False, False, False, False)
    _MATRIX = align.SubstitutionMatrix.std_protein_matrix()  # BLOSUM62
    # Gap penalties are passed through unmodified: biotite charges a length-L gap
    # as open + (L-1)*extend, which is exactly MMseqs2's convention, so 11/1 here
    # is the scoring system its hardcoded lambda/K were shipped with.
    _GAP = (-abs(gap_open), -abs(gap_extend))
    _ALP = alp_params
    if alp_params is not None:
        _ESTIMATOR = None
        _LAMBDA, _K = alp_params.lam, alp_params.k
    else:
        _ESTIMATOR = align.EValueEstimator(lam, k) if lam is not None else None
        _LAMBDA, _K = lam, k
    _DB_RESIDUES = db_residues
    _COMP_BIAS_SCALE = comp_bias_scale


def _align(query_protein, query_str: str, query_len: int,
           subject_protein, subject_str: str, subject_len: int,
           scoring=None) -> AlignmentMetrics:
    """Local alignment of one pair. ``scoring`` is ``(positional_query, matrix)``
    from :func:`_positional_query` when the compositional bias correction is on."""
    import biotite.sequence.align as align
    if scoring is None:
        aln = align.align_optimal(
            query_protein, subject_protein, _MATRIX,
            gap_penalty=_GAP, local=True, max_number=1,
        )[0]
    else:
        positional, matrix = scoring
        aln = align.align_optimal(
            positional, subject_protein, matrix,
            gap_penalty=_GAP, local=True, max_number=1,
        )[0]
        # Identity is measured on residues, not positions: re-attach the real query.
        aln = align.Alignment([query_protein, subject_protein], aln.trace, aln.score)
    score = int(aln.score)
    # Compute the bit score here so both return paths carry it. None when
    # significance is off; otherwise on whatever scale the active mode implies.
    bit_score = _bit_score(score, _LAMBDA, _K) if _LAMBDA is not None else None
    needs = _OUTPUT_NEEDS
    q_seq = query_str if needs.qseq else None
    t_seq = subject_str if needs.tseq else None
    trace = aln.trace
    aln_len = int(trace.shape[0])
    if aln_len == 0:
        return AlignmentMetrics(
            0.0, 0.0, 0.0, 0.0, 0, score, bit_score=bit_score,
            query_len=query_len, subject_len=subject_len,
            cigar=("" if needs.cigar else None),
            q_aln=("" if needs.qaln else None),
            t_aln=("" if needs.taln else None),
            q_seq=q_seq, t_seq=t_seq,
        )
    identity_aln = float(align.get_sequence_identity(aln, mode="all"))
    identity_shorter = float(align.get_sequence_identity(aln, mode="shortest"))
    q_aligned = int((trace[:, 0] != -1).sum())
    s_aligned = int((trace[:, 1] != -1).sum())
    (n_ident, mismatch, gap_open, q_start, q_end, t_start, t_end,
     cigar, q_aln, t_aln) = _derive_alignment_extras(trace, query_str, subject_str, needs)

    pvalue = evalue = None
    if _ALP is not None:
        # Calibrated: ALP constants + ALP's finite-size-corrected area.
        # Pairwise p-value: search space = the two sequences (n = subject length).
        # MMseqs2 itself emits no p-value; this is the ALP p-value of the
        # pairwise E, kept for continuity with the 'sampled' mode.
        pvalue = karlin_altschul.pvalue_from_evalue(
            karlin_altschul.evalue(score, query_len, subject_len, _ALP)
        )
        # Database E-value: this is the MMseqs2-comparable quantity — search
        # space = the whole target DB as one concatenated sequence, exactly as
        # MMseqs2 does (no per-sequence term, no multiply by sequence count).
        if _DB_RESIDUES:
            evalue = karlin_altschul.evalue(score, query_len, _DB_RESIDUES, _ALP)
    elif _ESTIMATOR is not None:
        # Sampled/relative-only: biotite's raw K*m*n*exp(-lambda*S), no edge correction.
        pvalue = _pvalue_from_evalue(
            _evalue_from_log(_ESTIMATOR.log_evalue(score, query_len, subject_len))
        )
        if _DB_RESIDUES:
            evalue = _evalue_from_log(
                _ESTIMATOR.log_evalue(score, query_len, _DB_RESIDUES)
            )

    return AlignmentMetrics(
        identity_aln=identity_aln,
        identity_shorter=identity_shorter,
        query_coverage=q_aligned / query_len if query_len else 0.0,
        subject_coverage=s_aligned / subject_len if subject_len else 0.0,
        aln_len=aln_len,
        score=score,
        bit_score=bit_score,
        pvalue=pvalue,
        evalue=evalue,
        query_len=query_len,
        subject_len=subject_len,
        n_ident=n_ident,
        mismatch=mismatch,
        gap_open=gap_open,
        q_start=q_start,
        q_end=q_end,
        t_start=t_start,
        t_end=t_end,
        cigar=cigar,
        q_aln=q_aln,
        t_aln=t_aln,
        q_seq=q_seq,
        t_seq=t_seq,
    )


def _align_query(task):
    """Align one query against a chunk of its candidate subjects. Runs in a worker.

    The chunk may be all of a query's candidates or a slice of them (see
    :func:`_chunk_candidate_tasks`); either way the query protein is built once
    for the chunk and a partial hit list is returned, keyed by ``query_id`` so
    the caller can regroup a query that was split across several chunks.
    """
    query_id, query_seq, candidates = task
    query_protein, query_clean, query_len = _to_protein(query_seq)
    if query_protein is None:
        return query_id, []
    scoring = None
    if _COMP_BIAS_SCALE is not None:
        # Built from the raw sequence so MMseqs2's letter folding (e.g. J->L)
        # applies; sanitize_sequence maps characters 1:1, so positions line up.
        scoring = _positional_query(query_protein, local_composition_bias(query_seq, _COMP_BIAS_SCALE))
    hits: List[Hit] = []
    for subject_id, subject_seq, emb_score in candidates:
        subject_protein, subject_clean, subject_len = _to_protein(subject_seq)
        if subject_protein is None:
            continue
        metrics = _align(
            query_protein, query_clean, query_len,
            subject_protein, subject_clean, subject_len,
            scoring=scoring,
        )
        hits.append(Hit(subject_id=subject_id, emb_score=emb_score, metrics=metrics))
    return query_id, hits


def _chunk_candidate_tasks(tasks, worker_budget: int, tasks_per_worker: int = 4):
    """Split per-query tasks into per-(query, candidate-chunk) tasks for the pool.

    Parallelizing over whole queries pins a single query with many candidates to
    one worker (every other core idle); splitting a query's candidate list into
    chunks lets the pool spread those alignments across all workers. Chunk size
    is chosen so there are roughly ``tasks_per_worker * worker_budget`` chunks in
    total — enough to keep every worker fed and to smooth out the load imbalance
    from variable-length (quadratic-cost) alignments — while still batching
    several candidates per chunk to amortize the per-chunk query rebuild and the
    task's pickling overhead.

    Queries with no candidates are dropped here (they emit no work); the caller
    re-seeds them so they still report an empty hit list.
    """
    total_pairs = sum(len(candidates) for _, _, candidates in tasks)
    if total_pairs == 0:
        return list(tasks)
    target_chunks = tasks_per_worker * max(1, worker_budget)
    chunk_size = max(1, math.ceil(total_pairs / target_chunks))
    chunked = []
    for query_id, query_seq, candidates in tasks:
        for start in range(0, len(candidates), chunk_size):
            chunked.append((query_id, query_seq, candidates[start:start + chunk_size]))
    return chunked


def align_candidates(
        query_sequences: Dict[str, str],
        prefilter_results: Dict[str, Tuple[List[str], List[float]]],
        fetch_subject_sequences,
        min_seq_identity: float = 0.3,
        min_coverage: float = 0.8,
        max_evalue: Optional[float] = 1e-3,
        gap_open: int = 11,
        gap_extend: int = 1,
        comp_bias_corr: bool = False,
        comp_bias_corr_scale: float = 1.0,
        num_workers: Optional[int] = None,
        subject_db_size: Optional[int] = None,
        compute_significance: bool = True,
        significance_mode: SignificanceMode = SignificanceMode.default,
        # 'sampled' mode only. NOTE: an earlier comment here claimed lambda/K had
        # converged to within ~1% of the 1000x1000 values by 500x500 — that is
        # measurably wrong (lambda 0.279 vs 0.222, K 0.062 vs 0.008; see the
        # table in the module header). Raising these does NOT buy a steadier
        # estimate, so treat 'sampled' absolute values as unreliable and prefer
        # significance_mode='default' wherever the gap penalties allow it.
        significance_sample_size: int = 500,
        significance_sample_length: int = 500,
        significance_seed: int = 0,
        output_fields: Optional[List[str]] = None,
) -> Dict[str, List[Hit]]:
    """Stage 2: pairwise-align each prefilter candidate and re-rank by identity.

    Args:
        query_sequences: ``{query_id: sequence}`` for every query.
        prefilter_results: Stage-1 output ``{query_id: ([subject_ids], [emb_scores])}``.
        fetch_subject_sequences: callable ``ids -> {id: sequence}`` (e.g.
            :meth:`SequenceStore.fetch`), used once to gather all candidate
            subject sequences across queries.
        min_seq_identity: drop hits whose ``identity_aln`` is below this.
        min_coverage: drop hits whose query *and* subject coverage are below this.
        max_evalue: drop hits whose database E-value is above this. ``None``
            or a non-finite value disables the filter. Because the E-value is a
            significance quantity, setting it forces ``compute_significance=True``
            and requires ``subject_db_size`` — so a caller with no search space
            (or one deliberately skipping the significance pass) must pass
            ``max_evalue=None``. These three thresholds default to the same
            values as the ``fm-search query`` options that expose them.
        gap_open / gap_extend: positive BLOSUM62 gap penalties (negated internally).
        comp_bias_corr: correct the Smith-Waterman scores for locally biased
            amino-acid composition, as MMseqs2's ``--comp-bias-corr`` does (see
            :func:`local_composition_bias`). Off by default (plain BLOSUM62);
            ``True`` makes raw scores, bit scores and E-values match a default
            ``mmseqs search`` (``--comp-bias-corr 1``).
        comp_bias_corr_scale: multiplier applied to the correction before
            rounding (``--comp-bias-corr-scale``, 0-1); ``0`` turns it off.
        num_workers: process-pool size. ``None`` (default) uses **all** CPUs:
            it first widens the process CPU-affinity mask (undoing any scheduler
            ``--cpu-bind`` pinning, kernel-clamped to the cgroup allocation) so a
            binding launcher like ``srun`` can't cap it below the node's cores.
            ``0`` or ``1`` runs serially in-process. An explicit ``N > 1`` uses
            N workers (also widening affinity so they can spread). Capped at the
            number of queries.
        subject_db_size: total residue count of the subject database, used as the
            search space for the database E-value. If ``None``, no E-value is
            reported (the pairwise p-value is still computed).
        compute_significance: when ``True``, annotate each hit with a bit score,
            a pairwise p-value and (if ``subject_db_size`` is given) a database
            E-value.
        significance_mode: a :class:`SignificanceMode` (a plain string with the
            same value is accepted and coerced).
            ``'default'`` uses calibrated Karlin-Altschul
            constants with a finite-size-corrected search space, giving exact,
            reproducible values; it supports only ``gap_open=11, gap_extend=1``
            and raises otherwise. ``'sampled'`` restores the previous behavior —
            lambda/K estimated by sampling, whose absolute values are unreliable
            (they do not converge with sample size) though the ranking is sound;
            use it for gap penalties ``'default'`` cannot serve. Neither mode is
            BLAST-comparable; see the module header.
        significance_sample_size / significance_sample_length / significance_seed: controls for the
            lambda/K sampling pass (``'sampled'`` mode only).
        output_fields: the ``--format-output`` columns that will be rendered
            (see :data:`~foldmatch.search.output.OUTPUT_FIELD_DESCRIPTIONS`);
            defaults to :data:`~foldmatch.search.output.DEFAULT_OUTPUT_FIELDS`.
            Only the heavy string columns actually
            requested (``cigar``/``qaln``/``taln``/``qseq``/``tseq``) are built
            and carried on each :class:`AlignmentMetrics`, so a default-format
            search stays lean. Every scalar column is always available.

    Returns:
        ``{query_id: [Hit, ...]}`` sorted by ``bit_score`` desc (equivalently
        raw alignment score, since lambda/K are shared across the run), then
        embedding score desc; with ``compute_significance=False`` the raw
        alignment score is used directly. Queries with no surviving hit map to
        ``[]``.
    """
    start_time = time.perf_counter()
    # An E-value threshold only means something once the significance pass has
    # run against a known search space, so turn it on rather than silently
    # dropping every hit (evalue stays None when it is off). A non-finite
    # threshold ('inf') is the documented way to switch the filter off.
    if max_evalue is not None and not math.isfinite(max_evalue):
        max_evalue = None
    if max_evalue is not None:
        if subject_db_size is None:
            raise ValueError(
                "max_evalue filtering requires subject_db_size (the database "
                "search space); pass it or set max_evalue=None."
            )
        compute_significance = True
    if not 0.0 <= comp_bias_corr_scale <= 1.0:
        raise ValueError(
            f"comp_bias_corr_scale must be between 0 and 1, got {comp_bias_corr_scale}."
        )
    comp_bias_scale = float(comp_bias_corr_scale) if comp_bias_corr and comp_bias_corr_scale > 0 else None
    # Resolve the requested output columns to the set of heavy strings the
    # workers must build (default format needs none of them).
    output_fields = list(output_fields) if output_fields else list(DEFAULT_OUTPUT_FIELDS)
    needs = _needs_from_fields(output_fields)
    # Validate up front (and coerce a plain string), so a typo fails immediately
    # rather than only when significance happens to be computed.
    try:
        significance_mode = SignificanceMode(significance_mode)
    except ValueError:
        raise ValueError(
            f"Unknown significance_mode {significance_mode!r}; expected one of: "
            f"{', '.join(m.value for m in SignificanceMode)}."
        ) from None

    # Gather every candidate subject sequence in a single random-access read.
    all_subject_ids = {
        sid
        for subject_ids, _ in prefilter_results.values()
        for sid in subject_ids
    }
    subject_sequences = fetch_subject_sequences(all_subject_ids)

    tasks = []
    for query_id, (subject_ids, emb_scores) in prefilter_results.items():
        query_seq = query_sequences.get(query_id)
        if query_seq is None:
            logger.warning(f"Query '{query_id}' has no sequence; skipping alignment")
            continue
        candidates = [
            (sid, subject_sequences[sid], float(score))
            for sid, score in zip(subject_ids, emb_scores)
            if sid in subject_sequences
        ]
        missing = len(subject_ids) - len(candidates)
        if missing:
            logger.warning(f"{missing} candidate(s) for query '{query_id}' missing from sequence store; skipped")
        tasks.append((query_id, query_seq, candidates))

    # Resolve the significance parameters once and share them with the workers.
    # 'default' is a table lookup (no sampling pass); 'sampled' runs the estimator.
    lam = k = alp_params = None
    if compute_significance and tasks:
        if significance_mode is SignificanceMode.default:
            try:
                alp_params = karlin_altschul.params_for(gap_open, gap_extend)
            except ValueError as exc:
                # Without an E-value threshold, these gap penalties would have
                # been fine whenever no significance column was requested — say
                # how to get that behaviour back.
                if max_evalue is None:
                    raise
                raise ValueError(
                    f"{exc} E-value filtering (max_evalue={max_evalue:g}) requires "
                    f"significance; disable it to use these gap penalties."
                ) from None
            logger.info(
                f"Using calibrated significance (BLOSUM62 {gap_open}/{gap_extend}: "
                f"lambda={alp_params.lam:.8f}, K={alp_params.k:.9f})"
            )
        elif significance_mode is SignificanceMode.sampled:
            lam, k = _estimate_lambda_k(
                gap_open, gap_extend, significance_sample_size, significance_sample_length, significance_seed
            )
        else:  # unreachable; guards against a new member added without a branch
            raise ValueError(f"Unhandled significance_mode {significance_mode!r}")

    # Unless explicitly serial, widen the affinity mask before sizing the pool
    # so a scheduler-pinned rank (e.g. srun-bound) still uses the whole node and
    # the forked workers inherit the widened mask. Kernel-clamped to the cgroup,
    # so this can't grab cores outside a real allocation.
    serial = num_workers is not None and num_workers <= 1
    if not serial and tasks:
        _expand_cpu_affinity()

    if comp_bias_scale is not None and tasks:
        logger.info(f"Applying compositional bias correction (scale {comp_bias_scale:g})")
    init_args = (gap_open, gap_extend, lam, k, subject_db_size, alp_params, needs, comp_bias_scale)
    if not serial and tasks:
        # Parallelize over (query, subject) *pairs* rather than whole queries by
        # chunking each query's candidate list, so even a single query with many
        # candidates uses the whole node instead of one core. Size the pool by
        # the number of chunks (capped, as before, at the worker budget).
        worker_budget = _available_cpus() if num_workers is None else num_workers
        pair_tasks = _chunk_candidate_tasks(tasks, worker_budget)
        n_workers = _resolve_num_workers(num_workers, len(pair_tasks))
    else:
        pair_tasks = tasks
        n_workers = 1

    if n_workers > 1:
        total_pairs = sum(len(candidates) for _, _, candidates in tasks)
        logger.info(
            f"Aligning {total_pairs} (query, subject) pairs from {len(tasks)} "
            f"queries in {len(pair_tasks)} chunks across {n_workers} CPU workers"
        )
        import multiprocessing as mp
        with mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=init_args,
        ) as pool:
            # chunksize=1: each candidate-chunk is its own dispatch unit (we've
            # already sized the chunks) so variable-length alignments stay
            # balanced; pool.map preserves task order.
            raw = pool.map(_align_query, pair_tasks, chunksize=1)
    else:
        _worker_init(*init_args)
        raw = [_align_query(task) for task in pair_tasks]

    # A query may now span several chunks, so accumulate its hits before
    # filtering/sorting. Pre-seed every query (in prefilter order) so those with
    # no candidates still map to [] and the result ordering is preserved.
    per_query: Dict[str, List[Hit]] = {query_id: [] for query_id, _, _ in tasks}
    for query_id, hits in raw:
        per_query[query_id].extend(hits)

    results: Dict[str, List[Hit]] = {}
    for query_id, hits in per_query.items():
        kept = [
            hit for hit in hits
            # A candidate with no positive-scoring local alignment yields an empty
            # trace (aln_len == 0) and thus no valid coordinates; it is not a hit,
            # so drop it outright (MMseqs2 never emits an alnlen=0 / position-0
            # row). This matters when the thresholds below are relaxed to 0.
            if hit.metrics.aln_len > 0
            and hit.metrics.identity_aln >= min_seq_identity
            and min(hit.metrics.query_coverage, hit.metrics.subject_coverage) >= min_coverage
            # evalue is guaranteed non-None here: a max_evalue forces the
            # significance pass and a subject_db_size above.
            and (max_evalue is None or hit.metrics.evalue <= max_evalue)
        ]
        # Rank by bit score (desc), embedding score breaking ties. When
        # significance is disabled there is no bit score, so fall back to the raw
        # alignment score — the same ordering, since the bit score is monotonic
        # in the raw score under the run-wide shared lambda/K.
        kept.sort(
            key=lambda h: (
                h.metrics.bit_score if h.metrics.bit_score is not None else h.metrics.score,
                h.emb_score,
            ),
            reverse=True,
        )
        results[query_id] = kept
    logger.info(f"Pairwise alignments completed in {time.perf_counter() - start_time:.2f}s")
    return results
