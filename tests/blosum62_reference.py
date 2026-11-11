"""Test oracle for the BLOSUM62 Stage 2: the per-pair biotite path it replaced. Never used in production.

Up to FoldMatch 0.10.0's native kernel (P9.S3), Stage 2 aligned each (query, candidate) pair with biotite 1.7.1
``align_optimal(..., local=True, max_number=1)`` and derived every output column from biotite's trace. The functions
below are that code (``alignment._to_protein``, ``_positional_query``, ``_derive_alignment_extras`` and ``_align`` of
release 562c718), with the per-process globals turned into arguments. The kernel path
(``alignment._align_query``) must give the same :class:`~foldmatch.search.alignment.AlignmentMetrics`, field for
field.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from foldmatch.search import alignment, karlin_altschul


def blosum62():
    import biotite.sequence.align as align
    return align.SubstitutionMatrix.std_protein_matrix()


def to_protein(sequence: str):
    """``(ProteinSequence, sanitized string, length)``; ``(None, "", 0)`` for an empty sequence."""
    import biotite.sequence as seq
    clean = alignment.sanitize_sequence(sequence)
    if not clean:
        return None, "", 0
    return seq.ProteinSequence(clean), clean, len(clean)


def positional_query(query_protein, bias: np.ndarray, matrix=None):
    """The query as a sequence over its own positions with an ``L x |protein alphabet|`` substitution matrix: the
    BLOSUM62 row of each residue plus its bias (biotite has no per-position scoring hook)."""
    import biotite.sequence as seq
    import biotite.sequence.align as align
    matrix = matrix if matrix is not None else blosum62()
    n = len(query_protein)
    alphabet = seq.Alphabet(list(range(n)))
    positional = seq.GeneralSequence(alphabet, list(range(n)))
    scores = matrix.score_matrix()[query_protein.code] + np.asarray(bias)[:, None]
    return positional, align.SubstitutionMatrix(alphabet, matrix.get_alphabet2(), scores.astype(np.int32))


def biotite_alignment(query: str, subject: str, gap: Tuple[int, int] = (11, 1), bias: Optional[np.ndarray] = None):
    """biotite's local alignment of two sanitized strings with positive gap costs ``gap``; with ``bias`` the query is
    scored position-specifically and the real query re-attached (identity is measured on residues)."""
    import biotite.sequence.align as align
    q_protein, _, _ = to_protein(query)
    s_protein, _, _ = to_protein(subject)
    penalty = (-abs(gap[0]), -abs(gap[1]))
    matrix = blosum62()
    if bias is None:
        return align.align_optimal(q_protein, s_protein, matrix, gap_penalty=penalty, local=True, max_number=1)[0]
    positional, pmatrix = positional_query(q_protein, bias, matrix)
    aln = align.align_optimal(positional, s_protein, pmatrix, gap_penalty=penalty, local=True, max_number=1)[0]
    return align.Alignment([q_protein, s_protein], aln.trace, aln.score)


def _count_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    return int(mask[0]) + int(np.count_nonzero(mask[1:] & ~mask[:-1]))


def _run_length_encode(ops: str) -> str:
    if not ops:
        return ""
    out = []
    prev, count = ops[0], 1
    for ch in ops[1:]:
        if ch == prev:
            count += 1
        else:
            out.append(f"{count}{prev}")
            prev, count = ch, 1
    out.append(f"{count}{prev}")
    return "".join(out)


def _derive_alignment_extras(trace, query_str: str, subject_str: str, needs):
    q_idx = trace[:, 0]
    s_idx = trace[:, 1]
    q_gap = q_idx == -1
    s_gap = s_idx == -1
    both = ~q_gap & ~s_gap
    q_arr = np.frombuffer(query_str.encode("ascii"), dtype=np.uint8)
    s_arr = np.frombuffer(subject_str.encode("ascii"), dtype=np.uint8)
    q_res = q_idx[~q_gap]
    s_res = s_idx[~s_gap]
    q_start, q_end = int(q_res.min()) + 1, int(q_res.max()) + 1
    t_start, t_end = int(s_res.min()) + 1, int(s_res.max()) + 1
    matches = q_arr[q_idx[both]] == s_arr[s_idx[both]]
    n_ident = int(np.count_nonzero(matches))
    mismatch = int(both.sum()) - n_ident
    gap_open = _count_runs(q_gap) + _count_runs(s_gap)
    cigar = q_aln = t_aln = None
    if needs.cigar:
        ops = np.empty(trace.shape[0], dtype="U1")
        ops[both] = "M"
        ops[q_gap] = "D"
        ops[s_gap] = "I"
        cigar = _run_length_encode("".join(ops.tolist()))
    if needs.qaln:
        q_aln = "".join(query_str[i] if i != -1 else "-" for i in q_idx)
    if needs.taln:
        t_aln = "".join(subject_str[i] if i != -1 else "-" for i in s_idx)
    return n_ident, mismatch, gap_open, q_start, q_end, t_start, t_end, cigar, q_aln, t_aln


def reference_metrics(query: str, subject: str, *, gap: Tuple[int, int] = (11, 1), comp_bias_scale=None,
                      needs=None, lam=None, k=None, alp=None, db_residues=None) -> Optional[alignment.AlignmentMetrics]:
    """The old ``_align`` of one (raw) query and subject: None when either sanitizes to nothing (no hit). ``alp``
    (calibrated) or ``lam``/``k`` (sampled, biotite's EValueEstimator) give the significance columns."""
    import biotite.sequence.align as align
    needs = needs if needs is not None else alignment._FieldNeeds(False, False, False, False, False)
    q_protein, query_str, query_len = to_protein(query)
    s_protein, subject_str, subject_len = to_protein(subject)
    if q_protein is None or s_protein is None:
        return None
    bias = None if comp_bias_scale is None else alignment.local_composition_bias(query, comp_bias_scale)
    aln = biotite_alignment(query, subject, gap, bias)
    score = int(aln.score)
    lam_k = (alp.lam, alp.k) if alp is not None else (lam, k)
    bit_score = alignment._bit_score(score, *lam_k) if lam_k[0] is not None else None
    q_seq = query_str if needs.qseq else None
    t_seq = subject_str if needs.tseq else None
    trace = aln.trace
    aln_len = int(trace.shape[0])
    if aln_len == 0:
        return alignment.AlignmentMetrics(
            0.0, 0.0, 0.0, 0.0, 0, score, bit_score=bit_score, query_len=query_len, subject_len=subject_len,
            cigar=("" if needs.cigar else None), q_aln=("" if needs.qaln else None),
            t_aln=("" if needs.taln else None), q_seq=q_seq, t_seq=t_seq)
    q_aligned = int((trace[:, 0] != -1).sum())
    s_aligned = int((trace[:, 1] != -1).sum())
    (n_ident, mismatch, gap_open, q_start, q_end, t_start, t_end,
     cigar, q_aln, t_aln) = _derive_alignment_extras(trace, query_str, subject_str, needs)
    pvalue = evalue = None
    if alp is not None:
        pvalue = karlin_altschul.pvalue_from_evalue(karlin_altschul.evalue(score, query_len, subject_len, alp))
        if db_residues:
            evalue = karlin_altschul.evalue(score, query_len, db_residues, alp)
    elif lam is not None:
        est = align.EValueEstimator(lam, k)
        pvalue = alignment._pvalue_from_evalue(alignment._evalue_from_log(est.log_evalue(score, query_len,
                                                                                         subject_len)))
        if db_residues:
            evalue = alignment._evalue_from_log(est.log_evalue(score, query_len, db_residues))
    return alignment.AlignmentMetrics(
        identity_aln=n_ident / aln_len, identity_shorter=n_ident / min(query_len, subject_len),
        query_coverage=q_aligned / query_len, subject_coverage=s_aligned / subject_len, aln_len=aln_len,
        score=score, bit_score=bit_score, pvalue=pvalue, evalue=evalue, query_len=query_len,
        subject_len=subject_len, n_ident=n_ident, mismatch=mismatch, gap_open=gap_open, q_start=q_start,
        q_end=q_end, t_start=t_start, t_end=t_end, cigar=cigar, q_aln=q_aln, t_aln=t_aln, q_seq=q_seq, t_seq=t_seq)
