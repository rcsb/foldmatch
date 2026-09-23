"""The stage-2 scoring system as one object: matrix, gaps, scale and statistics together.

A scoring system is not a handful of independent numbers.  The 3Di weight, the gap
penalties, the integer scale the matrix is rounded to, the representation the target's
structural state is stored in, and the Karlin-Altschul lambda/K are only meaningful *as a
set*: change any one and the others stop describing the same thing.  None of those
mismatches raises - they all produce plausible output - so this module makes the set a
single frozen, serializable value and derives the deployed numbers from it.

The two mismatches that motivated it, both measured rather than imagined:

* **Scale.** lambda is fitted on the *descaled* score (``alignment.score / scale``) because
  that is what the calibration harness returns, while biotite's aligner keeps the scaled
  integer.  Applying a lambda of 0.0210 to a scale-100 score is wrong by ``exp(0.0210*99*S)``
  - about ``e**600`` at S = 300.  Here ``lam_deployed`` divides by the scale, and
  ``gaps_deployed`` multiplies by it, and those are the only ways the numbers can be read.
* **Per-query lambda.** An earlier design computed lambda per query from its own profile, as
  PSI-BLAST does.  That was tested and retired: the analytic per-query lambda (~0.177)
  describes an i.i.d. null that never occurs, and deploying it puts 81% of genuinely
  unrelated pairs at E <= 0.01, against 1.4% for the fitted constant.  Real proteins from
  different folds share secondary structure, 3Di encodes exactly that, and the fitted lambda
  (~0.021) is the one that describes the null a search actually faces.  **lambda is a
  constant per (matrix, gaps, target representation, reference null)**, stamped at
  calibration time - not a per-query quantity.

``ScoringSystem`` itself is pure metadata: hashable, JSON round-trippable, and safe to store
in a database manifest.  The numpy objects an aligner needs are built by :func:`prepare`,
which returns a :class:`PreparedScoring`.

Nothing here imports torch, reads a checkpoint, or touches the network.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

#: Foldseek's 3Di states, in the order biotite's ``std_3di_matrix`` uses.
THREE_DI = "acdefghiklmnpqrstvwy"
N_STATES = len(THREE_DI)

#: From the header of biotite's ``matrix_data/3Di.mat``: the 3Di matrix is in bit/2 units
#: and this lambda converts a score to natural-log odds.  It is a property of that matrix,
#: unrelated to the Karlin-Altschul lambda of an assembled scoring system.
LAMBDA_3DI = 0.351568

#: Query representations.  ``states`` scores a single 3Di letter per residue (the hard
#: path, and what a structure query with known coordinates can use); ``profile`` scores a
#: per-residue distribution over the 20 states.
QUERY_REPRS = ("states", "profile")

#: Target representations.  ``amino`` is no structural term at all (plain BLOSUM62);
#: ``exact`` is a 3Di letter per residue read from coordinates; ``coded`` is one byte per
#: residue - a predicted state plus a confidence bucket - interpreted through a kernel.
TARGET_REPRS = ("amino", "exact", "coded")

PROFILE_MODES = ("argmax", "expected", "logodds")

_CACHE: Dict[tuple, object] = {}


def _base_scores() -> Tuple[np.ndarray, np.ndarray]:
    """BLOSUM62 over biotite's protein alphabet (24 x 24) and the 3Di matrix (20 x 20).

    The protein alphabet is biotite's own ``ACDEFGHIKLMNPQRSTVWYBZX*``, i.e. the one
    ``sanitize_sequence`` coerces into and the one ``_MATRIX`` in :mod:`.alignment` already
    scores against, so a paired alphabet built on it leaves ordinary alignments unchanged
    and only adds symbols.  Both matrices are re-indexed through their alphabets rather
    than used positionally, so a future reordering upstream cannot silently shift scores.
    """
    if "base" not in _CACHE:
        from biotite.sequence.align import SubstitutionMatrix

        blosum = SubstitutionMatrix.std_protein_matrix()
        amino_alphabet = blosum.get_alphabet1()
        amino = blosum.score_matrix()[np.ix_(
            [amino_alphabet.encode(c) for c in amino_alphabet.get_symbols()],
            [amino_alphabet.encode(c) for c in amino_alphabet.get_symbols()],
        )]
        structural_matrix = SubstitutionMatrix.std_3di_matrix()
        structural_alphabet = structural_matrix.get_alphabet1()
        structural = structural_matrix.score_matrix()[np.ix_(
            [structural_alphabet.encode(c) for c in THREE_DI],
            [structural_alphabet.encode(c) for c in THREE_DI],
        )]
        _CACHE["base"] = (amino, structural)
    return _CACHE["base"]  # type: ignore[return-value]


def amino_alphabet_size() -> int:
    """Number of symbols in the amino-acid alphabet the paired alphabet is built on."""
    return _base_scores()[0].shape[0]


def pair_alphabet(n_codes: int):
    """Alphabet of (amino acid, structural code) symbols, ``n_codes`` structural codes.

    ``n_codes = 1`` is the degenerate amino-acid-only case, so the same code path serves a
    plain BLOSUM62 system without special-casing.
    """
    from biotite.sequence import Alphabet

    key = ("pair_alphabet", n_codes)
    if key not in _CACHE:
        _CACHE[key] = Alphabet(list(range(amino_alphabet_size() * n_codes)))
    return _CACHE[key]


def pair_codes(amino_codes: np.ndarray, structural_codes: Optional[np.ndarray],
               n_codes: int) -> np.ndarray:
    """Encode residues as single symbols of the combined alphabet.

    ``amino_codes`` are biotite ``ProteinSequence.code`` values of the *sanitized*
    sequence - the sequence that is actually aligned - so the encoding cannot drift from
    what the aligner sees.  ``structural_codes`` is the 3Di state index for an ``exact``
    target or the stored byte for a ``coded`` one, and must be None when ``n_codes == 1``.

    Both halves are range-checked: an out-of-range symbol does not raise inside biotite,
    it reads past the end of the score matrix and returns a plausible number.
    """
    amino_codes = np.asarray(amino_codes)
    if amino_codes.size and (amino_codes.min() < 0
                             or amino_codes.max() >= amino_alphabet_size()):
        raise ValueError(f"amino codes outside [0, {amino_alphabet_size()})")
    if n_codes == 1:
        if structural_codes is not None:
            raise ValueError("amino-acid-only alphabet takes no structural codes")
        return amino_codes.astype(np.int64)
    if structural_codes is None:
        raise ValueError(f"n_codes={n_codes} requires structural codes")
    structural_codes = np.asarray(structural_codes)
    if len(structural_codes) != len(amino_codes):
        raise ValueError(
            f"length mismatch: {len(amino_codes)} residues, {len(structural_codes)} codes")
    if structural_codes.size and (structural_codes.min() < 0
                                  or structural_codes.max() >= n_codes):
        raise ValueError(f"structural codes outside [0, {n_codes})")
    return amino_codes.astype(np.int64) * n_codes + structural_codes.astype(np.int64)


def split_pair_codes(codes: np.ndarray, n_codes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pair_codes`: ``(amino_codes, structural_codes)``.

    A database that stores only the paired byte still has to rebuild the plain amino-acid
    sequence, because sequence identity is computed on residues, not on paired symbols.
    """
    codes = np.asarray(codes)
    if n_codes == 1:
        return codes.astype(np.int64), np.zeros(len(codes), dtype=np.int64)
    return codes.astype(np.int64) // n_codes, codes.astype(np.int64) % n_codes


def three_di_codes(three_di: str) -> np.ndarray:
    """State index of each 3Di letter; unknown letters fall back to 'd', as Foldseek writes."""
    fallback = THREE_DI.index("d")
    return np.array([THREE_DI.index(c) if c in THREE_DI else fallback for c in three_di])


def profile_term(profile: np.ndarray, mode: str,
                 kernel: Optional[np.ndarray] = None) -> np.ndarray:
    """(L, C) structural score of each query position against each target code.

    ``kernel`` is (C, 20): the distribution over true states each target code stands for.
    Without one, C = 20 exact states - a target whose 3Di is known.  With one, query
    uncertainty and target uncertainty are mixed in a single score.

    ``argmax``   the most probable state's matrix row (identical to the hard path).
    ``expected`` ``sum_k p_k M[k, s]``, which penalises uncertain positions.
    ``logodds``  ``(1/lambda) ln sum_k p_k exp(lambda M[k, s])``, the log-odds score of the
                 mixture, ~0 at the matrix background.  This is the Foldseek/MMseqs2-style
                 choice and the one that was calibrated.

    Both soft modes reduce to the argmax row exactly for a one-hot profile.
    """
    if mode not in PROFILE_MODES:
        raise ValueError(f"unknown profile mode {mode!r}")
    _, structural = _base_scores()
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[1] != N_STATES:
        raise ValueError(f"profile must be (L, {N_STATES}), got {profile.shape}")
    p = profile / profile.sum(axis=1, keepdims=True)
    # scores[k, c]: true state k against target code c.  No kernel -> the matrix itself.
    scores = structural if kernel is None else structural @ np.asarray(kernel).T
    top = scores[p.argmax(axis=1)]
    if mode == "argmax":
        return top.astype(np.float64)
    if mode == "expected":
        return p @ scores
    # Taken relative to the argmax row so that log(1) = 0 keeps a one-hot profile exact and
    # the exponents stay small.  With a kernel the mixture runs over (state, code) at once,
    # which is why the kernel multiplies exp(lambda M) rather than M.
    if kernel is None:
        relative = np.exp(LAMBDA_3DI * (structural[None, :, :] - top[:, None, :]))
        return top + np.log(np.einsum("lk,lks->ls", p, relative)) / LAMBDA_3DI
    mixed = np.exp(LAMBDA_3DI * structural) @ np.asarray(kernel).T
    return np.log(p @ mixed) / LAMBDA_3DI


@dataclass(frozen=True)
class ScoringSystem:
    """Everything that defines one stage-2 scoring system, and nothing that doesn't.

    Gaps and lambda are stored in *unscaled* units - the units they were tuned and fitted
    in - and converted for deployment by :attr:`gaps_deployed` and :attr:`lam_deployed`.
    Read them no other way.
    """

    name: str
    weight_amino: float = 1.4
    weight_3di: float = 2.1
    #: Float scores are multiplied by this and rounded before biotite sees them.
    score_scale: int = 100
    #: Magnitudes, in unscaled score units; both are charged as biotite/MMseqs2 charge them
    #: (a length-L gap costs ``open + (L-1)*extend``).
    gap_open: float = 30.0
    gap_extend: float = 3.0

    query_repr: str = "profile"
    profile_mode: str = "logodds"
    #: Temperature the query profile is softened with before scoring, as
    #: ``softmax(log p / T)``.  Applied by :meth:`PreparedScoring.soften`, which
    #: :meth:`PreparedScoring.query_matrix` calls - callers pass the model's raw
    #: distribution and must NOT pre-soften, or the temperature is applied twice.
    profile_temperature: float = 1.0
    #: Drop the structural term where the profile's max probability is below this.
    min_confidence: Optional[float] = None

    target_repr: str = "exact"
    #: Structural codes in the target alphabet: 1 (amino only), 20 (exact), or the kernel's
    #: row count (coded).
    n_codes: int = N_STATES
    #: sha256 of the kernel's contiguous float64 bytes - the identity of the data file, so a
    #: database built under one kernel cannot be searched under another unnoticed.
    kernel_sha256: Optional[str] = None

    #: Where this system's significance comes from.  ``alp`` means the finite-size
    #: corrected ALP tables in :mod:`.karlin_altschul` (only BLOSUM62 at gaps 11/1);
    #: ``karlin_altschul`` means the plain ``K*m*n*exp(-lambda*S)`` form calibrated here.
    #: The aligner dispatches on this rather than on the system's name.
    statistics: str = "karlin_altschul"
    #: Karlin-Altschul constants, fitted on DESCALED scores against ``reference_set``.
    lam: Optional[float] = None
    k: Optional[float] = None
    reference_set: Optional[str] = None
    calibration: Optional[str] = None

    def __post_init__(self) -> None:
        if self.query_repr not in QUERY_REPRS:
            raise ValueError(f"query_repr must be one of {QUERY_REPRS}")
        if self.target_repr not in TARGET_REPRS:
            raise ValueError(f"target_repr must be one of {TARGET_REPRS}")
        if self.profile_mode not in PROFILE_MODES:
            raise ValueError(f"profile_mode must be one of {PROFILE_MODES}")
        if not isinstance(self.score_scale, int) or self.score_scale < 1:
            raise ValueError("score_scale must be a positive integer")
        if self.gap_open <= 0 or self.gap_extend <= 0:
            raise ValueError("gap penalties are magnitudes and must be positive")
        if self.gap_extend > self.gap_open:
            raise ValueError("gap_extend above gap_open makes long gaps cheaper than short")
        if self.target_repr == "amino":
            if self.n_codes != 1:
                raise ValueError("an amino-acid-only system has n_codes=1")
            if self.weight_3di != 0.0:
                raise ValueError("an amino-acid-only system must have weight_3di=0")
            if self.query_repr != "states":
                raise ValueError("an amino-acid-only system has no query profile")
        elif self.target_repr == "exact":
            if self.n_codes != N_STATES:
                raise ValueError(f"an exact-3Di system has n_codes={N_STATES}")
            if self.kernel_sha256 is not None:
                raise ValueError("an exact-3Di system takes no kernel")
        else:  # coded
            if self.n_codes <= N_STATES:
                raise ValueError("a coded system needs more codes than there are states")
            if self.kernel_sha256 is None:
                raise ValueError("a coded system must name the kernel it was built with")
        if self.query_repr == "states" and self.profile_mode != "argmax":
            raise ValueError("a states query is the hard path; profile_mode must be argmax")
        if self.query_repr == "states" and self.target_repr == "coded":
            # The hard path needs a full (symbol x symbol) matrix, which for 240 codes is
            # 5760x5760 int32 = 127 MiB per worker, and a code is a distribution rather
            # than a symbol, so there is nothing sensible to put in it.
            raise ValueError("a coded target has no hard-query path; use a profile query")
        if self.min_confidence is not None and not 0.0 < self.min_confidence <= 1.0:
            raise ValueError("min_confidence is a probability in (0, 1]")
        if self.profile_temperature <= 0:
            raise ValueError("profile_temperature must be positive")
        if self.statistics not in ("alp", "karlin_altschul"):
            raise ValueError("statistics must be 'alp' or 'karlin_altschul'")
        if self.statistics == "alp" and self.lam is not None:
            raise ValueError("an ALP system takes its constants from karlin_altschul.py")
        if (self.lam is None) != (self.k is None):
            raise ValueError("lambda and K are calibrated together or not at all")
        if self.lam is not None and (self.lam <= 0 or self.k <= 0):
            raise ValueError("lambda and K must be positive")
        if self.lam is not None and self.reference_set is None:
            raise ValueError("calibrated constants must record the null they were fitted on")

    # -- derived, and the only sanctioned way to read the numbers ------------------------

    @property
    def weights(self) -> Tuple[float, float]:
        return (self.weight_amino, self.weight_3di)

    @property
    def gaps_deployed(self) -> Tuple[int, int]:
        """Gap penalties on the integer scale the matrix is rounded to, negative."""
        return (-round(self.gap_open * self.score_scale),
                -round(self.gap_extend * self.score_scale))

    @property
    def lam_deployed(self) -> Optional[float]:
        """Lambda for the SCALED integer score biotite reports.

        ``lam`` is fitted on the descaled float, and ``S_int = scale * S_float``, so
        ``lambda * S`` is preserved only if lambda is divided by the same scale.
        """
        return None if self.lam is None else self.lam / self.score_scale

    @property
    def is_calibrated(self) -> bool:
        return self.lam is not None

    def _require_calibration(self) -> None:
        if not self.is_calibrated:
            raise ValueError(
                f"{self.name} has no calibrated lambda/K"
                + (" (its significance comes from karlin_altschul.py)"
                   if self.statistics == "alp" else ""))

    def evalue(self, score: float, query_length: int, search_space: int) -> float:
        """``K * m * n * exp(-lambda * S)`` for a SCALED integer score.

        ``search_space`` is the target length for a pairwise E-value, or the database
        residue total for a database E-value.  No finite-size correction: the constants
        were fitted to exactly this form.
        """
        self._require_calibration()
        return self.k * query_length * search_space * float(
            np.exp(-self.lam_deployed * score))

    def bit_score(self, score: float) -> float:
        """``(lambda * S - ln K) / ln 2`` for a SCALED integer score.

        Exists so that no call site has to remember that lambda needs the scale division
        while K does not - the asymmetry that makes hand-assembling these two numbers a
        standing hazard.
        """
        self._require_calibration()
        return (self.lam_deployed * score - math.log(self.k)) / math.log(2.0)

    def pvalue(self, score: float, query_length: int, search_space: int) -> float:
        """``1 - exp(-E)``, the probability of at least one chance hit this good."""
        return -math.expm1(-self.evalue(score, query_length, search_space))

    # -- identity and serialization -------------------------------------------------------

    def digest(self) -> str:
        """Stable content hash; changes if any field changes.  Goes in a database manifest."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ScoringSystem":
        data = json.loads(text)
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown scoring-system fields: {sorted(unknown)}")
        return cls(**data)

    def calibrated(self, lam: float, k: float, reference_set: str,
                   calibration: Optional[str] = None) -> "ScoringSystem":
        """This system with freshly fitted constants; everything else identical."""
        return replace(self, lam=lam, k=k, reference_set=reference_set,
                       calibration=calibration)


class PreparedScoring:
    """A :class:`ScoringSystem` with its numpy objects built, ready for an aligner.

    The query matrix is built once per query, never per candidate: at 240 target codes a
    1000-residue query matrix is 24 x 240 x 1000 int32, about 23 MB.
    """

    def __init__(self, system: ScoringSystem, kernel: Optional[np.ndarray] = None):
        if system.target_repr == "coded":
            if kernel is None:
                raise ValueError(f"{system.name} needs its kernel")
            kernel = np.asarray(kernel, dtype=np.float64)
            if kernel.shape != (system.n_codes, N_STATES):
                raise ValueError(
                    f"kernel must be ({system.n_codes}, {N_STATES}), got {kernel.shape}")
            digest = hashlib.sha256(np.ascontiguousarray(kernel).tobytes()).hexdigest()
            if digest != system.kernel_sha256:
                raise ValueError(
                    f"kernel does not match {system.name}: expected "
                    f"{system.kernel_sha256}, got {digest}")
            # A code is (predicted state, confidence bucket), and the bucket is a function
            # of the profile's sharpness - so a kernel's rows are only meaningful at the
            # temperature it was fitted at.  Between T=1.0 and T=1.080, 26.8% of residues
            # fall in a different bucket.  Whenever the kernel's provenance is known,
            # refuse a system that would index those rows at another temperature.
            if digest in BUNDLED_KERNELS:
                fitted = kernel_provenance(digest).get("profile_temperature")
                if fitted is not None and not np.isclose(
                        fitted, system.profile_temperature):
                    raise ValueError(
                        f"{system.name} scores profiles at T={system.profile_temperature} "
                        f"but its kernel was fitted at T={fitted}; the confidence buckets "
                        f"would not mean the same thing. Refit the kernel or restore the "
                        f"temperature.")
        elif kernel is not None:
            raise ValueError(f"{system.name} takes no kernel")
        self.system = system
        self.kernel = kernel
        self.alphabet = pair_alphabet(system.n_codes)

    # -- target side ----------------------------------------------------------------------

    def target_sequence(self, amino_codes: np.ndarray,
                        structural_codes: Optional[np.ndarray] = None):
        """A biotite sequence over the paired alphabet, for use as the aligner's subject."""
        from biotite.sequence import GeneralSequence

        codes = pair_codes(amino_codes, structural_codes, self.system.n_codes)
        sequence = GeneralSequence(self.alphabet, [])
        sequence.code = codes
        return sequence

    # -- query side -----------------------------------------------------------------------

    def hard_matrix(self):
        """Substitution matrix over paired symbols, for a query with no profile.

        Only defined for ``amino`` and ``exact`` targets: a coded target's symbol is a
        distribution over states, not a state, so there is no symbol-by-symbol matrix for
        it (and at 240 codes one would be 5760x5760 int32 = 127 MiB per worker).
        """
        from biotite.sequence.align import SubstitutionMatrix

        system = self.system
        if system.target_repr == "coded":
            raise ValueError(
                f"{system.name}: a coded target has no hard matrix; use query_matrix")
        key = ("hard", system.weights, system.score_scale, system.n_codes)
        if key not in _CACHE:
            amino, structural = _base_scores()
            if system.n_codes == 1:
                scores = system.weight_amino * amino
            else:
                scores = (system.weight_amino * amino[:, None, :, None]
                          + system.weight_3di * structural[None, :, None, :])
            if system.score_scale != 1:
                scores = system.score_scale * scores
            size = amino_alphabet_size() * system.n_codes
            scores = np.rint(scores).astype(np.int32).reshape(size, size)
            _CACHE[key] = SubstitutionMatrix(self.alphabet, self.alphabet, scores)
        return _CACHE[key]

    def soften(self, profile: np.ndarray) -> np.ndarray:
        """Apply the system's profile temperature: ``softmax(log p / T)``.

        Equivalently ``p ** (1/T)`` renormalised.  This is the transformation the fitted
        lambda/K were calibrated under, so it belongs to the system rather than to the
        caller - a profile softened at a different temperature is a different scoring
        system wearing the same constants.
        """
        profile = np.asarray(profile, dtype=np.float64)
        profile = profile / profile.sum(axis=1, keepdims=True)
        if self.system.profile_temperature == 1.0:
            return profile
        scaled = np.log(profile) / self.system.profile_temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        p = np.exp(scaled)
        return p / p.sum(axis=1, keepdims=True)

    @property
    def buckets(self) -> int:
        """Confidence buckets per predicted state, for a ``coded`` target."""
        if self.system.target_repr != "coded":
            raise ValueError(f"{self.system.name} does not use confidence buckets")
        if self.system.n_codes % N_STATES:
            raise ValueError(
                f"{self.system.n_codes} codes is not a whole number of buckets "
                f"per {N_STATES} states")
        return self.system.n_codes // N_STATES

    def encode_target(self, profile: np.ndarray) -> np.ndarray:
        """One byte per residue: predicted state x confidence bucket.

        This is the target-side counterpart of :meth:`query_matrix`, and the single
        definition of the encoding in this package - the kernel's rows are indexed by
        exactly these codes, so a second implementation that bucketed differently would
        point every residue at the wrong row while still producing valid-looking codes.

        The bucket spans the *usable* confidence range: a 20-state distribution cannot be
        less confident than 1/20, so the range is rescaled from [1/20, 1] to [0, 1] before
        bucketing. The system's temperature is applied first, because the kernel was
        fitted on softened confidences.
        """
        buckets = self.buckets
        profile = self.soften(profile)
        state = profile.argmax(axis=1)
        confidence = profile.max(axis=1)
        lowest = 1.0 / N_STATES
        scaled = (confidence - lowest) / (1.0 - lowest)
        bucket = np.clip((scaled * buckets).astype(int), 0, buckets - 1)
        return (state * buckets + bucket).astype(np.uint8 if self.system.n_codes <= 256
                                                 else np.int64)

    def query_matrix(self, profile: np.ndarray, amino_codes: np.ndarray,
                     bias: Optional[np.ndarray] = None):
        """``(positional_sequence, matrix)`` scoring query position i against every target symbol.

        biotite has no per-position scoring hook, so the query becomes a sequence over its
        own positions with an ``L x (|amino| * n_codes)`` matrix - the same mechanism
        :func:`.alignment._positional_query` already uses for compositional bias, widened
        from amino acids to (amino acid, structural code) pairs.

        ``profile`` is the model's raw per-residue distribution over the 20 3Di states;
        the system's temperature is applied here, so do not pre-soften it.  ``bias`` is
        the optional per-position compositional-bias offset, in UNSCALED score units (it
        is scaled with everything else), matching MMseqs2's own units.

        Build this once per query, not once per candidate: at 240 codes a 1000-residue
        matrix is 24 x 240 x 1000 int32, about 22 MiB.
        """
        from biotite.sequence import PurePositionalSequence
        from biotite.sequence.align import SubstitutionMatrix

        system = self.system
        if system.query_repr != "profile":
            raise ValueError(f"{system.name} has no query profile")
        amino, _ = _base_scores()
        amino_codes = np.asarray(amino_codes)
        profile = np.asarray(profile, dtype=np.float64)
        if profile.ndim != 2 or len(profile) != len(amino_codes):
            raise ValueError(
                f"profile has {len(profile)} rows for {len(amino_codes)} residues")
        profile = self.soften(profile)
        structural = profile_term(profile, system.profile_mode, self.kernel)
        if system.min_confidence is not None:
            confidence = profile.max(axis=1)
            structural = np.where(
                confidence[:, None] < system.min_confidence, 0.0, structural)
        # Same expression and operand order as hard_matrix, so a one-hot profile rounds
        # bit-identically to the hard path (np.rint breaks .5 ties to even).
        scores = (system.weight_amino * amino[amino_codes][:, :, None]
                  + system.weight_3di * structural[:, None, :])
        scores = scores.reshape(len(amino_codes), amino_alphabet_size() * system.n_codes)
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float64)
            if len(bias) != len(amino_codes):
                raise ValueError(
                    f"bias has {len(bias)} entries for {len(amino_codes)} residues")
            scores = scores + bias[:, None]
        if system.score_scale != 1:
            scores = system.score_scale * scores
        query = PurePositionalSequence(len(amino_codes))
        matrix = SubstitutionMatrix(query.get_alphabet(), self.alphabet,
                                    np.rint(scores).astype(np.int32))
        return query, matrix


#: Kernels shipped with the package, indexed by the sha256 of their contiguous float64
#: bytes.  Content-addressed rather than named, so a system's ``kernel_sha256`` is enough
#: to find its kernel and there is no second identifier to keep in sync.
BUNDLED_KERNELS: Dict[str, str] = {
    "8b71484501d18bfaef9bdd27105bceb3bce9935797cb9733183363d62c8ea423":
        "three_di_kernel_v1.npz",
}


def _kernel_path(filename: str):
    from importlib.resources import files

    return files("foldmatch.search").joinpath("data", filename)


def load_kernel(sha256: str) -> np.ndarray:
    """The bundled kernel with this content hash, verified on load.

    The kernel is part of the scoring system's definition, not a tunable: the calibrated
    lambda/K of a coded system were fitted with one specific kernel, so shipping the two
    apart would let a database be searched under statistics that describe something else.
    It is 20 KB, so it travels in the wheel.
    """
    if sha256 not in BUNDLED_KERNELS:
        raise KeyError(
            f"no bundled kernel with sha256 {sha256}; pass the array explicitly. "
            f"Bundled: {sorted(BUNDLED_KERNELS)}")
    with np.load(str(_kernel_path(BUNDLED_KERNELS[sha256]))) as data:
        kernel = np.ascontiguousarray(data["kernel"].astype("<f8"))
    found = hashlib.sha256(kernel.tobytes()).hexdigest()
    if found != sha256:
        raise ValueError(
            f"bundled kernel {BUNDLED_KERNELS[sha256]} hashes to {found}, not {sha256} - "
            f"the installed file does not match the code that references it")
    return kernel


def kernel_provenance(sha256: str) -> dict:
    """Where a bundled kernel came from: head, split, temperature, fit script, date."""
    if sha256 not in BUNDLED_KERNELS:
        raise KeyError(f"no bundled kernel with sha256 {sha256}")
    with np.load(str(_kernel_path(BUNDLED_KERNELS[sha256]))) as data:
        return json.loads(str(data["provenance"]))


def prepare(system: ScoringSystem, kernel: Optional[np.ndarray] = None) -> PreparedScoring:
    """Build the numpy objects for a system, checking the kernel is the one it names.

    A coded system with no explicit kernel loads its bundled one, so the common path is
    ``prepare(get("3di-v1-coded"))`` with nothing else to locate or pass around.  An
    explicit array still wins, for experimenting with a refitted kernel - and is still
    hash-checked, so the experiment cannot be mistaken for the shipped system.
    """
    if kernel is None and system.target_repr == "coded":
        kernel = load_kernel(system.kernel_sha256)
    return PreparedScoring(system, kernel)


#: Reproduces FoldMatch's current behaviour exactly: plain BLOSUM62, gaps 11/1, no scaling.
#: Deliberately uncalibrated here - this system's E-values come from the ALP constants in
#: :mod:`.karlin_altschul`, which carry a finite-size correction this plain form does not.
BLOSUM62_11_1 = ScoringSystem(
    name="blosum62-11-1",
    weight_amino=1.0,
    weight_3di=0.0,
    score_scale=1,
    gap_open=11.0,
    gap_extend=1.0,
    query_repr="states",
    profile_mode="argmax",
    target_repr="amino",
    n_codes=1,
    statistics="alp",
)

#: Structure database: soft 3Di query profile against exact 3Di read from coordinates.
THREE_DI_V1_EXACT = ScoringSystem(
    name="3di-v1-exact",
    weight_amino=1.4,
    weight_3di=2.1,
    score_scale=100,
    gap_open=30.0,
    gap_extend=3.0,
    query_repr="profile",
    profile_mode="logodds",
    profile_temperature=1.080,
    target_repr="exact",
    n_codes=N_STATES,
    lam=0.0210,
    k=1.61e-4,
    reference_set="cath-val-2026-09-22",
    calibration="8000 cross-CATH-topology pairs, censored tail fit at q=0.9, "
                "descaled scores; observed null rate 1.09x nominal at E<=1e-2",
)

#: Sequence-only database: soft query profile against one byte per residue (predicted state
#: x confidence bucket), interpreted through the fitted kernel.
THREE_DI_V1_CODED = ScoringSystem(
    name="3di-v1-coded",
    weight_amino=1.4,
    weight_3di=2.1,
    score_scale=100,
    gap_open=30.0,
    gap_extend=3.0,
    query_repr="profile",
    profile_mode="logodds",
    profile_temperature=1.080,
    target_repr="coded",
    n_codes=240,
    kernel_sha256="8b71484501d18bfaef9bdd27105bceb3bce9935797cb9733183363d62c8ea423",
    lam=0.0186,
    k=1.10e-4,
    reference_set="cath-val-2026-09-22",
    calibration="8000 cross-CATH-topology pairs, censored tail fit at q=0.9, "
                "descaled scores; kernel fitted on the validation split at T=1.080",
)

#: The 3Di encoding a new ``build sequences --three-di`` database is written with. A
#: search does NOT read this - it reads the system recorded in the database's manifest -
#: so changing it affects only databases built afterwards.
DEFAULT_THREE_DI = "3di-v1-coded"

REGISTRY: Dict[str, ScoringSystem] = {
    s.name: s for s in (BLOSUM62_11_1, THREE_DI_V1_EXACT, THREE_DI_V1_CODED)
}


def get(name: str) -> ScoringSystem:
    if name not in REGISTRY:
        raise KeyError(f"unknown scoring system {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]
