"""Invariants of the stage-2 scoring system.

Each test here pins a failure that produces plausible numbers rather than an error, so it
would otherwise reach production unnoticed:

* the soft path drifting from the hard path (rounding, operand order);
* lambda applied on the wrong score scale (wrong by ``e**600`` at S = 300);
* the paired alphabet changing plain BLOSUM62 alignments;
* the kernel branch diverging from the exact branch;
* a log-odds profile that is not neutral at the background;
* gaps deployed unscaled against a scaled matrix (which makes gaps nearly free);
* a database searched under a different scoring system than it was built with.
"""
import hashlib
import json
import math
import unittest
from dataclasses import replace

import numpy as np

from biotite.sequence import ProteinSequence
from biotite.sequence.align import align_optimal

from foldmatch.search import scoring_system as ss

SEQUENCE = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGCEKAVQVKVKALPDAQFEVVHSLAKWKR"
            "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")


def _three_di_fixture(length):
    """A 3Di string of exactly ``length`` states, with runs like a real one.

    Generated rather than pasted so it can never drift out of step with SEQUENCE - a
    length mismatch there would fail the tests for a reason that has nothing to do with
    the invariant under test.
    """
    rng = np.random.default_rng(20260922)
    states = []
    while len(states) < length:
        states.extend([rng.integers(ss.N_STATES)] * int(rng.integers(1, 5)))
    return "".join(ss.THREE_DI[s] for s in states[:length])


THREE_DI_STRING = _three_di_fixture(len(SEQUENCE))


def one_hot(states):
    profile = np.zeros((len(states), ss.N_STATES))
    profile[np.arange(len(states)), states] = 1.0
    return profile


def amino_codes(sequence):
    return ProteinSequence(sequence).code


def soft_profile(length, seed=7):
    """A genuinely soft profile: no row is close to one-hot."""
    rng = np.random.default_rng(seed)
    profile = rng.dirichlet(np.full(ss.N_STATES, 0.8), size=length)
    return profile


def mixing_kernel(n_codes=40, seed=11):
    """A kernel whose rows are real mixtures, so the kernel branch is actually exercised."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(ss.N_STATES, 0.5), size=n_codes)


class TestOneHotMatchesHardPath(unittest.TestCase):
    """T1: a one-hot profile must reproduce the string path bit-identically at scale 1.

    This is the invariant that keeps the soft path honest: if it drifts, every comparison
    between "predicted 3Di" and "exact 3Di" is measuring the drift as well as the model.
    """

    def test_matrix_and_alignment_identical(self):
        system = ss.ScoringSystem(
            name="t1", score_scale=1, query_repr="profile", profile_mode="logodds",
            target_repr="exact", n_codes=ss.N_STATES)
        prepared = ss.prepare(system)

        states = ss.three_di_codes(THREE_DI_STRING)
        codes = amino_codes(SEQUENCE)
        query, soft_matrix = prepared.query_matrix(one_hot(states), codes)
        hard_matrix = prepared.hard_matrix()

        # Row i of the soft matrix must equal the hard matrix's row for symbol i.
        hard_rows = hard_matrix.score_matrix()[
            ss.pair_codes(codes, states, ss.N_STATES)]
        np.testing.assert_array_equal(soft_matrix.score_matrix(), hard_rows)

        target = prepared.target_sequence(codes, states)
        gaps = system.gaps_deployed
        soft = align_optimal(query, target, soft_matrix, gap_penalty=gaps,
                             local=True, max_number=1)[0]
        hard = align_optimal(prepared.target_sequence(codes, states), target, hard_matrix,
                             gap_penalty=gaps, local=True, max_number=1)[0]
        self.assertEqual(int(soft.score), int(hard.score))
        np.testing.assert_array_equal(soft.trace, hard.trace)

    def test_expected_mode_also_exact_for_one_hot(self):
        states = ss.three_di_codes(THREE_DI_STRING)
        argmax = ss.profile_term(one_hot(states), "argmax")
        for mode in ("expected", "logodds"):
            np.testing.assert_allclose(
                ss.profile_term(one_hot(states), mode), argmax, atol=1e-9,
                err_msg=f"{mode} must reduce to the argmax row for a one-hot profile")

    def test_expected_mode_on_a_soft_profile(self):
        # One-hot profiles make 'expected' and 'argmax' agree by construction, so they
        # cannot tell the two modes apart.  This pins the defining formula.
        _, structural = ss._base_scores()
        profile = soft_profile(24)
        np.testing.assert_allclose(
            ss.profile_term(profile, "expected"), profile @ structural, atol=1e-12)
        self.assertGreater(
            np.abs(ss.profile_term(profile, "expected")
                   - ss.profile_term(profile, "argmax")).max(), 1.0,
            "a soft profile must distinguish expected from argmax")


class TestScaleInvariance(unittest.TestCase):
    """T2: the same pair must get the same E-value at any score scale.

    This is the test that catches applying a lambda fitted on the descaled float to
    biotite's scaled integer.  It fails by hundreds of orders of magnitude, not subtly.
    """

    def _score(self, scale):
        system = ss.ScoringSystem(
            name=f"t2-{scale}", score_scale=scale, query_repr="profile",
            profile_mode="logodds", target_repr="exact", n_codes=ss.N_STATES,
            lam=0.0210, k=1.61e-4, reference_set="test")
        prepared = ss.prepare(system)
        states = ss.three_di_codes(THREE_DI_STRING)
        codes = amino_codes(SEQUENCE)
        query, matrix = prepared.query_matrix(one_hot(states), codes)
        target = prepared.target_sequence(codes, np.roll(states, 7))
        alignment = align_optimal(query, target, matrix,
                                  gap_penalty=system.gaps_deployed,
                                  local=True, max_number=1)[0]
        return system, int(alignment.score), len(codes), len(codes)

    def test_evalue_is_scale_invariant(self):
        evalues = []
        for scale in (1, 10, 100):
            system, score, m, n = self._score(scale)
            evalues.append(system.evalue(score, m, n))
        for scale, value in zip((10, 100), evalues[1:]):
            ratio = value / evalues[0]
            self.assertTrue(
                0.5 < ratio < 2.0,
                f"E at scale {scale} is {ratio:.3g}x the scale-1 value; "
                f"lambda is not being divided by the scale")

    def test_forgetting_the_scale_is_catastrophic(self):
        # Guards the guard: if lam_deployed ever stopped dividing, would we notice?
        system, score, m, n = self._score(100)
        correct = system.evalue(score, m, n)
        naive = system.k * m * n * math.exp(-system.lam * score)
        self.assertLess(naive, correct * 1e-50,
                        "the un-divided lambda must be dramatically different, "
                        "otherwise this invariant has no teeth")


class TestAminoOnlyReproducesBlosum(unittest.TestCase):
    """T3: the paired machinery must not change a plain BLOSUM62 alignment."""

    def test_blosum_alignment_unchanged(self):
        system = ss.BLOSUM62_11_1
        prepared = ss.prepare(system)
        codes_a = amino_codes(SEQUENCE)
        codes_b = amino_codes(SEQUENCE[30:] + SEQUENCE[:30])

        paired = align_optimal(
            prepared.target_sequence(codes_a), prepared.target_sequence(codes_b),
            prepared.hard_matrix(), gap_penalty=system.gaps_deployed,
            local=True, max_number=1)[0]

        from biotite.sequence.align import SubstitutionMatrix
        native = align_optimal(
            ProteinSequence(SEQUENCE), ProteinSequence(SEQUENCE[30:] + SEQUENCE[:30]),
            SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-11, -1),
            local=True, max_number=1)[0]

        self.assertEqual(int(paired.score), int(native.score))
        np.testing.assert_array_equal(paired.trace, native.trace)


class TestKernelBranch(unittest.TestCase):
    """T4: the kernel branch must equal the exact path when the kernel is degenerate, and
    the defining mixture formula when it is not."""

    def test_identity_expansion_kernel(self):
        profile = one_hot(ss.three_di_codes(THREE_DI_STRING[::-1]))
        exact = ss.profile_term(profile, "logodds", kernel=None)

        # 40 codes, each a duplicate of one state: code c means state c // 2.
        kernel = np.zeros((40, ss.N_STATES))
        kernel[np.arange(40), np.arange(40) // 2] = 1.0
        coded = ss.profile_term(profile, "logodds", kernel=kernel)

        np.testing.assert_allclose(coded[:, ::2], exact, atol=1e-9)
        np.testing.assert_allclose(coded[:, 1::2], exact, atol=1e-9)

    def test_mixing_kernel_matches_the_defining_formula(self):
        """A one-hot kernel cannot distinguish ``exp(lambda M) @ k.T`` from
        ``exp(lambda (M @ k.T))`` - they agree exactly when each row picks one state.
        Only a genuine mixture separates them, and they differ by thousands of score
        units when it does."""
        _, structural = ss._base_scores()
        profile = soft_profile(16)
        kernel = mixing_kernel(40)

        got = ss.profile_term(profile, "logodds", kernel=kernel)

        # (1/lambda) ln sum_k sum_l p_k q_l exp(lambda M[k,l]), computed the slow way.
        brute = np.empty((len(profile), len(kernel)))
        for i in range(len(profile)):
            for c in range(len(kernel)):
                total = 0.0
                for k in range(ss.N_STATES):
                    for l in range(ss.N_STATES):
                        total += (profile[i, k] * kernel[c, l]
                                  * math.exp(ss.LAMBDA_3DI * structural[k, l]))
                brute[i, c] = math.log(total) / ss.LAMBDA_3DI
        np.testing.assert_allclose(got, brute, rtol=1e-10)

        wrong = np.log(profile @ np.exp(ss.LAMBDA_3DI * (structural @ kernel.T)))
        self.assertGreater(np.abs(got - wrong / ss.LAMBDA_3DI).max(), 1.0,
                           "the two kernel placements must be distinguishable here")

    def test_expected_mode_with_kernel(self):
        _, structural = ss._base_scores()
        profile, kernel = soft_profile(16), mixing_kernel(40)
        np.testing.assert_allclose(
            ss.profile_term(profile, "expected", kernel=kernel),
            profile @ (structural @ kernel.T), atol=1e-12)

    def test_coded_path_aligns_end_to_end(self):
        """Nothing else ever runs an alignment through the 240-code alphabet."""
        kernel = mixing_kernel(240)
        digest = hashlib.sha256(np.ascontiguousarray(kernel).tobytes()).hexdigest()
        system = ss.ScoringSystem(
            name="t-coded", score_scale=100, gap_open=30.0, gap_extend=3.0,
            query_repr="profile", profile_mode="logodds", target_repr="coded",
            n_codes=240, kernel_sha256=digest, lam=0.0186, k=1.10e-4,
            reference_set="test")
        prepared = ss.prepare(system, kernel)
        codes = amino_codes(SEQUENCE)
        rng = np.random.default_rng(3)
        target_codes = rng.integers(0, 240, size=len(codes))

        query, matrix = prepared.query_matrix(soft_profile(len(codes)), codes)
        self.assertEqual(matrix.score_matrix().shape,
                         (len(codes), ss.amino_alphabet_size() * 240))
        target = prepared.target_sequence(codes, target_codes)
        alignment = align_optimal(query, target, matrix,
                                  gap_penalty=system.gaps_deployed,
                                  local=True, max_number=1)[0]
        self.assertGreater(int(alignment.score), 0)
        self.assertGreater(system.evalue(int(alignment.score), len(codes), len(codes)), 0)

    def test_hard_matrix_is_refused_for_coded(self):
        kernel = mixing_kernel(240)
        digest = hashlib.sha256(np.ascontiguousarray(kernel).tobytes()).hexdigest()
        system = ss.ScoringSystem(
            name="t-coded-hard", query_repr="profile", profile_mode="logodds",
            target_repr="coded", n_codes=240, kernel_sha256=digest)
        with self.assertRaises(ValueError):
            ss.prepare(system, kernel).hard_matrix()


class TestLogOddsBackground(unittest.TestCase):
    """T5: log-odds must be ~0 for a profile at the matrix background."""

    def test_background_profile_scores_near_zero(self):
        background = np.array([
            0.0489372, 0.0306991, 0.101049, 0.0329671, 0.0276149, 0.0416262, 0.0452521,
            0.030876, 0.0297251, 0.0607036, 0.0150238, 0.0215826, 0.0783843, 0.0512926,
            0.0264886, 0.0610702, 0.0201311, 0.215998, 0.0310265, 0.0295417])
        background = background / background.sum()
        profile = np.tile(background, (12, 1))
        term = ss.profile_term(profile, "logodds")
        # Averaged over target states drawn from the same background, this is the log-odds
        # of the background against itself: zero up to the matrix's own rounding.
        mean_logodds = float((term @ background).mean())
        # The measured value is ~4e-6, so 1e-3 is 200x headroom against a biotite matrix
        # revision while still pinning LAMBDA_3DI to about +/-0.1%.  A looser bound admits
        # a transposed digit (0.315568 for 0.351568) unnoticed.
        self.assertLess(abs(mean_logodds), 1e-3,
                        f"a background profile must score ~0 under log-odds, got "
                        f"{mean_logodds:.3g}")
        expected = float((ss.profile_term(profile, "expected") @ background).mean())
        self.assertGreaterEqual(mean_logodds + 1e-9, expected,
                                "log-odds is >= expected by Jensen's inequality")
        # And the expected score under the background must be clearly negative, or local
        # alignment leaves the Karlin-Altschul regime altogether.
        self.assertLess(expected, 0.0)


class TestTemperature(unittest.TestCase):
    """The calibrated constants were fitted at T = 1.080, so the system must apply it.

    A stored-but-unused temperature is precisely the silent mismatch this module exists
    to prevent: the caller would get plausible E-values computed under a different null.
    """

    def _prepared(self, temperature):
        return ss.prepare(ss.ScoringSystem(
            name=f"t-temp-{temperature}", score_scale=100, query_repr="profile",
            profile_mode="logodds", profile_temperature=temperature,
            target_repr="exact", n_codes=ss.N_STATES))

    def test_temperature_one_is_the_identity(self):
        profile = soft_profile(12)
        np.testing.assert_allclose(self._prepared(1.0).soften(profile), profile, atol=1e-12)

    def test_matches_the_calibrated_definition(self):
        # read_profiles() in the research code divides the LOG-probabilities by T and
        # renormalises; anything else is a different scoring system.
        profile = soft_profile(12)
        scaled = np.log(profile) / 1.080
        scaled -= scaled.max(axis=1, keepdims=True)
        expected = np.exp(scaled)
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(self._prepared(1.080).soften(profile), expected,
                                   atol=1e-12)

    def test_temperature_changes_the_matrix(self):
        profile, codes = soft_profile(len(SEQUENCE)), amino_codes(SEQUENCE)
        _, cold = self._prepared(1.0).query_matrix(profile, codes)
        _, warm = self._prepared(1.080).query_matrix(profile, codes)
        self.assertFalse(
            np.array_equal(cold.score_matrix(), warm.score_matrix()),
            "the temperature must reach the matrix, or the constants describe "
            "a different scoring system than the one being run")

    def test_registry_temperature_is_the_fitted_one(self):
        for system in (ss.THREE_DI_V1_EXACT, ss.THREE_DI_V1_CODED):
            self.assertAlmostEqual(system.profile_temperature, 1.080)


class TestMinConfidence(unittest.TestCase):
    """Masking must drop the structural term at the UNCERTAIN positions, not the confident ones."""

    def test_only_low_confidence_rows_are_zeroed(self):
        profile = np.vstack([
            one_hot(np.array([3, 11])),                       # confident
            np.full((2, ss.N_STATES), 1.0 / ss.N_STATES),     # flat
        ])
        codes = amino_codes(SEQUENCE[:4])
        system = ss.ScoringSystem(
            name="t-conf", score_scale=1, query_repr="profile", profile_mode="logodds",
            target_repr="exact", n_codes=ss.N_STATES, min_confidence=0.5)
        _, matrix = ss.prepare(system).query_matrix(profile, codes)
        scores = matrix.score_matrix().reshape(4, ss.amino_alphabet_size(), ss.N_STATES)

        # On a masked row every 3Di state scores alike: only the amino term survives.
        for row in (2, 3):
            self.assertEqual(len(np.unique(scores[row, 0])), 1,
                             f"row {row} is below min_confidence and must be amino-only")
        for row in (0, 1):
            self.assertGreater(len(np.unique(scores[row, 0])), 1,
                               f"row {row} is confident and must keep its 3Di term")

    def test_out_of_range_is_rejected(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                ss.ScoringSystem(name="bad", min_confidence=bad)


class TestDeployedNumbers(unittest.TestCase):
    """T6/T7: gaps scale with the matrix, and a system's identity is stable."""

    def test_gaps_are_scaled(self):
        self.assertEqual(ss.THREE_DI_V1_EXACT.gaps_deployed, (-3000, -300))
        self.assertEqual(ss.BLOSUM62_11_1.gaps_deployed, (-11, -1))

    def test_hard_matrix_is_scaled_too(self):
        """Scaling the matrix but not the gaps (or the reverse) makes gaps nearly free."""
        states = ss.three_di_codes(THREE_DI_STRING)
        codes = amino_codes(SEQUENCE)
        reference = None
        for scale in (1, 10, 100):
            system = ss.ScoringSystem(
                name=f"t-hard-{scale}", score_scale=scale, query_repr="states",
                profile_mode="argmax", target_repr="exact", n_codes=ss.N_STATES)
            prepared = ss.prepare(system)
            matrix = prepared.hard_matrix().score_matrix()
            self.assertEqual(matrix.shape,
                             (ss.amino_alphabet_size() * 20,) * 2)
            target = prepared.target_sequence(codes, np.roll(states, 11))
            score = int(align_optimal(
                prepared.target_sequence(codes, states), target, prepared.hard_matrix(),
                gap_penalty=system.gaps_deployed, local=True, max_number=1)[0].score)
            if reference is None:
                reference = score
            else:
                self.assertAlmostEqual(
                    score / scale, reference, delta=0.05 * abs(reference),
                    msg=f"the hard matrix at scale {scale} must scale with its gaps")

    def test_bit_score_and_pvalue(self):
        system = ss.THREE_DI_V1_EXACT
        score = 15000  # scaled
        self.assertAlmostEqual(
            system.bit_score(score),
            (system.lam_deployed * score - math.log(system.k)) / math.log(2.0))
        evalue = system.evalue(score, 200, 200)
        self.assertAlmostEqual(system.pvalue(score, 200, 200), -math.expm1(-evalue))
        with self.assertRaises(ValueError):
            ss.BLOSUM62_11_1.bit_score(score)

    def test_registry_constants_are_pinned(self):
        """Any drift in a shipped system's numbers must break a test, loudly.

        These digests are the identity a database manifest records; if one changes,
        every database built under it has to be re-examined, so the change should be a
        deliberate edit to this table rather than a side effect.
        """
        expected = {
            "blosum62-11-1": (1.0, 0.0, 1, 11.0, 1.0, None, None),
            "3di-v1-exact": (1.4, 2.1, 100, 30.0, 3.0, 0.0210, 1.61e-4),
            "3di-v1-coded": (1.4, 2.1, 100, 30.0, 3.0, 0.0186, 1.10e-4),
        }
        self.assertEqual(set(ss.REGISTRY), set(expected))
        for name, values in expected.items():
            system = ss.get(name)
            self.assertEqual(
                (system.weight_amino, system.weight_3di, system.score_scale,
                 system.gap_open, system.gap_extend, system.lam, system.k), values)

    def test_lambda_is_divided_by_scale(self):
        self.assertAlmostEqual(ss.THREE_DI_V1_EXACT.lam_deployed, 0.0210 / 100)
        self.assertIsNone(ss.BLOSUM62_11_1.lam_deployed)

    def test_digest_round_trip_and_sensitivity(self):
        for system in ss.REGISTRY.values():
            restored = ss.ScoringSystem.from_json(system.to_json())
            self.assertEqual(restored, system)
            self.assertEqual(restored.digest(), system.digest())
            self.assertEqual(restored.gaps_deployed, system.gaps_deployed)
        moved = ss.THREE_DI_V1_EXACT.calibrated(0.0211, 1.61e-4, "cath-val")
        self.assertNotEqual(moved.digest(), ss.THREE_DI_V1_EXACT.digest())

    def test_unknown_field_is_rejected(self):
        payload = json.loads(ss.THREE_DI_V1_EXACT.to_json())
        payload["gap_opne"] = 30.0
        with self.assertRaises(ValueError):
            ss.ScoringSystem.from_json(json.dumps(payload))


class TestBundledKernel(unittest.TestCase):
    """The kernel ships with the package, because it defines the scoring system.

    Its lambda/K were fitted with this exact kernel, so a coded system whose kernel had
    to be located out of band would be one misplaced file away from reporting confident
    E-values for a different scoring system.
    """

    def test_coded_system_prepares_with_no_arguments(self):
        prepared = ss.prepare(ss.get("3di-v1-coded"))
        self.assertEqual(prepared.kernel.shape, (240, ss.N_STATES))
        np.testing.assert_allclose(prepared.kernel.sum(axis=1), 1.0, atol=1e-9)

    def test_bundled_kernel_matches_the_registry_hash(self):
        kernel = ss.load_kernel(ss.THREE_DI_V1_CODED.kernel_sha256)
        digest = hashlib.sha256(np.ascontiguousarray(kernel).tobytes()).hexdigest()
        self.assertEqual(digest, ss.THREE_DI_V1_CODED.kernel_sha256)

    def test_every_bundled_kernel_is_loadable_and_self_consistent(self):
        for sha in ss.BUNDLED_KERNELS:
            kernel = ss.load_kernel(sha)
            self.assertEqual(
                hashlib.sha256(np.ascontiguousarray(kernel).tobytes()).hexdigest(), sha)
            np.testing.assert_allclose(kernel.sum(axis=1), 1.0, atol=1e-9)

    def test_provenance_records_what_it_was_fitted_on(self):
        record = ss.kernel_provenance(ss.THREE_DI_V1_CODED.kernel_sha256)
        for field in ("head", "fitted_on", "profile_temperature", "fit_script", "sha256"):
            self.assertIn(field, record)
        self.assertEqual(record["sha256"], ss.THREE_DI_V1_CODED.kernel_sha256)
        # The temperature in the file must be the one the system scores at, or the kernel
        # describes confidence buckets the aligner never produces.
        self.assertAlmostEqual(record["profile_temperature"],
                               ss.THREE_DI_V1_CODED.profile_temperature)

    def test_unknown_hash_is_refused(self):
        with self.assertRaises(KeyError):
            ss.load_kernel("0" * 64)

    def test_temperature_coupling_is_enforced_at_runtime(self):
        """A kernel's rows are indexed by confidence bucket, which depends on T.

        Retuning the query temperature without refitting the kernel would leave the codes
        pointing at rows fitted for a different sharpness - 26.8% of residues change
        bucket between T=1.0 and T=1.080 - so it must fail loudly rather than score.
        """
        mismatched = replace(ss.THREE_DI_V1_CODED, profile_temperature=1.0)
        with self.assertRaisesRegex(ValueError, "fitted at T="):
            ss.prepare(mismatched)
        # And equally when the same kernel is handed over explicitly.
        with self.assertRaisesRegex(ValueError, "fitted at T="):
            ss.prepare(mismatched, ss.load_kernel(ss.THREE_DI_V1_CODED.kernel_sha256))
        # The shipped pairing is consistent, so it prepares.
        self.assertIsNotNone(ss.prepare(ss.THREE_DI_V1_CODED).kernel)

    def test_kernel_file_is_packaged(self):
        """Catches a packaging regression here rather than after a release.

        The tests run from the source tree, where the file is present whether or not the
        build includes it; this asserts the path the installed package resolves.
        """
        for filename in ss.BUNDLED_KERNELS.values():
            self.assertTrue(ss._kernel_path(filename).is_file(),
                            f"{filename} is not where importlib.resources looks for it")

    def test_every_coded_registry_system_has_its_kernel_bundled(self):
        for system in ss.REGISTRY.values():
            if system.target_repr == "coded":
                self.assertIn(system.kernel_sha256, ss.BUNDLED_KERNELS,
                              f"{system.name} references a kernel that does not ship")


class TestKernelIdentityIsChecked(unittest.TestCase):
    """A database built under one kernel must not be searched under another."""

    def test_wrong_kernel_is_refused(self):
        kernel = np.full((240, ss.N_STATES), 1.0 / ss.N_STATES)
        with self.assertRaises(ValueError):
            ss.prepare(ss.THREE_DI_V1_CODED, kernel)

    def test_matching_kernel_is_accepted(self):
        kernel = np.full((240, ss.N_STATES), 1.0 / ss.N_STATES)
        digest = hashlib.sha256(np.ascontiguousarray(kernel).tobytes()).hexdigest()
        system = ss.ScoringSystem(
            name="t-kernel", query_repr="profile", profile_mode="logodds",
            target_repr="coded", n_codes=240, kernel_sha256=digest)
        prepared = ss.prepare(system, kernel)
        self.assertEqual(prepared.kernel.shape, (240, ss.N_STATES))

    def test_exact_system_refuses_a_kernel(self):
        with self.assertRaises(ValueError):
            ss.prepare(ss.THREE_DI_V1_EXACT, np.eye(20))


class TestValidation(unittest.TestCase):
    """Inconsistent sets must fail loudly at construction, not silently at search time."""

    def test_inconsistent_configurations_raise(self):
        cases = {
            "coded without kernel": dict(target_repr="coded", n_codes=240),
            "exact with kernel": dict(target_repr="exact", kernel_sha256="a" * 64),
            "amino with 3di weight": dict(target_repr="amino", n_codes=1,
                                          query_repr="states", profile_mode="argmax"),
            "lambda without K": dict(lam=0.02, reference_set="x"),
            "lambda without reference": dict(lam=0.02, k=1e-4),
            "negative gaps": dict(gap_open=-30.0),
            "extend above open": dict(gap_open=3.0, gap_extend=30.0),
            "fractional scale": dict(score_scale=1.5),
            "soft mode on a states query": dict(query_repr="states",
                                                profile_mode="logodds"),
            "hard query on a coded target": dict(query_repr="states",
                                                 profile_mode="argmax",
                                                 target_repr="coded", n_codes=240,
                                                 kernel_sha256="a" * 64),
            "zero temperature": dict(profile_temperature=0.0),
        }
        for label, overrides in cases.items():
            with self.subTest(label):
                with self.assertRaises((ValueError, TypeError)):
                    ss.ScoringSystem(name="bad", **overrides)

    def test_code_bounds_are_checked(self):
        codes = amino_codes(SEQUENCE[:10])
        with self.assertRaisesRegex(ValueError, "structural codes outside"):
            ss.pair_codes(codes, np.full(10, ss.N_STATES), ss.N_STATES)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            ss.pair_codes(codes, np.zeros(9), ss.N_STATES)
        # An out-of-range amino code does not raise inside biotite: the aligner reads
        # past the end of the score matrix and returns a plausible number.
        with self.assertRaisesRegex(ValueError, "amino codes outside"):
            ss.pair_codes(np.array([ss.amino_alphabet_size()]), np.zeros(1), ss.N_STATES)

    def test_pair_codes_round_trip(self):
        """A database storing only the paired byte must be able to recover both halves."""
        codes = amino_codes(SEQUENCE)
        states = ss.three_di_codes(THREE_DI_STRING)
        for n_codes in (1, ss.N_STATES, 240):
            structural = None if n_codes == 1 else states % n_codes
            paired = ss.pair_codes(codes, structural, n_codes)
            amino_back, structural_back = ss.split_pair_codes(paired, n_codes)
            np.testing.assert_array_equal(amino_back, codes)
            if n_codes != 1:
                np.testing.assert_array_equal(structural_back, structural)

    def test_three_di_fallback_is_pinned(self):
        # Foldseek writes 'd' for residues it cannot encode, and 'd' is also a real
        # state, so this coercion is unrecoverable downstream - it should not drift.
        np.testing.assert_array_equal(
            ss.three_di_codes("aXd"),
            [0, ss.THREE_DI.index("d"), ss.THREE_DI.index("d")])

    def test_statistics_field(self):
        self.assertEqual(ss.BLOSUM62_11_1.statistics, "alp")
        self.assertEqual(ss.THREE_DI_V1_EXACT.statistics, "karlin_altschul")
        with self.assertRaises(ValueError):
            ss.ScoringSystem(name="bad", statistics="made-up")
        with self.assertRaises(ValueError):
            ss.ScoringSystem(name="bad", statistics="alp", lam=0.02, k=1e-4,
                             reference_set="x")


if __name__ == "__main__":
    unittest.main()
