"""The in-memory esm-codes Stage 2 (``foldmatch.search.code_alignment.align_query_codes``) and the scoring-system
resolvers it uses (``scoring_system.resolve`` / ``resolve_system``).

No ESM3: the arrays are the F1 fixtures' (``pairs_v1.npz``) and the F-deg ones, as in ``test_code_alignment``, whose
helpers (``World``, ``_signature``, ``_path_from_hit``, the invariance workloads) this file reuses. The reference is
fm-search's own Stage 2, ``align_candidates_esm_codes``, on the same arrays.
"""
import dataclasses
import unittest
from unittest import mock

from foldmatch.search import scoring_system as ss

SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)
OTHER_STATISTICS = dataclasses.replace(SYSTEM, name="esm-codes-test-stats", calibration_version="test")


def _fake_prepared(system=SYSTEM):
    return ss.PreparedSystem(system=system, files={}, bundle={}, encoders_file="e", scoring_file="s")


# =====================================================================================================================
# C1: scoring_system.resolve_system / resolve
# =====================================================================================================================
class ResolveSystemTest(unittest.TestCase):
    """The scoring-system argument without I/O (``resolve_system``) and prepared (``resolve``)."""

    def test_every_accepted_form_gives_the_registry_entry(self):
        prepared = _fake_prepared()
        with mock.patch.object(ss, "prepare", side_effect=AssertionError("resolve_system did I/O")):
            self.assertIs(ss.resolve_system(SYSTEM.name), SYSTEM)
            self.assertIs(ss.resolve_system(SYSTEM), SYSTEM)
            self.assertIs(ss.resolve_system(prepared), SYSTEM)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "scoring_system is required: pass the name, EsmCodesSystem or "
                                                "PreparedSystem"):
            ss.resolve_system(None)
        with self.assertRaises(TypeError):
            ss.resolve_system(3)
        with self.assertRaises(ss.UnknownScoringSystemError):
            ss.resolve_system("esm-codes-v0")
        edited = dataclasses.replace(SYSTEM, calibration_version="edited")
        with self.assertRaisesRegex(ss.UnknownScoringSystemError, "calibration_version"):
            ss.resolve_system(edited)
        with self.assertRaises(ss.UnknownScoringSystemError):
            ss.resolve_system(_fake_prepared(OTHER_STATISTICS))
        with self.assertRaises(ss.UnknownScoringSystemError):
            ss.resolve(_fake_prepared(OTHER_STATISTICS))

    def test_resolve_prepares_once_and_keeps_a_prepared_system(self):
        prepared = _fake_prepared()
        with mock.patch.object(ss, "prepare", return_value=prepared) as prepare:
            self.assertIs(ss.resolve(prepared), prepared)
            prepare.assert_not_called()
            self.assertIs(ss.resolve(SYSTEM.name), prepared)
            prepare.assert_called_once_with(SYSTEM, local_files_only=False)
            ss.resolve(SYSTEM, local_files_only=True)
            prepare.assert_called_with(SYSTEM, local_files_only=True)
        with self.assertRaises(ValueError):
            ss.resolve(None)
