"""The 3Di sidecar and its manifest.

Every check here guards a mismatch that would otherwise produce plausible output: the
codes still index a kernel, the alignment still scores, the E-value still comes from a
calibrated-looking constant, and only the answer is wrong.
"""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from foldmatch.search import scoring_system as ss
from foldmatch.search.three_di_store import (
    MANIFEST_VERSION, ThreeDiStore, ThreeDiStoreError,
)

MODEL_REVISION = "6b88ce659351894bdb8954ad7a854d046db12527"
ESM_REVISION = "636acb420daf674df37ab0220f56061a5de2f169"

ENTRIES = {
    "p1": np.array([0, 13, 239, 7], dtype=np.uint8),
    "p2": np.arange(40, dtype=np.uint8),
    "p3": np.full(7, 239, dtype=np.uint8),
}


def build(folder, entries=None, system=None, **manifest_kwargs):
    store = ThreeDiStore(folder, "db")
    store.create(list((entries or ENTRIES).items()))
    store.write_manifest(
        system or ss.get("3di-v1-coded"),
        model_revision=manifest_kwargs.pop("model_revision", MODEL_REVISION),
        esm_revision=manifest_kwargs.pop("esm_revision", ESM_REVISION),
        entry_count=len(entries or ENTRIES),
        **manifest_kwargs,
    )
    return store


class RoundTripTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_codes_round_trip(self):
        store = build(self.dir)
        fetched = store.fetch(ENTRIES)
        self.assertEqual(set(fetched), set(ENTRIES))
        for name, codes in ENTRIES.items():
            np.testing.assert_array_equal(fetched[name], codes)
            self.assertEqual(fetched[name].dtype, np.uint8)
        self.assertEqual(store.count(), len(ENTRIES))

    def test_missing_ids_are_absent_not_an_error(self):
        store = build(self.dir)
        fetched = store.fetch(["p1", "nope"])
        self.assertEqual(set(fetched), {"p1"})

    def test_fetch_beyond_the_sqlite_variable_limit(self):
        many = {f"p{i}": np.full(3, i % 240, dtype=np.uint8) for i in range(2500)}
        store = build(self.dir, entries=many)
        fetched = store.fetch(many)
        self.assertEqual(len(fetched), 2500)

    def test_duplicate_ids_keep_the_first(self):
        store = ThreeDiStore(self.dir, "db")
        store.create([("p", np.array([1], np.uint8)), ("p", np.array([2], np.uint8))])
        np.testing.assert_array_equal(store.fetch(["p"])["p"], [1])


class BuildCompletionTest(unittest.TestCase):
    """The manifest is the build-complete signal; without it the store is not usable."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_store_without_manifest_reports_absent(self):
        store = ThreeDiStore(self.dir, "db")
        store.create(list(ENTRIES.items()))
        self.assertFalse(store.exists(), "an interrupted build must not look complete")
        self.assertTrue(store.is_partial())

    def test_manifest_completes_the_build(self):
        store = build(self.dir)
        self.assertTrue(store.exists())
        self.assertFalse(store.is_partial())

    def test_rebuild_clears_a_stale_manifest_first(self):
        """A manifest from the previous build would declare the new one finished."""
        store = build(self.dir)
        store.create(list(ENTRIES.items()), force=True)
        self.assertFalse(store.exists(), "the old manifest survived a rebuild")

    def test_create_refuses_to_clobber_without_force(self):
        store = build(self.dir)
        with self.assertRaises(FileExistsError):
            store.create(list(ENTRIES.items()))

    def test_manifest_write_leaves_no_temporary(self):
        build(self.dir)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_remove_deletes_both(self):
        store = build(self.dir)
        store.remove()
        self.assertFalse(store.path.exists())
        self.assertFalse(store.manifest_path.exists())


class ManifestContentTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_records_every_input_that_changes_the_bytes(self):
        store = build(self.dir)
        manifest = store.read_manifest()
        for field in ("manifest_version", "scoring_system", "scoring_system_digest",
                      "n_codes", "kernel_sha256", "profile_temperature",
                      "three_di_model_revision", "esm3_revision", "entry_count"):
            self.assertIn(field, manifest)
        self.assertEqual(manifest["three_di_model_revision"], MODEL_REVISION)
        self.assertEqual(manifest["esm3_revision"], ESM_REVISION)

    def test_stores_the_whole_scoring_system_not_only_its_digest(self):
        """A database must stay diagnosable after the shipped registry is refitted."""
        store = build(self.dir)
        system = ss.ScoringSystem.from_json(json.dumps(store.read_manifest()["scoring_system"]))
        self.assertEqual(system, ss.get("3di-v1-coded"))
        self.assertEqual(system.lam, ss.get("3di-v1-coded").lam)

    def test_manifest_is_plain_json(self):
        """Readable without importing foldmatch, unlike the pickled .metadata."""
        store = build(self.dir)
        json.loads(store.manifest_path.read_text())


class CompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.store = build(self.dir)
        self.system = ss.get("3di-v1-coded")

    def test_matching_system_is_accepted(self):
        manifest = self.store.check_compatible(
            self.system, model_revision=MODEL_REVISION, esm_revision=ESM_REVISION)
        self.assertEqual(manifest["entry_count"], len(ENTRIES))

    def test_different_kernel_is_refused(self):
        other = replace(self.system, kernel_sha256="f" * 64)
        with self.assertRaisesRegex(ThreeDiStoreError, "kernel_sha256"):
            self.store.check_compatible(other)

    def test_different_temperature_is_refused(self):
        other = replace(self.system, profile_temperature=1.0)
        with self.assertRaisesRegex(ThreeDiStoreError, "profile_temperature"):
            self.store.check_compatible(other)

    def test_different_code_count_is_refused(self):
        other = replace(self.system, n_codes=120)
        with self.assertRaisesRegex(ThreeDiStoreError, "n_codes"):
            self.store.check_compatible(other)

    def test_different_model_revision_is_refused(self):
        with self.assertRaisesRegex(ThreeDiStoreError, "3Di model"):
            self.store.check_compatible(self.system, model_revision="0" * 40)

    def test_different_esm_revision_is_refused(self):
        """The embeddings behind the codes come from that backbone."""
        with self.assertRaisesRegex(ThreeDiStoreError, "ESM3 backbone"):
            self.store.check_compatible(self.system, esm_revision="0" * 40)

    def test_unknown_manifest_version_is_refused(self):
        manifest = self.store.read_manifest()
        manifest["manifest_version"] = MANIFEST_VERSION + 1
        self.store.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ThreeDiStoreError, "manifest version"):
            self.store.check_compatible(self.system)

    def test_recalibrated_constants_warn_but_do_not_refuse(self):
        """Different lambda/K leave the BYTES valid; only the statistics differ."""
        refitted = self.system.calibrated(0.02, 1.2e-4, "cath-val-refit")
        with self.assertLogs("foldmatch.search.three_di_store", level="WARNING") as logs:
            self.store.check_compatible(refitted)
        self.assertIn("calibrated", "".join(logs.output))

    def test_revisions_are_only_checked_when_supplied(self):
        self.store.check_compatible(self.system)  # must not raise


if __name__ == "__main__":
    unittest.main()
