"""The vendored 3Di head and its release bundle.

The failure this file exists to prevent: ``nhead`` is not recoverable from the saved
parameters, so loading the head with the wrong value succeeds, matches every tensor, and
predicts nonsense - while the profile temperature, the target kernel and the fitted
lambda/K all keep describing the model that was supposed to be running.
"""
import tempfile
import unittest
from pathlib import Path

import torch

from foldmatch.model import three_di_head as th
from foldmatch.utils import model as model_utils

ARCHITECTURE = {
    "class": "TransformerThreeDiHead",
    "input_features": 1536,
    "hidden": 256,
    "nhead": 8,
    "num_layers": 2,
    "dim_feedforward": 1024,
    "with_ss8": False,
}


def bundle_for(architecture=None, fmt=model_utils.THREE_DI_FORMAT):
    architecture = dict(architecture or ARCHITECTURE)
    head = th.build_head(architecture)
    return {
        "format": fmt,
        "architecture": architecture,
        "provenance": {"name": "test", "profile_temperature": 1.080},
        "state_dict": head.state_dict(),
    }, head


class WrongArchitectureIsSilentTest(unittest.TestCase):
    """Why the bundle records the architecture instead of the loader hardcoding it."""

    def test_nhead_is_not_recoverable_from_the_weights(self):
        eight = th.TransformerThreeDiHead(**{k: v for k, v in ARCHITECTURE.items()
                                             if k != "class"}, dropout=0.0)
        four = th.TransformerThreeDiHead(
            **{**{k: v for k, v in ARCHITECTURE.items() if k != "class"}, "nhead": 4},
            dropout=0.0)
        # Every parameter matches by name and shape, so strict=True does not object.
        four.load_state_dict(eight.state_dict(), strict=True)
        self.assertEqual(
            {k: tuple(v.shape) for k, v in eight.state_dict().items()},
            {k: tuple(v.shape) for k, v in four.state_dict().items()})
        # ...but the model computes something different.
        eight.eval(), four.eval()
        probe = torch.randn(2, 31, 1536)
        with torch.no_grad():
            self.assertFalse(torch.allclose(eight(probe), four(probe), atol=1e-6),
                             "if these agreed, nhead would not matter and this whole "
                             "bundle format would be unnecessary")


class BundleRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_round_trip_is_bit_identical(self):
        bundle, head = bundle_for()
        head.eval()
        path = self.dir / "head.pt"
        torch.save(bundle, path)
        loaded, back = model_utils.load_three_di_head(path, device=torch.device("cpu"))
        probe = torch.randn(3, 41, 1536)
        with torch.no_grad():
            self.assertTrue(torch.equal(head(probe), loaded(probe)))
        self.assertEqual(back["provenance"]["profile_temperature"], 1.080)
        self.assertFalse(loaded.training, "a served head must be in eval mode")

    def test_every_architecture_round_trips(self):
        for architecture in (
            ARCHITECTURE,
            {"class": "LinearThreeDiHead", "input_features": 1536, "hidden": 512},
            {"class": "CnnThreeDiHead", "input_features": 1536, "hidden": 512,
             "kernel_size": 7, "num_layers": 2},
        ):
            with self.subTest(architecture["class"]):
                bundle, head = bundle_for(architecture)
                path = self.dir / f"{architecture['class']}.pt"
                torch.save(bundle, path)
                loaded, _ = model_utils.load_three_di_head(path, device=torch.device("cpu"))
                head.eval()
                probe = torch.randn(2, 23, 1536)
                with torch.no_grad():
                    self.assertTrue(torch.equal(head(probe), loaded(probe)))

    def test_weights_only_load_is_used(self):
        """The bundle must stay loadable under weights_only=True, as the aggregator is."""
        bundle, _ = bundle_for()
        path = self.dir / "head.pt"
        torch.save(bundle, path)
        back = torch.load(path, weights_only=True, map_location="cpu")
        self.assertEqual(back["architecture"], ARCHITECTURE)

    def test_bare_state_dict_is_refused_with_an_explanation(self):
        _, head = bundle_for()
        path = self.dir / "bare.pt"
        torch.save(head.state_dict(), path)
        with self.assertRaisesRegex(ValueError, "not a 3Di head bundle"):
            model_utils.load_three_di_head(path, device=torch.device("cpu"))

    def test_unknown_format_is_refused(self):
        bundle, _ = bundle_for(fmt="something-else/9")
        path = self.dir / "future.pt"
        torch.save(bundle, path)
        with self.assertRaisesRegex(ValueError, "unsupported bundle format"):
            model_utils.load_three_di_head(path, device=torch.device("cpu"))

    def test_mismatched_architecture_is_refused(self):
        """A bundle whose recorded architecture disagrees with its weights must fail."""
        bundle, _ = bundle_for()
        bundle["architecture"] = {**ARCHITECTURE, "hidden": 512}
        path = self.dir / "mismatch.pt"
        torch.save(bundle, path)
        with self.assertRaises(RuntimeError):
            model_utils.load_three_di_head(path, device=torch.device("cpu"))

    def test_unknown_head_class_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown head class"):
            th.build_head({"class": "NotAHead", "hidden": 8})


class DropoutIsNotServedTest(unittest.TestCase):
    def test_training_dropout_does_not_reach_inference(self):
        """The bundle records the trained dropout; the served head must ignore it."""
        head = th.build_head({**ARCHITECTURE, "dropout": 0.5})
        self.assertEqual(head.dropout.p, 0.0)


class AlphabetAgreesWithScoringTest(unittest.TestCase):
    def test_state_order_matches_the_substitution_matrix(self):
        from foldmatch.search import scoring_system as ss
        self.assertEqual(th.THREE_DI, ss.THREE_DI)
        self.assertEqual(th.N_STATES, ss.N_STATES)


class ReleasePinTest(unittest.TestCase):
    """Every weight this project serves is content-pinned.

    The models are frozen features: the aggregator was trained on ESM3's weights, and the
    3Di head's accuracy, its profile temperature, its target kernel and the
    Karlin-Altschul lambda/K were all calibrated against one specific backbone and one
    specific head. An unpinned download changes all of those without changing any code.
    """

    def test_unpinned_revision_is_refused(self):
        with self.assertRaisesRegex(ValueError, "THREE_DI_REVISION is unset"):
            model_utils.get_three_di_head(revision=None)

    def test_every_model_is_pinned_to_a_commit(self):
        from foldmatch.utils.esm import loaders
        for label, revision in (
            ("aggregator", model_utils.REVISION),
            ("3Di head", model_utils.THREE_DI_REVISION),
            ("ESM3 backbone", loaders.ESM_REVISION),
        ):
            with self.subTest(label):
                self.assertIsNotNone(revision, f"{label} is unpinned")
                self.assertRegex(revision, r"^[0-9a-f]{40}$",
                                 f"{label} must be pinned to a full commit sha")

    def test_three_di_head_points_at_its_own_repo(self):
        # The head lives in its own repository, not alongside the aggregator.
        self.assertEqual(model_utils.THREE_DI_REPO_ID, "rcsb/rcsb-three-di-head")
        self.assertNotEqual(model_utils.THREE_DI_REPO_ID, model_utils.REPO_ID)


if __name__ == "__main__":
    unittest.main()
