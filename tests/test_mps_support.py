import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import foldmatch
from foldmatch.model.residue_embedding_aggregator import ResidueEmbeddingAggregator
from foldmatch.modules import assembly_complete_module, chain_complete_module, esm_module
from foldmatch.modules.assembly_complete_module import AssemblyCompleteModule
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.modules.esm_module import EsmModule
from foldmatch.types.api_types import SrcEsmFrom, Accelerator
from foldmatch.utils.data import collate_seq_embeddings
from foldmatch.utils.model import get_aggregator_model, release_mps_cache

_TEST_PATH = os.path.dirname(__file__)
_FASTA = f"{_TEST_PATH}/resources/fasta/test_sequences.fasta"


def _legacy_forward(aggregator, x, x_mask=None):
    # ResidueEmbeddingAggregator.forward before G14, verbatim.
    return aggregator.embedding(aggregator.transformer(x, src_key_padding_mask=x_mask).sum(dim=x.dim() - 2))


def _max_rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


class AggregatorPaddingTest(unittest.TestCase):
    """G14: the aggregator gives the same chain embeddings with and without PyTorch's nested-tensor fast path."""

    @classmethod
    def setUpClass(cls):
        cls.aggregator = get_aggregator_model(torch.device("cpu")).eval()
        # Real ESM3 residue embeddings of different lengths (243 and 116 rows), so one item is padded.
        cls.residues = [
            torch.load(f"{_TEST_PATH}/resources/embeddings/{name}.pt", map_location="cpu", weights_only=True).float()
            for name in ("1acb.A", "2uzi.A")
        ]

    def test_fast_path_disabled_matches_fast_path_and_batch_size_1(self):
        x, x_mask = collate_seq_embeddings(self.residues)
        self.assertTrue(x_mask[1].any())
        transformer = self.aggregator.transformer
        with torch.inference_mode():
            fast = self.aggregator(x, x_mask)
            single = [self.aggregator(*collate_seq_embeddings([r]))[0] for r in self.residues]
            transformer.use_nested_tensor = False
            try:
                slow = self.aggregator(x, x_mask)
                legacy_slow = _legacy_forward(self.aggregator, x, x_mask)
            finally:
                transformer.use_nested_tensor = True
        for i in range(len(self.residues)):
            self.assertLessEqual(_max_rel(slow[i], fast[i]), 1e-5)
            self.assertLessEqual(_max_rel(slow[i], single[i]), 1e-5)
            self.assertLessEqual(_max_rel(fast[i], single[i]), 1e-5)
        # Before G14 the slow path (MPS with PYTORCH_ENABLE_MPS_FALLBACK=1) summed the padded rows.
        self.assertGreater(_max_rel(legacy_slow[1], single[1]), 1e-2)

    def test_cpu_output_byte_identical_to_legacy(self):
        x, x_mask = collate_seq_embeddings(self.residues)
        with torch.inference_mode():
            cases = [
                (x, x_mask),
                collate_seq_embeddings(self.residues[:1]),
                collate_seq_embeddings(self.residues[1:]),
                (self.residues[1], None),
            ]
            for xs, mask in cases:
                self.assertTrue(torch.equal(self.aggregator(xs, mask), _legacy_forward(self.aggregator, xs, mask)))

    def test_device_without_nested_tensor_kernels(self):
        # The meta device lacks aten::_nested_tensor_from_mask_left_aligned exactly as MPS does.
        with torch.device("meta"):
            aggregator = ResidueEmbeddingAggregator().eval()
        x = torch.zeros(2, 7, 1536, device="meta")
        x_mask = torch.zeros(2, 7, dtype=torch.bool, device="meta")
        x_mask[1, 4:] = True
        seen = []
        layer = aggregator.transformer.layers[0]
        with torch.inference_mode():
            with self.assertRaises(NotImplementedError):
                _legacy_forward(aggregator, x, x_mask)
            with mock.patch.object(
                    layer, "forward",
                    side_effect=lambda *a, _f=layer.forward, **k: seen.append(aggregator.transformer.use_nested_tensor) or _f(*a, **k)
            ):
                out = aggregator(x, x_mask)
        self.assertEqual(tuple(out.shape), (2, 1536))
        self.assertEqual(seen, [False])
        self.assertTrue(aggregator.transformer.use_nested_tensor)

    def test_fixture_fasta_cpu_byte_identical_at_batch_sizes_1_and_2(self):
        from foldmatch.inference.chain_complete_inference import predict
        calls = []
        forward = ResidueEmbeddingAggregator.forward

        def spy(aggregator, x, x_mask=None):
            out = forward(aggregator, x, x_mask)
            calls.append(torch.equal(out, _legacy_forward(aggregator, x, x_mask)))
            return out

        for batch_size in (1, 2):
            with mock.patch.object(ResidueEmbeddingAggregator, "forward", spy):
                predict(src_stream=_FASTA, src_from=SrcEsmFrom.fasta, batch_size=batch_size,
                        accelerator=Accelerator.cpu, devices=1)
        self.assertEqual(calls, [True, True, True])


class ReleaseMpsCacheTest(unittest.TestCase):
    """G14: every module that runs ESM3 releases the MPS cache after each batch, and only on MPS."""

    def test_release_only_on_mps(self):
        events = []
        with mock.patch.object(torch.mps, "synchronize", side_effect=lambda: events.append("synchronize")), \
                mock.patch.object(torch.mps, "empty_cache", side_effect=lambda: events.append("empty_cache")):
            for device in ("cpu", torch.device("cpu"), "cuda", "meta"):
                release_mps_cache(device)
            self.assertEqual(events, [])
            release_mps_cache(torch.device("mps"))
            release_mps_cache("mps:0")
        self.assertEqual(events, ["synchronize", "empty_cache"] * 2)

    def test_esm3_modules_release_once_per_batch(self):
        def embedding(_model, esm_prot):
            return torch.ones(len(esm_prot) + 2, 4)

        def aggregator(x, x_mask):
            return x.sum(dim=1)

        chains = [("AAAA", "a"), ("CC", "c")]
        cases = [
            (chain_complete_module, ChainCompleteModule(None, aggregator), chains),
            (esm_module, EsmModule(None), chains),
            (assembly_complete_module, AssemblyCompleteModule(None, aggregator), [(["AAAA", "CC"], "x"), (["CC"], "y")]),
        ]
        for module, lightning_module, batch in cases:
            with mock.patch.object(module, "per_residue_embedding", side_effect=embedding), \
                    mock.patch.object(module, "release_mps_cache") as release:
                lightning_module.predict_step(batch, 0)
                lightning_module.predict_step(batch, 1)
            self.assertEqual(release.call_args_list, [mock.call(lightning_module.device)] * 2, module.__name__)


@unittest.skipUnless(torch.backends.mps.is_available(), "Apple GPU (MPS) not available")
class MpsSmokeTest(unittest.TestCase):
    """G14: predict() on MPS runs without PYTORCH_ENABLE_MPS_FALLBACK and matches CPU."""

    _CHILD = r"""
import json, sys
import numpy as np, torch
from unittest import mock
from foldmatch.inference.chain_complete_inference import predict
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.types.api_types import SrcEsmFrom, Accelerator

fasta, out = sys.argv[1], sys.argv[2]
events = []
empty_cache, synchronize, predict_step = torch.mps.empty_cache, torch.mps.synchronize, ChainCompleteModule.predict_step

def step(module, batch, batch_idx):
    events.append("start")
    result = predict_step(module, batch, batch_idx)
    events.append("end")
    return result

result = {}
with mock.patch.object(torch.mps, "empty_cache", side_effect=lambda: events.append("empty_cache") or empty_cache()), \
        mock.patch.object(torch.mps, "synchronize", side_effect=lambda: events.append("synchronize") or synchronize()), \
        mock.patch.object(ChainCompleteModule, "predict_step", step):
    for accelerator, batch_size in (("cpu", 2), ("gpu", 1), ("gpu", 2)):
        events.clear()
        predictions = predict(src_stream=fasta, src_from=SrcEsmFrom.fasta, batch_size=batch_size,
                              accelerator=Accelerator(accelerator), devices=1)
        key = f"{accelerator}_b{batch_size}"
        result[key] = list(events)
        np.savez(f"{out}/{key}.npz", **{
            name: embedding.float().cpu().numpy()
            for embeddings, names in predictions for embedding, name in zip(embeddings, names)
        })
json.dump(result, open(f"{out}/events.json", "w"))
"""

    def test_predict_on_mps(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
        env["PYTHONPATH"] = os.pathsep.join(
            [str(Path(foldmatch.__file__).parents[1])] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        with tempfile.TemporaryDirectory() as out:
            run = subprocess.run(
                [sys.executable, "-c", self._CHILD, _FASTA, out], env=env, capture_output=True, text=True
            )
            self.assertEqual(run.returncode, 0, run.stderr[-3000:])
            events = json.load(open(f"{out}/events.json"))
            cpu = np.load(f"{out}/cpu_b2.npz")
            mps = {key: np.load(f"{out}/{key}.npz") for key in ("gpu_b1", "gpu_b2")}
            for name in cpu.files:
                for key in mps:
                    rel = np.abs(mps[key][name] - cpu[name]).max() / np.abs(cpu[name]).max()
                    self.assertLessEqual(rel, 1e-5, f"{key} {name}")
        per_batch = ["start", "synchronize", "empty_cache", "end"]
        self.assertEqual(events["gpu_b1"], per_batch * 2)
        self.assertEqual(events["gpu_b2"], per_batch)
        self.assertEqual(events["cpu_b2"], ["start", "end"])


if __name__ == "__main__":
    unittest.main()
