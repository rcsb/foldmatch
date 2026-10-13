import os
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from esm.sdk.api import ESMProtein, SamplingConfig

from foldmatch import FoldMatch
from foldmatch.dataset.esm_prot_from_fasta import parse_fasta
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.modules.esm_module import EsmModule
from foldmatch.types.api_types import SrcLocation
from foldmatch.utils.data import collate_seq_embeddings
from foldmatch.utils.esm import loaders
from foldmatch.utils.model import get_aggregator_model, get_residue_model


class GeomAttnOffTest(unittest.TestCase):
    """G13: sequence-only proteins run ESM3 with geometric attention off; embeddings are bit-identical."""

    __test_path = os.path.dirname(__file__)

    @classmethod
    def setUpClass(cls):
        cls.model = get_residue_model(torch.device("cpu"))
        cls.proteins = [
            (name, ESMProtein(sequence=sequence))
            for name, sequence in parse_fasta(f"{cls.__test_path}/resources/fasta/test_sequences.fasta")
        ]

    def _geom_blocks(self):
        return [i for i, block in enumerate(self.model.transformer.blocks) if block.use_geom_attn]

    def _geom_attn_spy(self):
        geom_attn = self.model.transformer.blocks[0].geom_attn
        return mock.patch.object(geom_attn, "forward", wraps=geom_attn.forward)

    def _with_geom_attn(self, esm_prot):
        # The pre-G13 call: geometric attention left on.
        return self.model.forward_and_sample(
            self.model.encode(esm_prot), SamplingConfig(return_per_residue_embeddings=True)
        ).per_residue_embedding

    def test_only_block_0_has_geometric_attention(self):
        self.assertEqual(self._geom_blocks(), [0])

    def test_sequence_only_embeddings_are_bit_identical(self):
        for name, esm_prot in self.proteins:
            with torch.inference_mode():
                with self._geom_attn_spy() as on_spy:
                    on = self._with_geom_attn(esm_prot)
                self.assertEqual(on_spy.call_count, 1)
                with self._geom_attn_spy() as off_spy:
                    off = loaders.per_residue_embedding(self.model, esm_prot)
                self.assertEqual(off_spy.call_count, 0)
            self.assertEqual(tuple(off.shape), (len(esm_prot) + 2, 1536))
            self.assertTrue(torch.equal(on, off), name)
            self.assertEqual(self._geom_blocks(), [0])

    def test_structure_input_keeps_geometric_attention(self):
        esm_prot, _name = next(iter(EsmProtFromStructure(
            src_stream=[("2uzi", f"{self.__test_path}/resources/pdb/2uzi.cif", "2uzi")],
            src_location=SrcLocation.stream
        )))
        self.assertIsNotNone(esm_prot.coordinates)
        with torch.inference_mode(), self._geom_attn_spy() as spy:
            expected = self._with_geom_attn(esm_prot)
            embedding = loaders.per_residue_embedding(self.model, esm_prot)
        self.assertEqual(spy.call_count, 2)
        self.assertTrue(torch.equal(expected, embedding))
        self.assertEqual(self._geom_blocks(), [0])

    def test_flag_restored_when_esm3_raises(self):
        with mock.patch.object(self.model, "forward_and_sample", side_effect=RuntimeError("out of memory")):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                loaders.per_residue_embedding(self.model, self.proteins[0][1])
        self.assertEqual(self._geom_blocks(), [0])

    def test_modules_run_sequence_only_proteins_without_geometric_attention(self):
        batch = [(esm_prot, name) for name, esm_prot in self.proteins]
        aggregator = get_aggregator_model(torch.device("cpu")).eval()
        chain_module = ChainCompleteModule(self.model, aggregator)
        with torch.inference_mode():
            with self._geom_attn_spy() as spy:
                chain_embeddings, chain_names = chain_module.predict_step(batch, 0)
                residue_embeddings, names = EsmModule(self.model).predict_step(batch, 0)
            self.assertEqual(spy.call_count, 0)
            expected = [self._with_geom_attn(esm_prot) for _name, esm_prot in self.proteins]
            expected_chain = aggregator(*collate_seq_embeddings(expected))
        self.assertEqual(names, tuple(name for name, _ in self.proteins))
        self.assertEqual(chain_names, names)
        for embedding, expected_embedding in zip(residue_embeddings, expected):
            self.assertTrue(torch.equal(expected_embedding, embedding))
        self.assertTrue(torch.equal(expected_chain, chain_embeddings))
        self.assertEqual(self._geom_blocks(), [0])

    def test_foldmatch_sequence_embedding_runs_without_geometric_attention(self):
        # FoldMatch.sequence_embedding, the Python API's sequence-only path, uses the same call.
        with mock.patch("foldmatch.foldmatch._load_res_model", return_value=self.model):
            api = FoldMatch()
            for name, esm_prot in self.proteins:
                with torch.inference_mode():
                    with self._geom_attn_spy() as spy:
                        embedding = api.sequence_embedding(esm_prot.sequence)
                    self.assertEqual(spy.call_count, 0)
                    expected = self._with_geom_attn(esm_prot)
                self.assertTrue(torch.equal(expected, embedding), name)
        self.assertEqual(self._geom_blocks(), [0])


class _FakeEsm3:
    """Stands in for ESM3 without weights: records the geometric-attention flag each forward sees.

    A forward of a protein named in ``hold`` signals ``inside`` and waits for ``release``.
    """

    def __init__(self, hold=()):
        self.block = SimpleNamespace(use_geom_attn=True)
        self.transformer = SimpleNamespace(blocks=[self.block])
        self.hold = hold
        self.inside, self.release = threading.Event(), threading.Event()
        self.seen = {}

    def encode(self, protein):
        return protein

    def forward_and_sample(self, protein, config):
        self.seen[protein.name] = self.block.use_geom_attn
        if protein.name in self.hold:
            self.inside.set()
            self.release.wait(10)
        return SimpleNamespace(per_residue_embedding=torch.zeros(3, 4))


class GeomAttnThreadSafetyTest(unittest.TestCase):
    """G13 switches a flag on the shared model: no other forward on it may run meanwhile. No weights."""

    SEQUENCE = SimpleNamespace(name="sequence", coordinates=None)
    STRUCTURE = SimpleNamespace(name="structure", coordinates=object())

    def _in_thread(self, model, protein):
        thread = threading.Thread(target=loaders.per_residue_embedding, args=(model, protein))
        thread.start()
        return thread

    def test_structure_forward_waits_for_a_sequence_forward_on_the_same_model(self):
        model = _FakeEsm3(hold=("sequence",))
        sequence = self._in_thread(model, self.SEQUENCE)
        self.assertTrue(model.inside.wait(10))
        structure = self._in_thread(model, self.STRUCTURE)
        structure.join(1.0)
        # Still waiting for the model while the sequence forward runs with the flag off.
        self.assertTrue(structure.is_alive())
        self.assertNotIn("structure", model.seen)
        model.release.set()
        sequence.join(10)
        structure.join(10)
        self.assertEqual(model.seen, {"sequence": False, "structure": True})
        self.assertTrue(model.block.use_geom_attn)

    def test_other_models_do_not_wait(self):
        held, other = _FakeEsm3(hold=("sequence",)), _FakeEsm3()
        sequence = self._in_thread(held, self.SEQUENCE)
        self.assertTrue(held.inside.wait(10))
        try:
            loaders.per_residue_embedding(other, self.STRUCTURE)
            loaders.per_residue_embedding(other, self.SEQUENCE)
            self.assertEqual(other.seen, {"structure": True, "sequence": False})
        finally:
            held.release.set()
            sequence.join(10)

    def test_foldmatch_structure_methods_take_the_same_path(self):
        """FoldMatch's structure methods go through per_residue_embedding, so they take the lock too."""
        test_path = os.path.dirname(__file__)
        cif = f"{test_path}/resources/pdb/2uzi.cif"
        fake = _FakeEsm3()
        fake.forward_and_sample = lambda protein, config: SimpleNamespace(
            per_residue_embedding=torch.zeros(len(protein) + 2, 4))
        spy = mock.Mock(wraps=loaders.per_residue_embedding)
        with mock.patch("foldmatch.foldmatch._load_res_model", return_value=fake), \
                mock.patch("foldmatch.foldmatch._load_model", return_value=lambda x: x), \
                mock.patch("foldmatch.foldmatch.per_residue_embedding", spy):
            api = FoldMatch()
            calls = {}
            for method in (api.residue_embedding, api.residue_embedding_by_chain,
                           api.residue_embedding_by_assembly):
                spy.reset_mock()
                method(cif)
                calls[method.__name__] = spy.call_count
                for call in spy.call_args_list:
                    self.assertIs(call.args[0], fake)
                    self.assertIsNotNone(call.args[1].coordinates)
        self.assertTrue(all(count > 0 for count in calls.values()), calls)


if __name__ == "__main__":
    unittest.main()
