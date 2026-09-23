"""The build-time 3Di channel: predict_step -> writer -> Parquet.

The failure this file guards is not a crash. It is that adding the head changes the
*embedding* the aggregator sees - the residue tensor carries BOS and EOS, the head wants
them stripped, and a slice applied one line too early silently shifts every stage-1
embedding in the database while every test still passes.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from foldmatch.model.three_di_head import TransformerThreeDiHead
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.search import scoring_system as ss
from foldmatch.writer.batch_writer import ParquetBatchWriter

LENGTHS = [37, 52, 19]


class FakeEsm:
    """Stands in for ESM3: returns a deterministic (L+2, 1536) tensor, BOS/EOS included."""

    def encode(self, protein):
        return protein

    def forward_and_sample(self, protein, config):
        length = protein["length"]
        generator = torch.Generator().manual_seed(length)
        embedding = torch.randn(length + 2, 1536, generator=generator)
        # Make the terminal rows unmistakable, so a missing slice is visible.
        embedding[0] = 100.0
        embedding[-1] = -100.0
        return type("Out", (), {"per_residue_embedding": embedding})()


class FakeAggregator(torch.nn.Module):
    """Records what it was handed, and returns a cheap summary of it."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, batch_embedding, batch_mask):
        self.seen.append(batch_embedding.detach().clone())
        return batch_embedding.sum(dim=1)[:, :8]


def build_module(with_head):
    head = scoring = None
    if with_head:
        system = ss.get("3di-v1-coded")
        scoring = ss.prepare(system)
        head = TransformerThreeDiHead(input_features=1536, hidden=256, nhead=8,
                                      num_layers=2, dim_feedforward=1024, dropout=0.0)
        head.eval()
    aggregator = FakeAggregator()
    module = ChainCompleteModule(res_model=FakeEsm(), aggregator_model=aggregator,
                                 three_di_head=head, three_di_scoring=scoring)
    batch = [({"length": n}, f"p{n}") for n in LENGTHS]
    return module, aggregator, batch


class AggregatorInputUnchangedTest(unittest.TestCase):
    """The headline invariant: enabling the head must not move a single stage-1 number."""

    def test_embedding_is_byte_identical_with_and_without_the_head(self):
        plain_module, plain_agg, batch = build_module(with_head=False)
        plain = plain_module.predict_step(batch, 0)

        head_module, head_agg, batch = build_module(with_head=True)
        with_head = head_module.predict_step(batch, 0)

        self.assertTrue(torch.equal(plain_agg.seen[0], head_agg.seen[0]),
                        "the aggregator received a different tensor once the head was "
                        "added - the [1:-1] slice has leaked out of the head branch")
        self.assertTrue(torch.equal(plain[0], with_head[0]))
        self.assertEqual(plain[1], with_head[1])

    def test_terminal_rows_still_reach_the_aggregator(self):
        _, aggregator, batch = build_module(with_head=True)
        module, aggregator, batch = build_module(with_head=True)
        module.predict_step(batch, 0)
        seen = aggregator.seen[0]
        self.assertTrue(torch.all(seen[0, 0] == 100.0), "BOS row was stripped")
        self.assertTrue(torch.all(seen[0, max(LENGTHS) + 1] == -100.0)
                        or torch.all(seen[0, LENGTHS[0] + 1] == -100.0),
                        "EOS row was stripped")


class ThreeDiOutputTest(unittest.TestCase):
    def test_codes_are_one_byte_per_residue(self):
        module, _, batch = build_module(with_head=True)
        aggregated, names, codes = module.predict_step(batch, 0)
        self.assertEqual(len(codes), len(LENGTHS))
        for length, code in zip(LENGTHS, codes):
            self.assertEqual(len(code), length,
                             "one code per residue, BOS/EOS excluded")
            self.assertEqual(code.dtype, np.uint8)
            self.assertTrue((code < 240).all(), "codes must index the 240-row kernel")

    def test_head_reads_the_sliced_tensor(self):
        """Feeding the head BOS/EOS would change its output; prove the slice is applied."""
        module, _, batch = build_module(with_head=True)
        _, _, sliced = module.predict_step(batch, 0)
        embedding = FakeEsm().forward_and_sample({"length": LENGTHS[0]}, None).per_residue_embedding
        with torch.no_grad():
            unsliced_logits = module.three_di_head(embedding.unsqueeze(0).float())[0]
        unsliced = module.three_di_scoring.encode_target(
            torch.softmax(unsliced_logits, -1).double().numpy())
        self.assertNotEqual(len(sliced[0]), len(unsliced))

    def test_two_element_return_without_a_head(self):
        module, _, batch = build_module(with_head=False)
        self.assertEqual(len(module.predict_step(batch, 0)), 2)

    def test_head_without_scoring_is_refused(self):
        with self.assertRaisesRegex(ValueError, "needs its scoring system"):
            ChainCompleteModule(res_model=FakeEsm(), aggregator_model=FakeAggregator(),
                                three_di_head=object())


class ParquetSchemaTest(unittest.TestCase):
    """An existing database's shard must keep exactly the columns it has today."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def _write(self, prediction, with_three_di):
        writer = ParquetBatchWriter(str(self.dir), f"shard-{with_three_di}",
                                    with_three_di=with_three_di)
        trainer = type("T", (), {"global_rank": 0})()
        writer.write_on_batch_end(trainer, None, prediction, None, None, 0, 0)
        writer.on_predict_end(trainer, None)
        return pq.read_table(self.dir / f"shard-{with_three_di}-0.parquet")

    def test_schema_unchanged_when_disabled(self):
        module, _, batch = build_module(with_head=False)
        table = self._write(module.predict_step(batch, 0), with_three_di=False)
        self.assertEqual(table.column_names, ["id", "embedding"])

    def test_three_element_prediction_into_a_plain_writer_drops_the_codes(self):
        """A 3-tuple must not corrupt a two-column shard."""
        module, _, batch = build_module(with_head=True)
        table = self._write(module.predict_step(batch, 0), with_three_di=False)
        self.assertEqual(table.column_names, ["id", "embedding"])

    def test_codes_round_trip_through_parquet(self):
        module, _, batch = build_module(with_head=True)
        prediction = module.predict_step(batch, 0)
        table = self._write(prediction, with_three_di=True)
        self.assertEqual(table.column_names, ["id", "embedding", "three_di"])
        stored = table.column("three_di").to_pylist()
        for original, blob in zip(prediction[2], stored):
            np.testing.assert_array_equal(np.frombuffer(blob, dtype=np.uint8), original)
        self.assertEqual(table.column("id").to_pylist(), list(prediction[1]))

    def test_enabled_writer_refuses_a_prediction_without_codes(self):
        module, _, batch = build_module(with_head=False)
        with self.assertRaisesRegex(ValueError, "returned no 3Di codes"):
            self._write(module.predict_step(batch, 0), with_three_di=True)


class SequenceOnlyTest(unittest.TestCase):
    """Predicted 3Di is for the sequence case, and only that case.

    A structure build has coordinates, so its 3Di can be read from them exactly - and
    measurably should be: predicted targets cost ~0.02-0.03 of detection at a fixed
    E-value against exact ones. Predicting there would spend a forward pass to make the
    database worse, so it is refused rather than left as a footgun.
    """

    def _predict(self, **kwargs):
        from foldmatch.inference import chain_complete_inference
        from foldmatch.types.api_types import OutFormat, SrcEsmFrom
        defaults = dict(src_stream=Path("unused.fasta"), src_from=SrcEsmFrom.fasta,
                        out_format=OutFormat.parquet, out_folder=Path("unused"),
                        three_di="3di-v1-coded")
        defaults.update(kwargs)
        return chain_complete_inference.predict(**defaults)

    def test_structure_source_is_refused(self):
        from foldmatch.types.api_types import SrcEsmFrom
        with self.assertRaisesRegex(ValueError, "sequence-only databases"):
            self._predict(src_from=SrcEsmFrom.structure)

    def test_non_parquet_output_is_refused(self):
        from foldmatch.types.api_types import OutFormat
        with self.assertRaisesRegex(ValueError, "only written to Parquet"):
            self._predict(out_format=OutFormat.csv)

    def test_non_coded_scoring_system_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not encode predicted targets"):
            self._predict(three_di="3di-v1-exact")

    def test_fasta_build_threads_the_option_through(self):
        """The sequence build must be able to ask for it at all."""
        import inspect
        from foldmatch.search.embedding_computer import EmbeddingComputer
        self.assertIn("three_di",
                      inspect.signature(EmbeddingComputer.compute_from_fasta).parameters)
        source = inspect.getsource(EmbeddingComputer.compute_from_fasta)
        self.assertIn("three_di=three_di", source,
                      "compute_from_fasta accepts three_di but does not pass it on")


if __name__ == "__main__":
    unittest.main()
