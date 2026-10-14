"""P5.1-P5.4: the residue-code channel of the ESM3 pass (plan §8 P5).

Most tests need no ESM3: a fake ``per_residue_embedding`` hands the module real ESM3 rows from the fixtures (F2c, the
FoldMatch CPU fp32 rows of ``test_sequences.fasta``; F2b, structure-derived rows used for arithmetic only), so the
module hook, the plumbing, the writer and the readers run with the real esm-codes-v1 encoders on known inputs.

The real encoders come from the scoring-system registry bundle (``esm_codes_bundle``): skipped only when offline and
not cached.
"""
import dataclasses
import functools
import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import esm_codes_bundle
import foldmatch
from foldmatch.model.residue_embedding_aggregator import ResidueEmbeddingAggregator
from foldmatch.modules import chain_complete_module
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.search import scoring_system as ss
from foldmatch.search import residue_shards as rs
from foldmatch.types.api_types import Accelerator, OutFormat, SrcEsmFrom
from foldmatch.utils.data import collate_seq_embeddings
from foldmatch.utils.model import get_aggregator_model, release_mps_cache

_TEST_PATH = Path(__file__).parent
_FIXTURES = _TEST_PATH / "resources" / "esm_codes"
_EMBEDDINGS = _TEST_PATH / "resources" / "embeddings"
_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
_SRC = str(Path(foldmatch.__file__).resolve().parent.parent)
_LAUNCH_ENV = ("FOLDMATCH_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_RESTART_COUNT", "SLURM_PROCID",
               "SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "RANK", "LOCAL_RANK", "NODE_RANK", "WORLD_SIZE")

SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)
# Another encoding (a different target encoder): what a stale shard of another system holds.
OTHER_ENCODING = dataclasses.replace(SYSTEM, name="esm-codes-test-other", target_encoder_sha256="0" * 64)
# The same encoding, other statistics: may reuse the residue codes, never the query encodings.
OTHER_STATISTICS = dataclasses.replace(SYSTEM, name="esm-codes-test-stats", calibration_version="test")

#: A4_THRESHOLDS.json (sha256 7eabf36e...), gate A4-xcpu: CPU Mac vs CPU Linux code bits, also the F2c threshold.
A4_XCPU_BITS = 0.999
#: A4_THRESHOLDS.json: code bits on every device class.
A4_DEVICE_BITS = 0.990
_TRAINER = SimpleNamespace(global_rank=0)


@contextmanager
def launch_env(**values):
    """Exactly ``values`` among the launch variables; the rest unset (only those variables are restored)."""
    saved = {name: os.environ.get(name) for name in _LAUNCH_ENV}
    try:
        for name in _LAUNCH_ENV:
            os.environ.pop(name, None)
        os.environ.update({k: str(v) for k, v in values.items()})
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# =====================================================================================================================
# fixtures
# =====================================================================================================================
@functools.lru_cache(maxsize=None)
def _f2c():
    """F2c: ``[(id, sequence, ESM3 rows [L + 2, 1536] fp32)]``, reference codes ``[303, 8]``, q ``[303, 64]``,
    offsets. The rows are stored fp16; S2 starts with fp16(x), so as fp32 they encode to the same codes."""
    z = np.load(_FIXTURES / "fasta_rows_codes_v1.npz", allow_pickle=False)
    ro = z["rows_offsets"]
    items = [(str(z["ids"][k]), str(z["sequences"][k]), torch.from_numpy(z["rows"][ro[k]:ro[k + 1]].astype(np.float32)))
             for k in range(len(ro) - 1)]
    return items, z["codes"], z["q"], z["offsets"]


@functools.lru_cache(maxsize=None)
def _f2b():
    """F2b (arithmetic only): structure-derived ESM3 rows ``[R, 1536]`` fp32 of two chains, standing in for an ESM3
    output with BOS and EOS, so the hook encodes rows ``1..R-2``. The reference encoded every row, and an inner slice
    encodes to the same values (FIXTURES.json ``inner_slice``)."""
    z = np.load(_FIXTURES / "ood_rows_codes_v1.npz", allow_pickle=False)
    names = [str(n) for n in z["ids"]]
    rows = [torch.load(_EMBEDDINGS / f"{name}.pt", map_location="cpu", weights_only=True) for name in names]
    # Any sequence of the right length: the codes depend on the rows only.
    items = [(f"f2b_{name}", "A" * (r.shape[0] - 2), r) for name, r in zip(names, rows)]
    return items, z["codes"], z["q"], z["offsets"]


@functools.lru_cache(maxsize=None)
def _prepared():
    esm_codes_bundle.registry_files()  # unittest.SkipTest only when offline and not cached
    return ss.prepare(SYSTEM)


@functools.lru_cache(maxsize=None)
def _head():
    return _prepared().load_head()


@functools.lru_cache(maxsize=None)
def _aggregator():
    return get_aggregator_model(torch.device("cpu")).eval()


def _bits_equal(a, b):
    return int((np.unpackbits(a, axis=1) == np.unpackbits(b, axis=1)).sum()), int(a.size * 8)


def _row_ulps(q, ref):
    """max |q - ref| in fp16 ulps of each reference row's largest |value| (the P2 measure)."""
    q32, r32 = q.astype(np.float32), ref.astype(np.float32)
    scale = np.spacing(np.abs(ref).max(axis=1).astype(np.float16)).astype(np.float32)[:, None]
    return float((np.abs(q32 - r32) / scale).max())


def _bitwise(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy().view(np.uint8)


def _sha1(sequence: str) -> bytes:
    return hashlib.sha1(sequence.encode("ascii")).digest()


def _fake_esm3(items):
    """A per_residue_embedding stand-in: the rows of the item whose sequence the protein has."""
    rows = {sequence: r for _, sequence, r in items}

    def per_residue_embedding(model, esm_prot):
        return rows[esm_prot.sequence].clone()
    return per_residue_embedding


def _batch(items):
    return [(SimpleNamespace(sequence=sequence, coordinates=None), name) for name, sequence, _ in items]


def _legacy_predict_step(module, prot_batch, batch_idx):
    # ChainCompleteModule.predict_step before P5.1 (e8b389c), verbatim but for the module-level name lookup, so the
    # same fake per_residue_embedding serves both.
    prot_embeddings = []
    prot_names = []
    for esm_prot, name in prot_batch:
        embeddings = chain_complete_module.per_residue_embedding(module.esm3, esm_prot)
        prot_embeddings.append(embeddings)
        prot_names.append(name)
    res_batch_embedding, res_batch_mask = collate_seq_embeddings(prot_embeddings)
    chain_embeddings = module.aggregator(res_batch_embedding, res_batch_mask)
    release_mps_cache(module.device)
    return chain_embeddings, tuple(prot_names)


class _RealHead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.head = _head()
        cls.aggregator = _aggregator()


# =====================================================================================================================
# P5.1 module hook
# =====================================================================================================================
class ModuleHookTest(_RealHead):
    """The hook: aggregator input untouched, payloads from rows 1..L, no raise in predict_step."""

    def _module(self, residue_output):
        return ChainCompleteModule(res_model=torch.nn.Identity(), aggregator_model=self.aggregator,
                                   residue_head=self.head if residue_output else None, residue_output=residue_output)

    def _run(self, residue_output, items, batch_size=2, esm3=None):
        """``(outputs per batch, aggregator inputs per batch)``."""
        module = self._module(residue_output)
        seen = []
        forward = ResidueEmbeddingAggregator.forward

        def spy(aggregator, x, x_mask=None):
            seen.append((x.clone(), None if x_mask is None else x_mask.clone()))
            return forward(aggregator, x, x_mask)

        outputs = []
        with mock.patch.object(chain_complete_module, "per_residue_embedding", esm3 or _fake_esm3(items)), \
                mock.patch.object(ResidueEmbeddingAggregator, "forward", spy), torch.inference_mode():
            for start in range(0, len(items), batch_size):
                outputs.append(module.predict_step(_batch(items[start:start + batch_size]), start // batch_size))
        return outputs, seen, module

    def test_a_module_without_a_head_returns_the_legacy_2_tuple(self):
        items = _f2c()[0] + _f2b()[0]
        outputs, _, module = self._run(None, items)
        with mock.patch.object(chain_complete_module, "per_residue_embedding", _fake_esm3(items)), \
                torch.inference_mode():
            legacy = [_legacy_predict_step(module, _batch(items[s:s + 2]), s // 2) for s in range(0, len(items), 2)]
        for got, want in zip(outputs, legacy):
            self.assertEqual(len(got), 2)
            self.assertEqual(got[1], want[1])
            np.testing.assert_array_equal(_bitwise(got[0]), _bitwise(want[0]))

    def test_aggregator_input_and_chain_embeddings_are_identical_with_and_without_codes(self):
        # Batches of two proteins of different lengths, so the aggregator input is padded and masked.
        items = _f2c()[0] + _f2b()[0]
        plain, plain_seen, _ = self._run(None, items)
        self.assertTrue(plain_seen[0][1].any())
        for residue_output in rs.RESIDUE_OUTPUTS:
            with self.subTest(residue_output=residue_output):
                outputs, seen, _ = self._run(residue_output, items)
                self.assertEqual(len(seen), len(plain_seen))
                for (x, m), (x0, m0) in zip(seen, plain_seen):
                    np.testing.assert_array_equal(_bitwise(x), _bitwise(x0))
                    np.testing.assert_array_equal(m.numpy(), m0.numpy())
                for got, want in zip(outputs, plain):
                    self.assertEqual(len(got), 4)
                    self.assertEqual(got[1], want[1])
                    np.testing.assert_array_equal(_bitwise(got[0]), _bitwise(want[0]))

    def test_f2c_rows_give_the_reference_codes(self):
        items, codes_ref, q_ref, offsets = _f2c()
        codes = b"".join(p for out in self._run(rs.RESIDUE_CODES, items)[0] for p in out[2])
        np.testing.assert_array_equal(np.frombuffer(codes, dtype=np.uint8).reshape(-1, 8), codes_ref)
        outputs = self._run(rs.QUERY_ENCODING, items)[0]
        q = np.frombuffer(b"".join(p for out in outputs for p in out[2]), dtype="<f2").reshape(-1, 64)
        self.assertLessEqual(_row_ulps(q, q_ref), 1.0)
        self.assertEqual([d for out in outputs for d in out[3]], [_sha1(s) for _, s, _ in items])

    def test_f2b_arithmetic(self):
        """F2b without ESM3: the payloads are the reference encodings of rows 1..R-2, packed as the spec says (S4:
        8 bytes per residue, MSB first; S5: 64 little-endian fp16 per residue)."""
        items, codes_ref, q_ref, offsets = _f2b()
        codes_out = self._run(rs.RESIDUE_CODES, items, batch_size=1)[0]
        q_out = self._run(rs.QUERY_ENCODING, items, batch_size=1)[0]
        for k, (_, sequence, rows) in enumerate(items):
            inner = slice(offsets[k] + 1, offsets[k + 1] - 1)
            payload = codes_out[k][2][0]
            self.assertEqual(len(payload), 8 * len(sequence))
            self.assertEqual(payload, codes_ref[inner].tobytes())
            q = np.frombuffer(q_out[k][2][0], dtype="<f2").reshape(-1, 64)
            self.assertLessEqual(_row_ulps(q, q_ref[inner]), 1.0)
            # bit for bit what the head gives for the same rows
            self.assertEqual(q.tobytes(), self.head.encode_query(rows[1:-1]).numpy().astype("<f2").tobytes())
            self.assertEqual(codes_out[k][3][0], _sha1(sequence))

    def test_a_length_mismatch_gives_an_empty_payload_and_no_raise(self):
        items = _f2c()[0]
        esm3 = _fake_esm3(items)

        def one_row_short(model, esm_prot):
            rows = esm3(model, esm_prot)
            return rows[:-1] if esm_prot.sequence == items[1][1] else rows

        outputs, _, module = self._run(rs.RESIDUE_CODES, items, esm3=one_row_short)
        (embeddings, names, payloads, digests), = outputs
        self.assertEqual(names, (items[0][0], items[1][0]))
        self.assertEqual(len(payloads[0]), 8 * len(items[0][1]))
        self.assertEqual(payloads[1], b"")
        self.assertEqual(digests[1], _sha1(items[1][1]))
        self.assertEqual(tuple(embeddings.shape), (2, 1536))
        self.assertEqual(module.residue_failures, 1)

    def test_an_encoder_failure_gives_an_empty_payload_and_no_raise(self):
        items = _f2c()[0]
        with mock.patch.object(type(self.head), "encode_target", side_effect=RuntimeError("boom")):
            outputs, _, module = self._run(rs.RESIDUE_CODES, items)
        self.assertEqual(outputs[0][2], (b"", b""))
        self.assertEqual(module.residue_failures, 2)

    def test_a_non_ascii_sequence_gives_an_empty_payload(self):
        name, sequence, rows = _f2c()[0][1]
        bad = sequence[:-1] + "É"
        outputs, _, _ = self._run(rs.RESIDUE_CODES, [(name, bad, rows)])
        self.assertEqual(outputs[0][2], (b"",))
        self.assertEqual(len(outputs[0][3][0]), 20)

    def test_seq_sha1_is_the_sha1_of_the_normalised_ascii_sequence(self):
        name, sequence, rows = _f2c()[0][1]
        outputs, _, _ = self._run(rs.RESIDUE_CODES, [(name, sequence, rows)])
        self.assertEqual(outputs[0][3][0], hashlib.sha1(sequence.encode("ascii")).digest())

    def test_constructor_refusals(self):
        with self.assertRaises(ValueError):
            ChainCompleteModule(torch.nn.Identity(), self.aggregator, residue_head=self.head,
                                residue_output="codes")
        with self.assertRaises(ValueError):
            ChainCompleteModule(torch.nn.Identity(), self.aggregator, residue_head=self.head)
        with self.assertRaises(ValueError):
            ChainCompleteModule(torch.nn.Identity(), self.aggregator, residue_output=rs.RESIDUE_CODES)

    def test_the_head_is_a_submodule(self):
        """Lightning moves the head with the module (CPU, CUDA or MPS, D18)."""
        module = self._module(rs.RESIDUE_CODES)
        self.assertIs(dict(module.named_children())["residue_head"], self.head)
        self.assertNotIn("residue_head", dict(self._module(None).named_children()))


# =====================================================================================================================
# P5.2 inference plumbing
# =====================================================================================================================
def _esm3_spy():
    """Patch the ESM3 loader of the pass; any call means ESM3 was reached."""
    return mock.patch("foldmatch.inference.chain_complete_inference.get_residue_model",
                      mock.Mock(side_effect=AssertionError("ESM3 was loaded")))


class InferenceRefusalTest(unittest.TestCase):
    """Every refusal of a residue pass comes before ESM3 loads."""

    def _refused(self, error, **kwargs):
        from foldmatch.inference.chain_complete_inference import predict
        kwargs.setdefault("src_stream", _FASTA)
        kwargs.setdefault("src_from", SrcEsmFrom.fasta)
        with _esm3_spy() as esm3, self.assertRaises(error) as ctx:
            predict(accelerator=Accelerator.cpu, devices=1, **kwargs)
        esm3.assert_not_called()
        return str(ctx.exception)

    def test_structure_sources_are_refused(self):
        message = self._refused(ValueError, src_stream=[("x", str(_EMBEDDINGS / "1acb.A.pt"), "x")],
                                src_from=SrcEsmFrom.structure, residue_output=rs.RESIDUE_CODES)
        self.assertIn("sequence (FASTA) input", message)

    def test_non_parquet_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for out_format in (OutFormat.csv, OutFormat.json, OutFormat.pt):
                with self.subTest(out_format=out_format):
                    message = self._refused(ValueError, residue_output=rs.RESIDUE_CODES, out_folder=tmp,
                                            out_format=out_format)
                    self.assertIn("Parquet", message)
            self.assertEqual(os.listdir(tmp), [])

    def test_a_system_other_than_the_registry_entry_is_refused(self):
        for system in (dataclasses.replace(SYSTEM, bundle_revision="f" * 40), OTHER_ENCODING, "esm-codes-v0"):
            with self.subTest(system=getattr(system, "name", system)):
                self._refused(ss.UnknownScoringSystemError, residue_output=rs.RESIDUE_CODES, scoring_system=system)

    def test_a_query_pass_needs_the_database_system(self):
        self.assertIn("database", self._refused(ValueError, residue_output=rs.QUERY_ENCODING))

    def test_unknown_residue_output_and_a_system_without_one_are_refused(self):
        self._refused(ValueError, residue_output="codes")
        self._refused(ValueError, scoring_system=SYSTEM)

    def test_a_bundle_problem_is_refused(self):
        with mock.patch.object(ss, "prepare", side_effect=ss.BundleUnavailableError("not cached")) as prepare:
            self._refused(ss.BundleUnavailableError, residue_output=rs.RESIDUE_CODES)
        prepare.assert_called_once()

    def test_fasta_the_method_cannot_take_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ss, "prepare") as prepare:
            prepare.return_value.system = SYSTEM
            for text, what in ((">a\nACDE\n>a\nACDE\n", "repeat"), (">a\nAC<mask>DE\n", "'<'"),
                               (">a\nACÉ\n", "non-ASCII")):
                with self.subTest(what=what):
                    path = Path(tmp) / "bad.fasta"
                    path.write_text(text, encoding="utf-8")
                    self.assertIn(what, self._refused(ValueError, src_stream=path, residue_output=rs.RESIDUE_CODES))
            prepare.return_value.load_head.assert_not_called()

    def test_a_plain_pass_is_not_strict(self):
        """Without a residue output the pre-flight problems are only warnings, as before (the spy is reached)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup.fasta"
            path.write_text(">a\nACDE\n>a\nACDE\n", encoding="utf-8")
            from foldmatch.inference.chain_complete_inference import predict
            with _esm3_spy() as esm3, self.assertRaisesRegex(AssertionError, "ESM3 was loaded"):
                predict(src_stream=path, src_from=SrcEsmFrom.fasta, accelerator=Accelerator.cpu, devices=1)
            esm3.assert_called_once()


class BundleFetchRuleTest(unittest.TestCase):
    """prepare(system, local_files_only=True) on every rank but the one process that starts the run (P4 outcome)."""

    def test_rule(self):
        from foldmatch.inference.chain_complete_inference import bundle_fetch_allowed
        cases = [
            ({}, True),
            ({"LOCAL_RANK": 1}, False),                                   # a child Lightning started
            ({"RANK": 0, "WORLD_SIZE": 1}, True),
            ({"RANK": 0, "WORLD_SIZE": 2}, False),                        # torchrun: all ranks start together
            ({"SLURM_JOB_ID": 5, "SLURM_STEP_ID": 0, "SLURM_STEP_NUM_TASKS": 2, "SLURM_PROCID": 0}, False),
            ({"SLURM_JOB_ID": 5, "SLURM_STEP_ID": 0, "SLURM_STEP_NUM_TASKS": 1, "SLURM_NTASKS": 4}, True),
            ({"SLURM_JOB_ID": 5, "SLURM_NTASKS": 4}, True),               # the batch step is one process
        ]
        for env, allowed in cases:
            with self.subTest(env=env), launch_env(**env):
                self.assertIs(bundle_fetch_allowed(), allowed)
                self.assertFalse(bundle_fetch_allowed(num_nodes=2))     # the other nodes start on their own

    def test_prepare_scoring_system(self):
        from foldmatch.inference.chain_complete_inference import prepare_scoring_system
        with mock.patch.object(ss, "prepare") as prepare:
            with launch_env():
                prepare_scoring_system()
            prepare.assert_called_with(SYSTEM, local_files_only=False)
            with launch_env(LOCAL_RANK=1):
                prepare_scoring_system(SYSTEM.name)
            prepare.assert_called_with(SYSTEM, local_files_only=True)
            prepare_scoring_system(SYSTEM, local_files_only=True)
            prepare.assert_called_with(SYSTEM, local_files_only=True)
        prepared = ss.PreparedSystem(system=SYSTEM, files={}, bundle={}, encoders_file="e", scoring_file="s")
        with mock.patch.object(ss, "prepare") as prepare:
            self.assertIs(prepare_scoring_system(prepared, residue_output=rs.QUERY_ENCODING), prepared)
        prepare.assert_not_called()
        foreign = ss.PreparedSystem(system=OTHER_STATISTICS, files={}, bundle={}, encoders_file="e", scoring_file="s")
        with self.assertRaises(ss.UnknownScoringSystemError):
            prepare_scoring_system(foreign)


class _FakeEsm3Pass(_RealHead):
    """predict() through Lightning on the CPU with ESM3 replaced by the F2c rows of the fixture FASTA."""

    @contextmanager
    def fake_models(self):
        with mock.patch("foldmatch.inference.chain_complete_inference.get_residue_model",
                        return_value=torch.nn.Identity()), \
                mock.patch("foldmatch.inference.chain_complete_inference.get_aggregator_model",
                           return_value=self.aggregator), \
                mock.patch.object(chain_complete_module, "per_residue_embedding", _fake_esm3(_f2c()[0])):
            yield

    def predict(self, **kwargs):
        from foldmatch.inference.chain_complete_inference import predict
        with self.fake_models(), launch_env():
            return predict(src_stream=_FASTA, src_from=SrcEsmFrom.fasta, accelerator=Accelerator.cpu, devices=1,
                           **kwargs)


class InferencePlumbingTest(_FakeEsm3Pass):

    def test_callers_without_the_keyword_get_the_legacy_module_and_outputs(self):
        made = []
        real = ChainCompleteModule

        def construct(*args, **kwargs):
            made.append(kwargs)
            return real(*args, **kwargs)

        with mock.patch("foldmatch.inference.chain_complete_inference.ChainCompleteModule", side_effect=construct):
            predictions = self.predict(batch_size=2)
        self.assertEqual(made, [{"res_model": mock.ANY, "aggregator_model": self.aggregator}])
        module = real(torch.nn.Identity(), self.aggregator)
        items = _f2c()[0]
        with mock.patch.object(chain_complete_module, "per_residue_embedding", _fake_esm3(items)), \
                torch.inference_mode():
            legacy = _legacy_predict_step(module, _batch(items), 0)
        (embeddings, names), = predictions
        self.assertEqual(names, legacy[1])
        np.testing.assert_array_equal(_bitwise(embeddings), _bitwise(legacy[0]))

    def test_residue_predictions(self):
        items, codes_ref, q_ref, offsets = _f2c()
        plain = self.predict(batch_size=2)
        for residue_output in rs.RESIDUE_OUTPUTS:
            with self.subTest(residue_output=residue_output):
                kwargs = {"scoring_system": SYSTEM} if residue_output == rs.QUERY_ENCODING else {}
                (embeddings, names, payloads, digests), = self.predict(batch_size=2, residue_output=residue_output,
                                                                       **kwargs)
                np.testing.assert_array_equal(_bitwise(embeddings), _bitwise(plain[0][0]))
                self.assertEqual(names, plain[0][1])
                self.assertEqual(list(digests), [_sha1(s) for _, s, _ in items])
                data = b"".join(payloads)
                if residue_output == rs.RESIDUE_CODES:
                    np.testing.assert_array_equal(np.frombuffer(data, dtype=np.uint8).reshape(-1, 8), codes_ref)
                else:
                    q = np.frombuffer(data, dtype="<f2").reshape(-1, 64)
                    self.assertLessEqual(_row_ulps(q, q_ref), 1.0)


# =====================================================================================================================
# P5.3 writer
# =====================================================================================================================
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from lightning.pytorch.callbacks import BasePredictionWriter  # noqa: E402

from foldmatch.writer import batch_writer  # noqa: E402
from foldmatch.writer.batch_writer import ParquetBatchWriter  # noqa: E402


class _LegacyParquetBatchWriter(BasePredictionWriter):
    """writer/batch_writer.py ParquetBatchWriter at e8b389c (before P5.3), verbatim."""

    def __init__(self, output_path, df_id, write_interval="batch", buffer_size=16384):
        super().__init__(write_interval)
        self.out_path = output_path
        self.df_id = df_id
        self.buffer_size = buffer_size
        self._writer = None
        self._id_buffer = []
        self._emb_buffer = []
        self._schema = pa.schema([
            ('id', pa.string()),
            ('embedding', pa.list_(pa.float32()))
        ])

    def _get_writer(self, rank):
        if self._writer is None:
            path = f"{self.out_path}/{self.df_id}-{rank}.parquet"
            self._writer = pq.ParquetWriter(path, self._schema, compression='snappy')
        return self._writer

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if prediction is None:
            return
        embeddings, dom_ids = prediction[0], prediction[1]
        self._id_buffer.extend(dom_ids)
        self._emb_buffer.extend(
            emb.detach().to('cpu', torch.float32).numpy().flatten()
            for emb in embeddings
        )
        if len(self._id_buffer) >= self.buffer_size:
            self._flush(trainer.global_rank)

    def _flush(self, rank):
        if not self._id_buffer:
            return
        table = pa.table(
            {
                'id': self._id_buffer,
                'embedding': pa.array(self._emb_buffer, type=pa.list_(pa.float32())),
            },
            schema=self._schema,
        )
        self._get_writer(rank).write_table(table)
        self._id_buffer.clear()
        self._emb_buffer.clear()

    def on_predict_end(self, trainer, pl_module):
        self._flush(trainer.global_rank)
        if self._writer is not None:
            self._writer.close()


def _rows(n, seed=0, residues=(3, 0, 5), dim=4):
    """``n`` synthetic rows: ``(id, embedding, codes payload, query payload, seq_sha1)``; the second has an empty
    payload (a length mismatch)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        length = residues[k % len(residues)]
        emb = torch.from_numpy(rng.normal(size=dim).astype(np.float32))
        codes = rng.integers(0, 256, size=(length, 8), dtype=np.uint8).tobytes()
        q = rng.normal(size=(length, 64)).astype("<f2").tobytes()
        out.append((f"s{seed}_{k}", emb, codes, q, hashlib.sha1(f"{seed}/{k}".encode()).digest()))
    return out


def _prediction(rows, residue_output=None):
    ids = tuple(r[0] for r in rows)
    embeddings = torch.stack([r[1] for r in rows])
    if residue_output is None:
        return embeddings, ids
    payloads = tuple(r[2] if residue_output == rs.RESIDUE_CODES else r[3] for r in rows)
    return embeddings, ids, payloads, tuple(r[4] for r in rows)


def _write_shard(folder, name, rows, residue_output=None, system=SYSTEM, rank=0, batch=2, **writer_kwargs):
    writer = ParquetBatchWriter(folder, name, residue_output=residue_output,
                                system=system if residue_output else None, **writer_kwargs)
    trainer = SimpleNamespace(global_rank=rank)
    module = SimpleNamespace(residue_output=residue_output)
    writer.on_predict_start(trainer, module)
    for start in range(0, len(rows), batch):
        writer.write_on_batch_end(trainer, module, _prediction(rows[start:start + batch], residue_output),
                                  None, None, 0, 0)
    writer.on_predict_end(trainer, module)
    return Path(folder) / f"{name}-{rank}.parquet"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pass_record(path):
    """The ``foldmatch.pass`` record in a shard's footer."""
    return json.loads(pq.read_schema(path).metadata[rs.META_PASS])


class WriterTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_plain_writer_keeps_the_legacy_schema_and_bytes(self):
        writer = ParquetBatchWriter(self.tmp, "x")
        self.assertEqual(writer.schema, pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.float32()))]))
        self.assertIsNone(writer.schema.metadata)
        self.assertIsNone(writer.byte_budget)
        rows = _rows(7)
        for buffer_size in (16384, 3):
            for size in (2, 4):
                with self.subTest(buffer_size=buffer_size, tuple_size=size):
                    legacy = _LegacyParquetBatchWriter(self.tmp, f"legacy{buffer_size}", buffer_size=buffer_size)
                    for start in range(0, 7, 2):
                        legacy.write_on_batch_end(_TRAINER, None, _prediction(rows[start:start + 2]), None, None, 0, 0)
                    legacy.on_predict_end(_TRAINER, None)
                    new = ParquetBatchWriter(self.tmp, f"new{buffer_size}_{size}", buffer_size=buffer_size)
                    for start in range(0, 7, 2):
                        prediction = _prediction(rows[start:start + 2], rs.RESIDUE_CODES if size == 4 else None)
                        new.write_on_batch_end(_TRAINER, None, prediction, None, None, 0, 0)
                    new.on_predict_end(_TRAINER, None)
                    self.assertEqual(_sha256(self.tmp / f"new{buffer_size}_{size}-0.parquet"),
                                     _sha256(self.tmp / f"legacy{buffer_size}-0.parquet"))

    def test_the_schema_is_fixed_at_construction(self):
        for residue_output in rs.RESIDUE_OUTPUTS:
            with self.subTest(residue_output=residue_output):
                schema = ParquetBatchWriter(self.tmp, "x", residue_output=residue_output, system=SYSTEM).schema
                self.assertEqual([(f.name, f.type) for f in schema], [
                    ("id", pa.string()), ("embedding", pa.list_(pa.float32())),
                    (residue_output, pa.large_binary()), ("seq_sha1", pa.binary(20))])
                metadata = dict(schema.metadata)
                record = json.loads(metadata.pop(b"foldmatch.pass"))
                self.assertEqual(metadata, {
                    b"foldmatch.residue_output": residue_output.encode(),
                    b"foldmatch.system": b"esm-codes-v1",
                    b"foldmatch.encoding_digest": SYSTEM.encoding_digest().encode(),
                    b"foldmatch.system_digest": SYSTEM.digest().encode(),
                })
                self.assertEqual(tuple(sorted(record)), tuple(sorted(rs.PASS_KEYS)))
                self.assertEqual({k: record[k] for k in ("min_res_n", "accelerator", "world_size", "esm_precision",
                                                         "esm_revision")}, dict.fromkeys(
                    ("min_res_n", "accelerator", "world_size", "esm_precision", "esm_revision")))  # not known yet
                self.assertEqual((record["encoder_precision"], record["foldmatch_version"]),
                                 ("fp32-no-tf32", foldmatch.__version__))
        self.assertEqual(SYSTEM.encoding_digest(), "92220d870638cb57")  # the P4 golden

    def test_the_pass_is_recorded(self):
        """foldmatch.pass: min_res_n and the ESM3 revision from the constructor, the device type (and so the ESM3
        precision) and the number of ranks from the start of the pass, in every shard's footer."""
        rows = _rows(3)
        for device, precision in (("cuda", "bf16-autocast"), ("cpu", "fp32"), ("mps", "fp32")):
            with self.subTest(device=device):
                writer = ParquetBatchWriter(self.tmp, device, residue_output=rs.RESIDUE_CODES, system=SYSTEM,
                                            min_res_n=7, esm_revision="r" * 40)
                trainer = SimpleNamespace(global_rank=1, world_size=4)
                module = SimpleNamespace(residue_output=rs.RESIDUE_CODES, device=torch.device(device))
                writer.on_predict_start(trainer, module)
                writer.write_on_batch_end(trainer, module, _prediction(rows, rs.RESIDUE_CODES), None, None, 0, 0)
                writer.on_predict_end(trainer, module)
                record = _pass_record(self.tmp / f"{device}-1.parquet")
                self.assertEqual(record, rs.pass_record(min_res_n=7, esm_revision="r" * 40, accelerator=device,
                                                        world_size=4))
                self.assertEqual((record["esm_precision"], record["encoder_precision"]), (precision, "fp32-no-tf32"))
                self.assertEqual(record["foldmatch_version"], foldmatch.__version__)
                self.assertTrue(record["torch"] and record["esm"] and record["platform"])
        self.assertEqual([rs.esm_precision(d) for d in ("cuda", "cpu", "mps", None)],
                         ["bf16-autocast", "fp32", "fp32", None])

    def test_round_trip(self):
        rows = _rows(7)
        for residue_output in rs.RESIDUE_OUTPUTS:
            with self.subTest(residue_output=residue_output):
                path = _write_shard(self.tmp, residue_output, rows, residue_output)
                table = pq.read_table(path)
                self.assertEqual(table.schema, rs.shard_schema(residue_output, SYSTEM))
                self.assertEqual(table.column("id").to_pylist(), [r[0] for r in rows])
                payloads = [r[2] if residue_output == rs.RESIDUE_CODES else r[3] for r in rows]
                self.assertEqual(table.column(residue_output).to_pylist(), payloads)
                self.assertEqual(b"", payloads[1])
                self.assertEqual(table.column("seq_sha1").to_pylist(), [r[4] for r in rows])
                np.testing.assert_array_equal(np.stack(table.column("embedding").to_numpy(zero_copy_only=False)),
                                              np.stack([r[1].numpy() for r in rows]))

    def test_byte_budget_and_row_count_flushes(self):
        self.assertEqual(ParquetBatchWriter(self.tmp, "d", residue_output=rs.RESIDUE_CODES, system=SYSTEM).byte_budget,
                         256 * 1024 * 1024)
        self.assertEqual(batch_writer.DEFAULT_BYTE_BUDGET, 256 * 1024 * 1024)
        rows = []
        for k in range(7):
            rows.append((f"{k}", torch.zeros(4), b"\x01" * 1000, b"", b"\x02" * 20))
        # Each row buffers 16 (embedding) + 1 (id) + 1000 (payload) + 20 (seq_sha1) = 1037 bytes: a 2,500-byte budget
        # is crossed by every third row.
        path = _write_shard(self.tmp, "budget", rows, rs.RESIDUE_CODES, batch=1, byte_budget=2500)
        meta = pq.ParquetFile(path).metadata
        self.assertEqual([meta.row_group(i).num_rows for i in range(meta.num_row_groups)], [3, 3, 1])
        path = _write_shard(self.tmp, "count", rows, rs.RESIDUE_CODES, batch=1, buffer_size=2)
        meta = pq.ParquetFile(path).metadata
        self.assertEqual([meta.row_group(i).num_rows for i in range(meta.num_row_groups)], [2, 2, 2, 1])
        self.assertEqual(pq.read_table(path).column("residue_codes").to_pylist(), [r[2] for r in rows])
        # A plain writer may be given a budget too.
        path = _write_shard(self.tmp, "plain", rows, None, batch=1, byte_budget=40)
        meta = pq.ParquetFile(path).metadata
        self.assertEqual([meta.row_group(i).num_rows for i in range(meta.num_row_groups)], [3, 3, 1])

    def test_a_payload_chunk_over_2_gib_mocked(self):
        """large_binary has 64-bit offsets: one row group may hold more than 2 GiB of payloads. Mocked: the 17 payloads
        of 128 MiB are one lazily zeroed object, the concatenated buffer a lazily zeroed array, and the Parquet writer
        only receives the table, so no memory is touched."""
        size, n = 1 << 27, 17
        payload = bytes(size)
        captured = []

        class FakeWriter:
            def __init__(self, path, schema, compression=None):
                self.schema = schema

            def write_table(self, table):
                captured.append(table)

            def close(self):
                pass

        rows = [(f"r{k}", torch.zeros(4), payload, b"", bytes(20)) for k in range(n)]
        with mock.patch.object(rs, "_concat_payloads", side_effect=lambda payloads, total: np.zeros(total, np.uint8)), \
                mock.patch.object(batch_writer.pq, "ParquetWriter", FakeWriter):
            _write_shard(self.tmp, "big", rows, rs.RESIDUE_CODES, batch=1, byte_budget=1 << 34)
        table, = captured
        column = table.column("residue_codes")
        self.assertEqual(column.num_chunks, 1)
        chunk = column.chunk(0)
        self.assertEqual(chunk.type, pa.large_binary())
        offsets = np.frombuffer(chunk.buffers()[1], dtype=np.int64)
        self.assertEqual(int(offsets[-1]), n * size)
        self.assertGreater(int(offsets[-1]), 2 ** 31)
        self.assertTrue(np.array_equal(np.diff(offsets), np.full(n, size)))
        chunk.validate(full=True)
        self.assertEqual(table.num_rows, n)
        with self.assertRaises((pa.ArrowCapacityError, pa.ArrowInvalid, OverflowError)):
            pa.BinaryArray.from_buffers(pa.binary(), n, [None, pa.py_buffer(offsets.astype(np.int32)),
                                                         chunk.buffers()[2]]).validate(full=True)

    def test_a_writer_with_a_column_refuses_predictions_without_it(self):
        rows = _rows(2)
        writer = ParquetBatchWriter(self.tmp, "x", residue_output=rs.RESIDUE_CODES, system=SYSTEM)
        for prediction in (_prediction(rows), _prediction(rows, rs.RESIDUE_CODES)[:3]):
            with self.subTest(size=len(prediction)), self.assertRaises(ValueError):
                writer.write_on_batch_end(_TRAINER, None, prediction, None, None, 0, 0)
        embeddings, ids, payloads, digests = _prediction(rows, rs.RESIDUE_CODES)
        for bad in ((embeddings, ids, payloads[:1], digests), (embeddings, ids, payloads, (b"x",) * 2)):
            with self.assertRaises(ValueError):
                writer.write_on_batch_end(_TRAINER, None, bad, None, None, 0, 0)
        # ... and, at the start of the pass, a module that does not produce it
        for produced in (None, rs.QUERY_ENCODING):
            with self.subTest(produced=produced), self.assertRaises(ValueError):
                writer.on_predict_start(_TRAINER, SimpleNamespace(residue_output=produced))
        with self.assertRaises(ValueError):
            writer.on_predict_start(_TRAINER, object())
        writer.on_predict_start(_TRAINER, SimpleNamespace(residue_output=rs.RESIDUE_CODES))
        # a plain writer takes any module
        ParquetBatchWriter(self.tmp, "y").on_predict_start(_TRAINER, SimpleNamespace(residue_output=rs.RESIDUE_CODES))
        writer.on_predict_end(_TRAINER, None)
        self.assertFalse((self.tmp / "x-0.parquet").exists())

    def test_a_plain_writer_drops_extra_tuple_elements(self):
        rows = _rows(5)
        plain = _write_shard(self.tmp, "two", rows)
        writer = ParquetBatchWriter(self.tmp, "four")
        writer.write_on_batch_end(_TRAINER, None, _prediction(rows, rs.QUERY_ENCODING), None, None, 0, 0)
        writer.on_predict_end(_TRAINER, None)
        self.assertTrue(pq.read_table(self.tmp / "four-0.parquet").equals(pq.read_table(plain)))

    def test_constructor_refusals(self):
        with self.assertRaises(ValueError):
            ParquetBatchWriter(self.tmp, "x", residue_output=rs.RESIDUE_CODES)
        with self.assertRaises(ValueError):
            ParquetBatchWriter(self.tmp, "x", system=SYSTEM)
        for kwargs in ({"min_res_n": 0}, {"esm_revision": "r"}):  # a plain shard keeps its legacy bytes
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ParquetBatchWriter(self.tmp, "x", **kwargs)
        with self.assertRaises(ValueError):
            ParquetBatchWriter(self.tmp, "x", residue_output="codes", system=SYSTEM)
        with self.assertRaises(ValueError):
            ParquetBatchWriter(self.tmp, "x", residue_output=rs.RESIDUE_CODES, system=SYSTEM, byte_budget=0)


class InferenceShardsTest(_FakeEsm3Pass):
    """predict() with a Parquet output: plain shards as before, residue shards with the columns."""

    def test_shards(self):
        items, codes_ref, q_ref, offsets = _f2c()
        with tempfile.TemporaryDirectory() as tmp:
            self.predict(batch_size=2, out_format=OutFormat.parquet, out_folder=tmp, out_name="plain",
                         return_predictions=False)
            plain = pq.read_table(Path(tmp) / "plain-0.parquet")
            self.assertEqual(plain.schema, rs.shard_schema(None))
            for residue_output in rs.RESIDUE_OUTPUTS:
                with self.subTest(residue_output=residue_output):
                    self.predict(batch_size=2, out_format=OutFormat.parquet, out_folder=tmp, out_name=residue_output,
                                 residue_output=residue_output, scoring_system=SYSTEM, return_predictions=False)
                    table = pq.read_table(Path(tmp) / f"{residue_output}-0.parquet")
                    self.assertEqual(table.schema, rs.shard_schema(residue_output, SYSTEM))
                    self.assertTrue(table.select(["id", "embedding"]).equals(plain))
                    # Lightning's start of the pass told the writer the device and the ranks.
                    from foldmatch.utils.esm.loaders import ESM_REVISION
                    record = _pass_record(Path(tmp) / f"{residue_output}-0.parquet")
                    self.assertEqual({k: record[k] for k in ("min_res_n", "accelerator", "world_size",
                                                             "esm_precision", "esm_revision")},
                                     {"min_res_n": 0, "accelerator": "cpu", "world_size": 1, "esm_precision": "fp32",
                                      "esm_revision": ESM_REVISION})
                    data = b"".join(table.column(residue_output).to_pylist())
                    if residue_output == rs.RESIDUE_CODES:
                        self.assertEqual(data, codes_ref.tobytes())
                    else:
                        q = np.frombuffer(data, dtype="<f2").reshape(-1, 64)
                        self.assertLessEqual(_row_ulps(q, q_ref), 1.0)
                    self.assertEqual(table.column("seq_sha1").to_pylist(), [_sha1(s) for _, s, _ in items])

if __name__ == "__main__":
    unittest.main()
