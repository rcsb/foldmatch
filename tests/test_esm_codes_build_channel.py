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

if __name__ == "__main__":
    unittest.main()
