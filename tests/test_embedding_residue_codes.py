"""P9.X: ``fm-embedding from-sequences residue|chain --output-codes-folder``: the residue codes in the same ESM3 pass,
in a folder of their own.

Most tests run the commands with ESM3 replaced by rows (``_esm3``: the F2c rows of the fixture FASTA, the FoldMatch CPU
fp32 ESM3 rows, and fixed pseudo-random rows for any other sequence), a stub aggregator and the real esm-codes-v1
encoders (the registry bundle, ``esm_codes_bundle``: skipped only when offline and not cached), so the codes of the
fixture are F2c's exactly. ``RealEsm3Test`` runs the real ESM3 on the CPU: the embedding files with and without
``--output-codes-folder`` in every format of both commands, the codes against ``fm-search build sequences
--residue-codes``, and fm-search on the chain output folders. ``TwoRankSmoke`` (slow) runs two ranks as a command
and as a library caller.
"""
import functools
import gc
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import zlib
from contextlib import chdir, contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from typer.testing import CliRunner

import esm_codes_bundle
import foldmatch
from foldmatch.cli import embedding, search
from foldmatch.errors import MissingResidueCodesError, RefusalError
# Imported before any test patches their model getters
from foldmatch.inference import chain_complete_inference, sequence_inference
from foldmatch.modules import chain_complete_module, esm_module
from foldmatch.modules.chain_complete_module import ChainCompleteModule
from foldmatch.modules.esm_module import EsmModule
from foldmatch.modules.residue_payloads import ResiduePayloads
from foldmatch.search import residue_shards as rs
from foldmatch.search import scoring_system as ss
from foldmatch.search.residue_code_store import ResidueCodeStore
from foldmatch.types.api_types import Accelerator, OutFormat, SrcEsmFrom
from foldmatch.utils import data, output_files
from foldmatch.utils.output_files import EXACT, Folding, default_codes_name, same_folder

_TEST_PATH = Path(__file__).parent
_FASTA = _TEST_PATH / "resources" / "fasta" / "test_sequences.fasta"
_F2C = _TEST_PATH / "resources" / "esm_codes" / "fasta_rows_codes_v1.npz"
_SRC = str(Path(foldmatch.__file__).resolve().parent.parent)
_CPU = ("--accelerator", "cpu", "--devices", "1")
_FORMATS = ("csv", "pt", "parquet", "json")
_COMMANDS = ("residue", "chain")
_LAUNCH_ENV = ("FOLDMATCH_RUN_ID", "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_RESTART_COUNT", "SLURM_PROCID",
               "SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "RANK", "LOCAL_RANK", "NODE_RANK", "WORLD_SIZE")
SYSTEM = ss.get(ss.DEFAULT_ESM_CODES)
#: A4_THRESHOLDS.json, gate A4-xcpu: CPU Mac vs CPU Linux code bits, also the F2c threshold.
A4_XCPU_BITS = 0.999


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


@functools.lru_cache(maxsize=None)
def _f2c():
    """F2c: ``{sequence: ESM3 rows [L + 2, 1536] fp32}``, ``{id: reference codes [L, 8]}``, the ids and sequences."""
    z = np.load(_F2C, allow_pickle=False)
    ro, offsets = z["rows_offsets"], z["offsets"]
    ids = [str(i) for i in z["ids"]]
    sequences = [str(s) for s in z["sequences"]]
    rows = {sequences[k]: torch.from_numpy(z["rows"][ro[k]:ro[k + 1]].astype(np.float32)) for k in range(len(ids))}
    codes = {ids[k]: z["codes"][offsets[k]:offsets[k + 1]] for k in range(len(ids))}
    return rows, codes, ids, sequences


def _esm3(model, protein):
    """Stands in for ESM3: the F2c rows of the fixture's sequences, fixed pseudo-random rows [L + 2, 1536] for any
    other."""
    rows = _f2c()[0].get(protein.sequence)
    if rows is not None:
        return rows.clone()
    generator = torch.Generator().manual_seed(zlib.crc32(protein.sequence.encode("utf-8", "surrogatepass")))
    return torch.randn(len(protein.sequence) + 2, 1536, generator=generator)


def _aggregate(x, x_mask):
    """Stands in for the aggregator: the sum of each item's rows."""
    return x.masked_fill(x_mask.unsqueeze(-1), 0.0).sum(dim=1)


def _flat(text: str) -> str:
    """Text on one line, typer's error panel borders and colour codes dropped."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


def _sha1(sequence: str) -> bytes:
    return hashlib.sha1(sequence.encode("ascii")).digest()


def _case_insensitive() -> bool:
    with tempfile.NamedTemporaryFile(prefix="CaseProbe-") as f:
        head, name = os.path.split(f.name)
        return os.path.exists(os.path.join(head, name.swapcase()))


_CASE_INSENSITIVE = _case_insensitive()


def read_codes(folder: Path, out_format: str, out_codes_name: str = "inference-codes", ids=None) -> dict:
    """``{id: uint8 [L, 8]}`` of the residue-code files of a run in its codes folder ``folder``, read without
    FoldMatch: ``<out_codes_name>-<rank>.parquet|json.gz``, or ``<id>.csv|pt`` for each of ``ids``."""
    out = {}
    if out_format == "parquet":
        for path in sorted(folder.glob(f"{out_codes_name}-*.parquet")):
            table = pq.read_table(path)
            for dom_id, residues in zip(table.column("id").to_pylist(), table.column("residue_codes").to_pylist()):
                out[dom_id] = np.array(residues, dtype=np.uint8).reshape(-1, 8)
    elif out_format == "json":
        for path in sorted(folder.glob(f"{out_codes_name}-*.json.gz")):
            for record in json.loads(gzip.decompress(path.read_bytes())):
                out[record["id"]] = np.array(record["residue_codes"], dtype=np.uint8).reshape(-1, 8)
    else:
        for dom_id in ids:
            path = folder / f"{dom_id}.{out_format}"
            if out_format == "pt":
                tensor = torch.load(path, weights_only=True)
                assert tensor.dtype == torch.uint8 and tensor.device.type == "cpu", (tensor.dtype, tensor.device)
                out[dom_id] = tensor.numpy()
            else:
                text = path.read_text()
                out[dom_id] = (np.zeros((0, 8), dtype=np.uint8) if not text else
                               pd.read_csv(path, header=None).to_numpy().astype(np.uint8))
    return out


def names(folder: Path) -> list:
    return sorted(p.name for p in folder.iterdir())


def same_bytes(a: Path, b: Path) -> bool:
    """Whether two output files hold the same bytes; a .json.gz but for the gzip header's mtime (bytes 4-7), which
    pandas sets to the time of writing."""
    x, y = a.read_bytes(), b.read_bytes()
    if a.name.endswith(".gz"):
        return x[:4] + x[8:] == y[:4] + y[8:]
    return x == y


class _Case(unittest.TestCase):
    """The commands with stub models (``_esm3``, ``_aggregate``) and the real encoders; ``refusing`` makes building a
    model an error, so a refused run shows that it stopped before ESM3."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        esm_codes_bundle.registry_files()  # unittest.SkipTest only when offline and not cached

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(launch_env())
        for module in (sequence_inference, chain_complete_inference):
            self.enterContext(mock.patch.object(module, "get_residue_model", self.model(torch.nn.Identity())))
        self.enterContext(mock.patch.object(chain_complete_inference, "get_aggregator_model",
                                            self.model(_aggregate)))
        for module in (esm_module, chain_complete_module):
            self.enterContext(mock.patch.object(module, "per_residue_embedding", side_effect=_esm3))
        self.prepare = self.enterContext(mock.patch.object(ss, "prepare", side_effect=ss.prepare))
        self.refusing = False

    def model(self, model):
        def get(**_):
            if self.refusing:
                raise AssertionError("a model was built")
            return model
        return get

    def folder(self, name: str = "out") -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.tmp))

    def fasta(self, *records, name=None) -> Path:
        path = Path(name) if name else self.tmp / f"q{len(list(self.tmp.iterdir()))}.fasta"
        path.write_text("".join(f">{header}\n{sequence}\n" for header, sequence in records), encoding="utf-8")
        return path

    def invoke(self, *args, app=None, cpu=True):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            return CliRunner().invoke(app or embedding.app, [str(a) for a in (*args, *(_CPU if cpu else ()))])
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)

    def run_ok(self, command, fasta, out, *extra):
        self.refusing = False
        result = self.invoke("from-sequences", command, "--fasta-file", fasta, "--output-folder", out, *extra)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        return result

    def run_search(self, *args):
        result = self.invoke(*args, app=search.app, cpu=False)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        return result

    def refused(self, *args, written=()) -> str:
        """The command exits 1 with a RefusalError (D20) before any model, writing nothing to ``written``; returns
        its message on one line."""
        before = {folder: sorted(p.name for p in folder.iterdir()) for folder in written}
        self.refusing = True
        try:
            result = self.invoke(*args)
        finally:
            self.refusing = False
        self.assertEqual(result.exit_code, 1, f"{result.output}\n{result.exception!r}")
        self.assertIsInstance(result.exception, RefusalError, repr(result.exception))
        for folder, names in before.items():
            self.assertEqual(sorted(p.name for p in folder.iterdir()), names, folder)
        return _flat(str(result.exception))


# =====================================================================================================================
# The options
# =====================================================================================================================
class OptionTest(unittest.TestCase):

    def help(self, command):
        result = CliRunner().invoke(embedding.app, ["from-sequences", command, "--help"], env={"COLUMNS": "200"})
        self.assertEqual(result.exit_code, 0, result.output)
        return _flat(result.output)

    def test_both_commands_have_the_options_with_one_help(self):
        texts = {command: self.help(command) for command in _COMMANDS}
        for command, text in texts.items():
            with self.subTest(command=command):
                self.assertIn("--output-codes-folder", text)
                self.assertIn("--output-codes-name", text)
                self.assertNotIn("--residue-codes --no-residue-codes", text)
                self.assertIn("Also compute each sequence's residue codes in the same ESM3 pass and write them to "
                              "this folder", text)
                self.assertIn("Used when output-format is parquet or json", text)
                self.assertIn("(default <output-name>-codes)", text)
                self.assertIn("must not be --output-folder (a subfolder of it will do):", text)
                self.assertNotIn(".parquet, .pt or .csv", text)
                self.assertIn("A sequence whose codes cannot be computed gets an empty table, and the run exits with "
                              "status 1 once every file is written.", text)
                self.assertIn("row i + 1 of the residue embeddings that 'from-sequences residue' writes", text)
                self.assertIn("on several nodes, or under srun or torchrun, run 'fm-embedding download-models' first",
                              text)
                self.assertIn("A run that would replace an embedding file there is refused.", text)
        start = lambda text: text[text.index("--output-codes-folder", text.index("--output-name")):  # noqa: E731
                                  text.index("--min-res-n")]
        self.assertEqual(start(texts["residue"]), start(texts["chain"]))
        for command in ("residue", "chain", "assembly"):
            result = CliRunner().invoke(embedding.app, ["from-structures", command, "--help"],
                                        env={"COLUMNS": "200"})
            texts[command] = _flat(result.output)
            self.assertNotIn("--output-codes-folder PATH", texts[command], command)
            self.assertNotIn("--output-codes-name", texts[command], command)
        for command, text in texts.items():
            # Z6: every command's output folder
            self.assertIn("Output folder to store predictions. A run that would replace a file of residue codes "
                          "there (as 'fm-embedding from-sequences ... --output-codes-folder' writes them) is refused.",
                          text, command)

    def test_the_codes_name_without_the_codes_folder_is_a_usage_error(self):
        """V5: --output-codes-name alone is typer.BadParameter, exit status 2, before anything runs."""
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(sequence_inference, "predict", side_effect=AssertionError("ran")), \
                mock.patch.object(chain_complete_inference, "predict", side_effect=AssertionError("ran")):
            for command in _COMMANDS:
                with self.subTest(command=command):
                    result = CliRunner().invoke(embedding.app, [
                        "from-sequences", command, "--fasta-file", str(_FASTA), "--output-folder", tmp,
                        "--output-codes-name", "x"])
                    self.assertEqual(result.exit_code, 2, result.output)
                    self.assertIn("--output-codes-name", _flat(result.output))
                    self.assertIn("names the residue-code files of --output-codes-folder, which is not given.",
                                  _flat(result.output))

    def test_the_codes_folder_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            for codes in (Path(tmp) / "missing", _FASTA):
                with self.subTest(codes=codes.name):
                    result = CliRunner().invoke(embedding.app, [
                        "from-sequences", "chain", "--fasta-file", str(_FASTA), "--output-folder", tmp,
                        "--output-codes-folder", str(codes)])
                    self.assertEqual(result.exit_code, 2, result.output)
                    self.assertIn("--output-codes-folder", _flat(result.output))

    def test_default_codes_name(self):
        self.assertEqual(default_codes_name("inference"), "inference-codes")
        self.assertEqual(default_codes_name("x"), "x-codes")


# =====================================================================================================================
# Without --output-codes-folder nothing changes
# =====================================================================================================================
class PlainRunTest(_Case):

    def test_a_plain_run_is_unchanged(self):
        """The legacy module and writers, no esm-codes bundle, no strict FASTA pre-flight (a '<' still runs)."""
        fasta = self.fasta(("a", "MK<mask>V"), ("b", "MKVLA"))
        made, writers = [], []
        real_trainer = sequence_inference.Trainer

        def trainer(**kwargs):
            writers.append([type(c).__name__ for c in kwargs["callbacks"] or []])
            return real_trainer(**kwargs)

        for command, module_path, module_class in (("residue", sequence_inference, EsmModule),
                                                   ("chain", chain_complete_inference, ChainCompleteModule)):
            for out_format in _FORMATS:
                with self.subTest(command=command, out_format=out_format):
                    made.clear()
                    writers.clear()
                    construct = lambda *a, _cls=module_class, **k: made.append(k) or _cls(*a, **k)  # noqa: E731
                    with mock.patch.object(module_path, module_class.__name__, side_effect=construct), \
                            mock.patch.object(module_path, "Trainer", side_effect=trainer), \
                            mock.patch.object(chain_complete_inference, "strict_fasta_preflight") as strict, \
                            mock.patch.object(chain_complete_inference, "residue_code_writer") as code_writer:
                        self.run_ok(command, fasta, self.folder(), "--output-format", out_format)
                    self.assertEqual(len(made), 1)
                    self.assertNotIn("residue_head", made[0])
                    self.assertNotIn("code_files", made[0])
                    self.assertEqual(len(writers[0]), 1, writers)
                    strict.assert_not_called()
                    code_writer.assert_not_called()
        self.prepare.assert_not_called()


# =====================================================================================================================
# The code files
# =====================================================================================================================
class CodeFilesTest(_Case):

    def test_every_format_of_both_commands(self):
        """U2/U4/V3/V4: the F2c codes of the fixture, one [L, 8] table per sequence, in the codes folder (csv and pt
        <id>.<format>, mirroring the embedding files; parquet and json <output-name>-codes-<rank>), and the output
        folder holds the embedding files of a run without --output-codes-folder, byte for byte, and nothing else."""
        _, reference, ids, _ = _f2c()
        for command in _COMMANDS:
            for out_format in _FORMATS:
                with self.subTest(command=command, out_format=out_format):
                    plain, out, codes = self.folder("plain"), self.folder("out"), self.folder("codes")
                    self.run_ok(command, _FASTA, plain, "--output-format", out_format)
                    self.run_ok(command, _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes)
                    self.assertEqual(names(out), names(plain))
                    for name in names(plain):
                        self.assertTrue(same_bytes(plain / name, out / name), name)
                    if out_format in ("csv", "pt"):
                        self.assertEqual(names(codes), names(plain))
                        self.assertEqual(names(codes), sorted(f"{i}.{out_format}" for i in ids))
                    else:
                        suffix = "parquet" if out_format == "parquet" else "json.gz"
                        self.assertEqual(names(codes), [f"inference-codes-0.{suffix}"])
                    got = read_codes(codes, out_format, ids=ids)
                    self.assertEqual(list(got), ids)
                    for dom_id in ids:
                        np.testing.assert_array_equal(got[dom_id], reference[dom_id])

    def test_the_layout_of_each_format(self):
        _, reference, ids, sequences = _f2c()
        out, codes = self.folder(), self.folder("codes")
        self.run_ok("chain", _FASTA, out, "--output-format", "csv", "--output-codes-folder", codes)
        lines = (codes / f"{ids[1]}.csv").read_text().splitlines()
        self.assertEqual(len(lines), 58)
        self.assertEqual(lines[0], ",".join(str(v) for v in reference[ids[1]][0]))
        out, codes = self.folder(), self.folder("codes")
        self.run_ok("residue", _FASTA, out, "--output-format", "pt", "--output-codes-folder", codes)
        tensor = torch.load(codes / f"{ids[0]}.pt", weights_only=True)
        self.assertEqual((tensor.dtype, tuple(tensor.shape), tensor.device.type), (torch.uint8, (245, 8), "cpu"))
        self.assertEqual(tensor.untyped_storage().nbytes(), 245 * 8)
        out, codes = self.folder(), self.folder("codes")
        self.run_ok("residue", _FASTA, out, "--output-format", "json", "--output-codes-folder", codes)
        records = json.loads(gzip.decompress((codes / "inference-codes-0.json.gz").read_bytes()))
        self.assertEqual([sorted(r) for r in records], [["id", "residue_codes"]] * 2)
        self.assertEqual(records[1]["residue_codes"][0], reference[ids[1]][0].tolist())

    def test_the_parquet_layout_and_provenance(self):
        _, reference, ids, sequences = _f2c()
        for command in _COMMANDS:
            with self.subTest(command=command):
                out, codes = self.folder(), self.folder("codes")
                self.run_ok(command, _FASTA, out, "--output-format", "parquet", "--output-codes-folder", codes,
                            "--min-res-n", "50")
                schema = pq.read_schema(codes / "inference-codes-0.parquet")
                self.assertEqual(schema.names, ["id", "residue_codes", "seq_sha1"])
                self.assertEqual(schema.field("residue_codes").type, pa.list_(pa.list_(pa.uint8(), 8)))
                self.assertEqual(schema.field("seq_sha1").type, pa.binary(20))
                meta = schema.metadata
                self.assertEqual(meta[b"foldmatch.residue_output"], b"residue_code_table")
                self.assertEqual(meta[b"foldmatch.system"], SYSTEM.name.encode())
                self.assertEqual(meta[b"foldmatch.encoding_digest"], SYSTEM.encoding_digest().encode())
                self.assertEqual(meta[b"foldmatch.system_digest"], SYSTEM.digest().encode())
                record = json.loads(meta[b"foldmatch.pass"])
                from foldmatch.utils.esm.loaders import ESM_REVISION
                self.assertEqual({k: record[k] for k in ("min_res_n", "accelerator", "world_size", "esm_precision",
                                                         "encoder_precision", "esm_revision")},
                                 {"min_res_n": 50, "accelerator": "cpu", "world_size": 1, "esm_precision": "fp32",
                                  "encoder_precision": "fp32-no-tf32", "esm_revision": ESM_REVISION})
                table = pq.read_table(codes / "inference-codes-0.parquet")
                self.assertEqual(table.column("seq_sha1").to_pylist(), [_sha1(s) for s in sequences])
                # The embeddings' Parquet is untouched: no metadata, the legacy columns for chain embeddings
                self.assertEqual(names(out), ["inference-0.parquet"])
                embeddings = pq.read_schema(out / "inference-0.parquet")
                self.assertIsNone(embeddings.metadata)
                if command == "chain":
                    self.assertEqual(embeddings.names, ["id", "embedding"])

    def test_min_res_n_filters_both(self):
        _, reference, ids, _ = _f2c()
        for out_format in _FORMATS:
            with self.subTest(out_format=out_format):
                out, codes = self.folder(), self.folder("codes")
                self.run_ok("residue", _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes,
                            "--min-res-n", "100")
                self.assertEqual(list(read_codes(codes, out_format, ids=ids[:1])), ids[:1])
                if out_format in ("csv", "pt"):
                    self.assertEqual(names(out), [f"{ids[0]}.{out_format}"])
                    self.assertEqual(names(codes), [f"{ids[0]}.{out_format}"])

    def test_the_codes_name(self):
        """V3/V4: --output-codes-name names the parquet and json files (by default <output-name>-codes; the output
        name itself is allowed, in another folder), and no file in csv and pt."""
        _, reference, ids, _ = _f2c()
        cases = [("csv", ("--output-codes-name", "esmc"), [f"{i}.csv" for i in ids]),
                 ("pt", (), [f"{i}.pt" for i in ids]),
                 ("parquet", ("--output-name", "emb"), ["emb-codes-0.parquet"]),
                 ("parquet", ("--output-codes-name", "inference"), ["inference-0.parquet"]),
                 ("json", ("--output-codes-name", "c"), ["c-0.json.gz"])]
        for out_format, extra, files in cases:
            with self.subTest(out_format=out_format, extra=extra):
                out, codes = self.folder(), self.folder("codes")
                self.run_ok("chain", _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes,
                            *extra)
                self.assertEqual(names(codes), sorted(files))
                self.assertNotIn("esmc", " ".join(names(out)))
        self.assertEqual(list(read_codes(codes, "json", "c")), ids)

    def test_nested_folders(self):
        """V2: one folder inside the other is allowed, both ways."""
        _, reference, ids, _ = _f2c()
        for out_format in ("csv", "parquet"):
            for name in ("codes", "codes.parquet", "codes.pt", "codes.csv"):   # W2: fm-search skips folders
                with self.subTest(out_format=out_format, codes=f"inside the output folder, {name}"):
                    out = self.folder()
                    (out / name).mkdir()
                    self.run_ok("chain", _FASTA, out, "--output-format", out_format, "--output-codes-folder",
                                out / name)
                    self.assertEqual(list(read_codes(out / name, out_format, ids=ids)), ids)
            with self.subTest(out_format=out_format, codes="around the output folder"):
                codes = self.folder("codes")
                (codes / "embeddings").mkdir()
                self.run_ok("chain", _FASTA, codes / "embeddings", "--output-format", out_format,
                            "--output-codes-folder", codes)
                self.assertEqual(list(read_codes(codes, out_format, ids=ids)), ids)

    def test_a_rerun_replaces_its_files(self):
        out, codes = self.folder(), self.folder("codes")
        for out_format in ("csv", "parquet"):
            for _ in range(2):
                self.run_ok("residue", _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes)

    def test_codes_equal_the_build_store(self):
        """U4: the codes are the target codes that 'fm-search build sequences --residue-codes' stores for the FASTA,
        same device and min_res_n."""
        _, _, ids, _ = _f2c()
        db = self.folder("db")
        shards = self.folder("shards")
        with launch_env(FOLDMATCH_RUN_ID="p9x-build"):
            result = self.invoke("build", "sequences", "--fasta-file", _FASTA, "--output-db", db / "db",
                                 "--tmp-embedding-folder", shards, "--residue-codes", app=search.app)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        store = ResidueCodeStore(db, "db")
        store.check_compatible()
        memmap = np.asarray(store.open_memmap())
        entries = store.lookup(ids)
        stored = {i: memmap[entries[i][0]:entries[i][0] + entries[i][1]] for i in ids}
        for command in _COMMANDS:
            for out_format in _FORMATS:
                with self.subTest(command=command, out_format=out_format):
                    codes = self.folder("codes")
                    self.run_ok(command, _FASTA, self.folder(), "--output-format", out_format,
                                "--output-codes-folder", codes)
                    got = read_codes(codes, out_format, ids=ids)
                    for dom_id in ids:
                        np.testing.assert_array_equal(got[dom_id], stored[dom_id])

    def test_a_protein_without_codes_gets_an_empty_table(self):
        """W4: ESM3 rows that do not match the sequence's length: an ERROR per protein and a count, an empty [0, 8]
        table, the embedding written as without --output-codes-folder, every file written; then the run ends with
        exit status 1, a MissingResidueCodesError (a RefusalError, D20) naming the count, the ids and where the empty
        tables are."""
        _, reference, ids, sequences = _f2c()

        def one_row_short(model, protein):
            rows = _esm3(model, protein)
            return rows[:-1] if protein.sequence == sequences[1] else rows

        for command, module in (("residue", esm_module), ("chain", chain_complete_module)):
            for out_format in _FORMATS:
                with self.subTest(command=command, out_format=out_format), \
                        mock.patch.object(module, "per_residue_embedding", side_effect=one_row_short):
                    plain = self.folder("plain")
                    self.run_ok(command, _FASTA, plain, "--output-format", out_format)
                    out, codes = self.folder(), self.folder("codes")
                    with self.assertLogs("foldmatch.modules.residue_payloads", "ERROR") as logs:
                        result = self.invoke("from-sequences", command, "--fasta-file", _FASTA, "--output-folder",
                                             out, "--output-format", out_format, "--output-codes-folder", codes)
                    self.assertEqual(result.exit_code, 1, f"{result.output}\n{result.exception!r}")
                    self.assertIsInstance(result.exception, MissingResidueCodesError)
                    self.assertEqual((result.exception.count, result.exception.ids), (1, (ids[1],)))
                    where = {"csv": f"{codes}/<id>.csv of each, an empty file",
                             "pt": f"{codes}/<id>.pt of each, a uint8 tensor [0, 8]",
                             "parquet": f"rows of {codes}/inference-codes-0.parquet without residues",
                             "json": f"records of {codes}/inference-codes-0.json.gz without residues"}[out_format]
                    self.assertEqual(str(result.exception),
                                     f"1 sequence(s) got no residue codes: {ids[1]!r}. ESM3 gave rows that do not "
                                     f"match the sequence's length, or the encoding failed (each one's reason is "
                                     f"logged above). Their codes are written as empty [0, 8] tables "
                                     f"({where}); their embeddings and every other file of the run are written. Check "
                                     f"those sequences, or leave them out of the FASTA file.")
                    got = read_codes(codes, out_format, ids=ids)
                    np.testing.assert_array_equal(got[ids[0]], reference[ids[0]])
                    self.assertEqual(got[ids[1]].shape, (0, 8))
                    if out_format == "csv":
                        self.assertEqual((codes / f"{ids[1]}.csv").read_bytes(), b"")
                    self.assertEqual(names(out), names(plain))   # the embeddings of a run without the option
                    for name in names(plain):
                        self.assertTrue(same_bytes(plain / name, out / name), name)
                    text = "\n".join(logs.output)
                    self.assertIn(f"{ids[1]}: no residue codes (ESM3 gave (59, 1536) rows for 58 residues", text)
                    self.assertIn("1 protein(s) on this rank got no residue codes: their codes are written as empty "
                                  "[0, 8] tables", text)

    def test_many_proteins_without_codes(self):
        """The first 5 ids named, the rest counted."""
        records = [(f"s{k}", "MKVLA" + "A" * k) for k in range(8)]
        fasta = self.fasta(*records)

        def short(model, protein):
            return _esm3(model, protein)[:-1]

        with mock.patch.object(esm_module, "per_residue_embedding", side_effect=short), \
                self.assertLogs("foldmatch.modules.residue_payloads", "ERROR"):
            result = self.invoke("from-sequences", "residue", "--fasta-file", fasta, "--output-folder",
                                 self.folder(), "--output-codes-folder", self.folder("codes"))
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertTrue(str(result.exception).startswith(
            "8 sequence(s) got no residue codes: 's0', 's1', 's2', 's3', 's4' and 3 more. "), str(result.exception))
        self.assertEqual(result.exception.ids, tuple(f"s{k}" for k in range(8)))


# =====================================================================================================================
# fm-search reads the output folder of a chain run with residue codes
# =====================================================================================================================
class FmSearchTest(_Case):
    """The reason for the codes folder (SPEC2): the output folder of 'from-sequences chain --output-codes-folder' is
    read by 'fm-search build embeddings' and 'query embedding' as that of a plain run: the same index bytes, the same
    hits; with the codes folder elsewhere or inside the output folder (fm-search reads a folder, not its subfolders),
    even under a name that ends like an embedding file's (P9.X1: fm-search skips it, with a WARNING)."""

    def test_build_and_query_embeddings(self):
        for out_format in ("parquet", "csv", "pt"):
            plain = self.folder("plain")
            self.run_ok("chain", _FASTA, plain, "--output-format", out_format)
            db = self.folder("db") / "db"
            self.run_search("build", "embeddings", "--embedding-folder", plain, "--output-db", db)
            want = {suffix: Path(f"{db}{suffix}").read_bytes() for suffix in (".index", ".metadata")}
            self.run_search("query", "embedding", "--db-path", db, "--embedding-folder", plain, "--output-file",
                            db.parent / "hits.tsv", "--threshold", "0")
            hits = (db.parent / "hits.tsv").read_bytes()
            self.assertTrue(hits)
            for where in ("elsewhere", "inside", "inside, named like an embedding file"):
                with self.subTest(out_format=out_format, codes=where):
                    out = self.folder("out")
                    codes = {"elsewhere": self.folder("codes"), "inside": out / "codes"}.get(
                        where, out / f"codes.{out_format}")
                    codes.mkdir(exist_ok=True)
                    self.run_ok("chain", _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes)
                    self.assertTrue(list(codes.iterdir()))
                    db = self.folder("db") / "db"
                    result = db.parent / "hits.tsv"
                    with self.assertLogs("foldmatch.utils.data", "INFO") as logs:
                        logging.getLogger("foldmatch.utils.data").info("start")
                        self.run_search("build", "embeddings", "--embedding-folder", out, "--output-db", db)
                        self.run_search("query", "embedding", "--db-path", db, "--embedding-folder", out,
                                        "--output-file", result, "--threshold", "0")
                    for suffix in (".index", ".metadata"):
                        self.assertEqual(Path(f"{db}{suffix}").read_bytes(), want[suffix], suffix)
                    self.assertEqual(result.read_bytes(), hits)
                    skipped = [line for line in logs.output if "folder(s) named like embedding files skipped" in line]
                    self.assertEqual(len(skipped), 2 if codes.name.endswith(f".{out_format}") else 0, logs.output)


# =====================================================================================================================
# Refusals, before ESM3
# =====================================================================================================================
class RefusalTest(_Case):

    def args(self, command, fasta, out, codes, *extra):
        return ("from-sequences", command, "--fasta-file", fasta, "--output-folder", out, "--output-codes-folder",
                codes, *extra)

    def same_folder_refused(self, out, given):
        """W1: the run with --output-codes-folder ``given`` (one folder with ``out``) is a usage error, exit status 2
        (typer.BadParameter on --output-codes-folder), before anything is imported or written: the message names both
        options and the folder."""
        message = None
        for command in _COMMANDS:
            before = names(Path(out))
            with mock.patch.object(sequence_inference, "predict", side_effect=AssertionError("ran")), \
                    mock.patch.object(chain_complete_inference, "predict", side_effect=AssertionError("ran")), \
                    mock.patch.object(output_files, "check_outputs", side_effect=AssertionError("checked")):
                self.refusing = True
                try:
                    result = CliRunner().invoke(embedding.app, [str(a) for a in (*self.args(command, _FASTA, out,
                                                                                            given), *_CPU)],
                                                env={"COLUMNS": "2000"})
                finally:
                    self.refusing = False
            self.assertEqual(result.exit_code, 2, f"{result.output}\n{result.exception!r}")
            self.assertEqual(names(Path(out)), before)
            message = _flat(result.output)
            self.assertRegex(message, r"Invalid value for '?--output-codes-folder'?: --output-folder and "
                                      r"--output-codes-folder are one folder \(.+\): the residue codes go to a folder "
                                      r"of their own, so that the embedding folder holds embeddings only \('fm-search "
                                      r"build embeddings' and 'fm-search query embedding' read every \.parquet, \.pt "
                                      r"and \.csv file of it\)\. Give --output-codes-folder another folder \(a "
                                      r"subfolder of --output-folder will do\)\.")
        return message

    def test_the_codes_folder_is_the_output_folder(self):
        """V2: by path, through a symbolic link, as '.' (a relative path), by a case variant on a case-insensitive
        volume; compared as files."""
        out = self.folder()
        self.assertIn(f"({out})", self.same_folder_refused(out, out))
        link = self.tmp / "link"
        link.symlink_to(out, target_is_directory=True)
        self.same_folder_refused(out, link)
        self.same_folder_refused(link, out)
        with chdir(out):
            self.same_folder_refused(out, ".")
        with chdir(out.parent):
            self.same_folder_refused(out, out.name)
            self.same_folder_refused(Path(out.name), out)

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_the_codes_folder_is_the_output_folder_in_another_case(self):
        out = self.folder("Out")
        variant = out.parent / out.name.swapcase()
        self.assertNotEqual(str(variant), str(out))
        self.same_folder_refused(out, variant)

    def test_the_same_folder_before_torch(self):
        """W1: the usage error comes before torch is imported (a fresh process)."""
        out = self.folder()
        code = ("import sys; from typer.testing import CliRunner; from foldmatch.cli import embedding; "
                "r = CliRunner().invoke(embedding.app, sys.argv[1:]); print(r.exit_code, 'torch' in sys.modules)")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([_SRC] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
        for command in _COMMANDS:
            with self.subTest(command=command):
                run = subprocess.run([sys.executable, "-c", code, *map(str, self.args(command, _FASTA, out, out))],
                                     env=env, capture_output=True, text=True, timeout=600)
                self.assertEqual(run.stdout.split(), ["2", "False"], run.stderr[-4000:])

    def test_a_folder_named_as_a_file_of_the_other(self):
        """One folder inside the other is allowed (V2), but not under the name of a file that the other's output
        writes (<id>.csv|pt, <name>-<rank>.parquet|json.gz), as that folder compares names: refused before ESM3 (exit
        status 1), not by the writer after the pass."""
        _, _, ids, _ = _f2c()
        for command in _COMMANDS:
            for out_format, name in (("csv", f"{ids[0]}.csv"), ("pt", f"{ids[1]}.pt"),
                                     ("parquet", "inference-0.parquet"), ("json", "inference-1.json.gz")):
                with self.subTest(command=command, out_format=out_format, codes="inside the output folder"):
                    out = self.folder()
                    codes = out / name
                    codes.mkdir()
                    message = self.refused(*self.args(command, _FASTA, out, codes, "--output-format", out_format),
                                           written=[out, codes])
                    self.assertTrue(message.startswith(f"The codes folder {codes} is named as one of the embedding "
                                                       f"files of the output folder {out} ("), message)
                    self.assertTrue(message.endswith(f"), which could not be written over a folder: give a folder "
                                                     f"of another name, or one outside {out}."), message)
                with self.subTest(command=command, out_format=out_format, codes="around the output folder"):
                    codes = self.folder("codes")
                    out = codes / name.replace("inference-", "inference-codes-")
                    out.mkdir()
                    message = self.refused(*self.args(command, _FASTA, out, codes, "--output-format", out_format),
                                           written=[out, codes])
                    self.assertTrue(message.startswith(f"The output folder {out} is named as one of the code files "
                                                       f"of the codes folder {codes} ("), message)
        out = self.folder()
        (out / f"{ids[0]}.pt").mkdir()   # a file name of another format is no output file
        self.run_ok("chain", _FASTA, out, "--output-format", "csv", "--output-codes-folder", out / f"{ids[0]}.pt")
        if _CASE_INSENSITIVE:
            with self.subTest(codes="a case variant on a case-insensitive volume"):
                out = self.folder()
                codes = out / f"{ids[0].swapcase()}.csv"
                codes.mkdir()
                message = self.refused(*self.args("chain", _FASTA, out, codes), written=[out, codes])
                self.assertIn(f"(the output {ids[0]!r})", message)
        with self.subTest(codes="through a symbolic link to such a folder"):
            out, elsewhere = self.folder(), self.folder("elsewhere")
            (out / f"{ids[1]}.csv").mkdir()
            (elsewhere / "link").symlink_to(out / f"{ids[1]}.csv", target_is_directory=True)
            self.refused(*self.args("residue", _FASTA, out, elsewhere / "link"), written=[out])

    def test_an_unusable_codes_name(self):
        """Checked whatever the format, as --output-name is."""
        for command in _COMMANDS:
            for out_format in _FORMATS:
                for name, problem in (("a/b", "holds '/'"), ("..", "is '..'"),
                                      ("x" * 241, "makes a file name of 256 bytes once a rank and a suffix are added")):
                    with self.subTest(command=command, out_format=out_format, name=name[:5]):
                        out, codes = self.folder(), self.folder("codes")
                        message = self.refused(*self.args(command, _FASTA, out, codes, "--output-format", out_format,
                                                          "--output-codes-name", name), written=[out, codes])
                        self.assertEqual(message, f"The codes name {name!r} {problem}: it names the files "
                                                  f"<name>-<rank>.parquet or <name>-<rank>.json.gz of the codes "
                                                  f"folder. Give a name that is not empty, without '/' or NUL, other "
                                                  f"than '.' and '..', of at most 240 bytes in UTF-8.")
        self.prepare.assert_not_called()

    def test_ids_that_the_codes_folder_takes_for_one_file(self):
        """csv and pt: the ids are checked as the codes folder compares names, which can differ from the output
        folder (a case-folding codes folder beside a case-sensitive output folder; probe_folding stubbed)."""
        fasta = self.fasta(("a", "MKVLA"), ("A", "MKVL"))
        for out_format in ("csv", "pt"):
            with self.subTest(out_format=out_format):
                out, codes = self.folder(), self.folder("codes")
                folding = lambda folder: (Folding(case=True, unicode=True, full=True)  # noqa: E731
                                          if same_folder(folder, codes) else EXACT)
                with mock.patch.object(output_files, "probe_folding", side_effect=folding):
                    message = self.refused(*self.args("chain", fasta, out, codes, "--output-format", out_format),
                                           written=[out, codes])
                self.assertEqual(message, f"{fasta}: 1 file(s) of the codes folder would each be written for more "
                                          f"than one FASTA id (the codes folder does not tell upper and lower case "
                                          f"apart): 'a' (record 1) and 'A' (record 2) give a.{out_format}. Each one "
                                          f"would replace the one before, as the folder takes those names for one: "
                                          f"write them to a folder of a file system that tells upper and lower case "
                                          f"apart or as parquet or json (which keep each FASTA id as given), or give "
                                          f"those records FASTA ids that differ otherwise.")

    def test_the_fasta_file_among_the_code_files(self):
        _, _, ids, _ = _f2c()
        for out_format, name in (("csv", f"{ids[0]}.csv"), ("pt", f"{ids[1]}.pt"),
                                 ("parquet", "inference-codes-0.parquet"), ("json", "inference-codes-3.json.gz")):
            with self.subTest(out_format=out_format):
                out, codes = self.folder(), self.folder("codes")
                fasta = codes / name
                shutil.copyfile(_FASTA, fasta)
                message = self.refused(*self.args("residue", fasta, out, codes, "--output-format", out_format),
                                       written=[out, codes])
                self.assertIn(f"1 input(s) of this run would be replaced by its output, in the codes folder {codes}: "
                              f"the FASTA file {fasta} by", message)

    def test_fasta_the_residue_pass_cannot_take(self):
        """The strict pre-flight of a residue pass, only with --output-codes-folder: before the bundle and ESM3."""
        for records, what in (((("a", "MK<mask>V"),), "contain '<' or '>'"), ((("a", "MKÉV"),), "non-ASCII")):
            fasta = self.fasta(*records)
            for command in _COMMANDS:
                for out_format in ("csv", "parquet"):
                    with self.subTest(what=what, command=command, out_format=out_format):
                        message = self.refused(*self.args(command, fasta, self.folder(), self.folder("codes"),
                                                          "--output-format", out_format))
                        self.assertIn(what, message)
                        self.run_ok(command, fasta, self.folder(), "--output-format", out_format)
        self.prepare.assert_not_called()

    def test_a_repeated_id(self):
        fasta = self.fasta(("a", "MKV"), ("a", "MKVL"))
        message = self.refused(*self.args("chain", fasta, self.folder(), self.folder("codes"), "--output-format",
                                          "parquet"))
        self.assertIn("1 record(s) repeat an earlier id (e.g. a)", message)

    def test_a_bundle_that_cannot_be_had(self):
        with mock.patch.object(ss, "prepare", side_effect=ss.BundleUnavailableError("not cached")):
            for command in _COMMANDS:
                with self.subTest(command=command):
                    message = self.refused(*self.args(command, _FASTA, self.folder(), self.folder("codes")))
                    self.assertEqual(message, "not cached")

    def test_only_the_process_that_starts_the_run_fetches_the_bundle(self):
        """The command of one process fetches the bundle; the pre-flight of every other rank finds it cached
        (bundle_fetch_allowed, as in builds)."""
        for command in _COMMANDS:
            with self.subTest(command=command):
                self.prepare.reset_mock()
                self.run_ok(command, _FASTA, self.folder(), "--output-format", "parquet", "--output-codes-folder",
                            self.folder("codes"))
                self.prepare.assert_called_once_with(SYSTEM, local_files_only=False)
        for env, num_nodes, local_only in (({}, 1, False), ({}, 2, True), ({"LOCAL_RANK": "1"}, 1, True),
                                           ({"WORLD_SIZE": "2", "RANK": "0"}, 1, True)):
            with self.subTest(env=env, num_nodes=num_nodes), launch_env(**env):
                self.prepare.reset_mock()
                chain_complete_inference.residue_codes_pass(_FASTA, self.folder("codes"), num_nodes=num_nodes)
                self.prepare.assert_called_once_with(SYSTEM, local_files_only=local_only)

    def test_a_refusal_is_a_refusal_error(self):
        """D20: every check of a run with --output-codes-folder exits 1 with a RefusalError, ValueErrors of the
        library included; the same FASTA without --output-codes-folder runs."""
        fasta = self.fasta(("a", "MK>V"))
        result = self.invoke(*self.args("residue", fasta, self.folder(), self.folder("codes")))
        self.assertEqual(result.exit_code, 1)
        self.assertIsInstance(result.exception, RefusalError)
        self.assertIsInstance(result.exception.__cause__, ValueError)


# =====================================================================================================================
# Files of the other kind (Z1-Z5): embeddings in the codes folder, residue codes in the output folder
# =====================================================================================================================
_CIF = _TEST_PATH / "resources" / "pdb" / "1acb.cif"
_SUFFIX = {"csv": "csv", "pt": "pt", "parquet": "parquet", "json": "json.gz"}


class OtherKindTest(_Case):
    """A run refuses, before ESM3 and before any file is written (RefusalError, exit status 1), to replace a file of
    the other kind: an embedding file in its codes folder, a residue-code file in its output folder, with or without
    --output-codes-folder, in every fm-embedding command. Only the files that the run would replace are read; files
    of no kind (empty, cut short, being written) are replaced as before; a rerun replaces its own files."""

    def setUp(self):
        super().setUp()
        # The models of the structure commands, never built here: every run with them is refused
        for module in ("esm_inference", "assembly_complete_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_residue_model",
                                         side_effect=AssertionError("a model was built")))
        self.enterContext(mock.patch("foldmatch.inference.assembly_complete_inference.get_aggregator_model",
                                     side_effect=AssertionError("a model was built")))

    def args(self, command, fasta, out, codes, *extra):
        return ("from-sequences", command, "--fasta-file", fasta, "--output-folder", out, "--output-codes-folder",
                codes, *extra)

    def code_files(self, out_format, *extra):
        """A codes folder that a run with --output-codes-folder filled (its output folder too), and the ids."""
        _, _, ids, _ = _f2c()
        out, codes = self.folder("emb"), self.folder("codes")
        self.run_ok("chain", _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes, *extra)
        return out, codes, ids

    def contents(self, *folders) -> dict:
        return {path: path.read_bytes() for folder in folders for path in sorted(folder.iterdir())}

    def test_an_embedding_folder_as_the_codes_folder(self):
        """Friday's accident: --output-codes-folder is the output folder of an earlier run (csv and pt: the code
        files would be <id>.<format> there, its embeddings): refused, nothing written, the embeddings intact."""
        _, _, ids, _ = _f2c()
        for command in _COMMANDS:
            for out_format in ("pt", "csv"):
                with self.subTest(command=command, out_format=out_format):
                    earlier, out = self.folder("earlier"), self.folder("out")
                    self.run_ok(command, _FASTA, earlier, "--output-format", out_format)
                    before = self.contents(earlier)
                    message = self.refused(*self.args(command, _FASTA, out, earlier, "--output-format", out_format),
                                           written=[earlier, out])
                    self.assertEqual(self.contents(earlier), before)
                    files = sorted(earlier / f"{i}.{out_format}" for i in ids)
                    self.assertEqual(message, (
                        f"2 file(s) of the codes folder {earlier} that this run would replace hold embeddings, not "
                        f"residue codes: {files[0]} (an embedding); {files[1]} (an embedding). The residue codes go "
                        f"to a folder of their own: give a codes folder for the codes only, without those embedding "
                        f"files, or move them out of it."))
        self.prepare.assert_not_called()

    def test_a_codes_folder_as_the_output_folder(self):
        """The reverse: a plain run (or one with codes) whose --output-folder holds the code files that it would
        replace: refused in every format of both commands; with other names (parquet, json) the folder is mixed and
        the run allowed."""
        for out_format in _FORMATS:
            _, codes, ids = self.code_files(out_format, "--output-codes-name", "inference")
            for command in _COMMANDS:
                with self.subTest(command=command, out_format=out_format):
                    before = self.contents(codes)
                    message = self.refused("from-sequences", command, "--fasta-file", _FASTA, "--output-folder",
                                           codes, "--output-format", out_format, written=[codes])
                    self.assertEqual(self.contents(codes), before)
                    many = out_format in ("csv", "pt")
                    files = sorted(f"{codes / i}.{out_format}" for i in ids) if many else \
                        [f"{codes}/inference-0.{_SUFFIX[out_format]}"]
                    self.assertEqual(message, (
                        f"{len(files)} file(s) of the output folder {codes} that this run would replace hold residue "
                        f"codes, as 'fm-embedding from-sequences ... --output-codes-folder' writes them, not "
                        f"embeddings: {'; '.join(f'{f} (residue codes)' for f in files)}. The embeddings and the "
                        f"residue codes go to folders of their own: give an output folder without those residue-code "
                        f"files, or move them out of it."))
                    other = self.folder("codes")
                    self.refused(*self.args(command, _FASTA, codes, other, "--output-format", out_format),
                                 written=[codes, other])
                    if not many:
                        self.run_ok(command, _FASTA, codes, "--output-format", out_format, "--output-name", "emb")

    def test_the_parquet_codes_name(self):
        """parquet and json: --output-codes-name equal to the earlier embeddings' name in the codes folder is refused;
        the default name writes beside them (a mixed folder: nothing is replaced, nothing lost)."""
        _, reference, ids, _ = _f2c()
        for out_format in ("parquet", "json"):
            for command in _COMMANDS:
                with self.subTest(command=command, out_format=out_format):
                    earlier, out = self.folder("earlier"), self.folder("out")
                    self.run_ok(command, _FASTA, earlier, "--output-format", out_format)
                    before = self.contents(earlier)
                    message = self.refused(*self.args(command, _FASTA, out, earlier, "--output-format", out_format,
                                                      "--output-codes-name", "inference"), written=[earlier, out])
                    self.assertIn(f"1 file(s) of the codes folder {earlier} that this run would replace hold "
                                  f"embeddings, not residue codes: {earlier}/inference-0.{_SUFFIX[out_format]} "
                                  f"(embeddings).", message)
                    self.run_ok(command, _FASTA, out, "--output-format", out_format, "--output-codes-folder",
                                earlier)
                    self.assertEqual(names(earlier), sorted([f"inference-0.{_SUFFIX[out_format]}",
                                                             f"inference-codes-0.{_SUFFIX[out_format]}"]))
                    self.assertEqual({p: d for p, d in self.contents(earlier).items() if p in before}, before)
                    got = read_codes(earlier, out_format)
                    for dom_id in ids:
                        np.testing.assert_array_equal(got[dom_id], reference[dom_id])
        # csv: embedding files of other ids in the codes folder are not looked at
        earlier, out = self.folder("earlier"), self.folder("out")
        fasta = self.fasta(("other", "MKVLA"))
        self.run_ok("chain", fasta, earlier, "--output-format", "csv")
        self.run_ok("chain", _FASTA, out, "--output-format", "csv", "--output-codes-folder", earlier)
        self.assertEqual(names(earlier), sorted(["other.csv"] + [f"{i}.csv" for i in ids]))

    def test_a_rerun_replaces_its_files(self):
        """A rerun into the same folders, every format of both commands: allowed, every file written again."""
        for command in _COMMANDS:
            for out_format in _FORMATS:
                with self.subTest(command=command, out_format=out_format):
                    out, codes = self.folder("out"), self.folder("codes")
                    self.run_ok(command, _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes)
                    files = sorted([*out.iterdir(), *codes.iterdir()])
                    for path in files:
                        os.utime(path, ns=(0, 0))
                    self.run_ok(command, _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes)
                    self.assertEqual(sorted([*out.iterdir(), *codes.iterdir()]), files)
                    for path in files:
                        self.assertGreater(path.stat().st_mtime_ns, 0, path)
                    self.run_ok(command, _FASTA, out, "--output-format", out_format)

    def test_files_of_no_kind_are_replaced(self):
        """Z4: a file cut short (as another rank of the run may be writing it) or of no kind (empty, foreign) is
        replaced, never refused, in both folders; an empty code csv (a table without residues) too."""
        _, _, ids, _ = _f2c()
        for out_format in _FORMATS:
            out, codes, _ = self.code_files(out_format)
            whole = self.contents(out, codes)
            for cut in (0, 1, 7, 30, 0.5, -1):
                with self.subTest(out_format=out_format, cut=cut):
                    for path, data in whole.items():
                        size = int(len(data) * cut) if isinstance(cut, float) else \
                            min(cut, len(data)) if cut >= 0 else len(data) + cut
                        path.write_bytes(data[:size])
                    self.run_ok("chain", _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes)
                    self.assertEqual(sorted(self.contents(out, codes)), sorted(whole))
        for out_format in ("csv", "pt"):
            with self.subTest(out_format=out_format, files="foreign"):
                out, codes = self.folder("out"), self.folder("codes")
                for folder in (out, codes):
                    (folder / f"{ids[0]}.{out_format}").write_bytes(b"not a FoldMatch file\n")
                    (folder / f"{ids[1]}.{out_format}").write_bytes(b"")
                self.run_ok("chain", _FASTA, out, "--output-format", out_format, "--output-codes-folder", codes)

    def test_another_rank_writes_while_the_check_reads(self):
        """Z4: the files that another rank of the same run is writing when this rank's check reads them: of the kind
        that the run writes there, or of none yet (truncated, then written in parts), so never refused. The check is
        made to read each file while it is being rewritten, at several points of the write."""
        reference = {}
        for out_format in _FORMATS:
            out, codes, _ = self.code_files(out_format)
            reference[out_format] = (out, codes, self.contents(out, codes))
        real = output_files.file_kind
        for out_format, (out, codes, whole) in reference.items():
            for share in (0.0, 0.1, 0.5, 0.9, 1.0):
                with self.subTest(out_format=out_format, written=share):
                    seen = []

                    def written_meanwhile(path, _whole=whole, _share=share):
                        data = _whole[Path(path)]
                        Path(path).write_bytes(data[:int(len(data) * _share)])
                        seen.append(real(path))
                        return seen[-1]

                    with mock.patch.object(output_files, "file_kind", side_effect=written_meanwhile):
                        self.run_ok("chain", _FASTA, out, "--output-format", out_format, "--output-codes-folder",
                                    codes)
                    self.assertTrue(seen)
                    self.assertTrue(set(seen) <= {None, "embeddings", "residue codes"})

    def test_parquet_files_of_other_ranks(self):
        """parquet and json: the files <name>-<rank> of every rank are among the files a run would replace (another
        rank's file of the other kind is refused; one being written, with no footer yet, is not)."""
        out, codes, _ = self.code_files("parquet")
        (codes / "inference-codes-0.parquet").rename(codes / "inference-codes-3.parquet")
        shutil.copyfile(out / "inference-0.parquet", codes / "inference-codes-5.parquet")
        message = self.refused(*self.args("chain", _FASTA, out, codes, "--output-format", "parquet"),
                               written=[out, codes])
        self.assertIn(f"1 file(s) of the codes folder {codes} that this run would replace hold embeddings, not residue "
                      f"codes: {codes}/inference-codes-5.parquet (embeddings).", message)
        data = (out / "inference-0.parquet").read_bytes()
        (codes / "inference-codes-5.parquet").write_bytes(data[:len(data) - 8])   # no footer yet
        self.run_ok("chain", _FASTA, out, "--output-format", "parquet", "--output-codes-folder", codes)

    def test_every_command_refuses_code_files_in_its_output_folder(self):
        """The output-folder half in every fm-embedding command that writes embedding files, and in the legacy
        'inference' command: code files under the names of the files it would write (<structure>.<chain>.<format>,
        <structure>-<assembly>.<format>, <name>.<format>, <output-name>-<rank>.<format>) are refused before any
        model; code files under other names are not looked at."""
        from foldmatch.cli import inference as legacy
        src = self.folder("src")
        shutil.copyfile(_CIF, src / "1acb.cif")
        chain_list = self.tmp / "chains.csv"
        chain_list.write_text(f"1acb,{_CIF},A,x\n")
        commands = {
            ("from-structures", "residue"): "1acb.A", ("from-structures", "chain"): "1acb.B",
            ("from-structures", "assembly"): "1acb-1", ("from-sequences", "residue"): _f2c()[2][0],
            ("from-sequences", "chain"): _f2c()[2][1], ("inference", "residue-embedding"): "x",
        }
        for out_format in _FORMATS:
            _, codes, ids = self.code_files(out_format, "--output-codes-name", "inference")
            one = (codes / f"{ids[0]}.{out_format}") if out_format in ("csv", "pt") else \
                codes / f"inference-0.{_SUFFIX[out_format]}"
            for (group, command), name in commands.items():
                with self.subTest(command=f"{group} {command}", out_format=out_format):
                    out = self.folder("out")
                    target = out / (f"{name}.{out_format}" if out_format in ("csv", "pt") else one.name)
                    shutil.copyfile(one, target)
                    if group == "from-structures":
                        args = (group, command, "--src-folder", src)
                    elif group == "from-sequences":
                        args = (group, command, "--fasta-file", _FASTA)
                    else:
                        args = (command, "--src-file", chain_list)
                    args += ("--output-folder", out, "--output-format", out_format)
                    before = self.contents(out)
                    self.refusing = True
                    try:
                        result = CliRunner().invoke(legacy.app if group == "inference" else embedding.app,
                                                    [str(a) for a in (*args, *_CPU)])
                    finally:
                        self.refusing = False
                    self.assertEqual(result.exit_code, 1, f"{result.output}\n{result.exception!r}")
                    self.assertIsInstance(result.exception, RefusalError, repr(result.exception))
                    self.assertTrue(str(result.exception).startswith(
                        f"1 file(s) of the output folder {out} that this run would replace hold residue codes"),
                        str(result.exception))
                    self.assertIn(f"{target} (residue codes)", str(result.exception))
                    self.assertEqual(self.contents(out), before)

    def test_the_library(self):
        """Both predict functions refuse either kind, with the codes or without; the other library entry points that
        write embeddings refuse code files in their output folder."""
        from foldmatch.inference import assembly_complete_inference, esm_inference
        from foldmatch.types.api_types import SrcLocation, SrcProteinFrom
        for out_format in (OutFormat.csv, OutFormat.pt, OutFormat.parquet, OutFormat.json):
            fmt = out_format.value
            earlier, codes, ids = self.code_files(fmt, "--output-codes-name", "inference")
            self.refusing = True
            files = sorted(earlier / f"{i}.{fmt}" for i in ids) if fmt in ("csv", "pt") else \
                [earlier / f"inference-0.{_SUFFIX[fmt]}"]
            for predict, kwargs in ((sequence_inference.predict, {"fasta_file": _FASTA}),
                                    (chain_complete_inference.predict, {"src_stream": _FASTA,
                                                                        "src_from": SrcEsmFrom.fasta})):
                with self.subTest(predict=predict.__module__, out_format=fmt):
                    out = self.folder("out")
                    before = self.contents(earlier, codes)
                    # Embeddings in the codes folder (the codes named as the embeddings in parquet and json)
                    with self.assertRaises(RefusalError) as refused:
                        predict(out_folder=out, out_codes_folder=earlier, out_format=out_format,
                                out_codes_name="inference", **kwargs)
                    self.assertTrue(str(refused.exception).startswith(
                        f"{len(files)} file(s) of the codes folder {earlier} that this run would replace hold "
                        f"embeddings, not residue codes: {files[0]} ("), str(refused.exception))
                    # Residue codes in the output folder, with and without the codes
                    for extra in ({}, {"out_codes_folder": self.folder("codes")}):
                        with self.assertRaises(RefusalError) as refused:
                            predict(out_folder=codes, out_format=out_format, **extra, **kwargs)
                        self.assertTrue(str(refused.exception).startswith(
                            f"{len(files)} file(s) of the output folder {codes} that this run would replace hold "
                            f"residue codes"), str(refused.exception))
                    self.assertEqual(self.contents(earlier, codes), before)
                    self.assertEqual(names(out), [])
            with self.subTest(predict="esm_inference and assembly_complete_inference", out_format=fmt):
                out = self.folder("out")
                for name in ("1acb.A", "1acb-1"):
                    if fmt in ("csv", "pt"):
                        shutil.copyfile(codes / f"{ids[0]}.{fmt}", out / f"{name}.{fmt}")
                    else:
                        shutil.copyfile(codes / f"inference-0.{_SUFFIX[fmt]}", out / f"inference-0.{_SUFFIX[fmt]}")
                stream = [("1acb", str(_CIF), "1acb")]
                with self.assertRaisesRegex(RefusalError, "hold residue codes"):
                    esm_inference.predict(src_stream=stream, src_location=SrcLocation.stream,
                                          src_from=SrcProteinFrom.structure, out_folder=out, out_format=out_format)
                with self.assertRaisesRegex(RefusalError, "hold residue codes"):
                    assembly_complete_inference.predict(src_stream=stream, src_location=SrcLocation.stream,
                                                        out_folder=out, out_format=out_format)


    def test_the_library_reads_the_files_of_the_records_kept(self):
        """csv and pt: the library's checks read the files of the records that the run keeps (min_res_n), the files
        it would replace: another kind under the name of a record left out is not looked at (allowed, untouched);
        under the name of a record kept it is refused."""
        from foldmatch.utils import file_kinds
        from foldmatch.utils.sequence import scan_fasta
        lengths = {name: length for _, name, length in scan_fasta(_FASTA)}
        kept, left = max(lengths, key=lengths.get), min(lengths, key=lengths.get)
        min_res_n = lengths[left] + 1
        for fmt in ("csv", "pt"):
            out_format = OutFormat(fmt)
            earlier, codes, _ = self.code_files(fmt)
            for predict, kwargs in ((sequence_inference.predict, {"fasta_file": _FASTA}),
                                    (chain_complete_inference.predict, {"src_stream": _FASTA,
                                                                        "src_from": SrcEsmFrom.fasta})):
                with self.subTest(predict=predict.__module__, out_format=fmt):
                    out, out_codes = self.folder("out"), self.folder("codes")
                    shutil.copyfile(codes / f"{left}.{fmt}", out / f"{left}.{fmt}")
                    shutil.copyfile(earlier / f"{left}.{fmt}", out_codes / f"{left}.{fmt}")
                    before = self.contents(out, out_codes)
                    self.refusing = False
                    predict(out_folder=out, out_codes_folder=out_codes, out_format=out_format, min_res_n=min_res_n,
                            accelerator=Accelerator.cpu, devices=1, **kwargs)
                    self.assertEqual(names(out), sorted([f"{kept}.{fmt}", f"{left}.{fmt}"]))
                    self.assertEqual({p: d for p, d in self.contents(out, out_codes).items() if p in before}, before)
                    self.assertEqual(file_kinds.file_kind(out / f"{kept}.{fmt}"), file_kinds.EMBEDDINGS)
                    for folder, source, kind in ((out, codes, "residue codes"), (out_codes, earlier, "embeddings")):
                        shutil.copyfile(source / f"{kept}.{fmt}", folder / f"{kept}.{fmt}")
                        self.refusing = True
                        with self.assertRaisesRegex(RefusalError, f"^1 file\\(s\\) .* hold {kind}"):
                            predict(out_folder=out, out_codes_folder=out_codes, out_format=out_format,
                                    min_res_n=min_res_n, **kwargs)
                        os.remove(folder / f"{kept}.{fmt}")

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_case_variants_of_the_files(self):
        """On a case-insensitive volume, the files that the folder takes for the run's (another case of the id, of
        the suffix) are the files it would replace: of the other kind, refused, named as they are on disk."""
        _, _, ids, _ = _f2c()
        for out_format in ("csv", "pt"):
            out, codes, _ = self.code_files(out_format)
            upper = out_format.upper()
            with self.subTest(out_format=out_format, folder="codes"):
                earlier = self.folder("earlier")
                for dom_id in ids:
                    shutil.copyfile(out / f"{dom_id}.{out_format}", earlier / f"{dom_id.upper()}.{upper}")
                before = self.contents(earlier)
                message = self.refused(*self.args("chain", _FASTA, self.folder("out"), earlier, "--output-format",
                                                  out_format), written=[earlier])
                files = sorted(f"{earlier / dom_id.upper()}.{upper}" for dom_id in ids)
                self.assertIn(f"2 file(s) of the codes folder {earlier} that this run would replace hold embeddings, "
                              f"not residue codes: {files[0]} (an embedding); {files[1]} (an embedding).", message)
                self.assertEqual(self.contents(earlier), before)
            with self.subTest(out_format=out_format, folder="output"):
                folder = self.folder("codes")
                shutil.copyfile(codes / f"{ids[0]}.{out_format}", folder / f"{ids[0].swapcase()}.{upper}")
                message = self.refused("from-sequences", "chain", "--fasta-file", _FASTA, "--output-folder", folder,
                                       "--output-format", out_format, written=[folder])
                self.assertIn(f"1 file(s) of the output folder {folder} that this run would replace hold residue "
                              f"codes", message)
                self.assertIn(f"{folder / ids[0].swapcase()}.{upper} (residue codes)", message)

    def test_each_file_is_read_once_per_run(self):
        """csv and pt: the command checks both folders before it calls the library, which checks them again; each
        file is read once in all (file_kinds.remembering), and nothing is remembered once the library has checked."""
        from foldmatch.utils import file_kinds
        for out_format in ("csv", "pt"):
            out, codes, ids = self.code_files(out_format)
            for command in _COMMANDS:
                with self.subTest(command=command, out_format=out_format):
                    file_kinds._KNOWN.clear()
                    reader = mock.Mock(wraps=file_kinds._KINDS[f".{out_format}"])
                    with mock.patch.dict(file_kinds._KINDS, {f".{out_format}": reader}):
                        self.run_ok(command, _FASTA, out, "--output-format", out_format, "--output-codes-folder",
                                    codes)
                    read = sorted(Path(c.args[0]).relative_to(self.tmp) for c in reader.call_args_list)
                    self.assertEqual(read, sorted(Path(folder).relative_to(self.tmp) / f"{i}.{out_format}"
                                                  for folder in (out, codes) for i in ids))
                    self.assertEqual(file_kinds._KNOWN, {})

# =====================================================================================================================
# The library
# =====================================================================================================================
class LibraryTest(_Case):

    def test_predictions_with_codes(self):
        """Y2: with out_codes_folder each prediction is (embeddings, ids, codes, seq_sha1s), the codes one CPU uint8
        tensor [L, 8] per id, the object of its .pt code file and the bytes of its row in the codes Parquet file; the
        embeddings are those of a plain run, whose predictions stay (embeddings, ids), without any conversion."""
        _, reference, ids, sequences = _f2c()
        converted = mock.patch.object(chain_complete_inference, "code_predictions",
                                      side_effect=chain_complete_inference.code_predictions)
        for predict, kwargs in ((sequence_inference.predict, {"fasta_file": _FASTA}),
                                (chain_complete_inference.predict, {"src_stream": _FASTA,
                                                                    "src_from": SrcEsmFrom.fasta})):
            with converted as conversion:
                plain = predict(accelerator=Accelerator.cpu, devices=1, batch_size=2, **kwargs)
            conversion.assert_not_called()
            self.assertEqual(len(plain), 1)
            self.assertEqual(len(plain[0]), 2)
            self.assertEqual(plain[0][1], tuple(ids))
            for out_format in (None, OutFormat.pt, OutFormat.parquet, OutFormat.json):
                with self.subTest(predict=predict.__module__, out_format=out_format):
                    codes = self.folder("codes")
                    # Without out_folder (no embedding files) and the default format, csv; else with both folders
                    given = {} if out_format is None else {"out_folder": self.folder(), "out_format": out_format}
                    with converted as conversion:
                        (embeddings, names_, tables, digests), = predict(
                            accelerator=Accelerator.cpu, devices=1, batch_size=2, out_codes_folder=codes, **given,
                            **kwargs)
                    conversion.assert_called_once()
                    self.assertEqual(names_, tuple(ids))
                    self.assertEqual(digests, tuple(_sha1(s) for s in sequences))
                    self.assertIsInstance(tables, tuple)
                    for k, dom_id in enumerate(ids):
                        table = tables[k]
                        self.assertIsInstance(table, torch.Tensor)
                        self.assertEqual((table.dtype, table.device.type, tuple(table.shape)),
                                         (torch.uint8, "cpu", (len(sequences[k]), 8)))
                        self.assertTrue(table.is_contiguous())
                        np.testing.assert_array_equal(table.numpy(), reference[dom_id])
                        self.assertTrue(torch.equal(plain[0][0][k], embeddings[k]))
                    if out_format in (None, OutFormat.json):
                        fmt = "csv" if out_format is None else "json"
                        if out_format is None:
                            self.assertEqual(names(codes), sorted(f"{i}.csv" for i in ids))
                        stored = read_codes(codes, fmt, ids=ids)
                        self.assertEqual(sorted(stored), sorted(ids))
                        for dom_id, table in zip(ids, tables):
                            np.testing.assert_array_equal(stored[dom_id], table.numpy())
                    elif out_format == OutFormat.pt:
                        for dom_id, table in zip(ids, tables):
                            stored = torch.load(codes / f"{dom_id}.pt", weights_only=True)
                            self.assertEqual((stored.dtype, stored.shape), (table.dtype, table.shape))
                            self.assertTrue(torch.equal(stored, table))
                    else:
                        parquet = pq.read_table(codes / "inference-codes-0.parquet")
                        self.assertEqual(parquet.column("id").to_pylist(), list(ids))
                        column = parquet.column("residue_codes").combine_chunks()
                        for k, table in enumerate(tables):
                            stored = column[k].values.flatten().to_numpy(zero_copy_only=False).tobytes()
                            self.assertEqual(table.numpy().tobytes(), stored)
                        self.assertEqual(parquet.column("seq_sha1").to_pylist(), list(digests))

    def test_the_shard_pass_returns_its_payloads_as_bytes(self):
        """Y2 leaves the shard pass of a build (residue_output) as it is: its predictions hold the raw payloads that
        its shards store, bytes, not converted."""
        _, reference, ids, _ = _f2c()
        out = self.folder()
        with mock.patch.object(chain_complete_inference, "code_predictions") as conversion:
            (_, names_, payloads, _), = chain_complete_inference.predict(
                src_stream=_FASTA, src_from=SrcEsmFrom.fasta, accelerator=Accelerator.cpu, devices=1, batch_size=2,
                out_folder=out, out_format=OutFormat.parquet, residue_output=rs.RESIDUE_CODES)
        conversion.assert_not_called()
        self.assertEqual(names_, tuple(ids))
        stored = pq.read_table(out / "inference-0.parquet").column(rs.RESIDUE_CODES).to_pylist()
        for k, dom_id in enumerate(ids):
            self.assertIsInstance(payloads[k], bytes)
            self.assertEqual(payloads[k], reference[dom_id].tobytes())
            self.assertEqual(payloads[k], stored[k])

    def test_library_refusals_before_esm3(self):
        self.refusing = True
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as codes:
            with self.assertRaisesRegex(ValueError, "out_codes_name and scoring_system are only used with "
                                                    "out_codes_folder, which is not given"):
                sequence_inference.predict(fasta_file=_FASTA, out_folder=tmp, out_codes_name="c")
            with self.assertRaisesRegex(ValueError, "out_codes_name names the residue-code files of out_codes_folder, "
                                                    "which is not given"):
                chain_complete_inference.predict(src_stream=_FASTA, src_from=SrcEsmFrom.fasta, out_folder=tmp,
                                                 out_codes_name="c")
            with self.assertRaisesRegex(ValueError, "pass one or the other"):
                chain_complete_inference.predict(src_stream=_FASTA, src_from=SrcEsmFrom.fasta, out_folder=tmp,
                                                 out_format=OutFormat.parquet, out_codes_folder=codes,
                                                 residue_output=rs.RESIDUE_CODES)
            with self.assertRaisesRegex(ValueError, "residue codes needs sequence \\(FASTA\\) input"):
                chain_complete_inference.predict(src_stream=[("1acb", str(_TEST_PATH / "resources/pdb/1acb.cif"),
                                                              "1acb")], src_location="stream", out_codes_folder=codes)
            for predict, kwargs in ((sequence_inference.predict, {"fasta_file": _FASTA}),
                                    (chain_complete_inference.predict, {"src_stream": _FASTA,
                                                                        "src_from": SrcEsmFrom.fasta})):
                with self.subTest(predict=predict.__module__):
                    # W1: a ValueError, as the library's other argument errors (the CLI's is a usage error)
                    with self.assertRaises(ValueError) as refused:
                        predict(out_folder=tmp, out_codes_folder=os.path.join(tmp, "."), **kwargs)
                    self.assertNotIsInstance(refused.exception, RefusalError)
                    self.assertTrue(str(refused.exception).startswith(
                        f"out_folder and out_codes_folder are one folder ({os.path.join(tmp, '.')}): the residue "
                        f"codes go to a folder of their own"), str(refused.exception))
                    # Compared as files: a symbolic link to the output folder is that folder (the library has no
                    # typer resolve_path), on any file system
                    link = os.path.join(codes, "link")
                    if not os.path.islink(link):
                        os.symlink(tmp, link, target_is_directory=True)
                    with self.assertRaises(ValueError) as refused:
                        predict(out_folder=tmp, out_codes_folder=link, **kwargs)
                    self.assertNotIsInstance(refused.exception, RefusalError)
                    self.assertTrue(str(refused.exception).startswith(
                        f"out_folder and out_codes_folder are one folder ({link}):"), str(refused.exception))
                    os.remove(link)
                    missing = os.path.join(codes, "missing")
                    with self.assertRaises(RefusalError) as refused:
                        predict(out_folder=tmp, out_codes_folder=missing, **kwargs)
                    self.assertEqual(str(refused.exception),
                                     f"out_codes_folder {missing} is not a folder: give an existing folder.")
                    with self.assertRaises(RefusalError) as refused:
                        predict(out_folder=tmp, out_codes_folder=_FASTA, **kwargs)
                    self.assertEqual(str(refused.exception),
                                     f"out_codes_folder {_FASTA} is not a folder: give an existing folder.")
                    with self.assertRaisesRegex(RefusalError, "The codes name 'a/b' holds '/'"):
                        predict(out_folder=tmp, out_codes_folder=codes, out_codes_name="a/b", **kwargs)
            # scoring_system: refused without the codes (or a residue_output), and an unknown name refused before
            # ESM3 by both functions
            with self.assertRaisesRegex(ValueError, "out_codes_name and scoring_system are only used with "
                                                    "out_codes_folder, which is not given"):
                sequence_inference.predict(fasta_file=_FASTA, out_folder=tmp, scoring_system="esm-codes-v1")
            with self.assertRaisesRegex(ValueError, "^scoring_system is only used with residue_output or "
                                                    "out_codes_folder, and neither is given$"):
                chain_complete_inference.predict(src_stream=_FASTA, src_from=SrcEsmFrom.fasta, out_folder=tmp,
                                                 scoring_system="esm-codes-v1")
            with self.assertRaises(ss.UnknownScoringSystemError):
                sequence_inference.predict(fasta_file=_FASTA, out_folder=tmp, out_codes_folder=codes,
                                           scoring_system="esm-codes-v0")
            with self.assertRaises(ss.UnknownScoringSystemError):
                chain_complete_inference.predict(src_stream=_FASTA, src_from=SrcEsmFrom.fasta, out_folder=tmp,
                                                 out_codes_folder=codes, scoring_system="esm-codes-v0")
            self.assertEqual(os.listdir(tmp), [])
            self.assertEqual(os.listdir(codes), [])

    def test_missing_codes_raise_at_the_end(self):
        """W4: predict raises MissingResidueCodesError once the files are written, with the count, the ids, the rank
        and the predictions (None without return_predictions)."""
        _, reference, ids, sequences = _f2c()

        def one_row_short(model, protein):
            rows = _esm3(model, protein)
            return rows[:-1] if protein.sequence == sequences[1] else rows

        for predict, module, kwargs in ((sequence_inference.predict, esm_module, {"fasta_file": _FASTA}),
                                        (chain_complete_inference.predict, chain_complete_module,
                                         {"src_stream": _FASTA, "src_from": SrcEsmFrom.fasta})):
            for return_predictions, out_format in ((True, "pt"), (False, "csv")):
                with self.subTest(predict=predict.__module__, return_predictions=return_predictions), \
                        mock.patch.object(module, "per_residue_embedding", side_effect=one_row_short), \
                        self.assertLogs("foldmatch.modules.residue_payloads", "ERROR"):
                    codes = self.folder("codes")
                    with self.assertRaises(MissingResidueCodesError) as raised:
                        predict(accelerator=Accelerator.cpu, devices=1, batch_size=2, out_codes_folder=codes,
                                out_format=OutFormat(out_format), return_predictions=return_predictions, **kwargs)
                    error = raised.exception
                    self.assertIsInstance(error, RefusalError)
                    self.assertEqual((error.count, error.ids, error.rank, error.world_size), (1, (ids[1],), 0, 1))
                    self.assertIn(f"({codes}/<id>.{out_format} of each, " +
                                  ("an empty file)" if out_format == "csv" else "a uint8 tensor [0, 8])"), str(error))
                    self.assertEqual(names(codes), sorted(f"{i}.{out_format}" for i in ids))
                    if return_predictions:
                        # Y2: the predictions of the call, the codes as the tensors of the .pt files, [0, 8] for
                        # the sequence without codes
                        (_, names_, tables, _), = error.predictions
                        self.assertEqual(names_, tuple(ids))
                        for dom_id, table in zip(ids, tables):
                            self.assertEqual(table.dtype, torch.uint8)
                            self.assertTrue(torch.equal(table, torch.load(codes / f"{dom_id}.pt", weights_only=True)))
                        self.assertEqual(tuple(tables[1].shape), (0, 8))
                        np.testing.assert_array_equal(tables[0].numpy(), reference[ids[0]])
                    else:
                        self.assertEqual((codes / f"{ids[1]}.csv").read_bytes(), b"")
                        self.assertIsNone(error.predictions)

    def test_the_ranks_report_to_rank_0(self):
        """W4 with several ranks: each rank posts its count and first ids in the process group's store and waits for
        every rank's; rank 0 raises for the run (the count, the ranks, the first ids by rank, the files of those
        ranks), the other ranks log theirs and return; a rank whose pass raised is reported by rank 0 as a
        RefusalError; a rank that does not report is waited for at most RANK_WAIT."""
        codes = chain_complete_inference.ResidueCodesPass(None, None, None, Path("/c"), "inference-codes", 0)
        key = chain_complete_inference._report_key

        def trainer(rank, world_size=2):
            return SimpleNamespace(global_rank=rank, world_size=world_size)

        def module(*ids):
            return SimpleNamespace(residue_failures=len(ids), residue_failed_ids=list(ids))

        def check(rank, own, others, out_format="parquet", world_size=2, read=True):
            """check_after_pass on ``rank`` with the other ranks' reports already posted (and read)."""
            store = torch.distributed.HashStore()
            pass_id = 7
            for r, report in others.items():
                store.set(key(pass_id, r), json.dumps(report))
            if read:
                store.add(key(pass_id, "read"), world_size - 1)
            with mock.patch.object(chain_complete_inference, "_default_store", return_value=store):
                result = codes.check_after_pass(trainer(rank, world_size), own, out_format, pass_id=pass_id)
            return result, store

        self.assertEqual(check(0, module(), {1: {"count": 0, "ids": []}})[0], None)
        with self.assertRaises(MissingResidueCodesError) as raised:
            check(0, module(), {1: {"count": 2, "ids": ["c", "d"]}})
        error = raised.exception
        self.assertEqual((error.count, error.ids, error.rank_counts, error.rank, error.world_size),
                         (2, ("c", "d"), (0, 2), 0, 2))
        self.assertEqual(str(error), "2 sequence(s) on rank 1 of 2 got no residue codes: 'c', 'd'. ESM3 gave rows that "
                                     "do not match the sequence's length, or the encoding failed (each one's reason "
                                     "is logged above, by its rank). Their codes are written as empty [0, 8] tables "
                                     "(rows of /c/inference-codes-1.parquet without residues); their embeddings and "
                                     "every other file of the run are written. Check those sequences, or leave them "
                                     "out of the FASTA file.")
        with self.assertRaises(MissingResidueCodesError) as raised:
            check(0, module("a", "b"), {1: {"count": 1, "ids": ["c"]}, 2: {"count": 0, "ids": []}}, "json", 3)
        self.assertEqual((raised.exception.count, raised.exception.ids, raised.exception.rank_counts),
                         (3, ("a", "b", "c"), (2, 1, 0)))
        self.assertTrue(str(raised.exception).startswith(
            "3 sequence(s) on 2 ranks of 3 (rank 0: 2, rank 1: 1) got no residue codes: 'a', 'b', 'c'. "))
        self.assertIn("(records of /c/inference-codes-0.json.gz and /c/inference-codes-1.json.gz without residues)",
                      str(raised.exception))
        with self.assertRaises(MissingResidueCodesError) as raised:
            check(0, module(*"abc"), {1: {"count": 9, "ids": list("defghijkl")}})
        self.assertEqual(raised.exception.ids, tuple("abcdefghij"))
        self.assertIn("got no residue codes: 'a', 'b', 'c', 'd', 'e' and 7 more.", str(raised.exception))
        # The other ranks log their count and return: rank 0 raises for them
        with self.assertLogs(chain_complete_inference.__name__, "ERROR") as logs:
            result, store = check(1, module("c"), {0: {"count": 0, "ids": []}})
        self.assertIsNone(result)
        self.assertEqual(logs.output, [f"ERROR:{chain_complete_inference.__name__}:1 sequence(s) on rank 1 of 2 got "
                                       f"no residue codes; rank 0 ends the run with an error that names them"])
        self.assertEqual(json.loads(store.get(key(7, 1))), {"count": 1, "ids": ["c"]})
        self.assertTrue(store.check([key(7, "all-read")]))
        # A rank whose pass raised: rank 0 refuses the run, after its own missing codes or alone
        with self.assertRaises(RefusalError) as raised:
            check(0, module(), {1: {"error": "RuntimeError"}})
        self.assertNotIsInstance(raised.exception, MissingResidueCodesError)
        self.assertEqual(str(raised.exception), "Rank(s) 1 of 2 ended the pass with an error, which they report "
                                                "themselves; their files may be incomplete. The files of the other "
                                                "ranks are written.")
        with self.assertRaises(MissingResidueCodesError) as raised:
            check(0, module("a"), {1: {"error": "RuntimeError"}})
        self.assertTrue(str(raised.exception).endswith(" Also, rank(s) 1 of 2 ended the pass with an error, which "
                                                       "they report themselves; their files may be incomplete."))
        self.assertIsNone(check(1, module(), {0: {"error": "RuntimeError"}})[0])
        # A rank that does not report
        with mock.patch.object(chain_complete_inference, "RANK_WAIT", timedelta(milliseconds=200)), \
                self.assertRaises(RefusalError) as raised:
            check(0, module(), {})
        self.assertTrue(str(raised.exception).startswith(
            "Rank 0 of 2 ended its pass, with its files written, but could not learn whether the other ranks' "
            "sequences got residue codes: rank(s) 1 did not report the end of its pass (a rank waits for that at "
            "most 200 ms); "), str(raised.exception))
        self.assertEqual(chain_complete_inference._duration(chain_complete_inference.RANK_WAIT), "1 day")
        self.assertEqual(chain_complete_inference._duration(chain_complete_inference.READ_WAIT), "10 minutes")
        # The store gone (rank 0's process held it and rank 0 ended): the other ranks say so, with the store's error
        gone = mock.Mock(**{f"{op}.side_effect": RuntimeError("Connection reset by peer")
                            for op in ("set", "wait", "get", "add", "check")})
        with mock.patch.object(chain_complete_inference, "_default_store", return_value=gone), \
                self.assertRaises(RefusalError) as raised:
            codes.check_after_pass(trainer(1), module(), "parquet", pass_id=7)
        self.assertEqual(str(raised.exception), "Rank 1 of 2 ended its pass, with its files written, but could not "
                                                "learn whether the other ranks' sequences got residue codes: the "
                                                "process group's store could not be reached (rank 0's process holds "
                                                "it under Lightning's launcher and srun: rank 0 has most likely "
                                                "ended, with an error of its own); RuntimeError: Connection reset by "
                                                "peer.")
        # Rank 0 waits for the others to read the reports (its process may hold the store), at most READ_WAIT
        with mock.patch.object(chain_complete_inference, "READ_WAIT", timedelta(milliseconds=200)), \
                self.assertLogs(chain_complete_inference.__name__, "WARNING") as logs:
            self.assertIsNone(check(0, module(), {1: {"count": 0, "ids": []}}, read=False)[0])
        self.assertIn("rank 0 of 2: not every rank read the residue-code reports within 200 ms", logs.output[0])
        # A rank whose pass raised never reads the reports: its post counts as read, so rank 0 does not wait
        # READ_WAIT for it (NERSC job 61254854: rank 0 sat out the 10 minutes while NCCL held rank 1's exit)
        store = torch.distributed.HashStore()
        with mock.patch.object(chain_complete_inference, "_default_store", return_value=store), \
                mock.patch.object(chain_complete_inference, "READ_WAIT", timedelta(seconds=30)), \
                self.assertNoLogs(chain_complete_inference.__name__, "WARNING"):
            chain_complete_inference._post_report(trainer(1), 7, {"error": "RuntimeError"})
            started = time.monotonic()
            with self.assertRaises(RefusalError) as raised:
                codes.check_after_pass(trainer(0), module(), "parquet", pass_id=7)
            self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(str(raised.exception).startswith("Rank(s) 1 of 2 ended the pass with an error"))
        self.assertTrue(store.check([key(7, "all-read")]))
        # Rank 0's pass raised: the other ranks read its report and end; the last reader sets all-read
        store = torch.distributed.HashStore()
        with mock.patch.object(chain_complete_inference, "_default_store", return_value=store):
            chain_complete_inference._post_report(trainer(0), 8, {"error": "RuntimeError"})
            self.assertFalse(store.check([key(8, "all-read")]))
            self.assertIsNone(codes.check_after_pass(trainer(1), module(), "parquet", pass_id=8))
        self.assertTrue(store.check([key(8, "all-read")]))
        # Several ranks without a process group (a spawn or fork strategy's parent): a WARNING, no error
        with mock.patch.object(chain_complete_inference, "_default_store", return_value=None), \
                self.assertLogs(chain_complete_inference.__name__, "WARNING") as logs:
            self.assertIsNone(codes.check_after_pass(trainer(0), module(), "parquet", pass_id=9))
        self.assertEqual(logs.output, [f"WARNING:{chain_complete_inference.__name__}:This process started its 2 ranks "
                                       f"with a spawn or fork strategy and does not learn their counts of sequences "
                                       f"without residue codes: it raises no MissingResidueCodesError for them. Each "
                                       f"rank has logged its own as ERROR lines; their codes are empty [0, 8] "
                                       f"tables."])

    def test_a_pass_that_raises_tells_the_other_ranks(self):
        """ResidueCodesPass.run: a rank whose trainer.predict raises posts {"error"} for its pass, then re-raises;
        one that ends posts its count and returns its predictions with the codes as uint8 tensors [L, 8]. The pass
        ids advance alike on every rank (one per pass)."""
        codes = chain_complete_inference.ResidueCodesPass(None, SimpleNamespace(system=SYSTEM), None, Path("/c"),
                                                          "inference-codes", 0)
        store = torch.distributed.HashStore()
        store.set(chain_complete_inference._report_key(100, 0), json.dumps({"count": 0, "ids": []}))
        store.add(chain_complete_inference._report_key(100, "read"), 1)

        def fails(*_, **__):
            raise RuntimeError("out of memory")

        with mock.patch.object(chain_complete_inference, "_default_store", return_value=store), \
                mock.patch.object(chain_complete_inference, "_PASSES", iter([99, 100])):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                codes.run(SimpleNamespace(global_rank=1, world_size=2, predict=fails), None, None, "csv", False)
            self.assertEqual(json.loads(store.get(chain_complete_inference._report_key(99, 1))),
                             {"error": "RuntimeError"})
            # ... and counts as having read the reports (it never will), so that rank 0 does not wait for it
            self.assertEqual(store.add(chain_complete_inference._report_key(99, "read"), 0), 1)
            embeddings = torch.zeros(2, 4)
            done = SimpleNamespace(global_rank=1, world_size=2, predict=lambda *_, **__: [
                (embeddings, ("a", "b"), (bytes(range(16)), b""), (bytes(20), bytes(20)))])
            (got,) = codes.run(done, SimpleNamespace(residue_failures=0, residue_failed_ids=[]), None, "csv", True)
            self.assertIs(got[0], embeddings)
            self.assertEqual(got[1], ("a", "b"))
            self.assertTrue(torch.equal(got[2][0], torch.arange(16, dtype=torch.uint8).reshape(2, 8)))
            self.assertEqual((got[2][1].dtype, tuple(got[2][1].shape)), (torch.uint8, (0, 8)))
            self.assertEqual(got[3], (bytes(20), bytes(20)))
            self.assertEqual(json.loads(store.get(chain_complete_inference._report_key(100, 1))),
                             {"count": 0, "ids": []})

    def test_a_folder_named_as_a_file_of_the_other_in_the_library(self):
        """A codes folder inside the output folder under the name of an embedding file (a library caller: the files
        <out_name>-<rank>.parquet|json.gz), or the reverse: refused before ESM3, nothing written."""
        self.refusing = True
        for predict, kwargs in ((sequence_inference.predict, {"fasta_file": _FASTA}),
                                (chain_complete_inference.predict, {"src_stream": _FASTA,
                                                                    "src_from": SrcEsmFrom.fasta})):
            for out_format, name in ((OutFormat.parquet, "inference-0.parquet"),
                                     (OutFormat.json, "inference-2.json.gz")):
                with self.subTest(predict=predict.__module__, out_format=out_format):
                    out = self.folder()
                    (out / name).mkdir()
                    with self.assertRaises(RefusalError) as refused:
                        predict(out_folder=out, out_codes_folder=out / name, out_format=out_format, **kwargs)
                    self.assertTrue(str(refused.exception).startswith(
                        f"The codes folder {out / name} is named as one of the embedding files of the output folder "
                        f"{out} (a file <name>-<rank>"), str(refused.exception))
                    self.assertEqual(names(out), [name])
                    self.assertEqual(names(out / name), [])
                    codes = self.folder("codes")
                    inner = codes / name.replace("inference-", "inference-codes-")
                    inner.mkdir()
                    with self.assertRaises(RefusalError) as refused:
                        predict(out_folder=inner, out_codes_folder=codes, out_format=out_format, **kwargs)
                    self.assertTrue(str(refused.exception).startswith(
                        f"The output folder {inner} is named as one of the code files of the codes folder {codes} "
                        f"(a file <name>-<rank>"), str(refused.exception))
                    self.assertEqual(names(inner), [])

    def test_refuse_same_folder(self):
        """Folders compared as files (device and inode), not as path strings: a link and '..' are the folder, on any
        file system; a missing folder is no folder; the message names the options given."""
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            link = os.path.join(other, "link")
            os.symlink(tmp, link, target_is_directory=True)
            os.mkdir(os.path.join(tmp, "sub"))
            for given in (link, os.path.join(tmp, "sub", ".."), os.path.join(link, ".")):
                with self.subTest(given=given), self.assertRaisesRegex(ValueError, r"^a and b are one folder"):
                    output_files.refuse_same_folder(tmp, given, ("a", "b"))
                self.assertTrue(output_files.same_folder_problem(tmp, given, ("a", "b")).startswith(
                    f"a and b are one folder ({given}): "))
            os.mkdir(os.path.join(tmp, "codes.parquet"))     # W2: fm-search skips such a subfolder
            for given in (other, os.path.join(tmp, "sub"), os.path.join(tmp, "missing"),
                          os.path.join(tmp, "codes.parquet")):
                with self.subTest(given=given):
                    output_files.refuse_same_folder(tmp, given)
                    self.assertIsNone(output_files.same_folder_problem(tmp, given))
            output_files.refuse_same_folder(os.path.join(tmp, "missing"), os.path.join(tmp, "missing"))


# =====================================================================================================================
# Modules and writers
# =====================================================================================================================
class ModuleTest(unittest.TestCase):

    def test_esm_module_with_codes(self):
        class Head(torch.nn.Module):
            calls = []

            def encode_target(self, rows):
                self.calls.append(rows)
                return torch.zeros(rows.shape[0], 8, dtype=torch.uint8)

        head = Head()
        batch = [(SimpleNamespace(sequence="MKV", coordinates=None), "a"),
                 (SimpleNamespace(sequence="MK", coordinates=None), "b")]
        rows = {"MKV": torch.randn(5, 4), "MK": torch.randn(4, 4)}
        with mock.patch.object(esm_module, "per_residue_embedding", lambda m, p: rows[p.sequence]):
            plain = EsmModule(None).predict_step(batch, 0)
            module = EsmModule(None, residue_head=head, residue_output=rs.RESIDUE_CODES, code_files=True)
            out = module.predict_step(batch, 0)
        self.assertEqual(len(plain), 2)
        self.assertEqual(out[1], ("a", "b"))
        self.assertTrue(all(x is y for x, y in zip(out[0], plain[0])))
        self.assertEqual(out[2], (bytes(24), bytes(16)))
        self.assertEqual(out[3], (_sha1("MKV"), _sha1("MK")))
        self.assertTrue(torch.equal(head.calls[0], rows["MKV"][1:-1]))
        self.assertIs(dict(module.named_children())["residue_head"], head)

    def test_failures_past_the_logged_ones(self):
        """After the first ten failures by name, one line, about residue codes for code files."""
        for code_files, line in ((True, "more proteins without residue codes; only counting them from now on"),
                                 (False, "more proteins without a residue_codes payload; only counting them from now "
                                         "on")):
            with self.subTest(code_files=code_files):
                payloads = ResiduePayloads()
                payloads._init_residue_payloads(torch.nn.Identity(), rs.RESIDUE_CODES, code_files=code_files)
                with self.assertLogs("foldmatch.modules.residue_payloads", "ERROR") as logs:
                    for k in range(12):
                        self.assertEqual(payloads._failed(f"p{k}", "a reason"), b"")
                messages = [record.getMessage() for record in logs.records]
                self.assertEqual(len(messages), 11)
                self.assertEqual(messages[-1], line)
                self.assertEqual(payloads.residue_failures, 12)

    def test_constructor_refusals(self):
        with self.assertRaises(ValueError):
            EsmModule(None, residue_head=torch.nn.Identity())
        with self.assertRaises(ValueError):
            EsmModule(None, residue_output=rs.RESIDUE_CODES)
        with self.assertRaises(ValueError):
            EsmModule(None, code_files=True)
        with self.assertRaises(ValueError):
            ChainCompleteModule(None, None, residue_head=torch.nn.Identity(), residue_output=rs.QUERY_ENCODING,
                                code_files=True)


# =====================================================================================================================
# 'build residue-codes --from-shards' refuses a code file
# =====================================================================================================================
class BuildFromShardsTest(_Case):
    """A code file among a run's shards is refused by 'build residue-codes --from-shards' (the readers of
    tests/test_residue_code_files.py)."""

    def codes_parquet(self, command="chain"):
        codes = self.folder("codes")
        self.run_ok(command, _FASTA, self.folder(), "--output-format", "parquet", "--output-codes-folder", codes)
        return codes

    def test_build_residue_codes_from_shards_refuses_a_code_file(self):
        db, shards = self.folder("db"), self.folder("shards")
        with launch_env(FOLDMATCH_RUN_ID="p9x-plain"):
            result = self.invoke("build", "sequences", "--fasta-file", _FASTA, "--output-db", db / "db",
                                 "--tmp-embedding-folder", shards, app=search.app)
        self.assertEqual(result.exit_code, 0, result.output)
        out = self.codes_parquet()
        stray = self.folder("stray")
        shutil.copyfile(out / "inference-codes-0.parquet", stray / "emb_p9x-codes-0.parquet")
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            result = CliRunner().invoke(search.app, ["build", "residue-codes", str(db), "db", "--from-shards",
                                                     str(stray), "--run-id", "p9x-codes"])
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("not an embedding shard", _flat(str(result.exception)))
        self.assertFalse(ResidueCodeStore(db, "db").exists())


# =====================================================================================================================
# The real ESM3 (CPU)
# =====================================================================================================================
class RealEsm3Test(unittest.TestCase):
    """The real ESM3 on the CPU, loaded once: in both commands and every format, the embedding files with and without
    --output-codes-folder (byte for byte; a .json.gz but for its gzip mtime), the codes against the residue-code store
    of 'fm-search build sequences --residue-codes' on the same FASTA (equal) and F2c (A4-xcpu), and 'fm-search build
    embeddings' and 'query embedding' on the chain output folders (the same bytes as from a plain run's)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        esm_codes_bundle.registry_files()
        gc.collect()
        from foldmatch.utils.model import get_residue_model
        cls.esm3 = get_residue_model(torch.device("cpu"))
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name).resolve()
        cls.results = {}
        with mock.patch.object(sequence_inference, "get_residue_model", lambda **_: cls.esm3), \
                mock.patch.object(chain_complete_inference, "get_residue_model", lambda **_: cls.esm3), \
                launch_env(FOLDMATCH_RUN_ID="p9x-real"):
            for command in _COMMANDS:
                for out_format in _FORMATS:
                    for kind in ("plain", "codes"):
                        out = cls.tmp / f"{command}-{out_format}-{kind}"
                        out.mkdir()
                        extra = ()
                        if kind == "codes":
                            (cls.tmp / f"{command}-{out_format}-code-files").mkdir()
                            extra = ("--output-codes-folder", cls.tmp / f"{command}-{out_format}-code-files")
                        cls.results[command, out_format, kind] = cls._invoke(
                            embedding.app, "from-sequences", command, "--fasta-file", _FASTA, "--output-folder", out,
                            "--output-format", out_format, "--batch-size", "2", *extra)
            (cls.tmp / "shards").mkdir()
            cls.results["build"] = cls._invoke(search.app, "build", "sequences", "--fasta-file", _FASTA,
                                               "--output-db", cls.tmp / "db" / "db", "--tmp-embedding-folder",
                                               cls.tmp / "shards", "--residue-codes", "--batch-size", "2")
        del cls.esm3
        gc.collect()

    @classmethod
    def _invoke(cls, app, *args):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            return CliRunner().invoke(app, [str(a) for a in (*args, *_CPU)])
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()
        super().tearDownClass()

    def setUp(self):
        for key, result in self.results.items():
            self.assertEqual(result.exit_code, 0, f"{key}: {result.output[-4000:]} {result.exception!r}")

    def test_the_embedding_files_are_those_of_a_run_without_codes(self):
        for command in _COMMANDS:
            for out_format in _FORMATS:
                with self.subTest(command=command, out_format=out_format):
                    plain = self.tmp / f"{command}-{out_format}-plain"
                    out = self.tmp / f"{command}-{out_format}-codes"
                    self.assertEqual(len(names(plain)), 2 if out_format in ("csv", "pt") else 1)
                    self.assertEqual(names(out), names(plain))
                    for name in names(plain):
                        self.assertTrue(same_bytes(plain / name, out / name), name)
                    if out_format in ("csv", "pt"):
                        self.assertEqual(names(self.tmp / f"{command}-{out_format}-code-files"), names(plain))

    def test_the_codes_are_the_build_store_codes(self):
        _, reference, ids, _ = _f2c()
        store = ResidueCodeStore(self.tmp / "db", "db")
        store.check_compatible()
        memmap = np.asarray(store.open_memmap())
        entries = store.lookup(ids)
        stored = {i: memmap[entries[i][0]:entries[i][0] + entries[i][1]] for i in ids}
        for command in _COMMANDS:
            for out_format in _FORMATS:
                with self.subTest(command=command, out_format=out_format):
                    got = read_codes(self.tmp / f"{command}-{out_format}-code-files", out_format, ids=ids)
                    self.assertEqual(list(got), ids)
                    for dom_id in ids:
                        np.testing.assert_array_equal(got[dom_id], stored[dom_id])
        codes = np.concatenate([stored[i] for i in ids])
        ref = np.concatenate([reference[i] for i in ids])
        equal = int((np.unpackbits(codes, axis=1) == np.unpackbits(ref, axis=1)).sum())
        print(f"\nfm-embedding --output-codes-folder F2c CPU code bits: {equal}/{ref.size * 8}", file=sys.stderr)
        self.assertGreaterEqual(equal / (ref.size * 8), A4_XCPU_BITS)

    def test_fm_search_reads_the_chain_output_folder(self):
        """The chain output folder of a run with --output-codes-folder: 'fm-search build embeddings' gives the index
        and metadata bytes of the plain run's folder, and 'query embedding' the same hits."""
        for out_format in ("parquet", "csv", "pt"):
            files = {}
            for kind in ("plain", "codes"):
                folder = self.tmp / f"chain-{out_format}-{kind}"
                db = self.tmp / f"db-{out_format}-{kind}"
                db.mkdir()
                hits = db / "hits.tsv"
                for args in (("build", "embeddings", "--embedding-folder", folder, "--output-db", db / "db"),
                             ("query", "embedding", "--db-path", db / "db", "--embedding-folder", folder,
                              "--output-file", hits, "--threshold", "0")):
                    root = logging.getLogger()
                    handlers, level = root.handlers[:], root.level
                    try:
                        result = CliRunner().invoke(search.app, [str(a) for a in args])
                    finally:
                        root.handlers[:] = handlers
                        root.setLevel(level)
                    self.assertEqual(result.exit_code, 0, f"{args}: {result.output} {result.exception!r}")
                files[kind] = {name: (db / name).read_bytes() for name in ("db.index", "db.metadata", "hits.tsv")}
            with self.subTest(out_format=out_format):
                self.assertTrue(files["plain"]["hits.tsv"])
                self.assertEqual(files["codes"], files["plain"])
                print(f"\nchain {out_format}: fm-search build embeddings / query embedding on the output folder of a "
                      f"run with --output-codes-folder: db.index {len(files['codes']['db.index'])} B, db.metadata "
                      f"{len(files['codes']['db.metadata'])} B, hits {len(files['codes']['hits.tsv'])} B, identical "
                      f"to the plain run's", file=sys.stderr)


# =====================================================================================================================
# Two ranks (slow)
# =====================================================================================================================
_SITECUSTOMIZE = r"""
import os
import numpy as np
import torch
from foldmatch.inference import chain_complete_inference, sequence_inference
from foldmatch.modules import chain_complete_module, esm_module
z = np.load(os.environ["FOLDMATCH_TEST_F2C"])
ro = z["rows_offsets"]
rows = {str(z["sequences"][k]): torch.from_numpy(z["rows"][ro[k]:ro[k + 1]].astype(np.float32))
        for k in range(len(ro) - 1)}
for module in (sequence_inference, chain_complete_inference):
    module.get_residue_model = lambda **_: torch.nn.Identity()
short = os.environ.get("FOLDMATCH_TEST_SHORT")   # the index of a sequence whose ESM3 rows lack one row
short = None if short is None else str(z["sequences"][int(short)])
slow = float(os.environ.get("FOLDMATCH_TEST_SLOW_RANK0") or 0)   # seconds that rank 0 spends on each sequence


def per_residue_embedding(model, protein):
    if slow and os.environ.get("LOCAL_RANK", "0") == "0":
        import time
        time.sleep(slow)
    return (rows[protein.sequence][:-1] if protein.sequence == short else rows[protein.sequence]).clone()


for module in (esm_module, chain_complete_module):
    module.per_residue_embedding = per_residue_embedding
"""
#: A library caller on two ranks (Lightning's launcher starts rank 1 by running this script again): predict with
#: out_codes_folder, then this rank's predictions (ids, codes, seq_sha1s) and the error it raised, if any, to
#: <dump>/rank-<rank>.pt.
_LIBRARY_DRIVER = r"""
import sys
from pathlib import Path
import torch
from foldmatch.errors import MissingResidueCodesError
from foldmatch.inference import chain_complete_inference, sequence_inference
from foldmatch.types.api_types import Accelerator, OutFormat, SrcEsmFrom

command, fasta, out, codes, out_format, dump = sys.argv[1:7]
kwargs = dict(accelerator=Accelerator.cpu, devices=2, batch_size=1, out_folder=Path(out),
              out_format=OutFormat(out_format), out_codes_folder=Path(codes))
if command == "residue":
    predict = sequence_inference.predict
    kwargs["fasta_file"] = Path(fasta)
else:
    predict = chain_complete_inference.predict
    kwargs.update(src_stream=Path(fasta), src_from=SrcEsmFrom.fasta)
error = None
try:
    predictions = predict(**kwargs)
except MissingResidueCodesError as e:
    predictions, error = e.predictions, str(e)
rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
torch.save({"predictions": [tuple(p[1:]) for p in predictions], "error": error}, f"{dump}/rank-{rank}.pt")
"""
_DDP_ENV_PREFIXES = ("MASTER_", "PL_", "TORCHELASTIC_")
_DDP_ENV = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK")


@unittest.skipUnless(os.environ.get("FOLDMATCH_SLOW_TESTS") == "1",
                     "slow: two CPU ranks as a command; set FOLDMATCH_SLOW_TESTS=1 to run")
class TwoRankSmoke(unittest.TestCase):
    """``--devices 2`` (CPU, gloo; ESM3 replaced by the F2c rows in both processes): a codes file per rank in parquet,
    a code file per id in csv, in the codes folder, the F2c codes; the output folder holds the embeddings only. W4:
    with one sequence's ESM3 rows a row short, on rank 1 (a process that Lightning's launcher, rank 0, started) or on
    rank 0, the run ends, no rank hangs, every file is written, and the exit status is 1, from rank 0, also when
    rank 0 ends its pass long after rank 1 (which then waits for it, rather than failing while rank 0 still writes,
    which would get rank 0 killed), under Lightning's launcher and under torchrun."""

    def run_two_ranks(self, tmp, env, command, out_format, short=None, slow=0, torchrun=False, out=None,
                      codes=None):
        """A run on two ranks, into new folders, or into ``out`` and ``codes`` (None: no codes) when given."""
        tag = f"{command}-{out_format}-{short}-{slow}-{'torchrun' if torchrun else 'lightning'}"
        if out is None:
            out, codes = tmp / tag, tmp / f"{tag}-codes"
            out.mkdir()
            codes.mkdir()
        env = dict(env)
        if short is not None:
            env["FOLDMATCH_TEST_SHORT"] = str(short)
        if slow:
            env["FOLDMATCH_TEST_SLOW_RANK0"] = str(slow)
        launcher = [sys.executable]
        if torchrun:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            launcher += ["-m", "torch.distributed.run", "--nproc-per-node", "2", "--master-addr", "127.0.0.1",
                         "--master-port", str(port)]
        start = time.monotonic()
        run = subprocess.run([*launcher, "-m", "foldmatch.cli.embedding", "from-sequences", command,
                              "--fasta-file", str(_FASTA), "--output-folder", str(out), "--output-format",
                              out_format, *(() if codes is None else ("--output-codes-folder", str(codes))),
                              "--accelerator", "cpu", "--devices", "2"],
                             env=env, capture_output=True, text=True, timeout=1800)
        print(f"\nTwoRankSmoke {tag}: exit status {run.returncode} in {time.monotonic() - start:.1f} s",
              file=sys.stderr)
        return run, out, codes

    def env(self, tmp):
        (tmp / "site").mkdir()
        (tmp / "site" / "sitecustomize.py").write_text(_SITECUSTOMIZE)
        env = {k: v for k, v in os.environ.items()
               if k not in _LAUNCH_ENV and k not in _DDP_ENV and not k.startswith(_DDP_ENV_PREFIXES)}
        env["PYTHONPATH"] = os.pathsep.join([str(tmp / "site"), _SRC] +
                                            [p for p in [os.environ.get("PYTHONPATH")] if p])
        env["FOLDMATCH_TEST_F2C"] = str(_F2C)
        return env

    def test_two_ranks(self):
        esm_codes_bundle.registry_files()
        _, reference, ids, _ = _f2c()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = self.env(tmp)
            for command, out_format in (("residue", "parquet"), ("chain", "csv"), ("chain", "pt")):
                with self.subTest(command=command, out_format=out_format):
                    run, out, codes = self.run_two_ranks(tmp, env, command, out_format)
                    self.assertEqual(run.returncode, 0, run.stderr[-4000:])
                    if out_format == "parquet":
                        self.assertEqual(names(out), ["inference-0.parquet", "inference-1.parquet"])
                        self.assertEqual(names(codes), ["inference-codes-0.parquet", "inference-codes-1.parquet"])
                        for rank in (0, 1):
                            record = json.loads(pq.read_schema(codes / f"inference-codes-{rank}.parquet")
                                                .metadata[b"foldmatch.pass"])
                            self.assertEqual(record["world_size"], 2)
                    else:
                        self.assertEqual(names(out), sorted(f"{i}.{out_format}" for i in ids))
                        self.assertEqual(names(codes), names(out))
                    got = read_codes(codes, out_format, ids=ids)
                    self.assertEqual(sorted(got), sorted(ids))
                    for dom_id in ids:
                        np.testing.assert_array_equal(got[dom_id], reference[dom_id])
                    # Z4: a rerun into the same folders on two ranks replaces its files (each rank checks the files
                    # of both ranks, which the other may be writing), with and without the codes
                    files = sorted([*out.iterdir(), *codes.iterdir()])
                    for path in files:
                        os.utime(path, ns=(0, 0))
                    rerun = self.run_two_ranks(tmp, env, command, out_format, out=out, codes=codes)[0]
                    self.assertEqual(rerun.returncode, 0, rerun.stderr[-4000:])
                    self.assertNotIn("would replace", rerun.stderr)
                    self.assertEqual(sorted([*out.iterdir(), *codes.iterdir()]), files)
                    self.assertTrue(all(path.stat().st_mtime_ns > 0 for path in files))
                    rerun = self.run_two_ranks(tmp, env, command, out_format, out=out, codes=None)[0]
                    self.assertEqual(rerun.returncode, 0, rerun.stderr[-4000:])

    def test_two_ranks_return_the_codes(self):
        """Y2 on two ranks, a library caller (Lightning's launcher): each rank's predictions hold its ids' codes as CPU
        uint8 tensors [L, 8], those of the .pt code files and the bytes of the codes Parquet files, for both predict
        functions; with a sequence without codes on rank 1, rank 0's MissingResidueCodesError carries rank 0's
        predictions and rank 1 returns its own, the empty table a [0, 8] tensor equal to its .pt file."""
        esm_codes_bundle.registry_files()
        _, reference, ids, sequences = _f2c()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = self.env(tmp)
            driver = tmp / "driver.py"
            driver.write_text(_LIBRARY_DRIVER)
            for command, out_format, short in (("residue", "pt", None), ("residue", "parquet", None),
                                               ("chain", "pt", None), ("chain", "parquet", None),
                                               ("chain", "pt", 1)):
                with self.subTest(command=command, out_format=out_format, short=short):
                    tag = f"lib-{command}-{out_format}-{short}"
                    out, codes, dump = tmp / tag, tmp / f"{tag}-codes", tmp / f"{tag}-dump"
                    for folder in (out, codes, dump):
                        folder.mkdir()
                    run_env = dict(env)
                    if short is not None:
                        run_env["FOLDMATCH_TEST_SHORT"] = str(short)
                    start = time.monotonic()
                    run = subprocess.run([sys.executable, str(driver), command, str(_FASTA), str(out), str(codes),
                                          out_format, str(dump)], env=run_env, capture_output=True, text=True,
                                         timeout=1800)
                    print(f"\nTwoRankSmoke {tag}: exit status {run.returncode} in {time.monotonic() - start:.1f} s",
                          file=sys.stderr)
                    self.assertEqual(run.returncode, 0, run.stderr[-4000:])
                    self.assertEqual(names(dump), ["rank-0.pt", "rank-1.pt"])
                    returned = {}
                    for rank in (0, 1):
                        record = torch.load(dump / f"rank-{rank}.pt", weights_only=False)
                        if short is None or rank == 1:
                            self.assertIsNone(record["error"])
                        else:
                            self.assertTrue(record["error"].startswith(
                                f"1 sequence(s) on rank 1 of 2 got no residue codes: {ids[short]!r}."),
                                record["error"])
                        for names_, tables, digests in record["predictions"]:
                            self.assertIsInstance(tables, tuple)
                            for dom_id, table, digest in zip(names_, tables, digests):
                                returned[dom_id] = (rank, table, digest)
                    self.assertEqual(sorted(returned), sorted(ids))
                    self.assertEqual({rank for rank, _, _ in returned.values()}, {0, 1})
                    if out_format == "parquet":
                        stored = {}
                        for rank in (0, 1):
                            parquet = pq.read_table(codes / f"inference-codes-{rank}.parquet")
                            column = parquet.column("residue_codes").combine_chunks()
                            for k, dom_id in enumerate(parquet.column("id").to_pylist()):
                                stored[dom_id] = (rank, column[k].values.flatten().to_numpy(zero_copy_only=False)
                                                  .tobytes())
                    for k, dom_id in enumerate(ids):
                        rank, table, digest = returned[dom_id]
                        self.assertEqual((table.dtype, table.device.type), (torch.uint8, "cpu"))
                        self.assertEqual(digest, _sha1(sequences[k]))
                        if dom_id == (ids[short] if short is not None else None):
                            self.assertEqual(tuple(table.shape), (0, 8))
                        else:
                            self.assertEqual(tuple(table.shape), (len(sequences[k]), 8))
                            np.testing.assert_array_equal(table.numpy(), reference[dom_id])
                        if out_format == "pt":
                            pt = torch.load(codes / f"{dom_id}.pt", weights_only=True)
                            self.assertEqual((pt.dtype, pt.shape), (table.dtype, table.shape))
                            self.assertTrue(torch.equal(pt, table))
                        else:
                            self.assertEqual(stored[dom_id], (rank, table.numpy().tobytes()))

    def test_two_ranks_with_missing_codes(self):
        """W4: the sequence of rank 1 (ids[1]) or of rank 0 (ids[0]) gets no codes; rank 0 also ends its pass 12 s
        after rank 1 (more than the 5 s that Lightning's child observer takes to kill every process once a rank exits
        with an error), under Lightning's launcher and under torchrun."""
        esm_codes_bundle.registry_files()
        _, reference, ids, _ = _f2c()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = self.env(tmp)
            for command, out_format, short, slow, torchrun in (
                    ("chain", "parquet", 1, 0, False), ("residue", "csv", 1, 0, False),
                    ("chain", "csv", 0, 0, False), ("residue", "parquet", 0, 0, False),
                    ("chain", "parquet", 1, 12, False), ("residue", "csv", 1, 12, True)):
                with self.subTest(command=command, out_format=out_format, short_rank=short, slow=slow,
                                  torchrun=torchrun):
                    run, out, codes = self.run_two_ranks(tmp, env, command, out_format, short, slow, torchrun)
                    stderr = _flat(run.stderr)
                    self.assertEqual(run.returncode, 1, run.stderr[-4000:])
                    message = f"Error: 1 sequence(s) on rank {short} of 2 got no residue codes: {ids[short]!r}."
                    self.assertEqual(stderr.count(message), 1, run.stderr[-4000:])
                    self.assertNotIn("Forcefully terminating", stderr)
                    if short == 1:
                        self.assertIn("1 sequence(s) on rank 1 of 2 got no residue codes; rank 0 ends the run with an "
                                      "error that names them", stderr)
                    if out_format == "parquet":
                        self.assertEqual(names(out), ["inference-0.parquet", "inference-1.parquet"])
                        self.assertEqual(names(codes), ["inference-codes-0.parquet", "inference-codes-1.parquet"])
                        self.assertIn(f"(rows of {codes.resolve()}/inference-codes-{short}.parquet without "
                                      f"residues)", stderr)
                    else:
                        self.assertEqual(names(out), sorted(f"{i}.{out_format}" for i in ids))
                        self.assertEqual(names(codes), names(out))
                        self.assertEqual((codes / f"{ids[short]}.csv").read_bytes(), b"")
                    got = read_codes(codes, out_format, ids=ids)
                    self.assertEqual(got[ids[short]].shape, (0, 8))
                    np.testing.assert_array_equal(got[ids[1 - short]], reference[ids[1 - short]])


if __name__ == "__main__":
    unittest.main()
