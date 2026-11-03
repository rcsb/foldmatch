"""P9.H: what the inference commands write, as their help says. ESM3 and the aggregator are never loaded: the commands run with stub models, as in
test_inference_follow_ups.py."""
import gzip
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq
import torch
from typer.testing import CliRunner

from foldmatch.cli import inference

_CIF = Path(__file__).parent / "resources" / "pdb" / "1acb.cif"
_CPU = ("--accelerator", "cpu", "--devices", "1")
_CHAINS = ["1acb.A", "1acb.B"]
_ASSEMBLIES = ["1acb-1"]


def _esm3(model, protein):
    """Stands in for ESM3: L + 2 rows (BOS and EOS, zeros), each residue's row its letter and its position."""
    rows = [[float(ord(aa)), float(k)] for k, aa in enumerate(protein.sequence)]
    return torch.tensor([[0.0] * 2] + rows + [[0.0] * 2])


def _aggregate(x, x_mask):
    """Stands in for the aggregator: the sum of each item's rows."""
    return x.masked_fill(x_mask.unsqueeze(-1), 0.0).sum(dim=1)


def _invoke(*args):
    """inference in-process, the logging setup restored after."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        return CliRunner().invoke(inference.app, [str(a) for a in args])
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def _ids(path: Path) -> list:
    """The embedding ids of a Parquet or JSON file."""
    if path.suffix == ".parquet":
        return pq.read_table(path).column("id").to_pylist()
    with gzip.open(path, "rt") as f:
        return [row["id"] for row in json.load(f)]


class _Lists(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.chains = self.tmp / "chains.csv"
        self.chains.write_text(f"1acb,{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B\n")
        self.assemblies = self.tmp / "assemblies.csv"
        self.assemblies.write_text(f"1acb,{_CIF},1,1acb-1\n")

    def folder(self) -> Path:
        return Path(tempfile.mkdtemp(dir=self.tmp))

    def run_ok(self, *args):
        result = _invoke(*args, *_CPU)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")


class _StubModels(_Lists):

    def setUp(self):
        super().setUp()
        for module in ("esm_inference", "structure_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_residue_model", return_value="ESM3"))
        for module in ("structure_inference", "chain_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_aggregator_model", return_value=_aggregate))
        for module in ("esm_module", "chain_complete_module"):
            self.enterContext(mock.patch(f"foldmatch.modules.{module}.per_residue_embedding", _esm3))

    def residues(self) -> Path:
        out = self.folder()
        self.run_ok("residue-embedding", "--src-file", self.chains, "--output-folder", out)
        return out


def _files(names: list, output_format: str, output_name: str) -> list:
    """The files that the help says a run writes: one per item for csv and pt, one per process for parquet and json."""
    return {"csv": sorted(f"{n}.csv" for n in names), "pt": sorted(f"{n}.pt" for n in names),
            "parquet": [f"{output_name}-0.parquet"], "json": [f"{output_name}-0.json.gz"]}[output_format]


class OutputFilesTest(_StubModels):
    """Each command writes what its help says, in each format it takes."""

    def assert_files(self, folder: Path, names: list, output_format: str, output_name: str):
        self.assertEqual(sorted(f.name for f in folder.iterdir()), _files(names, output_format, output_name))
        if output_format in ("parquet", "json"):
            self.assertCountEqual(_ids(next(folder.iterdir())), names)

    def test_residue_embedding_is_pt_by_default(self):
        self.assert_files(self.residues(), _CHAINS, "pt", "inference")

    def test_residue_embedding_in_each_format(self):
        for output_format in ("csv", "pt", "parquet", "json"):
            with self.subTest(output_format=output_format):
                out = self.folder()
                self.run_ok("residue-embedding", "--src-file", self.chains, "--output-folder", out,
                            "--output-format", output_format, "--output-name", "r")
                self.assert_files(out, _CHAINS, output_format, "r")

    def test_structure_embedding_is_parquet(self):
        """Its help said a pandas DataFrame file, and a JSON file."""
        out = self.folder()
        self.run_ok("structure-embedding", "--src-file", self.chains, "--output-folder", out, "--output-name", "s")
        self.assert_files(out, _CHAINS, "parquet", "s")

    def test_chain_and_assembly_embedding_are_csv_by_default(self):
        residues = self.residues()
        for command, src, names in (("chain-embedding", self.chains, _CHAINS),
                                    ("assembly-embedding", self.assemblies, _ASSEMBLIES)):
            with self.subTest(command=command):
                out = self.folder()
                self.run_ok(command, "--src-file", src, "--res-embedding-location", residues, "--output-folder", out)
                self.assert_files(out, names, "csv", "inference")

    def test_chain_and_assembly_embedding_in_each_format(self):
        residues = self.residues()
        for command, src, names in (("chain-embedding", self.chains, _CHAINS),
                                    ("assembly-embedding", self.assemblies, _ASSEMBLIES)):
            for output_format in ("csv", "pt", "parquet", "json"):
                with self.subTest(command=command, output_format=output_format):
                    out = self.folder()
                    self.run_ok(command, "--src-file", src, "--res-embedding-location", residues,
                                "--output-folder", out, "--output-format", output_format, "--output-name", "x")
                    self.assert_files(out, names, output_format, "x")

    def test_complete_embedding_in_each_format(self):
        """Residue embeddings as torch tensor files, chain and assembly embeddings in the format of --output-format;
        its help said a single JSON file, whatever the format."""
        for output_format in ("csv", "pt", "parquet", "json"):
            with self.subTest(output_format=output_format):
                res, chain, assembly = self.folder(), self.folder(), self.folder()
                self.run_ok("complete-embedding", "--src-chain-file", self.chains,
                            "--src-assembly-file", self.assemblies, "--output-res-path", res,
                            "--output-chain-path", chain, "--output-assembly-path", assembly,
                            "--output-format", output_format, "--output-chain-name", "c",
                            "--output-assembly-name", "a")
                self.assert_files(res, _CHAINS, "pt", "inference")
                self.assert_files(chain, _CHAINS, output_format, "c")
                self.assert_files(assembly, _ASSEMBLIES, output_format, "a")
