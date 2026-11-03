"""P9.H: what the inference commands write, as their help says, and what --structure-format and the default names of
complete-embedding do. ESM3 and the aggregator are never loaded: the commands run with stub models, as in
test_inference_follow_ups.py."""
import gzip
import json
import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq
import torch
from typer.testing import CliRunner

from foldmatch.cli import inference
# Imported before any test patches chain_inference.predict, which assembly_inferece binds at its import
from foldmatch.inference import assembly_inferece, chain_inference, esm_inference
from foldmatch.types.api_types import StructureFormat

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


def _flat(output: str) -> str:
    """The output on one line, typer's error panel borders dropped, and its colours (on with GITHUB_ACTIONS, FORCE_COLOR
    or PY_COLORS set)."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


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


class StructureFormatTest(_Lists):
    """chain-embedding reads residue embeddings and no structure, so it takes no --structure-format."""

    def test_chain_embedding_refuses_it(self):
        """Typer's usage error (exit status 2), before anything runs (before, it was taken and did nothing)."""
        with mock.patch.object(chain_inference, "predict") as chain:
            out = self.folder()
            result = _invoke("chain-embedding", "--src-file", self.chains, "--res-embedding-location", out,
                             "--output-folder", out, "--structure-format", "pdb")
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("No such option: --structure-format", _flat(result.output))
        chain.assert_not_called()

    def test_complete_embedding_keeps_it_for_the_steps_that_read_structures(self):
        """complete-embedding passes it to its residue and assembly steps, which read structures; its chain step does
        not get it."""
        out = self.folder()
        with mock.patch.object(esm_inference, "predict") as residue, \
                mock.patch.object(chain_inference, "predict") as chain, \
                mock.patch.object(assembly_inferece, "predict") as assembly:
            self.run_ok("complete-embedding", "--src-chain-file", self.chains, "--src-assembly-file", self.assemblies,
                        "--output-res-path", out, "--output-chain-path", out, "--output-assembly-path", out,
                        "--structure-format", "pdb")
        for step in (residue, chain, assembly):
            step.assert_called_once()
        self.assertEqual(residue.call_args.kwargs["structure_format"], StructureFormat.pdb)
        self.assertEqual(assembly.call_args.kwargs["structure_format"], StructureFormat.pdb)
        self.assertNotIn("structure_format", chain.call_args.kwargs)

    def test_chain_embedding_does_not_read_the_structure_file(self):
        """The list's structure files are not read: a list naming no file gives the chain embeddings of a list naming
        the real one, as the option's removal assumes."""
        with mock.patch("foldmatch.inference.chain_inference.get_aggregator_model", return_value=_aggregate):
            residues = self.folder()
            for name in _CHAINS:
                torch.save(torch.arange(6.0).reshape(3, 2) + ord(name[-1]), residues / f"{name}.pt")
            nowhere = self.tmp / "nowhere.csv"
            nowhere.write_text("1acb,/no/such/file.bcif,A,1acb.A\n1acb,,B,1acb.B\n")
            real, other = self.folder(), self.folder()
            self.run_ok("chain-embedding", "--src-file", self.chains, "--res-embedding-location", residues,
                        "--output-folder", real)
            self.run_ok("chain-embedding", "--src-file", nowhere, "--res-embedding-location", residues,
                        "--output-folder", other)
        for name in _CHAINS:
            self.assertEqual((real / f"{name}.csv").read_bytes(), (other / f"{name}.csv").read_bytes())


class CompleteEmbeddingNamesTest(_StubModels):
    """The default file names of complete-embedding's chain and assembly files differ: --output-assembly-name was
    chain-inference, the default of --output-chain-name, so that in one folder the assembly file replaced the chain
    file."""

    def run_complete(self, output_format: str, *names) -> Path:
        out = self.folder()
        self.run_ok("complete-embedding", "--src-chain-file", self.chains, "--src-assembly-file", self.assemblies,
                    "--output-res-path", self.folder(), "--output-chain-path", out, "--output-assembly-path", out,
                    "--output-format", output_format, *names)
        return out

    def test_one_folder_keeps_both_files(self):
        for output_format, postfix in (("parquet", "parquet"), ("json", "json.gz")):
            with self.subTest(output_format=output_format):
                out = self.run_complete(output_format)
                self.assertEqual(sorted(f.name for f in out.iterdir()),
                                 [f"assembly-inference-0.{postfix}", f"chain-inference-0.{postfix}"])
                self.assertCountEqual(_ids(out / f"chain-inference-0.{postfix}"), _CHAINS)
                self.assertCountEqual(_ids(out / f"assembly-inference-0.{postfix}"), _ASSEMBLIES)

    def test_the_names_given_are_used(self):
        out = self.run_complete("parquet", "--output-chain-name", "c", "--output-assembly-name", "a")
        self.assertEqual(sorted(f.name for f in out.iterdir()), ["a-0.parquet", "c-0.parquet"])

    def test_the_help_shows_the_default(self):
        result = CliRunner().invoke(inference.app, ["complete-embedding", "--help"], env={"COLUMNS": "400"})
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertRegex(_flat(result.output), r"--output-assembly-name TEXT .*\[default: assembly-inference\]")
