"""P9.N1: an inference run refuses names that conflict in its files.

One run never writes two of its outputs to one file, never writes an output onto one of its own inputs, and never
writes outside its output folder: such a run is refused (RefusalError, exit status 1) before any model is built. Each
refusal is checked before any model (the model getters fail if called), with its message, next to the same command
without the conflict, which still runs: reruns into the same folders, the default names of complete-embedding in one
folder, assembly names such as 1acb-1 beside residue files, Parquet ids with '/'. Lists give every row a name (D5).

ESM3 and the aggregator are never loaded: the commands run with stub models, as in tests/test_inference_outputs.py."""
import contextlib
import gzip
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from typer.testing import CliRunner

from foldmatch.cli import inference
from foldmatch.errors import RefusalError
# Imported before any test patches chain_inference.predict, which assembly_inferece binds at its import
from foldmatch.inference import assembly_inferece, chain_inference, esm_inference  # noqa: F401
from foldmatch.types.api_types import Accelerator, OutFormat, SrcLocation, SrcProteinFrom, SrcTensorFrom

_PDB = Path(__file__).parent / "resources" / "pdb"
_CIF, _CIF2 = _PDB / "1acb.cif", _PDB / "2uzi.cif"
_CPU = ("--accelerator", "cpu", "--devices", "1")
_LIB_CPU = {"accelerator": Accelerator.cpu, "devices": 1}
_FORMATS = ("csv", "pt", "parquet", "json")


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


def _flat(text: str) -> str:
    """Text on one line, typer's error panel borders and colour codes dropped."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


def _ids(path: Path) -> list:
    if path.suffix == ".parquet":
        return pq.read_table(path).column("id").to_pylist()
    with gzip.open(path, "rt") as f:
        return [row["id"] for row in json.load(f)]


def _tree(folder: Path) -> dict:
    """Every file under a folder, by relative path, with its bytes."""
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


def _case_insensitive() -> bool:
    """Whether the temporary folder takes 'A' and 'a' for one name (macOS APFS by default), found out here without
    FoldMatch, so that this module imports on trees without foldmatch.utils.output_files."""
    with tempfile.NamedTemporaryFile(prefix="CaseProbe-") as f:
        head, name = os.path.split(f.name)
        return os.path.exists(os.path.join(head, name.swapcase()))


def _output_files():
    """foldmatch.utils.output_files (P9.N), imported when a test needs it."""
    from foldmatch.utils import output_files
    return output_files


_CASE_INSENSITIVE = _case_insensitive()


class _Case(unittest.TestCase):

    def setUp(self):
        # Resolved, as typer resolves the paths the commands are given (/var is /private/var on macOS)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        for module in ("esm_inference", "structure_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_residue_model", self.residue_model))
        for module in ("structure_inference", "chain_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_aggregator_model", self.aggregator))
        for module in ("esm_module", "chain_complete_module"):
            self.enterContext(mock.patch(f"foldmatch.modules.{module}.per_residue_embedding", _esm3))
        self.models = 0
        self.refusing = False

    def residue_model(self, **_):
        self.models += 1
        if self.refusing:
            raise AssertionError("a model was built")
        return "ESM3"

    def aggregator(self, **_):
        self.models += 1
        if self.refusing:
            raise AssertionError("a model was built")
        return _aggregate

    def folder(self, name: str = "out") -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.tmp))

    def write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text)
        return path

    def run_ok(self, *args):
        self.refusing = False
        result = _invoke(*args, *_CPU)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        return result

    def refused(self, *args, written=()) -> str:
        """The command is refused (exit status 1, a RefusalError) before any model, and writes nothing in the folders
        given: its message, on one line."""
        before = {folder: _tree(folder) for folder in written}
        self.refusing = True
        try:
            result = _invoke(*args, *_CPU)
        finally:
            self.refusing = False
        self.assertEqual(result.exit_code, 1, f"{result.output}\n{result.exception!r}")
        self.assertIsInstance(result.exception, RefusalError, repr(result.exception))
        for folder, tree in before.items():
            self.assertEqual(_tree(folder), tree, folder)
        return _flat(str(result.exception))

    def residues(self, chains: Path, out_format: str = "pt") -> Path:
        out = self.folder("residues")
        self.run_ok("residue-embedding", "--src-file", chains, "--output-folder", out, "--output-format", out_format)
        return out

    def chain_list(self, *rows) -> Path:
        """A chain list of rows (structure name, chain ID, embedding name) of 1acb or 2uzi."""
        files = {"1acb": _CIF, "2uzi": _CIF2}
        return self.write(f"chains-{len(list(self.tmp.iterdir()))}.csv",
                          "".join(f"{s},{files[s]},{c},{n}\n" for s, c, n in rows))


class FileNameTest(_Case):
    """D1: with csv and pt, an embedding name is a file name of the output folder; the output name is one in every
    format."""

    def test_a_name_with_a_slash_is_refused(self):
        """'../victim' wrote outside the output folder (sweep case 10); 'Q9/1-100' stopped the run with a traceback.
        In Parquet and JSON an id is a string in a file, so both keep working there."""
        victim = self.write("victim.pt", "precious")
        chains = self.chain_list(("1acb", "A", "../victim"), ("1acb", "B", "Q9/1-100"))
        for out_format in ("csv", "pt"):
            with self.subTest(out_format=out_format):
                out = self.folder()
                message = self.refused("residue-embedding", "--src-file", chains, "--output-folder", out,
                                       "--output-format", out_format, written=[self.tmp])
                self.assertIn(f"2 embedding name(s) that cannot name a file <name>.{out_format} in the output folder: "
                              f"'../victim' (row 1) holds '/'; 'Q9/1-100' (row 2) holds '/'.", message)
                self.assertEqual(victim.read_text(), "precious")
        for out_format in ("parquet", "json"):
            with self.subTest(out_format=out_format):
                out = self.folder()
                self.run_ok("residue-embedding", "--src-file", chains, "--output-folder", out,
                            "--output-format", out_format)
                self.assertCountEqual(_ids(next(out.iterdir())), ["../victim", "Q9/1-100"])

    def test_dot_names_and_long_names(self):
        """'.', '..' and a name of more than 255 bytes with its suffix are refused; 251 bytes and '.csv' fit."""
        longest = "é" * 125 + "x"   # 251 bytes in UTF-8
        for name, problem in ((".", "is '.'"), ("..", "is '..'"), (longest + "y", "makes a file name of 256 bytes")):
            with self.subTest(name=name[:5]):
                message = self.refused("residue-embedding", "--src-file", self.chain_list(("1acb", "A", name)),
                                       "--output-folder", self.folder(), "--output-format", "csv")
                self.assertIn(f"{name!r} (row 1) {problem}", message)
                self.assertIn("of at most 251 bytes in UTF-8", message)
        out = self.folder()
        self.run_ok("residue-embedding", "--src-file", self.chain_list(("1acb", "A", longest)), "--output-folder", out,
                    "--output-format", "csv")
        self.assertEqual([f.name for f in out.iterdir()], [f"{longest}.csv"])

    def test_the_output_name_in_every_format(self):
        """--output-name '../up' wrote up-0.parquet outside the folder; it is refused in every format, as it names
        <name>-<rank>.parquet and .json.gz; 'inference' and a name of 240 bytes run."""
        chains = self.chain_list(("1acb", "A", "1acb.A"))
        for out_format in _FORMATS:
            with self.subTest(out_format=out_format):
                message = self.refused("residue-embedding", "--src-file", chains, "--output-folder", self.folder(),
                                       "--output-format", out_format, "--output-name", "../up", written=[self.tmp])
                self.assertEqual(message, "The output name '../up' holds '/': it names the files "
                                          "<name>-<rank>.parquet or <name>-<rank>.json.gz of the output folder. Give a "
                                          "name that is not empty, without '/' or NUL, other than '.' and '..', of at "
                                          "most 240 bytes in UTF-8.")
        message = self.refused("structure-embedding", "--src-file", chains, "--output-folder", self.folder(),
                               "--output-name", "x" * 241)
        self.assertIn(f"The output name {'x' * 241!r} makes a file name of 256 bytes once a rank and a suffix are "
                      f"added: it names the files", message)
        out = self.folder()
        self.run_ok("structure-embedding", "--src-file", chains, "--output-folder", out, "--output-name", "x" * 240)
        self.assertEqual([f.name for f in out.iterdir()], ["x" * 240 + "-0.parquet"])

    def test_an_output_name_that_is_empty_or_dots(self):
        """'', '.' and '..' are refused as what they are, not as what they would be with a rank and a suffix (the file
        would be '.-000000.parquet'), and the advice names the rule that refused ''."""
        chains = self.chain_list(("1acb", "A", "1acb.A"))
        for name, problem in (("", "is empty"), (".", "is '.'"), ("..", "is '..'")):
            with self.subTest(name=name):
                message = self.refused("residue-embedding", "--src-file", chains, "--output-folder", self.folder(),
                                       "--output-format", "parquet", "--output-name", name, written=[self.tmp])
                self.assertEqual(message, f"The output name {name!r} {problem}: it names the files "
                                          f"<name>-<rank>.parquet or <name>-<rank>.json.gz of the output folder. Give "
                                          f"a name that is not empty, without '/' or NUL, other than '.' and '..', of "
                                          f"at most 240 bytes in UTF-8.")

    def test_integer_ids_name_their_files(self):
        """chain_inference.predict on residue embeddings in Parquet whose id column is int64 writes 101.csv and 101.pt
        as it did; an integer id is not empty."""
        ints = self.tmp / "ints.parquet"
        pq.write_table(pa.table({"id": [101, 102], "embedding": [[[1.0, 2.0]], [[3.0, 4.0]]]},
                                schema=pa.schema([("id", pa.int64()),
                                                  ("embedding", pa.list_(pa.list_(pa.float32(), 2)))])), ints)
        for out_format in ("csv", "pt"):
            with self.subTest(out_format=out_format):
                out = self.folder()
                chain_inference.predict(src_stream=ints, src_from=SrcTensorFrom.parquet, out_folder=out,
                                        out_format=out_format, return_predictions=False, **_LIB_CPU)
                self.assertEqual(sorted(f.name for f in out.iterdir()), [f"101.{out_format}", f"102.{out_format}"])

    def test_the_rules(self):
        """name_problem, NUL included, and its 255-byte limit."""
        name_problem = _output_files().name_problem
        self.assertIsNone(name_problem("1acb.A", ".csv"))
        self.assertIsNone(name_problem("a" * 251, ".csv"))
        for name, problem in (("", "is empty"), (None, "is empty"), (float("nan"), "is empty"), (".", "is '.'"),
                              ("..", "is '..'"), ("a/b", "holds '/'"), ("a\0b", "holds a NUL character"),
                              ("a" * 252, "makes a file name of 256 bytes"), (pd.NA, "is empty"),
                              (np.float32("nan"), "is empty")):
            with self.subTest(name=repr(name)[:8]):
                self.assertEqual(name_problem(name, ".csv"), problem)
        for name in (101, np.int64(102), 0):
            with self.subTest(name=repr(name)):
                self.assertIsNone(name_problem(name, ".csv"))


class RepeatedNameTest(_Case):
    """D3: an embedding name given to more than one row is refused, in every format; with csv and pt, names that the
    output folder takes for one file too."""

    def test_residue_embedding(self):
        """Two rows named x kept the last one's embedding (sweep case 4); in Parquet and JSON two rows of id x."""
        chains = self.chain_list(("1acb", "A", "x"), ("1acb", "B", "x"))
        for out_format in _FORMATS:
            with self.subTest(out_format=out_format):
                message = self.refused("residue-embedding", "--src-file", chains, "--output-folder", self.folder(),
                                       "--output-format", out_format)
                self.assertEqual(message, f"{chains}: 1 embedding name(s) given to more than one row: 'x' (rows 1 and "
                                          f"2). Each row's embedding is written under its name (the file <name>.csv "
                                          f"or <name>.pt, or the id of its row in Parquet and JSON), so a repeated "
                                          f"name keeps one of them: give each row its own name (a missing or empty "
                                          f"name is <structure name>.<chain ID>).")
        out = self.folder()
        self.run_ok("residue-embedding", "--src-file", self.chain_list(("1acb", "A", "x"), ("1acb", "B", "y")),
                    "--output-folder", out)
        self.assertEqual(sorted(f.name for f in out.iterdir()), ["x.pt", "y.pt"])

    def test_every_list_command(self):
        """structure-embedding, chain-embedding and assembly-embedding, and complete-embedding's two lists; a default
        name repeated, an explicit one (rows counted from 1, 5 shown)."""
        residues = self.residues(self.chain_list(("1acb", "A", ""), ("1acb", "B", ""), ("2uzi", "A", ""),
                                                 ("2uzi", "B", ""), ("2uzi", "C", "")))
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "B", ""), ("1acb", "A", "1acb.A"))
        assemblies = self.write("assemblies.csv", f"1acb,{_CIF},1,\n2uzi,{_CIF2},1,1acb-1\n")
        found = "1 embedding name(s) given to more than one row: '1acb.A' (rows 1 and 3)"
        message = self.refused("structure-embedding", "--src-file", chains, "--output-folder", self.folder())
        self.assertIn(f"{chains}: {found}", message)
        message = self.refused("chain-embedding", "--src-file", chains, "--res-embedding-location", residues,
                               "--output-folder", self.folder(), "--output-format", "parquet")
        self.assertIn(f"{chains}: {found}", message)
        message = self.refused("assembly-embedding", "--src-file", assemblies, "--res-embedding-location", residues,
                               "--output-folder", self.folder(), "--output-format", "json")
        self.assertIn(f"{assemblies}: 1 embedding name(s) given to more than one row: '1acb-1' (rows 1 and 2)", message)
        self.assertIn("(a missing or empty name is <structure name>-<assembly ID>)", message)
        folders = [self.folder(name) for name in ("r", "c", "a")]
        for chain_list, assembly_list, source in ((chains, self.write("one.csv", f"1acb,{_CIF},1,\n"), chains),
                                                  (self.chain_list(("1acb", "A", "")), assemblies, assemblies)):
            with self.subTest(source=source.name):
                message = self.refused("complete-embedding", "--src-chain-file", chain_list, "--src-assembly-file",
                                       assembly_list, "--output-res-path", folders[0], "--output-chain-path",
                                       folders[1], "--output-assembly-path", folders[2], written=folders)
                self.assertIn(f"{source}: 1 embedding name(s) given to more than one row", message)
        many = self.chain_list(*[("1acb", "A", f"n{k // 2}") for k in range(14)])
        message = self.refused("structure-embedding", "--src-file", many, "--output-folder", self.folder())
        self.assertIn("7 embedding name(s) given to more than one row: 'n0' (rows 1 and 2); 'n1' (rows 3 and 4); 'n2' "
                      "(rows 5 and 6); 'n3' (rows 7 and 8); 'n4' (rows 9 and 10); and 2 more names.", message)
        self.run_ok("assembly-embedding", "--src-file", self.write("ok.csv", f"1acb,{_CIF},1,\n2uzi,{_CIF2},1,\n"),
                    "--res-embedding-location", residues, "--output-folder", self.folder())

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_names_that_differ_in_case_on_this_mac(self):
        """x and X give one file x.csv on a case-insensitive folder (the second replaced the first); in Parquet and
        JSON they are two ids, as on any file system; so are the NFC and NFD spellings of é."""
        nfc, nfd = "café", unicodedata.normalize("NFD", "café")
        for first, second, apart in (("x", "X", "upper and lower case"),
                                     (nfc, nfd, "Unicode spellings (NFC and NFD)")):
            chains = self.chain_list(("1acb", "A", first), ("1acb", "B", second))
            for out_format in ("csv", "pt"):
                with self.subTest(first=first, out_format=out_format):
                    message = self.refused("residue-embedding", "--src-file", chains, "--output-folder",
                                           self.folder(), "--output-format", out_format)
                    self.assertEqual(f"{chains}: 1 file(s) of the output folder would each be written for more than "
                                     f"one embedding name (the output folder does not tell {apart} apart): "
                                     f"{first!r} (row 1) and {second!r} (row 2) give {first}.{out_format}. Each one "
                                     f"would replace the one before, as the folder takes those names for one: write "
                                     f"them to a folder of a file system that tells {apart} apart or as parquet or "
                                     f"json (which keep each embedding name as given), or give those rows embedding "
                                     f"names that differ otherwise.", message)
            for out_format in ("parquet", "json"):
                with self.subTest(first=first, out_format=out_format):
                    out = self.folder()
                    self.run_ok("residue-embedding", "--src-file", chains, "--output-folder", out,
                                "--output-format", out_format)
                    self.assertCountEqual(_ids(next(out.iterdir())), [first, second])

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_names_equal_under_full_case_folding_on_this_mac(self):
        """APFS folds case fully: straße and strasse, ﬀ1 and ff1 are one file there (chain A's embedding was replaced
        by chain B's, exit status 0); refused with csv and pt. Parquet keeps both ids."""
        for first, second in (("straße", "strasse"), ("ﬀ1", "ff1")):
            chains = self.chain_list(("1acb", "A", first), ("1acb", "B", second))
            for out_format in ("csv", "pt"):
                with self.subTest(first=first, out_format=out_format):
                    out = self.folder()
                    message = self.refused("residue-embedding", "--src-file", chains, "--output-folder", out,
                                           "--output-format", out_format, written=[out])
                    self.assertIn(f"{first!r} (row 1) and {second!r} (row 2) give {first}.{out_format}.", message)
            out = self.folder()
            self.run_ok("residue-embedding", "--src-file", chains, "--output-folder", out, "--output-format", "parquet")
            self.assertCountEqual(_ids(next(out.iterdir())), [first, second])

    def test_full_and_simple_case_folding(self):
        """A folder that folds case fully (APFS) takes straße and strasse for one name; one that folds simply, one
        character for one, keeps them (the probe's result faked); İ and i stay two names under both."""
        of = _output_files()
        folder = self.folder()
        for full in (False, True):
            with self.subTest(full=full):
                folding = of.Folding(case=True, unicode=True, full=full)
                names = ["straße", "strasse", "ﬀ", "FF", "İ", "i"]
                repeats = of.OutputFiles(folder, folding).add_files(names, ".csv")
                self.assertEqual([file for file, _ in repeats], ["straße.csv", "ﬀ.csv"] if full else [])
                with mock.patch("foldmatch.utils.output_files.probe_folding", return_value=folding):
                    if full:
                        with self.assertRaises(RefusalError):
                            of.check_outputs(folder, OutFormat.pt, None, names=["straße", "STRASSE"])
                    else:
                        of.check_outputs(folder, OutFormat.pt, None, names=["straße", "STRASSE"])

    def test_the_comparison_on_a_case_sensitive_folder(self):
        """A folder that tells case apart (Linux; the probe's result faked here) keeps 'A' and 'a': lowercase chain IDs
        are real. One that does not refuses them; NFC and NFD only where the folder takes them for one name."""
        of = _output_files()
        EXACT, Folding, OutputFiles, name_key = of.EXACT, of.Folding, of.OutputFiles, of.name_key
        folder = self.folder()
        self.assertEqual(OutputFiles(folder, EXACT).add_files(["1abc.A", "1abc.a"], ".csv"), [])
        with mock.patch("foldmatch.utils.output_files.probe_folding", return_value=EXACT):
            outputs = of.check_outputs(folder, OutFormat.csv, None, names=["1abc.A", "1abc.a"])
        self.assertEqual(sorted(outputs.files), ["1abc.A.csv", "1abc.a.csv"])
        repeats = OutputFiles(folder, Folding(case=True)).add_files(["1abc.A", "1abc.a"], ".csv")
        self.assertEqual(repeats, [("1abc.A.csv", [("1abc.A", None), ("1abc.a", None)])])
        nfd = unicodedata.normalize("NFD", "é")
        self.assertEqual(OutputFiles(folder, Folding(case=True)).add_files(["é", nfd], ".pt"), [])
        self.assertEqual(len(OutputFiles(folder, Folding(unicode=True)).add_files(["é", nfd], ".pt")), 1)
        self.assertEqual(name_key("É", Folding(True, True)), "é")
        with mock.patch("foldmatch.utils.output_files.probe_folding", return_value=Folding(case=True)):
            with self.assertRaises(RefusalError):
                of.check_outputs(folder, OutFormat.csv, None, names=["1abc.A", "1abc.a"])

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_the_probe_of_this_mac(self):
        """The probe finds this folder case- and normalisation-insensitive, and leaves nothing behind."""
        of = _output_files()
        folder = self.folder()
        self.assertEqual(of.probe_folding(folder), of.Folding(case=True, unicode=True, full=True))
        self.assertEqual(list(folder.iterdir()), [])
        self.assertEqual(of.probe_folding(self.tmp / "missing"), of.EXACT)

    def test_the_library(self):
        """esm_inference.predict on a stream, chain_inference.predict on Parquet input with an id twice."""
        out = self.folder()
        self.refusing = True
        with self.assertRaises(RefusalError) as refused:
            esm_inference.predict(src_stream=[("1acb", str(_CIF), "A", "x"), ("1acb", str(_CIF), "B", "x")],
                                  src_location=SrcLocation.stream, out_folder=out, return_predictions=False,
                                  **_LIB_CPU)
        self.assertIn("The chain list stream: 1 embedding name(s) given to more than one row: 'x' (rows 1 and 2)",
                      str(refused.exception))
        twice = self.tmp / "twice.parquet"
        pq.write_table(pa.table({"id": ["x", "y", "x"], "embedding": [[[1.0, 2.0]]] * 3},
                                schema=pa.schema([("id", pa.string()),
                                                  ("embedding", pa.list_(pa.list_(pa.float32(), 2)))])), twice)
        for out_format in _FORMATS:
            with self.subTest(out_format=out_format):
                with self.assertRaises(RefusalError) as refused:
                    chain_inference.predict(src_stream=twice, src_from=SrcTensorFrom.parquet, out_folder=out,
                                            out_format=out_format, return_predictions=False, **_LIB_CPU)
                self.assertIn("1 id(s) held by more than one row: 'x'.", str(refused.exception))
        self.assertEqual(self.models, 0)
        self.assertEqual(list(out.iterdir()), [])
        self.refusing = False
        esm_inference.predict(src_stream=[("1acb", str(_CIF), "A", "x"), ("1acb", str(_CIF), "B", "y")],
                              src_location=SrcLocation.stream, out_folder=out, return_predictions=False, **_LIB_CPU)
        self.assertEqual(sorted(f.name for f in out.iterdir()), ["x.csv", "y.csv"])


class DefaultNameTest(_Case):
    """D5: every list gives every row a name; NA is a name."""

    def test_assembly_lists_without_names(self):
        """Two assemblies without a 4th column were both written to nan.csv (sweep case 5), and refused by pyarrow in
        Parquet (5b); they are now 1acb-1 and 2uzi-1, as fm-embedding names them. NA names NA.csv."""
        residues = self.residues(self.chain_list(("1acb", "A", ""), ("1acb", "B", ""), ("2uzi", "A", ""),
                                                 ("2uzi", "B", ""), ("2uzi", "C", "")))
        unnamed = self.write("unnamed.csv", f"1acb,{_CIF},1\n2uzi,{_CIF2},1,\n")
        for out_format in _FORMATS:
            with self.subTest(out_format=out_format):
                out = self.folder()
                self.run_ok("assembly-embedding", "--src-file", unnamed, "--res-embedding-location", residues,
                            "--output-folder", out, "--output-format", out_format)
                files = sorted(f.name for f in out.iterdir())
                if out_format in ("csv", "pt"):
                    self.assertEqual(files, [f"1acb-1.{out_format}", f"2uzi-1.{out_format}"])
                else:
                    self.assertCountEqual(_ids(out / files[0]), ["1acb-1", "2uzi-1"])
        out = self.folder()
        self.run_ok("assembly-embedding", "--src-file", self.write("na.csv", f"1acb,{_CIF},1,NA\n"),
                    "--res-embedding-location", residues, "--output-folder", out)
        self.assertEqual([f.name for f in out.iterdir()], ["NA.csv"])
        message = self.refused("assembly-embedding", "--src-file", self.write("twice.csv", f"1acb,{_CIF},1\n1acb,"
                                                                                           f"{_CIF},1,\n"),
                               "--res-embedding-location", residues, "--output-folder", self.folder())
        self.assertIn("'1acb-1' (rows 1 and 2)", message)

    def test_structure_lists_without_names(self):
        """A structure list without its 3rd column gave every structure the hidden files .A.csv, .B.csv (the last
        structure's); each now takes the structure name."""
        out = self.folder()
        esm_inference.predict(src_stream=self.write("structures.csv", f"1acb,{_CIF}\n2uzi,{_CIF2},\n"),
                              src_from=SrcProteinFrom.structure, out_folder=out, return_predictions=False,
                              **_LIB_CPU)
        self.assertEqual(sorted(f.name for f in out.iterdir()),
                         ["1acb.A.csv", "1acb.B.csv", "2uzi.A.csv", "2uzi.B.csv", "2uzi.C.csv"])
        self.refusing = True
        with self.assertRaises(RefusalError) as refused:
            esm_inference.predict(src_stream=self.write("same.csv", f"1acb,{_CIF},x\n2uzi,{_CIF2},x\n"),
                                  src_from=SrcProteinFrom.structure, out_folder=self.folder(),
                                  return_predictions=False, **_LIB_CPU)
        self.assertIn("1 embedding name(s) given to more than one row: 'x' (rows 1 and 2)", str(refused.exception))
        self.assertIn("(a missing or empty name is the structure name)", str(refused.exception))


class InputTest(_Case):
    """D2: an output never replaces an input of its run."""

    def test_chain_embedding_into_its_residue_folder(self):
        """In the residue format, every residue embedding became a chain vector (sweep case 6): refused, naming both
        options, also through a symbolic link to the folder; in another format the folder is shared as before."""
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "B", ""))
        for res_format, other in (("pt", "csv"), ("csv", "pt")):
            residues = self.residues(chains, res_format)
            link = self.tmp / f"link-{res_format}"
            link.symlink_to(residues)
            for folder in (residues, link):
                with self.subTest(res_format=res_format, folder=folder.name):
                    message = self.refused("chain-embedding", "--src-file", chains, "--res-embedding-location",
                                           residues, "--res-embedding-format", res_format, "--output-folder", folder,
                                           "--output-format", res_format, written=[residues])
                    self.assertEqual(message, f"--output-folder and --res-embedding-location are one folder "
                                              f"({folder.resolve()}), and --output-format is --res-embedding-format "
                                              f"({res_format}): each chain embedding <embedding name>.{res_format} "
                                              f"would replace the residue embedding it is computed from. Give another "
                                              f"--output-folder, or --output-format {other}, parquet or json.")
            self.run_ok("chain-embedding", "--src-file", chains, "--res-embedding-location", residues,
                        "--res-embedding-format", res_format, "--output-folder", residues, "--output-format", other)
            self.assertEqual(sorted(f.name for f in residues.iterdir()),
                             sorted(f"1acb.{c}.{fmt}" for c in "AB" for fmt in (res_format, other)))

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_a_folder_spelled_in_another_case(self):
        """On macOS, SHARED and shared are one folder: compared as files, not as strings."""
        shared = self.tmp / "shared"
        shared.mkdir()
        chains = self.chain_list(("1acb", "A", ""))
        self.run_ok("residue-embedding", "--src-file", chains, "--output-folder", shared)
        message = self.refused("chain-embedding", "--src-file", chains, "--res-embedding-location", shared,
                               "--output-folder", self.tmp / "SHARED", "--output-format", "pt")
        self.assertIn("--output-folder and --res-embedding-location are one folder", message)

    def test_the_list_and_the_structure_files(self):
        """A chain list in the output folder named like an output (x.csv, a row named x), and a structure file named
        like one: refused; a list named otherwise there runs."""
        out = self.folder()
        listed = out / "x.csv"
        listed.write_text(f"1acb,{_CIF},A,x\n")
        message = self.refused("residue-embedding", "--src-file", listed, "--output-folder", out, "--output-format",
                               "csv", written=[out])
        self.assertEqual(message, f"1 input(s) of this run would be replaced by its output, in the output folder "
                                  f"{out}: the chain list {listed} by the output 'x'. An output never replaces an "
                                  f"input of its run: write the output to another folder or under other names, or "
                                  f"move the input.")
        structure = Path(shutil.copyfile(_CIF, out / "s.A.pt"))
        message = self.refused("residue-embedding", "--src-file", self.write("s.csv", f"s,{structure},A,s.A\n"),
                               "--output-folder", out, "--output-format", "pt", written=[out])
        self.assertIn(f"the structure file of row 1 {structure} by the output 's.A'", message)
        message = self.refused("chain-embedding", "--src-file", listed, "--res-embedding-location", self.folder(),
                               "--output-folder", out, written=[out])
        self.assertIn(f"the chain list {listed} by the output 'x'", message)
        listed.rename(out / "list.csv")
        self.run_ok("residue-embedding", "--src-file", out / "list.csv", "--output-folder", out, "--output-format",
                    "csv")
        self.assertEqual(sorted(f.name for f in out.iterdir()), ["list.csv", "s.A.pt", "x.csv"])

    def test_assembly_names_beside_residue_files(self):
        """An assembly named <structure name>.<...>, written as pt to the pt residue folder, could replace a residue
        embedding: refused. 1acb-1 there runs, and so does 1acb.x in csv."""
        residues = self.residues(self.chain_list(("1acb", "A", ""), ("1acb", "B", "")))
        dotted = self.write("dotted.csv", f"1acb,{_CIF},1,1acb.A\n")
        message = self.refused("assembly-embedding", "--src-file", dotted, "--res-embedding-location", residues,
                               "--output-folder", residues, "--output-format", "pt", written=[residues])
        self.assertIn(f"{dotted}: 1 assembly embedding name(s) of the form <structure name>.<...>, written as pt files "
                      f"to the folder of the residue embeddings ({residues}): '1acb.A' (row 1).", message)
        self.run_ok("assembly-embedding", "--src-file", dotted, "--res-embedding-location", residues,
                    "--output-folder", residues, "--output-format", "csv")
        self.run_ok("assembly-embedding", "--src-file", self.write("plain.csv", f"1acb,{_CIF},1,\n"),
                    "--res-embedding-location", residues, "--output-folder", residues, "--output-format", "pt")
        self.assertEqual(sorted(f.name for f in residues.iterdir()),
                         ["1acb-1.pt", "1acb.A.csv", "1acb.A.pt", "1acb.B.pt"])

    def test_parquet_input_in_the_output_folder(self):
        """chain_inference.predict(src_from=parquet) on <F>/inference-0.parquet into F, default name: its output would
        replace its input; another name runs."""
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "B", ""))
        folder = self.residues(chains, "parquet")
        before = _tree(folder)
        self.refusing = True
        with self.assertRaises(RefusalError) as refused:
            chain_inference.predict(src_stream=folder / "inference-0.parquet", src_from=SrcTensorFrom.parquet,
                                    out_folder=folder, out_format=OutFormat.parquet, return_predictions=False,
                                    **_LIB_CPU)
        self.assertIn(f"the residue embedding Parquet file {folder / 'inference-0.parquet'} by a file "
                      f"<name>-<rank>.parquet of the output name 'inference'", str(refused.exception))
        self.assertEqual(_tree(folder), before)
        self.refusing = False
        chain_inference.predict(src_stream=folder / "inference-0.parquet", src_from=SrcTensorFrom.parquet,
                                out_folder=folder, out_format=OutFormat.parquet, out_name="chains",
                                return_predictions=False, **_LIB_CPU)
        self.assertCountEqual(_ids(folder / "chains-0.parquet"), ["1acb.A", "1acb.B"])


class CompleteEmbeddingTest(_Case):
    """complete-embedding checks its three steps before its residue step, and refuses two that would write one file."""

    def complete(self, folders, out_format: str, *names, refused: bool = False):
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "B", ""))
        assemblies = self.write("assemblies.csv", f"1acb,{_CIF},1,\n")
        args = ("complete-embedding", "--src-chain-file", chains, "--src-assembly-file", assemblies,
                "--output-res-path", folders[0], "--output-chain-path", folders[1], "--output-assembly-path",
                folders[2], "--output-format", out_format, *names)
        return self.refused(*args, written=set(folders)) if refused else self.run_ok(*args)

    def test_residues_and_chains_in_one_folder_as_pt(self):
        """The chain step replaced the residue files, and the assembly step failed (sweep case 3): refused before the
        residue step. In csv the folder is shared as before, and a rerun into the same folders runs."""
        shared = self.folder("shared")
        message = self.complete((shared, shared, self.folder("a")), "pt", refused=True)
        self.assertEqual(message, f"--output-res-path and --output-chain-path are one folder ({shared}), in which "
                                  f"complete-embedding's residue and chain embeddings would share 2 file(s): "
                                  f"1acb.A.pt; 1acb.B.pt. Each step would replace the other's files: give the steps "
                                  f"different folders, or an --output-format other than pt (residue embeddings are pt "
                                  f"files).")
        for _ in range(2):
            self.complete((shared, shared, shared), "csv")
        self.assertEqual(sorted(f.name for f in shared.iterdir()),
                         ["1acb-1.csv", "1acb.A.csv", "1acb.A.pt", "1acb.B.csv", "1acb.B.pt"])

    def test_one_name_for_chains_and_assemblies(self):
        """The recipe of P9.H2's CHANGELOG, --output-assembly-name chain-inference with one folder for chains and
        assemblies, lost the chain file (sweep case 1), and so did any one name (case 2): refused, in parquet and json.
        The default names in one folder run, in every format, and so do all three steps in one folder but in pt."""
        for out_format in ("parquet", "json"):
            for names in (("--output-assembly-name", "chain-inference"),
                          ("--output-chain-name", "X", "--output-assembly-name", "X")):
                with self.subTest(out_format=out_format, names=names):
                    shared = self.folder("shared")
                    message = self.complete((self.folder("r"), shared, shared), out_format, *names, refused=True)
                    self.assertIn(f"--output-chain-path and --output-assembly-path are one folder ({shared}), in which "
                                  f"complete-embedding's chain and assembly embeddings would share 1 file(s): "
                                  f"{names[-1]}-<rank>", message)
        for out_format in _FORMATS:
            with self.subTest(out_format=out_format, defaults=True):
                shared = self.folder("shared")
                for _ in range(2):
                    self.complete((self.folder("r"), shared, shared), out_format)
                self.assertEqual(len(list(shared.iterdir())), 3 if out_format in ("csv", "pt") else 2)
                if out_format != "pt":
                    everything = self.folder("everything")
                    self.complete((everything, everything, everything), out_format)

    def test_each_step_is_checked_first(self):
        """An assembly named 1acb.A beside pt residues, and an --output-chain-name that cannot name a file: refused
        before the residue step, naming the option."""
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "B", ""))
        shared = self.folder("shared")
        message = self.refused("complete-embedding", "--src-chain-file", chains, "--src-assembly-file",
                               self.write("dotted.csv", f"1acb,{_CIF},1,1acb.A\n"), "--output-res-path", shared,
                               "--output-chain-path", self.folder("c"), "--output-assembly-path", shared,
                               "--output-format", "pt", written=[shared])
        self.assertIn("1 assembly embedding name(s) of the form <structure name>.<...>", message)
        message = self.complete((self.folder("r"), self.folder("c"), self.folder("a")), "parquet",
                                "--output-chain-name", "a/b", refused=True)
        self.assertTrue(message.startswith("--output-chain-name 'a/b' holds '/'"), message)


class CompleteEmbeddingInputTest(_Case):
    """complete-embedding: no step's output replaces an input of the run, whichever step reads it (both lists and the
    structure files of both), and an assembly list that can be read once (/dev/fd/N, a pipe) gets every check."""

    def complete(self, chains: Path, assemblies, folders, out_format: str, *names, refused: bool = False,
                 kept=()):
        before = {path: path.read_bytes() for path in kept}
        args = ("complete-embedding", "--src-chain-file", chains, "--src-assembly-file", assemblies,
                "--output-res-path", folders[0], "--output-chain-path", folders[1], "--output-assembly-path",
                folders[2], "--output-format", out_format, *names)
        result = self.refused(*args) if refused else self.run_ok(*args)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content, path)
        return result

    def chains(self, path: Path = None) -> Path:
        path = path or self.tmp / "chains.csv"
        path.write_text(f"1acb,{_CIF},A,\n1acb,{_CIF},B,\n")
        return path

    def test_a_step_over_the_other_list(self):
        """The assembly embedding x.csv over the chain list ASM/x.csv (exit status 0, the list lost); the chain
        embedding 1acb.A.csv over the assembly list CHAIN/1acb.A.csv, and the residue embedding 1acb.A.pt over the
        assembly list RES/1acb.A.pt (each read later by the assembly step, which failed with a traceback): refused
        before any model, the lists unchanged. The same lists under other names run."""
        folders = [self.folder(name) for name in ("r", "c", "a")]
        chains = self.chains(folders[2] / "x.csv")
        named_x = self.write("x-assemblies.csv", f"1acb,{_CIF},1,x\n")
        message = self.complete(chains, named_x, folders, "csv", refused=True, kept=[chains])
        self.assertIn(f"the chain list {chains} by the output 'x'", message)
        for k, suffix in ((1, "csv"), (0, "pt")):
            assemblies = folders[k] / f"1acb.A.{suffix}"
            assemblies.write_text(f"1acb,{_CIF},1,\n")
            with self.subTest(step=k):
                message = self.complete(self.chains(), assemblies, folders, "csv", refused=True, kept=[assemblies])
                self.assertIn(f"the assembly list {assemblies} by the output '1acb.A'", message)
            assemblies.rename(folders[k] / "assemblies.txt")
            self.complete(self.chains(), folders[k] / "assemblies.txt", folders, "csv")
        self.assertEqual(sorted(f.name for f in folders[2].iterdir()), ["1acb-1.csv", "x.csv"])

    def test_a_structure_file_of_the_chain_list_under_an_assembly_name(self):
        """A structure file of the chain list, read by the residue step, in the assembly folder under the name of the
        assembly embedding: refused, naming the list."""
        folders = [self.folder(name) for name in ("r", "c", "a")]
        structure = Path(shutil.copyfile(_CIF, folders[2] / "1acb-1.csv"))
        chains = self.write("chains-structure.csv", f"1acb,{structure},A,\n1acb,{structure},B,\n")
        message = self.complete(chains, self.write("assemblies.csv", f"1acb,{_CIF},1,\n"), folders, "csv",
                                refused=True, kept=[structure])
        self.assertIn(f"the structure file of row 1 of the chain list {structure} by the output '1acb-1'", message)

    @unittest.skipUnless(os.path.isdir("/dev/fd"), "/dev/fd/N needs /dev/fd")
    def test_an_assembly_list_as_dev_fd(self):
        """An assembly list given as /dev/fd/N, with one folder for chains and assemblies: one name for both in
        Parquet or JSON, and an assembly named 1acb.A in csv or pt, replaced the chain embeddings with exit status 0.
        Refused as with the list in a regular file; the default names run, and the assembly is written."""
        cases = [(fmt, ("--output-chain-name", "X", "--output-assembly-name", "X"), "1acb-1", "X-<rank>")
                 for fmt in ("parquet", "json")]
        cases += [(fmt, (), "1acb.A", f"1acb.A.{fmt}") for fmt in ("csv", "pt")]
        for out_format, names, assembly, shared_file in cases:
            with self.subTest(out_format=out_format):
                listed = self.write(f"assemblies-{out_format}.csv", f"1acb,{_CIF},1,{assembly}\n")
                folders = [self.folder("r")] + [self.folder("shared")] * 2
                fd = os.open(listed, os.O_RDONLY)
                try:
                    message = self.complete(self.chains(), f"/dev/fd/{fd}", folders, out_format, *names, refused=True)
                finally:
                    os.close(fd)
                self.assertIn(f"--output-chain-path and --output-assembly-path are one folder ({folders[1]}), in "
                              f"which complete-embedding's chain and assembly embeddings would share 1 file(s): "
                              f"{shared_file}", message)
                self.assertEqual(list(folders[0].iterdir()), [])
        listed = self.write("assemblies-ok.csv", f"1acb,{_CIF},1,\n")
        folders = [self.folder("r")] + [self.folder("shared")] * 2
        fd = os.open(listed, os.O_RDONLY)
        try:
            self.complete(self.chains(), f"/dev/fd/{fd}", folders, "csv")
        finally:
            os.close(fd)
        self.assertEqual(sorted(f.name for f in folders[1].iterdir()), ["1acb-1.csv", "1acb.A.csv", "1acb.B.csv"])

    @unittest.skipUnless(os.path.isdir("/dev/fd"), "/dev/fd/N needs /dev/fd")
    def test_a_long_row_in_an_assembly_list_as_dev_fd(self):
        """A row of more than 4 fields in an assembly list given as /dev/fd/N is refused before the residue step too,
        naming the path given, also where os.path.realpath resolves /dev/fd/N to the file the descriptor is open on
        (Linux; emulated elsewhere)."""
        listed = self.write("long.csv", f"1acb,{_CIF},1,1acb-1,extra\n")
        realpath = os.path.realpath
        for linux in (False, True):
            with self.subTest(linux=linux):
                folders = [self.folder(name) for name in ("r", "c", "a")]
                fd = os.open(listed, os.O_RDONLY)
                given = f"/dev/fd/{fd}"
                as_linux = (lambda path, *args, **kwargs: os.fspath(listed) if os.fspath(path) == given
                            else realpath(path, *args, **kwargs))
                try:
                    with mock.patch("os.path.realpath", as_linux) if linux else contextlib.nullcontext():
                        message = self.complete(self.chains(), given, folders, "csv", refused=True)
                finally:
                    os.close(fd)
                self.assertTrue(message.startswith(f"{given}: "), message)
                self.assertEqual(list(folders[0].iterdir()), [])


class PrefixTest(_Case):
    """The files <embedding name>.<chain ID> and <embedding name>-<assembly ID> of a structure list, whose IDs are
    known only once the structure is read: looked up per name, and prefixes that the folder takes for one refused."""

    def test_structures_in_the_output_folder_scale(self):
        """N structure files in the output folder were checked against N prefixes each, N^2 steps (213 s for 50,000
        names); a lookup per separator now: 20,000 names in well under a second."""
        of = _output_files()
        folder = self.folder()
        names = [f"s{k}" for k in range(20000)]
        inputs = [("the structure file", folder / f"{name}.cif") for name in names]
        start = time.perf_counter()
        outputs = of.check_outputs(folder, OutFormat.pt, None, prefixes=names, inputs=inputs,
                                   folding=of.Folding(True, True, True))
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertEqual(outputs.holds("S7.A.pt"), "a file s7.<chain or assembly ID>.pt of the output 's7'")
        self.assertEqual(outputs.holds("s7.b.c.pt"), "a file s7.<chain or assembly ID>.pt of the output 's7'")
        for name in ("s7.pt", "s7..pt", ".A.pt", "s7.A.csv", "x7.A.pt", "s20000.A.pt"):
            self.assertIsNone(outputs.holds(name), name)
        nested = of.OutputFiles(folder, of.EXACT)
        nested.add_prefixes(["1abc.v2", "1abc"], "-", ".pt")
        self.assertIn("output '1abc'", nested.holds("1abc-1.pt"))
        self.assertIn("output '1abc.v2'", nested.holds("1abc.v2-1.pt"))

    def test_prefixes_that_differ_in_case(self):
        """Structures abc and ABC both wrote abc.A.pt on APFS, after the models, stopped by the writer only within one
        process: refused before work where the folder takes them for one; a case-sensitive folder keeps both."""
        of = _output_files()
        folder = self.folder()
        with self.assertRaises(RefusalError) as refused:
            of.check_outputs(folder, OutFormat.pt, None, prefixes=["abc", "x", "ABC"], separator="-",
                             labels=["row 1", "row 2", "row 3"], folding=of.Folding(case=True), source="list.csv")
        self.assertEqual(str(refused.exception),
                         "list.csv: 1 file(s) of the output folder would each be written for more than one embedding "
                         "name (the output folder does not tell upper and lower case apart): 'abc' (row 1) and 'ABC' "
                         "(row 3) give abc-<ID>.pt. Each one would replace the one before, as the folder takes those "
                         "names for one: write them to a folder of a file system that tells upper and lower case apart "
                         "or as parquet or json (which keep each embedding name as given), or give those rows "
                         "embedding names that differ otherwise.")
        of.check_outputs(folder, OutFormat.pt, None, prefixes=["abc", "ABC"], folding=of.EXACT)
        of.check_outputs(folder, OutFormat.parquet, None, prefixes=["abc", "ABC"], folding=of.Folding(case=True))

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_structure_names_that_differ_in_case_on_this_mac(self):
        """esm_inference.predict on a structure list of abc and ABC, as pt: refused before any model."""
        rows = [("abc", str(_CIF), "abc"), ("ABC", str(_CIF2), "ABC")]
        out = self.folder()
        self.refusing = True
        with self.assertRaises(RefusalError) as refused:
            esm_inference.predict(src_stream=rows, src_location=SrcLocation.stream, src_from=SrcProteinFrom.structure,
                                  out_folder=out, out_format=OutFormat.pt, return_predictions=False, **_LIB_CPU)
        self.assertIn("'abc' (row 1) and 'ABC' (row 2) give abc.<ID>.pt", str(refused.exception))
        self.assertEqual((self.models, list(out.iterdir())), (0, []))
        self.refusing = False
        esm_inference.predict(src_stream=rows, src_location=SrcLocation.stream, src_from=SrcProteinFrom.structure,
                              out_folder=out, out_format=OutFormat.parquet, return_predictions=False, **_LIB_CPU)

    def test_ranks_in_ascii_digits(self):
        """The writers spell the rank in ASCII digits: emb-٣.parquet (Arabic-Indic 3) is no output's file."""
        of = _output_files()
        outputs = of.OutputFiles(self.folder(), of.Folding(True, True, True))
        outputs.add_stem("emb", ".parquet")
        self.assertIsNotNone(outputs.holds("emb-3.parquet"))
        for digit in ("٣", "３", "𝟑", "๓"):
            self.assertIsNone(outputs.holds(f"emb-{digit}.parquet"), digit)


class FoldedNameTest(_Case):
    """Names that the folder takes for one: compared as the folder compares them, and the refusal says what keeps
    them apart."""

    def test_canonical_caseless_keys(self):
        """APFS takes 'ᾷ' and 'ᾼ' with U+0342 for one name; case folding before decomposing missed it."""
        of = _output_files()
        folding = of.Folding(True, True, True)
        for first, second in (("xᾷ", "xᾼ͂"), ("xῇ", "xῌ͂"), ("xῷ", "xῼ͂")):
            with self.subTest(first=first):
                self.assertEqual(of.name_key(first, folding), of.name_key(second, folding))
                self.assertNotEqual(of.name_key(first, of.EXACT), of.name_key(second, of.EXACT))
                with self.assertRaises(RefusalError):
                    of.check_outputs(self.folder(), OutFormat.pt, None, names=[first, second], folding=folding)
        self.assertEqual(of.name_key("STRAẞE", folding), of.name_key("strasse", folding))

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_complete_embedding_chains_a_and_a_on_this_mac(self):
        """complete-embedding names each chain <structure name>.<chain ID>, so its chains A and a of one structure
        cannot be renamed, and its residue step writes pt: the refusal names the folder as what keeps them apart."""
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "a", ""))
        folders = [self.folder(name) for name in ("r", "c", "a")]
        message = self.refused("complete-embedding", "--src-chain-file", chains, "--src-assembly-file",
                               self.write("asm.csv", f"1acb,{_CIF},1\n"), "--output-res-path", folders[0],
                               "--output-chain-path", folders[1], "--output-assembly-path", folders[2],
                               "--output-format", "parquet", written=folders)
        self.assertEqual(message, f"{chains}: 1 file(s) of the output folder would each be written for more than one "
                                  f"embedding name (the output folder does not tell upper and lower case apart): "
                                  f"'1acb.A' (row 1) and '1acb.a' (row 2) give 1acb.A.pt. Each one would replace the "
                                  f"one before, as the folder takes those names for one: give --output-res-path a "
                                  f"folder of a file system that tells upper and lower case apart (complete-embedding "
                                  f"names each chain <structure name>.<chain ID>, and its residue step writes pt "
                                  f"files).")

    def test_complete_embedding_chain_step(self):
        """The residue folder tells case apart, the chain folder does not (the probe's results faked): the chain step
        is refused with csv, naming its folder and Parquet or JSON."""
        of = _output_files()
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "a", ""))
        folders = [self.folder(name) for name in ("r", "c", "a")]

        def probe(folder):
            return of.EXACT if of.same_folder(folder, folders[0]) else of.Folding(case=True)
        with mock.patch("foldmatch.utils.output_files.probe_folding", side_effect=probe):
            message = self.refused("complete-embedding", "--src-chain-file", chains, "--src-assembly-file",
                                   self.write("asm.csv", f"1acb,{_CIF},1\n"), "--output-res-path", folders[0],
                                   "--output-chain-path", folders[1], "--output-assembly-path", folders[2],
                                   "--output-format", "csv", written=folders)
        self.assertTrue(message.endswith("as the folder takes those names for one: give --output-chain-path a folder "
                                         "of a file system that tells upper and lower case apart, or use "
                                         "--output-format parquet or json (complete-embedding names each chain "
                                         "<structure name>.<chain ID>)."), message)
        self.assertIn("'1acb.A' (row 1) and '1acb.a' (row 2) give 1acb.A.csv", message)


class LinkedInputTest(_Case):
    """D2: an input reached through a symbolic or a hard link from another folder is compared as a file."""

    def stage(self, residues: Path) -> Path:
        """A folder of links to the residue embeddings: 1acb.A.pt a symbolic link, 1acb.B.pt a hard link."""
        stage = self.folder("stage")
        (stage / "1acb.A.pt").symlink_to(residues / "1acb.A.pt")
        os.link(residues / "1acb.B.pt", stage / "1acb.B.pt")
        return stage

    def test_chain_embedding_over_linked_residues(self):
        """chain-embedding on links to residue embeddings in its output folder replaced each by its chain vector:
        refused, naming the file linked; into another folder, or in another format, it runs."""
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "B", ""))
        residues = self.residues(chains)
        stage = self.stage(residues)
        message = self.refused("chain-embedding", "--src-file", chains, "--res-embedding-location", stage,
                               "--output-folder", residues, "--output-format", "pt", written=[residues])
        self.assertIn(f"2 input(s) of this run would be replaced by its output, in the output folder {residues}: the "
                      f"residue embedding file of row 1 {stage}/1acb.A.pt (which is {residues}/1acb.A.pt) by the "
                      f"output '1acb.A'; the residue embedding file of row 2 {stage}/1acb.B.pt (which is "
                      f"{residues}/1acb.B.pt) by the output '1acb.B'.", message)
        for out_folder, out_format in ((self.folder(), "pt"), (residues, "csv")):
            self.run_ok("chain-embedding", "--src-file", chains, "--res-embedding-location", stage,
                        "--output-folder", out_folder, "--output-format", out_format)
        self.assertEqual(torch.load(residues / "1acb.A.pt").dim(), 2)

    def test_assembly_embedding_over_linked_residues(self):
        """An assembly named 1acb.A, written as pt to the folder whose residue embeddings are linked from the residue
        folder given: refused; 1acb-1 runs."""
        residues = self.residues(self.chain_list(("1acb", "A", ""), ("1acb", "B", "")))
        stage = self.stage(residues)
        dotted = self.write("dotted.csv", f"1acb,{_CIF},1,1acb.A\n")
        message = self.refused("assembly-embedding", "--src-file", dotted, "--res-embedding-location", stage,
                               "--output-folder", residues, "--output-format", "pt", written=[residues])
        self.assertIn(f"the residue embedding file {stage}/1acb.A.pt (which is {residues}/1acb.A.pt) by the output "
                      f"'1acb.A'", message)
        self.run_ok("assembly-embedding", "--src-file", self.write("plain.csv", f"1acb,{_CIF},1,\n"),
                    "--res-embedding-location", stage, "--output-folder", residues, "--output-format", "pt")

    def test_the_check(self):
        """check_outputs with a symbolic link, a hard link, and a file of another name linked."""
        of = _output_files()
        out, stage = self.folder(), self.folder("stage")
        (out / "y.pt").write_bytes(b"y")
        (stage / "x.pt").symlink_to(out / "y.pt")
        os.link(out / "y.pt", stage / "hard.pt")
        (stage / "other.pt").write_bytes(b"o")
        os.link(stage / "other.pt", stage / "other-too.pt")
        for name in ("x.pt", "hard.pt"):
            with self.subTest(name=name):
                with self.assertRaises(RefusalError) as refused:
                    of.check_outputs(out, OutFormat.pt, None, names=["y"], inputs=[("the input", stage / name)])
                self.assertIn(f"the input {stage / name} (which is {out / 'y.pt'}) by the output 'y'",
                              str(refused.exception))
        of.check_outputs(out, OutFormat.pt, None, names=["z"], inputs=[("the input", stage / "x.pt")])
        of.check_outputs(out, OutFormat.pt, None, names=["y"], inputs=[("the input", stage / "other.pt")])
        of.check_outputs(self.folder(), OutFormat.pt, None, names=["y"], inputs=[("the input", stage / "x.pt")])

    def test_an_output_file_linked_to_an_input(self):
        """An output's file that exists as a symbolic link to an input of another folder (out/x.pt to in/x.pt) is that
        input, which the writer, following the link, replaced: refused; a link to a file that is no input is not."""
        of = _output_files()
        out, inputs, other = self.folder(), self.folder("in"), self.folder("other")
        (inputs / "x.pt").write_bytes(b"x")
        (out / "x.pt").symlink_to(inputs / "x.pt")
        with self.assertRaises(RefusalError) as refused:
            of.check_outputs(out, OutFormat.pt, None, names=["x"], inputs=[("the input", inputs / "x.pt")])
        self.assertIn(f"the input {inputs / 'x.pt'} (which is {out / 'x.pt'}) by the output 'x'",
                      str(refused.exception))
        (other / "y.pt").write_bytes(b"y")
        unrelated = self.folder()
        (unrelated / "y.pt").symlink_to(other / "y.pt")
        of.check_outputs(unrelated, OutFormat.pt, None, names=["y"], inputs=[("the input", inputs / "x.pt")])

    def test_no_link_among_the_output_files(self):
        """Where no output's file that exists is a link, an input that is not a link is not looked up beyond its
        lstat: one lstat per output file and per input, and a stat per folder."""
        of = _output_files()
        out, inputs, n = self.folder(), self.folder("in"), 200
        for k in range(n):
            (out / f"s{k}.pt").write_bytes(b"o")
            (inputs / f"s{k}.cif").write_bytes(b"i")
        folding = of.probe_folding(out)
        with mock.patch.object(of.os, "stat", wraps=os.stat) as stat, \
                mock.patch.object(of.os, "lstat", wraps=os.lstat) as lstat:
            of.check_outputs(out, OutFormat.pt, None, names=[f"s{k}" for k in range(n)], folding=folding,
                             inputs=[("the structure file", inputs / f"s{k}.cif") for k in range(n)])
        self.assertLessEqual(stat.call_count + lstat.call_count, 2 * n + 4)

    def test_chain_embedding_into_links_to_its_residues(self):
        """chain-embedding into a folder whose 1acb.A.pt is a symbolic link to the residue embedding it reads
        replaced that residue embedding by a chain vector (exit status 0): refused, naming the link; a link to a file
        that is no input of the run is replaced through the link, as before."""
        chains = self.chain_list(("1acb", "A", ""), ("1acb", "B", ""))
        residues = self.residues(chains)
        out = self.folder()
        (out / "1acb.A.pt").symlink_to(residues / "1acb.A.pt")
        message = self.refused("chain-embedding", "--src-file", chains, "--res-embedding-location", residues,
                               "--output-folder", out, "--output-format", "pt", written=[residues, out])
        self.assertIn(f"1 input(s) of this run would be replaced by its output, in the output folder {out}: the "
                      f"residue embedding file of row 1 {residues}/1acb.A.pt (which is {out}/1acb.A.pt) by the "
                      f"output '1acb.A'.", message)
        other, out = self.folder("other"), self.folder()
        torch.save(torch.zeros(3), other / "1acb.A.pt")
        (out / "1acb.A.pt").symlink_to(other / "1acb.A.pt")
        before = _tree(residues)
        self.run_ok("chain-embedding", "--src-file", chains, "--res-embedding-location", residues,
                    "--output-folder", out, "--output-format", "pt")
        fresh = self.folder()
        self.run_ok("chain-embedding", "--src-file", chains, "--res-embedding-location", residues,
                    "--output-folder", fresh, "--output-format", "pt")
        self.assertTrue((out / "1acb.A.pt").is_symlink())
        self.assertEqual(torch.load(other / "1acb.A.pt").tolist(), torch.load(fresh / "1acb.A.pt").tolist())
        self.assertEqual(_tree(residues), before)


class ReadOnlyFolderTest(_Case):
    """A folder in which no file can be created, though its files can be written, is probed on the names it holds."""

    def read_only(self, *files) -> Path:
        folder = self.folder()
        for name in files:
            (folder / name).write_text("old\n")
        folder.chmod(0o555)
        self.addCleanup(folder.chmod, 0o755)
        if os.access(folder, os.W_OK):
            self.skipTest("the folder stays writable (root)")
        return folder

    def test_the_probe(self):
        """As the probe of a folder of the same file system that can be written."""
        of = _output_files()
        self.assertEqual(of.probe_folding(self.read_only("x.csv")), of.probe_folding(self.folder()))

    def test_a_sibling_that_links_to_the_folder(self):
        """On a case-sensitive file system, a sibling OUT that links to the folder out is not out in another case: the
        folder compares exactly, its entries probed first, and A and a are two files. Simulated here: the parent lists
        the sibling, and only the sibling names the folder."""
        of = _output_files()
        listdir = os.listdir
        for files in (("x.csv",), ()):
            folder = self.read_only(*files)
            alias = str(folder.parent / of._swap_case(folder.name))

            def names_it(spelling, path, folder=folder, alias=alias):
                return os.fspath(spelling) == os.fspath(path) or (os.fspath(spelling) == alias and
                                                                  os.fspath(path) == str(folder))

            def listed(path=".", folder=folder, alias=alias):
                entries = listdir(path)
                return entries + [os.path.basename(alias)] if os.fspath(path) == str(folder.parent) else entries

            with self.subTest(files=files), mock.patch.object(of, "_names_it", names_it), \
                    mock.patch.object(of.os, "listdir", listed):
                self.assertEqual(of.probe_folding(folder), of.EXACT)
                of.check_outputs(folder, OutFormat.csv, None, names=["A", "a"])

    @unittest.skipIf(_CASE_INSENSITIVE, "a case-sensitive file system")
    def test_a_sibling_that_links_to_the_folder_here(self):
        """As test_a_sibling_that_links_to_the_folder, on this case-sensitive file system."""
        of = _output_files()
        for files in (("x.csv",), ()):
            folder = self.read_only(*files)
            alias = folder.parent / of._swap_case(folder.name)
            alias.symlink_to(folder)
            with self.subTest(files=files):
                self.assertEqual(of.probe_folding(folder), of.EXACT)
                of.check_outputs(folder, OutFormat.csv, None, names=["A", "a"])

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_names_that_differ_in_case_on_this_mac(self):
        """X and x both went to the earlier run's x.csv, the second kept (exit status 0): refused. A rerun of x
        runs."""
        out = self.read_only("x.csv")
        message = self.refused("residue-embedding", "--src-file", self.chain_list(("1acb", "A", "X"),
                                                                                  ("1acb", "B", "x")),
                               "--output-folder", out, "--output-format", "csv", written=[out])
        self.assertIn("'X' (row 1) and 'x' (row 2) give X.csv", message)
        self.run_ok("residue-embedding", "--src-file", self.chain_list(("1acb", "B", "x")), "--output-folder", out,
                    "--output-format", "csv")
        self.assertEqual([f.name for f in out.iterdir()], ["x.csv"])


class ErrorLineTest(_Case):
    """The refusal reaches the shell as one 'Error:' line, exit status 1, no traceback."""

    def test_main(self):
        chains = self.chain_list(("1acb", "A", "x"), ("1acb", "B", "x"))
        stderr = io.StringIO()
        self.refusing = True
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            with mock.patch.object(sys, "argv", ["inference", "residue-embedding", "--src-file", str(chains),
                                                 "--output-folder", str(self.folder())]), \
                    contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exit_status:
                inference.main()
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
        self.assertEqual(exit_status.exception.code, 1)
        lines = re.sub(r"\x1b\[[0-9;]*m", "", stderr.getvalue()).splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith(f"Error: {chains}: 1 embedding name(s) given to more than one row"))
        self.assertEqual(self.models, 0)


if __name__ == "__main__":
    unittest.main()
