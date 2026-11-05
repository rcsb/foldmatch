"""P9.N2: fm-embedding refuses names that conflict in its files.

from-sequences refuses, before ESM3, a FASTA id repeated (every format), and with csv and pt an id that cannot name a
file of the output folder or that the folder takes for another one, and a FASTA file among its own output files; the
output name follows the file name rules in every format (from-structures too). CsvBatchWriter and TensorBatchWriter
refuse a second write of one file in a run (D4), the backstop for the chains of structures read whole, whose names are
known only once a structure is read (chains A and a are one file on a case-insensitive folder).

ESM3 and the aggregator are never loaded: the commands run with stub models (as in tests/test_structure_lists.py)."""
import logging
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pyarrow.parquet as pq
import torch
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure
from typer.testing import CliRunner

from foldmatch.cli import embedding
from foldmatch.errors import RefusalError
# Imported before any test patches their model getters
from foldmatch.inference import assembly_complete_inference, chain_complete_inference, esm_inference, \
    sequence_inference  # noqa: F401
from foldmatch.writer.batch_writer import CsvBatchWriter, TensorBatchWriter

_CIF = Path(__file__).parent / "resources" / "pdb" / "1acb.cif"
_CPU = ("--accelerator", "cpu", "--devices", "1")
_FORMATS = ("csv", "pt", "parquet", "json")


def _esm3(model, protein):
    """Stands in for ESM3: L + 2 rows (BOS and EOS, zeros), each residue's row its letter and its position."""
    rows = [[float(ord(aa)), float(k)] for k, aa in enumerate(protein.sequence)]
    return torch.tensor([[0.0] * 2] + rows + [[0.0] * 2])


def _aggregate(x, x_mask):
    """Stands in for the aggregator: the sum of each item's rows."""
    return x.masked_fill(x_mask.unsqueeze(-1), 0.0).sum(dim=1)


def _flat(text: str) -> str:
    """Text on one line, typer's error panel borders and colour codes dropped."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


def _ids(path: Path) -> list:
    return pq.read_table(path).column("id").to_pylist()


def _case_insensitive() -> bool:
    """Whether the temporary folder takes 'A' and 'a' for one name (macOS APFS by default), found out without
    FoldMatch."""
    with tempfile.NamedTemporaryFile(prefix="CaseProbe-") as f:
        head, name = os.path.split(f.name)
        return os.path.exists(os.path.join(head, name.swapcase()))


_CASE_INSENSITIVE = _case_insensitive()


class _Case(unittest.TestCase):

    def setUp(self):
        # Resolved, as typer resolves the paths the commands are given (/var is /private/var on macOS)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        for module in ("esm_inference", "sequence_inference", "chain_complete_inference",
                       "assembly_complete_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_residue_model", self.model("ESM3")))
        for module in ("chain_complete_inference", "assembly_complete_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_aggregator_model",
                                         self.model(_aggregate)))
        for module in ("esm_module", "chain_complete_module", "assembly_complete_module"):
            self.enterContext(mock.patch(f"foldmatch.modules.{module}.per_residue_embedding", _esm3))
        self.refusing = False
        self.models = 0

    def model(self, model):
        def get(**_):
            self.models += 1
            if self.refusing:
                raise AssertionError("a model was built")
            return model
        return get

    def folder(self, name: str = "out") -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.tmp))

    def fasta(self, *records, name: str = None) -> Path:
        path = Path(name) if name else self.tmp / f"q{len(list(self.tmp.iterdir()))}.fasta"
        path.write_text("".join(f">{header}\n{sequence}\n" for header, sequence in records))
        return path

    def invoke(self, *args):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            return CliRunner().invoke(embedding.app, [str(a) for a in (*args, *_CPU)])
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)

    def run_ok(self, *args):
        self.refusing = False
        result = self.invoke(*args)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")

    def refused(self, *args, written=()) -> str:
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


class FastaIdTest(_Case):

    def test_a_repeated_id_in_every_format(self):
        """A repeated id kept the last record's embedding in csv and pt, and two rows of one id in Parquet and JSON:
        refused before ESM3, as the residue-code pass refuses it; a record under min_res_n is not written, so its id
        may repeat a kept one's. A rerun into the same folder runs."""
        repeated = self.fasta(("a first", "MKV"), ("b", "MKVL"), ("a second", "MKVLA"))
        for command in ("residue", "chain"):
            for out_format in _FORMATS:
                with self.subTest(command=command, out_format=out_format):
                    message = self.refused("from-sequences", command, "--fasta-file", repeated, "--output-folder",
                                           self.folder(), "--output-format", out_format)
                    self.assertEqual(message, f"{repeated}: 1 record(s) repeat an earlier id (e.g. a). fm-embedding "
                                              f"writes each record's embedding under its id, the first word of its "
                                              f"header (the file <id>.csv or <id>.pt, or the id of its row in Parquet "
                                              f"and JSON), so a repeated id keeps one of them: give each record its "
                                              f"own id.")
            out = self.folder()
            for _ in range(2):
                self.run_ok("from-sequences", command, "--fasta-file", repeated, "--output-folder", out,
                            "--min-res-n", "4")
            self.assertEqual(sorted(f.name for f in out.iterdir()), ["a.csv", "b.csv"])

    def test_an_id_that_cannot_name_a_file(self):
        """'../precious' replaced a file outside the output folder, 'Q9/1-100' stopped the run with a traceback: refused
        in csv and pt, before ESM3, naming the records; Parquet keeps both ids."""
        precious = self.tmp / "precious.csv"
        precious.write_text("keep")
        fasta = self.fasta(("../precious", "MKV"), ("Q9/1-100", "MKVL"))
        for out_format in ("csv", "pt"):
            with self.subTest(out_format=out_format):
                message = self.refused("from-sequences", "chain", "--fasta-file", fasta, "--output-folder",
                                       self.folder(), "--output-format", out_format)
                self.assertIn(f"{fasta}: 2 FASTA id(s) that cannot name a file <name>.{out_format} in the output "
                              f"folder: '../precious' (record 1) holds '/'; 'Q9/1-100' (record 2) holds '/'. Each "
                              f"FASTA id names its own file there: give those records a FASTA id that is not empty, "
                              f"without '/'", message)
                self.assertEqual(precious.read_text(), "keep")
        out = self.folder()
        self.run_ok("from-sequences", "chain", "--fasta-file", fasta, "--output-folder", out, "--output-format",
                    "parquet")
        self.assertEqual(_ids(out / "inference-0.parquet"), ["../precious", "Q9/1-100"])

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_ids_that_differ_in_case_on_this_mac(self):
        """abc and ABC were one file, which held the second record's embedding: refused in csv and pt; Parquet keeps
        both."""
        fasta = self.fasta(("abc", "MKV"), ("ABC", "MKVL"))
        message = self.refused("from-sequences", "residue", "--fasta-file", fasta, "--output-folder", self.folder(),
                               "--output-format", "pt")
        self.assertIn("1 file(s) of the output folder would each be written for more than one FASTA id (the output "
                      "folder does not tell upper and lower case apart): 'abc' (record 1) and 'ABC' (record 2) give "
                      "abc.pt. Each one would replace the one before, as the folder takes those names for one: write "
                      "them to a folder of a file system that tells upper and lower case apart or as parquet or json "
                      "(which keep each FASTA id as given), or give those records FASTA ids that differ otherwise.",
                      message)
        out = self.folder()
        self.run_ok("from-sequences", "residue", "--fasta-file", fasta, "--output-folder", out, "--output-format",
                    "parquet")
        self.assertEqual(_ids(out / "inference-0.parquet"), ["abc", "ABC"])

    def test_the_fasta_file_among_the_outputs(self):
        """--fasta-file OUT/x.csv with a record x: the first embedding replaced the FASTA file, and the next record
        failed. Refused; under another name in that folder it runs."""
        out = self.folder()
        fasta = self.fasta(("x", "MKV"), ("y", "MKVL"), name=out / "x.csv")
        message = self.refused("from-sequences", "chain", "--fasta-file", fasta, "--output-folder", out,
                               written=[out])
        self.assertIn(f"the FASTA file {fasta} by the output 'x'", message)
        renamed = fasta.rename(out / "queries.fasta")
        self.run_ok("from-sequences", "chain", "--fasta-file", renamed, "--output-folder", out)
        self.assertEqual(sorted(f.name for f in out.iterdir()), ["queries.fasta", "x.csv", "y.csv"])

    def test_the_output_name(self):
        """--output-name '../up' wrote up-0.parquet outside the folder: refused, from sequences and from structures."""
        fasta = self.fasta(("x", "MKV"))
        structures = self.folder("structures")
        os.symlink(_CIF, structures / "1acb.cif")
        for args in (("from-sequences", "chain", "--fasta-file", fasta),
                     ("from-sequences", "residue", "--fasta-file", fasta),
                     ("from-structures", "chain", "--src-folder", structures),
                     ("from-structures", "assembly", "--src-folder", structures)):
            with self.subTest(command=args[:2]):
                message = self.refused(*args, "--output-folder", self.folder(), "--output-format", "parquet",
                                       "--output-name", "../up", written=[self.tmp])
                self.assertTrue(message.startswith("The output name '../up' holds '/'"), message)
        out = self.folder()
        self.run_ok("from-sequences", "chain", "--fasta-file", fasta, "--output-folder", out, "--output-format",
                    "parquet", "--output-name", "up")
        self.assertEqual([f.name for f in out.iterdir()], ["up-0.parquet"])


def _chains_a_and_a(path: Path) -> Path:
    """1acb's two protein chains (author fields E and I) renamed A and a, written as PDB."""
    atoms = get_structure(CIFFile.read(str(_CIF)), model=1, use_author_fields=True)
    atoms = atoms[~atoms.hetero]
    atoms.chain_id[atoms.chain_id == "E"] = "A"
    atoms.chain_id[atoms.chain_id == "I"] = "a"
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(str(path))
    return path


class WriterBackstopTest(_Case):
    """D4: a writer refuses a second write of one file in a run, naming the file; the first file is kept, the second
    item not written. Per process: not across ranks."""

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_chains_a_and_a_on_this_mac(self):
        """from-structures chain on a structure with chains A and a: cc.A.csv held chain a's data, and chain A was gone
        (sweep). Now the run stops at the second; Parquet keeps both."""
        structures = self.folder("structures")
        _chains_a_and_a(structures / "cc.pdb")
        for command in ("residue", "chain"):
            for out_format in ("csv", "pt"):
                with self.subTest(command=command, out_format=out_format):
                    out = self.folder()
                    result = self.invoke("from-structures", command, "--src-folder", structures, "--structure-format",
                                         "pdb", "--output-folder", out, "--output-format", out_format)
                    self.assertEqual(result.exit_code, 1, result.output)
                    self.assertIsInstance(result.exception, RefusalError)
                    written = [f.name for f in out.iterdir()]
                    self.assertEqual(len(written), 1, written)
                    self.assertRegex(_flat(str(result.exception)),
                                     f"^{re.escape(str(out))}/cc\\.(A|a)\\.{out_format} would be written twice by this "
                                     f"run, for 'cc\\.(A|a)' and then for 'cc\\.(A|a)' \\(the folder does not tell "
                                     f"upper and lower case or Unicode spellings \\(NFC and NFD\\) apart\\): the first "
                                     f"is kept, 'cc\\.(A|a)' is not written, and the run stops\\.")
        out = self.folder()
        self.run_ok("from-structures", "chain", "--src-folder", structures, "--structure-format", "pdb",
                    "--output-folder", out, "--output-format", "parquet")
        self.assertCountEqual(_ids(out / "inference-0.parquet"), ["cc.A", "cc.a"])

    def test_the_writers(self):
        """The same name twice is refused on any folder; A and a only where the folder takes them for one name (the
        probe's result faked: a case-sensitive folder keeps both)."""
        trainer = SimpleNamespace(global_rank=0)
        vectors = [torch.ones(3), torch.zeros(3)]
        for writer, postfix in ((CsvBatchWriter, "csv"), (TensorBatchWriter, "pt")):
            with self.subTest(writer=writer.__name__):
                out = self.folder()
                files = writer(str(out))
                with self.assertRaises(RefusalError) as refused:
                    files.write_on_batch_end(trainer, None, (vectors, ["x", "x"]), None, None, 0, 0)
                self.assertTrue(str(refused.exception).startswith(f"{out}/x.{postfix} would be written twice by this "
                                                                  f"run, for 'x' and then for 'x': the first is kept"))
                self.assertEqual([f.name for f in out.iterdir()], [f"x.{postfix}"])
                with self.assertRaises(RefusalError):
                    files._write_embedding(vectors[0], "../x")
                from foldmatch.utils.output_files import EXACT, Folding
                for folding in (EXACT, Folding(case=True)):
                    out = self.folder()
                    with mock.patch("foldmatch.writer.batch_writer.probe_folding", return_value=folding):
                        files = writer(str(out))
                        files._write_embedding(vectors[0], "A")
                        if folding.case:
                            with self.assertRaises(RefusalError):
                                files._write_embedding(vectors[1], "a")
                        else:
                            # Both written (on this Mac's case-insensitive folder, the file system makes them one)
                            files._write_embedding(vectors[1], "a")
                            self.assertEqual(len(list(out.iterdir())), 1 if _CASE_INSENSITIVE else 2)

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_the_writers_fold_case_fully_on_this_mac(self):
        """straße and strasse, ﬀ1 and ff1 are one file on APFS (the second replaced the first): the writers refuse the
        second; a folder that folds case simply (the probe's result faked) keeps both names."""
        trainer = SimpleNamespace(global_rank=0)
        vectors = [torch.ones(3), torch.zeros(3)]
        from foldmatch.utils.output_files import Folding
        for writer, postfix in ((CsvBatchWriter, "csv"), (TensorBatchWriter, "pt")):
            for first, second in (("straße", "strasse"), ("ﬀ1", "ff1")):
                with self.subTest(writer=writer.__name__, first=first):
                    out = self.folder()
                    with self.assertRaises(RefusalError) as refused:
                        writer(str(out)).write_on_batch_end(trainer, None, (vectors, [first, second]), None, None, 0,
                                                            0)
                    self.assertTrue(str(refused.exception).startswith(
                        f"{out}/{first}.{postfix} would be written twice by this run, for {first!r} and then for "
                        f"{second!r} (the folder does not tell upper and lower case"), str(refused.exception))
                    self.assertEqual([f.name for f in out.iterdir()], [f"{first}.{postfix}"])
                    with mock.patch("foldmatch.writer.batch_writer.probe_folding",
                                    return_value=Folding(case=True, unicode=True)):
                        files = writer(str(self.folder()))
                        files._write_embedding(vectors[0], first)
                        files._write_embedding(vectors[1], second)

    def test_runs_into_one_folder_still_run(self):
        """Not a conflict within a run: residue then chain embeddings of the same structures into one csv folder (the
        chain files replace the residue files, as before: an earlier run's files are not checked), and a rerun. A
        structure list given twice the same structure, which the writer would write twice, is refused by its reader."""
        structures = self.folder("structures")
        os.symlink(_CIF, structures / "1acb.cif")
        out = self.folder()
        for command in ("residue", "chain", "chain"):
            self.run_ok("from-structures", command, "--src-folder", structures, "--output-folder", out)
        self.assertEqual(sorted(f.name for f in out.iterdir()), ["1acb.A.csv", "1acb.B.csv"])
        self.refusing = True
        with self.assertRaises(RefusalError):
            chain_complete_inference.predict(src_stream=[("1acb", str(_CIF), "x"), ("2acb", str(_CIF), "x")],
                                             src_location="stream", out_folder=self.folder(), accelerator="cpu",
                                             devices=1, return_predictions=False)


if __name__ == "__main__":
    unittest.main()
