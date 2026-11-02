"""P9.A5: an assembly list row of more than 4 fields is refused.

The assembly lists of inference assembly-embedding and complete-embedding (--src-file, --src-assembly-file) and of the
library's ResidueAssemblyEmbeddingFromTensorFile and ResidueAssemblyEmbeddingFromParquet are CSV without a header, of
structure name, structure file or URL, assembly ID and embedding name, read by read_assembly_list. A row of more than 4
fields is refused (RefusalError), from a file and from a stream, naming the file or the stream and the row, as
read_chain_list refuses one, and so by both commands before any model is built: complete-embedding now reads an assembly
list in a regular file before its residue step (a pipe, /dev/stdin or /dev/fd/N, which can be read once, only in its
assembly step, as before). Before, pandas read the leading fields of a first such row as the index, which shifted the
columns of every row (a trailing comma was enough), stopped at a later one with a ParserError that named no file, and
refused a stream row of 5 elements with its ValueError; complete-embedding read its assembly list only in its assembly
step, after its residue and chain steps had written their files. Lists of at most 4 fields per row are read as before
(pandas' default NA values kept: an empty field or NA is NaN).

ESM3 and the aggregator are never loaded: the commands run with stub models (_esm3, _aggregate, as in
test_inference_follow_ups.py) or with model getters that fail if called."""
import errno
import io
import logging
import os
import re
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import typer
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from typer.testing import CliRunner

from foldmatch.cli import inference
from foldmatch.dataset.residue_assembly_embedding_from_parquet import ResidueAssemblyEmbeddingFromParquet
from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.errors import RefusalError
# Imported before any test patches their model getters
from foldmatch.inference import assembly_inferece, chain_inference, esm_inference, structure_inference  # noqa: F401
from foldmatch.types.api_types import SrcLocation, StructureFormat
from foldmatch.utils.structure_parser import filter_residues, get_structure_from_src, protein_chain_iter, \
    rename_atom_attr

_RESOURCES = Path(__file__).parent / "resources"
_CIF, _CIF2 = _RESOURCES / "pdb" / "1acb.cif", _RESOURCES / "pdb" / "2uzi.cif"
_EMBEDDINGS = _RESOURCES / "embeddings"
_CPU = ("--accelerator", "cpu", "--devices", "1")
_COLUMNS = ["stream_name", "stream", "assembly_id", "item_name"]

# What a refusal of an assembly list row of more than 4 fields says after what it found
_RULE = ("An assembly list row holds at most 4 fields (structure name, structure file or URL, assembly ID, embedding "
         "name): ")
_REMOVE_THE_EXTRA_FIELDS = "remove the extra fields (a comma after the embedding name makes a 5th)."

# (case, assembly list, what the refusal found, after the file's path: a regular expression)
_LISTS = (
    ("first row, a trailing comma", f"1acb,{_CIF},1,1acb-1,\n2uzi,{_CIF2},1,2uzi-1\n",
     re.escape("its first row has 5 fields")),
    ("first row, a 5th field of text", f"1acb,{_CIF},1,1acb-1,extra\n2uzi,{_CIF2},1,2uzi-1\n",
     re.escape("its first row has 5 fields")),
    ("first row of 6 fields", f"1acb,{_CIF},1,1acb-1,,extra\n2uzi,{_CIF2},1,2uzi-1\n",
     re.escape("its first row has 6 fields")),
    ("a later row of 5 fields", f"1acb,{_CIF},1,1acb-1\n2uzi,{_CIF2},1,2uzi-1,\n",
     r"a row has more than 4 fields \(pandas: [^()]*Expected 4 fields in line 2, saw 5\)"),
    ("a first row of 5, a later row of 6", f"1acb,{_CIF},1,1acb-1,\n2uzi,{_CIF2},1,2uzi-1,,\n",
     r"a row has more than 4 fields \(pandas: [^()]*Expected 5 fields in line 2, saw 6\)"),
    # pandas read this list's first column as the index, so it gave the files of the list without it
    ("an index column", f"0,1acb,{_CIF},1,1acb-1\n1,2uzi,{_CIF2},1,2uzi-1\n",
     re.escape("its first row has 5 fields")),
)
_READERS = ("read_assembly_list", "ResidueAssemblyEmbeddingFromTensorFile", "ResidueAssemblyEmbeddingFromParquet")
_STREAM_RULE = ("An assembly list row holds at most 4 fields (structure name, structure file or URL, assembly ID, "
                "embedding name): remove the extra elements.")


def _invoke(*args):
    """inference in-process (exit codes as in a shell, the exception kept), the logging setup restored after."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        return CliRunner().invoke(inference.app, [str(a) for a in args])
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def _esm3(model, protein):
    """Stands in for ESM3 in the inference modules: L + 2 rows (BOS and EOS, zeros), each residue's row its letter, its
    position and the chain's length (as in test_inference_follow_ups.py)."""
    rows = [[float(ord(aa)), float(k), float(len(protein))] for k, aa in enumerate(protein.sequence)]
    return torch.tensor([[0.0] * 3] + rows + [[0.0] * 3])


def _aggregate(x, x_mask):
    """Stands in for the aggregator: the sum of each item's rows."""
    return x.masked_fill(x_mask.unsqueeze(-1), 0.0).sum(dim=1)


def _csv(residues) -> str:
    """The embedding of these residue embeddings as CsvBatchWriter writes it: _aggregate's vector, one value a line."""
    return pd.DataFrame(residues.sum(dim=0).numpy()).to_csv(index=False, header=False)


def _protein(chain_id: str) -> ESMProtein:
    """Archive 1acb's protein chain, read as every path reads it."""
    atoms = next(protein_chain_iter(get_structure_from_src(str(_CIF), StructureFormat.mmcif), chain_id=chain_id))
    return ESMProtein.from_protein_chain(ProteinChain.from_atomarray(rename_atom_attr(filter_residues(atoms))))


def _read_assembly_list(*args):
    """foldmatch.dataset.utils.read_assembly_list, imported when called."""
    from foldmatch.dataset.utils import read_assembly_list
    return read_assembly_list(*args)


def _expected(rows: list, index: list) -> pd.DataFrame:
    """A frame of the assembly list columns, as strings, None read as NaN."""
    return pd.DataFrame(rows, index=pd.Index(index, dtype="int64"), columns=_COLUMNS, dtype=str)


def _write_once(fifo: Path, text: str, stop: threading.Event):
    """The writer of the named pipe fifo, until stop is set: its first reader reads text, a later one an end of file at
    once, as from a pipe whose writer has exited (rather than a wait for another writer)."""
    while not stop.is_set():
        try:
            # Opened only once a reader has opened it, so that stop always ends the loop
            fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            if e.errno != errno.ENXIO:
                raise
            stop.wait(0.01)
            continue
        os.set_blocking(fd, True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        text = ""
        stop.wait(0.01)


class _AssemblyListCase(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # Residue embeddings in Parquet, for ResidueAssemblyEmbeddingFromParquet, which reads them after the list
        self.parquet = self.tmp / "residues.parquet"
        pq.write_table(pa.table({"id": ["x.A"], "embedding": [[[0.0, 1.0, 2.0]]]},
                                schema=pa.schema([("id", pa.string()),
                                                  ("embedding", pa.list_(pa.list_(pa.float32(), 3)))])),
                       self.parquet)

    def assembly_list(self, case: str, text: str) -> Path:
        path = self.tmp / f"{case.replace(',', '').replace(' ', '_')}.csv"
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def readers(self, src, src_location: SrcLocation = SrcLocation.file) -> dict:
        """read_assembly_list and both readers (_READERS), each a callable that reads src, by name."""
        return {
            "read_assembly_list": lambda: _read_assembly_list(src, src_location),
            "ResidueAssemblyEmbeddingFromTensorFile": lambda: ResidueAssemblyEmbeddingFromTensorFile(
                src_stream=src, res_embedding_location=self.tmp, src_location=src_location),
            "ResidueAssemblyEmbeddingFromParquet": lambda: ResidueAssemblyEmbeddingFromParquet(
                src_stream=src, parquet_path=self.parquet, src_location=src_location),
        }

    def assert_refused(self, refusal: BaseException, source: str, found: str):
        """The refusal names the source and what it found, says that a row holds at most 4 fields, and what to do."""
        self.assertIsInstance(refusal, RefusalError)
        self.assertRegex(str(refusal), f"^{re.escape(source)}: {found}\\. "
                                       f"{re.escape(_RULE + _REMOVE_THE_EXTRA_FIELDS)}$")


class LongerRowTest(_AssemblyListCase):
    """D1: read_assembly_list and both readers refuse a row of more than 4 fields, from a path, an open file (named by
    its path) and a text buffer ("The assembly list"), and from a stream; any other ParserError too."""

    def test_a_file_row_of_more_than_4_fields(self):
        """A first row with a trailing comma, with a 5th field of text or with 6 fields, every row with an index column
        (pandas took the first one or two fields as the index), and a later row of 5, or of 6 after a first row of 5
        (pandas' ParserError, which names the line, quoted and kept as the cause)."""
        for case, text, found in _LISTS:
            path = self.assembly_list(case, text)
            for source, make, named in (("path", lambda: path, str(path)),
                                        ("open file", lambda: open(path, encoding="utf-8"), str(path)),
                                        ("text buffer", lambda: io.StringIO(text), "The assembly list")):
                for reader in _READERS:
                    with self.subTest(case=case, source=source, reader=reader):
                        src = make()
                        try:
                            with self.assertRaises(RefusalError) as refused:
                                self.readers(src)[reader]()
                        finally:
                            if hasattr(src, "close"):
                                src.close()
                        self.assert_refused(refused.exception, named, found)
                        if "later row" in case:
                            self.assertIsInstance(refused.exception.__cause__, pd.errors.ParserError)

    def test_a_stream_row_of_more_than_4_elements(self):
        """A stream: how many rows have more than 4 elements, and up to 5 of them, counted from 1 (before, pandas'
        ValueError named no row)."""
        for rows, found in (
                ([("1acb", str(_CIF), "1", "1acb-1"), ("2uzi", str(_CIF2), "1", "2uzi-1", "")],
                 "1 row(s) of more than 4 elements (row(s) 2"),
                ([("1acb", str(_CIF), "1", "1acb-1"), ("2uzi", str(_CIF2), "1", "2uzi-1", "", "x")],
                 "1 row(s) of more than 4 elements (row(s) 2"),
                ([("w", "f", "1")] + [("x", "f", "1", "x-1", "z")] * 5 + [("y",) * 6] * 2,
                 "7 row(s) of more than 4 elements (row(s) 2, 3, 4, 5, 6 and 2 more")):
            for reader, read in self.readers(rows, SrcLocation.stream).items():
                with self.subTest(rows=len(rows), longest=max(map(len, rows)), reader=reader):
                    with self.assertRaises(RefusalError) as refused:
                        read()
                    self.assertEqual(str(refused.exception),
                                     f"The assembly list stream: {found}, counted from 1). {_STREAM_RULE}")

    def test_another_parser_error_is_not_claimed_as_extra_fields(self):
        """Another pandas ParserError (an unclosed quote) is refused too, naming the file and quoting pandas, without
        saying that a row has more than 4 fields (before, pandas' bare ParserError)."""
        path = self.assembly_list("unclosed quote", f'1acb,"{_CIF},1,1acb-1\n2uzi,{_CIF2},1,2uzi-1\n')
        for reader, read in self.readers(path).items():
            with self.subTest(reader=reader):
                with self.assertRaises(RefusalError) as refused:
                    read()
                self.assertIsInstance(refused.exception.__cause__, pd.errors.ParserError)
                self.assertRegex(str(refused.exception),
                                 f"^{re.escape(str(path))}: pandas could not read it as CSV \\([^()]*EOF inside "
                                 f"string[^()]*\\)\\. " + re.escape(
                                     "An assembly list is CSV without a header, of at most 4 fields per row (structure "
                                     "name, structure file or URL, assembly ID, embedding name): correct it where "
                                     "pandas' message points.") + "$")
                self.assertNotIn("more than 4", str(refused.exception))


class CommandTest(_AssemblyListCase):
    """D1 and D3: inference assembly-embedding and complete-embedding refuse such a list (exit status 1, the
    RefusalError) before any model is built (the model getters fail if called) and write nothing; complete-embedding
    reads an assembly list in a regular file before its residue step, and so refuses a list that gives one structure
    name to two files before any model too."""

    def setUp(self):
        super().setUp()
        self.never = mock.Mock(side_effect=AssertionError("a model was built"))
        for getter in ("esm_inference.get_residue_model", "structure_inference.get_residue_model",
                       "structure_inference.get_aggregator_model", "chain_inference.get_aggregator_model"):
            self.enterContext(mock.patch(f"foldmatch.inference.{getter}", self.never))
        self.chains = self.assembly_list("chains", f"1acb,{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B\n")

    def folder(self, name: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.tmp))

    def complete_embedding(self, assemblies: Path):
        """inference complete-embedding on the conventional chain list and these assemblies: its result, and the files
        written in its residue, chain and assembly folders."""
        folders = [self.folder(name) for name in ("residues", "chains", "assemblies")]
        result = _invoke("complete-embedding", "--src-chain-file", self.chains, "--src-assembly-file", assemblies,
                         "--output-res-path", folders[0], "--output-chain-path", folders[1],
                         "--output-assembly-path", folders[2], *_CPU)
        return result, [f for folder in folders for f in folder.iterdir()]

    def test_assembly_embedding_refuses_it_before_any_model(self):
        """Before, the reader built shifted columns (the aggregator was then built, and the first item stopped on the
        assembly ID read as the structure file; 6 fields were refused as a shared name, before any model), or stopped on
        pandas' ParserError; an index column worked."""
        for case, text, found in _LISTS:
            with self.subTest(case=case):
                path = self.assembly_list(case, text)
                out = self.folder("out")
                result = _invoke("assembly-embedding", "--src-file", path, "--res-embedding-location", _EMBEDDINGS,
                                 "--output-folder", out, *_CPU)
                self.assertEqual(result.exit_code, 1, result.output)
                self.assert_refused(result.exception, str(path.resolve()), found)
                self.assertEqual(list(out.iterdir()), [])
        self.never.assert_not_called()

    def test_complete_embedding_refuses_it_before_any_model(self):
        """Before, the residue and chain steps ran and wrote their files, and the assembly step then failed or, for an
        index column, worked (on a chain list of both entries: this test's has 1acb only, so it stopped at 2uzi)."""
        for case, text, found in _LISTS:
            with self.subTest(case=case):
                path = self.assembly_list(case, text)
                result, written = self.complete_embedding(path)
                self.assertEqual(result.exit_code, 1, result.output)
                self.assert_refused(result.exception, str(path.resolve()), found)
                self.assertEqual(written, [])
        self.never.assert_not_called()

    def test_complete_embedding_refuses_a_shared_name_before_any_model(self):
        """A list that gives the name 1acb to two files: refused (refuse_shared_names) before the residue step, where
        before it was refused by the assembly step, after the residue and chain steps."""
        copy = Path(shutil.copyfile(_CIF, self.tmp / "1acb-copy.cif"))
        path = self.assembly_list("shared name", f"1acb,{_CIF},1,1acb-1\n1acb,{copy},1,1acb-x\n")
        result, written = self.complete_embedding(path)
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIsInstance(result.exception, RefusalError)
        self.assertIn(f"'1acb' names {min(str(_CIF), str(copy))} and {max(str(_CIF), str(copy))}",
                      str(result.exception))
        self.assertEqual(written, [])
        self.never.assert_not_called()

    def test_the_help_describes_the_columns(self):
        """D4: the help of assembly-embedding --src-file and complete-embedding --src-assembly-file."""
        commands = typer.main.get_command(inference.app).commands
        for command, option in (("assembly-embedding", "src_file"), ("complete-embedding", "src_assembly_file")):
            with self.subTest(command=command):
                self.assertEqual(next(p.help for p in commands[command].params if p.name == option),
                                 "CSV file of 4 columns, no header: Structure Name | Structure File Path or URL | "
                                 "Assembly Id | Embedding Name.")


class AsBeforeTest(_AssemblyListCase):
    """Lists of at most 4 fields per row read as before: the frames recorded from the readers of 9763f44 (pandas' call
    unchanged, then sorted by structure name), and complete-embedding's files with stub models, its assembly list in
    a regular file, a pipe or /dev/fd/N."""

    # (case, assembly list, the rows of the readers' frame, its index)
    LISTS = (
        ("rows of 2, 3 and 4 fields", "c,f.cif,3,c-3\nb,f.cif\na,g.cif,2\n",
         [["a", "g.cif", "2", None], ["b", "f.cif", None, None], ["c", "f.cif", "3", "c-3"]], [2, 1, 0]),
        ("an empty 4th field and NA", "b,f.cif,1,\nNA,g.cif,NA,a-1\na,NA,2,NA\n",
         [["a", None, "2", None], ["b", "f.cif", "1", None], [None, "g.cif", None, "a-1"]], [2, 0, 1]),
        ("a byte-order mark and CRLF", "﻿b,f.cif,1,b-1\r\na,g.cif,2,a-2\r\n",
         [["a", "g.cif", "2", "a-2"], ["b", "f.cif", "1", "b-1"]], [1, 0]),
        ("quoted fields", 'b,"f,1.cif",1,"n,1"\n"a\nz",g.cif,"2",\n',
         [["a\nz", "g.cif", "2", None], ["b", "f,1.cif", "1", "n,1"]], [1, 0]),
        ("blank lines", "\nb,f.cif,1,b-1\n\n  \n\t\na,g.cif,2,a-2\n\n",
         [["a", "g.cif", "2", "a-2"], ["b", "f.cif", "1", "b-1"]], [1, 0]),
        ("an empty file", "", [], []),
    )
    STREAM = [("b", "f.cif", "1"), ("a", "g.cif", "2", "a-2"), ("c", None, "3", None)]

    def test_the_readers_read_them_as_before(self):
        """Both readers' frames, from a path, an open file and a text buffer, and from a stream of 3 and 4 elements."""
        for case, text, rows, index in self.LISTS:
            path = self.assembly_list(case, text)
            for source, make in (("path", lambda: path), ("open file", lambda: open(path, encoding="utf-8")),
                                 ("text buffer", lambda: io.StringIO(text))):
                for reader in _READERS[1:]:
                    with self.subTest(case=case, source=source, reader=reader):
                        src = make()
                        try:
                            data = self.readers(src)[reader]().data
                        finally:
                            if hasattr(src, "close"):
                                src.close()
                        pd.testing.assert_frame_equal(data, _expected(rows, index), check_exact=True)
        for reader in _READERS[1:]:
            with self.subTest(stream=reader):
                data = self.readers(self.STREAM, SrcLocation.stream)[reader]().data
                pd.testing.assert_frame_equal(data, _expected([["a", "g.cif", "2", "a-2"], ["b", "f.cif", "1", None],
                                                               ["c", None, "3", None]], [1, 0, 2]), check_exact=True)

    def test_read_assembly_list_reads_them_as_pandas_did(self):
        """read_assembly_list: the frame of the readers' pandas call on 9763f44, in input order, its index pandas'
        (0, 1, ...); a stream as pandas reads its rows, inside a list, a tuple or a generator."""
        for case, text, rows, index in self.LISTS:
            path = self.assembly_list(case, text)
            unsorted = [row for _, row in sorted(zip(index, rows))]
            for source, make in (("path", lambda: path), ("text buffer", lambda: io.StringIO(text))):
                with self.subTest(case=case, source=source):
                    data = _read_assembly_list(make())
                    pd.testing.assert_frame_equal(data, _expected(unsorted, sorted(index)), check_exact=True)
                    self.assertIsInstance(data.index, pd.RangeIndex)
        for container in (list, tuple, iter):
            with self.subTest(stream=container.__name__):
                data = _read_assembly_list(container(self.STREAM), SrcLocation.stream)
                pd.testing.assert_frame_equal(data, _expected([["b", "f.cif", "1", None], ["a", "g.cif", "2", "a-2"],
                                                               ["c", None, "3", None]], [0, 1, 2]), check_exact=True)

    def assert_complete_embedding(self, assemblies: Path):
        """complete-embedding on a chain list of archive 1acb's chains A and B and these assemblies (1acb's assembly 1,
        named 1acb-1), stub models: the residue files, chain embeddings and assembly embedding that the stubs give
        through the unchanged readers and writers."""
        with mock.patch("foldmatch.inference.esm_inference.get_residue_model", return_value="ESM3"), \
                mock.patch("foldmatch.inference.chain_inference.get_aggregator_model", return_value=_aggregate), \
                mock.patch("foldmatch.modules.esm_module.per_residue_embedding", _esm3):
            residues_of = {c: _esm3(None, _protein(c)) for c in "AB"}
            chains = self.assembly_list("chains", f"1acb,{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B\n")
            folders = [Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.tmp))
                       for name in ("residues", "chains", "assemblies")]
            result = _invoke("complete-embedding", "--src-chain-file", chains, "--src-assembly-file", assemblies,
                             "--output-res-path", folders[0], "--output-chain-path", folders[1],
                             "--output-assembly-path", folders[2], *_CPU)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        written = {f.name: torch.load(f, weights_only=True) for f in folders[0].iterdir()}
        self.assertEqual(sorted(written), ["1acb.A.pt", "1acb.B.pt"])
        for c in "AB":
            self.assertTrue(torch.equal(written[f"1acb.{c}.pt"], residues_of[c]), c)
        self.assertEqual({f.name: f.read_text() for f in folders[1].iterdir()},
                         {f"1acb.{c}.csv": _csv(residues_of[c]) for c in "AB"})
        self.assertEqual({f.name: f.read_text() for f in folders[2].iterdir()},
                         {"1acb-1.csv": _csv(torch.cat([residues_of["A"], residues_of["B"]]))})

    def test_complete_embedding_on_conventional_lists(self):
        """complete-embedding on a chain list and an assembly list of 4 fields per row, stub models: the residue files,
        chain embeddings and assembly embedding that the stubs give through the unchanged readers and writers, as
        before."""
        self.assert_complete_embedding(self.assembly_list("assemblies", f"1acb,{_CIF},1,1acb-1\n"))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "a named pipe needs os.mkfifo")
    def test_complete_embedding_on_an_assembly_list_in_a_pipe(self):
        """D3': an assembly list given as a pipe (a named pipe here, or <(...) on macOS) can be read once:
        complete-embedding does not read it before its residue step, as it reads one in a regular file, so its assembly
        step reads it and writes the assembly embedding, as before (read before the residue step too, the assembly
        step would find it empty and write nothing, with exit status 0)."""
        fifo = self.tmp / "assemblies.fifo"
        os.mkfifo(fifo)
        stop = threading.Event()
        writer = threading.Thread(target=_write_once, args=(fifo, f"1acb,{_CIF},1,1acb-1\n", stop), daemon=True)
        writer.start()
        try:
            self.assert_complete_embedding(fifo)
        finally:
            stop.set()
            writer.join()

    @unittest.skipUnless(os.path.isdir("/dev/fd"), "/dev/fd/N needs /dev/fd")
    def test_complete_embedding_on_an_assembly_list_as_dev_fd(self):
        """D3'': on macOS, an assembly list in a regular file given as /dev/fd/N (or as /dev/stdin, which click makes
        /dev/fd/0) can be read once, as open shares the descriptor's offset: complete-embedding does not read a list
        under /dev before its residue step, so its assembly step reads it and writes the assembly embedding, as before
        (read before the residue step too, the assembly step would find it empty and write nothing, with exit status
        0)."""
        fd = os.open(self.assembly_list("assemblies", f"1acb,{_CIF},1,1acb-1\n"), os.O_RDONLY)
        try:
            self.assert_complete_embedding(Path(f"/dev/fd/{fd}"))
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
