"""P9.A6: a structure list row of more than 3 fields and a residue file list row of more than 2 are refused.

A structure list is CSV without a header, of structure name, structure file or URL and embedding name, read by
read_structure_list for EsmProtFromStructure, EsmAssemblyFromStructure, ResidueEmbeddingFromStructure and
ResidueAssemblyDatasetFromStructure. A residue file list is CSV without a header, of residue embedding file and
embedding name, read by read_residue_file_list for ResidueEmbeddingFromTensorFile. These lists are given in a file
through the library only; the commands give the readers streams. A row of more fields is refused (RefusalError), from a
file and from a stream, naming the file or the stream and the row, as read_chain_list and read_assembly_list refuse one,
and so by every entry point before any model is built. Before, pandas read the leading fields of a first such row as the
index, which shifted the columns of every row, stopped at a later one with a ParserError that named no file, and refused
a stream row of one element too many with its ValueError; a list whose every row began with an index column worked.
Lists of at most that many fields per row are read as before, but for the residue file list's strings (D3): it was the
only list read without dtype=str, and without it pandas 3 reads an index column 0, 1, 2 as a RangeIndex, which no check
can tell from no index, and the names 007 and 1.50 as 7.0 and 1.5.

ESM3 and the aggregator are never loaded: the entry points run with stub models (_esm3, _aggregate, as in
test_assembly_lists.py) or with model getters that fail if called."""
import errno
import gzip
import io
import json
import logging
import os
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from typer.testing import CliRunner

from foldmatch.cli import embedding
from foldmatch.dataset import resdiue_assembly_embedding_from_structure, residue_embedding_from_structure
from foldmatch.dataset.esm_assembly_from_structure import EsmAssemblyFromStructure
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.resdiue_assembly_embedding_from_structure import ResidueAssemblyDatasetFromStructure
from foldmatch.dataset.residue_embedding_from_structure import ResidueEmbeddingFromStructure
from foldmatch.dataset.residue_embedding_from_tensor_file import ResidueEmbeddingFromTensorFile
from foldmatch.errors import RefusalError
# Imported before any test patches their model getters
from foldmatch.inference import assembly_complete_inference, assembly_inferece, chain_complete_inference, \
    chain_inference, esm_inference, structure_inference
from foldmatch.types.api_types import Accelerator, OutFormat, SrcAssemblyFrom, SrcEsmFrom, SrcLocation, \
    SrcProteinFrom, SrcTensorFrom, StructureFormat
from foldmatch.utils.data import collate_seq_embeddings, scan_structure_folder
from foldmatch.utils.structure_parser import filter_residues, get_assemblies, get_structure_from_src, \
    protein_chain_iter, rename_atom_attr

_RESOURCES = Path(__file__).parent / "resources"
_PDB = _RESOURCES / "pdb"
_CIF, _CIF2 = _PDB / "1acb.cif", _PDB / "2uzi.cif"
_EMBEDDINGS = _RESOURCES / "embeddings"
_CPU = {"accelerator": Accelerator.cpu, "devices": 1}
_STRUCTURE_COLUMNS = ["stream_name", "stream", "item_name"]
_RESIDUE_FILE_COLUMNS = ["file", "item_name"]
_STRUCTURE_LIST, _RESIDUE_FILE_LIST = "structure list", "residue file list"

# What a refusal of a row of too many fields says after what it found, per list
_RULE = {
    _STRUCTURE_LIST: ("A structure list row holds at most 3 fields (structure name, structure file or URL, embedding "
                      "name)"),
    _RESIDUE_FILE_LIST: "A residue file list row holds at most 2 fields (residue embedding file, embedding name)",
}
_REMOVE_THE_EXTRA_FIELDS = {
    _STRUCTURE_LIST: "remove the extra fields (a comma after the embedding name makes a 4th).",
    _RESIDUE_FILE_LIST: "remove the extra fields (a comma after the embedding name makes a 3rd).",
}


def _first_row(fields: int) -> str:
    return re.escape(f"its first row has {fields} fields")


def _later_row(columns: int, expected: int, saw: int) -> str:
    return rf"a row has more than {columns} fields \(pandas: [^()]*Expected {expected} fields in line 2, saw {saw}\)"


# (case, list, what the refusal found, after the file's path: a regular expression)
_LISTS = {
    _STRUCTURE_LIST: (
        ("first row, a trailing comma", "a,f.cif,a-n,\nb,g.cif,b-n\n", _first_row(4)),
        ("first row, a 4th field of text", "a,f.cif,a-n,extra\nb,g.cif,b-n\n", _first_row(4)),
        ("first row of 5 fields", "a,f.cif,a-n,,extra\nb,g.cif,b-n\n", _first_row(5)),
        ("an index column", "0,a,f.cif,a-n\n1,b,g.cif,b-n\n2,c,h.cif,c-n\n", _first_row(4)),
        ("two index columns", "0,0,a,f.cif,a-n\n0,1,b,g.cif,b-n\n", _first_row(5)),
        ("a later row of 4 fields", "a,f.cif,a-n\nb,g.cif,b-n,\n", _later_row(3, 3, 4)),
        ("a first row of 4, a later row of 5", "a,f.cif,a-n,\nb,g.cif,b-n,,\n", _later_row(3, 4, 5)),
        ("a chain list", "a,f.cif,A,a.A\nb,g.cif,B,b.B\n", _first_row(4)),
    ),
    _RESIDUE_FILE_LIST: (
        ("first row, a trailing comma", "a.pt,a-n,\nb.pt,b-n\n", _first_row(3)),
        ("first row, a 3rd field of text", "a.pt,a-n,extra\nb.pt,b-n\n", _first_row(3)),
        ("first row of 4 fields", "a.pt,a-n,,extra\nb.pt,b-n\n", _first_row(4)),
        # Without dtype=str, pandas 3 reads these two as a RangeIndex, as it reads no index at all: the trap of D3
        ("an index column", "0,a.pt,a-n\n1,b.pt,b-n\n2,c.pt,c-n\n", _first_row(3)),
        ("an index column of other integers", "5,a.pt,a-n\n7,b.pt,b-n\n", _first_row(3)),
        ("two index columns", "0,0,a.pt,a-n\n0,1,b.pt,b-n\n", _first_row(4)),
        ("a later row of 3 fields", "a.pt,a-n\nb.pt,b-n,\n", _later_row(2, 2, 3)),
        ("a first row of 3, a later row of 4", "a.pt,a-n,\nb.pt,b-n,,\n", _later_row(2, 3, 4)),
        ("a structure list", "a,f.cif,a-n\nb,g.cif,b-n\n", _first_row(3)),
        ("a chain list", "a,f.cif,A,a.A\nb,g.cif,B,b.B\n", _first_row(4)),
    ),
}
_READERS = {
    _STRUCTURE_LIST: ("read_structure_list", "read_structure_list keep_default_na=False", "EsmProtFromStructure",
                      "EsmAssemblyFromStructure", "ResidueEmbeddingFromStructure",
                      "ResidueAssemblyDatasetFromStructure"),
    _RESIDUE_FILE_LIST: ("read_residue_file_list", "ResidueEmbeddingFromTensorFile"),
}
# A row of one element too many, and a list whose rows are longer still
_STREAMS = {
    _STRUCTURE_LIST: (
        ([("a", "f.cif", "a-n"), ("b", "g.cif", "b-n", "")], "1 row(s) of more than 3 elements (row(s) 2"),
        ([("a", "f.cif", "a-n"), ("b", "g.cif", "b-n", "", "x")], "1 row(s) of more than 3 elements (row(s) 2"),
        ([("w", "f.cif", "w-n")] + [("x", "g.cif", "x-n", "z")] * 5 + [("y",) * 5] * 2,
         "7 row(s) of more than 3 elements (row(s) 2, 3, 4, 5, 6 and 2 more"),
    ),
    _RESIDUE_FILE_LIST: (
        ([("a.pt", "a-n"), ("b.pt", "b-n", "")], "1 row(s) of more than 2 elements (row(s) 2"),
        ([("a.pt", "a-n"), ("b.pt", "b-n", "", "x")], "1 row(s) of more than 2 elements (row(s) 2"),
        ([("w.pt", "w-n")] + [("x.pt", "x-n", "z")] * 5 + [("y",) * 4] * 2,
         "7 row(s) of more than 2 elements (row(s) 2, 3, 4, 5, 6 and 2 more"),
    ),
}
_UNCLOSED_QUOTE = {
    _STRUCTURE_LIST: 'a,"f.cif,a-n\nb,g.cif,b-n\n',
    _RESIDUE_FILE_LIST: '"a.pt,a-n\nb.pt,b-n\n',
}
_CSV_RULE = {
    _STRUCTURE_LIST: ("A structure list is CSV without a header, of at most 3 fields per row (structure name, "
                      "structure file or URL, embedding name): correct it where pandas' message points."),
    _RESIDUE_FILE_LIST: ("A residue file list is CSV without a header, of at most 2 fields per row (residue embedding "
                         "file, embedding name): correct it where pandas' message points."),
}


def _read_structure_list(*args, **kwargs):
    """foldmatch.dataset.utils.read_structure_list, imported when called."""
    from foldmatch.dataset.utils import read_structure_list
    return read_structure_list(*args, **kwargs)


def _read_residue_file_list(*args):
    """foldmatch.dataset.utils.read_residue_file_list, imported when called."""
    from foldmatch.dataset.utils import read_residue_file_list
    return read_residue_file_list(*args)


def _esm3(model, protein):
    """Stands in for ESM3 in the inference modules: L + 2 rows (BOS and EOS, zeros), each residue's row its letter, its
    position and the chain's length (as in test_assembly_lists.py)."""
    rows = [[float(ord(aa)), float(k), float(len(protein))] for k, aa in enumerate(protein.sequence)]
    return torch.tensor([[0.0] * 3] + rows + [[0.0] * 3])


def _aggregate(x, x_mask):
    """Stands in for the aggregator: the sum of each item's rows."""
    return x.masked_fill(x_mask.unsqueeze(-1), 0.0).sum(dim=1)


def _chain_embedding(residues) -> np.ndarray:
    """_aggregate's vector for these residue embeddings, through the padding of a batch, as the modules give it."""
    return _aggregate(*collate_seq_embeddings([residues]))[0].numpy()


def _csv(values) -> str:
    """These values as CsvBatchWriter writes them."""
    return pd.DataFrame(values).to_csv(index=False, header=False)


def _proteins(cif: Path, assembly_id=None) -> dict:
    """The protein chains of a structure, or of one of its assemblies, by chain ID: read as every path reads them."""
    structure = get_structure_from_src(str(cif), StructureFormat.mmcif, assembly_id=assembly_id)
    return {atoms.chain_id[0]: ESMProtein.from_protein_chain(
        ProteinChain.from_atomarray(rename_atom_attr(filter_residues(atoms))))
        for atoms in protein_chain_iter(structure)}


def _expected(rows: list, index: list, columns: list) -> pd.DataFrame:
    """A frame of these columns, as strings, None read as NaN."""
    return pd.DataFrame(rows, index=pd.Index(index, dtype="int64"), columns=columns, dtype=str)


def _unsorted(rows: list, index: list) -> list:
    """The rows of a sorted frame, in input order."""
    return [row for _, row in sorted(zip(index, rows), key=lambda item: item[0])]


def _write_once(fifo: Path, text: str, stop: threading.Event):
    """The writer of the named pipe fifo, until stop is set: its first reader reads text, a later one an end of file at
    once, as from a pipe whose writer has exited (rather than a wait for another writer). As in
    test_assembly_lists.py."""
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


class _Frame(Exception):
    """Carries the frame that a reader gave refuse_shared_names: ResidueEmbeddingFromStructure and
    ResidueAssemblyDatasetFromStructure keep their list's own in no attribute, and read the structures right after."""

    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data


def _raise_frame(data, *_):
    raise _Frame(data)


class _ListCase(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def list_file(self, case: str, text: str) -> Path:
        path = self.tmp / f"{case.replace(',', '').replace(' ', '_')}.csv"
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def sources(self, kind: str, path: Path, text: str) -> tuple:
        """(source, what opens it, how a refusal names it): a path, an open file (named by its path), a text buffer."""
        return (("path", lambda: path, str(path)),
                ("open file", lambda: open(path, encoding="utf-8", newline=""), str(path)),
                ("text buffer", lambda: io.StringIO(text), f"The {kind}"))

    def readers(self, src, src_location: SrcLocation = SrcLocation.file) -> dict:
        """The read functions and the 5 readers (_READERS), each a callable that reads src, by name."""
        return {
            "read_structure_list": lambda: _read_structure_list(src, src_location),
            "read_structure_list keep_default_na=False": lambda: _read_structure_list(src, src_location,
                                                                                      keep_default_na=False),
            "EsmProtFromStructure": lambda: EsmProtFromStructure(src_stream=src, src_location=src_location),
            "EsmAssemblyFromStructure": lambda: EsmAssemblyFromStructure(src_stream=src, src_location=src_location),
            "ResidueEmbeddingFromStructure": lambda: ResidueEmbeddingFromStructure(
                src_stream=src, res_embedding_location=self.tmp, src_location=src_location),
            "ResidueAssemblyDatasetFromStructure": lambda: ResidueAssemblyDatasetFromStructure(
                src_stream=src, res_embedding_location=self.tmp, src_location=src_location),
            "read_residue_file_list": lambda: _read_residue_file_list(src, src_location),
            "ResidueEmbeddingFromTensorFile": lambda: ResidueEmbeddingFromTensorFile(src_stream=src,
                                                                                     src_location=src_location),
        }

    def frame(self, reader: str, src, src_location: SrcLocation = SrcLocation.file) -> pd.DataFrame:
        """The frame that this reader made of the list."""
        modules = {"ResidueEmbeddingFromStructure": residue_embedding_from_structure,
                   "ResidueAssemblyDatasetFromStructure": resdiue_assembly_embedding_from_structure}
        read = self.readers(src, src_location)[reader]
        if reader not in modules:
            return read().data
        with mock.patch.object(modules[reader], "refuse_shared_names", _raise_frame):
            with self.assertRaises(_Frame) as frame:
                read()
        return frame.exception.data

    def assert_refused(self, refusal: BaseException, kind: str, source: str, found: str):
        """The refusal names the source and what it found, says what a row holds, and what to do."""
        self.assertIsInstance(refusal, RefusalError)
        self.assertRegex(str(refusal), f"^{re.escape(source)}: {found}\\. "
                                       f"{re.escape(_RULE[kind] + ': ' + _REMOVE_THE_EXTRA_FIELDS[kind])}$")


class LongerRowTest(_ListCase):
    """D1: the read functions and the 5 readers refuse a row of more fields than their columns, from a path, an open
    file (named by its path) and a text buffer ("The structure list", "The residue file list"), and from a stream; any
    other ParserError too."""

    def test_a_file_row_of_too_many_fields(self):
        """A first row with a trailing comma, with a field of text or with two more fields, every row with one or two
        index columns (pandas took them as the index), a chain list or a structure list given by mistake, and a later
        row longer than the first, or longer still after a longer first row (pandas' ParserError, which names the line,
        quoted and kept as the cause)."""
        for kind, lists in _LISTS.items():
            for case, text, found in lists:
                path = self.list_file(f"{kind} {case}", text)
                for source, make, named in self.sources(kind, path, text):
                    for reader in _READERS[kind]:
                        with self.subTest(kind=kind, case=case, source=source, reader=reader):
                            src = make()
                            try:
                                with self.assertRaises(RefusalError) as refused:
                                    self.readers(src)[reader]()
                            finally:
                                if hasattr(src, "close"):
                                    src.close()
                            self.assert_refused(refused.exception, kind, named, found)
                            if "later row" in case:
                                self.assertIsInstance(refused.exception.__cause__, pd.errors.ParserError)

    def test_the_index_column_of_to_csv(self):
        """What pandas' to_csv(header=False) writes, an index column 0, 1, 2 before the columns, is refused, on every
        pandas: read without dtype=str, pandas 3 makes a RangeIndex of it."""
        rows = {_STRUCTURE_LIST: [(f"{c}", f"{c}.cif", f"{c}-n") for c in "abcd"],
                _RESIDUE_FILE_LIST: [(f"{c}.pt", f"{c}-n") for c in "abcd"]}
        for kind, listed in rows.items():
            path = self.tmp / f"{kind.replace(' ', '_')}_to_csv.csv"
            pd.DataFrame(listed).to_csv(path, header=False)
            for reader in _READERS[kind]:
                with self.subTest(kind=kind, reader=reader):
                    with self.assertRaises(RefusalError) as refused:
                        self.readers(path)[reader]()
                    self.assert_refused(refused.exception, kind, str(path), _first_row(len(listed[0]) + 1))

    def test_a_stream_row_of_too_many_elements(self):
        """A stream: how many rows have too many elements, and up to 5 of them, counted from 1 (before, pandas'
        ValueError named no row), in a list and in a generator."""
        for kind, streams in _STREAMS.items():
            for rows, found in streams:
                message = f"The {kind} stream: {found}, counted from 1). {_RULE[kind]}: remove the extra elements."
                for container in (list, iter):
                    for reader in _READERS[kind]:
                        with self.subTest(kind=kind, rows=len(rows), longest=max(map(len, rows)),
                                          container=container.__name__, reader=reader):
                            with self.assertRaises(RefusalError) as refused:
                                self.readers(container(rows), SrcLocation.stream)[reader]()
                            self.assertEqual(str(refused.exception), message)

    def test_another_parser_error_is_not_claimed_as_extra_fields(self):
        """Another pandas ParserError (an unclosed quote) is refused too, naming the file and quoting pandas, without
        saying that a row has too many fields (before, pandas' bare ParserError)."""
        for kind, text in _UNCLOSED_QUOTE.items():
            path = self.list_file(f"{kind} unclosed quote", text)
            for source, make, named in self.sources(kind, path, text):
                for reader in _READERS[kind]:
                    with self.subTest(kind=kind, source=source, reader=reader):
                        src = make()
                        try:
                            with self.assertRaises(RefusalError) as refused:
                                self.readers(src)[reader]()
                        finally:
                            if hasattr(src, "close"):
                                src.close()
                        self.assertIsInstance(refused.exception.__cause__, pd.errors.ParserError)
                        self.assertRegex(str(refused.exception),
                                         f"^{re.escape(named)}: pandas could not read it as CSV \\([^()]*EOF inside "
                                         f"string[^()]*\\)\\. {re.escape(_CSV_RULE[kind])}$")
                        self.assertNotIn("more than", str(refused.exception))


class EntryPointTest(_ListCase):
    """D1 and D4: each entry point refuses such a list, in a file and as a stream, before any model is built (the model
    getters fail if called), and writes nothing."""

    def setUp(self):
        super().setUp()
        self.never = mock.Mock(side_effect=AssertionError("a model was built"))
        for module in ("esm_inference", "structure_inference", "chain_complete_inference",
                       "assembly_complete_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_residue_model", self.never))
        for module in ("structure_inference", "chain_inference", "chain_complete_inference",
                       "assembly_complete_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_aggregator_model", self.never))
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.residues = self.tmp / "residues"
        self.residues.mkdir()

    def entry_points(self, src, src_location: SrcLocation, out_format: OutFormat = OutFormat.csv) -> dict:
        """Each entry point of a structure list on src, a callable by name (out_format: the writer it would make)."""
        common = {"src_stream": src, "src_location": src_location, "out_folder": self.out, **_CPU}
        written = {**common, "out_format": out_format}
        return {
            "esm_inference": lambda: esm_inference.predict(src_from=SrcProteinFrom.structure, **written),
            "structure_inference": lambda: structure_inference.predict(src_from=SrcProteinFrom.structure, **common),
            "chain_inference": lambda: chain_inference.predict(src_from=SrcTensorFrom.structure,
                                                               res_embedding_location=self.residues, **written),
            "assembly_inferece": lambda: assembly_inferece.predict(src_from=SrcAssemblyFrom.structure,
                                                                   res_embedding_location=self.residues, **written),
            "chain_complete_inference": lambda: chain_complete_inference.predict(src_from=SrcEsmFrom.structure,
                                                                                 **written),
            "assembly_complete_inference": lambda: assembly_complete_inference.predict(**written),
        }

    def assert_nothing_done(self):
        self.never.assert_not_called()
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(list(self.residues.iterdir()), [])

    def test_a_structure_list_in_a_file(self):
        for case, text, found in _LISTS[_STRUCTURE_LIST]:
            path = self.list_file(case, text)
            for entry_point, predict in self.entry_points(path, SrcLocation.file).items():
                with self.subTest(case=case, entry_point=entry_point):
                    with self.assertRaises(RefusalError) as refused:
                        predict()
                    self.assert_refused(refused.exception, _STRUCTURE_LIST, str(path), found)
                    self.assert_nothing_done()

    def test_a_residue_file_list_in_a_file(self):
        """chain_inference.predict from residue embedding files, its default."""
        for case, text, found in _LISTS[_RESIDUE_FILE_LIST]:
            path = self.list_file(case, text)
            with self.subTest(case=case):
                with self.assertRaises(RefusalError) as refused:
                    chain_inference.predict(src_stream=path, out_folder=self.out, **_CPU)
                self.assert_refused(refused.exception, _RESIDUE_FILE_LIST, str(path), found)
                self.assert_nothing_done()

    def test_a_stream(self):
        for rows, found in _STREAMS[_STRUCTURE_LIST]:
            for entry_point, predict in self.entry_points(rows, SrcLocation.stream).items():
                with self.subTest(rows=len(rows), longest=max(map(len, rows)), entry_point=entry_point):
                    with self.assertRaises(RefusalError) as refused:
                        predict()
                    self.assertIn(f"The structure list stream: {found}", str(refused.exception))
                    self.assert_nothing_done()
        for rows, found in _STREAMS[_RESIDUE_FILE_LIST]:
            with self.subTest(rows=len(rows), longest=max(map(len, rows)), entry_point="chain_inference, files"):
                with self.assertRaises(RefusalError) as refused:
                    chain_inference.predict(src_stream=rows, src_location=SrcLocation.stream, out_folder=self.out,
                                            **_CPU)
                self.assertIn(f"The residue file list stream: {found}", str(refused.exception))
                self.assert_nothing_done()

    def test_every_writer(self):
        """chain_complete_inference makes its writer before it reads its list: no writer leaves a file behind."""
        path = self.list_file("trailing comma", "a,f.cif,a-n,\n")
        for out_format in OutFormat:
            for entry_point, predict in self.entry_points(path, SrcLocation.file, out_format).items():
                with self.subTest(out_format=out_format.value, entry_point=entry_point):
                    with self.assertRaises(RefusalError):
                        predict()
                    self.assert_nothing_done()


class AsBeforeTest(_ListCase):
    """Lists of at most that many fields per row read as before: the frames recorded from the readers of 8ded017, and
    the streams of the commands."""

    # (case, structure list, the rows of EsmProtFromStructure's and EsmAssemblyFromStructure's frame,
    # keep_default_na=False and no sort; the rows of ResidueEmbeddingFromStructure's and
    # ResidueAssemblyDatasetFromStructure's, pandas' default NA values and sorted by structure name; that frame's index)
    STRUCTURE_LISTS = (
        ("rows of 1, 2 and 3 fields", "c,f.cif,c-n\nb,f.cif\na\n",
         [["c", "f.cif", "c-n"], ["b", "f.cif", ""], ["a", "", ""]],
         [["a", None, None], ["b", "f.cif", None], ["c", "f.cif", "c-n"]], [2, 1, 0]),
        ("an empty 3rd field and NA", "b,f.cif,\nNA,g.cif,a-n\na,NA,NA\n",
         [["b", "f.cif", ""], ["NA", "g.cif", "a-n"], ["a", "NA", "NA"]],
         [["a", None, None], ["b", "f.cif", None], [None, "g.cif", "a-n"]], [2, 0, 1]),
        ("a byte-order mark and CRLF", "﻿b,f.cif,b-n\r\na,g.cif,a-n\r\n",
         [["b", "f.cif", "b-n"], ["a", "g.cif", "a-n"]],
         [["a", "g.cif", "a-n"], ["b", "f.cif", "b-n"]], [1, 0]),
        ("quoted fields", 'b,"f,1.cif","n,1"\n"a\nz",g.cif,\n',
         [["b", "f,1.cif", "n,1"], ["a\nz", "g.cif", ""]],
         [["a\nz", "g.cif", None], ["b", "f,1.cif", "n,1"]], [1, 0]),
        ("blank lines", "\nb,f.cif,b-n\n\n  \n\t\na,g.cif,a-n\n\n",
         [["b", "f.cif", "b-n"], ["a", "g.cif", "a-n"]],
         [["a", "g.cif", "a-n"], ["b", "f.cif", "b-n"]], [1, 0]),
        ("an empty file", "", [], [], []),
    )
    # (case, residue file list, the rows of ResidueEmbeddingFromTensorFile's frame, sorted by file; its index)
    RESIDUE_FILE_LISTS = (
        ("rows of 1 and 2 fields", "c.pt,c-n\nb.pt\na.pt,a-n\n",
         [["a.pt", "a-n"], ["b.pt", None], ["c.pt", "c-n"]], [2, 1, 0]),
        ("an empty 2nd field and NA", "b.pt,\nNA,a-n\na.pt,NA\n",
         [["a.pt", None], ["b.pt", None], [None, "a-n"]], [2, 0, 1]),
        ("a byte-order mark and CRLF", "﻿b.pt,b-n\r\na.pt,a-n\r\n",
         [["a.pt", "a-n"], ["b.pt", "b-n"]], [1, 0]),
        ("quoted fields", 'b.pt,"n,1"\n"a\nz.pt",\n',
         [["a\nz.pt", None], ["b.pt", "n,1"]], [1, 0]),
        ("blank lines", "\nb.pt,b-n\n\n  \n\t\na.pt,a-n\n\n",
         [["a.pt", "a-n"], ["b.pt", "b-n"]], [1, 0]),
        ("an empty file", "", [], []),
    )

    def test_the_readers_columns(self):
        """D2: each reader keeps its COLUMNS, which are the read function's."""
        from foldmatch.dataset.utils.utils import RESIDUE_FILE_LIST_COLUMNS, STRUCTURE_LIST_COLUMNS
        self.assertEqual(STRUCTURE_LIST_COLUMNS, _STRUCTURE_COLUMNS)
        self.assertEqual(RESIDUE_FILE_LIST_COLUMNS, _RESIDUE_FILE_COLUMNS)
        for reader in (EsmProtFromStructure, EsmAssemblyFromStructure, ResidueEmbeddingFromStructure,
                       ResidueAssemblyDatasetFromStructure):
            self.assertEqual(reader.COLUMNS, STRUCTURE_LIST_COLUMNS, reader)
        self.assertEqual(ResidueEmbeddingFromTensorFile.COLUMNS, RESIDUE_FILE_LIST_COLUMNS)

    def test_the_structure_readers_read_them_as_before(self):
        """The 4 readers' frames, from a path, an open file and a text buffer, each with its NA handling and sort."""
        for case, text, kept, rows, index in self.STRUCTURE_LISTS:
            path = self.list_file(case, text)
            for source, make, _ in self.sources(_STRUCTURE_LIST, path, text):
                for reader, expected in (
                        ("EsmProtFromStructure", _expected(kept, list(range(len(kept))), _STRUCTURE_COLUMNS)),
                        ("EsmAssemblyFromStructure", _expected(kept, list(range(len(kept))), _STRUCTURE_COLUMNS)),
                        ("ResidueEmbeddingFromStructure", _expected(rows, index, _STRUCTURE_COLUMNS)),
                        ("ResidueAssemblyDatasetFromStructure", _expected(rows, index, _STRUCTURE_COLUMNS))):
                    with self.subTest(case=case, source=source, reader=reader):
                        src = make()
                        try:
                            data = self.frame(reader, src)
                        finally:
                            if hasattr(src, "close"):
                                src.close()
                        pd.testing.assert_frame_equal(data, expected, check_exact=True)

    def test_the_residue_file_reader_reads_them_as_before(self):
        """ResidueEmbeddingFromTensorFile's frame, from a path, an open file and a text buffer: its names as strings
        where they are not numeric (D3 changes nothing there), an empty name or NA as NaN. An empty file's columns were
        of object type on pandas 3: NamesTest."""
        for case, text, rows, index in self.RESIDUE_FILE_LISTS:
            path = self.list_file(case, text)
            for source, make, _ in self.sources(_RESIDUE_FILE_LIST, path, text):
                with self.subTest(case=case, source=source):
                    src = make()
                    try:
                        data = self.frame("ResidueEmbeddingFromTensorFile", src)
                    finally:
                        if hasattr(src, "close"):
                            src.close()
                    pd.testing.assert_frame_equal(data, _expected(rows, index, _RESIDUE_FILE_COLUMNS),
                                                  check_exact=True, check_dtype=bool(rows))

    def test_the_read_functions(self):
        """read_structure_list and read_residue_file_list: the frame of the readers' pandas call on 8ded017, in input
        order, its index pandas' (0, 1, ...). A residue file list's empty file: columns of strings (NamesTest)."""
        for case, text, kept, rows, index in self.STRUCTURE_LISTS:
            path = self.list_file(case, text)
            for source, src in (("path", path), ("text buffer", None)):
                with self.subTest(case=case, source=source, keep_default_na=False):
                    data = _read_structure_list(src or io.StringIO(text), keep_default_na=False)
                    pd.testing.assert_frame_equal(data, _expected(kept, list(range(len(kept))), _STRUCTURE_COLUMNS),
                                                  check_exact=True)
                    self.assertIsInstance(data.index, pd.RangeIndex)
                with self.subTest(case=case, source=source, keep_default_na=True):
                    data = _read_structure_list(src or io.StringIO(text))
                    pd.testing.assert_frame_equal(
                        data, _expected(_unsorted(rows, index), sorted(index), _STRUCTURE_COLUMNS), check_exact=True)
                    self.assertIsInstance(data.index, pd.RangeIndex)
        for case, text, rows, index in self.RESIDUE_FILE_LISTS:
            path = self.list_file(case, text)
            for source, src in (("path", path), ("text buffer", None)):
                with self.subTest(case=case, source=source):
                    data = _read_residue_file_list(src or io.StringIO(text))
                    pd.testing.assert_frame_equal(
                        data, _expected(_unsorted(rows, index), sorted(index), _RESIDUE_FILE_COLUMNS),
                        check_exact=True)
                    self.assertIsInstance(data.index, pd.RangeIndex)

    def assert_structure_stream(self, make, rows: list, index: list):
        """The 4 readers' frames of the stream that make gives (rows: in input order; index: of the sorted frame)."""
        unsorted_frame = _expected(rows, list(range(len(rows))), _STRUCTURE_COLUMNS)
        for reader, expected in (
                ("EsmProtFromStructure", unsorted_frame), ("EsmAssemblyFromStructure", unsorted_frame),
                ("ResidueEmbeddingFromStructure", unsorted_frame.loc[index]),
                ("ResidueAssemblyDatasetFromStructure", unsorted_frame.loc[index])):
            with self.subTest(reader=reader):
                pd.testing.assert_frame_equal(self.frame(reader, make(), SrcLocation.stream), expected,
                                              check_exact=True)

    def test_the_streams_of_the_commands(self):
        """D5: fm-embedding from-structures gives scan_structure_folder's tuple of (name, path as a string, name),
        fm-search build a list of (stem, Path, stem): read by pandas as before, and so are a generator and short
        rows."""
        scanned = scan_structure_folder(_PDB, StructureFormat.mmcif)
        self.assertEqual(scanned, (("1acb", str(_CIF), "1acb"), ("2uzi", str(_CIF2), "2uzi")))
        reversed_paths = [(Path(f).stem, Path(f), Path(f).stem) for _, f, _ in reversed(scanned)]
        as_read = [[Path(f).stem, f, Path(f).stem] for _, f, _ in reversed(scanned)]
        for case, make, rows, index in (
                ("scan_structure_folder", lambda: scanned, [list(row) for row in scanned], [0, 1]),
                ("(stem, Path, stem)", lambda: reversed_paths, as_read, [1, 0]),
                ("a generator", lambda: iter(reversed_paths), as_read, [1, 0]),
                ("short rows and None", lambda: [("b", "g.cif"), ("a", None, "x"), ("c",)],
                 [["b", "g.cif", None], ["a", None, "x"], ["c", None, None]], [1, 0, 2]),
                ("an empty list", lambda: [], [], [])):
            with self.subTest(case=case):
                self.assert_structure_stream(make, rows, index)

    def test_the_streams_of_the_residue_file_reader(self):
        """inference chain-embedding gives a list of (file, name), ResidueEmbeddingFromStructure its parent a tuple of
        them: read by pandas as before, numeric-looking names as they are."""
        rows = [("/x/b.pt", "007"), ("/x/a.pt", "1.50")]
        expected = _expected([["/x/a.pt", "1.50"], ["/x/b.pt", "007"]], [1, 0], _RESIDUE_FILE_COLUMNS)
        for case, make in (("a list", lambda: rows), ("a tuple", lambda: tuple(rows)),
                           ("a generator", lambda: iter(rows)),
                           ("Path elements", lambda: [(Path(f), n) for f, n in rows])):
            with self.subTest(case=case):
                pd.testing.assert_frame_equal(self.frame("ResidueEmbeddingFromTensorFile", make(), SrcLocation.stream),
                                              expected, check_exact=True)
        with self.subTest(case="short rows and None"):
            pd.testing.assert_frame_equal(
                self.frame("ResidueEmbeddingFromTensorFile", [("b.pt", "n"), ("a.pt",), ("c.pt", None)],
                           SrcLocation.stream),
                _expected([["a.pt", None], ["b.pt", "n"], ["c.pt", None]], [1, 0, 2], _RESIDUE_FILE_COLUMNS),
                check_exact=True)

    def test_to_csv_without_the_index(self):
        """What the CHANGELOG tells them to write: to_csv(header=False, index=False) gives the list as it was meant."""
        rows = [("b", "g.cif", "b-n"), ("a", "f.cif", "a-n")]
        path = self.tmp / "structures.csv"
        pd.DataFrame(rows).to_csv(path, header=False, index=False)
        for reader in ("EsmProtFromStructure", "EsmAssemblyFromStructure"):
            with self.subTest(reader=reader):
                pd.testing.assert_frame_equal(self.frame(reader, path), _expected(rows, [0, 1], _STRUCTURE_COLUMNS),
                                              check_exact=True)
        for reader in ("ResidueEmbeddingFromStructure", "ResidueAssemblyDatasetFromStructure"):
            with self.subTest(reader=reader):
                pd.testing.assert_frame_equal(self.frame(reader, path),
                                              _expected(rows[::-1], [1, 0], _STRUCTURE_COLUMNS), check_exact=True)
        rows = [("b.pt", "b-n"), ("a.pt", "a-n")]
        path = self.tmp / "residues.csv"
        pd.DataFrame(rows).to_csv(path, header=False, index=False)
        with self.subTest(reader="ResidueEmbeddingFromTensorFile"):
            pd.testing.assert_frame_equal(self.frame("ResidueEmbeddingFromTensorFile", path),
                                          _expected(rows[::-1], [1, 0], _RESIDUE_FILE_COLUMNS), check_exact=True)


class _StubModels(_ListCase):
    """The entry points with stub models: no ESM3, no aggregator."""

    def setUp(self):
        super().setUp()
        for module in ("esm_inference", "structure_inference", "chain_complete_inference",
                       "assembly_complete_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_residue_model", return_value="ESM3"))
        for module in ("structure_inference", "chain_inference", "chain_complete_inference",
                       "assembly_complete_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_aggregator_model",
                                         return_value=_aggregate))
        for module in ("esm_module", "chain_complete_module", "assembly_complete_module"):
            self.enterContext(mock.patch(f"foldmatch.modules.{module}.per_residue_embedding", _esm3))
        self.residues_of = {(cif.stem, chain): _esm3(None, protein) for cif in (_CIF, _CIF2)
                            for chain, protein in _proteins(cif).items()}

    def folder(self, name: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.tmp))

    def assert_files(self, out: Path, expected: dict):
        self.assertEqual({f.name: f.read_text() for f in out.iterdir()}, expected)

    def stored(self, entry: str, chain: str) -> torch.Tensor:
        """A residue embedding stored in tests/resources (1536 values per residue)."""
        return torch.load(_EMBEDDINGS / f"{entry}.{chain}.pt", map_location="cpu", weights_only=True)


class ConventionalListTest(_StubModels):
    """The files that a conventional list gives with stub models, through the unchanged writers, as before."""

    def test_the_entry_points_on_a_list_in_a_file(self):
        structures = self.list_file("structures", f"1acb,{_CIF},x\n")
        residues = {c: self.residues_of[("1acb", c)] for c in "AB"}
        out = self.folder("out")
        esm_inference.predict(src_stream=structures, src_from=SrcProteinFrom.structure, out_folder=out,
                              return_predictions=False, **_CPU)
        self.assert_files(out, {f"x.{c}.csv": _csv(residues[c].numpy()) for c in "AB"})

        out = self.folder("out")
        chain_complete_inference.predict(src_stream=structures, src_from=SrcEsmFrom.structure, out_folder=out,
                                         return_predictions=False, **_CPU)
        self.assert_files(out, {f"x.{c}.csv": _csv(_chain_embedding(residues[c])) for c in "AB"})

        out = self.folder("out")
        assembly_complete_inference.predict(src_stream=structures, out_folder=out, return_predictions=False, **_CPU)
        self.assert_files(out, {"x-1.csv": _csv(_chain_embedding(torch.cat([residues["A"], residues["B"]])))})

        out = self.folder("out")
        structure_inference.predict(src_stream=structures, src_from=SrcProteinFrom.structure, out_folder=out,
                                    return_predictions=False, **_CPU)
        table = pq.read_table(out / "inference-0.parquet")
        self.assertEqual(table["id"].to_pylist(), ["x.A", "x.B"])
        for c, row in zip("AB", table["embedding"].to_pylist()):
            np.testing.assert_array_equal(np.array(row, dtype=np.float32), _chain_embedding(residues[c]))

        stored = {c: self.stored("1acb", c) for c in "AB"}
        out = self.folder("out")
        chain_inference.predict(src_stream=structures, src_from=SrcTensorFrom.structure,
                                res_embedding_location=_EMBEDDINGS, out_folder=out, return_predictions=False, **_CPU)
        self.assert_files(out, {f"x.{c}.csv": _csv(_chain_embedding(stored[c])) for c in "AB"})

        out = self.folder("out")
        assembly_inferece.predict(src_stream=structures, src_from=SrcAssemblyFrom.structure,
                                  res_embedding_location=_EMBEDDINGS, out_folder=out, return_predictions=False, **_CPU)
        self.assert_files(out, {"x-1.csv": _csv(_chain_embedding(torch.cat([stored["A"], stored["B"]])))})

        files = self.list_file("files", f"{_EMBEDDINGS}/1acb.B.pt,r\n{_EMBEDDINGS}/1acb.A.pt,s\n")
        out = self.folder("out")
        chain_inference.predict(src_stream=files, out_folder=out, return_predictions=False, **_CPU)
        self.assert_files(out, {"r.csv": _csv(_chain_embedding(stored["B"])),
                                "s.csv": _csv(_chain_embedding(stored["A"]))})

    def from_structures(self, command: str, out: Path):
        """fm-embedding from-structures on tests/resources/pdb in-process, the logging setup restored after."""
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            result = CliRunner().invoke(embedding.app, ["from-structures", command, "--src-folder", str(_PDB),
                                                        "--output-folder", str(out), "--accelerator", "cpu",
                                                        "--devices", "1"])
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")

    def test_from_structures_residue(self):
        out = self.folder("out")
        self.from_structures("residue", out)
        self.assert_files(out, {f"{entry}.{c}.csv": _csv(residues.numpy())
                                for (entry, c), residues in self.residues_of.items()})

    def test_from_structures_chain(self):
        out = self.folder("out")
        self.from_structures("chain", out)
        self.assert_files(out, {f"{entry}.{c}.csv": _csv(_chain_embedding(residues))
                                for (entry, c), residues in self.residues_of.items()})

    def test_from_structures_assembly(self):
        out = self.folder("out")
        self.from_structures("assembly", out)
        expected = {}
        for cif in (_CIF, _CIF2):
            for assembly_id in get_assemblies(structure=str(cif), structure_format=StructureFormat.mmcif):
                structure = get_structure_from_src(str(cif), StructureFormat.mmcif, assembly_id=assembly_id)
                proteins = [ESMProtein.from_protein_chain(ProteinChain.from_atomarray(
                    rename_atom_attr(filter_residues(atoms)))) for atoms in protein_chain_iter(structure)]
                residues = torch.cat([_esm3(None, protein) for protein in proteins])
                expected[f"{cif.stem}-{assembly_id}.csv"] = _csv(_chain_embedding(residues))
        self.assertTrue(expected)
        self.assert_files(out, expected)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "a named pipe needs os.mkfifo")
    def test_a_list_in_a_pipe(self):
        """D4: a list given as a pipe can be read once: each reader reads it once, where it did, and no entry point
        reads it first, so a list in a named pipe reaches esm_inference.predict and chain_inference.predict, as on
        8ded017."""
        residues = self.residues_of[("1acb", "A")].numpy()
        stored = self.stored("1acb", "A")
        for case, text, predict, expected in (
                ("esm_inference", f"1acb,{_CIF},x\n",
                 lambda fifo, out: esm_inference.predict(src_stream=fifo, src_from=SrcProteinFrom.structure,
                                                         out_folder=out, return_predictions=False, **_CPU),
                 {"x.A.csv": _csv(residues), "x.B.csv": _csv(self.residues_of[("1acb", "B")].numpy())}),
                ("chain_inference", f"{_EMBEDDINGS}/1acb.A.pt,r\n",
                 lambda fifo, out: chain_inference.predict(src_stream=fifo, out_folder=out, return_predictions=False,
                                                           **_CPU),
                 {"r.csv": _csv(_chain_embedding(stored))})):
            with self.subTest(case=case):
                fifo = self.tmp / f"{case}.fifo"
                os.mkfifo(fifo)
                stop = threading.Event()
                writer = threading.Thread(target=_write_once, args=(fifo, text, stop), daemon=True)
                writer.start()
                out = self.folder("out")
                try:
                    predict(fifo, out)
                finally:
                    stop.set()
                    writer.join()
                self.assert_files(out, expected)


class NamesTest(_StubModels):
    """D3: the residue file list is read as strings, as every other list is, its names keeping their spelling. An empty
    name stays NaN (default NA values kept)."""

    def setUp(self):
        super().setUp()
        for name in "ab":
            torch.save(torch.ones(3, 2) * (ord(name) - 96), self.tmp / f"{name}.pt")
        self.a, self.b = _chain_embedding(torch.ones(3, 2)), _chain_embedding(torch.ones(3, 2) * 2)

    def names_list(self, *names) -> Path:
        return self.list_file("names", "".join(f"{self.tmp}/{f}.pt,{name}\n" for f, name in zip("ab", names)))

    def predict(self, names: tuple, out_format: OutFormat) -> Path:
        """chain_inference.predict on a list of a.pt and b.pt under these names, stub models: its output folder."""
        out = self.folder("out")
        chain_inference.predict(src_stream=self.names_list(*names), out_format=out_format, out_folder=out,
                                return_predictions=False, **_CPU)
        return out

    def test_the_names_as_read(self):
        """From a path, an open file and a text buffer, as strings (before, 7.0 and 1.5, the integers 1 and 2, the
        booleans True and False, or 1 and 2 without their space), and by each item of the dataset."""
        for names in (("007", "1.50"), ("1", "2"), ("007", "008"), ("1.0", "2.0"), ("TRUE", "false"), (" 1", "2 ")):
            path = self.names_list(*names)
            text = path.read_text()
            for source, make, _ in self.sources(_RESIDUE_FILE_LIST, path, text):
                with self.subTest(names=names, source=source):
                    src = make()
                    try:
                        dataset = ResidueEmbeddingFromTensorFile(src_stream=src)
                    finally:
                        if hasattr(src, "close"):
                            src.close()
                    self.assertEqual(list(dataset.data.item_name), list(names))
                    self.assertEqual([dataset[k][1] for k in range(len(dataset))], list(names))
                    self.assertIsInstance(dataset[0][1], str)

    def test_a_name_with_its_spelling_names_the_output(self):
        """007 and 1.50 give 007.csv and 1.50.csv (before, 7.0.csv and 1.5.csv), the same in pt, and the ids '007' and
        '1.50' in JSON (before, the numbers 7.0 and 1.5) and in Parquet (before, ArrowTypeError)."""
        names = ("007", "1.50")
        self.assert_files(self.predict(names, OutFormat.csv), {"007.csv": _csv(self.a), "1.50.csv": _csv(self.b)})
        self.assertEqual(sorted(f.name for f in self.predict(names, OutFormat.pt).iterdir()), ["007.pt", "1.50.pt"])
        self.assert_ids(names, names)

    def test_a_name_that_looks_like_a_boolean(self):
        """TRUE and false give TRUE.csv and false.csv (before, True.csv and False.csv), the same in pt, and the ids
        'TRUE' and 'false' in JSON (before, true and false) and in Parquet (before, ArrowTypeError)."""
        names = ("TRUE", "false")
        self.assert_files(self.predict(names, OutFormat.csv), {"TRUE.csv": _csv(self.a), "false.csv": _csv(self.b)})
        self.assertEqual(sorted(f.name for f in self.predict(names, OutFormat.pt).iterdir()), ["TRUE.pt", "false.pt"])
        self.assert_ids(names, names)

    def assert_ids(self, names: tuple, ids: tuple):
        """The ids in the Parquet and JSON output of these names."""
        with self.subTest(names=names, out_format="parquet"):
            table = pq.read_table(self.predict(names, OutFormat.parquet) / "inference-0.parquet")
            self.assertEqual(table["id"].to_pylist(), list(ids))
        with self.subTest(names=names, out_format="json"):
            with gzip.open(self.predict(names, OutFormat.json) / "inference-0.json.gz", "rt") as f:
                self.assertEqual([row["id"] for row in json.load(f)], list(ids))

    def test_plain_integers_in_parquet_and_json(self):
        """1 and 2 give the ids '1' and '2' in Parquet (before, ArrowTypeError, nothing written) and in JSON (before,
        the numbers 1 and 2)."""
        self.assert_ids(("1", "2"), ("1", "2"))

    def test_plain_integers_in_csv_and_pt(self):
        """1 and 2 name the same files as before."""
        self.assert_files(self.predict(("1", "2"), OutFormat.csv), {"1.csv": _csv(self.a), "2.csv": _csv(self.b)})
        self.assertEqual(sorted(f.name for f in self.predict(("1", "2"), OutFormat.pt).iterdir()), ["1.pt", "2.pt"])

    def test_an_empty_name_is_nan(self):
        """As before: an empty name or NA next to a text name is NaN, and names the file nan.csv."""
        for names in (("x", ""), ("x", "NA")):
            with self.subTest(names=names):
                dataset = ResidueEmbeddingFromTensorFile(src_stream=self.names_list(*names))
                self.assertEqual(dataset.data.item_name.iloc[0], "x")
                self.assertTrue(pd.isna(dataset.data.item_name.iloc[1]))
                self.assert_files(self.predict(names, OutFormat.csv), {"x.csv": _csv(self.a), "nan.csv": _csv(self.b)})

    def test_an_integer_next_to_an_empty_name(self):
        """An integer keeps its spelling next to an empty name, which is NaN: 1.csv and nan.csv (before, 1.0.csv)."""
        self.assert_files(self.predict(("1", ""), OutFormat.csv), {"1.csv": _csv(self.a), "nan.csv": _csv(self.b)})

    def test_the_empty_name_column_is_of_strings(self):
        """Empty names only: NaN in a column of strings (before, of floats)."""
        data = ResidueEmbeddingFromTensorFile(src_stream=self.names_list("", "")).data
        self.assertTrue(data.item_name.isna().all())
        pd.testing.assert_frame_equal(data, _expected([[f"{self.tmp}/a.pt", None], [f"{self.tmp}/b.pt", None]], [0, 1],
                                                      _RESIDUE_FILE_COLUMNS), check_exact=True)

    def test_an_empty_file_is_of_strings(self):
        """An empty file: no rows, its columns of strings, as every other list's (before, object on pandas 3)."""
        data = ResidueEmbeddingFromTensorFile(src_stream=self.list_file("empty", "")).data
        pd.testing.assert_frame_equal(data, _expected([], [], _RESIDUE_FILE_COLUMNS), check_exact=True)

    def test_a_file_column_of_numbers_sorts_as_text(self):
        """Contrived: a file column that holds only numbers sorts as text (before, as numbers)."""
        data = ResidueEmbeddingFromTensorFile(src_stream=self.list_file("numbers", "2.5,x\n10,y\n")).data
        pd.testing.assert_frame_equal(data, _expected([["10", "y"], ["2.5", "x"]], [1, 0], _RESIDUE_FILE_COLUMNS),
                                      check_exact=True)


if __name__ == "__main__":
    unittest.main()
