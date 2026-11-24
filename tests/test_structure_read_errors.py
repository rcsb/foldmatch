"""A structure file that cannot be read is refused (RefusalError, a ValueError); nothing else is.

structure_parser.get_structure_from_src and get_assemblies refuse what biotite raises because the file is not a valid
structure file of its format: malformed, truncated, of another format, compressed wrongly, without the assembly asked
for. They used to wrap every exception in a RuntimeError, and wrapping every exception in a RefusalError instead would
report the caller's mistakes, the machine's failures and FoldMatch's own bugs as the input's fault (an ``Error: ...``
line without a traceback in the command-line tools). So these propagate unchanged: an unknown format (ValueError), a
stream of the wrong mode, bytes, a file's content passed as a path or an assembly ID that is not a str (TypeError), a
path that cannot be opened (OSError), a MemoryError, an unexpected error type from biotite (also when biotite wraps it
in a DeserializationError); and a failure in FoldMatch's own processing is an InternalError.

The fixtures are 1acb (tests/resources/pdb) as mmCIF, as PDB with an assembly in REMARK 350, and as BinaryCIF, whole or
damaged. ESM3 is never loaded."""
import gzip
import io
import os
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np
from biotite import DeserializationError
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import BinaryCIFFile, CIFCategory, CIFFile, get_structure, set_structure

from foldmatch.errors import InternalError, RefusalError
from foldmatch.types.api_types import StructureFormat
from foldmatch.utils import structure_parser
from foldmatch.utils.structure_parser import get_assemblies, get_structure_from_src

_CIF_PATH = Path(__file__).parent / "resources" / "pdb" / "1acb.cif"
_CIF = _CIF_PATH.read_text()

# Assembly 1 of the PDB fixture: both protein chains (author chain IDs E and I) by the identity
_REMARKS = "".join(f"{line:<80}\n" for line in (
    "REMARK 300",
    "REMARK 300 BIOMOLECULE: 1",
    "REMARK 350",
    "REMARK 350 BIOMOLECULE: 1",
    "REMARK 350 APPLY THE FOLLOWING TO CHAINS: E, I",
    "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000",
    "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000",
    "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000",
))


def _pdb_text() -> str:
    pdb_file = PDBFile()
    pdb_file.set_structure(get_structure(CIFFile.read(str(_CIF_PATH)), model=1))
    text = io.StringIO()
    pdb_file.write(text)
    return _REMARKS + text.getvalue()


def _bcif_bytes() -> bytes:
    bcif_file = BinaryCIFFile()
    set_structure(bcif_file, get_structure(CIFFile.read(str(_CIF_PATH)), model=1))
    data = io.BytesIO()
    bcif_file.write(data)
    return data.getvalue()


_PDB = _pdb_text()
_BCIF = _bcif_bytes()


class _Files(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name: str, content) -> str:
        path = self.folder / name
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)
        return str(path)

    def assertRefused(self, call, *, cause=Exception, source=None, action="structure") -> RefusalError:
        with self.assertRaises(RefusalError) as refused:
            call()
        error = refused.exception
        self.assertIsInstance(error, ValueError)
        self.assertIsInstance(error.__cause__, cause)
        message = str(error)
        self.assertTrue(message.startswith(f"Error reading {action} from "), message)
        self.assertNotIn(" object at 0x", message)
        if source is not None:
            self.assertIn(f"from {source}: ", message)
        return error


class ReadableFilesTest(_Files):
    def test_each_format_from_a_stream_and_a_path(self):
        expected = get_structure(CIFFile.read(str(_CIF_PATH)), model=1, use_author_fields=False).array_length()
        for structure_format, content in (("mmcif", _CIF), ("pdb", _PDB), ("binarycif", _BCIF)):
            with self.subTest(structure_format):
                stream = io.StringIO(content) if isinstance(content, str) else io.BytesIO(content)
                self.assertEqual(get_structure_from_src(stream, structure_format).array_length(), expected)
                path = self.write(f"1acb.{structure_format}", content)
                self.assertEqual(get_structure_from_src(path, structure_format).array_length(), expected)
                self.assertEqual(get_structure_from_src(Path(path), structure_format).array_length(), expected)

    def test_a_structure_format_member(self):
        self.assertEqual(get_structure_from_src(io.StringIO(_CIF), StructureFormat.mmcif).array_length(),
                         get_structure_from_src(io.StringIO(_CIF), "mmcif").array_length())
        self.assertGreater(get_structure_from_src(io.BytesIO(_BCIF), StructureFormat.bciff).array_length(), 0)
        self.assertEqual(get_assemblies(io.StringIO(_PDB), StructureFormat.pdb), ("1",))

    def test_gzip_whatever_the_suffix_case(self):
        for name in ("1acb.cif.gz", "1acb.cif.GZ"):
            with self.subTest(name):
                path = self.write(name, gzip.compress(_CIF.encode()))
                self.assertEqual(get_structure_from_src(path, "mmcif").array_length(),
                                 get_structure_from_src(io.StringIO(_CIF), "mmcif").array_length())
                self.assertEqual(get_assemblies(path, "mmcif"), ("1",))

    def test_a_pdb_assembly(self):
        self.assertEqual(get_assemblies(io.StringIO(_PDB), "pdb"), ("1",))
        assembly = get_structure_from_src(io.StringIO(_PDB), "pdb", assembly_id="1")
        self.assertIn("sym_id", assembly.get_annotation_categories())
        self.assertNotIn("au_index", assembly.get_annotation_categories())

    def test_a_stream_is_read_from_its_start_each_time(self):
        stream = io.StringIO(_CIF)
        self.assertEqual(get_assemblies(stream, "mmcif"), ("1",))
        self.assertGreater(get_structure_from_src(stream, "mmcif", assembly_id="1").array_length(), 0)
        self.assertFalse(stream.closed)

    def test_a_path_is_closed_after_reading(self):
        path = self.write("1acb.cif.gz", gzip.compress(_CIF.encode()))
        opened = []
        real_open = gzip.open

        def recording_open(*args, **kwargs):
            opened.append(real_open(*args, **kwargs))
            return opened[-1]

        with mock.patch.object(structure_parser.gzip, "open", recording_open):
            get_structure_from_src(path, "mmcif")
            with self.assertRaises(RefusalError):
                get_structure_from_src(path, "pdb", assembly_id="9")
        self.assertEqual(len(opened), 2)
        self.assertTrue(all(f.closed for f in opened))


class MalformedFilesTest(_Files):
    """What biotite raises for files that are not valid structure files of their format, as measured by damaging the
    fixtures: each is refused, naming the source (never a stream's repr), with biotite's error as the cause."""

    def test_malformed_mmcif(self):
        lines = _CIF.splitlines(keepends=True)
        atom_site = next(k for k, line in enumerate(lines) if line.startswith("ATOM "))
        unquoted = lines[atom_site].split()
        unquoted[6] = "seg 0 water"  # label_asym_id with spaces, unquoted: more values than the loop has columns
        cases = {
            "empty": "",
            "an HTML error page": "<html><body>404 Not Found</body></html>\n",
            "truncated": _CIF[:len(_CIF) // 2],
            "no atom_site": _CIF.replace("_atom_site.", "_atom_sitx."),
            "a coordinate that is not a number": "".join(
                lines[:atom_site] + [lines[atom_site].replace(unquoted[10], "1x.0", 1)] + lines[atom_site + 1:]),
            "an unquoted value with spaces": "".join(
                lines[:atom_site] + [" ".join(unquoted) + "\n"] + lines[atom_site + 1:]),
        }
        for case, text in cases.items():
            with self.subTest(case):
                self.assertRefused(lambda: get_structure_from_src(io.StringIO(text), "mmcif"),
                                   source="a StringIO stream")

    def test_malformed_pdb(self):
        cases = {
            "not a PDB file": "hello\nworld\n",
            "a coordinate that is not a number": _PDB.replace(_PDB[_PDB.index("\nATOM") + 31:][:8], "  1x.000", 1),
        }
        for case, text in cases.items():
            with self.subTest(case):
                self.assertRefused(lambda: get_structure_from_src(io.StringIO(text), "pdb"))

    def test_an_integer_too_large_for_its_column(self):
        lines = _CIF.splitlines(keepends=True)
        atom_site = next(k for k, line in enumerate(lines) if line.startswith("ATOM "))
        values = lines[atom_site].split()
        values[8] = "9" * 20  # label_seq_id
        text = "".join(lines[:atom_site] + [" ".join(values) + "\n"] + lines[atom_site + 1:])
        self.assertRefused(lambda: get_structure_from_src(io.StringIO(text), "mmcif"), cause=OverflowError)

    def test_an_incomplete_remark_300(self):
        # biotite raises IndexError for a REMARK 300 without its BIOMOLECULE line
        text = _PDB.replace(f"{'REMARK 300 BIOMOLECULE: 1':<80}\n", "")
        self.assertRefused(lambda: get_assemblies(io.StringIO(text), "pdb"), cause=IndexError, action="assemblies")

    def test_a_non_ascii_character_in_a_fixed_pdb_column(self):
        # biotite's Rust PDB reader panics on a multi-byte character across a column boundary: a PanicException, a
        # BaseException that no "except Exception" catches
        start = _PDB.index("\nATOM") + 1
        text = _PDB[:start + 77] + "é" + _PDB[start + 78:]
        error = self.assertRefused(lambda: get_structure_from_src(io.StringIO(text), "pdb"), cause=BaseException)
        self.assertEqual(type(error.__cause__).__name__, "PanicException")

    def test_malformed_binarycif(self):
        cases = {
            "random bytes": bytes(range(256)) * 4,
            "an mmCIF file": _CIF.encode(),
            "truncated": _BCIF[:len(_BCIF) // 2],
            # valid msgpack of another shape, which biotite indexes without checking: TypeError or AttributeError
            "one byte (a msgpack number)": b"5",
            "an empty file": b"",
            "a msgpack list": b"\x93\x01\x02\x03",
            "a map whose dataBlocks is a number": b"\x81\xaadataBlocks\x03",
        }
        for case, data in cases.items():
            with self.subTest(case):
                self.assertRefused(lambda: get_structure_from_src(io.BytesIO(data), "binarycif"),
                                   source="a BytesIO stream")
                self.assertRefused(lambda: get_assemblies(io.BytesIO(data), "binarycif"), action="assemblies")

    def test_badly_compressed_or_encoded_paths(self):
        corrupt = bytearray(gzip.compress(_CIF.encode()))
        corrupt[30:46] = bytes(b ^ 0xFF for b in corrupt[30:46])
        cases = {
            "plain text named .gz": ("plain.cif.gz", _CIF, gzip.BadGzipFile),
            "a truncated .gz": ("truncated.cif.gz", gzip.compress(_CIF.encode())[:2000], EOFError),
            "a corrupt deflate stream": ("corrupt.cif.gz", bytes(corrupt), zlib.error),
            # UnicodeDecodeError (a ValueError) under a UTF-8 locale; under Latin-1, no data block (a ValueError)
            "binary content named .cif": ("binary.cif", bytes(range(128, 256)) * 30, ValueError),
        }
        for case, (name, content, cause) in cases.items():
            with self.subTest(case):
                path = self.write(name, content)
                # the message names the path, not the stream that decompressed it
                self.assertRefused(lambda: get_structure_from_src(path, "mmcif"), cause=cause, source=path)

    def test_an_assembly_the_file_does_not_have(self):
        for structure_format, content in (("mmcif", io.StringIO(_CIF)), ("pdb", io.StringIO(_PDB))):
            with self.subTest(structure_format):
                self.assertRefused(lambda: get_structure_from_src(content, structure_format, assembly_id="9"),
                                   cause=KeyError)

    def test_a_file_without_assemblies(self):
        self.assertRefused(lambda: get_assemblies(io.StringIO(_CIF.replace("_pdbx_struct_assembly.", "_x.")), "mmcif"),
                           action="assemblies")
        self.assertRefused(lambda: get_structure_from_src(io.BytesIO(_BCIF), "binarycif", assembly_id="1"))

    def test_a_named_stream_is_named(self):
        path = self.write("broken.cif", "")
        with open(path) as stream:
            self.assertRefused(lambda: get_structure_from_src(stream, "mmcif"), source=path)


class NotTheFilesFaultTest(_Files):
    """Errors that do not say the file is invalid propagate unchanged: never a RefusalError."""

    def assertNotRefused(self, call, expected):
        with self.assertRaises(expected) as raised:
            call()
        self.assertNotIsInstance(raised.exception, RefusalError)
        return raised.exception

    def test_an_unknown_format(self):
        for call in (lambda: get_structure_from_src(io.StringIO(_CIF), "xyz"),
                     lambda: get_assemblies(io.StringIO(_CIF), "xyz")):
            error = self.assertNotRefused(call, ValueError)
            self.assertIn("Unknown structure format 'xyz'", str(error))

    def test_a_stream_of_the_wrong_mode(self):
        self.assertNotRefused(lambda: get_structure_from_src(io.BytesIO(_CIF.encode()), "mmcif"), TypeError)
        self.assertNotRefused(lambda: get_structure_from_src(io.StringIO(_CIF), "binarycif"), TypeError)
        self.assertNotRefused(lambda: get_structure_from_src(None, "mmcif"), TypeError)

    def test_a_files_content_passed_as_a_path(self):
        error = self.assertNotRefused(lambda: get_structure_from_src(_CIF, "mmcif"), TypeError)
        self.assertLess(len(str(error)), 200)

    def test_bytes(self):
        for source, structure_format in ((_BCIF, "binarycif"), (_CIF.encode(), "mmcif"),
                                         (str(_CIF_PATH).encode(), "mmcif"), (bytearray(_BCIF), "binarycif")):
            with self.subTest(f"{type(source).__name__} for {structure_format}"):
                error = self.assertNotRefused(lambda: get_structure_from_src(source, structure_format), TypeError)
                self.assertLess(len(str(error)), 200)
                self.assertNotRefused(lambda: get_assemblies(source, structure_format), TypeError)

    def test_a_path_with_a_line_break_is_a_path(self):
        path = self.write("odd\nname.cif", _CIF)
        self.assertGreater(get_structure_from_src(path, "mmcif").array_length(), 0)

    def test_an_assembly_id_that_is_not_a_str(self):
        for assembly_id in (1, ["1"], b"1"):
            with self.subTest(repr(assembly_id)):
                self.assertNotRefused(
                    lambda: get_structure_from_src(io.StringIO(_CIF), "mmcif", assembly_id=assembly_id), TypeError)
        self.assertGreater(get_structure_from_src(io.StringIO(_CIF), "mmcif", assembly_id=np.str_("1"))
                           .array_length(), 0)

    def test_a_closed_stream(self):
        stream = io.StringIO(_CIF)
        stream.close()
        self.assertNotRefused(lambda: get_structure_from_src(stream, "mmcif"), ValueError)

    def test_a_path_that_cannot_be_opened(self):
        self.assertNotRefused(lambda: get_structure_from_src(str(self.folder / "missing.cif"), "mmcif"),
                              FileNotFoundError)
        self.assertNotRefused(lambda: get_assemblies(str(self.folder / "missing.cif.gz"), "mmcif"),
                              FileNotFoundError)
        self.assertNotRefused(lambda: get_structure_from_src(str(self.folder), "mmcif"), IsADirectoryError)
        path = self.write("unreadable.cif", _CIF)
        os.chmod(path, 0)
        try:
            if os.access(path, os.R_OK):
                self.skipTest("running as a user who can read any file")
            self.assertNotRefused(lambda: get_structure_from_src(path, "mmcif"), PermissionError)
        finally:
            os.chmod(path, 0o644)

    def test_out_of_memory_and_unexpected_biotite_errors(self):
        for error in (MemoryError(), TypeError("unexpected"), AttributeError("unexpected"), OSError(5, "I/O error")):
            with self.subTest(type(error).__name__):
                with mock.patch.object(structure_parser, "get_structure", side_effect=error):
                    self.assertNotRefused(lambda: get_structure_from_src(io.StringIO(_CIF), "mmcif"), type(error))

    def test_errors_under_biotites_lazy_parsing(self):
        # biotite parses a category when it is first used and wraps whatever that raises in a DeserializationError:
        # one over a MemoryError is the machine's failure, one over an unexpected type is not refused either
        with mock.patch.object(CIFCategory, "deserialize", side_effect=MemoryError()):
            self.assertNotRefused(lambda: get_structure_from_src(io.StringIO(_CIF), "mmcif"), MemoryError)
        with mock.patch.object(CIFCategory, "deserialize", side_effect=TypeError("unexpected")):
            error = self.assertNotRefused(lambda: get_structure_from_src(io.StringIO(_CIF), "mmcif"),
                                          DeserializationError)
        self.assertIsInstance(error.__context__, TypeError)

    def test_out_of_memory_in_foldmatchs_own_processing(self):
        class _Huge:
            @property
            def au_index(self):
                raise MemoryError()

        self.assertNotRefused(lambda: structure_parser._number_pdb_copies(_Huge()), MemoryError)

    def test_a_failure_in_foldmatchs_own_processing(self):
        # Numbering the copies of a PDB assembly, after biotite built it: here biotite gave no au_index
        with mock.patch.object(structure_parser, "get_pdb_assembly",
                               side_effect=lambda *a, **k: PDBFile.read(io.StringIO(_PDB)).get_structure(model=1)):
            error = self.assertNotRefused(
                lambda: get_structure_from_src(io.StringIO(_PDB), "pdb", assembly_id="1"), InternalError)
        self.assertIsNotNone(error.__cause__)

    def test_a_failure_in_foldmatchs_own_step_inside_biotites_call(self):
        # _IndexedPDBFile.get_structure runs inside biotite's get_assembly: a ValueError from FoldMatch's part of it is
        # FoldMatch's failure, not a malformed file, even though it is raised from a biotite call
        class _NoAnnotations:
            def array_length(self):
                return 3

            def set_annotation(self, *_):
                raise ValueError("annotation of the wrong length")

        with mock.patch.object(PDBFile, "get_structure", return_value=_NoAnnotations()):
            error = self.assertNotRefused(
                lambda: get_structure_from_src(io.StringIO(_PDB), "pdb", assembly_id="1"), InternalError)
        self.assertIsInstance(error.__cause__, ValueError)

    def test_a_refusal_raised_while_reading_is_not_wrapped_again(self):
        refusal = RefusalError("refused upstream")
        with mock.patch.object(structure_parser, "get_structure", side_effect=refusal):
            with self.assertRaises(RefusalError) as refused:
                get_structure_from_src(io.StringIO(_CIF), "mmcif")
        self.assertIs(refused.exception, refusal)


if __name__ == "__main__":
    unittest.main()
