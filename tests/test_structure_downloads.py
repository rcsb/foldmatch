"""A structure file at a URL is downloaded as the stream its format is read from.

data.stringio_from_url chose the stream by the URL's ending: only a URL ending in .bcif.gz gave a binary stream, so a
BinaryCIF file at any other URL (https://models.rcsb.org/1acb.bcif, an upper-case suffix, a query string) became text
and could never be read; and only a URL ending in .gz was gunzipped. It now chooses by the structure format (binary
for binarycif, UTF-8 text otherwise) and recognises gzip by its content. The stream is named by the URL, so that a
refused file is named in the message, and a gzip file that cannot be decompressed is refused at once, not downloaded
again five times.

The downloads are mocked (requests.get); the fixtures are 1acb (tests/resources/pdb) as mmCIF, PDB and BinaryCIF.
ESM3 is never loaded."""
import gzip
import inspect
import io
import unittest
from pathlib import Path
from unittest import mock

import requests
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile, get_structure, set_structure

from foldmatch.dataset import esm_assembly_from_structure, esm_prot_from_chain, esm_prot_from_structure, \
    resdiue_assembly_embedding_from_structure, residue_assembly_embedding_from_parquet, \
    residue_assembly_embedding_from_tensor_file, residue_embedding_from_structure
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.errors import RefusalError
from foldmatch.types.api_types import SrcLocation, StructureFormat
from foldmatch.utils import data
from foldmatch.utils.data import stringio_from_url
from foldmatch.utils.structure_parser import get_assemblies, get_structure_from_src

_CIF_PATH = Path(__file__).parent / "resources" / "pdb" / "1acb.cif"
_CIF = _CIF_PATH.read_bytes()


def _pdb() -> bytes:
    pdb_file = PDBFile()
    pdb_file.set_structure(get_structure(CIFFile.read(str(_CIF_PATH)), model=1))
    text = io.StringIO()
    pdb_file.write(text)
    return text.getvalue().encode()


def _bcif() -> bytes:
    bcif_file = BinaryCIFFile()
    set_structure(bcif_file, get_structure(CIFFile.read(str(_CIF_PATH)), model=1))
    content = io.BytesIO()
    bcif_file.write(content)
    return content.getvalue()


_PDB = _pdb()
_BCIF = _bcif()
_ATOMS = get_structure(CIFFile.read(str(_CIF_PATH)), model=1, use_author_fields=False).array_length()


def _response(content: bytes) -> mock.Mock:
    response = mock.Mock(content=content)
    response.raise_for_status.return_value = None
    return response


class _Downloads(unittest.TestCase):
    def setUp(self):
        # run_with_retries waits up to 16 minutes between attempts: a regression must fail, not sleep
        patcher = mock.patch.object(data.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def serve(self, content: bytes):
        """requests.get answers every URL with content."""
        patcher = mock.patch.object(data.requests, "get", return_value=_response(content))
        self.get = patcher.start()
        self.addCleanup(patcher.stop)


class StreamModeTest(_Downloads):
    def test_binarycif_at_any_url(self):
        for url in ("https://models.rcsb.org/1acb.bcif", "https://example.org/1ACB.BCIF", "https://example.org/1acb",
                    "https://example.org/1acb.bcif?download=1", "https://example.org/1acb.bcif.gz"):
            for content in (_BCIF, gzip.compress(_BCIF)):
                with self.subTest(url=url, gzip=content[:2] == b"\x1f\x8b"):
                    self.serve(content)
                    stream = stringio_from_url(url, StructureFormat.bciff)
                    self.assertIsInstance(stream, io.BytesIO)
                    self.assertEqual(get_structure_from_src(stream, "binarycif").array_length(), _ATOMS)

    def test_text_formats_at_any_url(self):
        for structure_format, content in ((StructureFormat.mmcif, _CIF), (StructureFormat.pdb, _PDB)):
            for url in ("https://example.org/1acb.cif", "https://example.org/1acb.cif.gz",
                        "https://example.org/1ACB.CIF.GZ", "https://example.org/1acb?format=gz"):
                for body in (content, gzip.compress(content)):
                    with self.subTest(format=structure_format.value, url=url, gzip=body[:2] == b"\x1f\x8b"):
                        self.serve(body)
                        stream = stringio_from_url(url, structure_format)
                        self.assertIsInstance(stream, io.TextIOBase)
                        self.assertEqual(get_structure_from_src(stream, structure_format).array_length(), _ATOMS)

    def test_the_format_as_its_value(self):
        self.serve(_BCIF)
        self.assertIsInstance(stringio_from_url("https://example.org/1acb.cif", "binarycif"), io.BytesIO)

    def test_without_a_format_binarycif_is_recognised_by_the_url(self):
        for url, binary in (("https://example.org/1acb.bcif", True), ("https://example.org/1ACB.BCIF.GZ", True),
                            ("https://example.org/1acb.bcif?x=1", True), ("https://example.org/1acb.cif", False),
                            ("https://example.org/1acb.cif.gz", False), ("https://example.org/bcif/1acb.cif", False)):
            with self.subTest(url=url):
                self.serve(_BCIF if binary else _CIF)
                self.assertEqual(isinstance(stringio_from_url(url), io.BytesIO), binary)

    def test_a_stream_is_read_more_than_once(self):
        # The assembly datasets list a download's assemblies, then read each of them from the same stream
        self.serve(gzip.compress(_CIF))
        stream = stringio_from_url("https://example.org/1acb.cif.gz", StructureFormat.mmcif)
        self.assertEqual(get_assemblies(stream, "mmcif"), ("1",))
        self.assertGreater(get_structure_from_src(stream, "mmcif", assembly_id="1").array_length(), 0)


class UnreadableDownloadsTest(_Downloads):
    def test_a_refusal_names_the_url(self):
        url = "https://example.org/broken.cif"
        self.serve(b"<html><body>Not a structure</body></html>\n")
        with self.assertRaises(RefusalError) as refused:
            get_structure_from_src(stringio_from_url(url, StructureFormat.mmcif), "mmcif")
        self.assertIn(f"from {url}: ", str(refused.exception))

    def test_an_empty_file_is_refused(self):
        url = "https://example.org/empty.cif"
        self.serve(b"")
        with self.assertRaises(RefusalError) as refused:
            get_structure_from_src(stringio_from_url(url, StructureFormat.mmcif), "mmcif")
        self.assertIn(f"from {url}: ", str(refused.exception))

    def test_text_that_is_not_utf8_is_refused(self):
        # as a local file is: decoded by the structure reader, so a decoding error is the file's
        url = "https://example.org/latin1.cif"
        self.serve(_CIF.replace(b"data_1ACB", b"data_1ACB\n# \xe9", 1))
        with self.assertRaises(RefusalError) as refused:
            get_structure_from_src(stringio_from_url(url, StructureFormat.mmcif), "mmcif")
        self.assertIsInstance(refused.exception.__cause__, UnicodeDecodeError)
        self.assertIn(url, str(refused.exception))

    def test_a_gzip_file_that_cannot_be_decompressed_is_refused_at_once(self):
        corrupt = bytearray(gzip.compress(_CIF))
        corrupt[30:46] = bytes(b ^ 0xFF for b in corrupt[30:46])
        bad_crc = bytearray(gzip.compress(_CIF))
        bad_crc[-8] ^= 0xFF
        bad_method = bytearray(gzip.compress(_CIF))
        bad_method[2] = 7
        # A bad CRC or compression method is a BadGzipFile, an OSError, which was downloaded again 5 times
        for case, body in (("corrupt", bytes(corrupt)), ("truncated", gzip.compress(_CIF)[:2000]),
                           ("bad crc", bytes(bad_crc)), ("bad method", bytes(bad_method))):
            with self.subTest(case):
                self.sleep.reset_mock()
                self.serve(body)
                url = f"https://example.org/{case.replace(' ', '_')}.cif.gz"
                with self.assertRaises(RefusalError) as refused:
                    stringio_from_url(url, StructureFormat.mmcif)
                self.assertIn(url, str(refused.exception))
                self.assertEqual(self.get.call_count, 1)
                self.sleep.assert_not_called()

    def test_network_errors_are_retried(self):
        responses = [requests.ConnectionError("reset"), requests.ConnectionError("reset"), _response(_BCIF)]
        with mock.patch.object(data.requests, "get", side_effect=responses) as get:
            stream = stringio_from_url("https://models.rcsb.org/1acb.bcif", StructureFormat.bciff)
        self.assertEqual(get.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)
        self.assertEqual(get_structure_from_src(stream, "binarycif").array_length(), _ATOMS)


_DATASETS = (esm_assembly_from_structure, esm_prot_from_chain, esm_prot_from_structure,
             resdiue_assembly_embedding_from_structure, residue_assembly_embedding_from_parquet,
             residue_assembly_embedding_from_tensor_file, residue_embedding_from_structure)


class DatasetTest(_Downloads):
    def test_the_chains_of_a_remote_binarycif_file(self):
        """fm-embedding from-structures / fm-search build structures with --structure-format binarycif on a list of
        URLs: every row failed before. The URL does not end in .bcif, so only the format can choose the stream."""
        self.serve(_BCIF)
        url = "https://example.org/download/1acb?format=bcif"
        names = [name for _, name in EsmProtFromStructure(
            src_stream=[("1acb", url, "1acb")], src_location=SrcLocation.stream,
            structure_format=StructureFormat.bciff)]
        self.assertEqual(len(names), 2)
        self.assertEqual(self.get.call_args.args[0], url)

    def test_every_dataset_passes_its_structure_format(self):
        for module in _DATASETS:
            with self.subTest(module.__name__):
                calls = [line.strip() for line in inspect.getsource(module).splitlines() if "stringio_from_url(" in line
                         and "import" not in line]
                self.assertTrue(calls)
                for call in calls:
                    self.assertIn("stringio_from_url(src_structure, self.structure_format)", call)


if __name__ == "__main__":
    unittest.main()
