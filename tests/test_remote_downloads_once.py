"""A structure file at a URL is downloaded once per entry, not once per chain or per assembly.

The datasets passed stringio_from_url(...) to StructureProvider.get_structure as an argument, so the file was downloaded
before the provider looked at what it held: EsmProtFromChain downloaded an entry once per chain (and parsed it once),
ResidueAssemblyEmbeddingFromTensorFile and ResidueAssemblyEmbeddingFromParquet once per assembly, and
ResidueAssemblyDatasetFromStructure once to list the assemblies of an entry and again for each of them. The datasets now
give the provider the download lazily (a callable): it is made only when the provider holds neither the structure nor
the file, and the file is kept for the other assemblies of the entry.

The downloads are mocked (requests.get, counted per URL); the fixture is 1acb with a second assembly, chain B alone.
ESM3 is never loaded."""
import io
import pickle
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.dataset.resdiue_assembly_embedding_from_structure import ResidueAssemblyDatasetFromStructure
from foldmatch.dataset.residue_assembly_embedding_from_parquet import ResidueAssemblyEmbeddingFromParquet
from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.types.api_types import SrcLocation
from foldmatch.utils import data
from foldmatch.utils.structure_provider import StructureProvider
from test_assembly_structures import _EMBEDDINGS, _chains, _embedding, _two_assemblies

_URL = "https://example.org/1acb.cif"
_OTHER_URL = "https://example.org/2acb.cif"


class _Named(StructureProvider):
    """A provider with state of its own, which its __init__ needs."""

    def __init__(self, name):
        super().__init__()
        self.name = name


class _Downloads(unittest.TestCase):
    """requests.get answers every URL with the two-assembly 1acb file; self.downloads counts the requests per URL."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cif = _two_assemblies(self.tmp)
        content = self.cif.read_bytes()
        self.downloads = Counter()

        def get(url, *args, **kwargs):
            self.downloads[url] += 1
            response = mock.Mock(content=content)
            response.raise_for_status.return_value = None
            return response

        # run_with_retries waits up to 16 minutes between attempts: a regression must fail, not sleep
        self.enterContext(mock.patch.object(data.time, "sleep"))
        self.enterContext(mock.patch.object(data.requests, "get", side_effect=get))
        self.expected = {"1acb-1": torch.cat([_embedding("A"), _embedding("B")]), "1acb-2": _embedding("B")}

    def items(self, dataset):
        return {name: value for value, name in (dataset[i] for i in range(len(dataset)))}

    def assert_assemblies(self, dataset):
        items = self.items(dataset)
        self.assertEqual(sorted(items), sorted(self.expected))
        for name, tensor in self.expected.items():
            with self.subTest(item=name):
                self.assertTrue(torch.equal(items[name].float(), tensor.float()))


class DatasetDownloadsTest(_Downloads):

    def test_chains_of_an_entry(self):
        """EsmProtFromChain: one download per entry, whatever its number of chains (two here); a chain of another
        entry downloads that entry."""
        dataset = EsmProtFromChain(
            src_stream=[("1acb", _URL, "A", "1acb.A"), ("1acb", _URL, "B", "1acb.B"),
                        ("2acb", _OTHER_URL, "B", "2acb.B")],
            src_location=SrcLocation.stream)
        local = EsmProtFromChain(src_stream=[("1acb", str(self.cif), "A", "1acb.A"),
                                             ("1acb", str(self.cif), "B", "1acb.B")],
                                 src_location=SrcLocation.stream)
        proteins = self.items(dataset)
        self.assertEqual(self.downloads, {_URL: 1, _OTHER_URL: 1})
        for name, protein in self.items(local).items():
            with self.subTest(chain=name):
                self.assertEqual(proteins[name].sequence, protein.sequence)
        self.assertEqual(proteins["2acb.B"].sequence, proteins["1acb.B"].sequence)

    def test_assemblies_from_tensor_files(self):
        self.assert_assemblies(ResidueAssemblyEmbeddingFromTensorFile(
            src_stream=[("1acb", _URL, "1", "1acb-1"), ("1acb", _URL, "2", "1acb-2")],
            res_embedding_location=_EMBEDDINGS, src_location=SrcLocation.stream))
        self.assertEqual(self.downloads, {_URL: 1})

    def test_assemblies_from_parquet(self):
        ids = ["1acb.A", "1acb.B"]
        kind = pa.list_(pa.list_(pa.float32(), _embedding("A").shape[1]))
        parquet = self.tmp / "residues.parquet"
        pq.write_table(pa.table({"id": ids, "embedding": [_embedding(i[-1]).numpy().tolist() for i in ids]},
                                schema=pa.schema([("id", pa.string()), ("embedding", kind)])), parquet)
        self.assert_assemblies(ResidueAssemblyEmbeddingFromParquet(
            src_stream=[("1acb", _URL, "1", "1acb-1"), ("1acb", _URL, "2", "1acb-2")],
            parquet_path=parquet, src_location=SrcLocation.stream))
        self.assertEqual(self.downloads, {_URL: 1})

    def test_assemblies_of_a_structure(self):
        """ResidueAssemblyDatasetFromStructure lists the assemblies of an entry and reads each from one download."""
        self.assert_assemblies(ResidueAssemblyDatasetFromStructure(
            src_stream=[("1acb", _URL, "1acb")], res_embedding_location=_EMBEDDINGS,
            src_location=SrcLocation.stream))
        self.assertEqual(self.downloads, {_URL: 1})

    def test_assemblies_of_several_structures(self):
        """With more than one entry, only the last file downloaded to list assemblies is still held when the items
        are read: an entry is downloaded at most twice (once to list its assemblies, once to read them), never once
        per assembly."""
        embeddings = self.tmp / "embeddings"
        embeddings.mkdir()
        for name in ("1acb", "2acb"):
            for chain in "AB":
                shutil.copy(_EMBEDDINGS / f"1acb.{chain}.pt", embeddings / f"{name}.{chain}.pt")
        dataset = ResidueAssemblyDatasetFromStructure(
            src_stream=[("1acb", _URL, "1acb"), ("2acb", _OTHER_URL, "2acb")], res_embedding_location=embeddings,
            src_location=SrcLocation.stream)
        self.assertEqual(self.downloads, {_URL: 1, _OTHER_URL: 1})
        self.assertEqual(sorted(self.items(dataset)), ["1acb-1", "1acb-2", "2acb-1", "2acb-2"])
        self.assertEqual(self.downloads, {_URL: 2, _OTHER_URL: 2})

    def test_a_dataset_that_holds_a_download_can_be_pickled(self):
        """A DataLoader worker gets its own copy of the dataset: what the provider holds (a stream, which cannot be
        pickled) is dropped from the copy, which downloads again what it reads."""
        dataset = ResidueAssemblyEmbeddingFromTensorFile(
            src_stream=[("1acb", _URL, "1", "1acb-1"), ("1acb", _URL, "2", "1acb-2")],
            res_embedding_location=_EMBEDDINGS, src_location=SrcLocation.stream)
        dataset[0]
        copy = pickle.loads(pickle.dumps(dataset))
        self.assert_assemblies(copy)
        self.assertEqual(self.downloads, {_URL: 2})


class LazySourceTest(unittest.TestCase):

    def setUp(self):
        self.cif = _two_assemblies(Path(self.enterContext(tempfile.TemporaryDirectory())))

    def opener(self):
        return mock.Mock(side_effect=lambda: io.StringIO(self.cif.read_text()))

    def test_opened_only_when_not_held(self):
        provider = StructureProvider()
        source = self.opener()
        for assembly_id, chains in (("1", ["A", "B", "C", "D"]), ("1", ["A", "B", "C", "D"]), ("2", ["B"])):
            with self.subTest(assembly_id=assembly_id):
                self.assertEqual(_chains(provider.get_structure("1acb", source, assembly_id=assembly_id)), chains)
        source.assert_called_once_with()

    def test_another_entry_or_format_is_opened(self):
        provider = StructureProvider()
        first, second, third = self.opener(), self.opener(), self.opener()
        provider.get_structure("1acb", first)
        provider.get_structure("2acb", second)
        provider.get_structure("2acb", third, "mmcif", assembly_id="2")  # the same entry and format: held
        provider.get_structure("1acb", first, assembly_id="2")
        self.assertEqual((first.call_count, second.call_count, third.call_count), (2, 1, 0))

    def test_a_failed_download_is_not_held(self):
        provider = StructureProvider()
        source = mock.Mock(side_effect=[IOError("unreachable"), io.StringIO(self.cif.read_text())])
        with self.assertRaises(IOError):
            provider.get_structure("1acb", source)
        self.assertEqual(_chains(provider.get_structure("1acb", source)), ["A", "B", "C", "D"])
        self.assertEqual(source.call_count, 2)

    def test_a_pickled_provider_drops_what_it_holds(self):
        """Under every pickle protocol, and for a subclass with its own state: the copy holds nothing (a stream cannot
        be pickled) and reads again what it is asked for."""
        for provider in (StructureProvider(), _Named("x")):
            source = self.opener()
            provider.get_structure("1acb", source, assembly_id="2")
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(provider=type(provider).__name__, protocol=protocol):
                    copy = pickle.loads(pickle.dumps(provider, protocol=protocol))
                    self.assertEqual(getattr(copy, "name", None), getattr(provider, "name", None))
                    other = self.opener()
                    self.assertEqual(_chains(copy.get_structure("1acb", other, assembly_id="2")), ["B"])
                    other.assert_called_once_with()
            source.assert_called_once_with()

    def test_paths_and_streams_are_read_as_given(self):
        provider = StructureProvider()
        self.assertEqual(provider.get_source("1acb", str(self.cif)), str(self.cif))
        with open(self.cif) as stream:
            self.assertIs(provider.get_source("1acb", stream), stream)


if __name__ == "__main__":
    unittest.main()
