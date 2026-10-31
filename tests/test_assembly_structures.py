"""P9.A: each assembly of an entry is read as itself.

StructureProvider kept the last structure read under the entry name alone, so assemblies of an entry read one after
another all got the first one's structure; EsmAssemblyFromStructure also passed the previous assembly's structure as
the source of the next, and read a downloaded file (a stream) to its end while listing its assemblies, before reading
any of them. The fixture is 1acb with a second assembly, chain B alone (the first holds A, B, C and D; A and B are
the protein chains, C and D water), and optionally a third, the water alone."""
import inspect
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import biotite.structure as struc
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from biotite.structure.io.pdbx import CIFCategory, CIFFile

from foldmatch.dataset import esm_assembly_from_structure
from foldmatch.dataset.esm_assembly_from_structure import EsmAssemblyFromStructure
from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.resdiue_assembly_embedding_from_structure import ResidueAssemblyDatasetFromStructure
from foldmatch.dataset.residue_assembly_embedding_from_parquet import ResidueAssemblyEmbeddingFromParquet
from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.dataset.residue_embedding_from_structure import ResidueEmbeddingFromStructure
from foldmatch.types.api_types import SrcLocation
from foldmatch.utils import structure_provider
from foldmatch.utils.structure_parser import get_assemblies, get_structure_from_src
from foldmatch.utils.structure_provider import StructureProvider

_RESOURCES = Path(__file__).parent / "resources"
_EMBEDDINGS = _RESOURCES / "embeddings"
_ALL = ["A", "B", "C", "D"]


def _two_assemblies(folder: Path, water: bool = False) -> Path:
    asym = ["A,B,C,D", "B"] + (["C,D"] if water else [])
    ids = [str(k + 1) for k in range(len(asym))]
    cif = CIFFile.read(str(_RESOURCES / "pdb" / "1acb.cif"))
    cif.block["pdbx_struct_assembly"] = CIFCategory({
        "id": ids, "details": ["author_and_software_defined_assembly"] * len(ids), "method_details": ["?"] * len(ids),
        "oligomeric_details": ["?"] * len(ids), "oligomeric_count": ["?"] * len(ids)})
    cif.block["pdbx_struct_assembly_gen"] = CIFCategory({
        "assembly_id": ids, "oper_expression": ["1"] * len(ids), "asym_id_list": asym})
    path = folder / "1acb.cif"
    cif.write(str(path))
    return path


def _chains(structure) -> list:
    return [str(chain) for chain in struc.get_chains(structure)]


def _embedding(chain: str) -> torch.Tensor:
    return torch.load(_EMBEDDINGS / f"1acb.{chain}.pt", map_location="cpu")


class StructureProviderTest(unittest.TestCase):

    def setUp(self):
        self.cif = _two_assemblies(Path(self.enterContext(tempfile.TemporaryDirectory())))

    def test_assemblies_of_one_entry(self):
        provider = StructureProvider()
        for assembly_id, chains in (("1", _ALL), ("2", ["B"]), ("1", _ALL)):
            with self.subTest(assembly_id=assembly_id):
                self.assertEqual(_chains(provider.get_structure("1acb", str(self.cif), assembly_id=assembly_id)),
                                 chains)
                self.assertEqual(_chains(get_structure_from_src(str(self.cif), assembly_id=assembly_id)), chains)

    def test_the_same_request_reads_the_file_once(self):
        provider = StructureProvider()
        with mock.patch.object(structure_provider, "get_structure_from_src", wraps=get_structure_from_src) as read:
            provider.get_structure("1acb", str(self.cif), assembly_id="1")
            chain = provider.get_structure("1acb", str(self.cif), chain_id="B", assembly_id="1")
            self.assertEqual(read.call_count, 1)
            self.assertEqual(_chains(chain), ["B"])
            provider.get_structure("1acb", str(self.cif), assembly_id="2")
            provider.get_structure("1acb", str(self.cif), assembly_id="1")
            self.assertEqual(read.call_count, 3)

    def test_a_stream_is_read_from_its_start(self):
        """A downloaded file is a stream: its assemblies are listed, then each assembly is read from it (and the
        listing works on a stream already read)."""
        stream = io.StringIO(self.cif.read_text())
        self.assertEqual(_chains(get_structure_from_src(stream)), _ALL)
        assemblies = get_assemblies(stream)
        self.assertEqual([str(a) for a in assemblies], ["1", "2"])
        provider = StructureProvider()
        self.assertEqual([_chains(provider.get_structure("1acb", stream, assembly_id=a)) for a in assemblies],
                         [_ALL, ["B"]])

    def test_each_dataset_has_its_own_provider(self):
        """The default was one StructureProvider per dataset class, built when the class was defined and shared by
        every instance of that class."""
        for cls in (EsmAssemblyFromStructure, ResidueAssemblyDatasetFromStructure,
                    ResidueAssemblyEmbeddingFromTensorFile, ResidueAssemblyEmbeddingFromParquet, EsmProtFromStructure,
                    ResidueEmbeddingFromStructure, EsmProtFromChain):
            with self.subTest(dataset=cls.__name__):
                self.assertIsNone(inspect.signature(cls.__init__).parameters["structure_provider"].default)


class AssemblyDatasetTest(unittest.TestCase):
    """Each assembly dataset on the two-assembly entry: assembly 2 holds chain B alone, read after assembly 1."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cif = _two_assemblies(self.tmp)
        self.expected = {"1acb-1": torch.cat([_embedding("A"), _embedding("B")]), "1acb-2": _embedding("B")}

    def assert_assemblies(self, dataset):
        items = {name: tensor for tensor, name in (dataset[i] for i in range(len(dataset)))}
        self.assertEqual(sorted(items), sorted(self.expected))
        for name, tensor in self.expected.items():
            with self.subTest(item=name):
                self.assertTrue(torch.equal(items[name].float(), tensor.float()), (name, tuple(items[name].shape)))

    def test_from_tensor_files(self):
        self.assert_assemblies(ResidueAssemblyEmbeddingFromTensorFile(
            src_stream=[("1acb", str(self.cif), "1", "1acb-1"), ("1acb", str(self.cif), "2", "1acb-2")],
            res_embedding_location=_EMBEDDINGS, src_location=SrcLocation.stream))

    def test_every_assembly_of_a_structure(self):
        self.assert_assemblies(ResidueAssemblyDatasetFromStructure(
            src_stream=[("1acb", str(self.cif), "1acb")], res_embedding_location=_EMBEDDINGS,
            src_location=SrcLocation.stream))

    def test_from_parquet(self):
        ids = ["1acb.A", "1acb.B"]
        embeddings = [_embedding(i[-1]).numpy().tolist() for i in ids]  # one list of D values per residue
        kind = pa.list_(pa.list_(pa.float32(), _embedding("A").shape[1]))
        parquet = self.tmp / "residues.parquet"
        pq.write_table(pa.table({"id": ids, "embedding": embeddings},
                                schema=pa.schema([("id", pa.string()), ("embedding", kind)])), parquet)
        self.assert_assemblies(ResidueAssemblyEmbeddingFromParquet(
            src_stream=[("1acb", str(self.cif), "1", "1acb-1"), ("1acb", str(self.cif), "2", "1acb-2")],
            parquet_path=parquet, src_location=SrcLocation.stream))

    def test_esm_proteins_of_every_assembly(self):
        """The ESM3 inputs of each assembly (fm-search build structures --granularity assembly, fm-embedding
        from-structures assembly), from a file and from a download."""
        url = "https://example.org/1acb.cif"
        for source in (str(self.cif), url):
            with self.subTest(source=source), mock.patch.object(
                    esm_assembly_from_structure, "stringio_from_url",
                    side_effect=lambda u: io.StringIO(self.cif.read_text())) as download:
                items = dict((name, proteins) for proteins, name in EsmAssemblyFromStructure(
                    src_stream=[("1acb", source, "1acb")], src_location=SrcLocation.stream))
                self.assertEqual(download.call_count, int(source == url))
                self.assertEqual(sorted(items), ["1acb-1", "1acb-2"])
                self.assertEqual(len(items["1acb-1"]), 2)
                self.assertEqual([p.sequence for p in items["1acb-2"]], [items["1acb-1"][1].sequence])
                self.assertNotEqual(items["1acb-1"][0].sequence, items["1acb-1"][1].sequence)

    def test_an_assembly_without_protein_chains_is_skipped(self):
        """The water-only assembly: skipped with a warning (before, it got assembly 1's chains; a first assembly
        without protein chains stopped the build)."""
        folder = self.tmp / "water"
        folder.mkdir()
        cif = _two_assemblies(folder, water=True)
        with self.assertLogs(esm_assembly_from_structure.logger, level="WARNING") as logs:
            items = [name for _, name in EsmAssemblyFromStructure(src_stream=[("1acb", str(cif), "1acb")],
                                                                  src_location=SrcLocation.stream)]
        self.assertEqual(items, ["1acb-1", "1acb-2"])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("1acb-3 has no protein chain", logs.output[0])


if __name__ == "__main__":
    unittest.main()
