"""P9.A: each assembly of an entry, and each copy of a chain in an assembly, is read as itself.

StructureProvider kept the last structure read under the entry name alone, so assemblies of an entry read one after
another all got the first one's structure; EsmAssemblyFromStructure also passed the previous assembly's structure as
the source of the next, and read a downloaded file (a stream) to its end while listing its assemblies, before reading
any of them. The fixture is 1acb with a second assembly, chain B alone (the first holds A, B, C and D; A and B are
the protein chains, C and D water), and optionally a third, the water alone.

EsmAssemblyFromStructure also read each protein chain of an assembly from the first piece of the atoms with its chain
ID, which the copies of a chain made by the assembly's operators share, so every copy got the first copy's atoms
(P9.A2). That fixture is 1acb with one assembly of two copies of A, B, C and D, made by the identity and by a 2-fold
about z then 40 Å along x, in mmCIF and in PDB format (author chain IDs E and I, the operators in REMARK 350)."""
import inspect
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import biotite.structure as struc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFCategory, CIFFile, get_structure

from foldmatch.dataset import esm_assembly_from_structure
from foldmatch.dataset.esm_assembly_from_structure import EsmAssemblyFromStructure
from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.resdiue_assembly_embedding_from_structure import ResidueAssemblyDatasetFromStructure
from foldmatch.dataset.residue_assembly_embedding_from_parquet import ResidueAssemblyEmbeddingFromParquet
from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.dataset.residue_embedding_from_structure import ResidueEmbeddingFromStructure
from foldmatch.types.api_types import SrcLocation, StructureFormat
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


# The operators of the two-copy assembly, as a rotation and a translation: the identity, and a 2-fold about z then 40 Å
# along x.
_OPERATORS = ((np.eye(3), np.zeros(3)), (np.diag([-1.0, -1.0, 1.0]), np.array([40.0, 0.0, 0.0])))


def _two_copies(folder: Path) -> Path:
    """1acb with one assembly: each of _OPERATORS applied to A, B, C and D."""
    cif = CIFFile.read(str(_RESOURCES / "pdb" / "1acb.cif"))
    operators = {"id": ["1", "2"], "type": ["identity operation", "point symmetry operation"]}
    for i in range(3):
        for j in range(3):
            operators[f"matrix[{i + 1}][{j + 1}]"] = [str(rotation[i, j]) for rotation, _ in _OPERATORS]
        operators[f"vector[{i + 1}]"] = [str(translation[i]) for _, translation in _OPERATORS]
    cif.block["pdbx_struct_oper_list"] = CIFCategory(operators)
    cif.block["pdbx_struct_assembly_gen"] = CIFCategory(
        {"assembly_id": ["1"], "oper_expression": ["1,2"], "asym_id_list": ["A,B,C,D"]})
    path = folder / "1acb.cif"
    cif.write(str(path))
    return path


def _two_copies_pdb(folder: Path, renumber: bool = False) -> Path:
    """The same assembly in PDB format: 1acb's asymmetric unit with author fields (chains E and I, each with its
    waters), as biotite writes it, after REMARK 300 and 350 records applying _OPERATORS to E and I. With renumber, E's
    residues 1-146 are numbered 1001-1146, so its numbering decreases and chain_iter cuts it in two pieces, of 144 and
    97 residues, with one chain ID."""
    atoms = get_structure(CIFFile.read(str(_RESOURCES / "pdb" / "1acb.cif")), model=1, use_author_fields=True)
    if renumber:
        atoms.res_id[(atoms.chain_id == "E") & ~atoms.hetero & (atoms.res_id <= 146)] += 1000
    remarks = ["REMARK 300", "REMARK 300 BIOMOLECULE: 1", "REMARK 350", "REMARK 350 BIOMOLECULE: 1",
               "REMARK 350 APPLY THE FOLLOWING TO CHAINS: E, I"]
    for k, (rotation, translation) in enumerate(_OPERATORS):
        for i in range(3):
            row = "".join(f"{value:10.6f}" for value in rotation[i])
            remarks.append(f"REMARK 350   BIOMT{i + 1} {k + 1:3d}{row}{translation[i]:15.5f}")
    pdb = PDBFile()
    pdb.set_structure(atoms)
    text = io.StringIO()
    pdb.write(text)
    path = folder / "1acb.pdb"
    path.write_text("\n".join(remarks) + "\n" + text.getvalue())
    return path


def _asymmetric_unit(path: Path, structure_format: StructureFormat = StructureFormat.mmcif) -> dict:
    """The chain-level reader's ESMProteins of the asymmetric unit, by chain ID."""
    return {name.split(".")[-1]: protein for protein, name in EsmProtFromStructure(
        src_stream=[("1acb", str(path), "1acb")], src_location=SrcLocation.stream, structure_format=structure_format)}


def _copy(protein, operator: int) -> tuple:
    """A copy as the assembly should hold it: an asymmetric-unit chain, moved by one of _OPERATORS."""
    return protein.sequence, protein.coordinates, operator


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
            again = provider.get_structure("1acb", str(self.cif), assembly_id="1")
            self.assertEqual(read.call_count, 1)
            self.assertEqual(_chains(again), _ALL)
            provider.get_structure("1acb", str(self.cif), assembly_id="2")
            provider.get_structure("1acb", str(self.cif), assembly_id="1")
            self.assertEqual(read.call_count, 3)

    def test_chain_id_is_gone(self):
        """chain_id, which selected a chain's atoms, is gone: a call that passes it fails, by name or by position
        (assembly_id, which took its place, is keyword-only, so a chain ID cannot land in it)."""
        provider = StructureProvider()
        for call in (lambda: get_structure_from_src(str(self.cif), "mmcif", chain_id="B"),
                     lambda: provider.get_structure("1acb", str(self.cif), "mmcif", chain_id="B")):
            with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'chain_id'"):
                call()
        with self.assertRaisesRegex(TypeError, "takes from 1 to 2 positional arguments but 3 were given"):
            get_structure_from_src(str(self.cif), "mmcif", "B")
        with self.assertRaisesRegex(TypeError, "takes from 3 to 4 positional arguments but 5 were given"):
            provider.get_structure("1acb", str(self.cif), "mmcif", "B")

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


class AssemblyCopiesTest(unittest.TestCase):
    """EsmAssemblyFromStructure on the two-copy assembly: each copy is read from its own atoms, its chain's in the
    asymmetric unit moved by its operator (before, every copy got the first copy's); and the stored-embedding
    datasets, which repeat each chain's stored residue embeddings for each of its copies."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cif = _two_copies(self.tmp)
        self.chains = _asymmetric_unit(self.cif)

    def proteins(self, source: str, structure_format: StructureFormat = StructureFormat.mmcif,
                 min_res_n: int = 0) -> list:
        items = list(EsmAssemblyFromStructure(src_stream=[("1acb", source, "1acb")], src_location=SrcLocation.stream,
                                              structure_format=structure_format, min_res_n=min_res_n))
        self.assertEqual([name for _, name in items], ["1acb-1"])
        return items[0][0]

    def assert_copies(self, proteins, copies):
        """The ESMProteins are these _copy()s, in order: their atom37 coordinates (NaN where an atom is missing) are
        the operator applied here, and the identity's copies are bit for bit the asymmetric unit's."""
        self.assertEqual([protein.sequence for protein in proteins], [sequence for sequence, _, _ in copies])
        for k, (protein, (_, coordinates, operator)) in enumerate(zip(proteins, copies)):
            with self.subTest(item=k):
                if operator == 0:
                    self.assertTrue(np.array_equal(protein.coordinates.numpy(), coordinates.numpy(), equal_nan=True))
                rotation, translation = _OPERATORS[operator]
                np.testing.assert_allclose(protein.coordinates.numpy(),
                                           coordinates.numpy().astype(np.float64) @ rotation.T + translation,
                                           rtol=0, atol=1e-4)

    def test_copies_in_order(self):
        """A, B, A, B, with their chains' sequences (as before)."""
        a, b = self.chains["A"], self.chains["B"]
        self.assertEqual([protein.sequence for protein in self.proteins(str(self.cif))], [a.sequence, b.sequence] * 2)

    def test_first_copies_are_the_chains_of_the_asymmetric_unit(self):
        """The identity's copies are bit for bit the chain-level reader's ESMProteins (as before)."""
        for protein, chain in zip(self.proteins(str(self.cif)), "AB"):
            with self.subTest(chain=chain):
                self.assertEqual(protein.sequence, self.chains[chain].sequence)
                self.assertTrue(np.array_equal(protein.coordinates.numpy(), self.chains[chain].coordinates.numpy(),
                                               equal_nan=True))

    def test_each_copy_from_its_own_atoms(self):
        """From a file and from a download."""
        a, b = self.chains["A"], self.chains["B"]
        url = "https://example.org/1acb.cif"
        for source in (str(self.cif), url):
            with self.subTest(source=source), mock.patch.object(
                    esm_assembly_from_structure, "stringio_from_url",
                    side_effect=lambda u: io.StringIO(self.cif.read_text())):
                self.assert_copies(self.proteins(source), [_copy(a, 0), _copy(b, 0), _copy(a, 1), _copy(b, 1)])

    def test_min_res_n_between_the_chain_lengths(self):
        """min_res_n 64, above B's 63 residues and below A's 241: the two copies of A, in order."""
        a = self.chains["A"]
        self.assert_copies(self.proteins(str(self.cif), min_res_n=64), [_copy(a, 0), _copy(a, 1)])

    def test_pdb_format(self):
        """The assembly in PDB format, read with StructureFormat.pdb; and with E's numbering decreasing, which cuts E
        in two pieces with one chain ID: each copy of E is read whole, its pieces joined, as in the plain file (P9.A3;
        before, each piece was read as itself, and before P9.A2 both got the first piece)."""
        pdb = _two_copies_pdb(self.tmp)
        chains = _asymmetric_unit(pdb, StructureFormat.pdb)
        e, i = chains["E"], chains["I"]
        renumbered = self.tmp / "renumbered"
        renumbered.mkdir()
        for path in (pdb, _two_copies_pdb(renumbered, renumber=True)):
            with self.subTest(file=str(path.relative_to(self.tmp))):
                self.assert_copies(self.proteins(str(path), StructureFormat.pdb),
                                   [_copy(e, 0), _copy(i, 0), _copy(e, 1), _copy(i, 1)])

    def test_stored_embeddings_of_each_copy(self):
        """The stored-embedding datasets (inference assembly-embedding, complete-embedding) repeat each chain's stored
        residue embeddings for each of its copies, in the build's order: A, B, A, B (as before)."""
        ids = ["1acb.A", "1acb.B"]
        embeddings = [_embedding(i[-1]).numpy().tolist() for i in ids]  # one list of D values per residue
        kind = pa.list_(pa.list_(pa.float32(), _embedding("A").shape[1]))
        parquet = self.tmp / "residues.parquet"
        pq.write_table(pa.table({"id": ids, "embedding": embeddings},
                                schema=pa.schema([("id", pa.string()), ("embedding", kind)])), parquet)
        assembly = [("1acb", str(self.cif), "1", "1acb-1")]
        expected = torch.cat([_embedding("A"), _embedding("B")] * 2)
        for dataset in (
                ResidueAssemblyEmbeddingFromTensorFile(src_stream=assembly, res_embedding_location=_EMBEDDINGS,
                                                       src_location=SrcLocation.stream),
                ResidueAssemblyEmbeddingFromParquet(src_stream=assembly, parquet_path=parquet,
                                                    src_location=SrcLocation.stream),
                ResidueAssemblyDatasetFromStructure(src_stream=[("1acb", str(self.cif), "1acb")],
                                                    res_embedding_location=_EMBEDDINGS,
                                                    src_location=SrcLocation.stream)):
            with self.subTest(dataset=type(dataset).__name__):
                self.assertEqual(len(dataset), 1)
                tensor, name = dataset[0]
                self.assertEqual(name, "1acb-1")
                self.assertTrue(torch.equal(tensor.float(), expected.float()), tuple(tensor.shape))


if __name__ == "__main__":
    unittest.main()
