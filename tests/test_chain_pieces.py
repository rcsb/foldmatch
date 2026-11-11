"""P9.A3: a chain that one chain ID cuts into pieces is read whole, and once, on every path.

In a PDB file (and in an mmCIF or BinaryCIF file not laid out as the archive's) one chain ID can cover several of
biotite's chain_iter pieces: where the chain's residue numbers decrease (a tag numbered 998-999 before 1, an insert
numbered from 1001), and where records of other chains lie between its records (its free amino-acid HETATM ligands
listed after every chain, say). The database (EsmProtFromStructure, EsmProtFromChain) embedded the first piece of the
atoms with the chain's ID, once for each piece listed; the query (FoldMatch.residue_embedding_by_chain) embedded every
piece and kept the last; FoldMatch.residue_embedding and the assembly paths embedded the pieces apart. Every path now
reads the protein chains of structure_parser.protein_chain_iter: for each chain ID (in an assembly, each copy), the
amino-acid atoms of its pieces joined in file order, without the pieces whose amino acids are all HETATM records when
it has amino acids in ATOM records, kept if their polymer residues reach the threshold together.

The fixtures are 1acb with author fields (chain E, 241 residues numbered 1-245, and chain I, 63 numbered 8-70, each
with its waters), edited, and written as PDB text (with REMARK 350 for assemblies) or as mmCIF by biotite. ESM3 is
never loaded: the paths' ESMProteins are compared, and FoldMatch embeds through a stub."""
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
from biotite.structure.io.pdbx import CIFCategory, CIFFile, get_structure, set_structure
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain

from foldmatch import FoldMatch
from foldmatch.dataset.esm_assembly_from_structure import EsmAssemblyFromStructure
from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.dataset.esm_prot_from_structure import EsmProtFromStructure
from foldmatch.dataset.residue_assembly_embedding_from_parquet import ResidueAssemblyEmbeddingFromParquet
from foldmatch.dataset.residue_assembly_embedding_from_tensor_file import ResidueAssemblyEmbeddingFromTensorFile
from foldmatch.dataset.residue_embedding_from_structure import ResidueEmbeddingFromStructure
from foldmatch.search.embedding_computer import compute_from_structure_file
from foldmatch.types.api_types import Accelerator, Granularity, SrcLocation, StructureFormat
from foldmatch.utils import structure_parser
from foldmatch.utils.structure_parser import filter_residues, get_protein_chains, get_structure_from_src, \
    protein_chain_iter, rename_atom_attr

_RESOURCES = Path(__file__).parent / "resources"
_PDB, _MMCIF = StructureFormat.pdb, StructureFormat.mmcif

_ATOMS = get_structure(CIFFile.read(str(_RESOURCES / "pdb" / "1acb.cif")), model=1, use_author_fields=True)
_E = _ATOMS[(_ATOMS.chain_id == "E") & ~_ATOMS.hetero]
_I = _ATOMS[(_ATOMS.chain_id == "I") & ~_ATOMS.hetero]
_WATER = _ATOMS[_ATOMS.hetero]

# The operators of the assemblies, as a rotation and a translation: the identity, and a 2-fold about z then 40 Å
# along x.
_OPERATORS = ((np.eye(3), np.zeros(3)), (np.diag([-1.0, -1.0, 1.0]), np.array([40.0, 0.0, 0.0])))


def _residues(atoms) -> list:
    starts = list(struc.get_residue_starts(atoms)) + [len(atoms)]
    return [atoms[a:b] for a, b in zip(starts[:-1], starts[1:])]


def _renumbered(atoms, first: int, last: int, shift: int):
    """A copy of the atoms, with the residues numbered first to last numbered shift more."""
    atoms = atoms.copy()
    atoms.res_id[(atoms.res_id >= first) & (atoms.res_id <= last)] += shift
    return atoms


def _hetero(residues, chain_id: str, res_id: int = None, shift: float = 40.0):
    """A copy of the residues as HETATM records with this chain ID, moved away (and numbered res_id, res_id + 1,
    ...)."""
    copies = []
    for k, residue in enumerate(residues):
        residue = residue.copy()
        residue.chain_id[:] = chain_id
        residue.hetero[:] = True
        if res_id is not None:
            residue.res_id[:] = res_id + k
        residue.coord += shift
        copies.append(residue)
    return struc.concatenate(copies)


def _glu(k: int):
    return [residue for residue in _residues(_E) if residue.res_name[0] == "GLU"][k]


# The asymmetric units, by name: 1acb's E, I and waters, edited, each with all of E and I, as the plain one.
_FIXTURES = {
    "plain": lambda: _E + _I + _WATER,
    # E's first two residues, 1 and 2, numbered 998 and 999: a tag before the chain's numbering
    "tag": lambda: _renumbered(_E, 1, 2, 997) + _I + _WATER,
    # E's residues 50-80 numbered 1050-1080: an insert numbered above 1000, then back
    "insert": lambda: _renumbered(_E, 50, 80, 1000) + _I + _WATER,
    # E's records in two blocks, I's between them, the numbering rising throughout
    "blocks": lambda: _E[_E.res_id <= 100] + _I + _E[_E.res_id > 100] + _WATER,
    # free amino-acid HETATM groups with chain ID E listed after I: GLU 500, and GLU 600 and 601, consecutive
    "ligands": lambda: _E + _I + _hetero([_glu(0)], "E", 500) + _hetero([_glu(0), _glu(1)], "E", 600) + _WATER,
    # a free amino acid with chain ID I listed before E
    "ligand first": lambda: _hetero([_glu(0)], "I", 500) + _E + _I + _WATER,
}


def _write_pdb(atoms, path: Path, remarks=()) -> Path:
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(str(path))
    if remarks:
        path.write_text("".join(f"{line}\n" for line in remarks) + path.read_text())
    return path


def _remarks(*lists) -> list:
    """REMARK 300 and 350 records of an assembly 1 that applies, for each (chain IDs, operators) list, those of
    _OPERATORS to those chains."""
    remarks = ["REMARK 300", "REMARK 300 BIOMOLECULE: 1", "REMARK 350", "REMARK 350 BIOMOLECULE: 1"]
    for chains, operators in lists:
        remarks.append(f"REMARK 350 APPLY THE FOLLOWING TO CHAINS: {', '.join(chains)}")
        for k in operators:
            rotation, translation = _OPERATORS[k]
            for i in range(3):
                row = "".join(f"{value:10.6f}" for value in rotation[i])
                remarks.append(f"REMARK 350   BIOMT{i + 1} {k + 1:3d}{row}{translation[i]:15.5f}")
    return remarks


def _assembly(cif: CIFFile, asym_ids: str):
    """Assembly 1 of the file: each of _OPERATORS applied to the asym IDs."""
    operators = {"id": ["1", "2"], "type": ["identity operation", "point symmetry operation"]}
    for i in range(3):
        for j in range(3):
            operators[f"matrix[{i + 1}][{j + 1}]"] = [str(rotation[i, j]) for rotation, _ in _OPERATORS]
        operators[f"vector[{i + 1}]"] = [str(translation[i]) for _, translation in _OPERATORS]
    cif.block["pdbx_struct_assembly"] = CIFCategory({
        "id": ["1"], "details": ["author_and_software_defined_assembly"], "method_details": ["?"],
        "oligomeric_details": ["?"], "oligomeric_count": ["?"]})
    cif.block["pdbx_struct_oper_list"] = CIFCategory(operators)
    cif.block["pdbx_struct_assembly_gen"] = CIFCategory(
        {"assembly_id": ["1"], "oper_expression": ["1,2"], "asym_id_list": [asym_ids]})


def _write_cif(atoms, path: Path, asym_ids: str = None) -> Path:
    """The atoms as biotite writes them in mmCIF (an asym ID per chain ID), with _assembly(asym_ids)."""
    cif = CIFFile()
    set_structure(cif, atoms)
    if asym_ids is not None:
        _assembly(cif, asym_ids)
    cif.write(str(path))
    return path


def _protein(atoms) -> ESMProtein:
    """The ESMProtein of a chain's atoms, processed as every path processes a chain."""
    return ESMProtein.from_protein_chain(ProteinChain.from_atomarray(rename_atom_attr(filter_residues(atoms))))


class _Esm3:
    """Stands in for ESM3 in FoldMatch: keeps the ESMProteins it is given, and returns L + 2 rows (BOS and EOS
    included), as per_residue_embedding does."""

    def __init__(self):
        self.proteins = []

    def __call__(self, model, protein):
        self.proteins.append(protein)
        return torch.zeros(len(protein) + 2, 4)


class _Paths(unittest.TestCase):
    """The chain-level and assembly paths on a file, FoldMatch's ESM3 and aggregator stubbed."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.esm3 = _Esm3()
        self.enterContext(mock.patch("foldmatch.foldmatch._load_res_model", return_value="ESM3"))
        self.enterContext(mock.patch("foldmatch.foldmatch._load_model", return_value=lambda x: x.mean(dim=0)))
        self.enterContext(mock.patch("foldmatch.foldmatch.per_residue_embedding", self.esm3))

    def pdb(self, name: str, *lists) -> Path:
        """One of _FIXTURES as a PDB file, with an assembly of these _remarks lists if given."""
        return _write_pdb(_FIXTURES[name](), self.tmp / f"{name.replace(' ', '_')}.pdb",
                          _remarks(*lists) if lists else ())

    def build(self, path: Path, structure_format: StructureFormat, min_res_n: int = 0) -> list:
        """EsmProtFromStructure (fm-search build structures --granularity chain, fm-embedding from-structures):
        (name, ESMProtein) of each chain, in order."""
        return [(name, protein) for protein, name in EsmProtFromStructure(
            src_stream=[("x", str(path), "x")], src_location=SrcLocation.stream, structure_format=structure_format,
            min_res_n=min_res_n)]

    def chain(self, path: Path, structure_format: StructureFormat, chain_id: str) -> ESMProtein:
        """EsmProtFromChain (inference residue-embedding, structure-embedding, complete-embedding)."""
        protein, name = EsmProtFromChain(src_stream=[("x", str(path), chain_id, f"x.{chain_id}")],
                                         src_location=SrcLocation.stream, structure_format=structure_format)[0]
        self.assertEqual(name, f"x.{chain_id}")
        return protein

    def query(self, path: Path, structure_format: StructureFormat, min_res: int = 10, chain_id: str = None) -> dict:
        """FoldMatch.residue_embedding_by_chain (fm-search query structure --granularity chain): the ESMProtein of
        each chain embedded, by chain ID, in order; each embedded once."""
        self.esm3.proteins.clear()
        embeddings = FoldMatch(min_res=min_res).residue_embedding_by_chain(str(path), structure_format, chain_id)
        self.assertEqual(len(self.esm3.proteins), len(embeddings), "ESM3 passes")
        return {str(c): protein for c, protein in zip(embeddings, self.esm3.proteins)}

    def api(self, path: Path, structure_format: StructureFormat, min_res: int = 10, chain_id: str = None,
            assembly_id: str = None) -> list:
        """FoldMatch.residue_embedding: the ESMProteins embedded, in order; the rows returned are theirs, BOS and EOS
        included, once each."""
        self.esm3.proteins.clear()
        rows = FoldMatch(min_res=min_res).residue_embedding(str(path), structure_format, chain_id, assembly_id)
        self.assertEqual(rows.shape[0], sum(len(protein) + 2 for protein in self.esm3.proteins))
        return list(self.esm3.proteins)

    def assembly_build(self, path: Path, structure_format: StructureFormat, min_res_n: int = 10) -> list:
        """EsmAssemblyFromStructure (fm-search build structures --granularity assembly): assembly 1's ESMProteins."""
        items = list(EsmAssemblyFromStructure(src_stream=[("x", str(path), "x")], src_location=SrcLocation.stream,
                                              structure_format=structure_format, min_res_n=min_res_n))
        self.assertEqual([name for _, name in items], ["x-1"])
        return items[0][0]

    def assembly_query(self, path: Path, structure_format: StructureFormat, min_res: int = 10) -> list:
        """FoldMatch.residue_embedding_by_assembly (fm-search query structure --granularity assembly)."""
        self.esm3.proteins.clear()
        embeddings = FoldMatch(min_res=min_res).residue_embedding_by_assembly(str(path), structure_format, "1")
        self.assertEqual(embeddings["1"].shape[0], sum(len(protein) + 2 for protein in self.esm3.proteins))
        return list(self.esm3.proteins)

    def assert_same(self, protein, expected, msg=None):
        """The same residues: the sequence, and the atom37 coordinates bit for bit (NaN where an atom is missing)."""
        self.assertEqual(protein.sequence, expected.sequence, msg)
        self.assertTrue(np.array_equal(protein.coordinates.numpy(), expected.coordinates.numpy(), equal_nan=True),
                        msg)

    def assert_same_proteins(self, proteins: list, expected: list, msg: str):
        self.assertEqual(len(proteins), len(expected), msg)
        for k, (protein, other) in enumerate(zip(proteins, expected)):
            self.assert_same(protein, other, f"{msg}, chain {k}")

    def assert_paths(self, path: Path, structure_format: StructureFormat, expected: dict, min_res: int = 10):
        """Every chain-level path gives these chains, each once, in order, with these residues: the build, the
        chain lookup, the query and residue_embedding, of every chain and of each alone."""
        names = [f"x.{c}" for c in expected]
        built = self.build(path, structure_format, min_res)
        self.assertEqual([name for name, _ in built], names)
        for (name, protein), chain in zip(built, expected):
            self.assert_same(protein, expected[chain], f"build {name}")
            self.assert_same(self.chain(path, structure_format, chain), expected[chain], f"chain lookup {name}")
            query = self.query(path, structure_format, min_res, chain)
            self.assertEqual(list(query), [chain])
            self.assert_same(query[chain], expected[chain], f"query of {name} alone")
            api = self.api(path, structure_format, min_res, chain)
            self.assertEqual(len(api), 1)
            self.assert_same(api[0], expected[chain], f"residue_embedding of {name} alone")
        query = self.query(path, structure_format, min_res)
        self.assertEqual(list(query), list(expected))
        api = self.api(path, structure_format, min_res)
        self.assertEqual(len(api), len(expected))
        for (chain, protein), from_query, from_api in zip(expected.items(), query.values(), api):
            self.assert_same(from_query, protein, f"query {chain}")
            self.assert_same(from_api, protein, f"residue_embedding {chain}")


class ChainPiecesTest(_Paths):
    """The chain-level paths on asymmetric units in PDB format (and one biotite-written mmCIF)."""

    def setUp(self):
        super().setUp()
        plain = get_structure_from_src(str(self.pdb("plain")), _PDB)
        self.plain = {c: _protein(plain[plain.chain_id == c]) for c in ("E", "I")}

    def test_one_chain_per_chain_id_on_every_path(self):
        """E cut where its numbering drops (a tag first, an insert in the middle), cut by I's records, or followed
        by its free amino-acid ligands after I (or I's before E): E and I once each, in that order, with the
        residues of the plain file's E and I (the ligands left out), on every path, at thresholds 0 and 10."""
        for name in ("tag", "insert", "blocks", "ligands", "ligand first"):
            path = self.pdb(name)
            for min_res in (0, 10):
                with self.subTest(fixture=name, min_res=min_res):
                    self.assert_paths(path, _PDB, self.plain, min_res)

    def test_the_rule_on_the_atoms(self):
        """protein_chain_iter yields each chain's amino-acid atoms of ATOM records, in file order (the tag and the
        insert keep their numbers), and get_protein_chains lists E and I once."""
        for name in ("plain", "tag", "insert", "blocks", "ligands", "ligand first"):
            with self.subTest(fixture=name):
                structure = get_structure_from_src(str(self.pdb(name)), _PDB)
                chains = list(protein_chain_iter(structure))
                self.assertEqual(get_protein_chains(structure), ("E", "I"))
                self.assertEqual(get_protein_chains(structure, 64), ("E",))
                self.assertEqual([chain.chain_id[0] for chain in chains], ["E", "I"])
                for chain in chains:
                    atoms = structure[(structure.chain_id == chain.chain_id[0]) & ~structure.hetero]
                    self.assertTrue(np.array_equal(chain.res_id, atoms.res_id))
                    self.assertTrue(np.array_equal(chain.coord, atoms.coord))

    def test_one_chain_id_alone(self):
        """With a chain ID, protein_chain_iter reads that ID's pieces alone (the chain lookup's way) and yields the
        chains that the whole iteration yields with that ID: one in an asymmetric unit, one per copy in an assembly;
        none for an ID without a protein chain."""
        for name, lists in (("plain", ()), ("tag", ()), ("blocks", ()), ("ligands", ()), ("ligand first", ()),
                            ("insert", ((("E", "I"), (0, 1)),)), ("plain", ((("E", "I"), (0,)), (("E",), (1,))))):
            path = self.pdb(name, *lists)
            for assembly_id in (None, "1") if lists else (None,):
                structure = get_structure_from_src(str(path), _PDB, assembly_id=assembly_id)
                chains = list(protein_chain_iter(structure, 10))
                for c in ("E", "I", "Z"):
                    with self.subTest(fixture=name, assembly_id=assembly_id, chain=c):
                        alone = list(protein_chain_iter(structure, 10, c))
                        same = [chain for chain in chains if chain.chain_id[0] == c]
                        self.assertEqual(len(alone), len(same))
                        for chain, other in zip(alone, same):
                            self.assertTrue(np.array_equal(chain.res_id, other.res_id))
                            self.assertTrue(np.array_equal(chain.coord, other.coord))

    def test_a_chain_made_only_of_hetatm_amino_acids_is_kept(self):
        """Chain P: I's first 12 residues as HETATM records, after I. It is a chain of its own on every path, with
        those 12 residues."""
        path = _write_pdb(_E + _I + _hetero(_residues(_I)[:12], "P", shift=60.0) + _WATER, self.tmp / "hetero.pdb")
        structure = get_structure_from_src(str(path), _PDB)
        expected = dict(self.plain, P=_protein(structure[structure.chain_id == "P"]))
        self.assertEqual(len(expected["P"]), 12)
        self.assertEqual(expected["P"].sequence, self.plain["I"].sequence[:12])
        self.assert_paths(path, _PDB, expected)

    def test_a_residue_of_a_hetatm_record_inside_a_chain(self):
        """E's methionines as selenomethionines (HETATM MSE) inside its ATOM records: kept by every path, read as X
        (residue_embedding dropped them, as it did not rename HETATM records to ATOM)."""
        atoms = _E.copy()
        met = atoms.res_name == "MET"
        atoms.res_name[met] = "MSE"
        atoms.hetero[met] = True
        path = _write_pdb(atoms + _I + _WATER, self.tmp / "mse.pdb")
        structure = get_structure_from_src(str(path), _PDB)
        expected = dict(self.plain, E=_protein(structure[structure.chain_id == "E"]))
        self.assertEqual(len(expected["E"]), len(self.plain["E"]))
        self.assertEqual(expected["E"].sequence.count("X"), len(_residues(_E[_E.res_name == "MET"])))
        self.assert_paths(path, _PDB, expected)

    def test_pieces_under_the_threshold_that_reach_it_together(self):
        """Chain S as two pieces of 6 residues (I's residues 8-13, then 14-19 numbered 1-6): a chain of 12 residues,
        kept at min_res_n and min_res 10 (and 12), not at 13."""
        residues = _residues(_I)
        second = struc.concatenate(residues[6:12]).copy()
        second.res_id = second.res_id - second.res_id[0] + 1
        pieces = [struc.concatenate(residues[:6]).copy(), second]
        for piece in pieces:
            piece.chain_id[:] = "S"
            piece.coord += 80.0
        path = _write_pdb(_E + _I + pieces[0] + pieces[1] + _WATER, self.tmp / "small.pdb")
        structure = get_structure_from_src(str(path), _PDB)
        expected = dict(self.plain, S=_protein(structure[structure.chain_id == "S"]))
        self.assertEqual(expected["S"].sequence, self.plain["I"].sequence[:12])
        self.assert_paths(path, _PDB, expected)
        self.assertEqual(get_protein_chains(structure, 12), ("E", "I", "S"))
        self.assertEqual(get_protein_chains(structure, 13), ("E", "I"))
        self.assertEqual([name for name, _ in self.build(path, _PDB, 13)], ["x.E", "x.I"])
        self.assertEqual(list(self.query(path, _PDB, 13)), ["E", "I"])

    def test_free_amino_acids_left_out_are_not_counted(self):
        """Chain I cut to its first 9 residues, then E, then two consecutive free GLU of chain I (HETATM 600 and 601,
        a piece of their own that filter_polymer counts as 2 polymer residues): left out, they do not count towards
        the threshold, so I is a chain of 9 residues on every path at 9 (before, the build read it with the two GLU),
        and not a chain at 10."""
        short = struc.concatenate(_residues(_I)[:9]).copy()
        path = _write_pdb(short + _E + _hetero([_glu(0), _glu(1)], "I", 600) + _WATER, self.tmp / "short.pdb")
        structure = get_structure_from_src(str(path), _PDB)
        expected = {"I": _protein(structure[(structure.chain_id == "I") & ~structure.hetero]), "E": self.plain["E"]}
        self.assertEqual(len(expected["I"]), 9)
        self.assert_paths(path, _PDB, expected, min_res=9)
        self.assertEqual(get_protein_chains(structure, 10), ("E",))
        self.assertEqual([name for name, _ in self.build(path, _PDB, 10)], ["x.E"])
        self.assertEqual(list(self.query(path, _PDB, 10)), ["E"])

    def test_a_lone_amino_acid_ligand_is_not_a_chain(self):
        """A free GLU with an asym ID of its own (G) in mmCIF, as the archive gives free amino acids: no chain at
        threshold 0 (it has no polymer residue), so the build and the query take A and B only (the query embedded it
        before)."""
        label = get_structure(CIFFile.read(str(_RESOURCES / "pdb" / "1acb.cif")), model=1, use_author_fields=False)
        glu = _hetero([_glu(0)], "G")
        path = _write_cif(label[np.isin(label.chain_id, ["A", "B"])] + glu + label[np.isin(label.chain_id, ["C", "D"])],
                          self.tmp / "ligand.cif")
        structure = get_structure_from_src(str(path), _MMCIF)
        self.assertEqual(sorted(set(structure.chain_id.tolist())), ["A", "B", "C", "D", "G"])
        self.assertEqual(get_protein_chains(structure), ("A", "B"))
        expected = {c: _protein(structure[structure.chain_id == c]) for c in ("A", "B")}
        self.assert_paths(path, _MMCIF, expected, min_res=0)


class AssemblyPiecesTest(_Paths):
    """The assembly paths: the build (EsmAssemblyFromStructure) and the query (residue_embedding_by_assembly) read
    each copy of a chain as one chain, its pieces joined, and take the same residues."""

    def setUp(self):
        super().setUp()
        plain = get_structure_from_src(str(self.pdb("plain")), _PDB)
        self.plain = {c: _protein(plain[plain.chain_id == c]) for c in ("E", "I")}

    def assert_copies(self, path: Path, structure_format: StructureFormat, chains: list):
        """The build's chains have these chains' sequences, in order, the identity's copies their residues bit for
        bit; and the query and residue_embedding of the assembly read the same residues."""
        built = self.assembly_build(path, structure_format)
        self.assertEqual([protein.sequence for protein in built], [self.plain[c].sequence for c in chains])
        for protein, chain in zip(built, chains[:len(set(chains))]):
            self.assert_same(protein, self.plain[chain], f"first copy of {chain}")
        self.assert_same_proteins(self.assembly_query(path, structure_format), built, "query")
        self.assert_same_proteins(self.api(path, structure_format, assembly_id="1"), built, "residue_embedding")

    def test_one_chain_per_copy(self):
        """E with an insert numbered 1050-1080, two copies of E and I: in PDB format (REMARK 350, two BIOMT
        operators) and in mmCIF as biotite writes it (asym IDs E and I): E, I, E, I, each E whole (before, the build
        and the query read E's two pieces apart in each copy)."""
        pdb = self.pdb("insert", (("E", "I"), (0, 1)))
        cif = _write_cif(_FIXTURES["insert"](), self.tmp / "insert.cif", "E,I")
        for path, structure_format in ((pdb, _PDB), (cif, _MMCIF)):
            with self.subTest(file=path.name):
                self.assertEqual(get_protein_chains(get_structure_from_src(str(path), structure_format,
                                                                           assembly_id="1")), ("E", "I", "E", "I"))
                self.assert_copies(path, structure_format, ["E", "I", "E", "I"])

    def test_a_chain_in_two_remark_350_lists(self):
        """REMARK 350 applies the identity to E and I, then the 2-fold to E alone. biotite numbers the copies of each
        list from 0, so the two copies of E share its sym_id; the reader numbers them apart: two chains, E, I, E (as
        before)."""
        path = self.pdb("plain", (("E", "I"), (0,)), (("E",), (1,)))
        self.assertEqual(get_protein_chains(get_structure_from_src(str(path), _PDB, assembly_id="1")), ("E", "I", "E"))
        self.assert_copies(path, _PDB, ["E", "I", "E"])

    def test_two_lists_with_the_numbers_rising_from_one_copy_into_the_next(self):
        """A chain in two REMARK 350 lists of one operator each, whose first amino acid is numbered above its last: E
        with its tag (998-999, then 3-245), alone, followed by its waters, or after I in the file; and I after a free
        amino acid of its own (GLU 500). biotite gives both copies sym_id 0, and as the numbers rise from the first copy
        into the second, chain_iter runs the two together: each copy is still a chain of its own, with the residues of
        the plain file's chain (before, chain_iter's pieces were read apart, the first piece of the second copy run into
        the last piece of the first)."""
        tag = _renumbered(_E, 1, 2, 997)
        for name, atoms, lists, chains in (
                ("tag", tag + _I, ((("E",), (0,)), (("E",), (1,))), ["E", "E"]),
                ("tag with waters", tag + _I + _WATER[_WATER.chain_id == "E"], ((("E",), (0,)), (("E",), (1,))),
                 ["E", "E"]),
                ("I first", _I + tag, ((("I", "E"), (0,)), (("E",), (1,))), ["I", "E", "E"]),
                ("ligand first", _hetero([_glu(0)], "I", 500) + _E + _I, ((("I",), (0,)), (("I",), (1,))),
                 ["I", "I"])):
            with self.subTest(fixture=name):
                path = _write_pdb(atoms, self.tmp / f"{name.replace(' ', '_')}_two_lists.pdb", _remarks(*lists))
                assembly = get_structure_from_src(str(path), _PDB, assembly_id="1")
                self.assertEqual(get_protein_chains(assembly), tuple(chains))
                self.assertEqual([struc.get_residue_count(chain) for chain in protein_chain_iter(assembly)],
                                 [{"E": 241, "I": 63}[c] for c in chains])
                self.assert_copies(path, _PDB, chains)

    def test_each_copy_of_a_chain_has_a_sym_id_of_its_own(self):
        """A PDB assembly as the reader gives it: biotite's, but for the sym_ids. With E and I in one REMARK 350 list
        and E alone in another, each of one operator, biotite gives every atom sym_id 0, and the reader gives the
        second copy of E sym_id 1, all its atoms; with one list of two operators, the sym_ids are biotite's (0, then
        1). The asymmetric unit has none, and the index in the asymmetric unit that numbers the copies is not kept."""
        two_lists, one_list = ((("E", "I"), (0,)), (("E",), (1,))), ((("E", "I"), (0, 1)),)
        for lists, biotite_sym_ids in ((two_lists, [0, 0]), (one_list, [0, 1])):
            path = self.pdb("tag", *lists)
            with self.subTest(lists=len(lists)):
                assembly = get_structure_from_src(str(path), _PDB, assembly_id="1")
                biotite = PDBFile.read(str(path)).get_assembly("1", model=1)
                self.assertEqual(set(assembly.get_annotation_categories()), set(biotite.get_annotation_categories()))
                for category in biotite.get_annotation_categories():
                    if category != "sym_id":
                        self.assertTrue(np.array_equal(assembly.get_annotation(category),
                                                       biotite.get_annotation(category)), category)
                self.assertTrue(np.array_equal(assembly.coord, biotite.coord))
                au = get_structure_from_src(str(path), _PDB)
                self.assertFalse({"sym_id", "au_index"} & set(au.get_annotation_categories()))
                # The atoms of each copy: E and I, then E alone (two lists), or E and I twice (one list)
                sizes = [np.isin(au.chain_id, chains).sum() for chains, operators in lists for _ in operators]
                self.assertTrue(np.array_equal(assembly.sym_id, np.repeat([0, 1], sizes)))
                self.assertTrue(np.array_equal(biotite.sym_id, np.repeat(biotite_sym_ids, sizes)))

    def test_the_pinned_biotite_keeps_the_index_in_the_assembly(self):
        """The reader numbers the copies of a PDB assembly by each atom's index in the asymmetric unit, which its PDB
        file class adds in get_structure, and which biotite's get_assembly (biotite 1.7.1, the release's pin) keeps:
        the assembly is the asymmetric unit's atoms of each REMARK 350 list, in file order, once for each operator,
        and biotite numbers the copies of each list from 0. A biotite that builds assemblies otherwise fails here."""
        path = self.pdb("tag", (("E", "I"), (0, 1)), (("E",), (1,)))
        au = PDBFile.read(str(path)).get_structure(model=1)
        e_i, e = np.flatnonzero(np.isin(au.chain_id, ["E", "I"])), np.flatnonzero(au.chain_id == "E")
        assembly = structure_parser._IndexedPDBFile.read(str(path)).get_assembly("1", model=1)
        self.assertTrue(np.array_equal(assembly.au_index, np.concatenate([e_i, e_i, e])))
        self.assertTrue(np.array_equal(assembly.sym_id, np.repeat([0, 1, 0], [len(e_i), len(e_i), len(e)])))
        self.assertTrue(np.array_equal(get_structure_from_src(str(path), _PDB, assembly_id="1").sym_id,
                                       np.repeat([0, 1, 2], [len(e_i), len(e_i), len(e)])))

    def test_a_copy_restarting_at_its_first_residue(self):
        """E's second part (from its first cysteine after residue 60) numbered again from 1, as its first part is, so
        both start with cysteine 1: the asymmetric unit has one E, and so has each copy of the assembly (before, each
        copy of E was read as its two pieces)."""
        residues = _residues(_E)
        k = next(i for i in range(60, len(residues)) if residues[i].res_name[0] == "CYS")
        self.assertEqual((residues[0].res_id[0], residues[0].res_name[0]), (1, "CYS"))
        second = struc.concatenate(residues[k:]).copy()
        second.res_id = second.res_id - second.res_id[0] + 1
        path = _write_pdb(struc.concatenate(residues[:k]) + second + _I + _WATER, self.tmp / "restart.pdb",
                          _remarks((("E", "I"), (0, 1))))
        self.assert_paths(path, _PDB, self.plain)
        self.assert_copies(path, _PDB, ["E", "I", "E", "I"])

    def test_a_copy_repeating_its_first_piece(self):
        """E followed by a copy of its first 20 residues, numbered from 1 as E is: the asymmetric unit has one E of 261
        residues, and so has each copy of the assembly, in the build, the query and residue_embedding, as each copy has
        a sym_id of its own (before, each copy of E was read as its two pieces)."""
        repeat = struc.concatenate(_residues(_E)[:20]).copy()
        repeat.coord += 60.0
        path = _write_pdb(_E + repeat + _I + _WATER, self.tmp / "repeat.pdb", _remarks((("E", "I"), (0, 1))))
        structure = get_structure_from_src(str(path), _PDB)
        expected = dict(self.plain, E=_protein(structure[structure.chain_id == "E"]))
        sequence = self.plain["E"].sequence
        self.assertEqual(expected["E"].sequence, sequence + sequence[:20])
        self.assert_paths(path, _PDB, expected)
        built = self.assembly_build(path, _PDB)
        self.assertEqual([protein.sequence for protein in built],
                         [expected["E"].sequence, self.plain["I"].sequence] * 2)
        self.assert_same(built[0], expected["E"], "first copy of E")
        self.assert_same_proteins(self.assembly_query(path, _PDB), built, "query")
        self.assert_same_proteins(self.api(path, _PDB, assembly_id="1"), built, "residue_embedding")

    def test_a_copy_whose_later_piece_differs_by_a_name_or_an_insertion_code(self):
        """The same file, with the first residue of E's later piece made a serine (SG named OG: as many atoms, numbered
        alike) or given insertion code A: each copy of E is one chain of 261 residues in the build, the query and
        residue_embedding too (before, each copy of E was read as its two pieces)."""
        def serine(atoms, first):
            atoms.res_name[first] = "SER"
            atoms.atom_name[first & (atoms.atom_name == "SG")] = "OG"
            atoms.element[first & (atoms.element == "S")] = "O"

        def insertion_code(atoms, first):
            atoms.ins_code[first] = "A"

        for edit in (serine, insertion_code):
            with self.subTest(edit=edit.__name__):
                repeat = struc.concatenate(_residues(_E)[:20]).copy()
                repeat.coord += 60.0
                edit(repeat, repeat.res_id == repeat.res_id[0])
                path = _write_pdb(_E + repeat + _I + _WATER, self.tmp / f"{edit.__name__}.pdb",
                                  _remarks((("E", "I"), (0, 1))))
                self.assertEqual(get_protein_chains(get_structure_from_src(str(path), _PDB, assembly_id="1")),
                                 ("E", "I", "E", "I"))
                built = self.assembly_build(path, _PDB)
                self.assertEqual([len(protein) for protein in built], [261, 63, 261, 63])
                self.assert_same_proteins(self.assembly_query(path, _PDB), built, "query")
                self.assert_same_proteins(self.api(path, _PDB, assembly_id="1"), built, "residue_embedding")

    def test_archive_layout_copies_of_an_asym_in_two_rows(self):
        """1acb's mmCIF with asym A in two pdbx_struct_assembly_gen rows (the identity on A-D, the 2-fold on A):
        A, B, A (as before)."""
        cif = CIFFile.read(str(_RESOURCES / "pdb" / "1acb.cif"))
        _assembly(cif, "A,B,C,D")
        cif.block["pdbx_struct_assembly_gen"] = CIFCategory(
            {"assembly_id": ["1", "1"], "oper_expression": ["1", "2"], "asym_id_list": ["A,B,C,D", "A"]})
        path = self.tmp / "1acb.cif"
        cif.write(str(path))
        structure = get_structure_from_src(str(path), _MMCIF)
        chains = {c: _protein(structure[structure.chain_id == c]) for c in ("A", "B")}
        built = self.assembly_build(path, _MMCIF)
        self.assertEqual([protein.sequence for protein in built], [chains[c].sequence for c in "ABA"])
        self.assert_same_proteins(self.assembly_query(path, _MMCIF), built, "query")


class ArchiveLayoutTest(_Paths):
    """Archive mmCIF, each protein chain one piece: every path reads what it read before, the chain's chain_iter
    piece."""

    def test_every_path_as_before(self):
        for name in ("1acb", "2uzi"):
            path = _RESOURCES / "pdb" / f"{name}.cif"
            structure = get_structure_from_src(str(path), _MMCIF)
            expected = {c: _protein(next(struc.chain_iter(structure[structure.chain_id == c])))
                        for c in get_protein_chains(structure, 10)}
            with self.subTest(file=path.name):
                self.assertEqual(list(expected), {"1acb": ["A", "B"], "2uzi": ["A", "B", "C"]}[name])
                self.assert_paths(path, _MMCIF, expected)
                if name == "1acb":
                    self.assert_same_proteins(self.assembly_build(path, _MMCIF), list(expected.values()), "build")
                    self.assert_same_proteins(self.assembly_query(path, _MMCIF), list(expected.values()), "query")


class StoredEmbeddingsTest(_Paths):
    """The datasets of stored residue embeddings list each chain once (in an assembly, once per copy), as the build
    writes one file per chain: <name>.<chain ID>.pt, and one Parquet row per chain."""

    def setUp(self):
        super().setUp()
        self.embeddings = {"E": torch.full((243, 4), 1.0), "I": torch.full((65, 4), 2.0)}
        for chain, tensor in self.embeddings.items():
            torch.save(tensor, self.tmp / f"x.{chain}.pt")

    def test_chains(self):
        """ResidueEmbeddingFromStructure (chain_inference from structures) on E cut by its tag: x.E and x.I, the
        build's names (before, x.E twice)."""
        path = self.pdb("tag")
        dataset = ResidueEmbeddingFromStructure(src_stream=[("x", str(path), "x")], res_embedding_location=self.tmp,
                                                src_location=SrcLocation.stream, structure_format=_PDB)
        items = [dataset[i] for i in range(len(dataset))]
        self.assertEqual([name for _, name in items], ["x.E", "x.I"])
        self.assertEqual([name for _, name in items], [name for name, _ in self.build(path, _PDB)])
        for tensor, name in items:
            self.assertTrue(torch.equal(tensor, self.embeddings[name[-1]]), name)

    def test_assembly_copies(self):
        """The tensor-file and Parquet assembly datasets (inference assembly-embedding, complete-embedding) on two
        copies of E (cut by its insert) and I: E, I, E, I (before, E's file twice for each copy)."""
        path = self.pdb("insert", (("E", "I"), (0, 1)))
        kind = pa.list_(pa.list_(pa.float32(), 4))
        parquet = self.tmp / "residues.parquet"
        pq.write_table(pa.table({"id": ["x.E", "x.I"],
                                 "embedding": [self.embeddings[c].numpy().tolist() for c in "EI"]},
                                schema=pa.schema([("id", pa.string()), ("embedding", kind)])), parquet)
        assembly = [("x", str(path), "1", "x-1")]
        expected = torch.cat([self.embeddings[c] for c in "EIEI"])
        for dataset in (
                ResidueAssemblyEmbeddingFromTensorFile(src_stream=assembly, res_embedding_location=self.tmp,
                                                       src_location=SrcLocation.stream, structure_format=_PDB),
                ResidueAssemblyEmbeddingFromParquet(src_stream=assembly, parquet_path=parquet,
                                                    src_location=SrcLocation.stream, structure_format=_PDB)):
            with self.subTest(dataset=type(dataset).__name__):
                tensor, name = dataset[0]
                self.assertEqual(name, "x-1")
                self.assertTrue(torch.equal(tensor.float(), expected), tuple(tensor.shape))


class NotAProteinChainTest(_Paths):
    """A requested chain ID with no protein chain: water (1acb's C), absent (Z), or a lone free amino acid (G); and a
    requested assembly with none."""

    def setUp(self):
        super().setUp()
        label = get_structure(CIFFile.read(str(_RESOURCES / "pdb" / "1acb.cif")), model=1, use_author_fields=False)
        self.ligand = _write_cif(label + _hetero([_glu(0)], "G"), self.tmp / "ligand.cif")
        self.cases = ((_RESOURCES / "pdb" / "1acb.cif", "C"), (_RESOURCES / "pdb" / "1acb.cif", "Z"),
                      (self.ligand, "G"))

    def test_chain_lookup(self):
        """EsmProtFromChain refuses it with an IOError, printed as OSError, that names the structure (the input's
        first column), the chain and the file (it returned the lone amino acid, and failed on the others with other
        messages)."""
        for path, chain in self.cases:
            with self.subTest(chain=chain):
                with self.assertRaises(IOError) as refused:
                    EsmProtFromChain(src_stream=[("1acb", str(path), chain, f"item.{chain}")],
                                     src_location=SrcLocation.stream, structure_format=_MMCIF)[0]
                self.assertEqual(type(refused.exception).__name__, "OSError")
                self.assertEqual(str(refused.exception),
                                 f"No protein chain found for 1acb.{chain} in structure {path}.")

    def test_query(self):
        """fm-search query structure --chain-id: today's message, at --min-res 0 too (the lone amino acid was
        embedded)."""
        for path, chain in self.cases:
            with self.subTest(chain=chain):
                with self.assertRaises(ValueError) as refused:
                    list(compute_from_structure_file(path, _MMCIF, Granularity.chain, chain_id=chain, min_res=0,
                                                     accelerator=Accelerator.cpu))
                self.assertEqual(str(refused.exception),
                                 f"Chain {chain} not found or does not meet minimum residue requirements")

    def test_residue_embedding(self):
        """FoldMatch.residue_embedding with that chain ID, or with no chain long enough: a ValueError that says so
        (before, torch.cat's on an empty list)."""
        for path, chain in self.cases:
            with self.subTest(chain=chain):
                with self.assertRaises(ValueError) as refused:
                    FoldMatch(min_res=0).residue_embedding(str(path), _MMCIF, chain)
                self.assertEqual(str(refused.exception),
                                 f"No protein chain {chain} of at least 0 residues found in the structure")
        with self.assertRaises(ValueError) as refused:
            FoldMatch(min_res=242).residue_embedding(str(_RESOURCES / "pdb" / "1acb.cif"), _MMCIF)
        self.assertEqual(str(refused.exception), "No protein chain of at least 242 residues found in the structure")

    def test_assembly_query(self):
        """fm-search query structure --granularity assembly on an assembly with no chain long enough (1acb's, at
        --min-res 242) or with no protein chain (two copies of 1acb's waters, at 0): residue_embedding_by_assembly
        gives an empty dict, as residue_embedding_by_chain does, so the query refuses the assembly with its own
        message, and nothing is embedded (before, torch.cat's error on an empty list)."""
        water = CIFFile.read(str(_RESOURCES / "pdb" / "1acb.cif"))
        _assembly(water, "C,D")
        water.write(str(self.tmp / "water.cif"))
        for path, min_res in ((_RESOURCES / "pdb" / "1acb.cif", 242), (self.tmp / "water.cif", 0)):
            with self.subTest(file=path.name):
                self.esm3.proteins.clear()
                self.assertEqual(FoldMatch(min_res=min_res).residue_embedding_by_assembly(str(path), _MMCIF, "1"), {})
                with self.assertRaises(ValueError) as refused:
                    list(compute_from_structure_file(path, _MMCIF, Granularity.assembly, assembly_id="1",
                                                     min_res=min_res, accelerator=Accelerator.cpu))
                self.assertEqual(str(refused.exception),
                                 "Assembly 1 not found or does not meet minimum residue requirements")
                self.assertEqual(self.esm3.proteins, [])


if __name__ == "__main__":
    unittest.main()
