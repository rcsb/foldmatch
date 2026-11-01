"""P9.A4: the four follow-ups of P9.A3.

F1. FoldMatch tests max_res against the protein chains a call is about to embed, counted as min_res counts them
(polymer amino-acid residues, summed over the pieces kept), all of them together: the chain given by chain_id (in an
assembly, each of its copies), else every protein chain of the structure or assembly. It counted every amino acid of the
structure, or of the atoms with the chain ID, free amino acids and chains under min_res included.
F2. inference residue-embedding, structure-embedding and chain-embedding no longer take --min-res-n, which selected
nothing there; complete-embedding passes it to its assembly step only; the library refuses a min_res_n it would ignore.
F3. One reader for the chain lists of inference and of EsmProtFromChain, whose embedding name defaults to
<structure name>.<chain ID>; chain-embedding reads and writes by the embedding name; complete-embedding refuses other
names before any model is built; and a row of more than 4 fields is refused, by every command and before any model.

The fixtures are archive 1acb (chains A, 241 residues, and B, 63) and 1acb with author fields (E and I), edited and
written as PDB text by biotite (with REMARK 350 for an assembly), as in test_chain_pieces.py. ESM3 and the aggregator
are never loaded: FoldMatch embeds through a stub, and the inference commands run with stub models."""
import io
import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import biotite.structure as struc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain
from typer.testing import CliRunner

from foldmatch import FoldMatch
from foldmatch.cli import inference
from foldmatch.dataset.esm_prot_from_chain import EsmProtFromChain
from foldmatch.errors import RefusalError
# Imported before any test patches chain_inference.predict, which assembly_inferece binds at its import
from foldmatch.inference import assembly_inferece, chain_inference, esm_inference, structure_inference
from foldmatch.search.embedding_computer import compute_from_structure_file
from foldmatch.types.api_types import Accelerator, Granularity, SrcLocation, SrcProteinFrom, SrcTensorFrom, \
    StructureFormat
from foldmatch.utils.structure_parser import filter_residues, get_protein_chains, get_structure_from_src, \
    protein_chain_iter, rename_atom_attr

_RESOURCES = Path(__file__).parent / "resources"
_CIF = _RESOURCES / "pdb" / "1acb.cif"
_PDB, _MMCIF = StructureFormat.pdb, StructureFormat.mmcif
_CPU = ("--accelerator", "cpu", "--devices", "1")

_ATOMS = get_structure(CIFFile.read(str(_CIF)), model=1, use_author_fields=True)
_E = _ATOMS[(_ATOMS.chain_id == "E") & ~_ATOMS.hetero]
_I = _ATOMS[(_ATOMS.chain_id == "I") & ~_ATOMS.hetero]
_WATER = _ATOMS[_ATOMS.hetero]


def _residues(atoms) -> list:
    starts = list(struc.get_residue_starts(atoms)) + [len(atoms)]
    return [atoms[a:b] for a, b in zip(starts[:-1], starts[1:])]


def _free_glu():
    """A GLU of E as a free amino acid with chain ID I: a HETATM record, residue 500, 40 Å away."""
    glu = next(residue for residue in _residues(_E) if residue.res_name[0] == "GLU").copy()
    glu.chain_id[:] = "I"
    glu.hetero[:] = True
    glu.res_id[:] = 500
    glu.coord += 40.0
    return glu


def _atoms():
    """A free GLU with chain ID I listed first, which the rule leaves out of I; E (241 residues) and I (63); S, I's
    first 6 residues as a chain of its own, under min_res 10; the waters. The old count: 311 amino acids in all, 64
    with chain ID I."""
    short = struc.concatenate(_residues(_I)[:6]).copy()
    short.chain_id[:] = "S"
    short.coord += 80.0
    return _free_glu() + _E + _I + short + _WATER


def _write_pdb(atoms, path: Path, operators: int = 0) -> Path:
    """The atoms as a PDB file; with operators, REMARK 350 of an assembly 1 that applies the first ones of the
    identity and a 2-fold about z then 40 Å along x to chains E and I."""
    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write(str(path))
    if operators:
        remarks = ["REMARK 300", "REMARK 300 BIOMOLECULE: 1", "REMARK 350", "REMARK 350 BIOMOLECULE: 1",
                   "REMARK 350 APPLY THE FOLLOWING TO CHAINS: E, I"]
        motions = ((np.eye(3), np.zeros(3)), (np.diag([-1.0, -1.0, 1.0]), np.array([40.0, 0.0, 0.0])))
        for k, (rotation, translation) in enumerate(motions[:operators]):
            for i in range(3):
                row = "".join(f"{value:10.6f}" for value in rotation[i])
                remarks.append(f"REMARK 350   BIOMT{i + 1} {k + 1:3d}{row}{translation[i]:15.5f}")
        path.write_text("".join(f"{line}\n" for line in remarks) + path.read_text())
    return path


def _invoke(*args):
    """inference in-process (exit codes as in a shell, the exception kept), the logging setup restored after."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    try:
        return CliRunner().invoke(inference.app, [str(a) for a in args])
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)


def _flat(output: str) -> str:
    """The output on one line, typer's error panel borders dropped."""
    return " ".join(output.translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


class _Esm3:
    """Stands in for ESM3 in FoldMatch: keeps the ESMProteins it is given, and returns L + 2 rows (BOS and EOS
    included), as per_residue_embedding does."""

    def __init__(self):
        self.proteins = []

    def __call__(self, model, protein):
        self.proteins.append(protein)
        return torch.zeros(len(protein) + 2, 4)


class MaxResTest(unittest.TestCase):
    """F1: max_res counts the residues of the protein chains a call embeds, as min_res counts them, all together."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.esm3 = _Esm3()
        self.enterContext(mock.patch("foldmatch.foldmatch._load_res_model", return_value="ESM3"))
        self.enterContext(mock.patch("foldmatch.foldmatch._load_model", return_value=lambda x: x.mean(dim=0)))
        self.enterContext(mock.patch("foldmatch.foldmatch.per_residue_embedding", self.esm3))
        self.path = _write_pdb(_atoms(), self.tmp / "f1.pdb")

    def embeds(self, call, *args) -> list:
        """The lengths of the chains the call embeds, one ESM3 pass each, in order."""
        self.esm3.proteins.clear()
        call(*args)
        return [len(protein) for protein in self.esm3.proteins]

    def refuses(self, call, *args) -> str:
        """The message of the call's ValueError, raised before any ESM3 pass."""
        self.esm3.proteins.clear()
        with self.assertRaises(ValueError) as refused:
            call(*args)
        self.assertEqual(self.esm3.proteins, [], "ESM3 passes")
        return str(refused.exception)

    def test_the_sizes_are_the_min_res_count(self):
        """protein_chains_with_sizes gives the chains of protein_chain_iter with the number that min_res_n is tested
        against: E 241, I 63 (its free GLU left out); S, 6, is not a chain at 10. Each chain is kept at a threshold of
        its size, not at one more."""
        from foldmatch.utils.structure_parser import protein_chains_with_sizes
        structure = get_structure_from_src(str(self.path), _PDB)
        sized = list(protein_chains_with_sizes(structure, 10))
        self.assertEqual([(str(chain.chain_id[0]), size) for chain, size in sized], [("E", 241), ("I", 63)])
        self.assertEqual([size for _, size in protein_chains_with_sizes(structure, 0, "S")], [6])
        chains = list(protein_chain_iter(structure, 10))
        self.assertEqual(len(chains), len(sized))
        for (chain, size), same in zip(sized, chains):
            self.assertTrue(np.array_equal(chain.coord, same.coord))
            self.assertIn(chain.chain_id[0], get_protein_chains(structure, size))
            self.assertNotIn(chain.chain_id[0], get_protein_chains(structure, size + 1))

    def test_a_chain_of_exactly_max_res(self):
        """A chain of max_res residues, counted as min_res counts them, is embedded, and refused at one fewer: E at
        241, I at 63 (the old count, with I's free GLU, refused I at 63), by each method that takes a chain ID."""
        path = str(self.path)
        for c, size in (("E", 241), ("I", 63)):
            with self.subTest(chain=c):
                within = FoldMatch(max_res=size)
                self.assertEqual(self.embeds(within.residue_embedding_by_chain, path, _PDB, c), [size])
                self.assertEqual(self.embeds(within.residue_embedding, path, _PDB, c), [size])
                self.assertEqual(self.embeds(within.structure_embedding, path, _PDB, c), [size])
                over = FoldMatch(max_res=size - 1)
                message = (f"Protein too long for embedding: protein chain {c} has {size} residues "
                           f"(max {size - 1} residues)")
                for call in (over.residue_embedding_by_chain, over.residue_embedding, over.structure_embedding):
                    self.assertEqual(self.refuses(call, path, _PDB, c), message, call.__name__)

    def test_residues_embedded_but_not_counted(self):
        """max_res counts as min_res counts, not the residues ESM3 is given: the free GLU right after I, in I's piece,
        or I's 11th residue alone between two numbering gaps, is embedded with I but is not a polymer residue, so it is
        not counted (63 and 62 counted, 64 and 63 embedded). A max_res of the count embeds I, one fewer refuses it, as
        min_res keeps I at the count and not at one more."""
        from foldmatch.utils.structure_parser import protein_chains_with_sizes
        lone = [residue.copy() for residue in _residues(_I)]
        lone[10].res_id[:] += 3
        for residue in lone[11:]:
            residue.res_id[:] += 6
        for case, atoms, size in (("free_glu", _E + _I + _free_glu() + _WATER, 63),
                                  ("lone_residue", _E + struc.concatenate(lone), 62)):
            with self.subTest(case=case):
                path = str(_write_pdb(atoms, self.tmp / f"{case}.pdb"))
                structure = get_structure_from_src(path, _PDB)
                self.assertEqual([n for _, n in protein_chains_with_sizes(structure, 10, "I")], [size])
                self.assertIn("I", get_protein_chains(structure, size))
                self.assertNotIn("I", get_protein_chains(structure, size + 1))
                self.assertEqual(self.embeds(FoldMatch(max_res=size).residue_embedding_by_chain, path, _PDB, "I"),
                                 [size + 1])
                self.assertEqual(self.refuses(FoldMatch(max_res=size - 1).residue_embedding_by_chain, path, _PDB, "I"),
                                 f"Protein too long for embedding: protein chain I has {size} residues "
                                 f"(max {size - 1} residues)")

    def test_a_chain_within_the_limit_in_a_structure_above_it(self):
        """The reported case: at max_res 100, between I's 63 residues and E's 241, I alone is embedded, by the query
        too (compute_from_structure_file with a chain ID); E alone is refused, and so are E and I together (304); the
        old count, the whole structure's 311 amino acids, refused I too."""
        path = str(self.path)
        api = FoldMatch(max_res=100)
        self.assertEqual(self.embeds(api.residue_embedding_by_chain, path, _PDB, "I"), [63])
        self.assertEqual(self.embeds(api.residue_embedding, path, _PDB, "I"), [63])
        query = dict(min_res=10, max_res=100, accelerator=Accelerator.cpu)
        ids, vectors = next(compute_from_structure_file(self.path, _PDB, Granularity.chain, chain_id="I", **query))
        self.assertEqual((ids, vectors.shape), (["f1.I"], (1, 4)))
        message = "Protein too long for embedding: protein chain E has 241 residues (max 100 residues)"
        self.assertEqual(self.refuses(api.residue_embedding_by_chain, path, _PDB, "E"), message)
        self.assertEqual(self.refuses(api.residue_embedding, path, _PDB, "E"), message)
        self.assertEqual(self.refuses(lambda: list(compute_from_structure_file(
            self.path, _PDB, Granularity.chain, chain_id="E", **query))), message)
        message = ("Protein too long for embedding: the 2 protein chains of the structure have 304 residues "
                   "(max 100 residues)")
        for call in (api.residue_embedding_by_chain, api.residue_embedding, api.residue_embedding_by_assembly):
            self.assertEqual(self.refuses(call, path, _PDB), message, call.__name__)

    def test_every_chain_together(self):
        """Without a chain ID, the chains embedded, together: 304 residues (S, under min_res, and the free GLU are not
        counted), which max_res 304 lets through (the old count, 311, refused it) and 303 refuses."""
        path = str(self.path)
        within, over = FoldMatch(max_res=304), FoldMatch(max_res=303)
        message = ("Protein too long for embedding: the 2 protein chains of the structure have 304 residues "
                   "(max 303 residues)")
        for method in ("residue_embedding_by_chain", "residue_embedding", "residue_embedding_by_assembly"):
            with self.subTest(method=method):
                self.assertEqual(self.embeds(getattr(within, method), path, _PDB), [241, 63])
                self.assertEqual(self.refuses(getattr(over, method), path, _PDB), message)

    def test_every_copy_in_an_assembly(self):
        """Assembly 1 with two copies of E and I: residue_embedding with chain ID I counts both copies of I (126),
        residue_embedding_by_assembly and residue_embedding without a chain ID every chain of every copy (608), the
        assembly query too; the old count added each copy's free GLU (128 and 610)."""
        path = _write_pdb(_atoms(), self.tmp / "f1_assembly.pdb", operators=2)
        self.assertEqual(self.embeds(FoldMatch(max_res=126).residue_embedding, str(path), _PDB, "I", "1"), [63, 63])
        self.assertEqual(self.refuses(FoldMatch(max_res=125).residue_embedding, str(path), _PDB, "I", "1"),
                         "Protein too long for embedding: the 2 copies of protein chain I in assembly 1 have 126 "
                         "residues (max 125 residues)")
        message = ("Protein too long for embedding: the 4 protein chains of assembly 1 have 608 residues "
                   "(max 607 residues)")
        for method, args in (("residue_embedding_by_assembly", ("1",)), ("residue_embedding", (None, "1"))):
            with self.subTest(method=method):
                self.assertEqual(self.embeds(getattr(FoldMatch(max_res=608), method), str(path), _PDB, *args),
                                 [241, 63, 241, 63])
                self.assertEqual(self.refuses(getattr(FoldMatch(max_res=607), method), str(path), _PDB, *args),
                                 message)
        query = dict(assembly_id="1", min_res=10, accelerator=Accelerator.cpu)
        ids, _ = next(compute_from_structure_file(path, _PDB, Granularity.assembly, max_res=608, **query))
        self.assertEqual(ids, ["f1_assembly-1"])
        self.assertEqual(self.refuses(lambda: list(compute_from_structure_file(
            path, _PDB, Granularity.assembly, max_res=607, **query))), message)

    def test_the_message_names_what_was_counted(self):
        """One copy, one chain: in the singular."""
        path = _write_pdb(_atoms(), self.tmp / "one_copy.pdb", operators=1)
        self.assertEqual(self.refuses(FoldMatch(max_res=62).residue_embedding, str(path), _PDB, "I", "1"),
                         "Protein too long for embedding: the 1 copy of protein chain I in assembly 1 has 63 residues "
                         "(max 62 residues)")
        self.assertEqual(self.refuses(FoldMatch(min_res=64, max_res=240).residue_embedding_by_chain, str(_CIF)),
                         "Protein too long for embedding: the 1 protein chain of the structure has 241 residues "
                         "(max 240 residues)")

    def test_nothing_to_embed(self):
        """No chain of at least min_res residues (242; E has 241): residue_embedding raises its own ValueError and the
        other two methods give an empty dict, at any max_res, a negative one included: nothing to embed is never too
        long (the old count refused the structure as too long)."""
        path = str(self.path)
        for max_res in (10, -1):
            with self.subTest(max_res=max_res):
                api = FoldMatch(min_res=242, max_res=max_res)
                self.assertEqual(self.refuses(api.residue_embedding, path, _PDB),
                                 "No protein chain of at least 242 residues found in the structure")
                self.assertEqual(self.refuses(api.residue_embedding, path, _PDB, "E"),
                                 "No protein chain E of at least 242 residues found in the structure")
                self.assertEqual(api.residue_embedding_by_chain(path, _PDB), {})
                self.assertEqual(api.residue_embedding_by_chain(path, _PDB, "E"), {})
                self.assertEqual(api.residue_embedding_by_assembly(path, _PDB), {})
                self.assertEqual(self.esm3.proteins, [])


class MinResNTest(unittest.TestCase):
    """F2: --min-res-n only where it selects chains."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.chains = self.tmp / "chains.csv"
        self.chains.write_text(f"1acb,{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B\n")
        self.assemblies = self.tmp / "assemblies.csv"
        self.assemblies.write_text(f"1acb,{_CIF},1,1acb-1\n")
        self.out = self.tmp / "out"
        self.out.mkdir()

    def test_the_chain_list_commands_refuse_it(self):
        """residue-embedding, structure-embedding and chain-embedding: Typer's usage error (exit status 2), before
        anything runs (before, it was taken and did nothing)."""
        never = mock.Mock(side_effect=AssertionError("predict was called"))
        with mock.patch.object(esm_inference, "predict", never), \
                mock.patch.object(structure_inference, "predict", never), \
                mock.patch.object(chain_inference, "predict", never):
            for command, extra in (("residue-embedding", ()), ("structure-embedding", ()),
                                   ("chain-embedding", ("--res-embedding-location", self.out))):
                with self.subTest(command=command):
                    result = _invoke(command, "--src-file", self.chains, "--output-folder", self.out, *extra,
                                     "--min-res-n", 10)
                    self.assertEqual(result.exit_code, 2, result.output)
                    self.assertIn("No such option: --min-res-n", _flat(result.output))
        never.assert_not_called()

    def test_assembly_and_complete_embedding_take_it(self):
        """assembly-embedding passes it to its predict, and complete-embedding to its assembly step only (before, to
        its residue and chain steps too, where it did nothing)."""
        with mock.patch.object(esm_inference, "predict") as residue, \
                mock.patch.object(chain_inference, "predict") as chain, \
                mock.patch.object(assembly_inferece, "predict") as assembly:
            result = _invoke("assembly-embedding", "--src-file", self.assemblies, "--res-embedding-location", self.out,
                             "--output-folder", self.out, "--min-res-n", 7)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(assembly.call_args.kwargs["min_res_n"], 7)
            assembly.reset_mock()
            result = _invoke("complete-embedding", "--src-chain-file", self.chains,
                             "--src-assembly-file", self.assemblies, "--output-res-path", self.out,
                             "--output-chain-path", self.out, "--output-assembly-path", self.out, "--min-res-n", 7)
            self.assertEqual(result.exit_code, 0, result.output)
            for step in (residue, chain, assembly):
                step.assert_called_once()
            self.assertNotIn("min_res_n", residue.call_args.kwargs)
            self.assertNotIn("min_res_n", chain.call_args.kwargs)
            self.assertEqual(assembly.call_args.kwargs["min_res_n"], 7)

    def test_the_library_refuses_a_min_res_n_it_would_ignore(self):
        """esm_inference and structure_inference given chains, chain_inference given residue files, Parquet or an
        inference_set: a RefusalError before the input is read (it does not exist) or any model is built. min_res_n 0,
        and min_res_n with structures read whole, go on to read the input, as before."""
        absent = self.tmp / "absent.csv"
        lead = ("min_res_n=10 selects the protein chains of structures read whole (src_from structure), but the "
                "chains of ")
        end = " are embedded as listed: leave min_res_n at 0, or list only the chains to embed."
        chain_list = lead + "a chain list (src_from chain)" + end
        listed = lead + "an inference_set or of residue embedding files or Parquet (src_from file or parquet)" + end
        never = mock.Mock(side_effect=AssertionError("a model was built"))
        with mock.patch.object(esm_inference, "get_residue_model", never), \
                mock.patch.object(structure_inference, "get_residue_model", never), \
                mock.patch.object(structure_inference, "get_aggregator_model", never), \
                mock.patch.object(chain_inference, "get_aggregator_model", never):
            for name, call, message in (
                    ("esm_inference", lambda n: esm_inference.predict(
                        absent, src_from=SrcProteinFrom.chain, min_res_n=n), chain_list),
                    ("structure_inference", lambda n: structure_inference.predict(
                        absent, src_from=SrcProteinFrom.chain, min_res_n=n), chain_list),
                    ("chain_inference, files", lambda n: chain_inference.predict(
                        absent, src_from=SrcTensorFrom.file, min_res_n=n), listed),
                    ("chain_inference, Parquet", lambda n: chain_inference.predict(
                        absent, src_from=SrcTensorFrom.parquet, min_res_n=n), listed),
                    ("chain_inference, inference_set", lambda n: chain_inference.predict(
                        absent, src_from=SrcTensorFrom.structure, min_res_n=n, inference_set=[]), listed)):
                with self.subTest(call=name):
                    with self.assertRaises(RefusalError) as refused:
                        call(10)
                    self.assertEqual(str(refused.exception), message)
                    if name != "chain_inference, inference_set":
                        with self.assertRaises((FileNotFoundError, RefusalError)) as read:
                            call(0)
                        self.assertIn(str(absent), str(read.exception))
            with self.assertRaises(FileNotFoundError):
                esm_inference.predict(absent, src_from=SrcProteinFrom.structure, min_res_n=10)
            with self.assertRaises(FileNotFoundError):
                structure_inference.predict(absent, src_from=SrcProteinFrom.structure, min_res_n=10)
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                chain_inference.predict(absent, res_embedding_location=self.tmp / "absent",
                                        src_from=SrcTensorFrom.structure, min_res_n=10)
        never.assert_not_called()

    def test_the_help_says_which_chains_it_selects(self):
        """assembly-embedding and complete-embedding: the chains of each assembly (complete-embedding said "When using
        all chains in a structure"); the help of the three chain-list commands no longer names the option."""
        for command, selects in (("assembly-embedding", True), ("complete-embedding", True),
                                 ("residue-embedding", False), ("structure-embedding", False),
                                 ("chain-embedding", False)):
            with self.subTest(command=command):
                result = CliRunner().invoke(inference.app, [command, "--help"], env={"COLUMNS": "400"})
                self.assertEqual(result.exit_code, 0, result.output)
                if selects:
                    self.assertIn("--min-res-n INTEGER Consider only assembly chains with at least <min_res_n> "
                                  "residues.", _flat(result.output))
                else:
                    self.assertNotIn("min-res", _flat(result.output))


def _esm3(model, protein):
    """Stands in for ESM3 in the inference modules: L + 2 rows (BOS and EOS, zeros), each residue's row its letter, its
    position and the chain's length, so that each chain has a tensor of its own (exact sums in float32)."""
    rows = [[float(ord(aa)), float(k), float(len(protein))] for k, aa in enumerate(protein.sequence)]
    return torch.tensor([[0.0] * 3] + rows + [[0.0] * 3])


def _aggregate(x, x_mask):
    """Stands in for the aggregator: the sum of each item's rows."""
    return x.masked_fill(x_mask.unsqueeze(-1), 0.0).sum(dim=1)


def _csv(residues) -> str:
    """The chain embedding of these residue embeddings as CsvBatchWriter writes it: _aggregate's vector, one value per
    line."""
    return pd.DataFrame(residues.sum(dim=0).numpy()).to_csv(index=False, header=False)


def _protein(chain_id: str) -> ESMProtein:
    """Archive 1acb's protein chain, read as every path reads it."""
    atoms = next(protein_chain_iter(get_structure_from_src(str(_CIF), _MMCIF), chain_id=chain_id))
    return ESMProtein.from_protein_chain(ProteinChain.from_atomarray(rename_atom_attr(filter_residues(atoms))))


class ChainListTest(unittest.TestCase):
    """F3: the inference commands on chain lists, with ESM3 and the aggregator stubbed (_esm3, _aggregate)."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for module in ("esm_inference", "structure_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_residue_model", return_value="ESM3"))
        for module in ("structure_inference", "chain_inference"):
            self.enterContext(mock.patch(f"foldmatch.inference.{module}.get_aggregator_model", return_value=_aggregate))
        for module in ("esm_module", "chain_complete_module"):
            self.enterContext(mock.patch(f"foldmatch.modules.{module}.per_residue_embedding", _esm3))
        # The residue embeddings of archive 1acb's chains, as the stub gives them
        self.residues_of = {c: _esm3(None, _protein(c)) for c in "AB"}

    def folder(self, name: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=self.tmp))

    def chain_list(self, text: str) -> Path:
        path = self.tmp / f"list-{len(list(self.tmp.glob('list-*.csv')))}.csv"
        path.write_text(text)
        return path

    def residues(self, chains: Path):
        """inference residue-embedding: its folder, and the tensors it wrote there, by file name."""
        out = self.folder("residues")
        result = _invoke("residue-embedding", "--src-file", chains, "--output-folder", out, *_CPU)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        return out, {f.name: torch.load(f, weights_only=True) for f in out.iterdir()}

    def chains(self, chains: Path, residues: Path, res_format: str = "pt"):
        """inference chain-embedding (CSV): its result, and the files it wrote, by name."""
        out = self.folder("chains")
        result = _invoke("chain-embedding", "--src-file", chains, "--res-embedding-location", residues,
                         "--res-embedding-format", res_format, "--output-folder", out, *_CPU)
        return result, {f.name: f.read_text() for f in out.iterdir()}

    def structures(self, chains: Path) -> dict:
        """inference structure-embedding: its chain embeddings, by id."""
        out = self.folder("structures")
        result = _invoke("structure-embedding", "--src-file", chains, "--output-folder", out, *_CPU)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        table = pq.read_table(out / "inference-0.parquet").to_pydict()
        return dict(zip(table["id"], table["embedding"]))

    def assert_residues(self, written: dict, expected: dict):
        """The residue files written, by name, hold the residue embeddings of these chains."""
        self.assertEqual(sorted(written), sorted(expected))
        for name, chain in expected.items():
            self.assertTrue(torch.equal(written[name], self.residues_of[chain]), name)

    def assert_chains(self, result, written: dict, expected: dict):
        """chain-embedding ran, and wrote the chain embeddings of these chains, by name."""
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        self.assertEqual(written, {name: _csv(self.residues_of[chain]) for name, chain in expected.items()})

    def vector(self, chain: str) -> list:
        return self.residues_of[chain].sum(dim=0).tolist()

    def test_custom_names_flow_from_residue_to_chain_embeddings(self):
        """residue-embedding writes each row under its embedding name, and chain-embedding reads that file and writes
        under that name (before, it read 1acb.A.pt and 1acb.B.pt, which were not there)."""
        chains = self.chain_list(f"1acb,{_CIF},A,first\n1acb,{_CIF},B,second\n")
        residues, written = self.residues(chains)
        self.assert_residues(written, {"first.pt": "A", "second.pt": "B"})
        self.assert_chains(*self.chains(chains, residues), {"first.csv": "A", "second.csv": "B"})
        self.assertEqual(self.structures(chains), {"first": self.vector("A"), "second": self.vector("B")})

    def test_a_missing_or_empty_name_is_structure_dot_chain(self):
        """A row of three fields, or with an empty 4th: 1acb.A and 1acb.B, in residue-embedding (before, the name ''
        made torch.save refuse the file name '.pt'), chain-embedding and structure-embedding (before, empty ids)."""
        chains = self.chain_list(f"1acb,{_CIF},A\n1acb,{_CIF},B,\n")
        residues, written = self.residues(chains)
        self.assert_residues(written, {"1acb.A.pt": "A", "1acb.B.pt": "B"})
        self.assert_chains(*self.chains(chains, residues), {"1acb.A.csv": "A", "1acb.B.csv": "B"})
        self.assertEqual(self.structures(chains), {"1acb.A": self.vector("A"), "1acb.B": self.vector("B")})

    def test_conventional_lists_as_before(self):
        """Embedding names <structure name>.<chain ID>: the names and contents that the stubs give through the
        unchanged readers and writers (EsmProtFromChain, TensorBatchWriter, CsvBatchWriter, ParquetBatchWriter),
        recorded here, so that this test passes before and after this change."""
        chains = self.chain_list(f"1acb,{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B\n")
        residues, written = self.residues(chains)
        self.assert_residues(written, {"1acb.A.pt": "A", "1acb.B.pt": "B"})
        self.assert_chains(*self.chains(chains, residues), {"1acb.A.csv": "A", "1acb.B.csv": "B"})
        self.assertEqual(self.structures(chains), {"1acb.A": self.vector("A"), "1acb.B": self.vector("B")})

    def test_the_recipe_of_residue_files(self):
        """The CHANGELOG's recipe for residue files NAME.CHAIN.pt or NAME.CHAIN.csv, a list of NAME,,CHAIN lines:
        chain embeddings NAME.CHAIN.csv, as before. The file column is not read, so neither an empty one nor two
        files under one name is refused."""
        residues = self.folder("recipe")
        for chain, tensor in self.residues_of.items():
            torch.save(tensor, residues / f"x.{chain}.pt")
        pd.DataFrame(self.residues_of["A"].numpy()).to_csv(residues / "x.A.csv", index=False, header=False)
        self.assert_chains(*self.chains(self.chain_list("x,,A\nx,,B\n"), residues), {"x.A.csv": "A", "x.B.csv": "B"})
        self.assert_chains(*self.chains(self.chain_list("x,,A\n"), residues, "csv"), {"x.A.csv": "A"})
        self.assert_chains(*self.chains(self.chain_list(f"x,{_CIF},A\nx,{self.tmp / 'other.cif'},B\n"), residues),
                           {"x.A.csv": "A", "x.B.csv": "B"})

    def test_a_quoted_field_is_read_alike(self):
        """Quoted fields: residue-embedding and chain-embedding read the same names, 1acb.A (the quotes of "1acb" and
        "A" dropped) and b,1 (a comma within quotes); before, chain-embedding split the lines on commas, and looked for
        "1acb"."A".pt."""
        chains = self.chain_list(f'"1acb",{_CIF},"A",\n1acb,{_CIF},B,"b,1"\n')
        residues, written = self.residues(chains)
        self.assert_residues(written, {"1acb.A.pt": "A", "b,1.pt": "B"})
        self.assert_chains(*self.chains(chains, residues), {"1acb.A.csv": "A", "b,1.csv": "B"})

    def test_a_missing_residue_file_is_named(self):
        """chain-embedding names the file it did not find, with the row's embedding name, and says what to do; a line
        of one or two fields gets an empty chain ID, and so a missing file (before, it was skipped without a word)."""
        residues = self.folder("empty")
        for text, name in ((f"1acb,{_CIF},A,first\n", "first"), (f"1acb,{_CIF}\n", "1acb."), ("1acb\n", "1acb.")):
            with self.subTest(line=text.strip()):
                result, written = self.chains(self.chain_list(text), residues)
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIsInstance(result.exception, FileNotFoundError)
                self.assertEqual(str(result.exception),
                                 f"No residue embedding file {residues.resolve() / f'{name}.pt'} for {name}: compute "
                                 f"it first, or check the folder and the format of the residue embeddings.")
                self.assertEqual(written, {})

    def test_complete_embedding_with_the_default_names(self):
        """complete-embedding on a list without the 4th column: the residue files, chain embeddings and assembly
        embedding under <structure name>.<chain ID> (before, the residue step stopped on the file name '.pt')."""
        chains = self.chain_list(f"1acb,{_CIF},A\n1acb,{_CIF},B\n")
        assemblies = self.chain_list(f"1acb,{_CIF},1,1acb-1\n")
        residues, chain_out, assembly_out = self.folder("residues"), self.folder("chains"), self.folder("assemblies")
        result = _invoke("complete-embedding", "--src-chain-file", chains, "--src-assembly-file", assemblies,
                         "--output-res-path", residues, "--output-chain-path", chain_out,
                         "--output-assembly-path", assembly_out, *_CPU)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        self.assert_residues({f.name: torch.load(f, weights_only=True) for f in residues.iterdir()},
                             {"1acb.A.pt": "A", "1acb.B.pt": "B"})
        self.assertEqual({f.name: f.read_text() for f in chain_out.iterdir()},
                         {f"1acb.{c}.csv": _csv(self.residues_of[c]) for c in "AB"})
        self.assertEqual({f.name: f.read_text() for f in assembly_out.iterdir()},
                         {"1acb-1.csv": _csv(torch.cat([self.residues_of["A"], self.residues_of["B"]]))})

    def test_assembly_embedding_with_min_res_n(self):
        """F2 through assembly_inferece and chain_inference, neither mocked: assembly-embedding --min-res-n 100 embeds
        assembly 1 of 1acb from A alone (B has 63 residues). chain_inference refuses a min_res_n given an
        inference_set, so assembly_inferece must not hand it on (complete-embedding's assembly step is this command)."""
        residues, _ = self.residues(self.chain_list(f"1acb,{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B\n"))
        out = self.folder("assemblies")
        result = _invoke("assembly-embedding", "--src-file", self.chain_list(f"1acb,{_CIF},1,1acb-1\n"),
                         "--res-embedding-location", residues, "--output-folder", out, "--min-res-n", 100, *_CPU)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        self.assertEqual({f.name: f.read_text() for f in out.iterdir()}, {"1acb-1.csv": _csv(self.residues_of["A"])})

    def test_complete_embedding_refuses_other_names_before_any_model(self):
        """A chain list with other embedding names: complete-embedding refuses it (RefusalError, exit status 1) before
        any step runs, naming up to 5 such rows (5 of the first list's 7) and how many, each field quoted, so that
        whitespace shows (the second list's first structure name starts with a space). Before, it ran the residue step,
        and failed at the chain step on the first list; it ran the second as if the space were not there."""
        names = [("A", "a1"), ("A", "a2"), ("B", "b1"), ("A", "a3"), ("B", "b2"), ("B", "b3"), ("A", "a4")]
        assemblies = self.chain_list(f"1acb,{_CIF},1,1acb-1\n")
        for case, text, found in (
                ("7 rows named otherwise",
                 f"1acb,{_CIF},A,1acb.A\n" + "".join(f"1acb,{_CIF},{c},{n}\n" for c, n in names),
                 "7 row(s) with an embedding name other than <structure name>.<chain ID>: "
                 + "; ".join(f"'1acb' chain '{c}' named '{n}'" for c, n in names[:5]) + "; and 2 more rows"),
                ("a structure name that starts with a space", f" 1acb,{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B\n",
                 "1 row(s) with an embedding name other than <structure name>.<chain ID>: "
                 "' 1acb' chain 'A' named '1acb.A'")):
            with self.subTest(case=case):
                chains = self.chain_list(text)
                out = self.folder("out")
                never = mock.Mock(side_effect=AssertionError("a step ran"))
                with mock.patch.object(esm_inference, "predict", never), \
                        mock.patch.object(chain_inference, "predict", never), \
                        mock.patch.object(assembly_inferece, "predict", never):
                    result = _invoke("complete-embedding", "--src-chain-file", chains, "--src-assembly-file",
                                     assemblies, "--output-res-path", out, "--output-chain-path", out,
                                     "--output-assembly-path", out, *_CPU)
                never.assert_not_called()
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIsInstance(result.exception, RefusalError)
                self.assertEqual(str(result.exception),
                                 f"{chains.resolve()}: {found}. The assembly step of complete-embedding reads each "
                                 f"chain's residue embedding as <structure name>.<chain ID>, as assembly-embedding "
                                 f"does: name those rows so, or leave their 4th column out.")
                self.assertEqual(list(out.iterdir()), [])

    def test_the_reader(self):
        """read_chain_list, of a file and of a stream: the four columns as strings, in input order (not sorted), each
        field as written, spaces included; a missing or empty embedding name, <structure name>.<chain ID>; quoted
        fields read as CSV; nothing refused, a blank chain ID included."""
        from foldmatch.dataset.utils import read_chain_list
        text = 'g\nb,f.cif,B\na,f.cif,A,named\nc,f.cif,C,\n"d,1",,"D",\ne,f.cif,,\nf,f.cif\n h , f.cif, H ,\n'
        expected = [["g", "", "", "g."], ["b", "f.cif", "B", "b.B"], ["a", "f.cif", "A", "named"],
                    ["c", "f.cif", "C", "c.C"], ["d,1", "", "D", "d,1.D"], ["e", "f.cif", "", "e."],
                    ["f", "f.cif", "", "f."], [" h ", " f.cif", " H ", " h . H "]]
        data = read_chain_list(self.chain_list(text))
        self.assertEqual(list(data.columns), ["stream_name", "stream", "chain_id", "item_name"])
        self.assertEqual(data.values.tolist(), expected)
        stream = [("h", "f.cif", "H", None), ("a", "f.cif", "A", "named"), ("b", "f.cif", "B"), ("c", "f.cif", "C", "")]
        self.assertEqual(read_chain_list(stream, SrcLocation.stream).values.tolist(),
                         [["h", "f.cif", "H", "h.H"], expected[2], expected[1], expected[3]])

    def test_a_blank_chain_id_is_a_chain_id(self):
        """A PDB file whose chain I has a blank chain ID, which biotite reads as '': EsmProtFromChain embeds it for a
        stream row of three fields, named x. (before, a stream row of three fields was refused by pandas)."""
        atoms = _E + _I
        atoms.chain_id[atoms.chain_id == "I"] = ""
        path = _write_pdb(atoms, self.tmp / "blank.pdb")
        protein, name = EsmProtFromChain(src_stream=[("x", str(path), "")], src_location=SrcLocation.stream,
                                         structure_format=_PDB)[0]
        self.assertEqual((name, len(protein)), ("x.", 63))


# What a refusal of a chain list row of more than 4 fields says after what it found
_AT_MOST_4_FIELDS = ("A chain list row holds at most 4 fields (structure name, structure file or URL, chain ID, "
                     "embedding name): ")
_REMOVE_THE_EXTRA_FIELDS = "remove the extra fields (a comma after the embedding name makes a 5th)."


class LongerRowTest(unittest.TestCase):
    """F3: a chain list row of more than 4 fields is refused (RefusalError) by read_chain_list, for a file and for a
    stream, naming the file or the stream and the row, and so by each command that reads a chain list and by
    EsmProtFromChain, before any model is built (here the model getters fail if called). Before, pandas read the leading
    fields of a first such row as the index, which shifted the columns of every row (by two for 6 fields), and stopped
    at a later one with a ParserError that named no file; chain-embedding ignored the fields after the 3rd. Lists of at
    most 4 fields per row are read as before."""

    # (case, chain list, what the refusal found, after the file's path: a regular expression)
    LISTS = (
        ("first row, a trailing comma", f"1acb,{_CIF},A,1acb.A,\n1acb,{_CIF},B,1acb.B\n",
         re.escape("its first row has 5 fields")),
        ("first row, a 5th field of text", f"1acb,{_CIF},A,1acb.A,extra\n1acb,{_CIF},B,1acb.B\n",
         re.escape("its first row has 5 fields")),
        ("first row of 6 fields", f"1acb,{_CIF},A,1acb.A,,extra\n1acb,{_CIF},B,1acb.B\n",
         re.escape("its first row has 6 fields")),
        ("a later row of 5 fields", f"1acb,{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B,\n",
         r"a row has more than 4 fields \(pandas: [^()]*Expected 4 fields in line 2, saw 5\)"),
        ("a first row of 5, a later row of 6", f"1acb,{_CIF},A,1acb.A,\n1acb,{_CIF},B,1acb.B,,\n",
         r"a row has more than 4 fields \(pandas: [^()]*Expected 5 fields in line 2, saw 6\)"),
    )

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.never = mock.Mock(side_effect=AssertionError("a model was built"))
        for getter in ("esm_inference.get_residue_model", "structure_inference.get_residue_model",
                       "structure_inference.get_aggregator_model", "chain_inference.get_aggregator_model"):
            self.enterContext(mock.patch(f"foldmatch.inference.{getter}", self.never))

    def chain_list(self, case: str, text: str) -> Path:
        path = self.tmp / f"{case.replace(',', '').replace(' ', '_')}.csv"
        path.write_text(text)
        return path

    def assert_refused(self, refusal: BaseException, source: str, found: str):
        """The refusal names the source and what it found, says that a row holds at most 4 fields, and what to do."""
        self.assertIsInstance(refusal, RefusalError)
        self.assertRegex(str(refusal), f"^{re.escape(source)}: {found}\\. "
                                       f"{re.escape(_AT_MOST_4_FIELDS + _REMOVE_THE_EXTRA_FIELDS)}$")

    def assert_frame(self, data: pd.DataFrame, expected: list):
        """The four columns, the rows as expected, every value a string, the index pandas gives (0, 1, ...)."""
        self.assertEqual(list(data.columns), ["stream_name", "stream", "chain_id", "item_name"])
        self.assertEqual(data.values.tolist(), expected)
        self.assertTrue(all(isinstance(value, str) for value in data.values.ravel()))
        self.assertIsInstance(data.index, pd.RangeIndex)
        self.assertEqual(list(data.index), list(range(len(expected))))

    def test_the_reader_refuses_rows_of_more_than_4_fields(self):
        """A first row with a trailing comma, with a 5th field of text or with 6 fields (pandas took its first one or
        two fields as the index), and a later row of 5, or of 6 after a first row of 5 (pandas' ParserError, which
        names the line, quoted and kept as the cause): from the path, from an open file (named by its path) and from a
        text buffer."""
        from foldmatch.dataset.utils import read_chain_list
        for case, text, found in self.LISTS:
            with self.subTest(case=case):
                path = self.chain_list(case, text)
                with self.assertRaises(RefusalError) as refused:
                    read_chain_list(path)
                self.assert_refused(refused.exception, str(path), found)
                if "later row" in case:
                    self.assertIsInstance(refused.exception.__cause__, pd.errors.ParserError)
                with open(path) as opened, self.assertRaises(RefusalError) as refused:
                    read_chain_list(opened)
                self.assert_refused(refused.exception, str(path), found)
                with self.assertRaises(RefusalError) as refused:
                    read_chain_list(io.StringIO(text))
                self.assert_refused(refused.exception, "The chain list", found)

    def test_a_stream_row_of_more_than_4_elements(self):
        """A stream: how many rows have more than 4 elements, and up to 5 of them, counted from 1 (before, pandas'
        ValueError named no row)."""
        from foldmatch.dataset.utils import read_chain_list
        rule = ("A chain list row holds at most 4 fields (structure name, structure file or URL, chain ID, embedding "
                "name): remove the extra elements.")
        for rows, found in (
                ([("x", "f", "A", "x.A"), ("y", "f", "B", "y.B", "")], "1 row(s) of more than 4 elements (row(s) 2"),
                ([("w", "f", "W")] + [("x", "f", "A", "x.A", "z")] * 5 + [("y",) * 6] * 2,
                 "7 row(s) of more than 4 elements (row(s) 2, 3, 4, 5, 6 and 2 more")):
            with self.subTest(rows=len(rows)):
                with self.assertRaises(RefusalError) as refused:
                    read_chain_list(rows, SrcLocation.stream)
                self.assertEqual(str(refused.exception), f"The chain list stream: {found}, counted from 1). {rule}")

    def test_another_parser_error_is_not_claimed_as_extra_fields(self):
        """Another pandas ParserError (an unclosed quote) is refused too, naming the file and quoting pandas, without
        saying that a row has more than 4 fields."""
        from foldmatch.dataset.utils import read_chain_list
        path = self.chain_list("unclosed quote", f'1acb,"{_CIF},A,1acb.A\n1acb,{_CIF},B,1acb.B\n')
        with self.assertRaises(RefusalError) as refused:
            read_chain_list(path)
        self.assertIsInstance(refused.exception.__cause__, pd.errors.ParserError)
        self.assertRegex(str(refused.exception),
                         f"^{re.escape(str(path))}: pandas could not read it as CSV \\([^()]*EOF inside string"
                         f"[^()]*\\)\\. " + re.escape("A chain list is CSV without a header, of at most 4 fields per "
                                                      "row (structure name, structure file or URL, chain ID, embedding "
                                                      "name): correct it where pandas' message points.") + "$")
        self.assertNotIn("more than 4", str(refused.exception))

    def test_each_command_refuses_it_before_any_model(self):
        """inference residue-embedding, structure-embedding, chain-embedding and complete-embedding: exit status 1 with
        the RefusalError, nothing written and no model built. Before, residue-embedding and structure-embedding read the
        shifted columns or stopped on pandas' ParserError, chain-embedding read each line's 1st and 3rd fields and went
        on, and complete-embedding's residue step did as residue-embedding did."""
        assemblies = self.chain_list("assemblies", f"1acb,{_CIF},1,1acb-1\n")
        for case, text, found in self.LISTS:
            path = self.chain_list(case, text)
            for command in ("residue-embedding", "structure-embedding", "chain-embedding", "complete-embedding"):
                with self.subTest(case=case, command=command):
                    out = Path(tempfile.mkdtemp(dir=self.tmp))
                    if command == "complete-embedding":
                        args = ("--src-chain-file", path, "--src-assembly-file", assemblies, "--output-res-path", out,
                                "--output-chain-path", out, "--output-assembly-path", out)
                    elif command == "chain-embedding":
                        args = ("--src-file", path, "--res-embedding-location", out, "--output-folder", out)
                    else:
                        args = ("--src-file", path, "--output-folder", out)
                    result = _invoke(command, *args, *_CPU)
                    self.assertEqual(result.exit_code, 1, result.output)
                    self.assert_refused(result.exception, str(path.resolve()), found)
                    self.assertEqual(list(out.iterdir()), [])
        self.never.assert_not_called()

    def test_esm_prot_from_chain_refuses_it(self):
        """The library's EsmProtFromChain, given a file or a stream (before: the shifted columns or pandas' ParserError
        for a file, pandas' ValueError for a stream)."""
        for case, text, found in self.LISTS:
            with self.subTest(case=case):
                path = self.chain_list(case, text)
                with self.assertRaises(RefusalError) as refused:
                    EsmProtFromChain(src_stream=path)
                self.assert_refused(refused.exception, str(path), found)
        with self.assertRaises(RefusalError) as refused:
            EsmProtFromChain(src_stream=[("1acb", str(_CIF), "A", "1acb.A"), ("1acb", str(_CIF), "B", "1acb.B", "x")],
                             src_location=SrcLocation.stream)
        self.assertTrue(str(refused.exception).startswith(
            "The chain list stream: 1 row(s) of more than 4 elements (row(s) 2, counted from 1)."), refused.exception)

    def test_lists_of_at_most_4_fields_read_as_before(self):
        """The frames that the reader gave before it refused longer rows (pandas' parse is unchanged), recorded here:
        rows of 1 to 4 fields, an empty 4th, a byte-order mark and CRLF, quoted fields with commas and newlines, blank
        and whitespace-only lines, NA, nan and None kept as text, an empty file and a file of blank lines, from a path
        and from a text buffer; and streams of 3 and 4 elements."""
        from foldmatch.dataset.utils import read_chain_list
        for case, text, expected in (
                ("4 fields", "x,f.cif,A,x.A\ny,f.cif,B,named\n",
                 [["x", "f.cif", "A", "x.A"], ["y", "f.cif", "B", "named"]]),
                ("3, 2 and 1 fields and an empty 4th", "x,f.cif,A\ny,f.cif\nz\nw,f.cif,W,\n",
                 [["x", "f.cif", "A", "x.A"], ["y", "f.cif", "", "y."], ["z", "", "", "z."],
                  ["w", "f.cif", "W", "w.W"]]),
                ("a byte-order mark and CRLF", "\ufeffx,f.cif,A,x.A\r\ny,f.cif,B\r\n",
                 [["x", "f.cif", "A", "x.A"], ["y", "f.cif", "B", "y.B"]]),
                ("quoted fields with commas and newlines", 'x,"a,b",A,"n,1"\n"y\nz",f.cif,"B",\n',
                 [["x", "a,b", "A", "n,1"], ["y\nz", "f.cif", "B", "y\nz.B"]]),
                ("blank and whitespace-only lines", "\nx,,A\n\n  \n\t\ny,,B\n\n",
                 [["x", "", "A", "x.A"], ["y", "", "B", "y.B"]]),
                ("NA, nan and None", "NA,nan,None\nN/A,NaN,,null\n",
                 [["NA", "nan", "None", "NA.None"], ["N/A", "NaN", "", "null"]]),
                ("an empty file", "", []),
                ("a file of blank lines", "\n\n\n", [])):
            path = self.chain_list(case, text)
            for source in ("path", "text buffer"):
                with self.subTest(case=case, source=source):
                    self.assert_frame(read_chain_list(path if source == "path" else io.StringIO(text)), expected)
        for rows in ([("x", "f.cif", "A"), ("y", "f.cif", "B")], [("x", "f.cif", "A", "x.A"), ("y", "f.cif", "B", "")]):
            with self.subTest(stream=len(rows[0])):
                self.assert_frame(read_chain_list(rows, SrcLocation.stream),
                                  [["x", "f.cif", "A", "x.A"], ["y", "f.cif", "B", "y.B"]])


if __name__ == "__main__":
    unittest.main()
