import gzip
import os
from pathlib import Path

import numpy as np
from biotite.structure import filter_amino_acids, filter_polymer, get_chain_starts, get_residue_count, \
    filter_peptide_backbone, residue_iter, array
from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure, get_assembly as get_pdb_assembly, list_assemblies as list_pdb_assemblies
from biotite.structure.io.pdbx import CIFFile, get_structure, get_assembly, BinaryCIFFile, list_assemblies


def get_structure_from_src(
        src_structure,
        structure_format="mmcif",
        *,
        assembly_id=None
):
    try:
        __rewind(src_structure)
        if __is_path(src_structure) and Path(src_structure).suffix == ".gz":
            if structure_format == "mmcif" or structure_format == "pdb":
                src_structure = gzip.open(src_structure, "rt")
            elif structure_format == "binarycif":
                src_structure = gzip.open(src_structure, "rb")
            else:
                raise RuntimeError(f"Unknown file format {structure_format}")
        if structure_format == "pdb":
            pdb_file = PDBFile.read(src_structure) if assembly_id is None else _IndexedPDBFile.read(src_structure)
            structure = __get_pdb_structure(pdb_file, assembly_id)
        elif structure_format == "mmcif":
            cif_file = CIFFile.read(src_structure)
            structure = __get_structure(cif_file, assembly_id)
        elif structure_format == "binarycif":
            cif_file = BinaryCIFFile.read(src_structure)
            structure = __get_structure(cif_file, assembly_id)
        else:
            raise RuntimeError(f"Unknown file format {structure_format}")
    except Exception as e:
        raise RuntimeError(f"Error reading structure from {src_structure}: {e}")

    return structure


def protein_chain_iter(structure, min_res_n=0, chain_id=None):
    """Each protein chain of the structure, in order, as one AtomArray. A protein chain is the amino-acid atoms of the
    chain_iter pieces with one chain ID (in an assembly, of one copy: one sym_id), joined in file order, without the
    pieces whose amino acids are all HETATM records if it has amino acids in ATOM records; it counts if they hold
    polymer amino-acid residues, at least min_res_n of them. With chain_id, the protein chains of that chain ID alone,
    reading only its pieces."""
    return (atom_ch for atom_ch, _ in protein_chains_with_sizes(structure, min_res_n, chain_id))


def protein_chains_with_sizes(structure, min_res_n=0, chain_id=None):
    """protein_chain_iter's protein chains, each with its size: (AtomArray, the number of polymer amino-acid residues
    that min_res_n is tested against, counted piece by piece and summed over the pieces kept)."""
    starts = get_chain_starts(structure, add_exclusive_stop=True)
    if chain_id is not None:
        # The atoms with that chain ID, cut where its pieces end (a selection alone would join pieces that other
        # chains' records separate): a few milliseconds on the largest structures
        atoms = np.flatnonzero(structure.chain_id == chain_id)
        piece_start = np.zeros(structure.array_length() + 1, dtype=bool)
        piece_start[starts] = True
        structure = structure[atoms]
        starts = np.append(np.flatnonzero(piece_start[atoms]), len(atoms))
    return __protein_chains(structure, starts, min_res_n)


def get_protein_chains(structure, min_res_n=0):
    """The chain ID of each protein chain of the structure (protein_chain_iter): in an assembly, once per copy."""
    return tuple(str(atom_ch.chain_id[0]) for atom_ch in protein_chain_iter(structure, min_res_n))


def get_assemblies(structure, structure_format="mmcif"):
    try:
        __rewind(structure)
        if __is_path(structure) and Path(structure).suffix == ".gz":
            if structure_format == "mmcif" or structure_format == "pdb":
                structure = gzip.open(structure, "rt")
            elif structure_format == "binarycif":
                structure = gzip.open(structure, "rb")
            else:
                raise RuntimeError(f"Unknown file format {structure_format}")
        if structure_format == "pdb":
            return tuple(list_pdb_assemblies(PDBFile.read(structure)))
        elif structure_format == "mmcif":
            return tuple(list_assemblies(CIFFile.read(structure)).keys())
        elif structure_format == "binarycif":
            return tuple(list_assemblies(BinaryCIFFile.read(structure)))
        else:
            raise RuntimeError(f"Unknown file format {structure_format}")
    except Exception as e:
        raise RuntimeError(f"Error reading assemblies from {structure}: {e}")


def rename_atom_attr(atom_ch):
    return array([__rename_atom(a) for a in atom_ch])


def filter_residues(atom_ch):
    return atom_ch[filter_amino_acids(atom_ch)]


def get_backbone_atoms(atom_ch):
    return np.array([(lambda x: [a.coord for a in x])(r) for r in  residue_iter(atom_ch[filter_peptide_backbone(atom_ch)])])


def __protein_chains(structure, starts, min_res_n):
    # The protein chains, and their sizes, of the structure whose chain_iter pieces start at starts (exclusive stop)
    aa = filter_amino_acids(structure)
    # The amino-acid atoms, and those of ATOM records, of every piece at once: the thousands of water and ligand pieces
    # of a large structure cost no Python loop
    n_aa = np.diff(np.concatenate(([0], np.cumsum(aa)))[starts])
    n_atom = np.diff(np.concatenate(([0], np.cumsum(aa & ~structure.hetero)))[starts])
    pieces = np.flatnonzero(n_aa > 0)
    keys = structure.chain_id[starts[pieces]].tolist()
    if "sym_id" in structure.get_annotation_categories():
        # In an assembly, each copy of a chain is a chain of its own: one sym_id (get_structure_from_src gives each copy
        # of a chain its own, in PDB files too)
        keys = list(zip(keys, structure.sym_id[starts[pieces]].tolist()))
    chains = {}
    for key, k in zip(keys, pieces.tolist()):
        chains.setdefault(key, []).append(k)
    # A piece whose amino acids are all HETATM records (a free amino-acid ligand) is left out if the chain has amino
    # acids in ATOM records, and a chain made only of HETATM amino acids keeps them all; the chains come in the order
    # of their first piece kept
    for kept in sorted(([k for k in ch if n_atom[k] > 0] or ch for ch in chains.values()), key=lambda kept: kept[0]):
        # Polymer amino-acid residues counted piece by piece, as each piece was counted before
        n_pol = 0
        for k in kept:
            piece = structure[starts[k]:starts[k + 1]]
            piece = piece[filter_polymer(piece)]
            n_pol += get_residue_count(piece[filter_amino_acids(piece)])
        if n_pol > 0 and n_pol >= min_res_n:
            atoms = np.concatenate([starts[k] + np.flatnonzero(aa[starts[k]:starts[k + 1]]) for k in kept])
            yield structure[atoms], n_pol


class _IndexedPDBFile(PDBFile):
    """A PDB file read for an assembly: its structure holds each atom's index in the asymmetric unit (au_index), which
    biotite's get_assembly keeps, as it builds the assembly from get_structure: the atoms of each REMARK 350 list, in
    file order, once for each of its operators."""

    def get_structure(self, *args, **kwargs):
        structure = super().get_structure(*args, **kwargs)
        structure.set_annotation("au_index", np.arange(structure.array_length()))
        return structure


def __get_pdb_structure(pdb_file, assembly_id=None):
    if assembly_id is None:
        return get_pdb_structure(
            pdb_file,
            model=1
        )
    assembly = get_pdb_assembly(
        pdb_file,
        assembly_id=assembly_id,
        model=1
    )
    # biotite numbers the copies (sym_id) of each REMARK 350 list ('APPLY THE FOLLOWING TO CHAINS') from 0, so a chain
    # in two lists has two copies with one sym_id. The index in the asymmetric unit (_IndexedPDBFile) rises within a
    # copy, and two copies of a chain repeat its atoms, so it stops rising between them: counting where it stops gives
    # each copy of a chain a sym_id of its own (biotite's, in an assembly of one list)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(assembly.au_index) <= 0) + 1, [assembly.array_length()]))
    assembly.set_annotation("sym_id", np.repeat(np.arange(len(starts) - 1), np.diff(starts)))
    assembly.del_annotation("au_index")
    return assembly


def __rename_atom(atom):
    atom.chain_id = "A"
    atom.hetero = False
    if len(atom.res_name) > 3:
        atom.res_name = 'UNK'
    return atom


def __get_structure(cif_file, assembly_id=None):
    return get_structure(
        cif_file,
        model=1,
        use_author_fields=False
    ) if assembly_id is None else get_assembly(
        cif_file,
        assembly_id=assembly_id,
        model=1,
        use_author_fields=False
    )

def __rewind(src):
    # A stream (a downloaded file) may be read more than once: its assemblies, then each assembly.
    seekable = getattr(src, "seekable", None)
    if seekable is not None and seekable():
        src.seek(0)


def __is_path(val):
    if isinstance(val, (str, os.PathLike)):
        return True
    else:
        return False