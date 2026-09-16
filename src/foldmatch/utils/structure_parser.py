import gzip
import os
from pathlib import Path

import numpy as np
from biotite.structure import filter_amino_acids, filter_polymer, chain_iter, get_chains, get_residues, \
    filter_peptide_backbone, residue_iter, array
from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure, get_assembly as get_pdb_assembly, list_assemblies as list_pdb_assemblies
from biotite.structure.io.pdbx import CIFFile, get_structure, get_assembly, BinaryCIFFile, list_assemblies


def get_structure_from_src(
        src_structure,
        structure_format="mmcif",
        chain_id=None,
        assembly_id=None
):
    try:
        if __is_path(src_structure) and Path(src_structure).suffix == ".gz":
            if structure_format == "mmcif" or structure_format == "pdb":
                src_structure = gzip.open(src_structure, "rt")
            elif structure_format == "binarycif":
                src_structure = gzip.open(src_structure, "rb")
            else:
                raise RuntimeError(f"Unknown file format {structure_format}")
        if structure_format == "pdb":
            pdb_file = PDBFile.read(src_structure)
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

    if chain_id is not None:
        return structure[structure.chain_id == chain_id]

    return structure


def get_protein_chains(structure, min_res_n=0):
    chain_ids = []
    for atom_ch in chain_iter(structure):
        atom_res = atom_ch[filter_polymer(atom_ch)]
        atom_res = atom_res[filter_amino_acids(atom_res)]
        if len(atom_res) > 0 and len(get_residues(atom_res)[0]) >= min_res_n:
            chain_ids.append(str(get_chains(atom_res)[0]))
    return tuple(chain_ids)


def get_assemblies(structure, structure_format="mmcif"):
    try:
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


def __get_pdb_structure(pdb_file, assembly_id=None):
    return get_pdb_structure(
        pdb_file,
        model=1
    ) if assembly_id is None else get_pdb_assembly(
        pdb_file,
        assembly_id=assembly_id,
        model=1
    )


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

def __is_path(val):
    if isinstance(val, (str, os.PathLike)):
        return True
    else:
        return False