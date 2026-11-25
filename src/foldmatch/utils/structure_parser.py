import gzip
import os
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from biotite import DeserializationError, InvalidFileError
from biotite.file import is_binary, is_text
from biotite.structure import filter_amino_acids, filter_polymer, get_chain_starts, get_residue_count, \
    filter_peptide_backbone, residue_iter, array, BadStructureError
from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure, get_assembly as get_pdb_assembly, list_assemblies as list_pdb_assemblies
from biotite.structure.io.pdbx import CIFFile, get_structure, get_assembly, BinaryCIFFile, list_assemblies

from foldmatch.errors import InternalError, RefusalError

STRUCTURE_FORMATS = ("mmcif", "pdb", "binarycif")

# What reading a file that is not a valid structure file of its format raises, in the calls made under _refusing alone
# (biotite's readers and the decompression and decoding below them). Measured on biotite 1.7.1 with truncated,
# byte-flipped and line-edited mmCIF, PDB and BinaryCIF files, and with damaged assembly records: biotite's own errors;
# ValueError (a number or hybrid-36 string that does not parse, no data block, invalid UTF-8, msgpack's unpack errors,
# all ValueErrors); OverflowError (an integer too large for its column); KeyError (a missing column, an unknown
# assembly ID); IndexError (an incomplete REMARK 300 in a PDB file); and, for .gz files, gzip's, zlib's and EOFError.
# Anything else (TypeError, AttributeError, MemoryError, an OSError other than a bad gzip stream) is not taken as the
# file's fault: it propagates unchanged.
_MALFORMED = (
    InvalidFileError, DeserializationError, BadStructureError,
    ValueError, LookupError, OverflowError,
    gzip.BadGzipFile, zlib.error, EOFError,
)
# A BinaryCIF file is msgpack, which biotite indexes without checking what it unpacked to: a file that unpacks to a
# value of another shape (a single byte unpacks to a number; a list; a map without its data blocks) raises TypeError
# or AttributeError. For binarycif these are the file's fault too: the stream's mode, the other source of a TypeError,
# is checked before the read (_check_source).
_MALFORMED_BINARYCIF = _MALFORMED + (TypeError, AttributeError)


def get_structure_from_src(
        src_structure,
        structure_format="mmcif",
        *,
        assembly_id=None
):
    """The structure (its first model) of a file, or of its assembly assembly_id: an AtomArray, in an assembly with
    each copy of a chain numbered by its sym_id.

    src_structure is a path (a str or a PathLike, decompressed if it ends in .gz) or an open stream, text for mmcif and
    pdb, binary for binarycif; a seekable stream is read from its start. assembly_id is a str.

    A file that is not a valid structure file of that format (malformed, truncated, of another format, without that
    assembly) is refused: RefusalError (a ValueError), with biotite's error as its __cause__. Everything else
    propagates unchanged, so that it is not mistaken for the input's fault: ValueError for an unknown
    structure_format; TypeError for a stream of the wrong mode, bytes, a str holding a file's content or an
    assembly_id that is not a str; OSError from opening a path (FileNotFoundError, PermissionError); MemoryError;
    InternalError for a failure in FoldMatch's own processing."""
    structure_format = _check_format(structure_format)
    _check_source(src_structure, structure_format)
    _check_assembly_id(assembly_id)
    with _opened(src_structure, structure_format) as src, _refusing("structure", src_structure, structure_format):
        # biotite's calls alone: only what they raise can be the file's fault
        if structure_format == "pdb":
            if assembly_id is None:
                structure = get_pdb_structure(PDBFile.read(src), model=1)
            else:
                structure = get_pdb_assembly(_IndexedPDBFile.read(src), assembly_id=assembly_id, model=1)
        else:
            cif_file = (CIFFile if structure_format == "mmcif" else BinaryCIFFile).read(src)
            if assembly_id is None:
                structure = get_structure(cif_file, model=1, use_author_fields=False)
            else:
                structure = get_assembly(cif_file, assembly_id=assembly_id, model=1, use_author_fields=False)

    if structure_format == "pdb" and assembly_id is not None:
        structure = _number_pdb_copies(structure)
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
    """The assembly IDs of a file (a path or a stream, as get_structure_from_src reads it), in file order. A file that
    is not a valid structure file of that format, or lists no assemblies, is refused (RefusalError); other errors
    propagate unchanged, as get_structure_from_src's do."""
    structure_format = _check_format(structure_format)
    _check_source(structure, structure_format)
    with _opened(structure, structure_format) as src, _refusing("assemblies", structure, structure_format):
        if structure_format == "pdb":
            return tuple(list_pdb_assemblies(PDBFile.read(src)))
        cif_file = (CIFFile if structure_format == "mmcif" else BinaryCIFFile).read(src)
        return tuple(list_assemblies(cif_file).keys())


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
        # biotite's reading, whose errors can be the file's fault (get_structure_from_src refuses them); then FoldMatch's
        # own step, which biotite's get_assembly runs from inside: a failure there is FoldMatch's, never the file's
        structure = super().get_structure(*args, **kwargs)
        with _internal("indexing the atoms of a PDB file's asymmetric unit"):
            structure.set_annotation("au_index", np.arange(structure.array_length()))
        return structure


def _number_pdb_copies(assembly):
    # biotite numbers the copies (sym_id) of each REMARK 350 list ('APPLY THE FOLLOWING TO CHAINS') from 0, so a chain
    # in two lists has two copies with one sym_id. The index in the asymmetric unit (_IndexedPDBFile) rises within a
    # copy, and two copies of a chain repeat its atoms, so it stops rising between them: counting where it stops gives
    # each copy of a chain a sym_id of its own (biotite's, in an assembly of one list)
    with _internal("numbering the copies of a PDB assembly"):
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


def _check_format(structure_format):
    # A format FoldMatch does not read is the caller's error, not the file's: a plain ValueError, never a refusal
    value = getattr(structure_format, "value", structure_format)  # a StructureFormat, or its value
    if value not in STRUCTURE_FORMATS:
        raise ValueError(f"Unknown structure format {value!r}: expected one of {', '.join(STRUCTURE_FORMATS)}")
    return value


def _check_source(src, structure_format):
    # The caller's mistakes in the source, before any reading: so that a TypeError biotite raises while reading never
    # means one of them. bytes would reach biotite, which opens them as a path inside _refusing (BinaryCIF content
    # would be refused as "embedded null byte"); a str with line breaks that names no file is a file's content passed
    # by mistake, which open() would reject with the whole content in its message
    if isinstance(src, (bytes, bytearray, memoryview)):
        raise TypeError("A structure source is a path (a str or a PathLike) or an open stream, not bytes: wrap a "
                        "file's content in io.BytesIO or io.StringIO")
    if isinstance(src, str):
        if ("\n" in src or "\r" in src) and not os.path.exists(src):
            raise TypeError("A structure file's content was given where a path or an open stream is expected")
        return
    if isinstance(src, os.PathLike):
        return
    # A stream, of the mode its format is read in (biotite's own test, made here rather than during the read)
    if structure_format == "binarycif":
        if not is_binary(src):
            raise TypeError("A file opened in 'binary' mode is required for binarycif")
    elif not is_text(src):
        raise TypeError(f"A file opened in 'text' mode is required for {structure_format}")


def _check_assembly_id(assembly_id):
    # An assembly ID of another type would be refused as one the file does not have ('1' is not 1)
    if assembly_id is not None and not isinstance(assembly_id, str):
        raise TypeError(f"assembly_id must be a str, not {type(assembly_id).__name__}")


@contextmanager
def _opened(src, structure_format):
    # The stream to read: a path opened here (decompressed if it ends in .gz, whatever its case) and closed after the
    # read, or the caller's stream, rewound and left open. Opening a path is not reading the file: its OSError
    # (FileNotFoundError, PermissionError, IsADirectoryError) propagates unchanged, outside _refusing
    mode = "rb" if structure_format == "binarycif" else "rt"
    if isinstance(src, (str, os.PathLike)):
        path = os.fspath(src)
        opener = gzip.open if Path(path).suffix.lower() == ".gz" else open
        with opener(path, mode) as f:
            yield f
    else:
        _rewind(src)
        yield src


@contextmanager
def _refusing(action, src, structure_format):
    # What the file's reading raises because the file is not a valid structure file, as a refusal naming the source
    malformed = _MALFORMED_BINARYCIF if structure_format == "binarycif" else _MALFORMED
    try:
        yield
    except RefusalError:
        raise
    except DeserializationError as e:
        # biotite parses a block, a category or a column when it is first used, and wraps whatever that raises in a
        # DeserializationError, without a cause: the error underneath says whether the file is at fault
        underneath = _underneath(e)
        if isinstance(underneath, MemoryError):
            raise underneath
        if underneath is not None and not isinstance(underneath, malformed):
            raise
        raise _refusal(action, src, e) from e
    except malformed as e:
        raise _refusal(action, src, e) from e
    except BaseException as e:
        # biotite's PDB reader is Rust: a non-ASCII character across one of its fixed columns makes it panic, which
        # pyo3 raises as a PanicException, a BaseException that no "except Exception" catches
        if type(e).__name__ == "PanicException" and type(e).__module__ == "pyo3_runtime":
            raise _refusal(action, src, e) from e
        raise


def _refusal(action, src, error):
    return RefusalError(f"Error reading {action} from {_describe(src)}: {str(error) or type(error).__name__}")


def _underneath(error):
    # The first error under a chain of biotite's DeserializationErrors, or None if one of them is the first
    seen = set()
    while isinstance(error, DeserializationError) and id(error) not in seen:
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return error


@contextmanager
def _internal(step):
    # A failure in FoldMatch's own processing of a structure that biotite read: an InternalError (a RuntimeError), so
    # that it is neither refused as the file's fault nor reported as a one-line error without its traceback. A
    # MemoryError is the machine's, not a FoldMatch bug: it propagates unchanged
    try:
        yield
    except (InternalError, MemoryError):
        raise
    except Exception as e:
        raise InternalError(f"FoldMatch failed {step}: {type(e).__name__}: {e}") from e


def _describe(src):
    # The source as a message names it: the path, the stream's file name, or the kind of stream (never its repr, which
    # holds a memory address, nor its content)
    if isinstance(src, (str, os.PathLike)):
        return str(os.fspath(src))
    try:
        name = getattr(src, "name", None)
    except Exception:  # noqa: BLE001  (a stream whose name cannot be read is named by its kind)
        name = None
    if isinstance(name, str) and name:
        return name
    return f"a {type(src).__name__} stream"


def _rewind(src):
    # A stream (a downloaded file) may be read more than once: its assemblies, then each assembly.
    seekable = getattr(src, "seekable", None)
    if seekable is not None and seekable():
        src.seek(0)