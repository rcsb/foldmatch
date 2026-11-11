import torch
from esm.sdk.api import ESMProtein
from esm.utils.structure.protein_chain import ProteinChain

from foldmatch.types.api_types import StreamSrc, StructureFormat
from foldmatch.utils.esm.loaders import per_residue_embedding
from foldmatch.utils.model import get_aggregator_model, get_residue_model
from foldmatch.utils.sequence import normalize_sequence
from foldmatch.utils.structure_parser import get_structure_from_src, filter_residues, rename_atom_attr, \
    protein_chains_with_sizes


class FoldMatch:

    def __init__(self, min_res = 10, max_res = None):
        self.__residue_embedding = None
        self.__aggregator_embedding = None
        self.min_res = min_res
        self.max_res = max_res

    def load_models(
            self,
            device: torch.device = None
    ):
        self.load_residue_embedding(device)
        self.load_aggregator_embedding(device)

    def load_residue_embedding(
            self,
            device: torch.device = None
    ):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__residue_embedding = _load_res_model(device)

    def load_aggregator_embedding(
            self,
            device: torch.device = None
    ):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__aggregator_embedding = _load_model(device)

    def structure_embedding(
            self,
            src_structure: StreamSrc,
            structure_format: StructureFormat = StructureFormat.mmcif,
            chain_id: str = None,
            assembly_id: str = None
    ):
        res_embedding = self.residue_embedding(src_structure, structure_format, chain_id, assembly_id)
        return self.aggregator_embedding(res_embedding)

    def residue_embedding(
            self,
            src_structure: StreamSrc,
            structure_format: StructureFormat = StructureFormat.mmcif,
            chain_id: str = None,
            assembly_id: str = None
    ):
        self.__check_residue_embedding()
        structure = get_structure_from_src(src_structure, structure_format, assembly_id=assembly_id)

        # The protein chains of the whole structure, or those with chain_id (in an assembly, one per copy)
        embedding_ch = [self.__chain_embedding(atom_ch)
                        for atom_ch in self.__protein_chains(structure, chain_id, assembly_id)]
        if not embedding_ch:
            chain = "protein chain" if chain_id is None else f"protein chain {chain_id}"
            raise ValueError(f"No {chain} of at least {self.min_res} residues found in the structure")
        return torch.cat(
            embedding_ch,
            dim=0
        )

    def residue_embedding_by_chain(
            self,
            src_structure: StreamSrc,
            structure_format: StructureFormat = StructureFormat.mmcif,
            chain_id: str = None
    ):
        self.__check_residue_embedding()
        structure = get_structure_from_src(src_structure, structure_format)

        # Each protein chain once, or the one with chain_id
        return {atom_ch.chain_id[0]: self.__chain_embedding(atom_ch)
                for atom_ch in self.__protein_chains(structure, chain_id)}

    def residue_embedding_by_assembly(
            self,
            src_structure: StreamSrc,
            structure_format: StructureFormat = StructureFormat.mmcif,
            assembly_id: str = None
    ):
        self.__check_residue_embedding()
        self.__check_aggregator_embedding()
        structure = get_structure_from_src(src_structure, structure_format, assembly_id=assembly_id)

        # Each protein chain of the assembly, one per copy; none, an empty dict (as residue_embedding_by_chain gives)
        embedding_ch = [self.__chain_embedding(atom_ch)
                        for atom_ch in self.__protein_chains(structure, assembly_id=assembly_id)]
        if not embedding_ch:
            return {}
        assembly_id_value = assembly_id if assembly_id is not None else "0"
        return {assembly_id_value: torch.cat(embedding_ch, dim=0)}

    def sequence_embedding(
            self,
            sequence: str
    ):
        self.__check_residue_embedding()

        if sequence.startswith(">"):
            sequence = "".join(line.strip() for line in sequence.splitlines() if not line.startswith(">"))
        # The string the FASTA paths embed (S15): lowercase letters and whitespace
        # would otherwise reach ESM3 as unknown residues.
        sequence = normalize_sequence(sequence, allow_non_ascii=True)

        if len(sequence) < self.min_res:
            raise ValueError(f"Sequence too short for embedding (min {self.min_res} residues)")

        if self.max_res is not None and len(sequence) > self.max_res:
            raise ValueError(f"Sequence too long for embedding (max {self.max_res} residues)")

        protein = ESMProtein(sequence=sequence)
        return per_residue_embedding(self.__residue_embedding, protein)

    def aggregator_embedding(
            self,
            residue_embedding: torch.Tensor
    ):
        self.__check_aggregator_embedding()
        return self.__aggregator_embedding(residue_embedding)

    def __protein_chains(self, structure, chain_id=None, assembly_id=None):
        # The protein chains to embed (protein_chain_iter's), refused before any ESM3 pass when they have more than
        # max_res residues together, counted as min_res counts them; none is never refused (each method then does what
        # it does without max_res)
        chains = list(protein_chains_with_sizes(structure, self.min_res, chain_id))
        n_res = sum(size for _, size in chains)
        if self.max_res is not None and chains and n_res > self.max_res:
            n = len(chains)
            if chain_id is None:
                where = "the structure" if assembly_id is None else f"assembly {assembly_id}"
                counted = f"the {n} protein chain{'s' if n > 1 else ''} of {where}"
            elif assembly_id is None:
                counted = f"protein chain {chain_id}"
            else:
                counted = f"the {n} cop{'ies' if n > 1 else 'y'} of protein chain {chain_id} in assembly {assembly_id}"
            raise ValueError(f"Protein too long for embedding: {counted} {'have' if n > 1 else 'has'} {n_res} "
                             f"residues (max {self.max_res} residues)")
        return [atom_ch for atom_ch, _ in chains]

    def __chain_embedding(self, atom_ch):
        # One protein chain, read as the build reads it (EsmProtFromStructure, EsmAssemblyFromStructure)
        atom_ch = filter_residues(atom_ch)
        atom_ch = rename_atom_attr(atom_ch)
        protein_chain = ProteinChain.from_atomarray(atom_ch)
        protein = ESMProtein.from_protein_chain(protein_chain)
        return per_residue_embedding(self.__residue_embedding, protein)

    def __check_residue_embedding(self):
        if self.__residue_embedding is None:
            self.load_residue_embedding()

    def __check_aggregator_embedding(self):
        if self.__aggregator_embedding is None:
            self.load_aggregator_embedding()


def _load_model(device=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregator_model = get_aggregator_model(device=device)
    aggregator_model.to(device)
    aggregator_model.eval()
    return aggregator_model


def _load_res_model(device=None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return get_residue_model(device)
