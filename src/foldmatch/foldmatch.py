import torch
from biotite.structure import get_residues, chain_iter, filter_amino_acids
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.structure.protein_chain import ProteinChain

from foldmatch.types.api_types import StreamSrc, StructureFormat
from foldmatch.utils.model import get_aggregator_model, get_residue_model
from foldmatch.utils.sequence import normalize_sequence
from foldmatch.utils.structure_parser import get_structure_from_src, filter_residues, rename_atom_attr


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
        structure = get_structure_from_src(src_structure, structure_format, chain_id, assembly_id)

        if self.max_res is not None and _protein_length(structure) > self.max_res:
            raise ValueError(f"Protein too long for embedding (max {self.max_res} residues)")

        embedding_ch = []
        for atom_ch in chain_iter(structure):
            atom_res = atom_ch[filter_amino_acids(atom_ch)]
            if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < self.min_res:
                continue
            protein_chain = ProteinChain.from_atomarray(atom_ch)
            protein = ESMProtein.from_protein_chain(protein_chain)
            protein_tensor = self.__residue_embedding.encode(protein)
            embedding_ch.append(self.__residue_embedding.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding)
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

        if self.max_res is not None and _protein_length(structure) > self.max_res:
            raise ValueError(f"Protein too long for embedding (max {self.max_res} residues)")

        embedding_by_chain = {}
        for atom_ch in chain_iter(structure):
            atom_ch = filter_residues(atom_ch)
            if len(atom_ch) == 0 or len(get_residues(atom_ch)[0]) < self.min_res:
                continue
            chain_id_value = atom_ch.chain_id[0]
            # Skip chains that don't match the requested chain_id (if specified)
            if chain_id is not None and chain_id_value != chain_id:
                continue
            atom_ch = rename_atom_attr(atom_ch)
            protein_chain = ProteinChain.from_atomarray(atom_ch)
            protein = ESMProtein.from_protein_chain(protein_chain)
            protein_tensor = self.__residue_embedding.encode(protein)
            embedding_by_chain[chain_id_value] = self.__residue_embedding.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding
        return embedding_by_chain

    def residue_embedding_by_assembly(
            self,
            src_structure: StreamSrc,
            structure_format: StructureFormat = StructureFormat.mmcif,
            assembly_id: str = None
    ):
        self.__check_residue_embedding()
        self.__check_aggregator_embedding()
        structure = get_structure_from_src(src_structure, structure_format, assembly_id=assembly_id)

        if self.max_res is not None and _protein_length(structure) > self.max_res:
            raise ValueError(f"Protein too long for embedding (max {self.max_res} residues)")

        embedding_ch = []
        for atom_ch in chain_iter(structure):
            atom_ch = filter_residues(atom_ch)
            if len(atom_ch) == 0 or len(get_residues(atom_ch)[0]) < self.min_res:
                continue
            atom_ch = rename_atom_attr(atom_ch)
            protein_chain = ProteinChain.from_atomarray(atom_ch)
            protein = ESMProtein.from_protein_chain(protein_chain)
            protein_tensor = self.__residue_embedding.encode(protein)
            embedding_ch.append(self.__residue_embedding.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            ).per_residue_embedding)
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
        protein_tensor = self.__residue_embedding.encode(protein)

        result = self.__residue_embedding.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )

        return result.per_residue_embedding

    def aggregator_embedding(
            self,
            residue_embedding: torch.Tensor
    ):
        self.__check_aggregator_embedding()
        return self.__aggregator_embedding(residue_embedding)

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


def _protein_length(structure):
    protein_atoms = structure[filter_amino_acids(structure)]
    return len(get_residues(protein_atoms)[0])
