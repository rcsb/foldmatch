import pandas as pd
from torch.utils.data import Dataset

from foldmatch.dataset.utils import RESIDUE_FILE_LIST_DEFAULT_NAME, list_source, read_residue_file_list, \
    refuse_repeated_names
from foldmatch.types.api_types import SrcLocation, ResEmbeddingFormat, FileOrStreamTuple
from foldmatch.utils.data import load_residue_embedding


class ResidueEmbeddingFromTensorFile(Dataset):

    FILE_ATTR = 'file'
    ITEM_NAME_ATTR = 'item_name'

    COLUMNS = [FILE_ATTR, ITEM_NAME_ATTR]

    def __init__(
            self,
            src_stream: FileOrStreamTuple,
            src_location: SrcLocation = SrcLocation.file,
            res_embedding_format: ResEmbeddingFormat = ResEmbeddingFormat.pt
    ):
        super().__init__()
        self.src_location = src_location
        self.res_embedding_format = res_embedding_format
        self.data = pd.DataFrame()
        self.__load_stream(src_stream)

    def __load_stream(self, src_stream):
        # The residue file list, whose columns are COLUMNS, as strings: a row of more than 2 fields is refused, a
        # missing or empty embedding name is the file's name without its extension, and an embedding name given to
        # more than one row is refused
        self.data = read_residue_file_list(src_stream, self.src_location)
        self.data = self.data.sort_values(by=self.data.columns[0])
        refuse_repeated_names(self.data, list_source(src_stream, self.src_location, "residue file list"),
                              RESIDUE_FILE_LIST_DEFAULT_NAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding_src = self.data.iloc[idx][ResidueEmbeddingFromTensorFile.FILE_ATTR]
        item_name = self.data.iloc[idx][ResidueEmbeddingFromTensorFile.ITEM_NAME_ATTR]
        try:
            embedding = load_residue_embedding(embedding_src, res_embedding_format=self.res_embedding_format)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No residue embedding file {embedding_src} for {item_name}: compute it first, or "
                                    f"check the folder and the format of the residue embeddings.") from e
        return embedding, item_name
