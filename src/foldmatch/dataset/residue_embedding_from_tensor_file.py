import pandas as pd
from torch.utils.data import Dataset

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
        self.data = pd.DataFrame(
            src_stream,
            dtype=str,
            columns=ResidueEmbeddingFromTensorFile.COLUMNS
        ) if self.src_location == SrcLocation.stream else pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            names=ResidueEmbeddingFromTensorFile.COLUMNS
        )
        self.data = self.data.sort_values(by=self.data.columns[0])

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
