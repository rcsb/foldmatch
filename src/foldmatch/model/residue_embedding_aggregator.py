
import torch.nn as nn
from collections import OrderedDict

from foldmatch.model.layers import ResBlock


class ResidueEmbeddingAggregator(nn.Module):
    dropout = 0.1

    def __init__(
            self,
            input_features=1536,
            dim_feedforward=3072,
            hidden_layer=1536,
            nhead=12,
            num_layers=6,
            res_block_layers=12
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if res_block_layers == 0:
            self.embedding = nn.Sequential(OrderedDict([
                ('norm', nn.LayerNorm(input_features)),
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ]))
        else:
            res_block = OrderedDict([(
                f'block{i}',
                ResBlock(input_features, hidden_layer, self.dropout)
            ) for i in range(res_block_layers)])
            res_block.update([
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ])
            self.embedding = nn.Sequential(res_block)

    def forward(self, x, x_mask=None):
        if x.dim() not in (2, 3):
            raise RuntimeError("Tensor dimension error. Allowed shapes (batch, sequence, residue_embeddings) or (sequence, residue_embeddings)")
        # PyTorch's nested-tensor fast path exists only on CPU and CUDA; elsewhere (Apple GPU)
        # its mask check calls aten::_nested_tensor_from_mask_left_aligned, which is not
        # implemented there, before the device is even checked. Skip it for the call.
        use_nested_tensor = self.transformer.use_nested_tensor
        if x.device.type not in ("cpu", "cuda"):
            self.transformer.use_nested_tensor = False
        try:
            x = self.transformer(x, src_key_padding_mask=x_mask)
        finally:
            self.transformer.use_nested_tensor = use_nested_tensor
        # The fast path returns padded rows as zeros; any other path leaves values there.
        # Zero them so every path sums the same rows (a no-op after the fast path).
        if x_mask is not None:
            x = x.masked_fill(x_mask.unsqueeze(-1), 0.0)
        return self.embedding(x.sum(dim=x.dim() - 2))
