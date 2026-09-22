"""Heads that predict Foldseek 3Di states from frozen ESM3 residue embeddings.

Vendored from ``rcsb-embedding-search/src/networks/three_di_head.py``, which is where
they are trained. Kept byte-compatible with that module's parameter names and shapes so a
checkpoint saved there loads here with ``strict=True``; the only changes are dropping the
repo-local import of the 3Di alphabet (inlined below) and the training-only helpers.

The stem is fixed for every architecture and matters:

* **LayerNorm on the input.** The stored embedding is the last block's output *before*
  ESM3's final LayerNorm, and it has massive-activation channels (dims 1361 and 1489 reach
  ~16,000 while the median is ~100). Feeding that to a linear layer unnormalized would let
  two channels dominate. ESM3's own heads read the normalized output, so this also matches
  how the backbone is used elsewhere.
* **Optional ss8 logits**, 11 numbers per residue from the same ESM3 pass, appended to the
  input when ``with_ss8`` is set.

``padding_mask`` is True where a position is padding, as in ``nn.Transformer``.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

#: Foldseek's 3Di states, in the order biotite's ``std_3di_matrix`` uses. Must stay in
#: step with :data:`foldmatch.search.scoring_system.THREE_DI`, which indexes the same
#: states in the substitution matrix.
THREE_DI = "acdefghiklmnpqrstvwy"
N_STATES = len(THREE_DI)
ESM3_DIM = 1536
SS8_DIM = 11


class ThreeDiHead(nn.Module):
    """Base class: input LayerNorm, projection, body, per-residue classifier."""

    def __init__(
            self,
            input_features: int = ESM3_DIM,
            hidden: int = 512,
            n_states: int = N_STATES,
            with_ss8: bool = False,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.with_ss8 = with_ss8
        self.n_states = n_states
        in_features = input_features + (SS8_DIM if with_ss8 else 0)
        self.norm = nn.LayerNorm(in_features)
        self.project = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_states)

    def _body(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(
            self,
            embedding: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            ss8: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_ss8:
            if ss8 is None:
                raise ValueError("head was built with with_ss8=True but no ss8 logits were given")
            embedding = torch.cat((embedding, ss8), dim=-1)
        elif ss8 is not None:
            # Silently ignoring them would make an ss8 ablation report a result it never ran.
            raise ValueError("ss8 logits were given but the head was built with with_ss8=False")
        x = self.dropout(self.project(self.norm(embedding)))
        x = self._body(x, padding_mask)
        return self.classifier(x)


class LinearThreeDiHead(ThreeDiHead):
    """Linear probe: how much 3Di is decodable from one residue's embedding alone."""

    def __init__(self, input_features: int = ESM3_DIM, hidden: int = 512,
                 with_ss8: bool = False, dropout: float = 0.0):
        super().__init__(input_features=input_features, hidden=hidden,
                         with_ss8=with_ss8, dropout=dropout)

    def _body(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return x


class CnnThreeDiHead(ThreeDiHead):
    """Two 1D convolutions - the architecture published 3Di heads use."""

    def __init__(self, input_features: int = ESM3_DIM, hidden: int = 512,
                 kernel_size: int = 7, num_layers: int = 2, with_ss8: bool = False,
                 dropout: float = 0.1):
        super().__init__(input_features=input_features, hidden=hidden,
                         with_ss8=with_ss8, dropout=dropout)
        if kernel_size % 2 == 0:
            # padding=kernel_size//2 only preserves the length for odd kernels.
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

    def _body(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.layer_norms):
            # Zero the padding before convolving: a kernel reaching past the end of a
            # chain must not pick up another sample's values or stale activations.
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            x = norm(x + layer(x.transpose(1, 2)).transpose(1, 2))
        return x


class TransformerThreeDiHead(ThreeDiHead):
    """A small transformer encoder over the frozen embeddings.

    ``nhead`` cannot be recovered from the saved parameters - ``nn.MultiheadAttention``
    packs ``in_proj_weight`` as ``(3*hidden, hidden)`` whatever it is - so a wrong value
    loads cleanly and predicts nonsense. That is why the bundle records the architecture
    rather than letting the loader guess it.
    """

    def __init__(self, input_features: int = ESM3_DIM, hidden: int = 256,
                 nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 1024,
                 with_ss8: bool = False, dropout: float = 0.1):
        super().__init__(input_features=input_features, hidden=hidden,
                         with_ss8=with_ss8, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _body(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        # A fully masked row comes back as NaN; keep it finite.
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


#: Every architecture a saved bundle may name.
HEADS = {
    "LinearThreeDiHead": LinearThreeDiHead,
    "CnnThreeDiHead": CnnThreeDiHead,
    "TransformerThreeDiHead": TransformerThreeDiHead,
}


def build_head(architecture: dict) -> ThreeDiHead:
    """Instantiate a head from a bundle's ``architecture`` block.

    ``dropout`` is forced to 0: the bundle records the value the head was *trained* with,
    which is a training hyper-parameter, and inference always runs the head in eval mode.
    Keeping it in the record and ignoring it here means the bundle documents the run
    without letting a training knob change a served prediction.
    """
    architecture = dict(architecture)
    name = architecture.pop("class", None)
    if name not in HEADS:
        raise ValueError(f"unknown head class {name!r}; have {sorted(HEADS)}")
    architecture.pop("dropout", None)
    return HEADS[name](dropout=0.0, **architecture)
