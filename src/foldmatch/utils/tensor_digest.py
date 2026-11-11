"""Content digests of named arrays (``foldmatch-tensor-digest/1``), independent of the file container.

numpy only (no torch), so the Stage-2 side can check the panel and the E-value table of an esm-codes bundle without
torch; :func:`foldmatch.model.residue_code_encoder.tensor_digest` applies the same digest to torch tensors.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

DIGEST_METHOD = "foldmatch-tensor-digest/1"


def tensor_digest(named: Mapping[str, object]) -> str:
    """sha256 over the ASCII tag ``foldmatch-tensor-digest/1`` and a newline, then, for every name in sorted order,
    an 8-byte little-endian length and the compact sorted-key JSON ``{"dtype", "name", "shape"}`` (numpy dtype
    string, little-endian), and an 8-byte little-endian length and the raw C-order little-endian bytes."""
    h = hashlib.sha256(DIGEST_METHOD.encode() + b"\n")
    for name in sorted(named):
        a = np.ascontiguousarray(np.asarray(named[name]))
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        head = json.dumps({"dtype": a.dtype.str, "name": name, "shape": list(a.shape)}, sort_keys=True,
                          separators=(",", ":")).encode()
        raw = a.tobytes(order="C")
        h.update(len(head).to_bytes(8, "little"))
        h.update(head)
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()
