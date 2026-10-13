"""ESM3 residue-code encoders of the ``esm-codes-v1`` scoring system (method spec S2-S5).

A residue's ESM3 embedding (``per_residue_embedding`` rows 1..L, ``[L, 1536]``) becomes

* a **target code**, 64 sign bits packed into 8 bytes (``uint8[L, 8]``), stored in the database, and
* a **query encoding**, 64 fp16 numbers (``float16[L, 64]``), used at search time.

The arithmetic is part of the method, so it is reproduced cast for cast from the research encoders
(``rcsb-embedding-search`` ``sw_mlp/sw_common.py``):

* S2, zcw50_clean transform: ``x16 = fp16(x)``; ``xs = fp32(x16[:, dims]) - offset``; ``xw = fp32(fp16(xs @ M))``,
  with ``dims`` = 0..1535 without 1361 and 1489, ``offset`` fp32[1534] and ``M`` fp32[1534, 1534]. The whitener is
  **not** folded into the encoders' first layer: folding changes the rounding, and with it the codes.
* S3, encoders: ``f(x) = lin(x) + fc2(GELU_exact(fc1(x)))`` (1534 -> 1024 -> 64), float32 with no TF32, no autocast
  and no dropout. ``f_q`` encodes queries, ``f_t`` targets.
* S4, target code: ``bit_k = (f_t(xw)_k >= 0)``, packed MSB-first as ``numpy.packbits(bits, axis=1)``.
* S5, query encoding: ``q = fp16(f_q(xw))``.

The weights come from the bundle file ``esm_codes_v1_encoders.pt`` (HF repo ``rcsb/foldmatch-esm-codes``), read with
``torch.load(weights_only=True)`` by :func:`load_esm_codes_encoders`. Which revision and sha256 to load is the
scoring-system registry's business; this module only refuses files that are not a well-formed bundle of a known
architecture, and checks a sha256 when given one.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ENCODERS_FORMAT = "foldmatch-esm-codes-encoders/1"
DIGEST_METHOD = "foldmatch-tensor-digest/1"
CODE_BITS = 64
CODE_BYTES = CODE_BITS // 8
STATE_KEYS = ("lin.weight", "lin.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")

#: Every encoder architecture a bundle may name, exactly as the bundle stores it (lists, not tuples: that is what a
#: saved dict holds). A bundle's ``architecture`` must equal one of these entries; the shapes are then checked tensor
#: by tensor with ``strict=True``.
ARCHITECTURES = {
    "esm-codes-mlp-v1": {
        "name": "esm-codes-mlp-v1",
        "d_embed": 1536,
        "dims_dropped": [1361, 1489],
        "d_in": 1534,
        "hidden": 1024,
        "d_out": 64,
        "activation": "gelu_exact",
        "skip": "linear",
        "bias": True,
        "dropout": 0.0,
        "encoders": ["f_q", "f_t"],
    },
}

#: The parts of a bundle's ``conventions`` that this implementation hard-codes. A bundle that documents anything else
#: describes a different method and is refused rather than silently run with this module's arithmetic.
IMPLEMENTED_CONVENTIONS = {
    ("transform", "formula"): "x16 = fp16(x); xs = fp32(x16[:, dims]) - offset; xw = fp32(fp16(xs @ M))",
    ("transform", "input_cast"): "float16",
    ("transform", "output_cast"): "float16",
    ("transform", "folded_into_encoders"): False,
    ("encoder", "formula"): "f(x) = lin(x) + fc2(GELU_exact(fc1(x)))",
    ("encoder", "dtype"): "float32",
    ("code", "encoder"): "f_t",
    ("code", "bits"): CODE_BITS,
    ("code", "bytes"): CODE_BYTES,
    ("code", "sign_rule"): "bit_k = (h_k >= 0)",
    ("code", "bitorder"): "big",
    ("query", "encoder"): "f_q",
    ("query", "formula"): "q = fp16(f_q(xw))",
    ("query", "dtype"): "float16",
}


class EsmCodesBundleError(ValueError):
    """The file is not a usable esm-codes encoder bundle (tampered, wrong format, wrong architecture...)."""


# =====================================================================================================================
# Precision context: fp32 matmuls, no TF32 / bf16, no autocast
# =====================================================================================================================
_PRECISION_PARENTS = ("", "cudnn", "mkldnn")       # saved and restored, never set ("" = torch.backends itself)
_PRECISION_CHILDREN = ("cuda.matmul", "cudnn.conv", "cudnn.rnn", "mkldnn.matmul", "mkldnn.conv", "mkldnn.rnn")


def _backend(path: str):
    obj = torch.backends
    for part in filter(None, path.split(".")):
        obj = getattr(obj, part)
    return obj


def _fp32_precision_handles(paths):
    """Backends exposing the per-backend ``fp32_precision`` setting (torch >= 2.9); empty on older torch."""
    handles = []
    for path in paths:
        try:
            obj = _backend(path)
            obj.fp32_precision
        except (AttributeError, RuntimeError):
            continue
        handles.append(obj)
    return handles


def _read(getter):
    """(True, value), or (False, None) when torch refuses the read (legacy and new TF32 APIs mixed by the caller)."""
    try:
        return True, getter()
    except RuntimeError:
        return False, None


@contextlib.contextmanager
def strict_fp32(*device_types: str):
    """Run float32 matmuls in full IEEE fp32 and disable autocast, then restore the caller's settings exactly.

    Inside: ``torch.set_float32_matmul_precision("highest")``, cuBLAS and cuDNN TF32 off, and on torch >= 2.9 every
    per-backend ``fp32_precision`` (cuda.matmul, cudnn.conv/rnn, mkldnn.matmul/conv/rnn) set to ``"ieee"``, so
    neither TF32 nor oneDNN's bf16 path can be used; autocast is disabled for ``"cpu"`` and every ``device_types``.

    torch >= 2.9 keeps the legacy flags (``allow_tf32``, the matmul precision) and the new per-backend settings side
    by side and raises when they disagree. Both are therefore saved and restored, each through its own API; a legacy
    value that torch refuses to read (the caller mixed the APIs) is left untouched. Not thread-safe: these are global
    torch settings, as for any caller of ``set_float32_matmul_precision``.
    """
    parents = _fp32_precision_handles(_PRECISION_PARENTS)
    children = _fp32_precision_handles(_PRECISION_CHILDREN)
    saved_new = [(h, h.fp32_precision) for h in parents + children]
    ok_prec, prec = _read(torch.get_float32_matmul_precision)
    ok_cublas, cublas = (False, None) if children else _read(lambda: torch.backends.cuda.matmul.allow_tf32)
    ok_cudnn, cudnn = _read(lambda: torch.backends.cudnn.allow_tf32)
    try:
        if ok_prec:
            torch.set_float32_matmul_precision("highest")
        if ok_cublas:
            torch.backends.cuda.matmul.allow_tf32 = False
        if ok_cudnn:
            torch.backends.cudnn.allow_tf32 = False
        for h in children:
            h.fp32_precision = "ieee"
        with contextlib.ExitStack() as stack:
            for device_type in dict.fromkeys(("cpu",) + tuple(device_types)):
                try:
                    stack.enter_context(torch.autocast(device_type=device_type, enabled=False))
                except RuntimeError:  # a device type without autocast support has nothing to disable
                    pass
            yield
    finally:
        if ok_prec:
            torch.set_float32_matmul_precision(prec)
        if ok_cublas:
            torch.backends.cuda.matmul.allow_tf32 = cublas
        if ok_cudnn:
            torch.backends.cudnn.allow_tf32 = cudnn
        for h, value in reversed(saved_new):
            h.fp32_precision = value


# =====================================================================================================================
# Modules
# =====================================================================================================================
_BIT_WEIGHTS = (128, 64, 32, 16, 8, 4, 2, 1)


def packbits_rows(bits: torch.Tensor) -> torch.Tensor:
    """``numpy.packbits(bits, axis=1)`` on the tensor's own device: bool ``[L, 8k]`` -> uint8 ``[L, k]``, the first
    bit of each byte in its most significant position."""
    if bits.dim() != 2 or bits.shape[1] % 8:
        raise ValueError(f"expected a [L, 8k] bit matrix, got shape {tuple(bits.shape)}")
    rows, width = bits.shape
    weights = torch.tensor(_BIT_WEIGHTS, dtype=torch.int32, device=bits.device)
    return (bits.reshape(rows, width // 8, 8).to(torch.int32) * weights).sum(dim=2).to(torch.uint8)


class Zcw50Transform(nn.Module):
    """S2 with every cast explicit: ``xw = fp32(fp16(fp32(fp16(x)[:, dims]) - offset) @ M))``."""

    def __init__(self, dims: torch.Tensor, offset: torch.Tensor, M: torch.Tensor, d_embed: int):
        super().__init__()
        self.d_embed = int(d_embed)
        self.register_buffer("dims", dims)
        self.register_buffer("offset", offset)
        self.register_buffer("M", M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.d_embed:
            raise ValueError(f"expected per-residue embeddings [L, {self.d_embed}], got shape {tuple(x.shape)}")
        x16 = x.to(torch.float16)                                                # x16 = fp16(x)
        xs = x16.index_select(1, self.dims).to(torch.float32) - self.offset    # fp32(x16[:, dims]) - offset
        return (xs @ self.M).to(torch.float16).to(torch.float32)                 # fp32(fp16(xs @ M))


class CodeEncoder(nn.Module):
    """S3: ``f(x) = lin(x) + fc2(GELU_exact(fc1(x)))`` (the research ``sw_common.Encoder``, mlp with skip)."""

    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        return y + self.fc2(F.gelu(self.fc1(x)))                                # GELU: approximate="none"


class ResidueCodeHead(nn.Module):
    """The transform plus both encoders. Build it with :func:`load_esm_codes_encoders`.

    Both ``encode_*`` methods take one protein's embeddings ``[L, 1536]`` (any float dtype; S2 casts to fp16 first),
    move them to the head's device, and run under :func:`strict_fp32` and ``torch.no_grad()``. Encode one protein at a
    time: that is how the reference values were made, and a BLAS call's rounding may depend on the batch shape.
    """

    def __init__(self, transform: Zcw50Transform, f_q: CodeEncoder, f_t: CodeEncoder, meta: Optional[dict] = None):
        super().__init__()
        self.transform = transform
        self.f_q = f_q
        self.f_t = f_t
        self.meta = dict(meta or {})

    @property
    def device(self) -> torch.device:
        return self.transform.M.device

    def _check_dtypes(self):
        bad = [name for name, t in list(self.named_parameters()) + [("transform.offset", self.transform.offset),
                                                                      ("transform.M", self.transform.M)]
               if t.dtype != torch.float32]
        if bad or self.transform.dims.dtype != torch.int64:
            raise RuntimeError(f"residue-code head tensors are no longer float32 / int64 ({bad or ['dims']}); "
                               "the method requires the exact fp32 weights (do not call .half() / .to(dtype))")

    def _xw(self, x: torch.Tensor) -> torch.Tensor:
        self._check_dtypes()
        return self.transform(x.to(self.device))

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """S4: uint8 ``[L, 8]`` packed sign bits of ``f_t``, on the head's device."""
        with strict_fp32(self.device.type), torch.no_grad():
            return packbits_rows(self.f_t(self._xw(x)) >= 0)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        """S5: float16 ``[L, 64]`` query encoding ``fp16(f_q(xw))``, on the head's device."""
        with strict_fp32(self.device.type), torch.no_grad():
            return self.f_q(self._xw(x)).to(torch.float16)

    def content_digests(self) -> dict:
        """Content hashes of the transform and both encoders (``foldmatch-tensor-digest/1``), as bundle.json lists them
        under ``content_digests``; independent of the file container."""
        t = self.transform
        return {
            "transform_sha256": tensor_digest({"dims": t.dims, "offset": t.offset, "M": t.M}),
            "query_encoder_sha256": tensor_digest(self.f_q.state_dict()),
            "target_encoder_sha256": tensor_digest(self.f_t.state_dict()),
        }


# =====================================================================================================================
# Loading
# =====================================================================================================================
def tensor_digest(named) -> str:
    """``foldmatch-tensor-digest/1``: sha256 over the ASCII tag and a newline, then, for every name in sorted order, an
    8-byte little-endian length and the compact sorted-key JSON ``{"dtype", "name", "shape"}`` (numpy dtype string,
    little-endian), and an 8-byte little-endian length and the raw C-order little-endian bytes."""
    h = hashlib.sha256(DIGEST_METHOD.encode() + b"\n")
    for name in sorted(named):
        value = named[name]
        a = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        a = np.ascontiguousarray(a)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        head = json.dumps({"dtype": a.dtype.str, "name": name, "shape": list(a.shape)}, sort_keys=True,
                          separators=(",", ":")).encode()
        raw = a.tobytes(order="C")
        h.update(len(head).to_bytes(8, "little"))
        h.update(head)
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _tensor(obj, where: str, dtype: torch.dtype, shape: tuple) -> torch.Tensor:
    if not isinstance(obj, torch.Tensor):
        raise EsmCodesBundleError(f"{where}: expected a tensor, got {type(obj).__name__}")
    if obj.dtype != dtype or tuple(obj.shape) != tuple(shape):
        raise EsmCodesBundleError(f"{where}: expected {dtype} {list(shape)}, got {obj.dtype} {list(obj.shape)}")
    if obj.is_floating_point() and not bool(torch.isfinite(obj).all()):
        raise EsmCodesBundleError(f"{where}: non-finite values")
    return obj


def _encoder(state, where: str, arch: dict, device) -> CodeEncoder:
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise EsmCodesBundleError(f"{where}: expected a state_dict (a dict of tensors)")
    with torch.device("meta"):     # no random init (and no global RNG draw) for weights that are replaced anyway
        module = CodeEncoder(arch["d_in"], arch["hidden"], arch["d_out"])
    try:
        module.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as e:
        raise EsmCodesBundleError(f"{where}: does not match architecture {arch['name']!r}: {e}") from e
    for name, p in module.named_parameters():
        _tensor(p.data, f"{where}.{name}", torch.float32, tuple(p.shape))
    return module.to(device).requires_grad_(False).eval()


def head_from_bundle(bundle, source: str = "<bundle>", device="cpu") -> ResidueCodeHead:
    """Build the head from an already ``weights_only``-loaded bundle dict (see :func:`load_esm_codes_encoders`)."""
    if not isinstance(bundle, dict):
        raise EsmCodesBundleError(f"{source}: not an esm-codes encoder bundle (a {type(bundle).__name__})")
    if "format" not in bundle:
        kind = ("a bare state_dict" if bundle and all(isinstance(v, torch.Tensor) for v in bundle.values())
                else "a dict without a format tag")
        raise EsmCodesBundleError(f"{source}: {kind}, not an esm-codes encoder bundle ({ENCODERS_FORMAT})")
    if bundle["format"] != ENCODERS_FORMAT:
        raise EsmCodesBundleError(f"{source}: unknown format {bundle['format']!r}; this FoldMatch reads "
                                  f"{ENCODERS_FORMAT!r} (a newer bundle needs a newer FoldMatch)")
    arch = bundle.get("architecture")
    name = arch.get("name") if isinstance(arch, dict) else None
    if name not in ARCHITECTURES:
        raise EsmCodesBundleError(f"{source}: unknown encoder architecture {name!r}; known: {sorted(ARCHITECTURES)}")
    if arch != ARCHITECTURES[name]:
        diff = sorted(k for k in set(arch) | set(ARCHITECTURES[name]) if arch.get(k) != ARCHITECTURES[name].get(k))
        raise EsmCodesBundleError(f"{source}: architecture {name!r} differs from the known one in {diff}")
    conventions = bundle.get("conventions")
    if not isinstance(conventions, dict):
        raise EsmCodesBundleError(f"{source}: no conventions")
    for (section, key), want in IMPLEMENTED_CONVENTIONS.items():
        got = conventions.get(section, {}).get(key) if isinstance(conventions.get(section), dict) else None
        if got != want or type(got) is not type(want):
            raise EsmCodesBundleError(f"{source}: convention {section}.{key} = {got!r}; this implementation does "
                                      f"{want!r}")
    missing = [k for k in ("transform", "f_q", "f_t") if k not in bundle]
    if missing:
        raise EsmCodesBundleError(f"{source}: missing {missing}")
    tr = bundle["transform"]
    if not isinstance(tr, dict) or set(tr) != {"dims", "offset", "M"}:
        raise EsmCodesBundleError(f"{source}: transform must hold exactly dims, offset and M")
    d_in = arch["d_in"]
    dims = _tensor(tr["dims"], f"{source}: transform.dims", torch.int64, (d_in,))
    keep = np.setdiff1d(np.arange(arch["d_embed"]), np.asarray(arch["dims_dropped"]))
    if len(keep) != d_in or not np.array_equal(dims.cpu().numpy(), keep):
        raise EsmCodesBundleError(f"{source}: transform.dims is not 0..{arch['d_embed'] - 1} without "
                                  f"{arch['dims_dropped']}")
    offset = _tensor(tr["offset"], f"{source}: transform.offset", torch.float32, (d_in,))
    M = _tensor(tr["M"], f"{source}: transform.M", torch.float32, (d_in, d_in))
    transform = Zcw50Transform(dims.clone(), offset.clone(), M.clone(), arch["d_embed"])
    f_q = _encoder(bundle["f_q"], f"{source}: f_q", arch, device)
    f_t = _encoder(bundle["f_t"], f"{source}: f_t", arch, device)
    meta = {k: bundle.get(k) for k in ("format", "system", "architecture", "conventions", "provenance")}
    meta["source"] = source
    head = ResidueCodeHead(transform, f_q, f_t, meta).to(device)
    for module in head.modules():  # the architecture has none; inference never drops anything
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    return head.eval()


def load_esm_codes_encoders(path, *, sha256: Optional[str] = None, device="cpu") -> ResidueCodeHead:
    """Load ``esm_codes_v1_encoders.pt`` with ``torch.load(weights_only=True)`` and build the head.

    Refused (:class:`EsmCodesBundleError`): a sha256 other than ``sha256`` (when given), anything ``weights_only``
    cannot read, a bare state_dict, an unknown format tag, an unknown or altered architecture, conventions this module
    does not implement, and tensors of the wrong dtype, shape or with non-finite values. Parameters are loaded with
    ``strict=True``; the head comes back in eval mode on ``device`` with gradients off.
    """
    path = os.fspath(path)
    if sha256 is not None:
        got = _sha256_file(path)
        if got != sha256.lower():
            raise EsmCodesBundleError(f"{path}: sha256 {got} does not match the expected {sha256.lower()} "
                                      "(tampered, truncated or a different file)")
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001  (pickle / zip / unsafe-global errors all mean "not a usable bundle")
        raise EsmCodesBundleError(f"{path}: not readable with torch.load(weights_only=True): "
                                  f"{type(e).__name__}: {e}") from e
    return head_from_bundle(bundle, source=path, device=device)
