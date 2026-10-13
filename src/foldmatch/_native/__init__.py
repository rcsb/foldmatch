"""Native Smith-Waterman kernel (``fmsw``): library discovery.

The kernel source lives in ``fmsw/fmsw.c`` and ``fmsw/fmsw.h``. It is compiled into one shared library per
instruction set, named ``_fmsw_<isa><suffix>`` and loaded with ctypes by :mod:`foldmatch.search.swkern`:

==========  =====================  ======================================
ISA         built on               extra compile flags
==========  =====================  ======================================
``sse2``    x86_64                 none (the x86_64 baseline)
``avx2``    x86_64                 ``-mavx2`` (no ``-mfma``)
``neon``    aarch64 / arm64        none
``generic`` every platform         ``-DFMSW_GENERIC`` (vector extensions)
==========  =====================  ======================================

Every x86_64 library is compiled with ``-march=x86-64 -mtune=generic`` first, so a compiler that defaults to a newer
``-march`` cannot put AVX2 code into the sse2 or generic library (see :mod:`foldmatch._native.build`).

Wheels ship the libraries next to this file. Editable installs and source checkouts build them in place with
``python -m foldmatch._native.build`` (the libraries are gitignored). This module only locates files; it never
compiles and never imports numpy or torch.
"""
from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

NATIVE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = NATIVE_DIR / "fmsw"
SOURCE_FILES = ("fmsw.h", "fmsw.c")
ISAS = ("sse2", "avx2", "neon", "generic")

REBUILD_COMMAND = "python -m foldmatch._native.build"
REINSTALL_COMMAND = "pip install --no-cache-dir --force-reinstall --no-binary foldmatch foldmatch"


def library_suffix() -> str:
    """Shared-library suffix of this platform."""
    return ".dylib" if sys.platform == "darwin" else ".so"


def library_filename(isa: str) -> str:
    if isa not in ISAS:
        raise ValueError(f"unknown fmsw ISA {isa!r}; expected one of {', '.join(ISAS)}")
    return f"_fmsw_{isa}{library_suffix()}"


def normalized_machine(machine: Optional[str] = None) -> str:
    """``x86_64``, ``arm64`` or the raw lower-cased ``platform.machine()`` for anything else."""
    m = (machine or platform.machine() or "").lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if m in ("arm64", "aarch64", "armv8l", "arm64e"):
        return "arm64"
    return m


def machine_isas(machine: Optional[str] = None) -> Tuple[str, ...]:
    """The ISA libraries built (and loadable) on a machine, in dispatch-preference order."""
    m = normalized_machine(machine)
    if m == "x86_64":
        return ("sse2", "avx2", "generic")
    if m == "arm64":
        return ("neon", "generic")
    return ("generic",)


def find_library(isa: str, search_dirs: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """Path of the ``isa`` library in ``search_dirs`` (default: this package directory), or None."""
    name = library_filename(isa)
    for d in (search_dirs if search_dirs is not None else (NATIVE_DIR,)):
        p = Path(d) / name
        if p.is_file():
            return p
    return None


def source_sha256(source_dir: Optional[Path] = None) -> Optional[str]:
    """sha256 identifying the kernel source: over each of ``fmsw.h`` and ``fmsw.c`` (in that order) the bytes
    ``<name> NUL <decimal size> NUL <content>``. The build injects it into the library (``fmsw_source_sha256()``);
    a library whose value differs from the tree's is stale. None when the source files are not present."""
    d = Path(source_dir) if source_dir is not None else SOURCE_DIR
    h = hashlib.sha256()
    for name in SOURCE_FILES:
        p = d / name
        if not p.is_file():
            return None
        data = p.read_bytes()
        h.update(name.encode("ascii") + b"\0" + str(len(data)).encode("ascii") + b"\0")
        h.update(data)
    return h.hexdigest()


def rebuild_hint() -> str:
    """How to get working libraries, for refusal messages."""
    return (f"rebuild the native kernel: in a source checkout or editable install run `{REBUILD_COMMAND}` "
            f"(needs a C compiler); for a pip install from an sdist, reinstall with a C compiler available: "
            f"`{REINSTALL_COMMAND}`")
