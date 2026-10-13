"""Developer build of the native Smith-Waterman kernel.

    python -m foldmatch._native.build [--isa ISA ...] [--out DIR] [--cc CC]

compiles ``fmsw/fmsw.c`` into one shared library per ISA of this machine (``sse2``, ``avx2`` and ``generic`` on
x86_64; ``neon`` and ``generic`` on arm64; ``generic`` elsewhere), in place into ``src/foldmatch/_native/`` (the
libraries are gitignored). Rebuild after every C edit: :mod:`foldmatch.search.swkern` refuses a library whose
embedded source sha256 differs from the tree.

Every build uses exactly ``-O3 -std=c11 -fPIC -shared -ffp-contract=off -fno-fast-math -fvisibility=hidden``
plus the ISA flag (``-mavx2`` for avx2, ``-DFMSW_GENERIC`` for generic) and the injected source sha256; never
``-march=native`` and never ``-mfma``. Every x86_64 library is compiled for the plain x86-64 baseline
(``-march=x86-64 -mtune=generic`` before the ISA flag), whatever ``-march`` the compiler or a wrapper defaults to
(e.g. ``x86-64-v3`` distribution compilers, ``-march=znver3`` from the Cray ``cc`` wrapper): the sse2 and generic
libraries must not contain AVX2 code, and avx2 adds exactly ``-mavx2``.

:func:`compile_library` is the single compile step, also used by the packaging build hook. This module uses the
standard library only and can be loaded by file path (without importing the foldmatch package).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:  # imported as foldmatch._native.build
    from . import ISAS, NATIVE_DIR, SOURCE_DIR, library_filename, machine_isas, normalized_machine, source_sha256
except ImportError:  # loaded by file path (packaging build hook): load the sibling __init__.py the same way
    _spec = importlib.util.spec_from_file_location("_foldmatch_native_init", Path(__file__).with_name("__init__.py"))
    _init = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_init)
    ISAS, NATIVE_DIR, SOURCE_DIR = _init.ISAS, _init.NATIVE_DIR, _init.SOURCE_DIR
    library_filename, machine_isas, source_sha256 = _init.library_filename, _init.machine_isas, _init.source_sha256
    normalized_machine = _init.normalized_machine

BASE_FLAGS = ("-O3", "-std=c11", "-fPIC", "-shared", "-ffp-contract=off", "-fno-fast-math", "-fvisibility=hidden")
ISA_FLAGS = {"sse2": (), "avx2": ("-mavx2",), "neon": (), "generic": ("-DFMSW_GENERIC",)}
#: Target-architecture baseline, placed before the ISA flag (see the module docstring).
MACHINE_FLAGS = {"x86_64": ("-march=x86-64", "-mtune=generic")}
_ISA_MACHINE = {"sse2": "x86_64", "avx2": "x86_64", "neon": "arm64"}


class NativeBuildError(RuntimeError):
    """The C compiler is missing or failed."""


def find_compiler(cc: Optional[str] = None) -> str:
    """``cc`` argument, else ``$CC``, else the first of cc / clang / gcc on PATH."""
    cand = cc or os.environ.get("CC")
    if cand:
        exe = shlex.split(cand)[0]
        if shutil.which(exe) is None and not Path(exe).is_file():
            raise NativeBuildError(f"C compiler {cand!r} not found")
        return cand
    for name in ("cc", "clang", "gcc"):
        if shutil.which(name):
            return name
    raise NativeBuildError("no C compiler found (tried $CC, cc, clang, gcc)")


def machine_flags(isa: str, machine: Optional[str] = None) -> List[str]:
    """Baseline flags of the library's target architecture: the ISA's own architecture for sse2 / avx2 / neon, else
    ``machine`` (default: this machine; the packaging hook passes the wheel's target)."""
    target = _ISA_MACHINE.get(isa) or normalized_machine(machine)
    return list(MACHINE_FLAGS.get(target, ()))


def compile_command(isa: str, output: Path, *, cc: Optional[str] = None, source_dir: Optional[Path] = None,
                    extra_flags: Sequence[str] = (), machine: Optional[str] = None) -> List[str]:
    """The exact compiler command line for one ISA library (the source sha256 is injected as a macro). ``machine``
    is the target architecture of a generic library (default: this machine)."""
    if isa not in ISA_FLAGS:
        raise ValueError(f"unknown fmsw ISA {isa!r}; expected one of {', '.join(ISAS)}")
    src_dir = Path(source_dir) if source_dir is not None else SOURCE_DIR
    sha = source_sha256(src_dir)
    if sha is None:
        raise NativeBuildError(f"kernel source not found in {src_dir}")
    return [*shlex.split(find_compiler(cc)), *BASE_FLAGS, *machine_flags(isa, machine), *ISA_FLAGS[isa], *extra_flags,
            f'-DFMSW_SOURCE_SHA256="{sha}"', "-o", str(output), str(src_dir / "fmsw.c"), "-lm"]


def compile_library(isa: str, out_dir: Path, *, cc: Optional[str] = None, source_dir: Optional[Path] = None,
                    extra_flags: Sequence[str] = (), machine: Optional[str] = None, quiet: bool = False) -> Path:
    """Compile the ``isa`` library into ``out_dir`` and return its path.

    The library is written to a temporary file in ``out_dir`` and moved into place atomically, so a process that has
    the previous library loaded keeps a valid mapping and a concurrent loader never sees a partial file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / library_filename(isa)
    fd, tmp = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=out_dir)
    os.close(fd)
    try:
        cmd = compile_command(isa, Path(tmp), cc=cc, source_dir=source_dir, extra_flags=extra_flags,
                              machine=machine)
        if not quiet:
            print(" ".join(shlex.quote(c) for c in cmd), flush=True)
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            raise NativeBuildError(f"cannot run the C compiler: {e}") from e
        if res.returncode != 0:
            raise NativeBuildError(f"compiling {final.name} failed (exit {res.returncode}):\n"
                                   f"{res.stdout}{res.stderr}")
        os.chmod(tmp, 0o755)
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return final


def library_is_current(path: Path, source_dir: Optional[Path] = None) -> bool:
    """True when ``path`` exists and embeds the tree's source sha256 (checked on the file bytes, without loading)."""
    sha = source_sha256(source_dir)
    p = Path(path)
    return sha is not None and p.is_file() and sha.encode("ascii") in p.read_bytes()


def build(isas: Optional[Iterable[str]] = None, out_dir: Optional[Path] = None, *, cc: Optional[str] = None,
          extra_flags: Sequence[str] = (), only_if_stale: bool = False, quiet: bool = False) -> List[Path]:
    """Build the libraries of ``isas`` (default: every ISA of this machine) into ``out_dir`` (default: in place).
    With ``only_if_stale`` a library that already embeds the tree's source sha256 is kept."""
    out = Path(out_dir) if out_dir is not None else NATIVE_DIR
    paths = []
    for isa in (tuple(isas) if isas is not None else machine_isas()):
        target = out / library_filename(isa)
        if only_if_stale and library_is_current(target):
            paths.append(target)
            continue
        paths.append(compile_library(isa, out, cc=cc, extra_flags=extra_flags, quiet=quiet))
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m foldmatch._native.build", description=__doc__.split("\n\n")[0])
    ap.add_argument("--isa", action="append", choices=ISAS,
                    help="ISA to build (repeatable; default: every ISA of this machine: "
                         f"{', '.join(machine_isas())})")
    ap.add_argument("--out", type=Path, default=None, help=f"output directory (default: {NATIVE_DIR})")
    ap.add_argument("--cc", default=None, help="C compiler (default: $CC, else cc / clang / gcc)")
    ap.add_argument("--if-stale", action="store_true", help="skip libraries that already match the source")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        paths = build(a.isa, a.out, cc=a.cc, only_if_stale=a.if_stale, quiet=a.quiet)
    except NativeBuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"fmsw source sha256 {source_sha256()}")
    for p in paths:
        print(f"built {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
