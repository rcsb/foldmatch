"""Hatchling build hook: compile the native Smith-Waterman kernel ``fmsw`` into the wheel (PLAN section 7, P3.2).

Standard wheels
    Every ISA library of the target machine (``sse2``, ``avx2``, ``generic`` on x86_64; ``neon``, ``generic`` on
    arm64; ``generic`` elsewhere) is compiled into a fresh temporary directory, never reusing libraries built in place,
    and shipped through ``build_data["force_include"]`` as ``foldmatch/_native/_fmsw_<isa><suffix>``. The wheel is
    tagged ``py3-none-<platform>``: the sysconfig platform on Linux (``linux_x86_64``, ``linux_aarch64``), and on
    macOS ``macosx_<target>_<arch>`` with the deployment target ``$MACOSX_DEPLOYMENT_TARGET`` (default 11.0, passed
    to the compiler as ``-mmacosx-version-min``). No manylinux tag is written here: auditwheel (Linux) and delocate
    (macOS) check the libraries and retag the wheel.

Editable installs
    The libraries are compiled in place into ``src/foldmatch/_native/`` (gitignored), exactly as the developer build
    ``python -m foldmatch._native.build`` does; the editable wheel itself carries none.

Compile failures
    ``FOLDMATCH_REQUIRE_NATIVE=1`` (CI, Docker, cibuildwheel) makes a missing or failing C compiler fatal. Without it
    the hook warns and the build falls back to a pure-Python wheel (``py3-none-any``): BLOSUM62 works, and
    ``--scoring esm-codes`` refuses with the reinstall command (pip caches that wheel, hence ``--no-cache-dir``).

Every library embeds the sha256 of the kernel source it was compiled from (``fmsw_source_sha256()``), which the
loader compares with the installed source. The compile step itself is ``foldmatch._native.build.compile_library``,
loaded by file path because the build environment has neither numpy nor the foldmatch dependencies.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

REQUIRE_NATIVE_ENV = "FOLDMATCH_REQUIRE_NATIVE"
DEFAULT_MACOS_TARGET = "11.0"
NATIVE_SRC = Path("src", "foldmatch", "_native")      # relative to the project root
NATIVE_WHEEL = "foldmatch/_native"                     # inside the wheel


class _TargetError(RuntimeError):
    """The build target cannot get native libraries (e.g. a universal2 macOS build)."""


def load_native_build(root: str) -> ModuleType:
    """``src/foldmatch/_native/build.py`` of the project at ``root``, loaded by file path (standard library only)."""
    path = Path(root) / NATIVE_SRC / "build.py"
    spec = importlib.util.spec_from_file_location("_foldmatch_native_build_hook", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _macos_arch() -> str:
    """Target architecture of a macOS build: a single ``-arch`` in ``$ARCHFLAGS`` (cross builds, cibuildwheel), else
    the architecture in ``$_PYTHON_HOST_PLATFORM``, else this machine."""
    archs = sorted(set(re.findall(r"-arch\s+(\S+)", os.environ.get("ARCHFLAGS", ""))))
    if len(archs) > 1:
        raise _TargetError(f"ARCHFLAGS={os.environ['ARCHFLAGS']!r} asks for a multi-architecture (universal2) "
                           "build; the native kernel is built per architecture, so build one wheel per architecture")
    if archs:
        return archs[0]
    host = os.environ.get("_PYTHON_HOST_PLATFORM", "")
    if host.startswith("macosx-"):
        arch = host.rsplit("-", 1)[-1]
        if arch in ("arm64", "x86_64"):
            return arch
    return platform.machine()


def _macos_target(arch: str) -> Tuple[str, str]:
    """(``-mmacosx-version-min`` value, wheel tag version ``major_minor``) for the deployment target."""
    target = (os.environ.get("MACOSX_DEPLOYMENT_TARGET") or DEFAULT_MACOS_TARGET).strip()
    m = re.fullmatch(r"(\d+)(?:\.(\d+))?(?:\.\d+)?", target)
    if not m:
        raise _TargetError(f"MACOSX_DEPLOYMENT_TARGET={target!r} is not a macOS version")
    major, minor = int(m.group(1)), int(m.group(2) or 0)
    if arch == "arm64" and major < 11:
        target, major, minor = "11.0", 11, 0      # the first macOS on Apple silicon
    if major >= 11:
        minor = 0                                  # pip only knows macosx_<major>_0 tags from macOS 11 on
    return target, f"{major}_{minor}"


def target_platform() -> Tuple[str, str, List[str]]:
    """(wheel platform tag, target machine, extra compiler flags) of this build."""
    if sys.platform == "darwin":
        arch = _macos_arch()
        version_min, tag_version = _macos_target(arch)
        flags = [f"-mmacosx-version-min={version_min}"]
        if arch != platform.machine():
            flags[:0] = ["-arch", arch]            # cross build (clang); a native build needs no -arch (gcc has none)
        return f"macosx_{tag_version}_{arch}", arch, flags
    plat = sysconfig.get_platform()                # honours _PYTHON_HOST_PLATFORM
    tag = re.sub(r"[^A-Za-z0-9_]", "_", plat)
    machine = plat.split("-", 1)[1] if "-" in plat else platform.machine()
    return tag, machine, []


class FoldMatchNativeBuildHook(BuildHookInterface):
    """Compiles the fmsw kernel libraries for the wheel target (see the module docstring)."""

    PLUGIN_NAME = "custom"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._tmp: Optional[str] = None

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return
        required = os.environ.get(REQUIRE_NATIVE_ENV) == "1"
        try:
            native = load_native_build(self.root)
        except Exception as e:  # a broken tree is never acceptable, but report it like a compile failure
            self._failed(f"cannot load the kernel build module: {e}", required, None)
            return
        try:
            if version == "editable":
                self._build_in_place(native)
            else:
                self._build_into_wheel(native, build_data)
        except (native.NativeBuildError, _TargetError) as e:
            self._cleanup()
            self._failed(str(e), required, native)

    # -----------------------------------------------------------------------------------------------------------------
    def _build_in_place(self, native: ModuleType) -> None:
        out = Path(self.root) / NATIVE_SRC
        paths = native.build(out_dir=out, quiet=True)
        self.app.display_info(f"fmsw: built {', '.join(p.name for p in paths)} in place in {out} "
                              f"(source sha256 {native.source_sha256()})")

    def _build_into_wheel(self, native: ModuleType, build_data: Dict[str, Any]) -> None:
        plat_tag, machine, flags = target_platform()
        isas = native.machine_isas(machine)
        self._tmp = tempfile.mkdtemp(prefix="foldmatch-fmsw-")
        built = []
        for isa in isas:
            path = native.compile_library(isa, Path(self._tmp), extra_flags=flags, machine=machine, quiet=True)
            if not native.library_is_current(path):
                raise native.NativeBuildError(f"{path.name} does not embed the source sha256 of this tree")
            built.append(path)
        # build_data changes only once every library is built: a failure part-way (one ISA does not compile) leaves
        # it untouched, so the fallback is the pure wheel, never a wheel pointing at deleted temporary files
        for path in built:
            build_data["force_include"][str(path)] = f"{NATIVE_WHEEL}/{path.name}"
        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = f"py3-none-{plat_tag}"
        self.app.display_info(f"fmsw: built {', '.join(p.name for p in built)} for {machine} (tag {build_data['tag']}, "
                              f"compiler {native.find_compiler()}, source sha256 {native.source_sha256()})")

    def _failed(self, reason: str, required: bool, native: Optional[ModuleType]) -> None:
        reinstall = native.REINSTALL_COMMAND if native is not None else \
            "pip install --no-cache-dir --force-reinstall --no-binary foldmatch foldmatch"
        if required:
            raise RuntimeError(f"FoldMatch native kernel build failed and {REQUIRE_NATIVE_ENV}=1: {reason}")
        self.app.display_warning(
            f"FoldMatch native kernel not built ({reason}). Building a pure-Python package: BLOSUM62 search works, "
            f"but --scoring esm-codes will refuse to run. Install a C compiler and reinstall with `{reinstall}`; "
            f"set {REQUIRE_NATIVE_ENV}=1 to make this an error.")

    def _cleanup(self) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None

    def finalize(self, version: str, build_data: Dict[str, Any], artifact_path: str) -> None:
        self._cleanup()
