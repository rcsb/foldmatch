"""Packaging of the native Smith-Waterman kernel (PLAN section 7, P3.2; test matrix: CI tier ``packaging``).

What is checked:
- wheel contents: every ``_fmsw_*`` library expected for the wheel's platform, built for that architecture, embedding
  the source sha256 of the shipped kernel source (= this tree), a non-pure ``py3-none-<platform>`` tag;
- sdist contents: the default hatchling set (no include list) with the hook, the C source and the test fixtures,
  and no compiled library;
- stale libraries: the in-place libraries of this tree embed its source sha256;
- ``selftest()`` from the installed wheel in a clean venv, and the refusal of a pure wheel;
- the build hook: pure-wheel fallback without a C compiler and after a partial compile failure,
  ``FOLDMATCH_REQUIRE_NATIVE=1`` makes both fatal, editable builds compile in place, the platform tag.

Inputs (environment):
- ``FOLDMATCH_DIST_DIR``: directory holding the sdist and the wheel(s) to check (CI: ``python -m build``, or the
  cibuildwheel wheelhouse). When unset, the sdist and a wheel built from it are made here with the build python
  (``python -m build --no-isolation``) if it has ``build`` and ``hatchling``; otherwise those tests skip.
- ``FOLDMATCH_BUILD_PYTHON``: interpreter with ``build``, ``hatchling`` and ``editables`` for the hook tests (default:
  the interpreter running the tests).

Clean venvs are created from the interpreter running the tests with ``--system-site-packages`` (so they see numpy and
importlib-metadata) and get the wheel with ``pip install --no-deps --no-index``; they run outside the tree without
``PYTHONPATH``, and every check asserts that ``foldmatch`` was imported from the venv.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SRC = ROOT / "src" / "foldmatch" / "_native"
DIST_ENV = "FOLDMATCH_DIST_DIR"
BUILD_PYTHON_ENV = "FOLDMATCH_BUILD_PYTHON"
REQUIRE_ENV = "FOLDMATCH_REQUIRE_NATIVE"
NO_CC = "/nonexistent/foldmatch-no-cc"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the tree's library-discovery module, loaded by path: these tests never import the foldmatch package itself
TREE = _load_by_path("_packaging_test_native", NATIVE_SRC / "__init__.py")
TREE_SHA = TREE.source_sha256()
DP_GOLDEN = re.search(r'^DP_GOLDEN_SYNTH_SHA256 = "([0-9a-f]{64})"',
                      (ROOT / "src" / "foldmatch" / "search" / "swkern.py").read_text(), re.M).group(1)


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _run(cmd, *, env=None, cwd=None, timeout=600) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _build_python() -> str:
    return os.environ.get(BUILD_PYTHON_ENV) or sys.executable


def _has_modules(python: str, *modules: str) -> bool:
    return _run([python, "-c", "import " + ", ".join(modules)]).returncode == 0


_BUILD_ENV_DROP = ("PYTHONPATH", REQUIRE_ENV, "CC", "ARCHFLAGS", "MACOSX_DEPLOYMENT_TARGET", "_PYTHON_HOST_PLATFORM")


def _build_env(**overrides: Optional[str]) -> Dict[str, str]:
    """Environment of a build: this one without PYTHONPATH, the compiler choice, the native requirement and the macOS
    cross-build variables, plus ``overrides``."""
    env = {k: v for k, v in os.environ.items() if k not in _BUILD_ENV_DROP}
    for k, v in overrides.items():
        if v is not None:
            env[k] = v
    return env


def _clean_env() -> Dict[str, str]:
    """Environment of a clean-venv process: no PYTHONPATH, no user site, no ISA override."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FOLDMATCH_SW_ISA", "PYTHONHOME")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _parse_wheel_name(path: Path) -> Tuple[str, str, str, str, str]:
    """(name, version, python tag, abi tag, platform tag) of a wheel file name (no build tag in ours)."""
    parts = path.name[:-len(".whl")].split("-")
    assert len(parts) == 5, f"unexpected wheel file name {path.name}"
    return tuple(parts)


def _wheel_arch(plat: str) -> str:
    """``x86_64`` or ``arm64`` from a wheel platform tag (all compressed tags must agree)."""
    archs = set()
    for p in plat.split("."):
        if p.endswith("x86_64"):
            archs.add("x86_64")
        elif p.endswith("aarch64") or p.endswith("arm64"):
            archs.add("arm64")
        else:
            archs.add(p)
    assert len(archs) == 1 and archs <= {"x86_64", "arm64"}, f"unsupported wheel platform {plat!r}"
    return archs.pop()


_ELF_MACHINE = {62: "x86_64", 183: "arm64"}
_MACHO_CPU = {0x01000007: "x86_64", 0x0100000C: "arm64"}


def _binary_kind(data: bytes) -> Tuple[str, str]:
    """(format, architecture) of a shared library: ('elf', arch) for ELF64 ET_DYN, ('macho', arch) for a 64-bit
    Mach-O MH_DYLIB; anything else fails."""
    if data[:4] == b"\x7fELF":
        assert data[4] == 2 and data[5] == 1, "not a 64-bit little-endian ELF"
        e_type, e_machine = int.from_bytes(data[16:18], "little"), int.from_bytes(data[18:20], "little")
        assert e_type == 3, f"ELF type {e_type} is not a shared object"
        return "elf", _ELF_MACHINE.get(e_machine, f"elf-machine-{e_machine}")
    if data[:4] == b"\xcf\xfa\xed\xfe":
        cpu, filetype = int.from_bytes(data[4:8], "little"), int.from_bytes(data[12:16], "little")
        assert filetype == 6, f"Mach-O file type {filetype} is not a dylib"
        return "macho", _MACHO_CPU.get(cpu, f"macho-cpu-{cpu:#x}")
    raise AssertionError(f"not an ELF64 / Mach-O 64 shared library (magic {data[:4]!r})")


def _installable_here(path: Path) -> bool:
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
    except ImportError:
        _n, _v, _py, _abi, plat = _parse_wheel_name(path)
        if plat == "any":
            return True
        machine = TREE.normalized_machine()
        mac = plat.startswith("macosx")
        return _wheel_arch(plat) == machine and mac == (sys.platform == "darwin")
    supported = set(sys_tags())
    return bool(set(parse_wheel_filename(path.name)[3]) & supported)


def _source_files_sha(files: Dict[str, bytes], tmp: Path) -> Optional[str]:
    """The kernel source sha256 (TREE.source_sha256 algorithm) of fmsw.h / fmsw.c given as bytes."""
    d = tmp / "fmsw_src"
    d.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    return TREE.source_sha256(d)


def _make_venv(where: Path) -> Path:
    for mod in ("numpy", "importlib_metadata"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} is not importable here; the clean venv borrows it through --system-site-packages")
    res = _run([sys.executable, "-m", "venv", "--system-site-packages", where])
    assert res.returncode == 0, res.stderr
    py = where / "bin" / "python"
    assert py.is_file()
    return py


def _install(py: Path, wheel: Path):
    # --ignore-installed: the venv sees the test interpreter's site-packages, which may hold a foldmatch of the same
    # version (then a plain install is a no-op and the probe would import that one)
    res = _run([py, "-m", "pip", "install", "--no-deps", "--no-index", "--ignore-installed", "--no-cache-dir",
                "--disable-pip-version-check", "--quiet", wheel], env=_clean_env())
    assert res.returncode == 0, res.stdout + res.stderr


_PROBE = r"""
import json, sys
import foldmatch
from foldmatch import _native
from foldmatch.search import swkern
out = {"file": foldmatch.__file__, "in_source_tree": _native.in_source_tree(), "source_sha256": _native.source_sha256(),
       "libraries": sorted(p.name for p in _native.NATIVE_DIR.glob("_fmsw_*"))}
try:
    out["selftest"] = swkern.selftest()
except swkern.KernelError as e:
    out["error"] = str(e)
print(json.dumps(out, default=str))
"""


def _probe(py: Path, cwd: Path, isa: Optional[str] = None) -> dict:
    env = _clean_env()
    if isa is not None:
        env["FOLDMATCH_SW_ISA"] = isa
    res = _run([py, "-c", _PROBE], env=env, cwd=cwd)
    assert res.returncode == 0, res.stdout + res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    venv = py.parent.parent.resolve()
    assert Path(out["file"]).resolve().is_relative_to(venv), f"foldmatch imported from {out['file']}, not {venv}"
    assert out["in_source_tree"] is False
    return out


# =====================================================================================================================
# the distribution under test
# =====================================================================================================================
@pytest.fixture(scope="module")
def dist_dir(tmp_path_factory) -> Path:
    given = os.environ.get(DIST_ENV)
    if given:
        d = Path(given).resolve()
        assert d.is_dir(), f"{DIST_ENV}={given} is not a directory"
        return d
    py = _build_python()
    if not _has_modules(py, "build", "hatchling"):
        pytest.skip(f"set {DIST_ENV} to a built dist, or {BUILD_PYTHON_ENV} to a python with build and hatchling")
    out = tmp_path_factory.mktemp("dist")
    res = _run([py, "-m", "build", "--no-isolation", "--outdir", out, ROOT], env=_build_env(**{REQUIRE_ENV: "1"}))
    assert res.returncode == 0, res.stdout + res.stderr
    return out


@pytest.fixture(scope="module")
def wheels(dist_dir) -> List[Path]:
    found = sorted(dist_dir.glob("foldmatch-*.whl"))
    if not found:
        pytest.skip(f"no foldmatch wheel in {dist_dir}")
    return found


@pytest.fixture(scope="module")
def sdist(dist_dir) -> Path:
    found = sorted(dist_dir.glob("foldmatch-*.tar.gz"))
    if not found:
        pytest.skip(f"no foldmatch sdist in {dist_dir}")
    assert len(found) == 1, f"several sdists in {dist_dir}: {found}"
    return found[0]


# =====================================================================================================================
# wheel contents
# =====================================================================================================================
def test_wheel_contents(wheels, tmp_path):
    for wheel in wheels:
        _name, version, py_tag, abi_tag, plat = _parse_wheel_name(wheel)
        assert (py_tag, abi_tag) == ("py3", "none"), wheel.name
        assert plat != "any", f"{wheel.name} is a pure wheel: the native kernel was not built"
        arch = _wheel_arch(plat)
        macos = plat.startswith("macosx")
        suffix = ".dylib" if macos else ".so"
        isas = TREE.machine_isas(arch)
        with zipfile.ZipFile(wheel) as z:
            names = z.namelist()
            assert len(names) == len(set(names)), f"duplicate entries in {wheel.name}"
            info = f"foldmatch-{version}.dist-info"
            meta = z.read(f"{info}/WHEEL").decode()
            assert "Root-Is-Purelib: false" in meta, meta
            assert sorted(re.findall(r"^Tag: (\S+)$", meta, re.M)) == \
                sorted(f"py3-none-{p}" for p in plat.split(".")), meta
            pkg = z.read(f"{info}/METADATA").decode()
            assert "Classifier: Operating System :: POSIX :: Linux" in pkg
            assert "Classifier: Operating System :: MacOS" in pkg
            assert "OS Independent" not in pkg

            libs = sorted(n for n in names if Path(n).name.startswith(("_fmsw_", "._fmsw_")))
            expected = sorted(f"foldmatch/_native/_fmsw_{isa}{suffix}" for isa in isas)
            assert libs == expected, f"{wheel.name}: libraries {libs}, expected {expected}"
            # auditwheel / delocate vendor external libraries here; the kernel needs none beyond libc / libm
            assert not [n for n in names if n.startswith(("foldmatch.libs/", "foldmatch/.dylibs/"))], wheel.name
            assert not [n for n in names if "__pycache__" in n or n.endswith((".pyc", ".tmp"))], wheel.name

            shipped = {n: z.read(f"foldmatch/_native/fmsw/{n}") for n in TREE.SOURCE_FILES}
            for n, data in shipped.items():
                assert data == (NATIVE_SRC / "fmsw" / n).read_bytes(), f"{wheel.name}: fmsw/{n} differs from the tree"
            for n in ("foldmatch/_native/__init__.py", "foldmatch/_native/build.py", "foldmatch/search/swkern.py"):
                assert z.read(n) == (ROOT / "src" / n).read_bytes(), f"{wheel.name}: {n} differs from the tree"
            sha = _source_files_sha(shipped, tmp_path / wheel.name)
            assert sha == TREE_SHA
            for lib in libs:
                data = z.read(lib)
                fmt, lib_arch = _binary_kind(data)
                assert fmt == ("macho" if macos else "elf") and lib_arch == arch, f"{lib}: {fmt} {lib_arch}"
                # stale-library check on the shipped binaries: compiled from exactly the shipped source
                embeds_sha, records_tmp = sha.encode("ascii") in data, b"/foldmatch-fmsw-" in data
                assert embeds_sha, f"{wheel.name}: {lib} does not embed source sha256 {sha}"
                assert not records_tmp, f"{wheel.name}: {lib} records its temporary build directory"


# =====================================================================================================================
# sdist contents
# =====================================================================================================================
def test_sdist_contents(sdist):
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    build = config.get("tool", {}).get("hatch", {}).get("build", {})
    for section in (build, build.get("targets", {}).get("sdist", {})):
        assert not {"include", "only-include"} & set(section), "the sdist must use hatchling's default file set"
    assert build["targets"]["wheel"]["hooks"]["custom"]["path"] == "hatch_build.py"

    with tarfile.open(sdist) as t:
        members = [m for m in t.getmembers() if m.isfile()]
        top = {m.name.split("/", 1)[0] for m in members}
        assert len(top) == 1, top
        prefix = top.pop() + "/"
        files = {m.name[len(prefix):]: m for m in members}

        def read(rel):
            return t.extractfile(files[rel]).read()

        required = ["pyproject.toml", "hatch_build.py", "README.md", "LICENSE.md", "PKG-INFO",
                    "src/foldmatch/_native/__init__.py", "src/foldmatch/_native/build.py",
                    "src/foldmatch/_native/fmsw/fmsw.c", "src/foldmatch/_native/fmsw/fmsw.h",
                    "src/foldmatch/search/swkern.py", "tests/conftest.py", "tests/test_swkern.py",
                    "tests/esm_codes_reference.py", "tests/test_packaging.py"]
        required += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "tests/resources/esm_codes").rglob("*"))
                     if p.is_file()]
        missing = [r for r in required if r not in files]
        assert not missing, f"missing from {sdist.name}: {missing}"
        for rel in ("pyproject.toml", "hatch_build.py", "src/foldmatch/_native/fmsw/fmsw.c",
                    "src/foldmatch/_native/fmsw/fmsw.h", "src/foldmatch/_native/build.py"):
            assert read(rel) == (ROOT / rel).read_bytes(), f"{sdist.name}: {rel} differs from the tree"
        compiled = [r for r in files if Path(r).name.startswith(("_fmsw_", "._fmsw_"))
                    or r.endswith((".so", ".dylib", ".pyc", ".o")) or "__pycache__" in r]
        assert not compiled, f"compiled files in {sdist.name}: {compiled}"
        pkg = read("PKG-INFO").decode()
        assert "Classifier: Operating System :: POSIX :: Linux" in pkg and "OS Independent" not in pkg


# =====================================================================================================================
# stale libraries (in place)
# =====================================================================================================================
@pytest.fixture
def tree_libs(request):
    """The in-place libraries of the tree (conftest ``fmsw_libs``), when foldmatch is importable (PYTHONPATH=src or an
    editable install)."""
    pytest.importorskip("foldmatch._native.build")
    return request.getfixturevalue("fmsw_libs")


@pytest.mark.usefixtures("tree_libs")
def test_in_place_libraries_match_the_tree():
    from foldmatch import _native
    from foldmatch._native import build as native_build
    from foldmatch.search import swkern
    if not _native.NATIVE_DIR.samefile(NATIVE_SRC):
        pytest.skip(f"foldmatch is imported from {_native.NATIVE_DIR}, not from this tree")
    assert _native.source_sha256() == TREE_SHA
    for isa in _native.machine_isas():
        path = _native.find_library(isa)
        assert path is not None, f"{isa} library missing after the developer build"
        assert native_build.library_is_current(path), f"{path} is stale"
    assert swkern.load().source_sha256 == TREE_SHA


# =====================================================================================================================
# selftest() from the installed wheel in a clean venv
# =====================================================================================================================
def test_wheel_selftest_in_clean_venv(wheels, tmp_path):
    here = [w for w in wheels if _installable_here(w)]
    if not here:
        pytest.skip(f"none of {[w.name for w in wheels]} installs on this machine")
    for wheel in here:
        py = _make_venv(tmp_path / f"venv-{wheel.stem}")
        _install(py, wheel)
        out = _probe(py, tmp_path)
        assert "error" not in out, out.get("error")
        report = out["selftest"]
        assert report["dp_golden_sha256"] == DP_GOLDEN
        assert report["s16_max_score_ratio"] <= 1.0
        kernel = report["kernel"]
        assert kernel["abi"] == 1 and kernel["source_sha256"] == TREE_SHA == out["source_sha256"]
        assert Path(kernel["path"]).resolve().is_relative_to(py.parent.parent.resolve())
        machine = TREE.normalized_machine()
        if machine == "arm64":
            assert kernel["isa"] == "neon"
        elif machine == "x86_64":
            assert kernel["isa"] in ("sse2", "avx2")
        # every other library of the wheel, through the validated override (avx2 may be refused on this CPU)
        for isa in TREE.machine_isas(machine):
            if isa == kernel["isa"]:
                continue
            o = _probe(py, tmp_path, isa=isa)
            if isa == "avx2" and "error" in o and "avx2 refused" in o["error"]:
                continue
            assert "error" not in o, o.get("error")
            assert (o["selftest"]["kernel"]["isa"], o["selftest"]["dp_golden_sha256"]) == (isa, DP_GOLDEN)


# =====================================================================================================================
# the build hook
# =====================================================================================================================
@pytest.fixture(scope="module")
def build_python() -> str:
    py = _build_python()
    if not _has_modules(py, "build", "hatchling"):
        pytest.skip(f"{BUILD_PYTHON_ENV} (default: this python) needs build and hatchling for the build-hook tests")
    return py


def _build_wheel(py: str, out: Path, **env: Optional[str]) -> subprocess.CompletedProcess:
    return _run([py, "-m", "build", "--wheel", "--no-isolation", "--outdir", out, ROOT], env=_build_env(**env))


def test_pure_wheel_without_a_compiler_and_its_refusal(build_python, tmp_path):
    out = tmp_path / "dist"
    res = _build_wheel(build_python, out, CC=NO_CC)
    assert res.returncode == 0, res.stdout + res.stderr
    log = res.stdout + res.stderr
    assert "FoldMatch native kernel not built" in log and TREE.REINSTALL_COMMAND in log, log
    (wheel,) = out.glob("*.whl")
    assert wheel.name.endswith("-py3-none-any.whl"), wheel.name
    with zipfile.ZipFile(wheel) as z:
        assert not [n for n in z.namelist() if "_fmsw_" in n]
        assert "Root-Is-Purelib: true" in z.read(next(n for n in z.namelist() if n.endswith(".dist-info/WHEEL"))).decode()
    py = _make_venv(tmp_path / "venv-pure")
    _install(py, wheel)
    o = _probe(py, tmp_path)
    assert o["libraries"] == []
    assert "not found" in o["error"] and "without its native kernel" in o["error"], o["error"]
    assert TREE.REINSTALL_COMMAND in o["error"], o["error"]


def test_require_native_makes_a_compile_failure_fatal(build_python, tmp_path):
    out = tmp_path / "dist"
    res = _build_wheel(build_python, out, CC=NO_CC, **{REQUIRE_ENV: "1"})
    assert res.returncode != 0
    assert f"{REQUIRE_ENV}=1" in res.stdout + res.stderr
    assert not list(out.glob("*.whl"))


def _cc_failing_for_generic(tmp_path: Path) -> str:
    """A C compiler wrapper that fails for the generic library only, which every machine with an ISA library builds
    last: the other libraries compile, then the build fails part-way."""
    real = next((shutil.which(c) for c in ("cc", "clang", "gcc") if shutil.which(c)), None)
    if real is None:
        pytest.skip("no C compiler")
    wrapper = tmp_path / "cc-failing-for-generic"
    wrapper.write_text("#!/bin/sh\n"
                       'for a in "$@"; do [ "$a" = "-DFMSW_GENERIC" ] && '
                       '{ echo "simulated compile failure (generic)" >&2; exit 1; }; done\n'
                       f'exec "{real}" "$@"\n')
    wrapper.chmod(0o755)
    return str(wrapper)


def test_partial_compile_failure(build_python, tmp_path):
    """One ISA fails after others compiled: without FOLDMATCH_REQUIRE_NATIVE the pure wheel and the warning (no
    library of the partial build), with it a clear build error."""
    cc = _cc_failing_for_generic(tmp_path)
    out = tmp_path / "dist"
    res = _build_wheel(build_python, out, CC=cc)
    log = res.stdout + res.stderr
    assert res.returncode == 0, log
    assert "FoldMatch native kernel not built" in log and "simulated compile failure" in log, log
    (wheel,) = out.glob("*.whl")
    assert wheel.name.endswith("-py3-none-any.whl"), wheel.name
    with zipfile.ZipFile(wheel) as z:
        assert not [n for n in z.namelist() if "_fmsw_" in n]
        wheel_meta = z.read(next(n for n in z.namelist() if n.endswith(".dist-info/WHEEL"))).decode()
        assert "Root-Is-Purelib: true" in wheel_meta
    out = tmp_path / "dist-required"
    res = _build_wheel(build_python, out, CC=cc, **{REQUIRE_ENV: "1"})
    log = res.stdout + res.stderr
    assert res.returncode != 0
    assert f"FoldMatch native kernel build failed and {REQUIRE_ENV}=1" in log, log
    assert "simulated compile failure" in log, log
    assert not list(out.glob("*.whl"))


def test_editable_build_compiles_in_place(build_python, tmp_path):
    if not _has_modules(build_python, "editables"):
        pytest.skip("the build python has no editables (hatchling's editable-build requirement)")
    proj = tmp_path / "proj"
    proj.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE.md", "hatch_build.py"):
        shutil.copy(ROOT / name, proj / name)
    shutil.copytree(ROOT / "src", proj / "src", ignore=shutil.ignore_patterns("_fmsw_*", "._fmsw_*", "__pycache__"))
    out = tmp_path / "dist"
    res = _run([build_python, "-m", "hatchling", "build", "-t", "wheel:editable", "-d", out], cwd=proj,
               env=_build_env(**{REQUIRE_ENV: "1"}))
    assert res.returncode == 0, res.stdout + res.stderr
    (wheel,) = out.glob("*.whl")
    with zipfile.ZipFile(wheel) as z:
        assert not [n for n in z.namelist() if "_fmsw_" in n], "the editable wheel must not carry libraries"
    native = proj / "src" / "foldmatch" / "_native"
    built = sorted(p.name for p in native.glob("_fmsw_*"))
    assert built == sorted(TREE.library_filename(isa) for isa in TREE.machine_isas())
    for name in built:
        current = TREE_SHA.encode("ascii") in (native / name).read_bytes()
        assert current, f"{name} does not embed source sha256 {TREE_SHA}"
    assert not list(native.glob("._fmsw_*")), "temporary build files left behind"


_TAG_PROBE = "import json, hatch_build; print(json.dumps(hatch_build.target_platform()))"


@pytest.mark.parametrize("env, expect", [
    ({}, "native"),
    ({"MACOSX_DEPLOYMENT_TARGET": "12.3"}, "macos-12"),
    ({"MACOSX_DEPLOYMENT_TARGET": "10.9"}, "macos-10.9"),
    ({"ARCHFLAGS": "-arch x86_64"}, "macos-x86_64"),
    ({"ARCHFLAGS": "-arch arm64 -arch x86_64"}, "macos-universal2"),
])
def test_hook_platform_tag(build_python, env, expect):
    if expect.startswith("macos") and sys.platform != "darwin":
        pytest.skip("macOS-only case")
    res = _run([build_python, "-c", _TAG_PROBE], cwd=ROOT, env=_build_env(**env))
    if expect == "macos-universal2":
        assert res.returncode != 0 and "universal2" in res.stderr, res.stderr
        return
    assert res.returncode == 0, res.stderr
    tag, machine, flags = json.loads(res.stdout)
    host = platform.machine()
    if sys.platform != "darwin":
        import sysconfig
        assert tag == re.sub(r"[^A-Za-z0-9_]", "_", sysconfig.get_platform()) and flags == []
        assert TREE.normalized_machine(machine) == TREE.normalized_machine()
        return
    if expect == "native":
        assert (tag, machine, flags) == (f"macosx_11_0_{host}", host, ["-mmacosx-version-min=11.0"])
    elif expect == "macos-12":
        assert (tag, flags) == (f"macosx_12_0_{host}", ["-mmacosx-version-min=12.3"])
    elif expect == "macos-10.9":
        assert tag == ("macosx_11_0_arm64" if host == "arm64" else "macosx_10_9_x86_64")
    elif expect == "macos-x86_64":
        assert tag == "macosx_11_0_x86_64" and machine == "x86_64"
        assert flags == (["-arch", "x86_64"] if host != "x86_64" else []) + ["-mmacosx-version-min=11.0"]
