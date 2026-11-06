import os

import pytest

# Plain help and error text from every CLI, whatever the environment. Typer forces a terminal, and rich then writes
# colour codes, when GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set (as on GitHub Actions); the codes split the words
# that the tests look for ('--min-res-n'). TERMINAL_WIDTH, when set, would override the COLUMNS that the help tests
# give the CLI. typer reads both once, when typer.rich_utils is first imported (at the first help or error), so they
# are set here, before any test runs; the CLIs that tests run as subprocesses inherit them.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ.pop("TERMINAL_WIDTH", None)


@pytest.fixture(scope="session")
def fmsw_libs():
    """The native Smith-Waterman kernel libraries of this machine, built in place by the developer build
    (``python -m foldmatch._native.build``) when they are missing or stale. Without a C compiler the tests that need
    the kernel are skipped, unless FOLDMATCH_REQUIRE_NATIVE=1 (CI), which makes it a failure."""
    from foldmatch._native import build
    try:
        return build.build(only_if_stale=True, quiet=True)
    except build.NativeBuildError as e:
        if os.environ.get("FOLDMATCH_REQUIRE_NATIVE") == "1":
            pytest.fail(f"native kernel build failed: {e}")
        pytest.skip(f"native kernel unavailable (no C compiler?): {e}")
