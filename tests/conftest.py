import os

import pytest


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
