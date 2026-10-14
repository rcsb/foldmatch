"""The esm-codes-v1 bundle for tests.

By default the bundle comes the way FoldMatch gets it: ``get_esm_codes_bundle(get(DEFAULT_ESM_CODES))``, i.e. the
registry entry's repo, revision and sha256s, downloaded into the Hugging Face cache once and verified on every call.
``FOLDMATCH_ESM_CODES_BUNDLE`` (a directory holding the bundle files) overrides it for the encoder tests, to test a
bundle before its upload.

Tests are skipped only when offline mode is on (``HF_HUB_OFFLINE=1``) and the bundle is not cached. Every other
failure is an error, never a skip: with offline mode off, a Hub that cannot be reached (an outage, a firewall, an
HTTP 5xx) fails the tests instead of turning them green without the acceptance checks, and a sha256 mismatch or a
file missing at the pinned revision always fails.
"""
import functools
import os
import unittest
from pathlib import Path

from huggingface_hub import constants as hf_constants

BUNDLE_ENV = "FOLDMATCH_ESM_CODES_BUNDLE"


@functools.lru_cache(maxsize=None)
def _fetch():
    """One fetch per test session: ``(files, None)``, or ``(None, error)`` (kept, so that a failing fetch is not
    repeated, with its network timeouts, by every test class)."""
    from foldmatch.search import scoring_system as ss
    from foldmatch.utils.model import get_esm_codes_bundle
    try:
        return get_esm_codes_bundle(ss.get(ss.DEFAULT_ESM_CODES)), None
    except Exception as e:  # noqa: BLE001  (classified by registry_files: skip only offline and not cached)
        return None, e


def registry_files():
    """``{file name: path}`` of the registry bundle (sha256-verified). ``unittest.SkipTest`` only when offline mode
    is on and the bundle is not cached; any other failure raises."""
    from foldmatch.search import scoring_system as ss
    files, error = _fetch()
    if error is None:
        return files
    if isinstance(error, ss.BundleUnavailableError) and hf_constants.HF_HUB_OFFLINE:
        raise unittest.SkipTest(f"offline (HF_HUB_OFFLINE=1) and the esm-codes-v1 bundle is not cached: {error}")
    raise RuntimeError(f"the esm-codes-v1 bundle could not be fetched ({type(error).__name__}); with offline mode "
                       f"off this is a test failure, not a skip: {error}") from error


def bundle_dir() -> Path:
    """The directory holding the bundle files: ``$FOLDMATCH_ESM_CODES_BUNDLE`` if set, else the registry bundle's
    Hugging Face snapshot directory."""
    d = os.environ.get(BUNDLE_ENV)
    if d:
        return Path(d)
    return Path(registry_files()["bundle.json"]).parent
