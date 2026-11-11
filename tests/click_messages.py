"""Click's usage errors as patterns that do not depend on its wording, which changes between releases: an unknown
option is "No such option: --x" in click 8.3 and "No such option '--x'." in click 8.5."""
from __future__ import annotations

import re


def no_such_option(option: str) -> str:
    """A pattern of click's error for the unknown ``option``, in either wording."""
    name = re.escape(option)
    return rf"No such option(: {name}\b| '{name}')"
