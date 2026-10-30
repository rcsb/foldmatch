import os
from urllib.parse import urlparse

import pandas as pd

from foldmatch.errors import RefusalError
from foldmatch.types.api_types import StructureLocation


def get_structure_location(s: str) -> str:
    # First, attempt to parse as URL
    parsed = urlparse(s)
    if parsed.scheme.lower() in {'http', 'https', 'ftp'} and parsed.netloc:
        return StructureLocation.remote

    # Next, test for an existing file or directory
    if os.path.exists(s):
        return StructureLocation.local

    # Neither URL nor existing file
    raise ValueError(f"Structure file source is neither a recognized URL nor file: {s!r}")

def refuse_shared_names(data: pd.DataFrame, name_column: str, source_column: str):
    """Refuse an input that gives one structure name to more than one source (file or URL).

    The name identifies the structure: StructureProvider keeps the last structure it read under the name (with the
    format and the assembly), so that the rows of one entry, one per chain or assembly, read it once, from a path or
    from a downloaded stream alike. A second source under the same name would get the first one's structure. One
    file under two spellings is one source (:func:`_source_key`). A row without a name or a source is left to fail
    where it is read, as before.
    """
    sources = data[[name_column, source_column]].dropna().astype(str).drop_duplicates()
    repeated = sources[sources.duplicated(name_column, keep=False)]
    conflicts = []
    for name, group in repeated.groupby(name_column, sort=True):
        if len({_source_key(source) for source in group[source_column]}) > 1:
            conflicts.append((name, sorted(group[source_column])))
    if conflicts:
        listed = "; ".join(f"'{name}' names {' and '.join(names)}" for name, names in conflicts[:5])
        more = f"; and {len(conflicts) - 5} more names" if len(conflicts) > 5 else ""
        raise RefusalError(
            f"A structure name must name one file or URL: {listed}{more}. The name identifies the structure, which is "
            f"read once for all of its rows (one per chain or assembly): give each file its own name, or, in a "
            f"folder of structures, keep one copy of each entry or choose one with the structure file extension "
            f"option."
        )


def _source_key(source: str):
    """A source as refuse_shared_names compares it: a URL with its scheme and host in lower case; an existing file by
    its device and inode, so that a relative path, a symbolic or hard link and, on a case-insensitive volume, a case
    variant are one file; any other path resolved."""
    parsed = urlparse(source)
    if parsed.scheme.lower() in {'http', 'https', 'ftp'} and parsed.netloc:
        return parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()).geturl()
    try:
        stat = os.stat(source)
    except OSError:
        return os.path.realpath(source)
    return stat.st_dev, stat.st_ino
