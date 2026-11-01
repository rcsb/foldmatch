import os
import re
from urllib.parse import urlparse

import pandas as pd

from foldmatch.errors import RefusalError
from foldmatch.types.api_types import SrcLocation, StructureLocation

#: The columns of a chain list (read_chain_list): structure name | structure file or URL | chain ID | embedding name.
CHAIN_LIST_COLUMNS = ['stream_name', 'stream', 'chain_id', 'item_name']
#: The fields of a chain list row, as read_chain_list's refusals name them.
_CHAIN_LIST_FIELDS = "structure name, structure file or URL, chain ID, embedding name"
_AT_MOST_4_FIELDS = f"A chain list row holds at most 4 fields ({_CHAIN_LIST_FIELDS})"
_REMOVE_THE_EXTRA_FIELDS = "remove the extra fields (a comma after the embedding name makes a 5th)"


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


def read_chain_list(src_stream, src_location: SrcLocation = SrcLocation.file) -> pd.DataFrame:
    """The rows of a chain list, in input order, one per chain, as strings (CHAIN_LIST_COLUMNS): the chain lists of
    inference residue-embedding, structure-embedding, chain-embedding and complete-embedding, and of EsmProtFromChain.

    A file is CSV without a header, read with pandas, a missing or empty field read as ''; a stream is a list of
    tuples, a short one read as a short line. The embedding name names what is computed for the row; when it is missing
    or empty, it is <structure name>.<chain ID>, the name under which the assembly readers look for a chain's residue
    embedding. A row of more than 4 fields is refused (RefusalError, naming the file or the stream and the row), and so
    is a file on which pandas raises any other ParserError (quoted): pandas reads the leading fields of a first row of
    more than 4 as the index, which shifts the columns of every row, and stops at a later one with a ParserError that
    names no file. Nothing else is refused: a blank chain ID (biotite reads one as '') is a chain ID.
    """
    if src_location == SrcLocation.stream:
        rows = [tuple(row) for row in src_stream]
        longer = [k for k, row in enumerate(rows, start=1) if len(row) > len(CHAIN_LIST_COLUMNS)]
        if longer:
            listed = ", ".join(str(k) for k in longer[:5]) + (f" and {len(longer) - 5} more" if len(longer) > 5 else "")
            raise RefusalError(f"The chain list stream: {len(longer)} row(s) of more than 4 elements (row(s) {listed}, "
                               f"counted from 1). {_AT_MOST_4_FIELDS}: remove the extra elements.")
        data = pd.DataFrame(
            [row + ("",) * (len(CHAIN_LIST_COLUMNS) - len(row)) for row in rows],
            dtype=str,
            columns=CHAIN_LIST_COLUMNS
        )
    else:
        source = _chain_list_source(src_stream)
        try:
            data = pd.read_csv(
                src_stream,
                header=None,
                index_col=None,
                keep_default_na=False,
                dtype=str,
                names=CHAIN_LIST_COLUMNS
            )
        except pd.errors.ParserError as e:
            message = " ".join(str(e).split())
            if re.search(r"Expected \d+ fields", message):
                raise RefusalError(f"{source}: a row has more than 4 fields (pandas: {message}). {_AT_MOST_4_FIELDS}: "
                                   f"{_REMOVE_THE_EXTRA_FIELDS}.") from e
            raise RefusalError(f"{source}: pandas could not read it as CSV ({message}). A chain list is CSV without a "
                               f"header, of at most 4 fields per row ({_CHAIN_LIST_FIELDS}): correct it where pandas' "
                               f"message points.") from e
        # pandas reads the leading fields of a first row of more than 4 as the index, one level each
        if not isinstance(data.index, pd.RangeIndex):
            raise RefusalError(f"{source}: its first row has {len(CHAIN_LIST_COLUMNS) + data.index.nlevels} fields. "
                               f"{_AT_MOST_4_FIELDS}: {_REMOVE_THE_EXTRA_FIELDS}.")
    unnamed = data['item_name'].isna() | (data['item_name'] == "")
    data.loc[unnamed, 'item_name'] = data['stream_name'][unnamed] + "." + data['chain_id'][unnamed]
    return data


def _chain_list_source(src_stream) -> str:
    """A chain list file as read_chain_list's refusals name it: its path, or the name of an open file; a text buffer is
    "The chain list"."""
    if isinstance(src_stream, (str, os.PathLike)):
        return os.fspath(src_stream)
    name = getattr(src_stream, "name", None)
    return name if isinstance(name, str) else "The chain list"


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
