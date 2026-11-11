import os
import re
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from foldmatch.errors import RefusalError
from foldmatch.types.api_types import SrcLocation, StructureLocation
from foldmatch.utils.output_files import shown

#: The columns of a chain list (read_chain_list): structure name | structure file or URL | chain ID | embedding name.
CHAIN_LIST_COLUMNS = ['stream_name', 'stream', 'chain_id', 'item_name']
#: The fields of a chain list row, as read_chain_list's refusals name them.
_CHAIN_LIST_FIELDS = "structure name, structure file or URL, chain ID, embedding name"
#: The columns of an assembly list (read_assembly_list): structure name | structure file or URL | assembly ID |
#: embedding name.
ASSEMBLY_LIST_COLUMNS = ['stream_name', 'stream', 'assembly_id', 'item_name']
#: The fields of an assembly list row, as read_assembly_list's refusals name them.
_ASSEMBLY_LIST_FIELDS = "structure name, structure file or URL, assembly ID, embedding name"
#: The columns of a structure list (read_structure_list): structure name | structure file or URL | embedding name.
STRUCTURE_LIST_COLUMNS = ['stream_name', 'stream', 'item_name']
#: The fields of a structure list row, as read_structure_list's refusals name them.
_STRUCTURE_LIST_FIELDS = "structure name, structure file or URL, embedding name"
#: The columns of a residue file list (read_residue_file_list): residue embedding file | embedding name.
RESIDUE_FILE_LIST_COLUMNS = ['file', 'item_name']
#: The fields of a residue file list row, as read_residue_file_list's refusals name them.
_RESIDUE_FILE_LIST_FIELDS = "residue embedding file, embedding name"
#: What a missing or empty embedding name is, per list, as refuse_repeated_names says it.
CHAIN_LIST_DEFAULT_NAME = "<structure name>.<chain ID>"
ASSEMBLY_LIST_DEFAULT_NAME = "<structure name>-<assembly ID>"
STRUCTURE_LIST_DEFAULT_NAME = "the structure name"
RESIDUE_FILE_LIST_DEFAULT_NAME = "the file's name without its extension"


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

    A file is CSV without a header, read with pandas, a missing or empty field read as '' (NA, nan or null are text);
    a stream is a list of tuples, a short one read as a short line. The embedding name names what is computed for the
    row; when it is missing or empty, it is <structure name>.<chain ID>, the name under which the assembly readers look
    for a chain's residue embedding. A row of more than 4 fields is refused (RefusalError, naming the file or the
    stream and the row), and so is a file on which pandas raises any other ParserError (quoted): pandas reads the
    leading fields of a first row of more than 4 as the index, which shifts the columns of every row, and stops at a
    later one with a ParserError that names no file. A blank chain ID (biotite reads one as '') is a chain ID. The
    readers refuse an embedding name given to more than one row (refuse_repeated_names).
    """
    if src_location == SrcLocation.stream:
        # A short row read as a short line
        src_stream = [tuple(row) + ("",) * (len(CHAIN_LIST_COLUMNS) - len(row)) for row in src_stream]
    data = _read_list(src_stream, src_location, CHAIN_LIST_COLUMNS, "chain list", _CHAIN_LIST_FIELDS)
    unnamed = data['item_name'].isna() | (data['item_name'] == "")
    data.loc[unnamed, 'item_name'] = data['stream_name'][unnamed] + "." + data['chain_id'][unnamed]
    return data


def read_assembly_list(src_stream, src_location: SrcLocation = SrcLocation.file) -> pd.DataFrame:
    """The rows of an assembly list, in input order, one per assembly (ASSEMBLY_LIST_COLUMNS): the assembly lists of
    inference assembly-embedding and complete-embedding, and of ResidueAssemblyEmbeddingFromTensorFile and
    ResidueAssemblyEmbeddingFromParquet.

    A file is CSV without a header, read with pandas as strings, a missing or empty field read as '' (NA, nan or null
    are text, as in a chain list); a stream is a list of tuples, read by pandas as such (a missing element is None).
    The embedding name names the row's output; when it is missing or empty, it is <structure name>-<assembly ID>, the
    name fm-embedding gives an assembly. A row of more than 4 fields is refused as read_chain_list refuses one
    (RefusalError, naming the file or the stream and the row), and so is a file on which pandas raises any other
    ParserError (quoted). The readers refuse an embedding name given to more than one row (refuse_repeated_names).
    """
    data = _read_list(src_stream, src_location, ASSEMBLY_LIST_COLUMNS, "assembly list", _ASSEMBLY_LIST_FIELDS)
    _name_unnamed(data, data['stream_name'].fillna("") + "-" + data['assembly_id'].fillna(""))
    return data


def read_structure_list(src_stream, src_location: SrcLocation = SrcLocation.file) -> pd.DataFrame:
    """The rows of a structure list, in input order, one per structure (STRUCTURE_LIST_COLUMNS): the file or stream of
    EsmProtFromStructure, EsmAssemblyFromStructure, ResidueEmbeddingFromStructure and
    ResidueAssemblyDatasetFromStructure.

    A file is CSV without a header, read with pandas as strings, a missing or empty field read as '' (NA, nan or null
    are text); a stream is a list of tuples, read by pandas as such (a missing element is None). The embedding name
    names the outputs of the structure's chains or assemblies (<embedding name>.<chain ID>, <embedding
    name>-<assembly ID>); when it is missing or empty, it is the structure name. A row of more than 3 fields is
    refused as read_chain_list refuses one (RefusalError, naming the file or the stream and the row), and so is a file
    on which pandas raises any other ParserError (quoted). The readers refuse an embedding name given to more than one
    row (refuse_repeated_names).
    """
    data = _read_list(src_stream, src_location, STRUCTURE_LIST_COLUMNS, "structure list", _STRUCTURE_LIST_FIELDS)
    _name_unnamed(data, data['stream_name'])
    return data


def read_residue_file_list(src_stream, src_location: SrcLocation = SrcLocation.file) -> pd.DataFrame:
    """The rows of a residue file list, in input order, one per residue embedding file (RESIDUE_FILE_LIST_COLUMNS): the
    file or stream of ResidueEmbeddingFromTensorFile.

    A file is CSV without a header, read with pandas as strings, so that a numeric-looking name keeps its spelling
    (007, 1.50), a missing or empty field read as '' (NA, nan or null are text); a stream is a list of tuples, read by
    pandas as such (a missing element is None). The embedding name names the row's output; when it is missing or
    empty, it is the residue embedding file's name without its extension (x for x.pt). A row of more than 2 fields is
    refused as read_chain_list refuses one (RefusalError, naming the file or the stream and the row), and so is a file
    on which pandas raises any other ParserError (quoted). The reader refuses an embedding name given to more than one
    row (refuse_repeated_names).
    """
    data = _read_list(src_stream, src_location, RESIDUE_FILE_LIST_COLUMNS, "residue file list",
                      _RESIDUE_FILE_LIST_FIELDS)
    _name_unnamed(data, data['file'].map(
        lambda f: os.path.splitext(os.path.basename(os.fspath(f)))[0] if isinstance(f, (str, os.PathLike)) else f))
    return data


def _name_unnamed(data: pd.DataFrame, default: pd.Series):
    """Each row whose embedding name is missing or empty gets its default (D5)."""
    unnamed = data['item_name'].isna() | (data['item_name'] == "")
    if unnamed.any():
        data.loc[unnamed, 'item_name'] = default[unnamed]


def _read_list(src_stream, src_location: SrcLocation, columns: list, name: str, fields: str) -> pd.DataFrame:
    """The rows of a list of these columns (its name and its fields as the refusals say them; the embedding name its
    last column), as pandas reads them: a file as CSV without a header, as strings; a stream as a list of rows. A row of
    more fields than columns is refused (RefusalError), and so is a file on which pandas raises any other ParserError:
    pandas reads the leading fields of a first row of more fields as the index, which shifts the columns of every row,
    and stops at a later one with a ParserError that names no file. A missing or empty field of a file is '', and NA,
    nan, null and the other words pandas takes for NA by default are text (keep_default_na=False)."""
    a_list = f"{'An' if name[0] in 'aeiou' else 'A'} {name}"
    rule = f"{a_list} row holds at most {len(columns)} fields ({fields})"
    next_field = "3rd" if len(columns) == 2 else f"{len(columns) + 1}th"
    remove_the_extra_fields = f"remove the extra fields (a comma after the embedding name makes a {next_field})"
    if src_location == SrcLocation.stream:
        rows = list(src_stream)
        longer = [k for k, row in enumerate(rows, start=1) if len(row) > len(columns)]
        if longer:
            listed = ", ".join(str(k) for k in longer[:5]) + (f" and {len(longer) - 5} more" if len(longer) > 5 else "")
            raise RefusalError(f"The {name} stream: {len(longer)} row(s) of more than {len(columns)} elements (row(s) "
                               f"{listed}, counted from 1). {rule}: remove the extra elements.")
        return pd.DataFrame(rows, dtype=str, columns=columns)
    source = _list_source(src_stream, name)
    try:
        data = pd.read_csv(
            src_stream,
            header=None,
            index_col=None,
            keep_default_na=False,
            dtype=str,
            names=columns
        )
    except pd.errors.ParserError as e:
        message = " ".join(str(e).split())
        if re.search(r"Expected \d+ fields", message):
            raise RefusalError(f"{source}: a row has more than {len(columns)} fields (pandas: {message}). {rule}: "
                               f"{remove_the_extra_fields}.") from e
        raise RefusalError(f"{source}: pandas could not read it as CSV ({message}). {a_list} is CSV without a header, "
                           f"of at most {len(columns)} fields per row ({fields}): correct it where pandas' message "
                           f"points.") from e
    # pandas reads the leading fields of a first row of more than len(columns) as the index, one level each. Read as
    # strings, they are never a RangeIndex; read without dtype=str, pandas 3 makes one of an index column of evenly
    # spaced integers (0, 1, 2).
    if not isinstance(data.index, pd.RangeIndex):
        raise RefusalError(f"{source}: its first row has {len(columns) + data.index.nlevels} fields. {rule}: "
                           f"{remove_the_extra_fields}.")
    return data


def _list_source(src_stream, name: str) -> str:
    """A list file as _read_list's refusals name it: its path, or the name of an open file; a text buffer is "The
    <name>" ("The chain list", say)."""
    if isinstance(src_stream, (str, os.PathLike)):
        return os.fspath(src_stream)
    file_name = getattr(src_stream, "name", None)
    return file_name if isinstance(file_name, str) else f"The {name}"


def list_source(src_stream, src_location: SrcLocation, name: str) -> str:
    """A list as refusals name it: the path or the name of an open file, "The <name>" for a text buffer, "The <name>
    stream" for a stream."""
    if src_location == SrcLocation.stream:
        return f"The {name} stream"
    return _list_source(src_stream, name)


def list_file(src_stream, src_location: SrcLocation) -> Optional[str]:
    """The path of a list given as a file (a path, or an open file's name), or None: an input of the run (D2)."""
    if src_location == SrcLocation.stream:
        return None
    if isinstance(src_stream, (str, os.PathLike)):
        return os.fspath(src_stream)
    file_name = getattr(src_stream, "name", None)
    return file_name if isinstance(file_name, str) and os.path.isfile(file_name) else None


def row_labels(data: pd.DataFrame) -> list:
    """The rows of a list's frame as refusals name them ("row 3", counted from 1 in input order): read_*_list's
    frames keep the input order as their index, sorted or not."""
    return [f"row {k + 1}" for k in data.index]


def refuse_repeated_names(data: pd.DataFrame, source: str, default: str, name_column: str = 'item_name'):
    """Refuse (RefusalError) a list that gives one embedding name to more than one row (D3), in any output format:
    csv and pt write each row's embedding to the file of its name, so a repeated name keeps the last one; Parquet and
    JSON store one row per name, which no reader can tell apart. Names are compared exactly here; the inference
    functions compare them as their output folder does. ``default`` says what an empty name is."""
    names = data[name_column]
    repeated = names[names.duplicated(keep=False) & names.notna()]
    if repeated.empty:
        return
    groups = sorted(((name, sorted(k + 1 for k in rows.index)) for name, rows in repeated.groupby(repeated)),
                    key=lambda group: group[1][0])

    def describe(group):
        name, rows = group
        return f"{name!r} (rows {', '.join(str(k) for k in rows[:-1])} and {rows[-1]})"
    raise RefusalError(
        f"{source}: {len(groups)} embedding name(s) given to more than one row: {shown(groups, describe, 'more names')}"
        f". Each row's embedding is written under its name (the file <name>.csv or <name>.pt, or the id of its row in "
        f"Parquet and JSON), so a repeated name keeps one of them: give each row its own name (a missing or empty name "
        f"is {default})."
    )


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
