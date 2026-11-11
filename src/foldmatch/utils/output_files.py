"""The files of one run, compared as the folder that holds them compares names (P9.N).

One run never writes two of its outputs to one file, never writes an output onto one of its own inputs, and never
writes outside its output folder. The checks here refuse such a run (:class:`RefusalError`) before any model is built,
on every rank alike: they read the inputs' names and the folders, and write nothing but the probe of a folder (a
temporary file, removed at once). A file that exists because an earlier run wrote it is not their concern: a rerun into
the same folder replaces its own earlier files, as before.

Names are compared as the folder's file system compares them: exactly on a case-sensitive one; case-folded and
Unicode-normalised (canonical caseless matching) on a folder that takes 'A' and 'a', or the NFC and NFD spellings of
'é', for one name (macOS APFS by default), with full case folding ('ß' as 'ss') where the folder folds so (APFS does),
which :func:`probe_folding` finds out per folder. Folders are compared as files (device and inode), never as strings:
a symbolic link and, on macOS, ``SHARED`` and ``shared`` are one folder; and an input and an output's file that
exists are one file when either is a link to the other from another folder.
"""
import glob
import os
import re
import stat
import tempfile
import unicodedata
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

from foldmatch.errors import RefusalError

#: The longest file name, in bytes of UTF-8, that the usual file systems take (ext4, XFS, APFS, Lustre, GPFS).
MAX_FILE_NAME_BYTES = 255
#: The digits reserved for the rank in <stem>-<rank>.parquet and <stem>-<rank>.json.gz.
RANK_DIGITS = 6
#: The suffix of the one file per process of each format that writes one.
STEM_SUFFIXES = {"parquet": ".parquet", "json": ".json.gz"}
#: The formats that write one file per item, <name>.<format>.
ITEM_FORMATS = ("csv", "pt")
#: How many examples a refusal shows.
SHOWN = 5


def shown(items: Sequence, describe=str, more: str = "more") -> str:
    """The first SHOWN items, described and joined by '; ', and "and N more" for the rest."""
    listed = "; ".join(describe(item) for item in items[:SHOWN])
    return listed + (f"; and {len(items) - SHOWN} {more}" if len(items) > SHOWN else "")


def format_value(out_format) -> str:
    """An OutFormat (a str Enum) or its value, as its value."""
    return getattr(out_format, "value", out_format)


# =====================================================================================================================
# D1: a name that becomes a file name
# =====================================================================================================================
def _missing(name) -> bool:
    """Whether a name is missing: None, or NaN (pandas' missing value, numpy's or float's), or pandas.NA."""
    if name is None:
        return True
    try:
        return bool(name != name)
    except TypeError:   # pandas.NA, whose comparisons are NA
        return True


def name_problem(name, suffix: str) -> Optional[str]:
    """Why ``<name><suffix>`` cannot be one file of the output folder, or None: an empty or missing name (None, NaN),
    '.', '..', a name that holds '/' (or the system's separators) or NUL, or a file name of more than
    MAX_FILE_NAME_BYTES bytes in UTF-8. Another scalar, such as an integer id of a Parquet file, is judged as the
    writers spell it in the file name (``f"{name}"``)."""
    if _missing(name):
        return "is empty"
    if not isinstance(name, str):
        name = f"{name}"
    if name == "":
        return "is empty"
    if name in (".", ".."):
        return f"is {name!r}"
    for separator in dict.fromkeys(("/", os.sep, os.altsep)):
        if separator is not None and separator in name:
            return f"holds {separator!r}"
    if "\0" in name:
        return "holds a NUL character"
    size = len((name + suffix).encode("utf-8", "surrogatepass"))
    if size > MAX_FILE_NAME_BYTES:
        return f"makes a file name of {size} bytes"
    return None


def _usable_rule(suffix: str) -> str:
    return (f"that is not empty, without '/' or NUL, other than '.' and '..', of at most "
            f"{MAX_FILE_NAME_BYTES - len(suffix.encode())} bytes in UTF-8")


def refuse_unusable_names(names: Sequence, suffix: str, *, source: str, labels: Optional[Sequence] = None,
                          what: str = "embedding name", given: str = "those rows", pattern: Optional[str] = None):
    """Refuse (RefusalError) names that cannot name the file ``<name><suffix>`` in the output folder (name_problem),
    naming the first ones (with their label, "row 3", say, when given); ``pattern`` is the file as the message names
    it, ``<name><suffix>`` by default."""
    bad = []
    for k, name in enumerate(names):
        problem = name_problem(name, suffix)
        if problem is not None:
            bad.append((name, problem, None if labels is None else labels[k]))
    if bad:
        def describe(item):
            name, problem, label = item
            return f"{name!r}{'' if label is None else f' ({label})'} {problem}"
        raise RefusalError(
            f"{source}: {len(bad)} {what}(s) that cannot name a file {pattern or '<name>' + suffix} in the output "
            f"folder: {shown(bad, describe)}. Each {what} names its own file there: give {given} "
            f"{'an' if what[0] in 'aeiou' else 'a'} {what} {_usable_rule(suffix)}."
        )


def refuse_unusable_stem(stem, what: str = "The output name"):
    """Refuse (RefusalError) a name that cannot name the files ``<stem>-<rank>.parquet`` and ``<stem>-<rank>.json.gz``
    of the output folder (a 6-digit rank reserved), whatever the format: name_problem's rules."""
    suffix = "-" + "0" * RANK_DIGITS + max(STEM_SUFFIXES.values(), key=len)
    # The name itself ('', '.', '..', '/', NUL), else its length once a rank and a suffix are added
    problem = name_problem(stem, "")
    if problem is None:
        problem = name_problem(stem, suffix)
        if problem is not None:
            problem += " once a rank and a suffix are added"
    if problem is not None:
        raise RefusalError(
            f"{what} {stem!r} {problem}: it names the files <name>-<rank>.parquet or <name>-<rank>.json.gz of the "
            f"output folder. Give a name {_usable_rule(suffix)}."
        )


# =====================================================================================================================
# How a folder compares names
# =====================================================================================================================
class Folding(NamedTuple):
    """How a folder compares file names: ``case``, 'A' and 'a' name one file; ``unicode``, the NFC and NFD spellings of
    'é' name one file; ``full``, with ``case``, 'ß' and 'ss' name one file too: full case folding (str.casefold, as
    macOS APFS folds), rather than simple case folding, one character for one (a folder that keeps 'ß' and 'ss'
    apart)."""
    case: bool = False
    unicode: bool = False
    full: bool = False

    def describe(self) -> str:
        """What the folder does not tell apart, for a refusal ('' for an exact folder)."""
        told = [part for part, on in (("upper and lower case", self.case), ("Unicode spellings (NFC and NFD)",
                                                                             self.unicode)) if on]
        return " or ".join(told)


EXACT = Folding()


def _names_it(spelling: str, path: str) -> bool:
    try:
        return os.path.samefile(spelling, path)
    except OSError:
        return False


def probe_folding(folder) -> Folding:
    """How ``folder`` compares names: a uniquely named temporary file is created in it (mkstemp) and removed, and its
    swapped-case (but for 'ß'), NFD and 'ß' as 'ss' spellings are tried. A folder in which no file can be created (it
    cannot be written, though it may hold files that can be) is probed without writing, on the names it holds
    (_probe_names); one that cannot be probed at all (it does not exist, say) compares exactly."""
    for prefix, unicode_too in ((".foldmatch-probe-Aéß-", True), (".foldmatch-probe-A-", False)):
        try:
            fd, path = tempfile.mkstemp(prefix=prefix, dir=folder)
        except (OSError, UnicodeError):
            continue
        try:
            os.close(fd)
            head, name = os.path.split(path)
            case = _names_it(os.path.join(head, _swap_case(name)), path)
            unicode = unicode_too and _names_it(os.path.join(head, unicodedata.normalize("NFD", name)), path)
            full = case and unicode_too and _names_it(os.path.join(head, name.replace("ß", "ss")), path)
            return Folding(case=case, unicode=unicode, full=full)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    return _probe_names(folder)


def _swap_case(name: str) -> str:
    """The name in swapped case, but for 'ß' (whose upper case is 'SS')."""
    return "".join(c if c == "ß" else c.swapcase() for c in name)


def _other_form(name: str) -> str:
    """The name in NFD, else in NFC (the name itself when it is both)."""
    nfd = unicodedata.normalize("NFD", name)
    return nfd if nfd != name else unicodedata.normalize("NFC", name)


def _sharp_s(name: str) -> str:
    """The name with its first 'ß' as 'ss', else its first 'ss' (in any case) as 'ß' (the name itself without
    either)."""
    if "ß" in name:
        return name.replace("ß", "ss", 1)
    at = name.lower().find("ss")
    return name if at < 0 else name[:at] + "ß" + name[at + 2:]


def _probe_names(folder) -> Folding:
    """How a folder in which no file can be created compares names, found out without writing, on names that exist:
    the folder's entries, then, for what they leave undecided, the folder's own name in its parent (when both are on
    one device: a folder may fold otherwise than its parent, as a case-folding ext4 folder does). A spelling of such a
    name in swapped case, in NFD (or NFC), or with 'ss' for 'ß' (or 'ß' for 'ss'), that names the same file and is not
    an entry of the folder that holds the name (a sibling of the folder that links to it, say) shows that the folder
    takes both spellings for one name; one that does not, that it does not. What no name shows is taken on the safe
    side, as the folders that fold case fold: Unicode spellings as one name where case is folded (APFS, HFS+,
    case-folding ext4), and case folded fully. A folder without such names (one that does not exist, say) compares
    exactly."""
    found = {"case": None, "unicode": None, "full": None}

    def candidates():
        try:
            entries = os.listdir(folder)
        except (OSError, ValueError):
            entries = []
        held = set(entries)
        for entry in entries:
            yield os.fspath(folder), entry, held
        real = os.path.realpath(os.fspath(folder))
        parent, base = os.path.split(real)
        here, there = file_id(real), file_id(parent)
        if base and here is not None and there is not None and here[0] == there[0]:
            try:
                siblings = set(os.listdir(parent))
            except (OSError, ValueError):
                return   # a spelling could be a sibling that links to the folder
            yield parent, base, siblings

    for head, name, others in candidates():
        path = os.path.join(head, name)
        if file_id(path) is None:   # a broken symbolic link, say
            continue
        for part, spell in (("case", _swap_case), ("unicode", _other_form), ("full", _sharp_s)):
            if found[part] is not None or (part == "full" and not found["case"]):
                continue
            other = spell(name)
            if other != name and other not in others:
                found[part] = _names_it(os.path.join(head, other), path)
        if found["case"] is False:
            found["full"] = False
        if None not in found.values():
            break
    if found["case"] is None and found["unicode"] is None:
        return EXACT
    case = bool(found["case"])
    unicode = case if found["unicode"] is None else found["unicode"]
    full = case if found["full"] is None else found["full"]
    return Folding(case=case, unicode=unicode, full=case and full)


def _fold_case(name: str, full: bool) -> str:
    """Full case folding (casefold: 'ß' is 'ss', 'ﬀ' is 'ff'), or simple case folding: each character's one-character
    fold (casefold), else its lower case."""
    if name.isascii():
        return name.lower()
    if full:
        return name.casefold()
    return "".join(c.casefold() if len(c.casefold()) == 1 else c.lower() for c in name)


def name_key(name: str, folding: Folding) -> str:
    """The name as a folder of this folding compares it. Where the folder folds case and Unicode spellings, as
    canonical caseless matching compares (Unicode's D145): decomposed (NFD), case folded, then composed (NFC). Case
    folding does not keep canonical equivalence, so 'ᾷ' and 'ᾼ' with U+0342 fold to one name, as APFS takes them,
    only once decomposed. Simple case folding (``full`` False) is Unicode's, which differs from the older tables of
    HFS+ and exFAT for a few letters outside ASCII ('σ' and 'ς', 'µ' and 'μ')."""
    if folding.unicode:
        name = unicodedata.normalize("NFD", name)
    if folding.case:
        name = _fold_case(name, folding.full)
    if folding.unicode:
        name = unicodedata.normalize("NFC", name)
    return name


def file_id(path) -> Optional[Tuple[int, int]]:
    """A file's (device, inode), or None when it cannot be read."""
    try:
        stat = os.stat(path)
    except (OSError, ValueError):
        return None
    return stat.st_dev, stat.st_ino


def same_folder(a, b) -> bool:
    """Whether two folders are one, compared as files (an existing folder and a missing one are not)."""
    first = file_id(a)
    return first is not None and first == file_id(b)


class InFolder:
    """Whether paths are in ``folder`` (their parent folder is it, as a file), the answer kept per parent."""

    def __init__(self, folder):
        self.id = file_id(folder)
        self._parents = {}

    def __call__(self, path) -> bool:
        parent = os.path.dirname(os.fspath(path)) or os.curdir
        if parent not in self._parents:
            self._parents[parent] = self.id is not None and file_id(parent) == self.id
        return self._parents[parent]


# =====================================================================================================================
# The files of one output
# =====================================================================================================================
class OutputFiles:
    """The files that one output of a run writes in its folder, by the names that the folder compares: ``<name>.<csv|
    pt>`` per item, ``<name>.<item>.<format>`` (or ``-``) per item of a name whose items are known only once its
    structure is read, and ``<stem>-<rank>.<parquet|json.gz>`` per process."""

    def __init__(self, folder, folding: Optional[Folding] = None):
        self.folder = os.fspath(folder)
        self.folding = probe_folding(self.folder) if folding is None else folding
        self.files = {}      # key -> (name, label, file name)
        self.stems = {}      # (stem key, suffix) -> stem
        self.prefixes = {}   # (separator, suffix) -> {prefix key: (prefix, label)}
        self._existing = None
        self.linked = False  # whether a file of existing() is a symbolic link or a hard link

    def key(self, name: str) -> str:
        return name_key(name, self.folding)

    def add_files(self, names: Sequence, suffix: str, labels: Optional[Sequence] = None) -> List[tuple]:
        """Add ``<name><suffix>`` per name: the repeats, as (file name, [(name, label), ...]) per file written twice."""
        groups = {}
        for k, name in enumerate(names):
            file = f"{name}{suffix}"
            key = self.key(file)
            label = None if labels is None else labels[k]
            if key in self.files:
                groups.setdefault(key, [self.files[key][:2]]).append((name, label))
            else:
                self.files[key] = (name, label, file)
        return [(f"{entries[0][0]}{suffix}", entries) for entries in groups.values()]

    def add_stem(self, stem: str, suffix: str):
        self.stems[(self.key(stem), suffix)] = stem

    def add_prefixes(self, prefixes: Sequence, separator: str, suffix: str,
                     labels: Optional[Sequence] = None) -> List[tuple]:
        """Add ``<prefix><separator><ID><suffix>`` per prefix, for the IDs known only once its structure is read: the
        prefixes that the folder takes for one, as (files, [(prefix, label), ...]) per prefix given twice (their IDs
        are not known yet, and those of one structure, or chain A of two, would give one file twice)."""
        index = self.prefixes.setdefault((separator, suffix), {})
        groups = {}
        for k, prefix in enumerate(prefixes):
            key = self.key(prefix)
            label = None if labels is None else labels[k]
            if key in index:
                groups.setdefault(key, [index[key]]).append((prefix, label))
            else:
                index[key] = (prefix, label)
        return [(f"{entries[0][0]}{separator}<ID>{suffix}", entries) for entries in groups.values()]

    def holds(self, name: str) -> Optional[str]:
        """The output that writes the file ``name`` of the folder, described, or None. A lookup per separator in the
        name, whatever the number of prefixes."""
        key = self.key(name)
        if key in self.files:
            return f"the output {self.files[key][0]!r}"
        for (stem, suffix), given in self.stems.items():
            # The rank as the writers spell it, in ASCII digits ('\d' takes any Unicode digit)
            if key.endswith(suffix) and re.fullmatch(re.escape(stem) + r"-[0-9]+", key[:len(key) - len(suffix)]):
                return f"a file <name>-<rank>{suffix} of the output name {given!r}"
        for (separator, suffix), index in self.prefixes.items():
            if not key.endswith(suffix):
                continue
            head = key[:len(key) - len(suffix)]
            at = head.find(separator)
            while at >= 0:
                # <prefix><separator><ID>, an ID of at least one character
                if 0 < at and at + len(separator) < len(head) and head[:at] in index:
                    given = index[head[:at]][0]
                    return f"a file {given}{separator}<chain or assembly ID>{suffix} of the output {given!r}"
                at = head.find(separator, at + 1)
        return None

    def existing(self) -> dict:
        """The files of the folder that the outputs would write and that exist already, as an earlier run left them:
        {(device, inode): (path, the output described)}, read once (one listing of the folder, and one lstat per such
        file, and a stat more per symbolic link). ``linked`` tells whether any of them is a symbolic link or a hard
        link (a file of more than one name), which may be an input of another folder."""
        if self._existing is None:
            self._existing = {}
            try:
                entries = os.listdir(self.folder)
            except (OSError, ValueError):
                entries = []
            for entry in entries:
                output = self.holds(entry)
                if output is None:
                    continue
                path = os.path.join(self.folder, entry)
                try:
                    link = os.lstat(path)
                except (OSError, ValueError):
                    continue
                if stat.S_ISLNK(link.st_mode):
                    found = file_id(path)
                    if found is None:   # a broken symbolic link
                        continue
                    self.linked = True
                else:
                    found = (link.st_dev, link.st_ino)
                    self.linked = self.linked or link.st_nlink > 1
                self._existing.setdefault(found, (path, output))
        return self._existing

    def shared_with(self, other: "OutputFiles") -> List[str]:
        """The files that this output and another one in the same folder would both write."""
        shared = [self.files[key][2] for key in self.files if key in other.files]
        return shared + [f"{stem}-<rank>{suffix}" for (key, suffix), stem in self.stems.items()
                         if (key, suffix) in other.stems]


def _folded_by(names: Sequence[str], folding: Folding) -> List[str]:
    """What the folder does not tell apart that makes names that differ one name: upper and lower case, Unicode
    spellings, or both."""
    for part, alone in (("upper and lower case", Folding(case=folding.case, full=folding.full)),
                        ("Unicode spellings (NFC and NFD)", Folding(unicode=folding.unicode))):
        if alone != EXACT and len({name_key(name, alone) for name in names}) == 1:
            return [part]
    return [part for part, on in (("upper and lower case", folding.case),
                                  ("Unicode spellings (NFC and NFD)", folding.unicode)) if on]


def repeated_files_message(source: str, repeats: List[tuple], folding: Folding, what: str = "embedding name",
                           given: str = "those rows", folded_advice: Optional[str] = None) -> str:
    """The refusal of names that give one file twice (OutputFiles.add_files' and add_prefixes' repeats). Names that
    differ, and that the folder only takes for one, are told so, with what can keep them apart: ``folded_advice``,
    whose ``{apart}`` is what the folder does not tell apart, or by default another folder, Parquet or JSON, or other
    names."""
    def describe(repeat):
        file, entries = repeat
        names = " and ".join(f"{name!r}{'' if label is None else f' ({label})'}" for name, label in entries)
        return f"{names} give {file}"
    folded = []
    for _, entries in repeats:
        names = list(dict.fromkeys(f"{name}" for name, _ in entries))
        if len(names) > 1:
            folded += [part for part in _folded_by(names, folding) if part not in folded]
    folded.sort(key=lambda part: part.startswith("Unicode"))
    apart = " or ".join(folded)
    note = f" (the output folder does not tell {apart} apart)" if folded else ""
    found = (f"{source}: {len(repeats)} file(s) of the output folder would each be written for more than one "
             f"{what}{note}: {shown(repeats, describe)}. Each one would replace the one before")
    if not folded:
        return f"{found}: give each its own {what}."
    advice = (folded_advice or "write them to a folder of a file system that tells {apart} apart or as parquet or "
              "json (which keep each {what} as given), or give {given} {what}s that differ otherwise")
    return f"{found}, as the folder takes those names for one: {advice.format(apart=apart, what=what, given=given)}."


def _linked_id(path) -> Optional[Tuple[int, int]]:
    """The (device, inode) of the file that ``path`` names when it may be another folder's file: through a symbolic
    link, or as a hard link (a file of more than one name); else None."""
    try:
        link = os.lstat(path)
    except (OSError, ValueError, TypeError):
        return None
    if stat.S_ISLNK(link.st_mode):
        return file_id(path)
    return (link.st_dev, link.st_ino) if link.st_nlink > 1 else None


def refuse_outputs_over_inputs(outputs: OutputFiles, inputs: Iterable[Tuple[str, object]], *, what: str = "run"):
    """Refuse (RefusalError) an input of the run (described, path) that is in the output folder under the name of one
    of its outputs, or, from another folder, one file with a file of the output folder that one of its outputs would
    write, through a symbolic or hard link either way (compared as files, with the files that exist: an input
    exists): the run would replace it."""
    in_folder = InFolder(outputs.folder)
    found, others = [], []
    for described, path in inputs:
        if path is None:
            continue
        output = outputs.holds(os.path.basename(os.fspath(path))) if in_folder(path) else None
        if output is not None:
            found.append((described, os.fspath(path), output))
        else:
            others.append((described, path))
    # Only an output's file that exists can be an input reached through a link (none in a new folder). Where none of
    # them is a link, only an input that is a link can be one (_linked_id); else any input can (file_id)
    existing = outputs.existing() if others else {}
    if existing:
        identify = file_id if outputs.linked else _linked_id
        for described, path in others:
            file = existing.get(identify(path))
            if file is not None:
                found.append((described, f"{os.fspath(path)} (which is {file[0]})", file[1]))
    if found:
        raise RefusalError(
            f"{len(found)} input(s) of this {what} would be replaced by its output, in the output folder "
            f"{outputs.folder}: {shown(found, lambda f: f'{f[0]} {f[1]} by {f[2]}')}. An output never replaces an "
            f"input of its run: write the output to another folder or under other names, or move the input."
        )


def check_outputs(out_folder, out_format, out_name, *, names: Optional[Sequence] = None,
                  labels: Optional[Sequence] = None, source: str = "The input", prefixes: Optional[Sequence] = None,
                  separator: str = ".", inputs: Iterable[Tuple[str, object]] = (),
                  folding: Optional[Folding] = None, stem_what: str = "The output name",
                  what: str = "embedding name", given: str = "those rows",
                  folded_advice: Optional[str] = None) -> Optional[OutputFiles]:
    """The checks of one output of a run, before any model is built (RefusalError): the output name ``out_name`` names
    files of the folder, whatever the format (D1); for csv and pt, each of ``names`` names one file ``<name>.<format>``
    (D1) of its own, as the folder compares names (D3), and each of ``prefixes`` names the files
    ``<prefix><separator><ID>.<format>`` of items known only once their structure is read, prefixes that the folder
    takes for one refused too; and no input (described, path) is one of the output files (D2). ``stem_what`` names the
    output name in a refusal, ``what`` a name and ``given`` what holds them, ``folded_advice`` what keeps names apart
    that the folder takes for one (repeated_files_message); ``folding`` is the folder's, when known. The output's
    files, or None without ``out_folder``."""
    if out_folder is None:
        return None
    fmt = format_value(out_format)
    outputs = OutputFiles(out_folder, folding)
    if out_name is not None:
        refuse_unusable_stem(out_name, stem_what)
        if fmt in STEM_SUFFIXES:
            outputs.add_stem(out_name, STEM_SUFFIXES[fmt])
    if fmt in ITEM_FORMATS:
        suffix = f".{fmt}"
        repeats = []
        if names is not None:
            refuse_unusable_names(names, suffix, source=source, labels=labels, what=what, given=given)
            repeats = outputs.add_files(names, suffix, labels)
        if prefixes is not None and not repeats:
            refuse_unusable_names(prefixes, separator + suffix, source=source, labels=labels,
                                  pattern=f"<name>{separator}<ID>{suffix}")
            repeats = outputs.add_prefixes(prefixes, separator, suffix, labels)
        if repeats:
            raise RefusalError(repeated_files_message(source, repeats, outputs.folding, what, given, folded_advice))
    refuse_outputs_over_inputs(outputs, inputs)
    return outputs


def refuse_assemblies_over_residues(names: Sequence, structure_names: Iterable, *, out_folder, out_format,
                                    res_folder, res_format, source: str, labels: Optional[Sequence] = None):
    """Refuse (RefusalError) assembly embeddings written in the format of the residue embeddings they are computed
    from, to their folder, under a name ``<structure name>.<anything>`` for a structure name of the list (D2):
    residue embeddings are named ``<structure name>.<chain ID>``, and the chain IDs are known only once the structure
    is read, so such a name could be a residue embedding's. Names such as ``1acb-1`` are allowed."""
    fmt = format_value(out_format)
    if out_folder is None or res_folder is None or fmt != format_value(res_format) or \
            not same_folder(out_folder, res_folder):
        return
    folding = probe_folding(out_folder)
    structures = {name_key(name, folding) for name in structure_names if isinstance(name, str)}
    found = []
    for k, name in enumerate(names):
        key = name_key(name, folding) if isinstance(name, str) else ""
        if any(key[:i] in structures for i, c in enumerate(key) if c == "."):
            found.append(f"{name!r}" + ("" if labels is None else f" ({labels[k]})"))
    if found:
        raise RefusalError(
            f"{source}: {len(found)} assembly embedding name(s) of the form <structure name>.<...>, written as {fmt} "
            f"files to the folder of the residue embeddings ({os.fspath(out_folder)}): {shown(found)}. Residue "
            f"embeddings are named <structure name>.<chain ID>.{fmt}, and the chain IDs are known only once the "
            f"structure is read, so such an assembly embedding could replace a residue embedding it is computed from: "
            f"name the assembly otherwise (<structure name>-<assembly ID>, the default, say), or write the assembly "
            f"embeddings to another folder or in another format."
        )


def residue_files(res_folder, structure_names: Iterable, res_format) -> List[Tuple[str, str]]:
    """The residue embedding files that a run on the structures named reads, as inputs (described, path): the files
    ``<structure name>.<chain ID>.<format>`` of ``res_folder`` that exist (the chains are known only once each
    structure is read). One listing of the folder."""
    if res_folder is None:
        return []
    index = OutputFiles(res_folder)
    index.add_prefixes([name for name in dict.fromkeys(structure_names) if isinstance(name, str) and name], ".",
                       f".{format_value(res_format)}")
    try:
        entries = sorted(os.listdir(res_folder))
    except (OSError, ValueError):
        return []
    return [("the residue embedding file", os.path.join(os.fspath(res_folder), entry)) for entry in entries
            if index.holds(entry) is not None]


#: The files of a database that the commands read, ``<db path><suffix>``: the index and its metadata and inverted lists,
#: the sequence store, and the residue-code store (``cli.search._SEQUENCE_DB_SUFFIXES``, ``search.residue_code_store``).
DATABASE_SUFFIXES = (".index", ".metadata", ".invlists", ".sequences", ".rescodes.bin", ".rescodes.idx",
                     ".rescodes.json")
#: The files of a database's residue-code store, ``<db path>.rescodes.*`` (``search.split_run._CODE_STORE_GLOB``).
CODE_STORE_GLOB = ".rescodes.*"


def database_files(db_path) -> List[str]:
    """The existing files of a database (DATABASE_SUFFIXES, then any other ``<db path>.rescodes.*``)."""
    db_path = os.fspath(db_path)
    db_dir, index_name = os.path.split(db_path)
    files = [db_path + suffix for suffix in DATABASE_SUFFIXES]
    pattern = os.path.join(glob.escape(db_dir or os.curdir), glob.escape(index_name) + CODE_STORE_GLOB)
    files += sorted(glob.glob(pattern))
    return [f for f in dict.fromkeys(files) if os.path.isfile(f)]


def refuse_output_file_over_inputs(output_file, *, files: Iterable[Tuple[str, object]] = (),
                                   databases: Iterable[Tuple[str, str]] = (), option: str = "--output-file"):
    """Refuse (RefusalError) an output file that is one of the command's inputs (P9.N D6), before any work: one of
    ``files`` (described, path), or a file of one of ``databases`` (described, path) that the commands read
    (database_files). Compared as files (os.path.samefile): an input exists, so only an existing output file can be
    one, and a symbolic link to it, or on macOS another case of its name, is. Any other file, such as the results of
    an earlier command named ``<db path>.hits.tsv`` beside the database, is not an input."""
    out = os.fspath(output_file)
    if not os.path.isfile(out):
        return
    found = []
    for described, path in files:
        if path is not None and _names_it(out, os.fspath(path)):
            found.append(f"{described} {os.fspath(path)}")
    for described, db_path in databases:
        found.extend(f"the file {path} of {described} {os.fspath(db_path)}" for path in database_files(db_path)
                     if _names_it(out, path))
    if found:
        raise RefusalError(
            f"{option} {out} is {'; and '.join(found)}: the results would replace an input of this command. Write "
            f"them to another file."
        )
