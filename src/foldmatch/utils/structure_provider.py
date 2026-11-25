from foldmatch.utils.structure_parser import get_structure_from_src


class StructureProvider:
    """The last structure read, kept so that consecutive requests for it (one per chain of an entry) read the file
    once. It is keyed by the source name, the format and the assembly: two assemblies of one entry are two
    structures.

    A source may be given lazily, as a callable that takes no argument and returns the path or stream to read (a
    download): it is called only when the structure is not the one kept, and the stream it returned is kept too, keyed
    by the source name and the format, so that the other assemblies of that entry are read from it rather than
    downloaded again. Only the last source and the last structure are kept.

    What is kept is dropped when the provider is pickled (sent to a DataLoader worker): each copy keeps its own, and a
    stream need not be pickled."""

    def __init__(self):
        self.__cached = None
        self.__source = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_StructureProvider__cached"] = None
        state["_StructureProvider__source"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_structure(
        self,
        src_name,
        src_structure,
        structure_format="mmcif",
        *,
        assembly_id=None
    ):
        key = (src_name, structure_format, assembly_id)
        cached = self.__cached
        if cached is None or cached[0] != key:
            cached = (key, get_structure_from_src(
                src_structure=self.get_source(src_name, src_structure, structure_format),
                structure_format=structure_format,
                assembly_id=assembly_id
            ))
            self.__cached = cached
        return cached[1]

    def get_source(self, src_name, src_structure, structure_format="mmcif"):
        """The path or stream to read src_name from: src_structure itself, unless it is a callable, whose result is
        kept for the next request for src_name in this format (the structure reader reads a stream from its start)."""
        if not callable(src_structure):
            return src_structure
        key = (src_name, structure_format)
        if self.__source is None or self.__source[0] != key:
            self.__source = None  # the previous download is let go before the next one
            self.__source = (key, src_structure())
        return self.__source[1]
