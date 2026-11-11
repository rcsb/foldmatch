from foldmatch.utils.structure_parser import get_structure_from_src


class StructureProvider:
    """The last structure read, kept so that consecutive requests for it (one per chain of an entry) read the file
    once. It is keyed by the source name, the format and the assembly: two assemblies of one entry are two
    structures."""

    def __init__(self):
        self.__cached = None

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
                src_structure=src_structure,
                structure_format=structure_format,
                assembly_id=assembly_id
            ))
            self.__cached = cached
        return cached[1]
