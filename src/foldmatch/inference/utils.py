import os
from typing import Optional

from foldmatch.errors import RefusalError


def is_distributed(devices, num_nodes) -> bool:
    """Multi-process (DDP) run → the post-inference barrier needs more than NCCL's 30-min default."""
    if num_nodes > 1:
        return True
    if isinstance(devices, int):
        return devices != 1            # >1, or -1 (= all visible)
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    return True


def refuse_min_res_n(min_res_n: int, listed: str):
    """min_res_n selects the protein chains of structures read whole (src_from structure): an input that lists its
    chains (``listed``) would ignore it, so a value other than 0 is refused, before anything is read or loaded."""
    if min_res_n != 0:
        raise RefusalError(
            f"min_res_n={min_res_n} selects the protein chains of structures read whole (src_from structure), but "
            f"{listed} are embedded as listed: leave min_res_n at 0, or list only the chains to embed."
        )


def list_inputs(data, src_stream=None, src_location=None, *, list_name: str = "list", column: Optional[str] = "stream",
                what: str = "structure file", rows: bool = True) -> list:
    """The inputs of a run from a list (D2), as (described, path): the list file (when the list is given as a file),
    and the local files of ``column`` of its frame (structure files, residue embedding files), with their rows when
    ``rows`` (the frame is the list given, not one a reader made of it)."""
    from foldmatch.dataset.utils import list_file
    inputs = []
    path = None if src_stream is None else list_file(src_stream, src_location)
    if path is not None:
        inputs.append((f"the {list_name}", path))
    if column is not None:
        for k, f in zip(data.index, data[column]):
            if isinstance(f, (str, os.PathLike)):
                inputs.append((f"the {what} of row {k + 1}" if rows else f"the {what}", f))
    return inputs
