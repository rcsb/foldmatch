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
