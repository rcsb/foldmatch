"""Exceptions shared by the command-line tools and the library (plan D20).

The command-line exit codes follow one rule:

* a refused combination of options is a ``typer.BadParameter`` and exits with status 2 (click's usage error);
* a refused run, because the data, the database or the installation cannot serve it, is a :class:`RefusalError`
  and exits with status 1, its message printed after ``Error: `` (``fm-search``'s ``main``).

Dependency-free, so any module (and Stage-2 workers) can import it.
"""


class RefusalError(ValueError):
    """A read-only check refused the run: the input, the database or the installation cannot serve the request.

    The message says what was found and what to do about it. Refusals come before any expensive or irreversible
    step where possible (before ESM3, before an existing file is replaced). A ``ValueError``, so callers and the
    command-line tools that already report ``ValueError`` as an ``Error: ...`` (exit status 1) handle it unchanged;
    the refused check's own exception, when there is one, is its ``__cause__``.
    """


class MissingResidueCodesError(RefusalError):
    """A pass with residue codes in files of their own (``fm-embedding from-sequences residue|chain
    --output-codes-folder``) ended with sequences whose codes could not be computed (ESM3 gave rows that do not match
    the sequence's length, or the encoding failed). Raised at the end, after every file is written: those sequences'
    codes are empty ``[0, 8]`` tables, their embeddings and every other output as in a complete run. A
    :class:`RefusalError` (exit status 1: the data cannot serve the run), raised after the pass rather than before it,
    so that a caller can tell it from a pre-flight refusal, after which nothing is written. With several ranks, rank 0
    raises it for the whole run, once every rank has ended its pass; the other ranks log their count and return.

    ``count`` is the number of such sequences in the run, ``ids`` the first of their ids (up to 10, by rank),
    ``rank_counts`` the count of each rank, ``rank`` and ``world_size`` the process's rank and the number of ranks,
    ``predictions`` what ``predict`` would have returned on this rank (None unless the caller asked for predictions):
    ``(embeddings, ids, codes, seq_sha1s)`` per batch, ``codes`` one CPU ``uint8`` tensor ``[L, 8]`` per id, as its
    ``.pt`` code file holds (``[0, 8]`` for those sequences).
    """

    def __init__(self, message: str, *, count: int = 0, ids=(), rank: int = 0, world_size: int = 1,
                 rank_counts=(), predictions=None):
        super().__init__(message)
        self.count = count
        self.ids = tuple(ids)
        self.rank = rank
        self.world_size = world_size
        self.rank_counts = tuple(rank_counts)
        self.predictions = predictions


class InternalError(RuntimeError):
    """An internal consistency check failed: a FoldMatch bug, not a problem with the input or the database.

    Raised instead of ``assert`` (which ``python -O`` removes) where a result would otherwise be wrong without notice,
    e.g. when the esm-codes Stage-2 traceback score differs from the round-1 score behind the reported E-value (P6.4).
    Not a ``ValueError``: the command-line tools do not report it as a one-line refusal.
    """
