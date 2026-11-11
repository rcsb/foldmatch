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


class InternalError(RuntimeError):
    """An internal consistency check failed: a FoldMatch bug, not a problem with the input or the database.

    Raised instead of ``assert`` (which ``python -O`` removes) where a result would otherwise be wrong without notice,
    e.g. when the esm-codes Stage-2 traceback score differs from the round-1 score behind the reported E-value (P6.4).
    Not a ``ValueError``: the command-line tools do not report it as a one-line refusal.
    """
