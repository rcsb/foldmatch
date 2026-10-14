"""Exceptions shared by the command-line tools and the library (plan D20).

The command-line exit codes follow one rule:

* a refused combination of options is a ``typer.BadParameter`` and exits with status 2 (click's usage error);
* a refused run, because the data, the database or the installation cannot serve it, is a :class:`RefusalError`
  and exits with status 1, its message printed on one line (``fm-search``'s ``main``).

Dependency-free, so any module (and Stage-2 workers) can import it.
"""


class RefusalError(ValueError):
    """A read-only check refused the run: the input, the database or the installation cannot serve the request.

    The message says what was found and what to do about it. Refusals come before any expensive or irreversible
    step where possible (before ESM3, before an existing file is replaced). A ``ValueError``, so callers and the
    command-line tools that already report ``ValueError`` as a one-line error (exit status 1) handle it unchanged;
    the refused check's own exception, when there is one, is its ``__cause__``.
    """
