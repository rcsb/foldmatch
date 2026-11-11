"""Where a database's files live, from the ``--db-path`` / ``--output-db`` string.

Dependency-free (no FAISS), so the CLI can check a database's files on every
rank before ESM3 without loading the vector library.
"""
from pathlib import Path


def parse_db_path(db_path: str) -> tuple[Path, str, str]:
    """Split a database path into (directory, index name, resolved path).

    The files are ``{resolved path}{suffix}``: ``.index``, ``.metadata``,
    ``.invlists`` and ``.sequences``. A bare name lives in the current
    directory, and an empty name is ``embeddings``.
    """
    db_path = Path(db_path)
    db_dir = db_path.parent
    index_name = db_path.name or "embeddings"
    if db_dir == Path('.'):
        db_dir = Path.cwd()
    return db_dir, index_name, str(db_dir / index_name)
