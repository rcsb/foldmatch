"""P9.N3: fm-search refuses an --output-file that is one of its inputs, and query embeddings that share an id.

Every command that takes --output-file (query structure, embedding, sequences, db, cluster, similarity-graph) refuses,
before any work, an output file that is a file of an input database (<db path>.index, .metadata, .invlists,
.sequences, .rescodes.*), the query FASTA or structure file, or a query embedding file it reads (D6): the results
replaced it with exit status 0. Compared as files (a symbolic link, and on macOS another case, name the same file). An
output file of an earlier command is not an input, even one named <db path>.hits.tsv beside the database: a rerun with
the same --output-file replaces it, as before. query embedding refuses two vectors of one id (D7): one of
the queries disappeared from the output.

No model is loaded: the databases are built from vectors (build embeddings), query structure's embedding is a stub,
and query sequences is refused or stopped before ESM3."""
import logging
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from typer.testing import CliRunner

from foldmatch.cli import search
from foldmatch.errors import RefusalError

_CIF = Path(__file__).parent / "resources" / "pdb" / "1acb.cif"


def _flat(text: str) -> str:
    """Text on one line, typer's error panel borders and colour codes dropped."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).translate({ord(c): " " for c in "│╭╮╰╯─"}).split())


def _case_insensitive() -> bool:
    """Whether the temporary folder takes 'A' and 'a' for one name (macOS APFS by default), found out without
    FoldMatch."""
    with tempfile.NamedTemporaryFile(prefix="CaseProbe-") as f:
        head, name = os.path.split(f.name)
        return os.path.exists(os.path.join(head, name.swapcase()))


_CASE_INSENSITIVE = _case_insensitive()


def _vectors(folder: Path, ids, offset: float = 0.0) -> Path:
    folder.mkdir(exist_ok=True)
    for k, name in enumerate(ids):
        torch.save(torch.arange(8, dtype=torch.float32) * (k + 1) + offset, folder / f"{name}.pt")
    return folder


def _parquet(path: Path, ids) -> Path:
    pq.write_table(pa.table({"id": list(ids), "embedding": [[float(k + 1)] * 8 for k in range(len(ids))]},
                            schema=pa.schema([("id", pa.string()), ("embedding", pa.list_(pa.float32()))])), path)
    return path


class _Case(unittest.TestCase):

    def setUp(self):
        # Resolved, as typer resolves the paths the commands are given (/var is /private/var on macOS)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.db = self.build("db", ["s1", "s2", "s3"])

    def invoke(self, *args):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        try:
            return CliRunner().invoke(search.app, [str(a) for a in args])
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)

    def run_ok(self, *args):
        result = self.invoke(*args)
        self.assertEqual(result.exit_code, 0, f"{result.output}\n{result.exception!r}")
        return result

    def build(self, name: str, ids) -> Path:
        db = self.tmp / name
        self.run_ok("build", "embeddings", "--embedding-folder", _vectors(self.tmp / f"{name}-vectors", ids),
                    "--output-db", db)
        return db

    def refused(self, *args, kept: Path) -> str:
        """Refused (exit status 1, a RefusalError) and the file ``kept`` unchanged: the message, on one line."""
        before = kept.read_bytes()
        result = self.invoke(*args)
        self.assertEqual(result.exit_code, 1, f"{result.output}\n{result.exception!r}")
        self.assertIsInstance(result.exception, RefusalError, repr(result.exception))
        self.assertEqual(kept.read_bytes(), before)
        return _flat(str(result.exception))


class DatabaseFileTest(_Case):
    """D6: an output file that is a file of an input database."""

    def test_query_embedding(self):
        """--output-file <db>.index replaced the index, and every later command failed (sweep case 11)."""
        queries = _vectors(self.tmp / "queries", ["q1", "q2"])
        db_files = sorted(f.name for f in self.tmp.iterdir() if f.name.startswith("db."))
        self.assertEqual(db_files, ["db.index", "db.metadata"])
        for name in db_files:
            with self.subTest(name=name):
                message = self.refused("query", "embedding", "--db-path", self.db, "--embedding-folder", queries,
                                       "--output-file", self.tmp / name, kept=self.tmp / name)
                self.assertEqual(message, f"--output-file {self.tmp / name} is the file {self.tmp / name} of the "
                                          f"database {self.db}: the results would replace an input of this command. "
                                          f"Write them to another file.")
        link = self.tmp / "results-link.tsv"
        link.symlink_to(self.tmp / "db.metadata")
        self.refused("query", "embedding", "--db-path", self.db, "--embedding-folder", queries, "--output-file", link,
                     kept=self.tmp / "db.metadata")
        message = self.refused("query", "embedding", "--db-path", self.db, "--embedding-folder", queries,
                               "--output-file", queries / "q1.pt", kept=queries / "q1.pt")
        self.assertIn(f"is the query embedding file {queries / 'q1.pt'}", message)
        results = self.tmp / "results.tsv"
        for _ in range(2):
            self.run_ok("query", "embedding", "--db-path", self.db, "--embedding-folder", queries, "--output-file",
                        results, "--threshold", "0")
        self.assertEqual(sorted({line.split("\t")[0] for line in results.read_text().splitlines()}), ["q1", "q2"])
        self.run_ok("stats", "--db-path", self.db)

    @unittest.skipUnless(_CASE_INSENSITIVE, "a case-insensitive file system (macOS APFS)")
    def test_another_case_on_this_mac(self):
        """DB.INDEX is db.index on macOS: refused."""
        queries = _vectors(self.tmp / "queries", ["q1"])
        message = self.refused("query", "embedding", "--db-path", self.db, "--embedding-folder", queries,
                               "--output-file", self.tmp / "DB.INDEX", kept=self.tmp / "db.index")
        self.assertIn(f"is the file {self.tmp / 'db.index'} of the database {self.db}", message)

    def test_query_db_cluster_and_graph(self):
        """query db (either database), cluster and similarity-graph; another file runs, twice."""
        other = self.build("other", ["o1", "o2"])
        cases = (("query", "db", "--query-db-path", other, "--subject-db-path", self.db),
                 ("cluster", "--db-path", self.db, "--threshold", "0"),
                 ("similarity-graph", "--db-path", self.db, "--threshold", "0"))
        for args in cases:
            targets = [self.tmp / "db.index"] + ([self.tmp / "other.metadata"] if args[0] == "query" else [])
            for target in targets:
                with self.subTest(command=args[:2], target=target.name):
                    message = self.refused(*args, "--output-file", target, kept=target)
                    self.assertIn(f"--output-file {target} is the file {target} of the ", message)
            out = self.tmp / f"{args[0]}-out.txt"
            for _ in range(2):
                self.run_ok(*args, "--output-file", out)
            self.assertTrue(out.exists())


    def test_results_named_after_the_database_rerun(self):
        """Results named <db path>.<anything> beside the database are not a file of it: query embedding, cluster and
        similarity-graph run twice into them (the second run was refused), and one command into another's results.
        A file of the residue-code store, <db path>.rescodes.*, is one."""
        queries = _vectors(self.tmp / "queries", ["q1"])
        cases = (("query", "embedding", "--db-path", self.db, "--embedding-folder", queries, "--threshold", "0"),
                 ("cluster", "--db-path", self.db, "--threshold", "0"),
                 ("similarity-graph", "--db-path", self.db, "--threshold", "0"))
        for args in cases:
            with self.subTest(command=args[0]):
                out = self.tmp / f"db.{args[0]}.tsv"
                for _ in range(2):
                    self.run_ok(*args, "--output-file", out)
                self.assertTrue(out.exists())
        self.run_ok(*cases[0], "--output-file", self.tmp / "db.cluster.tsv")
        store = self.tmp / "db.rescodes.bin"
        store.write_bytes(b"codes")
        message = self.refused(*cases[1], "--output-file", store, kept=store)
        self.assertIn(f"is the file {store} of the database {self.db}", message)


class QueryInputTest(_Case):
    """D6: the query FASTA and the query structure file."""

    def test_query_sequences(self):
        """--output-file q.fasta replaced the queries: refused before the ESM3 pass; another file gets past the check
        to the pass (stopped there, no model is loaded)."""
        fasta = self.tmp / "q.fasta"
        fasta.write_text(">q1\nMKV\n")
        tmp = self.tmp / "embeddings"
        tmp.mkdir()
        never = mock.Mock(side_effect=AssertionError("the ESM3 pass ran"))
        with mock.patch("foldmatch.search.embedding_computer.chain_predict", never):
            message = self.refused("query", "sequences", "--db-path", self.db, "--fasta-file", fasta,
                                   "--tmp-embedding-folder", tmp, "--output-file", fasta, kept=fasta)
        self.assertEqual(message, f"--output-file {fasta} is the FASTA file {fasta}: the results would replace an "
                                  f"input of this command. Write them to another file.")
        never.assert_not_called()
        stop = mock.Mock(side_effect=RuntimeError("the ESM3 pass"))
        with mock.patch("foldmatch.search.embedding_computer.chain_predict", stop):
            result = self.invoke("query", "sequences", "--db-path", self.db, "--fasta-file", fasta,
                                 "--tmp-embedding-folder", tmp, "--output-file", self.tmp / "results.tsv")
        stop.assert_called_once()
        self.assertIsInstance(result.exception, RuntimeError)

    def test_query_structure(self):
        """--output-file q.cif replaced the query structure: refused before its embedding; another file runs."""
        structure = Path(self.tmp / "q.cif")
        structure.write_bytes(_CIF.read_bytes())
        never = mock.Mock(side_effect=AssertionError("the query was embedded"))
        with mock.patch("foldmatch.search.embedding_computer.compute_from_structure_file", never):
            message = self.refused("query", "structure", "--db-path", self.db, "--query-structure", structure,
                                   "--output-file", structure, kept=structure)
        self.assertIn(f"is the query structure file {structure}", message)
        never.assert_not_called()
        stub = mock.Mock(return_value=[(["q.A"], np.arange(8, dtype=np.float32)[None, :])])
        with mock.patch("foldmatch.search.embedding_computer.compute_from_structure_file", stub):
            self.run_ok("query", "structure", "--db-path", self.db, "--query-structure", structure,
                        "--output-file", self.tmp / "results.tsv", "--threshold", "0")
        self.assertTrue((self.tmp / "results.tsv").read_text().startswith("q.A\t"))


class QueryIdTest(_Case):
    """D7: query embedding refuses two vectors of one id (one query disappeared from the output)."""

    def test_repeated_ids(self):
        queries = self.tmp / "queries"
        queries.mkdir()
        _parquet(queries / "a.parquet", ["x", "y", "x"])
        _parquet(queries / "b.parquet", ["z"])
        _vectors(queries, ["z"])
        results = self.tmp / "results.tsv"
        result = self.invoke("query", "embedding", "--db-path", self.db, "--embedding-folder", queries,
                             "--output-file", results)
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIsInstance(result.exception, RefusalError)
        self.assertEqual(_flat(str(result.exception)),
                         f"{queries}: 2 id(s) held by more than one query embedding: 'x' in 2 rows of a.parquet; 'z' "
                         f"in b.parquet and z.pt. The results are written by query id, so a repeated id would keep the "
                         f"hits of one of its embeddings and drop the others': give each embedding its own id, or "
                         f"query its files apart (--file-extension, or one folder per kind).")
        self.assertFalse(results.exists())
        self.run_ok("query", "embedding", "--db-path", self.db, "--embedding-folder", queries, "--file-extension",
                    ".pt", "--output-file", results, "--threshold", "0")
        (queries / "a.parquet").unlink()
        self.run_ok("query", "embedding", "--db-path", self.db, "--embedding-folder", queries, "--file-extension",
                    ".parquet", "--output-file", results, "--threshold", "0")
        self.assertEqual({line.split("\t")[0] for line in results.read_text().splitlines()}, {"z"})


if __name__ == "__main__":
    unittest.main()
