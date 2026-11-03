"""CLI help text that states what the code does (G10)."""
import unittest

import click
import pytest
import typer

from foldmatch.cli import embedding, inference
from foldmatch.cli.search import app


def _option_help(path, param_name):
    command = typer.main.get_command(app)
    for name in path:
        command = command.commands[name]
    return next(p.help for p in command.params if p.name == param_name)


class MinCoverageHelpTest(unittest.TestCase):

    def test_help_says_both_sides_must_reach_the_value(self):
        for path in (("query", "sequences"), ("query", "db")):
            with self.subTest(command=" ".join(path)):
                text = _option_help(path, "min_coverage")
                self.assertIn("both", text)
                self.assertIn("either side is below", text)
                self.assertNotIn("AND", text)

    def test_align_candidates_docstring_says_the_same(self):
        from foldmatch.search import alignment
        doc = " ".join(alignment.align_candidates.__doc__.split())
        self.assertIn("min_coverage: drop hits whose query or subject coverage is below this", doc)
        self.assertIn("min(qcov, tcov) >= min_coverage", doc)
        self.assertNotIn("*and*", doc)

    @pytest.mark.usefixtures("fmsw_libs")
    def test_filter_drops_a_hit_with_one_side_below(self):
        """The behaviour the help describes: min(qcov, tcov) >= --min-coverage."""
        from foldmatch.search import alignment
        query = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
        subject = query + "GSHMASMTGGQQMGRGSEFELRRQACGRTRAPPPPPLRSGC" * 2  # tcov ~ 0.44
        kwargs = dict(
            query_sequences={"q": query},
            prefilter_results={"q": (["s"], [0.9])},
            fetch_subject_sequences=lambda ids: {"s": subject},
            min_seq_identity=0.0,
            max_evalue=None,
            num_workers=1,
            compute_significance=False,
        )
        kept = alignment.align_candidates(min_coverage=0.4, **kwargs)["q"]
        self.assertEqual(len(kept), 1)
        metrics = kept[0].metrics
        self.assertGreaterEqual(metrics.query_coverage, 0.99)
        self.assertLess(metrics.subject_coverage, 0.8)
        self.assertEqual(alignment.align_candidates(min_coverage=0.8, **kwargs)["q"], [])


def _commands(command, path=()):
    """Every command under a click command, by path."""
    if isinstance(command, click.Group):
        for name, sub in command.commands.items():
            yield from _commands(sub, (*path, name))
    else:
        yield path, command


def _inference_help(name):
    """The help of an inference command and of each of its options, by parameter name."""
    command = typer.main.get_command(inference.app).commands[name]
    return command.help, {p.name: p.help for p in command.params}


class LogLevelHelpTest(unittest.TestCase):

    def test_every_log_level_help_is_about_logging(self):
        """--log-level read "Number of nodes to use for inference of embeddings." in 10 commands."""
        seen = 0
        for typer_app in (app, inference.app, embedding.app):
            for path, command in _commands(typer.main.get_command(typer_app)):
                for param in command.params:
                    if param.name == "log_level":
                        seen += 1
                        with self.subTest(command=" ".join(path)):
                            self.assertRegex(param.help, r"^Logging (verbosity )?level\.$")
        self.assertGreaterEqual(seen, 24)

    def test_no_other_option_reads_like_the_old_text(self):
        for typer_app in (app, inference.app, embedding.app):
            for path, command in _commands(typer.main.get_command(typer_app)):
                for param in command.params:
                    with self.subTest(command=" ".join(path), param=param.name):
                        self.assertNotIn("inference of embeddings", getattr(param, "help", None) or "")


class InferenceOutputHelpTest(unittest.TestCase):
    """The help of the inference commands says what they write (test_inference_outputs.py runs them)."""

    def test_structure_embedding_says_parquet(self):
        """It said a pandas DataFrame file, an option out-df-name that does not exist, and a JSON file."""
        text, options = _inference_help("structure-embedding")
        for help_text in (text, options["output_folder"], options["output_name"]):
            self.assertIn("<output-name>-<rank>.parquet", help_text)
        for help_text in (text, options["output_folder"], options["output_name"]):
            for stale in ("DataFrame", "out-df-name", "JSON"):
                self.assertNotIn(stale, help_text)

    def test_the_output_format_commands_say_so(self):
        """Each said one format, the default or none of them, whatever --output-format; complete-embedding a single
        JSON file."""
        for name, default, folders in (("residue-embedding", "torch tensor files by default",
                                        {"output_folder": "chain"}),
                                       ("chain-embedding", "csv files by default", {"output_folder": "chain"}),
                                       ("assembly-embedding", "csv files by default", {"output_folder": "assembly"}),
                                       ("complete-embedding", "csv files by default",
                                        {"output_chain_path": "chain", "output_assembly_path": "assembly"})):
            with self.subTest(command=name):
                text, options = _inference_help(name)
                self.assertIn("in the format of output-format", text)
                self.assertIn(default, text)
                for folder, item in folders.items():
                    self.assertIn("in the format of output-format", options[folder])
                    self.assertIn(f"one file per {item} for csv and pt", options[folder])
                    self.assertIn("one file per process for parquet and json", options[folder])
                    self.assertNotIn("single JSON file", options[folder])

    def test_the_residue_file_formats_are_named(self):
        """chain-embedding and assembly-embedding also read csv residue embeddings."""
        for name in ("chain-embedding", "assembly-embedding"):
            with self.subTest(command=name):
                self.assertIn("stored as torch tensor or csv files (res-embedding-format)", _inference_help(name)[0])

    def test_chain_embedding_does_not_list_structure_format(self):
        self.assertNotIn("structure_format", _inference_help("chain-embedding")[1])
        for name in ("residue-embedding", "structure-embedding", "assembly-embedding", "complete-embedding"):
            with self.subTest(command=name):
                self.assertIn("structure_format", _inference_help(name)[1])


if __name__ == "__main__":
    unittest.main()
