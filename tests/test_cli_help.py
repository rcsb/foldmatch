"""CLI help text that states what the code does (G10)."""
import unittest

import typer

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


if __name__ == "__main__":
    unittest.main()
