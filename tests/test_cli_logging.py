"""``set_log_level``, the logging set-up of every ``fm-search``, ``fm-embedding`` and ``inference`` command
(``--log-level info|warn|debug``)."""
import logging
import unittest
import warnings
from contextlib import contextmanager

from foldmatch.cli.args_utils import RankFilter, set_log_level
from foldmatch.types.api_types import LogLevel

#: The format of ``--log-level warn`` and ``--log-level debug``.
_DETAILED = '> %(asctime)s - [proc-%(rank)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
_LIGHTNING_LOGGERS = ("lightning.pytorch", "lightning.fabric.utilities.distributed")


@contextmanager
def _logging_restored():
    """The root logger's handlers and level, the Lightning loggers' levels and the warnings filters as they were."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    levels = {name: logging.getLogger(name).level for name in _LIGHTNING_LOGGERS}
    try:
        with warnings.catch_warnings():
            yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(level)
        for name, value in levels.items():
            logging.getLogger(name).setLevel(value)


class SetLogLevelTest(unittest.TestCase):

    def configured(self, level):
        """The root logger's level and its one handler's format after ``set_log_level(level)``."""
        with _logging_restored() as root:
            set_log_level(level)
            self.assertEqual(len(root.handlers), 1, root.handlers)
            [handler] = root.handlers
            self.assertTrue(any(isinstance(f, RankFilter) for f in handler.filters))
            return root.level, handler.formatter._fmt

    def test_every_level_sets_up_logging(self):
        """P9.L1: each value of ``--log-level`` has its configuration; ``warn`` had none (the branch tested
        ``'warning'``), so WARNINGs printed through Python's last-resort handler, without the format."""
        expected = {LogLevel.info: (logging.INFO, '[proc-%(rank)s] %(message)s'),
                    LogLevel.warn: (logging.WARNING, _DETAILED),
                    LogLevel.debug: (logging.DEBUG, _DETAILED)}
        self.assertEqual(set(expected), set(LogLevel))
        for level, configuration in expected.items():
            for value in (level, level.value):
                with self.subTest(level=value):
                    self.assertEqual(self.configured(value), configuration)

    def test_an_unknown_level_is_refused(self):
        for level in ("warning", "WARN", "", None):
            with self.subTest(level=level), _logging_restored() as root:
                before = list(root.handlers)
                with self.assertRaisesRegex(ValueError, "unknown log level .*: one of info, warn, debug"):
                    set_log_level(level)
                self.assertEqual(root.handlers, before)


if __name__ == "__main__":
    unittest.main()
