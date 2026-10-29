"""``set_log_level``, the logging set-up of every ``fm-search``, ``fm-embedding`` and ``inference`` command
(``--log-level info|warn|debug``)."""
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from pathlib import Path

import foldmatch
from foldmatch.cli import args_utils
from foldmatch.cli.args_utils import RankFilter, set_log_level
from foldmatch.types.api_types import LogLevel

_SRC = str(Path(foldmatch.__file__).resolve().parent.parent)

#: The format of ``--log-level warn`` and ``--log-level debug``.
_DETAILED = '> %(asctime)s - [proc-%(rank)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
_LIGHTNING_LOGGERS = ("lightning.pytorch", "lightning.fabric.utilities.distributed")


@contextmanager
def _logging_restored():
    """The root logger's handlers and level, the Lightning loggers' levels, the warnings filters and what a deferred
    ``set_log_level`` recorded, as they were."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    levels = {name: logging.getLogger(name).level for name in _LIGHTNING_LOGGERS}
    recorded = dict(args_utils._ROOT_BEFORE_LIGHTNING)
    try:
        with warnings.catch_warnings():
            yield root
    finally:
        args_utils._ROOT_BEFORE_LIGHTNING.clear()
        args_utils._ROOT_BEFORE_LIGHTNING.update(recorded)
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


# =====================================================================================================================
# P9.L2: set_log_level(..., defer_lightning=True), and Lightning imported before, by or after set_log_level
# =====================================================================================================================
#: A sitecustomize: any import of torch, Lightning, ESM, FAISS, transformers or huggingface_hub ends the process with
#: status 97, naming the module (as ``test_cli_split_build._NO_HEAVY_IMPORTS``).
_NO_HEAVY_IMPORTS = r"""
import os, sys
_BLOCKED = ("torch", "lightning", "lightning_fabric", "pytorch_lightning", "esm", "faiss", "transformers",
            "huggingface_hub")
class _NoHeavyImports:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in _BLOCKED:
            sys.stderr.write(f"FORBIDDEN IMPORT: {name}\n")
            sys.stderr.flush()
            os._exit(97)
        return None
sys.meta_path.insert(0, _NoHeavyImports())
"""

#: Sets up logging at $DRIVER_LEVEL as $DRIVER_MODE says, then logs FoldMatch's records, Lightning's own (a Trainer's
#: INFO lines among them), a warning; prints the logging, warnings and progress-bar state as JSON. The mode and level
#: come from the environment, so that every run has the same sys.argv (Lightning's srun hint, where srun exists,
#: quotes it).
_LIGHTNING_DRIVER = r"""
import json, logging, os, sys, warnings
mode, level = os.environ["DRIVER_MODE"], os.environ["DRIVER_LEVEL"]
from foldmatch.cli.args_utils import set_log_level
if mode.startswith("configured before"):
    logging.basicConfig(level=logging.WARNING, format="configured before: %(message)s")
if mode == "imported before":
    import lightning
elif "deferred" in mode:
    set_log_level(level, defer_lightning=True)
    if mode == "deferred, imported meanwhile":
        import lightning
set_log_level(level)
sys.stderr.write("=== set up\n")
sys.stderr.flush()
logging.debug("fm debug")
logging.info("fm info")
logging.warning("fm warning")
logging.getLogger("lightning.pytorch.core").error("a lightning error")
logging.getLogger("lightning.fabric.utilities.distributed").info("Initializing distributed: GLOBAL_RANK: 0")
warnings.warn("a user warning")
import lightning
lightning.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                  enable_model_summary=False)
from huggingface_hub.utils import are_progress_bars_disabled
import transformers.utils.logging as transformers_logging
def handlers(logger):
    return [(type(h).__name__, h.level, getattr(h.formatter, "_fmt", None), [type(f).__name__ for f in h.filters])
            for h in logger.handlers]
loggers = {name or "root": {"level": logging.getLogger(name).level, "propagate": logging.getLogger(name).propagate,
                            "handlers": handlers(logging.getLogger(name))}
           for name in ("", "lightning", "lightning.pytorch", "lightning.fabric",
                        "lightning.fabric.utilities.distributed")}
print(json.dumps({"loggers": loggers, "warnings": [repr(f) for f in warnings.filters],
                  "hub bars disabled": are_progress_bars_disabled(),
                  "transformers tqdm": transformers_logging._tqdm_active,
                  "env": {k: os.environ.get(k) for k in ("KMP_DUPLICATE_LIB_OK", "PYTORCH_NVML_BASED_CUDA_CHECK",
                                                         "HF_HUB_DISABLE_PROGRESS_BARS")}}))
"""

#: The deferred set-up alone: FoldMatch's records and a warning, then whether huggingface_hub was imported.
_DEFERRED_DRIVER = r"""
import json, logging, os, sys, warnings
from foldmatch.cli.args_utils import set_log_level
set_log_level(os.environ["DRIVER_LEVEL"], defer_lightning=os.environ["DRIVER_MODE"] == "deferred")
logging.debug("fm debug")
logging.info("fm info")
logging.warning("fm warning")
warnings.warn("a user warning")
print(json.dumps({"HF_HUB_DISABLE_PROGRESS_BARS": os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")}))
"""

_TIME = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}")


def _line(driver, statement):
    """The line number of ``statement`` in ``driver``, as the records it logs carry it (``<string>:N``)."""
    return driver.splitlines().index(statement) + 1


class LightningLoggingTest(unittest.TestCase):
    """P9.L2: the log output and the logging state are the same whether Lightning is imported by ``set_log_level``
    (as before), before it, or by a second ``set_log_level`` after a ``set_log_level(..., defer_lightning=True)``; and
    when Lightning was imported between the two, from the second call on. At ``info`` Lightning's INFO lines (a
    Trainer's 'GPU available: ...') stay hidden, at ``debug`` they show; the rank prefix is the launcher's."""

    MODES = ("by set_log_level", "imported before", "deferred", "deferred, imported meanwhile")
    CONFIGURED_BEFORE = ("configured before, by set_log_level", "configured before, deferred")

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def run_driver(self, driver, mode, level, hook=False):
        (self.tmp / "site").mkdir(exist_ok=True)
        (self.tmp / "site" / "sitecustomize.py").write_text(_NO_HEAVY_IMPORTS)
        path = ([str(self.tmp / "site")] if hook else []) + [_SRC]
        env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_DISABLE_PROGRESS_BARS", "PYTHONWARNINGS")}
        env = {k: v for k, v in env.items() if k not in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")}
        # GLOBAL_RANK gives the records' rank prefix; Lightning, which reads the others, logs as rank 0.
        env.update(PYTHONPATH=os.pathsep.join(path), GLOBAL_RANK="3", HF_HUB_OFFLINE="1", DRIVER_MODE=mode,
                   DRIVER_LEVEL=level)
        result = subprocess.run([sys.executable, "-c", driver], env=env, cwd=self.tmp, capture_output=True,
                                text=True, timeout=600)
        self.assertNotIn("FORBIDDEN IMPORT", result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        return _TIME.sub("TIME", result.stderr), json.loads(result.stdout)

    def test_lightning_imported_before_by_or_after_set_log_level(self):
        for level in LogLevel:
            runs = {mode: self.run_driver(_LIGHTNING_DRIVER, mode, level.value) for mode in self.MODES}
            stderr, state = runs["by set_log_level"]
            lines = stderr.splitlines()
            with self.subTest(level=level):
                self.assertEqual(lines[0], "=== set up", stderr)
                self.assertIn("a lightning error", lines)  # Lightning's own handler, as before
                if level is LogLevel.info:
                    self.assertEqual(lines[1:3], ["[proc-3] fm info", "[proc-3] fm warning"], stderr)
                    for hidden in ("GPU available", "Initializing distributed", "fm debug", "a user warning"):
                        self.assertNotIn(hidden, stderr)
                else:
                    warning, info = (_line(_LIGHTNING_DRIVER, f'logging.{name}("fm {name}")')
                                     for name in ("warning", "info"))
                    self.assertIn(f"> TIME - [proc-3] - WARNING - <string>:{warning} - fm warning", lines)
                    self.assertIn("Initializing distributed: GLOBAL_RANK: 0", lines)
                    self.assertTrue(any(line.startswith("GPU available: ") for line in lines), stderr)
                    self.assertIn("UserWarning: a user warning", stderr)
                    self.assertEqual(f"> TIME - [proc-3] - INFO - <string>:{info} - fm info" in lines,
                                     level is LogLevel.debug, stderr)
                self.assertEqual(state["loggers"]["lightning.pytorch"]["propagate"], False)
                self.assertEqual(state["loggers"]["lightning.pytorch"]["level"],
                                 logging.ERROR if level is LogLevel.info else logging.INFO)
                self.assertEqual(state["hub bars disabled"], level is LogLevel.info)
            for mode in self.MODES[1:]:
                with self.subTest(level=level, mode=mode):
                    other, other_state = runs[mode]
                    self.assertEqual(other_state, state)
                    if mode == "deferred, imported meanwhile":  # what that import logged is not compared
                        self.assertEqual(other.partition("=== set up")[2], stderr.partition("=== set up")[2])
                    else:
                        self.assertEqual(other, stderr)

    def test_a_root_logger_configured_before(self):
        """With the root logger configured before ``set_log_level`` (in process), Lightning gives its loggers no
        handler of their own, and a deferred set-up imports it with the root logger at the level it found: the same
        output and state as ``set_log_level(level)``."""
        for level in LogLevel:
            (stderr, state), (other, other_state) = (self.run_driver(_LIGHTNING_DRIVER, mode, level.value)
                                                     for mode in self.CONFIGURED_BEFORE)
            with self.subTest(level=level):
                self.assertEqual((state["loggers"]["lightning.pytorch"]["propagate"],
                                  state["loggers"]["lightning.pytorch"]["handlers"]), (True, []))
                self.assertEqual("GPU available" in stderr, level is not LogLevel.info, stderr)
                self.assertEqual((other, other_state), (stderr, state))

    def test_the_deferred_set_up_imports_nothing_heavy(self):
        """With ``defer_lightning=True`` the records come in the format of the level, the rank prefix included, as
        after ``set_log_level(level)``, with nothing imported that imports torch (the hook; ``RankFilter`` imported
        torch.distributed at the first record before P9.L2), and the progress bars are left alone (no
        HF_HUB_DISABLE_PROGRESS_BARS)."""
        for level in LogLevel:
            with self.subTest(level=level):
                deferred, state = self.run_driver(_DEFERRED_DRIVER, "deferred", level.value, hook=True)
                self.assertEqual(state, {"HF_HUB_DISABLE_PROGRESS_BARS": None})
                self.assertEqual(deferred, self.run_driver(_DEFERRED_DRIVER, "by set_log_level", level.value)[0])
                self.assertIn("[proc-3] fm warning" if level is LogLevel.info else "- [proc-3] - WARNING - ", deferred)

    def test_the_progress_bars_when_huggingface_hub_is_imported(self):
        """At ``info`` the deferred set-up disables the Hugging Face progress bars as before when huggingface_hub is
        already imported."""
        from huggingface_hub.utils import are_progress_bars_disabled, enable_progress_bars
        try:
            with _logging_restored():
                enable_progress_bars()
                set_log_level(LogLevel.debug, defer_lightning=True)
                self.assertFalse(are_progress_bars_disabled())
                set_log_level(LogLevel.info, defer_lightning=True)
                self.assertTrue(are_progress_bars_disabled())
        finally:
            enable_progress_bars()


if __name__ == "__main__":
    unittest.main()
