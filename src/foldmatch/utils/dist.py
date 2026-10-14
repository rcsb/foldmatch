"""Run identity shared by every rank of a distributed inference pass.

Each rank writes its own embedding shard, ``emb_{run_id}-{rank}.parquet``, and
rank 0 reads all of them back once the pass is over. That only works if every
rank derives the same ``run_id`` *before* the process group exists, from what
it was launched with:

* ``FOLDMATCH_RUN_ID`` when set (by the user, or inherited from the parent);
* in a task of an ``srun`` step, whose ranks SLURM starts itself,
  ``{SLURM_JOB_ID}_{SLURM_STEP_ID}_{SLURM_RESTART_COUNT}_{digest}``: the
  same on every task of the step and different for each step and requeue,
  while ``digest``, of the command's arguments, tells apart the commands that
  one step runs one after another (a build, then a query);
* otherwise a fresh random id, which only the process that starts the run may
  draw. It exports the id (:func:`exported_run_id`) around ``trainer.predict``
  so the child processes Lightning launches inherit it. A rank > 0 without an
  id refuses to start, rather than writing a shard rank 0 never reads.

The batch step of an ``sbatch`` script is not an srun step: it is one task,
so only Lightning can start more ranks there, and they inherit the id. Each
command it runs therefore draws its own id, instead of all of them sharing
the job's.

Dependency-free on purpose: it runs before torch or any model is loaded.
"""
import hashlib
import os
import re
import sys
from contextlib import contextmanager
from secrets import token_hex
from typing import Iterator, Optional

RUN_ID_ENV = "FOLDMATCH_RUN_ID"

# SLURM numbers ordinary steps below this; the values above it name the batch,
# extern, interactive and pending steps (SLURM_MAX_NORMAL_STEP_ID in slurm.h).
_SLURM_MAX_NORMAL_STEP_ID = 0xFFFFFFF0

# The id becomes part of file names and of a glob pattern, so it is restricted
# to characters that are neither path separators nor glob metacharacters.
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def is_run_id(value) -> bool:
    """Whether ``value`` is a well-formed run id (what :func:`run_id` accepts in ``FOLDMATCH_RUN_ID``)."""
    return isinstance(value, str) and _RUN_ID_RE.fullmatch(value) is not None


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def launch_rank() -> int:
    """This process's rank as given by its launch environment.

    Read from environment variables only, so it can be called before the
    process group is initialised. ``RANK`` (torchrun) is the global rank. If
    it is absent, the result is the largest of ``SLURM_PROCID``, ``NODE_RANK``
    and ``LOCAL_RANK`` (Lightning's launcher): exact on a single node or under
    SLURM, and in every case 0 exactly for the process that starts the run.
    """
    rank = _env_int("RANK")
    if rank is not None:
        return rank
    return max(_env_int("SLURM_PROCID") or 0, _env_int("NODE_RANK") or 0, _env_int("LOCAL_RANK") or 0)


def srun_step_id() -> Optional[str]:
    """``SLURM_STEP_ID`` when this process is a task of an ``srun`` step, else ``None``.

    The batch step of an sbatch script has no ``SLURM_STEP_ID``, or one of
    SLURM's special step values, and is not an srun step.
    """
    step = os.environ.get("SLURM_STEP_ID", "").strip()
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        return None
    if not step.isdigit() or int(step) >= _SLURM_MAX_NORMAL_STEP_ID:
        return None
    return step


def command_digest() -> str:
    """Eight hex digits of SHA-1 over this command's arguments (``sys.argv[1:]``).

    Every task of an srun step runs the same command line, so they agree;
    two different commands in one step do not. The program path is left out,
    so ``fm-search ...`` and ``python -m foldmatch.cli.search ...`` agree too.
    """
    arguments = b"\0".join(os.fsencode(argument) for argument in sys.argv[1:])
    return hashlib.sha1(arguments).hexdigest()[:8]


def run_id() -> str:
    """The id shared by every rank of this run (see the module docstring).

    Raises ``ValueError`` for a malformed ``FOLDMATCH_RUN_ID``, and on a rank
    > 0 that has no id to share: drawing its own would write a shard under a
    name rank 0 never reads, silently dropping that rank's embeddings.
    """
    explicit = os.environ.get(RUN_ID_ENV)
    if explicit:
        if not _RUN_ID_RE.fullmatch(explicit):
            raise ValueError(
                f"{RUN_ID_ENV}={explicit!r} is not a valid run id: use letters, digits, "
                f"'.', '_' and '-' only, starting with a letter or digit."
            )
        return explicit
    step_id = srun_step_id()
    if step_id is not None:
        job_id = os.environ["SLURM_JOB_ID"].strip()
        restarts = os.environ.get("SLURM_RESTART_COUNT", "").strip() or "0"
        return f"{job_id}_{step_id}_{restarts}_{command_digest()}"
    rank = launch_rank()
    if rank > 0:
        raise ValueError(
            f"This process is rank {rank} of a distributed run but has no run id to share "
            f"with rank 0, so rank 0 would never read its embeddings. Set {RUN_ID_ENV} to "
            f"the same value on every rank whenever neither srun nor Lightning starts the "
            f"ranks (torchrun, or one command per node with NODE_RANK set)."
        )
    return token_hex(8)


@contextmanager
def exported_run_id(rid: str) -> Iterator[None]:
    """Export ``rid`` as ``FOLDMATCH_RUN_ID`` for the duration of the block.

    Wrap ``trainer.predict`` in it: Lightning launches the other ranks as child
    processes that re-run the command and inherit the environment, so they
    compute the same :func:`run_id`. The previous value is restored afterwards:
    an id the user set stays set, and one exported here is cleared, so a later
    run in the same process draws a fresh id.
    """
    previous = os.environ.get(RUN_ID_ENV)
    os.environ[RUN_ID_ENV] = rid
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(RUN_ID_ENV, None)
        else:
            os.environ[RUN_ID_ENV] = previous
