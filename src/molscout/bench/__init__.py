"""Benchmark harness: run one tool on one crop dataset, then check, score and record the run."""

from __future__ import annotations


class BenchError(RuntimeError):
    """A benchmark run cannot start, its tool failed, or its output cannot be accepted."""


class Terminated(BaseException):
    """SIGTERM arrived during a run: a SLURM time limit or scancel.

    Like KeyboardInterrupt it is not an Exception, so nothing on the way can swallow it,
    and the run's cleanup (stop the tool, remove staging) still happens.
    """
