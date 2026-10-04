"""Benchmark harness: run one tool on one crop dataset, then check, score and record the run."""

from __future__ import annotations


class BenchError(RuntimeError):
    """A benchmark run cannot start, its tool failed, or its output cannot be accepted."""
