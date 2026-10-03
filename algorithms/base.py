"""Base abstraction for CPU scheduling algorithms.

Every scheduling algorithm in this project must subclass
:class:`SchedulingAlgorithm` and implement :meth:`schedule`.

Design notes
------------
* Algorithm-specific configuration (for example Round Robin's time quantum)
  belongs in each subclass's ``__init__``.  The base class intentionally
  takes **no** configuration so that simple algorithms (FCFS, SJF) are not
  forced to accept parameters they ignore.

* The ``schedule`` method receives a :class:`~core.models.Process` sequence
  and returns a :class:`~core.models.SimulationResult`.  This keeps all
  algorithm logic independent of Streamlit, visualization, and validation.

* The ``name`` property provides a human-readable label used in
  ``SimulationResult.algorithm_name``.

* Preemptive algorithms can be added later by simply subclassing this
  base—no redesign needed.  Nothing in the interface assumes non-preemptive
  behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from core.models import Process, SimulationResult


class SchedulingAlgorithm(ABC):
    """Common interface that every scheduling algorithm implements.

    Subclass contract
    -----------------
    1. Override :attr:`name` (or implement it as a property) to return the
       algorithm's display name (e.g. ``"FCFS"``, ``"Round Robin"``).
    2. Implement :meth:`schedule` to accept a sequence of
       :class:`~core.models.Process` objects and return a
       :class:`~core.models.SimulationResult`.
    3. Accept any algorithm-specific configuration (such as a time quantum)
       in ``__init__``, **not** in ``schedule``.

    Example skeleton (do NOT use as a real implementation)::

        class FCFS(SchedulingAlgorithm):
            name = "FCFS"

            def schedule(self, processes: Sequence[Process]) -> SimulationResult:
                ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name (e.g. ``'FCFS'``)."""

    @abstractmethod
    def schedule(self, processes: Sequence[Process]) -> SimulationResult:
        """Run the scheduling algorithm on the given processes.

        Parameters
        ----------
        processes:
            The processes to schedule.  Implementations must not mutate
            this sequence (``Process`` is frozen, but the container order
            should be preserved for the caller).

        Returns
        -------
        SimulationResult
            A fully populated result including execution segments,
            per-process metrics, and aggregate statistics.

        Notes
        -----
        * Deterministic tie-breaking rules are defined by each concrete
          algorithm (e.g. by arrival time, then by PID).
        * Implementations may assume that ``processes`` is non-empty and
          already validated.  Validation is handled by a separate layer.
        """
