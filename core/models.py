"""Shared data contracts for the CPU Scheduling Lab.

These models describe process input, CPU execution segments, per-process
metrics, and a complete simulation result. They contain no scheduling
algorithm logic, no Streamlit/UI coupling, and no metric calculations.

Intended consumers: algorithms, validation, metrics, Streamlit, Gantt
visualization, and tests.

Time values are discrete non-negative integers (time units). Full
validation is implemented elsewhere; documented constraints below are
the intended contract, not enforced here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Reserved process identifier for CPU idle time on a Gantt chart.
# Example: ExecutionSegment(process_id=IDLE_PROCESS_ID, start_time=3, end_time=5)
# renders conceptually as IDLE: 3 → 5.
IDLE_PROCESS_ID = "IDLE"


@dataclass(frozen=True)
class Process:
    """Input process for a scheduling simulation.

    Fields
    ------
    pid:
        Process identifier (for example ``\"P1\"``). Must not equal
        ``IDLE_PROCESS_ID``; that token is reserved for idle Gantt segments.
    arrival_time:
        Time unit when the process becomes ready. Intended: ``>= 0``.
    burst_time:
        CPU time required. Intended: ``> 0``.
    priority:
        Scheduling priority. Lower numbers mean higher priority. Unused by
        algorithms that ignore priority (for example FCFS); still required
        so every process uses the same input shape.

    This model is input data only. Do not attach dispatch or queue logic.
    """

    pid: str
    arrival_time: int
    burst_time: int
    priority: int


@dataclass(frozen=True)
class ExecutionSegment:
    """One continuous CPU interval on the Gantt chart.

    Fields
    ------
    process_id:
        ``Process.pid`` for a running process, or ``IDLE_PROCESS_ID``
        (``\"IDLE\"``) when the CPU is idle.
    start_time:
        Inclusive start of the interval. Intended: ``>= 0``.
    end_time:
        Exclusive end of the interval. Intended: ``>= start_time``.
        Zero-length segments (``start_time == end_time``) are allowed by
        the type; validation may reject them later if unused.

    Idle periods are first-class segments so a future Gantt chart can
    emit a sequence such as::

        P1: 0 → 3
        IDLE: 3 → 5
        P2: 5 → 8
    """

    process_id: str
    start_time: int
    end_time: int

    @property
    def is_idle(self) -> bool:
        """True when this segment represents CPU idle time."""
        return self.process_id == IDLE_PROCESS_ID


@dataclass(frozen=True)
class ProcessMetrics:
    """Final per-process metrics after a simulation.

    Values are stored, not computed, in this module.

    Fields
    ------
    pid, arrival_time, burst_time, priority:
        Copied from the originating :class:`Process` for table display.
    completion_time:
        Time when the process finished its last CPU burst.
    turnaround_time:
        Intended meaning: ``completion_time - arrival_time``.
    waiting_time:
        Intended meaning: ``turnaround_time - burst_time``.
    response_time:
        Optional. Time from arrival until first CPU dispatch. ``None``
        until metrics code fills it; added as an optional field so later
        work does not change the type shape.
    """

    pid: str
    arrival_time: int
    burst_time: int
    priority: int
    completion_time: int
    turnaround_time: int
    waiting_time: int
    response_time: int | None = None


@dataclass(frozen=True)
class SimulationResult:
    """Complete output of one scheduling run.

    Fields
    ------
    algorithm_name:
        Human-readable algorithm label (for example ``\"FCFS\"``).
    execution_segments:
        Ordered Gantt intervals, including idle segments.
    process_metrics:
        One :class:`ProcessMetrics` row per input process.
    average_waiting_time:
        Mean of per-process waiting times.
    average_turnaround_time:
        Mean of per-process turnaround times.
    average_response_time:
        Optional aggregate; ``None`` until metrics fill it.
    cpu_utilization:
        Optional fraction or percentage of busy CPU time; ``None`` until
        metrics fill it. Units will be decided by the metrics layer.
    throughput:
        Optional processes completed per time unit; ``None`` until
        metrics fill it.

    Sequences are tuples so the result is immutable after construction.
    Algorithms may build lists locally, then convert with ``tuple(...)``.
    """

    algorithm_name: str
    execution_segments: tuple[ExecutionSegment, ...]
    process_metrics: tuple[ProcessMetrics, ...]
    average_waiting_time: float
    average_turnaround_time: float
    average_response_time: float | None = None
    cpu_utilization: float | None = None
    throughput: float | None = None
