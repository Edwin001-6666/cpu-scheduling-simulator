"""Metrics enhancement layer for the CPU Scheduling Lab.

This module computes the three additional aggregate/per-process metrics
that the scheduling algorithms currently leave as ``None`` in their output:

* **Response time** — per-process: time from arrival until first CPU dispatch.
* **CPU utilization** — aggregate: fraction of total simulation time during
  which the CPU is executing a real process (not idle).
* **Throughput** — aggregate: number of processes completed per unit time.

Design
------
The public API is a single function :func:`compute_metrics` that accepts a
:class:`~core.models.SimulationResult` (as produced by any scheduling
algorithm) together with the original process list, and returns a *new*
``SimulationResult`` with all optional fields filled.

    enriched = compute_metrics(result, processes)

This keeps the metrics logic:

* **Independent of Streamlit** — pure Python, no UI coupling.
* **Independent of algorithm internals** — derives everything from the
  public ``SimulationResult`` / ``execution_segments`` contract.
* **Non-mutating** — input ``SimulationResult`` is frozen (dataclass); this
  function constructs and returns a new instance.
* **Reusable** — Laptop C (Streamlit UI) calls ``compute_metrics`` after
  calling ``algo.schedule()``.

Formulas
--------
Response time (per process)::

    response_time = first_dispatch_start_time - arrival_time

    where first_dispatch_start_time is the start_time of the earliest
    ExecutionSegment whose process_id equals the process's PID.

Average response time::

    average_response_time = mean(response_time for all processes)

CPU utilization (percentage, 0.0–100.0)::

    busy_time   = sum(seg.end_time - seg.start_time
                      for seg in execution_segments if not seg.is_idle)
    total_time  = last segment end_time - first segment start_time
    cpu_utilization = (busy_time / total_time) * 100.0

    Special case: if total_time == 0 (theoretically impossible given
    burst_time > 0, but handled defensively) → cpu_utilization = 0.0.

Throughput (processes per unit time)::

    total_time  = last segment end_time - first segment start_time
    throughput  = number_of_processes / total_time

    Special case: total_time == 0 → throughput = 0.0.

Idle time handling
------------------
The ``execution_segments`` tuple contains explicit ``IDLE`` segments for all
CPU idle periods.  Busy time is computed by summing only the non-idle
segment durations; idle segment durations are thus excluded automatically.
``total_time`` spans the full simulation interval (first start → last end),
which naturally includes idle gaps.

Round Robin response time
-------------------------
Round Robin is preemptive.  A process may have many execution segments.
Response time is determined by scanning ``execution_segments`` in order and
recording the *first* segment start_time for each PID.  The ``process_id``
field on each ``ExecutionSegment`` identifies which process ran.
"""

from __future__ import annotations

from typing import Sequence

from core.models import (
    IDLE_PROCESS_ID,
    Process,
    ProcessMetrics,
    SimulationResult,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(
    result: SimulationResult,
    processes: Sequence[Process],
) -> SimulationResult:
    """Return an enriched copy of *result* with all optional metrics filled.

    Parameters
    ----------
    result:
        A :class:`~core.models.SimulationResult` as returned by any
        scheduling algorithm.  The optional fields
        ``average_response_time``, ``cpu_utilization``, and ``throughput``
        may be ``None`` on input; they will be filled on output.
    processes:
        The original input process list passed to the scheduling algorithm.
        Used to look up ``arrival_time`` for response-time calculation.
        Must contain the same processes (same PIDs) as those referenced in
        *result*.

    Returns
    -------
    SimulationResult
        A new ``SimulationResult`` instance identical to *result* except that
        ``process_metrics`` entries all have ``response_time`` set,
        ``average_response_time`` is set, ``cpu_utilization`` is set, and
        ``throughput`` is set.

    Notes
    -----
    * The input *result* is not modified (it is a frozen dataclass).
    * *processes* is not modified.
    * Numeric precision follows Python's native ``float``.
    """
    segments = result.execution_segments

    # --- Build arrival map from the original process list -------------------
    arrival: dict[str, int] = {p.pid: p.arrival_time for p in processes}

    # --- Compute per-process response times from execution_segments ---------
    # Scan segments in order; record the FIRST start_time for each pid.
    first_dispatch: dict[str, int] = {}
    for seg in segments:
        if not seg.is_idle and seg.process_id not in first_dispatch:
            first_dispatch[seg.process_id] = seg.start_time

    # --- Rebuild ProcessMetrics with response_time filled -------------------
    enriched_metrics: list[ProcessMetrics] = []
    for m in result.process_metrics:
        pid = m.pid
        rt = first_dispatch[pid] - arrival[pid]
        enriched_metrics.append(
            ProcessMetrics(
                pid=m.pid,
                arrival_time=m.arrival_time,
                burst_time=m.burst_time,
                priority=m.priority,
                completion_time=m.completion_time,
                turnaround_time=m.turnaround_time,
                waiting_time=m.waiting_time,
                response_time=rt,
            )
        )

    # --- Aggregate: average response time -----------------------------------
    avg_rt = sum(m.response_time for m in enriched_metrics) / len(enriched_metrics)  # type: ignore[arg-type]

    # --- Aggregate: CPU utilization -----------------------------------------
    # Total simulation wall time (first segment start → last segment end).
    total_time: int = 0
    if segments:
        total_time = segments[-1].end_time - segments[0].start_time

    busy_time = sum(
        seg.end_time - seg.start_time
        for seg in segments
        if not seg.is_idle
    )

    cpu_utilization: float = 0.0
    if total_time > 0:
        cpu_utilization = (busy_time / total_time) * 100.0

    # --- Aggregate: throughput ----------------------------------------------
    n_processes = len(enriched_metrics)
    throughput: float = 0.0
    if total_time > 0:
        throughput = n_processes / total_time

    # --- Return enriched SimulationResult -----------------------------------
    return SimulationResult(
        algorithm_name=result.algorithm_name,
        execution_segments=result.execution_segments,
        process_metrics=tuple(enriched_metrics),
        average_waiting_time=result.average_waiting_time,
        average_turnaround_time=result.average_turnaround_time,
        average_response_time=avg_rt,
        cpu_utilization=cpu_utilization,
        throughput=throughput,
    )
