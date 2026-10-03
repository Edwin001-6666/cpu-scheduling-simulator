"""First Come First Serve (FCFS) scheduling algorithm.

Processes are dispatched in order of arrival time. When two or more
processes share the same arrival time, ties are broken by PID
(lexicographic order) for deterministic results.

FCFS is non-preemptive: once a process starts executing it runs to
completion before the next process is dispatched.
"""

from __future__ import annotations

from typing import Sequence

from algorithms.base import SchedulingAlgorithm
from core.models import (
    IDLE_PROCESS_ID,
    ExecutionSegment,
    Process,
    ProcessMetrics,
    SimulationResult,
)


class FCFS(SchedulingAlgorithm):
    """First Come First Serve scheduler."""

    @property
    def name(self) -> str:  # noqa: D401
        """Algorithm display name."""
        return "FCFS"

    def schedule(self, processes: Sequence[Process]) -> SimulationResult:
        """Schedule *processes* using FCFS and return a SimulationResult.

        Parameters
        ----------
        processes:
            Non-empty sequence of validated :class:`Process` objects.
            The sequence may arrive in any order; FCFS sorts internally
            by ``(arrival_time, pid)``.
        """
        if not processes:
            raise ValueError("FCFS requires at least one process.")

        # --- 1. Sort by arrival time, then PID for deterministic ordering ---
        sorted_procs = sorted(processes, key=lambda p: (p.arrival_time, p.pid))

        # --- 2. Simulate execution -----------------------------------------
        segments: list[ExecutionSegment] = []
        metrics: list[ProcessMetrics] = []
        current_time = 0

        for proc in sorted_procs:
            # Insert idle segment if the CPU must wait for this process.
            if proc.arrival_time > current_time:
                segments.append(
                    ExecutionSegment(
                        process_id=IDLE_PROCESS_ID,
                        start_time=current_time,
                        end_time=proc.arrival_time,
                    )
                )
                current_time = proc.arrival_time

            # Execute the process (non-preemptive: runs full burst).
            start = current_time
            end = start + proc.burst_time
            segments.append(
                ExecutionSegment(
                    process_id=proc.pid,
                    start_time=start,
                    end_time=end,
                )
            )
            current_time = end

            # Per-process metrics.
            completion_time = end
            turnaround_time = completion_time - proc.arrival_time
            waiting_time = turnaround_time - proc.burst_time

            metrics.append(
                ProcessMetrics(
                    pid=proc.pid,
                    arrival_time=proc.arrival_time,
                    burst_time=proc.burst_time,
                    priority=proc.priority,
                    completion_time=completion_time,
                    turnaround_time=turnaround_time,
                    waiting_time=waiting_time,
                    response_time=waiting_time,  # FCFS: response == waiting
                )
            )

        # --- 3. Aggregate metrics ------------------------------------------
        avg_waiting = sum(m.waiting_time for m in metrics) / len(metrics)
        avg_turnaround = sum(m.turnaround_time for m in metrics) / len(metrics)

        return SimulationResult(
            algorithm_name=self.name,
            execution_segments=tuple(segments),
            process_metrics=tuple(metrics),
            average_waiting_time=avg_waiting,
            average_turnaround_time=avg_turnaround,
        )
