"""Non-preemptive Priority scheduling algorithm.

At each scheduling decision point the algorithm selects the **arrived**
process with the highest priority.  Lower numeric ``priority`` values
represent higher priority (priority 1 beats priority 3).

Ties are broken deterministically by ``arrival_time``, then ``pid``
(lexicographic).  Effective selection key: ``priority → arrival_time → pid``.

Priority scheduling is non-preemptive: once a process begins executing
it runs its entire burst before the scheduler picks the next process.
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


class PriorityScheduler(SchedulingAlgorithm):
    """Non-preemptive Priority scheduler.

    Lower numeric ``priority`` value = higher scheduling priority.
    """

    @property
    def name(self) -> str:  # noqa: D401
        """Algorithm display name."""
        return "Priority"

    def schedule(self, processes: Sequence[Process]) -> SimulationResult:
        """Schedule *processes* using non-preemptive Priority.

        Parameters
        ----------
        processes:
            Non-empty sequence of validated :class:`Process` objects.
            May arrive in any order; the algorithm handles selection
            internally.
        """
        if not processes:
            raise ValueError("Priority scheduling requires at least one process.")

        remaining: list[Process] = list(processes)
        segments: list[ExecutionSegment] = []
        metrics: list[ProcessMetrics] = []
        current_time = 0

        while remaining:
            # Find processes that have arrived by current_time.
            ready = [p for p in remaining if p.arrival_time <= current_time]

            if not ready:
                # No process is ready — advance to the earliest arrival.
                next_arrival = min(p.arrival_time for p in remaining)
                segments.append(
                    ExecutionSegment(
                        process_id=IDLE_PROCESS_ID,
                        start_time=current_time,
                        end_time=next_arrival,
                    )
                )
                current_time = next_arrival
                ready = [p for p in remaining if p.arrival_time <= current_time]

            # Select highest priority (lowest numeric value);
            # ties broken by arrival_time, then pid.
            selected = min(ready, key=lambda p: (p.priority, p.arrival_time, p.pid))
            remaining.remove(selected)

            # Execute (non-preemptive: full burst).
            start = current_time
            end = start + selected.burst_time
            segments.append(
                ExecutionSegment(
                    process_id=selected.pid,
                    start_time=start,
                    end_time=end,
                )
            )
            current_time = end

            # Per-process metrics.
            completion_time = end
            turnaround_time = completion_time - selected.arrival_time
            waiting_time = turnaround_time - selected.burst_time

            metrics.append(
                ProcessMetrics(
                    pid=selected.pid,
                    arrival_time=selected.arrival_time,
                    burst_time=selected.burst_time,
                    priority=selected.priority,
                    completion_time=completion_time,
                    turnaround_time=turnaround_time,
                    waiting_time=waiting_time,
                    response_time=waiting_time,  # Non-preemptive: response == waiting
                )
            )

        # Aggregate metrics.
        avg_waiting = sum(m.waiting_time for m in metrics) / len(metrics)
        avg_turnaround = sum(m.turnaround_time for m in metrics) / len(metrics)

        return SimulationResult(
            algorithm_name=self.name,
            execution_segments=tuple(segments),
            process_metrics=tuple(metrics),
            average_waiting_time=avg_waiting,
            average_turnaround_time=avg_turnaround,
        )
