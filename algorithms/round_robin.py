"""Preemptive Round Robin scheduling algorithm.

Each process runs for at most ``time_quantum`` time units before being
preempted and placed at the back of the ready queue.  If a process
finishes within its quantum, it completes immediately without using the
remaining slice.

Ready-queue policy
------------------
* Processes are enqueued in arrival-time order.  When multiple processes
  share the same arrival time, they are enqueued in PID (lexicographic)
  order for determinism.
* When a quantum expires and the running process is preempted **at the
  same instant** new processes arrive, the **newly arriving processes are
  enqueued first**, then the preempted process is placed at the back.
  This is the standard textbook convention and avoids starvation of
  late-arriving short jobs.
* A process that arrives exactly at a quantum boundary is considered
  ready at that time and enters the queue before the preempted process.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from algorithms.base import SchedulingAlgorithm
from core.models import (
    IDLE_PROCESS_ID,
    ExecutionSegment,
    Process,
    ProcessMetrics,
    SimulationResult,
)


class RoundRobin(SchedulingAlgorithm):
    """Preemptive Round Robin scheduler.

    Parameters
    ----------
    time_quantum:
        Positive integer — maximum CPU time a process may use before
        being preempted.
    """

    def __init__(self, time_quantum: int) -> None:
        if not isinstance(time_quantum, int) or time_quantum <= 0:
            raise ValueError(
                f"time_quantum must be a positive integer, got {time_quantum!r}"
            )
        self._time_quantum = time_quantum

    @property
    def name(self) -> str:  # noqa: D401
        """Algorithm display name."""
        return "Round Robin"

    @property
    def time_quantum(self) -> int:
        """The configured time quantum."""
        return self._time_quantum

    def schedule(self, processes: Sequence[Process]) -> SimulationResult:
        """Schedule *processes* using preemptive Round Robin.

        Parameters
        ----------
        processes:
            Non-empty sequence of validated :class:`Process` objects.
            May arrive in any order.
        """
        if not processes:
            raise ValueError("Round Robin requires at least one process.")

        quantum = self._time_quantum

        # Sort a working copy by (arrival_time, pid) for deterministic
        # enqueue order.  We never mutate the caller's sequence.
        sorted_procs = sorted(processes, key=lambda p: (p.arrival_time, p.pid))

        # Remaining burst time for each process, keyed by pid.
        remaining_burst: dict[str, int] = {p.pid: p.burst_time for p in sorted_procs}

        # Completion time for each process, recorded when it finishes.
        completion_times: dict[str, int] = {}

        # Ready queue (FIFO of pids).
        ready: deque[str] = deque()

        # Pointer into sorted_procs: next process index to consider for
        # arrival.
        next_idx = 0

        segments: list[ExecutionSegment] = []
        current_time = 0

        def _enqueue_arrivals(up_to_time: int) -> None:
            """Enqueue all processes whose arrival_time <= up_to_time
            that haven't been enqueued yet."""
            nonlocal next_idx
            while next_idx < len(sorted_procs):
                p = sorted_procs[next_idx]
                if p.arrival_time <= up_to_time:
                    ready.append(p.pid)
                    next_idx += 1
                else:
                    break

        # Seed the ready queue at time 0.
        _enqueue_arrivals(current_time)

        while ready or next_idx < len(sorted_procs):
            # --- Handle CPU idle if the queue is empty ---
            if not ready:
                next_arrival = sorted_procs[next_idx].arrival_time
                segments.append(
                    ExecutionSegment(
                        process_id=IDLE_PROCESS_ID,
                        start_time=current_time,
                        end_time=next_arrival,
                    )
                )
                current_time = next_arrival
                _enqueue_arrivals(current_time)
                continue

            # --- Dispatch the front of the queue ---
            pid = ready.popleft()
            run_time = min(quantum, remaining_burst[pid])
            start = current_time
            end = start + run_time

            segments.append(
                ExecutionSegment(
                    process_id=pid,
                    start_time=start,
                    end_time=end,
                )
            )

            remaining_burst[pid] -= run_time
            current_time = end

            # Enqueue processes that arrived during this slice (up to and
            # including current_time) BEFORE re-queuing the preempted
            # process.
            _enqueue_arrivals(current_time)

            if remaining_burst[pid] > 0:
                # Process still has work — re-queue at the back.
                ready.append(pid)
            else:
                # Process finished.
                completion_times[pid] = current_time

        # --- Build per-process metrics ---
        metrics: list[ProcessMetrics] = []
        for proc in sorted_procs:
            ct = completion_times[proc.pid]
            tat = ct - proc.arrival_time
            wt = tat - proc.burst_time
            metrics.append(
                ProcessMetrics(
                    pid=proc.pid,
                    arrival_time=proc.arrival_time,
                    burst_time=proc.burst_time,
                    priority=proc.priority,
                    completion_time=ct,
                    turnaround_time=tat,
                    waiting_time=wt,
                )
            )

        avg_waiting = sum(m.waiting_time for m in metrics) / len(metrics)
        avg_turnaround = sum(m.turnaround_time for m in metrics) / len(metrics)

        return SimulationResult(
            algorithm_name=self.name,
            execution_segments=tuple(segments),
            process_metrics=tuple(metrics),
            average_waiting_time=avg_waiting,
            average_turnaround_time=avg_turnaround,
        )
