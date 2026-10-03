"""Tests for the preemptive Round Robin scheduling algorithm.

Each test constructs explicit Process inputs, runs RoundRobin with a
specific time quantum, and asserts exact expected values for execution
segments, per-process metrics, and aggregate averages.
"""

from __future__ import annotations

import unittest

from algorithms.round_robin import RoundRobin
from core.models import IDLE_PROCESS_ID, ExecutionSegment, Process, SimulationResult


class TestRoundRobin(unittest.TestCase):
    """Preemptive Round Robin algorithm correctness tests."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _seg(self, pid: str, start: int, end: int) -> ExecutionSegment:
        return ExecutionSegment(process_id=pid, start_time=start, end_time=end)

    def _idle(self, start: int, end: int) -> ExecutionSegment:
        return ExecutionSegment(process_id=IDLE_PROCESS_ID, start_time=start, end_time=end)

    # ------------------------------------------------------------------
    # 1. Single process
    # ------------------------------------------------------------------
    def test_single_process(self) -> None:
        rr = RoundRobin(time_quantum=3)
        procs = [Process(pid="P1", arrival_time=0, burst_time=5, priority=0)]
        result = rr.schedule(procs)

        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.algorithm_name, "Round Robin")

        # P1 runs 0→3 (quantum), then 3→5 (remaining 2).
        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P1", 3, 5),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        m = result.process_metrics[0]
        self.assertEqual(m.completion_time, 5)
        self.assertEqual(m.turnaround_time, 5)
        self.assertEqual(m.waiting_time, 0)

    # ------------------------------------------------------------------
    # 2. Multiple processes arriving at time 0
    # ------------------------------------------------------------------
    def test_multiple_at_time_zero(self) -> None:
        rr = RoundRobin(time_quantum=2)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=5, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=3, priority=0),
        ]
        result = rr.schedule(procs)

        # Enqueue order by PID: P1, P2
        # P1: 0→2 (rem 3), P2: 2→4 (rem 1), P1: 4→6 (rem 1),
        # P2: 6→7 (done), P1: 7→8 (done)
        expected_segs = (
            self._seg("P1", 0, 2),
            self._seg("P2", 2, 4),
            self._seg("P1", 4, 6),
            self._seg("P2", 6, 7),
            self._seg("P1", 7, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 3. Quantum smaller than burst time
    # ------------------------------------------------------------------
    def test_quantum_smaller_than_burst(self) -> None:
        rr = RoundRobin(time_quantum=1)
        procs = [Process(pid="P1", arrival_time=0, burst_time=3, priority=0)]
        result = rr.schedule(procs)

        expected_segs = (
            self._seg("P1", 0, 1),
            self._seg("P1", 1, 2),
            self._seg("P1", 2, 3),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        m = result.process_metrics[0]
        self.assertEqual(m.completion_time, 3)
        self.assertEqual(m.turnaround_time, 3)
        self.assertEqual(m.waiting_time, 0)

    # ------------------------------------------------------------------
    # 4. Quantum equal to burst time
    # ------------------------------------------------------------------
    def test_quantum_equals_burst(self) -> None:
        rr = RoundRobin(time_quantum=4)
        procs = [Process(pid="P1", arrival_time=0, burst_time=4, priority=0)]
        result = rr.schedule(procs)

        # Finishes in exactly one quantum.
        expected_segs = (self._seg("P1", 0, 4),)
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 5. Quantum larger than burst time
    # ------------------------------------------------------------------
    def test_quantum_larger_than_burst(self) -> None:
        rr = RoundRobin(time_quantum=10)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=5, priority=0),
        ]
        result = rr.schedule(procs)

        # Each finishes in one slice (quantum > burst).
        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P2", 3, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 6. Multiple rounds through the ready queue
    # ------------------------------------------------------------------
    def test_multiple_rounds(self) -> None:
        rr = RoundRobin(time_quantum=2)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=4, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=4, priority=0),
        ]
        result = rr.schedule(procs)

        # Round 1: P1 0→2, P2 2→4
        # Round 2: P1 4→6, P2 6→8
        expected_segs = (
            self._seg("P1", 0, 2),
            self._seg("P2", 2, 4),
            self._seg("P1", 4, 6),
            self._seg("P2", 6, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 7. Process arriving while another is executing
    # ------------------------------------------------------------------
    def test_arrival_during_execution(self) -> None:
        rr = RoundRobin(time_quantum=4)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=5, priority=0),
            Process(pid="P2", arrival_time=3, burst_time=3, priority=0),
        ]
        result = rr.schedule(procs)

        # t=0: P1 runs 0→4 (quantum expires, rem 1).
        #   At t=4 P2 (arrived at 3) is enqueued, then P1 re-queued.
        #   Queue: [P2, P1]
        # t=4: P2 runs 4→7 (burst 3, done).
        # t=7: P1 runs 7→8 (rem 1, done).
        expected_segs = (
            self._seg("P1", 0, 4),
            self._seg("P2", 4, 7),
            self._seg("P1", 7, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 8. Process arriving exactly at quantum boundary
    # ------------------------------------------------------------------
    def test_arrival_at_quantum_boundary(self) -> None:
        rr = RoundRobin(time_quantum=3)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=6, priority=0),
            Process(pid="P2", arrival_time=3, burst_time=2, priority=0),
        ]
        result = rr.schedule(procs)

        # t=0: P1 runs 0→3 (rem 3).
        #   At t=3 P2 arrives → enqueued first, then P1 re-queued.
        #   Queue: [P2, P1]
        # t=3: P2 runs 3→5 (done).
        # t=5: P1 runs 5→8 (rem 3, done).
        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P2", 3, 5),
            self._seg("P1", 5, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 9. CPU idle before first process
    # ------------------------------------------------------------------
    def test_idle_before_first_process(self) -> None:
        rr = RoundRobin(time_quantum=2)
        procs = [Process(pid="P1", arrival_time=3, burst_time=4, priority=0)]
        result = rr.schedule(procs)

        expected_segs = (
            self._idle(0, 3),
            self._seg("P1", 3, 5),
            self._seg("P1", 5, 7),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        m = result.process_metrics[0]
        self.assertEqual(m.completion_time, 7)
        self.assertEqual(m.turnaround_time, 4)  # 7 - 3
        self.assertEqual(m.waiting_time, 0)     # 4 - 4

    # ------------------------------------------------------------------
    # 10. CPU idle between workloads
    # ------------------------------------------------------------------
    def test_idle_between_workloads(self) -> None:
        rr = RoundRobin(time_quantum=3)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=2, priority=0),
            Process(pid="P2", arrival_time=5, burst_time=3, priority=0),
        ]
        result = rr.schedule(procs)

        expected_segs = (
            self._seg("P1", 0, 2),
            self._idle(2, 5),
            self._seg("P2", 5, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 11. Unsorted input
    # ------------------------------------------------------------------
    def test_unsorted_input(self) -> None:
        rr = RoundRobin(time_quantum=2)
        procs = [
            Process(pid="P2", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=4, priority=0),
        ]
        result = rr.schedule(procs)

        # Deterministic: sorted by (arrival, pid) → P1, P2.
        # P1: 0→2 (rem 2), P2: 2→4 (rem 1), P1: 4→6 (done),
        # P2: 6→7 (done).
        expected_segs = (
            self._seg("P1", 0, 2),
            self._seg("P2", 2, 4),
            self._seg("P1", 4, 6),
            self._seg("P2", 6, 7),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 12–16. Metric verification
    # ------------------------------------------------------------------
    def test_metrics_classic(self) -> None:
        """Classic example: P1(0,10), P2(0,4), P3(0,6) with quantum=3."""
        rr = RoundRobin(time_quantum=3)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=10, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=4, priority=0),
            Process(pid="P3", arrival_time=0, burst_time=6, priority=0),
        ]
        result = rr.schedule(procs)

        # P1: 0→3 (rem 7), P2: 3→6 (rem 1), P3: 6→9 (rem 3),
        # P1: 9→12 (rem 4), P2: 12→13 (done), P3: 13→16 (done),
        # P1: 16→19 (rem 1), P1: 19→20 (done).
        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P2", 3, 6),
            self._seg("P3", 6, 9),
            self._seg("P1", 9, 12),
            self._seg("P2", 12, 13),
            self._seg("P3", 13, 16),
            self._seg("P1", 16, 19),
            self._seg("P1", 19, 20),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        metrics = {m.pid: m for m in result.process_metrics}

        # P1: CT=20, TAT=20, WT=10
        self.assertEqual(metrics["P1"].completion_time, 20)
        self.assertEqual(metrics["P1"].turnaround_time, 20)
        self.assertEqual(metrics["P1"].waiting_time, 10)

        # P2: CT=13, TAT=13, WT=9
        self.assertEqual(metrics["P2"].completion_time, 13)
        self.assertEqual(metrics["P2"].turnaround_time, 13)
        self.assertEqual(metrics["P2"].waiting_time, 9)

        # P3: CT=16, TAT=16, WT=10
        self.assertEqual(metrics["P3"].completion_time, 16)
        self.assertEqual(metrics["P3"].turnaround_time, 16)
        self.assertEqual(metrics["P3"].waiting_time, 10)

        # Avg WT = (10+9+10)/3 ≈ 9.6667
        self.assertAlmostEqual(result.average_waiting_time, 29 / 3)
        # Avg TAT = (20+13+16)/3 ≈ 16.3333
        self.assertAlmostEqual(result.average_turnaround_time, 49 / 3)

    def test_metrics_staggered_arrivals(self) -> None:
        """Processes arrive at different times with quantum=2."""
        rr = RoundRobin(time_quantum=2)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=4, priority=0),
            Process(pid="P2", arrival_time=1, burst_time=3, priority=0),
            Process(pid="P3", arrival_time=3, burst_time=2, priority=0),
        ]
        result = rr.schedule(procs)

        # t=0: queue=[P1]. P1 runs 0→2 (rem 2).
        #   t=2: P2 (arr 1) enqueued, then P1 re-queued. queue=[P2, P1].
        # t=2: P2 runs 2→4 (rem 1).
        #   t=4: P3 (arr 3) enqueued, then P2 re-queued. queue=[P1, P3, P2].
        # t=4: P1 runs 4→6 (done).
        #   queue=[P3, P2].
        # t=6: P3 runs 6→8 (done).
        #   queue=[P2].
        # t=8: P2 runs 8→9 (done).
        expected_segs = (
            self._seg("P1", 0, 2),
            self._seg("P2", 2, 4),
            self._seg("P1", 4, 6),
            self._seg("P3", 6, 8),
            self._seg("P2", 8, 9),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        metrics = {m.pid: m for m in result.process_metrics}

        # P1: CT=6, TAT=6-0=6, WT=6-4=2
        self.assertEqual(metrics["P1"].completion_time, 6)
        self.assertEqual(metrics["P1"].turnaround_time, 6)
        self.assertEqual(metrics["P1"].waiting_time, 2)

        # P2: CT=9, TAT=9-1=8, WT=8-3=5
        self.assertEqual(metrics["P2"].completion_time, 9)
        self.assertEqual(metrics["P2"].turnaround_time, 8)
        self.assertEqual(metrics["P2"].waiting_time, 5)

        # P3: CT=8, TAT=8-3=5, WT=5-2=3
        self.assertEqual(metrics["P3"].completion_time, 8)
        self.assertEqual(metrics["P3"].turnaround_time, 5)
        self.assertEqual(metrics["P3"].waiting_time, 3)

        # Avg WT = (2+5+3)/3 ≈ 3.3333
        self.assertAlmostEqual(result.average_waiting_time, 10 / 3)
        # Avg TAT = (6+8+5)/3 ≈ 6.3333
        self.assertAlmostEqual(result.average_turnaround_time, 19 / 3)

    # ------------------------------------------------------------------
    # 17. Gantt segment correctness — spec example
    # ------------------------------------------------------------------
    def test_gantt_spec_example(self) -> None:
        """The example from the task spec: P1(0,5), P2(0,3), q=2."""
        rr = RoundRobin(time_quantum=2)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=5, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=3, priority=0),
        ]
        result = rr.schedule(procs)

        expected_segs = (
            self._seg("P1", 0, 2),
            self._seg("P2", 2, 4),
            self._seg("P1", 4, 6),
            self._seg("P2", 6, 7),
            self._seg("P1", 7, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 18. Invalid quantum = 0
    # ------------------------------------------------------------------
    def test_invalid_quantum_zero(self) -> None:
        with self.assertRaises(ValueError):
            RoundRobin(time_quantum=0)

    # ------------------------------------------------------------------
    # 19. Invalid negative quantum
    # ------------------------------------------------------------------
    def test_invalid_negative_quantum(self) -> None:
        with self.assertRaises(ValueError):
            RoundRobin(time_quantum=-1)

    def test_invalid_quantum_type(self) -> None:
        with self.assertRaises(ValueError):
            RoundRobin(time_quantum=2.5)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 20. Deterministic behavior
    # ------------------------------------------------------------------
    def test_deterministic_repeated_runs(self) -> None:
        """Running the same input twice must produce identical results."""
        procs = [
            Process(pid="P2", arrival_time=0, burst_time=4, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P3", arrival_time=2, burst_time=2, priority=0),
        ]
        rr = RoundRobin(time_quantum=2)
        r1 = rr.schedule(procs)
        r2 = rr.schedule(procs)
        self.assertEqual(r1.execution_segments, r2.execution_segments)
        self.assertEqual(r1.process_metrics, r2.process_metrics)
        self.assertEqual(r1.average_waiting_time, r2.average_waiting_time)
        self.assertEqual(r1.average_turnaround_time, r2.average_turnaround_time)

    def test_deterministic_pid_ordering(self) -> None:
        """Processes with same arrival time: PID order is deterministic."""
        procs = [
            Process(pid="P3", arrival_time=0, burst_time=2, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=2, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=2, priority=0),
        ]
        rr = RoundRobin(time_quantum=1)
        result = rr.schedule(procs)

        # Enqueue order: P1, P2, P3 (sorted by PID).
        # q=1: P1 0→1, P2 1→2, P3 2→3, P1 3→4, P2 4→5, P3 5→6.
        expected_segs = (
            self._seg("P1", 0, 1),
            self._seg("P2", 1, 2),
            self._seg("P3", 2, 3),
            self._seg("P1", 3, 4),
            self._seg("P2", 4, 5),
            self._seg("P3", 5, 6),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # Edge: empty input
    # ------------------------------------------------------------------
    def test_empty_input_raises(self) -> None:
        rr = RoundRobin(time_quantum=2)
        with self.assertRaises(ValueError):
            rr.schedule([])

    # ------------------------------------------------------------------
    # Result type and name
    # ------------------------------------------------------------------
    def test_returns_simulation_result(self) -> None:
        rr = RoundRobin(time_quantum=2)
        procs = [Process(pid="P1", arrival_time=0, burst_time=1, priority=0)]
        result = rr.schedule(procs)
        self.assertIsInstance(result, SimulationResult)

    def test_algorithm_name(self) -> None:
        rr = RoundRobin(time_quantum=2)
        self.assertEqual(rr.name, "Round Robin")

    def test_time_quantum_property(self) -> None:
        rr = RoundRobin(time_quantum=5)
        self.assertEqual(rr.time_quantum, 5)


if __name__ == "__main__":
    unittest.main()
