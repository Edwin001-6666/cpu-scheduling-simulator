"""Tests for the FCFS scheduling algorithm.

Each test constructs explicit Process inputs, runs FCFS, and asserts
exact expected values for execution segments, per-process metrics,
and aggregate averages.
"""

from __future__ import annotations

import unittest

from algorithms.fcfs import FCFS
from core.models import IDLE_PROCESS_ID, ExecutionSegment, Process, SimulationResult


class TestFCFS(unittest.TestCase):
    """FCFS algorithm correctness tests."""

    def setUp(self) -> None:
        self.fcfs = FCFS()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _seg(self, pid: str, start: int, end: int) -> ExecutionSegment:
        """Shorthand for an ExecutionSegment."""
        return ExecutionSegment(process_id=pid, start_time=start, end_time=end)

    def _idle(self, start: int, end: int) -> ExecutionSegment:
        return ExecutionSegment(process_id=IDLE_PROCESS_ID, start_time=start, end_time=end)

    # ------------------------------------------------------------------
    # 1. Single process
    # ------------------------------------------------------------------
    def test_single_process(self) -> None:
        procs = [Process(pid="P1", arrival_time=0, burst_time=5, priority=0)]
        result = self.fcfs.schedule(procs)

        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.algorithm_name, "FCFS")
        self.assertEqual(len(result.execution_segments), 1)
        self.assertEqual(result.execution_segments[0], self._seg("P1", 0, 5))

        m = result.process_metrics[0]
        self.assertEqual(m.pid, "P1")
        self.assertEqual(m.completion_time, 5)
        self.assertEqual(m.turnaround_time, 5)   # 5 - 0
        self.assertEqual(m.waiting_time, 0)       # 5 - 5
        self.assertEqual(result.average_waiting_time, 0.0)
        self.assertEqual(result.average_turnaround_time, 5.0)

    # ------------------------------------------------------------------
    # 2. Multiple processes arriving at time 0
    # ------------------------------------------------------------------
    def test_all_arrive_at_zero(self) -> None:
        """Three processes all arrive at 0; tie-break by PID."""
        procs = [
            Process(pid="P3", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=4, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=2, priority=0),
        ]
        result = self.fcfs.schedule(procs)

        # Sorted order: P1 (burst 4), P2 (burst 2), P3 (burst 3)
        expected_segs = (
            self._seg("P1", 0, 4),
            self._seg("P2", 4, 6),
            self._seg("P3", 6, 9),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        # P1: CT=4, TAT=4, WT=0
        # P2: CT=6, TAT=6, WT=4
        # P3: CT=9, TAT=9, WT=6
        metrics = {m.pid: m for m in result.process_metrics}
        self.assertEqual(metrics["P1"].completion_time, 4)
        self.assertEqual(metrics["P1"].turnaround_time, 4)
        self.assertEqual(metrics["P1"].waiting_time, 0)

        self.assertEqual(metrics["P2"].completion_time, 6)
        self.assertEqual(metrics["P2"].turnaround_time, 6)
        self.assertEqual(metrics["P2"].waiting_time, 4)

        self.assertEqual(metrics["P3"].completion_time, 9)
        self.assertEqual(metrics["P3"].turnaround_time, 9)
        self.assertEqual(metrics["P3"].waiting_time, 6)

        self.assertAlmostEqual(result.average_waiting_time, 10 / 3)
        self.assertAlmostEqual(result.average_turnaround_time, 19 / 3)

    # ------------------------------------------------------------------
    # 3. Processes with different arrival times
    # ------------------------------------------------------------------
    def test_different_arrival_times(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P2", arrival_time=1, burst_time=4, priority=0),
            Process(pid="P3", arrival_time=4, burst_time=2, priority=0),
        ]
        result = self.fcfs.schedule(procs)

        # P1: 0→3, P2: 3→7, P3: 7→9 (no idle gaps)
        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P2", 3, 7),
            self._seg("P3", 7, 9),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        metrics = {m.pid: m for m in result.process_metrics}
        # P1: CT=3, TAT=3-0=3, WT=3-3=0
        self.assertEqual(metrics["P1"].completion_time, 3)
        self.assertEqual(metrics["P1"].turnaround_time, 3)
        self.assertEqual(metrics["P1"].waiting_time, 0)
        # P2: CT=7, TAT=7-1=6, WT=6-4=2
        self.assertEqual(metrics["P2"].completion_time, 7)
        self.assertEqual(metrics["P2"].turnaround_time, 6)
        self.assertEqual(metrics["P2"].waiting_time, 2)
        # P3: CT=9, TAT=9-4=5, WT=5-2=3
        self.assertEqual(metrics["P3"].completion_time, 9)
        self.assertEqual(metrics["P3"].turnaround_time, 5)
        self.assertEqual(metrics["P3"].waiting_time, 3)

        self.assertAlmostEqual(result.average_waiting_time, 5 / 3)
        self.assertAlmostEqual(result.average_turnaround_time, 14 / 3)

    # ------------------------------------------------------------------
    # 4. CPU idle before first process
    # ------------------------------------------------------------------
    def test_idle_before_first_process(self) -> None:
        """First process arrives at t=3; CPU should be idle 0→3."""
        procs = [Process(pid="P1", arrival_time=3, burst_time=2, priority=0)]
        result = self.fcfs.schedule(procs)

        expected_segs = (
            self._idle(0, 3),
            self._seg("P1", 3, 5),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        m = result.process_metrics[0]
        self.assertEqual(m.completion_time, 5)
        self.assertEqual(m.turnaround_time, 2)  # 5 - 3
        self.assertEqual(m.waiting_time, 0)     # 2 - 2

    # ------------------------------------------------------------------
    # 5. CPU idle between processes
    # ------------------------------------------------------------------
    def test_idle_between_processes(self) -> None:
        """Gap between P1 finishing and P2 arriving."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P2", arrival_time=5, burst_time=2, priority=0),
        ]
        result = self.fcfs.schedule(procs)

        expected_segs = (
            self._seg("P1", 0, 3),
            self._idle(3, 5),
            self._seg("P2", 5, 7),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        metrics = {m.pid: m for m in result.process_metrics}
        # P1: CT=3, TAT=3, WT=0
        self.assertEqual(metrics["P1"].completion_time, 3)
        self.assertEqual(metrics["P1"].turnaround_time, 3)
        self.assertEqual(metrics["P1"].waiting_time, 0)
        # P2: CT=7, TAT=7-5=2, WT=2-2=0
        self.assertEqual(metrics["P2"].completion_time, 7)
        self.assertEqual(metrics["P2"].turnaround_time, 2)
        self.assertEqual(metrics["P2"].waiting_time, 0)

    # ------------------------------------------------------------------
    # 6. Equal arrival times — deterministic PID tie-break
    # ------------------------------------------------------------------
    def test_equal_arrival_times_deterministic(self) -> None:
        """Same arrival time; order must be determined by PID."""
        procs = [
            Process(pid="P2", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=5, priority=0),
        ]
        result = self.fcfs.schedule(procs)

        # P1 before P2 because "P1" < "P2" lexicographically
        expected_segs = (
            self._seg("P1", 0, 5),
            self._seg("P2", 5, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 7. Unsorted input order
    # ------------------------------------------------------------------
    def test_unsorted_input(self) -> None:
        """Processes given in random order; result must match arrival sort."""
        procs = [
            Process(pid="P3", arrival_time=4, burst_time=1, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P2", arrival_time=2, burst_time=2, priority=0),
        ]
        result = self.fcfs.schedule(procs)

        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P2", 3, 5),
            self._seg("P3", 5, 6),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 8–11. Metric spot-checks (completion, TAT, WT, averages)
    # ------------------------------------------------------------------
    def test_metrics_classic_example(self) -> None:
        """Classic textbook example: P1(0,6), P2(1,8), P3(2,7), P4(3,3)."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=6, priority=0),
            Process(pid="P2", arrival_time=1, burst_time=8, priority=0),
            Process(pid="P3", arrival_time=2, burst_time=7, priority=0),
            Process(pid="P4", arrival_time=3, burst_time=3, priority=0),
        ]
        result = self.fcfs.schedule(procs)

        metrics = {m.pid: m for m in result.process_metrics}

        # P1: CT=6,  TAT=6-0=6,   WT=6-6=0
        self.assertEqual(metrics["P1"].completion_time, 6)
        self.assertEqual(metrics["P1"].turnaround_time, 6)
        self.assertEqual(metrics["P1"].waiting_time, 0)

        # P2: CT=14, TAT=14-1=13, WT=13-8=5
        self.assertEqual(metrics["P2"].completion_time, 14)
        self.assertEqual(metrics["P2"].turnaround_time, 13)
        self.assertEqual(metrics["P2"].waiting_time, 5)

        # P3: CT=21, TAT=21-2=19, WT=19-7=12
        self.assertEqual(metrics["P3"].completion_time, 21)
        self.assertEqual(metrics["P3"].turnaround_time, 19)
        self.assertEqual(metrics["P3"].waiting_time, 12)

        # P4: CT=24, TAT=24-3=21, WT=21-3=18
        self.assertEqual(metrics["P4"].completion_time, 24)
        self.assertEqual(metrics["P4"].turnaround_time, 21)
        self.assertEqual(metrics["P4"].waiting_time, 18)

        # Averages: WT=(0+5+12+18)/4=8.75, TAT=(6+13+19+21)/4=14.75
        self.assertAlmostEqual(result.average_waiting_time, 8.75)
        self.assertAlmostEqual(result.average_turnaround_time, 14.75)

    # ------------------------------------------------------------------
    # 12. Execution segments completeness (idle + busy)
    # ------------------------------------------------------------------
    def test_segments_with_multiple_idle_gaps(self) -> None:
        """Multiple idle gaps: before first and between processes."""
        procs = [
            Process(pid="P1", arrival_time=2, burst_time=1, priority=0),
            Process(pid="P2", arrival_time=5, burst_time=2, priority=0),
            Process(pid="P3", arrival_time=10, burst_time=3, priority=0),
        ]
        result = self.fcfs.schedule(procs)

        expected_segs = (
            self._idle(0, 2),       # idle before P1
            self._seg("P1", 2, 3),
            self._idle(3, 5),       # idle between P1 and P2
            self._seg("P2", 5, 7),
            self._idle(7, 10),      # idle between P2 and P3
            self._seg("P3", 10, 13),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # Edge: empty input raises ValueError
    # ------------------------------------------------------------------
    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.fcfs.schedule([])

    # ------------------------------------------------------------------
    # Result type and name
    # ------------------------------------------------------------------
    def test_returns_simulation_result(self) -> None:
        procs = [Process(pid="P1", arrival_time=0, burst_time=1, priority=0)]
        result = self.fcfs.schedule(procs)
        self.assertIsInstance(result, SimulationResult)

    def test_algorithm_name(self) -> None:
        self.assertEqual(self.fcfs.name, "FCFS")

    # ------------------------------------------------------------------
    # Response time (for FCFS: same as waiting time)
    # ------------------------------------------------------------------
    def test_response_time_equals_waiting_time(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P2", arrival_time=1, burst_time=4, priority=0),
        ]
        result = self.fcfs.schedule(procs)
        for m in result.process_metrics:
            self.assertEqual(m.response_time, m.waiting_time)


if __name__ == "__main__":
    unittest.main()
