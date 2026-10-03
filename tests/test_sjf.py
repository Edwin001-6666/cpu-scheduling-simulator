"""Tests for the non-preemptive SJF scheduling algorithm.

Each test constructs explicit Process inputs, runs SJF, and asserts
exact expected values for execution segments, per-process metrics,
and aggregate averages.
"""

from __future__ import annotations

import unittest

from algorithms.sjf import SJF
from core.models import IDLE_PROCESS_ID, ExecutionSegment, Process, SimulationResult


class TestSJF(unittest.TestCase):
    """Non-preemptive SJF algorithm correctness tests."""

    def setUp(self) -> None:
        self.sjf = SJF()

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
        procs = [Process(pid="P1", arrival_time=0, burst_time=5, priority=0)]
        result = self.sjf.schedule(procs)

        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.algorithm_name, "SJF")
        self.assertEqual(len(result.execution_segments), 1)
        self.assertEqual(result.execution_segments[0], self._seg("P1", 0, 5))

        m = result.process_metrics[0]
        self.assertEqual(m.completion_time, 5)
        self.assertEqual(m.turnaround_time, 5)
        self.assertEqual(m.waiting_time, 0)
        self.assertEqual(result.average_waiting_time, 0.0)
        self.assertEqual(result.average_turnaround_time, 5.0)

    # ------------------------------------------------------------------
    # 2. Multiple processes arriving at time 0
    # ------------------------------------------------------------------
    def test_all_arrive_at_zero(self) -> None:
        """All arrive at 0 — shortest burst goes first."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=6, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=2, priority=0),
            Process(pid="P3", arrival_time=0, burst_time=4, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # Sorted by burst: P2(2), P3(4), P1(6)
        expected_segs = (
            self._seg("P2", 0, 2),
            self._seg("P3", 2, 6),
            self._seg("P1", 6, 12),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 3. Different burst times with staggered arrivals
    # ------------------------------------------------------------------
    def test_different_burst_times(self) -> None:
        """SJF should pick the shortest available burst at each decision."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=7, priority=0),
            Process(pid="P2", arrival_time=2, burst_time=4, priority=0),
            Process(pid="P3", arrival_time=4, burst_time=1, priority=0),
            Process(pid="P4", arrival_time=5, burst_time=4, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # t=0: only P1 ready → P1 runs 0→7
        # t=7: P2(burst 4), P3(burst 1), P4(burst 4) ready → P3 shortest → 7→8
        # t=8: P2(burst 4), P4(burst 4) → tie burst, P2 earlier arrival → 8→12
        # t=12: P4 → 12→16
        expected_segs = (
            self._seg("P1", 0, 7),
            self._seg("P3", 7, 8),
            self._seg("P2", 8, 12),
            self._seg("P4", 12, 16),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 4. Later-arriving process with shorter burst
    # ------------------------------------------------------------------
    def test_later_shorter_burst(self) -> None:
        """A process arriving later with a shorter burst should run next
        (non-preemptive: only considered after current finishes)."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=8, priority=0),
            Process(pid="P2", arrival_time=1, burst_time=1, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # t=0: only P1 → runs 0→8 (non-preemptive, P2 cannot interrupt)
        # t=8: P2 → runs 8→9
        expected_segs = (
            self._seg("P1", 0, 8),
            self._seg("P2", 8, 9),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 5. CPU idle before first process
    # ------------------------------------------------------------------
    def test_idle_before_first_process(self) -> None:
        procs = [Process(pid="P1", arrival_time=3, burst_time=2, priority=0)]
        result = self.sjf.schedule(procs)

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
    # 6. CPU idle between processes
    # ------------------------------------------------------------------
    def test_idle_between_processes(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=2, priority=0),
            Process(pid="P2", arrival_time=5, burst_time=3, priority=0),
        ]
        result = self.sjf.schedule(procs)

        expected_segs = (
            self._seg("P1", 0, 2),
            self._idle(2, 5),
            self._seg("P2", 5, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 7. Equal burst times — tie-break by arrival_time then pid
    # ------------------------------------------------------------------
    def test_equal_burst_times(self) -> None:
        """Same burst: prefer earlier arrival; if still tied, lower PID."""
        procs = [
            Process(pid="P3", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P2", arrival_time=1, burst_time=3, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # All burst 3.  t=0 ready: P3(arr 0), P1(arr 0) → same arrival →
        # PID tie-break → P1 first (P1 < P3).
        # t=3 ready: P3(arr 0), P2(arr 1) → P3 earlier arrival → P3 next.
        # t=6: P2 → 6→9.
        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P3", 3, 6),
            self._seg("P2", 6, 9),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 8. Equal arrival AND burst — pure PID tie-break
    # ------------------------------------------------------------------
    def test_equal_arrival_and_burst(self) -> None:
        procs = [
            Process(pid="P2", arrival_time=0, burst_time=4, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=4, priority=0),
        ]
        result = self.sjf.schedule(procs)

        expected_segs = (
            self._seg("P1", 0, 4),
            self._seg("P2", 4, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 9. Unsorted input order
    # ------------------------------------------------------------------
    def test_unsorted_input(self) -> None:
        """Input order should not affect result."""
        procs = [
            Process(pid="P3", arrival_time=4, burst_time=1, priority=0),
            Process(pid="P1", arrival_time=0, burst_time=7, priority=0),
            Process(pid="P2", arrival_time=2, burst_time=4, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # t=0: P1 only → 0→7
        # t=7: P2(burst 4), P3(burst 1) → P3 → 7→8
        # t=8: P2 → 8→12
        expected_segs = (
            self._seg("P1", 0, 7),
            self._seg("P3", 7, 8),
            self._seg("P2", 8, 12),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 10–14. Metric verification (CT, TAT, WT, averages)
    # ------------------------------------------------------------------
    def test_metrics_textbook_example(self) -> None:
        """Textbook-style example: P1(0,6), P2(0,8), P3(0,7), P4(0,3)."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=6, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=8, priority=0),
            Process(pid="P3", arrival_time=0, burst_time=7, priority=0),
            Process(pid="P4", arrival_time=0, burst_time=3, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # Order: P4(3), P1(6), P3(7), P2(8)
        metrics = {m.pid: m for m in result.process_metrics}

        # P4: CT=3, TAT=3, WT=0
        self.assertEqual(metrics["P4"].completion_time, 3)
        self.assertEqual(metrics["P4"].turnaround_time, 3)
        self.assertEqual(metrics["P4"].waiting_time, 0)

        # P1: CT=9, TAT=9, WT=3
        self.assertEqual(metrics["P1"].completion_time, 9)
        self.assertEqual(metrics["P1"].turnaround_time, 9)
        self.assertEqual(metrics["P1"].waiting_time, 3)

        # P3: CT=16, TAT=16, WT=9
        self.assertEqual(metrics["P3"].completion_time, 16)
        self.assertEqual(metrics["P3"].turnaround_time, 16)
        self.assertEqual(metrics["P3"].waiting_time, 9)

        # P2: CT=24, TAT=24, WT=16
        self.assertEqual(metrics["P2"].completion_time, 24)
        self.assertEqual(metrics["P2"].turnaround_time, 24)
        self.assertEqual(metrics["P2"].waiting_time, 16)

        # Avg WT = (0+3+9+16)/4 = 7.0
        self.assertAlmostEqual(result.average_waiting_time, 7.0)
        # Avg TAT = (3+9+16+24)/4 = 13.0
        self.assertAlmostEqual(result.average_turnaround_time, 13.0)

    def test_metrics_staggered_arrivals(self) -> None:
        """Verify metrics when processes arrive at different times."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P2", arrival_time=2, burst_time=6, priority=0),
            Process(pid="P3", arrival_time=4, burst_time=2, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # t=0: P1 only → 0→3
        # t=3: P2(burst 6) ready → P2 runs 3→9?  Wait — P3 arrives at 4.
        #   At t=3 only P2 is ready → P2 runs 3→9 (non-preemptive).
        # t=9: P3 → 9→11
        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P2", 3, 9),
            self._seg("P3", 9, 11),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        metrics = {m.pid: m for m in result.process_metrics}

        # P1: CT=3, TAT=3-0=3, WT=3-3=0
        self.assertEqual(metrics["P1"].completion_time, 3)
        self.assertEqual(metrics["P1"].turnaround_time, 3)
        self.assertEqual(metrics["P1"].waiting_time, 0)

        # P2: CT=9, TAT=9-2=7, WT=7-6=1
        self.assertEqual(metrics["P2"].completion_time, 9)
        self.assertEqual(metrics["P2"].turnaround_time, 7)
        self.assertEqual(metrics["P2"].waiting_time, 1)

        # P3: CT=11, TAT=11-4=7, WT=7-2=5
        self.assertEqual(metrics["P3"].completion_time, 11)
        self.assertEqual(metrics["P3"].turnaround_time, 7)
        self.assertEqual(metrics["P3"].waiting_time, 5)

        # Avg WT = (0+1+5)/3 = 2.0
        self.assertAlmostEqual(result.average_waiting_time, 2.0)
        # Avg TAT = (3+7+7)/3 ≈ 5.6667
        self.assertAlmostEqual(result.average_turnaround_time, 17 / 3)

    # ------------------------------------------------------------------
    # 15. Verify non-preemptive behavior
    # ------------------------------------------------------------------
    def test_non_preemptive(self) -> None:
        """A shorter job arriving mid-execution must NOT preempt the
        running process.  The running process should have one contiguous
        segment covering its full burst."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=10, priority=0),
            Process(pid="P2", arrival_time=3, burst_time=1, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # P1 must run 0→10 uninterrupted even though P2 (burst 1) arrives at 3.
        expected_segs = (
            self._seg("P1", 0, 10),
            self._seg("P2", 10, 11),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        # P1's segment is contiguous — exactly one segment for P1.
        p1_segs = [s for s in result.execution_segments if s.process_id == "P1"]
        self.assertEqual(len(p1_segs), 1)
        self.assertEqual(p1_segs[0].end_time - p1_segs[0].start_time, 10)

    # ------------------------------------------------------------------
    # Multiple idle gaps
    # ------------------------------------------------------------------
    def test_multiple_idle_gaps(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=2, burst_time=1, priority=0),
            Process(pid="P2", arrival_time=6, burst_time=2, priority=0),
        ]
        result = self.sjf.schedule(procs)

        expected_segs = (
            self._idle(0, 2),
            self._seg("P1", 2, 3),
            self._idle(3, 6),
            self._seg("P2", 6, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # Edge: empty input
    # ------------------------------------------------------------------
    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.sjf.schedule([])

    # ------------------------------------------------------------------
    # Result type and name
    # ------------------------------------------------------------------
    def test_returns_simulation_result(self) -> None:
        procs = [Process(pid="P1", arrival_time=0, burst_time=1, priority=0)]
        result = self.sjf.schedule(procs)
        self.assertIsInstance(result, SimulationResult)

    def test_algorithm_name(self) -> None:
        self.assertEqual(self.sjf.name, "SJF")

    # ------------------------------------------------------------------
    # SJF differs from FCFS
    # ------------------------------------------------------------------
    def test_sjf_differs_from_fcfs_order(self) -> None:
        """Confirm SJF actually picks shortest burst, not arrival order."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=5, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=2, priority=0),
        ]
        result = self.sjf.schedule(procs)

        # SJF: P2 first (shorter burst). FCFS would pick P1 first (same arr, P1 < P2).
        self.assertEqual(result.execution_segments[0].process_id, "P2")
        self.assertEqual(result.execution_segments[1].process_id, "P1")


if __name__ == "__main__":
    unittest.main()
