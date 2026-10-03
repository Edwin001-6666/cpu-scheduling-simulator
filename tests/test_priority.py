"""Tests for the non-preemptive Priority scheduling algorithm.

Each test constructs explicit Process inputs, runs PriorityScheduler,
and asserts exact expected values for execution segments, per-process
metrics, and aggregate averages.

Convention reminder: LOWER numeric priority = HIGHER scheduling priority.
"""

from __future__ import annotations

import unittest

from algorithms.priority import PriorityScheduler
from core.models import IDLE_PROCESS_ID, ExecutionSegment, Process, SimulationResult


class TestPriority(unittest.TestCase):
    """Non-preemptive Priority algorithm correctness tests."""

    def setUp(self) -> None:
        self.scheduler = PriorityScheduler()

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
        procs = [Process(pid="P1", arrival_time=0, burst_time=5, priority=2)]
        result = self.scheduler.schedule(procs)

        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.algorithm_name, "Priority")
        self.assertEqual(result.execution_segments, (self._seg("P1", 0, 5),))

        m = result.process_metrics[0]
        self.assertEqual(m.completion_time, 5)
        self.assertEqual(m.turnaround_time, 5)
        self.assertEqual(m.waiting_time, 0)
        self.assertEqual(result.average_waiting_time, 0.0)
        self.assertEqual(result.average_turnaround_time, 5.0)

    # ------------------------------------------------------------------
    # 2. Multiple processes with different priorities
    # ------------------------------------------------------------------
    def test_different_priorities(self) -> None:
        """Processes with distinct priorities — highest (lowest number) runs first."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=4, priority=3),
            Process(pid="P2", arrival_time=0, burst_time=3, priority=1),
            Process(pid="P3", arrival_time=0, burst_time=5, priority=2),
        ]
        result = self.scheduler.schedule(procs)

        # Order: P2(pri 1), P3(pri 2), P1(pri 3)
        expected_segs = (
            self._seg("P2", 0, 3),
            self._seg("P3", 3, 8),
            self._seg("P1", 8, 12),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 3. Lower numeric priority wins
    # ------------------------------------------------------------------
    def test_lower_numeric_priority_wins(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=3, priority=5),
            Process(pid="P2", arrival_time=0, burst_time=3, priority=1),
        ]
        result = self.scheduler.schedule(procs)

        # P2 (priority 1) should run before P1 (priority 5).
        self.assertEqual(result.execution_segments[0].process_id, "P2")
        self.assertEqual(result.execution_segments[1].process_id, "P1")

    # ------------------------------------------------------------------
    # 4. Higher numeric priority number loses
    # ------------------------------------------------------------------
    def test_higher_numeric_priority_loses(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=2, priority=10),
            Process(pid="P2", arrival_time=0, burst_time=2, priority=1),
            Process(pid="P3", arrival_time=0, burst_time=2, priority=5),
        ]
        result = self.scheduler.schedule(procs)

        # Order: P2(1), P3(5), P1(10)
        pids = [s.process_id for s in result.execution_segments]
        self.assertEqual(pids, ["P2", "P3", "P1"])

    # ------------------------------------------------------------------
    # 5. Priority decision among currently ready processes
    # ------------------------------------------------------------------
    def test_priority_among_ready(self) -> None:
        """Only arrived processes compete; a not-yet-arrived high-priority
        process does not affect earlier decisions."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=4, priority=3),
            Process(pid="P2", arrival_time=0, burst_time=3, priority=2),
            Process(pid="P3", arrival_time=5, burst_time=2, priority=1),  # arrives late
        ]
        result = self.scheduler.schedule(procs)

        # t=0: P1(pri 3), P2(pri 2) ready → P2 wins → 0→3
        # t=3: P1(pri 3) ready, P3 not yet → P1 → 3→7
        # t=7: P3(pri 1) ready → P3 → 7→9
        expected_segs = (
            self._seg("P2", 0, 3),
            self._seg("P1", 3, 7),
            self._seg("P3", 7, 9),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 6. High-priority process arriving later does NOT preempt
    # ------------------------------------------------------------------
    def test_no_preemption(self) -> None:
        """A priority-1 process arriving mid-execution must NOT interrupt
        the running process."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=10, priority=5),
            Process(pid="P2", arrival_time=3, burst_time=2, priority=1),
        ]
        result = self.scheduler.schedule(procs)

        # P1 runs 0→10 uninterrupted, then P2 runs 10→12.
        expected_segs = (
            self._seg("P1", 0, 10),
            self._seg("P2", 10, 12),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        # P1 must have exactly one contiguous segment.
        p1_segs = [s for s in result.execution_segments if s.process_id == "P1"]
        self.assertEqual(len(p1_segs), 1)
        self.assertEqual(p1_segs[0].end_time - p1_segs[0].start_time, 10)

    # ------------------------------------------------------------------
    # 7. CPU idle before first process
    # ------------------------------------------------------------------
    def test_idle_before_first_process(self) -> None:
        procs = [Process(pid="P1", arrival_time=4, burst_time=3, priority=1)]
        result = self.scheduler.schedule(procs)

        expected_segs = (
            self._idle(0, 4),
            self._seg("P1", 4, 7),
        )
        self.assertEqual(result.execution_segments, expected_segs)

        m = result.process_metrics[0]
        self.assertEqual(m.completion_time, 7)
        self.assertEqual(m.turnaround_time, 3)  # 7 - 4
        self.assertEqual(m.waiting_time, 0)     # 3 - 3

    # ------------------------------------------------------------------
    # 8. CPU idle between processes
    # ------------------------------------------------------------------
    def test_idle_between_processes(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=2, priority=1),
            Process(pid="P2", arrival_time=5, burst_time=3, priority=2),
        ]
        result = self.scheduler.schedule(procs)

        expected_segs = (
            self._seg("P1", 0, 2),
            self._idle(2, 5),
            self._seg("P2", 5, 8),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 9. Equal priorities — tie-break by arrival_time
    # ------------------------------------------------------------------
    def test_equal_priorities(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=2, burst_time=3, priority=1),
            Process(pid="P2", arrival_time=0, burst_time=4, priority=1),
        ]
        result = self.scheduler.schedule(procs)

        # Same priority → earlier arrival wins → P2 first.
        expected_segs = (
            self._seg("P2", 0, 4),
            self._seg("P1", 4, 7),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 10. Equal priority AND equal arrival — PID tie-break
    # ------------------------------------------------------------------
    def test_equal_priority_and_arrival(self) -> None:
        procs = [
            Process(pid="P3", arrival_time=0, burst_time=2, priority=1),
            Process(pid="P1", arrival_time=0, burst_time=4, priority=1),
            Process(pid="P2", arrival_time=0, burst_time=3, priority=1),
        ]
        result = self.scheduler.schedule(procs)

        # Same priority, same arrival → PID lexicographic: P1, P2, P3.
        expected_segs = (
            self._seg("P1", 0, 4),
            self._seg("P2", 4, 7),
            self._seg("P3", 7, 9),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 11. Unsorted input
    # ------------------------------------------------------------------
    def test_unsorted_input(self) -> None:
        procs = [
            Process(pid="P3", arrival_time=2, burst_time=1, priority=1),
            Process(pid="P1", arrival_time=0, burst_time=3, priority=3),
            Process(pid="P2", arrival_time=1, burst_time=2, priority=2),
        ]
        result = self.scheduler.schedule(procs)

        # t=0: only P1(pri 3) → 0→3
        # t=3: P2(pri 2), P3(pri 1) → P3 wins → 3→4
        # t=4: P2 → 4→6
        expected_segs = (
            self._seg("P1", 0, 3),
            self._seg("P3", 3, 4),
            self._seg("P2", 4, 6),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # 12–16. Metric verification (CT, TAT, WT, averages)
    # ------------------------------------------------------------------
    def test_metrics_textbook_example(self) -> None:
        """P1(0,10,3), P2(0,1,1), P3(0,2,4), P4(0,1,5), P5(0,5,2)."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=10, priority=3),
            Process(pid="P2", arrival_time=0, burst_time=1, priority=1),
            Process(pid="P3", arrival_time=0, burst_time=2, priority=4),
            Process(pid="P4", arrival_time=0, burst_time=1, priority=5),
            Process(pid="P5", arrival_time=0, burst_time=5, priority=2),
        ]
        result = self.scheduler.schedule(procs)

        # Order: P2(1), P5(2), P1(3), P3(4), P4(5)
        metrics = {m.pid: m for m in result.process_metrics}

        # P2: CT=1, TAT=1, WT=0
        self.assertEqual(metrics["P2"].completion_time, 1)
        self.assertEqual(metrics["P2"].turnaround_time, 1)
        self.assertEqual(metrics["P2"].waiting_time, 0)

        # P5: CT=6, TAT=6, WT=1
        self.assertEqual(metrics["P5"].completion_time, 6)
        self.assertEqual(metrics["P5"].turnaround_time, 6)
        self.assertEqual(metrics["P5"].waiting_time, 1)

        # P1: CT=16, TAT=16, WT=6
        self.assertEqual(metrics["P1"].completion_time, 16)
        self.assertEqual(metrics["P1"].turnaround_time, 16)
        self.assertEqual(metrics["P1"].waiting_time, 6)

        # P3: CT=18, TAT=18, WT=16
        self.assertEqual(metrics["P3"].completion_time, 18)
        self.assertEqual(metrics["P3"].turnaround_time, 18)
        self.assertEqual(metrics["P3"].waiting_time, 16)

        # P4: CT=19, TAT=19, WT=18
        self.assertEqual(metrics["P4"].completion_time, 19)
        self.assertEqual(metrics["P4"].turnaround_time, 19)
        self.assertEqual(metrics["P4"].waiting_time, 18)

        # Avg WT = (0+1+6+16+18)/5 = 41/5 = 8.2
        self.assertAlmostEqual(result.average_waiting_time, 8.2)
        # Avg TAT = (1+6+16+18+19)/5 = 60/5 = 12.0
        self.assertAlmostEqual(result.average_turnaround_time, 12.0)

    def test_metrics_staggered_arrivals(self) -> None:
        """Verify metrics when processes arrive at different times."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=3, priority=2),
            Process(pid="P2", arrival_time=2, burst_time=5, priority=1),
            Process(pid="P3", arrival_time=4, burst_time=2, priority=3),
        ]
        result = self.scheduler.schedule(procs)

        # t=0: P1(pri 2) only → 0→3
        # t=3: P2(pri 1), P3 not arrived → P2 → 3→8
        # t=8: P3(pri 3) → 8→10
        metrics = {m.pid: m for m in result.process_metrics}

        # P1: CT=3, TAT=3-0=3, WT=3-3=0
        self.assertEqual(metrics["P1"].completion_time, 3)
        self.assertEqual(metrics["P1"].turnaround_time, 3)
        self.assertEqual(metrics["P1"].waiting_time, 0)

        # P2: CT=8, TAT=8-2=6, WT=6-5=1
        self.assertEqual(metrics["P2"].completion_time, 8)
        self.assertEqual(metrics["P2"].turnaround_time, 6)
        self.assertEqual(metrics["P2"].waiting_time, 1)

        # P3: CT=10, TAT=10-4=6, WT=6-2=4
        self.assertEqual(metrics["P3"].completion_time, 10)
        self.assertEqual(metrics["P3"].turnaround_time, 6)
        self.assertEqual(metrics["P3"].waiting_time, 4)

        # Avg WT = (0+1+4)/3 ≈ 1.6667
        self.assertAlmostEqual(result.average_waiting_time, 5 / 3)
        # Avg TAT = (3+6+6)/3 = 5.0
        self.assertAlmostEqual(result.average_turnaround_time, 5.0)

    # ------------------------------------------------------------------
    # 17. Deterministic tie-breaking — full chain
    # ------------------------------------------------------------------
    def test_deterministic_tiebreak_full_chain(self) -> None:
        """Three processes: same priority, two share arrival → full
        priority → arrival_time → pid chain tested."""
        procs = [
            Process(pid="P3", arrival_time=0, burst_time=2, priority=2),
            Process(pid="P2", arrival_time=0, burst_time=3, priority=2),
            Process(pid="P1", arrival_time=1, burst_time=1, priority=2),
        ]
        result = self.scheduler.schedule(procs)

        # All pri 2.  t=0 ready: P3(arr 0), P2(arr 0).
        # Same pri, same arrival → PID: P2 < P3 → P2 first → 0→3.
        # t=3 ready: P3(arr 0), P1(arr 1).
        # Same pri → arrival: P3(0) < P1(1) → P3 → 3→5.
        # t=5: P1 → 5→6.
        expected_segs = (
            self._seg("P2", 0, 3),
            self._seg("P3", 3, 5),
            self._seg("P1", 5, 6),
        )
        self.assertEqual(result.execution_segments, expected_segs)

    # ------------------------------------------------------------------
    # Multiple idle gaps
    # ------------------------------------------------------------------
    def test_multiple_idle_gaps(self) -> None:
        procs = [
            Process(pid="P1", arrival_time=2, burst_time=1, priority=1),
            Process(pid="P2", arrival_time=6, burst_time=2, priority=2),
        ]
        result = self.scheduler.schedule(procs)

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
            self.scheduler.schedule([])

    # ------------------------------------------------------------------
    # Result type and name
    # ------------------------------------------------------------------
    def test_returns_simulation_result(self) -> None:
        procs = [Process(pid="P1", arrival_time=0, burst_time=1, priority=1)]
        result = self.scheduler.schedule(procs)
        self.assertIsInstance(result, SimulationResult)

    def test_algorithm_name(self) -> None:
        self.assertEqual(self.scheduler.name, "Priority")

    # ------------------------------------------------------------------
    # Priority differs from SJF / FCFS
    # ------------------------------------------------------------------
    def test_priority_differs_from_sjf(self) -> None:
        """Confirm Priority selects by priority value, not burst time."""
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=1, priority=3),  # short burst, low priority
            Process(pid="P2", arrival_time=0, burst_time=5, priority=1),  # long burst, high priority
        ]
        result = self.scheduler.schedule(procs)

        # Priority: P2 first (pri 1). SJF would pick P1 (burst 1).
        self.assertEqual(result.execution_segments[0].process_id, "P2")
        self.assertEqual(result.execution_segments[1].process_id, "P1")


if __name__ == "__main__":
    unittest.main()
