"""Cross-algorithm integration and hardening tests.

These tests verify properties that must hold for EVERY scheduling
algorithm, regardless of its specific policy.  They exercise the shared
contract (core/models.py, algorithms/base.py) and structural invariants
that individual algorithm test suites may not cover.

Each property test is run against all four algorithms using the same
input processes.
"""

from __future__ import annotations

import unittest
from typing import Sequence

from algorithms.base import SchedulingAlgorithm
from algorithms.fcfs import FCFS
from algorithms.sjf import SJF
from algorithms.priority import PriorityScheduler
from algorithms.round_robin import RoundRobin
from core.models import (
    IDLE_PROCESS_ID,
    ExecutionSegment,
    Process,
    ProcessMetrics,
    SimulationResult,
)


# ======================================================================
# Shared test inputs
# ======================================================================

# Basic workload: different arrivals, no idle gaps expected for most algos.
BASIC_PROCS = (
    Process(pid="P1", arrival_time=0, burst_time=6, priority=3),
    Process(pid="P2", arrival_time=2, burst_time=4, priority=1),
    Process(pid="P3", arrival_time=4, burst_time=2, priority=2),
)

# Workload with idle gaps: first process arrives late, gap between P1 and P2.
IDLE_GAP_PROCS = (
    Process(pid="P1", arrival_time=3, burst_time=2, priority=1),
    Process(pid="P2", arrival_time=8, burst_time=3, priority=2),
)

# All arrive at time 0.
ALL_AT_ZERO = (
    Process(pid="P1", arrival_time=0, burst_time=5, priority=2),
    Process(pid="P2", arrival_time=0, burst_time=3, priority=1),
    Process(pid="P3", arrival_time=0, burst_time=4, priority=3),
)

# Single process.
SINGLE_PROC = (
    Process(pid="P1", arrival_time=0, burst_time=7, priority=1),
)

# Large arrival gap.
LARGE_GAP = (
    Process(pid="P1", arrival_time=0, burst_time=2, priority=1),
    Process(pid="P2", arrival_time=100, burst_time=3, priority=2),
)

# Identical burst times and arrival times.
IDENTICAL = (
    Process(pid="P1", arrival_time=0, burst_time=3, priority=1),
    Process(pid="P2", arrival_time=0, burst_time=3, priority=2),
    Process(pid="P3", arrival_time=0, burst_time=3, priority=3),
)

# Single large burst.
LARGE_BURST = (
    Process(pid="P1", arrival_time=0, burst_time=1000, priority=1),
)


def _all_schedulers() -> list[SchedulingAlgorithm]:
    """Return one instance of every algorithm."""
    return [FCFS(), SJF(), PriorityScheduler(), RoundRobin(time_quantum=2)]


def _all_inputs() -> list[tuple[str, Sequence[Process]]]:
    """Return labelled test inputs."""
    return [
        ("basic", BASIC_PROCS),
        ("idle_gap", IDLE_GAP_PROCS),
        ("all_at_zero", ALL_AT_ZERO),
        ("single", SINGLE_PROC),
        ("large_gap", LARGE_GAP),
        ("identical", IDENTICAL),
        ("large_burst", LARGE_BURST),
    ]


class TestSharedContracts(unittest.TestCase):
    """CHECK 1 — Every algorithm satisfies the SchedulingAlgorithm contract."""

    def test_all_subclass_scheduling_algorithm(self) -> None:
        for algo in _all_schedulers():
            with self.subTest(algo=algo.name):
                self.assertIsInstance(algo, SchedulingAlgorithm)

    def test_all_return_simulation_result(self) -> None:
        procs = [Process("P1", 0, 3, 1)]
        for algo in _all_schedulers():
            with self.subTest(algo=algo.name):
                result = algo.schedule(procs)
                self.assertIsInstance(result, SimulationResult)

    def test_all_produce_execution_segments(self) -> None:
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    self.assertGreater(len(result.execution_segments), 0)
                    for seg in result.execution_segments:
                        self.assertIsInstance(seg, ExecutionSegment)

    def test_all_produce_process_metrics(self) -> None:
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    self.assertEqual(len(result.process_metrics), len(procs))
                    for m in result.process_metrics:
                        self.assertIsInstance(m, ProcessMetrics)

    def test_all_reject_empty_input(self) -> None:
        for algo in _all_schedulers():
            with self.subTest(algo=algo.name):
                with self.assertRaises(ValueError):
                    algo.schedule([])

    def test_algorithm_name_is_nonempty_string(self) -> None:
        for algo in _all_schedulers():
            with self.subTest(algo=algo.name):
                self.assertIsInstance(algo.name, str)
                self.assertGreater(len(algo.name), 0)


class TestTimelineIntegrity(unittest.TestCase):
    """CHECK 3 — Timeline structural invariants."""

    def test_segments_have_positive_duration(self) -> None:
        """Every segment must have start_time < end_time."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    for seg in result.execution_segments:
                        self.assertLess(
                            seg.start_time, seg.end_time,
                            f"{algo.name}/{label}: zero-length or negative segment {seg}",
                        )

    def test_no_process_executes_before_arrival(self) -> None:
        """No busy segment may start before the process's arrival time."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    arrival_map = {p.pid: p.arrival_time for p in procs}
                    for seg in result.execution_segments:
                        if not seg.is_idle:
                            self.assertGreaterEqual(
                                seg.start_time,
                                arrival_map[seg.process_id],
                                f"{algo.name}/{label}: {seg.process_id} executes "
                                f"at {seg.start_time} but arrives at "
                                f"{arrival_map[seg.process_id]}",
                            )

    def test_segments_do_not_overlap(self) -> None:
        """Each segment's start must equal the previous segment's end."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    segs = result.execution_segments
                    for i in range(1, len(segs)):
                        self.assertEqual(
                            segs[i].start_time, segs[i - 1].end_time,
                            f"{algo.name}/{label}: gap/overlap between "
                            f"seg {i-1} and seg {i}",
                        )

    def test_timeline_starts_at_zero(self) -> None:
        """The first segment must start at time 0."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    self.assertEqual(
                        result.execution_segments[0].start_time, 0,
                        f"{algo.name}/{label}: timeline does not start at 0",
                    )

    def test_total_cpu_time_equals_total_burst(self) -> None:
        """Sum of busy (non-idle) segment durations must equal sum of burst times."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    busy_time = sum(
                        seg.end_time - seg.start_time
                        for seg in result.execution_segments
                        if not seg.is_idle
                    )
                    total_burst = sum(p.burst_time for p in procs)
                    self.assertEqual(
                        busy_time, total_burst,
                        f"{algo.name}/{label}: busy time {busy_time} != "
                        f"total burst {total_burst}",
                    )

    def test_every_process_cpu_time_equals_its_burst(self) -> None:
        """Per-process sum of execution durations equals its burst_time."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    burst_map = {p.pid: p.burst_time for p in procs}
                    actual: dict[str, int] = {}
                    for seg in result.execution_segments:
                        if not seg.is_idle:
                            actual[seg.process_id] = (
                                actual.get(seg.process_id, 0)
                                + seg.end_time - seg.start_time
                            )
                    for pid, expected_burst in burst_map.items():
                        self.assertEqual(
                            actual.get(pid, 0), expected_burst,
                            f"{algo.name}/{label}: {pid} ran "
                            f"{actual.get(pid, 0)} but burst is {expected_burst}",
                        )

    def test_idle_segments_use_idle_process_id(self) -> None:
        """All idle segments must use IDLE_PROCESS_ID."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    pids_in_input = {p.pid for p in procs}
                    for seg in result.execution_segments:
                        if seg.process_id not in pids_in_input:
                            self.assertEqual(seg.process_id, IDLE_PROCESS_ID)

    def test_every_input_process_appears_in_metrics(self) -> None:
        """Every input pid must have exactly one ProcessMetrics row."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    metric_pids = {m.pid for m in result.process_metrics}
                    input_pids = {p.pid for p in procs}
                    self.assertEqual(metric_pids, input_pids)

    def test_completion_time_matches_last_segment(self) -> None:
        """Each process's completion_time must equal its last segment's end_time."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    # Find the last segment end_time for each process.
                    last_end: dict[str, int] = {}
                    for seg in result.execution_segments:
                        if not seg.is_idle:
                            last_end[seg.process_id] = seg.end_time
                    for m in result.process_metrics:
                        self.assertEqual(
                            m.completion_time,
                            last_end[m.pid],
                            f"{algo.name}/{label}/{m.pid}: CT {m.completion_time} "
                            f"!= last segment end {last_end[m.pid]}",
                        )


class TestMetricConsistency(unittest.TestCase):
    """CHECK 4 — Metric formulas hold for every algorithm and input."""

    def test_turnaround_formula(self) -> None:
        """TAT = CT - AT for every process."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    for m in result.process_metrics:
                        self.assertEqual(
                            m.turnaround_time,
                            m.completion_time - m.arrival_time,
                            f"{algo.name}/{label}/{m.pid}: TAT formula",
                        )

    def test_waiting_formula(self) -> None:
        """WT = TAT - BT for every process."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    for m in result.process_metrics:
                        self.assertEqual(
                            m.waiting_time,
                            m.turnaround_time - m.burst_time,
                            f"{algo.name}/{label}/{m.pid}: WT formula",
                        )

    def test_waiting_time_non_negative(self) -> None:
        """WT >= 0 for every process."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    for m in result.process_metrics:
                        self.assertGreaterEqual(
                            m.waiting_time, 0,
                            f"{algo.name}/{label}/{m.pid}: negative WT",
                        )

    def test_average_waiting_time(self) -> None:
        """avg_wt = sum(WT) / n."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    expected = sum(m.waiting_time for m in result.process_metrics) / len(
                        result.process_metrics
                    )
                    self.assertAlmostEqual(
                        result.average_waiting_time, expected,
                        msg=f"{algo.name}/{label}: avg WT",
                    )

    def test_average_turnaround_time(self) -> None:
        """avg_tat = sum(TAT) / n."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    expected = sum(
                        m.turnaround_time for m in result.process_metrics
                    ) / len(result.process_metrics)
                    self.assertAlmostEqual(
                        result.average_turnaround_time, expected,
                        msg=f"{algo.name}/{label}: avg TAT",
                    )

    def test_metrics_copy_input_fields(self) -> None:
        """ProcessMetrics.arrival_time/burst_time/priority match the input."""
        for algo in _all_schedulers():
            for label, procs in _all_inputs():
                with self.subTest(algo=algo.name, input=label):
                    result = algo.schedule(procs)
                    input_map = {p.pid: p for p in procs}
                    for m in result.process_metrics:
                        p = input_map[m.pid]
                        self.assertEqual(m.arrival_time, p.arrival_time)
                        self.assertEqual(m.burst_time, p.burst_time)
                        self.assertEqual(m.priority, p.priority)


class TestEdgeCases(unittest.TestCase):
    """CHECK 6 — Additional edge-case coverage."""

    def test_large_arrival_gap_idle_segment(self) -> None:
        """A 98-unit gap must produce an idle segment."""
        for algo in _all_schedulers():
            with self.subTest(algo=algo.name):
                result = algo.schedule(LARGE_GAP)
                idle_segs = [s for s in result.execution_segments if s.is_idle]
                self.assertGreater(
                    len(idle_segs), 0,
                    f"{algo.name}: no idle segment for large gap",
                )
                # The idle segment covering 2→100 must exist.
                durations = [s.end_time - s.start_time for s in idle_segs]
                self.assertIn(98, durations)

    def test_single_large_burst(self) -> None:
        """A single process with burst=1000 must complete at 1000."""
        for algo in _all_schedulers():
            with self.subTest(algo=algo.name):
                result = algo.schedule(LARGE_BURST)
                self.assertEqual(result.process_metrics[0].completion_time, 1000)

    def test_round_robin_quantum_larger_than_all_bursts(self) -> None:
        """When quantum > every burst, RR behaves like FCFS."""
        rr = RoundRobin(time_quantum=100)
        procs = [
            Process(pid="P1", arrival_time=0, burst_time=3, priority=0),
            Process(pid="P2", arrival_time=0, burst_time=5, priority=0),
        ]
        result = rr.schedule(procs)
        # Each process should have exactly one segment (no preemption).
        busy_segs = [s for s in result.execution_segments if not s.is_idle]
        self.assertEqual(len(busy_segs), 2)
        self.assertEqual(busy_segs[0].process_id, "P1")
        self.assertEqual(busy_segs[1].process_id, "P2")

    def test_unsorted_input_does_not_mutate_caller(self) -> None:
        """The input list must not be reordered by any algorithm."""
        original = [
            Process(pid="P3", arrival_time=4, burst_time=1, priority=3),
            Process(pid="P1", arrival_time=0, burst_time=3, priority=1),
            Process(pid="P2", arrival_time=2, burst_time=2, priority=2),
        ]
        for algo in _all_schedulers():
            with self.subTest(algo=algo.name):
                snapshot = list(original)
                algo.schedule(original)
                self.assertEqual(
                    original, snapshot,
                    f"{algo.name}: mutated the input list",
                )

    def test_deterministic_across_runs(self) -> None:
        """Running same input twice gives identical output."""
        procs = list(BASIC_PROCS)
        for algo in _all_schedulers():
            with self.subTest(algo=algo.name):
                r1 = algo.schedule(procs)
                r2 = algo.schedule(procs)
                self.assertEqual(r1.execution_segments, r2.execution_segments)
                self.assertEqual(r1.process_metrics, r2.process_metrics)


if __name__ == "__main__":
    unittest.main()
