"""Comprehensive unit tests for the metrics enhancement layer (core/metrics.py).

Tests are organised by concern:

TestResponseTime
    Per-process response_time and average_response_time.
    Covers FCFS, SJF, Priority, Round Robin (including preemption).

TestCPUUtilization
    cpu_utilization percentage, with and without idle time.

TestThroughput
    throughput in processes-per-unit-time.

TestComputeMetricsReturnType
    Structural guarantees: returns SimulationResult, original result
    unchanged, no input mutation.

TestComputeMetricsRegressionAllAlgorithms
    Round-trip: compute_metrics applied after every algorithm still
    preserves all existing fields (average_waiting_time,
    average_turnaround_time, execution_segments, etc.)

Each test uses actual scheduling algorithm output so that tests
are faithful to the real data contract, not synthetic stub data.
"""

from __future__ import annotations

import math
from typing import Sequence

import pytest

from algorithms.fcfs import FCFS
from algorithms.priority import PriorityScheduler
from algorithms.round_robin import RoundRobin
from algorithms.sjf import SJF
from core.metrics import compute_metrics
from core.models import (
    IDLE_PROCESS_ID,
    ExecutionSegment,
    Process,
    ProcessMetrics,
    SimulationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_process(
    pid: str = "P1",
    arrival_time: int = 0,
    burst_time: int = 5,
    priority: int = 1,
) -> Process:
    return Process(
        pid=pid,
        arrival_time=arrival_time,
        burst_time=burst_time,
        priority=priority,
    )


def metrics_by_pid(result: SimulationResult) -> dict[str, ProcessMetrics]:
    return {m.pid: m for m in result.process_metrics}


# ===========================================================================
# TestResponseTime
# ===========================================================================


class TestResponseTimeFCFS:
    """Response time via FCFS (non-preemptive: response == waiting for FCFS)."""

    def test_single_process_at_time_zero(self) -> None:
        """Single process arriving at 0, dispatched at 0 → response_time = 0."""
        procs = [make_process("P1", arrival_time=0, burst_time=5)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert metrics_by_pid(result)["P1"].response_time == 0

    def test_two_processes_no_idle(self) -> None:
        """P1 arrives at 0 (burst 4), P2 at 0 (burst 3).
        FCFS tie-break by PID: P1 first (0→4), P2 next (4→7).
        P1 response = 0-0 = 0; P2 response = 4-0 = 4."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=4),
            make_process("P2", arrival_time=0, burst_time=3),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 4

    def test_process_with_initial_idle(self) -> None:
        """Only process arrives at t=5. CPU idles 0→5, then P1 runs 5→10.
        Response time = 5 - 5 = 0."""
        procs = [make_process("P1", arrival_time=5, burst_time=5)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert metrics_by_pid(result)["P1"].response_time == 0

    def test_sequential_processes_with_gaps(self) -> None:
        """P1 arrives at 0 (burst 3), P2 arrives at 10 (burst 2).
        P1: 0→3 (response=0), idle 3→10, P2: 10→12 (response=10-10=0)."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=3),
            make_process("P2", arrival_time=10, burst_time=2),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 0

    def test_three_processes_response_times(self) -> None:
        """Three sequential processes all arriving at t=0.
        FCFS order: P1(0→5), P2(5→8), P3(8→12).
        Response times: 0, 5, 8."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=5),
            make_process("P2", arrival_time=0, burst_time=3),
            make_process("P3", arrival_time=0, burst_time=4),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        m = metrics_by_pid(result)
        # FCFS sorts by (arrival_time, pid) → P1, P2, P3
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 5
        assert m["P3"].response_time == 8

    def test_average_response_time(self) -> None:
        """avg_response_time = mean of per-process response times."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=5),
            make_process("P2", arrival_time=0, burst_time=3),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        m = metrics_by_pid(result)
        expected_avg = (m["P1"].response_time + m["P2"].response_time) / 2
        assert result.average_response_time == pytest.approx(expected_avg)

    def test_response_time_non_negative(self) -> None:
        """Response time must be >= 0 for all processes."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=6),
            make_process("P2", arrival_time=2, burst_time=4),
            make_process("P3", arrival_time=4, burst_time=2),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        for m in result.process_metrics:
            assert m.response_time is not None
            assert m.response_time >= 0

    def test_response_time_is_int(self) -> None:
        """Response time for FCFS is an integer (first dispatch - arrival)."""
        procs = [make_process("P1", arrival_time=3, burst_time=4)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        rt = metrics_by_pid(result)["P1"].response_time
        assert isinstance(rt, int)


class TestResponseTimeSJF:
    """Response time via non-preemptive SJF."""

    def test_sjf_selects_shortest_first(self) -> None:
        """P1(burst=6) and P2(burst=2) both arrive at 0.
        SJF picks P2 first: P2 0→2 (rt=0), P1 2→8 (rt=2)."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=6),
            make_process("P2", arrival_time=0, burst_time=2),
        ]
        result = compute_metrics(SJF().schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P2"].response_time == 0
        assert m["P1"].response_time == 2

    def test_sjf_with_staggered_arrivals(self) -> None:
        """P1(at=0, burst=8), P2(at=3, burst=2).
        At t=0 only P1 is ready → P1 runs 0→8.
        P2 is first dispatched at t=8: response = 8-3 = 5."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=8),
            make_process("P2", arrival_time=3, burst_time=2),
        ]
        result = compute_metrics(SJF().schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 5

    def test_sjf_idle_then_dispatch(self) -> None:
        """Only P1 arrives at t=10. CPU idles 0→10, P1 runs 10→14.
        response = 10 - 10 = 0."""
        procs = [make_process("P1", arrival_time=10, burst_time=4)]
        result = compute_metrics(SJF().schedule(procs), procs)
        assert metrics_by_pid(result)["P1"].response_time == 0


class TestResponseTimePriority:
    """Response time via non-preemptive Priority scheduling."""

    def test_priority_dispatches_highest_priority_first(self) -> None:
        """P1(prio=2, burst=5) and P2(prio=1, burst=3) arrive at t=0.
        Priority picks P2 first (lower number = higher priority).
        P2: 0→3 (rt=0), P1: 3→8 (rt=3)."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=5, priority=2),
            make_process("P2", arrival_time=0, burst_time=3, priority=1),
        ]
        result = compute_metrics(PriorityScheduler().schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P2"].response_time == 0
        assert m["P1"].response_time == 3

    def test_priority_with_delayed_arrival(self) -> None:
        """P1(at=0,prio=1,burst=5), P2(at=3,prio=1,burst=2).
        P1 dispatched at 0 (non-preemptive), P2 first dispatched at 5.
        P1 rt=0, P2 rt=5-3=2."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=5, priority=1),
            make_process("P2", arrival_time=3, burst_time=2, priority=1),
        ]
        result = compute_metrics(PriorityScheduler().schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 2


class TestResponseTimeRoundRobin:
    """Response time via preemptive Round Robin.
    Response time = start of first segment for each process - arrival_time.
    A process may receive CPU multiple times; we want only the FIRST."""

    def test_rr_single_process(self) -> None:
        """Single process arriving at 0 → dispatched at 0, response=0."""
        procs = [make_process("P1", arrival_time=0, burst_time=6)]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        assert metrics_by_pid(result)["P1"].response_time == 0

    def test_rr_two_processes_at_zero(self) -> None:
        """P1(at=0,burst=4) and P2(at=0,burst=4), quantum=2.
        Sorted by (at,pid): P1 first, P2 second.
        Segments: P1 0→2, P2 2→4, P1 4→6, P2 6→8.
        P1 rt = 0-0 = 0; P2 rt = 2-0 = 2."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=4),
            make_process("P2", arrival_time=0, burst_time=4),
        ]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 2

    def test_rr_multiple_preemptions_response_uses_first(self) -> None:
        """P1(burst=6) receives CPU at t=0,4,8 with quantum=2.
        response_time must be 0 (first dispatch), not 4 or 8."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=6),
            make_process("P2", arrival_time=0, burst_time=6),
        ]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P1"].response_time == 0   # first dispatch at t=0
        assert m["P2"].response_time == 2   # first dispatch at t=2

    def test_rr_delayed_arrival(self) -> None:
        """P1(at=0,burst=4), P2(at=5,burst=2), quantum=2.
        P1 runs 0→2, P1 runs 2→4 (done), idle 4→5, P2 runs 5→7 (done).
        P1 rt=0, P2 rt=5-5=0."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=4),
            make_process("P2", arrival_time=5, burst_time=2),
        ]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 0

    def test_rr_arrival_during_execution(self) -> None:
        """P1(at=0,burst=4), P2(at=1,burst=4), quantum=2.
        P1 runs 0→2. At t=2 P2 has arrived → enqueue P2 then P1.
        P2 runs 2→4, P1 runs 4→6.
        P2 rt = 2 - 1 = 1."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=4),
            make_process("P2", arrival_time=1, burst_time=4),
        ]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        m = metrics_by_pid(result)
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 1

    def test_rr_average_response_time_is_correct(self) -> None:
        """average_response_time = mean of per-process response times."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=4),
            make_process("P2", arrival_time=0, burst_time=4),
            make_process("P3", arrival_time=0, burst_time=4),
        ]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        m = metrics_by_pid(result)
        expected_avg = (
            m["P1"].response_time + m["P2"].response_time + m["P3"].response_time
        ) / 3
        assert result.average_response_time == pytest.approx(expected_avg)

    def test_rr_response_time_always_less_than_or_equal_to_waiting_time(self) -> None:
        """For any algorithm, response_time <= waiting_time is not necessarily
        true in general, but response_time <= turnaround_time always holds."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=6),
            make_process("P2", arrival_time=2, burst_time=4),
        ]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        for m in result.process_metrics:
            assert m.response_time is not None
            assert m.response_time <= m.turnaround_time


# ===========================================================================
# TestCPUUtilization
# ===========================================================================


class TestCPUUtilization:
    """cpu_utilization is a percentage (0.0–100.0) of busy CPU time."""

    def test_single_process_no_idle_full_utilization(self) -> None:
        """Single process arriving at 0: CPU is busy 100% of the time."""
        procs = [make_process("P1", arrival_time=0, burst_time=10)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.cpu_utilization == pytest.approx(100.0)

    def test_multiple_processes_no_idle_full_utilization(self) -> None:
        """All processes arrive at 0: no idle segments, 100% utilization."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=5),
            make_process("P2", arrival_time=0, burst_time=3),
            make_process("P3", arrival_time=0, burst_time=4),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.cpu_utilization == pytest.approx(100.0)

    def test_initial_idle_reduces_utilization(self) -> None:
        """P1 arrives at t=5 with burst=5.
        Total time = 10 (0→10). Busy time = 5. Utilization = 50%."""
        procs = [make_process("P1", arrival_time=5, burst_time=5)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.cpu_utilization == pytest.approx(50.0)

    def test_idle_gap_between_processes(self) -> None:
        """P1(at=0,burst=2), P2(at=10,burst=3).
        Total time = 13 (0→13). Busy time = 5. Utilization = 5/13 * 100."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=2),
            make_process("P2", arrival_time=10, burst_time=3),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        expected = (5 / 13) * 100.0
        assert result.cpu_utilization == pytest.approx(expected)

    def test_large_idle_gap(self) -> None:
        """P1(at=0,burst=2), P2(at=100,burst=3).
        Segments: P1 0→2, IDLE 2→100, P2 100→103.
        Total time = 103. Busy time = 5. Utilization = 5/103 * 100."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=2),
            make_process("P2", arrival_time=100, burst_time=3),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        expected = (5 / 103) * 100.0
        assert result.cpu_utilization == pytest.approx(expected)

    def test_utilization_between_0_and_100(self) -> None:
        """Utilization must always be in [0.0, 100.0]."""
        test_cases = [
            [make_process("P1", arrival_time=0, burst_time=5)],
            [make_process("P1", arrival_time=5, burst_time=5)],
            [
                make_process("P1", arrival_time=0, burst_time=3),
                make_process("P2", arrival_time=10, burst_time=2),
            ],
        ]
        for procs in test_cases:
            result = compute_metrics(FCFS().schedule(procs), procs)
            assert result.cpu_utilization is not None
            assert 0.0 <= result.cpu_utilization <= 100.0

    def test_rr_utilization_no_idle(self) -> None:
        """Round Robin with all processes arriving at 0: 100% utilization."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=4),
            make_process("P2", arrival_time=0, burst_time=4),
        ]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        assert result.cpu_utilization == pytest.approx(100.0)

    def test_rr_utilization_with_idle(self) -> None:
        """RR with P1(at=0,burst=2) and P2(at=10,burst=2), quantum=2.
        P1: 0→2, idle 2→10, P2: 10→12.
        Total=12, busy=4, utilization=4/12*100≈33.33%."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=2),
            make_process("P2", arrival_time=10, burst_time=2),
        ]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        expected = (4 / 12) * 100.0
        assert result.cpu_utilization == pytest.approx(expected)

    def test_utilization_all_algorithms_no_idle(self) -> None:
        """All algorithms: when no idle time, utilization must be 100%."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=5),
            make_process("P2", arrival_time=0, burst_time=3),
        ]
        for algo in [FCFS(), SJF(), PriorityScheduler(), RoundRobin(time_quantum=2)]:
            result = compute_metrics(algo.schedule(procs), procs)
            assert result.cpu_utilization == pytest.approx(100.0), algo.name

    def test_utilization_is_float(self) -> None:
        procs = [make_process("P1", arrival_time=0, burst_time=5)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert isinstance(result.cpu_utilization, float)


# ===========================================================================
# TestThroughput
# ===========================================================================


class TestThroughput:
    """throughput = n_processes / total_simulation_time."""

    def test_single_process(self) -> None:
        """P1(at=0, burst=5). Total time=5. Throughput = 1/5 = 0.2."""
        procs = [make_process("P1", arrival_time=0, burst_time=5)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.throughput == pytest.approx(1 / 5)

    def test_two_processes_no_idle(self) -> None:
        """P1(burst=5)+P2(burst=3), both at 0. Total=8. Throughput=2/8=0.25."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=5),
            make_process("P2", arrival_time=0, burst_time=3),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.throughput == pytest.approx(2 / 8)

    def test_throughput_with_idle(self) -> None:
        """P1(at=0,burst=2), P2(at=10,burst=3).
        Total=13. Throughput=2/13."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=2),
            make_process("P2", arrival_time=10, burst_time=3),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.throughput == pytest.approx(2 / 13)

    def test_throughput_delayed_single_process(self) -> None:
        """P1 arrives at t=5 (burst=5). Timeline 0→10. Throughput=1/10."""
        procs = [make_process("P1", arrival_time=5, burst_time=5)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.throughput == pytest.approx(1 / 10)

    def test_throughput_multiple_processes(self) -> None:
        """Three processes all at 0 with bursts 3,4,5. Total=12. TP=3/12=0.25."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=3),
            make_process("P2", arrival_time=0, burst_time=4),
            make_process("P3", arrival_time=0, burst_time=5),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.throughput == pytest.approx(3 / 12)

    def test_throughput_all_algorithms(self) -> None:
        """All algorithms: throughput = n / total_time."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=4),
            make_process("P2", arrival_time=0, burst_time=2),
        ]
        for algo in [FCFS(), SJF(), PriorityScheduler(), RoundRobin(time_quantum=2)]:
            result = compute_metrics(algo.schedule(procs), procs)
            segs = result.execution_segments
            expected_total = segs[-1].end_time - segs[0].start_time
            expected_tp = 2 / expected_total
            assert result.throughput == pytest.approx(expected_tp), algo.name

    def test_throughput_is_float(self) -> None:
        procs = [make_process("P1", arrival_time=0, burst_time=3)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert isinstance(result.throughput, float)

    def test_throughput_rr_single_process(self) -> None:
        """RR, single process: throughput = 1 / burst_time (no idle)."""
        procs = [make_process("P1", arrival_time=0, burst_time=6)]
        result = compute_metrics(RoundRobin(time_quantum=2).schedule(procs), procs)
        assert result.throughput == pytest.approx(1 / 6)


# ===========================================================================
# TestComputeMetricsReturnType
# ===========================================================================


class TestComputeMetricsReturnType:
    """Structural guarantees about the compute_metrics return value."""

    def test_returns_simulation_result(self) -> None:
        procs = [make_process("P1")]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert isinstance(result, SimulationResult)

    def test_original_result_unchanged(self) -> None:
        """compute_metrics must not mutate the input SimulationResult."""
        procs = [make_process("P1", arrival_time=0, burst_time=5)]
        original = FCFS().schedule(procs)
        # These are None before compute_metrics
        assert original.average_response_time is None
        assert original.cpu_utilization is None
        assert original.throughput is None
        # Run compute_metrics
        compute_metrics(original, procs)
        # Original still has None
        assert original.average_response_time is None
        assert original.cpu_utilization is None
        assert original.throughput is None

    def test_original_process_list_unchanged(self) -> None:
        """compute_metrics must not mutate the input process list."""
        procs = [make_process("P1", arrival_time=2, burst_time=5)]
        snapshot_pid = procs[0].pid
        snapshot_at = procs[0].arrival_time
        compute_metrics(FCFS().schedule(procs), procs)
        assert procs[0].pid == snapshot_pid
        assert procs[0].arrival_time == snapshot_at

    def test_all_optional_fields_filled(self) -> None:
        """After compute_metrics, no optional field is None."""
        procs = [make_process("P1"), make_process("P2", arrival_time=2)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert result.average_response_time is not None
        assert result.cpu_utilization is not None
        assert result.throughput is not None
        for m in result.process_metrics:
            assert m.response_time is not None

    def test_execution_segments_preserved(self) -> None:
        """The execution_segments tuple must be identical to the original."""
        procs = [make_process("P1"), make_process("P2", arrival_time=3)]
        original = FCFS().schedule(procs)
        result = compute_metrics(original, procs)
        assert result.execution_segments == original.execution_segments

    def test_average_waiting_and_turnaround_preserved(self) -> None:
        """Average waiting/turnaround times must be unchanged by compute_metrics."""
        procs = [make_process("P1"), make_process("P2", arrival_time=1)]
        original = FCFS().schedule(procs)
        result = compute_metrics(original, procs)
        assert result.average_waiting_time == pytest.approx(
            original.average_waiting_time
        )
        assert result.average_turnaround_time == pytest.approx(
            original.average_turnaround_time
        )

    def test_algorithm_name_preserved(self) -> None:
        procs = [make_process("P1")]
        original = FCFS().schedule(procs)
        result = compute_metrics(original, procs)
        assert result.algorithm_name == original.algorithm_name

    def test_process_metrics_count_preserved(self) -> None:
        procs = [make_process("P1"), make_process("P2", arrival_time=1)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert len(result.process_metrics) == 2


# ===========================================================================
# TestComputeMetricsRegressionAllAlgorithms
# ===========================================================================


class TestComputeMetricsRegressionAllAlgorithms:
    """Regression: compute_metrics applied over all four algorithms still
    satisfies the original scheduling engine's metric formulas."""

    PROCS = [
        Process(pid="P1", arrival_time=0, burst_time=6, priority=3),
        Process(pid="P2", arrival_time=2, burst_time=4, priority=1),
        Process(pid="P3", arrival_time=4, burst_time=2, priority=2),
    ]

    def _all_algorithms(self):
        return [FCFS(), SJF(), PriorityScheduler(), RoundRobin(time_quantum=2)]

    def test_turnaround_formula_holds_after_compute_metrics(self) -> None:
        """TAT = CT - AT for every process after enrichment."""
        for algo in self._all_algorithms():
            result = compute_metrics(algo.schedule(self.PROCS), self.PROCS)
            for m in result.process_metrics:
                assert m.turnaround_time == m.completion_time - m.arrival_time, (
                    f"{algo.name}/{m.pid}"
                )

    def test_waiting_formula_holds_after_compute_metrics(self) -> None:
        """WT = TAT - BT for every process after enrichment."""
        for algo in self._all_algorithms():
            result = compute_metrics(algo.schedule(self.PROCS), self.PROCS)
            for m in result.process_metrics:
                assert m.waiting_time == m.turnaround_time - m.burst_time, (
                    f"{algo.name}/{m.pid}"
                )

    def test_response_time_not_none(self) -> None:
        """All response times are filled (not None)."""
        for algo in self._all_algorithms():
            result = compute_metrics(algo.schedule(self.PROCS), self.PROCS)
            for m in result.process_metrics:
                assert m.response_time is not None, f"{algo.name}/{m.pid}"

    def test_response_time_non_negative(self) -> None:
        for algo in self._all_algorithms():
            result = compute_metrics(algo.schedule(self.PROCS), self.PROCS)
            for m in result.process_metrics:
                assert m.response_time >= 0, f"{algo.name}/{m.pid}"  # type: ignore[operator]

    def test_average_response_time_equals_mean_of_individual(self) -> None:
        for algo in self._all_algorithms():
            result = compute_metrics(algo.schedule(self.PROCS), self.PROCS)
            expected = sum(
                m.response_time for m in result.process_metrics  # type: ignore[misc]
            ) / len(result.process_metrics)
            assert result.average_response_time == pytest.approx(expected), algo.name

    def test_cpu_utilization_not_none(self) -> None:
        for algo in self._all_algorithms():
            result = compute_metrics(algo.schedule(self.PROCS), self.PROCS)
            assert result.cpu_utilization is not None, algo.name

    def test_throughput_not_none(self) -> None:
        for algo in self._all_algorithms():
            result = compute_metrics(algo.schedule(self.PROCS), self.PROCS)
            assert result.throughput is not None, algo.name

    def test_completion_time_unchanged(self) -> None:
        """compute_metrics must not alter completion_time for any process."""
        for algo in self._all_algorithms():
            original = algo.schedule(self.PROCS)
            enriched = compute_metrics(original, self.PROCS)
            orig_ct = {m.pid: m.completion_time for m in original.process_metrics}
            enr_ct = {m.pid: m.completion_time for m in enriched.process_metrics}
            assert orig_ct == enr_ct, algo.name

    def test_waiting_time_unchanged(self) -> None:
        for algo in self._all_algorithms():
            original = algo.schedule(self.PROCS)
            enriched = compute_metrics(original, self.PROCS)
            orig_wt = {m.pid: m.waiting_time for m in original.process_metrics}
            enr_wt = {m.pid: m.waiting_time for m in enriched.process_metrics}
            assert orig_wt == enr_wt, algo.name

    def test_response_time_leq_turnaround_time(self) -> None:
        """Response time can never exceed turnaround time."""
        for algo in self._all_algorithms():
            result = compute_metrics(algo.schedule(self.PROCS), self.PROCS)
            for m in result.process_metrics:
                assert m.response_time <= m.turnaround_time, (
                    f"{algo.name}/{m.pid}: rt={m.response_time} > tat={m.turnaround_time}"
                )


# ===========================================================================
# TestEdgeCasesMetrics
# ===========================================================================


class TestEdgeCasesMetrics:
    """Additional edge cases."""

    def test_single_process_large_burst(self) -> None:
        """Single process with burst=1000, arrival=0. All metrics are defined."""
        procs = [make_process("P1", arrival_time=0, burst_time=1000)]
        result = compute_metrics(FCFS().schedule(procs), procs)
        assert metrics_by_pid(result)["P1"].response_time == 0
        assert result.cpu_utilization == pytest.approx(100.0)
        assert result.throughput == pytest.approx(1 / 1000)
        assert result.average_response_time == pytest.approx(0.0)

    def test_all_processes_same_arrival_same_burst(self) -> None:
        """Identical processes: response times increment by burst_time each."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=3),
            make_process("P2", arrival_time=0, burst_time=3),
            make_process("P3", arrival_time=0, burst_time=3),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        m = metrics_by_pid(result)
        # FCFS order: P1(0→3), P2(3→6), P3(6→9)
        assert m["P1"].response_time == 0
        assert m["P2"].response_time == 3
        assert m["P3"].response_time == 6
        assert result.cpu_utilization == pytest.approx(100.0)
        assert result.throughput == pytest.approx(3 / 9)

    def test_response_time_equals_waiting_time_for_fcfs(self) -> None:
        """For FCFS, response_time == waiting_time for every process
        (because FCFS is non-preemptive and dispatches in FIFO order)."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=5),
            make_process("P2", arrival_time=0, burst_time=3),
        ]
        result = compute_metrics(FCFS().schedule(procs), procs)
        for m in result.process_metrics:
            assert m.response_time == m.waiting_time, (
                f"FCFS/{m.pid}: rt={m.response_time} != wt={m.waiting_time}"
            )

    def test_compute_metrics_idempotent(self) -> None:
        """Calling compute_metrics twice gives the same result."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=4),
            make_process("P2", arrival_time=2, burst_time=2),
        ]
        r1 = compute_metrics(FCFS().schedule(procs), procs)
        r2 = compute_metrics(r1, procs)
        # Key metrics must be identical
        assert r1.average_response_time == pytest.approx(r2.average_response_time)  # type: ignore[arg-type]
        assert r1.cpu_utilization == pytest.approx(r2.cpu_utilization)  # type: ignore[arg-type]
        assert r1.throughput == pytest.approx(r2.throughput)  # type: ignore[arg-type]
        for m1, m2 in zip(
            sorted(r1.process_metrics, key=lambda m: m.pid),
            sorted(r2.process_metrics, key=lambda m: m.pid),
        ):
            assert m1.response_time == m2.response_time

    def test_utilization_with_sjf_gap(self) -> None:
        """SJF where first process arrives at t=5.
        Timeline: idle 0→5, P1 5→8. Busy=3, total=8, util=3/8*100=37.5%."""
        procs = [make_process("P1", arrival_time=5, burst_time=3)]
        result = compute_metrics(SJF().schedule(procs), procs)
        assert result.cpu_utilization == pytest.approx(37.5)

    def test_throughput_with_priority_scheduler(self) -> None:
        """Priority scheduler: throughput = n / total_time."""
        procs = [
            make_process("P1", arrival_time=0, burst_time=3, priority=2),
            make_process("P2", arrival_time=0, burst_time=5, priority=1),
        ]
        result = compute_metrics(PriorityScheduler().schedule(procs), procs)
        segs = result.execution_segments
        total = segs[-1].end_time - segs[0].start_time
        assert result.throughput == pytest.approx(2 / total)
