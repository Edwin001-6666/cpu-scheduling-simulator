"""Comprehensive unit tests for the validation layer (core/validation.py).

Test organisation
-----------------
TestValidateProcesses:
    Valid inputs
    Invalid inputs — structural
    Invalid inputs — PID rules
    Invalid inputs — arrival_time rules
    Invalid inputs — burst_time rules
    Invalid inputs — priority rules
    Data integrity (validation must not modify input)

TestValidateRoundRobinQuantum:
    Valid quantum
    Invalid quantum — None / missing
    Invalid quantum — zero / negative
    Invalid quantum — non-integer types

All tests use only the public API:
    validate_processes(processes)       -> ValidationResult
    validate_round_robin_quantum(q)     -> ValidationResult
    ValidationResult.is_valid           -> bool
    ValidationResult.errors             -> list[ValidationError]
    ValidationError.field               -> str
    ValidationError.message             -> str
    ValidationError.process_id          -> str | None
"""

from __future__ import annotations

import pytest

from core.models import IDLE_PROCESS_ID, Process
from core.validation import (
    ValidationError,
    ValidationResult,
    validate_processes,
    validate_round_robin_quantum,
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
    """Return a valid Process with sensible defaults."""
    return Process(
        pid=pid,
        arrival_time=arrival_time,
        burst_time=burst_time,
        priority=priority,
    )


def error_fields(result: ValidationResult) -> list[str]:
    """Extract the list of field names from a ValidationResult's errors."""
    return [e.field for e in result.errors]


def error_pids(result: ValidationResult) -> list[str | None]:
    """Extract the list of process_id values from a ValidationResult's errors."""
    return [e.process_id for e in result.errors]


# ===========================================================================
# validate_processes
# ===========================================================================


class TestValidateProcessesValid:
    """Valid inputs must produce is_valid=True with no errors."""

    def test_single_valid_process(self) -> None:
        procs = [make_process("P1", 0, 5, 1)]
        result = validate_processes(procs)
        assert result.is_valid
        assert result.errors == []

    def test_multiple_valid_processes(self) -> None:
        procs = [
            make_process("P1", 0, 5, 2),
            make_process("P2", 2, 3, 1),
            make_process("P3", 4, 7, 3),
        ]
        result = validate_processes(procs)
        assert result.is_valid
        assert result.errors == []

    def test_unsorted_arrival_times_are_accepted(self) -> None:
        """Processes do not need to be sorted by arrival time."""
        procs = [
            make_process("P3", 10, 2, 1),
            make_process("P1", 0, 5, 2),
            make_process("P2", 5, 3, 3),
        ]
        result = validate_processes(procs)
        assert result.is_valid

    def test_arrival_time_zero_is_valid(self) -> None:
        procs = [make_process("P1", arrival_time=0)]
        result = validate_processes(procs)
        assert result.is_valid

    def test_arrival_time_large_value_is_valid(self) -> None:
        procs = [make_process("P1", arrival_time=10000)]
        result = validate_processes(procs)
        assert result.is_valid

    def test_burst_time_one_is_valid(self) -> None:
        procs = [make_process("P1", burst_time=1)]
        result = validate_processes(procs)
        assert result.is_valid

    def test_large_burst_time_is_valid(self) -> None:
        procs = [make_process("P1", burst_time=9999)]
        result = validate_processes(procs)
        assert result.is_valid

    def test_priority_low_numeric_is_high_priority(self) -> None:
        """Lower numeric priority value = higher priority (per DECISIONS.txt)."""
        procs = [
            make_process("P1", priority=1),
            make_process("P2", priority=10),
        ]
        result = validate_processes(procs)
        assert result.is_valid

    def test_priority_zero_is_valid(self) -> None:
        procs = [make_process("P1", priority=0)]
        result = validate_processes(procs)
        assert result.is_valid

    def test_negative_priority_is_valid(self) -> None:
        """Negative priority is a valid integer; convention is just lower = higher."""
        procs = [make_process("P1", priority=-5)]
        result = validate_processes(procs)
        assert result.is_valid

    def test_all_same_arrival_time_is_valid(self) -> None:
        procs = [
            make_process("P1", arrival_time=0),
            make_process("P2", arrival_time=0),
            make_process("P3", arrival_time=0),
        ]
        result = validate_processes(procs)
        assert result.is_valid

    def test_tuple_of_processes_is_valid(self) -> None:
        """Accepts a tuple as well as a list."""
        procs = (make_process("P1"), make_process("P2", arrival_time=1))
        result = validate_processes(procs)
        assert result.is_valid

    def test_bool_result_reflects_is_valid(self) -> None:
        """ValidationResult.__bool__ returns is_valid."""
        procs = [make_process("P1")]
        result = validate_processes(procs)
        assert bool(result) is True


# ---------------------------------------------------------------------------
# Structural / collection-level errors
# ---------------------------------------------------------------------------


class TestValidateProcessesStructural:
    """Tests for structural problems with the process collection."""

    def test_empty_list_is_invalid(self) -> None:
        result = validate_processes([])
        assert not result.is_valid
        assert len(result.errors) >= 1
        assert "processes" in error_fields(result)

    def test_empty_list_error_message_is_friendly(self) -> None:
        result = validate_processes([])
        assert any("empty" in e.message.lower() for e in result.errors)

    def test_none_input_is_invalid(self) -> None:
        result = validate_processes(None)
        assert not result.is_valid
        assert "processes" in error_fields(result)

    def test_string_input_is_invalid(self) -> None:
        result = validate_processes("P1")
        assert not result.is_valid
        assert "processes" in error_fields(result)

    def test_integer_input_is_invalid(self) -> None:
        result = validate_processes(42)
        assert not result.is_valid
        assert "processes" in error_fields(result)

    def test_non_process_items_are_flagged(self) -> None:
        """A list containing a dict instead of a Process is invalid."""
        procs = [{"pid": "P1", "arrival_time": 0, "burst_time": 5, "priority": 1}]
        result = validate_processes(procs)
        assert not result.is_valid
        assert "processes" in error_fields(result)

    def test_mixed_valid_and_non_process_items(self) -> None:
        """Mixed list: valid Process + dict — the dict should be reported."""
        procs = [make_process("P1"), {"pid": "P2", "burst_time": 3}]
        result = validate_processes(procs)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# PID errors
# ---------------------------------------------------------------------------


class TestValidateProcessesPID:
    """Tests for PID-specific validation rules."""

    def test_empty_string_pid_is_invalid(self) -> None:
        # Process is a frozen dataclass; we can construct one with "" pid
        procs = [Process(pid="", arrival_time=0, burst_time=5, priority=1)]
        result = validate_processes(procs)
        assert not result.is_valid
        assert "pid" in error_fields(result)

    def test_whitespace_only_pid_is_invalid(self) -> None:
        procs = [Process(pid="   ", arrival_time=0, burst_time=5, priority=1)]
        result = validate_processes(procs)
        assert not result.is_valid
        assert "pid" in error_fields(result)

    def test_reserved_idle_pid_is_invalid(self) -> None:
        procs = [Process(pid=IDLE_PROCESS_ID, arrival_time=0, burst_time=5, priority=1)]
        result = validate_processes(procs)
        assert not result.is_valid
        pids_in_errors = error_pids(result)
        assert IDLE_PROCESS_ID in pids_in_errors

    def test_reserved_idle_pid_error_message_mentions_reserved(self) -> None:
        procs = [Process(pid=IDLE_PROCESS_ID, arrival_time=0, burst_time=5, priority=1)]
        result = validate_processes(procs)
        pid_errors = [e for e in result.errors if e.field == "pid"]
        assert any("reserved" in e.message.lower() for e in pid_errors)

    def test_duplicate_pid_is_invalid(self) -> None:
        procs = [make_process("P1"), make_process("P1", arrival_time=2)]
        result = validate_processes(procs)
        assert not result.is_valid
        assert "pid" in error_fields(result)

    def test_duplicate_pid_error_mentions_the_pid(self) -> None:
        procs = [make_process("P2"), make_process("P2", arrival_time=1)]
        result = validate_processes(procs)
        pid_errors = [e for e in result.errors if e.field == "pid"]
        assert any("P2" in e.message for e in pid_errors)

    def test_duplicate_pid_process_id_field_set(self) -> None:
        procs = [make_process("P3"), make_process("P3", arrival_time=3)]
        result = validate_processes(procs)
        dup_errors = [e for e in result.errors if e.field == "pid" and e.process_id == "P3"]
        assert len(dup_errors) >= 1

    def test_three_processes_two_duplicates(self) -> None:
        """P1, P2, P1 — should report the duplicate."""
        procs = [
            make_process("P1"),
            make_process("P2", arrival_time=1),
            make_process("P1", arrival_time=2),
        ]
        result = validate_processes(procs)
        assert not result.is_valid

    def test_all_unique_pids_no_duplicate_error(self) -> None:
        procs = [make_process(f"P{i}", arrival_time=i) for i in range(1, 6)]
        result = validate_processes(procs)
        dup_errors = [e for e in result.errors if e.field == "pid" and "duplicate" in e.message.lower()]
        assert dup_errors == []


# ---------------------------------------------------------------------------
# Arrival time errors
# ---------------------------------------------------------------------------


class TestValidateProcessesArrivalTime:
    """Tests for arrival_time validation rules."""

    def test_negative_arrival_time_is_invalid(self) -> None:
        procs = [make_process("P1", arrival_time=-1)]
        result = validate_processes(procs)
        assert not result.is_valid
        assert "arrival_time" in error_fields(result)

    def test_negative_arrival_time_error_message(self) -> None:
        procs = [make_process("P1", arrival_time=-5)]
        result = validate_processes(procs)
        at_errors = [e for e in result.errors if e.field == "arrival_time"]
        assert at_errors
        assert any(">= 0" in e.message or "must be" in e.message for e in at_errors)

    def test_negative_arrival_time_process_id_reported(self) -> None:
        procs = [make_process("P9", arrival_time=-3)]
        result = validate_processes(procs)
        at_errors = [e for e in result.errors if e.field == "arrival_time"]
        assert any(e.process_id == "P9" for e in at_errors)

    def test_arrival_time_not_corrected_silently(self) -> None:
        """Validation must not auto-correct negative arrival_time."""
        original = make_process("P1", arrival_time=-2)
        procs = [original]
        validate_processes(procs)
        # The original process object must be unmodified (frozen dataclass)
        assert original.arrival_time == -2

    def test_multiple_processes_with_negative_arrival_times(self) -> None:
        procs = [
            make_process("P1", arrival_time=-1),
            make_process("P2", arrival_time=0),
            make_process("P3", arrival_time=-3),
        ]
        result = validate_processes(procs)
        assert not result.is_valid
        at_errors = [e for e in result.errors if e.field == "arrival_time"]
        reported_pids = {e.process_id for e in at_errors}
        assert "P1" in reported_pids
        assert "P3" in reported_pids
        assert "P2" not in reported_pids


# ---------------------------------------------------------------------------
# Burst time errors
# ---------------------------------------------------------------------------


class TestValidateProcessesBurstTime:
    """Tests for burst_time validation rules."""

    def test_zero_burst_time_is_invalid(self) -> None:
        procs = [make_process("P1", burst_time=0)]
        result = validate_processes(procs)
        assert not result.is_valid
        assert "burst_time" in error_fields(result)

    def test_zero_burst_time_error_message_is_friendly(self) -> None:
        procs = [make_process("P2", burst_time=0)]
        result = validate_processes(procs)
        bt_errors = [e for e in result.errors if e.field == "burst_time"]
        assert bt_errors
        # The message should mention the PID and the burst time value
        assert any("P2" in e.message for e in bt_errors)
        assert any("0" in e.message for e in bt_errors)

    def test_negative_burst_time_is_invalid(self) -> None:
        procs = [make_process("P1", burst_time=-1)]
        result = validate_processes(procs)
        assert not result.is_valid
        assert "burst_time" in error_fields(result)

    def test_negative_burst_time_large(self) -> None:
        procs = [make_process("P1", burst_time=-100)]
        result = validate_processes(procs)
        assert not result.is_valid

    def test_burst_time_process_id_reported(self) -> None:
        procs = [make_process("P7", burst_time=0)]
        result = validate_processes(procs)
        bt_errors = [e for e in result.errors if e.field == "burst_time"]
        assert any(e.process_id == "P7" for e in bt_errors)

    def test_multiple_processes_with_invalid_burst_times(self) -> None:
        procs = [
            make_process("P1", burst_time=5),
            make_process("P2", burst_time=0),
            make_process("P3", burst_time=-2),
        ]
        result = validate_processes(procs)
        assert not result.is_valid
        bt_errors = [e for e in result.errors if e.field == "burst_time"]
        pids = {e.process_id for e in bt_errors}
        assert "P2" in pids
        assert "P3" in pids
        assert "P1" not in pids


# ---------------------------------------------------------------------------
# Priority errors
# ---------------------------------------------------------------------------


class TestValidateProcessesPriority:
    """Tests for priority validation rules."""

    def test_valid_priority_values(self) -> None:
        """Any integer priority is valid (positive, zero, negative)."""
        procs = [
            make_process("P1", priority=1),
            make_process("P2", priority=0),
            make_process("P3", priority=-1),
            make_process("P4", priority=100),
        ]
        result = validate_processes(procs)
        assert result.is_valid

    def test_float_priority_is_invalid(self) -> None:
        """Python dataclasses do not enforce field types at runtime.
        A Process can be constructed with float priority; validation must catch it."""
        proc = Process(pid="P1", arrival_time=0, burst_time=5, priority=2.5)  # type: ignore[arg-type]
        result = validate_processes([proc])
        assert not result.is_valid
        assert "priority" in error_fields(result)

    def test_float_arrival_time_is_invalid(self) -> None:
        """Float arrival_time must be rejected even if it looks like an integer."""
        proc = Process(pid="P1", arrival_time=1.0, burst_time=5, priority=1)  # type: ignore[arg-type]
        result = validate_processes([proc])
        assert not result.is_valid
        assert "arrival_time" in error_fields(result)

    def test_float_burst_time_is_invalid(self) -> None:
        """Float burst_time must be rejected even if it looks like an integer."""
        proc = Process(pid="P1", arrival_time=0, burst_time=3.0, priority=1)  # type: ignore[arg-type]
        result = validate_processes([proc])
        assert not result.is_valid
        assert "burst_time" in error_fields(result)

    def test_priority_lower_means_higher_convention_accepted(self) -> None:
        """Explicitly verify the lower-value = higher-priority convention."""
        # Priority 1 = higher priority than priority 5
        procs = [
            make_process("P1", priority=1),
            make_process("P2", priority=5),
        ]
        result = validate_processes(procs)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestValidateProcessesDataIntegrity:
    """Validation must not modify, reorder, or discard input processes."""

    def test_original_list_not_mutated(self) -> None:
        procs = [make_process("P1"), make_process("P2", arrival_time=1)]
        original_pids = [p.pid for p in procs]
        validate_processes(procs)
        assert [p.pid for p in procs] == original_pids

    def test_original_process_fields_not_changed(self) -> None:
        proc = make_process("P1", arrival_time=3, burst_time=7, priority=2)
        validate_processes([proc])
        assert proc.pid == "P1"
        assert proc.arrival_time == 3
        assert proc.burst_time == 7
        assert proc.priority == 2

    def test_valid_processes_not_silently_discarded(self) -> None:
        procs = [make_process(f"P{i}", arrival_time=i) for i in range(1, 6)]
        result = validate_processes(procs)
        assert result.is_valid
        assert result.errors == []

    def test_invalid_input_not_silently_fixed(self) -> None:
        """Zero burst time must be reported, not silently changed to 1."""
        proc = Process(pid="P1", arrival_time=0, burst_time=0, priority=1)
        result = validate_processes([proc])
        assert not result.is_valid
        bt_errors = [e for e in result.errors if e.field == "burst_time"]
        assert bt_errors  # problem must be reported


# ---------------------------------------------------------------------------
# Multiple simultaneous errors
# ---------------------------------------------------------------------------


class TestValidateProcessesMultipleErrors:
    """One call can report multiple errors for multiple problems."""

    def test_multiple_field_errors_on_same_process_reported(self) -> None:
        """A process with both bad arrival_time and bad burst_time gets 2 errors."""
        proc = Process(pid="P1", arrival_time=-1, burst_time=0, priority=1)
        result = validate_processes([proc])
        assert not result.is_valid
        fields = error_fields(result)
        assert "arrival_time" in fields
        assert "burst_time" in fields

    def test_errors_from_multiple_processes_all_reported(self) -> None:
        procs = [
            make_process("P1", arrival_time=-1),  # bad arrival
            make_process("P2", burst_time=0),      # bad burst
        ]
        result = validate_processes(procs)
        assert not result.is_valid
        fields = error_fields(result)
        assert "arrival_time" in fields
        assert "burst_time" in fields

    def test_mixed_valid_invalid_processes(self) -> None:
        """The valid process should not generate errors; the invalid one should."""
        procs = [
            make_process("P1"),                   # valid
            make_process("P2", burst_time=-1),    # invalid
        ]
        result = validate_processes(procs)
        assert not result.is_valid
        bt_errors = [e for e in result.errors if e.field == "burst_time"]
        assert any(e.process_id == "P2" for e in bt_errors)


# ===========================================================================
# validate_round_robin_quantum
# ===========================================================================


class TestValidateRoundRobinQuantumValid:
    """Valid quantum values must produce is_valid=True with no errors."""

    def test_quantum_one_is_valid(self) -> None:
        result = validate_round_robin_quantum(1)
        assert result.is_valid
        assert result.errors == []

    def test_quantum_small_positive_is_valid(self) -> None:
        result = validate_round_robin_quantum(4)
        assert result.is_valid

    def test_quantum_large_positive_is_valid(self) -> None:
        result = validate_round_robin_quantum(1000)
        assert result.is_valid

    def test_quantum_bool_result_true(self) -> None:
        result = validate_round_robin_quantum(3)
        assert bool(result) is True


class TestValidateRoundRobinQuantumMissingNone:
    """None / missing quantum must be rejected."""

    def test_none_quantum_is_invalid(self) -> None:
        result = validate_round_robin_quantum(None)
        assert not result.is_valid
        assert "time_quantum" in error_fields(result)

    def test_none_error_message_mentions_required(self) -> None:
        result = validate_round_robin_quantum(None)
        assert any("required" in e.message.lower() for e in result.errors)


class TestValidateRoundRobinQuantumZeroNegative:
    """Zero and negative quantum values must be rejected."""

    def test_zero_quantum_is_invalid(self) -> None:
        result = validate_round_robin_quantum(0)
        assert not result.is_valid
        assert "time_quantum" in error_fields(result)

    def test_zero_quantum_error_message(self) -> None:
        result = validate_round_robin_quantum(0)
        qt_errors = [e for e in result.errors if e.field == "time_quantum"]
        assert qt_errors
        assert any("0" in e.message or "greater than" in e.message for e in qt_errors)

    def test_negative_one_quantum_is_invalid(self) -> None:
        result = validate_round_robin_quantum(-1)
        assert not result.is_valid

    def test_negative_large_quantum_is_invalid(self) -> None:
        result = validate_round_robin_quantum(-100)
        assert not result.is_valid

    def test_negative_quantum_error_field(self) -> None:
        result = validate_round_robin_quantum(-5)
        assert "time_quantum" in error_fields(result)


class TestValidateRoundRobinQuantumNonInteger:
    """Non-integer quantum values must be rejected."""

    def test_float_quantum_is_invalid(self) -> None:
        result = validate_round_robin_quantum(2.5)
        assert not result.is_valid
        assert "time_quantum" in error_fields(result)

    def test_float_quantum_that_looks_like_integer_is_invalid(self) -> None:
        """2.0 is a float and must be rejected even though it equals 2."""
        result = validate_round_robin_quantum(2.0)
        assert not result.is_valid

    def test_string_quantum_is_invalid(self) -> None:
        result = validate_round_robin_quantum("4")
        assert not result.is_valid
        assert "time_quantum" in error_fields(result)

    def test_string_quantum_error_message_mentions_type(self) -> None:
        result = validate_round_robin_quantum("4")
        qt_errors = [e for e in result.errors if e.field == "time_quantum"]
        assert qt_errors
        assert any(
            "integer" in e.message.lower() or "non-numeric" in e.message.lower()
            for e in qt_errors
        )

    def test_empty_string_quantum_is_invalid(self) -> None:
        result = validate_round_robin_quantum("")
        assert not result.is_valid

    def test_list_quantum_is_invalid(self) -> None:
        result = validate_round_robin_quantum([4])
        assert not result.is_valid

    def test_bool_quantum_is_invalid(self) -> None:
        """True and False are bools, not plain ints; must be rejected."""
        assert not validate_round_robin_quantum(True).is_valid
        assert not validate_round_robin_quantum(False).is_valid

    def test_none_is_distinct_from_zero(self) -> None:
        """None and 0 are both invalid but for different reasons."""
        none_result = validate_round_robin_quantum(None)
        zero_result = validate_round_robin_quantum(0)
        # Both invalid
        assert not none_result.is_valid
        assert not zero_result.is_valid
        # Messages should differ
        none_msg = none_result.errors[0].message
        zero_msg = zero_result.errors[0].message
        assert none_msg != zero_msg


# ---------------------------------------------------------------------------
# ValidationResult / ValidationError structure
# ---------------------------------------------------------------------------


class TestValidationResultStructure:
    """Verify the shape of the returned data structures."""

    def test_valid_result_has_empty_errors(self) -> None:
        result = validate_processes([make_process("P1")])
        assert result.errors == []

    def test_invalid_result_errors_are_validation_error_instances(self) -> None:
        result = validate_processes([])
        assert all(isinstance(e, ValidationError) for e in result.errors)

    def test_validation_error_has_field(self) -> None:
        result = validate_processes([])
        assert hasattr(result.errors[0], "field")

    def test_validation_error_has_message(self) -> None:
        result = validate_processes([])
        assert hasattr(result.errors[0], "message")
        assert isinstance(result.errors[0].message, str)
        assert len(result.errors[0].message) > 0

    def test_validation_error_has_process_id(self) -> None:
        result = validate_processes([make_process("P1", burst_time=0)])
        bt_errors = [e for e in result.errors if e.field == "burst_time"]
        assert bt_errors
        assert hasattr(bt_errors[0], "process_id")

    def test_collection_level_error_process_id_is_none(self) -> None:
        """Errors not tied to a specific process have process_id=None."""
        result = validate_processes([])
        assert result.errors[0].process_id is None

    def test_quantum_error_process_id_is_none(self) -> None:
        result = validate_round_robin_quantum(0)
        assert result.errors[0].process_id is None
