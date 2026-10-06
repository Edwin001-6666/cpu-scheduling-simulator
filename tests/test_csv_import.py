"""Comprehensive unit tests for the CSV import layer (core/csv_import.py).

Test organisation
-----------------
TestCsvParseValid
    Correct CSV → successful CsvParseResult with Process objects.

TestCsvParseErrors_Structure
    Empty input, header-only, missing columns, wrong column names.

TestCsvParseErrors_Numeric
    Non-integer text, decimal strings, empty numeric fields.

TestCsvParsePassThrough
    Values that are valid integers but semantically wrong (negative arrival,
    zero burst, duplicate PID) must be PARSED successfully; validation is
    left to B1 validate_processes().

TestCsvParseEdgeCases
    Blank lines, extra columns, quoted fields, file-like objects.

TestCsvParseResultStructure
    Shape/type guarantees on CsvParseResult and CsvParseError.

TestCsvParseIntegrationWithValidation
    Round-trip: parse → validate_processes() to confirm the two layers
    compose correctly without duplication of responsibility.
"""

from __future__ import annotations

import io

import pytest

from core.csv_import import CsvParseError, CsvParseResult, parse_processes_csv
from core.models import Process
from core.validation import validate_processes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_HEADER = "PID,Arrival Time,Burst Time,Priority\n"


def one_row(pid="P1", at=0, bt=5, prio=1) -> str:
    return VALID_HEADER + f"{pid},{at},{bt},{prio}\n"


def error_fields(result: CsvParseResult) -> list[str]:
    return [e.field for e in result.errors]


# ===========================================================================
# TestCsvParseValid
# ===========================================================================


class TestCsvParseValid:
    """Valid CSV inputs → is_valid=True, correct Process objects."""

    def test_single_process(self) -> None:
        result = parse_processes_csv(one_row("P1", 0, 5, 1))
        assert result.is_valid
        assert result.errors == []
        assert len(result.processes) == 1
        p = result.processes[0]
        assert p.pid == "P1"
        assert p.arrival_time == 0
        assert p.burst_time == 5
        assert p.priority == 1

    def test_multiple_processes(self) -> None:
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,0,5,2\n"
            "P2,2,3,1\n"
            "P3,4,7,3\n"
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert len(result.processes) == 3
        pids = [p.pid for p in result.processes]
        assert pids == ["P1", "P2", "P3"]

    def test_unsorted_arrival_times(self) -> None:
        """Arrival times do not need to be in order — parser preserves input order."""
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P3,10,2,1\n"
            "P1,0,5,2\n"
            "P2,5,3,3\n"
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert [p.pid for p in result.processes] == ["P3", "P1", "P2"]

    def test_numeric_conversion_int(self) -> None:
        """Numeric fields must be stored as int, not str."""
        result = parse_processes_csv(one_row("P1", 3, 7, 2))
        assert result.is_valid
        p = result.processes[0]
        assert isinstance(p.arrival_time, int)
        assert isinstance(p.burst_time, int)
        assert isinstance(p.priority, int)

    def test_arrival_time_zero(self) -> None:
        result = parse_processes_csv(one_row("P1", 0, 5, 1))
        assert result.is_valid
        assert result.processes[0].arrival_time == 0

    def test_large_values_parsed(self) -> None:
        result = parse_processes_csv(one_row("P1", 1000, 9999, 100))
        assert result.is_valid
        p = result.processes[0]
        assert p.arrival_time == 1000
        assert p.burst_time == 9999
        assert p.priority == 100

    def test_column_order_independent(self) -> None:
        """Columns may appear in any order."""
        csv = (
            "Priority,Burst Time,PID,Arrival Time\n"
            "2,5,P1,0\n"
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        p = result.processes[0]
        assert p.pid == "P1"
        assert p.arrival_time == 0
        assert p.burst_time == 5
        assert p.priority == 2

    def test_trailing_newline_handled(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,5,1\n\n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert len(result.processes) == 1

    def test_bool_result_true_when_valid(self) -> None:
        result = parse_processes_csv(one_row())
        assert bool(result) is True

    def test_processes_are_process_instances(self) -> None:
        result = parse_processes_csv(one_row("P1"))
        assert all(isinstance(p, Process) for p in result.processes)

    def test_whitespace_around_values_stripped(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\n P1 , 0 , 5 , 1 \n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        p = result.processes[0]
        assert p.pid == "P1"
        assert p.arrival_time == 0

    def test_whitespace_around_header_stripped(self) -> None:
        csv = " PID , Arrival Time , Burst Time , Priority \nP1,0,5,1\n"
        result = parse_processes_csv(csv)
        assert result.is_valid


# ===========================================================================
# TestCsvParseErrors_Structure
# ===========================================================================


class TestCsvParseErrors_Structure:
    """Structural CSV problems."""

    def test_empty_string_is_invalid(self) -> None:
        result = parse_processes_csv("")
        assert not result.is_valid
        assert len(result.errors) >= 1
        assert "csv" in error_fields(result)

    def test_empty_string_friendly_message(self) -> None:
        result = parse_processes_csv("")
        assert any("empty" in e.message.lower() for e in result.errors)

    def test_header_only_is_invalid(self) -> None:
        result = parse_processes_csv(VALID_HEADER)
        assert not result.is_valid
        assert "csv" in error_fields(result)

    def test_header_only_friendly_message(self) -> None:
        result = parse_processes_csv(VALID_HEADER)
        assert any("header" in e.message.lower() or "row" in e.message.lower()
                   for e in result.errors)

    def test_missing_pid_column(self) -> None:
        csv = "Arrival Time,Burst Time,Priority\n0,5,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any("PID" in e.message for e in result.errors)

    def test_missing_arrival_time_column(self) -> None:
        csv = "PID,Burst Time,Priority\nP1,5,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any("Arrival Time" in e.message for e in result.errors)

    def test_missing_burst_time_column(self) -> None:
        csv = "PID,Arrival Time,Priority\nP1,0,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any("Burst Time" in e.message for e in result.errors)

    def test_missing_priority_column(self) -> None:
        csv = "PID,Arrival Time,Burst Time\nP1,0,5\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any("Priority" in e.message for e in result.errors)

    def test_wrong_column_names(self) -> None:
        """Completely different header → all required columns missing."""
        csv = "Name,Start,Length,Level\nP1,0,5,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid

    def test_wrong_column_names_message_mentions_expected(self) -> None:
        csv = "Name,Start,Length,Level\nP1,0,5,1\n"
        result = parse_processes_csv(csv)
        msgs = " ".join(e.message for e in result.errors)
        assert "PID" in msgs or "Arrival Time" in msgs

    def test_partial_columns_lists_all_missing(self) -> None:
        """Only one column present → message mentions all three missing ones."""
        csv = "PID\nP1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        msg = result.errors[0].message
        assert "Arrival Time" in msg
        assert "Burst Time" in msg
        assert "Priority" in msg

    def test_bool_result_false_when_invalid(self) -> None:
        result = parse_processes_csv("")
        assert bool(result) is False

    def test_invalid_result_has_empty_processes(self) -> None:
        result = parse_processes_csv("")
        assert result.processes == []

    def test_whitespace_only_csv_is_invalid(self) -> None:
        result = parse_processes_csv("   \n   \n")
        assert not result.is_valid


# ===========================================================================
# TestCsvParseErrors_Numeric
# ===========================================================================


class TestCsvParseErrors_Numeric:
    """Non-convertible numeric fields."""

    def test_arrival_time_non_numeric(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,abc,5,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Arrival Time" for e in result.errors)

    def test_burst_time_non_numeric(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,xyz,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Burst Time" for e in result.errors)

    def test_priority_non_numeric(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,5,high\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Priority" for e in result.errors)

    def test_decimal_arrival_time_rejected(self) -> None:
        """3.5 is not an integer — must be a parse error."""
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,3.5,5,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Arrival Time" for e in result.errors)

    def test_decimal_burst_time_rejected(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,2.5,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Burst Time" for e in result.errors)

    def test_decimal_priority_rejected(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,5,1.5\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Priority" for e in result.errors)

    def test_empty_arrival_time_field(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,,5,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Arrival Time" for e in result.errors)

    def test_empty_burst_time_field(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,,1\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Burst Time" for e in result.errors)

    def test_empty_priority_field(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,5,\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any(e.field == "Priority" for e in result.errors)

    def test_numeric_error_reports_row_number(self) -> None:
        """CsvParseError.row must be set for data-row errors."""
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,abc,5,1\n"
        result = parse_processes_csv(csv)
        at_errors = [e for e in result.errors if e.field == "Arrival Time"]
        assert at_errors
        assert at_errors[0].row == 1

    def test_second_row_error_has_correct_row_number(self) -> None:
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,0,5,1\n"
            "P2,bad,3,2\n"
        )
        result = parse_processes_csv(csv)
        at_errors = [e for e in result.errors if e.field == "Arrival Time"]
        assert at_errors
        assert at_errors[0].row == 2

    def test_multiple_rows_collect_all_errors(self) -> None:
        """Errors from multiple rows are all collected, not stopped at first."""
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,abc,5,1\n"
            "P2,0,xyz,1\n"
        )
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert len(result.errors) >= 2

    def test_numeric_error_message_includes_field_name(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,abc,5,1\n"
        result = parse_processes_csv(csv)
        at_errors = [e for e in result.errors if e.field == "Arrival Time"]
        assert any("Arrival Time" in e.message for e in at_errors)

    def test_decimal_error_message_mentions_integer(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,3.5,5,1\n"
        result = parse_processes_csv(csv)
        at_errors = [e for e in result.errors if e.field == "Arrival Time"]
        assert any("integer" in e.message.lower() for e in at_errors)


# ===========================================================================
# TestCsvParsePassThrough
# ===========================================================================


class TestCsvParsePassThrough:
    """Values that are integers but semantically invalid must pass through
    the CSV parser successfully — semantic validation belongs to B1."""

    def test_negative_arrival_time_parsed(self) -> None:
        """arrival_time=-1 is a valid int; parser should not reject it."""
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,-1,5,1\n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert result.processes[0].arrival_time == -1

    def test_negative_burst_time_parsed(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,-5,1\n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert result.processes[0].burst_time == -5

    def test_zero_burst_time_parsed(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,0,1\n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert result.processes[0].burst_time == 0

    def test_negative_priority_parsed(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,5,-3\n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert result.processes[0].priority == -3

    def test_duplicate_pid_parsed(self) -> None:
        """Duplicate PIDs are a semantic problem — parser should not reject."""
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,0,5,1\n"
            "P1,2,3,2\n"
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert len(result.processes) == 2
        assert all(p.pid == "P1" for p in result.processes)

    def test_empty_pid_parsed(self) -> None:
        """Empty PID string is a semantic problem — parser does not reject."""
        csv = "PID,Arrival Time,Burst Time,Priority\n,0,5,1\n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert result.processes[0].pid == ""

    def test_negative_values_fail_b1_validation(self) -> None:
        """After parse, B1 validation correctly catches negative arrival."""
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,-1,5,1\n"
        parse_result = parse_processes_csv(csv)
        assert parse_result.is_valid   # parser accepts it
        val_result = validate_processes(parse_result.processes)
        assert not val_result.is_valid  # validation rejects it

    def test_zero_burst_fails_b1_validation(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,0,0,1\n"
        parse_result = parse_processes_csv(csv)
        assert parse_result.is_valid
        val_result = validate_processes(parse_result.processes)
        assert not val_result.is_valid

    def test_duplicate_pid_fails_b1_validation(self) -> None:
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,0,5,1\n"
            "P1,2,3,2\n"
        )
        parse_result = parse_processes_csv(csv)
        assert parse_result.is_valid
        val_result = validate_processes(parse_result.processes)
        assert not val_result.is_valid


# ===========================================================================
# TestCsvParseEdgeCases
# ===========================================================================


class TestCsvParseEdgeCases:
    """Blank lines, extra columns, quoted fields, file-like objects."""

    def test_blank_lines_between_rows_skipped(self) -> None:
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,0,5,1\n"
            "\n"
            "P2,2,3,2\n"
            "\n"
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert len(result.processes) == 2

    def test_blank_lines_only_with_header(self) -> None:
        """Header + only blank lines → header-only error."""
        csv = "PID,Arrival Time,Burst Time,Priority\n\n\n"
        result = parse_processes_csv(csv)
        assert not result.is_valid
        assert any("header" in e.message.lower() or "row" in e.message.lower()
                   for e in result.errors)

    def test_extra_columns_silently_ignored(self) -> None:
        csv = (
            "PID,Arrival Time,Burst Time,Priority,Notes,Department\n"
            "P1,0,5,1,some note,CS\n"
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert len(result.processes) == 1
        # Process must only have the four standard fields
        p = result.processes[0]
        assert p.pid == "P1"
        assert p.arrival_time == 0
        assert p.burst_time == 5
        assert p.priority == 1

    def test_extra_columns_surfaced_in_result(self) -> None:
        csv = (
            "PID,Arrival Time,Burst Time,Priority,Notes\n"
            "P1,0,5,1,test\n"
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert "Notes" in result.extra_columns

    def test_no_extra_columns_when_only_required(self) -> None:
        result = parse_processes_csv(one_row())
        assert result.extra_columns == []

    def test_quoted_fields_handled(self) -> None:
        """Quoted PID and numeric fields are handled by stdlib csv module."""
        csv = (
            'PID,Arrival Time,Burst Time,Priority\n'
            '"P1",0,5,1\n'
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert result.processes[0].pid == "P1"

    def test_quoted_numeric_fields(self) -> None:
        csv = (
            'PID,Arrival Time,Burst Time,Priority\n'
            'P1,"0","5","1"\n'
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        p = result.processes[0]
        assert p.arrival_time == 0
        assert p.burst_time == 5
        assert p.priority == 1

    def test_file_like_stringio_accepted(self) -> None:
        """parse_processes_csv accepts an io.StringIO file-like object."""
        csv_text = "PID,Arrival Time,Burst Time,Priority\nP1,0,5,1\n"
        file_like = io.StringIO(csv_text)
        result = parse_processes_csv(file_like)
        assert result.is_valid
        assert result.processes[0].pid == "P1"

    def test_file_like_multiple_processes(self) -> None:
        csv_text = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,0,5,1\n"
            "P2,2,3,2\n"
        )
        result = parse_processes_csv(io.StringIO(csv_text))
        assert result.is_valid
        assert len(result.processes) == 2

    def test_input_order_preserved(self) -> None:
        """Parser must not sort or reorder processes."""
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "PZ,10,1,1\n"
            "PA,0,5,2\n"
            "PM,5,3,3\n"
        )
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert [p.pid for p in result.processes] == ["PZ", "PA", "PM"]

    def test_windows_line_endings(self) -> None:
        """CRLF line endings are handled by stdlib csv."""
        csv = "PID,Arrival Time,Burst Time,Priority\r\nP1,0,5,1\r\n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        assert result.processes[0].pid == "P1"


# ===========================================================================
# TestCsvParseResultStructure
# ===========================================================================


class TestCsvParseResultStructure:
    """Shape/type guarantees on CsvParseResult and CsvParseError."""

    def test_valid_result_has_empty_errors(self) -> None:
        result = parse_processes_csv(one_row())
        assert result.errors == []

    def test_invalid_result_errors_are_csv_parse_error_instances(self) -> None:
        result = parse_processes_csv("")
        assert all(isinstance(e, CsvParseError) for e in result.errors)

    def test_csv_parse_error_has_field(self) -> None:
        result = parse_processes_csv("")
        assert hasattr(result.errors[0], "field")
        assert isinstance(result.errors[0].field, str)

    def test_csv_parse_error_has_message(self) -> None:
        result = parse_processes_csv("")
        assert hasattr(result.errors[0], "message")
        assert isinstance(result.errors[0].message, str)
        assert len(result.errors[0].message) > 0

    def test_csv_parse_error_has_row(self) -> None:
        result = parse_processes_csv("")
        assert hasattr(result.errors[0], "row")

    def test_structural_error_row_is_none(self) -> None:
        """Errors not tied to a specific data row have row=None."""
        result = parse_processes_csv("")
        assert result.errors[0].row is None

    def test_data_row_error_row_is_int(self) -> None:
        csv = "PID,Arrival Time,Burst Time,Priority\nP1,bad,5,1\n"
        result = parse_processes_csv(csv)
        at_errors = [e for e in result.errors if e.field == "Arrival Time"]
        assert isinstance(at_errors[0].row, int)

    def test_extra_columns_is_list(self) -> None:
        result = parse_processes_csv(one_row())
        assert isinstance(result.extra_columns, list)

    def test_processes_is_list(self) -> None:
        result = parse_processes_csv(one_row())
        assert isinstance(result.processes, list)


# ===========================================================================
# TestCsvParseIntegrationWithValidation
# ===========================================================================


class TestCsvParseIntegrationWithValidation:
    """Full pipeline: parse_processes_csv → validate_processes."""

    def test_valid_csv_passes_validation(self) -> None:
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,0,5,2\n"
            "P2,2,3,1\n"
        )
        parse_result = parse_processes_csv(csv)
        assert parse_result.is_valid
        val_result = validate_processes(parse_result.processes)
        assert val_result.is_valid

    def test_parse_failure_prevents_validation(self) -> None:
        """When CSV parse fails, processes list is empty — validation will
        report empty list error, not a crash."""
        parse_result = parse_processes_csv("")
        assert not parse_result.is_valid
        # processes is empty
        val_result = validate_processes(parse_result.processes)
        assert not val_result.is_valid

    def test_unsorted_processes_valid_after_parse(self) -> None:
        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P3,10,2,1\n"
            "P1,0,5,2\n"
            "P2,5,3,3\n"
        )
        parse_result = parse_processes_csv(csv)
        assert parse_result.is_valid
        val_result = validate_processes(parse_result.processes)
        assert val_result.is_valid

    def test_parser_does_not_mutate_process_fields(self) -> None:
        """Parsed Process objects are identical to what was in the CSV."""
        csv = "PID,Arrival Time,Burst Time,Priority\nP42,7,13,5\n"
        result = parse_processes_csv(csv)
        assert result.is_valid
        p = result.processes[0]
        assert p.pid == "P42"
        assert p.arrival_time == 7
        assert p.burst_time == 13
        assert p.priority == 5

    def test_full_pipeline_round_trip(self) -> None:
        """parse → validate → schedule → compute_metrics all succeed."""
        from algorithms.fcfs import FCFS
        from core.metrics import compute_metrics

        csv = (
            "PID,Arrival Time,Burst Time,Priority\n"
            "P1,0,5,1\n"
            "P2,2,3,2\n"
        )
        parse_result = parse_processes_csv(csv)
        assert parse_result.is_valid

        val_result = validate_processes(parse_result.processes)
        assert val_result.is_valid

        result = FCFS().schedule(parse_result.processes)
        enriched = compute_metrics(result, parse_result.processes)

        assert enriched.cpu_utilization is not None
        assert enriched.throughput is not None
        for m in enriched.process_metrics:
            assert m.response_time is not None
