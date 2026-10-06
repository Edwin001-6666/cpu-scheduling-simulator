"""CSV import layer for the CPU Scheduling Lab.

This module parses CSV text (or any file-like text source) into a list of
:class:`~core.models.Process` objects.  It is the first step in the full
pipeline:

    CSV text / file-like
        ↓
    parse_processes_csv()          ← this module (B3)
        ↓
    validate_processes()           ← core/validation.py (B1)
        ↓
    algorithm.schedule()           ← algorithms/ (Laptop A)
        ↓
    compute_metrics()              ← core/metrics.py (B2)

Design principles
-----------------
* **Parsing and semantic validation are separate.**
  This module converts CSV text → ``Process`` objects.  It does *not* call
  :func:`~core.validation.validate_processes` internally.  Callers are
  expected to run validation after parsing.  This keeps responsibilities
  cleanly separated and makes each layer independently testable.

* **In-memory only.**  The function accepts a ``str`` (CSV text) or any
  file-like object with a ``read()`` or iteration interface.  No file paths
  are opened; no data is written to disk.

* **No Streamlit coupling.**  Pure Python standard library only.

* **Clear, structured errors.**  Parsing problems (wrong columns, unconvertible
  numeric fields, malformed CSV) are collected into a
  :class:`CsvParseResult` and never raised as raw exceptions, so the UI can
  display them gracefully.

Supported CSV format
--------------------
The canonical column header row (case-sensitive, order-independent):

    PID,Arrival Time,Burst Time,Priority

* Column order does not matter; columns are matched by name.
* Extra columns beyond the four required ones are **silently ignored**.
  This allows CSV files with additional metadata columns to be parsed
  without error.
* Blank lines (lines that are empty or contain only whitespace) are skipped.
* Quoted fields are handled by the standard :mod:`csv` module (RFC 4180).

Numeric conversion
------------------
``Arrival Time``, ``Burst Time``, and ``Priority`` must be convertible to
``int`` via ``int(value.strip())``.  Decimal strings (e.g. ``"3.5"``) and
non-numeric strings (e.g. ``"abc"``) are parse errors.  The parsed value is
stored as-is; whether it is in-range (e.g. burst > 0) is *not* checked here —
that is :func:`~core.validation.validate_processes`'s responsibility.

Negative integers (e.g. ``"-1"``) *are* parsed successfully and passed
through so that B1 validation can produce the appropriate user-facing message.

PID handling
------------
``PID`` is kept as a stripped ``str``.  Whether it is empty, reserved, or
duplicate is left to B1 validation.

Extra column behaviour (documented decision)
--------------------------------------------
Extra columns are silently ignored.  The :class:`CsvParseResult` documents
the names of any extra columns found via :attr:`CsvParseResult.extra_columns`
so callers/UI can optionally surface a warning.

Public API
----------
::

    result = parse_processes_csv(csv_source)
    if not result.is_valid:
        for err in result.errors:
            print(err.message)
    else:
        processes = result.processes
        validation_result = validate_processes(processes)
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import IO, Union

from core.models import Process


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

# Type alias: accept either a str or a text file-like object.
CsvSource = Union[str, IO[str]]

# The canonical column names the parser requires.
_REQUIRED_COLUMNS = ("PID", "Arrival Time", "Burst Time", "Priority")


@dataclass(frozen=True)
class CsvParseError:
    """A single problem encountered while parsing the CSV.

    Attributes
    ----------
    row:
        1-based data row number (not counting the header).  ``None`` when
        the error is structural (e.g. missing column, empty input).
    field:
        Name of the CSV column that caused the problem, or ``"csv"`` for
        structural errors.
    message:
        Human-readable description suitable for direct UI display.
    """

    field: str
    message: str
    row: int | None = None


@dataclass
class CsvParseResult:
    """Outcome of a :func:`parse_processes_csv` call.

    Attributes
    ----------
    is_valid:
        ``True`` when parsing succeeded and :attr:`processes` is populated.
        ``False`` when one or more :attr:`errors` were found.
    processes:
        Ordered list of :class:`~core.models.Process` objects produced from
        the CSV.  Empty when ``is_valid`` is ``False``.
    errors:
        Ordered list of :class:`CsvParseError` objects.  Empty when
        ``is_valid`` is ``True``.
    extra_columns:
        Column names present in the CSV beyond the four required ones.
        May be non-empty even when ``is_valid`` is ``True`` (extra columns
        are silently ignored but surfaced here for optional UI warnings).
    """

    is_valid: bool
    processes: list[Process] = field(default_factory=list)
    errors: list[CsvParseError] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def parse_processes_csv(csv_source: CsvSource) -> CsvParseResult:
    """Parse *csv_source* into a list of :class:`~core.models.Process` objects.

    Parameters
    ----------
    csv_source:
        Either a ``str`` containing CSV text or a file-like text object
        (anything with an ``__iter__`` or ``read()`` method, such as the
        result of ``open(...)``, ``io.StringIO``, or Streamlit's
        ``UploadedFile``).

    Returns
    -------
    CsvParseResult
        ``is_valid=True`` with a populated ``processes`` list when parsing
        succeeds.  ``is_valid=False`` with one or more ``errors`` when
        structural or conversion problems are found.

    Notes
    -----
    * Parsing and semantic validation are **separate**.  After a successful
      parse, call :func:`~core.validation.validate_processes` on the
      returned ``processes`` list to check business rules (arrival_time >= 0,
      burst_time > 0, unique PIDs, etc.).

    * Extra CSV columns beyond the four required ones are silently ignored;
      their names are surfaced in ``CsvParseResult.extra_columns``.

    * Blank lines are silently skipped.

    * Quoted CSV fields (RFC 4180) are supported via the standard
      :mod:`csv` module.
    """
    errors: list[CsvParseError] = []
    processes: list[Process] = []
    extra_columns: list[str] = []

    # --- Normalise input to a line iterator ---------------------------------
    if isinstance(csv_source, str):
        lines: IO[str] = io.StringIO(csv_source)
    else:
        lines = csv_source  # file-like object

    # --- Read with the standard csv module ----------------------------------
    try:
        reader = csv.DictReader(lines)

        # Attempt to read the fieldnames (triggers header parse).
        # DictReader.fieldnames is None before any row is consumed and when
        # the source is completely empty.
        header = reader.fieldnames
    except csv.Error as exc:
        errors.append(
            CsvParseError(
                field="csv",
                message=f"CSV parse error in header: {exc}",
            )
        )
        return CsvParseResult(is_valid=False, errors=errors)

    # --- Handle completely empty input --------------------------------------
    if header is None:
        errors.append(
            CsvParseError(
                field="csv",
                message="CSV input is empty.  Expected a header row followed by at least one data row.",
            )
        )
        return CsvParseResult(is_valid=False, errors=errors)

    # Strip whitespace from header names (handles accidental spaces).
    stripped_header = [col.strip() for col in header]

    # --- Check required columns --------------------------------------------
    missing = [col for col in _REQUIRED_COLUMNS if col not in stripped_header]
    if missing:
        errors.append(
            CsvParseError(
                field="csv",
                message=(
                    f"Missing required column(s): {', '.join(missing)}.  "
                    f"Expected columns: {', '.join(_REQUIRED_COLUMNS)}."
                ),
            )
        )
        return CsvParseResult(is_valid=False, errors=errors)

    # --- Identify extra columns (informational only) -----------------------
    extra_columns = [
        col for col in stripped_header if col not in _REQUIRED_COLUMNS
    ]

    # --- Parse data rows ----------------------------------------------------
    data_row_count = 0
    try:
        for raw_row in reader:
            # DictReader produces {original_header: value} dicts; remap with
            # stripped keys so field lookup is whitespace-tolerant.
            row: dict[str, str] = {
                k.strip(): (v.strip() if v is not None else "")
                for k, v in raw_row.items()
                if k is not None
            }

            # Skip blank lines (all values empty after stripping).
            if all(v == "" for v in row.values()):
                continue

            data_row_count += 1
            row_num = data_row_count  # 1-based data row index

            row_errors: list[CsvParseError] = []

            # PID — kept as str, just strip whitespace.
            pid: str = row.get("PID", "")

            # Arrival Time — must be convertible to int.
            arrival_time = _parse_int(row.get("Arrival Time", ""), "Arrival Time", row_num, row_errors)

            # Burst Time — must be convertible to int.
            burst_time = _parse_int(row.get("Burst Time", ""), "Burst Time", row_num, row_errors)

            # Priority — must be convertible to int.
            priority = _parse_int(row.get("Priority", ""), "Priority", row_num, row_errors)

            if row_errors:
                errors.extend(row_errors)
                # Continue to collect errors from remaining rows.
                continue

            # All three numeric fields parsed successfully.
            processes.append(
                Process(
                    pid=pid,
                    arrival_time=arrival_time,  # type: ignore[arg-type]
                    burst_time=burst_time,       # type: ignore[arg-type]
                    priority=priority,           # type: ignore[arg-type]
                )
            )

    except csv.Error as exc:
        errors.append(
            CsvParseError(
                field="csv",
                message=f"CSV parse error at data row {data_row_count + 1}: {exc}",
            )
        )
        return CsvParseResult(is_valid=False, errors=errors, extra_columns=extra_columns)

    # --- Handle header-only CSV (no data rows) ------------------------------
    if data_row_count == 0 and not errors:
        errors.append(
            CsvParseError(
                field="csv",
                message=(
                    "CSV contains a header but no process rows.  "
                    "Add at least one data row."
                ),
            )
        )

    is_valid = len(errors) == 0
    return CsvParseResult(
        is_valid=is_valid,
        processes=processes if is_valid else [],
        errors=errors,
        extra_columns=extra_columns,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_int(
    raw: str,
    field_name: str,
    row_num: int,
    errors: list[CsvParseError],
) -> int | None:
    """Attempt to parse *raw* as a plain integer.

    Appends a :class:`CsvParseError` to *errors* and returns ``None`` on
    failure.  On success returns the ``int`` value.

    Decimal strings (e.g. ``"3.5"``) are rejected; so are empty strings and
    non-numeric text.
    """
    value = raw.strip()
    if value == "":
        errors.append(
            CsvParseError(
                field=field_name,
                row=row_num,
                message=(
                    f"Row {row_num}: '{field_name}' is empty.  "
                    f"Expected an integer value."
                ),
            )
        )
        return None

    # Reject decimal strings before trying int() so "3.5" is a clear error.
    if "." in value:
        errors.append(
            CsvParseError(
                field=field_name,
                row=row_num,
                message=(
                    f"Row {row_num}: '{field_name}' value {value!r} is not an integer.  "
                    f"Decimal values are not accepted; use a whole number."
                ),
            )
        )
        return None

    try:
        return int(value)
    except ValueError:
        errors.append(
            CsvParseError(
                field=field_name,
                row=row_num,
                message=(
                    f"Row {row_num}: '{field_name}' value {value!r} cannot be "
                    f"converted to an integer."
                ),
            )
        )
        return None
