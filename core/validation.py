"""Input validation layer for the CPU Scheduling Lab.

This module validates user-supplied scheduling inputs BEFORE they reach any
scheduling algorithm.  It is completely independent of Streamlit, the
scheduling algorithms, and any persistence/storage layer.

Design notes
------------
* :func:`validate_processes` checks a collection of :class:`~core.models.Process`
  objects and returns a :class:`ValidationResult`.
* :func:`validate_round_robin_quantum` checks a Round Robin time quantum value
  and returns a :class:`ValidationResult`.
* Both functions return the same :class:`ValidationResult` type so callers
  (e.g. Laptop C's Streamlit UI) can use a uniform pattern::

      result = validate_processes(processes)
      if not result.is_valid:
          for err in result.errors:
              display_error(err.message)

* Validation never silently modifies, corrects, or discards input data.
  All problems are reported to the caller for user-facing display.

* :class:`ValidationError` instances carry a ``field``, an optional
  ``process_id``, and a human-readable ``message``.  The UI can display
  friendly messages such as::

      "Process P2 has a burst time of 0.  Burst time must be greater than 0."

Priority convention (from DECISIONS.txt):
    LOWER numeric value = HIGHER priority.
    This convention is enforced here: priority must be a valid int (any integer
    is acceptable; negative priorities are technically permitted by the model).
    The validation layer does not impose an arbitrary lower/upper bound unless
    required by a future decision.

Time values (from core/models.py and DECISIONS.txt):
    arrival_time  >= 0  (int)
    burst_time    >  0  (int)
    time_quantum  >  0  (int, Round Robin only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from core.models import IDLE_PROCESS_ID, Process


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationError:
    """A single validation problem found in user input.

    Attributes
    ----------
    field:
        Name of the field or input that is invalid (e.g. ``\"burst_time\"``).
    message:
        Human-readable description of the problem, suitable for display
        directly in a UI without modification.
    process_id:
        Optional identifier of the process that caused this error.
        ``None`` when the error is not attributable to a specific process
        (e.g. an empty process list).
    """

    field: str
    message: str
    process_id: str | None = None


@dataclass
class ValidationResult:
    """Outcome of a validation call.

    Attributes
    ----------
    is_valid:
        ``True`` when no errors were found; ``False`` otherwise.
    errors:
        Ordered list of :class:`ValidationError` objects describing every
        problem that was detected.  Empty when ``is_valid`` is ``True``.
    """

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    def __bool__(self) -> bool:  # convenience: ``if result:``
        return self.is_valid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    """Return True if *value* is a plain Python ``int`` (not bool, float, etc.).

    ``bool`` is a subclass of ``int`` in Python, so we explicitly reject it
    because ``True`` / ``False`` are not valid process field values.
    """
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Public validation functions
# ---------------------------------------------------------------------------


def validate_processes(processes: Any) -> ValidationResult:
    """Validate a collection of :class:`~core.models.Process` objects.

    This function checks every process in *processes* against the project's
    input rules and returns a :class:`ValidationResult` describing all
    problems found.  It never raises; callers must inspect ``result.is_valid``
    and ``result.errors``.

    Parameters
    ----------
    processes:
        The collection of processes to validate.  Typically a ``list`` or
        ``tuple`` of :class:`~core.models.Process` objects, but any value is
        accepted so that type-level errors are reported uniformly.

    Returns
    -------
    ValidationResult
        ``is_valid=True`` with an empty error list when all inputs are valid.
        ``is_valid=False`` with one or more :class:`ValidationError` entries
        otherwise.

    Validation rules (from DECISIONS.txt and core/models.py)
    ---------------------------------------------------------
    1. The collection must be non-empty.
    2. Each element must be a :class:`~core.models.Process` instance.
    3. ``pid`` must be a non-empty string and must not equal ``IDLE_PROCESS_ID``
       (``"IDLE"`` is reserved for Gantt idle segments).
    4. All ``pid`` values must be unique across the collection.
    5. ``arrival_time`` must be an ``int`` >= 0.
    6. ``burst_time`` must be an ``int`` > 0.
    7. ``priority`` must be an ``int`` (any integer; lower = higher priority).

    Notes
    -----
    * Processes do **not** need to be sorted by arrival time.
    * Validation does **not** modify, reorder, or discard any process.
    """
    errors: list[ValidationError] = []

    # --- Rule 1: non-empty collection ---
    if not isinstance(processes, (list, tuple, Sequence)) or isinstance(
        processes, (str, bytes)
    ):
        errors.append(
            ValidationError(
                field="processes",
                message=(
                    "Process list must be a sequence (list or tuple), "
                    f"got {type(processes).__name__!r}."
                ),
            )
        )
        return ValidationResult(is_valid=False, errors=errors)

    # Convert to list for uniform handling (does not mutate caller's object)
    proc_list = list(processes)

    if len(proc_list) == 0:
        errors.append(
            ValidationError(
                field="processes",
                message="Process list must not be empty.  Add at least one process.",
            )
        )
        return ValidationResult(is_valid=False, errors=errors)

    # --- Per-element checks ---
    seen_pids: dict[str, int] = {}  # pid -> first occurrence index (0-based)

    for idx, proc in enumerate(proc_list):
        # Rule 2: each element must be a Process instance
        if not isinstance(proc, Process):
            errors.append(
                ValidationError(
                    field="processes",
                    message=(
                        f"Item at index {idx} is not a Process object "
                        f"(got {type(proc).__name__!r})."
                    ),
                )
            )
            # Cannot check individual fields; move to the next element.
            continue

        pid_label = f"Process {proc.pid!r}"

        # Rule 3a: pid must be a non-empty string
        if not isinstance(proc.pid, str) or proc.pid.strip() == "":
            errors.append(
                ValidationError(
                    field="pid",
                    process_id=None,
                    message=(
                        f"Item at index {idx} has an invalid PID {proc.pid!r}.  "
                        "PID must be a non-empty string."
                    ),
                )
            )
        # Rule 3b: pid must not be the reserved IDLE token
        elif proc.pid == IDLE_PROCESS_ID:
            errors.append(
                ValidationError(
                    field="pid",
                    process_id=proc.pid,
                    message=(
                        f"PID {proc.pid!r} is reserved for CPU idle segments "
                        "and cannot be used as a process identifier."
                    ),
                )
            )
        else:
            # Rule 4: pid must be unique
            if proc.pid in seen_pids:
                errors.append(
                    ValidationError(
                        field="pid",
                        process_id=proc.pid,
                        message=(
                            f"Duplicate PID {proc.pid!r} found at index {idx}.  "
                            f"PID first appeared at index {seen_pids[proc.pid]}.  "
                            "Each process must have a unique PID."
                        ),
                    )
                )
            else:
                seen_pids[proc.pid] = idx

        # Rule 5: arrival_time must be int >= 0
        if not _is_int(proc.arrival_time):
            errors.append(
                ValidationError(
                    field="arrival_time",
                    process_id=proc.pid if isinstance(proc.pid, str) else None,
                    message=(
                        f"{pid_label} has an arrival time of {proc.arrival_time!r}.  "
                        "Arrival time must be an integer."
                    ),
                )
            )
        elif proc.arrival_time < 0:
            errors.append(
                ValidationError(
                    field="arrival_time",
                    process_id=proc.pid if isinstance(proc.pid, str) else None,
                    message=(
                        f"{pid_label} has an arrival time of {proc.arrival_time}.  "
                        "Arrival time must be >= 0."
                    ),
                )
            )

        # Rule 6: burst_time must be int > 0
        if not _is_int(proc.burst_time):
            errors.append(
                ValidationError(
                    field="burst_time",
                    process_id=proc.pid if isinstance(proc.pid, str) else None,
                    message=(
                        f"{pid_label} has a burst time of {proc.burst_time!r}.  "
                        "Burst time must be an integer."
                    ),
                )
            )
        elif proc.burst_time <= 0:
            errors.append(
                ValidationError(
                    field="burst_time",
                    process_id=proc.pid if isinstance(proc.pid, str) else None,
                    message=(
                        f"{pid_label} has a burst time of {proc.burst_time}.  "
                        "Burst time must be greater than 0."
                    ),
                )
            )

        # Rule 7: priority must be int (any integer value; lower = higher priority)
        if not _is_int(proc.priority):
            errors.append(
                ValidationError(
                    field="priority",
                    process_id=proc.pid if isinstance(proc.pid, str) else None,
                    message=(
                        f"{pid_label} has a priority of {proc.priority!r}.  "
                        "Priority must be an integer "
                        "(lower numeric value = higher priority)."
                    ),
                )
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_round_robin_quantum(quantum: Any) -> ValidationResult:
    """Validate a Round Robin time quantum value.

    Parameters
    ----------
    quantum:
        The time quantum to validate.  Any Python value is accepted so that
        type-level problems are reported uniformly.

    Returns
    -------
    ValidationResult
        ``is_valid=True`` when *quantum* is a positive integer.
        ``is_valid=False`` with one or more :class:`ValidationError` entries
        otherwise.

    Validation rules (from DECISIONS.txt)
    --------------------------------------
    * quantum must be provided (not ``None``).
    * quantum must be a plain integer (not ``bool``, not ``float``).
    * quantum must be > 0.
    """
    errors: list[ValidationError] = []

    if quantum is None:
        errors.append(
            ValidationError(
                field="time_quantum",
                message=(
                    "A time quantum is required for Round Robin scheduling.  "
                    "Please provide a positive integer."
                ),
            )
        )
        return ValidationResult(is_valid=False, errors=errors)

    if not _is_int(quantum):
        errors.append(
            ValidationError(
                field="time_quantum",
                message=(
                    f"Time quantum must be a positive integer, got {quantum!r} "
                    f"({type(quantum).__name__}).  "
                    "Decimal values and non-numeric input are not accepted."
                ),
            )
        )
        return ValidationResult(is_valid=False, errors=errors)

    # quantum is a plain int here
    if quantum <= 0:
        errors.append(
            ValidationError(
                field="time_quantum",
                message=(
                    f"Time quantum must be greater than 0, got {quantum}.  "
                    "Round Robin requires a positive time quantum."
                ),
            )
        )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
