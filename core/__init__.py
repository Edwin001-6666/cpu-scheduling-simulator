"""Shared core package for the CPU Scheduling Lab.

This package holds UI-independent data contracts used by algorithms,
validation, metrics, visualization, and tests.
"""

from core.models import (
    IDLE_PROCESS_ID,
    ExecutionSegment,
    Process,
    ProcessMetrics,
    SimulationResult,
)
from core.validation import (
    ValidationError,
    ValidationResult,
    validate_processes,
    validate_round_robin_quantum,
)
from core.metrics import compute_metrics
from core.csv_import import (
    CsvParseError,
    CsvParseResult,
    parse_processes_csv,
)

__all__ = [
    "IDLE_PROCESS_ID",
    "ExecutionSegment",
    "Process",
    "ProcessMetrics",
    "SimulationResult",
    "ValidationError",
    "ValidationResult",
    "validate_processes",
    "validate_round_robin_quantum",
    "compute_metrics",
    "CsvParseError",
    "CsvParseResult",
    "parse_processes_csv",
]
