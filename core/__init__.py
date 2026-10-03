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

__all__ = [
    "IDLE_PROCESS_ID",
    "ExecutionSegment",
    "Process",
    "ProcessMetrics",
    "SimulationResult",
]
