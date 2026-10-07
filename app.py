"""CPU Scheduling Simulator — Streamlit Application (C1 Foundation).

This is the entry point for the web-based CPU scheduling simulator.
Run with:  ``streamlit run app.py``

Architecture
------------
The UI is a thin presentation layer that delegates all computation to the
existing backend modules:

    core.csv_import   — CSV text → Process objects    (B3)
    core.validation   — semantic input validation      (B1)
    core.metrics      — response time, utilization     (B2)
    algorithms.*      — FCFS, SJF, Priority, RR       (Laptop A)

No scheduling, validation, or metric logic is duplicated here.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Backend imports (Laptop A + Laptop B)
# ---------------------------------------------------------------------------
from core import (
    Process,
    SimulationResult,
    parse_processes_csv,
    validate_processes,
    validate_round_robin_quantum,
    compute_metrics,
)
from algorithms.fcfs import FCFS
from algorithms.sjf import SJF
from algorithms.priority import PriorityScheduler
from algorithms.round_robin import RoundRobin


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHMS: dict[str, type] = {
    "FCFS": FCFS,
    "SJF": SJF,
    "Priority": PriorityScheduler,
    "Round Robin": RoundRobin,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_algorithm(name: str, quantum: int | None = None):
    """Instantiate a scheduling algorithm by display name.

    Parameters
    ----------
    name:
        One of the keys in ``ALGORITHMS``.
    quantum:
        Required when *name* is ``"Round Robin"``.

    Returns
    -------
    An instance of the selected :class:`~algorithms.base.SchedulingAlgorithm`
    subclass.
    """
    cls = ALGORITHMS[name]
    if name == "Round Robin":
        return cls(time_quantum=quantum)
    return cls()


def run_simulation(
    processes: list[Process],
    algorithm_name: str,
    quantum: int | None = None,
) -> SimulationResult:
    """Execute the full pipeline: validate → schedule → compute_metrics.

    Returns an enriched :class:`~core.models.SimulationResult` with all
    optional fields populated.

    Raises
    ------
    ValueError
        When validation fails (with a user-friendly message).
    """
    # --- Validate inputs ---------------------------------------------------
    val_result = validate_processes(processes)
    if not val_result.is_valid:
        messages = "\n".join(f"• {e.message}" for e in val_result.errors)
        raise ValueError(f"Input validation failed:\n{messages}")

    if algorithm_name == "Round Robin":
        q_result = validate_round_robin_quantum(quantum)
        if not q_result.is_valid:
            messages = "\n".join(f"• {e.message}" for e in q_result.errors)
            raise ValueError(f"Quantum validation failed:\n{messages}")

    # --- Schedule ----------------------------------------------------------
    algo = get_algorithm(algorithm_name, quantum)
    result = algo.schedule(processes)

    # --- Enrich with B2 metrics --------------------------------------------
    return compute_metrics(result, processes)


def display_results(result: SimulationResult) -> None:
    """Render a SimulationResult in Streamlit."""
    st.subheader(f"📊 Results — {result.algorithm_name}")

    # --- Aggregate metrics -------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Waiting", f"{result.average_waiting_time:.2f}")
    col2.metric("Avg Turnaround", f"{result.average_turnaround_time:.2f}")
    if result.average_response_time is not None:
        col3.metric("Avg Response", f"{result.average_response_time:.2f}")
    if result.cpu_utilization is not None:
        col4.metric("CPU Utilization", f"{result.cpu_utilization:.1f}%")

    if result.throughput is not None:
        st.caption(f"Throughput: {result.throughput:.4f} processes/time unit")

    # --- Per-process metrics table -----------------------------------------
    st.markdown("**Per-Process Metrics**")
    table_data = []
    for m in result.process_metrics:
        row = {
            "PID": m.pid,
            "Arrival": m.arrival_time,
            "Burst": m.burst_time,
            "Priority": m.priority,
            "Completion": m.completion_time,
            "Turnaround": m.turnaround_time,
            "Waiting": m.waiting_time,
        }
        if m.response_time is not None:
            row["Response"] = m.response_time
        table_data.append(row)
    st.table(table_data)

    # --- Execution timeline (text, placeholder for Gantt) ------------------
    st.markdown("**Execution Timeline** *(Gantt visualization coming in C2/C3)*")
    timeline_parts = []
    for seg in result.execution_segments:
        label = seg.process_id if not seg.is_idle else "IDLE"
        timeline_parts.append(f"{label} [{seg.start_time}→{seg.end_time}]")
    st.code(" | ".join(timeline_parts))


# ---------------------------------------------------------------------------
# Smoke test — proves the backend integration works
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    """Run a small sample through the full pipeline and display results.

    This is embedded in C1 to catch import/API issues early.  It will be
    replaced by live user input in C2.
    """
    st.info("🔬 **Backend Integration Smoke Test** — "
            "running a sample workload through the full pipeline.")

    sample_processes = [
        Process(pid="P1", arrival_time=0, burst_time=5, priority=2),
        Process(pid="P2", arrival_time=1, burst_time=3, priority=1),
        Process(pid="P3", arrival_time=2, burst_time=4, priority=3),
    ]

    st.caption("Sample input: P1(at=0, bt=5, prio=2), "
               "P2(at=1, bt=3, prio=1), P3(at=2, bt=4, prio=3)")

    try:
        result = run_simulation(sample_processes, "FCFS")
        display_results(result)
        st.success("✅ Backend integration verified — all APIs working correctly.")
    except Exception as exc:
        st.error(f"❌ Smoke test failed: {exc}")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit application entry point."""
    # --- Page config -------------------------------------------------------
    st.set_page_config(
        page_title="CPU Scheduling Simulator",
        page_icon="⚙️",
        layout="wide",
    )

    # --- Title & description -----------------------------------------------
    st.title("⚙️ CPU Scheduling Simulator")
    st.markdown(
        "A web-based simulator for classic CPU scheduling algorithms. "
        "Enter processes, choose an algorithm, and view scheduling results "
        "including per-process metrics and execution timelines."
    )

    # --- Sidebar: Algorithm selection --------------------------------------
    with st.sidebar:
        st.header("Algorithm")
        selected_algo = st.selectbox(
            "Scheduling Algorithm",
            options=list(ALGORITHMS.keys()),
            index=0,
            help="Select which CPU scheduling algorithm to use.",
        )

        quantum = None
        if selected_algo == "Round Robin":
            quantum = st.number_input(
                "Time Quantum",
                min_value=1,
                value=2,
                step=1,
                help="Number of time units per Round Robin slice.",
            )

        st.divider()
        st.caption("Algorithms provided by Laptop A's scheduling engine.")

    # --- Main area ---------------------------------------------------------

    # Section 1: Process Input (placeholder for C2)
    st.header("📋 Process Input")
    st.markdown(
        "*Full CSV upload and manual entry will be available in C2.  "
        "The smoke test below uses a hardcoded sample dataset.*"
    )

    # Section 2: Simulation controls
    st.header("▶️ Simulation")
    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("Run Smoke Test", type="primary")
    with info_col:
        st.caption(
            f"Algorithm: **{selected_algo}**"
            + (f" (quantum={quantum})" if selected_algo == "Round Robin" else "")
        )

    # Section 3: Results
    st.header("📈 Results")
    if run_clicked:
        try:
            # Use the selected algorithm for the smoke test
            sample_processes = [
                Process(pid="P1", arrival_time=0, burst_time=5, priority=2),
                Process(pid="P2", arrival_time=1, burst_time=3, priority=1),
                Process(pid="P3", arrival_time=2, burst_time=4, priority=3),
            ]
            result = run_simulation(sample_processes, selected_algo, quantum)
            display_results(result)
            st.success("✅ Backend integration verified.")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")
    else:
        st.caption("Press **Run Smoke Test** to execute the simulation.")

    # --- Footer ------------------------------------------------------------
    st.divider()
    st.caption(
        "CPU Scheduling Lab — "
        "Engine: Laptop A · Validation/Metrics/CSV: Laptop B · UI: Laptop C"
    )


if __name__ == "__main__":
    main()
