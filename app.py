"""CPU Scheduling Simulator — Streamlit Application.

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

import io

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
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Ensure all required session-state keys exist with defaults.

    Called once at the top of every Streamlit rerun.  Keys are initialised
    only if absent so that existing state survives widget-driven reruns.
    """
    if "manual_processes" not in st.session_state:
        st.session_state.manual_processes = []  # list[dict]
    if "next_pid_num" not in st.session_state:
        st.session_state.next_pid_num = 1
    if "csv_processes" not in st.session_state:
        st.session_state.csv_processes = []  # list[Process]
    if "csv_extra_columns" not in st.session_state:
        st.session_state.csv_extra_columns = []
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None


# ---------------------------------------------------------------------------
# Backend pipeline helpers (unchanged from C1)
# ---------------------------------------------------------------------------


def get_algorithm(name: str, quantum: int | None = None):
    """Instantiate a scheduling algorithm by display name."""
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

    # --- Execution timeline (text, placeholder for Gantt in C3) ------------
    st.markdown("**Execution Timeline** *(Gantt visualization coming in C3)*")
    timeline_parts = []
    for seg in result.execution_segments:
        label = seg.process_id if not seg.is_idle else "IDLE"
        timeline_parts.append(f"{label} [{seg.start_time}→{seg.end_time}]")
    st.code(" | ".join(timeline_parts))


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def _render_sidebar() -> tuple[str, int | None]:
    """Render the sidebar and return (algorithm_name, quantum_or_None)."""
    with st.sidebar:
        st.header("⚙️ Algorithm")
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

    return selected_algo, quantum


def _render_manual_input() -> None:
    """Render the manual process entry form and current process table."""
    procs = st.session_state.manual_processes

    # --- Current process table ---------------------------------------------
    if procs:
        st.markdown("**Current Processes**")
        st.table([
            {
                "PID": p["pid"],
                "Arrival Time": p["arrival_time"],
                "Burst Time": p["burst_time"],
                "Priority": p["priority"],
            }
            for p in procs
        ])
    else:
        st.info("No processes added yet.  Use the form below to add processes.")

    # --- Add process form --------------------------------------------------
    st.markdown("**Add a Process**")
    with st.form("add_process_form", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            pid = st.text_input(
                "PID",
                value=f"P{st.session_state.next_pid_num}",
                help="Unique process identifier.",
            )
        with col2:
            arrival = st.number_input(
                "Arrival Time",
                min_value=0,
                value=0,
                step=1,
                help="Time unit when the process becomes ready (≥ 0).",
            )
        with col3:
            burst = st.number_input(
                "Burst Time",
                min_value=1,
                value=1,
                step=1,
                help="CPU time required (> 0).",
            )
        with col4:
            priority = st.number_input(
                "Priority",
                value=1,
                step=1,
                help="Lower number = higher priority.",
            )

        submitted = st.form_submit_button("➕ Add Process", type="primary")
        if submitted:
            st.session_state.manual_processes.append({
                "pid": pid.strip(),
                "arrival_time": int(arrival),
                "burst_time": int(burst),
                "priority": int(priority),
            })
            st.session_state.next_pid_num += 1
            st.rerun()

    # --- Remove / Clear controls -------------------------------------------
    if procs:
        rm_col, clear_col = st.columns(2)
        with rm_col:
            pid_options = [p["pid"] for p in procs]
            remove_pid = st.selectbox(
                "Remove process",
                options=pid_options,
                key="remove_pid_select",
            )
            if st.button("🗑️ Remove Selected"):
                st.session_state.manual_processes = [
                    p for p in procs if p["pid"] != remove_pid
                ]
                st.rerun()
        with clear_col:
            st.markdown("")  # spacer
            st.markdown("")  # align button vertically
            if st.button("🧹 Clear All Processes"):
                st.session_state.manual_processes = []
                st.session_state.next_pid_num = 1
                st.rerun()


def _get_manual_processes() -> list[Process]:
    """Convert the session-state manual process dicts to Process objects."""
    return [
        Process(
            pid=p["pid"],
            arrival_time=p["arrival_time"],
            burst_time=p["burst_time"],
            priority=p["priority"],
        )
        for p in st.session_state.manual_processes
    ]


def _render_csv_upload() -> None:
    """Render the CSV upload widget and parse the uploaded file."""
    st.markdown(
        "Upload a CSV file with columns: **PID, Arrival Time, Burst Time, Priority**"
    )
    st.caption("Column order does not matter.  Extra columns are silently ignored.")

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="csv_uploader",
        help="CSV with columns: PID, Arrival Time, Burst Time, Priority",
    )

    if uploaded is not None:
        # Decode and parse using B3
        try:
            csv_text = uploaded.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("❌ Could not decode the file as UTF-8.  "
                     "Please upload a UTF-8 encoded CSV file.")
            st.session_state.csv_processes = []
            return

        parse_result = parse_processes_csv(csv_text)

        if not parse_result.is_valid:
            st.error("❌ **CSV Parsing Errors:**")
            for err in parse_result.errors:
                st.error(f"• {err.message}")
            st.session_state.csv_processes = []
            return

        # Warn about extra columns (non-blocking)
        if parse_result.extra_columns:
            st.warning(
                f"ℹ️ Extra columns ignored: {', '.join(parse_result.extra_columns)}"
            )

        # Run B1 validation on parsed processes
        val_result = validate_processes(parse_result.processes)
        if not val_result.is_valid:
            st.error("❌ **Validation Errors in CSV data:**")
            for err in val_result.errors:
                st.error(f"• {err.message}")
            st.session_state.csv_processes = []
            return

        # All good — store processes
        st.session_state.csv_processes = parse_result.processes
        st.session_state.csv_extra_columns = parse_result.extra_columns
        st.success(f"✅ Parsed {len(parse_result.processes)} valid process(es) from CSV.")

        # Show parsed processes
        st.markdown("**Parsed Processes**")
        st.table([
            {
                "PID": p.pid,
                "Arrival Time": p.arrival_time,
                "Burst Time": p.burst_time,
                "Priority": p.priority,
            }
            for p in parse_result.processes
        ])
    else:
        st.session_state.csv_processes = []


def _render_sample_csv() -> None:
    """Show a downloadable sample CSV for user convenience."""
    sample = "PID,Arrival Time,Burst Time,Priority\nP1,0,5,2\nP2,1,3,1\nP3,2,4,3\n"
    st.download_button(
        label="📄 Download Sample CSV",
        data=sample,
        file_name="sample_processes.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit application entry point."""
    # --- Page config -------------------------------------------------------
    st.set_page_config(
        page_title="CPU Scheduling Simulator",
        page_icon="⚙️",
        layout="wide",
    )
    _init_session_state()

    # --- Title & description -----------------------------------------------
    st.title("⚙️ CPU Scheduling Simulator")
    st.markdown(
        "A web-based simulator for classic CPU scheduling algorithms.  "
        "Enter processes manually or upload a CSV, choose an algorithm, "
        "and view scheduling results including per-process metrics and "
        "execution timelines."
    )

    # --- Sidebar -----------------------------------------------------------
    selected_algo, quantum = _render_sidebar()

    # --- Process Input -----------------------------------------------------
    st.header("📋 Process Input")

    input_method = st.radio(
        "Input Method",
        options=["Manual Entry", "CSV Upload"],
        horizontal=True,
        help="Choose how to provide process data.",
    )

    if input_method == "Manual Entry":
        _render_manual_input()
        active_processes = _get_manual_processes()
    else:
        _render_csv_upload()
        _render_sample_csv()
        active_processes = list(st.session_state.csv_processes)

    # --- Simulation --------------------------------------------------------
    st.header("▶️ Simulation")

    info_parts = [f"Algorithm: **{selected_algo}**"]
    if selected_algo == "Round Robin":
        info_parts.append(f"Quantum: **{quantum}**")
    info_parts.append(f"Processes: **{len(active_processes)}**")
    st.caption(" · ".join(info_parts))

    run_clicked = st.button(
        "🚀 Run Simulation",
        type="primary",
        disabled=(len(active_processes) == 0),
    )

    # --- Results -----------------------------------------------------------
    st.header("📈 Results")

    if run_clicked and active_processes:
        try:
            result = run_simulation(active_processes, selected_algo, quantum)
            st.session_state.simulation_result = result
        except ValueError as ve:
            st.error(str(ve))
            st.session_state.simulation_result = None
        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")
            st.session_state.simulation_result = None

    if st.session_state.simulation_result is not None:
        display_results(st.session_state.simulation_result)
    elif not run_clicked:
        st.caption("Add processes and press **Run Simulation** to see results.")

    # --- Footer ------------------------------------------------------------
    st.divider()
    st.caption(
        "CPU Scheduling Lab — "
        "Engine: Laptop A · Validation/Metrics/CSV: Laptop B · UI: Laptop C"
    )


if __name__ == "__main__":
    main()
