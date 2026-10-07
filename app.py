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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# Deterministic, readable colour palette for process IDs.
_PROCESS_COLORS = [
    "#4C78A8",  # steel blue
    "#F58518",  # orange
    "#E45756",  # red
    "#72B7B2",  # teal
    "#54A24B",  # green
    "#EECA3B",  # yellow
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9D755D",  # brown
    "#BAB0AC",  # grey
]
_IDLE_COLOR = "#D3D3D3"  # light grey for IDLE segments


def _pid_color_map(segments) -> dict[str, str]:
    """Build a deterministic PID → colour mapping from execution segments."""
    seen: list[str] = []
    for seg in segments:
        if not seg.is_idle and seg.process_id not in seen:
            seen.append(seg.process_id)
    return {
        pid: _PROCESS_COLORS[i % len(_PROCESS_COLORS)]
        for i, pid in enumerate(seen)
    }


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Ensure all required session-state keys exist with defaults."""
    if "manual_processes" not in st.session_state:
        st.session_state.manual_processes = []
    if "next_pid_num" not in st.session_state:
        st.session_state.next_pid_num = 1
    if "csv_processes" not in st.session_state:
        st.session_state.csv_processes = []
    if "csv_extra_columns" not in st.session_state:
        st.session_state.csv_extra_columns = []
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None


# ---------------------------------------------------------------------------
# Backend pipeline helpers
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
    val_result = validate_processes(processes)
    if not val_result.is_valid:
        messages = "\n".join(f"• {e.message}" for e in val_result.errors)
        raise ValueError(f"Input validation failed:\n{messages}")

    if algorithm_name == "Round Robin":
        q_result = validate_round_robin_quantum(quantum)
        if not q_result.is_valid:
            messages = "\n".join(f"• {e.message}" for e in q_result.errors)
            raise ValueError(f"Quantum validation failed:\n{messages}")

    algo = get_algorithm(algorithm_name, quantum)
    result = algo.schedule(processes)
    return compute_metrics(result, processes)


# ---------------------------------------------------------------------------
# Gantt chart (C3)
# ---------------------------------------------------------------------------


def render_gantt_chart(result: SimulationResult) -> None:
    """Render a horizontal Gantt chart from execution_segments using matplotlib."""
    segments = result.execution_segments
    if not segments:
        st.warning("No execution segments to display.")
        return

    color_map = _pid_color_map(segments)
    total_time = segments[-1].end_time

    fig, ax = plt.subplots(figsize=(max(10, total_time * 0.6), 2.0))

    # Draw each segment as a horizontal bar on a single row (y=0).
    for seg in segments:
        duration = seg.end_time - seg.start_time
        if duration <= 0:
            continue
        color = _IDLE_COLOR if seg.is_idle else color_map[seg.process_id]
        label = "IDLE" if seg.is_idle else seg.process_id

        ax.barh(
            y=0,
            width=duration,
            left=seg.start_time,
            height=0.6,
            color=color,
            edgecolor="white",
            linewidth=1.5,
        )
        # Label inside the bar
        ax.text(
            seg.start_time + duration / 2,
            0,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#333333" if seg.is_idle else "white",
        )

    # Time markers along the bottom
    time_ticks = sorted({seg.start_time for seg in segments} | {segments[-1].end_time})
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([str(t) for t in time_ticks], fontsize=8)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_xlim(segments[0].start_time, total_time)
    ax.set_yticks([])
    ax.set_title(f"{result.algorithm_name} — Gantt Chart", fontsize=12, pad=10)

    # Legend
    legend_patches = []
    for pid, color in color_map.items():
        legend_patches.append(mpatches.Patch(color=color, label=pid))
    legend_patches.append(mpatches.Patch(color=_IDLE_COLOR, label="IDLE"))
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results dashboard (C3)
# ---------------------------------------------------------------------------


def display_metrics_dashboard(result: SimulationResult) -> None:
    """Render polished metric cards from the SimulationResult."""
    st.subheader(f"📊 {result.algorithm_name} — Metrics Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("⏱️ Avg Waiting Time", f"{result.average_waiting_time:.2f}")
    col2.metric("🔄 Avg Turnaround Time", f"{result.average_turnaround_time:.2f}")
    if result.average_response_time is not None:
        col3.metric("⚡ Avg Response Time", f"{result.average_response_time:.2f}")

    col4, col5 = st.columns(2)
    if result.cpu_utilization is not None:
        col4.metric("💻 CPU Utilization", f"{result.cpu_utilization:.1f}%")
    if result.throughput is not None:
        col5.metric("📈 Throughput", f"{result.throughput:.4f} proc/unit")


def display_process_table(result: SimulationResult) -> None:
    """Render the per-process metrics table."""
    st.subheader("📋 Per-Process Results")
    table_data = []
    for m in result.process_metrics:
        table_data.append({
            "PID": m.pid,
            "Arrival": m.arrival_time,
            "Burst": m.burst_time,
            "Priority": m.priority,
            "Completion": m.completion_time,
            "Turnaround": m.turnaround_time,
            "Waiting": m.waiting_time,
            "Response": m.response_time if m.response_time is not None else "—",
        })
    st.table(table_data)


def display_results(result: SimulationResult) -> None:
    """Render the full results section: dashboard + Gantt + table."""
    # 1. Metrics dashboard
    display_metrics_dashboard(result)

    st.divider()

    # 2. Gantt chart
    st.subheader("📊 Execution Timeline — Gantt Chart")
    render_gantt_chart(result)

    st.divider()

    # 3. Per-process table
    display_process_table(result)


# ---------------------------------------------------------------------------
# UI sections (C2 — unchanged)
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
            st.markdown("")
            st.markdown("")
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

        if parse_result.extra_columns:
            st.warning(
                f"ℹ️ Extra columns ignored: {', '.join(parse_result.extra_columns)}"
            )

        val_result = validate_processes(parse_result.processes)
        if not val_result.is_valid:
            st.error("❌ **Validation Errors in CSV data:**")
            for err in val_result.errors:
                st.error(f"• {err.message}")
            st.session_state.csv_processes = []
            return

        st.session_state.csv_processes = parse_result.processes
        st.session_state.csv_extra_columns = parse_result.extra_columns
        st.success(f"✅ Parsed {len(parse_result.processes)} valid process(es) from CSV.")

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
    st.set_page_config(
        page_title="CPU Scheduling Simulator",
        page_icon="⚙️",
        layout="wide",
    )
    _init_session_state()

    st.title("⚙️ CPU Scheduling Simulator")
    st.markdown(
        "A web-based simulator for classic CPU scheduling algorithms.  "
        "Enter processes manually or upload a CSV, choose an algorithm, "
        "and view scheduling results including Gantt charts, metrics dashboards, "
        "and per-process result tables."
    )

    # --- Sidebar -----------------------------------------------------------
    selected_algo, quantum = _render_sidebar()

    # --- Process Input (C2) ------------------------------------------------
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

    # --- Results (C3) ------------------------------------------------------
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
