## 👥 Team & Contributions

### 💻 Technical Contributions

| Team Member | Technical Area | Contribution |
|---|---|---|
| **Edwin** | Core Scheduling Engine | FCFS, SJF, Priority, Round Robin, core data models, scheduling logic, Gantt execution segments, core algorithm tests |
| **Akriti Srivastava** | Validation & Data Layer | Input validation, error handling, performance metrics, CSV input handling, validation/metrics tests |
| **Anish** | Frontend & Visualization | Streamlit interface, process input UI, algorithm controls, Gantt chart, metrics dashboard, algorithm comparison |
| **Akriti Raj** | Testing & Integration | End-to-end testing, integration testing, edge-case testing, output verification, application integration |

---

### 🧠 Edwin — Core Scheduling Engine

- Designed the core scheduling architecture
- Implemented:
  - FCFS
  - SJF
  - Priority Scheduling
  - Round Robin
- Created core process and simulation data models
- Generated structured Gantt-chart execution segments
- Implemented core scheduling calculations:
  - Completion Time
  - Turnaround Time
  - Waiting Time
- Added deterministic tie-breaking behavior
- Built the initial automated test suite
- Kept the scheduling engine independent from the UI

---

### 🛡️ Akriti Srivastava — Validation & Data Layer

- Built the input validation layer
- Validated:
  - Process ID
  - Arrival Time
  - Burst Time
  - Priority
  - Round Robin Time Quantum
- Detected:
  - Missing values
  - Invalid data types
  - Duplicate process IDs
  - Invalid numerical values
  - Invalid CSV structure
  - Incorrect CSV column names
- Added user-friendly validation/error messages
- Implemented performance metrics:
  - Response Time
  - CPU Utilization
  - Throughput
- Added CSV input and data-processing support
- Added automated tests for validation and metrics

---

### 🎨 Anish — Frontend & Visualization

- Built the Streamlit web interface
- Created the process input interface
- Added CSV upload interface
- Added scheduling algorithm selection
- Added Round Robin time-quantum controls
- Integrated the backend scheduling engine with the UI
- Built Gantt chart visualization
- Displayed process-level scheduling results
- Created the performance metrics dashboard
- Added algorithm comparison views
- Focused on making the application interactive and easy to understand

---

### 🧪 Akriti Raj — Testing & Integration

- Performed end-to-end application testing
- Tested all supported scheduling algorithms
- Tested valid and invalid inputs
- Tested edge cases and different process workloads
- Verified scheduling outputs against expected results
- Performed integration testing between:
  - Scheduling engine
  - Validation layer
  - Metrics layer
  - Streamlit UI
- Assisted with final application integration
- Verified the application before final deployment/demo

---

## 📚 Documentation & Presentation

The team collaboratively worked on:

- README and project documentation
- Sample inputs and outputs
- Hackathon presentation
- Live demonstration
- Backup demo/video
- Final project review

### 🤝 Team Workflow

The project was developed using a modular architecture:

```text
                    CPU Scheduling Lab
                           │
              ┌────────────┴────────────┐
              │                         │
        Backend / Logic             Frontend
              │                         │
      ┌───────┴───────┐           ┌─────┴─────┐
      │               │           │           │
   Scheduling     Validation    Streamlit   Testing
    Engine        & Metrics        UI       & Integration
      │               │           │           │
    Edwin      Akriti Srivastava   Anish    Akriti Raj
