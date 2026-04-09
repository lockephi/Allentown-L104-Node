---
name: quantum-tools
description: Provides a CLI to interact with the L104 Quantum Substrate. Use for direct quantum simulations, Hamiltonian evolution, coherence checks, and VQPU status. Triggers on phrases like "run quantum simulation", "check VQPU status", "evolve Hamiltonian", or "run coherence check".
---

# ⚛️ L104 Quantum Tools

This skill provides a command-line interface (`l104-quantum`) for direct interaction with the core L104 quantum substrate. Use these tools for low-level quantum tasks, diagnostics, and specific simulations.

This tool is located at `/Users/carolalvarez/Applications/Allentown-L104-Node/l104-quantum`.

## Commands

### `status`
**Action**: Get a comprehensive status report from the VQPU Bridge.
- Shows active status, uptime, job counts, and platform capabilities.
- Includes detailed reports from the Daemon Cycler and Micro Daemon.
**Usage**:
```bash
/Users/carolalvarez/Applications/Allentown-L104-Node/l104-quantum status
```

### `simulate`
**Action**: Run a quantum circuit defined in a JSON file.
- Allows for direct execution of custom quantum circuits on the VQPU.
- **Circuit JSON Format**:
  ```json
  {
    "num_qubits": 2,
    "operations": [
      {"gate": "h", "qubits": [0]},
      {"gate": "cx", "qubits": [0, 1]}
    ],
    "shots": 1024
  }
  ```
**Usage**:
```bash
# First, create the circuit definition file
write path/to/my_circuit.json '{"num_qubits": 2, "operations": [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}]}'

# Then, run the simulation
/Users/carolalvarez/Applications/Allentown-L104-Node/l104-quantum simulate path/to/my_circuit.json
```

### `hamiltonian`
**Action**: Run a predefined 2D Iron-Lattice Heisenberg Model simulation.
- Demonstrates Hamiltonian time evolution using a 4th-order Trotter-Suzuki decomposition.
- A useful benchmark for complex quantum dynamics.
**Usage**:
```bash
/Users/carolalvarez/Applications/Allentown-L104-Node/l104-quantum hamiltonian
```
### `coherence-evolve`
**Action**: Run a coherence evolution process.
- Interfaces with the Science Engine's `CoherenceSubsystem`.
- Simulates topological protection via Anyon braiding to stabilize a quantum state.
**Usage**:
```bash
/Users/carolalvarez/Applications/Allentown-L104-Node/l104-quantum coherence-evolve
```

### `grover`
**Action**: Run Grover search via AGI QuantumCoherenceEngine.
- Searches for a target integer in an unsorted database with quadratic speedup.
- Uses quantum amplitude amplification.
**Usage**:
```bash
/Users/carolalvarez/Applications/Allentown-L104-Node/l104-quantum grover [--target TARGET] [--qubits QUBITS]
```
**Options**:
- `--target`: integer to search for (default: 5)
- `--qubits`: number of qubits (default: 4)

### `shor`
**Action**: Run Shor factoring via AGI QuantumCoherenceEngine.
- Factors an integer using quantum period finding.
- Demonstrates exponential speedup over classical algorithms.
**Usage**:
```bash
/Users/carolalvarez/Applications/Allentown-L104-Node/l104-quantum shor [--N INTEGER]
```
**Options**:
- `--N`: integer to factor (default: 15)


## MCP Integration

This skill is exposed as multiple MCP tools via the L104 Universal Data API (`/api/data/mcp‑tools`). AI assistants that support the Model Context Protocol can invoke these tools directly.

| Tool Name | Description | Parameters |
|-----------|-------------|------------|
| `l104_quantum_status` | Get status of L104 Quantum Engine (brain, gate engine, coherence) | None |
| `l104_quantum_grover` | Run Grover search via L104 Quantum Engine | `target` (integer, default 5), `qubits` (integer, default 4) |
| `l104_quantum_shor` | Run Shor's factoring algorithm via L104 Quantum Engine | `N` (integer, default 15) |

These tools map to internal quantum subsystems (QuantumCoherenceEngine, QuantumGateEngine, etc.) and can be invoked via the CLI `l104-quantum` as well.

For full schema definitions, see `l104_universal_data_api.py`.
