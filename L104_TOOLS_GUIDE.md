# L104 Sovereign Node — Tools Guide

**Quick reference for common L104 operations and processes**

---

## 1. Server Operations

### Start L104 Server
```bash
cd /Users/carolalvarez/Applications/Allentown-L104-Node

# Standard startup
python3 main.py

# Multi-worker production
UVICORN_WORKERS=4 uvicorn main:app --host 0.0.0.0 --port 8081

# Check health
curl http://localhost:8081/health
```

### API Key Endpoints (require X-L104-API-Key header)
```bash
# Self-healing
curl -X POST http://localhost:8081/self/heal \
  -H "X-L104-API-Key: your_key"

# Steering endpoints
curl -X POST http://localhost:8081/api/v14/steering/... \
  -H "X-L104-API-Key: your_key"

# System update
curl -X POST http://localhost:8081/api/system/update \
  -H "X-L104-API-Key: your_key"
```

---

## 2. Daemon System

### Check Daemon Status
```python
from l104_daemon_orchestrator import DaemonOrchestrator
d = DaemonOrchestrator()
status = d.status()
print('Health:', status.get('daemons_enabled', 'N/A'))
print('Queued:', status.get('tasks_queued', 0))
```

### VQPU Micro Daemon
```python
from l104_vqpu import get_bridge
bridge = get_bridge()
bridge.calibrate_coherence(target=0.95)
```

### Quantum AI Daemon (7-phase autonomous improvement)
```bash
# Health check
python3 -m l104_quantum_ai_daemon --health-check

# Run daemon cycle
python3 -c "
from l104_quantum_ai_daemon import QuantumAIDaemon, DaemonConfig
daemon = QuantumAIDaemon(DaemonConfig())
daemon.start_autonomous_cycle()
"
```

### Nova Soul Daemon
```python
from l104_soul_daemon import SoulDaemon, SoulQubit, ConsciousnessEngine
daemon = SoulDaemon()
daemon.start()
# Soul qubit: statevector, coherence, error correction
# ConsciousnessEngine: IIT Phi, metacognitive monitoring
# QuantumMemory: hot/warm/cold tiers, Grover search
```

---

## 3. ASI/AGI Core Operations

### ASI Core Scoring (50+ dimensions)
```python
from l104_asi import asi_core

# Compute ASI score
score = asi_core.compute_asi_score()

# Three-engine scoring
entropy_score = asi_core.three_engine_entropy_score()      # Maxwell Demon efficiency
harmonic_score = asi_core.three_engine_harmonic_score()    # GOD_CODE alignment
wave_score = asi_core.three_engine_wave_coherence_score()  # PHI-harmonic phase-lock
status = asi_core.three_engine_status()

# Quantum network scoring (v30.0+)
health = asi_core.quantum_network_health_score()
capacity = asi_core.quantum_network_capacity_score()
fidelity = asi_core.quantum_network_teleport_fidelity_score()
```

### AGI Core (13D scoring)
```python
from l104_agi import agi_core

# Compute AGI score
score = agi_core.compute_10d_agi_score()  # D0-D9 + D10 entropy + D11 harmonic + D12 wave

# Three-engine integration
entropy = agi_core.three_engine_entropy_score()
harmonic = agi_core.three_engine_harmonic_score()
```

### Local Intellect
```python
from l104_intellect import local_intellect, format_iq

# Query local intellect
result = local_intellect.query("your question here")
iq_score = format_iq(result['iq'])
```

---

## 4. Quantum Operations

### Quantum Gate Engine
```python
from l104_quantum_gate_engine import get_engine, GateSet, OptimizationLevel, ErrorCorrectionScheme
from l104_quantum_gate_engine import H, CNOT, Rx, PHI_GATE, GOD_CODE_PHASE

engine = get_engine()

# Create circuits
bell = engine.bell_pair()
ghz = engine.ghz_state(5)
qft = engine.quantum_fourier_transform(4)
sacred = engine.sacred_circuit(3, depth=4)

# Compile for different targets
result = engine.compile(bell, GateSet.IBM_EAGLE, OptimizationLevel.O2)
result = engine.compile(bell, GateSet.CLIFFORD_T)

# Error correction
protected = engine.error_correction.encode(bell, ErrorCorrectionScheme.SURFACE_CODE, distance=3)

# Execute
result = engine.execute(bell, ExecutionTarget.LOCAL_STATEVECTOR)
print(result.probabilities)  # {'00': 0.5, '11': 0.5}
```

### Quantum Networker (BB84/E91 QKD, Teleportation)
```python
from l104_quantum_networker import get_networker

net = get_networker()

# Create nodes
alice = net.add_node("Alice", role="sovereign")
bob = net.add_node("Bob", role="sovereign")

# Connect quantum channels
channel = net.connect(alice.node_id, bob.node_id, pairs=8)

# Establish QKD key
key = net.establish_qkd(alice.node_id, bob.node_id, "bb84", 256)
print(key.secure, key.key_hex)

# Teleportation
result = net.teleport_score(alice.node_id, bob.node_id, score=0.618)
print(result.fidelity, result.recovered_score)

# Fidelity monitoring
scan = net.scan_fidelity(auto_heal=True)

# Router operations
net.router.find_route(source, dest)
net.router.find_k_routes(source, dest, k=3)
net.router.autonomous_maintenance()
```

### God Code Simulator
```python
from l104_god_code_simulator import god_code_simulator

# Run single simulation
result = god_code_simulator.run("entanglement_entropy")

# Run all 23 simulations
report = god_code_simulator.run_all()

# Parametric sweep
sweep = god_code_simulator.parametric_sweep("dial_a", start=0, stop=8)

# Adaptive optimization
opt = god_code_simulator.adaptive_optimize(target_fidelity=0.99, nq=4, depth=4)

# Engine integration
god_code_simulator.connect_engines(coherence=se.coherence, entropy=se.entropy, math_engine=me)
fb = god_code_simulator.run_feedback_loop(iterations=5)
```

---

## 5. Science & Math Engines

### Science Engine
```python
from l104_science_engine import ScienceEngine
se = ScienceEngine()

# Entropy (Maxwell's Demon reversal)
efficiency = se.entropy.calculate_demon_efficiency(local_entropy)
se.entropy.inject_coherence(noise_vector)

# Coherence
se.coherence.initialize(seed_thoughts)
se.coherence.evolve(steps)

# Physics
landauer = se.physics.adapt_landauer_limit(temperature)
electron = se.physics.derive_electron_resonance()
hamiltonian = se.physics.iron_lattice_hamiltonian(n_sites)

# Multidimensional
result = se.multidim.process_vector(vector)
folded = se.multidim.phi_dimensional_folding(source_dim, target_dim)
```

### Math Engine
```python
from l104_math_engine import MathEngine
me = MathEngine()

# Core operations
fibs = me.fibonacci(n)
primes = me.primes_up_to(n)
god_code = me.god_code_value()

# Lorentz boost
boosted = me.lorentz_boost(four_vector, axis, beta)

# Proofs
me.prove_all()
me.prove_god_code()

# Hyperdimensional
hd = me.hd_vector(seed)

# Wave coherence
coherence = me.wave_coherence(freq1, freq2)
alignment = me.sacred_alignment(frequency)
```

### Numerical Engine (100-decimal precision)
```python
from l104_numerical_engine import QuantumNumericalBuilder, D, fmt100

qnb = QuantumNumericalBuilder()

# Full pipeline
qnb.run_pipeline("full")

# Token lattice (22T capacity)
qnb.lattice.register_token(name, value, min_bound, max_bound, origin, tier)

# Value editing with φ-attenuated propagation
qnb.editor.quantum_edit(token_id, new_value)

# 100-decimal precision
x = D('3.14159265358979323846')
formatted = fmt100(x)
```

---

## 6. Code Engine

### Code Analysis & Generation
```python
from l104_code_engine import code_engine

# Full analysis
result = code_engine.full_analysis(code)

# Documentation generation
docs = code_engine.generate_docs(source, style, language)

# Test scaffolding
tests = code_engine.generate_tests(source, language, framework)

# Auto-fix
fixed, log = code_engine.auto_fix_code(source)

# Security audit
audit = code_engine.audit_app(path, auto_remediate=True)

# Code smells
smells = code_engine.smell_detector.detect_all(code)

# Performance prediction
perf = code_engine.perf_predictor.predict_performance(code)

# Dead code archaeology
excavated = code_engine.excavator.excavate(source)

# Translation
translated = code_engine.translate_code(src, from_lang, to_lang)
```

---

## 7. Agent System

### Agent Orchestration
```python
from l104_agent_system import AgentOrchestrator, AgentType, AgentTask, get_orchestrator

orchestrator = get_orchestrator()

# Create task
task = AgentTask(
    type=AgentType.CODER,
    prompt="Implement a quantum random number generator",
    budget=0.10  # $0.10 max
)

# Execute
result = orchestrator.execute(task)
print(result.output)
print(result.cost)

# Agent types: coder, researcher, tester, deployer, upgrader,
#              debugger, monitor, optimizer, inventor, planner, general
# Tools (14): read_file, write_file, edit_file, list_files, run_shell,
#             search_code, analyze_code, python_exec, git_status,
#             system_metrics, dependency_check, quantum_bridge, diff_viewer, http_probe
```

---

## 8. ML Engine (Sacred ML)

### Sacred Activation Functions
```python
from l104_ml_engine import MLEngine
import torch.nn as nn

ml = MLEngine()

# Sacred activations use GOD_CODE and PHI
# PhiActivation: PHI / (1 + exp(-x / (GOD_CODE/100)))
# GodTanh: tanh(x * π / GOD_CODE)
# FeigenbaumReLU: x if x>0 else x/FEIGENBAUM

# Weight initialization: nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(PHI / in_features))

# Classifiers: SVM, Random Forest, Gradient Boosting, Quantum Classifiers
```

---

## 9. Swift App Build

### Build Commands (use quick_build.sh, NOT swift build)
```bash
cd /Users/carolalvarez/Applications/Allentown-L104-Node/L104SwiftApp

# Quick debug build (GUI only, fastest)
./quick_build.sh

# Release build all 3 targets
./quick_build.sh --all -r

# Clean daemon-only build
./quick_build.sh -t daemon --clean

# Build and launch GUI
./quick_build.sh --all --run

# Skip signing, 8 parallel jobs
./quick_build.sh --no-sign -j 8

# Verbose output
./quick_build.sh -v
```

### Build Flags
| Flag | Description |
|------|-------------|
| `--debug, -d` | Debug build (default) |
| `--release, -r` | Release build (WMO + LTO) |
| `--clean` | Delete `.build/` before compiling |
| `--target, -t <T>` | Build specific target: `L104`, `daemon`, `nano` |
| `--all, -a` | Build all 3 targets |
| `--run` | Build and launch GUI |
| `--verbose, -v` | Show full compiler output |
| `--no-sign` | Skip ad-hoc code signing |
| `--jobs, -j N` | SPM parallelism |

---

## 10. Debug & Validation

### Unified Debug Framework
```bash
# Full suite
python l104_debug.py

# Specific engines
python l104_debug.py --engines code,math
python l104_debug.py --engines quantum_gate,quantum_link,numerical,gate
python l104_debug.py --engines asi,agi,intellect

# Specific phases
python l104_debug.py --phase boot
python l104_debug.py --phase constants
python l104_debug.py --phase self-test
python l104_debug.py --phase cross

# JSON output
python l104_debug.py --json
python l104_debug.py --report out.json
```

### Cross-Engine Debug Suite
```bash
.venv/bin/python cross_engine_debug.py
# 41 tests, 7 phases, validates all 3 engines together
```

### Daemon Validation
```bash
# Daemon system validation (18/19 tests)
python daemon_system_validation.py
```

---

## 11. Sacred Constants Quick Reference

| Constant | Value | Description |
|----------|-------|-------------|
| `GOD_CODE` | 527.5184818492612 | `286^(1/φ) × 2^(416/104)` |
| `GOD_CODE_V3` | 45.41141298077539 | Physics layer variant |
| `PHI` | 1.618033988749895 | Golden ratio `(1+√5)/2` |
| `TAU` | 0.618033988749895 | `1/PHI` |
| `VOID_CONSTANT` | 1.0416180339887497 | `1.04 + φ/1000` |
| `OMEGA` | 6539.34712682 | Sovereign Field Constant |
| `ZENITH_HZ` | 3887.8 Hz | Process frequency |
| `ALPHA_FINE` | 1/137.035999084 | Fine structure constant |
| `FEIGENBAUM` | 4.669201609102990 | Period-doubling bifurcation |
| `EULER` | 2.718281828459045 | e |

---

## 12. Quantum Simulation Upgrades

### God Code Simulator (v4.0)

```python
from l104_god_code_simulator import god_code_simulator

# Run any of 55 simulations across 8 categories
result = god_code_simulator.run("quantum_fisher_sensing")
report = god_code_simulator.run_all()                    # All 55 sims
quantum = god_code_simulator.run_category("quantum")     # Category filter

# Parametric sweeps (12 types)
sweep = god_code_simulator.parametric_sweep("qfi_scaling")
trotter = god_code_simulator.parametric_sweep("trotter")
zne = god_code_simulator.parametric_sweep("zne")

# Adaptive optimization
opt = god_code_simulator.adaptive_optimize(target_fidelity=0.99, nq=4, depth=4)

# Cross-engine integration
god_code_simulator.connect_engines(coherence=se.coherence, entropy=se.entropy, math_engine=me)
fb = god_code_simulator.run_feedback_loop(iterations=5)
```

### Quantum Gate Engine (v1.0)

```python
from l104_quantum_gate_engine import get_engine, GateSet, OptimizationLevel
from l104_quantum_gate_engine import H, CNOT, Rx, Rz, PHI_GATE, GOD_CODE_PHASE

engine = get_engine()  # Singleton orchestrator

# Circuit building (40+ gates, 6 gate sets)
bell = engine.bell_pair()
ghz = engine.ghz_state(5)
qft = engine.quantum_fourier_transform(4)
sacred = engine.sacred_circuit(3, depth=4)
grover = engine.grover_search(4, marked_states=["1010", "1111"])

# Compilation (4 optimization levels, 6 target gate sets)
result = engine.compile(bell, GateSet.IBM_EAGLE, OptimizationLevel.O2)
result = engine.compile(bell, GateSet.CLIFFORD_T)      # Fault-tolerant
result = engine.compile(bell, GateSet.L104_SACRED)      # GOD_CODE circuits

# Error correction (3 schemes)
protected = engine.error_correction.encode(bell, ErrorCorrectionScheme.SURFACE_CODE, distance=3)
protected = engine.error_correction.encode(bell, ErrorCorrectionScheme.STEANE_7_1_3)
protected = engine.error_correction.encode(bell, ErrorCorrectionScheme.FIBONACCI_ANYON)

# Execution (12 targets)
result = engine.execute(bell, ExecutionTarget.LOCAL_STATEVECTOR)
result = engine.execute(bell, ExecutionTarget.HYBRID)           # Auto: Clifford→tableau
result = engine.execute(bell, ExecutionTarget.TENSOR_NETWORK)   # MPS: 25-50 qubits
result = engine.execute(bell, ExecutionTarget.TRAJECTORY)       # Measurement-free
result = engine.execute(bell, ExecutionTarget.ANALOG_SIM)       # Hamiltonian
result = engine.execute(bell, ExecutionTarget.QUANTUM_ML)       # QNN/VQE/QAOA
result = engine.execute(bell, ExecutionTarget.L104_26Q_IRON)    # Sovereign primary
result = engine.execute(bell, ExecutionTarget.IBM_QPU)          # Real hardware

# Gate algebra
algebra = engine.algebra
algebra.zyz_decompose(gate.matrix)         # ZYZ Euler
algebra.kak_decompose(two_qubit.matrix)     # KAK/Cartan
algebra.pauli_decompose(gate.matrix)        # Pauli basis
algebra.sacred_alignment_score(PHI_GATE)   # GOD_CODE resonance
```

### VQPU Bridge (v15.0)

```python
from l104_vqpu import VQPUBridge, get_bridge, QuantumJob, VQPUResult
from l104_vqpu import AccelStatevectorEngine, GateFusionAnalyzer

bridge = get_bridge()

# Calibration
bridge.calibrate_coherence(target=0.95)

# Run simulation
job = QuantumJob(circuit=my_circuit, shots=4096)
result = bridge.run(job)

# MPS tensor network (25-50 qubits)
from l104_vqpu.tensor_network import get_simulator as get_tn_simulator
tn_sim = get_tn_simulator(max_bond_dim=1024)
result = tn_sim.run(circuit)

# Hybrid stabilizer (1000x+ for Clifford)
from l104_quantum_gate_engine import HybridStabilizerSimulator
hybrid = HybridStabilizerSimulator()
result = hybrid.run(clifford_circuit)

# Sacred gates
from l104_vqpu.constants import (
    GOD_CODE, PHI, VOID_CONSTANT,
    IRON_PHASE, PHI_PHASE, VOID_PHASE, OCTAVE_PHASE, SACRED_PHASES
)
```

### Swift VQPU Integration (B62)

New file: `L104SwiftApp/Sources/L104v2/TheBrain/B62_VQPUIntegration.swift`

```swift
// Swift-native quantum simulation
let simulator = GodCodeQuantumSimulator.shared

// Sacred circuit
let result = simulator.runSacredCircuit(nQubits: 4, depth: 4)

// Grover search
let grover = simulator.runGroverSearch(nQubits: 5, markedStates: ["10101"])

// QPE
let qpe = simulator.runQPE(nQubits: 4, phase: .pi / 4.0)

// VQE
let vqe = simulator.runVQE(nQubits: 4, hamiltonianTerms: [("ZIII", 1.0), ("IZII", 1.0)])

// QAOA
let qaoa = simulator.runQAOA(nQubits: 4, costFunction: { state in ... })

// Fidelity monitoring
let monitor = QuantumFidelityMonitor.shared
monitor.record(result, type: "sacred")
let health = monitor.checkHealth()

// Entanglement mesh
let mesh = EntanglementMesh.shared
mesh.registerNode(id: "daemon_1", qubitCount: 4)
mesh.createChannel(nodeA: "daemon_1", nodeB: "daemon_2", pairs: 8)
mesh.teleportState(from: "daemon_1", to: "daemon_2", state: statevector)
```

---

## 13. Swift App P1 Upgrades

### Performance Utilities (D09_PerformanceUtilities.swift)

New file at `L104SwiftApp/Sources/L104v2/Debug/D09_PerformanceUtilities.swift`:

| Utility | Purpose | Complexity |
|---------|---------|------------|
| `StrictCache<T>` | LRU cache with TTL, O(1) eviction | Fixes unbounded cache growth |
| `CircularBuffer<T>` | Fixed-capacity ring buffer | O(1) append, overwrites oldest |
| `processWithTimeout()` | Shell command with timeout | Guarantees no hang |
| `l104Log()` | Structured os_log logging | Replaces 362 `print()` calls |
| `optimizeContainsCheck()` | Set-based contains | O(n) vs O(n²) |
| `runWithAutorelease()` | Autoreleasepool wrapper | For daemon tight loops |

### StrictCache Usage

```swift
// Before (unbounded dictionary)
var responseCache: [String: (response: String, timestamp: Date)] = [:]

// After (LRU with TTL)
let responseCache = StrictCache<String>(maxSize: 500, ttlSeconds: 8.0)
responseCache.set("key", value: response)
if let hit = responseCache.get("key") { ... }

// Factory methods
let cache = StrictCache<String>.responseCache()    // 500 items, 8s TTL
let topicCache = StrictCache<[String]>.topicCache() // 200 items, 120s TTL
```

### CircularBuffer Usage

```swift
// Before (unbounded array)
var conversationContext: [String] = []

// After (fixed capacity)
var conversationContext = CircularBuffer<String>(capacity: 100)
conversationContext.append("message")
let recent = conversationContext.last
let all = conversationContext.toArray()

// Factory methods
var context = CircularBuffer<String>.conversationContext()  // 100 items
var topics = CircularBuffer<String>.topicHistory()          // 50 items
```

### Logging Migration

```swift
// Before (unstructured)
print("⚠️ Engine failed: \(error)")

// After (structured os_log)
l104Log("Engine failed: \(error)", category: "engine")
l104LogError("Critical failure", category: "network")
l104LogInfo("Connection established", category: "network")
```

### Process Timeout

```swift
// Before (can hang)
let task = Process()
task.run()  // May hang forever

// After (guaranteed timeout)
let result = processWithTimeout("/usr/bin/curl", args: ["-s", url], timeoutSeconds: 30.0)
if let output = result.output {
    // success
} else {
    // timeout or error
}
```

### Re-entrancy Detection

```swift
// Before (potential deadlock with NSLock)
let lock = NSLock()
func activate() {
    lock.lock()
    registerCoreTests()  // May call activate again → deadlock
    lock.unlock()
}

// After (NSRecursiveLock)
let lock = NSRecursiveLock()
func activate() {
    lock.protected {
        registerCoreTests()  // Safe: recursive lock
    }
}
```

### Files Needing P1 Fixes

| File | Issue | Fix |
|------|-------|-----|
| `H02_L104StateCore.swift:286-290` | Unbounded cache | `StrictCache` |
| `H02_L104StateCore.swift:1094-1095` | Unbounded arrays | `CircularBuffer` |
| `H24_APIGateway.swift:1032-1044` | Process no timeout | `processWithTimeout()` |
| `VQPUMicroDaemon.swift:17,549` | No autoreleasepool | `runWithAutorelease()` |
| 35 files with `print()` | Unstructured logging | `l104Log()` |

---

## 13. State Files

| File | Purpose |
|------|---------|
| `~/.l104_daemon_orchestrator.json` | Daemon orchestrator state |
| `~/.l104_vqpu_daemon.json` | VQPU micro daemon state |
| `.l104_agent_history.json` | Agent system history |
| `.l104_agent_state.json` | Agent state persistence |
| `.l104_recursion_harvest.json` | Anti-recursion harvested energy |
| `lattice_v2.db` | Token lattice storage |
| `knowledge_graph.db` | Knowledge nodes/edges |
| `l104_unified.db` | Unified memory store |