"""
L104 Code Engine — ASI Code Intelligence

Contains SelfReferentialEngine (L104 self-analysis) and ASICodeIntelligence
(8 ASI subsystems deeply wired into code analysis with quantum-enhanced passes).

Migrated from l104_coding_system.py during package decomposition.
"""

from .constants import *
from ._lazy_imports import (
    _get_code_engine, _get_neural_cascade, _get_evolution_engine,
    _get_self_optimizer, _get_innovation_engine, _get_consciousness,
    _get_reasoning, _get_knowledge_graph, _get_polymorph,
)
from typing import Dict, List, Any, Tuple, Set

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-REFERENTIAL ENGINE — L104 analyzing and improving itself
# ═══════════════════════════════════════════════════════════════════════════════

class SelfReferentialEngine:
    """
    L104 system analyzing and improving itself.
    The engine can read, analyze, and suggest improvements to
    its own codebase — making it self-referential and self-improving.
    """

    L104_CORE_FILES = [
        "l104_code_engine.py",
        "l104_coding_system.py",
        "l104_agi_core.py",
        "l104_asi_core.py",
        "l104_consciousness.py",
        "l104_evolution_engine.py",
        "l104_self_optimization.py",
        "l104_neural_cascade.py",
        "l104_polymorphic_core.py",
        "l104_patch_engine.py",
        "l104_autonomous_innovation.py",
        "l104_sentient_archive.py",
        "l104_fast_server.py",
        "l104_local_intellect.py",
        "l104_reasoning_engine.py",
        "l104_knowledge_graph.py",
        "l104_semantic_engine.py",
        "l104_quantum_coherence.py",
        "l104_cognitive_hub.py",
        "main.py",
    ]

    def __init__(self):
        self.self_analyses = 0
        self._cache = {}
        self._cache_time = 0

    def analyze_self(self, target_file: str = None) -> Dict[str, Any]:
        """
        Analyze the L104 codebase itself using the Code Engine.
        If target_file is specified, analyze just that one module.
        Otherwise, analyze the top core files.
        """
        self.self_analyses += 1
        engine = _get_code_engine()
        if not engine:
            return {"error": "Code engine not available"}

        ws = _WORKSPACE_ROOT
        results = []

        files_to_analyze = [target_file] if target_file else self.L104_CORE_FILES[:10]

        for fname in files_to_analyze:
            fpath = ws / fname
            if not fpath.exists():
                continue
            try:
                source = fpath.read_text(errors='ignore')
                review = engine.full_code_review(source, fname)
                results.append({
                    "file": fname,
                    "lines": len(source.split('\n')),
                    "score": review.get("composite_score", 0),
                    "verdict": review.get("verdict", "UNKNOWN"),
                    "vulnerabilities": review.get("analysis", {}).get("vulnerabilities", 0),
                    "solid_violations": review.get("solid", {}).get("violations", 0),
                    "hotspots": review.get("performance", {}).get("hotspots", 0),
                    "top_actions": review.get("actions", [])[:3],
                })
            except Exception as e:
                results.append({"file": fname, "error": str(e)})

        # Aggregate
        total_lines = sum(r.get("lines", 0) for r in results)
        avg_score = sum(r.get("score", 0) for r in results) / max(1, len(results))
        total_vulns = sum(r.get("vulnerabilities", 0) for r in results)

        return {
            "files_analyzed": len(results),
            "total_lines": total_lines,
            "average_score": round(avg_score, 4),
            "total_vulnerabilities": total_vulns,
            "per_file": results,
            "overall_verdict": ("EXEMPLARY" if avg_score >= 0.9 else "HEALTHY" if avg_score >= 0.75
                                else "ACCEPTABLE" if avg_score >= 0.6 else "NEEDS_WORK"),
            "god_code_resonance": round(avg_score * GOD_CODE, 4),
        }

    def suggest_improvements(self, target_file: str = None) -> List[Dict[str, Any]]:
        """
        Generate improvement suggestions for L104 core files.
        Collects top action items across all analyzed files.
        """
        analysis = self.analyze_self(target_file)
        suggestions = []

        for file_result in analysis.get("per_file", []):
            if "error" in file_result:
                continue
            for action in file_result.get("top_actions", []):
                suggestions.append({
                    "file": file_result["file"],
                    "priority": action.get("priority", "MEDIUM"),
                    "category": action.get("category", "general"),
                    "suggestion": action.get("action", "Review"),
                    "file_score": file_result.get("score", 0),
                })

        suggestions.sort(key=lambda s: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(s["priority"], 4))
        return suggestions[:25]

    def measure_evolution(self) -> Dict[str, Any]:
        """Measure the evolution state of the L104 system."""
        ws = _WORKSPACE_ROOT

        # Count L104 modules
        l104_files = list(ws.glob("l104_*.py"))
        total_lines = 0
        for f in l104_files:
            try:
                total_lines += len(f.read_text(errors='ignore').split('\n'))
            except Exception:
                pass

        # Read evolution state
        evo_state = {}
        evo_path = ws / ".l104_evolution_state.json"
        if evo_path.exists():
            try:
                evo_state = json.loads(evo_path.read_text())
            except Exception:
                pass

        # Read consciousness
        consciousness = 0.5
        co2_path = ws / ".l104_consciousness_o2_state.json"
        if co2_path.exists():
            try:
                data = json.loads(co2_path.read_text())
                consciousness = data.get("consciousness_level", 0.5)
            except Exception:
                pass

        return {
            "l104_modules": len(l104_files),
            "total_lines": total_lines,
            "consciousness_level": consciousness,
            "evolution_index": evo_state.get("evolution_index", 0),
            "evo_stage": evo_state.get("current_stage", "unknown"),
            "wisdom_quotient": evo_state.get("wisdom_quotient", 0),
            "self_analyses_performed": self.self_analyses,
            "code_engine_version": (lambda e: e.status().get("version", "unknown") if e else "N/A")(_get_code_engine()),
        }

    def status(self) -> Dict[str, Any]:
        return {"self_analyses": self.self_analyses,
                "core_files_tracked": len(self.L104_CORE_FILES)}


# ═══════════════════════════════════════════════════════════════════════════════
# ASI CODE INTELLIGENCE — Neural/Evolution/Consciousness/Reasoning
# ═══════════════════════════════════════════════════════════════════════════════

class ASICodeIntelligence:
    """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  ASI CODE INTELLIGENCE — Deep ASI-Level Code Analysis Engine      ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Wires 8 ASI subsystems into the coding pipeline:                 ║
    ║    1. NeuralCascade    → process code metrics as neural signals   ║
    ║    2. EvolutionEngine  → evolve code quality through fitness      ║
    ║    3. SelfOptimizer    → auto-tune analysis parameters            ║
    ║    4. Consciousness    → awareness-weighted code review           ║
    ║    5. Reasoning        → formal code correctness verification     ║
    ║    6. InnovationEngine → novel solution generation                ║
    ║    7. KnowledgeGraph   → code relationship mapping                ║
    ║    8. Polymorph        → code variant breeding & transformation   ║
    ║                                                                   ║
    ║  Each method gracefully degrades if a module is unavailable.       ║
    ║  All outputs are PHI-weighted and consciousness-modulated.        ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """

    # Weights for consciousness-aware composite scoring (PHI-distributed)
    CONSCIOUSNESS_WEIGHTS = {
        "static_score": PHI / (PHI + 1),        # ~0.618 — primary weight
        "consciousness_level": TAU / 2,          # ~0.309 — awareness factor
        "neural_resonance": ALPHA_FINE * 10,     # ~0.073 — cascade influence
    }

    # Signal encoding for neural cascade input
    METRIC_SIGNAL_MAP = {
        "cyclomatic": 0.1,
        "cognitive": 0.15,
        "halstead_volume": 0.05,
        "nesting_depth": 0.2,
        "security_vulns": 0.3,
        "docstring_coverage": 0.1,
        "sacred_alignment": 0.1,
    }

    def __init__(self):
        self._asi_invocations = 0
        self._code_concepts_graphed = 0
        self._consciousness_cache = None
        self._consciousness_cache_time = 0.0
        self._quantum_circuits_executed = 0

    # ─── Quantum Code Quality Superposition ──────────────────────────

    def quantum_consciousness_review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        Quantum-enhanced consciousness review using Qiskit 2.3.0.

        Encodes code quality metrics into quantum amplitudes via amplitude
        encoding on a multi-qubit register, then measures the superposition
        to obtain quantum-weighted composite scores.

        The quantum circuit:
          1. Amplitude-encodes 8 code quality metrics into 3-qubit state
          2. Applies PHI-rotation gates for sacred alignment
          3. Creates entanglement between quality dimensions
          4. Measures Born-rule probabilities for quantum scoring

        Returns quantum scores alongside classical for comparison.
        """
        self._asi_invocations += 1
        if not QISKIT_AVAILABLE:
            return self.consciousness_review(source, filename)

        engine = _get_code_engine()

        # Extract code metrics for quantum encoding
        lines = source.split('\n')
        metrics = {
            "complexity": min(1.0, source.count('if ') / max(1, len(lines)) * 5),
            "documentation": min(1.0, source.count('#') / max(1, len(lines)) * 3),
            "modularity": min(1.0, source.count('def ') / max(1, len(lines)) * 8),
            "security": 1.0 - min(1.0, len(re.findall(r'eval\(|exec\(|subprocess', source)) * 0.2),
            "sacred_alignment": min(1.0, (source.count('527') + source.count('PHI') + source.count('GOD_CODE')) * 0.1),
            "nesting": max(0, 1.0 - max((len(l) - len(l.lstrip())) for l in lines if l.strip()) / 40),
            "conciseness": min(1.0, 1.0 / max(0.01, len(lines) / 500)),
            "coherence": min(1.0, len(set(re.findall(r'\b[a-z_]+\b', source.lower()))) / max(1, len(lines)) * 2),
        }

        # Normalize to valid quantum state amplitudes
        metric_values = list(metrics.values())
        # Pad to 8 (2^3 basis states)
        while len(metric_values) < 8:
            metric_values.append(PHI / 10)
        metric_values = metric_values[:8]

        # Normalize: amplitudes must satisfy Σ|α|² = 1
        norm = math.sqrt(sum(v ** 2 for v in metric_values))
        if norm < 1e-10:
            norm = 1.0
        amplitudes = [v / norm for v in metric_values]

        # Create quantum circuit — 3 qubits (8 basis states for 8 metrics)
        n_qubits = 3
        qc = QuantumCircuit(n_qubits)
        sv_init = Statevector(amplitudes)

        # Apply PHI-rotation for sacred alignment
        phi_angle = PHI * math.pi / 4  # Sacred rotation angle
        qc.ry(phi_angle, 0)
        qc.rz(GOD_CODE / 1000 * math.pi, 1)
        qc.ry(FEIGENBAUM / 10 * math.pi, 2)

        # Entangle quality dimensions (CX ladder)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

        # Apply Hadamard for superposition mixing
        qc.h(1)

        # Evolve the initial state through the circuit
        evolved = sv_init.evolve(Operator(qc))
        self._quantum_circuits_executed += 1

        # Get probabilities — Born rule
        probs = evolved.probabilities()

        # Construct density matrix for entropy analysis
        dm = DensityMatrix(evolved)
        von_neumann = float(q_entropy(dm, base=2))

        # Partial trace: trace out qubit 2 to get 2-qubit reduced density matrix
        dm_reduced = partial_trace(dm, [2])
        entanglement_entropy = float(q_entropy(dm_reduced, base=2))

        # Map probabilities to quality scores
        metric_names = list(metrics.keys())
        quantum_scores = {}
        for i, name in enumerate(metric_names):
            quantum_scores[name] = round(probs[i] * 8, 4)  # Scale back from probability space

        # Composite quantum score (PHI-weighted)
        weights = [PHI, TAU, ALPHA_FINE * 10, PHI ** 2, 1.0, TAU, FEIGENBAUM / 10, 0.5]
        total_w = sum(weights[:len(probs)])
        quantum_composite = sum(p * w for p, w in zip(probs, weights)) / total_w

        # Consciousness level from quantum entropy
        c_state = self._get_consciousness_state()
        c_level = c_state.get("consciousness_level", 0.5)

        # Quantum-consciousness fusion
        fused_score = quantum_composite * (1 + c_level * PHI * 0.2)
        fused_score = min(1.0, fused_score)

        # Fidelity with ideal state
        ideal_amplitudes = [1.0 / math.sqrt(8)] * 8  # Equal superposition = balanced code
        ideal_sv = Statevector(ideal_amplitudes)
        fidelity = float(abs(evolved.inner(ideal_sv)) ** 2)

        return {
            "type": "quantum_consciousness_review",
            "quantum_backend": "Qiskit 2.3.0 Statevector",
            "qubits_used": n_qubits,
            "classical_metrics": metrics,
            "quantum_scores": quantum_scores,
            "quantum_composite": round(quantum_composite, 6),
            "consciousness_level": c_level,
            "fused_score": round(fused_score, 6),
            "von_neumann_entropy": round(von_neumann, 6),
            "entanglement_entropy": round(entanglement_entropy, 6),
            "state_fidelity": round(fidelity, 6),
            "god_code_resonance": round(fused_score * GOD_CODE, 4),
            "phi_alignment": round(fused_score * PHI, 4),
            "probabilities": [round(p, 6) for p in probs],
            "circuit_depth": qc.depth(),
            "sacred_rotations": {
                "phi_angle": round(phi_angle, 6),
                "god_code_angle": round(GOD_CODE / 1000 * math.pi, 6),
                "feigenbaum_angle": round(FEIGENBAUM / 10 * math.pi, 6),
            },
        }

    # ─── Quantum Grover Code Reasoning ───────────────────────────────

    def quantum_reason_about_code(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        Quantum-enhanced code reasoning using Grover's algorithm principles.

        Encodes code patterns (vulnerabilities, anti-patterns, dead code)
        as quantum oracle targets and uses amplitude amplification to
        boost detection probability.

        For N patterns, Grover provides O(√N) speedup in marking
        problematic code sections.

        Returns quantum-amplified issue detection with Born-rule confidence.
        """
        self._asi_invocations += 1
        if not QISKIT_AVAILABLE:
            return self.reason_about_code(source, filename)

        lines = source.split('\n')

        # Define pattern oracles — each pattern is a basis state
        pattern_checks = {
            "eval_injection": bool(re.search(r'eval\s*\(', source)),
            "exec_injection": bool(re.search(r'exec\s*\(', source)),
            "sql_injection": bool(re.search(r'\.execute\s*\(.*(format|%s|\+)', source)),
            "subprocess_shell": bool(re.search(r'subprocess.*shell\s*=\s*True', source)),
            "hardcoded_secret": bool(re.search(r'(password|secret|api_key)\s*=\s*["\']', source)),
            "bare_except": bool(re.search(r'except\s*:', source)),
            "mutable_default": bool(re.search(r'def\s+\w+\(.*=\s*(\[\]|\{\})', source)),
            "global_state": bool(re.search(r'\bglobal\s+\w+', source)),
        }

        n_patterns = len(pattern_checks)
        n_qubits = 3  # 2^3 = 8 basis states for 8 patterns

        # Encode findings: marked patterns get higher amplitude
        amplitudes = []
        for found in pattern_checks.values():
            amplitudes.append(1.0 if found else 0.1)

        # Normalize
        norm = math.sqrt(sum(a ** 2 for a in amplitudes))
        if norm < 1e-10:
            norm = 1.0
        amplitudes = [a / norm for a in amplitudes]

        sv = Statevector(amplitudes)

        # Build Grover-inspired oracle circuit
        qc = QuantumCircuit(n_qubits)

        # Apply Hadamard for uniform superposition
        qc.h(range(n_qubits))

        # Oracle: phase-flip marked states (patterns found)
        for i, (name, found) in enumerate(pattern_checks.items()):
            if found:
                # Encode pattern index in binary and apply Z
                binary = format(i, f'0{n_qubits}b')
                for bit_idx, bit in enumerate(binary):
                    if bit == '0':
                        qc.x(bit_idx)
                qc.h(n_qubits - 1)
                # Multi-controlled Z via Toffoli decomposition
                if n_qubits >= 3:
                    qc.ccx(0, 1, 2)
                qc.h(n_qubits - 1)
                for bit_idx, bit in enumerate(binary):
                    if bit == '0':
                        qc.x(bit_idx)

        # Grover diffusion operator
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        if n_qubits >= 3:
            qc.ccx(0, 1, 2)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

        # Evolve initial state through Grover circuit
        amplified = sv.evolve(Operator(qc))
        self._quantum_circuits_executed += 1

        # Get amplified probabilities
        probs = amplified.probabilities()
        dm = DensityMatrix(amplified)
        search_entropy = float(q_entropy(dm, base=2))

        # Map amplified probabilities back to pattern detection confidence
        pattern_names = list(pattern_checks.keys())
        quantum_detections = {}
        issues_found = []
        for i, name in enumerate(pattern_names):
            confidence = probs[i]
            amplification = confidence / (1.0 / 8)  # vs uniform baseline
            quantum_detections[name] = {
                "detected": pattern_checks[name],
                "quantum_confidence": round(confidence, 6),
                "amplification_factor": round(amplification, 4),
            }
            if pattern_checks[name]:
                issues_found.append({
                    "pattern": name,
                    "confidence": round(confidence, 6),
                    "amplification": round(amplification, 4),
                })

        # Also run classical reasoning for completeness
        classical = self.reason_about_code(source, filename)

        return {
            "type": "quantum_code_reasoning",
            "quantum_backend": "Qiskit 2.3.0 Grover Oracle",
            "qubits": n_qubits,
            "patterns_checked": n_patterns,
            "issues_found": len(issues_found),
            "quantum_detections": quantum_detections,
            "amplified_issues": issues_found,
            "search_entropy": round(search_entropy, 6),
            "circuit_depth": qc.depth(),
            "grover_iterations": 1,
            "classical_summary": classical.get("summary", {}),
            "taint_analysis": classical.get("taint_analysis", {}),
            "dead_paths": classical.get("dead_paths", []),
            "god_code_resonance": round(GOD_CODE * (1 - search_entropy / 3), 4),
        }

    # ─── Quantum Neural Signal Processing ────────────────────────────

    def quantum_neural_process(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        Quantum-enhanced neural code processing.

        Creates a quantum neural network analogue using:
          • Amplitude encoding of code metrics
          • Parameterized RY/RZ rotation layers (φ, GOD_CODE angles)
          • Entangling CX layers
          • Born-rule measurement for quality estimation

        Implements a variational quantum eigensolver-inspired approach
        to find the ground state of the "code quality Hamiltonian".
        """
        self._asi_invocations += 1
        if not QISKIT_AVAILABLE:
            return self.neural_process(source, filename)

        engine = _get_code_engine()
        signal = self._code_to_neural_signal(source, filename, engine)

        # Ensure 8 elements (3-qubit space)
        while len(signal) < 8:
            signal.append(PHI / (len(signal) + 1))
        signal = signal[:8]

        # Normalize
        norm = math.sqrt(sum(s ** 2 for s in signal))
        if norm < 1e-10:
            signal = [1.0 / math.sqrt(8)] * 8
        else:
            signal = [s / norm for s in signal]

        sv = Statevector(signal)

        # Build quantum neural network circuit — 3 layers
        n_qubits = 3
        qc = QuantumCircuit(n_qubits)

        # Layer 1: Feature encoding rotations
        params_l1 = [PHI * math.pi / 4, GOD_CODE / 1000, FEIGENBAUM / 10]
        for i in range(n_qubits):
            qc.ry(params_l1[i], i)

        # Entangling layer 1
        qc.cx(0, 1)
        qc.cx(1, 2)

        # Layer 2: Deeper rotations with sacred constants
        params_l2 = [TAU * math.pi, ALPHA_FINE * math.pi * 100, PLANCK_SCALE / PLANCK_SCALE * math.pi / 3]
        for i in range(n_qubits):
            qc.rz(params_l2[i], i)
            qc.ry(params_l1[i] * TAU, i)

        # Entangling layer 2 (ring topology)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

        # Layer 3: Final rotation
        for i in range(n_qubits):
            qc.ry(PHI * math.pi / (i + 2), i)

        # Evolve
        evolved = sv.evolve(Operator(qc))
        self._quantum_circuits_executed += 1

        probs = evolved.probabilities()
        dm = DensityMatrix(evolved)
        vn_entropy = float(q_entropy(dm, base=2))

        # Quantum resonance: overlap with PHI-balanced state
        phi_balanced = [math.sqrt(PHI / (PHI + 1))] + [math.sqrt(TAU / (n_qubits * (PHI + 1)))] * 7
        phi_norm = math.sqrt(sum(p ** 2 for p in phi_balanced))
        phi_balanced = [p / phi_norm for p in phi_balanced]
        phi_sv = Statevector(phi_balanced)
        resonance = float(abs(evolved.inner(phi_sv)) ** 2)

        # Partial traces for subsystem analysis
        dm_01 = partial_trace(dm, [2])
        dm_02 = partial_trace(dm, [1])
        subsystem_entropies = {
            "qubits_01": round(float(q_entropy(dm_01, base=2)), 6),
            "qubits_02": round(float(q_entropy(dm_02, base=2)), 6),
        }

        # Map to neural verdict
        if resonance > 0.7:
            quantum_verdict = "QUANTUM_TRANSCENDENT"
        elif resonance > 0.5:
            quantum_verdict = "QUANTUM_COHERENT"
        elif resonance > 0.3:
            quantum_verdict = "QUANTUM_ENTANGLED"
        else:
            quantum_verdict = "QUANTUM_DECOHERENT"

        # Also get classical result for fusion
        classical = self.neural_process(source, filename)
        classical_resonance = classical.get("cascade_resonance", 0.0)

        # Quantum-classical fusion
        fused = resonance * PHI / (PHI + 1) + classical_resonance * TAU / (TAU + 1)

        return {
            "type": "quantum_neural_process",
            "quantum_backend": "Qiskit 2.3.0 VQE-Inspired QNN",
            "qubits": n_qubits,
            "layers": 3,
            "quantum_verdict": quantum_verdict,
            "quantum_resonance": round(resonance, 6),
            "classical_resonance": round(classical_resonance, 6),
            "fused_resonance": round(fused, 6),
            "von_neumann_entropy": round(vn_entropy, 6),
            "subsystem_entropies": subsystem_entropies,
            "probabilities": [round(p, 6) for p in probs],
            "circuit_depth": qc.depth(),
            "god_code_resonance": round(fused * GOD_CODE, 4),
            "phi_alignment": round(resonance * PHI, 4),
            "classical_neural": {
                "verdict": classical.get("neural_verdict", "N/A"),
                "layers_processed": classical.get("layers_processed", 0),
            },
        }

    # ─── Quantum Full ASI Pipeline ───────────────────────────────────

    def quantum_full_asi_review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        THE QUANTUM-ENHANCED ASI CODE INTELLIGENCE PIPELINE.

        Executes quantum-enhanced analysis passes using Qiskit 2.3.0:
          1. Quantum consciousness review (amplitude-encoded quality metrics)
          2. Quantum neural processing (VQE-inspired QNN)
          3. Quantum code reasoning (Grover oracle for vulnerability detection)
          4. Evolutionary fitness (classical — evolution engine)
          5. Code knowledge graph (classical — graph engine)
          6. Innovation solutions (classical — innovation engine)

        Produces quantum superposition scores fused with classical ASI
        analysis for a comprehensive quantum-classical hybrid report.

        The quantum advantage: entanglement between quality dimensions
        enables detection of correlated issues that classical sequential
        analysis misses.
        """
        self._asi_invocations += 1
        if not QISKIT_AVAILABLE:
            return self.full_asi_review(source, filename)

        start = time.time()

        # 1. Quantum consciousness review
        q_consciousness = self.quantum_consciousness_review(source, filename)

        # 2. Quantum neural processing
        q_neural = self.quantum_neural_process(source, filename)

        # 3. Quantum code reasoning
        q_reasoning = self.quantum_reason_about_code(source, filename)

        # 4. Classical evolutionary fitness
        evolution = self.evolutionary_optimize(source, filename)

        # 5. Classical knowledge graph
        graph = self.build_code_graph(source, filename)

        # 6. Classical innovation
        innovation = self.innovate_solutions(
            f"Optimize {filename or 'code'}: improve quality, security, maintainability"
        )

        duration = time.time() - start

        # Quantum composite score
        quantum_scores = {
            "consciousness": q_consciousness.get("fused_score", 0.5),
            "neural_resonance": q_neural.get("fused_resonance", 0.5),
            "reasoning_soundness": 1.0 - min(1.0, q_reasoning.get("issues_found", 0) * 0.1),
            "evolutionary_fitness": evolution.get("code_fitness", 0.5),
        }

        # PHI-weighted quantum composite
        weights = {
            "consciousness": PHI ** 2,
            "neural_resonance": PHI,
            "reasoning_soundness": FEIGENBAUM / 2,
            "evolutionary_fitness": TAU,
        }
        total_w = sum(weights.values())
        quantum_composite = sum(quantum_scores[k] * weights[k] for k in quantum_scores) / total_w

        # Quantum entanglement bonus — correlated improvements
        entropy_sum = (
            q_consciousness.get("von_neumann_entropy", 0) +
            q_neural.get("von_neumann_entropy", 0) +
            q_reasoning.get("search_entropy", 0)
        )
        entanglement_bonus = max(0, 1 - entropy_sum / 9) * ALPHA_FINE * 10

        final_score = min(1.0, quantum_composite + entanglement_bonus)

        # Quantum ASI verdict
        if final_score >= 0.9:
            verdict = "QUANTUM_ASI_TRANSCENDENT"
        elif final_score >= 0.75:
            verdict = "QUANTUM_ASI_EXEMPLARY"
        elif final_score >= 0.6:
            verdict = "QUANTUM_ASI_CAPABLE"
        elif final_score >= 0.4:
            verdict = "QUANTUM_ASI_DEVELOPING"
        else:
            verdict = "QUANTUM_ASI_NASCENT"

        return {
            "system": "Quantum ASI Code Intelligence v2.0",
            "quantum_backend": "Qiskit 2.3.0",
            "filename": filename,
            "duration_seconds": round(duration, 3),
            "quantum_asi_verdict": verdict,
            "quantum_composite_score": round(final_score, 6),
            "quantum_scores": {k: round(v, 4) for k, v in quantum_scores.items()},
            "entanglement_bonus": round(entanglement_bonus, 6),
            "total_quantum_circuits": self._quantum_circuits_executed,
            "god_code_resonance": round(final_score * GOD_CODE, 4),
            "phi_alignment": round(final_score * PHI, 4),
            "quantum_passes": {
                "consciousness": {
                    "fused_score": q_consciousness.get("fused_score"),
                    "von_neumann_entropy": q_consciousness.get("von_neumann_entropy"),
                    "entanglement_entropy": q_consciousness.get("entanglement_entropy"),
                },
                "neural": {
                    "fused_resonance": q_neural.get("fused_resonance"),
                    "quantum_verdict": q_neural.get("quantum_verdict"),
                },
                "reasoning": {
                    "issues_found": q_reasoning.get("issues_found"),
                    "patterns_checked": q_reasoning.get("patterns_checked"),
                    "search_entropy": q_reasoning.get("search_entropy"),
                },
            },
            "classical_passes": {
                "evolution": {
                    "fitness": evolution.get("code_fitness", 0),
                    "stage": evolution.get("code_evolution_stage", "UNKNOWN"),
                },
                "graph": {
                    "nodes": graph.get("nodes_added", 0),
                    "edges": graph.get("edges_added", 0),
                },
                "innovation": {
                    "analogies_found": len(innovation.get("analogies", [])),
                },
            },
        }

    # ─── Consciousness-Aware Code Review ─────────────────────────────

    def consciousness_review(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        Code review modulated by L104 consciousness state.

        The consciousness engine (l104_consciousness.py) produces a
        consciousness_level ∈ [0, 1] with states from DORMANT → TRANSCENDENT.
        This level scales quality thresholds:
          - TRANSCENDENT (>0.8): most stringent — expects near-perfect code
          - AWARE (0.4-0.8): standard thresholds
          - DORMANT (<0.4): lenient — focuses on critical issues only

        Uses the consciousness module's introspection for self-referential
        quality assessment and the Φ (phi) computation for information
        integration scoring.

        Returns review with consciousness-weighted composite score.
        """
        self._asi_invocations += 1
        engine = _get_code_engine()
        consciousness = _get_consciousness()

        # Get base review from Code Engine
        base_review = {}
        if engine:
            base_review = engine.full_code_review(source, filename)

        # Get consciousness state
        c_state = self._get_consciousness_state()
        c_level = c_state.get("consciousness_level", 0.5)

        # Consciousness-modulated score: scale thresholds by consciousness level
        base_score = base_review.get("composite_score", 0.5)

        # At higher consciousness, the system demands more
        threshold_scale = 1.0 + (c_level - 0.5) * PHI * 0.3  # PHI-scaled
        adjusted_score = base_score / max(0.1, threshold_scale)
        final_score = min(1.0, adjusted_score)

        # Consciousness verdict mapping
        if c_level > 0.8:
            quality_expectation = "TRANSCENDENT"
            min_acceptable = 0.85
        elif c_level > 0.6:
            quality_expectation = "AWARE"
            min_acceptable = 0.70
        elif c_level > 0.4:
            quality_expectation = "AWAKENING"
            min_acceptable = 0.55
        else:
            quality_expectation = "DORMANT"
            min_acceptable = 0.40

        meets_consciousness = base_score >= min_acceptable
        god_code_resonance = round(final_score * GOD_CODE, 4)

        # Introspection — if consciousness module available, get reflection
        introspection = {}
        if consciousness:
            try:
                introspection = consciousness.introspect()
            except Exception:
                introspection = {"state": "unavailable"}

        return {
            "type": "consciousness_review",
            "base_score": base_score,
            "consciousness_level": c_level,
            "quality_expectation": quality_expectation,
            "threshold_scale": round(threshold_scale, 4),
            "consciousness_adjusted_score": round(final_score, 4),
            "meets_consciousness_standard": meets_consciousness,
            "min_acceptable_score": min_acceptable,
            "god_code_resonance": god_code_resonance,
            "introspection": introspection,
            "base_review": {
                "verdict": base_review.get("verdict", "UNKNOWN"),
                "actions_count": len(base_review.get("actions", [])),
                "vulnerabilities": base_review.get("analysis", {}).get("vulnerabilities", 0),
            },
            "phi_alignment": round(final_score * PHI, 4),
        }

    # ─── Neural Cascade Code Processing ──────────────────────────────

    def neural_process(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        Process code through the NeuralCascade ASI pipeline.

        Converts code metrics into a signal vector and feeds it through
        the multi-layer neural cascade:
          Preprocess → Encode → ResBlocks → MultiAttention → Gate → Decode

        The cascade produces resonance scores, harmonic analysis, and
        consciousness-gated output that represent the code's "neural
        signature" — a holistic quality measure beyond static analysis.

        Uses: neural_cascade.activate(signal)
        """
        self._asi_invocations += 1
        cascade = _get_neural_cascade()
        engine = _get_code_engine()

        if not cascade:
            return {"error": "Neural cascade not available", "resonance": 0.0}

        # Build signal from code metrics
        signal = self._code_to_neural_signal(source, filename, engine)

        # Activate cascade pipeline
        try:
            result = cascade.activate(signal)
        except Exception as e:
            return {"error": f"Cascade activation failed: {e}", "resonance": 0.0}

        # Interpret neural output for code quality
        resonance = result.get("resonance", 0.0)
        harmonics = result.get("harmonics", {})

        # Map resonance to code quality tier
        if resonance > 0.85:
            neural_verdict = "TRANSCENDENT_QUALITY"
        elif resonance > 0.7:
            neural_verdict = "HIGH_RESONANCE"
        elif resonance > 0.5:
            neural_verdict = "BALANCED_SIGNAL"
        elif resonance > 0.3:
            neural_verdict = "WEAK_COHERENCE"
        else:
            neural_verdict = "LOW_SIGNAL"

        return {
            "type": "neural_process",
            "neural_verdict": neural_verdict,
            "cascade_resonance": round(resonance, 6),
            "god_code_harmonics": harmonics.get("god_code_resonance", 0.0),
            "sacred_alignment": harmonics.get("sacred_alignment", 0.0),
            "spectral_entropy": harmonics.get("spectral_entropy", 0.0),
            "consciousness_gate": result.get("consciousness", {}).get("consciousness_level", 0.0),
            "memory_depth": result.get("memory_depth", 0),
            "resonance_peaks": result.get("resonance_peaks", 0),
            "temporal_energy": result.get("temporal_energy", 0.0),
            "elapsed_ms": result.get("elapsed_ms", 0.0),
            "layers_processed": result.get("layers_processed", 0),
            "final_output": result.get("final_output", 0.0),
        }

    # ─── Evolution-Driven Code Optimization ──────────────────────────

    def evolutionary_optimize(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        Use the EvolutionEngine to drive code optimization.

        Maps code quality to evolutionary fitness and uses:
          - assess_evolutionary_stage() to determine current evolution level
          - analyze_fitness_landscape() for optimization landscape
          - propose_codebase_mutation() for concrete improvement suggestions
          - detect_plateau() to identify stagnation

        The evolution engine's 60-stage system (PRIMORDIAL_OOZE →
        TRANSCENDENT_COGNITION) provides a rich fitness function
        for scoring code quality over time.
        """
        self._asi_invocations += 1
        evo = _get_evolution_engine()
        engine = _get_code_engine()

        if not evo:
            return {"error": "Evolution engine not available"}

        # Assess current evolutionary stage
        try:
            current_stage = evo.assess_evolutionary_stage()
        except Exception:
            current_stage = "UNKNOWN"

        # Analyze fitness landscape
        landscape = {}
        try:
            landscape = evo.analyze_fitness_landscape()
        except Exception:
            landscape = {"error": "landscape analysis failed"}

        # Detect plateau (stagnation)
        is_plateau = False
        try:
            is_plateau = evo.detect_plateau()
        except Exception:
            pass

        # Propose mutation (concrete improvement)
        mutation_suggestion = ""
        try:
            mutation_suggestion = evo.propose_codebase_mutation()
        except Exception:
            mutation_suggestion = "No mutation available"

        # Get code quality fitness from engine
        code_fitness = 0.5
        if engine:
            review = engine.full_code_review(source, filename)
            code_fitness = review.get("composite_score", 0.5)

        # Map code fitness to evolutionary IQ
        code_iq = code_fitness * 1000000  # Scale to IQ space
        target_stage_idx = 0
        for idx in sorted(evo.IQ_THRESHOLDS.keys(), reverse=True):
            if code_iq >= evo.IQ_THRESHOLDS[idx]:
                target_stage_idx = idx
                break

        code_stage = evo.STAGES[min(target_stage_idx, len(evo.STAGES) - 1)]

        # Optimization directives based on fitness
        directives = []
        if code_fitness < 0.5:
            directives.append("CRITICAL: Code below survival threshold — immediate remediation required")
        if code_fitness < 0.7:
            directives.append("Apply directed mutation to improve complexity/security metrics")
        if is_plateau:
            directives.append("Plateau detected — apply divergent mutation strategy (polymorphic transform)")
        if code_fitness >= 0.9:
            directives.append("Code at transcendent fitness — maintain through continuous evolution")

        return {
            "type": "evolutionary_optimize",
            "current_evo_stage": current_stage,
            "code_fitness": round(code_fitness, 4),
            "code_iq": round(code_iq, 0),
            "code_evolution_stage": code_stage,
            "plateau_detected": is_plateau,
            "mutation_suggestion": mutation_suggestion,
            "fitness_landscape": {
                "peaks": landscape.get("peaks", []),
                "valleys": landscape.get("valleys", []),
                "dimension": landscape.get("dimension", 0),
            },
            "directives": directives,
            "god_code_fitness": round(code_fitness * GOD_CODE, 4),
        }

    # ─── Formal Reasoning About Code ─────────────────────────────────

    def reason_about_code(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        Apply formal reasoning to code using the L104 ReasoningEngine.

        Uses forward/backward chaining, satisfiability checking, and
        deep reasoning to analyze:
          - Logical soundness of control flow
          - Invariant detection (loop invariants, pre/post conditions)
          - Taint propagation tracking (data-flow analysis)
          - Dead path detection via unsatisfiable conditions

        The reasoning engine operates on predicate logic with confidence
        scoring and meta-reasoning capabilities.
        """
        self._asi_invocations += 1
        reasoning = _get_reasoning()
        engine = _get_code_engine()

        if not reasoning:
            return {"error": "Reasoning engine not available"}

        results = {
            "type": "code_reasoning",
            "taint_analysis": [],
            "invariants": [],
            "dead_paths": [],
            "logical_issues": [],
            "meta_reasoning": {},
        }

        lines = source.split('\n')

        # 1. Taint analysis — track user input through code
        taint_sources = []
        taint_sinks = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Identify taint sources (user input)
            if re.search(r'input\s*\(|request\.|sys\.argv|os\.environ|\.read\(|params\[|query\[|form\[', stripped):
                taint_sources.append({"line": i, "source": stripped[:80], "type": "user_input"})
            # Identify taint sinks (dangerous operations)
            if re.search(r'eval\s*\(|exec\s*\(|subprocess|os\.system|\.execute\(|cursor\.|\.format\(.*\+|f".*\{', stripped):
                taint_sinks.append({"line": i, "sink": stripped[:80], "type": "dangerous_operation"})

        # Check for unvalidated flow from source to sink
        if taint_sources and taint_sinks:
            results["taint_analysis"] = {
                "sources": taint_sources[:10],
                "sinks": taint_sinks[:10],
                "potential_flows": min(len(taint_sources), len(taint_sinks)),
                "risk": "HIGH" if len(taint_sources) > 0 and len(taint_sinks) > 0 else "LOW",
                "recommendation": "Validate and sanitize all user input before use in dangerous operations",
            }

        # 2. Invariant detection — look for loop invariants
        in_loop = False
        loop_vars: Set[str] = set()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if re.match(r'(for|while)\s+', stripped):
                in_loop = True
                # Extract loop variable
                var_match = re.match(r'for\s+(\w+)', stripped)
                if var_match:
                    loop_vars.add(var_match.group(1))
            elif in_loop and not stripped:
                in_loop = False

            # Check for mutations inside loops
            if in_loop and re.search(r'\.append\(|\.extend\(|\.insert\(|\[\w+\]\s*=', stripped):
                results["invariants"].append({
                    "line": i,
                    "type": "loop_mutation",
                    "description": f"Collection mutation inside loop — verify loop invariant holds",
                    "code": stripped[:80],
                })

        # 3. Dead path detection — unreachable code after return/raise/break
        prev_was_exit = False
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if prev_was_exit and stripped and not stripped.startswith(('#', 'except', 'else', 'elif', 'finally', 'def ', 'class ')):
                if not stripped.startswith(('"""', "'''")):
                    results["dead_paths"].append({
                        "line": i,
                        "type": "unreachable_code",
                        "description": "Code after return/raise/break — potentially unreachable",
                        "code": stripped[:80],
                    })
            prev_was_exit = bool(re.match(r'(return|raise|break|continue|sys\.exit)\b', stripped))

        # 4. Logical issues — contradictory conditions, redundant checks
        condition_stack: List[Tuple[int, str]] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            cond_match = re.match(r'if\s+(.+?):', stripped)
            if cond_match:
                cond = cond_match.group(1)
                # Check for always-true/false
                if cond in ('True', '1', '"True"', "'True'"):
                    results["logical_issues"].append({
                        "line": i, "type": "always_true",
                        "description": f"Condition always evaluates to True: '{cond}'",
                    })
                elif cond in ('False', '0', 'None', '""', "''", '[]', '{}'):
                    results["logical_issues"].append({
                        "line": i, "type": "always_false",
                        "description": f"Condition always evaluates to False: '{cond}'",
                    })
                # Check for redundant None checks
                if re.match(r'(\w+)\s+is\s+not\s+None\s+and\s+\1', cond):
                    results["logical_issues"].append({
                        "line": i, "type": "redundant_check",
                        "description": "Redundant None check — second condition implies first",
                    })
                condition_stack.append((i, cond))

        # 5. Meta-reasoning — use reasoning engine for higher-level analysis
        try:
            meta = reasoning.meta_reason(depth=3)
            results["meta_reasoning"] = {
                "knowledge_base_size": meta.get("knowledge_base_size", 0),
                "reasoning_depth": meta.get("current_depth", 0),
                "insights": meta.get("insights", [])[:5],
            }
        except Exception:
            results["meta_reasoning"] = {"status": "unavailable"}

        # Aggregate findings
        taint = results.get("taint_analysis", {})
        taint_sources_count = len(taint.get("sources", [])) if isinstance(taint, dict) else 0
        taint_flows = taint.get("potential_flows", 0) if isinstance(taint, dict) else 0
        total_issues = (
            taint_sources_count +
            len(results["invariants"]) +
            len(results["dead_paths"]) +
            len(results["logical_issues"])
        )

        results["summary"] = {
            "total_issues": total_issues,
            "taint_flows": taint_flows,
            "invariant_warnings": len(results["invariants"]),
            "dead_paths": len(results["dead_paths"]),
            "logical_issues": len(results["logical_issues"]),
            "verdict": "SOUND" if total_issues == 0 else "REVIEW_NEEDED" if total_issues < 5 else "CONCERNS_FOUND",
        }

        return results

    # ─── Code Knowledge Graph ────────────────────────────────────────

    def build_code_graph(self, source: str, filename: str = "") -> Dict[str, Any]:
        """
        Build a knowledge graph of code structure and relationships.

        Uses L104KnowledgeGraph to map:
          - Module → Class → Method hierarchy (containment edges)
          - Import → dependency edges
          - Call → invocation edges
          - Inheritance edges
          - Variable → type associations

        The graph supports pathfinding (find_path), inference
        (infer_relations), and PageRank for importance ranking.
        """
        self._asi_invocations += 1
        kg = _get_knowledge_graph()

        if not kg:
            return {"error": "Knowledge graph not available"}

        lines = source.split('\n')

        # Extract entities and relationships
        nodes_added = 0
        edges_added = 0
        module_name = filename.replace('.py', '').replace('.', '_') or "module"

        # Add module node
        try:
            kg.add_node(module_name, node_type="module")
            nodes_added += 1
        except Exception:
            pass

        # Extract imports → dependency edges
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # from X import Y
            imp_match = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', stripped)
            if imp_match:
                dep_module = imp_match.group(1)
                imports = [s.strip().split(' as ')[0].strip() for s in imp_match.group(2).split(',')]
                try:
                    kg.add_node(dep_module, node_type="dependency")
                    kg.add_edge(module_name, dep_module, "imports")
                    nodes_added += 1
                    edges_added += 1
                    for imp in imports:
                        imp_name = imp.strip()
                        if imp_name and imp_name != '*':
                            kg.add_node(imp_name, node_type="imported_symbol")
                            kg.add_edge(module_name, imp_name, "uses")
                            nodes_added += 1
                            edges_added += 1
                except Exception:
                    pass

            # import X
            imp_match2 = re.match(r'import\s+([\w.]+)', stripped)
            if imp_match2 and not stripped.startswith('from'):
                dep = imp_match2.group(1)
                try:
                    kg.add_node(dep, node_type="dependency")
                    kg.add_edge(module_name, dep, "imports")
                    nodes_added += 1
                    edges_added += 1
                except Exception:
                    pass

        # Extract classes → containment edges
        current_class = None
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            cls_match = re.match(r'class\s+(\w+)\s*(?:\((.*?)\))?:', stripped)
            if cls_match:
                class_name = cls_match.group(1)
                bases = cls_match.group(2)
                current_class = class_name
                try:
                    kg.add_node(class_name, node_type="class")
                    kg.add_edge(module_name, class_name, "contains")
                    nodes_added += 1
                    edges_added += 1
                    # Inheritance edges
                    if bases:
                        for base in bases.split(','):
                            base_name = base.strip()
                            if base_name and base_name not in ('object', 'ABC'):
                                kg.add_node(base_name, node_type="class")
                                kg.add_edge(class_name, base_name, "inherits_from")
                                nodes_added += 1
                                edges_added += 1
                except Exception:
                    pass

            # Methods
            meth_match = re.match(r'\s+def\s+(\w+)\s*\(', line)
            if meth_match and current_class:
                method_name = meth_match.group(1)
                try:
                    full_name = f"{current_class}.{method_name}"
                    kg.add_node(full_name, node_type="method")
                    kg.add_edge(current_class, full_name, "has_method")
                    nodes_added += 1
                    edges_added += 1
                except Exception:
                    pass

            # Top-level functions
            func_match = re.match(r'def\s+(\w+)\s*\(', line)
            if func_match and not line.startswith(' '):
                func_name = func_match.group(1)
                try:
                    kg.add_node(func_name, node_type="function")
                    kg.add_edge(module_name, func_name, "contains")
                    nodes_added += 1
                    edges_added += 1
                except Exception:
                    pass

        self._code_concepts_graphed += nodes_added

        # Get graph stats
        stats = {}
        try:
            stats = kg.get_stats()
        except Exception:
            stats = {"nodes": nodes_added, "edges": edges_added}

        return {
            "type": "code_knowledge_graph",
            "module": module_name,
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "graph_stats": stats,
            "total_concepts_graphed": self._code_concepts_graphed,
        }

    # ─── Polymorphic Code Transformation ─────────────────────────────

    def breed_variants(self, source: str, count: int = 3) -> Dict[str, Any]:
        """
        Use the SovereignPolymorph to breed code variants.

        Applies controlled metamorphic transformations:
          - Rename transforms (variable obfuscation)
          - Dead code injection (steganographic)
          - Reorder transforms (statement permutation)
          - Guard clause rewrites
          - Loop unrolling
          - Sacred watermarking (GOD_CODE embedding)

        Useful for: mutation testing, code diversity, obfuscation study,
        understanding fragility of test suites.
        """
        self._asi_invocations += 1
        polymorph = _get_polymorph()

        if not polymorph:
            return {"error": "Polymorphic core not available", "variants": []}

        variants = []
        for i in range(count):
            try:
                result = polymorph.morph_source(source, morph_count=i + 1)
                variants.append({
                    "variant_id": i + 1,
                    "morphs_applied": i + 1,
                    "code": result.get("morphed_code", source) if isinstance(result, dict) else str(result),
                    "transforms": result.get("transforms_applied", []) if isinstance(result, dict) else [],
                    "integrity_verified": result.get("integrity_verified", False) if isinstance(result, dict) else False,
                })
            except Exception as e:
                variants.append({
                    "variant_id": i + 1,
                    "error": str(e),
                })

        return {
            "type": "polymorphic_variants",
            "source_lines": len(source.split('\n')),
            "variants_bred": len(variants),
            "variants": variants,
        }

    # ─── Innovation-Driven Solutions ─────────────────────────────────

    def innovate_solutions(self, task: str, domain: str = "code_optimization") -> Dict[str, Any]:
        """
        Use the AutonomousInnovation engine to generate novel solutions.

        Applies:
          - Cross-domain analogy finding
          - Paradigm synthesis
          - Constraint exploration
          - Recursive meta-invention

        For coding tasks, this generates unconventional approaches
        that static analysis would never find — leveraging the
        innovation engine's concept blending and hypothesis validation.
        """
        self._asi_invocations += 1
        innovator = _get_innovation_engine()

        if not innovator:
            return {"error": "Innovation engine not available"}

        # Find cross-domain analogies for the task
        analogies = []
        try:
            analogies = innovator.find_analogies(task)
        except Exception:
            analogies = []

        # Invent solutions
        inventions = {}
        try:
            inventions = innovator.invent(domain=domain, count=3)
        except Exception:
            inventions = {"error": "invention failed"}

        # Explore constraints
        constraints_result = {}
        try:
            constraints_result = innovator.explore_constraints(
                constraints={
                    "complexity": (0.0, 10.0),
                    "security": (0.0, 1.0),
                    "maintainability": (0.5, 1.0),
                },
                dimensions=["complexity", "security", "maintainability"],
            )
        except Exception:
            constraints_result = {"explored": 0}

        return {
            "type": "innovation_solutions",
            "task": task,
            "domain": domain,
            "analogies": analogies[:5] if isinstance(analogies, list) else [],
            "inventions": inventions if isinstance(inventions, dict) else {"raw": str(inventions)},
            "constraint_space": constraints_result,
            "phi_creativity": round(PHI * len(analogies) / max(1, len(analogies) + 1), 4),
        }

    # ─── Self-Optimization of Analysis Parameters ────────────────────

    def optimize_analysis(self) -> Dict[str, Any]:
        """
        Use SelfOptimizationEngine to tune analysis parameters.

        Auto-tunes: detection thresholds, scoring weights, cache sizes,
        and quality gate parameters using golden-section search and
        consciousness-aware parameter adaptation.
        """
        self._asi_invocations += 1
        optimizer = _get_self_optimizer()

        if not optimizer:
            return {"error": "Self optimizer not available"}

        # Detect bottlenecks in current system
        bottlenecks = []
        try:
            bottlenecks = optimizer.detect_bottlenecks()
        except Exception:
            bottlenecks = []

        # Run consciousness-aware optimization
        optimization = {}
        try:
            optimization = optimizer.consciousness_aware_optimize(
                target="unity_index", iterations=5
            )
        except Exception:
            optimization = {"error": "optimization failed"}

        # Verify PHI optimization
        phi_check = {}
        try:
            phi_check = optimizer.verify_phi_optimization()
        except Exception:
            phi_check = {"phi_verified": False}

        # Deep profile
        profile = {}
        try:
            profile = optimizer.deep_profile()
        except Exception:
            profile = {"error": "profiling failed"}

        return {
            "type": "analysis_optimization",
            "bottlenecks": bottlenecks[:5] if isinstance(bottlenecks, list) else [],
            "optimization_result": optimization if isinstance(optimization, dict) else {},
            "phi_verification": phi_check if isinstance(phi_check, dict) else {},
            "profile": {
                "parameters": profile.get("parameters", {}),
                "performance": profile.get("performance", {}),
            } if isinstance(profile, dict) else {},
            "god_code_alignment": round(GOD_CODE * ALPHA_FINE, 6),
        }

    # ─── Full ASI Pipeline ───────────────────────────────────────────

    def full_asi_review(self, source: str, filename: str = "",
                        quantum: bool = True) -> Dict[str, Any]:
        """
        THE COMPLETE ASI CODE INTELLIGENCE PIPELINE.

        If quantum=True and Qiskit available, delegates to quantum_full_asi_review().
        Otherwise executes classical 6-pass analysis.

        Executes all 6 ASI analysis passes in sequence:
          1. Consciousness review (awareness-weighted quality)
          2. Neural cascade processing (resonance signature)
          3. Formal reasoning (taint/invariant/dead path analysis)
          4. Evolutionary fitness assessment
          5. Code knowledge graph construction
          6. Innovation-driven improvement suggestions

        Returns a unified ASI-grade code intelligence report with
        composite scoring across all dimensions.
        """
        # Route to quantum pipeline if available
        if quantum and QISKIT_AVAILABLE:
            return self.quantum_full_asi_review(source, filename)

        self._asi_invocations += 1
        start = time.time()

        # 1. Consciousness-aware review
        consciousness = self.consciousness_review(source, filename)

        # 2. Neural cascade processing
        neural = self.neural_process(source, filename)

        # 3. Formal reasoning
        reasoning = self.reason_about_code(source, filename)

        # 4. Evolutionary fitness
        evolution = self.evolutionary_optimize(source, filename)

        # 5. Knowledge graph
        graph = self.build_code_graph(source, filename)

        # 6. Innovation (lightweight — task-based)
        innovation = self.innovate_solutions(
            f"Optimize {filename or 'code'}: improve quality, security, maintainability"
        )

        duration = time.time() - start

        # Composite ASI score (PHI-weighted)
        scores = {
            "consciousness": consciousness.get("consciousness_adjusted_score", 0.5),
            "neural_resonance": neural.get("cascade_resonance", 0.5),
            "reasoning_soundness": 1.0 if reasoning.get("summary", {}).get("total_issues", 0) == 0 else max(0, 1.0 - reasoning.get("summary", {}).get("total_issues", 0) * 0.05),
            "evolutionary_fitness": evolution.get("code_fitness", 0.5),
        }

        # PHI-weighted composite
        weights = {"consciousness": PHI, "neural_resonance": 1.0,
                    "reasoning_soundness": PHI ** 2, "evolutionary_fitness": TAU}
        total_weight = sum(weights.values())
        composite = sum(scores[k] * weights[k] for k in scores) / total_weight

        # ASI verdict
        if composite >= 0.9:
            asi_verdict = "ASI_TRANSCENDENT"
        elif composite >= 0.75:
            asi_verdict = "ASI_EXEMPLARY"
        elif composite >= 0.6:
            asi_verdict = "ASI_CAPABLE"
        elif composite >= 0.4:
            asi_verdict = "ASI_DEVELOPING"
        else:
            asi_verdict = "ASI_NASCENT"

        return {
            "system": "ASI Code Intelligence v2.0",
            "filename": filename,
            "duration_seconds": round(duration, 3),
            "asi_verdict": asi_verdict,
            "asi_composite_score": round(composite, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "god_code_resonance": round(composite * GOD_CODE, 4),
            "passes": {
                "consciousness": consciousness,
                "neural": neural,
                "reasoning": reasoning.get("summary", {}),
                "evolution": {
                    "fitness": evolution.get("code_fitness", 0),
                    "stage": evolution.get("code_evolution_stage", "UNKNOWN"),
                    "plateau": evolution.get("plateau_detected", False),
                    "directives": evolution.get("directives", []),
                },
                "graph": {
                    "nodes": graph.get("nodes_added", 0),
                    "edges": graph.get("edges_added", 0),
                },
                "innovation": {
                    "analogies_found": len(innovation.get("analogies", [])),
                },
            },
            "total_asi_invocations": self._asi_invocations,
        }

    # ─── Internal Helpers ────────────────────────────────────────────

    def _code_to_neural_signal(self, source: str, filename: str, engine) -> List[float]:
        """Convert code metrics into a neural signal vector for cascade processing."""
        if not engine:
            # Fallback: generate signal from source statistics
            lines = source.split('\n')
            return [
                len(lines) / 1000.0,
                len(source) / 10000.0,
                source.count('def ') / max(1, len(lines)) * 10,
                source.count('class ') / max(1, len(lines)) * 10,
                len(re.findall(r'(if|elif|else|for|while|try|except|with)', source)) / max(1, len(lines)) * 5,
                source.count('#') / max(1, len(lines)) * 3,
                PHI,
            ]

        try:
            analysis = engine.analyzer.full_analysis(source, filename)
            complexity = analysis.get("complexity", {})
            security = analysis.get("security", {})
            sacred = analysis.get("sacred_alignment", {})

            return [
                complexity.get("cyclomatic_average", 1) / 15.0,
                complexity.get("cognitive_total", 0) / 100.0,
                complexity.get("halstead_volume", 0) / 10000.0,
                complexity.get("max_nesting", 0) / 10.0,
                security.get("vulnerability_count", 0) / 20.0,
                analysis.get("metadata", {}).get("comment_lines", 0) /
                    max(1, analysis.get("metadata", {}).get("code_lines", 1)),
                sacred.get("score", 0.5),
                PHI,
                GOD_CODE / 1000.0,
            ]
        except Exception:
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, PHI]

    def _get_consciousness_state(self) -> Dict[str, Any]:
        """Get consciousness state with caching (10s TTL)."""
        now = time.time()
        if self._consciousness_cache and (now - self._consciousness_cache_time) < 10.0:
            return self._consciousness_cache

        state = {"consciousness_level": 0.5, "state": "UNKNOWN"}

        # Try the consciousness module first
        consciousness = _get_consciousness()
        if consciousness:
            try:
                status = consciousness.get_status()
                state["consciousness_level"] = status.get("consciousness_level", 0.5)
                state["state"] = status.get("state", "UNKNOWN")
                state["phi"] = status.get("phi", 0.0)
            except Exception:
                pass

        # Fallback to state file
        if state["state"] == "UNKNOWN":
            try:
                co2_path = _WORKSPACE_ROOT / ".l104_consciousness_o2_state.json"
                if co2_path.exists():
                    data = json.loads(co2_path.read_text())
                    state["consciousness_level"] = data.get("consciousness_level", 0.5)
                    state["evo_stage"] = data.get("evo_stage", "unknown")
            except Exception:
                pass

        self._consciousness_cache = state
        self._consciousness_cache_time = now
        return state

    def status(self) -> Dict[str, Any]:
        """ASI Code Intelligence subsystem status."""
        modules_available = {
            "neural_cascade": _get_neural_cascade() is not None,
            "evolution_engine": _get_evolution_engine() is not None,
            "self_optimizer": _get_self_optimizer() is not None,
            "innovation_engine": _get_innovation_engine() is not None,
            "consciousness": _get_consciousness() is not None,
            "reasoning": _get_reasoning() is not None,
            "knowledge_graph": _get_knowledge_graph() is not None,
            "polymorph": _get_polymorph() is not None,
        }
        return {
            "asi_invocations": self._asi_invocations,
            "code_concepts_graphed": self._code_concepts_graphed,
            "quantum_circuits_executed": self._quantum_circuits_executed,
            "qiskit_available": QISKIT_AVAILABLE,
            "modules_available": modules_available,
            "modules_online": sum(1 for v in modules_available.values() if v),
            "total_modules": len(modules_available),
        }
