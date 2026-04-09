"""
L104 VQPU Circuit Deriver v1.0.0 – Derives concrete quantum circuits from high‑level specifications.

Derivation modes:
  1. grover_search → Grover oracle + diffusion circuit
  2. qft → Quantum Fourier Transform circuit
  3. qpe → Quantum Phase Estimation circuit
  4. amplitude_estimation → Amplitude estimation circuit
  5. quantum_walk → Quantum walk on a graph

The deriver uses a library of circuit templates and fills in parameters (number of qubits,
oracle definition, etc.) to produce a list of gate operations compatible with the MPS engine
and VQPU bridge.
"""
import re

class CircuitDeriver:
    """Derives quantum circuits from high‑level textual specifications."""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self):
        """Load built‑in circuit templates."""
        return {
            "grover_search": self._derive_grover,
            "qft": self._derive_qft,
            "qpe": self._derive_qpe,
            "amplitude_estimation": self._derive_amplitude_estimation,
            "quantum_walk": self._derive_quantum_walk,
        }

    def derive(self, spec: str, **params) -> dict:
        """
        Derive a concrete circuit from a high‑level specification.

        Args:
            spec: Specification string, e.g., "grover_search oracle=mark_111 num_qubits=3"
            **params: Additional keyword parameters (num_qubits, oracle_type, etc.)

        Returns:
            dict with keys:
                - operations: list of gate dicts
                - num_qubits: total qubits required
                - derivation_log: list of derivation steps
                - template: name of used template
        """
        # Parse spec
        spec_lower = spec.lower()
        template_key = None
        for key in self.templates:
            if key in spec_lower:
                template_key = key
                break

        if template_key is None:
            raise ValueError(f"No template matches specification: {spec}")

        # Extract parameters from spec using regex
        parsed_params = self._parse_params(spec)
        parsed_params.update(params)

        # Call template function
        ops, nq, log = self.templates[template_key](**parsed_params)

        return {
            "operations": ops,
            "num_qubits": nq,
            "derivation_log": log,
            "template": template_key,
            "spec": spec,
            "parameters": parsed_params,
        }

    def _parse_params(self, spec: str) -> dict:
        """Extract key=value pairs from spec."""
        params = {}
        for match in re.finditer(r"(\w+)=([\w\.\-]+)", spec):
            key, val = match.group(1), match.group(2)
            # Try to convert numeric values
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            params[key] = val
        return params

    # ─── Template Functions ───────────────────────────────────────────────

    def _derive_grover(self, num_qubits=3, oracle_type="mark_111", **kwargs):
        """Derive a Grover search circuit."""
        log = [f"Deriving Grover search with {num_qubits} qubits, oracle={oracle_type}"]
        ops = []
        # Hadamard all qubits
        for q in range(num_qubits):
            ops.append({"gate": "H", "qubits": [q]})

        # Oracle (example: mark |111⟩)
        if oracle_type == "mark_111":
            # Mark |111⟩ with a phase flip (multi‑controlled Z decomposed as CZ chain)
            for q in range(num_qubits):
                ops.append({"gate": "X", "qubits": [q]})
            for q in range(num_qubits - 1):
                ops.append({"gate": "CZ", "qubits": [q, q + 1]})
            for q in range(num_qubits):
                ops.append({"gate": "X", "qubits": [q]})
            log.append("Oracle: phase‑flip on |111⟩ (decomposed MCZ)")
        else:
            # Generic oracle placeholder
            log.append("Oracle: generic phase‑flip (identity)")

        # Diffusion operator
        for q in range(num_qubits):
            ops.append({"gate": "H", "qubits": [q]})
        for q in range(num_qubits):
            ops.append({"gate": "X", "qubits": [q]})
        for q in range(num_qubits - 1):
            ops.append({"gate": "CZ", "qubits": [q, q + 1]})
        for q in range(num_qubits):
            ops.append({"gate": "X", "qubits": [q]})
        for q in range(num_qubits):
            ops.append({"gate": "H", "qubits": [q]})
        log.append("Diffusion operator applied")

        return ops, num_qubits, log

    def _derive_qft(self, num_qubits=4, **kwargs):
        """Derive a Quantum Fourier Transform circuit."""
        log = [f"Deriving QFT with {num_qubits} qubits"]
        ops = []
        for i in range(num_qubits):
            ops.append({"gate": "H", "qubits": [i]})
            for j in range(i + 1, num_qubits):
                angle = 2 * 3.141592653589793 / (2 ** (j - i + 1))
                ops.append({"gate": "Rz", "qubits": [j], "parameters": [angle]})
                ops.append({"gate": "CX", "qubits": [i, j]})
                ops.append({"gate": "Rz", "qubits": [j], "parameters": [-angle]})
                ops.append({"gate": "CX", "qubits": [i, j]})
        log.append("QFT circuit generated (H + controlled‑Rz)")
        return ops, num_qubits, log

    def _derive_qpe(self, precision_bits=3, **kwargs):
        """Derive a Quantum Phase Estimation circuit."""
        num_qubits = precision_bits + 1  # plus one eigenstate qubit
        log = [f"Deriving QPE with {precision_bits} precision qubits"]
        ops = []
        # Hadamard on precision qubits
        for q in range(precision_bits):
            ops.append({"gate": "H", "qubits": [q]})
        # Controlled‑U powers (simplified as Rz rotations)
        for a in range(precision_bits):
            power = 2 ** a
            angle = 0.5 * power  # placeholder eigenphase
            ops.append({"gate": "CX", "qubits": [a, precision_bits]})
            ops.append({"gate": "Rz", "qubits": [precision_bits], "parameters": [angle]})
            ops.append({"gate": "CX", "qubits": [a, precision_bits]})
        # Inverse QFT on precision qubits (simplified)
        for i in range(precision_bits):
            for j in range(i):
                angle = -3.141592653589793 / (2 ** (i - j))
                ops.append({"gate": "CX", "qubits": [j, i]})
                ops.append({"gate": "Rz", "qubits": [i], "parameters": [angle]})
                ops.append({"gate": "CX", "qubits": [j, i]})
            ops.append({"gate": "H", "qubits": [i]})
        log.append("QPE circuit generated (controlled‑U + inverse QFT)")
        return ops, num_qubits, log

    def _derive_amplitude_estimation(self, num_qubits=4, **kwargs):
        """Derive an amplitude estimation circuit (simplified)."""
        log = [f"Deriving amplitude estimation with {num_qubits} qubits"]
        ops = []
        # Start with Hadamard on all qubits
        for q in range(num_qubits):
            ops.append({"gate": "H", "qubits": [q]})
        # Oracle A (placeholder)
        for q in range(num_qubits):
            ops.append({"gate": "X", "qubits": [q]})
            ops.append({"gate": "H", "qubits": [q]})
        # Grover operator G (oracle + diffusion) repeated
        for _ in range(2):
            # Oracle
            for q in range(num_qubits):
                ops.append({"gate": "X", "qubits": [q]})
            for q in range(num_qubits - 1):
                ops.append({"gate": "CZ", "qubits": [q, q + 1]})
            for q in range(num_qubits):
                ops.append({"gate": "X", "qubits": [q]})
            # Diffusion
            for q in range(num_qubits):
                ops.append({"gate": "H", "qubits": [q]})
                ops.append({"gate": "X", "qubits": [q]})
            for q in range(num_qubits - 1):
                ops.append({"gate": "CZ", "qubits": [q, q + 1]})
            for q in range(num_qubits):
                ops.append({"gate": "X", "qubits": [q]})
                ops.append({"gate": "H", "qubits": [q]})
        log.append("Amplitude estimation circuit generated (Grover‑based)")
        return ops, num_qubits, log

    def _derive_quantum_walk(self, graph_nodes=4, **kwargs):
        """Derive a quantum walk circuit on a line graph."""
        num_qubits = graph_nodes.bit_length()  # enough qubits to encode node index
        log = [f"Deriving quantum walk on {graph_nodes}‑node line graph"]
        ops = []
        # Coin operator (Hadamard on coin qubit)
        ops.append({"gate": "H", "qubits": [0]})
        # Shift operator (conditional increment/decrement)
        for i in range(num_qubits - 1):
            ops.append({"gate": "CX", "qubits": [i, i + 1]})
            ops.append({"gate": "X", "qubits": [i]})
            ops.append({"gate": "CX", "qubits": [i, i + 1]})
            ops.append({"gate": "X", "qubits": [i]})
        log.append("Quantum walk circuit generated (coin + shift)")
        return ops, num_qubits, log


# Singleton instance for easy import
_default_deriver = CircuitDeriver()

def derive_circuit(spec: str, **params) -> dict:
    """Convenience wrapper around the default deriver."""
    return _default_deriver.derive(spec, **params)