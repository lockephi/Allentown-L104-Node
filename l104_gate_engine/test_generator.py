"""L104 Gate Engine — Automated Test Generator."""

import math
from typing import Any, Dict, List

from .constants import PHI, TAU, GOD_CODE, CALABI_YAU_DIM
from .models import LogicGate
from .gate_functions import (
    sage_logic_gate, quantum_logic_gate,
    entangle_values, higher_dimensional_dissipation,
)


class GateTestGenerator:
    """Automated test generation and execution for logic gates."""

    def generate_tests(self, gates: List[LogicGate]) -> List[Dict[str, Any]]:
        """Generate test cases for discovered logic gates."""
        tests = []

        for gate in gates:
            if gate.language == "python" and gate.gate_type == "function":
                tests.extend(self._generate_python_function_tests(gate))
            elif gate.language == "python" and gate.gate_type == "class":
                tests.append(self._generate_python_class_test(gate))

        return tests

    def _generate_python_function_tests(self, gate: LogicGate) -> List[Dict[str, Any]]:
        """Generate test cases for a Python function gate."""
        tests = []
        test_inputs = [0.0, 1.0, PHI, -1.0, GOD_CODE * 0.001, math.pi, TAU]

        for i, inp in enumerate(test_inputs[:3]):
            tests.append({
                "test_id": f"{gate.name}_input_{i}",
                "gate_name": gate.name,
                "source_file": gate.source_file,
                "test_type": "smoke",
                "input": inp,
                "description": f"Smoke test {gate.name} with input {inp}",
                "status": "pending",
            })

        return tests

    def _generate_python_class_test(self, gate: LogicGate) -> Dict[str, Any]:
        """Generate an instantiation test for a Python class gate."""
        return {
            "test_id": f"{gate.name}_instantiation",
            "gate_name": gate.name,
            "source_file": gate.source_file,
            "test_type": "instantiation",
            "description": f"Verify {gate.name} class can be instantiated",
            "status": "pending",
        }

    def run_builtin_gate_tests(self) -> List[Dict[str, Any]]:
        """Run tests on the built-in sage_logic_gate and quantum_logic_gate."""
        results = []
        operations = ["align", "filter", "amplify", "compress", "entangle", "dissipate", "inflect"]
        test_values = [0.0, 1.0, PHI, -PHI, GOD_CODE * 0.001, math.pi, 100.0, -100.0]

        for op in operations:
            for val in test_values:
                test_id = f"sage_gate_{op}_{val}"
                try:
                    result = sage_logic_gate(val, op)
                    passed = not (math.isnan(result) or math.isinf(result))
                    results.append({
                        "test_id": test_id,
                        "gate_name": "sage_logic_gate",
                        "operation": op,
                        "input": val,
                        "output": result,
                        "passed": passed,
                        "error": None,
                    })
                except Exception as e:
                    results.append({
                        "test_id": test_id,
                        "gate_name": "sage_logic_gate",
                        "operation": op,
                        "input": val,
                        "output": None,
                        "passed": False,
                        "error": str(e),
                    })

        # Quantum gate tests
        for depth in range(1, 6):
            for val in [1.0, PHI, GOD_CODE * 0.001]:
                test_id = f"quantum_gate_d{depth}_{val}"
                try:
                    result = quantum_logic_gate(val, depth)
                    passed = not (math.isnan(result) or math.isinf(result))
                    results.append({
                        "test_id": test_id,
                        "gate_name": "quantum_logic_gate",
                        "depth": depth,
                        "input": val,
                        "output": result,
                        "passed": passed,
                        "error": None,
                    })
                except Exception as e:
                    results.append({
                        "test_id": test_id,
                        "gate_name": "quantum_logic_gate",
                        "depth": depth,
                        "input": val,
                        "output": None,
                        "passed": False,
                        "error": str(e),
                    })

        # Entanglement test
        for a, b in [(1.0, PHI), (GOD_CODE, TAU), (0.0, 0.0)]:
            try:
                ea, eb = entangle_values(a, b)
                passed = not (math.isnan(ea) or math.isnan(eb))
                results.append({
                    "test_id": f"entangle_{a}_{b}",
                    "gate_name": "entangle_values",
                    "input": (a, b),
                    "output": (ea, eb),
                    "passed": passed,
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "test_id": f"entangle_{a}_{b}",
                    "gate_name": "entangle_values",
                    "input": (a, b),
                    "output": None,
                    "passed": False,
                    "error": str(e),
                })

        # Higher-dimensional dissipation test
        try:
            test_pool = [math.sin(i * PHI) for i in range(64)]
            result = higher_dimensional_dissipation(test_pool)
            passed = len(result) == CALABI_YAU_DIM and all(
                not (math.isnan(v) or math.isinf(v)) for v in result
            )
            results.append({
                "test_id": "higher_dim_dissipation",
                "gate_name": "higher_dimensional_dissipation",
                "input": "64-element entropy pool",
                "output": result,
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "higher_dim_dissipation",
                "gate_name": "higher_dimensional_dissipation",
                "input": "64-element entropy pool",
                "output": None,
                "passed": False,
                "error": str(e),
            })

        # ─── INTEGRITY TESTS ───

        # Idempotency: compress(compress(x)) should converge
        for val in [PHI, GOD_CODE * 0.01, 1.0]:
            try:
                r1 = sage_logic_gate(val, "compress")
                r2 = sage_logic_gate(r1, "compress")
                r3 = sage_logic_gate(r2, "compress")
                converging = abs(r3 - r2) <= abs(r2 - r1) + 1e-10
                results.append({
                    "test_id": f"idempotency_compress_{val}",
                    "gate_name": "sage_logic_gate",
                    "operation": "compress_idempotency",
                    "input": val,
                    "output": [r1, r2, r3],
                    "passed": converging,
                    "error": None if converging else "Compression diverges instead of converging",
                })
            except Exception as e:
                results.append({
                    "test_id": f"idempotency_compress_{val}",
                    "gate_name": "sage_logic_gate", "operation": "compress_idempotency",
                    "input": val, "output": None, "passed": False, "error": str(e),
                })

        # Sacred constant coherence: PHI * TAU should ≈ 1.0
        try:
            phi_tau = PHI * TAU
            passed = abs(phi_tau - 1.0) < 1e-10
            results.append({
                "test_id": "sacred_phi_tau_unity",
                "gate_name": "sacred_constants",
                "input": "PHI * TAU",
                "output": phi_tau,
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "sacred_phi_tau_unity",
                "gate_name": "sacred_constants",
                "input": "PHI * TAU", "output": None, "passed": False, "error": str(e),
            })

        # Boundary: entangle(0,0) should return (0,0)
        try:
            ea, eb = entangle_values(0.0, 0.0)
            passed = abs(ea) < 1e-10 and abs(eb) < 1e-10
            results.append({
                "test_id": "entangle_zero_boundary",
                "gate_name": "entangle_values",
                "input": (0.0, 0.0),
                "output": (ea, eb),
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "entangle_zero_boundary",
                "gate_name": "entangle_values",
                "input": (0.0, 0.0), "output": None, "passed": False, "error": str(e),
            })

        # Symmetry: align(PHI) should equal PHI (PHI is a lattice point)
        try:
            aligned = sage_logic_gate(PHI, "align")
            passed = abs(aligned - PHI) < 0.01
            results.append({
                "test_id": "align_phi_fixpoint",
                "gate_name": "sage_logic_gate",
                "operation": "align_fixpoint",
                "input": PHI,
                "output": aligned,
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "align_phi_fixpoint",
                "gate_name": "sage_logic_gate", "operation": "align_fixpoint",
                "input": PHI, "output": None, "passed": False, "error": str(e),
            })

        # Amplify should always increase magnitude
        for val in [0.5, 1.0, PHI, 10.0]:
            try:
                amplified = sage_logic_gate(val, "amplify")
                passed = abs(amplified) > abs(val)
                results.append({
                    "test_id": f"amplify_increases_{val}",
                    "gate_name": "sage_logic_gate",
                    "operation": "amplify_increases",
                    "input": val,
                    "output": amplified,
                    "passed": passed,
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "test_id": f"amplify_increases_{val}",
                    "gate_name": "sage_logic_gate", "operation": "amplify_increases",
                    "input": val, "output": None, "passed": False, "error": str(e),
                })

        # Dissipation should preserve total energy (conservation-ish)
        try:
            pool = [math.sin(i * PHI) for i in range(64)]
            input_energy = sum(v ** 2 for v in pool)
            result_proj = higher_dimensional_dissipation(pool)
            output_energy = sum(v ** 2 for v in result_proj)
            passed = 0 < output_energy < float('inf') and not math.isnan(output_energy)
            results.append({
                "test_id": "dissipation_energy_finite",
                "gate_name": "higher_dimensional_dissipation",
                "input": f"pool_energy={input_energy:.4f}",
                "output": f"proj_energy={output_energy:.4f}",
                "passed": passed,
                "error": None,
            })
        except Exception as e:
            results.append({
                "test_id": "dissipation_energy_finite",
                "gate_name": "higher_dimensional_dissipation",
                "input": "64-pool", "output": None, "passed": False, "error": str(e),
            })

        return results
