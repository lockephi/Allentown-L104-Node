"""
===============================================================================
L104 SIMULATOR — ZERO-NOISE EXTRAPOLATION (ZNE) ERROR MITIGATION
===============================================================================

Implements Zero-Noise Extrapolation (ZNE) for mitigating noise in quantum
simulations. ZNE runs a circuit at multiple noise amplification levels
and extrapolates to the zero-noise limit.

Methods:
  - Linear extrapolation (2+ points)
  - Richardson extrapolation (exact polynomial)
  - Polynomial fit (user-specified degree)

Usage:
    from l104_simulator import Simulator, QuantumCircuit
    from l104_simulator.error_mitigation import zne_extrapolate

    sim = Simulator(noise_model={"depolarizing": 0.01})
    qc = QuantumCircuit(4, "test")
    qc.h(0).cx(0,1).cx(1,2).cx(2,3)

    Z = np.array([[1,0],[0,-1]])
    mitigated = zne_extrapolate(sim, qc, Z, scale_factors=[1, 2, 3])

INVARIANT: 527.5184818492612 | PILOT: LONDEL
===============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union

from .simulator import Simulator, QuantumCircuit


def zne_extrapolate(
    simulator: Simulator,
    circuit: QuantumCircuit,
    observable: Union[np.ndarray, List[List]],
    scale_factors: Optional[List[float]] = None,
    method: str = "richardson",
) -> Dict[str, Any]:
    """Zero-Noise Extrapolation for error mitigation.

    Runs the circuit at multiple noise scale factors and extrapolates
    the expectation value of `observable` to the zero-noise limit.

    Args:
        simulator: Simulator with a noise_model configured
        circuit: The quantum circuit to mitigate
        observable: Hermitian operator (full Hilbert space or single-qubit)
        scale_factors: Noise amplification factors (default: [1, 2, 3])
        method: Extrapolation method — "linear", "richardson", or "polynomial"

    Returns:
        Dict with mitigated_value, raw_values, scale_factors, method
    """
    if scale_factors is None:
        scale_factors = [1.0, 2.0, 3.0]

    observable = np.array(observable, dtype=complex)

    # If observable is single-qubit (2×2), expand to full Hilbert space for qubit 0
    n = circuit.n_qubits
    if observable.shape == (2, 2) and n > 1:
        full_obs = observable
        for _ in range(n - 1):
            full_obs = np.kron(full_obs, np.eye(2, dtype=complex))
        observable = full_obs

    base_noise = dict(simulator.noise_model)
    raw_values = []

    for scale in scale_factors:
        # Scale the noise model
        scaled_noise = {k: v * scale for k, v in base_noise.items()}
        scaled_sim = Simulator(noise_model=scaled_noise)
        result = scaled_sim.run(circuit)
        exp_val = result.expectation(observable)
        raw_values.append(float(exp_val))

    # Extrapolate to zero noise
    if method == "linear":
        mitigated = _linear_extrapolate(scale_factors, raw_values)
    elif method == "richardson":
        mitigated = _richardson_extrapolate(scale_factors, raw_values)
    elif method == "polynomial":
        mitigated = _polynomial_extrapolate(scale_factors, raw_values)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'linear', 'richardson', or 'polynomial'")

    return {
        "mitigated_value": mitigated,
        "raw_values": raw_values,
        "scale_factors": scale_factors,
        "method": method,
        "base_noise": base_noise,
        "improvement": abs(raw_values[0] - mitigated) if raw_values else 0.0,
    }


def _linear_extrapolate(scales: List[float], values: List[float]) -> float:
    """Linear extrapolation to x=0 using least squares."""
    x = np.array(scales)
    y = np.array(values)
    # Fit y = a + b*x, extrapolate to x=0 → y = a
    coeffs = np.polyfit(x, y, 1)
    return float(np.polyval(coeffs, 0.0))


def _richardson_extrapolate(scales: List[float], values: List[float]) -> float:
    """Richardson extrapolation (exact polynomial through all points).

    For n points, fits degree-(n-1) polynomial and evaluates at x=0.
    This is the optimal extrapolation when noise scales polynomially.
    """
    x = np.array(scales)
    y = np.array(values)
    degree = len(scales) - 1
    coeffs = np.polyfit(x, y, degree)
    return float(np.polyval(coeffs, 0.0))


def _polynomial_extrapolate(scales: List[float], values: List[float],
                             degree: int = 2) -> float:
    """Polynomial fit extrapolation with specified degree."""
    x = np.array(scales)
    y = np.array(values)
    degree = min(degree, len(scales) - 1)
    coeffs = np.polyfit(x, y, degree)
    return float(np.polyval(coeffs, 0.0))


def zne_sweep(
    simulator: Simulator,
    circuit: QuantumCircuit,
    observable: Union[np.ndarray, List[List]],
    max_scale: float = 5.0,
    n_points: int = 5,
) -> Dict[str, Any]:
    """Run ZNE with automatic scale factor sweep.

    Linearly spaces scale factors from 1 to max_scale, runs all three
    extrapolation methods, and returns the best result.

    Args:
        simulator: Simulator with noise_model
        circuit: Circuit to mitigate
        observable: Observable to measure
        max_scale: Maximum noise amplification factor
        n_points: Number of scale factors to sample

    Returns:
        Dict with results from all three methods + recommendation
    """
    scales = np.linspace(1.0, max_scale, n_points).tolist()

    results = {}
    for method in ["linear", "richardson", "polynomial"]:
        results[method] = zne_extrapolate(
            simulator, circuit, observable, scales, method
        )

    # Recommend the method closest to the median prediction (consensus-based)
    mitigated_vals = {m: r["mitigated_value"] for m, r in results.items()}
    median_val = float(np.median(list(mitigated_vals.values())))
    deviations = {m: abs(v - median_val) for m, v in mitigated_vals.items()}
    best = min(deviations, key=deviations.get)

    return {
        "results": results,
        "recommended_method": best,
        "recommended_value": results[best]["mitigated_value"],
        "scale_factors": scales,
    }
